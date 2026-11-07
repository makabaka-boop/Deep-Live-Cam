import glob
import json
import mimetypes
import os
import shlex
import shutil
import subprocess
import urllib
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Any, Optional
from tqdm import tqdm

import modules.globals
from modules.model_downloader import _ssl_context

TEMP_FILE = "temp.mp4"
TEMP_DIRECTORY = "temp"
TIMING_MANIFEST_FILE = "timing.ffconcat"


@dataclass
class VideoTiming:
    """Per-frame timing information read from the source video."""

    pts: List[float]        # presentation timestamp of every frame (seconds)
    durations: List[float]  # how long every frame stays on screen (seconds)
    duration: float         # total video timeline duration (seconds)
    has_audio: bool         # whether the source contains an audio stream



def run_ffmpeg(args: List[str]) -> bool:
    """Run ffmpeg with hardware acceleration and optimized settings."""
    commands = [
        "ffmpeg",
        "-hide_banner",
        "-hwaccel", "auto",  # Auto-detect hardware acceleration
        "-hwaccel_output_format", "auto",  # Use hardware format when possible
        "-threads", str(modules.globals.execution_threads or 0),  # 0 = auto-detect optimal thread count
        "-loglevel", modules.globals.log_level,
    ]
    commands.extend(args)
    try:
        subprocess.check_output(commands, stderr=subprocess.STDOUT)
        return True
    except subprocess.CalledProcessError as error:
        output = error.output.decode(errors="ignore").strip()
        if output:
            print(output)
    except Exception as error:
        print(f"ffmpeg execution failed: {error}")
    return False


def detect_fps(target_path: str) -> float:
    command = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=r_frame_rate",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        target_path,
    ]
    try:
        output = subprocess.check_output(command, encoding="utf-8").strip().split("/")
        numerator, denominator = map(int, output)
        return numerator / denominator
    except (
        subprocess.CalledProcessError,
        OSError,
        UnicodeDecodeError,
        ValueError,
        ZeroDivisionError,
    ):
        pass
    return 30.0


def _ffprobe_json(target_path: str, entries_args: List[str]) -> dict:
    """Run ffprobe and return its JSON output as a dict."""
    command = ["ffprobe", "-v", "error", "-of", "json"]
    command.extend(entries_args)
    command.append(target_path)
    output = subprocess.check_output(command, encoding="utf-8")
    return json.loads(output)


def _ffprobe_time(value: Any) -> Optional[float]:
    """Parse an ffprobe time field; 'N/A'/missing/invalid becomes None."""
    if value is None:
        return None
    value = str(value).strip()
    if not value or value.upper() == "N/A":
        return None
    try:
        return float(value)
    except ValueError:
        return None


def probe_video_timing(target_path: str) -> Tuple[Optional[VideoTiming], Optional[str]]:
    """Read reliable per-frame presentation times from the source video.

    Returns ``(timing, None)`` on success or ``(None, reason)`` when the file
    is not a single-video-track MP4 or no trustworthy timing can be read.
    """
    if os.path.splitext(target_path)[1].lower() != ".mp4":
        return None, "Preserve timing only supports .mp4 files."

    try:
        meta = _ffprobe_json(
            target_path,
            ["-show_entries", "stream=codec_type,start_time,duration,nb_frames:format=duration,format_name"],
        )
        frames_data = _ffprobe_json(
            target_path,
            [
                "-select_streams", "v:0",
                "-show_entries", "frame=pts_time,best_effort_timestamp_time,duration_time",
            ],
        )
    except (subprocess.CalledProcessError, OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        return None, f"ffprobe could not read reliable frame timing ({error})."

    streams = meta.get("streams", [])
    video_streams = [stream for stream in streams if stream.get("codec_type") == "video"]
    if len(video_streams) != 1:
        return None, f"Expected exactly one video stream, found {len(video_streams)}."
    video_stream = video_streams[0]
    has_audio = any(stream.get("codec_type") == "audio" for stream in streams)

    frames = frames_data.get("frames", [])
    if not frames:
        return None, "No video frames with timing information were found."

    pts: List[float] = []
    frame_durations: List[Optional[float]] = []
    for frame in frames:
        timestamp = _ffprobe_time(frame.get("pts_time"))
        if timestamp is None:
            timestamp = _ffprobe_time(frame.get("best_effort_timestamp_time"))
        pts.append(timestamp)
        frame_durations.append(_ffprobe_time(frame.get("duration_time")))

    if any(timestamp is None for timestamp in pts):
        return None, "Some frames have no readable presentation timestamp."
    if abs(pts[0]) > 0.05:
        return None, "The video timeline does not start at 0; timing offset is unsupported."
    for previous, current in zip(pts, pts[1:]):
        if current <= previous + 1e-6:
            return None, "Frame timestamps stall or go backwards; timing is not reliable."

    declared_frames = video_stream.get("nb_frames")
    if declared_frames not in (None, "N/A"):
        try:
            if int(declared_frames) != len(frames):
                return None, (
                    f"Declared frame count ({declared_frames}) does not match "
                    f"the {len(frames)} timestamped frames."
                )
        except ValueError:
            pass

    stream_duration = _ffprobe_time(video_stream.get("duration"))
    if stream_duration is None:
        stream_duration = _ffprobe_time(meta.get("format", {}).get("duration"))

    # Prefer the per-frame durations stored in the container (they include the
    # last frame's hold time); derive them from PTS gaps otherwise.
    if all(duration is not None and duration > 0 for duration in frame_durations):
        for index in range(len(pts) - 1):
            gap = pts[index + 1] - pts[index]
            if abs(frame_durations[index] - gap) > max(0.005, gap * 0.1):
                return None, "Per-frame durations disagree with the frame timestamps."
        durations = list(frame_durations)
    else:
        durations = [pts[index + 1] - pts[index] for index in range(len(pts) - 1)]
        if stream_duration is None:
            return None, "The video duration is unknown, so the last frame hold time cannot be determined."
        last_duration = stream_duration - pts[-1]
        if last_duration <= 1e-4:
            return None, "The duration ends at or before the last frame; the last frame hold time is unknown."
        durations.append(last_duration)

    for duration in durations:
        if not 1e-4 < duration < 2.0:
            return None, f"Unreliable frame duration read from the file ({duration:.6f}s)."

    total_duration = sum(durations)
    if stream_duration is not None and abs(stream_duration - total_duration) > max(0.1, total_duration * 0.02):
        return None, "The declared video duration disagrees with the frame timestamps."

    return VideoTiming(pts=pts, durations=durations, duration=total_duration, has_audio=has_audio), None


def verify_video_timing(output_path: str, expected_frames: int, expected_duration: float) -> Optional[str]:
    """Confirm an encoded file matches the source frame count and timeline.

    Returns ``None`` when the output lines up, or a human-readable reason.
    """
    try:
        data = _ffprobe_json(
            output_path,
            [
                "-select_streams", "v:0",
                "-show_entries", "frame=pts_time:format=duration",
            ],
        )
    except (subprocess.CalledProcessError, OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        return f"ffprobe could not verify the encoded video ({error})."

    frames = [
        _ffprobe_time(frame.get("pts_time"))
        for frame in data.get("frames", [])
    ]
    frames = [timestamp for timestamp in frames if timestamp is not None]
    if len(frames) != expected_frames:
        return (
            f"The encoded video contains {len(frames)} frames but "
            f"{expected_frames} frames were processed."
        )
    if any(current <= previous for previous, current in zip(frames, frames[1:])):
        return "The encoded frame timestamps are not increasing."

    duration = _ffprobe_time(data.get("format", {}).get("duration"))
    tolerance = max(0.06, expected_duration * 0.02)
    if duration is None or abs(duration - expected_duration) > tolerance:
        return (
            f"The encoded duration ({duration:.3f}s if known) does not match "
            f"the source timing ({expected_duration:.3f}s)."
        )
    return None


def extract_frames(target_path: str) -> None:
    """Extract frames with hardware acceleration and optimized settings."""
    temp_directory_path = get_temp_directory_path(target_path)
    
    # Write a contiguous image sequence so the later "%04d.png" input pattern
    # used during encoding can consume every frame reliably.
    run_ffmpeg(
        [
            "-i", target_path,
            "-vf", "format=rgb24",  # Use video filter for format conversion (faster)
            "-vsync", "0",  # Prevent frame duplication
            os.path.join(temp_directory_path, "%04d.png"),
        ]
    )


def _encoder_candidates() -> List[Tuple[str, List[str], bool]]:
    """Encoders to try, in order: hardware encoder first, software fallback last.

    Each entry is ``(encoder, encoder_options, is_hardware)``. The software
    paths below encode exactly the same options as the historical
    ``create_video`` implementation.
    """
    encoder = modules.globals.video_encoder
    encoder_options: List[str] = []
    is_hardware = False

    # GPU-accelerated encoding options
    if 'CUDAExecutionProvider' in modules.globals.execution_providers:
        # NVIDIA GPU encoding
        if encoder == 'libx264':
            encoder = 'h264_nvenc'
            is_hardware = True
            encoder_options = [
                "-preset", "p7",  # Highest quality preset for NVENC
                "-tune", "hq",  # High quality tuning
                "-rc", "vbr",  # Variable bitrate
                "-cq", str(modules.globals.video_quality),  # Quality level
                "-b:v", "0",  # Let CQ control bitrate
                "-multipass", "fullres",  # Two-pass encoding for better quality
            ]
        elif encoder == 'libx265':
            encoder = 'hevc_nvenc'
            is_hardware = True
            encoder_options = [
                "-preset", "p7",
                "-tune", "hq",
                "-rc", "vbr",
                "-cq", str(modules.globals.video_quality),
                "-b:v", "0",
            ]
    elif 'DmlExecutionProvider' in modules.globals.execution_providers:
        # AMD/Intel GPU encoding (DirectML on Windows)
        if encoder == 'libx264':
            # Try AMD AMF encoder
            encoder = 'h264_amf'
            is_hardware = True
            encoder_options = [
                "-quality", "quality",  # Quality mode
                "-rc", "vbr_latency",
                "-qp_i", str(modules.globals.video_quality),
                "-qp_p", str(modules.globals.video_quality),
            ]
        elif encoder == 'libx265':
            encoder = 'hevc_amf'
            is_hardware = True
            encoder_options = [
                "-quality", "quality",
                "-rc", "vbr_latency",
                "-qp_i", str(modules.globals.video_quality),
                "-qp_p", str(modules.globals.video_quality),
            ]

    if is_hardware:
        candidates = [(encoder, encoder_options, True)]
        fallback_encoder = 'libx264' if 'h264' in encoder else 'libx265'
        fallback_options = ["-preset", "medium", "-crf", str(modules.globals.video_quality)]
        candidates.append((fallback_encoder, fallback_options, False))
        return candidates

    # CPU encoding with optimized settings
    if encoder == 'libx264':
        encoder_options = [
            "-preset", "medium",  # Balance speed/quality
            "-crf", str(modules.globals.video_quality),
            "-tune", "film",  # Optimize for film content
        ]
    elif encoder == 'libx265':
        encoder_options = [
            "-preset", "medium",
            "-crf", str(modules.globals.video_quality),
            "-x265-params", "log-level=error",
        ]
    elif encoder == 'libvpx-vp9':
        encoder_options = [
            "-crf", str(modules.globals.video_quality),
            "-b:v", "0",  # Constant quality mode
            "-cpu-used", "2",  # Speed vs quality (0-5, lower=slower/better)
        ]
    return [(encoder, encoder_options, False)]


def _common_encode_args(temp_output_path: str) -> List[str]:
    return [
        "-pix_fmt", "yuv420p",
        "-movflags", "+faststart",  # Enable fast start for web playback
        "-vf", "colorspace=bt709:iall=bt601-6-625:fast=1",
    ]


def create_video(target_path: str, fps: float = 30.0) -> bool:
    """Create video with hardware-accelerated encoding and optimized settings."""
    temp_output_path = get_temp_output_path(target_path)
    temp_directory_path = get_temp_directory_path(target_path)

    input_args = ["-r", str(fps), "-i", os.path.join(temp_directory_path, "%04d.png")]

    # Try with hardware encoder first, fallback to software if it fails
    for attempt, (encoder, encoder_options, is_hardware) in enumerate(_encoder_candidates()):
        if attempt > 0:
            print(f"Hardware encoding failed, falling back to software encoding ({encoder})...")

        # Build ffmpeg command
        ffmpeg_args = [*input_args, "-c:v", encoder]
        ffmpeg_args.extend(encoder_options)
        ffmpeg_args.extend(_common_encode_args(temp_output_path))
        ffmpeg_args.extend(["-y", temp_output_path])

        success = run_ffmpeg(ffmpeg_args)
        if success and os.path.isfile(temp_output_path):
            return True
        if not is_hardware:
            break
    return False


def _frame_path_index(frame_path: str) -> int:
    try:
        return int(os.path.splitext(os.path.basename(frame_path))[0])
    except ValueError:
        return 0


def create_video_vfr(target_path: str, durations: List[float]) -> bool:
    """Encode the processed PNG frames using the source per-frame durations.

    A concat demuxer manifest places every frame at its original time instead
    of exporting at one fixed frame rate, which preserves the playback rhythm
    of variable-frame-rate recordings (including the last frame's hold time).
    """
    temp_output_path = get_temp_output_path(target_path)
    temp_directory_path = get_temp_directory_path(target_path)

    frame_paths = sorted(get_temp_frame_paths(target_path), key=_frame_path_index)
    if len(frame_paths) != len(durations) or not frame_paths:
        print(
            f"[DLC.CORE] Timing data has {len(durations)} entries but "
            f"{len(frame_paths)} processed frames were found."
        )
        return False

    manifest_path = os.path.join(temp_directory_path, TIMING_MANIFEST_FILE)
    manifest_lines = ["ffconcat version 1.0"]
    for frame_path, duration in zip(frame_paths, durations):
        frame_name = shlex.quote(os.path.basename(frame_path))
        manifest_lines.append(f"file {frame_name}")
        manifest_lines.append(f"duration {duration:.9f}")
    # The concat demuxer needs the last file repeated for its duration to take
    # effect; this trailing entry emits no additional frame.
    manifest_lines.append(f"file {shlex.quote(os.path.basename(frame_paths[-1]))}")
    try:
        Path(manifest_path).write_text("\n".join(manifest_lines) + "\n", encoding="utf-8")
    except OSError as error:
        print(f"[DLC.CORE] Failed to write timing manifest: {error}")
        return False

    input_args = ["-f", "concat", "-safe", "0", "-i", manifest_path]

    for attempt, (encoder, encoder_options, is_hardware) in enumerate(_encoder_candidates()):
        if attempt > 0:
            print(f"Hardware encoding failed, falling back to software encoding ({encoder})...")

        ffmpeg_args = [*input_args, "-c:v", encoder]
        ffmpeg_args.extend(encoder_options)
        ffmpeg_args.extend(_common_encode_args(temp_output_path))
        # Keep the manifest's variable frame durations verbatim - never let the
        # encoder duplicate/drop frames to force a constant frame rate.
        ffmpeg_args.extend(["-vsync", "0", "-y", temp_output_path])

        success = run_ffmpeg(ffmpeg_args)
        if success and os.path.isfile(temp_output_path):
            return True
        if not is_hardware:
            break
    return False


def restore_audio(target_path: str, output_path: str) -> None:
    temp_output_path = get_temp_output_path(target_path)
    done = run_ffmpeg(
        [
            "-i",
            temp_output_path,
            "-i",
            target_path,
            "-c:v",
            "copy",
            "-map",
            "0:v:0",
            "-map",
            "1:a:0",
            "-y",
            output_path,
        ]
    )
    if not done:
        move_temp(target_path, output_path)


def restore_audio_timed(target_path: str, output_path: str) -> bool:
    """Mux the original audio without shifting it against the video.

    The encoded video reproduces the source timeline starting at 0, so the
    source audio stream can be copied 1:1 and stays in relative sync. If the
    source audio codec cannot be carried in MP4 as-is (rare for phone clips),
    it is transcoded to AAC, which keeps the same timestamps. Returns False
    (without writing a fallback file) only when both attempts fail.
    """
    temp_output_path = get_temp_output_path(target_path)
    base_args = [
        "-i",
        temp_output_path,
        "-i",
        target_path,
        "-map",
        "0:v:0",
        "-map",
        "1:a:0",
        "-c:v",
        "copy",
    ]
    if run_ffmpeg([*base_args, "-c:a", "copy", "-y", output_path]):
        return True
    print("[DLC.CORE] Audio stream copy failed, retrying with AAC audio...")
    return run_ffmpeg([*base_args, "-c:a", "aac", "-b:a", "192k", "-y", output_path])


def get_temp_frame_paths(target_path: str) -> List[str]:
    temp_directory_path = get_temp_directory_path(target_path)
    return glob.glob((os.path.join(glob.escape(temp_directory_path), "*.png")))


def get_temp_directory_path(target_path: str) -> str:
    target_name, _ = os.path.splitext(os.path.basename(target_path))
    target_directory_path = os.path.dirname(target_path)
    return os.path.join(target_directory_path, TEMP_DIRECTORY, target_name)


def get_temp_output_path(target_path: str) -> str:
    temp_directory_path = get_temp_directory_path(target_path)
    return os.path.join(temp_directory_path, TEMP_FILE)


def normalize_output_path(source_path: str, target_path: str, output_path: str) -> Any:
    if source_path and target_path:
        source_name, _ = os.path.splitext(os.path.basename(source_path))
        target_name, target_extension = os.path.splitext(os.path.basename(target_path))
        if os.path.isdir(output_path):
            return os.path.join(
                output_path, source_name + "-" + target_name + target_extension
            )
    return output_path


def create_temp(target_path: str) -> None:
    temp_directory_path = get_temp_directory_path(target_path)
    Path(temp_directory_path).mkdir(parents=True, exist_ok=True)


def move_temp(target_path: str, output_path: str) -> None:
    temp_output_path = get_temp_output_path(target_path)
    if os.path.isfile(temp_output_path):
        if os.path.isfile(output_path):
            os.remove(output_path)
        shutil.move(temp_output_path, output_path)


def clean_temp(target_path: str) -> None:
    temp_directory_path = get_temp_directory_path(target_path)
    parent_directory_path = os.path.dirname(temp_directory_path)
    if not modules.globals.keep_frames and os.path.isdir(temp_directory_path):
        shutil.rmtree(temp_directory_path)
    if os.path.exists(parent_directory_path) and not os.listdir(parent_directory_path):
        os.rmdir(parent_directory_path)


def has_image_extension(image_path: str) -> bool:
    # splitext so only the real extension counts (e.g. "photo.png.bak" is not
    # an image); the set is centralized in globals to stay in sync with dialogs.
    return os.path.splitext(image_path)[1].lower() in modules.globals.IMAGE_EXTENSIONS


def is_image(image_path: str) -> bool:
    if image_path and os.path.isfile(image_path):
        # Extension check first — Windows mimetypes doesn't always register webp
        if has_image_extension(image_path):
            return True
        mimetype, _ = mimetypes.guess_type(image_path)
        return bool(mimetype and mimetype.startswith("image/"))
    return False


def is_video(video_path: str) -> bool:
    if video_path and os.path.isfile(video_path):
        mimetype, _ = mimetypes.guess_type(video_path)
        return bool(mimetype and mimetype.startswith("video/"))
    return False


def conditional_download(download_directory_path: str, urls: List[str]) -> None:
    if not os.path.exists(download_directory_path):
        os.makedirs(download_directory_path)
    for url in urls:
        download_file_path = os.path.join(
            download_directory_path, os.path.basename(url)
        )
        if not os.path.exists(download_file_path):
            request = urllib.request.Request(url)

            # Verified TLS on every platform; never disable verification here —
            # model files are executed by the ONNX runtime (#1890).
            response = urllib.request.urlopen(request, context=_ssl_context())
            total = int(response.headers.get("Content-Length", 0))
            with tqdm(
                total=total,
                desc="Downloading",
                unit="B",
                unit_scale=True,
                unit_divisor=1024,
            ) as progress:
                with open(download_file_path, "wb") as f:
                    while True:
                        buffer = response.read(8192)
                        if not buffer:
                            break
                        f.write(buffer)
                        progress.update(len(buffer))


def resolve_relative_path(path: str) -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), path))


def get_video_dimensions(target_path: str) -> tuple:
    """Get video width and height using ffprobe."""
    command = [
        "ffprobe", "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=width,height",
        "-of", "csv=p=0:s=x",
        target_path,
    ]
    output = subprocess.check_output(command).decode().strip()
    width, height = map(int, output.split("x"))
    return width, height


def estimate_frame_count(target_path: str, fps: float = None) -> int:
    """Estimate total frame count from video duration and fps."""
    if fps is None:
        fps = detect_fps(target_path)
    command = [
        "ffprobe", "-v", "error",
        "-show_entries", "format=duration",
        "-of", "csv=p=0",
        target_path,
    ]
    try:
        output = subprocess.check_output(command).decode().strip()
        duration = float(output)
        return int(duration * fps)
    except Exception:
        return 0
