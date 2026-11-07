import glob
import json
import mimetypes
import os
import shutil
import subprocess
import urllib
from pathlib import Path
from typing import List, Any, Optional, Tuple
from tqdm import tqdm

import modules.globals
from modules.model_downloader import _ssl_context

TEMP_FILE = "temp.mp4"
TEMP_DIRECTORY = "temp"


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


def create_video(target_path: str, fps: float = 30.0) -> bool:
    """Create video with hardware-accelerated encoding and optimized settings."""
    temp_output_path = get_temp_output_path(target_path)
    temp_directory_path = get_temp_directory_path(target_path)
    
    # Determine optimal encoder based on available hardware
    encoder = modules.globals.video_encoder
    encoder_options = []
    
    # GPU-accelerated encoding options
    if 'CUDAExecutionProvider' in modules.globals.execution_providers:
        # NVIDIA GPU encoding
        if encoder == 'libx264':
            encoder = 'h264_nvenc'
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
            encoder_options = [
                "-quality", "quality",  # Quality mode
                "-rc", "vbr_latency",
                "-qp_i", str(modules.globals.video_quality),
                "-qp_p", str(modules.globals.video_quality),
            ]
        elif encoder == 'libx265':
            encoder = 'hevc_amf'
            encoder_options = [
                "-quality", "quality",
                "-rc", "vbr_latency",
                "-qp_i", str(modules.globals.video_quality),
                "-qp_p", str(modules.globals.video_quality),
            ]
    else:
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
    
    # Build ffmpeg command
    ffmpeg_args = [
        "-r", str(fps),
        "-i", os.path.join(temp_directory_path, "%04d.png"),
        "-c:v", encoder,
    ]
    
    # Add encoder-specific options
    ffmpeg_args.extend(encoder_options)
    
    # Add common options
    ffmpeg_args.extend([
        "-pix_fmt", "yuv420p",
        "-movflags", "+faststart",  # Enable fast start for web playback
        "-vf", "colorspace=bt709:iall=bt601-6-625:fast=1",
        "-y",
        temp_output_path,
    ])
    
    # Try with hardware encoder first, fallback to software if it fails
    success = run_ffmpeg(ffmpeg_args)
    
    if not success and encoder in ['h264_nvenc', 'hevc_nvenc', 'h264_amf', 'hevc_amf']:
        # Fallback to software encoding
        print(f"Hardware encoding with {encoder} failed, falling back to software encoding...")
        fallback_encoder = 'libx264' if 'h264' in encoder else 'libx265'
        ffmpeg_args_fallback = [
            "-r", str(fps),
            "-i", os.path.join(temp_directory_path, "%04d.png"),
            "-c:v", fallback_encoder,
            "-preset", "medium",
            "-crf", str(modules.globals.video_quality),
            "-pix_fmt", "yuv420p",
            "-movflags", "+faststart",
            "-vf", "colorspace=bt709:iall=bt601-6-625:fast=1",
            "-y",
            temp_output_path,
        ]
        success = run_ffmpeg(ffmpeg_args_fallback)
    return success and os.path.isfile(temp_output_path)


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


# ─── face swap frame range ────────────────────────────────────────────────

# Stable, whole-string status keys for invalid ranges. UI translations key on
# these exact strings; keep them static (no interpolated numbers) so gettext
# can resolve them.
FACE_SWAP_RANGE_INELIGIBLE = (
    "Face swap range requires a constant-frame-rate MP4 with a single video track."
)
FACE_SWAP_RANGE_INCOMPLETE = (
    "Please mark both the start frame and the end frame, "
    "or clear the marks to process the whole video."
)
FACE_SWAP_RANGE_REVERSED = "The start frame cannot be later than the end frame."
FACE_SWAP_RANGE_FRAME_NOT_FOUND = (
    "A frame marked for the face swap range cannot be found in the target video."
)


def probe_video_streams(target_path: str) -> List[dict]:
    """Return every video stream's ffprobe metadata as a list of dicts."""
    command = [
        "ffprobe", "-v", "error",
        "-show_streams",
        "-of", "json",
        target_path,
    ]
    output = subprocess.check_output(command).decode("utf-8")
    data = json.loads(output)
    return [
        stream
        for stream in data.get("streams", [])
        if stream.get("codec_type") == "video"
    ]


def _parse_frame_rate(value: Any) -> Optional[float]:
    """Parse ffprobe rate strings ('30000/1001', '30') into a float."""
    if not value or not isinstance(value, str):
        return None
    try:
        if "/" in value:
            numerator, denominator = value.split("/", 1)
            denominator = float(denominator)
            if denominator == 0:
                return None
            return float(numerator) / denominator
        return float(value)
    except (TypeError, ValueError, ZeroDivisionError):
        return None


def is_face_swap_range_supported(target_path: str) -> bool:
    """True only for an .mp4 container with exactly one constant-frame-rate
    video stream. Frame indices marked by the UI only line up with the
    sequential frame order used by both export pipelines under these
    conditions."""
    if not target_path or os.path.splitext(target_path)[1].lower() != ".mp4":
        return False
    try:
        streams = probe_video_streams(target_path)
    except Exception:
        return False
    if len(streams) != 1:
        return False
    stream = streams[0]
    real_rate = _parse_frame_rate(stream.get("r_frame_rate"))
    average_rate = _parse_frame_rate(stream.get("avg_frame_rate"))
    if real_rate is None or average_rate is None or average_rate <= 0:
        return False
    # ffprobe reports equal real/average rates for constant frame rate.
    return abs(real_rate - average_rate) < 1e-6


def get_video_frame_count(target_path: str) -> Optional[int]:
    """Authoritative frame count of the first video stream.

    MP4 containers expose the sample count directly via nb_frames; fall back
    to packet counting (still container-only, no decode) when it is absent.
    """
    commands = [
        [
            "ffprobe", "-v", "error",
            "-select_streams", "v:0",
            "-show_entries", "stream=nb_frames",
            "-of", "default=noprint_wrappers=1:nokey=1",
            target_path,
        ],
        [
            "ffprobe", "-v", "error",
            "-select_streams", "v:0",
            "-count_packets",
            "-show_entries", "stream=nb_read_packets",
            "-of", "default=noprint_wrappers=1:nokey=1",
            target_path,
        ],
    ]
    for command in commands:
        try:
            output = subprocess.check_output(command).decode("utf-8").strip()
            if output and output != "N/A":
                count = int(output)
                if count > 0:
                    return count
        except (subprocess.CalledProcessError, OSError, ValueError):
            continue
    return None


def frame_index_in_face_swap_range(
    index: int, start: Optional[int], end: Optional[int]
) -> bool:
    """Inclusive [start, end] test. Unset bounds mean 'process everything'."""
    if start is None or end is None:
        return True
    return start <= index <= end


def preview_position_to_frame_index(position: int) -> int:
    """Map a preview slider position to the 0-based decode index of the frame
    capturer.get_video_frame actually displays. Positions 0 and 1 both show
    the first frame; position N (N >= 2) shows decode index N - 1."""
    return 0 if position <= 1 else position - 1


def get_active_face_swap_range(
    target_path: str,
) -> Tuple[Optional[int], Optional[int]]:
    """Return the (start, end) marks that apply to target_path.

    Marks belong to the target they were taken from; marks left behind by a
    changed target are ignored rather than silently applied."""
    start = modules.globals.face_swap_start_frame
    end = modules.globals.face_swap_end_frame
    if start is None and end is None:
        return None, None
    if modules.globals.face_swap_range_target != target_path:
        return None, None
    return start, end


def validate_face_swap_range(
    target_path: str,
    start: Optional[int],
    end: Optional[int],
) -> Optional[str]:
    """Return a translatable error reason when the marked range cannot be
    exported, or None when processing should proceed.

    No marks (whole video) always passes. Otherwise reject:
      * target that is not a CFR single-video-track MP4,
      * only one end marked,
      * start later than end (or a negative index),
      * an end frame beyond the video's actual frame count.
    """
    if start is None and end is None:
        return None
    if not is_face_swap_range_supported(target_path):
        return FACE_SWAP_RANGE_INELIGIBLE
    if (start is None) != (end is None):
        return FACE_SWAP_RANGE_INCOMPLETE
    if start < 0 or end < 0 or start > end:
        return FACE_SWAP_RANGE_REVERSED
    total_frames = get_video_frame_count(target_path)
    if total_frames is None or end >= total_frames or start >= total_frames:
        return FACE_SWAP_RANGE_FRAME_NOT_FOUND
    return None


def select_temp_frames_in_range(
    temp_frame_paths: List[str],
    start: Optional[int],
    end: Optional[int],
) -> List[str]:
    """Sort extracted '%04d.png' frame files (1-based names) and keep those
    whose 0-based decode index lies in the inclusive [start, end].

    Frames outside the range are left untouched on disk, so the encoding
    step reassembles every frame in its original order. Both bounds unset
    returns the full, deterministically ordered list."""
    ordered = sorted(temp_frame_paths)
    if start is None or end is None:
        return ordered
    selected: List[str] = []
    for path in ordered:
        name = os.path.splitext(os.path.basename(path))[0]
        if not name.isdigit():
            # Extraction always writes %04d.png; process anything unexpected
            # rather than silently leaving it unswapped.
            selected.append(path)
            continue
        index = int(name) - 1  # extraction starts at 0001 for frame 0
        if start <= index <= end:
            selected.append(path)
    return selected
