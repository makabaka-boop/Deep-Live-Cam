import os
import sys
# single thread doubles cuda performance - needs to be set before torch import
if any(arg.startswith('--execution-provider') for arg in sys.argv):
    os.environ['OMP_NUM_THREADS'] = '6'
# reduce tensorflow log level
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import warnings
from typing import List
import platform
import signal
import shutil
import argparse
import time
try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
import onnxruntime
try:
    import tensorflow
    HAS_TENSORFLOW = True
except ImportError:
    HAS_TENSORFLOW = False

import modules.globals
import modules.metadata
import modules.ui as ui
from modules.processors.frame.core import get_frame_processors_modules, process_video_in_memory
from modules.utilities import has_image_extension, is_image, is_video, detect_fps, create_video, create_video_vfr, probe_video_timing, verify_video_timing, restore_audio, restore_audio_timed, extract_frames, get_temp_frame_paths, get_temp_output_path, create_temp, move_temp, clean_temp, normalize_output_path

if HAS_TORCH and 'ROCMExecutionProvider' in modules.globals.execution_providers:
    del torch

warnings.filterwarnings('ignore', category=FutureWarning, module='insightface')
if HAS_TORCH:
    warnings.filterwarnings('ignore', category=UserWarning, module='torchvision')


def parse_args() -> None:
    signal.signal(signal.SIGINT, lambda signal_number, frame: destroy())
    program = argparse.ArgumentParser()
    program.add_argument('-s', '--source', help='select an source image', dest='source_path')
    program.add_argument('-t', '--target', help='select an target image or video', dest='target_path')
    program.add_argument('-o', '--output', help='select output file or directory', dest='output_path')
    program.add_argument('--frame-processor', help='pipeline of frame processors', dest='frame_processor', default=['face_swapper'], choices=['face_swapper', 'face_enhancer', 'face_enhancer_gpen256', 'face_enhancer_gpen512'], nargs='+')
    program.add_argument('--keep-fps', help='keep original fps', dest='keep_fps', action='store_true', default=False)
    program.add_argument('--keep-timing', help='preserve original per-frame playback timing (variable frame rate mp4)', dest='keep_timing', action='store_true', default=False)
    program.add_argument('--keep-audio', help='keep original audio', dest='keep_audio', action='store_true', default=True)
    program.add_argument('--keep-frames', help='keep temporary frames', dest='keep_frames', action='store_true', default=False)
    program.add_argument('--many-faces', help='process every face', dest='many_faces', action='store_true', default=False)
    program.add_argument('--nsfw-filter', help='filter the NSFW image or video', dest='nsfw_filter', action='store_true', default=False)
    program.add_argument('--map-faces', help='map source target faces', dest='map_faces', action='store_true', default=False)
    program.add_argument('--mouth-mask', help='mask the mouth region', dest='mouth_mask', action='store_true', default=False)
    program.add_argument('--video-encoder', help='adjust output video encoder', dest='video_encoder', default='libx264', choices=['libx264', 'libx265', 'libvpx-vp9'])
    program.add_argument('--video-quality', help='adjust output video quality', dest='video_quality', type=int, default=18, choices=range(52), metavar='[0-51]')
    program.add_argument('-l', '--lang', help='Ui language', default="en")
    program.add_argument('--live-mirror', help='The live camera display as you see it in the front-facing camera frame', dest='live_mirror', action='store_true', default=False)
    program.add_argument('--live-resizable', help='The live camera frame is resizable', dest='live_resizable', action='store_true', default=False)
    program.add_argument('--max-memory', help='maximum amount of RAM in GB', dest='max_memory', type=int, default=suggest_max_memory())
    program.add_argument('--execution-provider', help='execution provider', dest='execution_provider', default=[suggest_default_execution_provider()], choices=suggest_execution_providers(), nargs='+')
    program.add_argument('--execution-threads', help='number of execution threads', dest='execution_threads', type=int, default=None)
    program.add_argument('--det-size', help='face detection input size (160, 320, or 640)', dest='det_size', type=int, default=modules.globals.DEFAULT_DET_SIZE, choices=[160, 320, 640])
    program.add_argument('-v', '--version', action='version', version=f'{modules.metadata.name} {modules.metadata.version}')

    # register deprecated args
    program.add_argument('-f', '--face', help=argparse.SUPPRESS, dest='source_path_deprecated')
    program.add_argument('--cpu-cores', help=argparse.SUPPRESS, dest='cpu_cores_deprecated', type=int)
    program.add_argument('--gpu-vendor', help=argparse.SUPPRESS, dest='gpu_vendor_deprecated')
    program.add_argument('--gpu-threads', help=argparse.SUPPRESS, dest='gpu_threads_deprecated', type=int)

    args = program.parse_args()

    modules.globals.source_path = args.source_path
    modules.globals.target_path = args.target_path
    modules.globals.output_path = normalize_output_path(modules.globals.source_path, modules.globals.target_path, args.output_path)
    modules.globals.frame_processors = args.frame_processor
    modules.globals.headless = args.source_path or args.target_path or args.output_path
    modules.globals.keep_fps = args.keep_fps
    modules.globals.keep_timing = args.keep_timing
    modules.globals.keep_audio = args.keep_audio
    modules.globals.keep_frames = args.keep_frames
    modules.globals.many_faces = args.many_faces
    modules.globals.mouth_mask = args.mouth_mask
    modules.globals.nsfw_filter = args.nsfw_filter
    modules.globals.map_faces = args.map_faces
    modules.globals.video_encoder = args.video_encoder
    modules.globals.video_quality = args.video_quality
    modules.globals.live_mirror = args.live_mirror
    modules.globals.live_resizable = args.live_resizable
    modules.globals.max_memory = args.max_memory
    modules.globals.execution_providers = decode_execution_providers(args.execution_provider)
    modules.globals.execution_threads = args.execution_threads
    modules.globals.det_size = args.det_size
    modules.globals.lang = args.lang

    # The argparse default (None) avoids evaluating suggest_execution_threads()
    # before providers are decoded, and deprecated-arg overrides above may
    # have already set execution_threads.
    if modules.globals.execution_threads is None:
        modules.globals.execution_threads = suggest_execution_threads()

    #for ENHANCER tumblers:
    for enhancer_key in ('face_enhancer', 'face_enhancer_gpen256', 'face_enhancer_gpen512'):
        modules.globals.fp_ui[enhancer_key] = enhancer_key in args.frame_processor

    # translate deprecated args
    if args.source_path_deprecated:
        print('\033[33mArgument -f and --face are deprecated. Use -s and --source instead.\033[0m')
        modules.globals.source_path = args.source_path_deprecated
        modules.globals.output_path = normalize_output_path(args.source_path_deprecated, modules.globals.target_path, args.output_path)
    if args.cpu_cores_deprecated:
        print('\033[33mArgument --cpu-cores is deprecated. Use --execution-threads instead.\033[0m')
        modules.globals.execution_threads = args.cpu_cores_deprecated
    if args.gpu_vendor_deprecated == 'apple':
        print('\033[33mArgument --gpu-vendor apple is deprecated. Use --execution-provider coreml instead.\033[0m')
        modules.globals.execution_providers = decode_execution_providers(['coreml'])
    if args.gpu_vendor_deprecated == 'nvidia':
        print('\033[33mArgument --gpu-vendor nvidia is deprecated. Use --execution-provider cuda instead.\033[0m')
        modules.globals.execution_providers = decode_execution_providers(['cuda'])
    if args.gpu_vendor_deprecated == 'amd':
        print('\033[33mArgument --gpu-vendor amd is deprecated. Use --execution-provider cuda instead.\033[0m')
        modules.globals.execution_providers = decode_execution_providers(['rocm'])
    if args.gpu_threads_deprecated:
        print('\033[33mArgument --gpu-threads is deprecated. Use --execution-threads instead.\033[0m')
        modules.globals.execution_threads = args.gpu_threads_deprecated


def encode_execution_providers(execution_providers: List[str]) -> List[str]:
    return [execution_provider.replace('ExecutionProvider', '').lower() for execution_provider in execution_providers]


def decode_execution_providers(execution_providers: List[str]) -> List[str]:
    return [provider for provider, encoded_execution_provider in zip(onnxruntime.get_available_providers(), encode_execution_providers(onnxruntime.get_available_providers()))
            if any(execution_provider in encoded_execution_provider for execution_provider in execution_providers)]


def suggest_max_memory() -> int:
    if platform.system().lower() == 'darwin':
        return 4
    return 16


def suggest_default_execution_provider() -> str:
    """Pick the best available provider: cuda > rocm > coreml > openvino > dml > cpu."""
    available = encode_execution_providers(onnxruntime.get_available_providers())
    for pref in ('cuda', 'rocm', 'coreml', 'openvino', 'dml'):
        if pref in available:
            return pref
    return 'cpu'


def suggest_execution_providers() -> List[str]:
    return encode_execution_providers(onnxruntime.get_available_providers())


def suggest_execution_threads() -> int:
    """Suggest optimal thread count based on hardware and execution provider."""
    import os
    
    # Get CPU count
    cpu_count = os.cpu_count() or 4
    
    if 'DmlExecutionProvider' in modules.globals.execution_providers:
        return 1
    if 'ROCMExecutionProvider' in modules.globals.execution_providers:
        return 1
    if 'CUDAExecutionProvider' in modules.globals.execution_providers:
        return 2
    if 'OpenVINOExecutionProvider' in modules.globals.execution_providers:
        return 1
    
    # For CPU execution, use most cores but leave some for system
    return max(4, min(cpu_count - 2, 16))


def limit_resources() -> None:
    # prevent tensorflow memory leak
    if HAS_TENSORFLOW:
        gpus = tensorflow.config.experimental.list_physical_devices('GPU')
        for gpu in gpus:
            tensorflow.config.experimental.set_memory_growth(gpu, True)
    # limit memory usage
    if modules.globals.max_memory:
        # setrlimit(RLIMIT_DATA) fails with EINVAL on macOS, crashing on launch.
        # See https://github.com/hacksider/Deep-Live-Cam/issues/1848
        if platform.system().lower() == 'darwin':
            return
        memory = modules.globals.max_memory * 1024 ** 3
        if platform.system().lower() == 'windows':
            import ctypes
            kernel32 = ctypes.windll.kernel32
            kernel32.SetProcessWorkingSetSize(-1, ctypes.c_size_t(memory), ctypes.c_size_t(memory))
        else:
            import resource
            resource.setrlimit(resource.RLIMIT_DATA, (memory, memory))


def release_resources() -> None:
    if 'CUDAExecutionProvider' in modules.globals.execution_providers and HAS_TORCH:
        torch.cuda.empty_cache()


def pre_check() -> bool:
    if sys.version_info < (3, 9):
        update_status('Python version is not supported - please upgrade to 3.9 or higher.')
        return False
    if not shutil.which('ffmpeg'):
        update_status('ffmpeg is not installed.')
        return False
    return True


def update_status(message: str, scope: str = 'DLC.CORE') -> None:
    print(f'[{scope}] {message}')
    if not modules.globals.headless:
        ui.update_status(message)

def _frame_path_index(frame_path: str) -> int:
    try:
        return int(os.path.splitext(os.path.basename(frame_path))[0])
    except ValueError:
        return 0


def _abort_timed_export(message: str, target_path: str = None, output_created: bool = False) -> None:
    """Report why the timing-preserving export stopped, without leaving an output."""
    update_status(message)
    ui.show_error(message)
    if target_path:
        clean_temp(target_path)
    if output_created and modules.globals.output_path and os.path.isfile(modules.globals.output_path):
        try:
            os.remove(modules.globals.output_path)
        except OSError as error:
            print(f"[DLC.CORE] Could not remove incomplete output: {error}")


def process_video_with_original_timing(start_time: float) -> None:
    """Export a single-video-track MP4 keeping every frame's original timing.

    Unlike the constant-frame-rate pipelines, each processed frame is placed at
    the presentation time recorded in the source, and the last frame keeps its
    original hold time. Any uncertainty about the source timing, or a mismatch
    between timing data and processed frames, aborts the export - no output file
    is produced.
    """
    target_path = modules.globals.target_path

    if modules.globals.map_faces:
        _abort_timed_export('Preserve timing is unavailable when "Map faces" is enabled.')
        return
    if os.path.splitext(target_path)[1].lower() != '.mp4':
        _abort_timed_export('Preserve timing only supports single-video-track MP4 files.')
        return

    update_status('Reading original frame timing...')
    timing, reason = probe_video_timing(target_path)
    if timing is None:
        _abort_timed_export(f'Cannot preserve the original timing: {reason} Export stopped, no output was created.')
        return

    update_status(
        f'Original timing: {len(timing.durations)} frames over {timing.duration:.3f}s '
        f'(variable frame rate).'
    )

    create_temp(target_path)
    update_status('Extracting frames...')
    extract_frames(target_path)

    temp_frame_paths = sorted(get_temp_frame_paths(target_path), key=_frame_path_index)
    if len(temp_frame_paths) != len(timing.durations):
        _abort_timed_export(
            f'Processed frame count ({len(temp_frame_paths)}) does not line up with the '
            f'original timing ({len(timing.durations)} frames). Export stopped, no output was created.',
            target_path,
        )
        return

    total_frames = len(temp_frame_paths)
    update_status(f'Processing {total_frames} frames with {modules.globals.execution_threads} threads...')
    for frame_processor in get_frame_processors_modules(modules.globals.frame_processors):
        update_status('Progressing...', frame_processor.NAME)
        frame_processor.process_video(modules.globals.source_path, temp_frame_paths)
        release_resources()

    update_status('Encoding video with the original per-frame timing...')
    if not create_video_vfr(target_path, timing.durations):
        _abort_timed_export(
            'Failed to encode the video with the original timing. Export stopped, no output was created.',
            target_path,
        )
        return

    mismatch = verify_video_timing(
        get_temp_output_path(target_path),
        total_frames,
        timing.duration,
    )
    if mismatch:
        _abort_timed_export(
            f'The encoded timing does not match the source: {mismatch} Export stopped, no output was created.',
            target_path,
        )
        return

    if modules.globals.keep_audio and timing.has_audio:
        update_status('Restoring audio with preserved timing...')
        if not restore_audio_timed(target_path, modules.globals.output_path):
            _abort_timed_export(
                'Failed to mux the original audio in sync. Export stopped, no output was created.',
                target_path,
                output_created=True,
            )
            return
    else:
        if modules.globals.keep_audio:
            update_status('The source has no audio track; exporting without audio.')
        move_temp(target_path, modules.globals.output_path)

    clean_temp(target_path)

    total_time = time.time() - start_time
    if os.path.isfile(modules.globals.output_path):
        update_status(f'Video processing succeeded with original timing! Total time: {total_time:.2f}s')
    else:
        update_status('Processing to video failed!')


def start() -> None:
    """Start processing with performance monitoring."""
    import time
    
    start_time = time.time()
    
    for frame_processor in get_frame_processors_modules(modules.globals.frame_processors):
        if not frame_processor.pre_start():
            return
    update_status('Processing...')
    
    # process image to image
    if has_image_extension(modules.globals.target_path):
        if modules.globals.nsfw_filter and ui.check_and_ignore_nsfw(modules.globals.target_path, destroy):
            return
        try:
            shutil.copy2(modules.globals.target_path, modules.globals.output_path)
        except Exception as e:
            print("Error copying file:", str(e))
        for frame_processor in get_frame_processors_modules(modules.globals.frame_processors):
            update_status('Progressing...', frame_processor.NAME)
            frame_processor.process_image(modules.globals.source_path, modules.globals.output_path, modules.globals.output_path)
            release_resources()
        if is_image(modules.globals.target_path):
            elapsed = time.time() - start_time
            update_status(f'Processing to image succeed! (Time: {elapsed:.2f}s)')
        else:
            update_status('Processing to image failed!')
        return
    
    # process image to videos
    if modules.globals.nsfw_filter and ui.check_and_ignore_nsfw(modules.globals.target_path, destroy):
        return

    # Preserve the source per-frame timing (variable frame rate MP4): use the
    # fidelity-first disk pipeline with explicit per-frame timestamps.
    if modules.globals.keep_timing:
        process_video_with_original_timing(start_time)
        return

    # Detect FPS early (needed by both pipelines)
    if modules.globals.keep_fps:
        update_status('Detecting fps...')
        fps = detect_fps(modules.globals.target_path)
    else:
        fps = 30.0

    video_created = False

    # --- In-memory pipeline (non-map_faces only) ---
    # Reads frames from FFmpeg pipe, processes in memory, encodes directly.
    # Eliminates all per-frame PNG disk I/O for a major speed-up.
    if not modules.globals.map_faces:
        update_status(f'Processing video in-memory at {fps} fps...')
        create_temp(modules.globals.target_path)

        processing_start = time.time()
        video_created = process_video_in_memory(
            modules.globals.source_path,
            modules.globals.target_path,
            fps,
        )
        processing_time = time.time() - processing_start
        release_resources()

        if video_created:
            update_status(f'In-memory processing + encoding completed in {processing_time:.2f}s')

    # --- Disk-based fallback (required for map_faces, or if pipe failed) ---
    if not video_created:
        if not modules.globals.map_faces:
            update_status('Falling back to disk-based processing...')

        extraction_start = time.time()
        create_temp(modules.globals.target_path)
        update_status('Extracting frames...')
        extract_frames(modules.globals.target_path)
        extraction_time = time.time() - extraction_start

        temp_frame_paths = get_temp_frame_paths(modules.globals.target_path)
        total_frames = len(temp_frame_paths)
        update_status(f'Processing {total_frames} frames with {modules.globals.execution_threads} threads...')

        processing_start = time.time()
        for frame_processor in get_frame_processors_modules(modules.globals.frame_processors):
            update_status('Progressing...', frame_processor.NAME)
            frame_processor.process_video(modules.globals.source_path, temp_frame_paths)
            release_resources()
        processing_time = time.time() - processing_start
        fps_processing = total_frames / processing_time if processing_time > 0 else 0
        update_status(f'Frame processing completed in {processing_time:.2f}s ({fps_processing:.2f} fps)')

        encoding_start = time.time()
        update_status(f'Creating video with {fps} fps...')
        video_created = create_video(modules.globals.target_path, fps)
        encoding_time = time.time() - encoding_start
        if video_created:
            update_status(f'Video encoding completed in {encoding_time:.2f}s')

    if not video_created:
        update_status('Video encoding failed. No temporary output video was created.')
        clean_temp(modules.globals.target_path)
        return
    
    # handle audio
    if modules.globals.keep_audio:
        if modules.globals.keep_fps:
            update_status('Restoring audio...')
        else:
            update_status('Restoring audio might cause issues as fps are not kept...')
        restore_audio(modules.globals.target_path, modules.globals.output_path)
    else:
        move_temp(modules.globals.target_path, modules.globals.output_path)
    
    # clean and validate
    clean_temp(modules.globals.target_path)
    
    total_time = time.time() - start_time
    if is_video(modules.globals.target_path) and modules.globals.output_path and os.path.isfile(modules.globals.output_path):
        update_status(f'Video processing succeeded! Total time: {total_time:.2f}s')
    else:
        update_status('Processing to video failed!')


def destroy(to_quit=True) -> None:
    if modules.globals.target_path:
        clean_temp(modules.globals.target_path)
    if to_quit:
        quit()


def run() -> None:
    parse_args()
    if not pre_check():
        return
    for frame_processor in get_frame_processors_modules(modules.globals.frame_processors):
        if not frame_processor.pre_check():
            return
    # Pre-load face analyser in main thread before GUI starts
    #from modules.face_analyser import get_face_analyser
    #get_face_analyser()
    limit_resources()
    if modules.globals.headless:
        start()
    else:
        window = ui.init(start, destroy, modules.globals.lang)
        window.mainloop()