import base64
import os
import time
from typing import List, Optional, Tuple
import math
import cv2
import numpy as np
from scenedetect import open_video, detect, SceneManager
from scenedetect.detectors import ContentDetector

from .utils import format_timestamp


def detect_scenes(video_path: str, threshold: float) -> List[Tuple[float, float]]:
    """整片跑一次 pyscenedetect，返回 [(s_start_sec, s_end_sec), ...]。"""

    raw = detect(video_path, ContentDetector(threshold=threshold))
    return [(scene[0].get_seconds(), scene[1].get_seconds()) for scene in raw]

Scene = Tuple[float, float]

def detect_scenes_multiscale(
    video_path: str,
    fast_threshold: float = 27.0,
    slow_threshold: float = 22.0,
    min_scene_len_fast: int = 5,
    min_scene_len_slow: int = 20,
    frame_skip_fast: int = 0,
    frame_skip_slow: int = 5,
    merge_distance: float = 0.5,
    min_scene_duration: float = 0.5,
) -> List[Scene]:
    """
    多时间尺度 scene detection:

    pass1:
        小 frame_skip
        检测 hard cut

    pass2:
        大 frame_skip
        检测 fade / dissolve

    最后 merge boundary。

    Returns:
        [(start_sec, end_sec), ...]
    """

    # ---------------------------------------------------
    # 内部函数：跑一次 detector
    # ---------------------------------------------------
    def run_detector(
        frame_skip: int,
        threshold: float,
        min_scene_len: int,
    ) -> List[Scene]:

        video = open_video(video_path)

        manager = SceneManager()
        manager.auto_downscale = True

        manager.add_detector(
            ContentDetector(
                threshold=threshold,
                min_scene_len=min_scene_len,
                weights=ContentDetector.Components(
                    delta_hue=1.0,
                    delta_sat=1.0,
                    delta_lum=1.0,
                    delta_edges=0.0,  # 提速
                ),
            )
        )

        manager.detect_scenes(
            video,
            frame_skip=frame_skip,
        )

        scenes = manager.get_scene_list()

        return [
            (
                s[0].get_seconds(),
                s[1].get_seconds(),
            )
            for s in scenes
        ]

    # ---------------------------------------------------
    # pass1: hard cuts
    # ---------------------------------------------------
    scenes_fast = run_detector(
        frame_skip=frame_skip_fast,
        threshold=fast_threshold,
        min_scene_len=min_scene_len_fast,
    )

    # ---------------------------------------------------
    # pass2: fades / dissolves
    # ---------------------------------------------------
    scenes_slow = run_detector(
        frame_skip=frame_skip_slow,
        threshold=slow_threshold,
        min_scene_len=min_scene_len_slow,
    )

    # ---------------------------------------------------
    # 提取 boundary
    # ---------------------------------------------------
    boundaries = set()

    for scenes in [scenes_fast, scenes_slow]:
        for start, _ in scenes:
            boundaries.add(round(start, 3))

    boundaries = sorted(boundaries)

    # ---------------------------------------------------
    # merge nearby boundaries
    # ---------------------------------------------------
    merged_boundaries = []

    for t in boundaries:

        if not merged_boundaries:
            merged_boundaries.append(t)
            continue

        prev = merged_boundaries[-1]

        # boundary 很近 -> 合并
        if abs(t - prev) <= merge_distance:
            merged_boundaries[-1] = (prev + t) / 2.0
        else:
            merged_boundaries.append(t)

    # ---------------------------------------------------
    # boundary -> scenes
    # ---------------------------------------------------
    scenes_final = []

    for i in range(len(merged_boundaries) - 1):

        start = merged_boundaries[i]
        end = merged_boundaries[i + 1]

        # 过滤超短 scene
        if (end - start) < min_scene_duration:
            continue

        scenes_final.append((start, end))

    return scenes_final


def get_target_timestamps(
    video_path: str,
    chunk_start: float,
    chunk_end: float,
    strategy: str,
    threshold: float,
    max_frames: int,
    log_prefix: str = "",
    precomputed_scenes: Optional[List[Tuple[float, float]]] = None,
    use_multiscale: bool = False,
    multiscale_fast_threshold: float = 27.0,
    multiscale_slow_threshold: float = 22.0,
) -> List[float]:
    print(f"\n{log_prefix}[序列构建] 分析 {format_timestamp(chunk_start)} 到 {format_timestamp(chunk_end)}。策略: {strategy}")
    if strategy == "scenedetect":
        if precomputed_scenes is None:
            if use_multiscale:
                scene_list_raw = detect_scenes_multiscale(
                    video_path,
                    fast_threshold=multiscale_fast_threshold,
                    slow_threshold=multiscale_slow_threshold,
                )
            else:
                scene_list_raw = detect_scenes(video_path, threshold)
        else:
            scene_list_raw = precomputed_scenes

        valid_scenes = []
        for s_start_raw, s_end_raw in scene_list_raw:
            s_start = max(s_start_raw, chunk_start)
            s_end = min(s_end_raw, chunk_end)
            if s_start < s_end:
                valid_scenes.append((s_start, s_end))
        if not valid_scenes:
            valid_scenes = [(chunk_start, chunk_end)]

        total_duration = sum(s_end - s_start for s_start, s_end in valid_scenes) or 1e-5
        timestamps: List[float] = []
        remaining_frames = max_frames
        for i, (s_start, s_end) in enumerate(valid_scenes):
            duration = s_end - s_start
            alloc = (
                remaining_frames
                if i == len(valid_scenes) - 1
                else max(1, int(round((duration / total_duration) * max_frames)))
            )
            alloc = min(alloc, remaining_frames)
            remaining_frames -= alloc
            if alloc == 1:
                timestamps.append(s_start + duration / 2.0)
            elif alloc > 1:
                timestamps.extend(np.linspace(s_start, s_end, alloc, endpoint=False).tolist())
        return sorted(list(set(timestamps)))
    else:
        return np.linspace(chunk_start, chunk_end, max_frames, endpoint=False).tolist()

def smart_resize(
    num_frames: int,
    height: int,
    width: int,
    temporal_factor: int = 2,
    factor: int = 32,
    min_pixels: int = 128 * 128,
    max_pixels: int = 16 * 16 * 2 * 2 * 2 * 6144,
):
    if num_frames < temporal_factor:
        raise ValueError(f"t:{num_frames} must be larger than temporal_factor:{temporal_factor}")
    if height < factor or width < factor:
        raise ValueError(f"height:{height} or width:{width} must be larger than factor:{factor}")
    elif max(height, width) / min(height, width) > 200:
        raise ValueError(
            f"absolute aspect ratio must be smaller than 200, got {max(height, width) / min(height, width)}"
        )
    h_bar = round(height / factor) * factor
    w_bar = round(width / factor) * factor
    t_bar = round(num_frames / temporal_factor) * temporal_factor

    if t_bar * h_bar * w_bar > max_pixels:
        beta = math.sqrt((num_frames * height * width) / max_pixels)
        h_bar = max(factor, math.floor(height / beta / factor) * factor)
        w_bar = max(factor, math.floor(width / beta / factor) * factor)
    elif t_bar * h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (num_frames * height * width))
        h_bar = math.ceil(height * beta / factor) * factor
        w_bar = math.ceil(width * beta / factor) * factor

    return h_bar, w_bar


def get_base64_frames_qwen(
    video_path: str,
    target_timestamps: List[float],
    max_total_pixels: int,
    jpg_quality: int,
    return_visual_tokens: bool = False,
) -> Tuple[List[float], List[str]]:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        if return_visual_tokens:
            return [], [], 0
        return [], []
    orig_fps = cap.get(cv2.CAP_PROP_FPS)
    orig_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    orig_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

    max_pixels_per_frame = 640 * 32 * 32
    actual_frame_count = len(target_timestamps)
    max_total_pixels = min(max_total_pixels, actual_frame_count * max_pixels_per_frame)
    target_h, target_w = smart_resize(actual_frame_count, orig_height, orig_width, max_pixels=max_total_pixels)
    visual_tokens = actual_frame_count * target_h * target_w // 32 // 32

    valid_timestamps: List[float] = []
    base64_frames: List[str] = []
    for t in target_timestamps:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(t * orig_fps))
        ret, frame = cap.read()
        if ret:
            resized = cv2.resize(frame, (target_w, target_h), interpolation=cv2.INTER_CUBIC)
            _, buffer = cv2.imencode('.jpg', resized, [int(cv2.IMWRITE_JPEG_QUALITY), jpg_quality])
            base64_frames.append(base64.b64encode(buffer).decode('utf-8'))
            valid_timestamps.append(t)
    cap.release()
    if return_visual_tokens:
        return valid_timestamps, base64_frames, visual_tokens
    return valid_timestamps, base64_frames


def get_base64_frames(
    video_path: str,
    target_timestamps: List[float],
    max_width: int,
    jpg_quality: int,
) -> Tuple[List[float], List[str]]:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return [], []
    orig_fps = cap.get(cv2.CAP_PROP_FPS)
    orig_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    orig_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

    scale = max_width / orig_width if orig_width > max_width else 1.0
    target_w, target_h = int(orig_width * scale), int(orig_height * scale)

    valid_timestamps: List[float] = []
    base64_frames: List[str] = []
    for t in target_timestamps:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(t * orig_fps))
        ret, frame = cap.read()
        if ret:
            resized = cv2.resize(frame, (target_w, target_h), interpolation=cv2.INTER_AREA)
            _, buffer = cv2.imencode('.jpg', resized, [int(cv2.IMWRITE_JPEG_QUALITY), jpg_quality])
            base64_frames.append(base64.b64encode(buffer).decode('utf-8'))
            valid_timestamps.append(t)
    cap.release()
    return valid_timestamps, base64_frames


def get_raw_chunk_video_base64(
    video_path: str,
    chunk_start: float,
    chunk_end: float,
    max_width: int,
    target_fps: float,
    temp_dir: str = ".",
) -> Tuple[List[float], str]:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return [], ""
    orig_fps = cap.get(cv2.CAP_PROP_FPS)
    orig_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    orig_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

    scale = max_width / orig_width if orig_width > max_width else 1.0
    target_w, target_h = int(orig_width * scale), int(orig_height * scale)

    os.makedirs(temp_dir, exist_ok=True)
    temp_filename = os.path.join(temp_dir, f"temp_chunk_{int(time.time() * 1000)}.mp4")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(temp_filename, fourcc, target_fps, (target_w, target_h))

    start_frame = int(chunk_start * orig_fps)
    end_frame = int(chunk_end * orig_fps)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    frame_interval = max(1, int(round(orig_fps / target_fps)))
    valid_timestamps: List[float] = []
    curr_frame = start_frame

    while curr_frame <= end_frame:
        ret, frame = cap.read()
        if not ret:
            break
        if (curr_frame - start_frame) % frame_interval == 0:
            resized = cv2.resize(frame, (target_w, target_h), interpolation=cv2.INTER_AREA)
            out.write(resized)
            valid_timestamps.append(curr_frame / orig_fps)
        curr_frame += 1

    out.release()
    cap.release()

    video_b64 = ""
    try:
        with open(temp_filename, "rb") as f:
            video_b64 = base64.b64encode(f.read()).decode('utf-8')
    finally:
        if os.path.exists(temp_filename):
            os.remove(temp_filename)

    return valid_timestamps, video_b64


def get_event_frames_base64(
    video_path: str,
    start_sec: float,
    end_sec: float,
    fps: float,
    max_frames: int,
    max_width: int,
    jpg_quality: int,
) -> Tuple[List[float], List[str]]:
    if end_sec <= start_sec:
        return [], []

    duration = end_sec - start_sec
    n_by_fps = max(1, int(round(duration * max(fps, 1e-6))))
    n_frames = min(n_by_fps, max(1, max_frames))

    if n_frames == 1:
        timestamps = [start_sec + duration / 2.0]
    else:
        timestamps = np.linspace(start_sec, end_sec, n_frames, endpoint=False).tolist()

    return get_base64_frames(video_path, timestamps, max_width, jpg_quality)


def get_event_frames_base64_qwen(
    video_path: str,
    start_sec: float,
    end_sec: float,
    fps: float,
    max_frames: int,
    max_total_pixels: int,
    jpg_quality: int,
    return_visual_tokens: bool = False,
) -> Tuple[List[float], List[str]]:
    if end_sec <= start_sec:
        if return_visual_tokens:
            return [], [], 0
        return [], []

    duration = end_sec - start_sec
    n_by_fps = max(1, int(round(duration * max(fps, 1e-6))))
    n_frames = min(n_by_fps, max(1, max_frames))

    if n_frames == 1:
        timestamps = [start_sec + duration / 2.0]
    else:
        timestamps = np.linspace(start_sec, end_sec, n_frames, endpoint=False).tolist()

    return get_base64_frames_qwen(
        video_path,
        timestamps,
        max_total_pixels,
        jpg_quality,
        return_visual_tokens=return_visual_tokens,
    )


def extract_single_frame_base64(video_path: str, timestamp_str: str, max_width=960, jpg_quality: int = 90) -> str:
    if not isinstance(timestamp_str, str):
        return ""
    clean_str = timestamp_str.strip('[] ')
    try:
        parts = clean_str.split(':')
        if len(parts) != 3:
            return ""
        sec_parts = parts[2].split('.')
        seconds = float(int(parts[0]) * 3600 + int(parts[1]) * 60 + int(sec_parts[0]))
        if len(sec_parts) > 1:
            seconds += float(f"0.{sec_parts[1]}")

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return ""

        orig_fps = cap.get(cv2.CAP_PROP_FPS)
        orig_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        orig_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

        if max_width is None or max_width <= 0 or orig_width <= max_width:
            scale = 1.0
        else:
            scale = max_width / orig_width

        cap.set(cv2.CAP_PROP_POS_FRAMES, int(seconds * orig_fps))
        ret, frame = cap.read()
        b64_str = ""
        if ret:
            if scale < 1.0:
                frame = cv2.resize(frame, (int(orig_width * scale), int(orig_height * scale)), interpolation=cv2.INTER_AREA)
            _, buffer = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), int(jpg_quality)])
            b64_str = base64.b64encode(buffer).decode('utf-8')
        cap.release()
        return b64_str
    except Exception as e:
        print(f"提取帧异常 {timestamp_str}: {e}")
        return ""


def get_video_duration(video_path: str) -> float:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"无法打开视频: {video_path}")
    duration = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) / cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    return float(duration)
