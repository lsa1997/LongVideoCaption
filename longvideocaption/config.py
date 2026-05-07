from dataclasses import dataclass, field, asdict
from typing import List

from .utils import sanitize_filename


@dataclass
class PipelineConfig:
    api_key: str = ""
    base_url: str = ""
    model_name: str = "gemini-3.1-pro-preview"
    llm_timeout: float = 6000.0
    max_retries: int = 3

    chunk_duration_sec: int = 90
    input_payload_format: str = "image_list"
    frame_extraction_strategy: str = "scenedetect"
    max_frames_per_chunk: int = 360
    scene_detect_threshold: float = 15.0
    target_fps: float = 1.0
    frame_max_width: int = 960
    max_total_pixels: int = 128 * 1024 * 32 * 32
    frame_jpg_quality: int = 95
    pass1_timestamp_mode: str = "second"  # second / millisecond / qwen_millisecond
    llm_temperature: float = 0.2
    llm_max_tokens: int = 32768
    prev_event_overlap_count: int = 1
    max_overlap_duration_sec: float = 30.0
    min_chunk_advance_sec: float = 5.0

    pass2_temperature: float = 0.1
    pass2_max_tokens: int = 32768
    pass2_confidence_threshold: int = 80
    pass2_hard_split_threshold: int = 50

    pass2_review_enable: bool = True
    pass2_review_frame_max_width: int = 0
    pass2_review_jpg_quality: int = 95
    pass2_review_max_frames_per_cluster: int = 3
    pass2_review_temperature: float = 0.1
    pass2_review_max_tokens: int = 32768
    pass2_review_batch_size: int = 0  # 0 或负数 = 一次性全部；>0 = 每批最多 N 个聚类

    pass3_temperature: float = 0.4
    pass3_max_tokens: int = 32768

    stage2_fps: float = 4.0
    stage2_max_frames: int = 1200
    stage2_temperature: float = 0.2
    stage2_max_tokens: int = 16384
    stage2_adjust_start_offset: float = 0.0
    stage2_frame_max_width: int = 960
    stage2_max_total_pixels: int = 128 * 1024 * 32 * 32
    stage2_frame_jpg_quality: int = 95
    stage2_parallel_max_workers: int = 4
    stage2_qwen_parallel_max_workers: int = 32
    stage2_qwen_parallel_visual_token_budget: int = 160 * 1024

    stage3_temperature: float = 0.4
    stage3_max_tokens: int = 65536
    stage3_completeness_max_retries: int = 3

    output_language: str = "zh"  # "zh" / "en"；en 模式下 Stage 3 输出英文，落盘 stage3_polished_en.json

    max_workers: int = 2
    stage1_only: bool = False
    stage2_only: bool = False
    stage3_only: bool = False
    hyper_sig_override: str = ""  # stage2-only/stage3-only 时显式指定前序阶段目录签名

    strict_failure: bool = True
    video_extensions: List[str] = field(
        default_factory=lambda: [".mp4", ".mkv", ".avi", ".mov", ".webm", ".flv"]
    )

    def to_dict(self) -> dict:
        return asdict(self)


def hyper_signature(cfg: PipelineConfig) -> str:
    model = sanitize_filename(cfg.model_name)
    scene = str(cfg.scene_detect_threshold).replace(".", "_")
    return (
        f"{model}"
        f"__chk{cfg.chunk_duration_sec}s"
        f"__{cfg.input_payload_format}"
        f"__mf{cfg.max_frames_per_chunk}"
        f"__sc{scene}"
        f"__fw{cfg.frame_max_width}"
        f"__ts{cfg.pass1_timestamp_mode}"
        f"__ovlp{cfg.prev_event_overlap_count}"
    )
