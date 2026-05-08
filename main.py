import argparse
import sys
import warnings

from longvideocaption.config import PipelineConfig
from longvideocaption.runner import run_batch


warnings.filterwarnings("ignore", message="Unverified HTTPS request")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Long Video Caption — 三遍扫描式长视频打标（支持文件夹批量 + 多线程 + 断点续打）"
    )
    p.add_argument("--input", default=None, help="视频文件或文件夹路径")
    p.add_argument("--output", required=True, help="输出根目录")
    p.add_argument("--video-root", default=None, help="视频根目录（与 --video-jsonl 配合使用）")
    p.add_argument("--video-jsonl", default=None, help="JSONL 文件，每行 dict[\"video_path\"] 为相对路径，与 --video-root 拼接为绝对路径")
    p.add_argument("--workers", type=int, default=2, help="并发视频数（默认 2）")
    stage_group = p.add_mutually_exclusive_group()
    stage_group.add_argument(
        "--stage1-only",
        action="store_true",
        help="只运行 Stage 1（Pass1 + Pass2 + Pass3），生成 pass3_final.json 后停止",
    )
    stage_group.add_argument(
        "--stage2-only",
        action="store_true",
        help="跳过 Stage 1，仅运行 Stage 2（帧精修），需 pass3_final.json 已存在",
    )
    stage_group.add_argument(
        "--stage3-only",
        action="store_true",
        help="跳过 Stage 1 和 Stage 2，仅运行 Stage 3（全局润色），需 stage2_refined.json 已存在",
    )

    p.add_argument("--hyper-sig", default=None, help="stage2-only/stage3-only 时显式指定前序阶段的 hyper_signature 目录名")

    p.add_argument("--model", default=None, help="覆盖 model_name")
    p.add_argument("--api-key", default=None, help="覆盖 api_key")
    p.add_argument("--base-url", default=None, help="覆盖 base_url")

    p.add_argument("--chunk", type=int, default=None, help="覆盖 chunk_duration_sec")
    p.add_argument("--payload", choices=["image_list", "video_base64"], default=None, help="覆盖 input_payload_format")
    p.add_argument("--max-frames", type=int, default=None, help="覆盖 max_frames_per_chunk")
    p.add_argument("--scene-thresh", type=float, default=None, help="覆盖 scene_detect_threshold")
    p.add_argument("--multiscale-detect", action="store_true", default=None, help="启用 multiscale 场景检测（fast + slow 两遍检测）")
    p.add_argument("--multiscale-fast-thresh", type=float, default=None, help="覆盖 multiscale_fast_threshold（hard cuts，默认 27.0）")
    p.add_argument("--multiscale-slow-thresh", type=float, default=None, help="覆盖 multiscale_slow_threshold（fades/dissolves，默认 22.0）")
    p.add_argument("--frame-width", type=int, default=None, help="覆盖 frame_max_width")
    p.add_argument("--target-fps", type=float, default=None, help="覆盖 target_fps")
    p.add_argument(
        "--pass1-timestamp-mode",
        choices=["second", "millisecond", "qwen_millisecond"],
        default=None,
        help=(
            "Pass1 给模型的 timestamp 白名单格式：second=hh:mm:ss，"
            "millisecond=hh:mm:ss.fff，qwen_millisecond=x.x seconds 提示词格式、毫秒格式落盘"
        ),
    )

    p.add_argument("--conf-thresh", type=int, default=None, help="覆盖 pass2_confidence_threshold")

    p.add_argument(
        "--output-lang",
        choices=["zh", "en"],
        default=None,
        help="Stage 3 输出语言：zh=中文（默认，落盘 stage3_polished.json）；en=英文（落盘 stage3_polished_en.json，同时翻译 video_summary / chapter_title / chapter_summary 与括号角色名）",
    )
    return p


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    jsonl_mode = bool(args.video_root and args.video_jsonl)
    input_mode = bool(args.input)
    if not input_mode and not jsonl_mode:
        parser.error("必须提供 --input 或 (--video-root + --video-jsonl)")
    if jsonl_mode and not args.video_root:
        parser.error("--video-jsonl 需要配合 --video-root 使用")
    if jsonl_mode and not args.video_jsonl:
        parser.error("--video-root 需要配合 --video-jsonl 使用")

    cfg = PipelineConfig()
    cfg.max_workers = max(1, args.workers)
    cfg.stage1_only = bool(args.stage1_only)
    cfg.stage2_only = bool(args.stage2_only)
    cfg.stage3_only = bool(args.stage3_only)
    if args.hyper_sig is not None:
        cfg.hyper_sig_override = args.hyper_sig

    if args.model is not None:
        cfg.model_name = args.model
    if args.api_key is not None:
        cfg.api_key = args.api_key
    if args.base_url is not None:
        cfg.base_url = args.base_url

    if args.chunk is not None:
        cfg.chunk_duration_sec = args.chunk
    if args.payload is not None:
        cfg.input_payload_format = args.payload
    if args.max_frames is not None:
        cfg.max_frames_per_chunk = args.max_frames
    if args.scene_thresh is not None:
        cfg.scene_detect_threshold = args.scene_thresh
    if args.multiscale_detect is not None:
        cfg.multiscale_detect = args.multiscale_detect
    if args.multiscale_fast_thresh is not None:
        cfg.multiscale_fast_threshold = args.multiscale_fast_thresh
    if args.multiscale_slow_thresh is not None:
        cfg.multiscale_slow_threshold = args.multiscale_slow_thresh
    if args.frame_width is not None:
        cfg.frame_max_width = args.frame_width
    if args.target_fps is not None:
        cfg.target_fps = args.target_fps
    if args.pass1_timestamp_mode is not None:
        cfg.pass1_timestamp_mode = args.pass1_timestamp_mode

    if args.conf_thresh is not None:
        cfg.pass2_confidence_threshold = args.conf_thresh

    if args.output_lang is not None:
        cfg.output_language = args.output_lang

    run_batch(cfg, args.input, args.output,
              video_root=args.video_root, video_jsonl=args.video_jsonl)
    return 0


if __name__ == "__main__":
    sys.exit(main())
