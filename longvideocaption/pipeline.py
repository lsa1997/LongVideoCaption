import json
import os
import time
import traceback
from datetime import datetime

from .config import PipelineConfig, hyper_signature
from .llm_client import build_client
from .pass1 import run_pass1
from .pass2 import run_pass2
from .pass3 import run_pass3
from .stage2_parallel import run_stage2_parallel as run_stage2
from .stage3 import run_stage3
from .token_tracker import TokenTracker, _format_seconds
from .utils import sanitize_filename


def _now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _load_json(path: str) -> dict:
    if os.path.exists(path):
        try:
            with open(path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception:
            pass
    return {}


def _load_meta(path: str) -> dict:
    return _load_json(path)


def _write_meta(path: str, meta: dict) -> None:
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)


def resolve_run_dir(cfg: PipelineConfig, video_path: str, output_root: str) -> str:
    video_basename = os.path.splitext(os.path.basename(video_path))[0]
    return os.path.join(
        output_root,
        sanitize_filename(video_basename),
        hyper_signature(cfg),
    )


def _resolve_previous_stage_dir(
    video_base_dir: str,
    current_run_dir: str,
    filename: str,
    hyper_sig_override: str = "",
) -> str:
    # 1) 显式指定 → 直接定位
    if hyper_sig_override:
        target_dir = os.path.join(video_base_dir, hyper_sig_override)
        target_path = os.path.join(target_dir, filename)
        if not os.path.exists(target_path):
            raise FileNotFoundError(
                f"通过 --hyper-sig={hyper_sig_override} 在 {target_dir} 下未找到 {filename}"
            )
        return target_dir

    # 2) 先试当前 run_dir
    if os.path.exists(os.path.join(current_run_dir, filename)):
        return current_run_dir

    # 3) 扫描 video_base_dir 下所有子目录
    if not os.path.isdir(video_base_dir):
        raise FileNotFoundError(
            f"视频输出目录不存在: {video_base_dir}\n"
            f"请先运行前一阶段，或通过 --hyper-sig 指定目录。"
        )

    candidates = []
    try:
        for entry in os.scandir(video_base_dir):
            if entry.is_dir():
                candidate = os.path.join(entry.path, filename)
                if os.path.exists(candidate):
                    candidates.append(entry.path)
    except OSError:
        pass

    if len(candidates) == 1:
        return candidates[0]
    if len(candidates) > 1:
        dirs = sorted(os.path.basename(d) for d in candidates)
        raise FileNotFoundError(
            f"在 {video_base_dir} 下找到多个包含 {filename} 的目录，"
            f"请通过 --hyper-sig 指定:\n  " + "\n  ".join(dirs)
        )

    raise FileNotFoundError(
        f"在 {video_base_dir} 下未找到 {filename}。"
        f"请先运行前一阶段，或通过 --hyper-sig 指定目录。"
    )


def process_single_video(cfg: PipelineConfig, video_path: str, output_root: str) -> dict:
    if not os.path.isfile(video_path):
        raise FileNotFoundError(f"视频文件不存在: {video_path}")

    video_tag = sanitize_filename(os.path.splitext(os.path.basename(video_path))[0])
    run_dir = resolve_run_dir(cfg, video_path, output_root)

    # stage2-only / stage3-only 时定位前序阶段产物所在目录
    read_dir = run_dir
    if cfg.stage2_only or cfg.stage3_only:
        filename = "stage2_refined.json" if cfg.stage3_only else "pass3_final.json"
        video_base_dir = os.path.join(output_root, video_tag)
        previous_dir = _resolve_previous_stage_dir(
            video_base_dir, run_dir, filename, cfg.hyper_sig_override,
        )
        read_dir = previous_dir
        if previous_dir != run_dir:
            print(
                f"🔍 [{video_tag}] 前序阶段产物位于: {previous_dir}\n"
                f"   当前 hyper_sig: {hyper_signature(cfg)}"
            )

    os.makedirs(run_dir, exist_ok=True)

    meta_path = os.path.join(run_dir, "run_meta.json")
    token_path = os.path.join(run_dir, "token_usage.json")

    meta = _load_meta(meta_path)
    if not meta:
        meta = {
            "video_path": video_path,
            "hyper_sig": hyper_signature(cfg),
            "config": cfg.to_dict(),
            "runs": [],
        }
    run_started_wall = time.time()
    this_run = {
        "started_at": _now_iso(),
        "ended_at": None,
        "duration_seconds": 0.0,
        "duration_hms": "00:00:00",
        "status": "running",
    }
    meta["runs"].append(this_run)
    _write_meta(meta_path, meta)

    tracker = TokenTracker()
    prev_token_dict = _load_json(token_path)
    if prev_token_dict:
        tracker.load_from_dict(prev_token_dict)
        prev_total = float(prev_token_dict.get("total_seconds", 0.0) or 0.0)
        prev_calls = int(prev_token_dict.get("grand_total", {}).get("calls", 0) or 0)
        print(
            f"🔄 [{video_tag}] 检测到既有 token_usage.json，基线累加："
            f"calls={prev_calls}  prev_total={_format_seconds(prev_total)}"
        )
    client = build_client(cfg)

    result = {
        "video_path": video_path,
        "run_dir": run_dir,
        "status": "running",
        "token_usage": None,
        "artifacts": {},
        "error": None,
    }

    try:
        print(f"\n{'#' * 70}\n# [{video_tag}] 处理视频: {video_path}\n# [{video_tag}] 运行目录: {run_dir}\n{'#' * 70}")

        # ── Stage 3 only：跳过 Stage 1 + Stage 2 ──
        if cfg.stage3_only:
            stage2_path = os.path.join(read_dir, "stage2_refined.json")
            if not os.path.exists(stage2_path):
                raise FileNotFoundError(
                    f"stage3_only 需要 stage2_refined.json，但未找到: {stage2_path}"
                )
            result["artifacts"]["stage2_refined"] = stage2_path
            print(f"⏭️  [{video_tag}] stage3_only=True，跳过 Stage 1 / Stage 2。")

            with tracker.stage_timer("stage3_global_polish"):
                stage3_path = run_stage3(
                    cfg, stage2_path, run_dir, client, tracker,
                    video_tag=video_tag, read_dir=read_dir,
                )
            result["artifacts"]["stage3_polished"] = stage3_path

            this_run["status"] = "completed"
            result["status"] = "completed"
            return result

        # ── Stage 2 only：跳过 Stage 1 ──
        if cfg.stage2_only:
            final_path = os.path.join(read_dir, "pass3_final.json")
            if not os.path.exists(final_path):
                raise FileNotFoundError(
                    f"stage2_only 需要 pass3_final.json，但未找到: {final_path}"
                )
            result["artifacts"]["pass3_final"] = final_path
            print(f"⏭️  [{video_tag}] stage2_only=True，跳过 Stage 1。")

            with tracker.stage_timer("stage2_frame_inspection"):
                stage2_path = run_stage2(cfg, video_path, final_path, run_dir, client, tracker, video_tag=video_tag)
            result["artifacts"]["stage2_refined"] = stage2_path

            this_run["status"] = "completed"
            result["status"] = "completed"
            return result

        # ── 完整 Stage 1（默认 / --stage1-only） ──
        with tracker.stage_timer("pass1_perception"):
            pass1_path = run_pass1(cfg, video_path, run_dir, client, tracker, video_tag=video_tag)
        result["artifacts"]["pass1_progress"] = pass1_path

        # pass2 内部分别计时 pass2_alignment 和 pass2_review
        aligned_path, bank_path = run_pass2(cfg, video_path, pass1_path, run_dir, client, tracker, video_tag=video_tag)
        result["artifacts"]["pass2_aligned"] = aligned_path
        result["artifacts"]["pass2_global_bank"] = bank_path

        with tracker.stage_timer("pass3_aggregation"):
            final_path = run_pass3(cfg, video_path, aligned_path, run_dir, client, tracker, video_tag=video_tag)
        result["artifacts"]["pass3_final"] = final_path

        if cfg.stage1_only:
            print(f"⏭️  [{video_tag}] stage1_only=True，已生成 pass3_final.json，跳过 Stage 2 / Stage 3。")
            this_run["status"] = "completed"
            result["status"] = "completed"
            result["artifacts"]["stage1_final"] = final_path
            return result

        with tracker.stage_timer("stage2_frame_inspection"):
            stage2_path = run_stage2(cfg, video_path, final_path, run_dir, client, tracker, video_tag=video_tag)
        result["artifacts"]["stage2_refined"] = stage2_path

        with tracker.stage_timer("stage3_global_polish"):
            stage3_path = run_stage3(cfg, stage2_path, run_dir, client, tracker, video_tag=video_tag)
        result["artifacts"]["stage3_polished"] = stage3_path

        this_run["status"] = "completed"
        result["status"] = "completed"

    except Exception as e:
        err_text = f"{type(e).__name__}: {e}\n{traceback.format_exc()}"
        print(f"\n💥 [{video_tag}] 视频处理失败: {video_path}\n{err_text}")
        this_run["status"] = "failed"
        this_run["error"] = err_text
        result["status"] = "failed"
        result["error"] = err_text

    finally:
        run_elapsed = max(0.0, time.time() - run_started_wall)
        this_run["ended_at"] = _now_iso()
        this_run["duration_seconds"] = round(run_elapsed, 2)
        this_run["duration_hms"] = _format_seconds(run_elapsed)

        token_dict = tracker.to_dict()
        meta["total_seconds"] = token_dict.get("total_seconds", 0.0)
        meta["total_seconds_hms"] = token_dict.get("total_seconds_hms", "00:00:00")

        result["token_usage"] = token_dict
        try:
            with open(token_path, 'w', encoding='utf-8') as f:
                json.dump(token_dict, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"⚠️ [{video_tag}] token_usage.json 写入失败: {e}")
        _write_meta(meta_path, meta)

        grand = token_dict.get("grand_total", {})
        print(
            f"⏱️  [{video_tag}] 本次耗时 {this_run['duration_hms']}  "
            f"累计耗时 {token_dict.get('total_seconds_hms', '00:00:00')}  "
            f"累计 calls={grand.get('calls', 0)}  total_tokens={grand.get('total_tokens', 0)}"
        )
        per_stage = token_dict.get("per_stage", {}) or {}
        for stage_name in (
            "pass1_perception", "pass2_alignment", "pass2_review",
            "pass3_aggregation", "stage2_frame_inspection", "stage3_global_polish",
        ):
            slot = per_stage.get(stage_name)
            if not slot:
                continue
            print(
                f"    └─ {stage_name}: tokens={slot.get('total_tokens', 0)}  "
                f"calls={slot.get('calls', 0)}  time={slot.get('seconds_hms', '00:00:00')}"
            )

    return result
