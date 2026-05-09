import json
import os
import re
from typing import Tuple

from .config import PipelineConfig
from .llm_client import request_llm_with_retry
from .token_tracker import TokenTracker
from .utils import parse_timestamp_to_seconds_strict


PASS_NAME = "pass3_aggregation"

_ROLE_PATTERN = re.compile(r'\[[^\[\]]+\]')
_TIMESTAMP_LIKE = re.compile(r'^[\d:.\s]+$')
_REQUIRED_CHAPTER_FIELDS = ("title", "chapter_summary", "start_time", "end_time")


def _log(video_tag: str, msg: str) -> None:
    print(f"[{video_tag}] {msg}" if video_tag else msg)


def _extract_event_characters(step3_text: str, name_to_desc: dict) -> list:
    if not step3_text:
        return []
    seen = []
    for name in _ROLE_PATTERN.findall(step3_text):
        inner = name[1:-1]
        if _TIMESTAMP_LIKE.match(inner) or "second" in inner:
            continue
        if name not in seen:
            seen.append(name)
    return [{"name": n, "desc": name_to_desc.get(n, "")} for n in seen]


def _init_confidence(raw_chapter_count: int = 0, event_count: int = 0) -> dict:
    return {
        "event_count": event_count,
        "chapter_count": 0,
        "llm_output_shape": {
            "raw_chapter_count": raw_chapter_count,
            "validated_chapter_count": 0,
            "fallback_used": False,
            "missing_required_field_count": 0,
        },
        "chapter_time_validity": {
            "invalid_start_count": 0,
            "invalid_end_count": 0,
            "invalid_range_count": 0,
        },
        "chapter_boundary_calibration": {
            "snap_count": 0,
            "total_abs_snap_delta_sec": 0.0,
            "max_abs_snap_delta_sec": 0.0,
        },
        "event_mounting": {
            "mounted_event_count": 0,
            "unbound_event_count": 0,
            "event_count_matches_input": False,
        },
        "chapter_distribution": {
            "empty_chapter_count": 0,
            "single_event_chapter_count": 0,
            "dominant_chapter_event_ratio": 0.0,
        },
    }


def _count_missing_required_chapter_fields(chapters: list) -> int:
    count = 0
    for ch in chapters or []:
        if not isinstance(ch, dict):
            count += len(_REQUIRED_CHAPTER_FIELDS)
            continue
        for field in _REQUIRED_CHAPTER_FIELDS:
            if not ch.get(field):
                count += 1
    return count


def _record_snap(confidence: dict, old_sec, new_sec) -> None:
    if confidence is None or old_sec is None or new_sec is None:
        return
    delta = abs(new_sec - old_sec)
    if delta <= 0.01:
        return
    calibration = confidence["chapter_boundary_calibration"]
    calibration["snap_count"] += 1
    calibration["total_abs_snap_delta_sec"] = round(
        calibration["total_abs_snap_delta_sec"] + delta, 3,
    )
    calibration["max_abs_snap_delta_sec"] = round(
        max(calibration["max_abs_snap_delta_sec"], delta), 3,
    )


def _chapter_start_sort_key(chapter: dict) -> float:
    start_sec = parse_timestamp_to_seconds_strict(chapter.get("start_time", ""))
    return start_sec if start_sec is not None else float("inf")

SYS_PROMPT = """你是一个资深的电影剧本统筹。
你将收到三层输入：
  ① 全局角色图鉴（已对齐的人物身份与外貌锚点）
  ② 分段叙事摘要（按 chunk 顺序的高密度脉络总结，跨段记忆锚点）
  ③ 完整底层时间轴（按时间排序的所有事件描述）

你的任务：综合理解这三层信息后，将整片切分成几个符合叙事逻辑的「大章节（Chapters）」，并产出全片总结。

【核心要求】
1. 必须基于三层输入综合理解：video_summary 与 chapter_summary 必须忠实地反映 ② 与 ③ 中实际发生的事件、人物与冲突，严禁脱离输入凭空想象或写通用化套话。
2. 切分逻辑：基于"起因、经过、转折、高潮、结局"等叙事节拍来切分章节，切忌按时间机械平分。
3. 时间边界：新章节的 start_time 必须严格等于它包含的【第一个事件】的 start_time；end_time 必须等于它包含的【最后一个事件】的 end_time。时间格式必须**逐字保留**输入里的字符串。
4. 角色一致性：summary 中提到角色时，请使用图鉴里的【角色名】（带方括号），保持与底层时间轴的指代一致。
5. 文本要求：输出丰富自然的剧情总结，严禁在最终的 summary 文本中遗留时间戳或事件ID。

请严格输出以下 JSON 结构：
{
  "video_summary": "<全片核心剧情深度总结，交代故事的起承转合，必须基于实际输入>",
  "chapters": [
    {
      "chapter_id": "ch_01",
      "title": "<极具概括性和文学性的章节标题>",
      "chapter_summary": "<本章节剧情深度复盘，写清前因后果和人物核心动作>",
      "start_time": "<本章第一个事件的 start_time，逐字照抄>",
      "end_time": "<本章最后一个事件的 end_time，逐字照抄>"
    }
  ]
}"""


def _validate_chapters(chapters: list, all_events: list, video_tag: str, confidence: dict = None) -> list:
    """Pass 3 章节校验：V2 排序 + V3 边界吸附到 event 边界 + V4 相邻首尾相连 + 重新编号。

    - V3 隐含格式归一化（snap 后字符串来自 event 原值，与 pass1 白名单一致）。
    - 首尾章节强制对齐 events[0].start / events[-1].end，避免漏首/漏尾。
    - V4 用区间内最接近中点的 event start 作为切点，无候选时退化为 cur_end / nxt_start。
    - 就地改写 start_time / end_time / chapter_id，返回排序后的新列表。
    """
    if not chapters or not all_events:
        return chapters

    event_starts = [
        (ev.get("start_time", ""), parse_timestamp_to_seconds_strict(ev.get("start_time", "")))
        for ev in all_events
    ]
    event_ends = [
        (ev.get("end_time", ""), parse_timestamp_to_seconds_strict(ev.get("end_time", "")))
        for ev in all_events
    ]
    event_starts = [(raw, sec) for raw, sec in event_starts if sec is not None]
    event_ends = [(raw, sec) for raw, sec in event_ends if sec is not None]
    if not event_starts or not event_ends:
        _log(video_tag, "  ⚠️ [Pass3校验] all_events 中没有可解析时间，跳过章节校验")
        return chapters

    sorted_chapters = sorted(
        chapters,
        key=_chapter_start_sort_key,
    )
    if sorted_chapters != chapters:
        _log(video_tag, "  ⚠️ [Pass3校验] 章节非按时间排序，已重排")

    for idx, ch in enumerate(sorted_chapters):
        s_raw_orig = ch.get("start_time", "")
        e_raw_orig = ch.get("end_time", "")
        s_sec = parse_timestamp_to_seconds_strict(s_raw_orig)
        e_sec = parse_timestamp_to_seconds_strict(e_raw_orig)
        if s_sec is None:
            if confidence is not None:
                confidence["chapter_time_validity"]["invalid_start_count"] += 1
            _log(video_tag, f"  ⚠️ [Pass3校验] ch[{idx}].start_time={s_raw_orig!r} 格式无效，按首个事件边界吸附")
            s_sec = event_starts[0][1]
        if e_sec is None:
            if confidence is not None:
                confidence["chapter_time_validity"]["invalid_end_count"] += 1
            _log(video_tag, f"  ⚠️ [Pass3校验] ch[{idx}].end_time={e_raw_orig!r} 格式无效，按最后事件边界吸附")
            e_sec = event_ends[-1][1]
        if e_sec <= s_sec and confidence is not None:
            confidence["chapter_time_validity"]["invalid_range_count"] += 1

        if idx == 0:
            new_start_raw, new_start_sec = event_starts[0]
        else:
            new_start_raw, new_start_sec = min(event_starts, key=lambda p: abs(p[1] - s_sec))

        if idx == len(sorted_chapters) - 1:
            new_end_raw, new_end_sec = event_ends[-1]
        else:
            new_end_raw, new_end_sec = min(event_ends, key=lambda p: abs(p[1] - e_sec))

        if new_start_raw != s_raw_orig or new_end_raw != e_raw_orig:
            _log(
                video_tag,
                f"  ⚠️ [Pass3校验] ch[{idx}] 边界吸附 [{s_raw_orig},{e_raw_orig}] → "
                f"[{new_start_raw},{new_end_raw}]",
            )
            if new_start_raw != s_raw_orig:
                _record_snap(confidence, parse_timestamp_to_seconds_strict(s_raw_orig), new_start_sec)
            if new_end_raw != e_raw_orig:
                _record_snap(confidence, parse_timestamp_to_seconds_strict(e_raw_orig), new_end_sec)
        ch["start_time"] = new_start_raw
        ch["end_time"] = new_end_raw

    for i in range(len(sorted_chapters) - 1):
        cur = sorted_chapters[i]
        nxt = sorted_chapters[i + 1]
        cur_end_raw = cur.get("end_time", "")
        nxt_start_raw = nxt.get("start_time", "")
        if cur_end_raw == nxt_start_raw:
            continue

        cur_end_sec = parse_timestamp_to_seconds_strict(cur_end_raw)
        nxt_start_sec = parse_timestamp_to_seconds_strict(nxt_start_raw)
        cur_start_sec = parse_timestamp_to_seconds_strict(cur.get("start_time", ""))
        nxt_end_sec = parse_timestamp_to_seconds_strict(nxt.get("end_time", ""))
        if None in (cur_end_sec, nxt_start_sec, cur_start_sec, nxt_end_sec):
            _log(video_tag, f"  ⚠️ [Pass3校验] ch[{i}] 或 ch[{i+1}] 时间格式无效，跳过相邻连续性修正")
            continue

        candidates = [p for p in event_starts if cur_start_sec < p[1] <= nxt_end_sec]
        mid_sec = (cur_end_sec + nxt_start_sec) / 2.0
        if candidates:
            chosen_raw, chosen_sec = min(candidates, key=lambda p: abs(p[1] - mid_sec))
        elif cur_end_sec < nxt_start_sec:
            chosen_raw = nxt_start_raw
            chosen_sec = nxt_start_sec
        else:
            chosen_raw = cur_end_raw
            chosen_sec = cur_end_sec

        if cur_end_sec < nxt_start_sec - 0.01:
            _log(
                video_tag,
                f"  ⚠️ [Pass3校验] ch[{i}].end={cur_end_raw} 与 ch[{i+1}].start={nxt_start_raw} "
                f"间存在 {nxt_start_sec - cur_end_sec:.2f}s gap，切点定为 {chosen_raw}",
            )
        elif cur_end_sec > nxt_start_sec + 0.01:
            _log(
                video_tag,
                f"  ⚠️ [Pass3校验] ch[{i}].end={cur_end_raw} 与 ch[{i+1}].start={nxt_start_raw} "
                f"重叠 {cur_end_sec - nxt_start_sec:.2f}s，切点定为 {chosen_raw}",
            )
        cur["end_time"] = chosen_raw
        nxt["start_time"] = chosen_raw
        _record_snap(confidence, cur_end_sec, chosen_sec)
        _record_snap(confidence, nxt_start_sec, chosen_sec)

    for idx, ch in enumerate(sorted_chapters):
        ch["chapter_id"] = f"ch_{idx+1:02d}"

    valid_chapters = []
    for idx, ch in enumerate(sorted_chapters):
        s_sec = parse_timestamp_to_seconds_strict(ch.get("start_time", ""))
        e_sec = parse_timestamp_to_seconds_strict(ch.get("end_time", ""))
        if s_sec is None or e_sec is None or e_sec <= s_sec:
            _log(
                video_tag,
                f"  ⚠️ [Pass3校验] 删除非法空章节 ch[{idx}] "
                f"({ch.get('start_time', '')} → {ch.get('end_time', '')})",
            )
            continue
        valid_chapters.append(ch)

    if not valid_chapters:
        _log(video_tag, "  ⚠️ [Pass3校验] 所有模型章节均非法，回退为单章节")
        if confidence is not None:
            confidence["llm_output_shape"]["fallback_used"] = True
        valid_chapters = [{
            "chapter_id": "ch_01",
            "title": "完整视频",
            "chapter_summary": "",
            "start_time": event_starts[0][0],
            "end_time": event_ends[-1][0],
        }]

    for idx, ch in enumerate(valid_chapters):
        ch["chapter_id"] = f"ch_{idx+1:02d}"

    if confidence is not None:
        confidence["llm_output_shape"]["validated_chapter_count"] = len(valid_chapters)

    return valid_chapters


def _build_character_bank_text(name_to_desc: dict) -> str:
    if not name_to_desc:
        return "（无角色图鉴）"
    lines = []
    for name, desc in name_to_desc.items():
        clean_desc = (desc or "").strip().replace("\n", " ")
        lines.append(f"- {name}: {clean_desc}" if clean_desc else f"- {name}")
    return "\n".join(lines)


def _build_chunk_summary_text(aligned_results: list) -> str:
    lines = []
    for idx, chunk in enumerate(aligned_results):
        time_range = chunk.get("chunk_time_range", f"chunk_{idx+1}")
        summary = (chunk.get("data", {}).get("chunk_summary", "") or "").strip().replace("\n", " ")
        if summary:
            lines.append(f"第{idx+1}段 {time_range}: {summary}")
        else:
            lines.append(f"第{idx+1}段 {time_range}: （无摘要）")
    return "\n".join(lines) if lines else "（无分段摘要）"


def _dump_debug(run_dir: str, filename: str, content) -> None:
    try:
        os.makedirs(run_dir, exist_ok=True)
        path = os.path.join(run_dir, filename)
        if isinstance(content, str):
            with open(path, 'w', encoding='utf-8') as f:
                f.write(content)
        else:
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(content, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"⚠️ 调试 dump 失败 {filename}: {e}")


def _write_pass3_confidence(run_dir: str, confidence: dict) -> str:
    confidence_path = os.path.join(run_dir, "pass3_confidence.json")
    with open(confidence_path, "w", encoding="utf-8") as f:
        json.dump(confidence, f, ensure_ascii=False, indent=2)
    return confidence_path


def _run_chapter_aggregation(
    cfg: PipelineConfig,
    aligned_results: list,
    name_to_desc: dict,
    run_dir: str,
    client,
    token_tracker: TokenTracker = None,
    video_tag: str = "",
) -> Tuple[dict, list]:
    _log(video_tag, "\n" + "=" * 50)
    _log(video_tag, "🎬 [Pass 3 · 子步骤 A] 宏观章节 (Chapter) 逻辑切分...")
    _log(video_tag, "=" * 50)

    all_events = []
    timeline_lines = []

    for chunk in aligned_results:
        for ev in chunk.get("data", {}).get("events", []):
            start_sec = parse_timestamp_to_seconds_strict(ev.get("start_time", ""))
            end_sec = parse_timestamp_to_seconds_strict(ev.get("end_time", ""))
            if start_sec is None or end_sec is None or end_sec <= start_sec:
                _log(
                    video_tag,
                    f"  ⚠️ [Pass3输入] 丢弃时间非法 event: "
                    f"{ev.get('start_time', '')} → {ev.get('end_time', '')}",
                )
                continue
            all_events.append(ev)

    all_events.sort(
        key=lambda ev: (
            parse_timestamp_to_seconds_strict(ev.get("start_time", "")) or 0.0,
            parse_timestamp_to_seconds_strict(ev.get("end_time", "")) or 0.0,
        )
    )

    for idx, ev in enumerate(all_events, 1):
        start_str = ev.get("start_time", "")
        end_str = ev.get("end_time", "")
        desc = ev.get("step3_synthesized_dense_caption", "")
        timeline_lines.append(f"E{idx:04d} {start_str} - {end_str} : {desc}")

    timeline_text = "\n".join(timeline_lines)
    if not all_events:
        _log(video_tag, "⚠️ Pass 3 未发现合法 event，跳过章节聚合 LLM。")
        return {"video_summary": "未发现可聚合的合法事件", "chapters": []}, []

    bank_text = _build_character_bank_text(name_to_desc)
    chunk_summary_text = _build_chunk_summary_text(aligned_results)

    user_prompt = (
        "【全局角色图鉴】\n"
        f"{bank_text}\n\n"
        "【分段叙事摘要（按 chunk 顺序，跨段记忆锚点）】\n"
        f"{chunk_summary_text}\n\n"
        "【完整底层时间轴】\n"
        f"{timeline_text}\n"
        "请综合以上三层输入进行章节切分与全片总结，输出 JSON。"
    )

    _log(video_tag, f"📥 Pass 3 输入规模: 角色 {len(name_to_desc)} 名 / 分段 {len(aligned_results)} 个 / 事件 {len(all_events)} 条 / prompt 字符数 {len(user_prompt)}")
    _dump_debug(run_dir, "_debug_pass3_input.txt", user_prompt)

    log_tag = f"[{video_tag}] 章节切分" if video_tag else "章节切分"
    chapter_agg_result = request_llm_with_retry(
        client=client, model=cfg.model_name,
        messages=[{"role": "system", "content": SYS_PROMPT}, {"role": "user", "content": user_prompt}],
        max_tokens=cfg.pass3_max_tokens, temperature=cfg.pass3_temperature,
        max_retries=cfg.max_retries, chunk_name=log_tag,
        token_tracker=token_tracker, stage=PASS_NAME,
        force_json=True,
    )
    _dump_debug(run_dir, "_debug_pass3_chapter_response.json", chapter_agg_result)
    return chapter_agg_result, all_events


def _assemble_final(
    video_path: str,
    chapter_agg_result: dict,
    all_events: list,
    name_to_desc: dict,
    video_tag: str = "",
    confidence: dict = None,
) -> dict:
    _log(video_tag, "\n" + "=" * 50)
    _log(video_tag, "🛠️ [Pass 3 · 子步骤 B] Chapter-Event 层级数据物理挂载...")
    _log(video_tag, "=" * 50)

    final_json = {
        "video_path": video_path,
        "video_summary": chapter_agg_result.get("video_summary", "全片剧情总结提取失败"),
        "chapters": [],
    }

    chapters_def = chapter_agg_result.get("chapters", [])
    if not all_events:
        final_json["chapters"] = []
        if confidence is not None:
            confidence["event_mounting"]["event_count_matches_input"] = True
        return final_json

    if not chapters_def:
        _log(video_tag, "❌ 未获取到章节定义，将采用兜底单章节模式。")
        if confidence is not None:
            confidence["llm_output_shape"]["fallback_used"] = True
        chapters_def = [{
            "chapter_id": "ch_01", "title": "完整视频", "chapter_summary": "兜底聚合",
            "start_time": all_events[0]["start_time"] if all_events else "[00:00:00.000]",
            "end_time": all_events[-1]["end_time"] if all_events else "[00:00:00.000]",
        }]

    chapter_objs = []
    ch_ranges = []
    for ch_idx, ch_def in enumerate(chapters_def):
        ch_start_str = ch_def.get("start_time", "")
        ch_end_str = ch_def.get("end_time", "")
        chapter_objs.append({
            "chapter_id": ch_def.get("chapter_id", f"ch_{ch_idx+1:02d}"),
            "title": ch_def.get("title", f"第 {ch_idx+1} 章"),
            "chapter_summary": ch_def.get("chapter_summary", ""),
            "start_time": ch_start_str,
            "end_time": ch_end_str,
            "events": [],
        })
        ch_start_sec = parse_timestamp_to_seconds_strict(ch_start_str)
        ch_end_sec = parse_timestamp_to_seconds_strict(ch_end_str)
        if ch_start_sec is None or ch_end_sec is None:
            _log(
                video_tag,
                f"⚠️ 章节时间非法，将跳过挂载范围: {ch_start_str} → {ch_end_str}",
            )
            ch_start_sec = float("inf")
            ch_end_sec = float("inf")
        if ch_idx == len(chapters_def) - 1:
            ch_end_sec = float("inf")
        ch_ranges.append((ch_start_sec, ch_end_sec))

    unbound = []
    for ev in all_events:
        ev_start_sec = parse_timestamp_to_seconds_strict(ev.get("start_time", ""))
        if ev_start_sec is None:
            unbound.append(ev)
            continue
        target = None
        for ch_idx, (cs, ce) in enumerate(ch_ranges):
            if cs <= ev_start_sec < ce:
                target = ch_idx
                break
        if target is None:
            unbound.append(ev)
            continue
        ch_obj = chapter_objs[target]
        step3_text = ev.get("step3_synthesized_dense_caption", "")
        ch_obj["events"].append({
            "event_id": f"ev_{(target+1):02d}_{len(ch_obj['events'])+1:03d}",
            "start_time": ev.get("start_time", ""),
            "end_time": ev.get("end_time", ""),
            "step1_objective_visual": ev.get("step1_objective_visual", ""),
            "step2_contextual_reasoning": ev.get("step2_contextual_reasoning", ""),
            "step3_synthesized_dense_caption": step3_text,
            "characters_in_event": _extract_event_characters(step3_text, name_to_desc),
            "key_frame_times": ev.get("key_frame_times", []),
        })

    if unbound:
        _log(video_tag, f"⚠️ 警告: 有 {len(unbound)} 个事件越界未挂载，已强制追加至最终章。")
        last_chapter = chapter_objs[-1]
        for ev in unbound:
            step3_text = ev.get("step3_synthesized_dense_caption", "")
            last_chapter["events"].append({
                "event_id": f"ev_fallback_{len(last_chapter['events'])+1:03d}",
                "start_time": ev.get("start_time", ""),
                "end_time": ev.get("end_time", ""),
                "step1_objective_visual": ev.get("step1_objective_visual", ""),
                "step2_contextual_reasoning": ev.get("step2_contextual_reasoning", ""),
                "step3_synthesized_dense_caption": step3_text,
                "characters_in_event": _extract_event_characters(step3_text, name_to_desc),
                "key_frame_times": ev.get("key_frame_times", []),
            })

    final_json["chapters"] = chapter_objs
    if confidence is not None:
        chapter_event_counts = [len(ch.get("events", [])) for ch in chapter_objs]
        mounted_event_count = sum(chapter_event_counts)
        confidence["chapter_count"] = len(chapter_objs)
        confidence["event_mounting"]["mounted_event_count"] = mounted_event_count
        confidence["event_mounting"]["unbound_event_count"] = len(unbound)
        confidence["event_mounting"]["event_count_matches_input"] = mounted_event_count == len(all_events)
        confidence["chapter_distribution"]["empty_chapter_count"] = sum(1 for c in chapter_event_counts if c == 0)
        confidence["chapter_distribution"]["single_event_chapter_count"] = sum(1 for c in chapter_event_counts if c == 1)
        confidence["chapter_distribution"]["dominant_chapter_event_ratio"] = (
            round(max(chapter_event_counts) / len(all_events), 3)
            if all_events and chapter_event_counts else 0.0
        )
    return final_json


def run_pass3(
    cfg: PipelineConfig,
    video_path: str,
    aligned_json_path: str,
    run_dir: str,
    client,
    token_tracker: TokenTracker = None,
    video_tag: str = "",
) -> str:
    os.makedirs(run_dir, exist_ok=True)
    final_output_path = os.path.join(run_dir, "pass3_final.json")

    if os.path.exists(final_output_path):
        _log(video_tag, "\n" + "=" * 50)
        _log(video_tag, "⏭️  Pass 3 终产物已存在，跳过装配。")
        _log(video_tag, "=" * 50)
        return final_output_path

    with open(aligned_json_path, 'r', encoding='utf-8') as f:
        aligned_results = json.load(f)

    bank_path = os.path.join(run_dir, "pass2_global_bank.json")
    name_to_desc = {}
    if os.path.exists(bank_path):
        try:
            with open(bank_path, 'r', encoding='utf-8') as f:
                lite_bank = json.load(f)
            name_to_desc = {entry["角色名"]: entry["外貌特征"] for entry in lite_bank}
            _log(video_tag, f"📖 已加载全局图鉴，收录 {len(name_to_desc)} 名角色，用于 event 级角色映射")
        except Exception as e:
            _log(video_tag, f"⚠️ 全局图鉴加载失败，event 角色字段将仅含 name 而无 desc: {e}")

    chapter_agg_result, all_events = _run_chapter_aggregation(cfg, aligned_results, name_to_desc, run_dir, client, token_tracker, video_tag)
    chapters = chapter_agg_result.get("chapters", [])
    if not isinstance(chapters, list):
        chapters = []
        chapter_agg_result["chapters"] = chapters

    confidence = _init_confidence(raw_chapter_count=len(chapters), event_count=len(all_events))
    confidence["llm_output_shape"]["missing_required_field_count"] = _count_missing_required_chapter_fields(chapters)
    chapters = [ch for ch in chapters if isinstance(ch, dict)]
    chapter_agg_result["chapters"] = chapters

    if chapters:
        chapter_agg_result["chapters"] = _validate_chapters(
            chapters, all_events, video_tag, confidence,
        )
    else:
        confidence["llm_output_shape"]["validated_chapter_count"] = 0

    final_storyboard = _assemble_final(video_path, chapter_agg_result, all_events, name_to_desc, video_tag, confidence)
    if confidence["llm_output_shape"]["validated_chapter_count"] == 0:
        confidence["llm_output_shape"]["validated_chapter_count"] = len(final_storyboard.get("chapters", []))

    with open(final_output_path, 'w', encoding='utf-8') as f:
        json.dump(final_storyboard, f, ensure_ascii=False, indent=2)

    confidence_path = _write_pass3_confidence(run_dir, confidence)
    _log(video_tag, f"📊 Pass 3 置信度报告已保存至: {confidence_path}")
    _log(video_tag, f"\n🎉 大功告成！最终的剧本级结构化数据已保存至: {final_output_path}")
    return final_output_path
