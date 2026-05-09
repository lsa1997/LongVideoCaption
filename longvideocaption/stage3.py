"""Stage 3 — 全局精修与指代统一（Global Polish）。

输入：stage2_refined.json（含每个 event 的 frame_caption）。
处理：将所有 chapters/events 拼成 {chapter_id, chapter_title, events:[{event_id, caption}]} 结构，
     一次性喂给纯文本 LLM，回填到每个 event 的 final_caption 字段。
输出：
- zh 模式（默认）：stage3_polished.json，仅更新 final_caption；未被模型回填的 event 退回 frame_caption。
- en 模式：stage3_polished_en.json，额外把 video_summary / chapter.title / chapter.chapter_summary
  以及 caption 中的括号角色名翻译为英文（拼音音译），并在 stage3_confidence 中记录
  character_name_map；完整性校验失败会按 cfg.stage3_completeness_max_retries 重试整轮调用，
  耗尽后抛错（不退回中文）。
"""

import json
import os

from .config import PipelineConfig
from .llm_client import request_llm_with_retry
from .prompts.stage3_v2 import SYS_PROMPT_STAGE3, build_stage3_user_prompt
from .prompts.stage3_v1_en import SYS_PROMPT_STAGE3_EN, build_stage3_user_prompt_en
from .token_tracker import TokenTracker


STAGE_NAME = "stage3_global_polish"


def _log(video_tag: str, msg: str) -> None:
    print(f"[{video_tag}] {msg}" if video_tag else msg)


def _output_filename(lang: str) -> str:
    return "stage3_polished_en.json" if lang == "en" else "stage3_polished.json"


def _build_input_payload(stage2_data: dict, lang: str) -> dict:
    chapters_payload = []
    for ch in stage2_data.get("chapters", []):
        events_payload = []
        for ev in ch.get("events", []):
            cap = ev.get("frame_caption", "") or ev.get("step3_synthesized_dense_caption", "")
            if not cap:
                continue
            events_payload.append({
                "event_id": ev.get("event_id", ""),
                "caption": cap,
            })
        if not events_payload:
            continue
        chapter_entry = {
            "chapter_id": ch.get("chapter_id", ""),
            "chapter_title": ch.get("title", ""),
            "events": events_payload,
        }
        if lang == "en":
            chapter_entry["chapter_summary"] = ch.get("chapter_summary", "")
        chapters_payload.append(chapter_entry)

    payload = {"chapters": chapters_payload}
    if lang == "en":
        payload["video_summary"] = stage2_data.get("video_summary", "")
    return payload


def _load_character_roster(run_dir: str) -> list:
    bank_path = os.path.join(run_dir, "pass2_global_bank.json")
    if not os.path.exists(bank_path):
        return []
    try:
        with open(bank_path, "r", encoding="utf-8") as f:
            bank = json.load(f)
    except Exception:
        return []
    names: list = []
    seen = set()

    def _add(raw_name: str) -> None:
        if not raw_name:
            return
        bracketed = raw_name if raw_name.startswith("[") else f"[{raw_name}]"
        if bracketed in seen:
            return
        seen.add(bracketed)
        names.append(bracketed)

    if isinstance(bank, list):
        for ent in bank:
            if isinstance(ent, dict):
                _add(ent.get("角色名") or ent.get("final_global_name") or ent.get("name") or "")
            elif isinstance(ent, str):
                _add(ent)
    elif isinstance(bank, dict):
        for key in ("characters", "global_bank", "global_characters"):
            entries = bank.get(key)
            if isinstance(entries, list):
                for ent in entries:
                    if isinstance(ent, dict):
                        _add(ent.get("角色名") or ent.get("final_global_name") or ent.get("name") or "")
                    elif isinstance(ent, str):
                        _add(ent)
    return names


def _event_count(chapters: list) -> int:
    count = 0
    for ch in chapters:
        if not isinstance(ch, dict):
            continue
        events = ch.get("events", [])
        if isinstance(events, list):
            count += len(events)
    return count


def _count_duplicates(values: list) -> list:
    seen = set()
    duplicates = set()
    for value in values:
        if value in seen:
            duplicates.add(value)
        seen.add(value)
    return sorted(duplicates)


def _extract_polished_map(polished_json) -> dict:
    polished_map: dict = {}
    if not isinstance(polished_json, dict):
        return polished_map
    polished_chapters = polished_json.get("chapters", [])
    if not isinstance(polished_chapters, list):
        return polished_map
    for ch in polished_chapters:
        if not isinstance(ch, dict):
            continue
        ch_id = ch.get("chapter_id", "")
        events = ch.get("events", [])
        if not isinstance(events, list):
            continue
        for ev in events:
            if not isinstance(ev, dict):
                continue
            ev_id = ev.get("event_id", "")
            cap = (ev.get("caption", "") or "").strip()
            if ev_id and cap:
                polished_map[(ch_id, ev_id)] = cap
    return polished_map


def _extract_chapter_meta_map(polished_json) -> dict:
    """en 模式：从 LLM 回包提取每个 chapter 的翻译后 title/summary。"""
    out: dict = {}
    if not isinstance(polished_json, dict):
        return out
    chapters = polished_json.get("chapters", [])
    if not isinstance(chapters, list):
        return out
    for ch in chapters:
        if not isinstance(ch, dict):
            continue
        ch_id = ch.get("chapter_id", "")
        if not ch_id:
            continue
        out[ch_id] = {
            "chapter_title": (ch.get("chapter_title") or ch.get("title") or "").strip(),
            "chapter_summary": (ch.get("chapter_summary") or "").strip(),
        }
    return out


def _check_en_completeness(
    input_payload: dict,
    polished_json,
    polished_map: dict,
    chapter_meta_map: dict,
    character_roster: list,
) -> dict:
    """en 模式严格完整性校验，返回 {ok, reasons:[...]}。"""
    reasons: list = []
    if not isinstance(polished_json, dict):
        return {"ok": False, "reasons": ["response_not_object"]}

    if not (polished_json.get("video_summary") or "").strip():
        reasons.append("video_summary_missing")

    char_map = polished_json.get("character_name_map", {})
    if character_roster:
        if not isinstance(char_map, dict) or not char_map:
            reasons.append("character_name_map_missing")
        else:
            missing_chars = [n for n in character_roster if n not in char_map]
            if missing_chars:
                reasons.append(f"character_name_map_incomplete:{missing_chars[:5]}")

    for ch in input_payload.get("chapters", []):
        ch_id = ch.get("chapter_id", "")
        meta = chapter_meta_map.get(ch_id, {})
        if not meta.get("chapter_title", ""):
            reasons.append(f"chapter_title_missing:{ch_id}")
        if not meta.get("chapter_summary", ""):
            reasons.append(f"chapter_summary_missing:{ch_id}")
        for ev in ch.get("events", []):
            key = (ch_id, ev.get("event_id", ""))
            if key not in polished_map:
                reasons.append(f"event_caption_missing:{key[0]}/{key[1]}")

    return {"ok": not reasons, "reasons": reasons}


def _build_stage3_confidence(input_payload: dict, polished_json, polished_map: dict) -> dict:
    input_chapters = input_payload.get("chapters", [])
    output_chapters = polished_json.get("chapters", []) if isinstance(polished_json, dict) else []
    if not isinstance(input_chapters, list):
        input_chapters = []
    if not isinstance(output_chapters, list):
        output_chapters = []

    input_chapter_count = len(input_chapters)
    output_chapter_count = len(output_chapters)
    input_event_count = _event_count(input_chapters)
    output_event_count = _event_count(output_chapters)

    chapter_mismatches = []
    for idx in range(min(input_chapter_count, output_chapter_count)):
        expected = input_chapters[idx]
        actual = output_chapters[idx]
        expected_title = expected.get("chapter_title", "")
        actual_title = actual.get("chapter_title", actual.get("title", ""))
        # en 模式 chapter_title 会被翻译，title 文本对比无意义；只比对 chapter_id
        if expected.get("chapter_id", "") != actual.get("chapter_id", ""):
            chapter_mismatches.append({
                "index": idx,
                "expected_chapter_id": expected.get("chapter_id", ""),
                "actual_chapter_id": actual.get("chapter_id", ""),
                "expected_title": expected_title,
                "actual_title": actual_title,
            })

    event_mismatches = []
    missing_events = []
    extra_events = []
    empty_caption_events = []
    input_keys = []
    output_keys = []

    for ch_idx, expected_ch in enumerate(input_chapters):
        actual_ch = output_chapters[ch_idx] if ch_idx < output_chapter_count else {}
        expected_events = expected_ch.get("events", [])
        actual_events = actual_ch.get("events", []) if isinstance(actual_ch, dict) else []
        if not isinstance(actual_events, list):
            actual_events = []
        chapter_id = expected_ch.get("chapter_id", "")
        actual_chapter_id = actual_ch.get("chapter_id", "") if isinstance(actual_ch, dict) else ""

        for ev_idx, expected_ev in enumerate(expected_events):
            expected_event_id = expected_ev.get("event_id", "")
            input_keys.append((chapter_id, expected_event_id))
            if ev_idx >= len(actual_events):
                missing_events.append({
                    "chapter_index": ch_idx,
                    "event_index": ev_idx,
                    "chapter_id": chapter_id,
                    "event_id": expected_event_id,
                })
                continue

            actual_ev = actual_events[ev_idx]
            if not isinstance(actual_ev, dict):
                actual_ev = {}
            actual_event_id = actual_ev.get("event_id", "")
            actual_key = (actual_chapter_id, actual_event_id)
            output_keys.append(actual_key)
            if actual_event_id != expected_event_id:
                event_mismatches.append({
                    "chapter_index": ch_idx,
                    "event_index": ev_idx,
                    "chapter_id": chapter_id,
                    "expected_event_id": expected_event_id,
                    "actual_event_id": actual_event_id,
                })
            if not (actual_ev.get("caption", "") or "").strip():
                empty_caption_events.append({
                    "chapter_index": ch_idx,
                    "event_index": ev_idx,
                    "chapter_id": actual_chapter_id,
                    "event_id": actual_event_id,
                })

        for ev_idx in range(len(expected_events), len(actual_events)):
            actual_ev = actual_events[ev_idx]
            if not isinstance(actual_ev, dict):
                actual_ev = {}
            actual_event_id = actual_ev.get("event_id", "")
            output_keys.append((actual_chapter_id, actual_event_id))
            extra_events.append({
                "chapter_index": ch_idx,
                "event_index": ev_idx,
                "chapter_id": actual_chapter_id,
                "event_id": actual_event_id,
            })
            if not (actual_ev.get("caption", "") or "").strip():
                empty_caption_events.append({
                    "chapter_index": ch_idx,
                    "event_index": ev_idx,
                    "chapter_id": actual_chapter_id,
                    "event_id": actual_event_id,
                })

    for ch_idx in range(input_chapter_count, output_chapter_count):
        actual_ch = output_chapters[ch_idx]
        if not isinstance(actual_ch, dict):
            actual_ch = {}
        actual_chapter_id = actual_ch.get("chapter_id", "")
        actual_events = actual_ch.get("events", [])
        if not isinstance(actual_events, list):
            actual_events = []
        for ev_idx, actual_ev in enumerate(actual_events):
            if not isinstance(actual_ev, dict):
                actual_ev = {}
            actual_event_id = actual_ev.get("event_id", "")
            output_keys.append((actual_chapter_id, actual_event_id))
            extra_events.append({
                "chapter_index": ch_idx,
                "event_index": ev_idx,
                "chapter_id": actual_chapter_id,
                "event_id": actual_event_id,
            })
            if not (actual_ev.get("caption", "") or "").strip():
                empty_caption_events.append({
                    "chapter_index": ch_idx,
                    "event_index": ev_idx,
                    "chapter_id": actual_chapter_id,
                    "event_id": actual_event_id,
                })

    chapter_count_matches = input_chapter_count == output_chapter_count
    event_count_matches = input_event_count == output_event_count
    chapter_id_title_matches = chapter_count_matches and not chapter_mismatches
    event_id_matches = event_count_matches and not event_mismatches and not missing_events and not extra_events
    polished_event_count = sum(1 for key in input_keys if key in polished_map)
    fallback_event_count = input_event_count - polished_event_count

    return {
        "ok": chapter_count_matches and event_count_matches and chapter_id_title_matches and event_id_matches and not empty_caption_events,
        "chapter_count_matches": chapter_count_matches,
        "event_count_matches": event_count_matches,
        "chapter_id_title_matches": chapter_id_title_matches,
        "event_id_matches": event_id_matches,
        "input_chapter_count": input_chapter_count,
        "output_chapter_count": output_chapter_count,
        "input_event_count": input_event_count,
        "output_event_count": output_event_count,
        "polished_event_count": polished_event_count,
        "fallback_event_count": fallback_event_count,
        "chapter_mismatches": chapter_mismatches,
        "event_mismatches": event_mismatches,
        "missing_events": missing_events,
        "extra_events": extra_events,
        "empty_caption_events": empty_caption_events,
        "duplicate_input_event_keys": [
            {"chapter_id": chapter_id, "event_id": event_id}
            for chapter_id, event_id in _count_duplicates(input_keys)
        ],
        "duplicate_output_event_keys": [
            {"chapter_id": chapter_id, "event_id": event_id}
            for chapter_id, event_id in _count_duplicates(output_keys)
        ],
    }


def _clone_with_final_caption(
    stage2_data: dict,
    polished_map: dict,
    lang: str,
    polished_json=None,
    strict_no_fallback: bool = False,
) -> dict:
    chapter_meta_map = _extract_chapter_meta_map(polished_json) if lang == "en" else {}
    video_summary = stage2_data.get("video_summary", "")
    if lang == "en" and isinstance(polished_json, dict):
        translated_summary = (polished_json.get("video_summary") or "").strip()
        if translated_summary:
            video_summary = translated_summary

    cloned = {
        "video_path": stage2_data.get("video_path", ""),
        "video_summary": video_summary,
        "chapters": [],
    }
    for ch in stage2_data.get("chapters", []):
        ch_id = ch.get("chapter_id", "")
        meta = chapter_meta_map.get(ch_id, {}) if lang == "en" else {}
        new_ch = {
            "chapter_id": ch_id,
            "title": meta.get("chapter_title") or ch.get("title", ""),
            "chapter_summary": meta.get("chapter_summary") or ch.get("chapter_summary", ""),
            "start_time": ch.get("start_time", ""),
            "end_time": ch.get("end_time", ""),
            "events": [],
        }
        for ev in ch.get("events", []):
            new_ev = dict(ev)
            ev_id = ev.get("event_id", "")
            polished = polished_map.get((ch_id, ev_id), "")
            if polished:
                new_ev["final_caption"] = polished
            elif strict_no_fallback:
                new_ev["final_caption"] = ""
            else:
                new_ev["final_caption"] = (
                    ev.get("frame_caption", "")
                    or ev.get("step3_synthesized_dense_caption", "")
                )
            new_ch["events"].append(new_ev)
        cloned["chapters"].append(new_ch)
    return cloned


def _call_llm(
    cfg: PipelineConfig,
    client,
    payload: dict,
    lang: str,
    character_roster: list,
    log_tag: str,
    token_tracker: TokenTracker,
):
    if lang == "en":
        sys_prompt = SYS_PROMPT_STAGE3_EN
        user_prompt = build_stage3_user_prompt_en(payload, character_roster)
    else:
        sys_prompt = SYS_PROMPT_STAGE3
        user_prompt = build_stage3_user_prompt(payload)

    return request_llm_with_retry(
        client=client, model=cfg.model_name,
        messages=[
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": user_prompt},
        ],
        max_tokens=cfg.stage3_max_tokens,
        temperature=cfg.stage3_temperature,
        max_retries=cfg.max_retries,
        chunk_name=log_tag,
        token_tracker=token_tracker,
        stage=STAGE_NAME,
        force_json=True,
    )


def run_stage3(
    cfg: PipelineConfig,
    stage2_path: str,
    run_dir: str,
    client,
    token_tracker: TokenTracker = None,
    video_tag: str = "",
    read_dir: str = "",
) -> str:
    os.makedirs(run_dir, exist_ok=True)
    lang = (cfg.output_language or "zh").lower()
    if lang not in ("zh", "en"):
        raise ValueError(f"unsupported output_language: {cfg.output_language!r} (expected 'zh' or 'en')")

    out_path = os.path.join(run_dir, _output_filename(lang))

    if os.path.exists(out_path):
        _log(video_tag, "\n" + "=" * 50)
        _log(video_tag, f"⏭️  Stage 3 终产物已存在（lang={lang}），跳过：{os.path.basename(out_path)}")
        _log(video_tag, "=" * 50)
        return out_path

    with open(stage2_path, 'r', encoding='utf-8') as f:
        stage2_data = json.load(f)

    payload = _build_input_payload(stage2_data, lang)
    if not payload["chapters"]:
        _log(video_tag, "⚠️ Stage 3：stage2 结果为空，无可润色 event，跳过 LLM 调用。")
        cloned = _clone_with_final_caption(stage2_data, {}, lang, polished_json=None)
        cloned["stage3_confidence"] = _build_stage3_confidence(payload, {"chapters": []}, {})
        cloned["stage3_confidence"]["skipped"] = True
        cloned["stage3_confidence"]["skip_reason"] = "no_caption_events"
        cloned["stage3_confidence"]["language"] = lang
        with open(out_path, 'w', encoding='utf-8') as f:
            json.dump(cloned, f, ensure_ascii=False, indent=2)
        return out_path

    character_roster = _load_character_roster(read_dir or run_dir) if lang == "en" else []

    _log(video_tag, "\n" + "=" * 50)
    total_events = sum(len(ch['events']) for ch in payload['chapters'])
    _log(video_tag, f"✨ Stage 3 启动（lang={lang}）：全局润色 {total_events} 个 event"
                    + (f"，角色名清单 {len(character_roster)} 项" if lang == "en" else ""))
    _log(video_tag, "=" * 50)

    log_tag = f"[{video_tag}] Stage3 全局润色" if video_tag else "Stage3 全局润色"

    polished_json = None
    polished_map: dict = {}
    en_check = None
    attempts_used = 0
    max_attempts = 1
    if lang == "en":
        max_attempts = max(1, 1 + int(cfg.stage3_completeness_max_retries or 0))

    last_failure_reasons: list = []
    for attempt in range(1, max_attempts + 1):
        attempts_used = attempt
        polished_json = _call_llm(cfg, client, payload, lang, character_roster, log_tag, token_tracker)
        polished_map = _extract_polished_map(polished_json)

        if lang != "en":
            break

        chapter_meta_map = _extract_chapter_meta_map(polished_json)
        en_check = _check_en_completeness(
            payload, polished_json, polished_map, chapter_meta_map, character_roster,
        )
        if en_check["ok"]:
            break

        last_failure_reasons = en_check["reasons"]
        _log(
            video_tag,
            f"⚠️ Stage 3 (en) 完整性校验未通过（第 {attempt}/{max_attempts} 次）："
            f"{last_failure_reasons[:5]}{' ...' if len(last_failure_reasons) > 5 else ''}",
        )

    if lang == "en" and en_check is not None and not en_check["ok"]:
        raise RuntimeError(
            f"Stage 3 (en) 完整性校验在 {attempts_used} 次调用后仍未通过。"
            f"前若干失败原因：{last_failure_reasons[:10]}"
        )

    cloned = _clone_with_final_caption(
        stage2_data,
        polished_map,
        lang,
        polished_json=polished_json,
        strict_no_fallback=(lang == "en"),
    )
    confidence = _build_stage3_confidence(payload, polished_json, polished_map)
    confidence["language"] = lang
    confidence["completeness_attempts"] = attempts_used
    if lang == "en":
        confidence["en_completeness"] = en_check or {"ok": False, "reasons": ["unknown"]}
        char_map = polished_json.get("character_name_map", {}) if isinstance(polished_json, dict) else {}
        if isinstance(char_map, dict):
            confidence["character_name_map"] = char_map
    cloned["stage3_confidence"] = confidence

    total_events = sum(len(ch.get("events", [])) for ch in cloned["chapters"])
    polished_count = sum(
        1
        for ch in cloned["chapters"]
        for ev in ch.get("events", [])
        if (ch.get("chapter_id", ""), ev.get("event_id", "")) in polished_map
    )

    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(cloned, f, ensure_ascii=False, indent=2)

    _log(video_tag, f"\n🎉 Stage 3 完成（lang={lang}）：润色 {polished_count}/{total_events} 个 event "
                    f"({attempts_used} 次调用) → {out_path}")
    if not confidence["ok"]:
        _log(video_tag, "⚠️ Stage 3 校验未完全通过，详情见输出文件的 stage3_confidence 字段。")
    return out_path
