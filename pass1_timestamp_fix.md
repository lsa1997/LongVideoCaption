# Pass 1 时间戳后处理逻辑

## 总览

Pass 1 的 timestamp 从 LLM 返回到最终落盘，经过 **7 个后处理步骤**，构成一条 "白名单吸附 → chunk 内连续性 → 跨 chunk 合并 → 跨 chunk 连续性 → chunk 内连续性（二次）" 的流水线。所有步骤均在当前 chunk 的 `global_results` append 之前就地修改数据。

## 处理链（主循环中的调用顺序）

```
LLM 返回 chunk_data
│
├─[1] _normalize_qwen_output_timestamps     (仅 qwen_millisecond 模式)
│     把模型输出的秒级/数字时间戳统一转为 [hh:mm:ss.fff]
│
├─[2] _validate_and_snap_event_times
│     逐个 event: snap 到白名单 → swap 起止 → 丢弃过短 → clamp 越界
│
├─[3] _enforce_event_continuity             (第 1 次)
│     chunk 内按 start 排序，修平相邻 event 的 overlap/gap
│
├─[4] _validate_revision_end_time
│     校验 prev_event_revision.end_time，吸附到当前 chunk 白名单
│
├─[5] _apply_prev_event_revision
│     若 need_merge=true，回溯 global_results 找到前段末 event 并原地修订
│
├─[6] _enforce_cross_chunk_continuity
│     回溯 global_results 找到最后一个含 events 的 chunk，修平跨 chunk 的 overlap/gap
│
├─[7] _enforce_event_continuity             (第 2 次)
│     跨 chunk 修平可能影响 chunk 内连续性，再跑一次
│
└─ 写入 global_results + pass1_progress.json
```

---

## 各步骤详解

### [1] `_normalize_qwen_output_timestamps`（仅 qwen_millisecond）

**触发条件**：`cfg.pass1_timestamp_mode == "qwen_millisecond"`

**处理对象**：chunk_data 中所有 `start_time` / `end_time` / `anchor_timestamp` 字段和 `key_frame_times` 列表。

**输入格式**（Qwen 可能输出）：
- `"12.5 seconds"` / `"<12.5 seconds>"` → 纯秒格式
- `"00:02:31.1 seconds"` → 混合格式
- `12.5`（float）→ 裸数字

**输出格式**：统一为 `[hh:mm:ss.fff]`，如 `"00:00:12.500"`

**容错**：
- float/int 直接 `format_timestamp(float(ts))`
- 无法识别的字符串保留原值不变

---

### [2] `_validate_and_snap_event_times`

**触发条件**：whitelist 非空且 events 非空

**白名单来源**：
- scenedetect 模式：整个 chunk 内的镜头切换时间点 + `chunk_start` + `chunk_end`（+ 重叠模式下的 `last_end_str`）
- 非 scenedetect 模式：抽帧时间戳 + `chunk_start`

**处理逻辑**（对每个 event 顺序执行）：

```
a. _snap(start_time) → 吸附到最近白名单点
b. _snap(end_time)   → 吸附到最近白名单点
   若任一格式无效 → 丢弃该 event

c. start > end → swap

d. end - start < 0.5s → 丢弃该 event（保证 stage2 fps4 至少采 2 帧）

e. _clamp_left(start)  → 越出 chunk_start-0.5s 时强制 clamp
f. _clamp_right(end)   → 越出 chunk_end+0.5s 时强制 clamp

g. 检查 key_frame_times 是否在 [start, end] 范围内，越界 WARN
```

**关键参数**：

| 参数 | 值 | 说明 |
|------|----|------|
| 白名单精确匹配容差 | 0.01s | `dist < 0.01` → 视为精确命中 |
| 最小 event 时长 | 0.5s | 低于此值丢弃 |
| 大偏移阈值 | 60s | snap 距离 > 60s 记大偏移 WARN |
| clamp 容差 | ±0.5s | 在此范围内不 clamp |

**_snap 吸附规则**：遍历全体白名单，取绝对距离最近的点。若距离 > 60s，打大偏移日志。

**_clamp_left 规则**：
1. `cur_sec >= chunk_start - 0.5` → 不处理
2. 找第一个 `>= chunk_start - 0.1` 的白名单点 → clamp
3. 找不到 → clamp 到 `chunk_start` 自身

**_clamp_right 规则**：对称，找最后一个 `<= chunk_end + 0.1` 的白名单点。

---

### [3] `_enforce_event_continuity`（第 1 次，chunk 内）

**目的**：修平 chunk 内相邻 event 的 overlap 和 gap，保证 `events[i+1].start == events[i].end`。

**处理流程**：

```
Step 1: 按 start_time 升序排列
Step 2: last_valid_idx = 0（第一个 event 作为锚点）
Step 3: 遍历 i = 1..n-1:
    cur = events[last_valid_idx]  ← 上一个保留的 event
    nxt = events[i]               ← 当前检查的 event

    若 nxt.start < cur.end（overlap）:
        nxt.start = cur.end  ← 吸附
    
    若 nxt.start > cur.end（gap）:
        nxt.start = cur.end  ← 吸附
    
    若 abs(nxt.start - cur.end) ≤ 0.01（adjacent）:
        不处理
    
    若 nxt.start >= nxt.end - 0.5（调整后剩余时长不足）:
        丢弃 nxt, last_valid_idx 不变
    否则:
        last_valid_idx = i
```

**关键设计：`last_valid_idx` 在丢弃时不更新**

```
丢弃前:
  ev[0]: |================|  end=10s   (last_valid)
  ev[1]:      |==|                    (将被丢弃)
  ev[2]:            |=======|  start=13s

丢弃后:
  ev[1] 移除, last_valid 仍是 ev[0]
  下一轮: ev[2].start(13s) vs ev[0].end(10s) → gap 3s → ev[2].start = 10s
```

这保证被丢弃 event 造成的空隙由下一个保留 event 回拉填补。

---

### [4] `_validate_revision_end_time`

**目的**：校验 `prev_event_revision.end_time`，吸附到当前 chunk 白名单。

**处理**：
- 仅当 `need_merge=True` 才处理
- 在 whitelist 中查找精确匹配（容差 0.01s）或最近点
- 原地修改 `revision["end_time"]`

**注意**：此函数修改的是 `chunk_data["prev_event_revision"]` 中的值，此时该字段尚未被 pop。

---

### [5] `_apply_prev_event_revision`

**目的**：将当前 chunk 的 LLM 判定（前一 chunk 的末事件需要延展）原地应用到 `global_results` 中。

**触发条件**：`chunk_data.prev_event_revision` 存在且 `need_merge=True`

**处理流程**：

```
Step 1: pop prev_event_revision 字段（避免污染当前 chunk 产物）
Step 2: 回溯 global_results → 找到最后一个含 events 的 chunk
Step 3: 匹配 revision.start_time 与 前段末 event 的 start_time（容差 0.01s）
Step 4: 冲突检测（revision.end > 当前 chunk events[0].start）:
    - 夹紧后 ≤ 原 end_time → 拒绝修订
    - 否则夹紧至 events[0].start
Step 5: 原地覆盖前段末 event 的全部字段（start_time 保留原值，其他用 revision 的值或回退）
```

**修订内容**：

```python
prev_events[-1] = {
    "start_time":    原 start_time（不变）,
    "end_time":      revision 提供的 end_time（被延长）,
    "step1/2/3":     revision 提供的值（fallback 到原值）,
    "key_frame_times": revision 提供的值（fallback 到原值）,
}
```

**回溯支持**：连续多个 chunk 触发合并时（如 chunk N+1、N+2 均为完全合并），`_apply_prev_event_revision` 会跳过中间的空 event chunk，直接修订最初包含 events 的 chunk 的末 event。每次修订都延长同一个 event 的 `end_time`。

---

### [6] `_enforce_cross_chunk_continuity`

**目的**：修平跨 chunk 边界的 overlap 和 gap。

**处理流程**：

```
Step 1: 回溯 global_results → 找到最后一个含 events 的 chunk → prev_events
Step 2: prev_end = prev_events[-1].end_time
Step 3: 遍历当前 chunk events:
    
    ev.end <= prev_end（完全在范围内）:
        → 丢弃该 event，继续下一个
    
    ev.start < prev_end（overlap）:
        → ev.start = prev_end（吸附当前 event 起点），继续下一个
    
    ev.start > prev_end（gap）:
        → prev_events[-1].end_time = ev.start（延展前段末 event），停止遍历
    
    相邻（容差 0.01s 内）:
        → 不处理，停止遍历
```

**方向选择**：

| 情况 | 操作方向 | 原因 |
|------|---------|------|
| overlap | 当前 event ← 前段 | 当前 event 开始过早，属于模型错误 |
| gap | 前段 → 当前 event | 前段 event 结束不够晚，延展它比回拉当前 event 更尊重 LLM 意图 |

---

### [7] `_enforce_event_continuity`（第 2 次）

与 [3] 完全相同的逻辑。因跨 chunk 修平或合并可能改变当前 chunk events 的状态（start 被吸附、event 被丢弃），再跑一次确保 chunk 内连续性。

---

## 配置参数

| 参数 | 默认值 | 位置 | 说明 |
|------|-------|------|------|
| `prev_event_overlap_count` | `1` | config.py | 前情提要末尾 event 个数 |
| `max_overlap_duration_sec` | `30.0` | config.py | 前情提要最大时长（秒），超出截断 |
| `min_chunk_advance_sec` | `5.0` | config.py | 合并后 chunk_start 最小推进量，防止死循环 |
| 白名单精确匹配容差 | `0.01s` | pass1.py | snap 距离 < 此值视为精确命中 |
| 最小 event 时长 | `0.5s` | pass1.py | 保证 stage2 fps4 至少采 2 帧 |
| chunk 边界 clamp 容差 | `±0.5s` | pass1.py | 在此范围内不 clamp |
| snap 大偏移阈值 | `60s` | pass1.py | 超出此值记大偏移 WARN |

---

## 时间戳流向图

```
LLM 输出（可能为 qwen 秒格式 / 裸数字 / 标准格式）
│
▼ [1] normalize (qwen only)
│   float 12.5 → "00:00:12.500"
│   "12.5 seconds" → "00:00:12.500"
│
▼ [2] snap + validate
│   每个 event 吸附到最近白名单点
│   丢弃 duration < 0.5s 的 event
│   swap start>end, clamp 越界
│
▼ [3] chunk 内连续性
│   修平 overlap/gap, 丢弃调整后过短的 event
│
▼ [4] revision end_time snap
│   prev_event_revision.end → whitelist
│
▼ [5] 跨 chunk 合并
│   修订前段末 event 的 end_time + caption
│
▼ [6] 跨 chunk 连续性
│   overlap: ev.start → prev_end
│   gap:     prev_end → ev.start
│
▼ [7] chunk 内连续性（二次）
│   再次修平，确保跨 chunk 修改后的内部一致性
│
▼ 落盘
  pass1_progress.json
```