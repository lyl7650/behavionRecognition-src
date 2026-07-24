# auto_presort 置信度分档 + 审核页滑块 —— 设计文档

日期：2026-07-20
关联：`docs/superpowers/specs/2026-07-16-auto-presort-model-assisted-cut-label-design.md`

> 目标：让审核页能看到模型在**不同把握程度**下的预标注，并用一个滑块选择显示哪一档。
> 现状痛点：`2a` 只用固定门槛 0.90 产段，审核时只能看到"很有把握"的段，0.5–0.9 之间的内容根本没生成、看不到。

---

## 1. 背景与动机

现在的链路：

- `2a`（`auto_presort_label.py`）用**单一门槛**（默认 0.90）把逐窗打分切成段，存进 `.autopresort/<video>.json`。
- `2b`（审核网页）只能显示这一套已经切死的段。
- `2c`（apply）落地被批准视频的 `doc["segments"]`。

用户希望看到不同 conf 梯度（0.5–0.6 / 0.6–0.7 / … / 0.9–1.0）的标注，并在网页上按梯度筛选。

**关键认知（讨论中澄清）**：切段发生在 `2a`（后端），段的边界在存进 JSON 时就冻结。审核页的滑块无法对"逐秒海床"重新切段，只能对"已冻好的段"做整块显隐。而段的 conf 取的是**段内最低窗口值**（`min(confs)`），所以单纯降门槛会让一个低分窗口把整段 conf 拉低、把原本的高分段合并成一个低分大段——滑块拉回高档时整段消失，不是"高分段浮现"。

**结论**：不在前端实时切段（会与人工编辑冲突、且要把切段逻辑搬去 JS 重写）。改为 **`2a` 预先按多个门槛各切好一套段**，`2b` 滑块只挑显示哪一档。切段逻辑全部留在 Python（已有测试覆盖）。

## 2. 目标与非目标

**目标**
- `2a`：模型逐窗打分只跑一次；用 0.5 / 0.6 / 0.7 / 0.8 / 0.9 五个门槛各合并+裁剪一遍，把五套段都存进 doc。
- `2b`：时间轴上方一个 min 滑块（5 档），选中哪档就载入哪档的段来显示和编辑。默认停在 **0.9**（保持现有观感）。
- `2c`：无需改动——落地照旧切 `doc["segments"]`，即用户在选定档位上审核批准的那套（天然满足"只切看过/保留的段"）。
- 色块内显示 conf（本次会话前置改动已完成，保留）。

**非目标**
- 不在前端实时切段（海平面连续涨落）——改为 5 个离散档位。
- 不改 trim_margin / min_duration / enter_run 等切段参数（只多跑几个门槛）。
- 不改 YOLO / teacher / 预处理 / 蒸馏逻辑。
- 不做换档时的"未保存编辑"确认弹窗（采用简单版：直接换，改动进 undo 栈）。

## 3. 数据结构变更

### 3.1 门槛档位（单一事实来源）

在 `auto_presort/timeline.py` 定义：

```python
CONF_LEVELS = [0.5, 0.6, 0.7, 0.8, 0.9]
```

档位 key 用字符串一位小数：`"0.5"`, `"0.6"`, …, `"0.9"`（JSON 对象 key 必须是字符串）。

### 3.2 doc 新增字段 `segment_levels`

`build_video_doc` 输出新增：

```json
{
  "schema_version": "autopresort-2",
  "segments": [ ...0.9 档那套... ],
  "segment_levels": {
    "0.5": [ ... ],
    "0.6": [ ... ],
    "0.7": [ ... ],
    "0.8": [ ... ],
    "0.9": [ ... ]
  },
  ...其余字段不变（gaps / duration / meta / source_id / approved）...
}
```

**兼容性设计（关键）**：
- `doc["segments"]` **保留**，语义 = "当前正在审核/将被落地的那套段"。`2a` 初始把它设为默认档（0.9）的那套。
- 用户在网页拖滑块换档 → 前端把 `doc["segments"]` 替换成该档的段（`source="model"` 的原样，人工编辑仍写这里）。
- 用户 save/approve → 照旧只写 `doc["segments"]`。`2c` 只读 `doc["segments"]`，**完全不用改**。
- `segment_levels` 是"每档的原始模型建议"，是只读参考池；`segments` 是"人在某档上编辑后的结果"。
- `schema_version` 升到 `"autopresort-2"`。旧的 `"autopresort-1"` doc 没有 `segment_levels` 字段——前端读到时回退：滑块禁用或只显示 `segments`（见 6.3）。

### 3.3 gaps 归因

`gaps` 目前依赖 `boundary_spans` / `no_animal_spans`，由默认档（0.9）计算并存储，语义不变（时间轴上的灰色区）。低门槛档不重算 gaps（gaps 只服务于"这块为什么没段"的提示，用默认档足够；避免 doc 膨胀）。

## 4. `2a` 引擎改动（`engine.py` / `timeline.py` / `auto_presort_label.py`）

### 4.1 逐窗打分与合并解耦

现在 `label_video` 里 `preds`（逐窗 `(center, label, conf)`）已经是"海床"，只是紧接着用单一 `params["threshold"]` 合并。改为：

1. 逐窗打分得到 `preds` —— **不变，只跑一次**（这是唯一的贵操作）。
2. 对每个 `level in CONF_LEVELS`：用 `merge_windows(preds, threshold=level, ...)` → `split_out_no_animal` → `trim_segments`，得到该档的 `trimmed` 段。
3. 默认档（`params["threshold"]`，即 0.9）那套同时作为 `doc["segments"]`，并用它算 `boundary_spans` / `gaps`（保持现状）。
4. `build_video_doc` 接收 `segment_levels` 一并写入。

> 注意：`split_out_no_animal` 和 `trim_segments` 都是纯函数、无副作用，对每档重复调用安全且廉价。

### 4.2 门槛默认值

**决策：`doc["segments"]` 恒等于 `segment_levels["0.9"]`**（0.9 档 = 现有观感），逻辑最简。

`CONF_LEVELS` 固定为 5 档，独立于任何 CLI 参数。`--threshold` CLI 参数**退化为兼容占位**，不再单独驱动切段（切段由 `CONF_LEVELS` 全量覆盖）。gaps/boundary 用 0.9 档计算。未来若要改默认档，只改 `label_video` 里取哪个 key。

### 4.3 signature 变更

`label_video` 内部改动，对外签名不变（仍返回一个 doc）。`build_video_doc` 增加参数 `segment_levels`（带默认 `None` 以兼容 timeline 单测）。

## 5. `2b` 审核页改动（`review_app.html`）

### 5.1 滑块 UI

时间轴上方（`#timeline` 之前或 `#legend` 附近）加一行：

```
把握程度 ≥ [====●====]  0.9    显示 N/M 段
```

- HTML `<input type="range" min="0" max="4" step="1">`，5 个刻度对应 `CONF_LEVELS` 索引。
- 旁边文字显示当前档（`0.5`…`0.9`）和"该档段数"。
- 默认值 = 索引 4（0.9）。

### 5.2 换档逻辑

滑块 `oninput`：
1. `pushUndo(snapNow())` —— 换档前把当前 `doc.segments` 推进 undo 栈（`Ctrl+Z` 可撤回）。
2. `doc.segments = deepCopy(doc.segment_levels[levelKey])` —— 载入该档段（深拷贝，避免污染参考池）。
3. `selected = -1; render();`
4. **不自动 save**——换档只改内存中的显示。用户满意后手动编辑/批准时才走 save（save 会 `approved=False`，符合现有硬性规则）。

> 简单版取舍：若用户在当前档已手动改过段，换档会用该档原始模型段覆盖（改动进了 undo 栈，可 Ctrl+Z）。符合"先滑到满意档，再动手改"的正常流程。

### 5.3 与现有编辑的交互

- 拖边界 / 删段 / 改标签 / 新建人工段：全部照旧作用于 `doc.segments`（当前档载入的那套）。
- save 时 POST `doc.segments` → 服务端只写 `doc["segments"]`，`segment_levels` 原样保留（服务端 update_doc 不动它）。
- 色块文字 `label conf`（已实现）不变。

### 5.4 兼容旧 doc

`loadVideo` 后：若 `doc.segment_levels` 不存在（旧 `autopresort-1`），滑块**禁用并置灰**，只显示 `doc.segments`（现状行为）。避免旧数据报错。

## 6. `2c` 落地（`apply_clips.py`）

**不改动**。`plan_cuts` 照旧只读 `doc["segments"]`。因为 `segments` 恒等于"用户在选定档审核批准的那套"，天然满足用户要求的 B 方案（只切看过/保留的段）。

## 7. store / review_server 改动

### 7.1 store.py

`update_doc(action="save")` 现在只覆盖 `doc["segments"]`。需确认它**不会丢弃** `doc["segment_levels"]`——当前实现是 `load_doc` → 改字段 → `save_doc`，会保留其他字段，**天然安全，无需改动**。仅需保证前端 POST 的 payload 不含 `segment_levels`（不覆盖）。

### 7.2 review_server.py

- `/api/doc` GET 返回完整 doc（已含 `segment_levels`）——无需改。
- `/api/videos` 的 `n_segments` 用 `len(d["segments"])`（默认档段数）——语义可接受，无需改。
- 无新增端点。

## 8. 测试计划

### 8.1 timeline / engine（Python，`conda run -n AR python -m pytest`）

- `test_build_video_doc_includes_segment_levels`：doc 含 `segment_levels`，5 个 key，每个是 list。
- `test_segment_levels_lower_threshold_merges_more`：构造带一个低分窗口（如 0.60）夹在高分窗口间的 `preds`；断言 `"0.9"` 档切成两段、`"0.5"` 档合成一段（验证核心行为）。
- `test_segments_equals_default_level`：`doc["segments"] == doc["segment_levels"]["0.9"]`。
- `test_schema_version_bumped`：`SCHEMA_VERSION == "autopresort-2"`。
- 现有 timeline 单测：`build_video_doc` 新增参数带默认值，旧测试不传 `segment_levels` 仍通过。

### 8.2 review_server（Python）

- `test_api_doc_returns_segment_levels`：seed 一个带 `segment_levels` 的 doc，GET `/api/doc` 返回该字段。
- `test_save_preserves_segment_levels`：POST save（payload 只含 segments）后，重新 load 的 doc 仍有 `segment_levels`（未被覆盖）。

### 8.3 前端（手动验证）

- 起 `2b`，拖滑块 0.9→0.5，观察段变多/合并；0.5→0.9 观察段变少/拆分。
- 换档后 `Ctrl+Z` 能撤回。
- 在某档删段、批准，确认 `2c` 只切该档留下的段。
- 载入旧 `autopresort-1` doc，滑块置灰、不报错。

## 9. 落地顺序（供 writing-plans 细化）

1. `timeline.py`：加 `CONF_LEVELS`，`SCHEMA_VERSION→autopresort-2`，`build_video_doc` 加 `segment_levels` 参数。
2. `engine.py`：`label_video` 循环 5 档产 `segment_levels`，`segments` 取 0.9 档。
3. Python 测试（timeline / engine / review_server），跑绿。
4. `review_app.html`：滑块 UI + 换档逻辑 + 旧 doc 兼容。
5. 手动验证前端全流程。
6. `2a`/`2b`/`2c` `.bat` 无需改（入口不变）。

## 10. 风险与取舍

- **doc 体积**：5 档段全存，doc 变大 ~5×。段是轻量 JSON（每段 5 字段），可接受。
- **换档丢编辑**：简单版取舍，靠 undo 兜底。已与用户确认接受。
- **默认档语义**：`segments` 恒 = 0.9 档，`--threshold` CLI 参数退化为兼容占位。若未来要改默认档，改 `label_video` 里取哪个 key 即可。
