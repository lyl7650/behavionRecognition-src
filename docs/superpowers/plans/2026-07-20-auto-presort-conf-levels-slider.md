# auto_presort 置信度分档 + 审核页滑块 实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 让 `2a` 预标注时按 5 个置信度门槛各切一套段全存进 doc，`2b` 审核页用滑块选择显示/编辑哪一档，`2c` 无需改动。

**Architecture:** 模型逐窗打分（贵操作）只跑一次；对 `CONF_LEVELS=[0.5,0.6,0.7,0.8,0.9]` 每档复用现有纯函数 `merge_windows→split_out_no_animal→trim_segments` 各切一套段，存进 doc 新字段 `segment_levels`；`doc["segments"]` 恒等于 0.9 档，作为"当前审核中"的那套。前端滑块换档 = 把该档段深拷贝载入 `doc.segments`。

**Tech Stack:** Python 3.12（conda env `AR`），stdlib `http.server`，原生 JS（无框架），pytest。

## Global Constraints

- 测试命令：`conda run -n AR python -m pytest <path> -v`（从项目根 `E:\ActionRecognition\behavionRecognition-src`）。
- 本仓库**不是 git 仓库**——"任务完成" = 相关测试全绿，**不做 git commit**。
- `timeline.py` 是纯逻辑，**不得** import torch/cv2/ultralytics。
- 门槛档位单一事实来源：`timeline.CONF_LEVELS = [0.5, 0.6, 0.7, 0.8, 0.9]`，档位 key 为字符串一位小数 `"0.5"`…`"0.9"`。
- `doc["segments"]` 恒等于 `segment_levels["0.9"]`；`2c`/`apply_clips.py` 零改动。
- `.bat` 入口不变，不改。
- 修改代码文件前需用户同意（用户已批准本计划整体，逐任务实现按 subagent-driven 流程审查）。

---

## File Structure

- `auto_presort/timeline.py`（修改）：加 `CONF_LEVELS` 常量、`SCHEMA_VERSION` 升版、`build_video_doc` 加 `segment_levels` 参数。纯逻辑。
- `auto_presort/engine.py`（修改）：`label_video` 循环 5 档产 `segment_levels`，`segments` 取 0.9 档。
- `auto_presort/review_app.html`（修改）：滑块 UI + 换档逻辑 + 旧 doc 兼容。
- `tests/test_autopresort_timeline.py`（修改）：加 doc 含 `segment_levels`、升版断言。
- `tests/test_autopresort_engine.py`（修改）：加 `label_video` 产多档、`segments == 0.9 档` 断言。
- `tests/test_autopresort_review_server.py`（修改）：加 GET 返回 `segment_levels`、save 保留 `segment_levels`。

---

### Task 1: timeline 加档位常量、升 schema 版本、build_video_doc 支持 segment_levels

**Files:**
- Modify: `auto_presort/timeline.py`（`SCHEMA_VERSION` 第 3 行；`build_video_doc` 第 148-159 行）
- Test: `tests/test_autopresort_timeline.py`

**Interfaces:**
- Produces:
  - `timeline.CONF_LEVELS: list[float]` = `[0.5, 0.6, 0.7, 0.8, 0.9]`
  - `timeline.SCHEMA_VERSION: str` = `"autopresort-2"`
  - `timeline.build_video_doc(rel_path, duration, segments, boundary_spans, no_animal_spans, source_id, approved, meta, segment_levels=None)` → doc dict；当 `segment_levels` 非 None 时 doc 含 `"segment_levels"` 键，否则不含该键（向后兼容旧调用）。

- [ ] **Step 1: 写失败测试**

在 `tests/test_autopresort_timeline.py` 末尾追加：

```python
def test_conf_levels_constant():
    tl = importlib.import_module("auto_presort.timeline")
    assert tl.CONF_LEVELS == [0.5, 0.6, 0.7, 0.8, 0.9]


def test_schema_version_bumped_to_v2():
    tl = importlib.import_module("auto_presort.timeline")
    assert tl.SCHEMA_VERSION == "autopresort-2"


def test_build_video_doc_includes_segment_levels_when_given():
    tl = importlib.import_module("auto_presort.timeline")
    segs09 = [tl.make_segment(1.0, 3.0, "eat", 0.95, "model")]
    levels = {"0.5": segs09, "0.6": segs09, "0.7": segs09,
              "0.8": segs09, "0.9": segs09}
    doc = tl.build_video_doc(
        rel_path="v.mp4", duration=4.0, segments=segs09,
        boundary_spans=[], no_animal_spans=[], source_id="s1",
        approved=False, meta={"class_names": ["eat"]},
        segment_levels=levels,
    )
    assert set(doc["segment_levels"].keys()) == {"0.5", "0.6", "0.7", "0.8", "0.9"}
    assert doc["segment_levels"]["0.9"] == segs09


def test_build_video_doc_omits_segment_levels_when_none():
    tl = importlib.import_module("auto_presort.timeline")
    segs = [tl.make_segment(1.0, 3.0, "eat", 0.95, "model")]
    doc = tl.build_video_doc(
        rel_path="v.mp4", duration=4.0, segments=segs,
        boundary_spans=[], no_animal_spans=[], source_id="s1",
        approved=False, meta={"class_names": ["eat"]},
    )
    assert "segment_levels" not in doc
```

- [ ] **Step 2: 运行确认失败**

Run: `conda run -n AR python -m pytest tests/test_autopresort_timeline.py -v -k "conf_levels or bumped or segment_levels"`
Expected: FAIL —`CONF_LEVELS` 不存在 / `SCHEMA_VERSION` 是 `"autopresort-1"` / `build_video_doc` 无 `segment_levels` 参数。

同时注意现有 `test_schema_version_is_exposed`（第 13-15 行）只断言是非空 str，升版后仍通过，无需改。

- [ ] **Step 3: 实现**

改 `auto_presort/timeline.py`：

第 3 行 `SCHEMA_VERSION = "autopresort-1"` 改为：

```python
SCHEMA_VERSION = "autopresort-2"

CONF_LEVELS = [0.5, 0.6, 0.7, 0.8, 0.9]
```

`build_video_doc`（第 148 行起）签名与 body 改为：

```python
def build_video_doc(rel_path, duration, segments, boundary_spans,
                    no_animal_spans, source_id, approved, meta,
                    segment_levels=None):
    doc = {
        "schema_version": SCHEMA_VERSION,
        "rel_path": rel_path,
        "source_id": source_id,
        "duration": float(duration),
        "approved": bool(approved),
        "segments": segments,
        "gaps": compute_gaps(duration, segments, boundary_spans, no_animal_spans),
        "meta": meta,
    }
    if segment_levels is not None:
        doc["segment_levels"] = segment_levels
    return doc
```

- [ ] **Step 4: 运行确认通过**

Run: `conda run -n AR python -m pytest tests/test_autopresort_timeline.py -v`
Expected: PASS（全部，含旧测试）。

---

### Task 2: engine 循环 5 档产 segment_levels，segments 取 0.9 档

**Files:**
- Modify: `auto_presort/engine.py`（`label_video` 第 21-48 行）
- Test: `tests/test_autopresort_engine.py`

**Interfaces:**
- Consumes: `timeline.CONF_LEVELS`、`timeline.merge_windows`、`timeline.split_out_no_animal`、`timeline.trim_segments`、`timeline.build_video_doc(..., segment_levels=...)`（Task 1）。
- Produces: `engine.label_video(...)` 返回的 doc 现在含 `"segment_levels"`（5 档），且 `doc["segments"] == doc["segment_levels"]["0.9"]`。签名不变。

- [ ] **Step 1: 写失败测试**

在 `tests/test_autopresort_engine.py` 末尾追加：

```python
def test_label_video_produces_all_conf_levels():
    eng = importlib.import_module("auto_presort.engine")
    tl = importlib.import_module("auto_presort.timeline")
    track = {"fps": 10.0, "frames": 90, "boxes": [(0, 0, 10, 10)] * 90,
             "flags": [True] * 90, "no_animal_spans": []}
    doc = eng.label_video(
        video_path="fake.mp4", rel_path="fake.mp4", source_id="src1",
        track=track,
        read_window_frames_fn=lambda vp, s, e, fps: ([None], [None]),
        classify_frames_fn=lambda frames, boxes: ("eat", 0.95),
        meta={"class_names": ["eat"]}, params=_params(),
    )
    assert set(doc["segment_levels"].keys()) == {"0.5", "0.6", "0.7", "0.8", "0.9"}
    # segments 恒等于 0.9 档
    assert doc["segments"] == doc["segment_levels"]["0.9"]


def test_label_video_lower_threshold_merges_across_dip():
    # 中间一个低分窗口(0.60)在 0.9 档打断成两段、在 0.5 档合成一段。
    eng = importlib.import_module("auto_presort.engine")
    # 逐窗 conf 由 classifier 依 window 序号给出：高-高-高-低-高-高-高
    confs = iter([0.95, 0.93, 0.94, 0.60, 0.94, 0.96, 0.95])

    def classifier(frames, boxes):
        return ("eat", next(confs))

    # 需要 7 个窗口：duration 使 iter_windows 产出 7 窗（window 3s, stride 1s → 起点 0..6，end<=dur）
    # end = start+3 <= dur → 最大 start=6 需 dur>=9；7 窗 start=0..6 → dur=9.0
    track = {"fps": 10.0, "frames": 90, "boxes": [(0, 0, 10, 10)] * 90,
             "flags": [True] * 90, "no_animal_spans": []}
    doc = eng.label_video(
        video_path="fake.mp4", rel_path="fake.mp4", source_id="src1",
        track=track,
        read_window_frames_fn=lambda vp, s, e, fps: ([None], [None]),
        classify_frames_fn=classifier,
        meta={"class_names": ["eat"]}, params=_params(),
    )
    # 0.9 档：0.60 打断 → 两个 run（前3窗、后3窗），各自 enter_run>=2 → 两段
    assert len(doc["segment_levels"]["0.9"]) == 2
    # 0.5 档：0.60 也够格 → 七窗连成一段 → trim 后一段
    assert len(doc["segment_levels"]["0.5"]) == 1
```

> 说明：`_params()`（文件顶部第 4-7 行）已存在，`threshold=0.90`。`iter_windows(9.0, 3.0, 1.0)` 产 7 窗（start 0..6），与 `confs` 的 7 个值一一对应。

- [ ] **Step 2: 运行确认失败**

Run: `conda run -n AR python -m pytest tests/test_autopresort_engine.py -v -k "all_conf_levels or merges_across_dip"`
Expected: FAIL —doc 无 `segment_levels` 键（KeyError）。

- [ ] **Step 3: 实现**

改 `auto_presort/engine.py` 的 `label_video`（第 21-48 行）。把"用单一 threshold 合并"改成"对每档合并"，并保留 0.9 档用于 gaps/boundary：

```python
def label_video(video_path, rel_path, source_id, track,
                read_window_frames_fn, classify_frames_fn, meta, params):
    duration = track["frames"] / track["fps"] if track["fps"] else 0.0
    windows = tl.iter_windows(duration, params["window_sec"], params["stride_sec"])
    no_animal = track["no_animal_spans"]

    preds = []
    for start, end, center in windows:
        if _window_no_animal_ratio(start, end, no_animal) >= params["no_animal_overlap"]:
            preds.append((center, None, 0.0))
            continue
        frames_bgr, boxes = read_window_frames_fn(video_path, start, end, track["fps"])
        label, conf = classify_frames_fn(frames_bgr, boxes)
        preds.append((center, label, conf))

    # 逐档切段：merge → split_out_no_animal → trim，复用现有纯函数。
    segment_levels = {}
    default_trimmed, default_boundary = [], []
    for level in tl.CONF_LEVELS:
        merged = tl.merge_windows(preds, level, params["enter_run"])
        merged = tl.split_out_no_animal(merged, no_animal)
        trimmed, boundary_spans = tl.trim_segments(
            merged, params["trim_margin"], params["min_duration"])
        key = f"{level:.1f}"
        segment_levels[key] = trimmed
        if level == 0.9:
            default_trimmed, default_boundary = trimmed, boundary_spans

    return tl.build_video_doc(
        rel_path=rel_path, duration=duration, segments=default_trimmed,
        boundary_spans=default_boundary, no_animal_spans=no_animal,
        source_id=source_id, approved=False,
        meta={"class_names": meta["class_names"],
              "checkpoint_sha256": meta.get("sha256")},
        segment_levels=segment_levels,
    )
```

> `segments` 与 `gaps`/`boundary` 均取 0.9 档，行为与改动前默认 `threshold=0.90` 完全一致——现有 engine 测试（`test_label_video_produces_trimmed_segment` 等）应保持通过。

- [ ] **Step 4: 运行确认通过**

Run: `conda run -n AR python -m pytest tests/test_autopresort_engine.py -v`
Expected: PASS（含旧测试；旧测试断言 `doc["segments"]` 与 gaps，取 0.9 档后不变）。

- [ ] **Step 5: 跑整个 auto_presort 测试面确认无回归**

Run: `conda run -n AR python -m pytest tests/test_autopresort_timeline.py tests/test_autopresort_engine.py -v`
Expected: PASS（全部）。

---

### Task 3: review_server 保留 segment_levels（读取返回 + save 不丢失）

**Files:**
- Test: `tests/test_autopresort_review_server.py`
- （`review_server.py` / `store.py` 预期**无需改动**——本任务先用测试证明现有实现已正确保留 `segment_levels`；若测试失败再改。）

**Interfaces:**
- Consumes: `store.save_doc` / `store.update_doc` / `/api/doc` GET·POST（现有）。
- Produces: 保证 doc 往返（GET 返回、save 后重载）都保留 `segment_levels`。

- [ ] **Step 1: 写测试**

在 `tests/test_autopresort_review_server.py` 末尾追加：

```python
def _seed_with_levels(batch_dir):
    st = importlib.import_module("auto_presort.store")
    segs09 = [{"start": 2.5, "end": 6.5, "label": "eat",
               "confidence": 0.95, "source": "model"}]
    segs05 = [{"start": 1.5, "end": 7.5, "label": "eat",
               "confidence": 0.60, "source": "model"}]
    st.save_doc(str(batch_dir), {
        "schema_version": "autopresort-2", "rel_path": "vL.mp4",
        "source_id": "lev123", "duration": 10.0, "approved": False,
        "segments": segs09, "gaps": [],
        "segment_levels": {"0.5": segs05, "0.6": segs05, "0.7": segs09,
                           "0.8": segs09, "0.9": segs09},
        "meta": {"class_names": ["eat", "drink"]},
    })


def test_api_doc_returns_segment_levels(tmp_path):
    _seed_with_levels(tmp_path)
    server, base = _start(tmp_path)
    try:
        with urllib.request.urlopen(f"{base}/api/doc?rel=vL.mp4") as r:
            doc = json.loads(r.read())
        assert set(doc["segment_levels"].keys()) == {"0.5", "0.6", "0.7", "0.8", "0.9"}
        assert doc["segment_levels"]["0.5"][0]["confidence"] == 0.60
    finally:
        server.shutdown()


def test_save_preserves_segment_levels(tmp_path):
    _seed_with_levels(tmp_path)
    server, base = _start(tmp_path)
    try:
        # 前端 save 只发 segments（不含 segment_levels）
        body = json.dumps({"rel_path": "vL.mp4", "action": "save", "segments": [
            {"start": 2.0, "end": 5.0, "label": "drink",
             "confidence": 0.7, "source": "human"}]}).encode()
        req = urllib.request.Request(f"{base}/api/doc", data=body,
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req) as r:
            doc = json.loads(r.read())
        # segments 被更新，但 segment_levels 仍在（未被覆盖）
        assert doc["segments"][0]["label"] == "drink"
        assert "segment_levels" in doc
        assert set(doc["segment_levels"].keys()) == {"0.5", "0.6", "0.7", "0.8", "0.9"}
    finally:
        server.shutdown()
```

- [ ] **Step 2: 运行**

Run: `conda run -n AR python -m pytest tests/test_autopresort_review_server.py -v -k "segment_levels"`
Expected: PASS（`store.update_doc` 走 load→改 segments→save，其他字段天然保留；`/api/doc` GET 返回完整 doc）。

- [ ] **Step 3: 若失败才改**

若 `test_save_preserves_segment_levels` FAIL（说明 update_doc 丢了字段），检查 `store.update_doc`（`store.py` 第 53-64 行）确保它是"load 完整 doc → 只改 segments/approved → save 完整 doc"。当前实现已如此，正常应直接通过。**不通过再定位，不预先改。**

- [ ] **Step 4: 跑全部 auto_presort 后端测试**

Run: `conda run -n AR python -m pytest tests/ -v -k "autopresort"`
Expected: PASS（全部）。

---

### Task 4: review_app.html 加置信度滑块 + 换档逻辑 + 旧 doc 兼容

**Files:**
- Modify: `auto_presort/review_app.html`（HTML body 第 160-180 行区域加滑块；JS `loadVideo` 第 300-315 行、`render` 邻近加换档逻辑）

**Interfaces:**
- Consumes: `doc.segment_levels`（Task 2 产出，`/api/doc` 返回）、现有 `doc.segments` / `render()` / `pushUndo` / `snapNow` / `save`。
- Produces: 前端交互（无对外接口）。手动验证。

- [ ] **Step 1: 加滑块 HTML**

在 `auto_presort/review_app.html` 的 `#timeline`（第 163 行）之后、`#legend`（第 164 行）之前插入：

```html
  <div id="conflevel" style="margin-top:12px; display:flex; align-items:center;
       gap:12px; font-family:var(--mono); font-size:12px; color:var(--fg-dim);">
    <span>把握程度 ≥</span>
    <input id="confSlider" type="range" min="0" max="4" step="1" value="4"
           style="flex:1; max-width:320px;">
    <span id="confVal" style="color:var(--accent); font-weight:600; min-width:2.5em;">0.9</span>
    <span id="confSegCount"></span>
  </div>
```

- [ ] **Step 2: 加 JS 常量与状态**

在 `<script>` 顶部状态声明区（第 183 行 `let videos = ...` 附近）加：

```javascript
const CONF_LEVELS = [0.5, 0.6, 0.7, 0.8, 0.9];
let confLevelIdx = 4; // 默认 0.9 档
```

- [ ] **Step 3: 换档函数**

在 `render()` 函数（第 316 行）之前加：

```javascript
function levelKey() { return CONF_LEVELS[confLevelIdx].toFixed(1); }

function hasLevels() {
  return doc && doc.segment_levels &&
         Object.prototype.hasOwnProperty.call(doc.segment_levels, levelKey());
}

function applyLevel() {
  // 换档：把当前 segments 推进 undo 栈，再载入该档段（深拷贝，勿污染参考池）。
  if (!hasLevels()) return;
  pushUndo(snapNow());
  doc.segments = JSON.parse(JSON.stringify(doc.segment_levels[levelKey()]));
  selected = -1;
  render();
}

function syncConfUI() {
  const slider = $("confSlider");
  const enabled = doc && doc.segment_levels;
  slider.disabled = !enabled;
  $("conflevel").style.opacity = enabled ? "1" : "0.4";
  $("confVal").textContent = CONF_LEVELS[confLevelIdx].toFixed(1);
  $("confSegCount").textContent =
    doc ? `显示 ${doc.segments.length} 段` : "";
}
```

- [ ] **Step 4: 绑定滑块事件**

在脚本底部（第 443 行 `window.onresize` 附近）加：

```javascript
$("confSlider").oninput = e => {
  confLevelIdx = parseInt(e.target.value, 10);
  applyLevel();
  syncConfUI();
};
```

- [ ] **Step 5: loadVideo 里初始化滑块**

在 `loadVideo` 函数（第 300-315 行）末尾、`render(); loadList();`（第 314 行）**之前**加一行，让每次载入视频时滑块状态与 UI 同步。把第 314 行：

```javascript
  render(); loadList();
```

改为：

```javascript
  syncConfUI();
  render(); loadList();
```

> 注意：`loadVideo` 里 `doc` 已从 `/api/doc` 载入，`doc.segments` 已是 0.9 档（后端保证）。滑块默认 `confLevelIdx=4`（0.9），与 `doc.segments` 一致，无需在载入时强制 applyLevel。旧 `autopresort-1` doc 无 `segment_levels` → `syncConfUI` 置灰滑块，`doc.segments` 照常显示。

- [ ] **Step 6: render 后刷新段数显示**

`render()` 内部会重画段。为保证换档/编辑后"显示 N 段"及时更新，在 `save`（第 419-428 行）成功后也刷新。把 `save` 函数末尾第 427 行 `render(); loadList();` 改为：

```javascript
  render(); loadList(); syncConfUI();
```

- [ ] **Step 7: 手动验证**

启动审核服务（需要一份已有 `.autopresort/` 且含 `autopresort-2` doc 的目录；若暂无，可先跑一次 `2a` 生成，或用 Task 3 的 seed 思路造一个测试目录）：

```bash
conda run -n AR python run_presort_review.py
```

在浏览器逐项确认：
1. 时间轴上方出现滑块，默认停在 **0.9**，"显示 N 段"与时间轴段数一致。
2. 拖滑块 0.9→0.5：段变多/合并（低门槛把断点连起来）。
3. 拖 0.5→0.9：段变少/拆分。
4. 换档后按 `Ctrl+Z`：撤回到上一档。
5. 在某档删一个段、点批准；停服后跑 `2c`（`run_presort_apply.py`）确认只切该档留下的段。
6. 若有旧 `autopresort-1` doc：滑块置灰、不报错，段照常显示。

> 前端无自动化测试（项目现状 UI 无 e2e）。本步为人工验证，验证不通过则回到对应 step 修复。

- [ ] **Step 8: 全后端测试回归**

Run: `conda run -n AR python -m pytest tests/ -v -k "autopresort"`
Expected: PASS（前端改动不影响后端，此步确认无意外牵连）。

---

## Self-Review

**1. Spec coverage:**
- §3.1 CONF_LEVELS → Task 1 ✓
- §3.2 segment_levels 字段 + schema 升版 → Task 1（结构）+ Task 2（填充）✓
- §3.2 segments 恒 = 0.9 档 → Task 2 ✓
- §4.1 逐窗只跑一次 + 逐档切 → Task 2 ✓
- §4.2 threshold 退化、segments=0.9 档 → Task 2 ✓
- §5.1 滑块 UI → Task 4 Step 1 ✓
- §5.2 换档逻辑（pushUndo + 深拷贝载入 + 不自动 save）→ Task 4 Step 3 ✓
- §5.4 旧 doc 兼容（滑块置灰）→ Task 4 Step 3 `syncConfUI` + Step 5 ✓
- §7.1 store 保留 segment_levels → Task 3 ✓
- §7.2 review_server GET 返回 → Task 3 ✓
- §6 apply_clips 零改动 → 无任务（刻意）✓
- §8.3 前端手动验证 → Task 4 Step 7 ✓

**2. Placeholder scan:** 每个 code step 均有完整代码；Task 3 Step 3 是条件分支（"失败才改"），已明确当前实现应通过、给了定位指引，非占位。无 TBD/TODO。

**3. Type consistency:**
- `segment_levels` key 全程为字符串一位小数（`f"{level:.1f}"` / `.toFixed(1)` / `"0.5"`…）一致 ✓
- `build_video_doc(..., segment_levels=None)` 参数名 Task 1 定义、Task 2 调用一致 ✓
- 前端 `levelKey()` / `hasLevels()` / `applyLevel()` / `syncConfUI()` 命名前后引用一致 ✓
- `CONF_LEVELS` 后端 `[0.5,...,0.9]` list[float]，前端同名同值 ✓
