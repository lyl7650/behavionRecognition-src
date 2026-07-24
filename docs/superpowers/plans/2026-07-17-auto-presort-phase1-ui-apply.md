# auto_presort Phase 1（审核 UI + apply 落地 + 泄漏修复）实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在已完成的 Phase 0 引擎之上，交付可用的最小闭环：`label`（真实入口）→ 网页审核（时间线 + 拖拽编辑 + 批准）→ `apply`（ffmpeg 精确切片到 `sorted-<日期>/<类别>/`）→ `run_crop` 泄漏修复（同源 clip 不跨 train/val）。

**Architecture:** JSON 存取层（`store.py`，纯逻辑：任何编辑自动使 approved 失效）+ stdlib `http.server` 审核服务（静态单页 + `/api/videos`、`/api/doc`、带 Range 的 `/video/`）+ apply 规划层（纯逻辑：批准闸门过滤、`<srcid>__<label>__<start>-<end>.mp4` 命名、ffmpeg 精确重编码命令构造）+ 对 `rebalance_val`/`health_report` 的受控小改（按来源 id 分组/检漏，无 `__` 的旧文件回退现行为）。审核前端是单文件 HTML/JS，无法 pytest，用人工浏览器清单验收；所有 Python 逻辑走 TDD。

**Tech Stack:** Python 3.12（conda `AR`）；stdlib（http.server/json/subprocess/hashlib/webbrowser）；ffmpeg（已在 PATH，8.0.1）；测试用 cv2 合成视频。review/apply 不依赖 torch/YOLO（spec：轻量步骤）。

**运行环境：** 测试命令一律 `conda run -n AR python -m pytest <path> -v`（项目根目录）。**不是 git 仓库**：Task 完成 = 对应测试全绿。汉字打印经 `conda run` 会触发 cp1252 编码错误——入口脚本 print 一律用英文（与 .bat 控制台英文约定一致）。

**实现状态（2026-07-17 完成）：** Task 1–9 全部完成，全仓库 136 测试通过。最终代码评审确认 6 条硬性规则全部满足；评审发现的 3 个中等问题已修复并加回归测试：(1) 畸形/越界 Range 返回 416/整文件而非损坏响应；(2) 前端视频 URL 逐段 encodeURIComponent + 列表 textContent 防注入；(3) 未知标签显示醒目红而非与首类同色。低优先级备忘（未修，YAGNI）：同 doc 内完全相同 label+span 的两段输出同名会覆盖；store 扁平化对含 `__` 的原始文件名有理论碰撞。**剩余人工节点**：浏览器验收清单（Task 6 Step 4，7 项）+ 真实 GPU 跑 label。
**2026-07-17 修订**：类别顺序改为直接取 `pipeline_config.SPECIES[species]["class_names"]`（用户确认标签体系按品种固定），sidecar `.meta.json` 降级为可选交叉校验（存在则必须哈希/类别一致），跨品种拿错 checkpoint 由 head 维度检查兜底（3 类 vs 4 类）。`checkpoint_guard.load_sidecar_meta` → `resolve_meta(checkpoint_path, class_names)`；`verify_and_load_teacher` 增加 `class_names` 参数；label/calibrate 入口同步。全仓库 138 测试通过。

**关键约束（spec + 已批准设计，不得违背）：**
- **Review-all 闸门**：apply 只处理 `approved=true` 的视频，跳过必打印清单。
- **任何编辑使 approved 失效**（防"改完忘确认"）：由 store 层强制，不依赖前端自觉。
- **ffmpeg 必须精确重编码**（`-ss` + 重编码，禁止 `-c copy` 关键帧吸附抵消 inward trim）；切完校验输出时长误差。
- **文件名内嵌来源 id**：`<srcid>__<label>__<start>-<end>.mp4`；`rebalance_val` 按 srcid 分组，同源全进同一 split；`health_report` 新增同源跨 split 检测。旧命名（无 `__`）回退现行为，向后兼容。
- **JSON 相对路径**，不写绝对盘符。
- **置信度门/阈值是便利参数非安全保证**——gate 参数化，本阶段不调（用户明确：后期边用边调）。

---

## 文件结构

**新增：**
- `auto_presort/store.py` — `.autopresort/` JSON 存取：列举、加载、保存（save 强制 approved=false / approve 单独动作）、manifest。纯 stdlib。
- `auto_presort/apply_clips.py` — apply 纯逻辑：闸门过滤、输出命名、ffmpeg 命令构造、时长校验判断。纯 stdlib。
- `auto_presort/review_server.py` — stdlib HTTP 服务：静态页 + `/api/videos` + `/api/doc` GET/POST + `/video/<rel>`（206 Range）。
- `auto_presort/review_app.html` — 单文件前端：左视频列表+状态徽标，右播放器+彩色时间线，拖边界/改标签/删除/批准。
- `auto_presort_label.py` — 入口（重）：YOLO+teacher 真实装配，批量产出 JSON。
- `auto_presort_review.py` — 入口（轻）：起服务开浏览器。
- `auto_presort_apply.py` — 入口（轻）：跑 ffmpeg 落地到 `sorted-<日期>/<类别>/`。
- `tests/test_autopresort_store.py`、`tests/test_autopresort_apply_clips.py`、`tests/test_autopresort_review_server.py`、`tests/test_autopresort_apply_ffmpeg.py`
- 修改测试：`tests/test_abnormal_crop_video.py`（追加分组测试）、`tests/test_snapshot_merge.py`（追加同源检漏测试）

**修改：**
- `utils/abnormal_crop_video.py::rebalance_val` — 按来源 id 分组（受控小改）
- `snapshot_merge.py::health_report` — 新增 `source_leak` 字段

---

## Task 1: store.py（JSON 存取 + 编辑失效 approved）

**Files:**
- Create: `auto_presort/store.py`
- Test: `tests/test_autopresort_store.py`

- [ ] **Step 1: 写失败测试**

```python
# tests/test_autopresort_store.py
import importlib
import json


def _mkdoc(rel="v1.mp4", approved=False, segs=None):
    return {
        "schema_version": "autopresort-1", "rel_path": rel, "source_id": "abc123",
        "duration": 10.0, "approved": approved,
        "segments": segs if segs is not None else
            [{"start": 1.0, "end": 4.0, "label": "eat", "confidence": 0.95, "source": "model"}],
        "gaps": [], "meta": {"class_names": ["eat"]},
    }


def test_save_and_load_roundtrip(tmp_path):
    st = importlib.import_module("auto_presort.store")
    st.save_doc(str(tmp_path), _mkdoc())
    doc = st.load_doc(str(tmp_path), "v1.mp4")
    assert doc["rel_path"] == "v1.mp4"
    assert (tmp_path / ".autopresort" / "v1.mp4.json").is_file()


def test_nested_rel_path_is_flattened_safely(tmp_path):
    st = importlib.import_module("auto_presort.store")
    st.save_doc(str(tmp_path), _mkdoc(rel="sub/v2.mp4"))
    doc = st.load_doc(str(tmp_path), "sub/v2.mp4")
    assert doc["rel_path"] == "sub/v2.mp4"
    # 文件名不含路径分隔符
    files = [p.name for p in (tmp_path / ".autopresort").iterdir()]
    assert all("/" not in f and "\\" not in f for f in files)


def test_list_docs_returns_all(tmp_path):
    st = importlib.import_module("auto_presort.store")
    st.save_doc(str(tmp_path), _mkdoc(rel="a.mp4"))
    st.save_doc(str(tmp_path), _mkdoc(rel="b.mp4", approved=True))
    docs = st.list_docs(str(tmp_path))
    assert sorted(d["rel_path"] for d in docs) == ["a.mp4", "b.mp4"]


def test_update_doc_edit_forces_approved_false(tmp_path):
    st = importlib.import_module("auto_presort.store")
    st.save_doc(str(tmp_path), _mkdoc(approved=True))
    new_segs = [{"start": 1.5, "end": 4.0, "label": "eat",
                 "confidence": 0.95, "source": "human"}]
    doc = st.update_doc(str(tmp_path), "v1.mp4", segments=new_segs, action="save")
    assert doc["approved"] is False          # 编辑强制失效
    assert doc["segments"] == new_segs
    on_disk = st.load_doc(str(tmp_path), "v1.mp4")
    assert on_disk["approved"] is False


def test_update_doc_approve_action_sets_true(tmp_path):
    st = importlib.import_module("auto_presort.store")
    st.save_doc(str(tmp_path), _mkdoc(approved=False))
    doc = st.update_doc(str(tmp_path), "v1.mp4", segments=None, action="approve")
    assert doc["approved"] is True
    assert st.load_doc(str(tmp_path), "v1.mp4")["approved"] is True


def test_update_doc_unknown_action_raises(tmp_path):
    st = importlib.import_module("auto_presort.store")
    st.save_doc(str(tmp_path), _mkdoc())
    try:
        st.update_doc(str(tmp_path), "v1.mp4", segments=None, action="nope")
        assert False, "should raise"
    except ValueError:
        pass
```

- [ ] **Step 2: 跑测试确认失败**

Run: `conda run -n AR python -m pytest tests/test_autopresort_store.py -v`
Expected: FAIL（`ModuleNotFoundError: auto_presort.store`）

- [ ] **Step 3: 最小实现**

```python
# auto_presort/store.py
"""`.autopresort/` JSON 存取层。纯 stdlib。

约定：doc 文件名 = rel_path 扁平化（分隔符→"__"）+ ".json"，存于 <batch_dir>/.autopresort/。
update_doc(action="save") 强制 approved=False（任何编辑使批准失效，硬性规则）；
update_doc(action="approve") 仅置 approved=True，不改 segments。
"""
import json
import os

STORE_DIRNAME = ".autopresort"


def _store_dir(batch_dir):
    return os.path.join(batch_dir, STORE_DIRNAME)


def _doc_filename(rel_path):
    flat = rel_path.replace("\\", "/").replace("/", "__")
    return flat + ".json"


def _doc_path(batch_dir, rel_path):
    return os.path.join(_store_dir(batch_dir), _doc_filename(rel_path))


def save_doc(batch_dir, doc):
    os.makedirs(_store_dir(batch_dir), exist_ok=True)
    path = _doc_path(batch_dir, doc["rel_path"])
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(doc, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)
    return path


def load_doc(batch_dir, rel_path):
    with open(_doc_path(batch_dir, rel_path), encoding="utf-8") as f:
        return json.load(f)


def list_docs(batch_dir):
    d = _store_dir(batch_dir)
    if not os.path.isdir(d):
        return []
    docs = []
    for name in sorted(os.listdir(d)):
        if name.endswith(".json") and name != "manifest.json":
            with open(os.path.join(d, name), encoding="utf-8") as f:
                docs.append(json.load(f))
    return docs


def update_doc(batch_dir, rel_path, segments, action):
    doc = load_doc(batch_dir, rel_path)
    if action == "save":
        if segments is not None:
            doc["segments"] = segments
        doc["approved"] = False
    elif action == "approve":
        doc["approved"] = True
    else:
        raise ValueError(f"unknown action: {action}")
    save_doc(batch_dir, doc)
    return doc


def save_manifest(batch_dir, manifest):
    os.makedirs(_store_dir(batch_dir), exist_ok=True)
    path = os.path.join(_store_dir(batch_dir), "manifest.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    return path


def load_manifest(batch_dir):
    path = os.path.join(_store_dir(batch_dir), "manifest.json")
    with open(path, encoding="utf-8") as f:
        return json.load(f)
```

- [ ] **Step 4: 跑测试确认通过**

Run: `conda run -n AR python -m pytest tests/test_autopresort_store.py -v`
Expected: PASS（6 passed）

---

## Task 2: apply_clips.py（闸门过滤 + 命名 + ffmpeg 命令构造）

**Files:**
- Create: `auto_presort/apply_clips.py`
- Test: `tests/test_autopresort_apply_clips.py`

- [ ] **Step 1: 写失败测试**

```python
# tests/test_autopresort_apply_clips.py
import importlib


def _doc(rel, approved, segs, source_id="a1b2c3d4"):
    return {"rel_path": rel, "approved": approved, "source_id": source_id,
            "segments": segs, "duration": 60.0}


def test_plan_cuts_skips_unapproved_and_reports():
    ac = importlib.import_module("auto_presort.apply_clips")
    docs = [
        _doc("v1.mp4", True,  [{"start": 2.5, "end": 6.5, "label": "eat"}]),
        _doc("v2.mp4", False, [{"start": 1.0, "end": 5.0, "label": "active"}]),
    ]
    cuts, skipped = ac.plan_cuts(docs)
    assert skipped == ["v2.mp4"]
    assert len(cuts) == 1
    assert cuts[0]["rel_path"] == "v1.mp4"
    assert cuts[0]["label"] == "eat"


def test_clip_output_name_embeds_source_id_label_and_span():
    ac = importlib.import_module("auto_presort.apply_clips")
    name = ac.clip_output_name("a1b2c3d4", "eat", 12.3, 19.8)
    assert name == "a1b2c3d4__eat__12.3-19.8.mp4"
    assert "__" in name  # rebalance_val 分组依赖


def test_build_ffmpeg_cmd_reencodes_accurately():
    ac = importlib.import_module("auto_presort.apply_clips")
    cmd = ac.build_ffmpeg_cmd("in.mp4", "out/eat/x.mp4", start=2.5, end=6.5)
    s = " ".join(cmd)
    assert cmd[0] == "ffmpeg"
    assert "-c copy" not in s and "copy" not in s   # 禁止 stream copy
    assert "libx264" in s                            # 精确重编码
    assert "-ss" in cmd and "2.5" in cmd
    assert "-t" in cmd and "4.0" in cmd              # duration = end-start
    assert cmd[-1] == "out/eat/x.mp4"


def test_duration_ok_within_tolerance():
    ac = importlib.import_module("auto_presort.apply_clips")
    assert ac.duration_ok(expected=4.0, actual=4.05, tolerance=0.25) is True
    assert ac.duration_ok(expected=4.0, actual=5.0, tolerance=0.25) is False


def test_source_id_from_rel_path_is_stable_and_short():
    ac = importlib.import_module("auto_presort.apply_clips")
    a = ac.source_id_for("sub/video_001.mp4")
    b = ac.source_id_for("sub/video_001.mp4")
    c = ac.source_id_for("sub\\video_001.mp4")   # Windows 分隔符归一
    assert a == b == c
    assert len(a) == 8 and "__" not in a
```

- [ ] **Step 2: 跑测试确认失败**

Run: `conda run -n AR python -m pytest tests/test_autopresort_apply_clips.py -v`
Expected: FAIL（`ModuleNotFoundError`）

- [ ] **Step 3: 最小实现**

```python
# auto_presort/apply_clips.py
"""apply 纯逻辑：Review-all 闸门、输出命名（内嵌来源 id）、ffmpeg 精确切片命令。

命名 <srcid>__<label>__<start>-<end>.mp4 —— "__" 是 rebalance_val 按来源分组的分隔符，
srcid 内不得出现 "__"。禁止 -c copy（关键帧吸附会抵消 inward trim）。
"""
import hashlib


def source_id_for(rel_path):
    norm = rel_path.replace("\\", "/")
    return hashlib.sha1(norm.encode("utf-8")).hexdigest()[:8]


def plan_cuts(docs):
    """返回 (cuts, skipped_rel_paths)。只收 approved=True 的 doc（Review-all 闸门）。"""
    cuts, skipped = [], []
    for doc in docs:
        if not doc.get("approved"):
            skipped.append(doc["rel_path"])
            continue
        for seg in doc["segments"]:
            cuts.append({
                "rel_path": doc["rel_path"],
                "source_id": doc["source_id"],
                "label": seg["label"],
                "start": float(seg["start"]),
                "end": float(seg["end"]),
            })
    return cuts, skipped


def clip_output_name(source_id, label, start, end):
    return f"{source_id}__{label}__{round(start, 3):g}-{round(end, 3):g}.mp4"


def build_ffmpeg_cmd(input_path, output_path, start, end):
    """精确重编码切片：-ss 前置 + 重编码（帧级精确），-t 用时长，无音轨。"""
    duration = round(end - start, 6)
    return [
        "ffmpeg", "-y",
        "-ss", f"{start:g}",
        "-i", input_path,
        "-t", str(duration),
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "18",
        "-an",
        output_path,
    ]


def duration_ok(expected, actual, tolerance=0.25):
    return abs(actual - expected) <= tolerance
```

- [ ] **Step 4: 跑测试确认通过**

Run: `conda run -n AR python -m pytest tests/test_autopresort_apply_clips.py -v`
Expected: PASS（5 passed）

---

## Task 3: 泄漏修复 — rebalance_val 按来源 id 分组

**Files:**
- Modify: `utils/abnormal_crop_video.py`（`rebalance_val` 前加 `source_key` + 分组逻辑）
- Test: `tests/test_abnormal_crop_video.py`（追加）

背景：现有 `rebalance_val(video_records, cfg)` 中 `video_records = {class: [(video_name, clip_files), ...]}`，按 record（=输入文件）挑 ~20% 进 val。auto_presort 的输出文件名为 `<srcid>__<label>__<span>.mp4`，同一原始视频会产出多个文件——必须同 split。

- [ ] **Step 1: 写失败测试（追加到 tests/test_abnormal_crop_video.py）**

```python
def test_source_key_extracts_srcid_or_falls_back():
    crop_video = importlib.import_module("utils.abnormal_crop_video")
    assert crop_video.source_key("a1b2c3d4__eat__12.3-19.8.mp4") == "a1b2c3d4"
    assert crop_video.source_key("a1b2c3d4__active__20-25.mp4") == "a1b2c3d4"
    # 旧命名（无 __）：整个文件名即 key（回退现行为）
    assert crop_video.source_key("CAM1_20260605.mp4") == "CAM1_20260605.mp4"


def test_rebalance_val_keeps_same_source_in_one_split(tmp_path, monkeypatch):
    crop_video = importlib.import_module("utils.abnormal_crop_video")
    out = tmp_path / "snap"
    # 同一来源 a1b2c3d4 的两个输入文件 + 另一来源 ffff0000 的两个输入文件
    records = {"eat": [
        ("a1b2c3d4__eat__0-5.mp4",   ["eat/a1b2c3d4__eat__0-5_clip-1.mp4"]),
        ("ffff0000__eat__0-5.mp4",   ["eat/ffff0000__eat__0-5_clip-1.mp4"]),
        ("a1b2c3d4__eat__10-15.mp4", ["eat/a1b2c3d4__eat__10-15_clip-1.mp4"]),
        ("ffff0000__eat__9-14.mp4",  ["eat/ffff0000__eat__9-14_clip-1.mp4"]),
    ]}
    for _, clips in records["eat"]:
        for c in clips:
            p = out / "train" / c
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes(b"x")
    cfg = crop_video.default_config()
    cfg["output_base_dir"] = str(out)
    crop_video.rebalance_val(records, cfg)
    # 断言：任一来源的 clip 不得同时出现在 train 和 val
    for src in ("a1b2c3d4", "ffff0000"):
        in_train = list((out / "train").rglob(f"{src}__*"))
        in_val = list((out / "val").rglob(f"{src}__*"))
        assert not (in_train and in_val), f"source {src} leaked across splits"
```

- [ ] **Step 2: 跑测试确认失败**

Run: `conda run -n AR python -m pytest tests/test_abnormal_crop_video.py -v -k "source_key or same_source"`
Expected: FAIL（`AttributeError: source_key`）

- [ ] **Step 3: 修改 rebalance_val（受控小改）**

在 `rebalance_val` 定义**之前**加：

```python
def source_key(video_name):
    """从 auto_presort 命名 <srcid>__<label>__<span> 提取来源 id；无 __ 的旧命名回退整名。"""
    return video_name.split("__")[0] if "__" in video_name else video_name


def _group_records_by_source(records):
    """[(video_name, clips)] → 按 source_key 合并为 [(key, merged_clips)]，保持首见顺序。"""
    grouped, order = {}, []
    for video_name, clip_files in records:
        key = source_key(video_name)
        if key not in grouped:
            grouped[key] = []
            order.append(key)
        grouped[key].extend(clip_files)
    return [(k, grouped[k]) for k in order]
```

然后把 `rebalance_val` 循环体开头的 `records` 换成分组后的（只改一行 + 注释）：

```python
def rebalance_val(video_records, cfg=None):
    """按视频粒度将约 20% 的片段从 train/ 移到 val/，均匀抽取视频避免泄露。"""
    output_base_dir = OUTPUT_BASE_DIR if cfg is None else cfg["output_base_dir"]
    for class_name, records in video_records.items():
        # 同源分组：auto_presort 会把一条原始视频切成多个输入文件（<srcid>__...），
        # 必须整组同 split，否则秒级相邻片段跨 train/val 造成泄漏。
        records = _group_records_by_source(records)
        total_segs = sum(len(clip_files) for _, clip_files in records)
        ...  # 其余逻辑完全不变
```

（即：在 `total_segs = ...` 之前插入一行 `records = _group_records_by_source(records)` 及注释，其余不动。）

- [ ] **Step 4: 跑全部 crop 测试确认通过（含原有测试无回归）**

Run: `conda run -n AR python -m pytest tests/test_abnormal_crop_video.py tests/test_crop_video_run.py -v`
Expected: PASS（原有全部 + 新增 2 个）

---

## Task 4: 泄漏修复 — health_report 同源跨 split 检测

**Files:**
- Modify: `snapshot_merge.py`（`health_report` 加 `source_leak`）
- Test: `tests/test_snapshot_merge.py`（追加）

- [ ] **Step 1: 写失败测试（追加到 tests/test_snapshot_merge.py；先读该文件看现有 fixture 风格，保持一致）**

```python
def test_health_report_detects_source_leak(tmp_path):
    import snapshot_merge as sm
    # 同一来源 a1b2c3d4 的不同 clip 分别落在 train 和 val
    for split, fn in [("train", "a1b2c3d4__eat__0-5_clip-1.mp4"),
                      ("val",   "a1b2c3d4__eat__10-15_clip-1.mp4")]:
        p = tmp_path / split / "eat" / fn
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(b"x")
    report = sm.health_report(str(tmp_path), ["eat"], frames=False)
    assert "a1b2c3d4" in report["source_leak"]
    # 不同名文件，旧 basename 检测抓不到
    assert report["cross_split_leak"] == []


def test_health_report_no_source_leak_for_legacy_names(tmp_path):
    import snapshot_merge as sm
    for split, fn in [("train", "CAM1_video_clip-1.mp4"),
                      ("val",   "CAM2_video_clip-1.mp4")]:
        p = tmp_path / split / "eat" / fn
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(b"x")
    report = sm.health_report(str(tmp_path), ["eat"], frames=False)
    assert report["source_leak"] == []
```

- [ ] **Step 2: 跑测试确认失败**

Run: `conda run -n AR python -m pytest tests/test_snapshot_merge.py -v -k source_leak`
Expected: FAIL（`KeyError: 'source_leak'`）

- [ ] **Step 3: 修改 health_report**

在 `health_report` 中（复用现有遍历循环，不加新循环）：

```python
def health_report(new_root, class_names, frames):
    per_class = {c: 0 for c in class_names}
    unrecognized, seen_split, cross_split_leak, label_mismatch = [], {}, [], []
    seen_source_split, source_leak = {}, []          # 新增
    for split in ("train", "val"):
        for rel, path in _iter_samples(os.path.join(new_root, split), frames):
            base = os.path.basename(rel)
            if base in seen_split and seen_split[base] != split:
                cross_split_leak.append(base)
            seen_split[base] = split
            # 新增：同源跨 split 检测（auto_presort 命名 <srcid>__...）
            if "__" in base:
                src = base.split("__")[0]
                if src in seen_source_split and seen_source_split[src] != split:
                    if src not in source_leak:
                        source_leak.append(src)
                seen_source_split[src] = split
            ...  # 其余现有逻辑不变
    return {"per_class": per_class, "cross_split_leak": cross_split_leak,
            "unrecognized": unrecognized, "label_mismatch": label_mismatch,
            "source_leak": source_leak}              # 新增字段
```

同时在 `run_train_distill.py` 体检打印区（`if report["cross_split_leak"]:` 之后）加：

```python
    if report.get("source_leak"):
        print(f"[SOURCE LEAK] Same source video appears in both train/val: {report['source_leak']}")
```

- [ ] **Step 4: 跑测试确认通过（含 run_train_distill 原有测试无回归）**

Run: `conda run -n AR python -m pytest tests/test_snapshot_merge.py tests/test_run_train_distill.py -v`
Expected: PASS（原有全部 + 新增 2 个）

---

## Task 5: review_server.py（API + Range 视频服务）

**Files:**
- Create: `auto_presort/review_server.py`
- Test: `tests/test_autopresort_review_server.py`

- [ ] **Step 1: 写失败测试**

```python
# tests/test_autopresort_review_server.py
import importlib
import json
import threading
import urllib.request


def _start(batch_dir):
    rs = importlib.import_module("auto_presort.review_server")
    server = rs.make_server(str(batch_dir), port=0)  # port=0 → 随机可用端口
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()
    return server, f"http://127.0.0.1:{server.server_address[1]}"


def _seed(batch_dir):
    st = importlib.import_module("auto_presort.store")
    st.save_doc(str(batch_dir), {
        "schema_version": "autopresort-1", "rel_path": "v1.mp4",
        "source_id": "abc123", "duration": 10.0, "approved": False,
        "segments": [{"start": 1.0, "end": 4.0, "label": "eat",
                      "confidence": 0.95, "source": "model"}],
        "gaps": [], "meta": {"class_names": ["eat", "drink"]},
    })


def test_api_videos_lists_docs(tmp_path):
    _seed(tmp_path)
    server, base = _start(tmp_path)
    try:
        with urllib.request.urlopen(f"{base}/api/videos") as r:
            data = json.loads(r.read())
        assert data[0]["rel_path"] == "v1.mp4"
        assert data[0]["approved"] is False
    finally:
        server.shutdown()


def test_api_doc_get_and_post_save_invalidates_approved(tmp_path):
    _seed(tmp_path)
    server, base = _start(tmp_path)
    try:
        body = json.dumps({"rel_path": "v1.mp4", "action": "approve"}).encode()
        req = urllib.request.Request(f"{base}/api/doc", data=body,
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req) as r:
            assert json.loads(r.read())["approved"] is True

        body = json.dumps({"rel_path": "v1.mp4", "action": "save", "segments": [
            {"start": 1.5, "end": 4.0, "label": "drink",
             "confidence": 0.95, "source": "human"}]}).encode()
        req = urllib.request.Request(f"{base}/api/doc", data=body,
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req) as r:
            doc = json.loads(r.read())
        assert doc["approved"] is False           # 编辑使批准失效
        assert doc["segments"][0]["label"] == "drink"
    finally:
        server.shutdown()


def test_video_range_request_returns_206(tmp_path):
    _seed(tmp_path)
    (tmp_path / "v1.mp4").write_bytes(b"0123456789ABCDEF")
    server, base = _start(tmp_path)
    try:
        req = urllib.request.Request(f"{base}/video/v1.mp4",
                                     headers={"Range": "bytes=4-7"})
        with urllib.request.urlopen(req) as r:
            assert r.status == 206
            assert r.read() == b"4567"
            assert r.headers["Content-Range"] == "bytes 4-7/16"
    finally:
        server.shutdown()


def test_video_path_traversal_rejected(tmp_path):
    _seed(tmp_path)
    server, base = _start(tmp_path)
    try:
        try:
            urllib.request.urlopen(f"{base}/video/../secret.txt")
            status = 200
        except urllib.error.HTTPError as e:
            status = e.code
        assert status in (400, 403, 404)
    finally:
        server.shutdown()
```

- [ ] **Step 2: 跑测试确认失败**

Run: `conda run -n AR python -m pytest tests/test_autopresort_review_server.py -v`
Expected: FAIL（`ModuleNotFoundError`）

- [ ] **Step 3: 最小实现**

```python
# auto_presort/review_server.py
"""审核服务：stdlib http.server。

GET  /               → review_app.html
GET  /api/videos     → [{rel_path, approved, duration, n_segments, class_names}]
GET  /api/doc?rel=X  → 完整 doc
POST /api/doc        → {rel_path, action: save|approve, segments?} → 更新后的 doc
GET  /video/<rel>    → 视频文件（支持 Range/206，长视频拖动必需）
"""
import json
import os
import posixpath
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse, parse_qs, unquote

from auto_presort import store

_HTML_PATH = os.path.join(os.path.dirname(__file__), "review_app.html")


def make_server(batch_dir, port=8765):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *a):  # 静默访问日志
            pass

        def _json(self, obj, status=200):
            body = json.dumps(obj, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _safe_video_path(self, rel):
            rel = unquote(rel).replace("\\", "/")
            norm = posixpath.normpath(rel)
            if norm.startswith("..") or posixpath.isabs(norm):
                return None
            path = os.path.join(batch_dir, *norm.split("/"))
            real = os.path.realpath(path)
            if not real.startswith(os.path.realpath(batch_dir)):
                return None
            return real if os.path.isfile(real) else None

        def do_GET(self):
            parsed = urlparse(self.path)
            if parsed.path == "/":
                with open(_HTML_PATH, "rb") as f:
                    body = f.read()
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            elif parsed.path == "/api/videos":
                docs = store.list_docs(batch_dir)
                self._json([{
                    "rel_path": d["rel_path"], "approved": d["approved"],
                    "duration": d["duration"], "n_segments": len(d["segments"]),
                    "class_names": d.get("meta", {}).get("class_names", []),
                } for d in docs])
            elif parsed.path == "/api/doc":
                rel = parse_qs(parsed.query).get("rel", [""])[0]
                try:
                    self._json(store.load_doc(batch_dir, rel))
                except FileNotFoundError:
                    self._json({"error": "not found"}, 404)
            elif parsed.path.startswith("/video/"):
                self._serve_video(parsed.path[len("/video/"):])
            else:
                self._json({"error": "not found"}, 404)

        def do_POST(self):
            parsed = urlparse(self.path)
            if parsed.path != "/api/doc":
                self._json({"error": "not found"}, 404)
                return
            length = int(self.headers.get("Content-Length", 0))
            payload = json.loads(self.rfile.read(length))
            try:
                doc = store.update_doc(
                    batch_dir, payload["rel_path"],
                    segments=payload.get("segments"),
                    action=payload["action"])
                self._json(doc)
            except FileNotFoundError:
                self._json({"error": "not found"}, 404)
            except (KeyError, ValueError) as e:
                self._json({"error": str(e)}, 400)

        def _serve_video(self, rel):
            path = self._safe_video_path(rel)
            if path is None:
                self._json({"error": "bad path"}, 404)
                return
            size = os.path.getsize(path)
            rng = self.headers.get("Range")
            start, end = 0, size - 1
            status = 200
            if rng and rng.startswith("bytes="):
                spec = rng[len("bytes="):].split("-")
                if spec[0]:
                    start = int(spec[0])
                    if spec[1]:
                        end = min(int(spec[1]), size - 1)
                elif spec[1]:                       # bytes=-N（末尾 N 字节）
                    start = max(0, size - int(spec[1]))
                status = 206
            length = end - start + 1
            self.send_response(status)
            self.send_header("Content-Type", "video/mp4")
            self.send_header("Accept-Ranges", "bytes")
            self.send_header("Content-Length", str(length))
            if status == 206:
                self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
            self.end_headers()
            with open(path, "rb") as f:
                f.seek(start)
                remaining = length
                while remaining > 0:
                    chunk = f.read(min(1 << 16, remaining))
                    if not chunk:
                        break
                    try:
                        self.wfile.write(chunk)
                    except (ConnectionAbortedError, BrokenPipeError):
                        return                       # 浏览器拖动会中断连接，正常
                    remaining -= len(chunk)

    return ThreadingHTTPServer(("127.0.0.1", port), Handler)
```

- [ ] **Step 4: 跑测试确认通过**

Run: `conda run -n AR python -m pytest tests/test_autopresort_review_server.py -v`
Expected: PASS（4 passed）
注意：Step 3 引用了 `review_app.html`（Task 6 创建）。本 Task 的测试不请求 `/`，不受影响；若实现顺序需要，先 `touch` 一个空 html 占位再由 Task 6 覆盖。

---

## Task 6: review_app.html + auto_presort_review.py 入口（人工验收）

**Files:**
- Create: `auto_presort/review_app.html`
- Create: `auto_presort_review.py`

前端无法 pytest——本 Task 交付完整文件 + 人工浏览器验收清单。JS 直接内联（单文件，零依赖，file:// 不适用因为要 fetch——必须经 server 打开）。

- [ ] **Step 1: 创建 auto_presort/review_app.html（完整文件）**

```html
<!DOCTYPE html>
<html lang="zh">
<head>
<meta charset="utf-8">
<title>auto_presort review</title>
<style>
  :root { --bg:#1e1e1e; --panel:#252526; --fg:#ddd; --accent:#0a84ff; }
  * { box-sizing: border-box; }
  body { margin:0; display:flex; height:100vh; background:var(--bg); color:var(--fg);
         font:14px/1.4 system-ui, sans-serif; }
  #sidebar { width:300px; background:var(--panel); overflow-y:auto; padding:8px; flex-shrink:0; }
  #main { flex:1; padding:16px; display:flex; flex-direction:column; min-width:0; }
  .vitem { padding:8px; border-radius:6px; cursor:pointer; margin-bottom:4px; }
  .vitem:hover { background:#333; }
  .vitem.active { background:#094771; }
  .badge { float:right; }
  video { width:100%; max-height:55vh; background:#000; border-radius:6px; }
  #timeline { position:relative; height:56px; background:#3a3a3a; border-radius:6px;
              margin-top:10px; user-select:none; }
  .seg { position:absolute; top:4px; bottom:4px; border-radius:4px; opacity:.85;
         cursor:pointer; overflow:hidden; font-size:11px; color:#fff;
         display:flex; align-items:center; justify-content:center; }
  .seg.selected { outline:2px solid #fff; }
  .handle { position:absolute; top:0; bottom:0; width:8px; cursor:ew-resize; }
  .handle.l { left:0; } .handle.r { right:0; }
  #cursor { position:absolute; top:0; bottom:0; width:2px; background:#fff; pointer-events:none; }
  #legend { margin-top:8px; } .chip { display:inline-block; padding:2px 10px;
    border-radius:10px; margin-right:8px; font-size:12px; color:#fff; }
  #editor { margin-top:10px; display:flex; gap:8px; align-items:center; flex-wrap:wrap; }
  button, select { background:#3a3a3a; color:var(--fg); border:1px solid #555;
    border-radius:6px; padding:6px 14px; cursor:pointer; font-size:14px; }
  button.primary { background:var(--accent); border-color:var(--accent); }
  button.danger { background:#8b2e2e; border-color:#8b2e2e; }
  #status { margin-left:auto; opacity:.7; }
  .gapinfo { font-size:12px; opacity:.6; margin-top:6px; }
</style>
</head>
<body>
<div id="sidebar"><h3 style="margin:4px">Videos</h3><div id="vlist"></div></div>
<div id="main">
  <video id="player" controls></video>
  <div id="timeline"></div>
  <div id="legend"></div>
  <div class="gapinfo">灰色 = 模型不确定/无动物/边界区（未生成 clip）。彩色块可拖两端调整，点击选中后可改标签/删除。</div>
  <div id="editor">
    <select id="labelSel"></select>
    <button id="relabelBtn">改标签</button>
    <button id="deleteBtn" class="danger">删除选中段</button>
    <button id="approveBtn" class="primary">✔ Approve 本视频</button>
    <span id="status"></span>
  </div>
</div>
<script>
const COLORS = ["#2d7dd2","#97cc04","#eeb902","#f45d01","#9b5de5","#00bbf9"];
let videos = [], curRel = null, doc = null, selected = -1, classNames = [];
let drag = null; // {idx, side}

const $ = id => document.getElementById(id);
const player = $("player"), tlEl = $("timeline");

async function api(path, opts) {
  const r = await fetch(path, opts);
  if (!r.ok) throw new Error(await r.text());
  return r.json();
}
function colorOf(label) {
  const i = classNames.indexOf(label);
  return COLORS[(i >= 0 ? i : 0) % COLORS.length];
}
async function loadList() {
  videos = await api("/api/videos");
  const el = $("vlist"); el.innerHTML = "";
  for (const v of videos) {
    const d = document.createElement("div");
    d.className = "vitem" + (v.rel_path === curRel ? " active" : "");
    d.innerHTML = `${v.rel_path} <span class="badge">${v.approved ? "✅" : "⬜"} ${v.n_segments}段</span>`;
    d.onclick = () => loadVideo(v.rel_path);
    el.appendChild(d);
  }
}
async function loadVideo(rel) {
  curRel = rel; selected = -1;
  doc = await api("/api/doc?rel=" + encodeURIComponent(rel));
  classNames = (doc.meta && doc.meta.class_names) || [];
  player.src = "/video/" + rel;
  $("labelSel").innerHTML = classNames.map(c => `<option>${c}</option>`).join("");
  $("legend").innerHTML = classNames.map(c =>
    `<span class="chip" style="background:${colorOf(c)}">${c}</span>`).join("")
    + `<span class="chip" style="background:#666">gap</span>`;
  render(); loadList();
}
function render() {
  tlEl.innerHTML = "<div id='cursor' style='left:0'></div>";
  const W = tlEl.clientWidth, D = doc.duration || 1;
  doc.segments.forEach((s, i) => {
    const el = document.createElement("div");
    el.className = "seg" + (i === selected ? " selected" : "");
    el.style.left = (s.start / D * W) + "px";
    el.style.width = Math.max(6, (s.end - s.start) / D * W) + "px";
    el.style.background = colorOf(s.label);
    el.title = `${s.label} ${s.start.toFixed(1)}–${s.end.toFixed(1)}s conf=${(s.confidence||0).toFixed(2)}`;
    el.textContent = s.label;
    el.innerHTML += "<div class='handle l'></div><div class='handle r'></div>";
    el.onmousedown = e => {
      if (e.target.classList.contains("handle"))
        drag = { idx: i, side: e.target.classList.contains("l") ? "l" : "r" };
      else { selected = i; player.currentTime = s.start; render(); }
      e.preventDefault();
    };
    tlEl.appendChild(el);
  });
}
tlEl.onclick = e => {          // 点击空白处（gap）→ 跳转播放
  if (e.target !== tlEl) return;
  const rect = tlEl.getBoundingClientRect();
  player.currentTime = (e.clientX - rect.left) / rect.width * doc.duration;
};
window.onmousemove = e => {
  if (!drag) return;
  const rect = tlEl.getBoundingClientRect();
  const t = Math.max(0, Math.min(doc.duration,
    (e.clientX - rect.left) / rect.width * doc.duration));
  const s = doc.segments[drag.idx];
  if (drag.side === "l") s.start = Math.min(t, s.end - 0.2);
  else s.end = Math.max(t, s.start + 0.2);
  s.source = "human";
  render();
};
window.onmouseup = async () => {
  if (!drag) return;
  drag = null;
  await save("save");          // 拖完即存；save 会使 approved 失效
};
player.ontimeupdate = () => {
  const W = tlEl.clientWidth;
  $("cursor").style.left = (player.currentTime / (doc.duration || 1) * W) + "px";
};
async function save(action, extra) {
  const payload = Object.assign(
    { rel_path: curRel, action: action }, extra || {});
  if (action === "save") payload.segments = doc.segments;
  doc = await api("/api/doc", { method: "POST",
    headers: {"Content-Type": "application/json"}, body: JSON.stringify(payload) });
  $("status").textContent = action === "approve" ? "已批准 ✅"
    : "已保存（批准已失效，需重新 Approve）";
  render(); loadList();
}
$("relabelBtn").onclick = async () => {
  if (selected < 0) return alert("先点击选中一个段");
  doc.segments[selected].label = $("labelSel").value;
  doc.segments[selected].source = "human";
  await save("save");
};
$("deleteBtn").onclick = async () => {
  if (selected < 0) return alert("先点击选中一个段");
  doc.segments.splice(selected, 1); selected = -1;
  await save("save");
};
$("approveBtn").onclick = () => save("approve");
window.onresize = () => doc && render();
loadList().then(() => videos.length && loadVideo(videos[0].rel_path));
</script>
</body>
</html>
```

- [ ] **Step 2: 创建 auto_presort_review.py 入口**

```python
"""Step 2 of auto_presort: start the local review server and open the browser.

Light entry point: stdlib only (no torch/YOLO). Console prints in English
(cp1252-safe, consistent with .bat convention).
Usage: python auto_presort_review.py --dir <raw_batch_dir> [--port 8765]
"""
import argparse
import sys
import webbrowser


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True, help="raw batch dir (contains .autopresort/)")
    ap.add_argument("--port", type=int, default=8765)
    a = ap.parse_args(argv)

    from auto_presort.review_server import make_server
    server = make_server(a.dir, port=a.port)
    url = f"http://127.0.0.1:{server.server_address[1]}/"
    print(f"[OK] Review server running at {url}")
    print("[NOTE] Approve each video in the browser; then run auto_presort_apply.py")
    print("Press Ctrl+C to stop.")
    webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[OK] Server stopped.")


if __name__ == "__main__":
    main(sys.argv[1:])
```

- [ ] **Step 3: 自动化冒烟（服务能起、页面能取）**

Run: `conda run -n AR python -c "from auto_presort.review_server import make_server; import threading, urllib.request, tempfile, os; d=tempfile.mkdtemp(); s=make_server(d, port=0); t=threading.Thread(target=s.serve_forever, daemon=True); t.start(); html=urllib.request.urlopen(f'http://127.0.0.1:{s.server_address[1]}/').read(); s.shutdown(); print('html ok', len(html)>1000)"`
Expected: 打印 `html ok True`

- [ ] **Step 4: 人工浏览器验收清单（记录在完成报告里，不阻塞后续 Task）**

用一个真实批次（跑过 label 后）执行 `python auto_presort_review.py --dir <批次目录>`，核对：
1. 左侧列表显示全部视频 + ⬜/✅ 徽标 + 段数
2. 点视频 → 右侧加载播放器 + 彩色时间线（颜色对应图例）
3. 点彩色块 → 播放器跳到该段起点；块高亮选中
4. 拖块左/右边缘 → 宽度实时变化，松手后状态栏显示"已保存（批准已失效）"
5. 选中块 → 改标签下拉 + 按钮生效；删除按钮移除块
6. Approve → 徽标变 ✅；再做任何编辑 → 徽标回 ⬜（approved 失效）
7. 长视频拖进度条流畅（Range 生效）

---

## Task 7: apply 入口 + 真实 ffmpeg 集成测试

**Files:**
- Create: `auto_presort_apply.py`
- Test: `tests/test_autopresort_apply_ffmpeg.py`

- [ ] **Step 1: 写失败测试（真实 ffmpeg + cv2 合成视频，ffmpeg 已在 PATH）**

```python
# tests/test_autopresort_apply_ffmpeg.py
import importlib
import json
import shutil
import subprocess

import pytest

cv2 = pytest.importorskip("cv2")
np = pytest.importorskip("numpy")

pytestmark = pytest.mark.skipif(shutil.which("ffmpeg") is None,
                                reason="ffmpeg not on PATH")


def _make_video(path, seconds=10, fps=10, w=160, h=120):
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    wr = cv2.VideoWriter(str(path), fourcc, fps, (w, h))
    for i in range(seconds * fps):
        f = np.full((h, w, 3), i % 255, dtype=np.uint8)
        wr.write(f)
    wr.release()


def _probe_duration(path):
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", str(path)],
        capture_output=True, text=True)
    return float(out.stdout.strip())


def test_apply_cuts_only_approved_and_names_with_source_id(tmp_path):
    apply_mod = importlib.import_module("auto_presort_apply")
    st = importlib.import_module("auto_presort.store")

    batch = tmp_path / "batch"; batch.mkdir()
    _make_video(batch / "v1.mp4")
    _make_video(batch / "v2.mp4")
    st.save_doc(str(batch), {
        "schema_version": "autopresort-1", "rel_path": "v1.mp4",
        "source_id": "aaaa1111", "duration": 10.0, "approved": True,
        "segments": [{"start": 2.0, "end": 6.0, "label": "eat",
                      "confidence": 0.95, "source": "model"}],
        "gaps": [], "meta": {"class_names": ["eat"]}})
    st.save_doc(str(batch), {
        "schema_version": "autopresort-1", "rel_path": "v2.mp4",
        "source_id": "bbbb2222", "duration": 10.0, "approved": False,
        "segments": [{"start": 1.0, "end": 5.0, "label": "eat",
                      "confidence": 0.95, "source": "model"}],
        "gaps": [], "meta": {"class_names": ["eat"]}})

    out_dir = tmp_path / "sorted-2026-07-17"
    result = apply_mod.run_apply(str(batch), str(out_dir))

    assert result["skipped"] == ["v2.mp4"]
    outs = list(out_dir.rglob("*.mp4"))
    assert len(outs) == 1
    assert outs[0].name == "aaaa1111__eat__2-6.mp4"
    assert outs[0].parent.name == "eat"
    assert abs(_probe_duration(outs[0]) - 4.0) <= 0.25   # 精确切片校验


def test_apply_reports_duration_mismatch(tmp_path):
    apply_mod = importlib.import_module("auto_presort_apply")
    st = importlib.import_module("auto_presort.store")
    batch = tmp_path / "batch"; batch.mkdir()
    _make_video(batch / "v1.mp4", seconds=5)
    # segment 超出视频实际长度 → 输出时长不足 → 记入 warnings
    st.save_doc(str(batch), {
        "schema_version": "autopresort-1", "rel_path": "v1.mp4",
        "source_id": "cccc3333", "duration": 5.0, "approved": True,
        "segments": [{"start": 3.0, "end": 9.0, "label": "eat",
                      "confidence": 0.95, "source": "model"}],
        "gaps": [], "meta": {"class_names": ["eat"]}})
    result = apply_mod.run_apply(str(batch), str(tmp_path / "out"))
    assert result["warnings"]
```

- [ ] **Step 2: 跑测试确认失败**

Run: `conda run -n AR python -m pytest tests/test_autopresort_apply_ffmpeg.py -v`
Expected: FAIL（`ModuleNotFoundError: auto_presort_apply`）

- [ ] **Step 3: 创建 auto_presort_apply.py**

```python
"""Step 3 of auto_presort: cut approved segments into sorted-<date>/<label>/.

Light entry: stdlib + ffmpeg on PATH. Skips unapproved videos (Review-all gate).
Accurate re-encode cutting (never -c copy). English console prints (cp1252-safe).
Usage: python auto_presort_apply.py --dir <raw_batch_dir> --out <sorted_dir>
"""
import argparse
import os
import subprocess
import sys

from auto_presort import store
from auto_presort.apply_clips import plan_cuts, clip_output_name, build_ffmpeg_cmd, duration_ok


def _probe_duration(path):
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", path],
        capture_output=True, text=True)
    try:
        return float(out.stdout.strip())
    except ValueError:
        return -1.0


def run_apply(batch_dir, out_dir, tolerance=0.25):
    docs = store.list_docs(batch_dir)
    cuts, skipped = plan_cuts(docs)
    if skipped:
        print(f"[GATE] Skipped {len(skipped)} unapproved video(s): {skipped}")
    written, warnings = [], []
    for cut in cuts:
        src = os.path.join(batch_dir, *cut["rel_path"].replace("\\", "/").split("/"))
        if not os.path.isfile(src):
            warnings.append(f"missing source video: {cut['rel_path']}")
            continue
        name = clip_output_name(cut["source_id"], cut["label"], cut["start"], cut["end"])
        dst = os.path.join(out_dir, cut["label"], name)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        cmd = build_ffmpeg_cmd(src, dst, cut["start"], cut["end"])
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0 or not os.path.isfile(dst):
            warnings.append(f"ffmpeg failed: {name}: {proc.stderr[-300:]}")
            continue
        actual = _probe_duration(dst)
        expected = cut["end"] - cut["start"]
        if not duration_ok(expected, actual, tolerance):
            warnings.append(
                f"duration mismatch: {name} expected={expected:.2f}s actual={actual:.2f}s")
        written.append(dst)
    print(f"[OK] Wrote {len(written)} clip(s) to {out_dir}")
    for w in warnings:
        print(f"[WARN] {w}")
    return {"written": written, "skipped": skipped, "warnings": warnings}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True, help="raw batch dir (contains .autopresort/)")
    ap.add_argument("--out", required=True,
                    help="output sorted dir, e.g. dataset/raw_videos/rabbit/hkbr-record/sorted-2026-07-17")
    a = ap.parse_args(argv)
    run_apply(a.dir, a.out)


if __name__ == "__main__":
    main(sys.argv[1:])
```

- [ ] **Step 4: 跑测试确认通过**

Run: `conda run -n AR python -m pytest tests/test_autopresort_apply_ffmpeg.py -v`
Expected: PASS（2 passed）

---

## Task 8: label 入口（真实装配，冒烟验证）

**Files:**
- Create: `auto_presort_label.py`

真实 GPU/YOLO/teacher 路径，无单测（既定风格）；复用 Phase 0 已测的 `detect_track.build_track`、`engine.label_video`、`checkpoint_guard.verify_and_load_teacher`、Task 2 的 `source_id_for`、Task 1 的 store。

- [ ] **Step 1: 创建 auto_presort_label.py**

```python
"""Step 1 of auto_presort: model pre-labels a batch of raw videos into .autopresort/ JSONs.

Heavy entry: needs GPU + torch + YOLO + teacher checkpoint (with .meta.json sidecar).
English console prints (cp1252-safe).
Usage:
  python auto_presort_label.py --species rabbit --dir <raw_batch_dir> \
      --checkpoint <teacher.pth> [--threshold 0.90] [--trim-margin 1.0]
"""
import argparse
import os
import sys

VIDEO_EXTS = (".mp4", ".avi", ".mov", ".webm")


def collect_videos(batch_dir):
    """Recursively find raw videos; returns rel paths with forward slashes."""
    rels = []
    for root, dirs, files in os.walk(batch_dir):
        dirs[:] = [d for d in dirs if not d.startswith(".")]
        for fn in sorted(files):
            if fn.lower().endswith(VIDEO_EXTS):
                rel = os.path.relpath(os.path.join(root, fn), batch_dir)
                rels.append(rel.replace("\\", "/"))
    return rels


def default_params(threshold, trim_margin):
    return dict(window_sec=3.0, stride_sec=1.0, threshold=threshold, enter_run=2,
                trim_margin=trim_margin, min_duration=2.0, imgsz=224,
                temporal_window=16, expand_ratio=1.25, no_animal_overlap=0.5)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--species", required=True, choices=["pet", "rabbit"])
    ap.add_argument("--dir", required=True, help="raw batch dir with long videos")
    ap.add_argument("--checkpoint", required=True, help="teacher .pth (sidecar .meta.json required)")
    ap.add_argument("--threshold", type=float, default=0.90)
    ap.add_argument("--trim-margin", type=float, default=1.0)
    a = ap.parse_args(argv)

    import pipeline_config as pc
    from ultralytics import YOLO
    from auto_presort import detect_track, engine, store
    from auto_presort.apply_clips import source_id_for
    from auto_presort.checkpoint_guard import verify_and_load_teacher

    sp = pc.SPECIES[a.species]
    rels = collect_videos(a.dir)
    if not rels:
        print(f"[ERROR] No videos found under: {a.dir}")
        return 1

    print(f"[1/3] Verifying teacher checkpoint (hard-fail on any mismatch)...")
    model, meta = verify_and_load_teacher(a.checkpoint)
    if meta["class_names"] != sp["class_names"]:
        print(f"[ERROR] sidecar class_names {meta['class_names']} != "
              f"SPECIES[{a.species}] {sp['class_names']}. Refusing to label.")
        return 1

    print(f"[2/3] Loading YOLO: {sp['yolo_model']}")
    yolo = YOLO(sp["yolo_model"])

    params = default_params(a.threshold, a.trim_margin)
    reader = engine.build_real_window_reader()
    classifier = engine.build_real_classifier(model, meta["class_names"], params)

    print(f"[3/3] Labeling {len(rels)} video(s)...")
    for i, rel in enumerate(rels, 1):
        vp = os.path.join(a.dir, *rel.split("/"))
        print(f"  ({i}/{len(rels)}) {rel}")
        track = detect_track.build_track(vp, yolo, sp["target_classes"])
        doc = engine.label_video(
            video_path=vp, rel_path=rel, source_id=source_id_for(rel), track=track,
            read_window_frames_fn=lambda v, s, e, fps, _b=track["boxes"]:
                reader(v, s, e, fps, _b),
            classify_frames_fn=classifier, meta=meta, params=params)
        store.save_doc(a.dir, doc)
        print(f"      segments={len(doc['segments'])} gaps={len(doc['gaps'])}")

    store.save_manifest(a.dir, {
        "species": a.species, "checkpoint_sha256": meta["sha256"],
        "class_names": meta["class_names"], "params": params,
        "n_videos": len(rels)})
    print(f"[DONE] JSONs written to {os.path.join(a.dir, '.autopresort')}")
    print("Next: python auto_presort_review.py --dir <this dir>")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
```

- [ ] **Step 2: 冒烟（导入 + collect_videos 纯逻辑 + help 不触发重依赖）**

Run: `conda run -n AR python -c "import auto_presort_label as m; import tempfile, os; d=tempfile.mkdtemp(); os.makedirs(os.path.join(d,'sub')); open(os.path.join(d,'sub','a.mp4'),'w').close(); open(os.path.join(d,'skip.txt'),'w').close(); print(m.collect_videos(d)); print(m.default_params(0.9,1.0)['threshold'])"`
Expected: 打印 `['sub/a.mp4']` 和 `0.9`

Run: `conda run -n AR python auto_presort_label.py --help`
Expected: 打印 usage，退出码 0（argparse 在 import 重依赖之前）

---

## Task 9: Phase 1 全量回归

**Files:** 无新增。

- [ ] **Step 1: Phase 1 新测试全绿**

Run: `conda run -n AR python -m pytest tests/test_autopresort_store.py tests/test_autopresort_apply_clips.py tests/test_autopresort_review_server.py tests/test_autopresort_apply_ffmpeg.py -v`
Expected: 全 PASS

- [ ] **Step 2: 全仓库回归（Phase 0 + 现有套件 + 改动的 crop/snapshot 测试）**

Run: `conda run -n AR python -m pytest tests/ -q`
Expected: 全 PASS，0 失败

- [ ] **Step 3: 三入口冒烟**

Run: `conda run -n AR python auto_presort_review.py --help && conda run -n AR python auto_presort_apply.py --help && conda run -n AR python auto_presort_label.py --help`
Expected: 三个 usage 各自打印，无异常

---

## 自检（Self-Review）

**Spec 覆盖：**
- §5 审核（单页整批、列表徽标、时间线、拖边界、改标签、删除、Approve、编辑失效批准、Range）→ Task 1/5/6 ✅
- §6 apply（闸门跳过打印、精确重编码非 -c copy、时长校验、`<srcid>__<label>__<span>.mp4`、不抠图、相对路径）→ Task 2/7 ✅
- §4.1 泄漏修复（rebalance_val 同源同 split + health_report 同源检漏 + run_train_distill 打印）→ Task 3/4 ✅
- §3 label 入口（sidecar 硬校验 + 类别顺序比对 SPECIES 一致才放行、manifest 记录）→ Task 8 ✅
- 明确延后（spec §2 非目标）：gap 拖拽新建、同事打包/portable ffmpeg、非 H.264 转码 → 不在本计划 ✅
- 用户指示：gate 阈值参数化不调 → apply/label 均为 CLI 参数，无 gate 逻辑 ✅

**占位符扫描：** 无 TBD/TODO；HTML/JS 为完整文件；Task 3/4 对既有函数的改动给出了精确插入位置与代码。

**类型/签名一致性：** `store.save_doc(batch_dir, doc)`/`load_doc(batch_dir, rel)`/`update_doc(batch_dir, rel, segments, action)` 在 Task 1 定义、Task 5 server 与 Task 7/8 入口一致复用；`plan_cuts(docs)->(cuts, skipped)`、`clip_output_name(srcid,label,start,end)`、`build_ffmpeg_cmd(in,out,start,end)`、`source_id_for(rel)` 在 Task 2 定义、Task 7/8 一致复用；`make_server(batch_dir, port)` Task 5 定义、Task 6 入口复用；doc 字段沿用 Phase 0 `build_video_doc` 输出结构。

**已知取舍：** 前端 JS 无自动化测试（人工清单验收）；label 入口真实 GPU 路径无单测（冒烟 + Phase 0 集成测试覆盖装配数据流）；`run_train_distill.py` 的 source_leak 打印行无独立测试（`health_report` 的返回字段已测）。
