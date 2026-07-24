# Task 3 Brief: review_server 保留 segment_levels（读取返回 + save 不丢失）

这是你的需求来源，使用其中的确切数值/命名（逐字照抄，勿改）。

## Files
- Test: `tests/test_autopresort_review_server.py`（追加测试）
- `auto_presort/review_server.py` / `auto_presort/store.py` 预期**无需改动**——本任务先用测试证明现有实现已正确保留 `segment_levels`；**只有当测试失败时才改**。

## 背景
- `doc` 现在多了一个 `segment_levels` 字段（5 档段，key `"0.5"`…`"0.9"`）。
- `doc["segments"]` 是"当前审核中"的那套（初始 = 0.9 档）。
- 前端 save 只发 `segments`（不含 `segment_levels`）。服务端 `store.update_doc` 走 load 完整 doc → 只改 segments/approved → save 完整 doc，理论上天然保留其他字段。
- `/api/doc` GET 返回完整 doc，应含 `segment_levels`。
- 本任务用两个测试锁定这两点。

## Step 1: 写测试

在 `tests/test_autopresort_review_server.py` 末尾追加（文件顶部已有 `_start` 和 `import` 语句：`importlib, json, threading, urllib.error, urllib.request`）：

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

## Step 2: 运行
`conda run --no-capture-output -n AR python -m pytest tests/test_autopresort_review_server.py -v -k "segment_levels"`
Expected: PASS（现有实现应直接通过）。

## Step 3: 只有失败才改
若 `test_save_preserves_segment_levels` FAIL（说明 `update_doc` 丢了字段），检查 `auto_presort/store.py` 的 `update_doc`（当前第 53-64 行）确保它是"load 完整 doc → 只改 segments/approved → save 完整 doc"。当前实现已如此，正常应直接通过。**不通过再定位，不预先改。** 若两个测试都直接通过，本任务不改任何生产代码，正常。

## Step 4: 跑全部 autopresort 后端测试确认无回归
`conda run --no-capture-output -n AR python -m pytest tests/ -v -k "autopresort"`
Expected: PASS（全部）。

## 硬约束
- 本仓库**不是 git 仓库**——不要执行任何 git 命令，不要 commit。"完成" = 上述测试全绿。
- 测试命令固定：`conda run --no-capture-output -n AR python -m pytest <path> -v`（注意本环境非 ASCII 输出会让 conda run 崩，必须带 `--no-capture-output`）。
- 默认只加测试；生产代码仅在测试失败时按 Step 3 定位后最小改动。
