# Task 1 Brief: Backend — Add conf_tier to /api/videos

这是你的需求来源，使用其中的确切代码/命名（逐字照抄，勿改）。

## 背景
auto_presort 审核服务的 `/api/videos` 接口返回视频列表。每个视频的 doc 里现在有一个 `segment_levels` 字段（模型预切的 5 档段，key `"0.5"`…`"0.9"`，每档是一个段数组，可能为空）。要新加一个 `conf_tier` 字段，表示"这个视频最高撑到哪一档"（从 0.9 往下找第一个非空档）。旧格式 doc 无 `segment_levels` → `conf_tier` 为 null。

## Files
- Test: `tests/test_autopresort_review_server.py`（末尾追加测试）
- Modify: `auto_presort/review_server.py`（`/api/videos` 处理分支，当前第 83-89 行）

## Step 1: 写失败测试

在 `tests/test_autopresort_review_server.py` 末尾追加（文件顶部已有 `importlib, json, threading, urllib.error, urllib.request` 和 `_start` 辅助函数）：

```python
def test_api_videos_includes_conf_tier(tmp_path):
    """Verify /api/videos includes conf_tier field computed from segment_levels."""
    st = importlib.import_module("auto_presort.store")

    # Video 1: has segments at 0.8 level (not at 0.9) → tier = "0.8"
    st.save_doc(str(tmp_path), {
        "schema_version": "autopresort-2", "rel_path": "v1.mp4",
        "source_id": "s1", "duration": 10.0, "approved": False,
        "segments": [], "gaps": [],
        "segment_levels": {
            "0.5": [{"start":1,"end":3,"label":"eat","confidence":0.6,"source":"model"}],
            "0.6": [{"start":1,"end":3,"label":"eat","confidence":0.65,"source":"model"}],
            "0.7": [{"start":2,"end":4,"label":"eat","confidence":0.75,"source":"model"}],
            "0.8": [{"start":2,"end":4,"label":"eat","confidence":0.85,"source":"model"}],
            "0.9": []  # empty at 0.9 → highest non-empty is 0.8
        },
        "meta": {"class_names": ["eat"]},
    })

    # Video 2: has segments at 0.9 level → tier = "0.9"
    st.save_doc(str(tmp_path), {
        "schema_version": "autopresort-2", "rel_path": "v2.mp4",
        "source_id": "s2", "duration": 12.0, "approved": True,
        "segments": [{"start":1,"end":5,"label":"drink","confidence":0.95,"source":"model"}],
        "gaps": [],
        "segment_levels": {
            "0.5": [{"start":1,"end":5,"label":"drink","confidence":0.95,"source":"model"}],
            "0.9": [{"start":1,"end":5,"label":"drink","confidence":0.95,"source":"model"}],
        },
        "meta": {"class_names": ["drink"]},
    })

    # Video 3: no segment_levels (old format) → tier = null
    st.save_doc(str(tmp_path), {
        "schema_version": "autopresort-1", "rel_path": "v3.mp4",
        "source_id": "s3", "duration": 8.0, "approved": False,
        "segments": [], "gaps": [], "meta": {"class_names": ["eat"]},
    })

    server, base = _start(tmp_path)
    try:
        with urllib.request.urlopen(f"{base}/api/videos") as r:
            items = json.loads(r.read())

        assert len(items) == 3

        v1 = next(x for x in items if x["rel_path"] == "v1.mp4")
        assert v1["conf_tier"] == "0.8", "Should pick highest non-empty tier"

        v2 = next(x for x in items if x["rel_path"] == "v2.mp4")
        assert v2["conf_tier"] == "0.9", "Should pick 0.9 when available"

        v3 = next(x for x in items if x["rel_path"] == "v3.mp4")
        assert v3["conf_tier"] is None, "No segment_levels → null tier"
    finally:
        server.shutdown()
```

## Step 2: 运行确认失败
`conda run --no-capture-output -n AR python -m pytest tests/test_autopresort_review_server.py::test_api_videos_includes_conf_tier -v`
Expected: FAIL（`conf_tier` 字段还不存在）

## Step 3: 实现

在 `auto_presort/review_server.py`，把 `/api/videos` 处理分支（当前第 83-89 行）整体替换为：

```python
            elif parsed.path == "/api/videos":
                docs = store.list_docs(batch_dir)
                items = []
                for d in docs:
                    # Calculate conf_tier: highest level with non-empty segments
                    conf_tier = None
                    if "segment_levels" in d and d["segment_levels"]:
                        for level in [0.9, 0.8, 0.7, 0.6, 0.5]:
                            key = f"{level:.1f}"
                            if key in d["segment_levels"] and len(d["segment_levels"][key]) > 0:
                                conf_tier = key
                                break
                    items.append({
                        "rel_path": d["rel_path"],
                        "approved": d["approved"],
                        "duration": d["duration"],
                        "n_segments": len(d["segments"]),
                        "class_names": d.get("meta", {}).get("class_names", []),
                        "conf_tier": conf_tier,
                    })
                self._json(items)
```

替换前先用 `Read` 确认第 83-89 行确实是原来的 `/api/videos` 分支（原样是一个 `self._json([{...} for d in docs])` 列表推导）。

## Step 4: 运行确认通过
`conda run --no-capture-output -n AR python -m pytest tests/test_autopresort_review_server.py::test_api_videos_includes_conf_tier -v`
Expected: PASS

## Step 5: 跑全部 autopresort 测试确认无回归
`conda run --no-capture-output -n AR python -m pytest tests/ -v -k "autopresort"`
Expected: 全部 PASS

## 硬约束
- 本仓库**不是 git 仓库**——不要执行任何 git 命令，不要 commit。"完成" = 上述测试全绿。
- 测试命令固定：`conda run --no-capture-output -n AR python -m pytest <path> -v`（本环境非 ASCII 输出会让 conda run 崩，必须带 `--no-capture-output`）。
- 只改这两个文件。保持既有代码风格（4 空格缩进、现有注释语言）。
- 不要新增依赖，不要改 2a/2c 或其他接口。
