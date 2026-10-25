"""scene_retrain 编排器测试：重 I/O 全 mock，遵循项目 unittest+pytest 约定。"""
import json
import os
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import pytest

from scripts.scene_retrain import pipeline as pl
from scripts.scene_retrain.job_spec import JobSpec
from scripts.scene_retrain.paths import build_retrain_context


# ---------- paths / job_spec ----------

def test_build_retrain_context_flat_paths_no_site():
    from scripts.pipeline import pipeline_config as pc
    ctx = build_retrain_context("rabbit", "2026-09-10")
    # SPECIES 里是 Windows 风格硬编码路径，期望值必须用同一来源拼接
    expected_ws = os.path.join(pc.SPECIES["rabbit"]["raw_root"],
                               "_retrain", "2026-09-10")
    assert ctx["workspace"] == expected_ws
    assert ctx["sorted_dir"] == os.path.join(expected_ws, "sorted-2026-09-10")
    assert ctx["state_file"] == os.path.join(expected_ws, "_review_state.json")
    assert ctx["snapshot_dir"] == pc.snapshot_dir("rabbit", "2026-09-10")
    assert "site" not in ctx
    assert ctx["classes"] == ["resting", "active", "eating", "drinking"]


def test_build_retrain_context_frames_snapshot_has_no_videos_layer():
    from scripts.pipeline import pipeline_config as pc
    ctx = build_retrain_context("pet", "2026-09-10", frames=True)
    assert ctx["snapshot_dir"] == pc.snapshot_dir("pet", "2026-09-10", frames=True)
    assert not ctx["snapshot_dir"].endswith("videos")
    assert ctx["frames"] is True


def test_job_spec_roundtrip(tmp_path):
    job = JobSpec(species="pet", date="2026-09-10", videos_dir="V",
                  teacher_checkpoint="t.pth", classes=["eat"], frames=True,
                  threshold=0.8)
    p = tmp_path / "job.json"
    job.write(p)
    got = JobSpec.read(p)
    assert got == job


# ---------- stage 1 ----------

def test_resolve_videos_dir_rejects_missing(tmp_path):
    with pytest.raises(SystemExit):
        pl.resolve_videos_dir(str(tmp_path / "nope"))


def test_resolve_videos_dir_rejects_empty(tmp_path):
    with pytest.raises(SystemExit):
        pl.resolve_videos_dir(str(tmp_path))


def test_resolve_videos_dir_accepts_batch(tmp_path):
    (tmp_path / "a.mp4").write_bytes(b"x")
    (tmp_path / ".hidden").mkdir()
    (tmp_path / ".hidden" / "b.mp4").write_bytes(b"x")
    got = pl.resolve_videos_dir(str(tmp_path))
    assert got == str(tmp_path)


def test_group_videos_by_animal_profile(tmp_path):
    for folder, profile in (("one", "pet"), ("two", "rabbit"), ("three", "pet")):
        record = tmp_path / folder
        record.mkdir()
        (record / "video.mp4").write_bytes(b"video")
        (record / "metadata.json").write_text(
            json.dumps({"animal_profile": profile}), encoding="utf-8")

    groups = pl.group_videos_by_animal_profile(str(tmp_path))
    assert {key: len(value) for key, value in groups.items()} == {"pet": 2, "rabbit": 1}


def test_group_videos_by_animal_profile_fails_on_missing_or_invalid_profile(tmp_path):
    record = tmp_path / "record"
    record.mkdir()
    (record / "video.mp4").write_bytes(b"video")
    (record / "metadata.json").write_text("{}", encoding="utf-8")

    with pytest.raises(SystemExit):
        pl.group_videos_by_animal_profile(str(tmp_path))


def test_stage_profile_videos_hardlinks_video_and_copies_metadata(tmp_path):
    source = tmp_path / "source" / "record"
    source.mkdir(parents=True)
    video = source / "video.mp4"
    metadata = source / "metadata.json"
    video.write_bytes(b"video")
    metadata.write_text('{"animal_profile":"pet"}', encoding="utf-8")
    target = tmp_path / "stage"

    pl.stage_profile_videos([(video, metadata)], target, species="pet", batch_id="2026-10-05")

    staged_video = target / "00000000" / "video.mp4"
    staged_metadata = target / "00000000" / "metadata.json"
    assert os.stat(staged_video).st_ino == os.stat(video).st_ino
    assert staged_metadata.read_text(encoding="utf-8") == metadata.read_text(encoding="utf-8")
    from scripts.pipeline.lineage import store
    ingests = store.find_records(store.INGEST_PATH,
                                 raw_path=os.path.abspath(staged_video))
    assert len(ingests) == 1
    assert ingests[-1]["source_path"] == os.path.abspath(video)
    assert ingests[-1]["species"] == "pet"
    assert ingests[-1]["batch_id"] == "2026-10-05"


def test_auto_profile_batch_runs_larger_group_first_with_species_defaults(
    monkeypatch, tmp_path
):
    root = tmp_path / "raw"
    root.mkdir()
    records = [(Path("video"), Path("metadata"))]
    monkeypatch.setattr(pl, "group_videos_by_animal_profile", lambda _root: {
        "pet": records,
        "rabbit": records * 2,
    })
    monkeypatch.setattr(pl, "stage_profile_videos", lambda _records, stage, *, species, batch_id: str(stage))
    monkeypatch.setattr(pl, "resolve_videos_dir", lambda path: path)
    jobs = []
    monkeypatch.setattr(pl, "run_job", jobs.append)

    pl.run_auto_profile_batch(str(root), "2026-10-05")

    assert [job.species for job in jobs] == ["rabbit", "pet"]
    assert all(job.frames is False for job in jobs)
    assert all(job.date == "2026-10-05" for job in jobs)
    assert all(job.teacher_checkpoint == pl.build_retrain_context(
        job.species, job.date
    )["teacher_checkpoint"] for job in jobs)


def test_ask_job_inputs_uses_default_teacher_and_clip_mode(tmp_path):
    (tmp_path / "a.mp4").write_bytes(b"x")
    ctx = build_retrain_context("rabbit", "2026-09-10")
    answers = iter([str(tmp_path), "", ""])
    job = pl.ask_job_inputs(ctx, input_fn=lambda _p: next(answers))
    assert job.teacher_checkpoint == ctx["teacher_checkpoint"]
    assert job.frames is False
    assert job.videos_dir == str(tmp_path)


def test_ask_job_inputs_frames_mode_and_pasted_checkpoint(tmp_path):
    (tmp_path / "a.mp4").write_bytes(b"x")
    ctx = build_retrain_context("rabbit", "2026-09-10")
    answers = iter([str(tmp_path), '"E:\\ckpt\\teacher.pth"', "2"])
    job = pl.ask_job_inputs(ctx, input_fn=lambda _p: next(answers))
    assert job.teacher_checkpoint == "E:\\ckpt\\teacher.pth"
    assert job.frames is True


# ---------- stage 3: review gate ----------

class _FakeServer:
    def __init__(self, die=False):
        self.returncode = 0 if not die else 1
        self.terminated = False
        self._die = die

    def poll(self):
        return 1 if self._die else None

    def terminate(self):
        self.terminated = True

    def wait(self, timeout=None):
        return 0


def test_poll_review_state_finished_returns(tmp_path):
    state = tmp_path / "state.json"
    state.write_text(json.dumps(
        {"status": "finished", "stats": {"approved_videos": 2, "labeled_videos": 3,
                                         "total_videos": 4}}),
        encoding="utf-8")
    pl.poll_review_state(state, _FakeServer())  # must not raise


def test_poll_review_state_server_died_aborts(tmp_path):
    with pytest.raises(SystemExit):
        pl.poll_review_state(tmp_path / "missing.json", _FakeServer(die=True))


def test_poll_review_state_pending_then_finished(tmp_path, monkeypatch):
    state = tmp_path / "state.json"
    state.write_text(json.dumps({"status": "pending", "stats": {}}),
                     encoding="utf-8")
    sleeps = []

    def fake_sleep(_s):
        sleeps.append(_s)
        state.write_text(json.dumps(
            {"status": "finished", "stats": {"approved_videos": 1}}),
            encoding="utf-8")

    monkeypatch.setattr(pl.time, "sleep", fake_sleep)
    pl.poll_review_state(state, _FakeServer())
    assert sleeps  # polled at least once before finishing


def test_run_review_stops_server_after_finish(tmp_path, monkeypatch):
    videos = tmp_path / "videos"
    videos.mkdir()
    state = tmp_path / "_review_state.json"
    state.write_text(json.dumps({"status": "finished", "stats": {}}),
                     encoding="utf-8")

    launched = {}

    class _Proc(_FakeServer):
        stdout = iter(["[OK] Review server running", ""])  # type: ignore[var-annotated]

    def fake_popen(cmd, **_kw):
        launched["cmd"] = cmd
        return _Proc()

    monkeypatch.setattr(pl.subprocess, "Popen", fake_popen)
    pl.run_review(str(videos), state)
    assert launched["cmd"][1:3] == ["-m", "scripts.presort.auto_presort_review"]
    assert "--state-file" in launched["cmd"]
    assert launched["cmd"][launched["cmd"].index("--dir") + 1] == str(videos)


# ---------- stage 4 ----------

def test_run_apply_and_crop_aborts_on_zero_clips(tmp_path, monkeypatch):
    job = JobSpec(species="rabbit", date="2026-09-10", videos_dir="V",
                  teacher_checkpoint="t.pth")
    with mock.patch("scripts.presort.auto_presort_apply.run_apply",
                    return_value={"written": [], "skipped": ["v.mp4"],
                                  "warnings": []}):
        with pytest.raises(SystemExit):
            pl.run_apply_and_crop(job, str(tmp_path / "sorted"),
                                  str(tmp_path / "snap"))


def test_run_apply_and_crop_clip_mode_builds_cfg(tmp_path, monkeypatch):
    job = JobSpec(species="rabbit", date="2026-09-10", videos_dir="V",
                  teacher_checkpoint="t.pth")
    sorted_dir = str(tmp_path / "sorted")
    snap = str(tmp_path / "snap")

    with mock.patch("scripts.presort.auto_presort_apply.run_apply",
                    return_value={"written": ["x.mp4"], "skipped": [],
                                  "warnings": []}), \
         mock.patch("scripts.pipeline.run_crop.validate_sorted") as m_val, \
         mock.patch("scripts.pipeline.run_crop.assert_snapshot_empty") as m_emp:
        fake_crop = SimpleNamespace(
            default_config=lambda: {},
            run=lambda cfg: None)
        captured = {}
        fake_crop.run = lambda cfg: captured.update(cfg)
        import utils as utils_pkg
        # 同时替换包属性与 sys.modules：utils.abnormal_crop_video 可能已被
        # 其他测试真实 import，仅 setitem sys.modules 会被包属性短路。
        monkeypatch.setattr(utils_pkg, "abnormal_crop_video", fake_crop,
                            raising=False)
        monkeypatch.setitem(__import__("sys").modules,
                            "utils.abnormal_crop_video", fake_crop)
        got = pl.run_apply_and_crop(job, sorted_dir, snap)

    assert got == snap
    m_val.assert_called_once()
    m_emp.assert_called_once_with(snap)
    assert captured["input_base_dir"] == sorted_dir
    assert captured["output_base_dir"] == snap
    assert captured["yolo_model_path"].endswith(".pt")
    assert "label_map" not in captured  # clip mode carries no label_map


# ---------- stage 5 ----------

def _report(per_class, **kw):
    base = {"per_class": per_class, "cross_split_leak": [], "unrecognized": [],
            "label_mismatch": [], "source_leak": []}
    base.update(kw)
    return base


def test_check_health_gates_aborts_on_empty_class():
    with pytest.raises(SystemExit):
        pl.check_health_gates("5/6", _report({"eat": 3, "drink": 0}),
                              ["eat", "drink"])


def test_check_health_gates_warns_but_continues():
    rep = _report({"eat": 3, "drink": 2},
                  cross_split_leak=["a.mp4"], unrecognized=["junk.bin"])
    pl.check_health_gates("5/6", rep, ["eat", "drink"])  # no raise


def test_run_merge_uses_unattended_mode(tmp_path, monkeypatch):
    job = JobSpec(species="rabbit", date="2026-09-10", videos_dir="V",
                  teacher_checkpoint="t.pth")
    ctx = build_retrain_context("rabbit", "2026-09-10")
    calls = {}

    def fake_merge(root, date, snap, frames, classes, assume_yes=False):
        calls["assume_yes"] = assume_yes
        calls["root"] = root
        return _report({c: 5 for c in classes})

    monkeypatch.setattr(pl, "merge_and_report", fake_merge)
    pl.run_merge(job, ctx, str(tmp_path / "snap"))
    assert calls["assume_yes"] is True
    assert calls["root"] == ctx["dataset_root"]


def test_run_merge_no_old_snapshot_falls_back_to_health(tmp_path, monkeypatch):
    job = JobSpec(species="rabbit", date="2026-09-10", videos_dir="V",
                  teacher_checkpoint="t.pth")
    ctx = build_retrain_context("rabbit", "2026-09-10")

    import scripts.pipeline.snapshot_merge as sm
    monkeypatch.setattr(pl, "merge_and_report", lambda *_a, **_k: None)
    monkeypatch.setattr(sm, "health_report",
                        lambda *_a, **_k: _report({c: 1 for c in ctx["classes"]}))

    report = pl.run_merge(job, ctx, str(tmp_path / "snap"))
    assert report["per_class"]["resting"] == 1


# ---------- stage 6 wiring (run_train_distill reuse) ----------

def _patch_ctx_builder(monkeypatch, tmp_path, wrap=None):
    """main() 会重建 ctx；把 build_retrain_context 的 workspace/state_file
    重定向到 tmp_path，避免测试写进真实数据目录。wrap 可再改 frames 等字段。"""
    def fake(species, date, frames=False):
        ctx = build_retrain_context(species, date, frames=frames)
        ctx["workspace"] = str(tmp_path / "ws")
        ctx["state_file"] = str(tmp_path / "ws" / "state.json")
        if wrap:
            wrap(ctx, frames)
        return ctx

    monkeypatch.setattr(pl, "build_retrain_context", fake)


def test_pipeline_main_wires_six_stages(monkeypatch, tmp_path):
    """main() 的编排顺序：job -> label -> review -> apply+crop -> merge -> train。"""
    _patch_ctx_builder(monkeypatch, tmp_path)
    calls = []

    monkeypatch.setattr(pl, "ask_retrain_context", lambda: {})
    monkeypatch.setattr(pl, "ask_job_inputs", lambda _c, input_fn=None: JobSpec(
        species="rabbit", date="2026-09-10", videos_dir="V",
        teacher_checkpoint="t.pth"))
    monkeypatch.setattr(pl, "run_subprocess",
                        lambda stage, cmd: calls.append(stage))
    monkeypatch.setattr(pl, "run_review",
                        lambda *_a: calls.append("review"))
    monkeypatch.setattr(pl, "run_apply_and_crop",
                        lambda *a: calls.append("apply+crop") or a[2])
    monkeypatch.setattr(pl, "run_merge", lambda *a: calls.append("merge"))
    monkeypatch.setattr(pl, "train_and_distill",
                        lambda *a, **k: calls.append("train") or "teacher.pth")

    pl.main()
    assert calls == ["2/6 auto label", "review", "apply+crop", "merge", "train"]
    assert (tmp_path / "ws" / "job.json").is_file()


def test_pipeline_main_rebuilds_ctx_for_frames_mode(monkeypatch, tmp_path):
    """frames 模式必须重算快照路径（不能用 ask_retrain_context 的 clip 默认）。"""
    from scripts.pipeline import pipeline_config as pc

    captured = {}
    _patch_ctx_builder(monkeypatch, tmp_path)
    job = JobSpec(species="rabbit", date="2026-09-10", videos_dir="V",
                  teacher_checkpoint="t.pth", frames=True)

    monkeypatch.setattr(pl, "ask_retrain_context", lambda: {})
    monkeypatch.setattr(pl, "ask_job_inputs", lambda _c, input_fn=None: job)
    monkeypatch.setattr(pl, "run_subprocess", lambda *a: None)
    monkeypatch.setattr(pl, "run_review", lambda *a: None)
    monkeypatch.setattr(pl, "run_apply_and_crop",
                        lambda j, sorted_dir, snap: captured.setdefault("snap", snap))
    monkeypatch.setattr(pl, "run_merge", lambda *a: None)
    monkeypatch.setattr(pl, "train_and_distill", lambda *a, **k: "t.pth")

    pl.main()
    expected = pc.snapshot_dir("rabbit", "2026-09-10", frames=True)
    assert captured["snap"] == expected
    assert not captured["snap"].endswith("videos")


def test_pipeline_main_clears_stale_review_state(monkeypatch, tmp_path):
    """上次运行残留的 finished 状态文件必须清掉，否则闸门被静默跳过。"""
    _patch_ctx_builder(monkeypatch, tmp_path)
    state = Path(tmp_path / "ws" / "state.json")
    state.parent.mkdir(parents=True)
    state.write_text(json.dumps({"status": "finished", "stats": {}}),
                     encoding="utf-8")

    monkeypatch.setattr(pl, "ask_retrain_context", lambda: {})
    monkeypatch.setattr(pl, "ask_job_inputs", lambda _c, input_fn=None: JobSpec(
        species="rabbit", date="2026-09-10", videos_dir="V",
        teacher_checkpoint="t.pth"))
    for name in ("run_subprocess", "run_review", "run_apply_and_crop",
                 "run_merge", "train_and_distill"):
        monkeypatch.setattr(pl, name, lambda *a, **k: None)

    pl.main()
    assert not state.exists()
