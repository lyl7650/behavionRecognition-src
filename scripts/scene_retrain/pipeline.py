"""编排器（在 AR 环境运行）：一次启动跑完全流程（扁平，无 site 维度）。

    python -m scripts.scene_retrain.pipeline

一批新视频进来，直接与旧快照合并训练；物种/日期之外不再询问场地。
阶段（控制台全程打印 [n/6 ...] 进度）：
  1/6 job setup    species/date + videos dir + teacher + clip/frame mode -> job.json
  2/6 auto label   scripts.presort.auto_presort_label subprocess (YOLO + teacher)
  3/6 review gate  scripts.presort.auto_presort_review subprocess; poll the
                   state file until the page's Finish button is pressed
  4/6 apply+crop   auto_presort_apply.run_apply -> YOLO-crop into the date snapshot
  5/6 merge+health backfill the latest old snapshot (unattended: warn, no pause);
                   hard-fail when any class ends up with zero samples
  6/6 train+distill config rewrite -> teacher finetune -> student distillation

唯一人工闸门是 3/6 的浏览器审核；Finish 之后全程无人值守。
必须从项目根运行（config.yaml 是 cwd 相对路径）。
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import NoReturn

from scripts.pipeline import pipeline_config as pc
from scripts.pipeline.run_train_distill import merge_and_report, train_and_distill
from scripts.scene_retrain.job_spec import JobSpec
from scripts.scene_retrain.paths import ask_retrain_context, build_retrain_context

REVIEW_PORT = int(os.environ.get("SCENE_RETRAIN_REVIEW_PORT", "8765"))
REPO_ROOT = Path(__file__).resolve().parents[2]


def log(msg: str) -> None:
    print(msg, flush=True)


def _fail(stage: str, reason: str) -> NoReturn:
    log(f"[{stage}] ABORT: {reason}")
    raise SystemExit(1)


def run_subprocess(stage: str, cmd: list[str]) -> None:
    """Run a child process in the same AR env, streaming its stdout."""
    log(f"[{stage}] launch: {' '.join(cmd)}")
    started = time.time()
    proc = subprocess.Popen(
        cmd, cwd=str(REPO_ROOT), stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, encoding="utf-8", errors="replace",
    )
    assert proc.stdout is not None

    def _forward() -> None:
        for line in proc.stdout:  # type: ignore[union-attr]
            log(f"[{stage}] {line.rstrip()}")

    forwarder = threading.Thread(target=_forward, daemon=True)
    forwarder.start()
    proc.wait()
    if proc.returncode != 0:
        _fail(stage, f"subprocess exited with code {proc.returncode}")
    log(f"[{stage}] done in {time.time() - started:.0f}s")


# ---------- stage 1: job setup ----------

def resolve_videos_dir(path_text: str) -> str:
    """校验视频批次目录（沿用 auto_presort 的扩展名与隐藏目录约定）。"""
    from scripts.presort.auto_presort_label import collect_videos

    path = Path(path_text.strip().strip('"'))
    if not path.is_dir():
        _fail("1/6 job setup", f"videos dir does not exist: {path}")
    rels = collect_videos(str(path))
    if not rels:
        _fail("1/6 job setup", f"no videos under: {path}")
    log(f"[1/6 job setup] {len(rels)} video(s) found under {path}")
    return str(path)


def ask_job_inputs(ctx: dict, input_fn=None) -> JobSpec:
    if input_fn is None:
        input_fn = input
    videos_dir = resolve_videos_dir(input_fn("Videos dir (raw batch folder): "))
    ckpt = input_fn(
        f"Teacher checkpoint [default {ctx['teacher_checkpoint']}]: "
    ).strip().strip('"')
    mode = input_fn("Snapshot mode: 1) video clips (default) 2) frames [1]: ").strip()
    frames = mode == "2"
    return JobSpec(
        species=ctx["species"], date=ctx["date"], videos_dir=videos_dir,
        teacher_checkpoint=ckpt or ctx["teacher_checkpoint"],
        classes=list(ctx["classes"]), frames=frames,
    )


# ---------- stage 3: review gate ----------

def _review_counts(state: dict) -> str:
    stats = state.get("stats", {})
    return (f"approved {stats.get('approved_videos', 0)}"
            f" / labeled {stats.get('labeled_videos', 0)}"
            f" / total {stats.get('total_videos', 0)}")


def poll_review_state(state_file: Path, server) -> None:
    """轮询状态文件直到 finished；server 进程先死则中止。

    独立成函数便于单测（mock server 与状态文件）。
    """
    stage = "3/6 review gate"
    while True:
        if server.poll() is not None:
            _fail(stage, f"review server died unexpectedly (code {server.returncode})")
        if state_file.exists():
            with open(state_file, encoding="utf-8") as f:
                state = json.load(f)
            if state.get("status") == "finished":
                log(f"[{stage}] review finished: {_review_counts(state)}")
                return
            log(f"[{stage}] waiting for human review... {_review_counts(state)}")
        else:
            log(f"[{stage}] waiting for review server to start...")
        time.sleep(30)


def run_review(videos_dir: str, state_file: Path) -> None:
    stage = "3/6 review gate"
    state_file.parent.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, "-m", "scripts.presort.auto_presort_review",
           "--dir", videos_dir, "--port", str(REVIEW_PORT),
           "--state-file", str(state_file)]
    log(f"[{stage}] starting review server (port {REVIEW_PORT})...")
    server = subprocess.Popen(
        cmd, cwd=str(REPO_ROOT), stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, encoding="utf-8", errors="replace",
    )
    assert server.stdout is not None

    def _forward() -> None:
        for line in server.stdout:  # type: ignore[union-attr]
            log(f"[{stage}] {line.rstrip()}")

    threading.Thread(target=_forward, daemon=True).start()
    try:
        log(f"[{stage}] review page: http://localhost:{REVIEW_PORT}")
        log(f"[{stage}] approve videos in the browser, then press the Finish button...")
        poll_review_state(state_file, server)
    finally:
        # review server is light (no GPU weights) but must release port 8765
        # and stop serving before the long unattended stages.
        if server.poll() is None:
            server.terminate()
            try:
                server.wait(timeout=10)
            except subprocess.TimeoutExpired:
                server.kill()
        log(f"[{stage}] review server stopped")


# ---------- stage 4: apply + crop ----------

def run_apply_and_crop(job: JobSpec, sorted_dir: str, snapshot_dir: str) -> str:
    stage = "4/6 apply+crop"
    from scripts.presort.auto_presort_apply import run_apply

    log(f"[{stage}] cutting approved segments into {sorted_dir} ...")
    result = run_apply(job.videos_dir, sorted_dir)
    if not result["written"]:
        _fail(stage, "no clips were produced (nothing approved?)")

    from scripts.pipeline.run_crop import assert_snapshot_empty, validate_sorted

    sp = pc.SPECIES[job.species]
    validate_sorted(sorted_dir, sp["class_names"])
    assert_snapshot_empty(snapshot_dir)

    if job.frames:
        from utils import abnormal_crop as crop
    else:
        from utils import abnormal_crop_video as crop
    cfg = crop.default_config()
    cfg.update({
        "input_base_dir": sorted_dir,
        "output_base_dir": snapshot_dir,
        "class_names": sp["class_names"],
        "yolo_model_path": sp["yolo_model"],
        "target_classes": sp["target_classes"],
    })
    if job.frames:
        cfg["label_map"] = sp["label_map"]
    log(f"[{stage}] YOLO-cropping into snapshot {snapshot_dir} ...")
    crop.run(cfg)
    log(f"[{stage}] done, snapshot: {snapshot_dir}")
    return snapshot_dir


# ---------- stage 5: merge + health ----------

def check_health_gates(stage: str, report: dict, classes: list[str]) -> None:
    """无人值守下的硬闸门：任何一类样本数为 0 就中止（多半是标签写错或整类缺失）。

    leak/mismatch 类问题沿用脚本 B 的语义：打印 [WARN] 继续（历史行为如此，
    人工模式也只是确认后放行）。
    """
    empty = [c for c in classes if report["per_class"].get(c, 0) == 0]
    if empty:
        _fail(stage, f"zero samples for class(es) {empty} after merge; "
                     f"refusing to train on a broken snapshot")
    if report["cross_split_leak"]:
        log(f"[{stage}] [WARN] cross-split leak: {report['cross_split_leak']}")
    if report.get("source_leak"):
        log(f"[{stage}] [WARN] source leak: {report['source_leak']}")
    if report["label_mismatch"]:
        log(f"[{stage}] [WARN] label mismatch: {report['label_mismatch']}")
    if report["unrecognized"]:
        log(f"[{stage}] [WARN] unclassifiable samples skipped: {report['unrecognized']}")


def run_merge(job: JobSpec, ctx: dict, snapshot_dir: str) -> dict:
    stage = "5/6 merge+health"
    from scripts.pipeline import snapshot_merge as sm

    log(f"[{stage}] backfilling latest old snapshot from {ctx['dataset_root']} ...")
    report = merge_and_report(
        ctx["dataset_root"], job.date, snapshot_dir, job.frames, job.classes,
        assume_yes=True)
    if report is None:
        report = sm.health_report(snapshot_dir, job.classes, job.frames)
        from scripts.pipeline.run_train_distill import print_health_report
        print_health_report(report, job.classes)
    check_health_gates(stage, report, job.classes)
    log(f"[{stage}] health checks passed")
    return report


# ---------- entry ----------

def main() -> None:
    ctx = ask_retrain_context()
    job = ask_job_inputs(ctx)
    # ask_retrain_context 在问模式之前建 ctx（默认 clip 快照）；模式确定后必须
    # 按 job.frames 重建，否则 frames 模式会裁进 <date>/videos 这个 clip 快照。
    ctx = build_retrain_context(job.species, job.date, frames=job.frames)
    workspace = Path(ctx["workspace"])
    workspace.mkdir(parents=True, exist_ok=True)
    job_path = workspace / "job.json"
    job.write(job_path)
    # 上次运行的 Finish 状态文件必须清掉：残留的 status=finished 会让本次的
    # Review 闸门被静默跳过（唯一人工环节失效）。
    state_file = Path(ctx["state_file"])
    if state_file.exists():
        state_file.unlink()
        log("[1/6 job setup] removed stale review state file from a previous run")
    log(f"=== Scene Retrain: {job.species} / {job.date} ===")
    log(f"workspace: {workspace}")

    run_subprocess("2/6 auto label", [
        sys.executable, "-m", "scripts.presort.auto_presort_label",
        "--species", job.species, "--dir", job.videos_dir,
        "--checkpoint", job.teacher_checkpoint,
        "--threshold", str(job.threshold),
        "--trim-margin", str(job.trim_margin),
    ])
    run_review(job.videos_dir, Path(ctx["state_file"]))
    snapshot_dir = run_apply_and_crop(job, ctx["sorted_dir"], ctx["snapshot_dir"])
    run_merge(job, ctx, snapshot_dir)
    teacher_path = train_and_distill(job.species, snapshot_dir, frames=job.frames)

    log("=== all done ===")
    log(f"teacher: {teacher_path}")
    log("[WARNING] the deployment artifact is the student's *_ema.pth "
        "(under outputs/<species>/<date>/), not the teacher checkpoint.")
    log("Next: evaluate it with 5_run_test.bat before deployment.")


if __name__ == "__main__":
    main()
