"""Overnight orchestrator: wait for downloads -> label 8B -> label 32B ->
static report -> serve the comparison page. Designed to run DETACHED so it
survives the launching session. Safe to re-run (per-clip JSON resume).

Run (PowerShell, detached):
  Start-Process -WindowStyle Hidden <venv python> -ArgumentList \
    '-m','scripts.vlm_label.run_overnight' -WorkingDirectory <project root> \
    -RedirectStandardOutput logs\vlm_overnight.log \
    -RedirectStandardError logs\vlm_overnight.err.log
"""
import json
import os
import subprocess
import sys
import threading
import time

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
VENV_PY = os.path.join(PROJECT_ROOT, ".venv-vlm", "Scripts", "python.exe")
BATCH_DIR = r"G:\aws_backup\2026-10-07\raw\rabbit"
HF_ENDPOINT = "https://hf-mirror.com"
HF_HOME = os.path.join(PROJECT_ROOT, ".hf_cache")
PORT = 8767
# v3 = 9-class spec (rabbit-9class-json-v1, schema vlm-qwen3vl-3). New tags
# write fresh doc dirs so the schema-v2 results (.vlm-qwen3vl-8b/-32b/...)
# stay untouched for comparison on the review page.
TAG_A, TAG_B = "qwen3vl-8b-v3", "qwen3vl-32b-v3"
# only label the first N sorted clips of BATCH_DIR (resume skips any clip
# that already has a doc, so existing results are never recomputed)
LIMIT = 50
# third comparison panel: .autopresort docs from the 10-06 scene-retrain
# batch (same clip naming -> mapped by rel_path onto the page videos)
AP_DIR = r"G:\aws_backup\2026-10-06\scene-retrain-28wfyca8\rabbit\.autopresort"
AP_LEVEL = "0.5"
# second VLM round: YOLO-cropped input (rabbit box * expand 2, miss ->
# full frame), same 50 filtered clips, docs go to .vlm-<tag>-crop
TAG_A_CROP, TAG_B_CROP = "qwen3vl-8b-v3-crop", "qwen3vl-32b-v3-crop"
# third prompt: compact 9-class one-liner (schema v4, label+confidence
# only, ~15 output tokens/window -> ~20x faster than the full-spec JSON);
# full-frame only, docs go to .vlm-<tag>-v4
TAG_A2, TAG_B2 = "qwen3vl-8b-v4", "qwen3vl-32b-v4"
YOLO_RABBIT = (r"E:\ActionRecognition\detection-src\rabbit-detect\runs"
               r"\rabbit-yolo11n-0603\weights\rabbit-yolo11n-0603.pt")
YOLO_EXPAND = 2.0
MODEL_A = os.path.join(PROJECT_ROOT, "models-vlm", "qwen3-vl-8b")
MODEL_B = os.path.join(PROJECT_ROOT, "models-vlm", "qwen3-vl-32b")
GIT_BASH = r"C:\Users\PPCUSER\.workbuddy\binaries\PortableGit\versions\1.2.0\bin\bash.exe"
DOWNLOAD_SH = os.path.join(PROJECT_ROOT, "scripts", "vlm_label", "curl_download.sh")
REPO_B = "Qwen/Qwen3-VL-32B-Instruct"
LIST_B = os.path.join(PROJECT_ROOT, "logs", "list_32b_lf.tsv")


def _pick_log_path():
    """Files created inside the WorkBuddy sandbox can be locked for normal
    (user-launched) processes; probe candidates and use the first writable."""
    candidates = [os.environ.get("VLM_ORCH_LOG"),
                  os.path.join(PROJECT_ROOT, "logs", "vlm_overnight.log")]
    for c in candidates:
        if not c:
            continue
        try:
            with open(c, "a", encoding="utf-8"):
                return c
        except OSError:
            continue
    return os.path.join(PROJECT_ROOT, "logs",
                        f"vlm_overnight_{time.strftime('%Y%m%d_%H%M%S')}.log")


LOG_PATH = _pick_log_path()


def log(msg):
    print(msg, flush=True)
    try:
        with open(LOG_PATH, "a", encoding="utf-8") as f:
            f.write(f"[{time.strftime('%m-%d %H:%M:%S')}] {msg}\n")
    except OSError:
        pass


def marker(model_dir):
    return os.path.join(model_dir, ".download_done")


def wait_for_marker(model_dir, timeout_s):
    mp = marker(model_dir)
    name = os.path.basename(model_dir)
    t0 = time.time()
    if os.path.isfile(mp):
        log(f"marker exists for {name}")
        return True
    log(f"waiting for download marker {mp} (timeout {timeout_s / 3600:.1f}h)")
    while time.time() - t0 < timeout_s:
        if os.path.isfile(mp):
            log(f"download finished for {name} after {time.time() - t0:.0f}s")
            return True
        time.sleep(30)
    log(f"[ERROR] timeout waiting for {name} download")
    return False


def next_morning_7am():
    import datetime
    now = datetime.datetime.now()
    d = (now + datetime.timedelta(days=1)).replace(hour=7, minute=0, second=0,
                                                   microsecond=0)
    return d.timestamp()


LOCK_PATH = os.path.join(PROJECT_ROOT, "logs", "vlm_overnight.lock")


def pid_alive(pid):
    try:
        out = subprocess.run(["tasklist", "/FI", f"PID eq {pid}"],
                             capture_output=True, text=True).stdout
        return str(pid) in out
    except Exception:  # noqa: BLE001
        return False


def acquire_lock():
    """Single-instance guard; stale locks from dead PIDs are taken over."""
    global LOCK_PATH
    candidates = [LOCK_PATH,
                  os.path.join(PROJECT_ROOT, "logs",
                               f"vlm_overnight_{os.getpid()}.lock")]
    for c in candidates:
        try:
            old = 0
            if os.path.isfile(c):
                with open(c, encoding="utf-8") as f:
                    old = int(f.read().strip() or 0)
                if old and pid_alive(old):
                    log(f"another orchestrator (pid {old}) is running; exiting")
                    return False
                log(f"stale lock from pid {old}; taking over")
            with open(c, "w", encoding="utf-8") as f:
                f.write(str(os.getpid()))
            LOCK_PATH = c
            return True
        except (ValueError, OSError):
            continue
    log("[WARN] cannot write lock file; continuing without single-instance guard")
    return True


def log_vram(tag):
    """Best-effort VRAM snapshot before a round -- the 10-09 overnight run
    showed 32B NF4 spilling into WDDM shared memory (23.8/24.5 GiB used,
    100% util, ~400s/window) when desktop apps crowded the GPU."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.free,memory.total",
             "--format=csv,noheader"], capture_output=True, text=True,
            timeout=15).stdout.strip().replace("\n", " | ")
        log(f"vram[{tag}] free/total: {out}")
    except Exception as e:  # noqa: BLE001 - diagnostics only
        log(f"vram[{tag}] query failed: {e}")


def run_label(tag, model_id, dtype, gpu_mem_gib, deadline_epoch,
              yolo=False, min_duration=0.0, prompt="full"):
    env = dict(os.environ)
    env.update({
        "HF_ENDPOINT": HF_ENDPOINT, "HF_HOME": HF_HOME,
        "HF_HUB_OFFLINE": "1", "VLM_DEADLINE_EPOCH": str(deadline_epoch),
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        "PYTHONIOENCODING": "utf-8",
    })
    args = [VENV_PY, "-m", "scripts.vlm_label.vlm_label",
            "--dir", BATCH_DIR, "--tag", tag, "--model-id", model_id,
            "--dtype", dtype, "--limit", str(LIMIT), "--prompt", prompt]
    if gpu_mem_gib:
        args += ["--gpu-mem-gib", str(gpu_mem_gib)]
    if yolo:
        args += ["--yolo-crop", "--yolo-weights", YOLO_RABBIT,
                 "--yolo-expand", str(YOLO_EXPAND)]
    if min_duration:
        args += ["--min-duration", str(min_duration)]
    log_vram(tag)
    log(f"=== labeling start: {tag} ({model_id}, {dtype}, limit={LIMIT}, "
        f"yolo={yolo}, prompt={prompt}) ===")
    t0 = time.time()
    r = subprocess.run(args, cwd=PROJECT_ROOT, env=env)
    log(f"=== labeling end: {tag} rc={r.returncode} in {time.time() - t0:.0f}s ===")
    return r.returncode


def serve_review():
    from scripts.vlm_label import vlm_review
    try:
        vlm_review.serve(
            dir_path=BATCH_DIR, port=PORT, in_thread=True,
            ap_dir=AP_DIR, ap_level=AP_LEVEL, ap_limit=LIMIT,
            a_name=f".vlm-{TAG_A}", b_name=f".vlm-{TAG_B}",
            c_name=f".vlm-{TAG_A_CROP}", d_name=f".vlm-{TAG_B_CROP}",
            v4_a_name=f".vlm-{TAG_A2}", v4_b_name=f".vlm-{TAG_B2}")
        log(f"review server serving on http://127.0.0.1:{PORT} "
            f"(ap={AP_DIR or 'off'}, level={AP_LEVEL}, limit={LIMIT}, "
            f"crop=[.vlm-{TAG_A_CROP}, .vlm-{TAG_B_CROP}], "
            f"v4=[.vlm-{TAG_A2}, .vlm-{TAG_B2}])")
    except Exception as e:  # noqa: BLE001
        log(f"[ERROR] review server failed: {e}")


def ensure_download_b():
    """If the 32B model is not fully downloaded, start the resumable curl
    downloader as a child process; wait_for_marker polls its .download_done."""
    if os.path.isfile(marker(MODEL_B)):
        log("32B marker exists; no download needed")
        return
    log(f"32B marker missing; starting curl downloader -> {MODEL_B}")
    subprocess.Popen([GIT_BASH, DOWNLOAD_SH, REPO_B, MODEL_B, LIST_B])


def main():
    os.makedirs(os.path.join(PROJECT_ROOT, "logs"), exist_ok=True)
    log("orchestrator starting")
    if not acquire_lock():
        return 0
    deadline = next_morning_7am()
    log(f"labeling deadline epoch={deadline:.0f} "
        f"(~07:00 next morning)")

    if not wait_for_marker(MODEL_A, timeout_s=3 * 3600):
        pass  # continue anyway; labeling will fail loudly in its own log
    serve_review()  # start early so the comparison page is watchable live
    ensure_download_b()  # overlap 32B download with 8B labeling
    rc_a = run_label(TAG_A, MODEL_A, "bf16", 0, deadline)
    if not wait_for_marker(MODEL_B, timeout_s=5 * 3600):
        pass
    # No mem cap for 32B: bnb 4bit refuses CPU-dispatched modules, so any
    # max_memory cap pushes leftovers to CPU and crashes the load (rc=1 in
    # 6s on 10-08). NF4 32B needs ~18 GiB and fits free VRAM uncapped.
    # compact prompt rounds BEFORE the slow 32B full-spec continuation:
    # ~15 output tokens/window keeps 32B at seconds-per-window, so the new
    # page tab fills quickly; the deadline then guards the slow rounds.
    rc_a2 = run_label(TAG_A2, MODEL_A, "bf16", 0, deadline, prompt="compact")
    rc_b2 = run_label(TAG_B2, MODEL_B, "bnb4", 0, deadline, prompt="compact")
    rc_b = run_label(TAG_B, MODEL_B, "bnb4", 0, deadline)

    # YOLO-cropped input on the same filtered 50 clips (full-spec prompt)
    rc_ac = run_label(TAG_A_CROP, MODEL_A, "bf16", 0, deadline,
                      yolo=True, min_duration=0.5)
    rc_bc = run_label(TAG_B_CROP, MODEL_B, "bnb4", 0, deadline,
                      yolo=True, min_duration=0.5)

    env = dict(os.environ)
    env.update({"HF_HOME": HF_HOME, "PYTHONIOENCODING": "utf-8"})
    r = subprocess.run([VENV_PY, "-m", "scripts.vlm_label.build_static_report",
                        "--dir", BATCH_DIR,
                        "--a-name", f".vlm-{TAG_A}",
                        "--b-name", f".vlm-{TAG_B}"], cwd=PROJECT_ROOT, env=env)
    log(f"static report rc={r.returncode}")

    serve_review()
    summary = {"rc_8b": rc_a, "rc_32b": rc_b,
               "rc_8b_crop": rc_ac, "rc_32b_crop": rc_bc,
               "rc_8b_v4": rc_a2, "rc_32b_v4": rc_b2,
               "port": PORT,
               "report": os.path.join(BATCH_DIR, ".vlm-report", "report.html"),
               "finished_at": time.strftime("%Y-%m-%d %H:%M:%S")}
    with open(os.path.join(PROJECT_ROOT, "logs", "vlm_overnight_done.json"),
              "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    log(f"ALL DONE {summary}; keeping review server alive")
    while True:
        time.sleep(3600)


if __name__ == "__main__":
    sys.exit(main())
