"""One-click orchestrator: PID lock -> serve the review page (8768).

Directory-mode review tool: open a folder from the page (or --dir), run
inference from the page (7 only), edit labels with the 2b interaction set.

  python -m scripts.vlm_screener.run_all                # serve (GPU ok)
  python -m scripts.vlm_screener.run_all --dir <path>   # prefill dir on page
  python -m scripts.vlm_screener.run_all --view-only    # no infer endpoints

The legacy fixed-pool pipeline is unchanged and can still be run manually:
  python -m scripts.vlm_screener.prepare && python -m scripts.vlm_screener.run_combos
"""
import argparse
import os
import subprocess
import sys
import time

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
LOG_DIR = os.path.join(PROJECT_ROOT, "logs")
LOCK_PATH = os.path.join(LOG_DIR, "vlm_screener.lock")
PORT = 8768


def log(msg):
    line = f"[{time.strftime('%m-%d %H:%M:%S')}] {msg}"
    print(line, flush=True)
    try:
        os.makedirs(LOG_DIR, exist_ok=True)
        with open(os.path.join(LOG_DIR, "vlm_screener.log"), "a",
                  encoding="utf-8") as f:
            f.write(line + "\n")
    except OSError:
        pass


def pid_alive(pid):
    try:
        out = subprocess.run(["tasklist", "/FI", f"PID eq {pid}"],
                             capture_output=True, text=True).stdout
        return str(pid) in out
    except Exception:  # noqa: BLE001
        return False


def acquire_lock():
    candidates = [LOCK_PATH,
                  os.path.join(LOG_DIR, f"vlm_screener_{os.getpid()}.lock")]
    for c in candidates:
        try:
            old = 0
            if os.path.isfile(c):
                with open(c, encoding="utf-8") as f:
                    old = int(f.read().strip() or 0)
                if old and old != os.getpid() and pid_alive(old):
                    log(f"another run_all (pid {old}) is alive; exiting")
                    return False
                if old:
                    log(f"stale lock from pid {old}; taking over")
            with open(c, "w", encoding="utf-8") as f:
                f.write(str(os.getpid()))
            return True
        except (ValueError, OSError):
            continue
    log("[WARN] cannot write lock; continuing without guard")
    return True


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--view-only", action="store_true",
                    help="no inference endpoints (no GPU needed)")
    ap.add_argument("--dir", default=None,
                    help="video directory to prefill on the review page")
    ap.add_argument("--port", type=int, default=PORT)
    ap.add_argument("--species", choices=("rabbit", "pet"), default="rabbit")
    a = ap.parse_args(argv)
    os.makedirs(LOG_DIR, exist_ok=True)
    log(f"run_all starting (species={a.species})")
    if not acquire_lock():
        return 0
    from scripts.vlm_screener.server import serve
    log(f"serving review page on http://127.0.0.1:{a.port}")
    serve(a.port, open_browser=True, target=a.dir, no_infer=a.view_only,
          species=a.species)
    return 0


if __name__ == "__main__":
    sys.exit(main())
