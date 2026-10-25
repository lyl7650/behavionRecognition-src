"""Download Qwen3-VL models into the shared HF cache (mirror-friendly).

Writes a <cache_root>/<repo-name>.done marker when finished so the
overnight orchestrator can wait for completion.

Usage:
  python download_models.py --repo Qwen/Qwen3-VL-8B-Instruct
Env:
  HF_ENDPOINT (e.g. https://hf-mirror.com), HF_HOME (big disk).
"""
import argparse
import json
import os
import sys
import time

IGNORE = ["original/*", "*.pth", "*.onnx", "*.gguf", "*.bin", "*.mp4",
          "*.jpg", "*.png", "*.webp"]


def marker_path(repo):
    cache = os.environ.get("HF_HOME", "")
    name = "models--" + repo.replace("/", "--")
    return os.path.join(cache, name + ".done") if cache else name + ".done"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    a = ap.parse_args()
    t0 = time.time()
    from huggingface_hub import snapshot_download
    path = snapshot_download(a.repo, ignore_patterns=IGNORE, max_workers=8)
    files = [f for f in os.listdir(path) if f.endswith(".safetensors")]
    total_gb = sum(os.path.getsize(os.path.join(path, f)) for f in files) / 1e9
    mp = marker_path(a.repo)
    with open(mp, "w", encoding="utf-8") as f:
        json.dump({"repo": a.repo, "snapshot": path,
                   "safetensors_gb": round(total_gb, 2)}, f)
    print(f"[DONE] {a.repo} -> {path} ({total_gb:.1f}GB safetensors) "
          f"in {time.time() - t0:.0f}s, marker {mp}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
