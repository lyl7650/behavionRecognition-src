"""Standalone multi-dir review entry (stdlib only). Serves one or more other/ dirs as a
merged list — no pipeline_config, no conda, no GPU. This is the entry shipped inside a
colleague review-pack (launched as: --root data).

Two ways to say which dirs to serve:
  --root <dir>   discover every other/ dir under <dir> (recursively)
  --dir  <dir>   serve this exact batch dir; repeat the flag for several

Unlike auto_presort_review.py this serves through make_multi_server, so videos with no
model pre-labels still show up in the sidebar's default filter.
Usage: python auto_presort_review_multidir.py --root data [--port 8765]
"""
import argparse
import sys
import webbrowser

from auto_presort.other_split import find_other_dirs
from auto_presort.review_server import make_multi_server


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=None, help="discover every other/ dir under this root")
    ap.add_argument("--dir", action="append", default=[], dest="dirs",
                    help="serve this exact batch dir; repeat the flag for several")
    ap.add_argument("--port", type=int, default=8765)
    a = ap.parse_args(argv)

    dirs = list(a.dirs)
    if a.root:
        dirs.extend(find_other_dirs(a.root))
    if not dirs:
        print("[ERROR] No dirs to serve. Pass --root <dir> or one/more --dir <dir>.")
        return 1
    print(f"[INFO] Serving {len(dirs)} dir(s)")

    try:
        server = make_multi_server(dirs, port=a.port)
    except OSError:
        print(f"[ERROR] Port {a.port} is already in use. Close the other window and retry.")
        return 1
    url = f"http://127.0.0.1:{server.server_address[1]}/"
    print(f"[OK] Review server running at {url}")
    print("[NOTE] Ctrl+drag on the timeline to draw a segment; click a color to label it; Enter to approve.")
    print("Press Ctrl+C to stop.")
    webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[OK] Server stopped.")


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]) or 0)
