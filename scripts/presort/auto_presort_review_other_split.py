"""Step 2b2 of the other-split workflow: review server across ALL scattered other/ dirs.

Light entry point: stdlib only (no torch/YOLO). Reviews every other/ directory under
the pet raw_root in one browser session instead of one server run per site/date folder.
Usage: python auto_presort_review_other_split.py [--root <dir>] [--port 8765]
"""
import argparse
import sys
import webbrowser

from auto_presort.other_split import find_other_dirs


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=None,
                    help="raw root to scan (default: pet raw_root from pipeline_config)")
    ap.add_argument("--port", type=int, default=8765)
    a = ap.parse_args(argv)

    from scripts.pipeline import pipeline_config as pc
    root = a.root or pc.SPECIES["pet"]["raw_root"]

    other_dirs = find_other_dirs(root)
    if not other_dirs:
        print(f"[ERROR] No 'other' directories found under: {root}")
        return 1
    print(f"[INFO] Reviewing {len(other_dirs)} 'other' dir(s) under {root}")

    from auto_presort.review_server import make_multi_server
    try:
        server = make_multi_server(other_dirs, port=a.port)
    except OSError:
        print(f"[ERROR] Port {a.port} is already in use.")
        print("        Another review window is probably still open. Close it and try again,")
        print(f"        or pass a different port:  python auto_presort_review_other_split.py --port {a.port + 1}")
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
