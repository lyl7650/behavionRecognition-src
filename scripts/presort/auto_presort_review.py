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
    try:
        server = make_server(a.dir, port=a.port)
    except OSError:
        print(f"[ERROR] Port {a.port} is already in use.")
        print("        Another review window is probably still open. Close it and try again,")
        print(f"        or pass a different port:  python auto_presort_review.py --dir <dir> --port {a.port + 1}")
        return 1
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
    sys.exit(main(sys.argv[1:]) or 0)
