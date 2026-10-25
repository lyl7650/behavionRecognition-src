"""Build a self-contained static comparison report (no server needed).

Embeds the aggregated JSON into vlm_review.html as INLINE_DATA and points
the video player at file:/// URLs, so the page works from disk even if the
overnight review server is not running. Also writes a short summary.md.

Usage:
  python -m scripts.vlm_label.build_static_report --dir <batch_dir>
"""
import argparse
import json
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
HTML_PATH = os.path.join(PROJECT_ROOT, "scripts", "vlm_label", "vlm_review.html")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--a-name", default=".vlm-qwen3vl-8b")
    ap.add_argument("--b-name", default=".vlm-qwen3vl-32b")
    ap.add_argument("--old-a-name", default=".vlm-qwen3vl-8b",
                    help="legacy 5-class (v2 prompt) doc dir")
    ap.add_argument("--old-b-name", default=".vlm-qwen3vl-32b",
                    help="legacy 5-class (v2 prompt) doc dir")
    ap.add_argument("--v4-a-name", default=".vlm-qwen3vl-8b-v4",
                    help="compact 9-class (schema v4) doc dir")
    ap.add_argument("--v4-b-name", default=".vlm-qwen3vl-32b-v4",
                    help="compact 9-class (schema v4) doc dir")
    a = ap.parse_args(argv)

    from scripts.vlm_label.vlm_review import build_data
    data = build_data(a.dir, a.a_name, a.b_name,
                      old_a_name=a.old_a_name, old_b_name=a.old_b_name,
                      v4_a_name=a.v4_a_name, v4_b_name=a.v4_b_name)

    out_dir = os.path.join(a.dir, ".vlm-report")
    os.makedirs(out_dir, exist_ok=True)
    with open(HTML_PATH, encoding="utf-8") as f:
        html = f.read()
    file_base = "file:///" + a.dir.replace("\\", "/").rstrip("/")
    inline = (f'<script>window.INLINE_DATA = '
              f'{json.dumps(data, ensure_ascii=False)}; '
              f'window.FILE_BASE = "{file_base}";</script>\n')
    marker = "<body>\n"
    idx = html.find(marker)
    if idx < 0:
        print("[ERROR] <body> tag not found in review html")
        return 1
    html = html[:idx + len(marker)] + inline + html[idx + len(marker):]
    out_html = os.path.join(out_dir, "report.html")
    with open(out_html, "w", encoding="utf-8") as f:
        f.write(html)

    lines = ["# Qwen3-VL 8B vs 32B rabbit clip labeling", ""]
    st = data.get("status", {})
    for k, name in (("a", "Qwen3-VL-8B (bf16)"), ("b", "Qwen3-VL-32B (nf4)")):
        s = st.get(k)
        if not s:
            lines.append(f"- {name}: no results")
            continue
        lines.append(
            f"- {name}: {s['n_videos']} clips, {s.get('n_windows')} windows, "
            f"labels {s.get('label_counts')}, incomplete={s['incomplete']}, "
            f"finished {s.get('finished_at')}")
    lines += ["", f"- Review page (server): http://127.0.0.1:8767",
              f"- Static report: {out_html}", ""]
    out_md = os.path.join(out_dir, "summary.md")
    with open(out_md, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"[DONE] {out_html}\n[DONE] {out_md}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
