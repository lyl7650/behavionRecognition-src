"""问答式包装：物种/机构/日期 → 对同一 records_dir 开审核服务器。"""
import sys
from scripts.pipeline import pipeline_config as pc


def build_review_argv(ctx):
    records = pc.records_dir(ctx["species"], ctx["site"], ctx["date"])
    return ["--dir", records]


def main():
    ctx = pc.ask_context()
    argv = build_review_argv(ctx)
    from . import auto_presort_review
    return auto_presort_review.main(argv)


if __name__ == "__main__":
    sys.exit(main() or 0)
