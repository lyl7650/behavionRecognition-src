"""问答式包装：物种/机构/日期 → records_dir 作输入、sorted_dir 作输出，切片落地。

sorted_dir 正是 3_run_crop 读取的输入目录，故 apply 完可直接接 run_crop。
"""
import sys
from scripts.pipeline import pipeline_config as pc


def build_apply_argv(ctx):
    records = pc.records_dir(ctx["species"], ctx["site"], ctx["date"])
    sorted_out = pc.sorted_dir(ctx["species"], ctx["site"], ctx["date"])
    return ["--dir", records, "--out", sorted_out]


def main():
    ctx = pc.ask_context()
    argv = build_apply_argv(ctx)
    from . import auto_presort_apply
    return auto_presort_apply.main(argv)


if __name__ == "__main__":
    sys.exit(main() or 0)
