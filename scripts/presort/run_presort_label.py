"""问答式包装：物种/机构/日期 + teacher → 调 auto_presort_label。

records_dir（长视频下载区）自动作为 --dir；teacher 默认按物种取，回车用默认或粘贴覆盖。
纯路径推导抽成 build_label_argv/resolve_checkpoint 便于单测；main() 委派给现有 CLI。
"""
import sys
from scripts.pipeline import pipeline_config as pc


def resolve_checkpoint(species, input_fn=None):
    """回车用 SPECIES 默认；粘贴则用粘贴的路径（剥掉 Windows 复制路径常带的双引号）。"""
    if input_fn is None:
        input_fn = input
    default = pc.SPECIES[species]["teacher_checkpoint"]
    ans = input_fn(f"Teacher checkpoint [default {default}]: ").strip().strip('"')
    return ans if ans else default


def build_label_argv(ctx, checkpoint):
    records = pc.records_dir(ctx["species"], ctx["site"], ctx["date"])
    return ["--species", ctx["species"], "--dir", records,
            "--checkpoint", checkpoint]


def main():
    ctx = pc.ask_context()
    checkpoint = resolve_checkpoint(ctx["species"])
    argv = build_label_argv(ctx, checkpoint)
    from . import auto_presort_label
    return auto_presort_label.main(argv)


if __name__ == "__main__":
    sys.exit(main() or 0)
