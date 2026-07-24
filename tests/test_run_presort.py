from scripts.pipeline import pipeline_config as pc
from scripts.presort import run_presort_apply as apl
from scripts.presort import run_presort_label as lbl
from scripts.presort import run_presort_review as rvw


def test_resolve_checkpoint_uses_default_on_blank():
    got = lbl.resolve_checkpoint("pet", input_fn=lambda p: "")
    assert got == pc.SPECIES["pet"]["teacher_checkpoint"]


def test_resolve_checkpoint_uses_pasted_override():
    got = lbl.resolve_checkpoint("rabbit", input_fn=lambda p: r"D:\tmp\new.pth")
    assert got == r"D:\tmp\new.pth"


def test_resolve_checkpoint_strips_surrounding_quotes():
    # Windows "复制文件地址" 粘贴出来带双引号，必须剥掉否则 isfile 失败
    got = lbl.resolve_checkpoint("pet", input_fn=lambda p: '"D:\\tmp\\new.pth"')
    assert got == "D:\\tmp\\new.pth"


def test_build_label_argv_derives_records_dir():
    ctx = {"species": "rabbit", "site": "hkbr", "date": "2026-07-15"}
    argv = lbl.build_label_argv(ctx, r"D:\tmp\new.pth")
    expected_dir = pc.records_dir("rabbit", "hkbr", "2026-07-15")
    assert argv == ["--species", "rabbit", "--dir", expected_dir,
                    "--checkpoint", r"D:\tmp\new.pth"]


def test_build_review_argv_derives_records_dir():
    ctx = {"species": "pet", "site": "fotan", "date": "2026-07-15"}
    argv = rvw.build_review_argv(ctx)
    expected_dir = pc.records_dir("pet", "fotan", "2026-07-15")
    assert argv == ["--dir", expected_dir]


def test_build_apply_argv_derives_in_and_out_dirs():
    ctx = {"species": "rabbit", "site": "hkbr", "date": "2026-07-15"}
    argv = apl.build_apply_argv(ctx)
    in_dir = pc.records_dir("rabbit", "hkbr", "2026-07-15")
    out_dir = pc.sorted_dir("rabbit", "hkbr", "2026-07-15")
    assert argv == ["--dir", in_dir, "--out", out_dir]
