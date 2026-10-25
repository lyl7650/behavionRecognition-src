import os
from scripts.pipeline import pipeline_config as pc


def test_species_table_has_both_species():
    assert set(pc.SPECIES.keys()) == {"pet", "rabbit"}


def test_pet_species_fields():
    pet = pc.SPECIES["pet"]
    assert pet["class_names"] == ["eat", "drink", "active", "resting"]
    assert pet["label_map"] == {"eat": 0, "drink": 1, "active": 2, "resting": 3}
    assert pet["target_classes"] == [1, 2]
    assert pet["path_token"] == "pet"
    assert pet["raw_root"] == r"dataset\raw_videos\pet\normal"
    assert pet["dataset_root"] == r"dataset\pet\normal"
    assert pet["yolo_model"].endswith("yolo11n-ssb-0710.pt")


def test_rabbit_species_fields():
    rb = pc.SPECIES["rabbit"]
    assert rb["class_names"] == ["resting", "active", "eating", "drinking"]
    assert rb["label_map"] == {"resting": 0, "active": 1, "eating": 2, "drinking": 3}
    assert rb["target_classes"] == [0]
    assert rb["path_token"] == "rabbit"
    assert rb["raw_root"] == r"dataset\raw_videos\rabbit"
    assert rb["dataset_root"] == r"dataset\rabbit"
    assert rb["yolo_model"].endswith(".pt") and "rabbit" in rb["yolo_model"]


def test_records_dir():
    got = pc.records_dir("rabbit", "hkbr", "2026-07-15")
    assert got == os.path.join(r"dataset\raw_videos\rabbit", "hkbr-record", "records", "2026-07-15")


def test_sorted_dir_and_class_dirs():
    base = pc.sorted_dir("pet", "hkbr", "2026-07-15")
    assert base == os.path.join(r"dataset\raw_videos\pet\normal", "hkbr-record", "sorted-2026-07-15")
    class_dirs = pc.sorted_class_dirs("pet", "hkbr", "2026-07-15")
    assert class_dirs == [os.path.join(base, c) for c in ["eat", "drink", "active", "resting"]]


def test_snapshot_dir_video_default_includes_videos_layer():
    got = pc.snapshot_dir("rabbit", "2026-07-15", frames=False)
    assert got == os.path.join(r"dataset\rabbit", "2026-07-15", "videos")


def test_snapshot_dir_frames_has_no_videos_layer():
    got = pc.snapshot_dir("rabbit", "2026-07-15", frames=True)
    assert got == os.path.join(r"dataset\rabbit", "2026-07-15")


def test_ask_choice_accepts_number_and_legacy_name():
    assert pc._ask_choice("Species", ["pet", "rabbit"], input_fn=lambda _p: "2") == "rabbit"
    assert pc._ask_choice("Species", ["pet", "rabbit"], input_fn=lambda _p: "pet") == "pet"


def test_ask_choice_retries_on_invalid():
    answers = iter(["x", "9", "1"])
    assert pc._ask_choice("Species", ["pet", "rabbit"], input_fn=lambda _p: next(answers)) == "pet"


def test_ask_context_collects_three_fields():
    # 2 = rabbit in the shared 1/2 menu
    answers = iter(["2", "hkbr", "2026-07-14"])
    ctx = pc.ask_context(input_fn=lambda prompt: next(answers), today_fn=lambda: "2026-07-15")
    assert ctx == {"species": "rabbit", "site": "hkbr", "date": "2026-07-14"}


def test_ask_context_date_defaults_to_today_when_blank():
    answers = iter(["1", "fotan", ""])
    ctx = pc.ask_context(input_fn=lambda prompt: next(answers), today_fn=lambda: "2026-07-15")
    assert ctx["date"] == "2026-07-15"
    assert ctx["species"] == "pet"


def test_ask_context_rejects_bad_species_then_retries():
    answers = iter(["dog", "rabbit", "hkbr", ""])
    ctx = pc.ask_context(input_fn=lambda prompt: next(answers), today_fn=lambda: "2026-07-15")
    assert ctx["species"] == "rabbit"


def test_ask_context_no_site_skips_site_prompt():
    answers = iter(["2", "2026-07-14"])
    ctx = pc.ask_context_no_site(input_fn=lambda prompt: next(answers), today_fn=lambda: "2026-07-15")
    assert ctx == {"species": "rabbit", "date": "2026-07-14"}


def test_ask_context_no_site_date_defaults_to_today():
    answers = iter(["1", ""])
    ctx = pc.ask_context_no_site(input_fn=lambda prompt: next(answers), today_fn=lambda: "2026-07-15")
    assert ctx == {"species": "pet", "date": "2026-07-15"}


def test_species_has_teacher_checkpoint_defaults():
    pet = pc.SPECIES["pet"]["teacher_checkpoint"]
    rabbit = pc.SPECIES["rabbit"]["teacher_checkpoint"]
    # Only the shape is asserted: the concrete dated checkpoint is a local
    # deployment choice that changes with each retrain batch.
    assert pet.endswith(".pth") and "pet_teacher_" in pet
    assert "outputs" in pet and "pet" in pet
    assert rabbit.endswith(".pth")
    assert "outputs" in rabbit and "rabbit" in rabbit
