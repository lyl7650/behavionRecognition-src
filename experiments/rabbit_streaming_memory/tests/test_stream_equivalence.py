import importlib.util
from pathlib import Path

import torch

LAB = Path(__file__).resolve().parents[1]
MODEL_PATH = LAB / "src/timm_gru_student.py"
SPEC = importlib.util.spec_from_file_location("experiment_student", MODEL_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_stream_step_outputs_match_sequence_forward():
    torch.manual_seed(20261024)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MODULE.TimmGRUStudent("repvit_m1", 4, hidden_dim=256, pretrained=False)
    model = model.to(device).eval()
    clip = torch.randn(1, 3, 3, 224, 224, device=device)

    with torch.inference_mode():
        expected, _ = model.forward_stream(clip)
        hidden = None
        actual = []
        for index in range(clip.shape[1]):
            logits, hidden = model(clip[:, index], hidden)
            actual.append(logits)
        actual = torch.stack(actual, dim=1)

    torch.testing.assert_close(actual, expected, rtol=2e-4, atol=2e-6)


def test_stream_helpers_preserve_existing_checkpoint_keys():
    model = MODULE.TimmGRUStudent("repvit_m1", 4, hidden_dim=256, pretrained=False)
    keys = model.state_dict()
    assert "backbone.stem.conv1.c.weight" in keys
    assert "gru.weight_ih_l0" in keys
    assert "fc.weight" in keys
    assert not any(key.startswith("core.") for key in keys)


def test_step_loss_and_uniform_loss_are_finite():
    logits = torch.randn(1, 48, 4)
    labels = torch.zeros(1, 48, dtype=torch.long)
    uniform = torch.nn.functional.cross_entropy(logits[:, 40:].reshape(-1, 4), labels[:, 40:].reshape(-1))
    step = torch.nn.functional.cross_entropy(logits[:, -1], labels[:, -1])
    assert torch.isfinite(uniform)
    assert torch.isfinite(step)


def test_replay_reset_gap_is_recorded():
    import importlib.util
    replay_path = LAB / "src/replay_eval.py"
    spec = importlib.util.spec_from_file_location("replay_eval", replay_path)
    replay = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(replay)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MODULE.TimmGRUStudent("repvit_m1", 4, hidden_dim=256, pretrained=False).to(device).eval()
    frames = torch.randn(3, 3, 224, 224)
    rows = replay.replay_stream(model, frames, device, reset_after=0.5, fps=1.0)
    assert rows[0]["reset_reason"] is None
    assert rows[1]["reset_reason"] == "gap>0.5s"
    assert rows[2]["reset_reason"] == "gap>0.5s"


def test_isolated_fsm_debounces_and_accepts_valid_transition():
    import importlib.util
    path = LAB / "src/rabbit_fsm.py"
    spec = importlib.util.spec_from_file_location("rabbit_fsm", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    fsm = module.RabbitFSM(fps=2.0)
    states = []
    for index in range(8):
        state, _ = fsm.update("EATING", index / 2.0)
        states.append(state)
    assert states[-1] == "EATING"
    assert states[0] == "RESTING"


def test_replay_forced_reset_is_recorded():
    import importlib.util
    replay_path = LAB / "src/replay_eval.py"
    spec = importlib.util.spec_from_file_location("replay_eval_forced", replay_path)
    replay = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(replay)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MODULE.TimmGRUStudent("repvit_m1", 4, hidden_dim=256, pretrained=False).to(device).eval()
    frames = torch.randn(4, 3, 224, 224)
    rows = replay.replay_stream(model, frames, device, reset_after=None, fps=2.0, force_reset_at={2})
    assert [row["reset_reason"] for row in rows] == [None, None, "forced_reset", None]


def test_t0_window_loss_uses_window_endpoint_label():
    import importlib.util
    trainer_path = LAB / "src/stream_trainer.py"
    spec = importlib.util.spec_from_file_location("stream_trainer_test", trainer_path)
    trainer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(trainer)
    logits = torch.randn(2, 4)
    labels = torch.tensor([[0] * 16, [1] * 16])
    expected = torch.nn.functional.cross_entropy(logits, labels[:, -1])
    torch.testing.assert_close(trainer.loss_for_window(logits, labels), expected)


def test_training_selection_manifest_is_exact_and_fail_fast(tmp_path):
    import json
    import hashlib
    import importlib.util
    module_path = LAB / "src/data_selection.py"
    spec = importlib.util.spec_from_file_location("data_selection_test", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    data_root = tmp_path / "dataset"
    train_bad = data_root / "train/active/bad.mp4"
    train_dup = data_root / "train/eating/copy.mp4"
    val_dup = data_root / "val/eating/original.mp4"
    audit_path = tmp_path / "audit.json"
    selection_path = tmp_path / "selection.json"
    audit = {"full_video_hashes_computed": True, "dataset_root": str(data_root), "invalid_videos": [{"path": str(train_bad)}], "duplicate_sha256_across_splits": [[{"path": str(train_dup), "split": "train"}, {"path": str(val_dup), "split": "val"}]]}
    selection = {"source_audit": str(audit_path), "approved": True, "excluded_invalid_paths": [str(train_bad)], "excluded_train_duplicate_paths": [str(train_dup)], "retained_validation_duplicate_paths": [str(val_dup)]}
    audit_path.write_text(json.dumps(audit), encoding="utf-8")
    selection["selection_sha256_basis"] = {"audit_sha256": hashlib.sha256(audit_path.read_bytes()).hexdigest()}
    selection_path.write_text(json.dumps(selection), encoding="utf-8")
    train_excluded, val_excluded = module.load_split_exclusions(audit_path, selection_path, data_root)
    assert train_excluded == {str(train_bad.resolve()).lower(), str(train_dup.resolve()).lower()}
    assert val_excluded == set()
    selection["approved"] = False
    selection_path.write_text(json.dumps(selection), encoding="utf-8")
    try:
        module.load_split_exclusions(audit_path, selection_path, data_root)
    except RuntimeError as error:
        assert "not approved" in str(error)
    else:
        raise AssertionError("unapproved selection manifest must fail")


def test_short_stream_videos_keep_one_supervised_step():
    import importlib.util
    path = LAB / "src/stream_dataset.py"
    spec = importlib.util.spec_from_file_location("stream_dataset_lengths", path)
    dataset = importlib.util.module_from_spec(spec)
    import sys
    sys.modules[spec.name] = dataset
    spec.loader.exec_module(dataset)
    assert dataset.effective_block_lengths(96, 96, 40) == (96, 40)
    assert dataset.effective_block_lengths(24, 96, 40) == (24, 23)
    assert dataset.effective_block_lengths(1, 96, 40) == (1, 0)


def test_target_fps_sampling_maps_to_native_source_frames():
    import importlib.util
    path = LAB / "src/stream_dataset.py"
    spec = importlib.util.spec_from_file_location("stream_dataset_sampling", path)
    dataset = importlib.util.module_from_spec(spec)
    import sys
    sys.modules[spec.name] = dataset
    spec.loader.exec_module(dataset)
    assert [dataset.target_to_source_frame(i, 30.0, 8.0) for i in range(5)] == [0, 4, 8, 11, 15]
    assert [dataset.target_to_source_frame(i, 16.0, 8.0) for i in range(5)] == [0, 2, 4, 6, 8]
    assert [dataset.target_to_source_frame(i, 8.0, 8.0) for i in range(5)] == [0, 1, 2, 3, 4]


def test_dashboard_fusion_snapshot_uses_production_candidates_and_fsm():
    import importlib.util
    path = LAB / "src/dashboard_fusion_snapshot.py"
    spec = importlib.util.spec_from_file_location("dashboard_fusion_test", path)
    module = importlib.util.module_from_spec(spec)
    import sys
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    agent = module.DashboardFusionSnapshot()
    box = [0, 0, 100, 100]
    low = agent.update(box, [0.1, 0.1, 0.65, 0.15], 0.0)
    assert low["candidate"] == "RESTING"
    high_agent = module.DashboardFusionSnapshot()
    high = high_agent.update(box, [0.1, 0.1, 0.75, 0.15], 0.0)
    assert high["candidate"] == "EATING"
    moving = high_agent.update([100, 0, 200, 100], [0.0, 1.0, 0.0, 0.0], 0.1)
    assert moving["rule_state"] == "FAST_MOVING"
    assert moving["candidate"] == "ACTIVE"


def test_dashboard_fusion_snapshot_resets_hidden_bookkeeping_after_gap():
    import importlib.util
    path = LAB / "src/dashboard_fusion_snapshot.py"
    spec = importlib.util.spec_from_file_location("dashboard_fusion_reset_test", path)
    module = importlib.util.module_from_spec(spec)
    import sys
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    agent = module.DashboardFusionSnapshot()
    agent.update([0, 0, 100, 100], [0.1, 0.1, 0.7, 0.1], 0.1)
    agent.mamba_state = object()
    agent.seq_idx = 4
    row = agent.update([0, 0, 100, 100], [0.1, 0.1, 0.7, 0.1], 2.2)
    assert row["reset_after_gap"] is True
    assert agent.mamba_state is None
    assert row["model_ran"] is True


def test_dashboard_predictor_cadence_and_hidden_reset():
    import importlib.util
    import sys
    path = LAB / "src/dashboard_fusion_snapshot.py"
    spec = importlib.util.spec_from_file_location("dashboard_fusion_predictor_test", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    agent = module.DashboardFusionSnapshot()
    calls = []

    def predictor(hidden):
        calls.append(hidden)
        return [0.05, 0.05, 0.85, 0.05], object()

    first = agent.update_with_predictor([0, 0, 100, 100], predictor, 0.1)
    assert first["model_ran"] and first["model_prob_used"]
    assert calls == [None]
    second = agent.update_with_predictor([0, 0, 100, 100], predictor, 2.2)
    assert second["reset_after_gap"]
    assert calls == [None, None]

    moving_agent = module.DashboardFusionSnapshot()
    moving_agent.update_with_predictor([0, 0, 100, 100], predictor, 0.1)
    skipped = moving_agent.update_with_predictor([100, 0, 200, 100], predictor, 0.2)
    assert skipped["rule_state"] == "FAST_MOVING"
    assert not skipped["model_ran"]
