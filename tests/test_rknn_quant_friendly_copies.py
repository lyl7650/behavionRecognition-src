import unittest

import torch
import torch.nn as nn

from training.config_rknn import build_distill_config_rknn
from training.rknn_reparameterization import verify_reparameterization
from training.rknn_optim import adamw_groups_with_bn_gamma_decay


class _ReparameterizableIdentity(nn.Module):
    """Stand-in for the student: folding shifts the backbone output by ``delta``."""

    def __init__(self, delta=0.0):
        super().__init__()
        self.delta = delta

    def backbone(self, frames):
        return [frames + self.delta]

    def forward_seq(self, x):
        return x.flatten(0, 1) + self.delta

    def reparameterize(self):
        return self


class RknnCopyTests(unittest.TestCase):
    def test_bn_gamma_has_its_own_mild_weight_decay(self):
        model = nn.Sequential(nn.Conv2d(3, 4, 1), nn.BatchNorm2d(4), nn.Linear(4, 2))
        groups = adamw_groups_with_bn_gamma_decay(model, base_weight_decay=0.05, bn_gamma_weight_decay=1e-4)
        gamma_group = next(group for group in groups if group["weight_decay"] == 1e-4)
        self.assertEqual({id(p) for p in gamma_group["params"]}, {id(model[1].weight)})
        no_decay_group = next(group for group in groups if group["weight_decay"] == 0.0)
        self.assertIn(id(model[1].bias), {id(p) for p in no_decay_group["params"]})

    def test_reparameterization_requires_numerical_alignment(self):
        model = _ReparameterizableIdentity()
        error = verify_reparameterization(model, torch.ones(2, 3, 4), tolerance=1e-3)
        self.assertEqual(error, 0.0)

    def test_reparameterization_tolerates_float_noise_on_large_weights(self):
        """Folding is exact, so residual float32 noise must not fail the guard."""

        class NoisyModel(_ReparameterizableIdentity):
            def reparameterize(self):
                self.delta = 1e-5
                return self

        error = verify_reparameterization(NoisyModel(), torch.ones(2, 3, 4), tolerance=1e-3)
        self.assertLess(error, 1e-3)

    def test_reparameterization_rejects_large_change(self):
        class BadModel(_ReparameterizableIdentity):
            def reparameterize(self):
                self.delta = 0.5
                return self

        with self.assertRaisesRegex(RuntimeError, "exceeds tolerance"):
            verify_reparameterization(BadModel(), torch.ones(2, 3, 4), tolerance=1e-3)

    def test_reparameterization_rejects_non_finite_output(self):
        class NanModel(_ReparameterizableIdentity):
            def reparameterize(self):
                self.delta = float("nan")
                return self

        with self.assertRaisesRegex(RuntimeError, "non-finite"):
            verify_reparameterization(NanModel(), torch.ones(2, 3, 4), tolerance=1e-3)

    def test_rknn_config_defaults_and_overrides(self):
        cfg = build_distill_config_rknn(
            "normal",
            config_dict={
                "distillation": {"teacher_checkpoint": "teacher.pth"},
                "task_configs": {
                    "normal": {
                        "dataset_path": "dataset", "class_names": ["eat", "drink", "active", "resting"],
                        "recognizer": {"imgsz": 224, "temporal_window": 16},
                        "distillation": {"rknn": {"finetune_epochs": 5, "finetune_lr_ratio": 0.05}},
                    }
                },
            },
        )
        self.assertEqual(cfg["rknn_finetune_epochs"], 5)
        self.assertEqual(cfg["rknn_finetune_lr_ratio"], 0.05)
        self.assertEqual(cfg["rknn_bn_gamma_weight_decay"], 1e-4)

    def test_rknn_config_rejects_out_of_range_finetune_epochs(self):
        with self.assertRaisesRegex(ValueError, "between 3 and 5"):
            build_distill_config_rknn(
                "normal",
                config_dict={
                    "rknn": {"finetune_epochs": 2},
                    "task_configs": {"normal": {"dataset_path": "dataset", "class_names": ["resting"], "recognizer": {}}},
                },
            )


if __name__ == "__main__":
    unittest.main()
