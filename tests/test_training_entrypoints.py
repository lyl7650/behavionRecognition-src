import unittest
import importlib
import sys
from types import SimpleNamespace
from unittest.mock import Mock, patch

from training.entrypoints import run_distill, run_train


class RunTrainTests(unittest.TestCase):
    def test_run_train_dispatches_resolved_config_to_mode_module(self):
        with patch("training.entrypoints.build_train_config", return_value={"mode": "normal"}) as build_cfg:
            fake_module = SimpleNamespace(train=Mock())
            with patch("training.entrypoints.importlib.import_module", return_value=fake_module) as importer:
                run_train("normal")
        build_cfg.assert_called_once()
        importer.assert_called_once_with("train.train_finetune_normal_3")
        fake_module.train.assert_called_once_with(config_override={"mode": "normal"})

    def test_run_train_calls_train_with_config_override(self):
        fake_module = SimpleNamespace(train=Mock())
        with patch(
            "training.entrypoints.build_train_config",
            return_value={"mode": "abnormal", "batch_size": 4},
        ):
            with patch("training.entrypoints.importlib.import_module", return_value=fake_module):
                run_train("abnormal")
        fake_module.train.assert_called_once_with(config_override={"mode": "abnormal", "batch_size": 4})


class RunDistillTests(unittest.TestCase):
    def test_run_distill_dispatches_resolved_config_to_mode_module(self):
        fake_module = SimpleNamespace(train_distill=Mock())
        resolved = {"mode": "normal", "temperature": 4.0}
        with patch("training.entrypoints.build_distill_config", return_value=resolved) as build_cfg:
            with patch("training.entrypoints.importlib.import_module", return_value=fake_module) as importer:
                run_distill("normal")
        build_cfg.assert_called_once()
        importer.assert_called_once_with("distill.distill_normal_3")
        fake_module.train_distill.assert_called_once_with(config_override=resolved)


class ThinScriptDelegationTests(unittest.TestCase):
    def _import_with_stubs(self, module_name, stubs):
        original = {name: sys.modules.get(name) for name in stubs}
        try:
            sys.modules.update(stubs)
            sys.modules.pop(module_name, None)
            return importlib.import_module(module_name)
        finally:
            sys.modules.pop(module_name, None)
            for name, value in original.items():
                if value is None:
                    sys.modules.pop(name, None)
                else:
                    sys.modules[name] = value

    def test_normal_teacher_script_delegates_to_shared_trainer(self):
        train_finetune_normal_3 = self._import_with_stubs(
            "train.train_finetune_normal_3",
            {
                "training.teacher_finetune": SimpleNamespace(run_teacher_finetune=Mock()),
            },
        )

        cfg = {
            "imgsz": 224,
            "temporal_window": 16,
            "class_names": ["eating", "moving", "resting", "drinking"],
            "output_model_name": "pet_teacher_0703_ema.pth",
        }
        with patch.object(train_finetune_normal_3, "load_train_config", return_value=cfg) as load_cfg:
            with patch.object(train_finetune_normal_3, "run_teacher_finetune") as runner:
                train_finetune_normal_3.train()

        load_cfg.assert_called_once_with(config_path=None)
        runner.assert_called_once()
        args, kwargs = runner.call_args
        self.assertEqual(args[0], cfg)
        self.assertEqual(args[2], cfg["output_model_name"])
        self.assertTrue(kwargs["print_config"])
        self.assertFalse(kwargs["pass_num_classes_to_teacher"])

    def test_abnormal_teacher_script_delegates_to_shared_trainer(self):
        train_finetune_ab_5 = self._import_with_stubs(
            "train.train_finetune_ab_5",
            {
                "training.teacher_finetune": SimpleNamespace(run_teacher_finetune=Mock()),
            },
        )

        cfg = {
            "imgsz": 224,
            "temporal_window": 16,
            "class_names": ["normal", "convulsion", "limp", "sneeze", "vomit"],
        }
        with patch.object(train_finetune_ab_5, "build_train_config", return_value=cfg) as build_cfg:
            with patch.object(train_finetune_ab_5, "run_teacher_finetune") as runner:
                train_finetune_ab_5.train()

        build_cfg.assert_called_once_with("abnormal", config_path=None)
        runner.assert_called_once()
        args, kwargs = runner.call_args
        self.assertEqual(args[0], cfg)
        self.assertEqual(args[2], train_finetune_ab_5.BEST_MODEL_NAME)
        self.assertFalse(kwargs["print_config"])
        self.assertTrue(kwargs["pass_num_classes_to_teacher"])

    def test_normal_distill_script_delegates_to_shared_trainer(self):
        distill_normal_3 = self._import_with_stubs(
            "distill.distill_normal_3",
            {
                "training.distillation": SimpleNamespace(run_distillation_training=Mock()),
            },
        )

        runtime_cfg = {
            "class_names": ["eating", "moving", "resting", "drinking"],
            "loss_weights": {"gt": 1.0, "kd": 8.0, "feature": 1.0},
            "temperature": 4.0,
            "output_model_name": "repvit_best_distilled_0703.pth",
            "output_ema_model_name": "repvit_best_distilled_0703_ema.pth",
        }
        with patch.object(distill_normal_3, "_build_runtime_config", return_value=runtime_cfg) as build_cfg:
            with patch.object(distill_normal_3, "run_distillation_training") as runner:
                distill_normal_3.train_distill(config_override={"mode": "normal"})

        build_cfg.assert_called_once_with(config_override={"mode": "normal"}, config_path=None)
        runner.assert_called_once()
        kwargs = runner.call_args.kwargs
        self.assertEqual(kwargs["runtime_cfg"], runtime_cfg)
        self.assertEqual(kwargs["best_model_name"], runtime_cfg["output_model_name"])
        self.assertEqual(kwargs["best_ema_model_name"], runtime_cfg["output_ema_model_name"])

    def test_abnormal_distill_script_delegates_to_shared_trainer(self):
        distill_abnormal_5 = self._import_with_stubs(
            "distill.distill_abnormal_5",
            {
                "training.distillation": SimpleNamespace(run_distillation_training=Mock()),
            },
        )

        runtime_cfg = {
            "class_names": ["normal", "convulsion", "limp", "sneeze", "vomit"],
            "loss_weights": {"gt": 1.0, "kd": 8.0, "feature": 1.0},
            "temperature": 4.0,
        }
        with patch.object(distill_abnormal_5, "_build_runtime_config", return_value=runtime_cfg) as build_cfg:
            with patch.object(distill_abnormal_5, "run_distillation_training") as runner:
                distill_abnormal_5.train_distill(config_override={"mode": "abnormal"})

        build_cfg.assert_called_once_with(config_override={"mode": "abnormal"}, config_path=None)
        runner.assert_called_once()
        kwargs = runner.call_args.kwargs
        self.assertEqual(kwargs["runtime_cfg"], runtime_cfg)
        self.assertEqual(kwargs["best_model_name"], distill_abnormal_5.BEST_MODEL_NAME)
        self.assertEqual(kwargs["best_ema_model_name"], distill_abnormal_5.BEST_EMA_MODEL_NAME)


if __name__ == "__main__":
    unittest.main()
