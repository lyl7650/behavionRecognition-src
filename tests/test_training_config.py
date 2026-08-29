import ast
import os
import unittest
from datetime import datetime

from training.config import build_distill_config
from training.config import build_train_config


class BuildDistillConfigTests(unittest.TestCase):
    def test_build_distill_config_merges_global_defaults_and_task_overrides(self):
        cfg = build_distill_config(
            "abnormal",
            config_dict={
                "distillation": {
                    "temperature": 4.0,
                    "alpha": 1.0,
                    "loss_weights": {"kd": 8.0},
                },
                "task_configs": {
                    "abnormal": {
                        "dataset_path": "dataset/abnormal/5_cls",
                        "class_names": ["normal", "convulsion", "limp", "sneeze", "vomit"],
                        "recognizer": {"temporal_window": 16, "imgsz": 224},
                        "distillation": {
                            "temperature": 6.0,
                            "loss_weights": {"feature": 2.0},
                        },
                    }
                },
            },
        )
        self.assertEqual(cfg["temperature"], 6.0)
        self.assertEqual(cfg["loss_weights"], {"gt": 1.0, "kd": 8.0, "feature": 2.0})

    def test_build_distill_config_preserves_task_specific_output_dir(self):
        cfg = build_distill_config(
            "normal",
            config_dict={
                "distillation": {"output_dir": "outputs/default", "temperature": 4.0},
                "task_configs": {
                    "normal": {
                        "dataset_path": "dataset/rabbit/0507",
                        "class_names": ["eating", "moving", "resting", "drinking"],
                        "recognizer": {"temporal_window": 16, "imgsz": 224},
                        "distillation": {"output_dir": "outputs/distilled_final/rabbit/0508"},
                    }
                },
            },
        )
        self.assertEqual(cfg["output_dir"], os.path.normpath("outputs/distilled_final/rabbit/0508"))

    def test_build_distill_config_resolves_output_model_names_with_date(self):
        date_tag = datetime.now().strftime("%Y-%m-%d")
        cfg = build_distill_config(
            "normal",
            config_dict={
                "distillation": {"output_model_name": "global_{date}.pth"},
                "task_configs": {
                    "normal": {
                        "dataset_path": "dataset/rabbit/0507",
                        "class_names": ["eating", "moving", "resting", "drinking"],
                        "recognizer": {"temporal_window": 16, "imgsz": 224},
                        "distillation": {"output_model_name": "repvit_best_distilled_{date}.pth"},
                    }
                },
            },
        )
        self.assertEqual(cfg["output_model_name"], f"repvit_best_distilled_{date_tag}.pth")
        self.assertEqual(cfg["output_ema_model_name"], f"repvit_best_distilled_{date_tag}_ema.pth")

    def test_build_distill_config_prefers_task_specific_teacher_checkpoint_and_output_dir(self):
        cfg = build_distill_config(
            "abnormal",
            config_dict={
                "distillation": {
                    "teacher_checkpoint": "outputs/global_teacher.pth",
                    "output_dir": "outputs/global_distill",
                    "output_model_name": "global_distill_{date}.pth",
                    "loss_weights": {"kd": 8.0},
                },
                "task_configs": {
                    "abnormal": {
                        "dataset_path": "dataset/abnormal/5_cls",
                        "class_names": ["normal", "convulsion", "limp", "sneeze", "vomit"],
                        "recognizer": {"temporal_window": 16, "imgsz": 224},
                        "distillation": {
                            "teacher_checkpoint": "outputs/task_teacher.pth",
                            "output_dir": "outputs/task_distill",
                            "output_model_name": "task_distill_{date}.pth",
                        },
                    }
                },
            },
        )
        date_tag = datetime.now().strftime("%Y-%m-%d")
        self.assertEqual(cfg["teacher_checkpoint"], os.path.normpath("outputs/task_teacher.pth"))
        self.assertEqual(cfg["output_dir"], os.path.normpath("outputs/task_distill"))
        self.assertEqual(cfg["output_model_name"], f"task_distill_{date_tag}.pth")
        self.assertEqual(cfg["output_ema_model_name"], f"task_distill_{date_tag}_ema.pth")

    def test_build_distill_config_keeps_batch_size_and_weights_for_legacy_adapter(self):
        cfg = build_distill_config(
            "normal",
            config_dict={
                "distillation": {
                    "temperature": 5.0,
                    "batch_size": 12,
                    "epochs": 20,
                    "learning_rate": 1e-4,
                    "loss_weights": {"gt": 1.2, "kd": 7.0, "feature": 0.8},
                },
                "task_configs": {
                    "normal": {
                        "dataset_path": "dataset/rabbit/0507",
                        "class_names": ["eating", "moving", "resting", "drinking"],
                        "recognizer": {"temporal_window": 16, "imgsz": 224},
                    }
                },
            },
        )
        self.assertEqual(cfg["batch_size"], 12)
        self.assertEqual(cfg["epochs"], 20)
        self.assertEqual(cfg["learning_rate"], 1e-4)
        self.assertEqual(cfg["loss_weights"], {"gt": 1.2, "kd": 7.0, "feature": 0.8})

    def test_distill_normal_script_uses_shared_config_fields_and_resolved_save_names(self):
        parsed = _load_module_ast(os.path.join("distill", "distill_normal_3.py"))
        self.assertEqual(_find_constant(parsed, "DEFAULT_MODE"), "normal")
        self.assertEqual(_find_train_distill_args(parsed), ["config_override", "config_path"])
        self.assertEqual(
            _find_runtime_config_keys(parsed),
            [
                "dataset_path",
                "class_names",
                "recognizer",
                "teacher_checkpoint",
                "temperature",
                "loss_weights",
                "learning_rate",
                "batch_size",
                "epochs",
                "num_workers",
                "output_dir",
                "output_model_name",
                "output_ema_model_name",
            ],
        )

    def test_distill_abnormal_script_uses_shared_config_fields_and_expected_save_names(self):
        parsed = _load_module_ast(os.path.join("distill", "distill_abnormal_5.py"))
        self.assertEqual(_find_constant(parsed, "BEST_MODEL_NAME"), "repvit_best_distilled_5cls.pth")
        self.assertEqual(_find_constant(parsed, "BEST_EMA_MODEL_NAME"), "repvit_best_distilled_5cls_ema.pth")
        self.assertEqual(_find_constant(parsed, "DEFAULT_MODE"), "abnormal")
        self.assertEqual(_find_train_distill_args(parsed), ["config_override", "config_path"])
        self.assertEqual(
            _find_runtime_config_keys(parsed),
            [
                "dataset_path",
                "class_names",
                "recognizer",
                "teacher_checkpoint",
                "temperature",
                "loss_weights",
                "learning_rate",
                "batch_size",
                "epochs",
                "num_workers",
                "output_dir",
            ],
        )


class BuildTrainConfigTests(unittest.TestCase):
    def test_build_train_config_populates_abnormal_recognizer_from_class_names(self):
        cfg = build_train_config(
            "abnormal",
            config_dict={
                "task_configs": {
                    "abnormal": {
                        "dataset_path": "dataset/abnormal/5_cls",
                        "class_names": ["normal", "convulsion", "limp", "sneeze", "vomit"],
                        "finetune": {
                            "checkpoint_path": "outputs/pretrained/teacher.pth",
                            "output_dir": "outputs/teacher_finetuned_5cls",
                            "batch_size": 6,
                        },
                        "recognizer": {
                            "imgsz": 256,
                            "temporal_window": 24,
                        },
                    }
                }
            },
        )
        self.assertEqual(cfg["class_names"], ["normal", "convulsion", "limp", "sneeze", "vomit"])
        self.assertEqual(cfg["recognizer"]["class_names"], cfg["class_names"])
        self.assertEqual(cfg["recognizer"]["num_classes"], 5)
        self.assertEqual(cfg["imgsz"], 256)
        self.assertEqual(cfg["temporal_window"], 24)

    def test_build_train_config_preserves_recognizer_augment_settings(self):
        cfg = build_train_config(
            "normal",
            config_dict={
                "task_configs": {
                    "normal": {
                        "dataset_path": "dataset/pet/normal/0702",
                        "class_names": ["eat", "drink", "active", "resting"],
                        "recognizer": {
                            "imgsz": 224,
                            "temporal_window": 16,
                            "augment": {"fliplr": 1.0, "brightness": 0.1},
                        },
                    }
                }
            },
        )
        self.assertEqual(cfg["recognizer"]["augment"], {"fliplr": 1.0, "brightness": 0.1})


def _load_module_ast(path):
    with open(path, "r", encoding="utf-8") as handle:
        return ast.parse(handle.read(), filename=path)


def _find_constant(parsed, name):
    for node in parsed.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == name:
                    return ast.literal_eval(node.value)
    raise AssertionError(f"Constant {name} not found")


def _find_train_distill_args(parsed):
    for node in parsed.body:
        if isinstance(node, ast.FunctionDef) and node.name == "train_distill":
            return [arg.arg for arg in node.args.args]
    raise AssertionError("train_distill not found")


def _find_runtime_config_keys(parsed):
    for node in parsed.body:
        if isinstance(node, ast.FunctionDef) and node.name == "_build_runtime_config":
            for child in ast.walk(node):
                if isinstance(child, ast.Return) and isinstance(child.value, ast.Dict):
                    keys = []
                    for key in child.value.keys:
                        keys.append(ast.literal_eval(key))
                    return keys
    raise AssertionError("_build_runtime_config return dict not found")


if __name__ == "__main__":
    unittest.main()
