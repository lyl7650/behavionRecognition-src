import sys
import types
import unittest
from pathlib import Path
from importlib import import_module


ROOT_DIR = Path(__file__).resolve().parents[1]

EXPECTED_IMPORTS = {
    "train/train_finetune_normal_3.py": [
        "from training.teacher_finetune import run_teacher_finetune",
    ],
    "train/train_finetune_ab_5.py": [
        "from training.teacher_finetune import run_teacher_finetune",
    ],
    "distill/distill_normal_3.py": [
        "from training.distillation import run_distillation",
    ],
    "distill/distill_abnormal_5.py": [
        "from training.distillation import run_distillation",
    ],
    "utils/export_onnx.py": [
        "from models.student import RepViT_GRU",
    ],
    "train/train_qat.py": [
        "from models.student import RepViT_GRU",
    ],
}


class ModelModuleMigrationTests(unittest.TestCase):
    def test_new_model_modules_exist_and_old_paths_reexport(self):
        if "timm" not in sys.modules:
            timm_stub = types.ModuleType("timm")
            timm_stub.create_model = lambda *args, **kwargs: None
            sys.modules["timm"] = timm_stub

        from distill.swin_wrapper import SwinTeacher as legacy_teacher
        from models.teacher import SwinTeacher as new_teacher
        from models.student import RepViT_GRU as new_student
        from recognizers.gru_model import RepViT_GRU as legacy_student

        self.assertIs(legacy_teacher, new_teacher)
        self.assertIs(legacy_student, new_student)

    def test_runtime_modules_can_import_from_new_and_legacy_paths(self):
        if "timm" not in sys.modules:
            timm_stub = types.ModuleType("timm")
            timm_stub.create_model = lambda *args, **kwargs: None
            sys.modules["timm"] = timm_stub

        teacher_module = import_module("models.teacher")
        student_module = import_module("models.student")
        legacy_teacher_module = import_module("distill.swin_wrapper")
        legacy_student_module = import_module("recognizers.gru_model")

        self.assertTrue(hasattr(teacher_module, "SwinTeacher"))
        self.assertTrue(hasattr(student_module, "RepViT_GRU"))
        self.assertIs(legacy_teacher_module.SwinTeacher, teacher_module.SwinTeacher)
        self.assertIs(legacy_student_module.RepViT_GRU, student_module.RepViT_GRU)

    def test_entrypoints_import_new_model_paths(self):
        for relative_path, expected_lines in EXPECTED_IMPORTS.items():
            content = (ROOT_DIR / relative_path).read_text(encoding="utf-8")
            for expected_line in expected_lines:
                self.assertIn(expected_line, content, f"{relative_path} missing {expected_line!r}")

    def test_train_entrypoints_no_longer_swallow_teacher_import_errors(self):
        for relative_path in (
            "train/train_finetune_normal_3.py",
            "train/train_finetune_ab_5.py",
        ):
            content = (ROOT_DIR / relative_path).read_text(encoding="utf-8")
            self.assertNotIn("SwinTeacher = None", content, f"{relative_path} still swallows teacher import")
            self.assertNotIn("except ImportError", content, f"{relative_path} still keeps ImportError fallback")


if __name__ == "__main__":
    unittest.main()
