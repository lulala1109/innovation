"""Offline regression tests for migrated public APIs and model creation."""

import subprocess
import sys
import textwrap
import unittest
from pathlib import Path
from types import ModuleType
from unittest.mock import MagicMock, patch

import torch

import models


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class LazyPublicExportTests(unittest.TestCase):
    def _run_isolated(self, source: str) -> None:
        completed = subprocess.run(
            [sys.executable, "-c", textwrap.dedent(source)],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        if completed.returncode != 0:
            self.fail(
                "isolated export check failed:\n"
                f"stdout:\n{completed.stdout}\n"
                f"stderr:\n{completed.stderr}"
            )

    def test_attacks_exports_are_lazy_and_exclude_removed_algorithms(self):
        self._run_isolated(
            """
            import sys
            import attacks

            assert "attacks.base" not in sys.modules
            assert "attacks.pgd" not in sys.modules
            assert "attacks.layer_adaptive_pgd" not in sys.modules
            assert {
                "RLPGDAttacker",
                "TwoStageAttacker",
                "TwoStageResult",
            }.isdisjoint(attacks.__all__)
            for name in attacks.__all__:
                assert getattr(attacks, name) is not None, name
            assert "attacks.pgd" in sys.modules
            assert "attacks.layer_adaptive_pgd" in sys.modules
            for old_name in ("RLPGDAttacker", "TwoStageAttacker"):
                try:
                    getattr(attacks, old_name)
                except AttributeError:
                    pass
                else:
                    raise AssertionError(old_name)
            """
        )

    def test_core_exports_are_lazy_and_exclude_old_trackers_and_reward(self):
        self._run_isolated(
            """
            import sys
            import core

            for module_name in (
                "core.activations",
                "core.artifacts",
                "core.audio",
                "core.safety_state",
            ):
                assert module_name not in sys.modules, module_name
            assert {
                "RewardComputer",
                "RLPGDTracker",
                "SemanticTracker",
                "DifferentiableMelTransform",
            }.isdisjoint(core.__all__)
            for name in core.__all__:
                assert getattr(core, name) is not None, name
            for module_name in (
                "core.activations",
                "core.artifacts",
                "core.audio",
                "core.safety_state",
            ):
                assert module_name in sys.modules, module_name
            for old_name in ("RewardComputer", "RLPGDTracker", "SemanticTracker"):
                try:
                    getattr(core, old_name)
                except AttributeError:
                    pass
                else:
                    raise AssertionError(old_name)
            """
        )

    def test_model_classes_are_lazy_and_direct_mel_exports_are_absent(self):
        self._run_isolated(
            """
            import sys
            import models

            assert "models.qwen" not in sys.modules
            assert {
                "QwenMelAttackWrapper",
                "PhiMelAttackWrapper",
                "VoxtralMelAttackWrapper",
            }.isdisjoint(models.__all__)
            for name in models.__all__:
                assert getattr(models, name) is not None, name
            assert "models.qwen" in sys.modules
            for old_name in (
                "QwenMelAttackWrapper",
                "PhiMelAttackWrapper",
                "VoxtralMelAttackWrapper",
            ):
                try:
                    getattr(models, old_name)
                except AttributeError:
                    pass
                else:
                    raise AssertionError(old_name)
            """
        )


class ModelFactoryTests(unittest.TestCase):
    def _fake_adapter_module(self):
        qwen_module = ModuleType("models.qwen")
        qwen_module.QwenModel = MagicMock(name="QwenModel")
        return qwen_module

    def test_supported_models_match_portable_default_ids(self):
        self.assertEqual(
            models.SUPPORTED_MODELS,
            ("qwen-3b", "qwen-7b"),
        )
        self.assertEqual(
            set(models.SUPPORTED_MODELS), set(models.DEFAULT_MODEL_IDS)
        )
        for checkpoint in models.DEFAULT_MODEL_IDS.values():
            self.assertFalse(checkpoint.startswith("/"), checkpoint)

    def test_unsupported_model_type_is_rejected_before_adapter_import(self):
        with patch.dict(sys.modules, {}, clear=False):
            with self.assertRaisesRegex(ValueError, "Unknown model type"):
                models.create_model("phi-4b", device="cpu")

    def test_model_id_override_and_constructor_options_are_forwarded(self):
        qwen_module = self._fake_adapter_module()
        expected = object()
        qwen_module.QwenModel.return_value = expected

        with patch.dict(sys.modules, {"models.qwen": qwen_module}):
            result = models.create_model(
                "QWEN-7B",
                device="cpu",
                dtype=torch.float32,
                token="offline-token",
                model_id="local/test-qwen",
            )

        self.assertIs(result, expected)
        qwen_module.QwenModel.assert_called_once_with(
            model_id="local/test-qwen",
            device="cpu",
            dtype=torch.float32,
            token="offline-token",
        )

    def test_each_supported_model_uses_its_declared_default_checkpoint(self):
        qwen_module = self._fake_adapter_module()
        with patch.dict(sys.modules, {"models.qwen": qwen_module}):
            for model_type in models.SUPPORTED_MODELS:
                with self.subTest(model_type=model_type):
                    qwen_module.QwenModel.reset_mock()
                    models.create_model(
                        model_type,
                        device="cpu",
                        dtype=torch.float32,
                        token=None,
                    )
                    qwen_module.QwenModel.assert_called_once_with(
                        model_id=models.DEFAULT_MODEL_IDS[model_type],
                        device="cpu",
                        dtype=torch.float32,
                        token=None,
                    )


if __name__ == "__main__":
    unittest.main()
