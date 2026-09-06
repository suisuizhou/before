import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from torch.utils.data import TensorDataset


class TargetEntryTests(unittest.TestCase):
    def test_variants_have_exact_cumulative_switches_and_reject_unknown(self):
        from main_tta_sde_evmt_r18 import variant_switches

        expected = {
            "R2": (False, False, False, False, False, False),
            "R3": (True, True, False, False, False, False),
            "R4": (True, True, True, False, False, False),
            "R5": (True, True, True, True, True, False),
            "R6": (True, True, True, True, True, True),
        }
        keys = ("multiview_teacher", "mean_teacher", "evidence", "memory", "pcl", "ncl")
        for variant, values in expected.items():
            switches = variant_switches(variant)
            self.assertEqual(tuple(switches[key] for key in keys), values)
            self.assertEqual(switches["feature_adapter"], variant == "R6")
        with self.assertRaises(ValueError):
            variant_switches("R7")

    def test_checkpoint_loader_requires_r1_and_matching_signature(self):
        from main_src_sde_evmt_r18 import model_signature
        from main_tta_sde_evmt_r18 import load_source_checkpoint
        from sde_evmt_r18.model import SDEEVMTResNet18

        model = SDEEVMTResNet18(num_classes=3)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "source.pt"
            checkpoint = {
                "model": model.state_dict(),
                "epoch": 1,
                "source_accuracy": 50.0,
                "config": {"variant": "R1"},
                "model_signature": model_signature(model),
            }
            torch.save(checkpoint, path)
            loaded = load_source_checkpoint(SDEEVMTResNet18(num_classes=3), path)
            self.assertEqual(len(loaded["sha256"]), 64)

            checkpoint["config"]["variant"] = "R0"
            torch.save(checkpoint, path)
            with self.assertRaises(ValueError):
                load_source_checkpoint(SDEEVMTResNet18(num_classes=3), path)

            checkpoint["config"]["variant"] = "R1"
            checkpoint["model_signature"] = {"architecture": "wrong"}
            torch.save(checkpoint, path)
            with self.assertRaises(ValueError):
                load_source_checkpoint(SDEEVMTResNet18(num_classes=3), path)

    def test_output_directories_are_unique_across_task_seed_and_variant(self):
        from main_tta_sde_evmt_r18 import target_output_dir

        root = Path("/tmp/results")
        paths = {
            target_output_dir(root, 0, 1, seed, variant)
            for seed in (1, 2)
            for variant in ("R2", "R6")
        }
        self.assertEqual(len(paths), 4)

    def test_tiny_run_writes_config_batches_summary_and_log(self):
        from main_src_sde_evmt_r18 import model_signature
        from main_tta_sde_evmt_r18 import run_target
        from sde_evmt_r18.model import SDEEVMTResNet18

        torch.manual_seed(9)
        dataset = TensorDataset(torch.rand(6, 512), torch.arange(6) % 3)
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            checkpoint_path = tmp_path / "source.pt"
            source_model = SDEEVMTResNet18(num_classes=3)
            torch.save(
                {
                    "model": source_model.state_dict(),
                    "epoch": 1,
                    "source_accuracy": 50.0,
                    "config": {"variant": "R1"},
                    "model_signature": model_signature(source_model),
                },
                checkpoint_path,
            )
            cfg = {
                "variant": "R2",
                "only_task": [0, 1],
                "seed": 5,
                "device": "cpu",
                "num_classes": 3,
                "source_checkpoint": str(checkpoint_path),
                "output_root": str(tmp_path / "outputs"),
                "smoke_batches": 2,
                "data": {"batch_size": 3, "num_workers": 0},
                "model": {"input_len": 512, "bottleneck_dim": 256},
                "reliability": {
                    "confidence_min": 0.0,
                    "min_view_agreement": 0.0,
                    "min_class_samples": 1,
                },
            }
            with patch(
                "main_tta_sde_evmt_r18.load_pu4d_domain", return_value=dataset
            ):
                summary = run_target(cfg)

            output = Path(summary["output_dir"])
            self.assertTrue((output / "config.yaml").is_file())
            self.assertTrue((output / "batches.jsonl").is_file())
            self.assertTrue((output / "summary.json").is_file())
            self.assertTrue((output / "run.log").is_file())
            rows = [json.loads(line) for line in (output / "batches.jsonl").read_text().splitlines()]
            self.assertEqual(len(rows), 2)
            self.assertEqual(summary["seen_samples"], 6)
            self.assertEqual(summary["passes"], 1)
            self.assertEqual(summary["batch_rows"], 2)


if __name__ == "__main__":
    unittest.main()
