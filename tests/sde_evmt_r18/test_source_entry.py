import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from torch.utils.data import TensorDataset


class SourceEntryTests(unittest.TestCase):
    def test_train_source_writes_reloadable_checkpoint_contract(self):
        from main_src_sde_evmt_r18 import train_source

        torch.manual_seed(7)
        dataset = TensorDataset(torch.rand(8, 512), torch.arange(8) % 4)
        with tempfile.TemporaryDirectory() as tmp, patch(
            "main_src_sde_evmt_r18.load_pu4d_domain", return_value=dataset
        ):
            cfg = {
                "variant": "R0",
                "source": 0,
                "epochs": 1,
                "smoke_batches": 1,
                "seed": 3,
                "device": "cpu",
                "num_classes": 4,
                "output_root": tmp,
                "training": {
                    "batch_size": 4,
                    "eval_batch_size": 8,
                    "lr": 0.001,
                    "weight_decay": 0.00001,
                    "num_workers": 0,
                    "label_smoothing": 0.1,
                    "mixup_alpha": 0.2,
                    "mixup_prob": 0.0,
                },
            }
            summary = train_source(cfg)
            output = Path(summary["output_dir"])
            checkpoint = torch.load(output / "best.pt", map_location="cpu", weights_only=False)
            self.assertEqual(
                set(("model", "epoch", "source_accuracy", "config", "model_signature"))
                - set(checkpoint),
                set(),
            )
            self.assertEqual(len(summary["checkpoint_sha256"]), 64)
            self.assertTrue((output / "history.json").is_file())
            self.assertEqual(json.loads((output / "summary.json").read_text())["variant"], "R0")
            self.assertEqual(checkpoint["source_accuracy"], summary["best_source_accuracy"])

    def test_r0_and_r1_have_unique_output_directories(self):
        from main_src_sde_evmt_r18 import source_output_dir

        root = Path("/tmp/output")
        self.assertNotEqual(
            source_output_dir(root, 0, 1, "R0"),
            source_output_dir(root, 0, 1, "R1"),
        )


if __name__ == "__main__":
    unittest.main()
