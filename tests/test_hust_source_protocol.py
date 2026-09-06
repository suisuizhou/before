from __future__ import annotations

import json
import math
from pathlib import Path
import subprocess
import sys

import pytest
import torch
from omegaconf import OmegaConf
from torch.utils.data import TensorDataset

from Lib.hust_source_training import (
    FORMAL_SOURCE_HYPERPARAMETERS,
    source_loss,
    train_hust_source,
)
from Lib.hust_strict_protocol import (
    ensure_hust_adaptation_carrier,
    hust_checkpoint_dir,
    resolve_hust_checkpoint,
    strict_load_hust_checkpoint,
)
from Lib.pu4d_vanilla_protocol import reset_and_freeze_adaptation_carrier


ROOT = Path(__file__).resolve().parents[1]


def test_ordinary_loss_uses_only_clean_logits():
    clean = torch.tensor([[2.0, 0.0]], requires_grad=True)
    loss, terms = source_loss("ordinary", clean, torch.tensor([0]), num_classes=2)

    assert set(terms) == {"clean"}
    assert loss.item() == terms["clean"].item()


def test_robust_loss_exposes_all_0711_terms():
    logits = torch.tensor([[2.0, 0.0]], requires_grad=True)
    loss, terms = source_loss(
        "robust",
        logits,
        torch.tensor([0]),
        num_classes=2,
        style_logits=logits + 0.1,
        warp_logits=logits - 0.1,
        style_warp_logits=logits + 0.2,
        clean_feature=torch.tensor([[1.0, 0.0]]),
        augmented_features=[torch.tensor([[0.9, 0.1]])] * 3,
    )

    assert {"clean", "style", "warp", "style_warp", "symmetric_kl", "feature"} == set(terms)
    assert torch.isfinite(loss)


def test_robust_loss_has_exact_approved_weighting_and_feature_gradient_direction():
    import torch.nn.functional as F

    labels = torch.tensor([0, 1])
    clean_logits = torch.tensor([[1.5, -0.5], [-0.2, 1.2]], requires_grad=True)
    style_logits = torch.tensor([[1.0, 0.0], [0.1, 0.9]], requires_grad=True)
    warp_logits = torch.tensor([[1.2, -0.1], [-0.1, 1.0]], requires_grad=True)
    combo_logits = torch.tensor([[0.9, 0.2], [0.0, 0.8]], requires_grad=True)
    clean_feature = torch.tensor([[1.0, 0.0], [0.0, 1.0]], requires_grad=True)
    augmented = [
        torch.tensor([[0.8, 0.2], [0.1, 0.9]], requires_grad=True),
        torch.tensor([[0.9, 0.1], [0.2, 0.8]], requires_grad=True),
        torch.tensor([[0.7, 0.3], [0.3, 0.7]], requires_grad=True),
    ]

    total, terms = source_loss(
        "robust",
        clean_logits,
        labels,
        num_classes=2,
        style_logits=style_logits,
        warp_logits=warp_logits,
        style_warp_logits=combo_logits,
        clean_feature=clean_feature,
        augmented_features=augmented,
    )
    expected_clean = F.cross_entropy(clean_logits, labels, label_smoothing=0.1)
    expected_style = F.cross_entropy(style_logits, labels, label_smoothing=0.1)
    expected_warp = F.cross_entropy(warp_logits, labels, label_smoothing=0.1)
    expected_combo = F.cross_entropy(combo_logits, labels, label_smoothing=0.1)
    expected = (
        expected_clean
        + 0.5 * expected_style
        + 0.5 * expected_warp
        + 0.25 * expected_combo
        + 0.03 * terms["symmetric_kl"]
        + 0.02 * terms["feature"]
    )

    assert torch.allclose(terms["clean"], expected_clean)
    assert torch.allclose(terms["style"], expected_style)
    assert torch.allclose(terms["warp"], expected_warp)
    assert torch.allclose(terms["style_warp"], expected_combo)
    assert torch.allclose(total, expected)
    terms["feature"].backward()
    assert clean_feature.grad is None
    assert all(feature.grad is not None for feature in augmented)


def test_checkpoint_paths_separate_variants(tmp_path):
    ordinary = hust_checkpoint_dir(tmp_path, "ordinary", 0, 2025)
    robust = hust_checkpoint_dir(tmp_path, "robust", 0, 2025)

    assert ordinary != robust
    assert ordinary.parts[-3:] == ("ordinary", "source_0", "seed_2025")
    assert robust.parts[-3:] == ("robust", "source_0", "seed_2025")


class _TinyBackbone(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.band_scale = torch.nn.Parameter(torch.ones(1, 1, 2))
        self.band_bias = torch.nn.Parameter(torch.ones(1, 1, 2))
        self.projection = torch.nn.Linear(8, 4)

    def forward(self, x):
        return self.projection(x.flatten(1))


def _tiny_model():
    return torch.nn.Sequential(
        _TinyBackbone(),
        torch.nn.Linear(4, 3),
        torch.nn.Linear(3, 2),
    )


def test_source_optimizer_excludes_every_frozen_carrier_tensor():
    import Lib.hust_source_training as source_training

    model = _tiny_model()
    carrier_names = ensure_hust_adaptation_carrier(model)
    reset_and_freeze_adaptation_carrier(model)
    optimizer, optimized_names = source_training.build_source_optimizer(model)
    optimized_ids = {
        id(parameter)
        for group in optimizer.param_groups
        for parameter in group["params"]
    }
    named = dict(model.named_parameters())

    assert set(carrier_names) == {"0.band_scale", "0.band_bias", "0.warp_ctrl"}
    assert set(optimized_names) == set(named) - set(carrier_names)
    assert optimized_ids == {id(named[name]) for name in optimized_names}
    assert all(not named[name].requires_grad for name in carrier_names)
    assert all(id(named[name]) not in optimized_ids for name in carrier_names)


def _smoke_cfg(root: Path):
    return OmegaConf.create(
        {
            "only_source": 0,
            "seed_run": 2025,
            "batch_size": 128,
            "num_workers": 0,
            "src_epoch": 999,
            "smoke_mode": True,
            "smoke_epochs": 1,
            "hust_checkpoint_root": str(root),
            "model_name": "tiny.pt",
            "Dataset": {
                "data_name": "HUSTStrict",
                "data_path": "unused",
                "TL_list": [0, 1, 2, 3],
                "TL_Task": [0, 1],
                "input_kind": "fft",
            },
            "Model": {
                "model_name": "ResNet18_1D_SDE",
                "use_spectral_adapter": True,
                "band_num": 256,
                "input_len": 512,
            },
            "Opt": {"name": "adamw", "lr_src": 9.0, "weight_decay_src": 9.0},
        }
    )


def _install_tiny_source(monkeypatch, source_training, *, empty=False):
    import Dataset

    class SourceOnlyHUST:
        num_classes = 2

        def _load(self, domain):
            assert int(domain) == self.source
            count = 0 if empty else 4
            x = torch.arange(count * 8, dtype=torch.float32).reshape(count, 1, 8)
            if count:
                x = x / float(count * 8)
            return TensorDataset(
                x,
                torch.tensor([0, 1, 0, 1], dtype=torch.int64)[:count],
                torch.arange(count),
            )

    monkeypatch.setattr(Dataset, "HUSTStrict", SourceOnlyHUST)
    monkeypatch.setattr(source_training, "get_model", lambda **_kwargs: _tiny_model())


@pytest.mark.parametrize(
    ("variant", "expected_terms"),
    [
        ("ordinary", {"clean"}),
        (
            "robust",
            {"clean", "style", "warp", "style_warp", "symmetric_kl", "feature"},
        ),
    ],
)
def test_source_runtime_record_reports_actual_aggregated_losses(
    variant, expected_terms, tmp_path, monkeypatch, capsys
):
    import Lib.hust_source_training as source_training

    _install_tiny_source(monkeypatch, source_training)
    actual_terms = []
    real_source_loss = source_training.source_loss

    def recording_source_loss(*args, **kwargs):
        loss, terms = real_source_loss(*args, **kwargs)
        actual_terms.append(
            {name: float(value.detach().cpu().item()) for name, value in terms.items()}
        )
        return loss, terms

    monkeypatch.setattr(source_training, "source_loss", recording_source_loss)
    root = tmp_path / f"{variant.upper()}_SMOKE"

    returned = train_hust_source(_smoke_cfg(root), source=0, variant=variant)
    checkpoint = resolve_hust_checkpoint(root, variant, 0, 2025, "tiny.pt")
    stored = json.loads(
        (checkpoint.parent / "source_training_summary.json").read_text(encoding="utf-8")
    )
    lines = capsys.readouterr().out.splitlines()
    result_lines = [line for line in lines if line.startswith("HUST_SOURCE_RESULT_JSON=")]

    assert len(actual_terms) == 1
    assert len(result_lines) == 1
    record = json.loads(result_lines[0].split("=", 1)[1])
    assert result_lines[0] == "HUST_SOURCE_RESULT_JSON=" + json.dumps(
        record, allow_nan=False, separators=(",", ":"), sort_keys=True
    )
    assert record == returned["source_result"] == stored["source_result"]
    assert set(record) == {
        "route",
        "source",
        "seed",
        "epochs",
        "optimizer_steps",
        "target_labels_consumed",
        "carrier_identity",
        "loss_terms",
    }
    assert record["route"] == variant
    assert record["source"] == 0
    assert record["seed"] == 2025
    assert record["epochs"] == 1
    assert record["optimizer_steps"] == 1
    assert record["target_labels_consumed"] is False
    assert record["carrier_identity"] is True
    assert set(record["loss_terms"]) == expected_terms
    for name, value in record["loss_terms"].items():
        assert math.isfinite(value)
        assert value == pytest.approx(actual_terms[0][name])


def test_source_training_rejects_an_empty_loader(tmp_path, monkeypatch, capsys):
    import Lib.hust_source_training as source_training

    _install_tiny_source(monkeypatch, source_training, empty=True)
    root = tmp_path / "EMPTY_SMOKE"

    with pytest.raises(RuntimeError, match="empty source loader"):
        train_hust_source(_smoke_cfg(root), source=0, variant="ordinary")

    assert "HUST_SOURCE_RESULT_JSON=" not in capsys.readouterr().out
    assert not hust_checkpoint_dir(root, "ordinary", 0, 2025).exists()


def test_smoke_train_reads_source_only_keeps_carrier_identity_and_formal_metadata(
    tmp_path, monkeypatch
):
    import Dataset
    import Lib.hust_source_training as source_training

    class SourceOnlyHUST:
        num_classes = 2
        target_load_attempts = 0

        def __init__(self, **_kwargs):
            raise AssertionError("constructing HUSTStrict validates target label tensors")

        def _load(self, domain):
            if int(domain) != self.source:
                type(self).target_load_attempts += 1
                raise AssertionError("target labels were inspected")
            x = torch.arange(32, dtype=torch.float32).reshape(4, 1, 8) / 32.0
            y = torch.tensor([0, 1, 0, 1])
            index = torch.arange(4)
            return TensorDataset(x, y, index)

        def data_generator(self):
            raise AssertionError("data_generator would construct target data")

    monkeypatch.setattr(Dataset, "HUSTStrict", SourceOnlyHUST)
    monkeypatch.setattr(source_training, "get_model", lambda **_kwargs: _tiny_model())
    root = tmp_path / "HUST_SOURCE_SMOKE"

    summary = train_hust_source(_smoke_cfg(root), source=0, variant="ordinary")

    assert SourceOnlyHUST.target_load_attempts == 0
    assert summary["target_labels_consumed"] is False
    assert summary["route"] == "ordinary"
    assert summary["hyperparameters"] == FORMAL_SOURCE_HYPERPARAMETERS
    assert summary["epochs"] == 50
    assert summary["execution"]["epochs_run"] == 1
    assert summary["execution"]["smoke_mode"] is True
    assert summary["carrier_identity_check"] is True
    assert summary["optimizer_steps"] == 1

    checkpoint = resolve_hust_checkpoint(root, "ordinary", 0, 2025, "tiny.pt")
    loaded_model = _tiny_model()
    metadata = strict_load_hust_checkpoint(loaded_model, checkpoint)
    state = loaded_model.state_dict()
    assert metadata == summary
    assert torch.count_nonzero(state["0.band_scale"]) == 0
    assert torch.count_nonzero(state["0.band_bias"]) == 0
    assert torch.count_nonzero(state["0.warp_ctrl"]) == 0


def test_smoke_override_is_rejected_outside_isolated_root(tmp_path):
    cfg = _smoke_cfg(tmp_path / "formal")

    with pytest.raises(ValueError, match="_SMOKE"):
        train_hust_source(cfg, source=0, variant="ordinary")


def test_strict_load_rejects_hash_and_route_metadata_tampering(tmp_path, monkeypatch):
    import Dataset
    import Lib.hust_source_training as source_training

    class SourceOnlyHUST:
        num_classes = 2

        def __init__(self, **_kwargs):
            raise AssertionError("whole-cache construction is forbidden")

        def _load(self, domain):
            assert int(domain) == self.source
            return TensorDataset(
                torch.randn(2, 1, 8), torch.tensor([0, 1]), torch.arange(2)
            )

    monkeypatch.setattr(Dataset, "HUSTStrict", SourceOnlyHUST)
    monkeypatch.setattr(source_training, "get_model", lambda **_kwargs: _tiny_model())
    root = tmp_path / "CONTRACT_SMOKE"
    train_hust_source(_smoke_cfg(root), source=0, variant="robust")
    checkpoint = resolve_hust_checkpoint(root, "robust", 0, 2025, "tiny.pt")
    summary_path = checkpoint.parent / "source_training_summary.json"
    metadata = json.loads(summary_path.read_text())
    original = dict(metadata)
    metadata["route"] = "ordinary"
    summary_path.write_text(json.dumps(metadata), encoding="utf-8")

    with pytest.raises(ValueError, match="route|metadata"):
        strict_load_hust_checkpoint(_tiny_model(), checkpoint)

    original["source_accuracy"] = original["source_accuracy"] + 1.0
    summary_path.write_text(json.dumps(original), encoding="utf-8")
    with pytest.raises(ValueError, match="metadata"):
        strict_load_hust_checkpoint(_tiny_model(), checkpoint)


@pytest.mark.parametrize("damage", ["missing_term", "extra_term", "non_finite"])
def test_strict_load_rejects_invalid_source_loss_diagnostics(
    damage, tmp_path, monkeypatch
):
    import Lib.hust_source_training as source_training

    _install_tiny_source(monkeypatch, source_training)
    variant = "ordinary" if damage == "extra_term" else "robust"
    root = tmp_path / f"LOSS_{damage.upper()}_SMOKE"
    train_hust_source(_smoke_cfg(root), source=0, variant=variant)
    checkpoint = resolve_hust_checkpoint(root, variant, 0, 2025, "tiny.pt")
    summary_path = checkpoint.parent / "source_training_summary.json"
    metadata = json.loads(summary_path.read_text(encoding="utf-8"))
    terms = metadata["source_result"]["loss_terms"]
    if damage == "missing_term":
        terms.pop("feature")
    elif damage == "extra_term":
        terms["style"] = terms["clean"]
    else:
        terms["feature"] = float("nan")
    summary_path.write_text(json.dumps(metadata), encoding="utf-8")

    with pytest.raises(ValueError, match="loss"):
        strict_load_hust_checkpoint(_tiny_model(), checkpoint)


def test_checkpoint_publish_failure_is_invisible_and_retryable(tmp_path, monkeypatch):
    import Dataset
    import Lib.hust_source_training as source_training

    class SourceOnlyHUST:
        num_classes = 2

        def _load(self, domain):
            assert int(domain) == self.source
            return TensorDataset(
                torch.randn(2, 1, 8), torch.tensor([0, 1]), torch.arange(2)
            )

    monkeypatch.setattr(Dataset, "HUSTStrict", SourceOnlyHUST)
    monkeypatch.setattr(source_training, "get_model", lambda **_kwargs: _tiny_model())
    original_write_text = Path.write_text
    calls = 0

    def fail_once(path, *args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise OSError("injected summary failure")
        return original_write_text(path, *args, **kwargs)

    monkeypatch.setattr(Path, "write_text", fail_once)
    root = tmp_path / "ATOMIC_SMOKE"
    cfg = _smoke_cfg(root)
    final_dir = hust_checkpoint_dir(root, "ordinary", 0, 2025)

    with pytest.raises(OSError, match="injected"):
        train_hust_source(cfg, source=0, variant="ordinary")
    assert not final_dir.exists()
    assert not list(final_dir.parent.glob(".seed_2025.tmp-*"))

    summary = train_hust_source(cfg, source=0, variant="ordinary")
    assert final_dir.is_dir()
    assert strict_load_hust_checkpoint(_tiny_model(), final_dir / "tiny.pt") == summary


def _tree_bytes(root: Path) -> dict[str, bytes | None]:
    if not root.exists():
        return {}
    return {
        str(path.relative_to(root)): (None if path.is_dir() else path.read_bytes())
        for path in sorted(root.rglob("*"))
    }


@pytest.mark.parametrize("layout", ["partial", "unrecognized"])
def test_preexisting_incomplete_contract_fails_closed_without_moving_or_rewriting_destination(
    layout, tmp_path, monkeypatch
):
    import Dataset
    import Lib.hust_source_training as source_training

    class SourceOnlyHUST:
        num_classes = 2

        def _load(self, domain):
            assert int(domain) == self.source
            return TensorDataset(
                torch.randn(2, 1, 8), torch.tensor([0, 1]), torch.arange(2)
            )

    monkeypatch.setattr(Dataset, "HUSTStrict", SourceOnlyHUST)
    monkeypatch.setattr(source_training, "get_model", lambda **_kwargs: _tiny_model())
    root = tmp_path / "PARTIAL_SMOKE"
    final_dir = hust_checkpoint_dir(root, "ordinary", 0, 2025)
    final_dir.mkdir(parents=True)
    unrelated = final_dir / "keep-me.txt"
    unrelated.write_text("user data", encoding="utf-8")
    if layout == "partial":
        (final_dir / "tiny.pt").write_bytes(b"partial")

    before = _tree_bytes(final_dir.parent)

    with pytest.raises(
        FileExistsError, match="pre-existing|partial|invalid|unrelated"
    ):
        train_hust_source(_smoke_cfg(root), source=0, variant="ordinary")

    assert _tree_bytes(final_dir.parent) == before
    assert not list(final_dir.parent.glob(".seed_2025.incomplete-*"))


@pytest.mark.parametrize("damage", ["truncated_summary", "invalid_hash", "corrupt_checkpoint"])
def test_damaged_complete_contract_fails_closed_without_mutating_destination(
    damage, tmp_path, monkeypatch
):
    import Dataset
    import Lib.hust_source_training as source_training
    from Lib.hust_strict_protocol import sha256_file

    class SourceOnlyHUST:
        num_classes = 2

        def _load(self, domain):
            assert int(domain) == self.source
            return TensorDataset(
                torch.randn(2, 1, 8), torch.tensor([0, 1]), torch.arange(2)
            )

    monkeypatch.setattr(Dataset, "HUSTStrict", SourceOnlyHUST)
    monkeypatch.setattr(source_training, "get_model", lambda **_kwargs: _tiny_model())
    root = tmp_path / "DAMAGED_SMOKE"
    cfg = _smoke_cfg(root)
    train_hust_source(cfg, source=0, variant="ordinary")
    final_dir = hust_checkpoint_dir(root, "ordinary", 0, 2025)
    checkpoint = final_dir / "tiny.pt"
    summary_path = final_dir / "source_training_summary.json"
    unrelated = final_dir / "keep-me.txt"
    unrelated.write_text("user data", encoding="utf-8")

    if damage == "truncated_summary":
        summary_path.write_text('{"route":', encoding="utf-8")
    elif damage == "invalid_hash":
        metadata = json.loads(summary_path.read_text(encoding="utf-8"))
        metadata["checkpoint_sha256"] = "0" * 64
        summary_path.write_text(json.dumps(metadata), encoding="utf-8")
    else:
        checkpoint.write_bytes(b"not a torch checkpoint")
        metadata = json.loads(summary_path.read_text(encoding="utf-8"))
        metadata["checkpoint_sha256"] = sha256_file(checkpoint)
        summary_path.write_text(json.dumps(metadata), encoding="utf-8")

    before = _tree_bytes(final_dir.parent)

    with pytest.raises(FileExistsError, match="pre-existing|invalid|damaged"):
        train_hust_source(cfg, source=0, variant="ordinary")

    assert _tree_bytes(final_dir.parent) == before
    assert not list(final_dir.parent.glob(".seed_2025.incomplete-*"))


@pytest.mark.parametrize(
    ("different_field", "different_value"),
    [("route", "robust"), ("source", 1), ("seed", 2026), ("batch_size", 64)],
)
def test_valid_complete_contract_with_different_expectation_is_protected(
    different_field, different_value, tmp_path, monkeypatch
):
    import Dataset
    import Lib.hust_source_training as source_training
    from Lib.hust_strict_protocol import sha256_file

    class SourceOnlyHUST:
        num_classes = 2

        def _load(self, domain):
            assert int(domain) == self.source
            return TensorDataset(
                torch.randn(2, 1, 8), torch.tensor([0, 1]), torch.arange(2)
            )

    monkeypatch.setattr(Dataset, "HUSTStrict", SourceOnlyHUST)
    monkeypatch.setattr(source_training, "get_model", lambda **_kwargs: _tiny_model())
    root = tmp_path / "PROTECTED_SMOKE"
    cfg = _smoke_cfg(root)
    summary = train_hust_source(cfg, source=0, variant="ordinary")
    final_dir = hust_checkpoint_dir(root, "ordinary", 0, 2025)
    checkpoint = final_dir / "tiny.pt"
    before_hash = sha256_file(checkpoint)
    different_expected = {
        "route": "ordinary",
        "source": 0,
        "seed": 2025,
        "epochs": 50,
        "hyperparameters": summary["hyperparameters"],
    }
    if different_field == "batch_size":
        different_expected["hyperparameters"] = {
            **summary["hyperparameters"],
            "batch_size": different_value,
        }
        expected_message = "hyperparameters"
    else:
        different_expected[different_field] = different_value
        expected_message = different_field

    with pytest.raises(FileExistsError, match=expected_message):
        source_training._existing_contract(
            _tiny_model(), checkpoint, different_expected
        )

    assert sha256_file(checkpoint) == before_hash
    assert not list(final_dir.parent.glob(".seed_2025.incomplete-*"))


def test_checkpoint_permission_error_is_not_quarantined(tmp_path, monkeypatch):
    import Dataset
    import Lib.hust_source_training as source_training

    class SourceOnlyHUST:
        num_classes = 2

        def _load(self, domain):
            assert int(domain) == self.source
            return TensorDataset(
                torch.randn(2, 1, 8), torch.tensor([0, 1]), torch.arange(2)
            )

    monkeypatch.setattr(Dataset, "HUSTStrict", SourceOnlyHUST)
    monkeypatch.setattr(source_training, "get_model", lambda **_kwargs: _tiny_model())
    root = tmp_path / "PERMISSION_SMOKE"
    cfg = _smoke_cfg(root)
    train_hust_source(cfg, source=0, variant="ordinary")
    final_dir = hust_checkpoint_dir(root, "ordinary", 0, 2025)
    summary_path = final_dir / "source_training_summary.json"
    original_read_text = Path.read_text

    def denied(path, *args, **kwargs):
        if path == summary_path:
            raise PermissionError("injected permission denial")
        return original_read_text(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", denied)
    with pytest.raises(PermissionError, match="injected"):
        train_hust_source(cfg, source=0, variant="ordinary")
    assert final_dir.is_dir()
    assert not list(final_dir.parent.glob(".seed_2025.incomplete-*"))


def test_resolve_never_crosses_variant_tree(tmp_path):
    robust_dir = hust_checkpoint_dir(tmp_path, "robust", 0, 2025)
    robust_dir.mkdir(parents=True)
    (robust_dir / "model.pt").touch()

    with pytest.raises(FileNotFoundError):
        resolve_hust_checkpoint(tmp_path, "ordinary", 0, 2025, "model.pt")


@pytest.mark.parametrize(
    ("module_name", "expected_variant"),
    [
        ("main_src_dtcc_hust_strict", "ordinary"),
        ("main_src_0711_hust_strict", "robust"),
    ],
)
def test_source_entrypoints_force_formal_contract_and_route(
    module_name, expected_variant, monkeypatch
):
    import importlib

    module = importlib.import_module(module_name)
    cfg = OmegaConf.create(
        {
            "only_source": 2,
            "seed_run": 7,
            "batch_size": 3,
            "num_workers": 0,
            "src_epoch": 4,
            "Dataset": {"data_name": "wrong", "TL_list": [9]},
            "Model": {"model_name": "wrong", "use_spectral_adapter": False},
            "Opt": {"name": "sgd", "lr_src": 9.0, "weight_decay_src": 9.0},
        }
    )
    observed = {}

    def fake_train(actual_cfg, source, variant):
        observed.update(cfg=actual_cfg, source=source, variant=variant)
        return {"route": variant}

    monkeypatch.setattr(module, "train_hust_source", fake_train)
    result = module.run.__wrapped__(cfg)

    assert result == {"route": expected_variant}
    assert observed["source"] == 2
    assert observed["variant"] == expected_variant
    assert cfg.seed_run == 2025
    assert cfg.batch_size == 128
    assert cfg.num_workers == 4
    assert cfg.src_epoch == 50
    assert cfg.Dataset.data_name == "HUSTStrict"
    assert list(cfg.Dataset.TL_list) == [0, 1, 2, 3]
    assert cfg.Model.model_name == "ResNet18_1D_SDE"
    assert cfg.Model.use_spectral_adapter is True
    assert cfg.Opt.name == "adamw"
    assert cfg.Opt.lr_src == 0.001
    assert cfg.Opt.weight_decay_src == 0.0001


def test_source_entrypoint_rejects_missing_only_source():
    import main_src_dtcc_hust_strict as module

    cfg = OmegaConf.create({"Dataset": {}, "Model": {}, "Opt": {}})
    with pytest.raises(ValueError, match="only_source"):
        module.run.__wrapped__(cfg)


@pytest.mark.parametrize(
    "script", ["main_src_dtcc_hust_strict.py", "main_src_0711_hust_strict.py"]
)
def test_exact_hydra_source_command_composes_without_plus_only_source(script):
    command = [
        sys.executable,
        str(ROOT / script),
        "Model=ResNet18_1D_SDE",
        "Dataset=HUSTStrict",
        "only_source=0",
        "batch_size=128",
        "src_epoch=50",
        "seed=2025",
        "Opt.lr_src=0.001",
        "Opt.weight_decay_src=0.0001",
        "--cfg",
        "job",
    ]

    completed = subprocess.run(
        command, cwd=ROOT, text=True, capture_output=True, check=False
    )

    assert completed.returncode == 0, completed.stderr
    assert "only_source: 0" in completed.stdout
    assert "data_name: HUSTStrict" in completed.stdout
    assert "model_name: ResNet18_1D_SDE" in completed.stdout
    assert "batch_size: 128" in completed.stdout
    assert "src_epoch: 50" in completed.stdout
    assert "lr_src: 0.001" in completed.stdout
    assert "weight_decay_src: 0.0001" in completed.stdout
