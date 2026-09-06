import ast
import csv
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
from types import SimpleNamespace

import omegaconf
import pytest
import torch

from Lib.hust_physical_fault_evidence import build_hust_physical_masks
from Lib.physical_fault_evidence import build_physical_masks
import main_tta_0711_strict_randomstream as base


ROOT = Path(__file__).resolve().parents[1]


def test_pu4d_default_hooks_preserve_physical_masks_without_metadata():
    trainer = base.Strict0711ResNetTrainer.__new__(base.Strict0711ResNetTrainer)
    trainer.target_domain = 1
    trainer.physical_config = base.PhysicalEvidenceConfig(spectrum_length=32)
    trainer.geometry = base.BearingGeometry()
    labels = torch.tensor([0, 7, 18, 25])

    actual = trainer.build_evidence_masks(labels, batch_metadata=None)
    expected = build_physical_masks(
        pseudo_labels=labels,
        target_domain=1,
        config=trainer.physical_config,
        geometry=trainer.geometry,
    )

    assert trainer.evidence_batch_metadata(torch.tensor([3, 1])) is None
    assert torch.equal(actual[0], expected[0])
    assert torch.equal(actual[1], expected[1])


def test_hust_metadata_and_masks_use_sample_indices_and_shaft_hz():
    import main_tta_0711_hust_strict as hust

    trainer = hust.HUST0711Trainer.__new__(hust.HUST0711Trainer)
    trainer.datasets = {
        "target_data": SimpleNamespace(shaft_hz=torch.tensor([10.0, 20.0, 30.0]))
    }
    trainer.device = torch.device("cpu")
    trainer.target_domain = 2
    trainer.physical_config = hust.HUSTPhysicalEvidenceConfig()

    metadata = trainer.evidence_batch_metadata(torch.tensor([2, 0]))
    labels = torch.tensor([1, 2])
    actual = trainer.build_evidence_masks(labels, metadata)
    expected = build_hust_physical_masks(
        labels, 2, torch.tensor([30.0, 10.0]), trainer.physical_config
    )

    assert metadata["shaft_hz"].tolist() == [30.0, 10.0]
    assert torch.equal(actual[0], expected[0])
    assert torch.equal(actual[1], expected[1])


def test_hust_checkpoint_routes_are_explicit_and_resolve_existing_file(tmp_path):
    from Lib.hust_strict_protocol import hust_checkpoint_dir, resolve_hust_checkpoint

    ordinary = hust_checkpoint_dir(tmp_path, "ordinary", source=1, seed=2025)
    robust = hust_checkpoint_dir(tmp_path, "robust", source=1, seed=2025)
    assert ordinary != robust

    robust.mkdir(parents=True)
    checkpoint = robust / "best_source_model.pt"
    checkpoint.write_bytes(b"checkpoint")
    assert resolve_hust_checkpoint(tmp_path, "robust", 1, 2025, "model.pt") == checkpoint

    with pytest.raises(ValueError, match="source_variant"):
        hust_checkpoint_dir(tmp_path, "typo", source=1, seed=2025)


def test_hust_trainer_rejects_invalid_source_variant(tmp_path):
    import main_tta_0711_hust_strict as hust

    trainer = hust.HUST0711Trainer.__new__(hust.HUST0711Trainer)
    trainer.cfg = SimpleNamespace(
        source_variant="unknown",
        hust_checkpoint_root=str(tmp_path),
        seed_run=2025,
        model_name="model.pt",
        Dataset=SimpleNamespace(TL_Task=[0, 1]),
    )
    with pytest.raises(ValueError, match="invalid source_variant"):
        trainer._checkpoint_path()


def test_hust_beginning_only_returns_before_optimizer_creation(monkeypatch, capsys):
    import main_tta_0711_hust_strict as hust

    trainer = hust.HUST0711Trainer.__new__(hust.HUST0711Trainer)
    trainer.cfg = SimpleNamespace(
        beginning_only=True,
        Dataset=SimpleNamespace(TL_Task=[0, 1]),
    )
    trainer.dataloaders = {"target_data": object()}
    trainer.student = object()
    trainer.initialize_models = lambda: None
    trainer.configure_trainable_parameters = lambda: pytest.fail(
        "Beginning-only mode created an optimizer"
    )
    monkeypatch.setattr(base, "cal_acc", lambda *_args, **_kwargs: (42.5,))

    result = trainer.adapt()

    assert result == {"before": 42.5}
    assert "Beginning Acc T = 42.50%" in capsys.readouterr().out


def test_hust_entrypoint_forces_protocol_configuration():
    import main_tta_0711_hust_strict as hust

    cfg = omegaconf.OmegaConf.create(
        {
            "Dataset": {"data_name": "wrong"},
            "Model": {
                "model_name": "wrong",
                "use_spectral_adapter": False,
                "band_num": 2,
                "input_len": 2,
            },
            "TTA0711": {"passes": 9, "stream_seed": 2026},
        }
    )
    hust.prepare_hust_config(cfg)

    assert cfg.Dataset.data_name == "HUSTStrict"
    assert cfg.Dataset.data_path == "Dataset/HUST_STRICT_CACHE_V2"
    assert list(cfg.Dataset.TL_list) == [0, 1, 2, 3]
    assert cfg.Model.model_name == "ResNet18_1D_SDE"
    assert cfg.Model.use_spectral_adapter is True
    assert cfg.Model.band_num == 256
    assert cfg.Model.input_len == 512
    assert cfg.batch_size == 128
    assert cfg.num_workers == 4
    assert cfg.TTA0711.passes == 1
    assert cfg.TTA0711.stream_seed == 2026
    assert cfg.TTA0711.min_pcl_classes == 3
    assert cfg.TTA0711.min_ncl_classes == 5
    assert cfg.TTA0711.min_ncl_entries == 20


def test_0711_trainable_parameters_are_adapter_and_warp_only():
    class Carrier(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.band_scale = torch.nn.Parameter(torch.zeros(1))
            self.band_bias = torch.nn.Parameter(torch.zeros(1))
            self.warp_ctrl = torch.nn.Parameter(torch.zeros(1))
            self.classifier = torch.nn.Linear(1, 1)

    trainer = base.Strict0711ResNetTrainer.__new__(base.Strict0711ResNetTrainer)
    trainer.student = Carrier()
    trainer.cfg = SimpleNamespace(
        Opt=SimpleNamespace(lr_tar=1e-3, weight_decay_tar=0.0)
    )
    trainer.tcfg = {}
    trainer.configure_trainable_parameters()

    trainable = {
        name for name, parameter in trainer.student.named_parameters()
        if parameter.requires_grad
    }
    assert trainable == {"band_scale", "band_bias", "warp_ctrl"}


def test_adapt_routes_indices_to_metadata_before_reliability_and_labels_only_to_diagnostics():
    tree = ast.parse((ROOT / "main_tta_0711_strict_randomstream.py").read_text())
    adapt = next(
        node for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "adapt"
    )
    calls = [node for node in ast.walk(adapt) if isinstance(node, ast.Call)]

    metadata_call = next(
        call for call in calls
        if isinstance(call.func, ast.Attribute)
        and call.func.attr == "evidence_batch_metadata"
    )
    reliability_call = next(
        call for call in calls
        if isinstance(call.func, ast.Attribute) and call.func.attr == "teacher_reliability"
    )
    diagnostics_call = next(
        call for call in calls
        if isinstance(call.func, ast.Attribute)
        and call.func.attr == "offline_diagnostics_update"
    )

    assert metadata_call.lineno < reliability_call.lineno < diagnostics_call.lineno
    assert {keyword.arg for keyword in reliability_call.keywords} == {
        "x", "iter_num", "batch_metadata"
    }
    assert all(
        not (isinstance(value, ast.Name) and value.id == "y")
        for value in [*reliability_call.args, *(kw.value for kw in reliability_call.keywords)]
    )


def test_dtcc_hust_route_resolves_only_ordinary_checkpoint(tmp_path):
    import main_tta_dtcc_hust_strict as hust_dtcc

    cfg = SimpleNamespace(
        hust_checkpoint_root=str(tmp_path),
        seed_run=2025,
        model_name="model.pt",
        Dataset=SimpleNamespace(TL_Task=[2, 1]),
    )
    ordinary = tmp_path / "ordinary" / "source_2" / "seed_2025"
    robust = tmp_path / "robust" / "source_2" / "seed_2025"
    ordinary.mkdir(parents=True)
    robust.mkdir(parents=True)
    expected = ordinary / "best_source_model.pt"
    expected.write_bytes(b"ordinary")
    (robust / "best_source_model.pt").write_bytes(b"robust")

    trainer = hust_dtcc.HUSTDtCCTrainer.__new__(hust_dtcc.HUSTDtCCTrainer)
    trainer.cfg = cfg

    assert trainer.checkpoint_path() == expected


def test_dtcc_hust_strict_load_records_verified_hash(tmp_path, monkeypatch, capsys):
    import main_tta_dtcc_hust_strict as hust_dtcc

    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"strict checkpoint")
    expected_hash = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    model = torch.nn.Linear(2, 2)
    trainer = hust_dtcc.HUSTDtCCTrainer.__new__(hust_dtcc.HUSTDtCCTrainer)
    trainer.model = model
    trainer.checkpoint_path = lambda: checkpoint

    def strict_loader(actual_model, actual_path):
        assert actual_model is model
        assert actual_path == checkpoint
        return {
            "route": "ordinary",
            "checkpoint_sha256": expected_hash,
            "carrier_parameters": ["0.band_scale", "0.band_bias", "0.warp_ctrl"],
        }

    monkeypatch.setattr(hust_dtcc, "strict_load_hust_checkpoint", strict_loader)

    trainer.initialize_common_source()

    assert trainer.checkpoint_metadata["route"] == "ordinary"
    assert trainer.checkpoint_metadata["checkpoint_sha256"] == expected_hash
    assert f"checkpoint_sha256={expected_hash}" in capsys.readouterr().out


def test_dtcc_hust_beginning_only_performs_no_optimizer_setup(capsys):
    import main_tta_dtcc_hust_strict as hust_dtcc

    trainer = hust_dtcc.HUSTDtCCTrainer.__new__(hust_dtcc.HUSTDtCCTrainer)
    trainer.cfg = SimpleNamespace(
        beginning_only=True,
        Dataset=SimpleNamespace(TL_Task=[0, 1]),
    )
    trainer.model = object()
    trainer.initialize_common_source = lambda: None
    trainer.evaluate = lambda _model: {
        "accuracy": 37.5,
        "macro_precision": 0.0,
        "macro_recall": 0.0,
        "macro_f1": 0.0,
    }
    trainer.configure_adaptation = lambda: pytest.fail(
        "Beginning-only mode configured adaptation"
    )

    result = trainer.adapt()

    assert result == {"before": trainer.evaluate(trainer.model)}
    assert "Beginning Acc T = 37.50%" in capsys.readouterr().out


def test_dtcc_hust_entrypoint_forces_protocol_but_preserves_declared_stream_seed():
    import main_tta_dtcc_hust_strict as hust_dtcc

    cfg = omegaconf.OmegaConf.create(
        {
            "batch_size": 3,
            "stream_seed": 9,
            "Dataset": {"data_name": "wrong"},
            "Model": {
                "model_name": "wrong",
                "use_spectral_adapter": False,
                "band_num": 2,
                "input_len": 2,
            },
            "DtCC": {
                "optim_steps": 7,
                "filter_k": 7,
                "neighbor_k": 7,
                "alpha": 7.0,
                "ncl_temperature": 7.0,
            },
        }
    )

    hust_dtcc.prepare_hust_config(cfg)

    assert cfg.Dataset.data_name == "HUSTStrict"
    assert cfg.Dataset.data_path == "Dataset/HUST_STRICT_CACHE_V2"
    assert list(cfg.Dataset.TL_list) == [0, 1, 2, 3]
    assert cfg.Model.model_name == "ResNet18_1D_SDE"
    assert cfg.Model.use_spectral_adapter is True
    assert cfg.Model.band_num == 256
    assert cfg.Model.input_len == 512
    assert cfg.batch_size == 128
    assert cfg.num_workers == 4
    assert cfg.stream_seed == 9
    assert cfg.DtCC.optim_steps == 2
    assert cfg.DtCC.filter_k == 50
    assert cfg.DtCC.neighbor_k == 5
    assert cfg.DtCC.alpha == 2.0
    assert cfg.DtCC.ncl_temperature == 0.1


def test_offline_diagnostics_are_detached_write_only_and_seven_class():
    import main_tta_dtcc_hust_strict as hust_dtcc

    truth = torch.tensor([0, 1, 2, 6], device="cpu")
    pseudo = torch.tensor([0, 2, 2, 6], device="cpu")
    certain = torch.tensor([True, False, True, True], device="cpu")
    diagnostics = hust_dtcc.OfflineDiagnosticsAccumulator(num_classes=7)

    assert diagnostics.update(truth, pseudo, certain) is None
    truth[0] = 5
    pseudo[0] = 5
    certain[0] = False
    metrics = diagnostics.metrics()

    assert metrics["pseudo_purity"] == pytest.approx(75.0)
    assert metrics["certain_purity"] == pytest.approx(100.0)
    assert metrics["class_coverage"] == 4
    assert metrics["confusion_matrix"].shape == (7, 7)
    assert int(metrics["confusion_matrix"].sum()) == 4


def test_dtcc_adapt_scores_pre_update_and_labels_only_feed_diagnostics():
    tree = ast.parse((ROOT / "main_tta_dtcc_hust_strict.py").read_text())
    trainer = next(
        node for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "HUSTDtCCTrainer"
    )
    adapt = next(
        node for node in trainer.body
        if isinstance(node, ast.FunctionDef) and node.name == "adapt"
    )
    calls = [node for node in ast.walk(adapt) if isinstance(node, ast.Call)]

    online_call = next(
        call for call in calls
        if isinstance(call.func, ast.Attribute) and call.func.attr == "update"
        and isinstance(call.func.value, ast.Name) and call.func.value.id == "online"
    )
    optimizer_zero = next(
        call for call in calls
        if isinstance(call.func, ast.Attribute) and call.func.attr == "zero_grad"
    )
    diagnostics_calls = sorted((
        call for call in calls
        if isinstance(call.func, ast.Attribute) and call.func.attr == "update"
        and isinstance(call.func.value, ast.Name)
        and call.func.value.id == "diagnostics"
    ), key=lambda call: call.lineno)

    assert len(diagnostics_calls) == 2
    assert (
        online_call.lineno
        < diagnostics_calls[0].lineno
        < optimizer_zero.lineno
        < diagnostics_calls[1].lineno
    )
    y_uses = [
        call for call in calls
        if any(isinstance(arg, ast.Name) and arg.id == "y" for arg in call.args)
    ]
    assert sorted(y_uses, key=lambda call: call.lineno) == [
        online_call,
        *diagnostics_calls,
    ]


def test_0711_hust_source_load_records_strict_checkpoint_hash(
    tmp_path, monkeypatch, capsys
):
    import main_tta_0711_hust_strict as hust

    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"strict 0711 checkpoint")
    expected_hash = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    model = torch.nn.Linear(2, 2)
    trainer = hust.HUST0711Trainer.__new__(hust.HUST0711Trainer)

    def strict_loader(actual_model, actual_path):
        assert actual_model is model
        assert actual_path == checkpoint
        return {
            "route": "robust",
            "checkpoint_sha256": expected_hash,
            "carrier_parameters": ["0.band_scale", "0.band_bias", "0.warp_ctrl"],
        }

    monkeypatch.setattr(hust, "strict_load_hust_checkpoint", strict_loader)

    trainer.load_source_checkpoint(model, checkpoint)

    assert trainer.checkpoint_metadata["route"] == "robust"
    assert trainer.checkpoint_metadata["checkpoint_sha256"] == expected_hash
    assert f"checkpoint_sha256={expected_hash}" in capsys.readouterr().out


@pytest.mark.parametrize("module_name", ["main_tta_dtcc_hust_strict", "main_tta_0711_hust_strict"])
@pytest.mark.parametrize(
    ("selector", "path", "domains"),
    [
        ("bearing", "Dataset/HUST_STRICT_CACHE_V2", [0, 1, 2, 3]),
        ("load", "Dataset/HUST_STRICT_LOAD_CACHE_V2", [0, 1, 2]),
    ],
)
def test_hust_protocol_split_is_closed_and_preserves_declared_dataset(module_name, selector, path, domains):
    module = __import__(module_name)
    cfg = omegaconf.OmegaConf.create({
        "hust_protocol_split": selector,
        "Dataset": {"data_name": "ignored", "data_path": "arbitrary", "TL_list": [9]},
        "Model": {}, "batch_size": 1,
    })
    module.prepare_hust_config(cfg)
    assert cfg.Dataset.data_path == path
    assert list(cfg.Dataset.TL_list) == domains
    cfg.hust_protocol_split = "arbitrary"
    with pytest.raises(ValueError, match="hust_protocol_split"):
        module.prepare_hust_config(cfg)


def test_0711_hust_offline_diagnostics_report_protocol_without_returning_metrics(capsys):
    import main_tta_0711_hust_strict as hust

    trainer = hust.HUST0711Trainer.__new__(hust.HUST0711Trainer)
    trainer.num_classes = 7
    trainer.stream_seed = 2025
    trainer.checkpoint_metadata = {
        "route": "robust",
        "checkpoint_sha256": "a" * 64,
    }
    trainer.student = torch.nn.Sequential(torch.nn.BatchNorm1d(2))
    trainer.student.requires_grad_(False)
    trainer.student[0].weight.requires_grad_(True)
    trainer.student[0].bias.requires_grad_(True)
    trainer._offline_finite_losses = True
    trainer._offline_losses_seen = 4
    trainer.offline_diagnostics_update(
        torch.tensor([0, 1, 6]),
        {
            "pseudo": torch.tensor([0, 2, 6]),
            "certain": torch.tensor([True, False, True]),
            "evidence_applicable": torch.tensor([False, True, True]),
        },
    )

    assert trainer.offline_diagnostics_finalize() is None
    output = capsys.readouterr().out

    assert "[HUST OFFLINE DIAGNOSTICS] pseudo_purity=66.67" in output
    assert "certain_purity=100.00" in output
    assert "confusion_matrix_7x7=" in output
    assert "checkpoint_route=robust" in output
    assert "checkpoint_sha256=" + "a" * 64 in output
    assert "passes=1 pre_update_scoring=True stream_seed=2025" in output
    assert "finite_losses=True losses_seen=4" in output
    assert "metadata_evidence_used=True" in output


def test_0711_keeps_student_online_and_teacher_pseudo_confusions_distinct(capsys):
    import main_tta_0711_hust_strict as hust

    trainer = hust.HUST0711Trainer.__new__(hust.HUST0711Trainer)
    trainer.num_classes = 7
    trainer.stream_seed = 2025
    trainer.checkpoint_metadata = {
        "route": "robust",
        "checkpoint_sha256": "a" * 64,
    }
    trainer.student = torch.nn.Sequential(torch.nn.BatchNorm1d(2))
    trainer.student.requires_grad_(False)
    trainer.student[0].weight.requires_grad_(True)
    trainer.student[0].bias.requires_grad_(True)
    trainer._offline_finite_losses = True
    trainer._offline_losses_seen = 0
    truth = torch.tensor([0, 1, 6])

    trainer.strict_online_diagnostics_update(
        truth, torch.tensor([1, 1, 6])
    )
    trainer.offline_diagnostics_update(
        truth,
        {
            "pseudo": torch.tensor([0, 2, 6]),
            "certain": torch.tensor([True, False, True]),
            "evidence_applicable": torch.tensor([False, True, True]),
        },
    )
    trainer.offline_diagnostics_finalize()

    online = trainer._final_online_metrics["confusion_matrix"]
    teacher = trainer._final_offline_metrics["confusion_matrix"]
    assert online.tolist() == [
        [0, 1, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 1],
    ]
    assert teacher.tolist() == [
        [1, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 1],
    ]
    assert not torch.equal(online, teacher)
    assert trainer._final_offline_metrics["pseudo_purity"] == pytest.approx(66.6666667)


def test_0711_post_stream_diagnostics_compute_all_macro_metrics():
    import main_tta_0711_hust_strict as hust

    trainer = hust.HUST0711Trainer.__new__(hust.HUST0711Trainer)
    trainer.num_classes = 7
    evaluation = (
        2.0 / 3.0 * 100.0,
        torch.empty(0),
        torch.empty(0),
        torch.tensor([0, 1, 6]),
        torch.tensor([0, 2, 6]),
    )

    metrics = trainer.post_stream_diagnostics(evaluation)

    assert metrics == {
        "post_macro_precision": pytest.approx(100.0 * 2.0 / 7.0),
        "post_macro_recall": pytest.approx(100.0 * 2.0 / 7.0),
        "post_macro_f1": pytest.approx(100.0 * 2.0 / 7.0),
    }


@pytest.mark.parametrize(
    ("field", "bad_value"),
    [
        ("loss_sem", float("nan")),
        ("loss_mt", float("inf")),
        ("loss_pcl", float("nan")),
        ("loss_ncl", float("inf")),
        ("loss_adapter", float("nan")),
        ("loss_warp", float("inf")),
        ("component_loss", float("nan")),
        ("total_loss", float("inf")),
    ],
)
def test_0711_hust_composed_loss_hook_reports_every_nonfinite_path(
    field, bad_value
):
    import main_tta_0711_hust_strict as hust

    trainer = hust.HUST0711Trainer.__new__(hust.HUST0711Trainer)
    trainer._offline_finite_losses = True
    trainer._offline_losses_seen = 0
    losses = {
        "loss_sem": torch.tensor(1.0),
        "loss_mt": torch.tensor(2.0),
        "loss_pcl": torch.tensor(3.0),
        "loss_ncl": torch.tensor(4.0),
        "loss_adapter": torch.tensor(5.0),
        "loss_warp": torch.tensor(6.0),
        "component_loss": torch.tensor(7.0),
        "total_loss": torch.tensor(8.0),
    }
    losses[field] = torch.tensor(bad_value)
    before = {name: value.clone() for name, value in losses.items()}

    assert trainer.offline_loss_diagnostics_update(**losses) is None

    assert trainer._offline_finite_losses is False
    assert trainer._offline_losses_seen == 8
    for name, value in losses.items():
        assert torch.allclose(value, before[name], equal_nan=True)


def test_pu4d_default_composed_loss_hook_is_noop():
    trainer = base.Strict0711ResNetTrainer.__new__(base.Strict0711ResNetTrainer)
    losses = {
        name: torch.tensor(float(index), requires_grad=True)
        for index, name in enumerate(
            (
                "loss_sem",
                "loss_mt",
                "loss_pcl",
                "loss_ncl",
                "loss_adapter",
                "loss_warp",
                "component_loss",
                "total_loss",
            ),
            start=1,
        )
    }

    assert trainer.offline_loss_diagnostics_update(**losses) is None
    assert [float(loss.detach()) for loss in losses.values()] == list(
        map(float, range(1, 9))
    )
    assert all(loss.requires_grad for loss in losses.values())


class _FakeOptimizer:
    def __init__(self, events=None):
        self.zero_calls = 0
        self.step_calls = 0
        self.events = events

    def zero_grad(self, **_kwargs):
        self.zero_calls += 1

    def step(self):
        self.step_calls += 1
        if self.events is not None:
            self.events.append("step")


class _FakeMemory:
    supports = torch.zeros(1, 2)
    scores = torch.ones(1, 7) / 7

    def __init__(self):
        self.update_calls = 0

    def snapshot(self):
        return object()

    def update(self, *_args, **_kwargs):
        self.update_calls += 1
        return None

    def prototypes(self):
        return torch.zeros(7, 2)

    def slim(self, _filter_k):
        return None

    def covered_classes(self):
        return 1

    def __len__(self):
        return 1


@pytest.mark.parametrize(
    ("batch_sizes", "expected_samples", "expected_steps", "expected_coverage"),
    [([1], 1, 0, 1), ([2, 1], 3, 2, 3)],
)
def test_dtcc_singletons_are_scored_once_but_never_adapted(
    batch_sizes,
    expected_samples,
    expected_steps,
    expected_coverage,
    monkeypatch,
    capsys,
):
    import main_tta_dtcc_hust_strict as hust_dtcc

    trainer = hust_dtcc.HUSTDtCCTrainer.__new__(hust_dtcc.HUSTDtCCTrainer)
    trainer.cfg = SimpleNamespace(
        beginning_only=False,
        batch_size=2,
        Dataset=SimpleNamespace(TL_Task=[0, 1]),
        DtCC=SimpleNamespace(
            optim_steps=2,
            filter_k=50,
            neighbor_k=1,
            alpha=2.0,
            ncl_temperature=0.1,
            log_interval=25,
        ),
    )
    trainer.run = None
    trainer.model = torch.nn.Sequential(torch.nn.Linear(1, 1))
    trainer.num_classes = 7
    trainer.device = torch.device("cpu")
    trainer.stream_seed = 2025
    trainer.checkpoint_metadata = {
        "route": "ordinary",
        "checkpoint_sha256": "b" * 64,
    }
    labels = iter(range(expected_samples))
    trainer.stream_loader = [
        (
            torch.arange(size, dtype=torch.float32).reshape(size, 1, 1),
            torch.tensor([next(labels) for _ in range(size)]),
            torch.arange(size),
        )
        for size in batch_sizes
    ]
    optimizer = _FakeOptimizer()
    memory = _FakeMemory()
    trainer.initialize_common_source = lambda: None
    trainer.evaluate = lambda _model: {
        "accuracy": 0.0,
        "macro_precision": 0.0,
        "macro_recall": 0.0,
        "macro_f1": 0.0,
    }
    trainer.configure_adaptation = lambda: (optimizer, memory)

    forward_batch_sizes = []

    def fake_forward(_model, x):
        forward_batch_sizes.append(x.size(0))
        feature = x.reshape(x.size(0), 1).repeat(1, 2).requires_grad_(True)
        logits = torch.full((x.size(0), 7), -5.0, requires_grad=True)
        logits = logits + torch.nn.functional.one_hot(
            torch.arange(x.size(0)) % 7, 7
        ).float() * 10.0
        return feature, logits

    monkeypatch.setattr(hust_dtcc, "forward_parts", fake_forward)
    monkeypatch.setattr(
        hust_dtcc,
        "dynamic_data_division",
        lambda probabilities, _entropy: (
            torch.ones(probabilities.size(0), dtype=torch.bool),
            torch.zeros(probabilities.size(0), dtype=torch.bool),
        ),
    )
    monkeypatch.setattr(
        hust_dtcc,
        "spectral_entropy",
        lambda x: torch.zeros(x.size(0)),
    )
    monkeypatch.setattr(
        hust_dtcc,
        "dtcc_sem_loss",
        lambda probabilities, _certain, alpha: probabilities.sum() * 0.0,
    )
    monkeypatch.setattr(
        hust_dtcc,
        "dtcc_pcl_loss",
        lambda *args, **kwargs: args[0].sum() * 0.0,
    )
    monkeypatch.setattr(
        hust_dtcc,
        "dtcc_ncl_loss",
        lambda features, *args, **kwargs: features.sum() * 0.0,
    )

    trainer.adapt()
    output = capsys.readouterr().out

    assert optimizer.step_calls == expected_steps
    assert optimizer.zero_calls == expected_steps
    assert memory.update_calls == expected_steps
    assert f"sample_count={expected_samples}" in output
    assert f"class_coverage={expected_coverage}/7" in output
    expected_size_two_forwards = 1 if batch_sizes == [1] else 4
    assert sum(size == 2 for size in forward_batch_sizes) == expected_size_two_forwards


@pytest.mark.parametrize(
    ("batch_sizes", "expected_samples", "expected_steps"),
    [([1], 1, 0), ([2, 1], 3, 1)],
)
def test_0711_singletons_are_scored_once_but_never_adapted(
    batch_sizes, expected_samples, expected_steps, monkeypatch, capsys
):
    import main_tta_0711_hust_strict as hust

    trainer = hust.HUST0711Trainer.__new__(hust.HUST0711Trainer)
    trainer.cfg = SimpleNamespace(
        beginning_only=False,
        process_wandb=False,
        Dataset=SimpleNamespace(TL_Task=[0, 1]),
    )
    trainer.run = None
    trainer.tcfg = {}
    trainer.device = torch.device("cpu")
    trainer.num_classes = 7
    trainer.stream_seed = 2025
    trainer.student = torch.nn.Linear(1, 7)
    trainer.dataloaders = {"target_data": object()}
    labels = iter(range(expected_samples))
    trainer.target_dataloader = [
        (
            torch.arange(size, dtype=torch.float32).reshape(size, 1),
            torch.tensor([next(labels) for _ in range(size)]),
            torch.arange(size),
        )
        for size in batch_sizes
    ]
    trainer.checkpoint_metadata = {
        "route": "robust",
        "checkpoint_sha256": "c" * 64,
    }
    optimizer = _FakeOptimizer()
    memory = _FakeMemory()
    events = []
    trainer.initialize_models = lambda: None
    trainer.configure_trainable_parameters = lambda: optimizer
    trainer._set_student_adaptation_mode = lambda: None
    trainer.evidence_batch_metadata = lambda indices: {"indices": indices}

    def reliability(x, iter_num, batch_metadata):
        count = x.size(0)
        return {
            "reliability": torch.ones(count),
            "pseudo": torch.arange(count) % 7,
            "certain": torch.ones(count, dtype=torch.bool),
            "teacher_prob": torch.ones(count, 7) / 7,
            "teacher_feature": torch.ones(count, 1),
            "evidence_applicable": torch.ones(count, dtype=torch.bool),
            "evidence": torch.ones(count),
            "confidence": torch.ones(count),
            "view_consistency": torch.ones(count),
            "mask_ratio": torch.zeros(count),
            "evidence_active": True,
        }

    trainer.teacher_reliability = reliability
    trainer.forward_parts = lambda model, x: (x, model(x))
    trainer._auxiliary_schedule = lambda _iteration: {
        "ramp": 0.0,
        "pcl_on": False,
        "ncl_on": False,
        "bank_size": 0,
        "bank_classes": 0,
    }
    trainer.adapter_reg_loss = lambda: torch.tensor(0.0)
    trainer.warp_reg_loss = lambda: torch.tensor(0.0)
    trainer.ema_update_teacher = lambda: None
    trainer.warp_delta_max = lambda: 0.0
    trainer.memory = memory
    monkeypatch.setattr(base, "cal_acc", lambda *_args, **_kwargs: (0.0,))
    monkeypatch.setattr(
        base,
        "reliability_weighted_sem",
        lambda logits, **_kwargs: (
            logits.sum() * 0.0,
            {"l_te": torch.tensor(0.0), "l_div": torch.tensor(0.0)},
        ),
    )
    monkeypatch.setattr(
        base,
        "reliability_weighted_mt",
        lambda logits, *_args: logits.sum() * 0.0,
    )

    original_hook = trainer.offline_loss_diagnostics_update

    def ordered_hook(**losses):
        events.append("observe")
        assert all(not loss.requires_grad for loss in losses.values())
        return original_hook(**losses)

    trainer.offline_loss_diagnostics_update = ordered_hook
    trainer.student.weight.register_hook(lambda _grad: events.append("backward"))
    optimizer.events = events

    trainer.adapt()
    output = capsys.readouterr().out

    assert optimizer.step_calls == expected_steps
    assert optimizer.zero_calls == expected_steps
    assert memory.update_calls == expected_steps
    assert f"sample_count={expected_samples}" in output
    assert trainer._final_online_metrics["sample_count"] == expected_samples
    assert trainer._final_offline_metrics["sample_count"] == expected_samples
    if expected_steps:
        assert events == ["observe", "backward", "step"]
    else:
        assert events == []


def _fake_launcher_environment(tmp_path):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    trace = tmp_path / "argv.trace"
    fake = bin_dir / "fake-command"
    fake.write_text(
        "#!/bin/bash\n"
        "printf '%s' \"${0##*/}\" >> \"$HUST_LAUNCHER_TRACE\"\n"
        "printf ' %q' \"$@\" >> \"$HUST_LAUNCHER_TRACE\"\n"
        "printf '\\n' >> \"$HUST_LAUNCHER_TRACE\"\n"
        "if [[ -n \"${HUST_FAIL_COMMAND:-}\" "
        "&& \"${0##*/}\" == \"$HUST_FAIL_COMMAND\" ]]; then exit 41; fi\n"
        "if [[ -n \"${HUST_FAIL_MATCH:-}\" "
        "&& \" $* \" == *\"$HUST_FAIL_MATCH\"* ]]; then exit 42; fi\n",
        encoding="utf-8",
    )
    fake.chmod(0o755)
    for name in ("bash", "nvidia-smi", "pytest", "python"):
        (bin_dir / name).symlink_to(fake)
    env = os.environ.copy()
    for name in ("RUN_DIR", "STAGE", "GPUS", "DRY_RUN"):
        env.pop(name, None)
    env.update(
        {
            "PATH": f"{bin_dir}:{os.environ['PATH']}",
            "HUST_LAUNCHER_TRACE": str(trace),
        }
    )
    return env, trace


@pytest.mark.parametrize(
    ("name", "hostile_value"),
    (
        ("RUN_DIR", "/tmp/formal-outer-run"),
        ("STAGE", "report"),
        ("GPUS", "7,5"),
        ("DRY_RUN", "1"),
    ),
)
def test_fake_launcher_environment_clears_ambient_launcher_controls(
    name, hostile_value, tmp_path, monkeypatch
):
    monkeypatch.setenv(name, hostile_value)
    monkeypatch.setenv("HUST_UNRELATED_SENTINEL", "preserved")

    env, trace = _fake_launcher_environment(tmp_path)

    assert name not in env
    assert env["HUST_UNRELATED_SENTINEL"] == "preserved"
    assert env["PATH"].split(os.pathsep)[0] == str(tmp_path / "bin")
    assert env["HUST_LAUNCHER_TRACE"] == str(trace)


def _launcher_marker(run_dir, *, repo=ROOT, config=None, basename=None):
    config = config or repo / "Configs/Experiments/HUST0711_strict_tuning.yaml"
    basename = basename or run_dir.name
    return (
        "schema=1\n"
        f"repo={repo.resolve()}\n"
        f"config={config.resolve()}\n"
        f"run={basename}"
    )


def _owned_run_dir(path, *, with_state=False):
    path.mkdir(parents=True)
    (path / ".hust_dtcc_0711_launcher").write_text(
        _launcher_marker(path), encoding="utf-8"
    )
    if with_state:
        state = path / "state"
        state.mkdir()
        (state / "formal.json").write_text("{}\n", encoding="utf-8")
    return path


def _run_launcher(env, launcher=None, **values):
    return subprocess.run(
        ["/bin/bash", launcher or ROOT / "run_hust_dtcc_0711_strict.sh"],
        cwd=ROOT,
        env={**env, **values},
        text=True,
        capture_output=True,
        check=False,
    )


def _write_formal_source_states(run_dir: Path, *, load_split: bool = False) -> list[Path]:
    evidence = run_dir / ("load-source-evidence" if load_split else "source-evidence")
    evidence.mkdir()
    inputs = {
        "cache_manifest": evidence / "manifest.json",
        "runner_script": evidence / "source_runner.py",
        "experiment_config": evidence / "experiment.yaml",
    }
    split = "load" if load_split else "bearing"
    inputs["cache_manifest"].write_text(json.dumps({"protocol": split}) + "\n")
    inputs["runner_script"].write_text("# controlled source runner\n")
    inputs["experiment_config"].write_text("source_seed: 2025\n")

    digest = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
    artifact_hashes = {str(path.resolve()): digest(path) for path in inputs.values()}
    state_dir = run_dir / "state"
    state_dir.mkdir(exist_ok=True)
    checkpoints = []
    for variant in ("ordinary", "robust"):
        for source in range(3 if load_split else 4):
            candidate = (
                f"load-source-{variant}-{source}"
                if load_split
                else f"source-{variant}-{source}-2025"
            )
            checkpoint = evidence / f"{candidate}.pt"
            checkpoint.write_bytes(f"unique checkpoint {candidate}\n".encode())
            checkpoints.append(checkpoint)
            summary = evidence / f"{candidate}.summary.json"
            metrics = {
                "route": variant,
                "source": source,
                "seed": 2025,
                "checkpoint_sha256": digest(checkpoint),
                "target_labels_consumed": False,
                "best_accuracy": 50.0 + source,
            }
            summary.write_text(json.dumps(metrics, sort_keys=True) + "\n")
            command = ["python", str(inputs["runner_script"]), candidate]
            command_path = evidence / f"{candidate}.attempt-1.txt"
            command_path.write_text(" ".join(command) + "\n")
            log_path = evidence / f"{candidate}.attempt-1.log"
            log_path.write_text("source complete\n")
            state = {
                "kind": "source",
                "stage": "load-audit-source" if load_split else "source",
                "route": f"source_{variant}",
                "variant": variant, "candidate_id": candidate, "source": source,
                "source_seed": 2025, "config_sha256": "a" * 64,
                "command": command,
                "command_sha256": hashlib.sha256(
                    json.dumps(command, sort_keys=True, separators=(",", ":")).encode()
                ).hexdigest(),
                "command_path": str(command_path.resolve()),
                "command_log_sha256": digest(command_path),
                "artifacts": list(artifact_hashes), "artifact_hashes": artifact_hashes,
                "gpu": source % 2, "status": "succeeded", "attempt": 1,
                "started_at": 1.0, "ended_at": 2.0, "returncode": 0,
                "log_path": str(log_path.resolve()), "log_sha256": digest(log_path),
                "metrics": metrics,
                "cache_manifest_path": str(inputs["cache_manifest"].resolve()),
                "cache_manifest_sha256": digest(inputs["cache_manifest"]),
                "runner_script_path": str(inputs["runner_script"].resolve()),
                "runner_script_sha256": digest(inputs["runner_script"]),
                "experiment_config_path": str(inputs["experiment_config"].resolve()),
                "experiment_config_sha256": digest(inputs["experiment_config"]),
                "source_checkpoint_path": str(checkpoint.resolve()),
                "source_checkpoint_sha256": digest(checkpoint),
                "source_summary_path": str(summary.resolve()),
                "source_summary_sha256": digest(summary),
                "expected_result_contract": "source", "expected_result_kind": "source",
                "output_artifact_hashes": {
                    str(checkpoint.resolve()): digest(checkpoint),
                    str(summary.resolve()): digest(summary),
                },
                "load_split": load_split,
            }
            (state_dir / f"{candidate}.json").write_text(
                json.dumps(state, sort_keys=True, indent=2) + "\n"
            )
    return checkpoints


def _rewrite_source_summary(state: dict) -> None:
    checkpoint = Path(state["source_checkpoint_path"])
    summary_path = Path(state["source_summary_path"])
    checkpoint_hash = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    summary = {
        **state["metrics"],
        "route": state["variant"],
        "source": state["source"],
        "checkpoint_sha256": checkpoint_hash,
    }
    summary_path.write_text(json.dumps(summary, sort_keys=True) + "\n")
    summary_hash = hashlib.sha256(summary_path.read_bytes()).hexdigest()
    state["metrics"] = summary
    state["source_checkpoint_sha256"] = checkpoint_hash
    state["source_summary_sha256"] = summary_hash
    state["output_artifact_hashes"] = {
        str(checkpoint.resolve()): checkpoint_hash,
        str(summary_path.resolve()): summary_hash,
    }


def _mutate_source_inventory(run_dir: Path, defect: str) -> None:
    state_paths = sorted((run_dir / "state").glob("*.json"))
    states = [json.loads(path.read_text(encoding="utf-8")) for path in state_paths]
    if defect == "seven_sources":
        state_paths[-1].unlink()
        return
    if defect == "duplicate_identity":
        states[1].update(
            variant=states[0]["variant"],
            source=states[0]["source"],
            route=states[0]["route"],
            candidate_id=states[0]["candidate_id"],
        )
        _rewrite_source_summary(states[1])
    elif defect == "duplicate_checkpoint_hash":
        Path(states[1]["source_checkpoint_path"]).write_bytes(
            Path(states[0]["source_checkpoint_path"]).read_bytes()
        )
        _rewrite_source_summary(states[1])
    elif defect == "unexpected_source":
        states[3].update(source=4, candidate_id="source-ordinary-4")
        _rewrite_source_summary(states[3])
    elif defect == "wrong_stage":
        states[0]["stage"] = "baseline"
    elif defect == "load_audit_source":
        states[0].update(
            stage="load-audit-source",
            load_split=True,
            candidate_id="load-source-ordinary-0",
        )
    elif defect == "persisted_checkpoint_sha":
        states[0]["source_checkpoint_sha256"] = "f" * 64
    elif defect == "persisted_summary_sha":
        states[0]["source_summary_sha256"] = "f" * 64
    else:
        raise AssertionError(f"unknown defect fixture: {defect}")
    for path, state in zip(state_paths, states, strict=True):
        path.write_text(json.dumps(state, sort_keys=True, indent=2) + "\n")


@pytest.mark.parametrize("existing_manifest", (False, True))
def test_hust_launcher_report_rebuilds_authoritative_checkpoint_manifest(
    tmp_path, existing_manifest
):
    run_dir = tmp_path / "HUST_DTCC_0711_STRICT_20260823_120017"
    _owned_run_dir(run_dir)
    checkpoints = _write_formal_source_states(run_dir)
    manifest_path = run_dir / "checkpoint_manifest.json"
    if existing_manifest:
        manifest_path.write_text("{}\n", encoding="utf-8")
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    nvidia_smi = bin_dir / "nvidia-smi"
    nvidia_smi.write_text("#!/bin/sh\nexit 99\n", encoding="utf-8")
    nvidia_smi.chmod(0o755)
    env = os.environ.copy()
    env["PATH"] = f"{bin_dir}:{env['PATH']}"

    completed = _run_launcher(
        env, STAGE="report", GPUS="auto", RUN_DIR=str(run_dir), DRY_RUN="0"
    )

    assert completed.returncode != 0
    assert "final inventory" in completed.stderr
    if existing_manifest:
        assert json.loads(manifest_path.read_text(encoding="utf-8")) == {}
    else:
        assert not manifest_path.exists()


def test_hust_launcher_report_separates_valid_load_sources_from_primary_manifest(tmp_path):
    run_dir = tmp_path / "HUST_DTCC_0711_STRICT_20260823_120017"
    _owned_run_dir(run_dir)
    primary_checkpoints = _write_formal_source_states(run_dir)
    load_checkpoints = _write_formal_source_states(run_dir, load_split=True)

    completed = _run_launcher(
        os.environ.copy(), STAGE="report", GPUS="auto", RUN_DIR=str(run_dir), DRY_RUN="0"
    )

    assert completed.returncode != 0
    assert "final inventory" in completed.stderr
    assert not (run_dir / "checkpoint_manifest.json").exists()
    assert not (run_dir / "supplementary_load_source_inventory.csv").exists()


def test_hust_launcher_report_still_formally_validates_load_source_evidence(tmp_path):
    run_dir = tmp_path / "HUST_DTCC_0711_STRICT_20260823_120017"
    _owned_run_dir(run_dir)
    _write_formal_source_states(run_dir)
    _write_formal_source_states(run_dir, load_split=True)
    load_state_path = next((run_dir / "state").glob("load-source-*.json"))
    load_state = json.loads(load_state_path.read_text())
    load_state["source_checkpoint_sha256"] = "f" * 64
    load_state_path.write_text(json.dumps(load_state, sort_keys=True, indent=2) + "\n")

    completed = _run_launcher(
        os.environ.copy(), STAGE="report", GPUS="auto", RUN_DIR=str(run_dir), DRY_RUN="0"
    )

    assert completed.returncode != 0
    assert "final inventory" in completed.stderr
    assert not (run_dir / "checkpoint_manifest.json").exists()
    assert not (run_dir / "supplementary_load_source_inventory.csv").exists()


@pytest.mark.parametrize(
    "defect",
    (
        "seven_sources",
        "duplicate_identity",
        "duplicate_checkpoint_hash",
        "unexpected_source",
        "wrong_stage",
        "load_audit_source",
        "persisted_checkpoint_sha",
        "persisted_summary_sha",
    ),
)
def test_hust_launcher_report_rejects_untrusted_primary_source_inventory_before_publication(
    tmp_path, defect
):
    run_dir = tmp_path / "HUST_DTCC_0711_STRICT_20260823_120017"
    _owned_run_dir(run_dir)
    _write_formal_source_states(run_dir)
    _mutate_source_inventory(run_dir, defect)
    env = os.environ.copy()

    completed = _run_launcher(
        env, STAGE="report", GPUS="auto", RUN_DIR=str(run_dir), DRY_RUN="0"
    )

    assert completed.returncode != 0, defect
    for artifact in (
        "metrics.csv",
        "beginning_audit.csv",
        "checkpoint_manifest.json",
        "protocol_audit.json",
        "report_summary.json",
        "report.md",
    ):
        assert not (run_dir / artifact).exists(), (defect, artifact)


def test_hust_launcher_dry_run_is_side_effect_free_and_shell_quotes_arguments(tmp_path):
    env, trace = _fake_launcher_environment(tmp_path)
    run_dir = (
        tmp_path
        / "parent with space"
        / "HUST_DTCC_0711_STRICT_20260823_120000"
    )

    completed = _run_launcher(
        env,
        DRY_RUN="1",
        STAGE="final",
        GPUS="2,0",
        RUN_DIR=str(run_dir),
    )

    assert completed.returncode == 0, completed.stderr
    assert not trace.exists()
    assert not run_dir.exists()
    assert "tools/tune_hust_0711_strict.py" in completed.stdout
    assert "Configs/Experiments/HUST0711_strict_tuning.yaml" in completed.stdout
    assert "--stage final" in completed.stdout
    assert "--gpus 2\\,0" in completed.stdout
    assert "parent\\ with\\ space" in completed.stdout


@pytest.mark.parametrize(
    "stage",
    ("source", "beginning", "baseline", "tune", "final", "load-audit", "all"),
)
def test_hust_launcher_delegates_every_formal_stage_with_exact_config(stage, tmp_path):
    env, trace = _fake_launcher_environment(tmp_path)
    run_dir = tmp_path / "HUST_DTCC_0711_STRICT_20260823_120001"
    _owned_run_dir(run_dir)

    completed = _run_launcher(
        env,
        STAGE=stage,
        GPUS="3,1",
        RUN_DIR=str(run_dir),
        DRY_RUN="0",
    )

    assert completed.returncode == 0, completed.stderr
    calls = trace.read_text(encoding="utf-8").splitlines()
    assert any(call.startswith("pytest ") for call in calls)
    assert any(call.startswith("bash -n ") for call in calls)
    assert any(call.startswith("nvidia-smi ") for call in calls)
    expected = (
        "python tools/tune_hust_0711_strict.py"
        " --config Configs/Experiments/HUST0711_strict_tuning.yaml"
        f" --stage {stage} --run-dir {run_dir} --gpus 3\\,1"
    )
    assert calls[-1] == expected


def test_hust_launcher_cache_stage_builds_only_primary_cache_without_gpu_access(tmp_path):
    env, trace = _fake_launcher_environment(tmp_path)
    run_dir = tmp_path / "HUST_DTCC_0711_STRICT_20260823_120002"

    completed = _run_launcher(
        env,
        STAGE="cache",
        RUN_DIR=str(run_dir),
        DRY_RUN="0",
    )

    assert completed.returncode == 0, completed.stderr
    calls = trace.read_text(encoding="utf-8").splitlines()
    assert not any(call.startswith("nvidia-smi ") for call in calls)
    assert calls[-1] == (
        "python tools/build_hust_strict_cache.py --raw-root Dataset/HUST"
        " --output Dataset/HUST_STRICT_CACHE_V2 --seed 2025"
    )


def test_hust_launcher_runs_checkpoint_audit_only_for_checkpoint_consumers(tmp_path):
    env, trace = _fake_launcher_environment(tmp_path)
    run_dir = tmp_path / "HUST_DTCC_0711_STRICT_20260823_120003"
    _owned_run_dir(run_dir)

    source = _run_launcher(env, STAGE="source", RUN_DIR=str(run_dir), DRY_RUN="0")
    source_calls = trace.read_text(encoding="utf-8").splitlines()
    trace.unlink()
    beginning = _run_launcher(
        env, STAGE="beginning", RUN_DIR=str(run_dir), DRY_RUN="0"
    )
    beginning_calls = trace.read_text(encoding="utf-8").splitlines()

    assert source.returncode == beginning.returncode == 0
    assert not any("_source_output" in call for call in source_calls)
    assert any("_source_output" in call for call in beginning_calls)


def test_hust_launcher_load_audit_does_not_require_preexisting_load_checkpoints(tmp_path):
    env, trace = _fake_launcher_environment(tmp_path)
    run_dir = tmp_path / "HUST_DTCC_0711_STRICT_20260823_120005"
    _owned_run_dir(run_dir)

    completed = _run_launcher(
        env, STAGE="load-audit", GPUS="0", RUN_DIR=str(run_dir), DRY_RUN="0"
    )

    assert completed.returncode == 0, completed.stderr
    calls = trace.read_text(encoding="utf-8").splitlines()
    assert any("Dataset/HUST_STRICT_LOAD_CACHE_V2" in call for call in calls)
    assert not any("_source_output" in call for call in calls)
    assert calls[-1].startswith("python tools/tune_hust_0711_strict.py")
    assert "--stage load-audit" in calls[-1]


@pytest.mark.parametrize("gpus", ("", "0,", ",0", "0,,1", "-1", "+1", "0, 1", "gpu0"))
def test_hust_launcher_rejects_malformed_gpu_list_before_side_effects(gpus, tmp_path):
    env, trace = _fake_launcher_environment(tmp_path)
    run_dir = tmp_path / "HUST_DTCC_0711_STRICT_20260823_120006"

    completed = _run_launcher(
        env, STAGE="source", GPUS=gpus, RUN_DIR=str(run_dir), DRY_RUN="0"
    )

    assert completed.returncode == 2
    assert "GPUS must be auto" in completed.stderr
    assert not trace.exists()
    assert not run_dir.exists()


def test_hust_launcher_rejects_markerless_or_mismatched_existing_run_dir(tmp_path):
    env, trace = _fake_launcher_environment(tmp_path)
    markerless = tmp_path / "HUST_DTCC_0711_STRICT_20260823_120007"
    markerless.mkdir()
    (markerless / "unrelated.txt").write_text("user data", encoding="utf-8")
    mismatched = tmp_path / "HUST_DTCC_0711_STRICT_20260823_120008"
    _owned_run_dir(mismatched)
    (mismatched / ".hust_dtcc_0711_launcher").write_text(
        _launcher_marker(mismatched, basename="HUST_DTCC_0711_STRICT_19990101_000000"),
        encoding="utf-8",
    )

    first = _run_launcher(
        env, STAGE="source", RUN_DIR=str(markerless), DRY_RUN="0"
    )
    second = _run_launcher(
        env, STAGE="source", RUN_DIR=str(mismatched), DRY_RUN="0"
    )

    assert first.returncode == second.returncode == 2
    assert "launcher ownership" in first.stderr
    assert "launcher ownership" in second.stderr
    assert (markerless / "unrelated.txt").read_text(encoding="utf-8") == "user data"
    assert not trace.exists()


@pytest.mark.parametrize(
    ("failure_env", "stage"),
    (
        ({"HUST_FAIL_COMMAND": "pytest"}, "source"),
        ({"HUST_FAIL_MATCH": "validate_cache"}, "source"),
        ({"HUST_FAIL_MATCH": "_source_output"}, "beginning"),
        ({"HUST_FAIL_COMMAND": "nvidia-smi"}, "source"),
        ({"HUST_FAIL_MATCH": "--dry-run"}, "cache"),
    ),
)
def test_hust_launcher_preflight_failure_leaves_new_run_dir_absent(
    failure_env, stage, tmp_path
):
    env, trace = _fake_launcher_environment(tmp_path)
    run_dir = tmp_path / "HUST_DTCC_0711_STRICT_20260823_120009"

    completed = _run_launcher(
        {**env, **failure_env}, STAGE=stage, RUN_DIR=str(run_dir), DRY_RUN="0"
    )

    assert completed.returncode != 0
    assert trace.exists()
    assert not run_dir.exists()
    calls = trace.read_text(encoding="utf-8").splitlines()
    assert not any(
        call.startswith("python tools/tune_hust_0711_strict.py") for call in calls
    )


def test_hust_launcher_report_is_owned_cpu_only_and_skips_training_preflight(tmp_path):
    env, trace = _fake_launcher_environment(tmp_path)
    run_dir = tmp_path / "HUST_DTCC_0711_STRICT_20260823_120010"
    _owned_run_dir(run_dir, with_state=True)

    completed = _run_launcher(
        env, STAGE="report", GPUS="auto", RUN_DIR=str(run_dir), DRY_RUN="0"
    )

    assert completed.returncode == 0, completed.stderr
    calls = trace.read_text(encoding="utf-8").splitlines()
    assert calls == [
        "python tools/tune_hust_0711_strict.py"
        " --config Configs/Experiments/HUST0711_strict_tuning.yaml"
        f" --stage report --run-dir {run_dir} --gpus auto"
    ]


def test_hust_launcher_all_audits_checkpoints_after_source_before_all(tmp_path):
    env, trace = _fake_launcher_environment(tmp_path)
    run_dir = tmp_path / "HUST_DTCC_0711_STRICT_20260823_120011"
    _owned_run_dir(run_dir)

    completed = _run_launcher(
        env, STAGE="all", GPUS="0,2", RUN_DIR=str(run_dir), DRY_RUN="0"
    )

    assert completed.returncode == 0, completed.stderr
    calls = trace.read_text(encoding="utf-8").splitlines()
    source_index = next(i for i, call in enumerate(calls) if "--stage source" in call)
    audit_index = next(i for i, call in enumerate(calls) if "_source_output" in call)
    all_index = next(i for i, call in enumerate(calls) if "--stage all" in call)
    assert source_index < audit_index < all_index


def test_hust_launcher_all_checkpoint_failure_stops_before_consuming_stages(tmp_path):
    env, trace = _fake_launcher_environment(tmp_path)
    env["HUST_FAIL_MATCH"] = "_source_output"
    run_dir = tmp_path / "HUST_DTCC_0711_STRICT_20260823_120012"

    completed = _run_launcher(
        env, STAGE="all", GPUS="0", RUN_DIR=str(run_dir), DRY_RUN="0"
    )

    assert completed.returncode != 0
    calls = trace.read_text(encoding="utf-8").splitlines()
    assert any("--stage source" in call for call in calls)
    assert any("_source_output" in call for call in calls)
    assert not any("--stage all" in call for call in calls)


def test_hust_launcher_new_run_dir_gets_matching_atomic_ownership_marker(tmp_path):
    env, _trace = _fake_launcher_environment(tmp_path)
    run_dir = tmp_path / "HUST_DTCC_0711_STRICT_20260823_120013"

    completed = _run_launcher(
        env, STAGE="source", GPUS="auto", RUN_DIR=str(run_dir), DRY_RUN="0"
    )

    assert completed.returncode == 0, completed.stderr
    marker = run_dir / ".hust_dtcc_0711_launcher"
    assert marker.read_text(encoding="utf-8") == _launcher_marker(run_dir)
    assert not list(run_dir.glob(".hust_dtcc_0711_launcher.tmp.*"))


def test_hust_launcher_default_timestamp_collision_never_reuses_directory(tmp_path):
    env, trace = _fake_launcher_environment(tmp_path)
    project = tmp_path / "project"
    project.mkdir()
    (project / "logs").mkdir()
    launcher = project / "run_hust_dtcc_0711_strict.sh"
    shutil.copy2(ROOT / "run_hust_dtcc_0711_strict.sh", launcher)
    collision = project / "logs/HUST_DTCC_0711_STRICT_21000101_000000"
    collision.mkdir()

    completed = _run_launcher(
        {**env, "SOURCE_DATE_EPOCH": "4102444800"},
        launcher=launcher,
        STAGE="source",
        DRY_RUN="0",
    )

    assert completed.returncode == 2
    assert "collision" in completed.stderr
    assert not trace.exists()
    assert list(collision.iterdir()) == []


def test_hust_launcher_resolves_project_directory_symlink_physically(tmp_path):
    env, _trace = _fake_launcher_environment(tmp_path)
    linked_project = tmp_path / "linked-project"
    linked_project.symlink_to(ROOT, target_is_directory=True)
    run_dir = tmp_path / "HUST_DTCC_0711_STRICT_20260823_120015"

    completed = _run_launcher(
        env,
        launcher=linked_project / "run_hust_dtcc_0711_strict.sh",
        STAGE="source",
        GPUS="0",
        RUN_DIR=str(run_dir),
        DRY_RUN="0",
    )

    assert completed.returncode == 0, completed.stderr
    assert (run_dir / ".hust_dtcc_0711_launcher").read_text(
        encoding="utf-8"
    ) == _launcher_marker(run_dir)


def test_hust_launcher_load_cache_failure_stops_before_run_dir_and_delegation(tmp_path):
    env, trace = _fake_launcher_environment(tmp_path)
    env["HUST_FAIL_MATCH"] = "validate_cache"
    run_dir = tmp_path / "HUST_DTCC_0711_STRICT_20260823_120014"

    completed = _run_launcher(
        env, STAGE="load-audit", GPUS="auto", RUN_DIR=str(run_dir), DRY_RUN="0"
    )

    assert completed.returncode != 0
    assert not run_dir.exists()
    calls = trace.read_text(encoding="utf-8").splitlines()
    assert any("Dataset/HUST_STRICT_LOAD_CACHE_V2" in call for call in calls)
    assert not any(
        call.startswith("python tools/tune_hust_0711_strict.py") for call in calls
    )


def test_hust_launcher_rejects_invalid_run_basename_before_side_effects(tmp_path):
    env, trace = _fake_launcher_environment(tmp_path)
    run_dir = tmp_path / "ordinary-results"

    completed = _run_launcher(
        env, STAGE="source", GPUS="auto", RUN_DIR=str(run_dir), DRY_RUN="0"
    )

    assert completed.returncode == 2
    assert "RUN_DIR must end" in completed.stderr
    assert not trace.exists()
    assert not run_dir.exists()


def test_hust_launcher_report_rejects_missing_state_without_external_calls(tmp_path):
    env, trace = _fake_launcher_environment(tmp_path)
    run_dir = tmp_path / "HUST_DTCC_0711_STRICT_20260823_120016"
    _owned_run_dir(run_dir)

    completed = _run_launcher(
        env, STAGE="report", GPUS="auto", RUN_DIR=str(run_dir), DRY_RUN="0"
    )

    assert completed.returncode == 2
    assert "state inputs" in completed.stderr
    assert not trace.exists()


def test_hust_launcher_rejects_unknown_stage_before_commands_or_artifacts(tmp_path):
    env, trace = _fake_launcher_environment(tmp_path)
    run_dir = tmp_path / "HUST_DTCC_0711_STRICT_20260823_120004"

    completed = _run_launcher(
        env, STAGE="typo", RUN_DIR=str(run_dir), DRY_RUN="0"
    )

    assert completed.returncode != 0
    assert "unsupported STAGE" in completed.stderr
    assert not trace.exists()
    assert not run_dir.exists()


def test_hust_launcher_has_strict_shell_wiring_and_no_destructive_commands():
    text = (ROOT / "run_hust_dtcc_0711_strict.sh").read_text(encoding="utf-8")
    assert "set -euo pipefail" in text
    assert "rm -rf" not in text
    assert "kill" not in text
    assert "reset" not in text
