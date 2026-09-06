import hashlib
import json
from pathlib import Path
import re
import warnings

import numpy as np
from omegaconf import OmegaConf
import pytest
import torch
from scipy.io import savemat

from Lib.hust_strict_protocol import LABEL_MAP, parse_hust_filename, validate_cache
from tools.build_hust_strict_cache import (
    balanced_window_indices,
    build_cache,
    fft_window,
)


@pytest.fixture
def synthetic_raw_hust(tmp_path):
    root = tmp_path / "raw"
    root.mkdir()
    signal = np.sin(2 * np.pi * 8 * np.arange(4096) / 2048).astype(np.float32)
    for bearing in range(4, 9):
        for fault in LABEL_MAP:
            if bearing == 4 and fault in {"B", "IB"}:
                continue
            for load_code in ("00", "02", "04"):
                savemat(
                    root / f"{fault}{bearing}{load_code}.mat",
                    {"data": signal[:, None], "fs": [[24.0]]},
                )
    return root


@pytest.mark.parametrize(
    ("name", "fault", "bearing", "load"),
    [
        ("IB504.mat", "IB", 6205, 400),
        ("IO602.mat", "IO", 6206, 200),
        ("OB700.mat", "OB", 6207, 0),
        ("N800.mat", "N", 6208, 0),
    ],
)
def test_parse_hust_filename_uses_longest_prefix(name, fault, bearing, load):
    row = parse_hust_filename(name)
    assert (row.fault, row.label, row.bearing, row.load_w) == (
        fault,
        LABEL_MAP[fault],
        bearing,
        load,
    )


@pytest.mark.parametrize(
    "name",
    ["IB50.mat", "N501.mat", "N506.mat", "X500.mat", "N500.csv"],
)
def test_parse_rejects_runup_or_unknown_names(name):
    with pytest.raises(ValueError, match="unsupported HUST filename"):
        parse_hust_filename(name)


def test_fft_window_has_exact_grid_and_scale():
    n = np.arange(2048)
    signal = np.sin(2 * np.pi * 8 * n / 2048).astype(np.float32)
    spectrum = fft_window(signal)
    assert spectrum.shape == (1, 512)
    assert spectrum.dtype == np.float32
    assert int(spectrum[0].argmax()) == 8
    assert spectrum[0, 8] == pytest.approx(0.5, rel=1e-5)


def test_fft_window_rejects_wrong_length():
    with pytest.raises(ValueError, match="2048"):
        fft_window(np.zeros(2047, dtype=np.float32))


def test_balanced_indices_are_deterministic_and_equal():
    available = {"a": 499, "b": 430, "c": 470}
    first = balanced_window_indices(available, seed=2025)
    second = balanced_window_indices(available, seed=2025)
    reordered = balanced_window_indices(
        dict(reversed(list(available.items()))), seed=2025
    )
    different_seed = balanced_window_indices(available, seed=2026)
    assert first.keys() == second.keys()
    assert all(np.array_equal(first[k], second[k]) for k in first)
    assert all(np.array_equal(first[k], reordered[k]) for k in first)
    assert any(not np.array_equal(first[k], different_seed[k]) for k in first)
    assert {len(v) for v in first.values()} == {430}
    assert all(np.all(np.diff(v) >= 0) for v in first.values())


def test_bearing_split_builds_four_balanced_domains_and_valid_manifest(
    tmp_path, synthetic_raw_hust
):
    output = tmp_path / "bearing"
    manifest = build_cache(synthetic_raw_hust, output, seed=2025)

    assert manifest["split"] == "bearing"
    assert manifest["domain_map"] == {
        "0": "6205",
        "1": "6206",
        "2": "6207",
        "3": "6208",
    }
    assert manifest["balance_count"] == 3
    assert len(manifest["source_files"]) == 84
    assert all(len(row["sha256"]) == 64 for row in manifest["source_files"])
    for domain, row in manifest["domains"].items():
        assert row["shape"] == [63, 1, 512]
        assert set(row["class_counts"]) == {str(i) for i in range(7)}
        assert set(row["load_counts"]) == {"0", "200", "400"}
        obj = torch.load(output / f"domain_{domain}.pt", weights_only=True)
        assert set(obj) == {
            "data",
            "label",
            "shaft_hz",
            "load_w",
            "recording_id",
            "window_offset",
        }
        assert obj["window_offset"].unique().tolist() == [0, 1024, 2048]

    validated = validate_cache(output)
    expected_sha = hashlib.sha256((output / "manifest.json").read_bytes()).hexdigest()
    assert validated["manifest_sha256"] == expected_sha


def test_v2_manifest_binds_every_domain_tensor_and_rejects_same_shape_value_tamper(
    tmp_path, synthetic_raw_hust
):
    """Removing domain SHA verification would let a value-only tensor edit pass."""
    output = tmp_path / "bearing-v2"
    manifest = build_cache(synthetic_raw_hust, output, seed=2025)

    assert manifest["version"] == 2
    assert re.fullmatch(r"[0-9a-f]{64}", manifest["content_sha256"])
    for domain, summary in manifest["domains"].items():
        tensor = output / f"domain_{domain}.pt"
        assert summary["tensor_file"] == tensor.name
        assert summary["tensor_sha256"] == hashlib.sha256(tensor.read_bytes()).hexdigest()

    tensor = output / "domain_0.pt"
    payload = torch.load(tensor, map_location="cpu", weights_only=True)
    payload["data"][0, 0, 0] += 0.125
    torch.save(payload, tensor)

    with pytest.raises(ValueError, match="tensor hash"):
        validate_cache(output)


@pytest.mark.parametrize("mutation", ["missing", "swapped"])
def test_v2_manifest_rejects_missing_or_swapped_domain_file(
    tmp_path, synthetic_raw_hust, mutation
):
    """File-name/content binding must catch both absence and a valid tensor in the wrong slot."""
    output = tmp_path / f"bearing-{mutation}"
    build_cache(synthetic_raw_hust, output, seed=2025)
    first, second = output / "domain_0.pt", output / "domain_1.pt"
    if mutation == "missing":
        first.unlink()
    else:
        first_bytes, second_bytes = first.read_bytes(), second.read_bytes()
        first.write_bytes(second_bytes)
        second.write_bytes(first_bytes)

    with pytest.raises(ValueError, match="tensor"):
        validate_cache(output)


def test_load_split_contract_has_three_complete_domains(tmp_path, synthetic_raw_hust):
    manifest = build_cache(
        synthetic_raw_hust, tmp_path / "load", seed=2025, split="load"
    )
    assert manifest["split"] == "load"
    assert manifest["domain_map"] == {"0": "0W", "1": "200W", "2": "400W"}
    for row in manifest["domains"].values():
        assert set(row["class_counts"]) == {"0", "1", "2", "3", "4", "5", "6"}
        assert len(set(row["class_counts"].values())) == 1


@pytest.mark.parametrize(
    "payload",
    [
        {"fs": [[24.0]]},
        {"data": np.zeros((4096, 1), dtype=np.float32)},
        {"data": np.zeros((4096, 1), dtype=np.float32), "fs": [[0.0]]},
        {"data": np.zeros((4096, 1), dtype=np.float32), "fs": [[np.nan]]},
        {"data": np.zeros((4096, 1), dtype=np.float32), "fs": [[24.0, 25.0]]},
    ],
)
def test_builder_rejects_invalid_mat_contract(tmp_path, payload):
    raw = tmp_path / "raw"
    raw.mkdir()
    savemat(raw / "N500.mat", payload)
    with pytest.raises(ValueError):
        build_cache(raw, tmp_path / "out", seed=2025)


def test_existing_identical_cache_is_reused_but_nonidentical_is_refused(
    tmp_path, synthetic_raw_hust
):
    output = tmp_path / "cache"
    first = build_cache(synthetic_raw_hust, output, seed=2025)
    second = build_cache(synthetic_raw_hust, output, seed=2025)
    assert second["manifest_sha256"] == validate_cache(output)["manifest_sha256"]
    assert second["manifest_sha256"] == first["manifest_sha256"]

    manifest_path = output / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["seed"] = 7
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(FileExistsError, match="non-identical"):
        build_cache(synthetic_raw_hust, output, seed=2025)


def test_builder_rejects_unknown_mat_among_primary_recordings(
    tmp_path, synthetic_raw_hust
):
    savemat(
        synthetic_raw_hust / "runup.mat",
        {"data": np.zeros((4096, 1), dtype=np.float32), "fs": [[24.0]]},
    )
    with pytest.raises(ValueError, match="unsupported HUST filename"):
        build_cache(synthetic_raw_hust, tmp_path / "out", seed=2025)


def test_validate_cache_rejects_manifest_contract_mismatch(
    tmp_path, synthetic_raw_hust
):
    output = tmp_path / "cache"
    build_cache(synthetic_raw_hust, output, seed=2025)
    manifest_path = output / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["processing"]["window_stride"] = 1
    manifest["domains"]["0"]["class_counts"]["0"] = 0
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError):
        validate_cache(output)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("data", np.ones((4096, 1), dtype=np.complex64) * (1 + 2j)),
        ("fs", np.asarray([[24.0 + 2.0j]], dtype=np.complex64)),
    ],
)
def test_builder_rejects_complex_mat_values_without_warnings(
    tmp_path, synthetic_raw_hust, field, value
):
    path = synthetic_raw_hust / "N500.mat"
    payload = {
        "data": np.zeros((4096, 1), dtype=np.float32),
        "fs": np.asarray([[24.0]], dtype=np.float32),
    }
    payload[field] = value
    savemat(path, payload)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with pytest.raises(ValueError, match="real"):
            build_cache(synthetic_raw_hust, tmp_path / "out", seed=2025)
    assert caught == []


def _rewrite_domain(output, domain, mutate):
    path = output / f"domain_{domain}.pt"
    obj = torch.load(path, weights_only=True)
    mutate(obj)
    torch.save(obj, path)
    manifest_path = output / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    labels = obj["label"]
    loads = obj["load_w"]
    manifest["domains"][str(domain)] = {
        "shape": list(obj["data"].shape),
        "class_counts": {
            str(label): int((labels == label).sum()) for label in range(7)
        },
        "load_counts": {
            str(load): int((loads == load).sum()) for load in (0, 200, 400)
        },
        "tensor_file": path.name,
        "tensor_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }
    tensor_identity = {
        key: {
            "tensor_file": row["tensor_file"],
            "tensor_sha256": row["tensor_sha256"],
        }
        for key, row in manifest["domains"].items()
    }
    manifest["content_sha256"] = hashlib.sha256(
        json.dumps(
            {"version": 2, "split": manifest["split"], "domains": tensor_identity},
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")


def test_validate_cache_rejects_empty_domain_even_with_matching_summary(
    tmp_path, synthetic_raw_hust
):
    output = tmp_path / "cache"
    build_cache(synthetic_raw_hust, output, seed=2025)

    def empty(obj):
        for key in obj:
            obj[key] = obj[key][:0]

    _rewrite_domain(output, 0, empty)
    with pytest.raises(ValueError, match="empty"):
        validate_cache(output)


@pytest.mark.parametrize("split", ["bearing", "load"])
def test_validate_cache_rejects_unequal_classes_for_each_split(
    tmp_path, synthetic_raw_hust, split
):
    output = tmp_path / split
    build_cache(synthetic_raw_hust, output, seed=2025, split=split)
    _rewrite_domain(output, 0, lambda obj: obj["label"].__setitem__(0, 1))
    with pytest.raises(ValueError, match="class balance"):
        validate_cache(output)


def test_validate_cache_rejects_unequal_primary_loads(
    tmp_path, synthetic_raw_hust
):
    output = tmp_path / "cache"
    build_cache(synthetic_raw_hust, output, seed=2025)
    _rewrite_domain(output, 0, lambda obj: obj["load_w"].__setitem__(0, 200))
    with pytest.raises(ValueError, match="load balance"):
        validate_cache(output)


def test_validate_cache_rejects_unknown_load_value(tmp_path, synthetic_raw_hust):
    output = tmp_path / "cache"
    build_cache(synthetic_raw_hust, output, seed=2025)
    _rewrite_domain(output, 0, lambda obj: obj["load_w"].__setitem__(0, 123))
    with pytest.raises(ValueError, match="load"):
        validate_cache(output)


@pytest.mark.parametrize("recording_id", [-1, 84])
def test_validate_cache_rejects_recording_id_out_of_bounds(
    tmp_path, synthetic_raw_hust, recording_id
):
    output = tmp_path / "cache"
    build_cache(synthetic_raw_hust, output, seed=2025)
    _rewrite_domain(
        output,
        0,
        lambda obj: obj["recording_id"].__setitem__(0, recording_id),
    )
    with pytest.raises(ValueError, match="recording_id"):
        validate_cache(output)


def test_validate_cache_rejects_recording_from_wrong_bearing_domain(
    tmp_path, synthetic_raw_hust
):
    output = tmp_path / "cache"
    manifest = build_cache(synthetic_raw_hust, output, seed=2025)
    wrong_id = next(
        index
        for index, row in enumerate(manifest["source_files"])
        if row["path"] == "B600.mat"
    )
    _rewrite_domain(
        output,
        0,
        lambda obj: obj["recording_id"].__setitem__(0, wrong_id),
    )
    with pytest.raises(ValueError, match="domain"):
        validate_cache(output)


@pytest.mark.parametrize("offset", [1, 1024 * 99])
def test_validate_cache_rejects_unaligned_or_out_of_range_offset(
    tmp_path, synthetic_raw_hust, offset
):
    output = tmp_path / "cache"
    build_cache(synthetic_raw_hust, output, seed=2025)
    _rewrite_domain(
        output,
        0,
        lambda obj: obj["window_offset"].__setitem__(0, offset),
    )
    with pytest.raises(ValueError, match="window offset"):
        validate_cache(output)


def test_validate_cache_rejects_recording_manifest_inconsistency(
    tmp_path, synthetic_raw_hust
):
    output = tmp_path / "cache"
    build_cache(synthetic_raw_hust, output, seed=2025)
    manifest_path = output / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["recording_count"] -= 1
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="recording count"):
        validate_cache(output)


def test_validate_cache_rejects_source_metadata_inconsistent_with_filename(
    tmp_path, synthetic_raw_hust
):
    output = tmp_path / "cache"
    build_cache(synthetic_raw_hust, output, seed=2025)
    manifest_path = output / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["source_files"][0]["bearing"] = 9999
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="source metadata"):
        validate_cache(output)


def test_hust_strict_dataset_returns_stable_index_and_metadata(tmp_path):
    from Dataset.HUSTStrict import HUSTStrictTensorDataset

    obj = {
        "data": torch.arange(1024, dtype=torch.float32).reshape(2, 1, 512),
        "label": torch.tensor([1, 4]),
        "shaft_hz": torch.tensor([24.8, 23.1]),
        "load_w": torch.tensor([0, 400]),
        "recording_id": torch.tensor([7, 8]),
        "window_offset": torch.tensor([0, 1024]),
    }
    ds = HUSTStrictTensorDataset(obj)
    x, y, index = ds[1]
    assert (y, index) == (4, 1)
    assert x.shape == (1, 512)
    assert float(x.mean()) == pytest.approx(0.0, abs=1e-5)
    assert float(x.std(unbiased=False)) == pytest.approx(1.0, rel=1e-5)
    assert ds.shaft_hz[index].item() == pytest.approx(23.1)
    assert ds.load_w.dtype == torch.int64
    assert ds.recording_id.dtype == torch.int64
    assert ds.window_offset.dtype == torch.int64
    assert torch.equal(ds.load_w, torch.tensor([0, 400], dtype=torch.int64))
    assert torch.equal(ds.recording_id, torch.tensor([7, 8], dtype=torch.int64))
    assert torch.equal(ds.window_offset, torch.tensor([0, 1024], dtype=torch.int64))
    assert ds.load_w[index].item() == 400
    assert ds.recording_id[index].item() == 8
    assert ds.window_offset[index].item() == 1024


def test_hust_strict_loader_selects_domains_and_uses_safe_cpu_load(
    tmp_path, synthetic_raw_hust, monkeypatch
):
    import importlib

    from Dataset.HUSTStrict import HUSTStrict

    output = tmp_path / "cache"
    build_cache(synthetic_raw_hust, output, seed=2025)
    expected_source = torch.load(
        output / "domain_2.pt", map_location="cpu", weights_only=True
    )
    expected_target = torch.load(
        output / "domain_0.pt", map_location="cpu", weights_only=True
    )
    cfg = OmegaConf.create(
        {
            "Dataset": {
                "data_name": "HUSTStrict",
                "data_path": str(output),
                "TL_list": [0, 1, 2, 3],
                "TL_Task": [2, 0],
                "input_kind": "fft",
                "norm_kind": "mean-std",
            }
        }
    )

    module = importlib.import_module("Dataset.HUSTStrict")
    loader = HUSTStrict(**cfg.Dataset)
    real_load = torch.load
    load_calls = []

    def recording_load(path, *args, **kwargs):
        load_calls.append((Path(path).name, args, kwargs.copy()))
        return real_load(path, *args, **kwargs)

    monkeypatch.setattr(module.torch, "load", recording_load)
    source, target = loader.data_generator()

    assert len(source) == len(expected_source["label"]) == 63
    assert len(target) == len(expected_target["label"]) == 63
    assert torch.equal(source.data, expected_source["data"])
    assert torch.equal(target.data, expected_target["data"])
    for key in ("shaft_hz", "load_w", "recording_id", "window_offset"):
        assert torch.equal(getattr(source, key), expected_source[key])
        assert torch.equal(getattr(target, key), expected_target[key])
    source_x, source_y, source_index = source[5]
    target_x, target_y, target_index = target[7]
    assert (source_y, source_index) == (int(expected_source["label"][5]), 5)
    assert (target_y, target_index) == (int(expected_target["label"][7]), 7)
    assert source_x.shape == target_x.shape == (1, 512)
    assert source.recording_id[5] == expected_source["recording_id"][5]
    assert target.recording_id[7] == expected_target["recording_id"][7]
    assert load_calls == [
        ("domain_2.pt", (), {"map_location": "cpu", "weights_only": True}),
        ("domain_0.pt", (), {"map_location": "cpu", "weights_only": True}),
    ]


def test_hust_strict_loader_rejects_missing_manifest(tmp_path, synthetic_raw_hust):
    from Dataset.HUSTStrict import HUSTStrict

    output = tmp_path / "cache"
    build_cache(synthetic_raw_hust, output, seed=2025)
    (output / "manifest.json").unlink()

    with pytest.raises(ValueError, match="invalid HUST cache manifest"):
        HUSTStrict(data_path=output, TL_Task=[0, 1])
