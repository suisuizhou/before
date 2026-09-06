import pytest
import numpy as np
import torch

from Lib.hust_physical_fault_evidence import (
    HUST_GEOMETRIES,
    HUSTPhysicalEvidenceConfig,
    build_hust_physical_masks,
    characteristic_frequencies,
)


@pytest.mark.parametrize(
    ("domain", "shaft", "rolling_diameter", "pitch_diameter"),
    [
        (0, 25.0, 7.8, 38.5),
        (1, 24.0, 9.0, 46.0),
        (2, 23.0, 11.0, 53.5),
        (3, 22.0, 12.0, 60.0),
    ],
)
def test_characteristic_frequencies_use_each_hust_bearing_geometry(
    domain, shaft, rolling_diameter, pitch_diameter
):
    values = characteristic_frequencies(
        torch.tensor([shaft], dtype=torch.float64), HUST_GEOMETRIES[domain]
    )
    ratio = rolling_diameter / pitch_diameter

    assert values["shaft"].item() == pytest.approx(shaft)
    assert values["bpfo"].item() == pytest.approx(0.5 * 9 * shaft * (1 - ratio))
    assert values["bpfi"].item() == pytest.approx(0.5 * 9 * shaft * (1 + ratio))
    assert values["bsf"].item() == pytest.approx(
        (pitch_diameter / (2 * rolling_diameter)) * shaft * (1 - ratio**2)
    )
    assert values["ftf"].item() == pytest.approx(0.5 * shaft * (1 - ratio))


def test_single_fault_masks_follow_predicted_class_and_per_sample_shaft_speed():
    cfg = HUSTPhysicalEvidenceConfig(
        sampling_rate_hz=51200,
        fft_size=2048,
        spectrum_length=512,
        harmonics=1,
        inner_sideband_orders=(0,),
        outer_sideband_orders=(0,),
        ball_sideband_orders=(0,),
        mask_sigma_bins=0.2,
    )
    labels = torch.tensor([1, 2, 3, 1])  # I, O, B, I
    shaft = torch.tensor([20.0, 20.0, 20.0, 30.0])

    masks, applicable = build_hust_physical_masks(labels, 0, shaft, cfg)
    frequencies = characteristic_frequencies(shaft, HUST_GEOMETRIES[0])

    assert applicable.tolist() == [True, True, True, True]
    for row, component in enumerate(("bpfi", "bpfo", "bsf", "bpfi")):
        expected_bin = round(frequencies[component][row].item() / 25.0)
        assert masks[row].argmax().item() == expected_bin
    assert not torch.equal(masks[0], masks[3])


def test_normal_is_not_applicable_and_composites_union_components():
    cfg = HUSTPhysicalEvidenceConfig(
        sampling_rate_hz=51200, fft_size=2048, spectrum_length=512
    )
    shaft = torch.tensor([24.8, 24.8, 24.8, 24.8])
    labels = torch.tensor([0, 1, 3, 4])  # N, I, B, IB

    masks, applicable = build_hust_physical_masks(labels, 0, shaft, cfg)

    assert applicable.tolist() == [False, True, True, True]
    assert torch.count_nonzero(masks[0]) == 0
    expected_union = torch.maximum(masks[1], masks[2])
    assert torch.equal(masks[3] > 0, expected_union > 0)
    assert (masks[3] >= cfg.mask_activity_threshold).sum() <= int(
        cfg.max_mask_ratio * 512
    )


@pytest.mark.parametrize(
    ("composite", "components"), [(4, (1, 3)), (5, (1, 2)), (6, (2, 3))]
)
def test_each_composite_is_the_capped_union_of_its_components(composite, components):
    cfg = HUSTPhysicalEvidenceConfig(
        sampling_rate_hz=51200,
        fft_size=2048,
        spectrum_length=512,
        harmonics=2,
        max_mask_ratio=0.5,
    )
    labels = torch.tensor([components[0], components[1], composite])
    shaft = torch.full((3,), 25.0)

    masks, _ = build_hust_physical_masks(labels, 2, shaft, cfg)

    assert torch.equal(masks[2], torch.maximum(masks[0], masks[1]))


@pytest.mark.parametrize("bad_domain", [-1, 4])
def test_masks_reject_invalid_target_domains(bad_domain):
    with pytest.raises(ValueError, match="target domain"):
        build_hust_physical_masks(
            torch.tensor([1]), bad_domain, torch.tensor([25.0]), HUSTPhysicalEvidenceConfig()
        )


@pytest.mark.parametrize(
    "bad_domain", [0.5, 3.9, True, False, torch.tensor([1]), torch.tensor(1.0)]
)
def test_masks_reject_non_integral_or_non_scalar_target_domains(bad_domain):
    with pytest.raises(ValueError, match="target_domain must be an integer scalar"):
        build_hust_physical_masks(
            torch.tensor([1]), bad_domain, torch.tensor([25.0]), HUSTPhysicalEvidenceConfig()
        )


@pytest.mark.parametrize("valid_domain", [0, np.int64(2), torch.tensor(3)])
def test_masks_accept_exact_integer_like_target_domains(valid_domain):
    masks, applicable = build_hust_physical_masks(
        torch.tensor([1]), valid_domain, torch.tensor([25.0]), HUSTPhysicalEvidenceConfig()
    )

    assert masks.shape == (1, 512)
    assert applicable.tolist() == [True]


@pytest.mark.parametrize("bad_label", [-1, 7])
def test_masks_reject_invalid_predicted_labels(bad_label):
    with pytest.raises(ValueError, match="label"):
        build_hust_physical_masks(
            torch.tensor([bad_label]), 0, torch.tensor([25.0]), HUSTPhysicalEvidenceConfig()
        )


@pytest.mark.parametrize(
    ("labels", "shaft", "message"),
    [
        (torch.tensor([[1]]), torch.tensor([25.0]), "pseudo_labels must be 1-D"),
        (torch.tensor([1]), torch.tensor([[25.0]]), "shaft_hz must be 1-D"),
        (torch.tensor([1, 2]), torch.tensor([25.0]), "same length"),
    ],
)
def test_masks_reject_shape_mismatches(labels, shaft, message):
    with pytest.raises(ValueError, match=message):
        build_hust_physical_masks(labels, 0, shaft, HUSTPhysicalEvidenceConfig())


@pytest.mark.parametrize("bad_shaft", [0.0, -1.0, float("nan"), float("inf")])
def test_masks_reject_non_positive_or_non_finite_shaft_values(bad_shaft):
    with pytest.raises(ValueError, match="shaft_hz"):
        build_hust_physical_masks(
            torch.tensor([1]), 0, torch.tensor([bad_shaft]), HUSTPhysicalEvidenceConfig()
        )


@pytest.mark.parametrize(
    ("field", "orders"),
    [
        ("inner_sideband_orders", ()),
        ("outer_sideband_orders", (-1,)),
        ("ball_sideband_orders", (0.5,)),
        ("inner_sideband_orders", (float("nan"),)),
        ("outer_sideband_orders", (float("inf"),)),
    ],
)
def test_masks_reject_invalid_public_sideband_orders(field, orders):
    cfg = HUSTPhysicalEvidenceConfig(**{field: orders})

    with pytest.raises(ValueError, match=field):
        build_hust_physical_masks(torch.tensor([1]), 0, torch.tensor([25.0]), cfg)


@pytest.mark.parametrize(
    ("label", "frequency_name", "orders_field"),
    [
        (1, "bpfi", "inner_sideband_orders"),
        (2, "bpfo", "outer_sideband_orders"),
    ],
)
def test_inner_and_outer_masks_include_hand_derived_shaft_sideband_bins(
    label, frequency_name, orders_field
):
    cfg_kwargs = {
        "sampling_rate_hz": 51200,
        "fft_size": 2048,
        "spectrum_length": 512,
        "harmonics": 1,
        "inner_sideband_orders": (0,),
        "outer_sideband_orders": (0,),
        "ball_sideband_orders": (0,),
        "mask_sigma_bins": 0.25,
    }
    cfg_kwargs[orders_field] = (1,)
    cfg = HUSTPhysicalEvidenceConfig(**cfg_kwargs)
    shaft_hz = 25.0
    ratio = 7.8 / 38.5
    base_hz = {
        "bpfi": 0.5 * 9 * shaft_hz * (1 + ratio),
        "bpfo": 0.5 * 9 * shaft_hz * (1 - ratio),
    }[frequency_name]
    expected_bins = {
        round((base_hz - shaft_hz) / 25.0),
        round((base_hz + shaft_hz) / 25.0),
    }

    masks, _ = build_hust_physical_masks(
        torch.tensor([label]), 0, torch.tensor([shaft_hz]), cfg
    )
    active_bins = set(
        torch.nonzero(masks[0] >= cfg.mask_activity_threshold).flatten().tolist()
    )

    assert active_bins == expected_bins


def test_ball_mask_includes_hand_derived_ftf_sideband_bins():
    cfg = HUSTPhysicalEvidenceConfig(
        sampling_rate_hz=51200,
        fft_size=2048,
        spectrum_length=512,
        harmonics=1,
        inner_sideband_orders=(0,),
        outer_sideband_orders=(0,),
        ball_sideband_orders=(1,),
        mask_sigma_bins=0.25,
    )
    shaft_hz = 25.0
    ratio = 7.8 / 38.5
    bsf_hz = (38.5 / (2 * 7.8)) * shaft_hz * (1 - ratio**2)
    ftf_hz = 0.5 * shaft_hz * (1 - ratio)
    expected_bins = {
        round((bsf_hz - ftf_hz) / 25.0),
        round((bsf_hz + ftf_hz) / 25.0),
    }

    masks, _ = build_hust_physical_masks(
        torch.tensor([3]), 0, torch.tensor([shaft_hz]), cfg
    )
    active_bins = set(
        torch.nonzero(masks[0] >= cfg.mask_activity_threshold).flatten().tolist()
    )

    assert active_bins == expected_bins


def test_composite_cap_is_applied_after_union_not_to_each_component_first():
    cfg = HUSTPhysicalEvidenceConfig(
        sampling_rate_hz=51200,
        fft_size=2048,
        spectrum_length=64,
        harmonics=1,
        inner_sideband_orders=(0,),
        outer_sideband_orders=(0,),
        ball_sideband_orders=(0,),
        mask_sigma_bins=0.3,
        max_mask_ratio=1 / 64,
    )
    ratio = 7.8 / 38.5
    bpfo_order = 0.5 * 9 * (1 - ratio)
    shaft_hz = 25.0 / bpfo_order  # Places BPFO exactly on bin 1.
    bpfi_hz = 0.5 * 9 * shaft_hz * (1 + ratio)
    expected_inner_bin = round(bpfi_hz / 25.0)
    expected_outer_bin = 1

    masks, _ = build_hust_physical_masks(
        torch.tensor([1, 2, 5]), 0, torch.full((3,), shaft_hz), cfg
    )
    active = [
        torch.nonzero(row >= cfg.mask_activity_threshold).flatten().tolist()
        for row in masks
    ]

    assert active[0] == [expected_inner_bin]
    assert active[1] == [expected_outer_bin]
    assert active[2] == [expected_outer_bin]


def test_empty_batch_returns_empty_float32_masks_and_bool_applicability():
    masks, applicable = build_hust_physical_masks(
        torch.empty(0, dtype=torch.long),
        0,
        torch.empty(0, dtype=torch.float64),
        HUSTPhysicalEvidenceConfig(),
    )

    assert masks.shape == (0, 512)
    assert masks.dtype == torch.float32
    assert masks.device.type == "cpu"
    assert applicable.shape == (0,)
    assert applicable.dtype == torch.bool


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_cross_device_shaft_metadata_returns_masks_on_prediction_device():
    labels = torch.tensor([1, 3], device="cuda")
    shaft_hz = torch.tensor([25.0, 24.0], device="cpu", dtype=torch.float64)

    masks, applicable = build_hust_physical_masks(
        labels, 0, shaft_hz, HUSTPhysicalEvidenceConfig()
    )

    assert masks.device == labels.device
    assert masks.dtype == torch.float32
    assert applicable.device == labels.device
    assert applicable.dtype == torch.bool
