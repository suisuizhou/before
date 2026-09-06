import torch
from torch.utils.data import TensorDataset

from Lib.fixed_random_stream import make_fixed_random_stream_loader


def _stream_order(seed: int):
    dataset = TensorDataset(torch.arange(24))
    loader = make_fixed_random_stream_loader(
        dataset=dataset,
        batch_size=5,
        seed=seed,
        num_workers=0,
        pin_memory=False,
    )
    order = []
    for (values,) in loader:
        order.extend(values.tolist())
    return order


def test_fixed_random_stream_is_reproducible_for_same_seed():
    assert _stream_order(2025) == _stream_order(2025)


def test_fixed_random_stream_changes_with_seed():
    assert _stream_order(2025) != _stream_order(2026)


def test_fixed_random_stream_visits_every_sample_exactly_once():
    order = _stream_order(2025)
    assert len(order) == 24
    assert sorted(order) == list(range(24))
    assert len(set(order)) == 24
