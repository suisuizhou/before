from pathlib import Path
import torch
import torch.nn as nn

from Lib.pu4d_vanilla_protocol import (
    vanilla_checkpoint_dir,
    choose_dummy_target,
    reset_and_freeze_adaptation_carrier,
)


def test_checkpoint_dir_is_method_independent():
    p = vanilla_checkpoint_dir(Path('TTA_Model_VANILLA'), 2, 2025)
    assert p == Path('TTA_Model_VANILLA/PU4D/source_2/seed_2025')


def test_choose_dummy_target_never_equals_source():
    for s in range(4):
        t = choose_dummy_target(s, [0, 1, 2, 3])
        assert t != s
        assert t in [0, 1, 2, 3]


class Dummy(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(2))
        self.band_scale = nn.Parameter(torch.ones(2))
        self.band_bias = nn.Parameter(torch.ones(2))
        self.warp_ctrl = nn.Parameter(torch.ones(2))


def test_reset_and_freeze_adaptation_carrier():
    model = Dummy()
    frozen = reset_and_freeze_adaptation_carrier(model)
    assert set(frozen) == {'band_scale', 'band_bias', 'warp_ctrl'}
    assert torch.equal(model.band_scale, torch.zeros_like(model.band_scale))
    assert torch.equal(model.band_bias, torch.zeros_like(model.band_bias))
    assert torch.equal(model.warp_ctrl, torch.zeros_like(model.warp_ctrl))
    assert not model.band_scale.requires_grad
    assert not model.band_bias.requires_grad
    assert not model.warp_ctrl.requires_grad
    assert model.weight.requires_grad
