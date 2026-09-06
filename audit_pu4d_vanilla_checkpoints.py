#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import torch

from Lib.pu4d_vanilla_protocol import resolve_vanilla_checkpoint


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', default='TTA_Model_VANILLA')
    ap.add_argument('--seed', type=int, default=2025)
    ap.add_argument('--model-name', default='ResNet18_1D_SDE2025fft_Linear.pt')
    args = ap.parse_args()
    for source in range(4):
        p = resolve_vanilla_checkpoint(args.root, source, args.seed, args.model_name)
        state = torch.load(p, map_location='cpu')
        if isinstance(state, dict) and 'state_dict' in state:
            state = state['state_dict']
        if not isinstance(state, dict):
            raise TypeError(f'Unsupported checkpoint object for source {source}: {type(state)}')
        bad = []
        for name, tensor in state.items():
            if any(token in name for token in ('band_scale','band_bias','warp_ctrl')):
                if torch.is_tensor(tensor) and float(tensor.abs().max()) > 1e-8:
                    bad.append((name, float(tensor.abs().max())))
        if bad:
            raise RuntimeError(f'Source {source} target-only adaptation carrier is not identity: {bad[:5]}')
        print(f'[AUDIT PASS] source={source} tensors={len(state)} sha256={sha256(p)} checkpoint={p}')
    print('[AUDIT COMPLETE] four vanilla source checkpoints are present and target-only carriers are identity')


if __name__ == '__main__':
    main()
