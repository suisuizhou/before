#!/usr/bin/python
# -*- coding: UTF-8 -*-
"""Audit staged Protocol-A ViT source checkpoints before target adaptation."""
from __future__ import annotations

import argparse
from pathlib import Path
import torch

from Lib.vit_protocol_a import method_checkpoint_dir


def _unwrap_state(obj):
    if isinstance(obj, dict) and "state_dict" in obj:
        obj = obj["state_dict"]
    if not isinstance(obj, dict):
        raise TypeError(f"Unsupported checkpoint object: {type(obj)}")
    return {(k[7:] if str(k).startswith("module.") else str(k)): v for k, v in obj.items()}


def inspect_state_dict(state):
    state = _unwrap_state(state)
    prompt_keys = [k for k in state if k.endswith("prompt_embed")]
    if len(prompt_keys) != 1:
        raise ValueError(f"Expected exactly one prompt_embed tensor, found {prompt_keys}")
    prompt_shape = tuple(state[prompt_keys[0]].shape)
    if prompt_shape != (1, 3, 256):
        raise ValueError(f"Unexpected prompt shape {prompt_shape}; expected (1, 3, 256)")

    pos_keys = [k for k in state if k.endswith("pos_embed")]
    pos_shape = tuple(state[pos_keys[0]].shape) if pos_keys else None
    if pos_shape is not None and (len(pos_shape) != 3 or pos_shape[-1] != 256):
        raise ValueError(f"Unexpected pos_embed shape {pos_shape}")
    return {"prompt_key": prompt_keys[0], "prompt_shape": prompt_shape, "pos_shape": pos_shape, "tensors": len(state)}


def audit_method(method: str, seed: int = 2025, root: str = "TTA_Model_ViT_A"):
    rows = []
    model_name = f"ViT1D{seed}fft_Linear.pt"
    for source in range(4):
        directory = method_checkpoint_dir(method, "PU4D", source, seed, root=root)
        path = directory / ("best_source_" + model_name)
        if not path.exists():
            raise FileNotFoundError(path)
        state = torch.load(path, map_location="cpu")
        info = inspect_state_dict(state)
        rows.append((source, path, info))
        print(
            f"[AUDIT PASS] method={method} source={source} prompt={info['prompt_shape']} "
            f"pos={info['pos_shape']} tensors={info['tensors']} path={path}"
        )
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", required=True, choices=["DTCC_VIT", "0711_FULL_VIT"])
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument("--root", default="TTA_Model_ViT_A")
    args = parser.parse_args()
    audit_method(args.method, args.seed, args.root)


if __name__ == "__main__":
    main()
