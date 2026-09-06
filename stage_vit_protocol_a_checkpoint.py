#!/usr/bin/python
# -*- coding: UTF-8 -*-
"""Stage a just-trained ViT source checkpoint into a method-specific directory."""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path
from typing import Iterable

from Lib.vit_protocol_a import method_checkpoint_dir, source_checkpoint_candidates


def stage_from_candidates(candidates: Iterable[Path], destination: Path) -> Path:
    for candidate in [Path(p) for p in candidates]:
        if candidate.exists():
            destination = Path(destination)
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(candidate, destination)
            return candidate
    raise FileNotFoundError("No source checkpoint found among: " + ", ".join(map(str, candidates)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", required=True, choices=["DTCC_VIT", "0711_FULL_VIT"])
    parser.add_argument("--source", required=True, type=int)
    parser.add_argument("--target", required=True, type=int, help="Representative source-training target domain")
    parser.add_argument("--seed", default=2025, type=int)
    parser.add_argument("--root", default="TTA_Model_ViT_A")
    args = parser.parse_args()

    destination_dir = method_checkpoint_dir(
        args.method, "PU4D", args.source, args.seed, root=args.root
    )
    model_name = f"ViT1D{args.seed}fft_Linear.pt"
    best_destination = destination_dir / ("best_source_" + model_name)
    chosen = stage_from_candidates(
        source_checkpoint_candidates(args.source, args.target, args.seed),
        best_destination,
    )

    # Also expose a final-name alias so both target runners accept the staged model.
    final_destination = destination_dir / model_name
    shutil.copy2(best_destination, final_destination)
    print(f"[STAGE PASS] method={args.method} source={args.source} from={chosen} to={best_destination}")


if __name__ == "__main__":
    main()
