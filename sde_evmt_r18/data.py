from pathlib import Path

import torch
from torch.utils.data import TensorDataset


PU4D_DOMAINS = (
    "D1_1500_0.7_1000",
    "D2_900_0.7_1000",
    "D3_1500_0.1_1000",
    "D4_1500_0.7_400",
)


def load_pu4d_domain(cache_root, domain, input_kind="fft"):
    domain = int(domain)
    if domain not in range(len(PU4D_DOMAINS)):
        raise ValueError(f"domain must be one of 0,1,2,3; got {domain}")
    path = Path(cache_root) / f"{PU4D_DOMAINS[domain]}_{input_kind}.pt"
    if not path.is_file():
        raise FileNotFoundError(path)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or not {"x", "y"} <= payload.keys():
        raise ValueError(f"invalid PU4D cache payload: {path}")
    x = torch.as_tensor(payload["x"], dtype=torch.float32).contiguous()
    y = torch.as_tensor(payload["y"], dtype=torch.long).flatten().contiguous()
    if x.ndim != 2 or x.shape[1] != 512 or len(x) != len(y):
        raise ValueError(f"expected x=[N,512], y=[N], got {x.shape}, {y.shape}")
    if not bool(torch.isfinite(x).all()) or bool((x < 0).any()):
        raise ValueError("PU4D FFT cache must contain finite non-negative magnitudes")
    if len(y) and (int(y.min()) < 0 or int(y.max()) >= 32):
        raise ValueError("PU4D labels must be in [0,31]")
    return TensorDataset(x, y)
