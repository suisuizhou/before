from pathlib import Path

import torch
from torch.utils.data import Dataset

from Lib.hust_strict_protocol import validate_cache


class HUSTStrictTensorDataset(Dataset):
    def __init__(self, obj):
        self.data = obj["data"].float()
        self.label = obj["label"].long()
        self.shaft_hz = obj["shaft_hz"].float()
        self.load_w = obj["load_w"].long()
        self.recording_id = obj["recording_id"].long()
        self.window_offset = obj["window_offset"].long()

    def __len__(self):
        return int(self.label.numel())

    def __getitem__(self, index):
        x = self.data[index]
        mean = x.mean()
        std = x.std(unbiased=False).clamp_min(1e-8)
        return (x - mean) / std, int(self.label[index]), int(index)


class HUSTStrict:
    inputchannel = 1
    num_classes = 7

    def __init__(self, data_path, TL_Task=(0, 1), **_kwargs):
        self.root = Path(data_path)
        self.source, self.target = (int(TL_Task[0]), int(TL_Task[1]))
        validate_cache(self.root)

    def _load(self, domain):
        obj = torch.load(
            self.root / f"domain_{domain}.pt", map_location="cpu", weights_only=True
        )
        return HUSTStrictTensorDataset(obj)

    def data_generator(self):
        return self._load(self.source), self._load(self.target)
