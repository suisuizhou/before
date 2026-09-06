#!/usr/bin/python
# -*- coding: UTF-8 -*-

from pathlib import Path
import numpy as np
import pandas as pd
import torch
from scipy.io import loadmat
from tqdm import tqdm

from Dataset.SequenceDatasets import dataset
from Dataset.sequence_aug import *

signal_size = 1024

DOMAINS = [
    "D1_1500_0.7_1000",  # 0
    "D2_900_0.7_1000",   # 1
    "D3_1500_0.1_1000",  # 2
    "D4_1500_0.7_400",   # 3
]

BEARINGS = [
    "K001", "K002", "K003", "K004", "K005", "K006",
    "KA01", "KA03", "KA04", "KA05", "KA06", "KA07", "KA08", "KA09", "KA15", "KA16", "KA22", "KA30",
    "KB23", "KB24", "KB27",
    "KI01", "KI03", "KI04", "KI05", "KI07", "KI08", "KI14", "KI16", "KI17", "KI18", "KI21",
]
LABEL_MAP = {b: i for i, b in enumerate(BEARINGS)}

BAD_FILES = {
    "N15_M01_F10_KA08_2.mat",
}


def parse_tl_task(tl_task):
    src, tar = tl_task[0], tl_task[1]
    source_domains = [src] if isinstance(src, int) else list(src)
    target_domains = [tar] if isinstance(tar, int) else list(tar)
    return source_domains, target_domains


def is_cache_root(root):
    root = Path(root)
    return all((root / f"{d}_fft.pt").exists() for d in DOMAINS)


def _load_raw_signal_original_style(mat_path):
    """
    稳定优先：
    直接完整 loadmat 一次，仍保留原始 PU 的固定索引 [0][0][2][0][6][2]
    """
    mat_path = str(mat_path)
    stem = Path(mat_path).stem

    mat = loadmat(mat_path)
    keys = [k for k in mat.keys() if not k.startswith("__")]
    if len(keys) == 0:
        raise RuntimeError(f"No user variables found in {mat_path}")

    key = stem if stem in keys else keys[0]
    obj = mat[key]

    fl = obj[0][0][2][0][6][2]
    fl = np.asarray(fl).reshape(-1)
    if fl.size < signal_size:
        raise RuntimeError(f"Signal too short: {mat_path}, len={fl.size}")
    return fl


def load_one_file(mat_path, input_kind="fft"):
    fl = _load_raw_signal_original_style(mat_path)

    data = []
    start, end = 0, signal_size
    while end <= fl.shape[0]:
        x = fl[start:end]
        if input_kind == 'fft':
            x = np.fft.fft(x)
            x = np.abs(x) / len(x)
            x = x[: x.shape[0] // 2]
        x = x.reshape(-1, 1)
        data.append(x)
        start += signal_size
        end += signal_size

    return data


def collect_domain_files(root, domain_idx, input_kind="fft"):
    domain_name = DOMAINS[domain_idx]
    domain_dir = Path(root) / domain_name
    if not domain_dir.exists():
        raise FileNotFoundError(f"Domain dir not found: {domain_dir}")

    data, lab = [], []
    bad_files = []

    for bearing in tqdm(BEARINGS, desc=f"loading {domain_name}"):
        bearing_dir = domain_dir / bearing
        if not bearing_dir.exists():
            continue

        mat_files = sorted(bearing_dir.glob("*.mat"))
        for mat_path in mat_files:
            if mat_path.name in BAD_FILES:
                print(f"[WARN] skip known bad file: {mat_path}")
                continue

            try:
                xs = load_one_file(str(mat_path), input_kind=input_kind)
            except Exception as e:
                bad_files.append((str(mat_path), str(e)))
                print(f"[WARN] skip bad file: {mat_path}")
                continue

            data.extend(xs)
            lab.extend([LABEL_MAP[bearing]] * len(xs))

    if bad_files:
        print(f"[WARN] {domain_name}: skipped {len(bad_files)} bad files")
        for fp, msg in bad_files[:10]:
            print(f"  - {fp}")
            print(f"    reason: {msg}")

    return data, lab


def collect_domain_cache_files(root, domain_idx, input_kind="fft"):
    domain_name = DOMAINS[domain_idx]
    cache_path = Path(root) / f"{domain_name}_{input_kind}.pt"
    if not cache_path.exists():
        raise FileNotFoundError(f"Cache file not found: {cache_path}")

    obj = torch.load(cache_path, map_location="cpu")
    x = obj["x"]          # [N, 512]
    y = obj["y"]          # [N]

    if isinstance(x, torch.Tensor):
        x = x.numpy()
    if isinstance(y, torch.Tensor):
        y = y.numpy()

    data = [arr.astype(np.float32).reshape(-1, 1) for arr in x]
    lab = [int(v) for v in y.tolist()]

    print(f"[CACHE] loaded {domain_name} from {cache_path}, samples={len(data)}")
    return data, lab


def get_files(root, domain_indices, input_kind='fft'):
    data, lab = [], []
    root = Path(root)

    use_cache = is_cache_root(root)
    if use_cache:
        print(f"[INFO] PU4D using CACHE root: {root}")
    else:
        print(f"[INFO] PU4D using RAW MAT root: {root}")

    for d in domain_indices:
        if use_cache:
            d_data, d_lab = collect_domain_cache_files(root, d, input_kind=input_kind)
        else:
            d_data, d_lab = collect_domain_files(root, d, input_kind=input_kind)
        data += d_data
        lab += d_lab
    return [data, lab]


class PU4D(object):
    inputchannel = 1
    num_classes = len(BEARINGS)

    def __init__(self, TL_list, data_path, TL_Task, seed_run=None, TL_kind=None,
                 data_name='PU4D', norm_kind="mean-std", input_kind='fft', **kwargs):
        self.TL_kind = TL_kind
        self.TL_list = TL_list
        self.data_name = data_name
        self.data_path = data_path
        self.source_domains, self.target_domains = parse_tl_task(TL_Task)
        self.norm_kind = norm_kind
        self.input_kind = input_kind

        self.data_transforms = {
            'train': Compose([
                Reshape(),
                Normalize(self.norm_kind),
                Retype(),
            ]),
        }

    def data_generator(self):
        src_list = get_files(self.data_path, self.source_domains, self.input_kind)
        src_pd = pd.DataFrame({"data": src_list[0], "label": src_list[1]})
        source_data = dataset(list_data=src_pd, transform=self.data_transforms['train'])

        tar_list = get_files(self.data_path, self.target_domains, self.input_kind)
        tar_pd = pd.DataFrame({"data": tar_list[0], "label": tar_list[1]})
        target_data = dataset(list_data=tar_pd, transform=self.data_transforms['train'])

        return source_data, target_data