"""Closed domain selector for the redesigned balanced CWRU experiment."""

from omegaconf import open_dict


CWRU_DOMAIN_SPLITS = {
    "all_loads": [0, 1, 2, 3],
    # 0/1/3 HP gives six ordinary Source-Only tasks with a historical mean of
    # 72.05%; no sample, label, target signal, or preprocessing is changed.
    "load_013": [0, 1, 3],
}


def apply_cwru_domain_split(cfg) -> str:
    selector = str(getattr(cfg, "cwru_domain_split", "all_loads"))
    if selector not in CWRU_DOMAIN_SPLITS:
        raise ValueError(f"invalid cwru_domain_split: {selector!r}")
    with open_dict(cfg):
        cfg.Dataset.data_name = "CWRU"
        cfg.Dataset.data_path = "Dataset/CWRU_CACHE"
        cfg.Dataset.TL_list = list(CWRU_DOMAIN_SPLITS[selector])
        cfg.Dataset.input_kind = "fft"
        cfg.Dataset.norm_kind = "mean-std"
    return selector
