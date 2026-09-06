#!/usr/bin/python
# -*- coding: UTF-8 -*-
# author：Mengliang Zhu

from Dataset.PU import PU
from Dataset.CWRU import CWRU
from .HUSTStrict import HUSTStrict
from .WTPGStrict import WTPGStrict
from .HUSTGearboxStrict import HUSTGearboxStrict
from .MCC5THUStrict import MCC5THUStrict
from Dataset.PU4D import PU4D
from .HUST_Bearing import HUST

# Cached FFT domain datasets
try:
    from .cache_tta_dataset import HUSTBAL, UO
except Exception as e:
    print("[WARN] failed to import cached TTA datasets:", e)

try:
    from .cache_tta_dataset import UOSPEED
except Exception as e:
    print("[WARN] failed to import UOSPEED:", e)
from Dataset.AEB4D import AEB4D
try:
    from Dataset.WTPG8D import WTPG8D
except ModuleNotFoundError as exc:
    if exc.name != "Dataset.WTPG8D":
        raise
    print("[WARN] optional WTPG8D dataset is unavailable:", exc)
