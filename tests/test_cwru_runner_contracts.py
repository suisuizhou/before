from pathlib import Path


def text(name):
    return Path(name).read_text(encoding="utf-8")


def test_dtcc_source_is_50_epochs_and_cwru():
    s = text("main_src_dtcc_cwru_r18.py")
    assert "dtcc_src_epoch = 50" in s
    assert 'cfg.Dataset.data_name = "CWRU"' in s


def test_0711_source_is_60_epochs():
    s = text("main_src_0711_full_cwru_r18.py")
    assert "cfg.src_epoch = 60" in s
    assert 'cfg.Dataset.data_name = "CWRU"' in s


def test_0711_target_keeps_adapter_and_fwarp_and_cwru_evidence():
    s = text("main_tta_0711_full_cwru_r18.py")
    assert "band_scale" in s and "band_bias" in s and "warp_ctrl" in s
    assert "cwru_physical_fault_evidence" in s
    assert "min_pcl_classes" in s and "min_ncl_classes" in s
