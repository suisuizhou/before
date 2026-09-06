from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def text(name):
    return (ROOT / name).read_text()


def test_source_protocol_contract():
    s = text('main_src_pu4d_vanilla_r18.py')
    assert 'src_epoch = 50' in s or 'src_epoch=50' in s
    assert 'label_smoothing=0.1' in s.replace(' ', '')
    assert 'drop_last=False' in s.replace(' ', '')
    assert "lr_scheduler = 'designed'" in s or 'lr_scheduler="designed"' in s
    low = s.lower()
    for forbidden in ('mixup', 'ssp_', 'use_sde', 'lambda_sde'):
        assert forbidden not in low


def test_dtcc_temperature_is_one():
    s = text('main_tta_dtcc_resnet18_vanilla.py')
    assert 'ncl_temperature", 1.0' in s or "ncl_temperature', 1.0" in s


def test_0711_uses_vanilla_root():
    s = text('main_tta_0711_resnet18_vanilla.py')
    assert 'TTA_Model_VANILLA' in s
    assert 'vanilla_checkpoint_dir' in s


def test_runner_has_12_tasks_and_common_source():
    s = text('run_pu4d_vanilla_dtcc_0711_gpu01.sh')
    for pair in ('0 1','0 2','0 3','1 0','1 2','1 3','2 0','2 1','2 3','3 0','3 1','3 2'):
        assert pair in s
    assert 'DtCC.ncl_temperature=1.0' in s
    assert 'TTA_Model_VANILLA' in s
    assert 'DRY_RUN' in s


def test_summarizer_enforces_before_equality():
    s = text('summarize_pu4d_vanilla_results.py')
    assert 'BEFORE_MISMATCH' in s
    assert 'before_tolerance' in s
