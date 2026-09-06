from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def read(name):
    return (ROOT / name).read_text(encoding="utf-8")


def test_source_epochs_are_60():
    text = read("run_vit_protocol_a_pu4d_gpu01.sh")
    assert "src_epoch=60" in text


def test_dtcc_target_is_prompt_only_and_two_steps():
    text = read("main_tta_dtcc_vit_prompt.py")
    assert 'select_trainable_parameters(model, method="dtcc")' in text
    assert "optim_steps = 2" in text


def test_0711_full_forces_prompt_adapter_warp():
    text = read("main_tta_0711_full_vit_prompt.py")
    assert "update_prompt = True" in text
    assert "update_bn_affine = False" in text
    assert "use_frequency_warp = True" in text
    assert "use_spectral_adapter = True" in text
    assert 'mode = "full"' in text


def test_target_runner_uses_fixed_random_stream_seed():
    text1 = read("main_tta_dtcc_vit_prompt.py")
    text2 = read("main_tta_0711_full_vit_prompt.py")
    assert "stream_seed" in text1 and "2025" in text1
    assert "stream_seed" in text2 and "2025" in text2


def test_dtcc_source_wrapper_is_label_smoothing_only():
    text = read("main_src_dtcc_vit_prompt.py")
    assert "src_epoch = 60" in text
    assert "prompt_len_src = 3" in text
    assert "label_smoothing = 0.1" in text
    assert "mixup_prob = 0.0" in text
    assert "use_ssp_lite = False" in text


def test_0711_source_wrapper_preserves_sde_source_recipe():
    text = read("main_src_0711_full_vit_prompt.py")
    assert "src_epoch = 60" in text
    assert "prompt_len_src = 3" in text
    assert "use_ssp_lite = True" in text
    assert "use_sde_lite = True" in text
    assert "mixup_prob = 0.5" in text


def test_runner_has_dry_run_and_both_methods():
    text = read("run_vit_protocol_a_pu4d_gpu01.sh")
    assert "DRY_RUN" in text
    assert "main_src_dtcc_vit_prompt.py" in text
    assert "main_src_0711_full_vit_prompt.py" in text
    assert "main_tta_dtcc_vit_prompt.py" in text
    assert "main_tta_0711_full_vit_prompt.py" in text
    assert "summarize_vit_protocol_a.py" in text


def test_summarizer_parses_standard_result_line():
    import sys
    sys.path.insert(0, str(ROOT))
    from summarize_vit_protocol_a import parse_result_line
    row = parse_result_line(
        "[RESULT] method=DTCC_VIT task=[2,1] before=18.1 online=43.2 post=49.9 "
        "online_f1=41.0 post_f1=47.5 batch_ms=123.4 peak_mb=210.0"
    )
    assert row["method"] == "DTCC_VIT"
    assert row["task"] == "[2,1]"
    assert row["online"] == 43.2


def test_checkpoint_stager_copies_first_existing_candidate(tmp_path):
    import sys
    sys.path.insert(0, str(ROOT))
    from stage_vit_protocol_a_checkpoint import stage_from_candidates
    missing = tmp_path / "missing.pt"
    src = tmp_path / "source.pt"
    src.write_bytes(b"checkpoint")
    dst = tmp_path / "out" / "best_source_ViT1D2025fft_Linear.pt"
    chosen = stage_from_candidates([missing, src], dst)
    assert chosen == src
    assert dst.read_bytes() == b"checkpoint"


def test_runner_dry_run_succeeds_in_fresh_project_dir(tmp_path):
    import os, subprocess
    env = os.environ.copy()
    env.update({"DRY_RUN": "1", "PROJECT_ROOT": str(tmp_path), "RUN_DIR": str(tmp_path / "run")})
    result = subprocess.run(
        ["bash", str(ROOT / "run_vit_protocol_a_pu4d_gpu01.sh")],
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    assert result.returncode == 0, result.stdout
    assert (tmp_path / "logs" / "latest_vit_protocol_a_run_dir.txt").exists()


def test_0711_vit_wraps_actual_strict_trainer_and_ema_includes_prompt():
    text = read("main_tta_0711_full_vit_prompt.py")
    assert "class ViT0711FullTrainer(base0711.Strict0711ResNetTrainer)" in text
    assert "base0711.Strict0711ResNetTrainer = ViT0711FullTrainer" in text
    assert '"prompt_embed", "band_scale", "band_bias", "warp_ctrl"' in text
    assert 'select_trainable_parameters(self.student, method="0711_full")' in text


def test_summarizer_fills_0711_efficiency_from_strict_diagnostics(tmp_path):
    import sys
    sys.path.insert(0, str(ROOT))
    from summarize_vit_protocol_a import collect
    log = tmp_path / "x.log"
    log.write_text(
        "[STRICT DIAGNOSTICS] mean_batch_ms=88.50 | peak_memory_mb=333.25 | pcl_active_batches=1\n"
        "[RESULT] method=0711_FULL_VIT task=[0,1] before=20 online=40 post=45 "
        "online_f1=NA post_f1=NA batch_ms=NA peak_mb=NA\n",
        encoding="utf-8",
    )
    rows = collect(tmp_path)
    assert rows[0]["batch_ms"] == 88.5
    assert rows[0]["peak_mb"] == 333.25


def test_checkpoint_audit_requires_three_prompt_tokens():
    import sys, torch
    sys.path.insert(0, str(ROOT))
    from audit_vit_protocol_a_checkpoints import inspect_state_dict
    ok = {"0.prompt_embed": torch.zeros(1, 3, 256), "0.pos_embed": torch.zeros(1, 36, 256)}
    info = inspect_state_dict(ok)
    assert info["prompt_shape"] == (1, 3, 256)
    bad = {"0.prompt_embed": torch.zeros(1, 5, 256)}
    import pytest
    with pytest.raises(ValueError):
        inspect_state_dict(bad)
