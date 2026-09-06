import torch


def load_vit_prompt_compatible_ckpt(model, ckpt_path, map_location="cpu"):
    state = torch.load(ckpt_path, map_location=map_location)
    current = model.state_dict()

    # ---------- pos_embed 兼容 ----------
    if "0.pos_embed" in state and "0.pos_embed" in current:
        old_pos = state["0.pos_embed"]
        new_pos = current["0.pos_embed"]

        if old_pos.shape != new_pos.shape:
            out = new_pos.clone()

            # CLS token
            out[:, 0:1, :] = old_pos[:, 0:1, :]

            # patch positional embedding
            old_patch = old_pos[:, 1:, :]
            new_prompt_len = new_pos.shape[1] - old_pos.shape[1]
            out[:, 1 + new_prompt_len:, :] = old_patch

            # 新增 prompt 位置保留当前初始化
            state["0.pos_embed"] = out

    # ---------- 普通 prompt -> split prompt ----------
    if (
        "0.prompt_embed" in state
        and "0.prompt_embed_inv" in current
        and "0.prompt_embed_inv" not in state
    ):
        old_prompt = state["0.prompt_embed"]
        if old_prompt.shape == current["0.prompt_embed_inv"].shape:
            state["0.prompt_embed_inv"] = old_prompt.clone()
            if "0.prompt_embed_pla" in current:
                state["0.prompt_embed_pla"] = torch.zeros_like(current["0.prompt_embed_pla"])

    # ---------- split prompt -> 普通 prompt ----------
    if (
        "0.prompt_embed" in current
        and "0.prompt_embed" not in state
        and "0.prompt_embed_inv" in state
    ):
        inv = state["0.prompt_embed_inv"]
        if "0.prompt_embed_pla" in state and state["0.prompt_embed_pla"].shape == inv.shape:
            pla = state["0.prompt_embed_pla"]
            merged = inv + pla
        else:
            merged = inv

        if merged.shape == current["0.prompt_embed"].shape:
            state["0.prompt_embed"] = merged.clone()

    # ---------- 删除形状不匹配键 ----------
    bad_keys = []
    for k in list(state.keys()):
        if k in current and state[k].shape != current[k].shape:
            bad_keys.append(k)

    for k in bad_keys:
        if k != "0.pos_embed":
            del state[k]

    msg = model.load_state_dict(state, strict=False)
    print("[load_vit_prompt_compatible_ckpt]")
    print("  ckpt:", ckpt_path)
    print("  missing/unexpected:", msg)
    return msg
