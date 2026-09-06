import torch
import torch.nn as nn
import torch.nn.functional as F


class PatchEmbed1D(nn.Module):
    def __init__(self, input_len=512, patch_size=16, in_chans=1, embed_dim=256):
        super().__init__()
        assert input_len % patch_size == 0, "input_len must be divisible by patch_size"
        self.input_len = input_len
        self.patch_size = patch_size
        self.num_patches = input_len // patch_size
        self.proj = nn.Conv1d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        # accept [B, L], [B, 1, L], [B, L, 1]
        if x.dim() == 2:
            x = x.unsqueeze(1)
        elif x.dim() == 3 and x.shape[-1] == 1:
            x = x.transpose(1, 2)

        if x.dim() != 3:
            raise ValueError(f"Unexpected input shape: {x.shape}")

        if x.shape[-1] != self.input_len:
            x = F.interpolate(x, size=self.input_len, mode="linear", align_corners=False)

        x = self.proj(x)          # [B, C, N]
        x = x.transpose(1, 2)     # [B, N, C]
        return x



class ViT1D(nn.Module):
    def __init__(self, cfg=None):
        super().__init__()

        mcfg = cfg.Model if hasattr(cfg, "Model") else cfg

        self.input_len = int(getattr(mcfg, "input_len", 512))
        self.patch_size = int(getattr(mcfg, "patch_size", 16))
        self.in_chans = int(getattr(mcfg, "in_chans", 1))
        self.embed_dim = int(getattr(mcfg, "embed_dim", 256))
        self.depth = int(getattr(mcfg, "depth", 4))
        self.num_heads = int(getattr(mcfg, "num_heads", 4))
        self.mlp_ratio = float(getattr(mcfg, "mlp_ratio", 4.0))
        self.drop_rate = float(getattr(mcfg, "drop_rate", 0.1))
        self.prompt_len = int(getattr(mcfg, "prompt_len", 0))
        self.use_split_prompt = bool(getattr(mcfg, "use_split_prompt", False))
        self.use_spectral_adapter = bool(getattr(mcfg, "use_spectral_adapter", False))
        self.band_num = int(getattr(mcfg, "band_num", 16))
        assert self.input_len % self.band_num == 0, "input_len must be divisible by band_num"
        self.band_width = self.input_len // self.band_num
        self.adapter_delta = float(getattr(mcfg, "adapter_delta", 0.1))

        self.patch_embed = PatchEmbed1D(
            input_len=self.input_len,
            patch_size=self.patch_size,
            in_chans=self.in_chans,
            embed_dim=self.embed_dim,
        )
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
        if self.prompt_len > 0:
            if self.use_split_prompt:
                self.prompt_embed_inv = nn.Parameter(torch.zeros(1, self.prompt_len, self.embed_dim))
                self.prompt_embed_pla = nn.Parameter(torch.zeros(1, self.prompt_len, self.embed_dim))
                self.prompt_embed = None
            else:
                self.prompt_embed = nn.Parameter(torch.zeros(1, self.prompt_len, self.embed_dim))
                self.prompt_embed_inv = None
                self.prompt_embed_pla = None
        else:
            self.prompt_embed = None
            self.prompt_embed_inv = None
            self.prompt_embed_pla = None

        if self.use_spectral_adapter:
            self.band_scale = nn.Parameter(torch.zeros(1, 1, self.band_num))
            self.band_bias = nn.Parameter(torch.zeros(1, 1, self.band_num))
        else:
            self.band_scale = None
            self.band_bias = None

        total_tokens = 1 + self.prompt_len + num_patches
        self.pos_embed = nn.Parameter(torch.zeros(1, total_tokens, self.embed_dim))
        self.pos_drop = nn.Dropout(self.drop_rate)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.embed_dim,
            nhead=self.num_heads,
            dim_feedforward=int(self.embed_dim * self.mlp_ratio),
            dropout=self.drop_rate,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.blocks = nn.TransformerEncoder(encoder_layer, num_layers=self.depth)
        self.norm = nn.LayerNorm(self.embed_dim)

        self.output_num = self.embed_dim

        self._init_weights()

    def apply_spectral_adapter(self, x):
        if (not self.use_spectral_adapter) or (self.band_scale is None):
            return x

        if x.dim() == 2:
            x = x.unsqueeze(1)
        elif x.dim() == 3 and x.shape[-1] == 1:
            x = x.transpose(1, 2)

        B, C, L = x.shape
        if L != self.input_len:
            x = F.interpolate(x, size=self.input_len, mode="linear", align_corners=False)
            L = self.input_len

        x = x.view(B, C, self.band_num, self.band_width)

        scale = 1.0 + self.adapter_delta * torch.tanh(self.band_scale)
        bias = self.adapter_delta * self.band_bias

        x = x * scale.unsqueeze(-1) + bias.unsqueeze(-1)
        x = x.view(B, C, L)
        return x


    def _init_weights(self):
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        if self.prompt_embed is not None:
            nn.init.trunc_normal_(self.prompt_embed, std=0.02)
        if self.prompt_embed_inv is not None:
            nn.init.trunc_normal_(self.prompt_embed_inv, std=0.02)
        if self.prompt_embed_pla is not None:
            nn.init.zeros_(self.prompt_embed_pla)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def _get_prompt_tokens(self, B, prompt_mode="both"):
        if self.prompt_len <= 0:
            return None

        if self.use_split_prompt:
            inv = self.prompt_embed_inv.expand(B, -1, -1)
            pla = self.prompt_embed_pla.expand(B, -1, -1)
            if prompt_mode == "inv_only":
                return inv
            elif prompt_mode == "pla_only":
                return pla
            else:
                return inv + pla
        else:
            return self.prompt_embed.expand(B, -1, -1)

    def forward_features(self, x, prompt_mode="both"):
        x = self.apply_spectral_adapter(x)
        x = self.patch_embed(x)
        B = x.shape[0]

        cls_tokens = self.cls_token.expand(B, -1, -1)

        prompt_tokens = self._get_prompt_tokens(B, prompt_mode=prompt_mode)
        if prompt_tokens is not None:
            x = torch.cat((cls_tokens, prompt_tokens, x), dim=1)
        else:
            x = torch.cat((cls_tokens, x), dim=1)

        x = x + self.pos_embed[:, :x.size(1), :]
        x = self.pos_drop(x)
        x = self.blocks(x)
        x = self.norm(x)

        cls_feat = x[:, 0]
        return cls_feat

    def forward_tokens(self, x, prompt_mode="both"):
        x = self.apply_spectral_adapter(x)
        x = self.patch_embed(x)
        B = x.shape[0]

        cls_tokens = self.cls_token.expand(B, -1, -1)

        prompt_tokens = self._get_prompt_tokens(B, prompt_mode=prompt_mode)
        if prompt_tokens is not None:
            x = torch.cat((cls_tokens, prompt_tokens, x), dim=1)
        else:
            x = torch.cat((cls_tokens, x), dim=1)

        x = x + self.pos_embed[:, :x.size(1), :]
        x = self.pos_drop(x)
        x = self.blocks(x)
        x = self.norm(x)

        cls_feat = x[:, 0]
        if self.prompt_len > 0:
            patch_tokens = x[:, 1 + self.prompt_len:, :]
        else:
            patch_tokens = x[:, 1:, :]

        return cls_feat, patch_tokens


    def forward(self, x):
        return self.forward_features(x, prompt_mode="both")


if __name__ == "__main__":
    model = ViT1D(cfg=type("cfg", (), {
        "Model": type("m", (), {
            "input_len": 512,
            "patch_size": 16,
            "in_chans": 1,
            "embed_dim": 256,
            "depth": 4,
            "num_heads": 4,
            "mlp_ratio": 4.0,
            "drop_rate": 0.1,
            "prompt_len": 0,
        })()
    })())
    y = model(torch.randn(2, 1, 512))
    print(y.shape)
