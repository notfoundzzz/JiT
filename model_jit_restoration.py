import torch
import torch.nn as nn

from condition_encoder_qwenvl import QwenVLConditionEncoder
from model_jit import JiT


class JiTRestoration(JiT):
    def __init__(self, *args, qwen_model_path, **kwargs):
        super().__init__(*args, **kwargs)
        self.cond_encoder = QwenVLConditionEncoder(
            model_path=qwen_model_path,
            hidden_size=self.hidden_size,
            num_tokens=self.in_context_len,
        )
        self.cond_token_norm = nn.LayerNorm(self.hidden_size)
        self.cond_global_norm = nn.LayerNorm(self.hidden_size)

    def forward(self, x, t, cond_img):
        """
        x: (N, C, H, W)
        t: (N,)
        cond_img: (N, C, H, W)
        """
        t_emb = self.t_embedder(t)
        cond_global, cond_tokens = self.cond_encoder(cond_img)
        cond_global = self.cond_global_norm(cond_global)
        cond_tokens = self.cond_token_norm(cond_tokens)
        c = t_emb + cond_global

        x = self.x_embedder(x)
        x += self.pos_embed

        for i, block in enumerate(self.blocks):
            if self.in_context_len > 0 and i == self.in_context_start:
                in_context_tokens = cond_tokens + self.in_context_posemb
                x = torch.cat([in_context_tokens, x], dim=1)
            x = block(x, c, self.feat_rope if i < self.in_context_start else self.feat_rope_incontext)

        x = x[:, self.in_context_len:]
        x = self.final_layer(x, c)
        output = self.unpatchify(x, self.patch_size)
        return output


def build_restoration_model(model_name, img_size, attn_dropout, proj_dropout, qwen_model_path):
    kwargs = dict(
        input_size=img_size,
        in_channels=3,
        num_classes=1000,
        attn_drop=attn_dropout,
        proj_drop=proj_dropout,
        qwen_model_path=qwen_model_path,
    )
    if model_name == "JiT-B/16":
        return JiTRestoration(depth=12, hidden_size=768, num_heads=12,
                              bottleneck_dim=128, in_context_len=32, in_context_start=4, patch_size=16, **kwargs)
    if model_name == "JiT-B/32":
        return JiTRestoration(depth=12, hidden_size=768, num_heads=12,
                              bottleneck_dim=128, in_context_len=32, in_context_start=4, patch_size=32, **kwargs)
    if model_name == "JiT-L/16":
        return JiTRestoration(depth=24, hidden_size=1024, num_heads=16,
                              bottleneck_dim=128, in_context_len=32, in_context_start=8, patch_size=16, **kwargs)
    if model_name == "JiT-L/32":
        return JiTRestoration(depth=24, hidden_size=1024, num_heads=16,
                              bottleneck_dim=128, in_context_len=32, in_context_start=8, patch_size=32, **kwargs)
    if model_name == "JiT-H/16":
        return JiTRestoration(depth=32, hidden_size=1280, num_heads=16,
                              bottleneck_dim=256, in_context_len=32, in_context_start=10, patch_size=16, **kwargs)
    if model_name == "JiT-H/32":
        return JiTRestoration(depth=32, hidden_size=1280, num_heads=16,
                              bottleneck_dim=256, in_context_len=32, in_context_start=10, patch_size=32, **kwargs)
    raise KeyError(f"Unknown model: {model_name}")
