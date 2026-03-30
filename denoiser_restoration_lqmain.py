import torch
import torch.nn as nn
import torch.nn.functional as F

from lora_util import apply_lora_to_modules
from model_jit_restoration import build_restoration_model


class RestorationDenoiserLQMain(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.net = build_restoration_model(
            model_name=args.model,
            img_size=args.img_size,
            attn_dropout=args.attn_dropout,
            proj_dropout=args.proj_dropout,
            qwen_model_path=args.qwen_model_path,
        )
        self.img_size = args.img_size
        self.recon_weight = args.recon_weight
        self.ema_decay1 = args.ema_decay1
        self.ema_decay2 = args.ema_decay2
        self.lora_rank = args.lora_rank
        self.lora_alpha = args.lora_alpha
        self.lora_dropout = args.lora_dropout
        self.ema_params1 = None
        self.ema_params2 = None
        self.lora_modules = []
        if self.lora_rank > 0:
            self.lora_modules = apply_lora_to_modules(
                self.net,
                target_suffixes=("qkv", "proj", "w12", "w3"),
                rank=self.lora_rank,
                alpha=self.lora_alpha,
                dropout=self.lora_dropout,
            )
        self._freeze_pretrained_weights()

    def _freeze_pretrained_weights(self):
        for param in self.net.parameters():
            param.requires_grad = False

        trainable_modules = [
            self.net.cond_encoder.projector,
            self.net.cond_encoder.global_proj,
            self.net.cond_encoder.token_norm,
            self.net.cond_encoder.global_norm,
            self.net.cond_token_norm,
            self.net.cond_global_norm,
        ]
        for module in trainable_modules:
            for param in module.parameters():
                param.requires_grad = True

        for module_name in self.lora_modules:
            module = dict(self.net.named_modules())[module_name]
            for param in module.parameters():
                if param is not module.base.weight and param is not module.base.bias:
                    param.requires_grad = True

    def forward(self, x, cond_img):
        t = torch.zeros(x.size(0), device=x.device, dtype=x.dtype)
        x_pred = self.net(cond_img, t, cond_img)
        recon_loss = F.l1_loss(x_pred, x)
        return self.recon_weight * recon_loss

    @torch.no_grad()
    def generate(self, cond_img):
        t = torch.zeros(cond_img.size(0), device=cond_img.device, dtype=cond_img.dtype)
        return self.net(cond_img, t, cond_img)

    @torch.no_grad()
    def update_ema(self):
        source_params = list(self.parameters())
        for targ, src in zip(self.ema_params1, source_params):
            targ.detach().mul_(self.ema_decay1).add_(src, alpha=1 - self.ema_decay1)
        for targ, src in zip(self.ema_params2, source_params):
            targ.detach().mul_(self.ema_decay2).add_(src, alpha=1 - self.ema_decay2)
