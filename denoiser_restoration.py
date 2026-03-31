import torch
import torch.nn as nn
import torch.nn.functional as F

from lora_util import apply_lora_to_modules
from model_jit_restoration import build_restoration_model


class RestorationDenoiser(nn.Module):
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
        self.P_mean = args.P_mean
        self.P_std = args.P_std
        self.t_eps = args.t_eps
        self.noise_scale = args.noise_scale
        self.method = args.sampling_method
        self.steps = args.num_sampling_steps
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

    def sample_t(self, n, device=None):
        z = torch.randn(n, device=device) * self.P_std + self.P_mean
        return torch.sigmoid(z)

    def forward(self, x, cond_img):
        t = self.sample_t(x.size(0), device=x.device).view(-1, *([1] * (x.ndim - 1)))
        e = torch.randn_like(x) * self.noise_scale
        z = t * x + (1 - t) * e
        x_pred = self.net(z, t.flatten(), cond_img)
        recon_loss = F.l1_loss(x_pred, x)
        return self.recon_weight * recon_loss

    @torch.no_grad()
    def generate(self, cond_img):
        device = cond_img.device
        bsz = cond_img.size(0)
        z = self.noise_scale * torch.randn(bsz, 3, self.img_size, self.img_size, device=device)
        timesteps = torch.linspace(0.0, 1.0, self.steps + 1, device=device).view(-1, *([1] * z.ndim)).expand(-1, bsz, -1, -1, -1)

        if self.method == "euler":
            stepper = self._euler_step
        elif self.method == "heun":
            stepper = self._heun_step
        else:
            raise NotImplementedError

        for i in range(self.steps - 1):
            z = stepper(z, timesteps[i], timesteps[i + 1], cond_img)
        z = self._euler_step(z, timesteps[-2], timesteps[-1], cond_img)
        return z

    @torch.no_grad()
    def _forward_sample(self, z, t, cond_img):
        x_pred = self.net(z, t.flatten(), cond_img)
        return (x_pred - z) / (1.0 - t).clamp_min(self.t_eps)

    @torch.no_grad()
    def _euler_step(self, z, t, t_next, cond_img):
        v_pred = self._forward_sample(z, t, cond_img)
        return z + (t_next - t) * v_pred

    @torch.no_grad()
    def _heun_step(self, z, t, t_next, cond_img):
        v_pred_t = self._forward_sample(z, t, cond_img)
        z_next_euler = z + (t_next - t) * v_pred_t
        v_pred_t_next = self._forward_sample(z_next_euler, t_next, cond_img)
        v_pred = 0.5 * (v_pred_t + v_pred_t_next)
        return z + (t_next - t) * v_pred

    @torch.no_grad()
    def update_ema(self):
        source_params = list(self.parameters())
        for targ, src in zip(self.ema_params1, source_params):
            targ.detach().mul_(self.ema_decay1).add_(src, alpha=1 - self.ema_decay1)
        for targ, src in zip(self.ema_params2, source_params):
            targ.detach().mul_(self.ema_decay2).add_(src, alpha=1 - self.ema_decay2)
