import torch
import torch.nn as nn
import torch.nn.functional as F

from model_jit_restoration import build_restoration_model


class RestorationDenoiser(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.net = build_restoration_model(
            model_name=args.model,
            img_size=args.img_size,
            attn_dropout=args.attn_dropout,
            proj_dropout=args.proj_dropout,
            cond_channels=3,
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
        self.ema_params1 = None
        self.ema_params2 = None

    def sample_t(self, n, device=None):
        z = torch.randn(n, device=device) * self.P_std + self.P_mean
        return torch.sigmoid(z)

    def forward(self, x, cond_img):
        t = self.sample_t(x.size(0), device=x.device).view(-1, *([1] * (x.ndim - 1)))
        e = torch.randn_like(x) * self.noise_scale
        z = t * x + (1 - t) * e
        v = (x - z) / (1 - t).clamp_min(self.t_eps)

        x_pred = self.net(z, t.flatten(), cond_img)
        v_pred = (x_pred - z) / (1 - t).clamp_min(self.t_eps)

        diffusion_loss = ((v - v_pred) ** 2).mean(dim=(1, 2, 3)).mean()
        recon_loss = F.l1_loss(x_pred, x)
        return diffusion_loss + self.recon_weight * recon_loss

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
