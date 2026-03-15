import argparse
from types import SimpleNamespace

import torch

from denoiser import Denoiser


def parse_args():
    parser = argparse.ArgumentParser("Verify a pretrained JiT checkpoint")
    parser.add_argument("--checkpoint", required=True, type=str)
    parser.add_argument("--model", default="JiT-L/32", type=str)
    parser.add_argument("--img_size", default=512, type=int)
    parser.add_argument("--class_num", default=1000, type=int)
    parser.add_argument("--noise_scale", default=2.0, type=float)
    parser.add_argument("--sampling_method", default="heun", type=str)
    parser.add_argument("--num_sampling_steps", default=50, type=int)
    parser.add_argument("--cfg", default=2.5, type=float)
    parser.add_argument("--interval_min", default=0.1, type=float)
    parser.add_argument("--interval_max", default=1.0, type=float)
    parser.add_argument("--label_drop_prob", default=0.1, type=float)
    parser.add_argument("--P_mean", default=-0.8, type=float)
    parser.add_argument("--P_std", default=0.8, type=float)
    parser.add_argument("--t_eps", default=5e-2, type=float)
    parser.add_argument("--attn_dropout", default=0.0, type=float)
    parser.add_argument("--proj_dropout", default=0.0, type=float)
    parser.add_argument("--ema_key", default="model_ema1", choices=["model", "model_ema1", "model_ema2"])
    parser.add_argument("--device", default="cuda", type=str)
    parser.add_argument("--label", default=0, type=int)
    parser.add_argument("--timestep", default=0.5, type=float)
    parser.add_argument("--seed", default=0, type=int)
    return parser.parse_args()


def build_denoiser_args(args):
    return SimpleNamespace(
        model=args.model,
        img_size=args.img_size,
        class_num=args.class_num,
        noise_scale=args.noise_scale,
        sampling_method=args.sampling_method,
        num_sampling_steps=args.num_sampling_steps,
        cfg=args.cfg,
        interval_min=args.interval_min,
        interval_max=args.interval_max,
        label_drop_prob=args.label_drop_prob,
        P_mean=args.P_mean,
        P_std=args.P_std,
        t_eps=args.t_eps,
        attn_dropout=args.attn_dropout,
        proj_dropout=args.proj_dropout,
        ema_decay1=0.9999,
        ema_decay2=0.9996,
    )


def main():
    args = parse_args()
    torch.manual_seed(args.seed)

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available")

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    if args.ema_key not in checkpoint:
        raise KeyError(f"{args.ema_key} not found in checkpoint. Keys: {sorted(checkpoint.keys())}")

    model = Denoiser(build_denoiser_args(args))
    model.load_state_dict(checkpoint[args.ema_key], strict=True)
    model.eval()
    model.to(args.device)

    z = torch.randn(1, 3, args.img_size, args.img_size, device=args.device) * args.noise_scale
    t = torch.tensor([args.timestep], device=args.device)
    y = torch.tensor([args.label], dtype=torch.long, device=args.device)

    with torch.no_grad():
        x_pred = model.net(z, t, y)

    print("checkpoint keys:", sorted(checkpoint.keys()))
    print("ema key:", args.ema_key)
    print("model:", args.model)
    print("img_size:", args.img_size)
    print("noise_scale:", args.noise_scale)
    print("input z shape:", tuple(z.shape))
    print("input t shape:", tuple(t.shape), "value:", float(t.item()))
    print("input y shape:", tuple(y.shape), "value:", int(y.item()))
    print("output shape:", tuple(x_pred.shape))
    print("output dtype:", x_pred.dtype)
    print("output min/max:", float(x_pred.min().item()), float(x_pred.max().item()))


if __name__ == "__main__":
    main()
