import argparse
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from PIL import Image

from denoiser import Denoiser


def parse_args():
    parser = argparse.ArgumentParser("Black-box JiT inference for single-image restoration and frame interpolation")
    parser.add_argument("--checkpoint", required=True, type=str)
    parser.add_argument("--mode", choices=["spatial", "temporal"], required=True)
    parser.add_argument("--input", type=str, help="Input image for spatial mode")
    parser.add_argument("--frame0", type=str, help="First frame for temporal mode")
    parser.add_argument("--frame1", type=str, help="Second frame for temporal mode")
    parser.add_argument("--output_dir", default="./blackbox_outputs", type=str)
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
    parser.add_argument("--sr_scale", default=1, type=int, help="Spatial degradation factor before bicubic upsampling")
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


def load_rgb_image(path, img_size):
    image = Image.open(path).convert("RGB").resize((img_size, img_size), Image.Resampling.BICUBIC)
    tensor = torch.from_numpy(np.array(image)).permute(2, 0, 1).to(torch.float32).div_(255.0)
    return tensor * 2.0 - 1.0


def degrade_for_spatial_sr(image_tensor, scale):
    if scale <= 1:
        return image_tensor.clone()
    low_size = max(1, image_tensor.shape[-1] // scale)
    image = tensor_to_pil(image_tensor)
    low_res = image.resize((low_size, low_size), Image.Resampling.BICUBIC)
    upsampled = low_res.resize((image_tensor.shape[-1], image_tensor.shape[-1]), Image.Resampling.BICUBIC)
    tensor = torch.from_numpy(np.array(upsampled)).permute(2, 0, 1).to(torch.float32).div_(255.0)
    return tensor * 2.0 - 1.0


def tensor_to_pil(tensor):
    tensor = ((tensor + 1.0) / 2.0).clamp(0.0, 1.0)
    array = tensor.mul(255).byte().permute(1, 2, 0).cpu().numpy()
    return Image.fromarray(array)


def save_image(tensor, path):
    tensor_to_pil(tensor).save(path)


def make_spatial_input(args):
    if not args.input:
        raise ValueError("--input is required in spatial mode")
    clean = load_rgb_image(args.input, args.img_size)
    blackbox_input = degrade_for_spatial_sr(clean, args.sr_scale)
    stem = Path(args.input).stem if args.sr_scale <= 1 else f"{Path(args.input).stem}_x{args.sr_scale}"
    return blackbox_input, clean, stem


def make_temporal_input(args):
    if not args.frame0 or not args.frame1:
        raise ValueError("--frame0 and --frame1 are required in temporal mode")
    frame0 = load_rgb_image(args.frame0, args.img_size)
    frame1 = load_rgb_image(args.frame1, args.img_size)
    mean_frame = 0.5 * (frame0 + frame1)
    stem = f"{Path(args.frame0).stem}_to_{Path(args.frame1).stem}"
    return mean_frame, (frame0, frame1), stem


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

    if args.mode == "spatial":
        z_image, aux_inputs, stem = make_spatial_input(args)
    else:
        z_image, aux_inputs, stem = make_temporal_input(args)

    z = z_image.unsqueeze(0).to(args.device)
    t = torch.tensor([args.timestep], device=args.device)
    y = torch.tensor([args.label], dtype=torch.long, device=args.device)

    with torch.no_grad():
        x_pred = model.net(z, t, y)[0]

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.mode == "spatial":
        save_image(aux_inputs, output_dir / f"{stem}_input.png")
        save_image(z_image, output_dir / f"{stem}_blackbox_input.png")
    else:
        frame0, frame1 = aux_inputs
        save_image(frame0, output_dir / f"{stem}_frame0.png")
        save_image(frame1, output_dir / f"{stem}_frame1.png")
        save_image(z_image, output_dir / f"{stem}_mean_input.png")

    save_image(x_pred, output_dir / f"{stem}_pred.png")

    print("mode:", args.mode)
    print("checkpoint:", args.checkpoint)
    print("ema key:", args.ema_key)
    print("input shape:", tuple(z.shape))
    print("timestep:", float(t.item()))
    print("label:", int(y.item()))
    print("output shape:", tuple(x_pred.shape))
    print("saved to:", output_dir.resolve())


if __name__ == "__main__":
    main()
