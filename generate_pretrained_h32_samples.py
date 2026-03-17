import argparse
from pathlib import Path
from types import SimpleNamespace

import torch
from PIL import Image

from denoiser import Denoiser
from util.amp import get_cuda_autocast_kwargs


def parse_args():
    parser = argparse.ArgumentParser("Generate images from a pretrained JiT-H/32 checkpoint")
    parser.add_argument(
        "--checkpoint",
        default="/home/zhahl/JiT-cloud-smoke/JiT-H-32/checkpoint-last.pth",
        type=str,
    )
    parser.add_argument("--output_dir", default="./generated_pretrained_h32", type=str)
    parser.add_argument("--model", default="JiT-H/32", type=str)
    parser.add_argument("--img_size", default=512, type=int)
    parser.add_argument("--class_num", default=1000, type=int)
    parser.add_argument("--noise_scale", default=2.0, type=float)
    parser.add_argument("--sampling_method", default="heun", type=str)
    parser.add_argument("--num_sampling_steps", default=50, type=int)
    parser.add_argument("--cfg", default=2.3, type=float)
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
    parser.add_argument("--labels", default="0", type=str)
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


def save_tensor_image(image_tensor, output_path):
    image_tensor = ((image_tensor + 1.0) / 2.0).clamp(0.0, 1.0)
    image_array = image_tensor.mul(255).byte().permute(1, 2, 0).cpu().numpy()
    Image.fromarray(image_array).save(output_path)


def main():
    args = parse_args()
    torch.manual_seed(args.seed)

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available")

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    state_dict = checkpoint[args.ema_key]

    model = Denoiser(build_denoiser_args(args))
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    model.to(args.device)

    label_values = [int(item.strip()) for item in args.labels.split(",") if item.strip()]
    labels = torch.tensor(label_values, dtype=torch.long, device=args.device)

    autocast_kwargs = get_cuda_autocast_kwargs(args.device)
    with torch.no_grad():
        with torch.amp.autocast(**autocast_kwargs):
            samples = model.generate(labels)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for index, (sample, label) in enumerate(zip(samples, labels.tolist())):
        save_tensor_image(sample, output_dir / f"label_{label:03d}_seed_{args.seed}_{index:02d}.png")

    print("checkpoint:", args.checkpoint)
    print("ema key:", args.ema_key)
    print("labels:", label_values)
    print("saved to:", output_dir.resolve())


if __name__ == "__main__":
    main()
