import argparse
import os
from pathlib import Path

import torch
from PIL import Image

from denoiser import Denoiser


def parse_args():
    parser = argparse.ArgumentParser("Minimal JiT sampler")
    parser.add_argument("--checkpoint", default="./output_local/checkpoint-last.pth", type=str)
    parser.add_argument("--output_dir", default="./samples_local", type=str)
    parser.add_argument("--num_samples", default=4, type=int)
    parser.add_argument("--labels", default="", type=str)
    parser.add_argument("--cfg", default=None, type=float)
    parser.add_argument("--steps", default=None, type=int)
    parser.add_argument("--device", default="cuda", type=str)
    parser.add_argument("--seed", default=0, type=int)
    return parser.parse_args()


def save_tensor_image(image_tensor, output_path):
    image_tensor = ((image_tensor + 1.0) / 2.0).clamp(0.0, 1.0)
    image_array = image_tensor.mul(255).byte().permute(1, 2, 0).cpu().numpy()
    Image.fromarray(image_array).save(output_path)


def main():
    args = parse_args()
    torch.manual_seed(args.seed)

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available")

    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    train_args = checkpoint["args"]

    model = Denoiser(train_args)
    ema_state_dict = checkpoint.get("model_ema1", checkpoint["model"])
    model.load_state_dict(ema_state_dict, strict=True)

    if args.cfg is not None:
        model.cfg_scale = args.cfg
    if args.steps is not None:
        model.steps = args.steps

    model.eval()
    model.to(args.device)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.labels:
        label_values = [int(item.strip()) for item in args.labels.split(",") if item.strip()]
        if not label_values:
            raise ValueError("No valid labels were provided")
        labels = torch.tensor(label_values, dtype=torch.long, device=args.device)
    else:
        labels = torch.zeros(args.num_samples, dtype=torch.long, device=args.device)

    args.num_samples = int(labels.shape[0])

    with torch.no_grad():
        with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=args.device.startswith("cuda")):
            samples = model.generate(labels)

    for index, (sample, label) in enumerate(zip(samples, labels.tolist())):
        save_tensor_image(sample, output_dir / f"class{label}_sample_{index:02d}.png")

    print(f"Saved {args.num_samples} samples to {output_dir.resolve()}")


if __name__ == "__main__":
    main()
