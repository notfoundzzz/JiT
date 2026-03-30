import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.serialization import add_safe_globals

from denoiser_restoration_lqmain import RestorationDenoiserLQMain
from util.amp import get_cuda_autocast_kwargs
from util.crop import center_crop_arr


def parse_args():
    parser = argparse.ArgumentParser("Run JiT LQ-main restoration inference")
    parser.add_argument("--checkpoint", required=True, type=str)
    parser.add_argument("--input", required=True, type=str)
    parser.add_argument("--output", required=True, type=str)
    parser.add_argument("--model", default="JiT-L/32", type=str)
    parser.add_argument("--img_size", default=256, type=int)
    parser.add_argument("--qwen_model_path", required=True, type=str)
    parser.add_argument("--noise_scale", default=1.0, type=float)
    parser.add_argument("--sampling_method", default="heun", type=str)
    parser.add_argument("--num_sampling_steps", default=1, type=int)
    parser.add_argument("--P_mean", default=-0.8, type=float)
    parser.add_argument("--P_std", default=0.8, type=float)
    parser.add_argument("--t_eps", default=5e-2, type=float)
    parser.add_argument("--attn_dropout", default=0.0, type=float)
    parser.add_argument("--proj_dropout", default=0.0, type=float)
    parser.add_argument("--ema_decay1", default=0.9999, type=float)
    parser.add_argument("--ema_decay2", default=0.9996, type=float)
    parser.add_argument("--recon_weight", default=1.0, type=float)
    parser.add_argument("--lora_rank", default=8, type=int)
    parser.add_argument("--lora_alpha", default=16.0, type=float)
    parser.add_argument("--lora_dropout", default=0.0, type=float)
    parser.add_argument("--ema_key", default="model_ema1", choices=["model", "model_ema1", "model_ema2"])
    parser.add_argument("--device", default="cuda", type=str)
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--disable_amp", action="store_true")
    return parser.parse_args()


def load_image(path, img_size):
    image = Image.open(path).convert("RGB")
    image = center_crop_arr(image, img_size)
    array = np.array(image)
    tensor = torch.from_numpy(array).permute(2, 0, 1).to(torch.float32).div_(255.0)
    return tensor * 2.0 - 1.0


def save_image(tensor, path):
    tensor = ((tensor + 1.0) / 2.0).clamp(0.0, 1.0)
    array = tensor.mul(255).byte().permute(1, 2, 0).cpu().numpy()
    Image.fromarray(array).save(path)


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available")

    model = RestorationDenoiserLQMain(args)
    add_safe_globals([argparse.Namespace])
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint[args.ema_key], strict=True)
    model.eval()
    model.to(args.device)

    cond_img = load_image(args.input, args.img_size).unsqueeze(0).to(args.device)
    with torch.no_grad():
        if args.disable_amp:
            sample = model.generate(cond_img)[0]
        else:
            autocast_kwargs = get_cuda_autocast_kwargs(args.device)
            with torch.amp.autocast(**autocast_kwargs):
                sample = model.generate(cond_img)[0]

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_image(sample, output_path)
    print("saved to:", output_path.resolve())


if __name__ == "__main__":
    main()
