import argparse
from pathlib import Path

import torch
from PIL import Image, ImageDraw
from torch.serialization import add_safe_globals

from denoiser_restoration import RestorationDenoiser
from infer_jit_restoration import load_image, save_image
from util.amp import get_cuda_autocast_kwargs


def parse_args():
    parser = argparse.ArgumentParser("Run batch JiT restoration inference with triptych outputs")
    parser.add_argument("--checkpoint", required=True, type=str)
    parser.add_argument("--input_dir", required=True, type=str)
    parser.add_argument("--target_dir", required=True, type=str)
    parser.add_argument("--output_dir", required=True, type=str)
    parser.add_argument("--model", default="JiT-L/32", type=str)
    parser.add_argument("--img_size", default=256, type=int)
    parser.add_argument("--qwen_model_path", required=True, type=str)
    parser.add_argument("--noise_scale", default=1.0, type=float)
    parser.add_argument("--sampling_method", default="heun", type=str)
    parser.add_argument("--num_sampling_steps", default=50, type=int)
    parser.add_argument("--P_mean", default=-0.8, type=float)
    parser.add_argument("--P_std", default=0.8, type=float)
    parser.add_argument("--t_eps", default=5e-2, type=float)
    parser.add_argument("--attn_dropout", default=0.0, type=float)
    parser.add_argument("--proj_dropout", default=0.0, type=float)
    parser.add_argument("--ema_decay1", default=0.9999, type=float)
    parser.add_argument("--ema_decay2", default=0.9996, type=float)
    parser.add_argument("--recon_weight", default=1.0, type=float)
    parser.add_argument("--lora_rank", default=0, type=int)
    parser.add_argument("--lora_alpha", default=16.0, type=float)
    parser.add_argument("--lora_dropout", default=0.0, type=float)
    parser.add_argument("--ema_key", default="model_ema1", choices=["model", "model_ema1", "model_ema2"])
    parser.add_argument("--device", default="cuda", type=str)
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--disable_amp", action="store_true")
    parser.add_argument("--limit", default=0, type=int, help="0 means all images")
    return parser.parse_args()


def to_pil_from_path(path, img_size):
    tensor = load_image(path, img_size)
    tensor = ((tensor + 1.0) / 2.0).clamp(0.0, 1.0)
    array = tensor.mul(255).byte().permute(1, 2, 0).cpu().numpy()
    return Image.fromarray(array)


def save_triptych(input_path, restored_path, target_path, output_path, img_size):
    input_img = to_pil_from_path(input_path, img_size)
    restored_img = to_pil_from_path(restored_path, img_size)
    target_img = to_pil_from_path(target_path, img_size)

    width, height = input_img.size
    label_h = 24
    canvas = Image.new("RGB", (width * 3, height + label_h), "white")
    canvas.paste(input_img, (0, 0))
    canvas.paste(restored_img, (width, 0))
    canvas.paste(target_img, (width * 2, 0))

    draw = ImageDraw.Draw(canvas)
    draw.text((6, height + 4), "LQ input", fill="black")
    draw.text((width + 6, height + 4), "Restored", fill="black")
    draw.text((width * 2 + 6, height + 4), "HQ target", fill="black")
    canvas.save(output_path)


def collect_input_files(input_dir):
    files = sorted(
        [p for p in input_dir.iterdir() if p.is_file() and p.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp", ".bmp"}]
    )
    return files


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available")

    input_dir = Path(args.input_dir)
    target_dir = Path(args.target_dir)
    output_dir = Path(args.output_dir)
    restored_dir = output_dir / "restored"
    triptych_dir = output_dir / "triptych"
    restored_dir.mkdir(parents=True, exist_ok=True)
    triptych_dir.mkdir(parents=True, exist_ok=True)

    input_files = collect_input_files(input_dir)
    if args.limit > 0:
        input_files = input_files[: args.limit]
    if not input_files:
        raise RuntimeError(f"No input images found under {input_dir}")

    print(f"Found {len(input_files)} input images")
    print("Loading model once for batch inference...")
    model = RestorationDenoiser(args)
    add_safe_globals([argparse.Namespace])
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint[args.ema_key], strict=True)
    model.eval()
    model.to(args.device)

    autocast_kwargs = get_cuda_autocast_kwargs(args.device)
    for idx, input_path in enumerate(input_files, start=1):
        target_path = target_dir / input_path.name
        if not target_path.exists():
            print(f"[{idx}/{len(input_files)}] skipping {input_path.name}: target not found")
            continue

        restored_path = restored_dir / f"{input_path.stem}_restored.png"
        triptych_path = triptych_dir / f"{input_path.stem}_triptych.png"

        cond_img = load_image(str(input_path), args.img_size).unsqueeze(0).to(args.device)
        with torch.no_grad():
            if args.disable_amp:
                sample = model.generate(cond_img)[0]
            else:
                with torch.amp.autocast(**autocast_kwargs):
                    sample = model.generate(cond_img)[0]

        save_image(sample, restored_path)
        save_triptych(str(input_path), str(restored_path), str(target_path), triptych_path, args.img_size)
        print(f"[{idx}/{len(input_files)}] saved {restored_path.name} and {triptych_path.name}")

    print("Batch inference complete.")
    print("Restored dir:", restored_dir.resolve())
    print("Triptych dir:", triptych_dir.resolve())


if __name__ == "__main__":
    main()
