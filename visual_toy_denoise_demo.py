import argparse
from pathlib import Path

import torch
from PIL import Image, ImageDraw
from torchvision.io import read_image

from denoiser import Denoiser


def parse_args():
    parser = argparse.ArgumentParser("Visual toy denoising demo")
    parser.add_argument("--checkpoint", default="./output_visual_toy/checkpoint-last.pth", type=str)
    parser.add_argument("--data_dir", default="./toy_visual_data/train", type=str)
    parser.add_argument("--output_dir", default="./denoise_visual_toy", type=str)
    parser.add_argument("--device", default="cuda", type=str)
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--timestep", default=0.5, type=float)
    return parser.parse_args()


def to_pil(image_tensor):
    image_tensor = ((image_tensor + 1.0) / 2.0).clamp(0.0, 1.0)
    image_array = image_tensor.mul(255).byte().permute(1, 2, 0).cpu().numpy()
    return Image.fromarray(image_array)


def make_triptych(clean, noisy, pred, title):
    clean_img = to_pil(clean)
    noisy_img = to_pil(noisy)
    pred_img = to_pil(pred)

    width, height = clean_img.size
    label_h = 22
    canvas = Image.new("RGB", (width * 3, height + label_h), "white")
    canvas.paste(clean_img, (0, 0))
    canvas.paste(noisy_img, (width, 0))
    canvas.paste(pred_img, (width * 2, 0))

    draw = ImageDraw.Draw(canvas)
    draw.text((4, height + 3), f"{title} | clean")
    draw.text((width + 4, height + 3), "noisy")
    draw.text((width * 2 + 4, height + 3), "pred")
    return canvas


def load_demo_examples(data_dir):
    examples = []
    for class_dir in sorted(Path(data_dir).glob("class_*")):
        image_path = sorted(class_dir.glob("*.png"))[0]
        label = int(class_dir.name.split("_")[-1])
        image = read_image(str(image_path)).to(torch.float32).div_(255.0)
        image = image * 2.0 - 1.0
        examples.append((label, image))
    return examples


def main():
    args = parse_args()
    torch.manual_seed(args.seed)

    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    train_args = checkpoint["args"]
    model = Denoiser(train_args)
    model.load_state_dict(checkpoint.get("model_ema1", checkpoint["model"]), strict=True)
    model.to(args.device)
    model.eval()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    examples = load_demo_examples(args.data_dir)
    t = torch.tensor([args.timestep], device=args.device)

    for label, clean in examples:
        clean = clean.unsqueeze(0).to(args.device)
        labels = torch.tensor([label], dtype=torch.long, device=args.device)
        noise = torch.randn_like(clean) * train_args.noise_scale
        z = t.view(1, 1, 1, 1) * clean + (1.0 - t.view(1, 1, 1, 1)) * noise

        with torch.no_grad():
            with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=args.device.startswith("cuda")):
                pred = model.net(z, t, labels)

        triptych = make_triptych(clean[0], z[0], pred[0], f"class_{label}")
        triptych.save(output_dir / f"class_{label}_triptych.png")

    print(f"Saved denoising demos to {output_dir.resolve()}")


if __name__ == "__main__":
    main()
