import argparse
from pathlib import Path

import torch
from PIL import Image, ImageDraw
from torchvision.io import read_image

from model_jit import JiT_models
from util.amp import get_cuda_autocast_kwargs


def parse_args():
    parser = argparse.ArgumentParser("Render a display-first reconstruction demo")
    parser.add_argument("--checkpoint", default="./output_display_demo/checkpoint-last.pth", type=str)
    parser.add_argument("--data_dir", default="./toy_display_data/train", type=str)
    parser.add_argument("--output_dir", default="./display_demo_outputs", type=str)
    parser.add_argument("--device", default="cuda", type=str)
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--timestep", default=0.35, type=float)
    return parser.parse_args()


def to_pil(image_tensor):
    image_tensor = ((image_tensor + 1.0) / 2.0).clamp(0.0, 1.0)
    image_array = image_tensor.mul(255).byte().permute(1, 2, 0).cpu().numpy()
    return Image.fromarray(image_array)


def load_examples(data_dir):
    items = []
    for class_dir in sorted(Path(data_dir).glob("class_*")):
        image_path = sorted(class_dir.glob("*.png"))[0]
        label = int(class_dir.name.split("_")[-1])
        image = read_image(str(image_path)).to(torch.float32).div_(255.0)
        image = image * 2.0 - 1.0
        items.append((label, image))
    return items


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
    draw.text((4, height + 3), f"{title} | clean", fill="black")
    draw.text((width + 4, height + 3), "noisy", fill="black")
    draw.text((width * 2 + 4, height + 3), "pred", fill="black")
    return canvas


def make_contact_sheet(images):
    cols = 2
    rows = (len(images) + cols - 1) // cols
    width, height = images[0].size
    sheet = Image.new("RGB", (cols * width, rows * height), (245, 245, 245))
    for index, image in enumerate(images):
        sheet.paste(image, ((index % cols) * width, (index // cols) * height))
    return sheet


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    torch.set_float32_matmul_precision("high")
    autocast_kwargs = get_cuda_autocast_kwargs(args.device)

    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    train_args = checkpoint["args"]
    config = train_args if isinstance(train_args, dict) else vars(train_args)

    model = JiT_models[config["model"]](
        input_size=config["img_size"],
        in_channels=3,
        num_classes=config["class_num"],
        attn_drop=0.0,
        proj_drop=0.0,
    )
    model.load_state_dict(checkpoint["model"], strict=True)
    model.to(args.device)
    model.eval()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    triptychs = []
    t = torch.tensor([args.timestep], device=args.device)
    for label, clean in load_examples(args.data_dir):
        clean = clean.unsqueeze(0).to(args.device)
        labels = torch.tensor([label], dtype=torch.long, device=args.device)
        noise = torch.randn_like(clean) * config["noise_scale"]
        noisy = t.view(1, 1, 1, 1) * clean + (1.0 - t.view(1, 1, 1, 1)) * noise
        with torch.no_grad():
            with torch.amp.autocast(**autocast_kwargs):
                pred = model(noisy, t, labels)
        triptych = make_triptych(clean[0], noisy[0], pred[0], f"class_{label}")
        triptychs.append(triptych)
        triptych.save(output_dir / f"class_{label}_triptych.png")

    make_contact_sheet(triptychs).save(output_dir / "contact_sheet.png")
    print(f"Saved outputs to {output_dir.resolve()}")


if __name__ == "__main__":
    main()
