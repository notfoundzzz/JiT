import argparse
from pathlib import Path

from PIL import Image, ImageDraw

from infer_jit_restoration import load_image


def parse_args():
    parser = argparse.ArgumentParser("Create a LQ / restored / HQ triptych")
    parser.add_argument("--input", required=True, type=str)
    parser.add_argument("--restored", required=True, type=str)
    parser.add_argument("--target", required=True, type=str)
    parser.add_argument("--output", required=True, type=str)
    parser.add_argument("--img_size", default=512, type=int)
    return parser.parse_args()


def to_pil(path, img_size):
    tensor = load_image(path, img_size)
    tensor = ((tensor + 1.0) / 2.0).clamp(0.0, 1.0)
    array = tensor.mul(255).byte().permute(1, 2, 0).cpu().numpy()
    return Image.fromarray(array)


def main():
    args = parse_args()
    input_img = to_pil(args.input, args.img_size)
    restored_img = to_pil(args.restored, args.img_size)
    target_img = to_pil(args.target, args.img_size)

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

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)
    print("saved triptych to:", output_path.resolve())


if __name__ == "__main__":
    main()
