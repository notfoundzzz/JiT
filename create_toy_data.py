import argparse
import math
import os
import random

import numpy as np
from PIL import Image, ImageDraw, ImageFilter


def parse_args():
    parser = argparse.ArgumentParser("Create structured toy data for JiT")
    parser.add_argument("--output_dir", default="toy_data", type=str)
    parser.add_argument("--img_size", default=64, type=int)
    parser.add_argument("--train_per_class", default=16, type=int)
    parser.add_argument("--val_per_class", default=4, type=int)
    parser.add_argument("--seed", default=0, type=int)
    return parser.parse_args()


def add_noise(image, rng):
    array = np.asarray(image).astype(np.int16)
    noise = rng.integers(-10, 11, size=array.shape)
    array = np.clip(array + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(array)


def draw_circle(draw, size, rng):
    cx = size // 2 + rng.randint(-4, 4)
    cy = size // 2 + rng.randint(-4, 4)
    radius = rng.randint(size // 5, size // 3)
    bbox = (cx - radius, cy - radius, cx + radius, cy + radius)
    draw.ellipse(bbox, fill=(225, 70, 70), outline=(255, 240, 240), width=2)


def draw_square(draw, size, rng):
    half = rng.randint(size // 5, size // 3)
    cx = size // 2 + rng.randint(-5, 5)
    cy = size // 2 + rng.randint(-5, 5)
    bbox = (cx - half, cy - half, cx + half, cy + half)
    draw.rounded_rectangle(bbox, radius=4, fill=(70, 120, 235), outline=(245, 248, 255), width=2)


def draw_stripes(draw, size, rng):
    spacing = rng.randint(8, 12)
    offset = rng.randint(0, spacing)
    for start in range(-size, size * 2, spacing):
        draw.line(
            [(start + offset, 0), (start + offset - size, size)],
            fill=(70, 180, 110),
            width=4,
        )


def draw_diamond(draw, size, rng):
    cx = size // 2 + rng.randint(-4, 4)
    cy = size // 2 + rng.randint(-4, 4)
    radius = rng.randint(size // 5, size // 3)
    points = [
        (cx, cy - radius),
        (cx + radius, cy),
        (cx, cy + radius),
        (cx - radius, cy),
    ]
    draw.polygon(points, fill=(240, 200, 70), outline=(255, 250, 230))


def draw_sunburst(draw, size, rng):
    cx = size // 2 + rng.randint(-3, 3)
    cy = size // 2 + rng.randint(-3, 3)
    radius = rng.randint(size // 7, size // 5)
    for angle in range(0, 360, 30):
        radians = math.radians(angle + rng.randint(-5, 5))
        length = radius + rng.randint(size // 7, size // 5)
        x = cx + int(math.cos(radians) * length)
        y = cy + int(math.sin(radians) * length)
        draw.line([(cx, cy), (x, y)], fill=(255, 210, 80), width=2)
    draw.ellipse((cx - radius, cy - radius, cx + radius, cy + radius), fill=(255, 225, 120))


def render_image(class_id, size, rng):
    backgrounds = {
        0: (248, 240, 232),
        1: (236, 242, 252),
        2: (232, 246, 236),
        3: (248, 244, 224),
    }
    image = Image.new("RGB", (size, size), backgrounds[class_id])
    draw = ImageDraw.Draw(image)

    if class_id == 0:
        draw_circle(draw, size, rng)
    elif class_id == 1:
        draw_square(draw, size, rng)
    elif class_id == 2:
        draw_stripes(draw, size, rng)
    elif class_id == 3:
        draw_diamond(draw, size, rng)
        draw_sunburst(draw, size, rng)
    else:
        raise ValueError(f"Unsupported class id: {class_id}")

    if rng.random() < 0.35:
        image = image.filter(ImageFilter.GaussianBlur(radius=0.4 + rng.random() * 0.6))
    return add_noise(image, np.random.default_rng(rng.randint(0, 10_000)))


def create_split(split_dir, per_class, size, rng):
    for class_id in range(4):
        class_dir = os.path.join(split_dir, f"class_{class_id}")
        os.makedirs(class_dir, exist_ok=True)
        for index in range(per_class):
            image = render_image(class_id, size, rng)
            image.save(os.path.join(class_dir, f"sample_{index:03d}.png"))


def main():
    args = parse_args()
    rng = random.Random(args.seed)

    train_dir = os.path.join(args.output_dir, "train")
    val_dir = os.path.join(args.output_dir, "val")
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)

    create_split(train_dir, args.train_per_class, args.img_size, rng)
    create_split(val_dir, args.val_per_class, args.img_size, rng)

    print(f"Structured toy dataset ready at {args.output_dir}")


if __name__ == "__main__":
    main()
