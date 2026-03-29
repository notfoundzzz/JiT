import argparse
import random
import time
from io import BytesIO
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter

from util.crop import center_crop_arr


def parse_args():
    parser = argparse.ArgumentParser("Create a minimal paired restoration dataset")
    parser.add_argument("--input_dir", required=True, type=str, help="Directory with source natural images")
    parser.add_argument("--output_dir", required=True, type=str, help="Output root containing hq/ and lq/")
    parser.add_argument("--img_size", default=256, type=int)
    parser.add_argument("--num_samples", default=200, type=int)
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--extensions", default="jpg,jpeg,png,webp,bmp", type=str)
    parser.add_argument("--scan_log_freq", default=5000, type=int)
    parser.add_argument("--save_log_freq", default=100, type=int)
    parser.add_argument("--max_source_images", default=4096, type=int)
    return parser.parse_args()


def sample_source_images(input_dir, extensions, target_count, scan_log_freq, rng):
    suffixes = {f".{item.strip().lower()}" for item in extensions.split(",") if item.strip()}
    images = []
    matched = 0
    started = time.time()
    print(f"scanning source directory: {input_dir}")
    for idx, path in enumerate(sorted(Path(input_dir).rglob("*")), start=1):
        if path.is_file() and path.suffix.lower() in suffixes:
            matched += 1
            if len(images) < target_count:
                images.append(path)
            else:
                replace_idx = rng.randrange(matched)
                if replace_idx < target_count:
                    images[replace_idx] = path
            if matched == 1 or matched % scan_log_freq == 0:
                elapsed = max(time.time() - started, 1e-6)
                print(
                    f"scan matched {matched} images after {idx} entries ({elapsed:.1f}s) | "
                    f"sampled {len(images)}"
                )
    print(
        f"scan complete: matched {matched} images, sampled {len(images)} source images "
        f"in {time.time() - started:.1f}s"
    )
    return images


def random_degrade(image, rng):
    degraded = image.copy()

    if rng.random() < 0.9:
        radius = rng.uniform(0.2, 1.8)
        degraded = degraded.filter(ImageFilter.GaussianBlur(radius=radius))

    scale = rng.choice([2, 3, 4])
    low_w = max(16, degraded.size[0] // scale)
    low_h = max(16, degraded.size[1] // scale)
    down_method = rng.choice([Image.Resampling.BILINEAR, Image.Resampling.BICUBIC, Image.Resampling.BOX])
    degraded = degraded.resize((low_w, low_h), down_method)
    up_method = rng.choice([Image.Resampling.BILINEAR, Image.Resampling.BICUBIC])
    degraded = degraded.resize(image.size, up_method)

    if rng.random() < 0.8:
        quality = int(rng.integers(35, 86))
        buffer = BytesIO()
        degraded.save(buffer, format="JPEG", quality=quality)
        buffer.seek(0)
        degraded = Image.open(buffer).convert("RGB")

    array = np.array(degraded).astype(np.float32)
    if rng.random() < 0.8:
        sigma = rng.uniform(1.0, 8.0)
        noise = rng.normal(0.0, sigma, size=array.shape)
        array = np.clip(array + noise, 0, 255)

    if rng.random() < 0.3:
        gain = rng.uniform(0.85, 1.15)
        bias = rng.uniform(-10.0, 10.0)
        array = np.clip(array * gain + bias, 0, 255)

    return Image.fromarray(array.astype(np.uint8))


def main():
    args = parse_args()
    rng = random.Random(args.seed)
    np_rng = np.random.default_rng(args.seed)

    source_images = sample_source_images(
        args.input_dir,
        args.extensions,
        min(args.num_samples, args.max_source_images),
        args.scan_log_freq,
        rng,
    )
    if not source_images:
        raise RuntimeError(f"No source images found under {args.input_dir}")

    output_root = Path(args.output_dir)
    hq_dir = output_root / "hq"
    lq_dir = output_root / "lq"
    hq_dir.mkdir(parents=True, exist_ok=True)
    lq_dir.mkdir(parents=True, exist_ok=True)

    started = time.time()
    skipped = 0
    print(f"starting paired generation: target={args.num_samples}, img_size={args.img_size}")
    for index in range(args.num_samples):
        source_path = source_images[index % len(source_images)]
        try:
            image = Image.open(source_path).convert("RGB")
        except Exception as exc:
            print(f"skip {source_path}: {exc}")
            skipped += 1
            continue

        hq = center_crop_arr(image, args.img_size)
        lq = random_degrade(hq, np_rng)

        sample_name = f"sample_{index:05d}.png"
        hq.save(hq_dir / sample_name)
        lq.save(lq_dir / sample_name)

        if index == 0 or (index + 1) % args.save_log_freq == 0 or index + 1 == args.num_samples:
            elapsed = max(time.time() - started, 1e-6)
            rate = (index + 1) / elapsed
            remaining = args.num_samples - (index + 1)
            eta = remaining / rate if rate > 0 else float("inf")
            print(
                f"saved pair {index + 1}/{args.num_samples} | "
                f"rate {rate:.2f} pairs/s | eta {eta/60:.1f} min | skipped {skipped}"
            )

    print("ready:")
    print("hq:", hq_dir.resolve())
    print("lq:", lq_dir.resolve())


if __name__ == "__main__":
    main()
