import argparse
import json
from pathlib import Path

import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms

from model_jit import JiT_models
from util.crop import center_crop_arr


def parse_args():
    parser = argparse.ArgumentParser("Train a display-first JiT demo")
    parser.add_argument("--model", default="JiT-Tiny/16", type=str)
    parser.add_argument("--img_size", default=64, type=int)
    parser.add_argument("--class_num", default=4, type=int)
    parser.add_argument("--data_path", default="./toy_display_data", type=str)
    parser.add_argument("--output_dir", default="./output_display_demo", type=str)
    parser.add_argument("--epochs", default=80, type=int)
    parser.add_argument("--batch_size", default=16, type=int)
    parser.add_argument("--lr", default=3e-4, type=float)
    parser.add_argument("--num_workers", default=0, type=int)
    parser.add_argument("--noise_scale", default=1.0, type=float)
    parser.add_argument("--t_min", default=0.2, type=float)
    parser.add_argument("--t_max", default=0.7, type=float)
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--device", default="cuda", type=str)
    return parser.parse_args()


def save_checkpoint(model, args):
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "model": model.state_dict(),
        "args": vars(args),
    }
    torch.save(checkpoint, output_dir / "checkpoint-last.pth")
    with open(output_dir / "args.json", "w", encoding="ascii") as handle:
        json.dump(vars(args), handle, indent=2)


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    torch.set_float32_matmul_precision("high")

    transform = transforms.Compose([
        transforms.Lambda(lambda img: center_crop_arr(img, args.img_size)),
        transforms.PILToTensor(),
    ])
    dataset = datasets.ImageFolder(Path(args.data_path) / "train", transform=transform)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        drop_last=True,
    )

    device = torch.device(args.device)
    model = JiT_models[args.model](
        input_size=args.img_size,
        in_channels=3,
        num_classes=args.class_num,
        attn_drop=0.0,
        proj_drop=0.0,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=0.0)

    print(f"Training {args.model} for display demo")
    print(f"Samples: {len(dataset)}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6:.3f}M")

    for epoch in range(args.epochs):
        running_loss = 0.0
        for images, labels in loader:
            images = images.to(device, non_blocking=True).to(torch.float32).div_(255.0)
            images = images * 2.0 - 1.0
            labels = labels.to(device, non_blocking=True)

            t = torch.empty(images.size(0), device=device).uniform_(args.t_min, args.t_max)
            t_view = t.view(-1, 1, 1, 1)
            noise = torch.randn_like(images) * args.noise_scale
            noisy = t_view * images + (1.0 - t_view) * noise

            with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=args.device.startswith("cuda")):
                pred = model(noisy, t, labels)
                loss = torch.nn.functional.smooth_l1_loss(pred, images)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        avg_loss = running_loss / len(loader)
        print(f"epoch={epoch:03d} loss={avg_loss:.5f}")

    save_checkpoint(model, args)
    print(f"Saved checkpoint to {Path(args.output_dir).resolve()}")


if __name__ == "__main__":
    main()
