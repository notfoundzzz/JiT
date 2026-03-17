import argparse
from pathlib import Path
from types import SimpleNamespace

import torch
from PIL import Image, ImageDraw

from denoiser import Denoiser
from util.amp import get_cuda_autocast_kwargs


def parse_args():
    parser = argparse.ArgumentParser("Visualize how different noise seeds affect pretrained JiT generation")
    parser.add_argument(
        "--checkpoint",
        default="/home/zhahl/JiT-cloud-smoke/JiT-l-32/checkpoint-last.pth",
        type=str,
    )
    parser.add_argument("--output_dir", default="./generated_pretrained/noise_triptych_l32", type=str)
    parser.add_argument("--model", default="JiT-L/32", type=str)
    parser.add_argument("--img_size", default=512, type=int)
    parser.add_argument("--class_num", default=1000, type=int)
    parser.add_argument("--noise_scale", default=2.0, type=float)
    parser.add_argument("--sampling_method", default="heun", type=str)
    parser.add_argument("--num_sampling_steps", default=50, type=int)
    parser.add_argument("--cfg", default=2.5, type=float)
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
    parser.add_argument("--label", default=0, type=int)
    parser.add_argument("--seeds", default="0,1,2", type=str)
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


def to_uint8_image(tensor):
    tensor = tensor.clamp(0.0, 1.0)
    return Image.fromarray(tensor.mul(255).byte().permute(1, 2, 0).cpu().numpy())


def noise_to_image(noise, noise_scale):
    # Map approximately [-2 * noise_scale, 2 * noise_scale] into [0, 1] for display.
    vis = (noise / (4.0 * noise_scale) + 0.5).clamp(0.0, 1.0)
    return to_uint8_image(vis)


def sample_from_noise(model, labels, z0):
    z = z0.clone()
    timesteps = torch.linspace(
        0.0, 1.0, model.steps + 1, device=z.device
    ).view(-1, *([1] * z.ndim)).expand(-1, z.size(0), -1, -1, -1)

    if model.method == "euler":
        stepper = model._euler_step
    elif model.method == "heun":
        stepper = model._heun_step
    else:
        raise NotImplementedError

    for i in range(model.steps - 1):
        z = stepper(z, timesteps[i], timesteps[i + 1], labels)
    z = model._euler_step(z, timesteps[-2], timesteps[-1], labels)
    return z


def build_triptych(noise_img, gen_img, seed, label):
    width, height = noise_img.size
    text_h = 28
    canvas = Image.new("RGB", (width * 3, height + text_h), "white")
    slots = [
        ("noise", noise_img),
        ("generated", gen_img),
        (f"seed={seed}, label={label}", gen_img),
    ]
    draw = ImageDraw.Draw(canvas)
    for idx, (title, img) in enumerate(slots):
        x = idx * width
        canvas.paste(img, (x, 0))
        draw.text((x + 8, height + 8), title, fill="black")
    return canvas


def main():
    args = parse_args()

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available")

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    state_dict = checkpoint[args.ema_key]

    model = Denoiser(build_denoiser_args(args))
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    model.to(args.device)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    seed_values = [int(item.strip()) for item in args.seeds.split(",") if item.strip()]
    label_tensor = torch.tensor([args.label], dtype=torch.long, device=args.device)
    autocast_kwargs = get_cuda_autocast_kwargs(args.device)

    for seed in seed_values:
        torch.manual_seed(seed)
        z0 = args.noise_scale * torch.randn(1, 3, args.img_size, args.img_size, device=args.device)
        with torch.no_grad():
            with torch.amp.autocast(**autocast_kwargs):
                sample = sample_from_noise(model, label_tensor, z0)[0]

        noise_img = noise_to_image(z0[0].float().cpu(), args.noise_scale)
        gen_img = to_uint8_image(((sample.float().cpu() + 1.0) / 2.0).clamp(0.0, 1.0))
        triptych = build_triptych(noise_img, gen_img, seed, args.label)

        noise_img.save(output_dir / f"label_{args.label:03d}_seed_{seed}_noise.png")
        gen_img.save(output_dir / f"label_{args.label:03d}_seed_{seed}_generated.png")
        triptych.save(output_dir / f"label_{args.label:03d}_seed_{seed}_triptych.png")
        print(f"saved seed={seed} to {output_dir.resolve()}")


if __name__ == "__main__":
    main()
