import argparse
import copy
import datetime
import os
import time
import warnings
from pathlib import Path

import numpy as np
import torch
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter

import util.misc as misc
from denoiser_restoration_lqmain import RestorationDenoiserLQMain
from engine_restoration import train_one_epoch_restoration
from paired_image_dataset import PairedImageDataset
from util.amp import get_cuda_autocast_kwargs


def get_args_parser():
    parser = argparse.ArgumentParser("JiT restoration LQ-main", add_help=False)
    parser.add_argument("--model", default="JiT-L/32", type=str)
    parser.add_argument("--img_size", default=256, type=int)
    parser.add_argument("--attn_dropout", default=0.0, type=float)
    parser.add_argument("--proj_dropout", default=0.0, type=float)
    parser.add_argument("--epochs", default=50, type=int)
    parser.add_argument("--warmup_epochs", default=2, type=int)
    parser.add_argument("--batch_size", default=8, type=int)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--blr", type=float, default=1e-4)
    parser.add_argument("--min_lr", type=float, default=0.0)
    parser.add_argument("--lr_schedule", type=str, default="constant")
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--ema_decay1", type=float, default=0.9999)
    parser.add_argument("--ema_decay2", type=float, default=0.9996)
    parser.add_argument("--P_mean", default=-0.8, type=float)
    parser.add_argument("--P_std", default=0.8, type=float)
    parser.add_argument("--noise_scale", default=1.0, type=float)
    parser.add_argument("--t_eps", default=5e-2, type=float)
    parser.add_argument("--recon_weight", default=1.0, type=float)
    parser.add_argument("--lora_rank", default=8, type=int)
    parser.add_argument("--lora_alpha", default=16.0, type=float)
    parser.add_argument("--lora_dropout", default=0.0, type=float)
    parser.add_argument("--sampling_method", default="heun", type=str)
    parser.add_argument("--num_sampling_steps", default=1, type=int)
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--start_epoch", default=0, type=int)
    parser.add_argument("--num_workers", default=4, type=int)
    parser.add_argument("--pin_mem", action="store_true")
    parser.add_argument("--no_pin_mem", action="store_false", dest="pin_mem")
    parser.set_defaults(pin_mem=True)
    parser.add_argument("--disable_amp", action="store_true")
    parser.add_argument("--data_path", required=True, type=str)
    parser.add_argument("--qwen_model_path", required=True, type=str)
    parser.add_argument("--output_dir", default="./output_restoration_lqmain", type=str)
    parser.add_argument("--resume", default="", type=str)
    parser.add_argument("--pretrained_checkpoint", default="", type=str)
    parser.add_argument("--ema_key", default="model_ema1", choices=["model", "model_ema1", "model_ema2"])
    parser.add_argument("--save_last_freq", type=int, default=5)
    parser.add_argument("--log_freq", default=100, type=int)
    parser.add_argument("--device", default="cuda", type=str)
    parser.add_argument("--world_size", default=1, type=int)
    parser.add_argument("--local_rank", default=-1, type=int)
    parser.add_argument("--dist_on_itp", action="store_true")
    parser.add_argument("--dist_url", default="env://", type=str)
    return parser


def maybe_load_pretrained(model_without_ddp, args):
    if not args.pretrained_checkpoint:
        return
    checkpoint = torch.load(args.pretrained_checkpoint, map_location="cpu", weights_only=True)
    state_dict = checkpoint[args.ema_key]
    model_state = model_without_ddp.state_dict()
    filtered_state = {}
    skipped = []
    for key, value in state_dict.items():
        if key not in model_state:
            skipped.append((key, "missing in target model"))
            continue
        if model_state[key].shape != value.shape:
            skipped.append((key, f"shape {tuple(value.shape)} -> {tuple(model_state[key].shape)}"))
            continue
        filtered_state[key] = value

    missing, unexpected = model_without_ddp.load_state_dict(filtered_state, strict=False)
    print("Loaded pretrained checkpoint:", args.pretrained_checkpoint)
    print("Missing keys:", len(missing))
    print("Unexpected keys:", len(unexpected))
    if skipped:
        print("Skipped pretrained keys:", len(skipped))
        for key, reason in skipped[:20]:
            print(f"  {key}: {reason}")


def main(args):
    warnings.filterwarnings(
        "ignore",
        message=".*does not support bfloat16 compilation natively, skipping.*",
        category=UserWarning,
    )
    misc.init_distributed_mode(args)
    print("Job directory:", os.path.dirname(os.path.realpath(__file__)))
    print("Arguments:\n{}".format(args).replace(", ", ",\n"))

    device = torch.device(args.device)
    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    cudnn.benchmark = True

    num_tasks = misc.get_world_size()
    global_rank = misc.get_rank()
    if global_rank == 0 and args.output_dir is not None:
        os.makedirs(args.output_dir, exist_ok=True)
        log_writer = SummaryWriter(log_dir=args.output_dir)
    else:
        log_writer = None

    dataset_train = PairedImageDataset(args.data_path, args.img_size)
    print("Dataset:", dataset_train.__class__.__name__)
    print("Training samples:", len(dataset_train))
    print("Qwen model path:", args.qwen_model_path)

    sampler_train = torch.utils.data.DistributedSampler(
        dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
    )

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train,
        sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
    )

    model = RestorationDenoiserLQMain(args)
    maybe_load_pretrained(model, args)
    model.to(device)
    if args.disable_amp:
        print("AMP dtype: disabled")
    else:
        amp_dtype = get_cuda_autocast_kwargs(device).get("dtype")
        print("AMP dtype:", str(amp_dtype).replace("torch.", "") if amp_dtype is not None else "disabled")
    print("Recon weight:", args.recon_weight)
    print("LoRA rank:", args.lora_rank)
    print("LoRA alpha:", args.lora_alpha)
    print("LoRA dropout:", args.lora_dropout)

    eff_batch_size = args.batch_size * misc.get_world_size()
    if args.lr is None:
        args.lr = args.blr * eff_batch_size / 256
    print("Actual lr:", args.lr)

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[args.gpu],
            find_unused_parameters=True,
        )
        model_without_ddp = model.module
    else:
        model_without_ddp = model

    param_groups = misc.add_weight_decay(model_without_ddp, args.weight_decay)
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr, betas=(0.9, 0.95))

    checkpoint_path = os.path.join(args.resume, "checkpoint-last.pth") if args.resume else None
    if checkpoint_path and os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        model_without_ddp.load_state_dict(checkpoint["model"])
        ema_state_dict1 = checkpoint["model_ema1"]
        ema_state_dict2 = checkpoint["model_ema2"]
        model_without_ddp.ema_params1 = [ema_state_dict1[name].cuda() for name, _ in model_without_ddp.named_parameters()]
        model_without_ddp.ema_params2 = [ema_state_dict2[name].cuda() for name, _ in model_without_ddp.named_parameters()]
        if "optimizer" in checkpoint and "epoch" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer"])
            args.start_epoch = checkpoint["epoch"] + 1
        print("Resumed checkpoint from", args.resume)
    else:
        model_without_ddp.ema_params1 = copy.deepcopy(list(model_without_ddp.parameters()))
        model_without_ddp.ema_params2 = copy.deepcopy(list(model_without_ddp.parameters()))
        print("Training from scratch")
    if model_without_ddp.lora_modules:
        print("LoRA modules:", len(model_without_ddp.lora_modules))
        for name in model_without_ddp.lora_modules[:20]:
            print("  ", name)
    else:
        print("LoRA modules: 0")

    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)
        train_one_epoch_restoration(model, model_without_ddp, data_loader_train, optimizer, device, epoch, log_writer=log_writer, args=args)
        if epoch % args.save_last_freq == 0 or epoch + 1 == args.epochs:
            misc.save_model(args=args, model_without_ddp=model_without_ddp, optimizer=optimizer, epoch=epoch, epoch_name="last")
        if misc.is_main_process() and log_writer is not None:
            log_writer.flush()

    total_time = time.time() - start_time
    print("Training time:", str(datetime.timedelta(seconds=int(total_time))))
    if args.distributed:
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    args = get_args_parser().parse_args()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
