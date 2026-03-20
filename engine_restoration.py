import math
import sys

import torch

import util.lr_sched as lr_sched
import util.misc as misc
from util.amp import get_cuda_autocast_kwargs


def train_one_epoch_restoration(model, model_without_ddp, data_loader, optimizer, device, epoch, log_writer=None, args=None):
    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", misc.SmoothedValue(window_size=1, fmt="{value:.6f}"))
    header = f"Epoch: [{epoch}]"
    print_freq = 20

    if log_writer is not None:
        print(f"log_dir: {log_writer.log_dir}")

    autocast_kwargs = get_cuda_autocast_kwargs(device)
    for data_iter_step, (lq, hq) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)

        lq = lq.to(device, non_blocking=True)
        hq = hq.to(device, non_blocking=True)

        with torch.amp.autocast(**autocast_kwargs):
            loss = model(hq, lq)

        loss_value = loss.item()
        if not math.isfinite(loss_value):
            print(f"Loss is {loss_value}, stopping training")
            sys.exit(1)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        torch.cuda.synchronize()
        model_without_ddp.update_ema()

        metric_logger.update(loss=loss_value)
        lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(lr=lr)
        loss_value_reduce = misc.all_reduce_mean(loss_value)

        if log_writer is not None:
            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
            if data_iter_step % args.log_freq == 0:
                log_writer.add_scalar("train_loss", loss_value_reduce, epoch_1000x)
                log_writer.add_scalar("lr", lr, epoch_1000x)
