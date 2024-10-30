import math
import time
import os
import datetime
import sys
from typing import Iterable
import torch
import numpy as np
from timm.utils import accuracy

import utils


def get_loss_scale_for_deepspeed(model):
    optimizer = model.optimizer
    return optimizer.loss_scale if hasattr(optimizer, "loss_scale") else optimizer.cur_scale


def train_one_epoch(model: torch.nn.Module, args,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, start_steps=None, 
                    lr_schedule_values=None, wd_schedule_values=None,
                    num_training_steps_per_epoch=None):
    model.train(True)
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('min_lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10

    model.zero_grad()
    model.micro_steps = 0

    criterion =torch.nn.CrossEntropyLoss()

    for step, data in enumerate(metric_logger.log_every(data_loader, print_freq, header)):      
        if step >= num_training_steps_per_epoch:
            continue
        it = start_steps + step  # global training iteration
        # Update LR & WD for the first acc
        if lr_schedule_values is not None or wd_schedule_values is not None:
        # if lr_schedule_values is not None or wd_schedule_values is not None and data_iter_step % update_freq == 0:
            for i, param_group in enumerate(optimizer.param_groups):
                if lr_schedule_values is not None:
                    param_group["lr"] = lr_schedule_values[it] * param_group["lr_scale"]
                if wd_schedule_values is not None and param_group["weight_decay"] > 0:
                    param_group["weight_decay"] = wd_schedule_values[it]
       
        samples = data[0].to(device, non_blocking=True)
        samples = samples.unsqueeze(1).permute(0,1,4,2,3)
        targets = data[1].to(device, non_blocking=True)
        samples = samples.half()

        outputs = model(samples)
        loss = criterion(outputs, targets)
        loss_value = loss.item()
        outputs = outputs.detach()
        class_acc = (outputs.max(-1)[-1] == targets).float().mean()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        model.backward(loss)
        model.step()
        grad_norm = None
        loss_scale_value = get_loss_scale_for_deepspeed(model)

        torch.cuda.synchronize()

        metric_logger.update(loss=loss_value)
        metric_logger.update(class_acc=class_acc)
        metric_logger.update(loss_scale=loss_scale_value)
        min_lr = 10.
        max_lr = 0.
        for group in optimizer.param_groups:
            min_lr = min(min_lr, group["lr"])
            max_lr = max(max_lr, group["lr"])

        metric_logger.update(lr=max_lr)
        metric_logger.update(min_lr=min_lr)
        weight_decay_value = None
        for group in optimizer.param_groups:
            if group["weight_decay"] > 0:
                weight_decay_value = group["weight_decay"]
        metric_logger.update(weight_decay=weight_decay_value)
        metric_logger.update(grad_norm=grad_norm)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def validation_one_epoch(data_loader, model, device, args):
    criterion = torch.nn.CrossEntropyLoss()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Val:'

    # switch to evaluation mode
    model.eval()

    for step, data in enumerate(metric_logger.log_every(data_loader, 10, header)):
        samples = data[0]
        samples = samples.unsqueeze(1).permute(0,1,4,2,3)
        targets = data[1]
        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        samples = samples.half()

        outputs = model(samples)
        loss = criterion(outputs, targets)
        acc1, acc5 = accuracy(outputs, targets, topk=(1, 5))

        batch_size = samples.shape[0]
        metric_logger.update(loss=loss.item())
        metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
        metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print('* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} loss {losses.global_avg:.3f}'
          .format(top1=metric_logger.acc1, top5=metric_logger.acc5, losses=metric_logger.loss))

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}