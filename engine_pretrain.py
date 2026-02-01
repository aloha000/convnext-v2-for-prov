# Copyright (c) Meta Platforms, Inc. and affiliates.

# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import os
import math
import sys
from typing import Iterable

import torch
import torch.nn.functional as F
import utils
from utils import visualize_loss_patches

def train_one_epoch(model: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler,
                    log_writer=None,
                    args=None):
    model.train(True)
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 20

    update_freq = args.update_freq

    optimizer.zero_grad()
    for data_iter_step, (samples, labels) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
         # we use a per iteration (instead of per epoch) lr scheduler
        if data_iter_step % update_freq == 0:
           utils.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)
        if not isinstance(samples, list):
            samples = samples.to(device, non_blocking=True)
        # @qinluozheng: add interplotion for alignment with original FCMAE
        # samples = F.interpolate(samples, size=56, mode='bilinear', align_corners=False)
        
        labels = labels.to(device, non_blocking=True)

        loss, _, _ = model(samples, labels, mask_ratio=args.mask_ratio)

        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)
    
        loss /= update_freq
        loss_scaler(loss, optimizer, parameters=model.parameters(),
                    update_grad=(data_iter_step + 1) % update_freq == 0)
        if (data_iter_step + 1) % update_freq == 0:
            optimizer.zero_grad()
            torch.cuda.empty_cache() # clear the GPU cache at a regular interval for training ME network
        
        metric_logger.update(loss=loss_value)

        lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(lr=lr)
        
        loss_value_reduce = utils.all_reduce_mean(loss_value)
        if log_writer is not None and (data_iter_step + 1) % update_freq == 0:
            """ We use epoch_1000x as the x-axis in tensorboard.
            This calibrates different curves when batch size changes.
            """
            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
            log_writer.update(train_loss=loss_value_reduce, head="loss", step=epoch_1000x)
            log_writer.update(lr=lr, head="opt", step=epoch_1000x)
    
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


def train_one_epoch_all_farm(model: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler,
                    log_writer=None,
                    args=None):
    model.train(True)
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 20

    update_freq = args.update_freq

    optimizer.zero_grad()
    for data_iter_step, batch_data in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
         # we use a per iteration (instead of per epoch) lr scheduler
        samples = batch_data["nwp"]
        labels = torch.zeros(5)

        if data_iter_step % update_freq == 0:
           utils.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)
        if not isinstance(samples, list):
            samples = samples.to(device, non_blocking=True)
        # @qinluozheng: add interplotion for alignment with original FCMAE
        # samples = F.interpolate(samples, size=128, mode='bilinear', align_corners=False)
        labels = labels.to(device, non_blocking=True)
        loss, _, _ = model(samples, labels, mask_ratio=args.mask_ratio)

        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)
    
        loss /= update_freq
        loss_scaler(loss, optimizer, parameters=model.parameters(),
                    update_grad=(data_iter_step + 1) % update_freq == 0)
        if (data_iter_step + 1) % update_freq == 0:
            optimizer.zero_grad()
            torch.cuda.empty_cache() # clear the GPU cache at a regular interval for training ME network
        
        metric_logger.update(loss=loss_value)

        lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(lr=lr)
        
        loss_value_reduce = utils.all_reduce_mean(loss_value)
        if log_writer is not None and (data_iter_step + 1) % update_freq == 0:
            """ We use epoch_1000x as the x-axis in tensorboard.
            This calibrates different curves when batch size changes.
            """
            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
            log_writer.update(train_loss=loss_value_reduce, head="loss", step=epoch_1000x)
            log_writer.update(lr=lr, head="opt", step=epoch_1000x)
            
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate_one_epoch_all_farm(model: torch.nn.Module, 
                       data_loader: Iterable,
                       device: torch.device,
                       epoch: int, log_writer=None, save_visualize=False, args=None):
    """
    Pure evaluation: no LR scheduling, no optimizer steps, no loss scaling.
    Returns averaged metrics from MetricLogger.
    """
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = f"Eval @ Epoch: [{epoch}]"
    print_freq = 20  # define locally for eval

    # track loss explicitly
    # (MetricLogger will create meters on first update, but we can be explicit)
    # metric_logger.add_meter('loss', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))

    for data_iter_step, batch_data in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        # unpack
        samples = batch_data["nwp"]
        labels = torch.zeros(5)  # keep same stub shape as training for compatibility

        # device move
        if not isinstance(samples, list):
            samples = samples.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        # align with original FCMAE
        # samples = F.interpolate(samples, size=128, mode='bilinear', align_corners=False)

        # forward only
        if save_visualize:
            loss, _, _, loss_per_patch = model(samples, labels, mask_ratio=args.mask_ratio, return_per_patch_loss=True)
            visualization_path = os.path.join(args.output_dir, "visualized")
            # os.makedirs(visualization_path, exist_ok=True) NOTE:explicitly ensured in utils.visualize_loss_patches
            visualize_loss_patches(loss_per_patch, data_iter_step, visualization_path)
        else:
            loss, _, _ = model(samples, labels, mask_ratio=args.mask_ratio)
        
        

        loss_value = loss.item()
        if not math.isfinite(loss_value):
            print(f"Eval loss is {loss_value}, aborting.")
            sys.exit(1)

        # update meters
        metric_logger.update(loss=loss_value)

        # cross-process reduce for logging (avg across ranks)
        loss_value_reduce = utils.all_reduce_mean(loss_value)

        if log_writer is not None:
            # keep eval on the same 1000x axis convention
            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
            log_writer.update(val_loss=loss_value_reduce, head="loss", step=epoch_1000x)
            
    # sync across processes & print summary
    metric_logger.synchronize_between_processes()
    print("Eval averaged stats:", metric_logger)

    # return dict of global averages
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
