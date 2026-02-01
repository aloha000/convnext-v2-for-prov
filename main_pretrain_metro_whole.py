# train_fcmae_ddp.py

# Copyright (c) Meta Platforms
# (license as in original)

import argparse
import datetime
import numpy as np
import time
import json
import os
from pathlib import Path

import torch
import torch.backends.cudnn as cudnn

import timm
import timm.optim.optim_factory as optim_factory

from engine_pretrain import train_one_epoch_all_farm, evaluate_one_epoch_all_farm
import models.fcmae as fcmae

import utils
from utils import NativeScalerWithGradNormCount as NativeScaler
from utils import str2bool

from dataloader_ss import create_dataloader


def get_args_parser():
    parser = argparse.ArgumentParser('FCMAE pre-training', add_help=False)
    parser.add_argument('--batch_size', default=32, type=int,
                        help='Per GPU batch size')
    parser.add_argument('--patch_size', default=4, type=int,
                        help='patch size')
    parser.add_argument('--epochs', default=100, type=int)
    parser.add_argument('--warmup_epochs', type=int, default=40, metavar='N',
                        help='epochs to warmup LR')
    parser.add_argument('--update_freq', default=1, type=int,
                        help='gradient accumulation step')

    # Model
    parser.add_argument('--model', default='convnextv2_base', type=str, metavar='MODEL')
    parser.add_argument('--input_size', default="60,92", type=str)
    parser.add_argument('--mask_ratio', default=0.6, type=float)
    parser.add_argument('--norm_pix_loss', action='store_true')
    parser.set_defaults(norm_pix_loss=True)
    parser.add_argument('--decoder_depth', type=int, default=1)
    parser.add_argument('--decoder_embed_dim', type=int, default=512)

    # Optimizer
    parser.add_argument('--weight_decay', type=float, default=0.05)
    parser.add_argument('--lr', type=float, default=None, metavar='LR',
                        help='absolute lr; if None uses blr scaling rule')
    parser.add_argument('--blr', type=float, default=1.5e-4, metavar='LR',
                        help='base lr: absolute_lr = blr * total_batch / 256')
    parser.add_argument('--min_lr', type=float, default=0.)

    # Dataset
    parser.add_argument('--output_dir', default='')
    parser.add_argument('--log_dir', default=None)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--resume', default='')
    parser.add_argument('--auto_resume', type=str2bool, default=True)
    parser.add_argument('--save_ckpt', type=str2bool, default=True)
    parser.add_argument('--save_ckpt_freq', default=1, type=int)
    parser.add_argument('--save_ckpt_num', default=-1, type=int)
    parser.add_argument('--start_epoch', default=0, type=int)
    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument('--pin_mem', type=str2bool, default=True)

    # distributed
    parser.add_argument('--world_size', default=1, type=int)
    parser.add_argument('--local-rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', type=str2bool, default=False)
    parser.add_argument('--dist_url', default='env://')

    # metro configs
    parser.add_argument('--farm_id', default="1001")
    parser.add_argument('--farm_type', default='solar')  # solar|wind|all
    parser.add_argument("--metro_feat_channel", default=3, type=int)
    parser.add_argument("--metro_fcmae_dims", default="3,6,12,24")

    parser.add_argument("--data_dir", default="/inspire/ssd/project/sais-mtm/public/qlz/linan/dl/ConvNeXt-V2/self_supervised/processed_wind_1.2")
    parser.add_argument("--data_statistic_dir", default="/inspire/ssd/project/sais-mtm/public/qlz/linan/dl/ConvNeXt-V2/Guangdong_spatial_mean_std_fast_1.2.npz")

    # evaluation
    parser.add_argument('--eval_freq', type=int, default=1, help='evaluate every N epochs (0 to disable)')
    parser.add_argument('--evaluate_and_visualize', type=str2bool, default=False, help='if do evaluate only.')
    # parser.add_argument('--eval_mask_ratio', type=float, default=None, help='override mask ratio during eval (defaults to train mask_ratio)')
    # NOTE:this param is set but not used, eval_mask_ratio always equal to mask_ratio
    return parser


def parse_fcamae_dims(fcmae_dims_str):
    return [int(x.strip()) for x in fcmae_dims_str.split(',') if x.strip() != ""]




def main(args):
    utils.init_distributed_mode(args)
    device = torch.device(args.device)

    # perf knobs
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    try:
        torch.set_float32_matmul_precision('high')
    except Exception:
        pass

    print(args)

    # seed (offset by rank for DDP)
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)

    # --------- Dataloaders (DDP-friendly)
    train_loader, test_loader, train_sampler, test_sampler = create_dataloader(
        dataset_type=args.farm_type,
        data_dir=args.data_dir,
        data_stat_dir=args.data_statistic_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        distributed=args.distributed,
        pin_memory=args.pin_mem,
        seed=args.seed + utils.get_rank(),
        patch_size=args.patch_size
    )



    # --------- Logger
    log_writer = utils.TensorboardLogger(log_dir=args.log_dir)

    # --------- Model
    H, W = map(int, args.input_size.split(","))
    input_size = (H, W)
    # sample_batch = next(iter(train_loader))
    # real_H = sample_batch["nwp"].shape[-2]
    # real_W = sample_batch["nwp"].shape[-1]
    # print("Real input size:", real_H, real_W)
    # input_size = (real_H, real_W)
    
    dims = parse_fcamae_dims(args.metro_fcmae_dims)
    model = fcmae.FCMAE(
        img_size=input_size,
        in_chans=args.metro_feat_channel,
        depths=[2, 2, 6, 2],
        dims=dims,
        mask_ratio=args.mask_ratio,
        patch_size=args.patch_size,
        decoder_depth=args.decoder_depth,
        decoder_embed_dim=args.decoder_embed_dim,
        norm_pix_loss=args.norm_pix_loss
    ).to(device)

    model_without_ddp = model
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Model = %s" % str(model_without_ddp))
    print('number of params:', n_parameters)

    eff_batch_size = args.batch_size * args.update_freq * utils.get_world_size()
    num_training_steps_per_epoch = len(train_loader)

    if args.lr is None:
        args.lr = args.blr * eff_batch_size / 16.0
    print("base lr: %.2e" % (args.lr * 16.0 / eff_batch_size))
    print("actual lr: %.2e" % args.lr)
    print("accumulate grad iterations: %d" % args.update_freq)
    print("effective batch size: %d" % eff_batch_size)

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.gpu], find_unused_parameters=False
        )
        model_without_ddp = model.module

    # --------- Optimizer & scaler
    param_groups = optim_factory.add_weight_decay(model_without_ddp, args.weight_decay)
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr, betas=(0.9, 0.95))
    loss_scaler = NativeScaler()
    print(optimizer)

    # --------- (Optional) Resume
    utils.auto_load_model(
        args=args, model=model, model_without_ddp=model_without_ddp,
        optimizer=optimizer, loss_scaler=loss_scaler)

    # --------- Train
    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    for epoch in range(0, args.epochs):
        if args.evaluate_and_visualize:
            eval_stats = {}
            eval_stats = evaluate_one_epoch_all_farm(model, test_loader, device, epoch, save_visualize=True, log_writer=log_writer, args=args)
            print("Evaluation Finished")
            break
        if args.distributed:
            # important! makes per-epoch shuffling distinct across ranks
            train_sampler.set_epoch(epoch)

        if log_writer is not None:
            log_writer.set_step(epoch * num_training_steps_per_epoch * args.update_freq)
        print(epoch)
        train_stats = train_one_epoch_all_farm(
            model, train_loader, optimizer, device, epoch, loss_scaler,
            log_writer=log_writer, args=args
        )

        # save ckpt
        if args.output_dir and args.save_ckpt:
            if (epoch + 1) % args.save_ckpt_freq == 0 or (epoch + 1) == args.epochs:
                utils.save_model(
                    args=args, model=model, model_without_ddp=model_without_ddp,
                    optimizer=optimizer, loss_scaler=loss_scaler, epoch=epoch
                )

        # evaluate
        do_eval = (args.eval_freq > 0) and ((epoch + 1) % args.eval_freq == 0 or (epoch + 1) == args.epochs)
        eval_stats = {}
        if do_eval:
            eval_stats = evaluate_one_epoch_all_farm(model, test_loader, device, epoch, log_writer=log_writer, args=args)

        # logging (only main process)
        log_stats = {
            **{f'train_{k}': v for k, v in train_stats.items()},
            **eval_stats,
            'epoch': epoch,
            'n_parameters': n_parameters
        }
        if args.output_dir and utils.is_main_process():
            if log_writer is not None:
                # also push val to TB if available
                log_writer.flush()
            with open(os.path.join(args.output_dir, "log.txt"), "a", encoding="utf-8") as f:
                f.write(json.dumps(log_stats) + "\n")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == '__main__':
    
    args = get_args_parser()
    args = args.parse_args()
    # if args.debug:
    #     import debugpy
    #     debugpy.listen(("0.0.0.0", 5678))
    #     debugpy.wait_for_client()

    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
