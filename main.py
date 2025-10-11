import torch
import torch.backends.cudnn as cudnn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import os
import time
import argparse
import json
import datetime
import numpy as np
import yaml
import random
from pathlib import Path
from loguru import logger


from optimizer import build_optimizer, build_scheduler
from dataset import ForgeryDataset
from net import FSDFormer
from opt import train_one_epoch, evaluate_fn
import utils


def get_args_parser():
    parser = argparse.ArgumentParser("Image Manipulation Localization", add_help=False)
    parser.add_argument("--batch-size", default=1, type=int)
    parser.add_argument("--epochs", default=800, type=int)

    parser.add_argument("--finetune", default="", help="finetune from checkpoint")
    parser.add_argument(
        "--device", default="cpu", help="device to use for training / testing"
    )
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--resume", default="", help="resume from checkpoint")
    parser.add_argument(
        "--start_epoch", default=0, type=int, metavar="N", help="start epoch"
    )
    parser.add_argument("--eval", action="store_true", help="Perform evaluation only")
    parser.add_argument("--num_workers", default=4, type=int)

    parser.add_argument(
        "--cfg",
        type=str,
        default="configs/casiav2.yaml",
        help="Path to config file",
    )

    parser.add_argument("--print_freq", default=1, type=int, help="print frequency")
    parser.add_argument("--save_all", action="store_true", help="save all images")
    parser.add_argument(
        "--eval_train", action="store_true", help="Evaluate in training"
    )

    return parser


def main(args, cfg):
    model_dir = cfg["training"]["model_dir"]
    log_file = os.path.join(
        model_dir, f"run_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.log"
    )
    # Setup distributed
    is_distributed, rank, local_rank, world_size = utils.setup_distributed()

    if is_distributed:
        device = torch.device(
            f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu"
        )
    else:
        device = torch.device(
            args.device
            if (
                args.device
                and (str(args.device).startswith("cuda") or args.device == "cpu")
            )
            else ("cuda" if torch.cuda.is_available() else "cpu")
        )

    seed = args.seed + utils.get_rank()
    # Set seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = False
    cfg_data = cfg["data"]

    try:
        model = FSDFormer(device=device, **cfg["model"])
        # Convert BN to SyncBatchNorm if using CUDA DDP for stabler stats
        model = model.to(device)

        # Wrap with DDP if needed
        if is_distributed:
            if device.type == "cuda":
                model = DDP(
                    model,
                    device_ids=[local_rank],
                    output_device=local_rank,
                    find_unused_parameters=True,
                )
            else:
                model = DDP(model, find_unused_parameters=True)

        model_for_params = model.module if is_distributed else model
        n_parameters = utils.count_model_parameters(model_for_params)
        if rank == 0:
            print(f"Model created successfully with {n_parameters:,} parameters")
    except Exception as e:
        print(f"Error creating model: {e}")
        print("Model config:", cfg["model"])
        raise

    if rank == 0:
        print(f"Number of parameters: {n_parameters}")

    if rank == 0:
        input_shape = (
            args.batch_size,
            3,
            cfg_data["image_size"],
            cfg_data["image_size"],
        )
        model_info = utils.get_model_info(model_for_params, input_shape, device)

        print("Model Information:")
        print(f"  Total parameters: {model_info['total_params']:,}")
        print(f"  Trainable parameters: {model_info['trainable_params']:,}")
        print(f"  Non-trainable parameters: {model_info['non_trainable_params']:,}")
        print(f"  FLOPs: {model_info['flops_str']}")
        print()

    # Create datasets
    train_data = ForgeryDataset(split="train", cfg=cfg_data)
    test_data = ForgeryDataset(split="test", cfg=cfg_data)

    # Samplers for DDP
    train_sampler = None
    test_sampler = None
    if is_distributed:
        train_sampler = DistributedSampler(train_data, shuffle=True)
        test_sampler = DistributedSampler(test_data, shuffle=False)

    train_dataloader = DataLoader(
        train_data,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        collate_fn=train_data.data_collator,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        pin_memory=True,
        drop_last=True,
    )

    test_dataloader = DataLoader(
        test_data,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        collate_fn=test_data.data_collator,
        sampler=test_sampler,
        pin_memory=True,
    )

    if args.finetune:
        if rank == 0:
            print(f"Finetuning from {args.finetune}")
        checkpoint = torch.load(args.finetune, map_location="cpu")
        target_model = model.module if is_distributed else model
        ret = target_model.load_state_dict(checkpoint["model_state_dict"], strict=False)
        if rank == 0:
            print("Missing keys: \n", "\n".join(ret.missing_keys))
            print("Unexpected keys: \n", "\n".join(ret.unexpected_keys))

    optimizer = build_optimizer(
        config=cfg["training"]["optimization"],
        model=(model.module if is_distributed else model),
    )

    for group in optimizer.param_groups:
        if "initial_lr" not in group:
            group["initial_lr"] = group["lr"]

    cfg["training"]["optimization"]["scheduler"]["T_max"] = args.epochs

    scheduler_last_epoch = -1
    if args.resume:
        if rank == 0:
            print(f"Resume training from {args.resume}")
        checkpoint = torch.load(args.resume, map_location="cpu")
        target_model = model.module if is_distributed else model
        ret = target_model.load_state_dict(checkpoint["model_state_dict"], strict=False)
        if "epoch" in checkpoint:
            scheduler_last_epoch = checkpoint["epoch"]
        args.start_epoch = checkpoint["epoch"] + 1
        if rank == 0:
            print("Missing keys: \n", "\n".join(ret.missing_keys))
            print("Unexpected keys: \n", "\n".join(ret.unexpected_keys))

    scheduler, scheduler_type = build_scheduler(
        config=cfg["training"]["optimization"],
        optimizer=optimizer,
        last_epoch=scheduler_last_epoch,
    )

    if args.resume:
        if (
            not args.eval
            and "optimizer_state_dict" in checkpoint
            and "scheduler_state_dict" in checkpoint
        ):
            if rank == 0:
                print("Loading optimizer and scheduler")
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            if hasattr(scheduler, "load_state_dict"):
                scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
            if rank == 0:
                print(f"New learning rate : {scheduler.get_last_lr()[0]}")

    args.output_dir = model_dir
    args.save_images = cfg["evaluation"]["save_images"]
    args.device = device

    output_dir = Path(cfg["training"]["model_dir"])

    if args.eval:
        if not args.resume:
            logger.warning(
                "Please specify the trained model: --resume /path/to/best_checkpoint.pth"
            )

        test_results = evaluate_fn(
            args,
            test_dataloader,
            model,
            epoch=0,
            print_freq=args.print_freq,
            log_file=log_file,
        )
        if rank == 0:
            print(
                f"Test loss of the network on the {len(test_dataloader)} test images: {test_results['iou']:.3f} iou"
            )
        return

    if rank == 0:
        print(f"Training on {device}")
        print(
            f"Start training for {args.epochs} epochs and start epoch: {args.start_epoch}"
        )
    start_time = time.time()
    best_iou = 0.0

    for epoch in range(args.start_epoch, args.epochs):
        if is_distributed and train_sampler is not None:
            train_sampler.set_epoch(epoch)
        train_results = train_one_epoch(
            args,
            model,
            train_dataloader,
            optimizer,
            epoch,
            print_freq=args.print_freq,
            log_file=log_file,
            eval_train=args.eval_train,
        )
        scheduler.step()

        # Save checkpoint
        checkpoint_paths = [output_dir / f"checkpoint_{epoch}.pth"]
        prev_chkpt = output_dir / f"checkpoint_{epoch - 1}.pth"
        if rank == 0 and os.path.exists(prev_chkpt):
            os.remove(prev_chkpt)
        for checkpoint_path in checkpoint_paths:
            utils.save_on_master(
                {
                    "model_state_dict": (
                        model.module.state_dict()
                        if is_distributed
                        else model.state_dict()
                    ),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "epoch": epoch,
                },
                checkpoint_path,
            )
        if rank == 0:
            print()

        if not args.eval_train:
            test_results = evaluate_fn(
                args,
                test_dataloader,
                model,
                epoch,
                print_freq=args.print_freq,
                log_file=log_file,
            )
        else:
            test_results = train_results

        if test_results["iou"] > best_iou:
            best_iou = test_results["iou"]
            checkpoint_paths = [output_dir / "best_checkpoint.pth"]
            for checkpoint_path in checkpoint_paths:
                utils.save_on_master(
                    {
                        "model_state_dict": (
                            model.module.state_dict()
                            if is_distributed
                            else model.state_dict()
                        ),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "scheduler_state_dict": scheduler.state_dict(),
                        "epoch": epoch,
                    },
                    checkpoint_path,
                )
        if rank == 0:
            print(f"* TEST iou {test_results['iou']:.3f} Best iou {best_iou:.3f}")

        # Log results
        log_results = {
            **{f"train_{k}": v for k, v in train_results.items()},
            **{f"test_{k}": v for k, v in test_results.items()},
            "epoch": epoch,
            "n_parameters": n_parameters,
        }
        if rank == 0:
            print()
        with (output_dir / "log.txt").open("a") as f:
            f.write(json.dumps(log_results) + "\n")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    if rank == 0:
        print("Training time {}".format(total_time_str))
    utils.cleanup_distributed()


if __name__ == "__main__":
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    parser = argparse.ArgumentParser("Image DeBluring", parents=[get_args_parser()])
    args = parser.parse_args()

    with open(args.cfg, "r+", encoding="utf-8") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    Path(config["training"]["model_dir"]).mkdir(parents=True, exist_ok=True)
    main(args, config)
