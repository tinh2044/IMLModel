import argparse
import os
from pathlib import Path
from typing import List, Tuple, Dict

import torch
import torch.utils.data as data
import torchvision.transforms.functional as TF
from PIL import Image
import yaml

from loguru import logger

from net import ANRTNet
from metrics import compute_metrics
from logger import MetricLogger, SmoothedValue
import utils


def _pair_by_stem(
    raw_dir, mask_dir, orig_dir=None, exts=None
):
    if exts is None:
        exts = [".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"]
    raw_index = {}
    orig_index = None
    for p in raw_dir.iterdir():
        if p.suffix.lower() in exts and p.is_file():
            raw_index[p.stem.replace("_gt", "")] = p

    mask_index = {}
    for p in mask_dir.iterdir():
        if p.suffix.lower() in exts and p.is_file():
            mask_index[p.stem.replace("_gt", "")] = p

    if orig_dir is not None:
        orig_index = {}
        for p in orig_dir.iterdir():
            if p.suffix.lower() in exts and p.is_file():
                orig_index[p.stem.replace("_gt", "")] = p

    keys = sorted(set(raw_index.keys()) & set(mask_index.keys()))
    return [
        (
            raw_index[k],
            mask_index[k],
            orig_index[k.split("_")[-1]] if orig_index is not None else None,
        )
        for k in keys
    ]


class PairFolderDataset(data.Dataset):
    def __init__(
        self,
        raw_dir,
        mask_dir,
        orig_dir = None,
        image_size = 320,
        center_crop = False,
    ):
        self.raw_dir = Path(raw_dir)
        self.mask_dir = Path(mask_dir)
        self.orig_dir = Path(orig_dir) if orig_dir is not None else None
        if not self.raw_dir.is_dir() or not self.mask_dir.is_dir():
            raise ValueError("raw_dir and mask_dir must be valid directories")

        self.image_size = int(image_size)
        self.center_crop = bool(center_crop)
        self.samples = _pair_by_stem(self.raw_dir, self.mask_dir, self.orig_dir)
        logger.info(
            f"PairFolderDataset: {len(self.samples)} pairs from {self.raw_dir} and {self.mask_dir}"
        )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, mask_path, orig_path = self.samples[idx]
        img = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")
        orig = Image.open(orig_path).convert("RGB") if orig_path is not None else None

        size = (self.image_size, self.image_size)
        img = TF.resize(img, size, interpolation=TF.InterpolationMode.BILINEAR)
        mask = TF.resize(mask, size, interpolation=TF.InterpolationMode.NEAREST)
        orig = (
            TF.resize(orig, size, interpolation=TF.InterpolationMode.BILINEAR)
            if orig_path is not None
            else None
        )

        if self.center_crop:
            img = TF.center_crop(img, size)
            mask = TF.center_crop(mask, size)
            orig = TF.center_crop(orig, size) if orig_path is not None else None
        img_t = TF.to_tensor(img)
        img_t = TF.normalize(
            img_t, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
        )

        mask_t = TF.to_tensor(mask)
        mask_t = (mask_t > 0.5).float()
        if mask_t.shape[0] > 1:
            mask_t = mask_t[0:1, ...]

        orig_t = TF.to_tensor(orig) if orig_path is not None else None

        return {
            "images": img_t,
            "masks": mask_t,
            "origs": orig_t,
            "filenames": img_path.name,
        }

    @staticmethod
    def collate_fn(batch):
        images = torch.stack([b["images"] for b in batch])
        masks = torch.stack([b["masks"] for b in batch])
        if batch[0]["origs"] is not None:
            origs = torch.stack([b["origs"] for b in batch])
        else:
            origs = None
        filenames = [b["filenames"] for b in batch]
        return {
            "images": images,
            "masks": masks,
            "filenames": filenames,
            "origs": origs if origs is not None else None,
        }


def build_argparser():
    p = argparse.ArgumentParser("Interface Pair Inference")
    p.add_argument("-i", type=str, required=True)
    p.add_argument("--cfg_path", type=str, default="configs/casiav2.yaml")
    p.add_argument("--resume", type=str, required=True, help="path to checkpoint .pth")
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--save_all", action="store_true")
    p.add_argument("--output_dir", type=str, default=None)
    p.add_argument("--use_orig", action="store_true")
    return p


def main(args):
    raw_dir = f"{args.i}/raw"
    mask_dir = f"{args.i}/mask"
    if args.use_orig:
        orig_dir = f"{args.i}/origin"
    else:
        orig_dir = None
    with open(args.cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)

    model_cfg = cfg.get("model", {})
    data_cfg = cfg.get("data", {})
    image_size = int(data_cfg.get("image_size", 320))

    device_str = args.device
    if device_str == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA device requested but not available. Falling back to CPU.")
        device_str = "cpu"

    device = torch.device(device_str)

    ds = PairFolderDataset(
        raw_dir,
        mask_dir,
        orig_dir=orig_dir,
        image_size=image_size,
        center_crop=bool(data_cfg.get("center_crop_eval", False)),
    )
    dl = torch.utils.data.DataLoader(
        ds,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=False,
        pin_memory=True,
        collate_fn=PairFolderDataset.collate_fn,
    )

    model = ANRTNet(device=device, **model_cfg).to(device)
    ckpt = torch.load(args.resume, map_location="cpu")
    ret = model.load_state_dict(ckpt["model_state_dict"], strict=False)
    logger.info(
        f"Loaded checkpoint with missing keys: {len(ret.missing_keys)}, unexpected: {len(ret.unexpected_keys)}"
    )
    model.eval()

    output_dir = Path(args.output_dir or (cfg["training"]["model_dir"]))
    (output_dir / "evaluation" / "epoch_0").mkdir(parents=True, exist_ok=True)

    metric_logger = MetricLogger(
        delimiter="  ", log_file=str(output_dir / "inference.log")
    )

    with torch.no_grad():
        for batch_idx, batch in enumerate(
            metric_logger.log_every(dl, print_freq=50, header="Infer")
        ):
            inputs = batch["images"].to(device)
            targets = batch["masks"].to(device)
            origs = batch["origs"].to(device) if batch["origs"] is not None else None
            filenames = batch["filenames"]

            outputs = model(inputs, gt_mask=targets)
            pred_masks = outputs["mask"]

            # metrics
            metrics = compute_metrics(pred_masks, targets, threshold=0.5)
            for k, v in metrics.items():
                metric_logger.update(**{k: v})

            # utils.save_eval_images(
            #     inputs,
            #     pred_masks,
            #     targets,
            #     filenames,
            #     epoch=0,
            #     output_dir=str(output_dir),
            #     save_all=args.save_all,
            #     origs=origs,
            # )

    metric_logger.synchronize_between_processes()
    logger.info(f"Final: {metric_logger}")


if __name__ == "__main__":
    parser = build_argparser()
    args = parser.parse_args()
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    main(args)
