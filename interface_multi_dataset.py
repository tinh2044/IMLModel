import argparse
import os
from pathlib import Path
from typing import List, Tuple, Dict
import json

import torch
import torch.utils.data as data
import torchvision.transforms.functional as TF
from PIL import Image
import yaml

from loguru import logger

from net import FSDFormer
from metrics import compute_metrics
from logger import MetricLogger, SmoothedValue
import utils


def _pair_by_stem(
    raw_dir: Path, mask_dir: Path, orig_dir=None, exts=None
) -> List[Tuple[Path, Path, Path]]:
    if exts is None:
        exts = [".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"]
    raw_index: Dict[str, Path] = {}
    for p in raw_dir.iterdir():
        if p.suffix.lower() in exts and p.is_file():
            raw_index[p.stem.replace("_gt", "")] = p

    mask_index: Dict[str, Path] = {}
    for p in mask_dir.iterdir():
        if p.suffix.lower() in exts and p.is_file():
            mask_index[p.stem.replace("_gt", "")] = p

    if orig_dir is not None:
        orig_index: Dict[str, Path] = {}
        for p in orig_dir.iterdir():
            if p.suffix.lower() in exts and p.is_file():
                orig_index[p.stem.replace("_gt", "")] = p
    else:
        orig_index = None

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
        raw_dir: str,
        mask_dir: str,
        orig_dir: str = None,
        image_size: int = 320,
        center_crop: bool = False,
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
        origs = (
            torch.stack([b["origs"] for b in batch])
            if batch[0]["origs"] is not None
            else None
        )
        filenames = [b["filenames"] for b in batch]
        return {
            "images": images,
            "masks": masks,
            "filenames": filenames,
            "origs": origs,
        }


def discover_datasets(datasets_dir: Path) -> List[Dict[str, str]]:
    """
    Tự động phát hiện các dataset con trong thư mục datasets_dir.
    Mỗi dataset phải có thư mục raw và mask.
    """
    datasets = []

    for dataset_dir in datasets_dir.iterdir():
        if not dataset_dir.is_dir():
            continue

        raw_dir = dataset_dir / "raw"
        mask_dir = dataset_dir / "mask"
        orig_dir = dataset_dir / "origin"  # optional

        # Kiểm tra xem có thư mục raw và mask không
        if raw_dir.exists() and mask_dir.exists():
            dataset_info = {
                "name": dataset_dir.name,
                "raw_dir": str(raw_dir),
                "mask_dir": str(mask_dir),
                "orig_dir": str(orig_dir) if orig_dir.exists() else None,
                "dataset_path": str(dataset_dir),
            }
            datasets.append(dataset_info)
            logger.info(f"Found dataset: {dataset_dir.name}")
        else:
            logger.warning(
                f"Skipping {dataset_dir.name}: missing raw or mask directory"
            )

    return datasets


def test_single_dataset(
    dataset_info: Dict[str, str],
    model: FSDFormer,
    device: torch.device,
    args,
    cfg: Dict,
    base_output_dir: Path,
) -> Dict[str, float]:
    """
    Test một dataset đơn lẻ và trả về metrics.
    """
    dataset_name = dataset_info["name"]
    logger.info(f"Testing dataset: {dataset_name}")

    # Tạo thư mục output riêng cho dataset này
    dataset_output_dir = base_output_dir / dataset_name
    dataset_output_dir.mkdir(parents=True, exist_ok=True)

    # Cấu hình dataset
    data_cfg = cfg.get("data", {})
    image_size = int(data_cfg.get("image_size", 320))

    try:
        # Tạo dataset
        ds = PairFolderDataset(
            dataset_info["raw_dir"],
            dataset_info["mask_dir"],
            orig_dir=None,
            image_size=image_size,
            center_crop=bool(data_cfg.get("center_crop_eval", False)),
        )

        if len(ds) == 0:
            logger.warning(f"No samples found in dataset {dataset_name}")
            return {}

        # Tạo dataloader
        dl = torch.utils.data.DataLoader(
            ds,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            shuffle=False,
            pin_memory=True,
            collate_fn=PairFolderDataset.collate_fn,
        )

        # Tạo metric logger riêng cho dataset này
        log_file = dataset_output_dir / "inference.log"
        metric_logger = MetricLogger(delimiter="  ", log_file=str(log_file))

        # Tạo thư mục evaluation
        (dataset_output_dir / "evaluation" / "epoch_0").mkdir(
            parents=True, exist_ok=True
        )

        # Test dataset
        with torch.no_grad():
            for batch_idx, batch in enumerate(
                metric_logger.log_every(
                    dl, print_freq=1, header=f"Infer {dataset_name}"
                )
            ):
                inputs = batch["images"].to(device)
                targets = batch["masks"].to(device)
                origs = batch["origs"]
                if origs is not None:
                    origs = origs.to(device)

                filenames = batch["filenames"]

                outputs = model(inputs, gt_mask=targets)
                pred_masks = outputs["mask"]

                # metrics
                metrics = compute_metrics(pred_masks, targets, threshold=0.5)
                for k, v in metrics.items():
                    metric_logger.update(**{k: v})

                utils.save_eval_images(
                    inputs,
                    pred_masks,
                    targets,
                    filenames,
                    epoch=0,
                    output_dir=str(dataset_output_dir),
                    save_all=args.save_all,
                    origs=origs,
                )
                # utils.save_features_per_channel(
                #     inputs,
                #     pred_masks,
                #     targets,
                #     outputs,
                #     filenames,
                #     epoch=0,
                #     output_dir=str(dataset_output_dir),
                # )
                break

        metric_logger.synchronize_between_processes()
        logger.info(f"Dataset {dataset_name} - Final: {metric_logger}")

        # Trả về metrics cuối cùng
        final_metrics = {}
        for k, v in metric_logger.meters.items():
            if hasattr(v, "global_avg"):
                final_metrics[k] = v.global_avg
            else:
                final_metrics[k] = v.avg

        return final_metrics

    except Exception as e:
        logger.error(f"Error testing dataset {dataset_name}: {str(e)}")
        return {}


def build_argparser():
    p = argparse.ArgumentParser("Multi Dataset Interface Inference")
    p.add_argument(
        "-i",
        "--input_dir",
        type=str,
        required=True,
        help="Path to directory containing multiple datasets",
    )
    p.add_argument("--cfg", type=str, default="configs/casiav2.yaml")
    p.add_argument("--resume", type=str, required=True, help="path to checkpoint .pth")
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--save_all", action="store_true")
    p.add_argument("--output_dir", type=str, default=None)
    p.add_argument("--use_orig", action="store_true")
    p.add_argument(
        "--datasets",
        type=str,
        nargs="+",
        default=None,
        help="Specific datasets to test (by name). If not provided, test all discovered datasets.",
    )
    return p


def main(args):
    datasets_dir = Path(args.input_dir)
    if not datasets_dir.exists():
        raise ValueError(f"Input directory {datasets_dir} does not exist")

    # Phát hiện các dataset
    all_datasets = discover_datasets(datasets_dir)

    if not all_datasets:
        raise ValueError(f"No valid datasets found in {datasets_dir}")

    # Lọc dataset nếu có chỉ định cụ thể
    if args.datasets:
        datasets_to_test = [d for d in all_datasets if d["name"] in args.datasets]
        if not datasets_to_test:
            raise ValueError(f"None of the specified datasets {args.datasets} found")
    else:
        datasets_to_test = all_datasets

    logger.info(
        f"Found {len(datasets_to_test)} datasets to test: {[d['name'] for d in datasets_to_test]}"
    )

    # Load config
    with open(args.cfg, "r", encoding="utf-8") as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)

    model_cfg = cfg.get("model", {})

    device_str = args.device
    if device_str == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA device requested but not available. Falling back to CPU.")
        device_str = "cpu"

    device = torch.device(device_str)

    # Load model
    model = FSDFormer(**model_cfg).to(device)
    ckpt = torch.load(args.resume, map_location="cpu")
    ret = model.load_state_dict(ckpt["model_state_dict"], strict=False)
    logger.info(
        f"Loaded checkpoint with missing keys: {len(ret.missing_keys)}, unexpected: {len(ret.unexpected_keys)}"
    )
    model.eval()

    # Tạo thư mục output chính
    base_output_dir = Path(args.output_dir or (cfg["training"]["model_dir"]))
    base_output_dir.mkdir(parents=True, exist_ok=True)

    # Lưu trữ kết quả của tất cả dataset
    all_results = {}

    # Test từng dataset
    for dataset_info in datasets_to_test:
        logger.info(f"\n{'=' * 50}")
        logger.info(f"Testing dataset: {dataset_info['name']}")
        logger.info(f"{'=' * 50}")

        results = test_single_dataset(
            dataset_info, model, device, args, cfg, base_output_dir
        )

        all_results[dataset_info["name"]] = results

        # Lưu kết quả riêng cho dataset này
        dataset_output_dir = base_output_dir / dataset_info["name"]
        results_file = dataset_output_dir / "results.json"
        with open(results_file, "w") as f:
            json.dump(results, f, indent=2)

    # Tạo summary report cho tất cả dataset
    summary_file = base_output_dir / "summary_results.json"
    with open(summary_file, "w") as f:
        json.dump(all_results, f, indent=2)

    # In summary ra console
    logger.info(f"\n{'=' * 60}")
    logger.info("SUMMARY RESULTS")
    logger.info(f"{'=' * 60}")

    for dataset_name, results in all_results.items():
        logger.info(f"\nDataset: {dataset_name}")
        for metric, value in results.items():
            logger.info(f"  {metric}: {value:.4f}")

    logger.info(f"\nDetailed results saved to: {base_output_dir}")
    logger.info(f"Summary results saved to: {summary_file}")


if __name__ == "__main__":
    parser = build_argparser()
    args = parser.parse_args()
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    main(args)
