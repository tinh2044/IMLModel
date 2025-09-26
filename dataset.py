import os
import torch
import torch.utils.data as data
import torchvision.transforms.functional as TF
from PIL import Image, UnidentifiedImageError
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple


def normalize_basename(filename: str) -> str:
    stem = Path(filename).stem
    if stem.endswith("_gt"):
        stem = stem[:-3]
    return stem


class ForgeryDataset(data.Dataset):
    def __init__(self, cfg: Dict, split: str = "train"):
        self.root = Path(cfg["root"])
        self.cfg = cfg or {}
        self.split = cfg["train_dir"] if split == "train" else cfg["test_dir"]
        print(f"ForgeryDataset[{split}] -> {self.split}")
        split_dir = self.root / self.split
        if split_dir.exists():
            self.raw_dir = split_dir / "raw"
            self.mask_dir = split_dir / "mask"
            self.base_dir = split_dir
        else:
            raise ValueError(f"Split directory {split_dir} does not exist")

        self.image_size = int(self.cfg.get("image_size", 320))
        self.crop_size = int(self.cfg.get("crop_size", self.image_size))
        self.use_center_crop_eval = bool(self.cfg.get("center_crop_eval", False))
        self.return_paths = bool(self.cfg.get("return_paths", False))

        self.split_list: Optional[set] = None
        split_list_file = self.cfg.get(f"{split}_list")  # e.g., train_list, val_list
        if split_list_file is not None:
            split_list_path = Path(split_list_file)
            if split_list_path.is_file():
                items = [
                    line.strip()
                    for line in split_list_path.read_text(encoding="utf-8").splitlines()
                    if line.strip()
                ]
                self.split_list = set(items)
        self.ignore_dataset = self.cfg.get("ignore_dataset", [])

        self.samples = self._build_samples()
        random.shuffle(self.samples)

        print(f"ForgeryDataset[{self.split}] -> {len(self.samples)} samples")
        # if self.base_dir != self.root:
        #     print(f"  Base dir: {self.base_dir}")
        # print(f"  Raw: {self.raw_dir} ({len(self._list_files(self.raw_dir))} files)")
        # print(f"  Mask: {self.mask_dir} ({len(self._list_files(self.mask_dir))} files)")

    def _safe_open_image(self, path: Path, mode: str) -> Optional[Image.Image]:
        try:
            img = Image.open(path)
            if mode == "RGB":
                return img.convert("RGB")
            if mode == "L":
                return img.convert("L")
            return img
        except (UnidentifiedImageError, OSError, ValueError) as e:
            print(f"Warning: failed to open image '{path}': {e}")
            return None

    def _build_samples(self) -> List[Tuple[Path, Optional[Path]]]:
        samples: List[Tuple[Path, Optional[Path]]] = []

        if self.raw_dir.exists() and self.mask_dir.exists():
            raw_dir = str(self.raw_dir)
            mask_dir = str(self.mask_dir)
            list_files = os.listdir(raw_dir)
            list_files = [x for x in list_files if os.path.exists(f"{mask_dir}/{x}")]
            filter_files = []
            for file in list_files:
                is_ignore = False
                for ignore_dataset in self.ignore_dataset:
                    if ignore_dataset in file:
                        is_ignore = True
                        break
                if not is_ignore:
                    filter_files.append(file)
            samples = [
                (Path(raw_dir) / file, Path(mask_dir) / file) for file in filter_files
            ]
        return samples

    def __len__(self):
        return len(self.samples)

    def _get_pair_transforms(self, is_train: bool):
        size = (self.image_size, self.image_size)
        crop = (self.crop_size, self.crop_size)

        def transform_pair(
            img: Image.Image, mask: Image.Image
        ) -> Tuple[torch.Tensor, torch.Tensor]:
            img = TF.resize(img, size, interpolation=TF.InterpolationMode.BILINEAR)
            mask = TF.resize(mask, size, interpolation=TF.InterpolationMode.NEAREST)

            if self.use_center_crop_eval and crop != size:
                img = TF.center_crop(img, crop)
                mask = TF.center_crop(mask, crop)

            img_t = TF.to_tensor(img)
            img_t = TF.normalize(
                img_t, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
            )

            mask_t = TF.to_tensor(mask)
            mask_t = (mask_t > 0.5).float()
            if mask_t.shape[0] > 1:
                mask_t = mask_t[0:1, ...]

            return img_t, mask_t

        return transform_pair

    def __getitem__(self, idx):
        img_path, mask_path = self.samples[idx]

        img = self._safe_open_image(img_path, "RGB")
        if img is None:
            return None

        mask = self._safe_open_image(mask_path, "L")
        if mask is None:
            return None

        try:
            transform_pair = self._get_pair_transforms(is_train=(self.split == "train"))
            image_tensor, mask_tensor = transform_pair(img, mask)
        except Exception as e:
            print(f"Warning: failed to transform sample '{img_path.name}': {e}")
            return None

        sample = {
            "image": image_tensor,
            "mask": mask_tensor,
            "filename": img_path.name,
            "is_tampered": bool(mask_tensor.max().item() > 0.5),
            "has_mask": True,
        }

        if self.return_paths:
            sample["image_path"] = str(img_path)
            sample["mask_path"] = str(mask_path) if mask_path is not None else None

        return sample

    def data_collator(self, batch):
        valid_batch = [
            item
            for item in batch
            if item is not None and "image" in item and "mask" in item
        ]
        if not valid_batch:
            return None
        try:
            images = torch.stack([item["image"] for item in valid_batch])
            masks = torch.stack([item["mask"] for item in valid_batch])
            filenames = [item["filename"] for item in valid_batch]

            batch_out = {
                "images": images,
                "masks": masks,
                "filenames": filenames,
            }
            if "image_path" in valid_batch[0]:
                batch_out["image_paths"] = [
                    item.get("image_path") for item in valid_batch
                ]
                batch_out["mask_paths"] = [
                    item.get("mask_path") for item in valid_batch
                ]
            return batch_out
        except Exception as e:
            print(f"Error in data collator: {e}")
            return None
