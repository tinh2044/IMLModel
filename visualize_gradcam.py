import argparse
import time
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from PIL import Image
import yaml
from net import CMFDNet
from dataset import normalize_basename


def _to_numpy_image_from_tensor(img_t: torch.Tensor) -> np.ndarray:
    """Convert normalized tensor (C,H,W) in ImageNet space to numpy [0,1] RGB."""
    if isinstance(img_t, torch.Tensor):
        img = img_t.detach().cpu()
    else:
        raise TypeError("Expected torch.Tensor")
    if img.ndim == 4:
        img = img[0]
    assert img.ndim == 3 and img.shape[0] == 3
    mean = torch.tensor([0.485, 0.456, 0.406])[:, None, None]
    std = torch.tensor([0.229, 0.224, 0.225])[:, None, None]
    img = (img * std + mean).clamp(0, 1)
    img = img.permute(1, 2, 0).numpy()
    return img


def _prepare_image(img: Image.Image, size: int) -> torch.Tensor:
    img = TF.resize(img, (size, size), interpolation=TF.InterpolationMode.BILINEAR)
    t = TF.to_tensor(img)
    t = TF.normalize(t, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    return t


def _prepare_mask(mask: Image.Image, size: int) -> torch.Tensor:
    mask = TF.resize(mask, (size, size), interpolation=TF.InterpolationMode.NEAREST)
    t = TF.to_tensor(mask)
    if t.shape[0] > 1:
        t = t[0:1]
    t = (t > 0.5).float()
    return t


def _collect_pairs(
    data_dir: Path, exts=(".png", ".jpg", ".jpeg", ".bmp", ".tiff")
) -> List[Tuple[Path, Path]]:
    raw_dir = data_dir / "raw"
    mask_dir = data_dir / "mask"
    if not raw_dir.exists() or not mask_dir.exists():
        raise FileNotFoundError(f"Expect subfolders 'raw' and 'mask' in {data_dir}")

    raw_index = {}
    for p in raw_dir.rglob("*"):
        if p.is_file() and p.suffix.lower() in exts:
            raw_index.setdefault(normalize_basename(p.name), []).append(p)

    mask_index = {}
    for p in mask_dir.rglob("*"):
        if p.is_file() and p.suffix.lower() in exts:
            mask_index.setdefault(normalize_basename(p.name), []).append(p)

    pairs: List[Tuple[Path, Path]] = []
    matched = sorted(set(raw_index.keys()) & set(mask_index.keys()))
    for base in matched:
        img_path = sorted(raw_index[base])[0]
        mask_path = sorted(mask_index[base])[0]
        pairs.append((img_path, mask_path))
    return pairs


class GradCAM:
    def __init__(self, model: CMFDNet, target_layer: torch.nn.Module):
        self.model = model
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None
        self.hooks = []
        self._register_hooks()

    def _register_hooks(self):
        def fwd_hook(module, inp, out):
            self.activations = out

            def bwd_hook(grad):
                self.gradients = grad

            out.register_hook(bwd_hook)

        self.hooks.append(self.target_layer.register_forward_hook(fwd_hook))

    def remove(self):
        for h in self.hooks:
            h.remove()
        self.hooks = []

    def generate(
        self, inputs: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run model and compute Grad-CAM map.

        Returns: pred_mask (B,1,H,W), cam (B,H,W), mask_logits (B,2,H,W)
        """
        self.model.zero_grad(set_to_none=True)
        outputs = self.model(inputs)
        mask_logits = outputs["mask_logits"]  # (B,2,H,W)
        score = mask_logits[:, 1].mean()  # positive class average
        score.backward()

        acts = self.activations  # (B,C,h,w)
        grads = self.gradients  # (B,C,h,w)
        assert acts is not None and grads is not None, (
            "Hooks didn't capture activations/gradients"
        )

        weights = grads.mean(dim=(2, 3), keepdim=True)  # (B,C,1,1)
        cam = torch.relu((weights * acts).sum(dim=1))  # (B,h,w)
        # Normalize per-sample to [0,1]
        B = cam.size(0)
        cam_norm = []
        for i in range(B):
            c = cam[i]
            c = (c - c.min()) / (c.max() - c.min() + 1e-6)
            cam_norm.append(c)
        cam = torch.stack(cam_norm, dim=0)

        pred_mask = outputs["mask"].detach()
        return pred_mask, cam.detach(), mask_logits.detach()


def _overlay_cam_on_image(
    img_np: np.ndarray, cam_np: np.ndarray, alpha: float = 0.5
) -> np.ndarray:
    """Overlay a single-channel CAM [0,1] onto RGB image [0,1]."""
    import cv2  # local import to avoid hard dependency on import time

    cam_uint8 = np.uint8(255 * cam_np)
    heatmap = cv2.applyColorMap(cam_uint8, cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    heatmap = heatmap.astype(np.float32) / 255.0
    overlay = (1 - alpha) * img_np + alpha * heatmap
    overlay = np.clip(overlay, 0.0, 1.0)
    return overlay


def _save_image(path: Path, arr: np.ndarray, cmap: str = None):
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    if arr.ndim == 2:
        plt.imsave(path, arr, cmap=cmap or "gray")
    else:
        plt.imsave(path, arr)


def load_config(cfg: str) -> dict:
    with open(cfg, "r+", encoding="utf-8") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    return config["model"]


def load_model(checkpoint: str, device: torch.device, cfg: str) -> CMFDNet:
    model_cfg = load_config(cfg)
    model = CMFDNet(**(model_cfg or {}))
    model.to(device)
    model.eval()

    if checkpoint:
        state = torch.load(checkpoint, map_location="cpu")
        if isinstance(state, dict) and "model_state_dict" in state:
            sd = state["model_state_dict"]
        elif isinstance(state, dict) and "state_dict" in state:
            sd = state["state_dict"]
        else:
            sd = state
        missing_unexp = model.load_state_dict(sd, strict=False)
        try:
            mk = "\n".join(missing_unexp.missing_keys)
            uk = "\n".join(missing_unexp.unexpected_keys)
            print("Missing keys:\n", mk)
            print("Unexpected keys:\n", uk)
        except Exception:
            pass

    return model


def main():
    parser = argparse.ArgumentParser("Grad-CAM visualization for CMFDNet")
    parser.add_argument(
        "--data_dir", type=str, required=True, help="Path with subdirs raw/ and mask/"
    )
    parser.add_argument(
        "--checkpoint", type=str, required=True, help="Path to model checkpoint .pth"
    )
    parser.add_argument(
        "--image_size", type=int, default=256, help="Resize square size for inference"
    )
    parser.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="gradcam_outputs",
        help="Where to save outputs",
    )
    parser.add_argument(
        "--alpha", type=float, default=0.5, help="Overlay alpha for heatmap"
    )
    parser.add_argument(
        "--cfg",
        type=str,
        default="configs/imd20.yaml",
        help="Config file",
    )
    parser.add_argument(
        "--print_freq",
        type=int,
        default=1,
        help="Print progress every N samples",
    )

    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    model = load_model(args.checkpoint, device=device, cfg=args.cfg)
    target_layer = model.fuse.net[-1]  # last ReLU in fuse head
    gradcam = GradCAM(model, target_layer)

    data_dir = Path(args.data_dir)
    pairs = _collect_pairs(data_dir)
    print(f"Found {len(pairs)} pairs in {data_dir}")

    out_root = Path(args.output_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    total = len(pairs)
    last_end = time.time()
    iter_total = 0.0
    data_total = 0.0

    for idx, (img_path, mask_path) in enumerate(pairs):
        base = Path(img_path).stem
        out_dir = out_root / base
        out_dir.mkdir(parents=True, exist_ok=True)

        data_time = time.time() - last_end
        data_total += data_time

        img = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        x = _prepare_image(img, args.image_size).unsqueeze(0).to(device)
        y = _prepare_mask(mask, args.image_size).unsqueeze(0).to(device)

        with torch.enable_grad():
            pred_mask, cam, _ = gradcam.generate(x)

        # Resize CAM to input resolution
        cam_up = F.interpolate(
            cam.unsqueeze(1), size=x.shape[-2:], mode="bilinear", align_corners=True
        ).squeeze(1)

        # Prepare numpy images
        input_np = _to_numpy_image_from_tensor(x[0])
        gt_np = y[0, 0].detach().cpu().numpy()
        pred_np = (pred_mask[0, 0].detach().cpu().numpy() > 0.5).astype(np.float32)
        cam_np = cam_up[0].detach().cpu().numpy()

        overlay = _overlay_cam_on_image(input_np, cam_np, alpha=args.alpha)

        # Save outputs
        _save_image(out_dir / "input.png", input_np)
        _save_image(out_dir / "gt_mask.png", gt_np, cmap="gray")
        _save_image(out_dir / "pred_mask.png", pred_np, cmap="gray")
        _save_image(out_dir / "gradcam_overlay.png", overlay)

        # Progress logging to terminal only
        iter_time = time.time() - last_end
        iter_total += iter_time
        last_end = time.time()

        if idx % args.print_freq == 0 or idx == total - 1:
            avg_iter = iter_total / (idx + 1)
            eta_seconds = avg_iter * (total - idx - 1)
            eta_string = str(time.strftime("%H:%M:%S", time.gmtime(eta_seconds)))
            avg_data = data_total / (idx + 1)
            if torch.cuda.is_available() and device.type == "cuda":
                mb = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
                print(
                    f"Step [{idx + 1}/{total}], ETA: {eta_string}, time: {avg_iter:.4f}, data: {avg_data:.4f}, max mem: {mb:.0f}MB"
                )
            else:
                print(
                    f"Step [{idx + 1}/{total}], ETA: {eta_string}, time: {avg_iter:.4f}, data: {avg_data:.4f}"
                )

    gradcam.remove()
    print(f"Saved Grad-CAM results to {out_root}")


if __name__ == "__main__":
    main()
