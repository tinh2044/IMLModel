import torch
import torch.distributed as dist
import numpy as np
import matplotlib.pyplot as plt
import os
from pathlib import Path
from fvcore.nn import FlopCountAnalysis


def save_img(tensor, path, normalize=True):
    """Save tensor as image file"""
    if isinstance(tensor, torch.Tensor):
        # Convert tensor to numpy
        if tensor.is_cuda:
            tensor = tensor.cpu()
        img = tensor.detach().numpy()

        # Handle different tensor shapes
        if img.ndim == 4:  # Batch of images
            img = img[0]  # Take first image
        if img.ndim == 3:  # CHW format
            if img.shape[0] == 1:  # Single channel (mask)
                img = img[0]  # HW
            elif img.shape[0] == 3:  # RGB
                img = img.transpose(1, 2, 0)  # HWC
            else:
                raise ValueError(f"Unsupported channel number: {img.shape[0]}")
        elif img.ndim == 2:  # HW format (mask)
            pass
        else:
            raise ValueError(f"Unsupported tensor dimensions: {img.ndim}")

    # Normalize if needed
    if normalize and img.dtype != np.uint8:
        if img.max() <= 1.0:
            img = (img * 255).astype(np.uint8)
        else:
            img = img.astype(np.uint8)

    # Ensure directory exists
    os.makedirs(os.path.dirname(path), exist_ok=True)

    # Save image
    if len(img.shape) == 2:  # Grayscale
        plt.imsave(path, img, cmap="gray")
    else:  # RGB
        plt.imsave(path, img)


def save_sample_images(inputs, pred_masks, targets, batch_idx, epoch, output_dir):
    """Save sample images during training for visualization"""
    sample_dir = Path(output_dir) / "samples"
    sample_dir.mkdir(parents=True, exist_ok=True)

    # Convert tensors to images
    input_img = inputs[0].cpu()  # (3, H, W)
    target_mask = targets[0].cpu()  # (1, H, W)
    pred_mask = pred_masks[0].cpu()  # (1, H, W)

    # Create visualization
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Input image
    input_np = input_img.permute(1, 2, 0).numpy()
    input_np = (input_np - input_np.min()) / (input_np.max() - input_np.min())
    axes[0].imshow(input_np)
    axes[0].set_title("Input Image")
    axes[0].axis("off")

    # Ground truth mask
    target_np = target_mask.squeeze().numpy()
    axes[1].imshow(target_np, cmap="gray")
    axes[1].set_title("Ground Truth")
    axes[1].axis("off")

    # Predicted mask
    pred_np = (pred_mask.squeeze().detach().numpy() > 0.5).astype(float)
    axes[2].imshow(pred_np, cmap="gray")
    axes[2].set_title("Prediction")
    axes[2].axis("off")

    plt.tight_layout()
    plt.savefig(
        sample_dir / f"epoch_{epoch}_batch_{batch_idx}.png",
        bbox_inches="tight",
        dpi=150,
    )
    plt.close()


def save_eval_images(
    inputs, pred_masks, targets, filenames, epoch, output_dir, save_all, origs=None
):
    """Save evaluation images with metrics"""
    eval_dir = Path(output_dir) / "evaluation" / f"epoch_{epoch}"
    eval_dir.mkdir(parents=True, exist_ok=True)

    if save_all:
        index = range(len(inputs))
    else:
        index = [np.random.randint(0, len(inputs))]
    for i in index:
        input_img = inputs[i].cpu()
        target_mask = targets[i].cpu()
        pred_mask = pred_masks[i].cpu()
        if origs is not None:
            orig_img = origs[i].cpu()
        filename = filenames[i]

        # Calculate metrics for this sample
        pred_binary = (pred_mask > 0.5).float()
        target_binary = (target_mask > 0.5).float()

        intersection = (pred_binary * target_binary).sum()
        union = pred_binary.sum() + target_binary.sum() - intersection
        iou = (intersection / (union + 1e-8)).item()

        if origs is not None:
            j = 1
        else:
            j = 0

        fig, axes = plt.subplots(1, 4 + j, figsize=(20, 5))

        if origs is not None:
            # Original image
            orig_np = orig_img.permute(1, 2, 0).numpy()
            orig_np = (orig_np - orig_np.min()) / (orig_np.max() - orig_np.min())
            axes[0].imshow(orig_np)
            axes[0].set_title("Original Image")
            axes[0].axis("off")

        input_np = input_img.permute(1, 2, 0).numpy()
        input_np = (input_np - input_np.min()) / (input_np.max() - input_np.min())
        axes[0 + j].imshow(input_np)
        axes[0 + j].set_title("Input Image")
        axes[0 + j].axis("off")

        # Ground truth mask
        target_np = target_mask.squeeze().numpy()
        axes[1 + j].imshow(target_np, cmap="gray")
        axes[1 + j].set_title("Ground Truth")
        axes[1 + j].axis("off")

        # Predicted mask
        pred_np = pred_mask.squeeze().numpy()
        axes[2 + j].imshow(pred_np, cmap="gray")
        axes[2 + j].set_title("Prediction")
        axes[2 + j].axis("off")

        # Overlay
        overlay = input_np.copy()
        overlay[target_np > 0.5] = [1, 0, 0]  # Red for ground truth
        overlay[pred_np > 0.5] = [0, 1, 0]  # Green for prediction
        axes[3 + j].imshow(overlay)
        axes[3 + j].set_title(f"Overlay (IoU: {iou:.3f})")
        axes[3 + j].axis("off")

        plt.suptitle(f"File: {filename}", fontsize=12)
        plt.tight_layout()
        plt.savefig(
            eval_dir / f"eval_{i}_{Path(filename).stem}.png",
            bbox_inches="tight",
            dpi=150,
        )
        plt.close()


def _to_numpy_image(t: torch.Tensor) -> np.ndarray:
    """Convert a tensor (C,H,W) or (H,W) to numpy image in [0,1] range."""
    if isinstance(t, torch.Tensor):
        t = t.detach().cpu()
    arr = t.numpy()
    if arr.ndim == 3:  # C,H,W
        if arr.shape[0] == 3:
            arr = arr.transpose(1, 2, 0)
            # Min-max normalize per image
            mn, mx = arr.min(), arr.max()
            if mx > mn:
                arr = (arr - mn) / (mx - mn)
        elif arr.shape[0] == 1:
            arr = arr[0]
            mn, mx = arr.min(), arr.max()
            if mx > mn:
                arr = (arr - mn) / (mx - mn)
        else:
            # For multi-channel non-RGB, reduce with min-max on each channel when saved per-channel
            mn, mx = arr.min(), arr.max()
            if mx > mn:
                arr = (arr - mn) / (mx - mn)
            # Keep as (C,H,W); caller may slice per channel
    elif arr.ndim == 2:
        mn, mx = arr.min(), arr.max()
        if mx > mn:
            arr = (arr - mn) / (mx - mn)
    return arr


def save_features_per_channel(
    inputs: torch.Tensor,
    pred_masks: torch.Tensor,
    targets: torch.Tensor,
    outputs: dict,
    filenames,
    epoch: int,
    output_dir,
):
    """Save inputs, targets, preds, and per-channel feature maps to per-filename folders.

    Directory layout:
      output_dir/filenames/<name>/
        input.png
        target.png
        pred.png
        <feature_name>/ch_000.png, ch_001.png, ... (for multi-channel)
        <feature_name>.png (for single-channel)
    """
    base = Path(output_dir) / "features"
    base.mkdir(parents=True, exist_ok=True)

    B = inputs.size(0)
    # Ensure filenames is a list of strings
    if not isinstance(filenames, (list, tuple)):
        filenames = [str(filenames)] * B

    for i in range(B):
        name = Path(str(filenames[i])).stem
        out_dir = base / name
        out_dir.mkdir(parents=True, exist_ok=True)

        # Save input image
        in_img = _to_numpy_image(inputs[i])
        plt.imsave(
            out_dir / "input.png",
            in_img if in_img.ndim == 3 else in_img,
            cmap=None if in_img.ndim == 3 else "gray",
        )

        # Save target mask
        tgt = targets[i]
        tgt_np = _to_numpy_image(
            tgt.squeeze(0) if tgt.ndim == 3 and tgt.shape[0] == 1 else tgt
        )
        plt.imsave(out_dir / "target.png", tgt_np, cmap="gray")

        # Save predicted mask
        pred = pred_masks[i]
        pred_np = _to_numpy_image(
            pred.squeeze(0) if pred.ndim == 3 and pred.shape[0] == 1 else pred
        )
        plt.imsave(out_dir / "pred.png", pred_np, cmap="gray")

        # Iterate over output features
        for key, tensor in outputs.items():
            if not isinstance(tensor, torch.Tensor):
                continue
            if tensor.ndim == 4 and tensor.size(0) > i:
                feat = tensor[i]
                C = feat.size(0)
                H, W = feat.size(-2), feat.size(-1)
                # Only save image-like maps
                if H >= 2 and W >= 2:
                    if C == 1:
                        arr = _to_numpy_image(feat[0])
                        plt.imsave(out_dir / f"{key}.png", arr, cmap="gray")
                    else:
                        ch_dir = out_dir / key
                        ch_dir.mkdir(parents=True, exist_ok=True)
                        for c in range(C):
                            arr = _to_numpy_image(feat[c])
                            plt.imsave(ch_dir / f"ch_{c:03d}.png", arr, cmap="gray")


def is_dist_avail_and_initialized():
    """Check if distributed training is available and initialized"""
    if not torch.distributed.is_available():
        return False
    if not torch.distributed.is_initialized():
        return False
    return True


def get_rank():
    import torch.distributed as dist

    if not torch.distributed.is_available() or not dist.is_initialized():
        return 0
    return dist.get_rank()


def is_main_process():
    return get_rank() == 0


def save_on_master(obj, path):
    if is_main_process():
        from pathlib import Path as _Path

        p = _Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        torch.save(obj, p)


def setup_distributed():
    """Initialize distributed training (torchrun-friendly)."""
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size > 1:
        rank = int(os.environ.get("RANK", "0"))
        local_rank = int(
            os.environ.get(
                "LOCAL_RANK", os.environ.get("OMPI_COMM_WORLD_LOCAL_RANK", "0")
            )
        )

        backend = "nccl" if torch.cuda.is_available() and os.name != "nt" else "gloo"
        init_method = os.environ.get("DIST_INIT_METHOD", "env://")

        if not dist.is_initialized():
            dist.init_process_group(backend=backend, init_method=init_method)

        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)

        print(
            f"Setting up distributed training: rank={rank}, local_rank={local_rank}, world_size={world_size}, backend={backend}"
        )

        return True, rank, local_rank, world_size
    return False, 0, 0, 1


def cleanup_distributed():
    if dist.is_initialized():
        dist.destroy_process_group()


def count_model_parameters(model) -> int:
    """Count total number of parameters in a model."""
    return sum(p.numel() for p in model.parameters())


def get_model_info(model, input_shape, device) -> dict:
    """Return model parameter counts and, if possible, FLOPs and MACs.

    input_shape: tuple like (batch, channels, height, width)
    """
    info = {}
    total_params = count_model_parameters(model)
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    info["total_params"] = total_params
    info["trainable_params"] = trainable_params
    info["non_trainable_params"] = total_params - trainable_params
    dummy_input = torch.randn(input_shape).to(device)

    flops = FlopCountAnalysis(model, dummy_input)
    flops_total = flops.total()
    flops_str = f"{flops_total / 1e9:.3f} GFLOPs"
    info.update(
        {
            "flops": int(flops_total),
            "flops_str": flops_str,
            "params_str": f"{total_params:,.3f}",
        }
    )
    return info


def check_state_dict(model, state_dict: dict) -> bool:
    """Basic compatibility check between a model and a checkpoint state_dict.

    Returns True if all keys in checkpoint exist in the model and shapes match
    for the overlapping keys.
    """
    model_state = model.state_dict()
    model_keys = set(model_state.keys())
    ckpt_keys = set(state_dict.keys())

    if not ckpt_keys:
        return False

    if not ckpt_keys.issubset(model_keys):
        return False

    for k in ckpt_keys:
        if k in model_state and model_state[k].shape != state_dict[k].shape:
            return False
    return True
