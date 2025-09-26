import numpy as np
from sklearn.metrics import f1_score, roc_auc_score
import torch
from typing import Dict


def calculate_iou(
    pred_mask: torch.Tensor, gt_mask: torch.Tensor, threshold: float = 0.5
) -> float:
    """
    Calculate Intersection over Union (IoU) for binary segmentation
    Args:
        pred_mask: (B, 1, H, W) - probabilities [0,1] for positive class
        gt_mask: (B, 1, H, W) - binary mask [0,1]
        threshold: threshold for binary classification
    Returns:
        IoU score
    """
    if pred_mask.dim() == 4:
        pred_mask = pred_mask.squeeze(1)  # (B,1,H,W) -> (B,H,W)
    if gt_mask.dim() == 4:
        gt_mask = gt_mask.squeeze(1)  # (B,1,H,W) -> (B,H,W)

    # pred_mask is already probabilities [0,1], apply threshold
    pred_binary = (pred_mask > threshold).float()
    gt_binary = (gt_mask > threshold).float()

    intersection = (pred_binary * gt_binary).sum()
    union = pred_binary.sum() + gt_binary.sum() - intersection

    return (intersection / (union + 1e-8)).item()


def calculate_f1(
    pred_mask: torch.Tensor, gt_mask: torch.Tensor, threshold: float = 0.5
) -> float:
    """
    Calculate F1 score for binary segmentation
    Args:
        pred_mask: (B, 1, H, W) - probabilities [0,1] for positive class
        gt_mask: (B, 1, H, W) - binary mask [0,1]
        threshold: threshold for binary classification
    Returns:
        F1 score
    """
    if pred_mask.dim() == 4:
        pred_mask = pred_mask.squeeze(1)  # (B,1,H,W) -> (B,H,W)
    if gt_mask.dim() == 4:
        gt_mask = gt_mask.squeeze(1)  # (B,1,H,W) -> (B,H,W)

    pred_mask = pred_mask.detach().cpu().numpy()
    gt_mask = gt_mask.detach().cpu().numpy()
    score = []
    for yy_true, yy_pred in zip(gt_mask, pred_mask):
        this = f1_score(
            (yy_true > 0.5).astype("int").ravel(), (yy_pred > 0.5).astype("int").ravel()
        )
        that = f1_score(
            (yy_true > 0.5).astype("int").ravel(),
            (1 - yy_pred > 0.5).astype("int").ravel(),
        )
        score.append(max(this, that))
    return float(np.mean(score).astype("float32"))


def np_auc(pred_mask: torch.Tensor, gt_mask: torch.Tensor):
    pred_mask = pred_mask.detach().cpu().numpy()
    gt_mask = gt_mask.detach().cpu().numpy()

    score = []

    for yy_true, yy_pred in zip(gt_mask, pred_mask):
        try:
            this = roc_auc_score((yy_true > 0.5).astype("int").ravel(), yy_pred.ravel())
            that = roc_auc_score(
                (yy_true > 0.5).astype("int").ravel(), 1 - yy_pred.ravel()
            )
            score.append(max(this, that))
        except Exception as e:
            print(e)
            score.append(0)
    return float(np.mean(score).astype("float32"))


def calculate_precision(
    pred_mask: torch.Tensor, gt_mask: torch.Tensor, threshold: float = 0.5
) -> float:
    """
    Calculate precision for binary segmentation
    Args:
        pred_mask: (B, 1, H, W) - probabilities [0,1] for positive class
        gt_mask: (B, 1, H, W) - binary mask [0,1]
        threshold: threshold for binary classification
    Returns:
        Precision score
    """
    if pred_mask.dim() == 4:
        pred_mask = pred_mask.squeeze(1)  # (B,1,H,W) -> (B,H,W)
    if gt_mask.dim() == 4:
        gt_mask = gt_mask.squeeze(1)  # (B,1,H,W) -> (B,H,W)

    # pred_mask is already probabilities [0,1], apply threshold
    pred_binary = (pred_mask > threshold).float()
    gt_binary = (gt_mask > threshold).float()

    tp = ((pred_binary == 1) & (gt_binary == 1)).sum().float()
    fp = ((pred_binary == 1) & (gt_binary == 0)).sum().float()

    return (tp / (tp + fp + 1e-8)).item()


def calculate_recall(
    pred_mask: torch.Tensor, gt_mask: torch.Tensor, threshold: float = 0.5
) -> float:
    """
    Calculate recall for binary segmentation
    Args:
        pred_mask: (B, 1, H, W) - probabilities [0,1] for positive class
        gt_mask: (B, 1, H, W) - binary mask [0,1]
        threshold: threshold for binary classification
    Returns:
        Recall score
    """
    if pred_mask.dim() == 4:
        pred_mask = pred_mask.squeeze(1)  # (B,1,H,W) -> (B,H,W)
    if gt_mask.dim() == 4:
        gt_mask = gt_mask.squeeze(1)  # (B,1,H,W) -> (B,H,W)

    # pred_mask is already probabilities [0,1], apply threshold
    pred_binary = (pred_mask > threshold).float()
    gt_binary = (gt_mask > threshold).float()

    tp = ((pred_binary == 1) & (gt_binary == 1)).sum().float()
    fn = ((pred_binary == 0) & (gt_binary == 1)).sum().float()

    return (tp / (tp + fn + 1e-8)).item()


def calculate_accuracy(
    pred_mask: torch.Tensor, gt_mask: torch.Tensor, threshold: float = 0.5
) -> float:
    """
    Calculate pixel accuracy for binary segmentation
    Args:
        pred_mask: (B, 1, H, W) - probabilities [0,1] for positive class
        gt_mask: (B, 1, H, W) - binary mask [0,1]
        threshold: threshold for binary classification
    Returns:
        Pixel accuracy score
    """
    if pred_mask.dim() == 4:
        pred_mask = pred_mask.squeeze(1)  # (B,1,H,W) -> (B,H,W)
    if gt_mask.dim() == 4:
        gt_mask = gt_mask.squeeze(1)  # (B,1,H,W) -> (B,H,W)

    # pred_mask is already probabilities [0,1], apply threshold
    pred_binary = (pred_mask > threshold).float()
    gt_binary = (gt_mask > threshold).float()

    correct = (pred_binary == gt_binary).sum().float()
    total = torch.numel(gt_binary)

    return (correct / total).item()


def compute_metrics(
    pred_mask: torch.Tensor, gt_mask: torch.Tensor, threshold: float = 0.5
) -> Dict[str, float]:
    """
    Compute all forgery detection metrics
    Args:
        pred_mask: (B, 1, H, W) - probabilities [0,1] for positive class
        gt_mask: (B, 1, H, W) - binary mask [0,1]
        threshold: threshold for binary classification
    Returns:
        dict with keys: 'iou', 'f1', 'auc', 'precision', 'recall', 'accuracy'
    """
    return {
        "iou": calculate_iou(pred_mask, gt_mask, threshold),
        "f1": calculate_f1(pred_mask, gt_mask, threshold),
        "auc": np_auc(pred_mask, gt_mask),
        "precision": calculate_precision(pred_mask, gt_mask, threshold),
        "recall": calculate_recall(pred_mask, gt_mask, threshold),
        "accuracy": calculate_accuracy(pred_mask, gt_mask, threshold),
    }
