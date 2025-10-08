import torch
from torch.nn import functional as F


def _binary_dice_score(
    pred: torch.Tensor, target: torch.Tensor, smooth: float = 1.0
) -> torch.Tensor:
    """Soft Dice score for binary maps in [0,1].

    pred/target: (B,1,H,W)
    returns scalar tensor
    """
    pred = pred.contiguous().view(pred.size(0), -1)
    target = target.contiguous().view(target.size(0), -1)
    intersection = (pred * target).sum(dim=1)
    denom = pred.sum(dim=1) + target.sum(dim=1)
    dice = (2.0 * intersection + smooth) / (denom + smooth)
    return dice.mean()


def _soft_iou_loss(
    pred: torch.Tensor, target: torch.Tensor, smooth: float = 1.0
) -> torch.Tensor:
    """Soft IoU (Jaccard) loss for binary maps in [0,1]."""
    pred = pred.contiguous().view(pred.size(0), -1)
    target = target.contiguous().view(target.size(0), -1)
    intersection = (pred * target).sum(dim=1)
    union = pred.sum(dim=1) + target.sum(dim=1) - intersection
    iou = (intersection + smooth) / (union + smooth)
    return (1.0 - iou).mean()


def _focal_tversky_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    alpha: float = 0.7,
    beta: float = 0.3,
    gamma: float = 0.75,
    smooth: float = 1.0,
) -> torch.Tensor:
    """Focal Tversky Loss (emphasizes recall for minority class when alpha>beta)."""
    pred = pred.contiguous().view(pred.size(0), -1)
    target = target.contiguous().view(target.size(0), -1)
    tp = (pred * target).sum(dim=1)
    fp = (pred * (1.0 - target)).sum(dim=1)
    fn = ((1.0 - pred) * target).sum(dim=1)
    tversky = (tp + smooth) / (tp + alpha * fn + beta * fp + smooth)
    return (1.0 - tversky).pow(gamma).mean()


class FocalLoss(torch.nn.Module):
    """
    Focal loss for binary segmentation (per-pixel).
    Accepts preds in probability space [0,1]. If you have logits, set from_logits=True.
    """

    def __init__(
        self,
        alpha: float = 0.25,
        gamma: float = 2.0,
        eps: float = 1e-6,
        from_logits: bool = False,
    ):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.eps = eps
        self.from_logits = from_logits

    def forward(self, preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        preds: [B,1,H,W] or [B,H,W] probabilities (or logits if from_logits=True)
        targets: same spatial shape as preds, values in {0,1}
        returns: scalar loss (mean over batch & pixels)
        """
        # ensure shapes: make preds and targets shape [B, H, W]
        if preds.dim() == 4 and preds.size(1) == 1:
            preds = preds[:, 0]
        if targets.dim() == 4 and targets.size(1) == 1:
            targets = targets[:, 0]

        if self.from_logits:
            preds = torch.sigmoid(preds)

        preds = preds.clamp(self.eps, 1.0 - self.eps)

        # BCE per pixel
        pt = torch.where(targets == 1, preds, 1 - preds)  # p_t
        w = torch.where(targets == 1, self.alpha, 1.0 - self.alpha)
        loss = -w * (1 - pt) ** self.gamma * torch.log(pt)
        return loss.mean()


def multiscale_focal_loss(
    gt_mask,
    masks_scale,
    focal_params=None,
    weights=None,
):
    """
    Compute focal loss for multi-scale supervision.

    Args:
      outputs: dict returned by model, must contain:
         - "mask": final mask prediction tensor [B,1,H0,W0] (probabilities)
         - "masks_scale": tuple/list of per-scale preds (p_mask1..p_mask4)
             each p_mask_i: [B,1,H_i,W_i] (probabilities)
      gt_mask: ground-truth mask [B,1,H0,W0] with values {0,1} (or floats in [0,1])
      focal_params: dict passed to FocalLoss (alpha,gamma,eps,from_logits)
      weights: tuple of length 5 for weighting loss terms:
         (w_final, w_p1, w_p2, w_p3, w_p4).
         If None, defaults to (1.0, 0.5, 0.75, 1.0, 1.25) as an example.

    Returns:
      total_loss: scalar
      details: dict with per-scale losses and final
    """
    if focal_params is None:
        focal_params = {"alpha": 0.25, "gamma": 2.0, "eps": 1e-6, "from_logits": False}
    loss_fn = FocalLoss(**focal_params)

    if weights is None:
        weights = (1.0, 0.5, 0.75, 1.0, 1.25)

    B, _, H0, W0 = gt_mask.shape
    # ensure gt is float {0,1}
    gt_mask = gt_mask.float()

    total_loss = 0.0
    # per-scale masks: assume list/tuple in order (p_mask1, p_mask2, p_mask3, p_mask4)
    # weights indices: 1..4
    for idx, p in enumerate(masks_scale, start=1):
        pred = p  # [B,1,Hi,Wi]
        pred_up = F.interpolate(
            pred, size=gt_mask.shape[2:], mode="bilinear", align_corners=False
        )
        l = loss_fn(pred_up, gt_mask)

        total_loss = total_loss + weights[idx] * l

    return total_loss


class FullLoss(torch.nn.Module):
    def __init__(self, loss_cfg: dict = {}):
        super(FullLoss, self).__init__()
        self.loss_pos_weight = float(loss_cfg.get("pos_weight", 5.0))  # static fallback
        self.loss_dynamic_pos_weight = bool(loss_cfg.get("dynamic_pos_weight", True))
        self.loss_lambda_dice = float(loss_cfg.get("lambda_dice", 1.0))
        self.loss_lambda_iou = float(loss_cfg.get("lambda_iou", 0.0))
        self.loss_lambda_ft = float(loss_cfg.get("lambda_focal_tversky", 0.0))
        self.tversky_alpha = float(loss_cfg.get("tversky_alpha", 0.7))
        self.tversky_beta = float(loss_cfg.get("tversky_beta", 0.3))
        self.tversky_gamma = float(loss_cfg.get("tversky_gamma", 0.75))

        self.focal_params = {
            "alpha": float(loss_cfg.get("focal_alpha", 0.25)),
            "gamma": float(loss_cfg.get("focal_gamma", 2.0)),
            "eps": float(loss_cfg.get("focal_eps", 1e-6)),
            "from_logits": bool(loss_cfg.get("focal_from_logits", True)),
        }
        self.focal_weights = tuple(
            float(w) for w in loss_cfg.get("focal_weights", (1.0, 0.5, 0.75, 1.0, 1.25))
        )

    def forward(self, gt_mask, outputs_pred):
        mask_logits = outputs_pred["mask_logits"]
        mask_probs = outputs_pred["mask"]
        masks_scale = outputs_pred["masks_scale"]
        device = mask_logits.device
        loss_dict = {}
        with torch.no_grad():
            if self.loss_dynamic_pos_weight:
                total = float(gt_mask.numel())
                pos = float(gt_mask.sum().item())
                neg = max(total - pos, 0.0)
                w1 = 1.0 if pos <= 0.0 else max(1.0, min(neg / (pos + 1e-6), 20.0))
            else:
                w1 = self.loss_pos_weight
        weight = torch.tensor([w1], device=device, dtype=torch.float32)
        ce = F.binary_cross_entropy_with_logits(mask_logits, gt_mask, pos_weight=weight)

        p_pos = mask_probs
        t_pos = gt_mask.float()

        dice_loss = 1.0 - _binary_dice_score(p_pos, t_pos)
        iou_loss = _soft_iou_loss(p_pos, t_pos)
        ft_loss = _focal_tversky_loss(
            p_pos, t_pos, self.tversky_alpha, self.tversky_beta, self.tversky_gamma
        )

        focal_loss = multiscale_focal_loss(
            masks_scale=masks_scale,
            gt_mask=gt_mask,
            focal_params=self.focal_params,
            weights=self.focal_weights,
        )

        total_loss = ce
        total_loss = total_loss + self.loss_lambda_dice * dice_loss
        total_loss = total_loss + self.loss_lambda_iou * iou_loss
        total_loss = total_loss + self.loss_lambda_ft * ft_loss
        total_loss = total_loss + focal_loss

        loss_dict["ce"] = ce
        loss_dict["dice"] = dice_loss
        loss_dict["iou"] = iou_loss
        loss_dict["focal_tversky"] = ft_loss
        loss_dict["focal"] = focal_loss
        loss_dict["total"] = total_loss

        return loss_dict
