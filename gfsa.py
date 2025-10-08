# gfsa.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class LayerNormChannel(nn.Module):
    """
    LayerNorm over channel dimension for tensor [B, C, H, W].
    Implemented by permuting to [B, H, W, C], applying nn.LayerNorm(C), then permuting back.
    """

    def __init__(self, num_channels: int, eps: float = 1e-6):
        super().__init__()
        self.ln = nn.LayerNorm(num_channels, eps=eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, H, W]
        B, C, H, W = x.shape
        x_perm = x.permute(0, 2, 3, 1).contiguous()  # [B, H, W, C]
        x_ln = self.ln(x_perm)
        return x_ln.permute(0, 3, 1, 2).contiguous()  # [B, C, H, W]


class SE_Bottleneck(nn.Module):
    """
    Simple SE-like bottleneck: D -> D_mid -> D, returns scale vector in (0,1)^D.
    Input is channel vector z (D) or tensor [B, D].
    """

    def __init__(self, D: int, reduction: int = 16):
        super().__init__()
        mid = max(1, D // reduction)
        self.net = nn.Sequential(
            nn.Linear(D, mid, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(mid, D, bias=True),
            nn.Sigmoid(),
        )
        # init last bias to 0 to start gate at ~0.5
        nn.init.constant_(self.net[2].bias, 0.0)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        # z: [B, D] or [D] -> ensure shape [B, D]
        if z.dim() == 1:
            z = z.unsqueeze(0)
        return self.net(z)  # [B, D]


class GFSA(nn.Module):
    """
    Guided Frequency-Spatial Aggregation module.

    Inputs:
      dec: [B, D, H, W]  feature from DCFS
      guide: [B, gC, H, W] guide map, gC typically 1 or small

    Outputs:
      Y_out: [B, D, H, W] aggregated feature
      M_pred: [B, 1, H, W] mask prediction at this scale
    """

    def __init__(
        self,
        D: int,
        gC: int = 1,
        reduction: int = 16,
        use_bn: bool = True,
        mid_ch: Optional[int] = None,
    ):
        super().__init__()
        self.D = D
        self.gC = gC
        self.reduction = reduction
        mid = mid_ch if mid_ch is not None else max(8, D // 4)

        # Normalization on dec (channel-wise LayerNorm)
        self.norm_chan = LayerNormChannel(D)

        # Guide projection to single channel and sigmoid to [0,1]
        if gC == 1:
            # pass-through or light conv to refine
            self.guide_proj = nn.Sequential(
                nn.Conv2d(1, 1, kernel_size=3, padding=1, bias=True), nn.Sigmoid()
            )
        else:
            self.guide_proj = nn.Sequential(
                nn.Conv2d(gC, 1, kernel_size=1, bias=True), nn.Sigmoid()
            )

        # Branch A: local spatial refinement
        layers_a = []
        layers_a.append(nn.Conv2d(D, mid, kernel_size=3, padding=1, bias=False))
        if use_bn:
            layers_a.append(nn.BatchNorm2d(mid))
        layers_a.append(nn.ReLU(inplace=True))
        layers_a.append(nn.Conv2d(mid, D, kernel_size=3, padding=1, bias=False))
        if use_bn:
            layers_a.append(nn.BatchNorm2d(D))
        layers_a.append(nn.ReLU(inplace=True))
        self.branchA = nn.Sequential(*layers_a)

        # Branch B: frequency-guided channel modulation (SE-like)
        # We'll produce channel-wise scale s of size D via bottleneck conditioned on pooled dec and scalar pooled guide
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.se = SE_Bottleneck(D, reduction=reduction)

        # Branch C: guided spatial attention map
        self.branchC_reduce = nn.Conv2d(D, 1, kernel_size=1, bias=True)  # hat{D}
        # No biasing to preserve sign; we'll normalize via softmax later

        # Channel gating for combining branches: compute per-channel gates gA,gB,gC
        self.gA_fc = nn.Sequential(
            nn.Linear(D, max(8, D // 4)),
            nn.ReLU(inplace=True),
            nn.Linear(max(8, D // 4), D),
            nn.Sigmoid(),
        )
        self.gB_fc = nn.Sequential(
            nn.Linear(D, max(8, D // 4)),
            nn.ReLU(inplace=True),
            nn.Linear(max(8, D // 4), D),
            nn.Sigmoid(),
        )
        self.gC_fc = nn.Sequential(
            nn.Linear(D, max(8, D // 4)),
            nn.ReLU(inplace=True),
            nn.Linear(max(8, D // 4), D),
            nn.Sigmoid(),
        )

        # Final merge conv and residual scaling gamma
        self.merge_conv = nn.Conv2d(D, D, kernel_size=1, bias=False)
        self.merge_bn = nn.BatchNorm2d(D) if use_bn else nn.Identity()
        self.gamma = nn.Parameter(torch.tensor(1.0))

        # Mask head modules: small conv stack -> 1 channel sigmoid
        self.mask_head = nn.Sequential(
            nn.Conv2d(D, mid, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid) if use_bn else nn.Identity(),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid, 1, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )

        # Initialize convs
        self._init_weights()

        self.last_norm = LayerNormChannel(D)

    def _init_weights(self):
        # Kaiming for convs
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
            if isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)

    def forward(
        self, dec: torch.Tensor, guide: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        dec: [B, D, H, W]
        guide: [B, gC, H, W]
        returns:
          Y_out: [B, D, H, W]
          M_pred: [B, 1, H, W]
        """
        B, D, H, W = dec.shape
        assert D == self.D, "channel mismatch"

        # Step 1: normalize dec
        dec_norm = self.norm_chan(dec)  # [B, D, H, W]

        # Step 1b: project guide to single channel G in [0,1]
        G = self.guide_proj(guide)  # [B, 1, H, W]

        # Branch A: local spatial refinement
        B_A = self.branchA(dec_norm)  # [B, D, H, W]

        # Branch B: frequency-guided modulation (channel-wise scaling)
        z = self.pool(dec_norm).view(B, D)  # [B, D]
        g_scalar = self.pool(G).view(B)  # [B]
        # Multiply z by scalar g (broadcast)
        zg = z * g_scalar.unsqueeze(1)  # [B, D]
        s = self.se(zg)  # [B, D] scale in (0,1)
        B_B = dec_norm * s.unsqueeze(-1).unsqueeze(-1)  # [B, D, H, W]

        # Branch C: guided spatial attention
        hatD = self.branchC_reduce(dec_norm)  # [B, 1, H, W]
        # Compute element-wise product with guide, then softmax over spatial positions
        prod = (hatD * G).view(B, -1)  # [B, H*W]
        attn = F.softmax(prod, dim=-1).view(B, 1, H, W)  # [B, 1, H, W]
        # Multiply attn to dec_norm
        B_C = dec_norm * attn  # [B, D, H, W]

        # Step 3: gating compute per-channel gates via global pooling of each branch
        gp_A = self.pool(B_A).view(B, D)  # [B, D]
        gp_B = self.pool(B_B).view(B, D)  # [B, D]
        gp_C = self.pool(B_C).view(B, D)  # [B, D]

        gA = self.gA_fc(gp_A)  # [B, D] in (0,1)
        gB = self.gB_fc(gp_B)  # [B, D]
        gC = self.gC_fc(gp_C)  # [B, D]

        # Combine branches by per-channel gating
        # Expand gates to spatial dims
        gA_map = gA.unsqueeze(-1).unsqueeze(-1)  # [B, D, 1, 1]
        gB_map = gB.unsqueeze(-1).unsqueeze(-1)
        gC_map = gC.unsqueeze(-1).unsqueeze(-1)

        B_mix = gA_map * B_A + gB_map * B_B + gC_map * B_C  # [B, D, H, W]

        # Merge and residual
        Y_pre = self.merge_conv(B_mix)  # [B, D, H, W]
        Y_pre = (
            self.merge_bn(Y_pre) if isinstance(self.merge_bn, nn.BatchNorm2d) else Y_pre
        )
        Y_out = dec + self.gamma * Y_pre  # residual
        # Apply final channel LayerNorm via LayerNormChannel for stability
        Y_out = self.last_norm(Y_out)
        Y_out = F.gelu(Y_out)

        # Mask head
        M_pred = self.mask_head(Y_out)  # [B, 1, H, W]

        return Y_out, M_pred


if __name__ == "__main__":
    # quick shape test
    B, D, H, W = 2, 64, 64, 64
    gC = 1
    x = torch.randn(B, D, H, W)
    guide = torch.randn(B, gC, H, W)
    gfsa = GFSA(D=D, gC=gC)
    y, m = gfsa(x, guide)
    print("Y_out shape", y.shape)
    print("M_pred shape", m.shape)
