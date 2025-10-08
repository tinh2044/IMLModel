# model_complete.py
from functools import partial
from fvcore.nn import flop_count_table
import torch
import torch.nn as nn
import torch.nn.functional as F

from block import (
    FSD4,
    FSD3,
    FSD2,
    FSD1,
)
from dcfs import (
    DCFS,
)
from gfsa import (
    GFSA,
)
from loss import FullLoss
from pvt_v2 import (
    PyramidVisionTransformerV2,
)


class Projector(nn.Module):
    """Multi-branch atrous-like projector returning D channels."""

    def __init__(self, in_ch, out_ch, atrous_rates=(1, 3, 5)):
        super().__init__()
        self.branches = nn.ModuleList()
        for r in atrous_rates:
            self.branches.append(
                nn.Sequential(
                    nn.Conv2d(
                        in_ch, out_ch, kernel_size=3, padding=r, dilation=r, bias=False
                    ),
                    nn.BatchNorm2d(out_ch),
                    nn.ReLU(inplace=True),
                )
            )
        self.proj = nn.Conv2d(out_ch * len(atrous_rates), out_ch, kernel_size=1)

    def forward(self, x):
        # print(f"x shape: {x.shape}")
        outs = [b(x) for b in self.branches]
        out = torch.cat(outs, dim=1)
        out = self.proj(out)
        return out


class MaskHead(nn.Module):
    def __init__(self, in_ch, mid_ch=64):
        super().__init__()
        self.head = nn.Sequential(
            nn.Conv2d(in_ch, mid_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(mid_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_ch, 1, 1),
        )

    def forward(self, x):
        return torch.sigmoid(self.head(x))  # B x 1 x H x W


class DecoderStage(nn.Module):
    """
    One decoder stage that accepts:
      - fsd_i : B x D x H x W (frequency feature)
      - proj_i: B x D x H x W (projector spatial feature)
      - up_in : B x D x H x W (upsampled output from previous deeper decoder stage)
    Internally uses DCFS and GFSA, exposes mask head.
    """

    def __init__(self, D, d_token=None):
        super().__init__()
        self.D = D
        d_token = D if d_token is None else d_token
        # DCFS expects two BxD maps and returns BxD map
        self.dcfs = DCFS(dim=D)  # adjust constructor names to match your dcfs.py
        self.gfsa = GFSA(D=D, gC=1)  # adjust to match gfsa.py signature
        self.mask_head = MaskHead(D, mid_ch=max(32, D // 2))
        self.guide_reduce = nn.Conv2d(D, 1, kernel_size=1)

    def forward(self, fsd_i, proj_in):
        # DCFS fusion
        dec_feat = self.dcfs(
            proj_in, fsd_i
        )  # expects (spatial, frequency), returns BxD x H x W
        # GFSA guided aggregation and mask
        guide = self.guide_reduce(proj_in)
        out_feat, mask = self.gfsa(dec_feat, guide)  # assume gfsa returns (feat, mask)
        # if gfsa returns only feat, then mask = self.mask_head(out_feat)
        if mask is None:
            mask = self.mask_head(out_feat)
        return out_feat, mask


class FSDFormer(nn.Module):
    def __init__(self, D=64, stem_ch=32, **kwargs):
        super().__init__()
        self.C4 = stem_ch * 8
        self.C3 = stem_ch * 4
        self.C2 = stem_ch * 2
        self.C1 = stem_ch
        self.encoder = PyramidVisionTransformerV2(
            img_size=512,
            patch_size=4,
            embed_dims=[self.C1, self.C2, self.C3, self.C4],
            num_heads=[1, 2, 4, 8],
            mlp_ratios=[8, 8, 4, 4],
            qkv_bias=True,
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
            depths=[2, 2, 2, 2],
            sr_ratios=[8, 4, 2, 1],
        )

        # channel counts from encoder

        # projectors
        self.proj4 = Projector(self.C4, D)

        self.fsd4 = FSD4(c4=self.C4, D=D)
        self.fsd3 = FSD3(c3=self.C3, D=D)
        self.fsd2 = FSD2(c2=self.C2, D=D)
        self.fsd1 = FSD1(c1=self.C1, D=D)

        # decoder stages
        self.dec4 = DecoderStage(D)
        self.dec3 = DecoderStage(D)
        self.dec2 = DecoderStage(D)
        self.dec1 = DecoderStage(D)

        # final fusion head
        self.fuse_conv = nn.Sequential(
            nn.Conv2d(4, 16, 3, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, 1),
        )
        self.loss_fn = FullLoss({})

    def forward(self, x, gt_mask=None):
        # x: B x 3 x H0 x W0
        # stem = self.stem(x)  # B x stem_ch x H0/2 x W0/2
        f1, f2, f3, f4 = self.encoder(x)

        p4 = self.proj4(f4)  # B x D x H4 x W4

        # print(p4.shape)

        # FSD pipeline top-down routing
        fsd4 = self.fsd4(f4)  # B x D x H4 x W4
        fsd4_up = F.interpolate(
            fsd4, size=f3.shape[2:], mode="bilinear", align_corners=False
        )
        fsd4_up = (
            F.conv2d(
                fsd4_up,
                weight=torch.eye(self.fsd4.block.D)
                .view(self.fsd4.block.D, self.fsd4.block.D, 1, 1)
                .to(fsd4_up.device),
            )
            if False
            else fsd4_up
        )
        # Note: smoothing conv omitted here. Use a conv layer if needed.

        fsd3 = self.fsd3(fsd4_up, f3)  # B x D x H3 x W3
        fsd4_up_to2 = F.interpolate(
            fsd4, size=f2.shape[2:], mode="bilinear", align_corners=False
        )
        fsd3_up = F.interpolate(
            fsd3, size=f2.shape[2:], mode="bilinear", align_corners=False
        )
        fsd2 = self.fsd2(fsd4_up_to2, fsd3_up, f2)  # B x D x H2 x W2

        fsd4_up_to1 = F.interpolate(
            fsd4, size=f1.shape[2:], mode="bilinear", align_corners=False
        )
        fsd3_up_to1 = F.interpolate(
            fsd3, size=f1.shape[2:], mode="bilinear", align_corners=False
        )
        fsd2_up = F.interpolate(
            fsd2, size=f1.shape[2:], mode="bilinear", align_corners=False
        )
        fsd1 = self.fsd1(fsd4_up_to1, fsd3_up_to1, fsd2_up, f1)  # B x D x H1 x W1

        # Decoder stage 4
        dec4_feat, p_mask4 = self.dec4(fsd4, p4)
        dec4_up = F.interpolate(
            dec4_feat, size=f3.shape[2:], mode="bilinear", align_corners=False
        )

        # Decoder stage 3
        dec3_feat, p_mask3 = self.dec3(fsd3, dec4_up)
        dec3_up = F.interpolate(
            dec3_feat, size=f2.shape[2:], mode="bilinear", align_corners=False
        )

        # Decoder stage 2
        dec2_feat, p_mask2 = self.dec2(fsd2, dec3_up)
        dec2_up = F.interpolate(
            dec2_feat, size=f1.shape[2:], mode="bilinear", align_corners=False
        )

        # Decoder stage 1
        dec1_feat, p_mask1 = self.dec1(fsd1, dec2_up)

        # Upsample masks to input resolution H0 x W0
        H0, W0 = x.shape[2], x.shape[3]
        P1_up = F.interpolate(
            p_mask1, size=(H0, W0), mode="bilinear", align_corners=False
        )
        P2_up = F.interpolate(
            p_mask2, size=(H0, W0), mode="bilinear", align_corners=False
        )
        P3_up = F.interpolate(
            p_mask3, size=(H0, W0), mode="bilinear", align_corners=False
        )
        P4_up = F.interpolate(
            p_mask4, size=(H0, W0), mode="bilinear", align_corners=False
        )

        # multi-scale fusion
        # multi = torch.cat([P1_up, P2_up, P3_up, P4_up], dim=1)  # B x 4 x H0 x W0
        # fuse = self.fuse_conv(multi)  # B x 1 x H0 x W0
        fuse = P1_up
        out_mask = torch.sigmoid(fuse)
        output = {
            "mask_logits": fuse,
            "mask": out_mask,
            # "masks_scale": (p_mask1, p_mask2, p_mask3, p_mask4),
            "features": {
                "fsd": (fsd1, fsd2, fsd3, fsd4),
                "dec": (dec1_feat, dec2_feat, dec3_feat, dec4_feat),
            },
        }
        if gt_mask is not None:
            loss = self.loss_fn(
                gt_mask,
                output,
            )
        else:
            loss = {}

        return {
            **output,
            "loss": loss,
        }


if __name__ == "__main__":
    from fvcore.nn import FlopCountAnalysis, flop_count_table

    model = FSDFormer(D=64, stem_ch=64)
    input_tensor = torch.randn(2, 3, 512, 512)
    gt_mask = torch.randn(2, 1, 512, 512)
    # print(flop_count_table(FlopCountAnalysis(model, input_tensor), max_depth=2))
    out = model(input_tensor, gt_mask)
    print("out mask shape", out["mask"].shape)  # expect [2,1,256,256]
    for i, m in enumerate(out["masks_scale"], 1):
        print(f"mask P{i} shape", m.shape)
    # features shapes
    # print("fsd shapes", [f.shape for f in out["features"]["fsd"]])
    # print("dec shapes", [d.shape for d in out["features"]["dec"]])

    for k, v in out["loss"].items():
        print(k, v)
