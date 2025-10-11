# model_complete.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import (
    resnet18,
    ResNet18_Weights,
)

from torchvision import models


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
        # DCFS fusion (expects (frequency, spatial))
        dec_feat = self.dcfs(fsd_i, proj_in)
        # GFSA guided aggregation and mask
        guide = self.guide_reduce(proj_in)
        out_feat, mask = self.gfsa(dec_feat, guide)  # assume gfsa returns (feat, mask)
        # if gfsa returns only feat, then mask = self.mask_head(out_feat)
        if mask is None:
            mask = self.mask_head(out_feat)
        return out_feat, mask


class ResNet18Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        backbone = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        self.conv1 = backbone.conv1
        self.bn1 = backbone.bn1
        self.relu = backbone.relu
        self.maxpool = backbone.maxpool
        self.layer1 = backbone.layer1  # 64 ch, stride 1 (overall /4)
        self.layer2 = backbone.layer2  # 128 ch, stride 2 (overall /8)
        self.layer3 = backbone.layer3  # 256 ch, stride 2 (overall /16)
        self.layer4 = backbone.layer4  # 512 ch, stride 2 (overall /32)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        f1 = self.layer1(x)
        f2 = self.layer2(f1)
        f3 = self.layer3(f2)
        f4 = self.layer4(f3)
        return f1, f2, f3, f4


class EfficientNetEncoder(nn.Module):
    def __init__(self, pretrained: bool = True, variant: str = "b0"):
        super().__init__()
        try:
            weights = (
                getattr(models, f"EfficientNet_{variant.upper()}_Weights").IMAGENET1K_V1
                if pretrained
                else None
            )
            backbone = getattr(models, f"efficientnet_{variant}")(weights=weights)
        except AttributeError:
            raise ValueError(f"Variant {variant} not found in torchvision.models")

        self.stem = nn.Sequential(backbone.features[0])  # /2
        # We will group features to approx /4, /8, /16, /32
        # EfficientNet blocks and downsample points: [0]/2, [1]/4, [2]/4, [3]/8, [4]/16, [5]/16, [6]/32
        self.block1 = nn.Sequential(backbone.features[1], backbone.features[2])  # ~ /4
        self.block2 = nn.Sequential(backbone.features[3])  # ~ /8
        self.block3 = nn.Sequential(backbone.features[4], backbone.features[5])  # ~ /16
        self.block4 = nn.Sequential(backbone.features[6])  # ~ /32

        # Get actual output channels from the backbone
        # For EfficientNet, we need to check the last layer of each block
        self.ch1 = (
            backbone.features[2][-1].out_channels
            if hasattr(backbone.features[2][-1], "out_channels")
            else 24
        )
        self.ch2 = (
            backbone.features[3][-1].out_channels
            if hasattr(backbone.features[3][-1], "out_channels")
            else 40
        )
        self.ch3 = (
            backbone.features[5][-1].out_channels
            if hasattr(backbone.features[5][-1], "out_channels")
            else 112
        )
        self.ch4 = (
            backbone.features[6][-1].out_channels
            if hasattr(backbone.features[6][-1], "out_channels")
            else 320
        )

    def get_output_channels(self):
        return self.ch1, self.ch2, self.ch3, self.ch4

    def forward(self, x):
        x = self.stem(x)  # /2
        f1 = self.block1(x)  # /4
        f2 = self.block2(f1)  # /8
        f3 = self.block3(f2)  # /16
        f4 = self.block4(f3)  # /32

        return f1, f2, f3, f4


class FSDFormer(nn.Module):
    def __init__(self, D=64, stem_ch=32, **kwargs):
        super().__init__()
        encoder_name = kwargs.get("encoder", "efficientnet")
        pretrained_backbone = kwargs.get("pretrained_backbone", True)

        if encoder_name == "efficientnet":
            variant = kwargs.get("variant", "b0")
            self.encoder = EfficientNetEncoder(
                pretrained=pretrained_backbone,
                variant=variant,
            )
        else:
            self.encoder = ResNet18Encoder()

        self.C1, self.C2, self.C3, self.C4 = self.encoder.get_output_channels()

        # projectors
        self.proj4 = Projector(self.C4, D)
        # self.proj3 = Projector(self.C3, D)
        # self.proj2 = Projector(self.C2, D)
        # self.proj1 = Projector(self.C1, D)

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
            nn.Conv2d(1, 16, 3, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, 1),
        )
        self.loss_fn = FullLoss({})

    def forward(self, x, gt_mask=None):
        f1, f2, f3, f4 = self.encoder(x)
        p4 = self.proj4(f4)  # B x D x H4 x W4
        # p3 = self.proj3(f3)  # B x D x H3 x W3
        # p2 = self.proj2(f2)  # B x D x H2 x W2
        # p1 = self.proj1(f1)  # B x D x H1 x W1

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
        # print(f"dec1_feat and fsd1 shape: {dec1_feat.shape}, {fsd1.shape}")
        # print(f"dec2_feat and fsd2 shape: {dec2_feat.shape}, {fsd2.shape}")
        # print(f"dec3_feat and fsd3 shape: {dec3_feat.shape}, {fsd3.shape}")
        # print(f"dec4_feat and fsd4 shape: {dec4_feat.shape}, {fsd4.shape}")

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
        # multi = torch.cat(P1_up, dim=1)  # B x 1 x H0 x W0
        fuse = self.fuse_conv(P1_up)  # B x 1 x H0 x W0 (logits)
        out_mask = torch.sigmoid(fuse)
        output = {
            "mask_logits": fuse,
            "mask": out_mask,
            "masks_scale": (p_mask1, p_mask2, p_mask3, p_mask4),
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
    out = model(input_tensor, gt_mask)
    print(flop_count_table(FlopCountAnalysis(model, input_tensor), max_depth=2))
    # print("out mask shape", out["mask"].shape)  # expect [2,1,256,256]
    # for i, m in enumerate(out["masks_scale"], 1):
    #     print(f"mask P{i} shape", m.shape)
    # # features shapes
    # print("fsd shapes", [f.shape for f in out["features"]["fsd"]])
    # print("dec shapes", [d.shape for d in out["features"]["dec"]])

    for k, v in out["loss"].items():
        print(k, v)
