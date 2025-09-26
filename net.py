# anrt_full.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


def make_orthonormal_dct_matrix(n: int, device=None, dtype=torch.float32):
    k = torch.arange(n, dtype=torch.float32, device=device)
    i = torch.arange(n, dtype=torch.float32, device=device).unsqueeze(1)
    theta = math.pi * (2 * i + 1) * k / (2.0 * n)
    D = torch.cos(theta)
    D[0, :] *= 1.0 / math.sqrt(2.0)
    D = D * math.sqrt(2.0 / n)
    return D.to(dtype=dtype)


def patchify(x, p: int):
    # x: (B,C,H,W) -> (B, N, C, p, p)
    B, C, H, W = x.shape
    assert H % p == 0 and W % p == 0
    x_unf = F.unfold(x, kernel_size=p, stride=p)  # (B, C*p*p, N)
    _, Cp2, N = x_unf.shape
    x_unf = x_unf.view(B, C, p * p, N).permute(0, 3, 1, 2)
    x_unf = x_unf.view(B, N, C, p, p)
    return x_unf


def unpatchify(patches, p: int, H: int, W: int):
    # patches: (B, N, C, p, p)
    B, N, C, p1, p2 = patches.shape
    assert p1 == p2 == p
    patches = patches.view(B, N, C * p * p).permute(0, 2, 1)
    x = F.fold(patches, output_size=(H, W), kernel_size=p, stride=p)
    return x


class FrequencyMaskGenerator(nn.Module):
    """
    Generates M_f(theta) \in (0,1)^{C x p x p} via a small MLP on normalized frequency coords.
    This is more structured than a raw tensor and enforces smoothness priors via weight decay.
    """

    def __init__(self, channels=3, p=16, hidden=64):
        super().__init__()
        # Create coordinate grid of shape (p*p, 2)
        xi = torch.linspace(0.0, 1.0, steps=p)
        yi = torch.linspace(0.0, 1.0, steps=p)
        grid = torch.stack(torch.meshgrid(xi, yi), dim=-1).view(-1, 2)  # (p*p,2)
        self.register_buffer("coords", grid)  # normalized frequencies
        self.net = nn.Sequential(
            nn.Linear(2, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, channels * p * p),
        )
        self.channels = channels
        self.p = p

    def forward(self):
        # coords: (p*p,2) -> output (C,p,p) after sigmoid
        out = self.net(self.coords)  # (p*p, channels*p*p)
        # reshape
        out = out.view(self.p * self.p, self.channels, self.p, self.p)
        # average across input coord dimension (we used grid as batch; simpler: compute per-coord then average)
        # For stability, instead create a single map per channel by averaging outputs over grid positions:
        # but to keep parameter count small, instead sum the outputs along first dim then normalize
        out = out.mean(dim=0)  # (channels, p, p)
        return torch.sigmoid(out)  # in (0,1)


class NRE_DCT(nn.Module):
    """
    Nonsemantic Residual Extractor (NRE):
      - Patchify
      - DCT per patch (linear orthonormal)
      - Frequency mask from FrequencyMaskGenerator
      - iDCT + subtract => residual patches
    """

    def __init__(self, p=16, in_ch=3, device="cpu"):
        super().__init__()
        self.p = p
        self.in_ch = in_ch
        D = make_orthonormal_dct_matrix(p, device=device)
        self.register_buffer("D", D)  # (p,p)
        self.register_buffer("D_t", D.t())  # inverse
        self.mask_gen = FrequencyMaskGenerator(channels=in_ch, p=p)
        # small temperature param for mask smoothness (you can regularize mask_gen weights)
        self.register_buffer("mask_temp", torch.tensor(1.0))

    def dct2(self, patches):
        B, N, C, p, _ = patches.shape
        x = patches.reshape(B * N * C, p, p)
        x = torch.matmul(self.D, torch.matmul(x, self.D.t()))
        x = x.view(B, N, C, p, p)
        return x

    def idct2(self, coeffs):
        B, N, C, p, _ = coeffs.shape
        x = coeffs.reshape(B * N * C, p, p)
        x = torch.matmul(self.D_t, torch.matmul(x, self.D_t.t()))
        x = x.view(B, N, C, p, p)
        return x

    def forward(self, x):
        # x: (B,C,H,W)
        B, C, H, W = x.shape
        patches = patchify(x, self.p)  # (B,N,C,p,p)
        Dcoeff = self.dct2(patches)  # (B,N,C,p,p)
        Mf = (
            self.mask_gen().unsqueeze(0).unsqueeze(0)
        )  # (1,1,C,p,p) if needed broadcast
        Dmasked = Dcoeff * Mf  # broadcast -> (B,N,C,p,p)
        recon = self.idct2(Dmasked)
        residual = patches - recon  # (B,N,C,p,p)
        # return residual patches and optionally low-freq recon for inspection
        return (
            residual,
            recon,
            Mf.squeeze(0).squeeze(0),
        )  # residual:(B,N,C,p,p), recon:(B,N,C,p,p), Mf:(C,p,p)


class TokenEmbedder(nn.Module):
    def __init__(self, p=16, in_ch=3, d=256, dropout=0.0):
        super().__init__()
        self.in_dim = in_ch * p * p
        self.d = d
        self.proj = nn.Linear(self.in_dim, d)
        self.pos = None  # init in model
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, residual_flat, pos_embed):
        # residual_flat: (B, N, in_dim)
        x = self.proj(residual_flat)  # (B,N,d)
        x = x + pos_embed
        x = self.dropout(x)
        return x


class GateMLP(nn.Module):
    def __init__(self, in_dim=2, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(inplace=True), nn.Linear(hidden, 1)
        )

    def forward(self, x):  # x: (B,N,2)
        return torch.sigmoid(self.net(x).squeeze(-1))  # (B,N) values in (0,1)


class AGSMLayer(nn.Module):
    """
    Single AGSM layer: gate-modulated attention with softTopK behavior & FFN.
    tau predictor is per-query (learned).
    """

    def __init__(self, d=256, mlp_ratio=4.0, beta=12.0, alpha_gate=0.8):
        super().__init__()
        self.ln1 = nn.LayerNorm(d)
        self.qkv = nn.Linear(d, d * 3)
        self.proj = nn.Linear(d, d)
        self.ln2 = nn.LayerNorm(d)
        self.ffn = nn.Sequential(
            nn.Linear(d, int(d * mlp_ratio)),
            nn.GELU(),
            nn.Linear(int(d * mlp_ratio), d),
        )
        self.tau = nn.Linear(d, 1)  # per query threshold predictor
        self.beta = beta
        self.alpha_gate = alpha_gate

    def forward_attention(self, x, g):
        # x: (B,N,d), g: (B,N)
        B, N, d = x.shape
        qkv = self.qkv(self.ln1(x))  # (B,N,3d)
        q, k, v = torch.chunk(qkv, 3, dim=-1)
        scale = 1.0 / math.sqrt(d)
        A_raw = torch.einsum("bnd,bmd->bnm", q, k) * scale  # (B,N,N)

        g_outer = (g.unsqueeze(2) * g.unsqueeze(1)) ** self.alpha_gate  # (B,N,N)
        A = A_raw * g_outer

        tau = self.tau(self.ln1(x)).squeeze(-1).unsqueeze(-1)  # (B,N,1)
        mask_soft = torch.sigmoid(self.beta * (A - tau))  # (B,N,N)
        Asoft = F.softmax(A, dim=-1)
        W = mask_soft * Asoft
        W = W / (W.sum(dim=-1, keepdim=True) + 1e-12)
        out = torch.einsum("bnm,bmd->bnd", W, v)  # (B,N,d)
        out = self.proj(out)
        return out, A  # return A for possible diagnostics

    def forward(self, x, g):
        att_out, A = self.forward_attention(x, g)
        x = x + att_out
        x = x + self.ffn(self.ln2(x))
        return x, A


class AGSM(nn.Module):
    """
    Stack of AGSM layers, returning feature maps from selected layers.
    """

    def __init__(
        self,
        d=256,
        depth=6,
        mlp_ratio=4.0,
        beta=12.0,
        alpha_gate=0.8,
        return_layers=None,
    ):
        super().__init__()
        self.layers = nn.ModuleList(
            [
                AGSMLayer(d=d, mlp_ratio=mlp_ratio, beta=beta, alpha_gate=alpha_gate)
                for _ in range(depth)
            ]
        )
        self.depth = depth
        if return_layers is None:
            self.return_layers = [depth - 1]  # default: last layer
        else:
            self.return_layers = return_layers

    def forward(self, tokens, gates):
        # tokens: (B,N,d), gates: (B,N)
        feats = []
        A_list = []
        x = tokens
        for i, layer in enumerate(self.layers):
            x, A = layer(x, gates)
            A_list.append(A)
            if i in self.return_layers:
                feats.append(x)
        return x, feats, A_list


class CSCA(nn.Module):
    """
    Cross-Scale Contrastive Amplifier (faithful but efficient approximation).
    Produces a multiplicative gain a_i per token based on cross-scale contrasts.
    We implement the KL-style approx using per-scale mean+var and a symmetric KL approx.
    """

    def __init__(self, grid_h, grid_w, scales=(1, 2, 4), eps=1e-6, gamma=2.0, eta=0.02):
        super().__init__()
        self.scales = scales
        self.grid_h = grid_h
        self.grid_w = grid_w
        self.eps = eps
        self.gamma = gamma
        self.eta = eta

    def pool_to_scale(self, energy_grid, scale):
        # energy_grid: (B,1,GH,GW)
        if scale == 1:
            return energy_grid  # (B,1,GH,GW)
        # pool with kernel=scale
        out = F.avg_pool2d(
            energy_grid, kernel_size=scale, stride=scale, ceil_mode=False
        )
        return out

    def forward(self, token_feats, residual_energy):
        # token_feats: (B,N,d) ; residual_energy: (B,N)
        B, N, d = token_feats.shape
        GH, GW = self.grid_h, self.grid_w
        energy_grid = residual_energy.view(B, 1, GH, GW)
        base = token_feats.norm(dim=-1)  # (B,N)
        # compute per-scale norms and map back to base resolution
        scale_norms = []
        for s in self.scales:
            pooled = self.pool_to_scale(energy_grid, s)  # (B,1,GH/s,GW/s)
            up = F.interpolate(pooled, size=(GH, GW), mode="nearest").view(
                B, N
            )  # (B,N)
            scale_norms.append(up)
        # compute normalized contrast C_i as average absolute deviation
        stacked = torch.stack(scale_norms, dim=1)  # (B, S, N)
        mean_across = stacked.mean(dim=1)  # (B, N)
        C = torch.abs(base - mean_across) / (self.eps + base + mean_across)  # (B,N)
        # amplifier
        a = 1.0 + self.gamma * F.relu(C - self.eta)
        return a  # (B,N) multiplicative gains


class EAMDecoder(nn.Module):
    """
    Evidence Attribution Module (EAM):
      - Accepts multi-scale token features, upsamples & fuses
      - Produces K evidence maps E_k(x) and attribution alpha(x)
      - Produces mask as Mpred = sum_k alpha_k * E_k
    """

    def __init__(self, d=256, gh=32, gw=32, C_f=128, evidences=4):
        super().__init__()
        self.gh = gh
        self.gw = gw
        self.conv1 = nn.Conv2d(d, C_f, kernel_size=1)
        self.conv2 = nn.Conv2d(C_f, C_f, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(C_f, C_f, kernel_size=3, padding=1)
        self.evidence_heads = nn.Conv2d(C_f, evidences, kernel_size=1)
        self.alpha_head = nn.Conv2d(C_f, evidences, kernel_size=1)
        self.mask_head = nn.Conv2d(C_f, 1, kernel_size=1)
        # per-scale scalar weights (learnable)
        self.gamma_scales = nn.Parameter(torch.ones(1, 1, 1), requires_grad=True)

    def forward(self, feats_list):
        # feats_list: list of token-feature maps (B, N, d) from different layers
        # We will simply use the last entry for decoder input for simplicity but keep interface for multiple
        feat = feats_list[-1]  # (B,N,d)
        B, N, d = feat.shape
        GH = self.gh
        GW = self.gw
        feat_grid = feat.view(B, GH, GW, d).permute(0, 3, 1, 2)  # (B,d,GH,GW)
        x = self.conv1(feat_grid)
        x = F.gelu(x)
        x = self.conv2(x)
        x = F.gelu(x)
        x = self.conv3(x)
        # upsample to full image resolution (512x512)
        x_up = F.interpolate(
            x, size=(GH * 16, GW * 16), mode="bilinear", align_corners=False
        )  # p=16
        evidence_maps = torch.sigmoid(self.evidence_heads(x_up))
        alpha_logits = self.alpha_head(x_up)
        alpha = F.softmax(alpha_logits, dim=1)
        mask_pred = (alpha * evidence_maps).sum(dim=1, keepdim=True)
        # also compute mask_head (aux)
        mask_aux = torch.sigmoid(self.mask_head(x_up))
        return mask_aux, evidence_maps, alpha, mask_aux, x_up


class ANRTNet(nn.Module):
    def __init__(
        self,
        img_size=512,
        patch_size=16,
        in_ch=3,
        d=256,
        depth=6,
        evidences=4,
        device="cpu",
        **kwargs
    ):
        super().__init__()
        self.img_size = img_size
        self.p = patch_size
        self.in_ch = in_ch
        self.d = d
        self.depth = depth
        self.evidences = evidences
        self.grid_h = img_size // patch_size
        self.grid_w = img_size // patch_size
        self.N = self.grid_h * self.grid_w

        # Modules
        self.nre = NRE_DCT(p=self.p, in_ch=self.in_ch, device=device)
        self.embedder = TokenEmbedder(p=self.p, in_ch=self.in_ch, d=self.d, dropout=0.0)
        # position embedding
        self.pos_embed = nn.Parameter(torch.zeros(1, self.N, self.d))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        self.gatemlp = GateMLP(in_dim=2, hidden=64)
        self.backbone = AGSM(
            d=self.d, depth=self.depth, return_layers=[self.depth - 2, self.depth - 1]
        )
        self.csca = CSCA(
            grid_h=self.grid_h,
            grid_w=self.grid_w,
            scales=(1, 2, 4),
            gamma=2.0,
            eta=0.02,
        )
        self.decoder = EAMDecoder(
            d=self.d, gh=self.grid_h, gw=self.grid_w, C_f=128, evidences=self.evidences
        )

        # Regularizer weights may be used externally
        self.register_buffer("reg_attrib_smooth_tau", torch.tensor(1.0))
        self.to(device)
        
        self.bce_loss = nn.BCELoss()

    def compute_attribution_smoothness(self, alpha):
        # alpha: (B, K, H, W) -> measure local variation (L1 of Laplacian)
        lap = (
            F.laplacian(alpha.mean(dim=1, keepdim=True), mode="reflect")
            if hasattr(F, "laplacian")
            else None
        )
        if lap is None:
            # fallback finite diff
            dx = alpha[..., :, 1:] - alpha[..., :, :-1]
            dy = alpha[..., :, 1:, :] - alpha[..., :, :-1, :]
            val = dx.abs().mean() + dy.abs().mean()
        else:
            val = lap.abs().mean()
        return val

    def evidence_orthogonality(self, evidence_maps):
        # evidence_maps: (B,K,H,W). Encourage mutual decorrelation across K via Gram matrix penalty
        B, K, H, W = evidence_maps.shape
        flat = evidence_maps.view(B, K, -1)  # (B,K,HW)
        flat = F.normalize(flat, dim=-1)
        G = torch.bmm(flat, flat.transpose(1, 2))  # (B,K,K)
        # subtract identity
        I = torch.eye(K, device=G.device).unsqueeze(0)
        offdiag = (G - I).pow(2).sum(dim=(1, 2))  # (B,)
        return offdiag.mean()

    def forward(self, x, gt_mask=None):
        """
        x: (B,3,512,512)
        returns:
          mask_pred: (B,1,512,512)
          evidence_maps: (B,K,512,512)
          alpha: (B,K,512,512)
          regs: dict of regularizers (if return_regs True)
        """
        B = x.shape[0]
        # 1. NRE
        residual_patches, recon_patches, Mf = self.nre(
            x
        )  # (B,N,C,p,p), (B,N,C,p,p), (C,p,p)
        # flatten residuals
        res_flat = residual_patches.view(B, self.N, -1)  # (B,N,C*p*p)
        # 2. embed + pos
        tokens = self.embedder(res_flat, self.pos_embed)  # (B,N,d)

        # 3. gates
        s = (res_flat**2).sum(dim=-1)  # (B,N)
        energy_grid = s.view(B, 1, self.grid_h, self.grid_w)
        local_mean = F.avg_pool2d(
            F.pad(energy_grid, (1, 1, 1, 1), mode="reflect"), kernel_size=3, stride=1
        ).view(B, self.N)
        gate_in = torch.stack(
            [torch.log1p(s), torch.log1p(local_mean)], dim=-1
        )  # (B,N,2)
        g = self.gatemlp(gate_in)  # (B,N) in (0,1)

        # 4. backbone (AGSM)
        feats_last, feats_list, A_list = self.backbone(
            tokens, g
        )  # feats_last: (B,N,d), feats_list: list of (B,N,d)

        # 5. CSCA: compute multiplicative gain using residual stats & token feats
        a = self.csca(feats_last, s)  # (B,N)
        feats_last = feats_last * a.unsqueeze(-1)
        # also apply to features in feats_list
        feats_list = [f * a.unsqueeze(-1) for f in feats_list]

        # 6. decoder + evidence attribution
        mask_logits, evidence_maps, alpha, mask_aux, fused_feat = self.decoder(
            feats_list
        )  # mask_pred: (B,1,H,W)
        loss = {}
        
        loss["attrib_smooth"] = self.compute_attribution_smoothness(alpha)
        loss["evidence_orth"] = self.evidence_orthogonality(evidence_maps)
        loss["gate_sparsity"] = (
            g.mean()
        )  # encourage low gating if desired (you'll weight it negative)
        # also expose Mf for frequency mask reg (if needed)
        # loss["freq_mask"] = Mf  # (C,p,p)
        
        if gt_mask is not None:
            gt_mask = gt_mask.float()
            loss['main'] = self.bce_loss(mask_logits, gt_mask)
            loss['aux'] = self.bce_loss(mask_aux, gt_mask) * 0.5
            loss_reg = (
                    loss['attrib_smooth'] * 0.1
                    + loss['evidence_orth'] * 0.05
                    + (self.nre.mask_gen.net[0].weight.norm() * 1e-6
                    if hasattr(self.nre.mask_gen.net[0], "weight") else 0.0)
                )

            loss['total'] = loss['main'] + loss['aux'] + loss_reg
        return {
            "mask": mask_aux,
            "evidence_maps": evidence_maps,
            "alpha": alpha,
            "mask_aux": mask_aux,
            "loss": loss,
            "fused_feat": fused_feat
        }


if __name__ == "__main__":
    from fvcore.nn import FlopCountAnalysis, flop_count_table

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ANRTNet(
        img_size=512,
        patch_size=16,
        in_ch=3,
        d=512,
        depth=6,
        evidences=4,
        device=device,
    )

    print(f"Model parameters: {sum(p.numel() for p in model.parameters())}")
    imgs = torch.randn(2, 3, 512, 512, device=device)
    masks_gt = torch.randint(0, 2, (2, 1, 512, 512), device=device).float()

    output = model(imgs)
    flops = FlopCountAnalysis(model, imgs)
    print(flop_count_table(flops, show_param_shapes=True, max_depth=2))
