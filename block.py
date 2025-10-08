import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fft as fft


class ConvBNAct(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, s=1, p=1):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, k, s, p)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.GELU()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class LinearAttention(nn.Module):
    def __init__(
        self, chan, chan_out=None, key_dim=64, value_dim=64, heads=8, norm_queries=True
    ):
        super().__init__()
        self.chan = chan
        chan_out = chan if chan_out is None else chan_out
        self.key_dim = key_dim
        self.value_dim = value_dim
        self.heads = heads
        self.norm_queries = norm_queries

        self.to_q = nn.Conv2d(chan, key_dim * heads, 1)
        self.to_k = nn.Conv2d(chan, key_dim * heads, 1)
        self.to_v = nn.Conv2d(chan, value_dim * heads, 1)
        self.to_out = nn.Conv2d(value_dim * heads, chan_out, 1)

        self.norm_q = nn.LayerNorm(key_dim)
        self.norm_k = nn.LayerNorm(key_dim)
        self.norm_v = nn.LayerNorm(value_dim)

    def forward(self, x, context=None):
        """
        x: [B, C, H, W]
        context: [B, C, H, W] (same spatial H,W)
        returns out: [B, C_out, H, W]
        """
        b, c, h, w = x.shape
        heads, k_dim, v_dim = self.heads, self.key_dim, self.value_dim

        context = x if context is None else context

        q = self.to_q(x)  # [B, heads*k_dim, H, W]
        k = self.to_k(context)
        v = self.to_v(context)

        # reshape to [B, heads, k_dim, N]
        q = q.view(b, heads, k_dim, h * w)
        k = k.view(b, heads, k_dim, h * w)
        v = v.view(b, heads, v_dim, h * w)

        # layernorm on last dim (feature dim) -> apply over k_dim / v_dim
        # torch LayerNorm expects last dim, so transpose to [..., feature]
        q = self.norm_q(q.permute(0, 1, 3, 2)).permute(0, 1, 3, 2).contiguous()
        k = self.norm_k(k.permute(0, 1, 3, 2)).permute(0, 1, 3, 2).contiguous()
        v = self.norm_v(v.permute(0, 1, 3, 2)).permute(0, 1, 3, 2).contiguous()

        # normalize queries/keys as in linear attention trick
        q = F.softmax(q * (self.key_dim**-0.25), dim=2)  # softmax over feature dim
        k = F.softmax(k, dim=3)  # softmax over spatial dim
        # context matrix: shape [B, heads, k_dim, v_dim]
        context_mat = torch.einsum("bhkn,bhvn->bhkv", k, v)
        # out: [B, heads, k_dim, N]
        out = torch.einsum("bhkn,bhkv->bhvn", q, context_mat)
        # reshape to [B, heads*v_dim, H, W]
        out = out.reshape(b, -1, h, w)
        out = self.to_out(out)
        return out


class FSDCore(nn.Module):
    """
    Frequency-Spatial Decomposition core using LinearAttention on patch-grid maps.

    - in_ch: total input channels (concatenated sources)
    - embed_dim: D
    - patch_sizes: list of p_s
    - strides: list of stride r_s
    - freq_bands: R
    """

    def __init__(
        self, in_ch, embed_dim=64, patch_sizes=(8, 4), strides=(4, 2), freq_bands=4
    ):
        super().__init__()
        assert len(patch_sizes) == len(strides)
        self.D = embed_dim
        self.patch_sizes = list(patch_sizes)
        self.strides = list(strides)
        self.num_stages = len(self.patch_sizes)
        self.R = freq_bands

        # project to D
        self.proj = nn.Conv2d(in_ch, self.D, kernel_size=1)

        # per-stage modules
        self.freq_fcs = nn.ModuleList()
        self.spa_fcs = nn.ModuleList()
        self.attns = nn.ModuleList()
        self.reconstructs = nn.ModuleList()

        for p in self.patch_sizes:
            self.freq_fcs.append(
                nn.Sequential(
                    nn.Linear(self.D * self.R + 2 * self.D, self.D),
                    nn.GELU(),
                    nn.Linear(self.D, self.D),
                )
            )
            spa_in_dim = self.D * (p * p)
            self.spa_fcs.append(
                nn.Sequential(
                    nn.Linear(spa_in_dim, self.D), nn.GELU(), nn.Linear(self.D, self.D)
                )
            )
            # use LinearAttention operating on maps [B, D, Hn, Wn]
            self.attns.append(
                LinearAttention(
                    self.D, chan_out=self.D, key_dim=32, value_dim=32, heads=4
                )
            )
            # reconstruction: predict real+imag delta flattened per patch token
            self.reconstructs.append(nn.Linear(self.D, 2 * self.D * p * p))

        self.refine = ConvBNAct(self.D, self.D, 3, 1, 1)

    def _compute_grid_size(self, H_full, W_full, p, r):
        # number of patches along height and width
        Hn = (H_full - p) // r + 1
        Wn = (W_full - p) // r + 1
        return Hn, Wn

    def forward(self, x):
        """
        x: [B, C_in, H, W]
        returns: [B, D, H, W]
        """
        B, _, H_full, W_full = x.shape
        F_prev = self.proj(x)  # [B, D, H, W]

        for s in range(self.num_stages):
            p = self.patch_sizes[s]
            r = self.strides[s]
            attn_layer = self.attns[s]
            freq_fc = self.freq_fcs[s]
            spa_fc = self.spa_fcs[s]
            rec_lin = self.reconstructs[s]

            # Unfold to patches: [B, D*p*p, N]
            patches = F.unfold(F_prev, kernel_size=p, stride=r)  # [B, D*p*p, N]
            N = patches.shape[-1]
            if N == 0:
                continue

            # compute grid dims Hn, Wn (so N = Hn * Wn)
            Hn, Wn = self._compute_grid_size(H_full, W_full, p, r)
            assert Hn * Wn == N, f"grid mismatch: {Hn}x{Wn} != {N}"

            # reshape patches -> [B, N, D, p, p]
            patches = patches.transpose(1, 2).contiguous().view(B, N, self.D, p, p)

            # FFT per patch-channel -> complex tensor [B,N,D,p,p]
            fft_patches = fft.fft2(patches, dim=(-2, -1))
            mag = torch.abs(fft_patches)
            ph = torch.angle(fft_patches)

            # normalize magnitude per patch-channel
            denom = torch.sqrt((mag**2).mean(dim=(-2, -1), keepdim=True) + 1e-8)
            mag_n = mag / denom

            # pooling into R bands (simple tiling along freq dims)
            freq_tokens = []
            step = max(1, p // self.R)
            for r_idx in range(self.R):
                u0 = r_idx * step
                u1 = min(p, (r_idx + 1) * step)
                region = mag_n[:, :, :, u0:u1, u0:u1]  # [B,N,D,bs,bs]
                pooled = region.mean(dim=(-2, -1))  # [B,N,D]
                freq_tokens.append(pooled)
            freq_vec = torch.cat(freq_tokens, dim=-1)  # [B,N, D*R]

            # phase summary: cos & sin pooled
            cos_mean = torch.cos(ph).mean(dim=(-2, -1))  # [B,N,D]
            sin_mean = torch.sin(ph).mean(dim=(-2, -1))  # [B,N,D]
            phase_token = torch.cat([cos_mean, sin_mean], dim=-1)  # [B,N, 2D]

            # freq embedding: [B,N, D]
            freq_input = torch.cat([freq_vec, phase_token], dim=-1)
            freq_emb = freq_fc(freq_input)  # [B,N,D]

            # spatial token embedding from raw patch content
            spa_flat = patches.reshape(B, N, self.D * p * p)  # [B,N, D*p*p]
            spa_emb = spa_fc(spa_flat)  # [B,N,D]

            # reshape tokens to maps [B, D, Hn, Wn] for LinearAttention
            # spa_emb: [B,N,D] -> [B,D,Hn,Wn]
            spa_map = spa_emb.permute(0, 2, 1).contiguous().view(B, self.D, Hn, Wn)
            freq_map = freq_emb.permute(0, 2, 1).contiguous().view(B, self.D, Hn, Wn)

            # Cross-attention: spatial(query) asks frequency(context)
            O_spa_map = attn_layer(spa_map, context=freq_map)  # [B, D, Hn, Wn]
            # Cross-attention: frequency(query) asks spatial(context)
            O_freq_map = attn_layer(freq_map, context=spa_map)  # [B, D, Hn, Wn]

            # flatten back to tokens [B, N, D]
            O_spa = O_spa_map.view(B, self.D, -1).permute(0, 2, 1).contiguous()
            O_freq = O_freq_map.view(B, self.D, -1).permute(0, 2, 1).contiguous()

            # combine
            O = 0.5 * (O_spa + O_freq)  # [B,N,D]

            # reconstruct complex delta per patch token
            delta_all = rec_lin(O)  # [B,N, 2*D*p*p]
            delta_all = delta_all.view(B, N, 2, self.D, p, p)  # [B,N,2,D,p,p]
            delta_real = delta_all[:, :, 0]  # [B,N,D,p,p]
            delta_imag = delta_all[:, :, 1]  # [B,N,D,p,p]
            delta_complex = torch.complex(delta_real, delta_imag)

            # update FFT coefficients
            fft_new = fft_patches + delta_complex

            # inverse FFT -> spatial patches
            patches_ifft = fft.ifft2(fft_new, dim=(-2, -1)).real  # [B,N,D,p,p]

            # fold back to map [B, D, H_full, W_full]
            patches_ifft_flat = (
                patches_ifft.view(B, N, self.D * p * p).transpose(1, 2).contiguous()
            )
            F_new = F.fold(
                patches_ifft_flat, output_size=(H_full, W_full), kernel_size=p, stride=r
            )

            # coverage normalization
            ones = torch.ones((B, 1, H_full, W_full), device=x.device, dtype=x.dtype)
            cover = F.fold(
                F.unfold(ones, kernel_size=p, stride=r),
                output_size=(H_full, W_full),
                kernel_size=p,
                stride=r,
            )
            cover = cover.clamp(min=1.0)
            F_new = F_new / cover

            # residual update
            F_prev = F_prev + F_new

        out = self.refine(F_prev)
        return out  # [B, D, H, W]


class FSD4(nn.Module):
    def __init__(self, c4, D=64):
        super().__init__()
        self.block = FSDCore(
            in_ch=c4, embed_dim=D, patch_sizes=(8, 4), strides=(4, 2), freq_bands=4
        )

    def forward(self, f4):
        return self.block(f4)


class FSD3(nn.Module):
    def __init__(self, c3, D=64):
        super().__init__()
        # input will be cat(fsd4_up, f3) -> channel = D + c3
        self.block = FSDCore(
            in_ch=c3 + D, embed_dim=D, patch_sizes=(8, 4), strides=(4, 2), freq_bands=4
        )

    def forward(self, fsd4_up, f3):
        x = torch.cat([fsd4_up, f3], dim=1)
        return self.block(x)


class FSD2(nn.Module):
    def __init__(self, c2, D=64):
        super().__init__()
        # input will be cat(fsd4_up, fsd3_up, f2) -> channel = 2*D + c2
        self.block = FSDCore(
            in_ch=c2 + 2 * D,
            embed_dim=D,
            patch_sizes=(8, 4),
            strides=(4, 2),
            freq_bands=4,
        )

    def forward(self, fsd4_up, fsd3_up, f2):
        x = torch.cat([fsd4_up, fsd3_up, f2], dim=1)
        return self.block(x)


class FSD1(nn.Module):
    def __init__(self, c1, D=64):
        super().__init__()
        # input cat(fsd4_up, fsd3_up, fsd2_up, f1) -> channel = 3*D + c1
        self.block = FSDCore(
            in_ch=c1 + 3 * D,
            embed_dim=D,
            patch_sizes=(16, 8, 4),
            strides=(8, 4, 2),
            freq_bands=4,
        )

    def forward(self, fsd4_up, fsd3_up, fsd2_up, f1):
        x = torch.cat([fsd4_up, fsd3_up, fsd2_up, f1], dim=1)
        return self.block(x)


if __name__ == "__main__":
    device = torch.device("cpu")
    B = 1
    f4 = torch.randn(B, 512, 16, 16, device=device)
    f3 = torch.randn(B, 256, 32, 32, device=device)
    f2 = torch.randn(B, 128, 64, 64, device=device)
    f1 = torch.randn(B, 64, 128, 128, device=device)

    D = 64
    fsd4_m = FSD4(512, D=D).to(device)
    fsd3_m = FSD3(256, D=D).to(device)
    fsd2_m = FSD2(128, D=D).to(device)
    fsd1_m = FSD1(64, D=D).to(device)

    out4 = fsd4_m(f4)  # [B, D, 16, 16]
    out4_up = F.interpolate(out4, size=(32, 32), mode="bilinear", align_corners=False)

    out3 = fsd3_m(out4_up, f3)  # [B, D, 32, 32]
    out3_up = F.interpolate(out3, size=(64, 64), mode="bilinear", align_corners=False)
    out4_up2 = F.interpolate(out4, size=(64, 64), mode="bilinear", align_corners=False)

    out2 = fsd2_m(out4_up2, out3_up, f2)  # [B, D, 64, 64]
    out2_up = F.interpolate(out2, size=(128, 128), mode="bilinear", align_corners=False)
    out3_up2 = F.interpolate(
        out3, size=(128, 128), mode="bilinear", align_corners=False
    )
    out4_up3 = F.interpolate(
        out4, size=(128, 128), mode="bilinear", align_corners=False
    )

    out1 = fsd1_m(out4_up3, out3_up2, out2_up, f1)  # [B, D, 128, 128]
    print("out1.shape", out1.shape)
