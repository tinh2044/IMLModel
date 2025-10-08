import torch
import torch.nn as nn
import torch.nn.functional as F


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
        b, c, h, w = x.shape
        heads, _, _ = self.heads, self.key_dim, self.value_dim

        context = x if context is None else context

        q, k, v = self.to_q(x), self.to_k(context), self.to_v(context)
        q, k, v = map(lambda t: t.reshape(b, heads, -1, h * w), (q, k, v))  # [B,h,d,N]

        q = self.norm_q(q.transpose(-1, -2)).transpose(-1, -2)
        k = self.norm_k(k.transpose(-1, -2)).transpose(-1, -2)
        v = self.norm_v(v.transpose(-1, -2)).transpose(-1, -2)

        q, k = map(lambda x: F.softmax(x * (self.key_dim**-0.25), dim=-2), (q, k))
        k = F.softmax(k, dim=-1)

        context = torch.einsum("bhdn,bhen->bhde", k, v)  # context matrix
        out = torch.einsum("bhdn,bhde->bhen", q, context)
        out = out.reshape(b, -1, h, w)
        out = self.to_out(out)
        return out


class DCFS(nn.Module):
    def __init__(self, dim, heads=8, key_dim=64, value_dim=64, ff_dim=None):
        super().__init__()
        self.dim = dim
        ff_dim = ff_dim or dim * 2

        # Projection for both streams
        self.freq_proj = nn.Conv2d(dim, dim, 1)
        self.spat_proj = nn.Conv2d(dim, dim, 1)

        # Linear attention both directions
        self.cross_f2s = LinearAttention(
            dim, chan_out=dim, key_dim=key_dim, value_dim=value_dim, heads=heads
        )
        self.cross_s2f = LinearAttention(
            dim, chan_out=dim, key_dim=key_dim, value_dim=value_dim, heads=heads
        )

        # Feed-forward blocks
        self.ffn_s = nn.Sequential(
            nn.Conv2d(dim, ff_dim, 1), nn.GELU(), nn.Conv2d(ff_dim, dim, 1)
        )
        self.ffn_f = nn.Sequential(
            nn.Conv2d(dim, ff_dim, 1), nn.GELU(), nn.Conv2d(ff_dim, dim, 1)
        )

        # Gating
        self.gate = nn.Conv2d(dim * 2, dim, 1)
        self.norm = nn.BatchNorm2d(dim)
        self.gamma = nn.Parameter(torch.ones(1))

    def forward(self, x_freq, x_spat):
        # Project
        f = self.freq_proj(x_freq)
        s = self.spat_proj(x_spat)

        # Cross attention
        s_out = self.cross_f2s(s, context=f)  # F -> S
        f_out = self.cross_s2f(f, context=s)  # S -> F

        # FFN refinement
        s_ref = s_out + self.ffn_s(self.norm(s_out))
        f_ref = f_out + self.ffn_f(self.norm(f_out))

        # Gating fusion
        g = torch.sigmoid(self.gate(torch.cat([s_ref, f_ref], dim=1)))
        y = g * s_ref + (1 - g) * f_ref

        out = self.norm(x_spat + self.gamma * y)
        return out


if __name__ == "__main__":
    from fvcore.nn import FlopCountAnalysis, flop_count_table

    x_freq = torch.randn(1, 512, 64, 64)
    x_spat = torch.randn(1, 512, 64, 64)

    model = DCFS(dim=512)
    print(flop_count_table(FlopCountAnalysis(model, (x_freq, x_spat)), max_depth=5))
