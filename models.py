"""Two selective-SSM families on a shared parity harness.

arch="mamba1": per-(channel, state) decay, sequential scan (as in the original Mamba).
arch="mamba2": one decay per head, computed with the chunked SSD algorithm (as in Mamba-2).

Decay parameterisations (variant):
  pos    a = exp(-dt * A)                        in (0, 1)    standard Mamba
  signed a = 2 * exp(-dt * A) - 1                in (-1, 1)   Grazzi et al. 2025
  split  a = tanh(g) * exp(-dt * A)              in (-1, 1)   separate sign gate g and magnitude;
         "keep" (a=+1) and "flip" (a=-1) both sit at dt -> 0, so |a| can stay at 1 for both.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from signed_ssd import ssd_positive, ssd_signed


def decay(log_mag, gate, variant):
    """log_mag = -dt*A (<= 0). Returns a real-valued a, or the log for the positive case."""
    if variant == "pos":
        return log_mag
    if variant == "signed":
        return 2 * log_mag.exp() - 1
    if variant == "split":
        return torch.tanh(gate) * log_mag.exp()
    raise ValueError(variant)


def mamba_dt_bias(n, lo=1e-3, hi=1e-1):
    dt0 = torch.exp(torch.empty(n).uniform_(math.log(lo), math.log(hi)))
    return dt0 + torch.log(-torch.expm1(-dt0))                          # inverse softplus


class Mamba1Layer(nn.Module):
    def __init__(self, d, variant, init, d_state=8, expand=2):
        super().__init__()
        di = expand * d
        self.variant, self.di, self.N = variant, di, d_state
        self.norm = nn.LayerNorm(d)
        self.in_proj = nn.Linear(d, 2 * di)
        extra = di if variant == "split" else 0                         # sign gate per channel
        self.x_proj = nn.Linear(di, 2 * d_state + di + extra)
        self.A_log = nn.Parameter(torch.log(torch.arange(1, d_state + 1).float()).repeat(di, 1))
        if init == "mamba":
            with torch.no_grad():
                self.x_proj.bias[2 * d_state:2 * d_state + di] = mamba_dt_bias(di)
        self.D = nn.Parameter(torch.ones(di))
        self.out_proj = nn.Linear(di, d)

    def forward(self, x):
        u, z = self.in_proj(self.norm(x)).chunk(2, -1)
        u = F.silu(u)
        parts = self.x_proj(u).split([self.N, self.N, self.di] + ([self.di] if self.variant == "split" else []), -1)
        B, C, dt = parts[:3]
        gate = parts[3].unsqueeze(-1) if self.variant == "split" else None
        dt = F.softplus(dt)
        log_mag = -dt.unsqueeze(-1) * self.A_log.exp()                  # (b,l,di,N)
        a = decay(log_mag, gate, self.variant)
        a = a.exp() if self.variant == "pos" else a
        bu = dt.unsqueeze(-1) * B.unsqueeze(2) * u.unsqueeze(-1)
        h = x.new_zeros(x.shape[0], self.di, self.N)
        ys = []
        for t in range(x.shape[1]):
            h = a[:, t] * h + bu[:, t]
            ys.append((h * C[:, t].unsqueeze(1)).sum(-1))
        y = torch.stack(ys, 1) + self.D * u
        return x + self.out_proj(y * F.silu(z))


class Mamba2Layer(nn.Module):
    def __init__(self, d, variant, init, heads=4, headdim=16, d_state=16, chunk=16):
        super().__init__()
        self.variant, self.h, self.p, self.n, self.chunk = variant, heads, headdim, d_state, chunk
        di = heads * headdim
        self.norm = nn.LayerNorm(d)
        self.extra = heads if variant == "split" else 0
        self.in_proj = nn.Linear(d, 2 * di + 2 * d_state + heads + self.extra)
        self.A_log = nn.Parameter(torch.zeros(heads))
        if init == "mamba":
            self.A_log.data = torch.log(torch.empty(heads).uniform_(1, 16))
            with torch.no_grad():
                self.in_proj.bias[2 * di + 2 * d_state:2 * di + 2 * d_state + heads] = mamba_dt_bias(heads)
        self.D = nn.Parameter(torch.ones(heads))
        self.out_proj = nn.Linear(di, d)

    def forward(self, u):
        b, l, _ = u.shape
        parts = self.in_proj(self.norm(u)).split(
            [self.h * self.p, self.h * self.p, self.n, self.n, self.h] + ([self.h] if self.extra else []), -1)
        x, z, B, C, dt = parts[:5]
        gate = parts[5] if self.extra else None
        x = F.silu(x).view(b, l, self.h, self.p)
        dt = F.softplus(dt)
        B = B[:, :, None].expand(b, l, self.h, self.n)
        C = C[:, :, None].expand(b, l, self.h, self.n)
        xdt = x * dt[..., None]
        log_mag = -dt * self.A_log.exp()
        if self.variant == "pos":
            y = ssd_positive(xdt, log_mag, B, C, self.chunk)
        else:
            y = ssd_signed(xdt, decay(log_mag, gate, self.variant), B, C, self.chunk)
        y = (y + x * self.D[:, None]).reshape(b, l, -1)
        return u + self.out_proj(y * F.silu(z))


class ParityModel(nn.Module):
    def __init__(self, arch, variant, init="mamba", d=32, layers=2, heads=4):
        super().__init__()
        self.emb = nn.Embedding(3, d)
        if arch == "mamba1":
            mk = lambda: Mamba1Layer(d, variant, init)
        elif arch == "mamba2":
            mk = lambda: Mamba2Layer(d, variant, init, heads=heads, headdim=64 // heads)
        else:
            raise ValueError(arch)
        self.layers = nn.ModuleList(mk() for _ in range(layers))
        self.head = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, 2))

    def forward(self, tok):
        x = self.emb(tok)
        for layer in self.layers:
            x = layer(x)
        return self.head(x)
