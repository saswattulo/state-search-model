"""Signed SSD: run negative-eigenvalue SSMs through an unmodified positive-only (log-space) SSD kernel.

Recurrence (Mamba-2 style, scalar decay per head):
    h_t = a_t * h_{t-1} + x_t B_t^T          h: (P x N),  a_t in (-1, 1)
    y_t = h_t C_t

Gauge lemma. Let s_t = prod_{k<=t} sign(a_k) in {+1,-1} and h~_t = s_t h_t. Then
    h~_t = |a_t| h~_{t-1} + (s_t x_t) B_t^T        (because s_t a_t = |a_t| s_{t-1})
    y_t  = s_t (h~_t C_t)
so  SSD_signed(a, x) = s * SSD_positive(|a|, s * x).
Matrix view: the 1-semiseparable mask factorises as L_signed = diag(s) L_|a| diag(s).

Credit: this is the rotation-by-pi case of Mamba-3's "RoPE trick" (arXiv 2603.15569, Prop. 3 / Eq. 10).
s_t x_t B_t^T = x_t (s_t B_t)^T and s_t (h~_t C_t) = h~_t (s_t C_t): the running sign is applied to B and C.
"""
import torch
import torch.nn.functional as F


def sequential_scan(x, a, B, C):
    """Reference recurrence. x (b,l,h,p), a (b,l,h) any real, B,C (b,l,h,n) -> y (b,l,h,p)."""
    b, l, h, p = x.shape
    st = x.new_zeros(b, h, p, B.shape[-1])
    ys = []
    for t in range(l):
        st = a[:, t, :, None, None] * st + x[:, t, :, :, None] * B[:, t, :, None, :]
        ys.append((st * C[:, t, :, None, :]).sum(-1))
    return torch.stack(ys, 1)


def segsum(x):
    """x (..., T) -> (..., T, T): out[i,j] = sum_{k=j+1..i} x_k for i >= j, -inf above the diagonal."""
    T = x.size(-1)
    xc = x.cumsum(-1)
    out = xc[..., :, None] - xc[..., None, :]
    mask = torch.ones(T, T, dtype=torch.bool, device=x.device).tril()
    return out.masked_fill(~mask, float("-inf"))


def ssd_positive(x, log_a, B, C, chunk=64):
    """Mamba-2 chunked SSD (the "ssd_minimal" algorithm). log_a must be the log of a POSITIVE decay.
    x (b,l,h,p), log_a (b,l,h), B,C (b,l,h,n). Any l: the tail is zero-padded to a multiple of chunk."""
    b, l, h, p = x.shape
    pad = (-l) % chunk
    if pad:  # causal, so padding at the end does not change the first l outputs
        x, B, C = (F.pad(t, (0, 0, 0, 0, 0, pad)) for t in (x, B, C))
        log_a = F.pad(log_a, (0, 0, 0, pad))
    L = l + pad
    c = L // chunk
    X = x.reshape(b, c, chunk, h, p)
    Bc = B.reshape(b, c, chunk, h, -1)
    Cc = C.reshape(b, c, chunk, h, -1)
    A = log_a.reshape(b, c, chunk, h).permute(0, 3, 1, 2)             # (b,h,c,Q)
    A_cum = A.cumsum(-1)
    # 1) intra-chunk: attention-like quadratic form, all matmuls
    scores = torch.einsum("bclhn,bcshn->bhcls", Cc, Bc) * torch.exp(segsum(A))
    Y_diag = torch.einsum("bhcls,bcshp->bclhp", scores, X)
    # 2) state at the end of each chunk
    decay_states = torch.exp(A_cum[..., -1:] - A_cum)
    states = torch.einsum("bclhn,bhcl,bclhp->bchpn", Bc, decay_states, X)
    # 3) pass states between chunks
    states = torch.cat([torch.zeros_like(states[:, :1]), states], 1)
    decay_chunk = torch.exp(segsum(F.pad(A_cum[..., -1], (1, 0))))
    states = torch.einsum("bhzc,bchpn->bzhpn", decay_chunk, states)[:, :-1]
    # 4) contribution of the carried-in state
    Y_off = torch.einsum("bclhn,bchpn,bhcl->bclhp", Cc, states, torch.exp(A_cum))
    return (Y_diag + Y_off).reshape(b, L, h, p)[:, :l]


def ssd_signed(x, a, B, C, chunk=64):
    """Negative-eigenvalue SSD via the gauge transform. a: (b,l,h) real, typically in (-1, 1)."""
    neg = (a < 0).to(torch.int64)
    s = (1 - 2 * (neg.cumsum(1) % 2)).to(x.dtype)                      # running sign, exact
    log_abs = torch.log(a.abs().clamp_min(torch.finfo(x.dtype).tiny))
    return ssd_positive(x * s[..., None], log_abs, B, C, chunk) * s[..., None]
