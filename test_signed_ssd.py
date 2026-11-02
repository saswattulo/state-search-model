"""Correctness checks for the gauge-transformed SSD. Run: python test_signed_ssd.py  (or pytest)."""
import torch
import math
from signed_ssd import sequential_scan, sequential_scan_rotary, ssd_positive, ssd_rotary, ssd_signed


def make(b, l, h, p, n, dtype, seed=0):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(b, l, h, p, generator=g, dtype=dtype)
    B = torch.randn(b, l, h, n, generator=g, dtype=dtype) / n ** 0.5
    C = torch.randn(b, l, h, n, generator=g, dtype=dtype) / n ** 0.5
    mag = torch.rand(b, l, h, generator=g, dtype=dtype) * 0.25 + 0.75
    sign = torch.where(torch.rand(b, l, h, generator=g) < 0.5, -1.0, 1.0).to(dtype)
    return x, mag * sign, B, C


def test_forward_matches_loop():
    for l in (1024, 1000):                                  # 1000: not a multiple of the chunk size
        x, a, B, C = make(2, l, 4, 8, 16, torch.float64)
        ref = sequential_scan(x, a, B, C)
        rel = ((ssd_signed(x, a, B, C) - ref).norm() / ref.norm()).item()
        assert rel < 1e-12, rel


def test_gradients_match_loop():
    x, a, B, C = make(1, 200, 2, 4, 8, torch.float64)
    grads = []
    for f in (sequential_scan, ssd_signed):
        xs = [t.clone().requires_grad_() for t in (x, a, B, C)]
        f(*xs).pow(2).sum().backward()
        grads.append([t.grad for t in xs])
    for g1, g2 in zip(*grads):
        assert (g1 - g2).abs().max().item() < 1e-10


def test_positive_case_unchanged():
    x, a, B, C = make(2, 256, 4, 8, 16, torch.float64)
    a = a.abs()
    assert torch.allclose(ssd_signed(x, a, B, C), ssd_positive(x, a.log(), B, C))


def test_stock_ssd_breaks_on_negative():
    x, a, B, C = make(1, 128, 2, 4, 8, torch.float32)
    assert torch.isnan(ssd_positive(x, torch.log(a), B, C)).any()


def make_rotary(b, l, h, p, n, dtype, seed=0):
    x, a, B, C = make(b, l, h, p, n, dtype, seed)
    g = torch.Generator().manual_seed(seed + 1)
    theta = (torch.rand(b, l, h, n // 2, generator=g, dtype=dtype) * 2 - 1) * math.pi
    return x, a.abs().log(), theta, B, C


def test_rotary_matches_loop():
    x, la, th, B, C = make_rotary(2, 1000, 4, 8, 16, torch.float64)
    ref = sequential_scan_rotary(x, la, th, B, C)
    rel = ((ssd_rotary(x, la, th, B, C) - ref).norm() / ref.norm()).item()
    assert rel < 1e-12, rel


def test_rotary_gradients_match_loop():
    x, la, th, B, C = make_rotary(1, 200, 2, 4, 8, torch.float64)
    grads = []
    for f in (sequential_scan_rotary, ssd_rotary):
        xs = [t.clone().requires_grad_() for t in (x, la, th, B, C)]
        f(*xs).pow(2).sum().backward()
        grads.append([t.grad for t in xs])
    for g1, g2 in zip(*grads):
        assert (g1 - g2).abs().max().item() < 1e-10


def test_rotation_by_pi_equals_sign_flip():
    """theta = pi on every pair is a = -|a|, so ssd_rotary must agree with ssd_signed."""
    x, la, th, B, C = make_rotary(2, 256, 4, 8, 16, torch.float64)
    flip = torch.rand(la.shape, generator=torch.Generator().manual_seed(3)) < 0.5
    th = (flip[..., None].to(la.dtype) * math.pi).expand_as(th)   # float64 pi; a float32 pi drifts
    a = torch.where(flip, -la.exp(), la.exp())
    assert torch.allclose(ssd_rotary(x, la, th, B, C), ssd_signed(x, a, B, C), atol=1e-10)


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn(); print("PASS", name)
