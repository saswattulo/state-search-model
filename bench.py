"""Speed: sequential scan vs signed SSD (training step), and signed vs stock positive SSD overhead.
  python bench.py --threads 12
"""
import argparse, statistics, time
import torch
from signed_ssd import sequential_scan, ssd_positive, ssd_signed
from test_signed_ssd import make


def step(f, *t):
    xs = [v.clone().requires_grad_() for v in t]
    f(*xs).sum().backward()


def timeit(fn, reps):
    fn()
    ts = []
    for _ in range(reps):
        t = time.perf_counter(); fn(); ts.append(time.perf_counter() - t)
    return statistics.median(ts)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--threads", type=int, default=torch.get_num_threads())
    ap.add_argument("--loop-max-len", type=int, default=2048, help="the loop gets very slow beyond this")
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    print(f"threads={args.threads}  b=4 heads=8 headdim=32 state=16, float32, forward+backward\n")
    print(f"{'L':>6} {'loop s':>9} {'signed SSD s':>13} {'speedup':>8}")
    for L in (512, 1024, 2048, 4096, 8192):
        x, a, B, C = make(4, L, 8, 32, 16, torch.float32)
        t2 = timeit(lambda: step(ssd_signed, x, a, B, C), 5)
        if L <= args.loop_max_len:
            t1 = timeit(lambda: step(sequential_scan, x, a, B, C), 1)
            print(f"{L:6d} {t1:9.3f} {t2:13.3f} {t1/t2:7.1f}x")
        else:
            print(f"{L:6d} {'skipped':>9} {t2:13.3f}")
    print(f"\n{'L':>6} {'stock SSD ms':>13} {'signed SSD ms':>14} {'overhead':>9}   (interleaved medians)")
    for L in (1024, 4096, 16384):
        x, a, B, C = make(2, L, 8, 32, 16, torch.float32)
        la = a.abs().log()
        ts = {"pos": [], "sgn": []}
        step(ssd_positive, x, la, B, C); step(ssd_signed, x, a, B, C)
        for i in range(12):
            pair = [("pos", lambda: step(ssd_positive, x, la, B, C)), ("sgn", lambda: step(ssd_signed, x, a, B, C))]
            for k, fn in (pair if i % 2 == 0 else pair[::-1]):
                t = time.perf_counter(); fn(); ts[k].append(time.perf_counter() - t)
        p, s = (statistics.median(ts[k]) * 1e3 for k in ("pos", "sgn"))
        print(f"{L:6d} {p:13.1f} {s:14.1f} {100*(s/p-1):8.1f}%")
