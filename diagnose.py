"""Read the learned decays of saved Mamba-2-style parity models and relate them to extrapolation.

  python diagnose.py checkpoints/*.pt

Each model sees fresh random bit strings; every head's real decay a_t is recorded and averaged
separately over 0-bit and 1-bit tokens. A clean "flip head" keeps on 0-bits (a ~ +1) and flips on
1-bits (a ~ -1); its flip error is max(1 - mean a|0, 1 + mean a|1), 0 for a perfect flip.
One row per model, sorted by accuracy at the longest evaluated length.
"""
import argparse
import torch
from models import ParityModel
from train import batch


def load(path):
    ck = torch.load(path, map_location="cpu")
    c = ck["config"]
    model = ParityModel(c["arch"], c["variant"], c["init"], c["d_model"], c["layers"], c["heads"],
                        c.get("out_norm", "none"))
    model.load_state_dict(ck["state_dict"])
    return model.eval(), c, {int(k): v for k, v in ck["acc"].items()}


@torch.no_grad()
def head_decays(model, L, n):
    """Per layer: (mean a on 0-bits, mean a on 1-bits), each of shape (heads,)."""
    tok, _ = batch(n, L, "cpu")
    for layer in model.layers:
        layer.record = True
    model(tok)
    bits = tok[:, 1:]                                                   # drop BOS
    out = []
    for layer in model.layers:
        a = layer.recorded[:, 1:]                                       # (n, L-1, heads)
        out.append((a[bits == 0].mean(0), a[bits == 1].mean(0)))
        layer.record = False
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("checkpoints", nargs="+")
    ap.add_argument("--len", type=int, default=256, help="sequence length used to read the decays")
    ap.add_argument("--n", type=int, default=64, help="number of sequences")
    ap.add_argument("--clean", type=float, default=0.05, help="flip error below this counts as a clean flip head")
    args = ap.parse_args()
    torch.manual_seed(0)

    rows = []
    for path in args.checkpoints:
        model, c, acc = load(path)
        if c["arch"] != "mamba2" or c["variant"] not in ("signed", "split"):
            print(f"skipping {path}: only mamba2 signed/split models have real signed decays")
            continue
        layers = []
        for a0, a1 in head_decays(model, args.len, args.n):
            err = torch.maximum(1 - a0, 1 + a1)
            best = int(err.argmin())
            layers.append(dict(clean=int((err < args.clean).sum()), err=err[best].item(),
                               a0=a0[best].item(), a1=a1[best].item()))
        rows.append((c, acc, layers))

    if not rows:
        return
    longest = max(max(acc) for _, acc, _ in rows)
    rows.sort(key=lambda r: -r[1].get(longest, 0))
    lens = sorted({L for _, acc, _ in rows for L in acc})
    print(f"\ndecays read at L={args.len} on {args.n} sequences; clean = heads with flip error < {args.clean}")
    print(f"{'variant':7} {'heads':>5} {'seed':>4}  " + " ".join(f"{'L=' + str(L):>7}" for L in lens)
          + "   per layer: clean heads | best head (a|0, a|1) err")
    for c, acc, layers in rows:
        cells = " ".join(f"{acc.get(L, float('nan')):7.3f}" for L in lens)
        lay = "   ".join(f"{d['clean']:3d} | ({d['a0']:+.3f}, {d['a1']:+.3f}) {d['err']:.3f}" for d in layers)
        print(f"{c['variant']:7} {c['heads']:5} {c['seed']:4}  {cells}   {lay}")


if __name__ == "__main__":
    main()
