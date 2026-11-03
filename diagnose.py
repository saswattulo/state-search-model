"""Read the learned decays of saved Mamba-2-style parity models and relate them to extrapolation.

  python diagnose.py checkpoints/*.pt

Each model sees fresh random bit strings; every head's real decay a_t is recorded per token.
A "flip head" has the parity sign pattern on every token: a > 0 on 0-bits and a < 0 on 1-bits.
Its decay length L_e = 1 / (-mean log|a_t|) is the number of tokens over which that head's signal
shrinks by a factor of e. Per layer the table shows how many heads are flip heads (sign right on
>= 99.9% of tokens) and the flip head with the longest L_e, with its mean decay on 0s and 1s.
One row per model, sorted by accuracy at the longest evaluated length.

Rule under test (fixed on 2026-11-03 from seeds 3-6, before seeds 7-8 and the split runs finished):
a model passes length L (accuracy >= 0.95) iff L <= RATIO * max L_e over all flip heads, RATIO = 5.
The last column prints how many evaluated lengths the rule gets right for that model.
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
def head_stats(model, L, n):
    """Per layer, per head: mean a on 0-bits, mean a on 1-bits, sign accuracy, decay length L_e."""
    tok, _ = batch(n, L, "cpu")
    for layer in model.layers:
        layer.record = True
    model(tok)
    bits = tok[:, 1:]                                                   # drop BOS
    out = []
    for layer in model.layers:
        a = layer.recorded[:, 1:]                                       # (n, L-1, heads)
        want_neg = (bits == 1)[..., None]
        sign_ok = torch.where(want_neg, a < 0, a > 0).float().flatten(0, 1).mean(0)
        log_mag = a.abs().clamp_min(1e-30).log().flatten(0, 1).mean(0)
        out.append(dict(a0=a[bits == 0].mean(0), a1=a[bits == 1].mean(0), sign_ok=sign_ok,
                        L_e=-1 / log_mag.clamp_max(-1e-9)))
        layer.record = False
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("checkpoints", nargs="+")
    ap.add_argument("--len", type=int, default=256, help="sequence length used to read the decays")
    ap.add_argument("--n", type=int, default=64, help="number of sequences")
    ap.add_argument("--sign-acc", type=float, default=0.999, help="sign accuracy that makes a head a flip head")
    ap.add_argument("--ratio", type=float, default=5.0, help="rule: pass at L iff L <= ratio * max L_e")
    ap.add_argument("--pass-acc", type=float, default=0.95, help="accuracy that counts as passing a length")
    args = ap.parse_args()
    torch.manual_seed(0)

    rows = []
    for path in args.checkpoints:
        model, c, acc = load(path)
        if c["arch"] != "mamba2" or c["variant"] not in ("signed", "split"):
            print(f"skipping {path}: only mamba2 signed/split models have real signed decays")
            continue
        layers = []
        for st in head_stats(model, args.len, args.n):
            flip = st["sign_ok"] >= args.sign_acc
            if flip.any():
                best = int(torch.where(flip, st["L_e"], torch.zeros_like(st["L_e"])).argmax())
                layers.append(dict(n=int(flip.sum()), L_e=st["L_e"][best].item(),
                                   a0=st["a0"][best].item(), a1=st["a1"][best].item()))
            else:
                layers.append(dict(n=0, sign=st["sign_ok"].max().item()))
        rows.append((c, acc, layers))

    if not rows:
        return
    longest = max(max(acc) for _, acc, _ in rows)
    rows.sort(key=lambda r: -r[1].get(longest, 0))
    lens = sorted({L for _, acc, _ in rows for L in acc})
    print(f"\ndecays read at L={args.len} on {args.n} sequences; flip head = sign right on >= {args.sign_acc:.1%} of tokens")
    print(f"{'variant':7} {'heads':>5} {'seed':>4}  " + " ".join(f"{'L=' + str(L):>7}" for L in lens)
          + "   per layer: flip heads | longest-lived flip head: L_e (a|0, a|1)   rule")
    right = total = 0
    for c, acc, layers in rows:
        cells = " ".join(f"{acc.get(L, float('nan')):7.3f}" for L in lens)
        lay = "   ".join(f"{d['n']:2d} | L_e {d['L_e']:7.1f} ({d['a0']:+.3f}, {d['a1']:+.3f})" if d["n"]
                         else f"{0:2d} | none (best sign acc {d['sign']:.3f})" + " " * 6 for d in layers)
        max_le = max((d["L_e"] for d in layers if d["n"]), default=0.0)
        hits = [(L <= args.ratio * max_le) == (a >= args.pass_acc) for L, a in acc.items()]
        right += sum(hits); total += len(hits)
        print(f"{c['variant']:7} {c['heads']:5} {c['seed']:4}  {cells}   {lay}   {sum(hits)}/{len(hits)}")
    print(f"\nrule 'pass at L iff L <= {args.ratio:g} x max L_e' matches {right}/{total} (model, length) cells")


if __name__ == "__main__":
    main()
