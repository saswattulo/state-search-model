"""Train one selective-SSM on running parity and evaluate length generalisation.

Example:
  python train.py --arch mamba2 --variant signed --seed 0
Prints one JSON line with the config and accuracies (also appended to --out if given).
"""
import argparse, json, time
import torch
import torch.nn as nn
import torch.nn.functional as F
from models import ParityModel


def batch(b, L, device):
    """BOS + (L-1) random bits; target at each position = parity of bits so far."""
    bits = torch.randint(0, 2, (b, L - 1), device=device)
    tok = torch.cat([torch.full((b, 1), 2, device=device), bits], 1)
    tgt = torch.cat([torch.zeros(b, 1, dtype=torch.long, device=device), bits.cumsum(1) % 2], 1)
    return tok, tgt


@torch.no_grad()
def evaluate(model, L, n, micro, device):
    """Accuracy of the final-position parity on n fresh sequences, in micro-batches to bound memory."""
    model.eval()
    correct = 0
    for i in range(0, n, micro):
        tok, tgt = batch(min(micro, n - i), L, device)
        correct += (model(tok)[:, -1].argmax(-1) == tgt[:, -1]).sum().item()
    model.train()
    return correct / n


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arch", choices=["mamba1", "mamba2"], default="mamba2")
    ap.add_argument("--variant", choices=["pos", "signed", "split", "rotate"], default="signed")
    ap.add_argument("--init", choices=["naive", "mamba"], default="mamba",
                    help="naive: dt bias 0 (what section 8 used); mamba: dt~logU[1e-3,1e-1], A~U[1,16]")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--heads", type=int, default=4, help="mamba2 only; head dim = 64 / heads")
    ap.add_argument("--out-norm", choices=["none", "gated-rms"], default="none",
                    help="mamba2 only; gated-rms = Mamba-2's RMSNorm(y * silu(z)) before out_proj")
    ap.add_argument("--d-model", type=int, default=32)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--min-len", type=int, default=4)
    ap.add_argument("--max-len", type=int, default=64)
    ap.add_argument("--eval-lens", default="64,256,1024,4096")
    ap.add_argument("--eval-n", type=int, default=256)
    ap.add_argument("--eval-micro", type=int, default=32, help="eval micro-batch; lower it if RAM is tight")
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--device", default="cpu", help="cpu, cuda or mps")
    ap.add_argument("--out", default=None, help="append the JSON result line to this file")
    args = ap.parse_args(argv)

    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    dev = torch.device(args.device)
    model = ParityModel(args.arch, args.variant, args.init, args.d_model, args.layers, args.heads,
                        args.out_norm).to(dev)
    opt = torch.optim.AdamW(model.parameters(), args.lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.steps)

    t0 = time.time()
    for step in range(args.steps):
        L = int(torch.randint(args.min_len, args.max_len + 1, ()))
        tok, tgt = batch(args.batch, L, dev)
        loss = F.cross_entropy(model(tok).reshape(-1, 2), tgt.reshape(-1))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sched.step()
    train_s = time.time() - t0

    acc = {int(L): evaluate(model, int(L), args.eval_n, args.eval_micro, dev) for L in args.eval_lens.split(",")}
    res = {**{k: v for k, v in vars(args).items() if k not in ("out",)},
           "params": sum(p.numel() for p in model.parameters()),
           "final_loss": round(loss.item(), 4), "train_s": round(train_s, 1),
           "total_s": round(time.time() - t0, 1), "acc": acc}
    line = json.dumps(res)
    print(line, flush=True)
    if args.out:
        with open(args.out, "a") as f:
            f.write(line + "\n")
    return res


if __name__ == "__main__":
    main()
