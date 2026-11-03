"""Run a grid of train.py jobs in parallel, one CPU thread per job, and print a summary table.

  python sweep.py --preset main           # the experiments from the write-up + the split variant
  python sweep.py --preset quick          # 2-minute smoke test
  python sweep.py --summary results.jsonl # just print the table for existing results

Results are appended to results.jsonl; jobs already in that file are skipped, so a sweep can
be interrupted and restarted.
"""
import argparse, itertools, json, os, subprocess, sys
from concurrent.futures import ThreadPoolExecutor, as_completed

HERE = os.path.dirname(os.path.abspath(__file__))
GB_PER_JOB = 1.5  # measured peak ~1.46 GB (16-head mamba2 rotate, eval at L=4096); most jobs need less

PRESETS = {
    # (arch, variant, init, heads) x seeds
    "quick": dict(grid=[("mamba2", "pos", "mamba", 4), ("mamba2", "signed", "mamba", 4)],
                  seeds=[0], extra=["--steps", "200", "--eval-lens", "64,256"]),
    "main": dict(grid=[
        ("mamba1", "pos", "naive", 4), ("mamba1", "signed", "naive", 4), ("mamba1", "split", "naive", 4),
        ("mamba2", "pos", "mamba", 4), ("mamba2", "signed", "mamba", 4), ("mamba2", "split", "mamba", 4),
        ("mamba2", "signed", "mamba", 16), ("mamba2", "split", "mamba", 16),
    ], seeds=[0, 1, 2], extra=[]),
    # the Mamba-2 half of "main" again, with Mamba-2's gated RMSNorm on the SSM output
    "outnorm": dict(grid=[
        ("mamba2", "pos", "mamba", 4), ("mamba2", "signed", "mamba", 4), ("mamba2", "split", "mamba", 4),
        ("mamba2", "signed", "mamba", 16), ("mamba2", "split", "mamba", 16),
    ], seeds=[0, 1, 2], extra=["--out-norm", "gated-rms"]),
    # six more seeds of the per-channel configs, with checkpoints for diagnose.py
    "seeds64": dict(grid=[("mamba2", "signed", "mamba", 64), ("mamba2", "split", "mamba", 64)],
                    seeds=[3, 4, 5, 6, 7, 8], extra=[], save=True),
    # unit-modulus rotations (Mamba-3 style), and one decay per channel (head dim 1)
    "rotation": dict(grid=[
        ("mamba2", "rotate", "mamba", 4), ("mamba2", "rotate", "mamba", 16),
        ("mamba2", "signed", "mamba", 64), ("mamba2", "split", "mamba", 64),
    ], seeds=[0, 1, 2], extra=[]),
}
# mamba1 uses a Python-loop scan, so evaluate it at shorter lengths to keep runtime sane
MAMBA1_EXTRA = ["--max-len", "40", "--min-len", "4", "--eval-lens", "40,128,512"]


def key(r):
    return (r["arch"], r["variant"], r["init"], r["heads"] if r["arch"] == "mamba2" else 4,
            r.get("out_norm", "none"), r["seed"])


def load(path):
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]


def available_ram_gb():
    """Free memory right now (MemAvailable), not total RAM: other apps may already be using most of it."""
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) / 2**20
    except OSError:
        pass
    try:
        return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_AVPHYS_PAGES") / 2**30
    except (ValueError, OSError, AttributeError):
        return 8.0


def summary(rows):
    groups = {}
    for r in rows:
        groups.setdefault(key(r)[:5], []).append(r)
    print(f"\n{'arch':7} {'variant':7} {'init':6} {'heads':>5} {'out_norm':9}  seeds  "
          "accuracy per eval length (min-max over seeds)")
    for g, rs in sorted(groups.items()):
        lens = sorted({int(L) for r in rs for L in r["acc"]})
        cells = []
        for L in lens:
            v = [r["acc"][str(L)] for r in rs if str(L) in r["acc"]]   # runs may use different eval lengths
            cells.append(f"L={L}: {min(v):.2f}-{max(v):.2f}" + (f" (n={len(v)})" if len(v) < len(rs) else ""))
        print(f"{g[0]:7} {g[1]:7} {g[2]:6} {g[3]:5} {g[4]:9}  {len(rs):5}  " + "   ".join(cells))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--preset", choices=list(PRESETS), default="main")
    ap.add_argument("--workers", type=int, default=None,
                    help=f"parallel jobs (default: min(cores, free RAM / {GB_PER_JOB} GB))")
    ap.add_argument("--out", default=os.path.join(HERE, "results.jsonl"))
    ap.add_argument("--summary", metavar="FILE", nargs="+", help="only print the summary table of FILE(s)")
    ap.add_argument("--device", default="cpu", help="cpu, cuda or mps; passed to every train.py job")
    args = ap.parse_args()

    if args.summary:
        summary([r for f in args.summary for r in load(f)]); return

    cores = os.cpu_count() or 1
    free_gb = available_ram_gb()
    workers = args.workers or max(1, min(cores, int(free_gb / GB_PER_JOB)))
    if args.device != "cpu" and not args.workers:
        workers = min(workers, 4)   # tiny models: a few processes share one GPU well

    p = PRESETS[args.preset]
    out_norm = p["extra"][p["extra"].index("--out-norm") + 1] if "--out-norm" in p["extra"] else "none"
    done = {key(r) for r in load(args.out)}
    jobs = []
    for (arch, variant, init, heads), seed in itertools.product(p["grid"], p["seeds"]):
        if (arch, variant, init, heads, out_norm, seed) in done:
            continue
        cmd = [sys.executable, os.path.join(HERE, "train.py"), "--arch", arch, "--variant", variant,
               "--init", init, "--heads", str(heads), "--seed", str(seed), "--threads", "1",
               "--out", args.out] + (MAMBA1_EXTRA if arch == "mamba1" else []) + p["extra"] \
              + (["--eval-micro", "8"] if heads >= 32 else []) \
              + ["--device", args.device]
        if p.get("save"):
            cmd += ["--save", os.path.join(HERE, "checkpoints", f"{arch}-{variant}-h{heads}-{out_norm}-s{seed}.pt")]
        jobs.append(((arch, variant, init, heads, out_norm, seed), cmd))
    print(f"{len(jobs)} jobs to run ({len(done)} already done), {workers} in parallel "
          f"on {cores} cores / {free_gb:.1f} GB free RAM, device {args.device}", flush=True)
    if workers < cores and not args.workers:
        print(f"  limited by free memory; close other apps or pass --workers N to override", flush=True)

    env = {**os.environ, "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}
    def run(job):
        name, cmd = job
        r = subprocess.run(cmd, capture_output=True, text=True, env=env)
        return name, r
    with ThreadPoolExecutor(workers) as ex:
        for fut in as_completed([ex.submit(run, j) for j in jobs]):
            name, r = fut.result()
            if r.returncode == 0:
                acc = json.loads(r.stdout.strip().splitlines()[-1])["acc"]
                print("done", name, " ".join(f"{k}:{v:.3f}" for k, v in acc.items()), flush=True)
            else:
                print("FAILED", name, r.stderr.strip().splitlines()[-1:] or r.returncode, flush=True)
    summary(load(args.out))


if __name__ == "__main__":
    main()
