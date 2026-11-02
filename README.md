# state-search-model

Small, CPU-friendly experiments on selective state space models (Mamba-style).

- **The eigenvalue flaw.** Standard Mamba keeps every decay in (0, 1), so the model cannot solve
  parity. Allowing (−1, 1) fixes it (Grazzi et al., ICLR 2025).
- **Signed SSD.** A gauge transform that runs the (−1, 1) fix through Mamba-2's fast chunked SSD
  algorithm *unchanged*. The stock algorithm works in log-space and returns NaN for negative
  decays. With `s_t = Π sign(a_k)`: `SSD_signed(a, x) = s ⊙ SSD_positive(|a|, s ⊙ x)`.
- **Split** (new, not yet tested at scale): `a = tanh(g) · exp(−Δ·A)`. A separate sign gate lets
  "keep" (+1) and "flip" (−1) both hold |a| ≈ 1. This targets the length-extrapolation gap of the
  Mamba-2-style model.

The full explainer (theory, lineage, applications, results) is in
[`docs/explainer.html`](docs/explainer.html). Open it in a browser.

## Setup (once)

Needs Python 3.9+. CPU-only PyTorch is enough.

```bash
python3 -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -r requirements.txt --index-url https://download.pytorch.org/whl/cpu
python test_signed_ssd.py            # 4 PASS lines, ~5 s
```

With an NVIDIA GPU you can install the default CUDA build (`pip install torch`) and pass
`--device cuda` to `train.py`. For models this small, the parallel CPU sweep is usually the
better use of the machine.

## Run

```bash
python sweep.py --preset quick       # smoke test, a few minutes
python sweep.py --preset main        # full grid: 8 configs × 3 seeds = 24 jobs
python sweep.py --summary results.jsonl
```

`sweep.py` runs one single-threaded job per worker. It picks the worker count from **free**
memory (about 1.3 GB per job at peak), capped at the core count, and prints the number it chose.
If other apps are using most of your RAM it will run fewer jobs at once; close them or pass
`--workers N`. Results are appended to `results.jsonl`, and finished jobs are skipped on restart,
so you can stop and resume a sweep at any time.

## Single runs

```bash
python train.py --arch mamba2 --variant signed --seed 0
python train.py --arch mamba2 --variant split  --seed 0 --heads 16
python train.py --arch mamba1 --variant signed --init naive --max-len 40 --eval-lens 40,128,512
python train.py --help                # steps, lengths, lr, d-model, layers, device, ...
```

| flag | choices | meaning |
|---|---|---|
| `--arch` | `mamba1` / `mamba2` | per-(channel, state) decay with a sequential scan / one decay per head with chunked SSD |
| `--variant` | `pos` / `signed` / `split` | a ∈ (0,1) / `2·exp(−ΔA)−1` / `tanh(g)·exp(−ΔA)` |
| `--init` | `naive` / `mamba` | dt bias 0 / Mamba-2 init (dt ~ logU[1e-3, 1e-1], A ~ U[1, 16]) |

Training uses lengths `--min-len`..`--max-len` (default 4–64). Evaluation reports final-position
parity accuracy at each of `--eval-lens`; 0.5 is chance.

## Benchmark

```bash
python bench.py --threads 12
```

Times a training step (sequential scan vs signed SSD) and the overhead of signed vs stock
positive SSD. The loop baseline is PyTorch autograd over a Python loop, so very large speedups at
long L mostly measure loop overhead, not a GPU kernel.

## Files

| file | contents |
|---|---|
| `signed_ssd.py` | reference scan, Mamba-2 chunked SSD, the gauge-transformed signed SSD |
| `models.py` | Mamba-1-style and Mamba-2-style layers with the three decay variants |
| `train.py` | train and evaluate one configuration; prints one JSON line |
| `sweep.py` | parallel grid runner and summary table |
| `test_signed_ssd.py` | correctness tests (forward, gradients, NaN check) |
| `bench.py` | speed benchmarks |
| `docs/explainer.html` | the write-up |

## Results so far (4-core cloud CPU, 3 seeds each)

| model | train len | ~3–4× | ~13–16× |
|---|---|---|---|
| mamba1 pos (naive init) | 0.52–0.57 | 0.50–0.52 | 0.49–0.55 |
| mamba1 signed (naive init) | **1.00** | **1.00** | **1.00** |
| mamba2 pos (mamba init) | 0.48–0.52 | 0.48–0.54 | 0.45–0.49 |
| mamba2 signed (mamba init) | **1.00** | 1.00, 1.00, 0.51 | 0.47–0.52 |

Signed SSD vs the sequential loop, float64: relative error 5e-16, gradients within 6e-14.
Training step speedup over the loop on CPU: 16× at L=512, 57× at L=2048.

Open question: why does the per-head (Mamba-2) model extrapolate worse? The `split` variant and
`--heads 16` runs in the `main` preset test two hypotheses.
