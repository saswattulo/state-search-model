# state-search-model

Small, CPU-friendly experiments on selective state space models (Mamba-style).

- **The eigenvalue flaw.** Standard Mamba keeps every decay in (0, 1), so the model cannot solve
  parity. Allowing (−1, 1) fixes it (Grazzi et al., ICLR 2025).
- **Signed SSD.** A gauge transform that runs the (−1, 1) fix through Mamba-2's fast chunked SSD
  algorithm *unchanged*. The stock algorithm works in log-space and returns NaN for negative
  decays. With `s_t = Π sign(a_k)`: `SSD_signed(a, x) = s ⊙ SSD_positive(|a|, s ⊙ x)`.
  This is the rotation-by-π special case of Mamba-3's "RoPE trick" (see Related work).
- **Split** (new, not yet tested at scale): `a = tanh(g) · exp(−Δ·A)`. A separate sign gate lets
  "keep" (+1) and "flip" (−1) both hold |a| ≈ 1. This targets the length-extrapolation gap of the
  Mamba-2-style model.

The full explainer (theory, lineage, applications, results) is in
[`docs/explainer.html`](docs/explainer.html). Open it in a browser.

## Related work

- Grazzi et al., [Unlocking State-Tracking in Linear RNNs Through Negative Eigenvalues](https://arxiv.org/abs/2411.12537),
  ICLR 2025: the (−1, 1) fix and the parity result. They patched Mamba-1's CUDA scan by hand and
  left log-space (Mamba-2 style) kernels to future work.
- [Mamba-3](https://arxiv.org/abs/2603.15569), March 2026: complex (rotational) state updates run
  through the unmodified Mamba-2 SSD kernel by keeping the decay real and positive and applying the
  running product of rotations to B and C (Proposition 3, Eq. 10, the "RoPE trick"). Multiplying
  x and y by the running sign `s_t` is the same as multiplying B and C by it, so `ssd_signed` is
  that trick with every rotation equal to ±1. An earlier version of this repo did not credit this.
  What the sign version adds is narrow: it runs Grazzi et al.'s real (−1, 1) decays exactly, with
  integer signs and no extra state dimension.

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
| `--variant` | `pos` / `signed` / `split` / `rotate` | a ∈ (0,1) / `2·exp(−ΔA)−1` / `tanh(g)·exp(−ΔA)` / `exp(−ΔA)·R(θ)`: Mamba-3-style unit-modulus rotations of state-dim pairs with a free angle θ (mamba2 only) |
| `--init` | `naive` / `mamba` | dt bias 0 / Mamba-2 init (dt ~ logU[1e-3, 1e-1], A ~ U[1, 16]) |
| `--out-norm` | `none` / `gated-rms` | mamba2 only: no norm / Mamba-2's `RMSNorm(y · silu(z))` before the output projection |

`python sweep.py --preset outnorm` reruns the Mamba-2 half of `main` with `--out-norm gated-rms`.
`python sweep.py --preset rotation` runs `rotate` (4 and 16 heads) and per-channel decay
(`--heads 64`, head dim 1) for `signed` and `split`.

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
| `signed_ssd.py` | reference scans, Mamba-2 chunked SSD, signed SSD (sign gauge), rotary SSD (RoPE trick) |
| `models.py` | Mamba-1-style and Mamba-2-style layers with the three decay variants |
| `train.py` | train and evaluate one configuration; prints one JSON line |
| `sweep.py` | parallel grid runner and summary table |
| `test_signed_ssd.py` | correctness tests (forward, gradients, NaN check, rotation-by-π = sign flip) |
| `bench.py` | speed benchmarks |
| `docs/explainer.html` | the write-up |

## Results

`python sweep.py --preset main`, run locally (6-core CPU, 3 parallel jobs, ~75 min). Raw data in
[`results.jsonl`](results.jsonl). Final-position parity accuracy; 0.5 is chance; ranges are
min–max over 3 seeds.

**Mamba-1-style** (per-(channel, state) decay, sequential scan), trained on lengths 4–40:

| variant | L=40 (train) | L=128 (3×) | L=512 (13×) |
|---|---|---|---|
| pos | 0.43–0.54 | 0.44–0.47 | 0.45–0.54 |
| signed | **1.00** ×3 | 0.98–1.00 | 1.00, 1.00, 0.535 |
| split | **1.00** ×3 | **1.00** ×3 | **1.00** ×3 |

**Mamba-2-style** (one decay per head, chunked SSD), trained on lengths 4–64:

| variant | heads | L=64 (train) | L=256 (4×) | L=1024 (16×) | L=4096 (64×) |
|---|---|---|---|---|---|
| pos | 4 | 0.50–0.54 | 0.52–0.54 | 0.46–0.51 | 0.46–0.52 |
| signed | 4 | **1.00** ×3 | 1.00, 0.75, 0.46 | 0.47–0.55 | 0.48–0.54 |
| split | 4 | **1.00** ×3 | 1.00, 0.75, 0.61 | 0.45–0.54 | 0.50–0.55 |
| signed | 16 | **1.00** ×3 | 1.00, 1.00, 0.64 | 0.51–0.85 | 0.46–0.55 |
| split | 16 | **1.00** ×3 | **1.00** ×3 | 0.50–0.56 | 0.49–0.54 |

What this shows:

- Positive decays (standard Mamba) never learn parity, even at training length. Negative
  eigenvalues are required in both architectures.
- Mamba-1-style: `split` is the only variant perfect at 13× on every seed. `signed` failed on one
  seed. (An earlier cloud run had `signed` 3/3 at L=512 with a slightly different length setup:
  40 bits + BOS instead of 39 + BOS.)
- Mamba-2-style: every signed or split model learns parity in distribution, and more heads plus
  the split gate help at 4×, but **nothing generalises to 16×**. So "|a| cannot reach 1" (split)
  and "too few decays" (16 heads) are not the main cause.

Signed SSD vs the sequential loop, float64: relative error 5e-16, gradients within 6e-14.
Training step speedup over the loop on CPU: 16× at L=512, 57× at L=2048.

### Follow-up: Mamba-2's gated RMSNorm (`--preset outnorm`)

Hypothesis: the state's *sign* still tracks parity at long lengths but its *magnitude* decays,
so the signal sinks below the residual stream. Real Mamba-2 applies `RMSNorm(y · silu(z))` before
the output projection, which would make the readout depend only on the sign. The Mamba-2 half of
the grid was rerun with that norm (`--out-norm gated-rms`):

| variant | heads | perfect at L=256, no norm | with gated RMSNorm | L=1024, with norm |
|---|---|---|---|---|
| pos | 4 | 0/3 | 0/3 | 0.51–0.54 |
| signed | 4 | 1/3 | 0/3 (best 0.996) | 0.51–0.61 |
| split | 4 | 1/3 | 0/3 | 0.49–0.53 |
| signed | 16 | 2/3 | 0/3 (best 0.93) | 0.48–0.54 |
| split | 16 | **3/3** | 0/3 (best 0.84) | 0.47–0.56 |

**Refuted.** The norm does not help at 16× and makes 4× worse (7/12 → 0/12 perfect runs). Every
model still learns parity in distribution. Decay magnitude fading below the residual stream is
not the main cause.

### Follow-up: rotations and per-channel decay (`--preset rotation`)

Two more hypotheses for the Mamba-2-style extrapolation gap:

- **Flips shrink the state.** With `signed` and `split`, every flip has |a| < 1. The `rotate`
  variant (Mamba-3 style) flips by rotating a pair of state dims by θ = π with modulus exactly 1.
- **Decays are too coarse.** Mamba-1 has a separate decay for each of 64 × 8 = 512 (channel,
  state) pairs; Mamba-2 one per head. `--heads 64` (head dim 1) gives one decay per channel.

| variant | heads | L=64 (train) | L=256 (4×) | L=1024 (16×) | L=4096 (64×) |
|---|---|---|---|---|---|
| rotate | 4 | **1.00** ×3 | 0.46–0.54 | 0.47–0.55 | 0.45–0.52 |
| rotate | 16 | **1.00** ×3 | 0.52–0.54 | 0.50–0.55 | 0.45–0.51 |
| split | 64 | **1.00** ×3 | 1.00, 0.95, 0.55 | 0.46–0.50 | 0.52–0.57 |
| **signed** | **64** | **1.00** ×3 | **1.00, 1.00**, 0.93 | **1.00, 0.988**, 0.48 | 0.85, 0.53, 0.48 |

- **Per-channel decay is what helps `signed`.** No per-head `signed` or `split` run in the 27
  above got past 0.85 at 16×; with one decay per channel, 2 of 3 `signed` seeds are near-perfect
  at 16× and the best holds 0.85 at 64×. These are the first Mamba-2-style models here that
  extrapolate.
- **`split` does not benefit** from per-channel decay (0/3 at 16×).
- **Unit-modulus rotations extrapolate worst** (0/6 at 4×). Training only sees up to ~32 flips,
  so it does not pin the learned angle to exactly π, and angle error adds up with every flip.
  (Mamba-3 reports parity success with rotations; its training lengths, angle parameterisation
  and model size differ from this setup, so the two results are not directly comparable.)

### Open question

Why does finer decay granularity help `signed` but not `split`? Three seeds per config is thin,
and `signed`/`split` at 64 heads differ in parameter count (19.6k vs 23.8k). Next: more seeds for
the 64-head configs, and a diagnostic that reads the learned decays on 0-bits vs 1-bits in each
channel, to see which channels implement a clean flip (a ≈ −1 on 1-bits, a ≈ +1 on 0-bits) in the
models that extrapolate versus the ones that do not.

The explainer in `docs/explainer.html` is a snapshot of the earlier cloud runs.
