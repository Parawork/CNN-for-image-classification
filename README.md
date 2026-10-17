# Tiny CNNs for Edge Devices: EN3150 Assignment 03

> How small can an image classifier get before it stops being useful?

We're **Group PatternX**. For this assignment we built two CNNs from scratch: a
standard one, and a slimmed-down version that fits in **under 64k parameters
(about 250 KB)**. We then put them up against two industry-standard lightweight
networks, MobileNetV2 and ShuffleNetV2. The question throughout is the trade-off
between **accuracy, memory and compute** when the target is a tiny device
rather than a GPU server.

| Index no. | Name |
|---|---|
| 220303E | Kariyawasam U.G.R.T. |
| 220370E | Maddumage M.M.T.D. |
| 220522A | Rathnayaka R.M.P.B. |
| 220525K | Rathnayake R.M.D.D. |

---

## The dataset: why EuroSAT?

**EuroSAT (RGB)** is 27,000 Sentinel-2 satellite patches covering 10 land-use classes
(forest, river, highway, residential, and so on). It suits the edge-device idea well.
Picture a small satellite or drone that has to decide *on board* what it's looking at,
instead of beaming every image to the ground. The images are natively 64×64, which is
exactly the assignment's limit, so we lose no resolution.

- **Split:** 70% train / 15% validation / 15% test, stratified, with a fixed seed.
- `data/splits.json` is committed, so every model, ours and the pre-trained ones, uses exactly the same images.

## The models at a glance

| Model | What it is | Params | MACs |
|---|---|---:|---:|
| **Model A** | 4 × (Conv 3×3 → BN → ReLU → MaxPool) → GAP → FC 64 → FC 10 | 405,994 | 60.2 M |
| **Model B** | Same skeleton, with depthwise separable convolutions | **63,659** | **7.3 M** |
| MobileNetV2 | ImageNet pre-trained, fine-tuned | ~2.2 M | – |
| ShuffleNetV2 ×1.0 | ImageNet pre-trained, fine-tuned | ~1.3 M | – |

Models A and B are identical except for the convolution type. That was deliberate: any
difference between them comes from the depthwise separable trick and nothing else.
The full layer-by-layer maths is in [docs/architecture.md](docs/architecture.md).

---

## Getting set up (once)

You need **Python 3.10+**. An NVIDIA GPU helps a lot but isn't required.

```powershell
# 1. Create and activate a virtual environment
python -m venv .venv
.venv\Scripts\activate                 # Windows
# source .venv/bin/activate            # Linux / macOS

# 2. Install PyTorch with GPU support (skip the --index-url part for CPU-only)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126

# 3. Install everything else
pip install -r requirements.txt

# 4. Sanity check: should print True and your GPU name
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

> **Windows gotcha:** make sure your prompt starts with `(.venv)` before running `pip install`.
> If it doesn't, PyTorch lands in the Microsoft Store Python folder, and you get
> `DLL load failed while importing _C`. Ask us how we know.

---

## Running it

### The short version

```powershell
python run_all.py --quick     # 2-epoch smoke test, a few minutes; checks everything works
python run_all.py             # the real thing, about 1-1.5 h on an RTX 3060 Ti
```

### All the commands

| What you want | Command |
|---|---|
| Run everything | `python run_all.py` |
| Quick smoke test (2 epochs) | `python run_all.py --quick` |
| Resume from a step | `python run_all.py --from 4` |
| Run a single step | `python run_all.py --only 3` |
| Delete all outputs (keeps the dataset) | `python run_all.py --clean` |
| Clean, then re-run from scratch | `python run_all.py --clean` then `python run_all.py` |
| See a step's options | `python scripts/04_train_custom.py --help` |

### Or step by step

| # | Command | Assignment task | Rough time* |
|---|---|---|---|
| 1 | `python scripts/01_prepare_data.py` | Data preparation | 2–5 min (first run only) |
| 2 | `python scripts/02_model_summary.py` | Architecture and parameter counts | seconds |
| 3 | `python scripts/03_optimizer_study.py` | Optimizer selection (7 runs) | 20–40 min |
| 4 | `python scripts/04_train_custom.py` | Train and evaluate A and B | 10–15 min |
| 5 | `python scripts/05_finetune_sota.py` | Fine-tune MobileNetV2 and ShuffleNetV2 | 15–25 min |
| 6 | `python scripts/06_compare.py` | Comparison tables and trade-off plot | about 1 min |

<sub>*Estimates for an RTX 3060 Ti. On a CPU, expect several times longer.</sub>

Useful flags: `--epochs N`, `--batch-size N` (drop to 32 if you hit `CUDA out of memory`),
`--models model_b`, and `--input-size 128` (step 5 only).

---

## What you get

Everything lands in `outputs/`:

```
outputs/
├── figures/       loss curves, confusion matrices, optimizer plots, accuracy-vs-size chart
├── results/       JSON metrics per model + comparison.md (tables ready for the report)
└── checkpoints/   trained weights; their file sizes are the "size on disk" numbers (not in git)
```

## Project layout

```
.
├── run_all.py            ← start here
├── scripts/              the six pipeline steps, numbered in run order
├── src/
│   ├── paths.py          every input/output folder in one place
│   ├── data.py           download, 64×64 cache, stratified split, augmentation
│   ├── models.py         Model A, Model B, MobileNetV2, ShuffleNetV2
│   ├── engine.py         training loop, metrics, plots
│   └── utils.py          seeding, param/MAC counting, size on disk, latency
├── docs/architecture.md  hand-worked parameter and MAC calculations
├── report/               report drafts, plus a map of which output feeds which section
├── data/                 dataset cache (git-ignored, except splits.json)
└── outputs/              everything the runs produce
```

## Troubleshooting

| Problem | Fix |
|---|---|
| `DLL load failed while importing _C` | PyTorch went into the wrong Python. Delete `.venv`, recreate it, activate it, and reinstall. |
| `torch.cuda.is_available()` is `False` | You installed the CPU build. Reinstall with the `--index-url .../cu126` line (or `cu124`). |
| `CUDA out of memory` | Add `--batch-size 32`. |
| EuroSAT download fails | The mirror is occasionally down. Wait and retry, or let us know. |
| A step crashed halfway | Fix it, then `python run_all.py --from <that step>`. |
