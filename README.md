# EN3150 Assignment 03: Resource-Constrained CNN for Edge Image Classification

**Group: PatternX**

| Index no. | Name |
|---|---|
| 220303E | Kariyawasam U.G.R.T. |
| 220370E | Maddumage M.M.T.D. |
| 220522A | Rathnayaka R.M.P.B. |
| 220525K | Rathnayake R.M.D.D. |

## Dataset

**EuroSAT (RGB)**: Sentinel-2 satellite patches, 27,000 images, 10 land-use classes.
Images are used at **64 × 64 × 3** (the maximum the assignment allows) and split
**70 / 15 / 15** (train / val / test), stratified with a fixed seed.
The split indices are stored in `data/splits.json`, so every model uses exactly the same images.

## Models

| Model | Description | Params |
|---|---|---:|
| Model A | 4 × (Conv3×3–BN–ReLU–MaxPool) → GAP → FC64 → FC10 | 405,994 |
| Model B | Same, with depthwise separable convs | 63,659 |
| MobileNetV2 | ImageNet pre-trained, fine-tuned | ~2.2 M |
| ShuffleNetV2 x1.0 | ImageNet pre-trained, fine-tuned | ~1.3 M |

The layer-by-layer parameter and MAC calculations, and the reasons for the activation
choices, are in [docs/architecture.md](docs/architecture.md).

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows  (Linux/macOS: source .venv/bin/activate)
pip install -r requirements.txt  # for CUDA, install torch from pytorch.org first
```

## Running (in order)

| Step | Command | Assignment task | Outputs |
|---|---|---|---|
| 1 | `python prepare_data.py` | 1. Data preparation | `results/dataset_summary.json`, `figures/dataset_samples.png` |
| 2 | `python model_summary.py` | 2. Architecture / parameter counts | layer tables, `results/model_summary.json` |
| 3 | `python optimizer_study.py` | 3. Optimizer selection | `figures/optim_*.png`, `results/optimizer_study_model_b.json` |
| 4 | `python train_custom.py` | 4. Train/evaluate A & B | `figures/model_*_loss.png`, `*_confusion.png`, `results/model_*.json` |
| 5 | `python finetune_sota.py` | 5. Fine-tune SOTA | `results/mobilenet_v2.json`, `results/shufflenet_v2.json` |
| 6 | `python compare.py` | 4 & 6. Comparison tables | `results/comparison.md`, `figures/accuracy_vs_size.png` |

Every script accepts `--epochs` and `--batch-size`; run `python <script> --help` to see all options.
A GPU (or Google Colab) is recommended for steps 3 to 5. Step 3 trains 7 runs × 20 epochs.

## Repository layout

```
src/data.py      dataset download, 64x64 caching, stratified split, loaders
src/models.py    Model A, Model B, MobileNetV2, ShuffleNetV2
src/engine.py    training loop, metrics, plots, experiment runner
src/utils.py     seeding, parameter / MAC counting, size on disk, latency
docs/            architecture and parameter calculations
```
