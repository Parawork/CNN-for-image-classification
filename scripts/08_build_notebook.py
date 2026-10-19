"""Build the standalone submission notebook from the repo's src/ + scripts/ code."""
import nbformat as nbf
from pathlib import Path

C = []          # (kind, source)


def md(s):
    C.append(("md", s.strip("\n")))


def code(s):
    C.append(("code", s.strip("\n")))


md(r"""
# EN3150 Assignment 03 - Resource-Constrained CNN for Edge Image Classification

**Group PatternX**

| Index no. | Name |
|---|---|
| 220303E | Kariyawasam U.G.R.T. |
| 220370E | Maddumage M.M.T.D. |
| 220522A | Rathnayaka R.M.P.B. |
| 220525K | Rathnayake R.M.D.D. |

**Repository:** https://github.com/Parawork/Pattern

---

This notebook is the complete, self-contained version of our code. Running it top to
bottom reproduces every number, table and figure in the report:

| Section | Assignment task | Marks |
|---|---|---|
| 1. Data preparation | Task 1 | - |
| 2. Architecture design (Model A / Model B) | Task 2 | 20 |
| 3. Optimizer selection & tuning | Task 3 | 15 |
| 4. Training & evaluation of the custom models | Task 4 | 25 |
| 5. Fine-tuning lightweight SOTA networks | Task 5 | 20 |
| 6. Final comparison & trade-off discussion | Task 6 | 20 |

**Dataset:** EuroSAT (RGB) - 27,000 Sentinel-2 satellite patches, 10 land-use classes,
natively 64x64 (the maximum resolution the assignment allows). Not CIFAR-10.

> **Before you run:** section 3 trains 7 models and sections 4-5 train 4 more.
> On an RTX 3060 Ti the whole notebook takes about 1-1.5 hours. Set `QUICK = True`
> in the configuration cell for a 2-epoch smoke test that finishes in a few minutes.
""")

md("## 0. Setup\n\nInstall requirements (skip if the environment is already prepared):")

code(r"""
# Run once if torch / torchvision / scikit-learn are not already installed.
# For a CUDA build use e.g.:
#   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
# %pip install torch torchvision scikit-learn matplotlib numpy
""")

code(r"""
import json, os, random, time, copy
from collections import Counter
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             precision_score, recall_score)
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from torchvision import models as tvm
from torchvision.transforms import functional as TF

%matplotlib inline
print("torch", torch.__version__, "| CUDA available:", torch.cuda.is_available())
""")

md("""
### Configuration

Everything that controls the run lives here. `QUICK = True` shortens every training run to
2 epochs so the whole notebook can be checked end to end in a few minutes; the numbers in
the report come from `QUICK = False`.
""")

code(r'''
QUICK = False           # True -> 2-epoch smoke test

SEED        = 42
IMG_SIZE    = 64        # assignment limit: max 64 x 64
BATCH_SIZE  = 64        # drop to 32 if you hit CUDA out of memory
NUM_CLASSES = 10

EPOCHS_OPTIM  = 2 if QUICK else 20   # section 3: optimizer study (7 runs)
EPOCHS_CUSTOM = 2 if QUICK else 30   # section 4: Model A and Model B (>= 20 required)
EPOCHS_SOTA   = 2 if QUICK else 20   # section 5: MobileNetV2 and ShuffleNetV2

ROOT     = Path.cwd()
DATA_DIR = ROOT / "data";           DATA_DIR.mkdir(parents=True, exist_ok=True)
OUT_DIR  = ROOT / "outputs";        OUT_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR  = OUT_DIR / "figures";     FIG_DIR.mkdir(parents=True, exist_ok=True)
RES_DIR  = OUT_DIR / "results";     RES_DIR.mkdir(parents=True, exist_ok=True)
CKPT_DIR = OUT_DIR / "checkpoints"; CKPT_DIR.mkdir(parents=True, exist_ok=True)

CACHE_FILE = DATA_DIR / "eurosat64.pt"   # 64x64 uint8 cache of the whole dataset
SPLIT_FILE = DATA_DIR / "splits.json"    # fixed 70/15/15 indices, shared by every model

# ImageNet statistics: required by the pre-trained networks and harmless for the custom
# ones, so one normalisation is used everywhere and the splits stay directly comparable.
MEAN = (0.485, 0.456, 0.406)
STD  = (0.229, 0.224, 0.225)


def set_seed(seed=SEED):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def sync(device):
    """Wait for queued GPU work so wall-clock timings are honest."""
    if device.type == "cuda":
        torch.cuda.synchronize()


def save_json(obj, path):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


DEVICE = get_device()
set_seed()
print("device:", DEVICE, "| quick mode:", QUICK)
''')

# ---------------------------------------------------------------- Task 1
md(r"""
---
## 1. Data preparation  *(Task 1)*

**Why EuroSAT?** It is a labelled Sentinel-2 land-cover dataset (AnnualCrop, Forest,
HerbaceousVegetation, Highway, Industrial, Pasture, PermanentCrop, Residential, River,
SeaLake) published for exactly the kind of problem an edge device faces: a small
satellite or drone that must decide *on board* what it is looking at instead of
transmitting every frame to the ground. The images are natively 64x64 RGB, so we meet
the assignment's resolution limit without discarding any information.

**Pipeline**

1. Download once, resize to 64x64 and cache the whole dataset as a `uint8` tensor
   (~330 MB in RAM - small enough to keep in memory, which removes disk I/O from the
   training-time measurements).
2. Stratified **70 / 15 / 15** split with a fixed seed, written to `data/splits.json`.
   Every model in this notebook - ours and the pre-trained ones - sees exactly the same
   train / validation / test images.
3. Augmentation on the training split only: random horizontal/vertical flips and 90-degree
   rotations. Satellite imagery has no canonical orientation, so these are
   label-preserving and essentially free to compute.
""")

code(r'''
def load_eurosat():
    """Download EuroSAT once, downscale to 64x64 and cache as a uint8 tensor.

    Returns (images[N,3,64,64] uint8, labels[N] int64, class_names).
    """
    if CACHE_FILE.exists():
        cache = torch.load(CACHE_FILE)
        return cache["images"], cache["labels"], cache["classes"]

    print("First run: downloading EuroSAT (~90 MB) and caching it at 64x64 - happens only once.")
    ds = datasets.EuroSAT(root=str(DATA_DIR), download=True)
    resize = transforms.Resize((IMG_SIZE, IMG_SIZE), antialias=True)

    images = torch.empty((len(ds), 3, IMG_SIZE, IMG_SIZE), dtype=torch.uint8)
    labels = torch.empty(len(ds), dtype=torch.long)
    for i in range(len(ds)):
        img, y = ds[i]
        images[i] = TF.pil_to_tensor(resize(img.convert("RGB")))
        labels[i] = y

    torch.save({"images": images, "labels": labels, "classes": ds.classes}, CACHE_FILE)
    return images, labels, ds.classes


def make_splits(labels, seed=SEED):
    """Stratified 70/15/15 split, created once and then re-loaded from disk."""
    if SPLIT_FILE.exists():
        with open(SPLIT_FILE) as f:
            return json.load(f)

    idx, y = list(range(len(labels))), labels.tolist()
    train_idx, temp_idx = train_test_split(idx, test_size=0.30, stratify=y, random_state=seed)
    temp_y = [y[i] for i in temp_idx]
    val_idx, test_idx = train_test_split(temp_idx, test_size=0.50, stratify=temp_y,
                                         random_state=seed)

    splits = {"train": sorted(train_idx), "val": sorted(val_idx), "test": sorted(test_idx)}
    with open(SPLIT_FILE, "w") as f:
        json.dump(splits, f)
    return splits


class EuroSATTensorDataset(Dataset):
    """In-memory dataset.

    augment    : random flips / 90-degree rotations (training split only).
    input_size : the custom CNNs use 64. The pre-trained networks can optionally be fed an
                 upsampled copy (e.g. 128); the information content is still that of the
                 64x64 sensor image.
    """

    def __init__(self, images, labels, augment=False, input_size=IMG_SIZE):
        self.images, self.labels = images, labels
        self.augment, self.input_size = augment, input_size
        self.normalize = transforms.Normalize(MEAN, STD)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        x = self.images[i].float() / 255.0
        if self.augment:
            if torch.rand(1).item() < 0.5:
                x = torch.flip(x, dims=[2])
            if torch.rand(1).item() < 0.5:
                x = torch.flip(x, dims=[1])
            x = torch.rot90(x, k=int(torch.randint(0, 4, (1,))), dims=[1, 2])
        if self.input_size != IMG_SIZE:
            x = TF.resize(x, [self.input_size, self.input_size], antialias=True)
        return self.normalize(x), self.labels[i]


def get_dataloaders(batch_size=BATCH_SIZE, input_size=IMG_SIZE, augment=True, num_workers=0):
    """Return ({'train','val','test'} -> DataLoader, class_names)."""
    images, labels, classes = load_eurosat()
    splits = make_splits(labels)

    loaders = {}
    for name, idx in splits.items():
        idx = torch.tensor(idx)
        ds = EuroSATTensorDataset(images[idx], labels[idx],
                                  augment=augment and name == "train",
                                  input_size=input_size)
        loaders[name] = DataLoader(ds, batch_size=batch_size, shuffle=(name == "train"),
                                   num_workers=num_workers,
                                   pin_memory=torch.cuda.is_available())
    return loaders, classes
''')

md("Build the splits and print the per-class table (this is what goes into the dataset sheet):")

code(r'''
images, labels, CLASSES = load_eurosat()
splits = make_splits(labels)

print(f"Dataset   : EuroSAT RGB, {len(labels)} images, {len(CLASSES)} classes")
print(f"Image size: {IMG_SIZE}x{IMG_SIZE}x3\n")

summary = {"classes": CLASSES, "total": len(labels), "image_size": IMG_SIZE, "splits": {}}
counts = {s: Counter(labels[idx].tolist()) for s, idx in splits.items()}

header = f"{'class':<22}" + "".join(f"{s:>8}" for s in splits)
print(header + "\n" + "-" * len(header))
for c, name in enumerate(CLASSES):
    print(f"{name:<22}" + "".join(f"{counts[s][c]:>8}" for s in splits))
print("-" * len(header))
print(f"{'TOTAL':<22}" + "".join(f"{len(idx):>8}" for idx in splits.values()))

for s, idx in splits.items():
    summary["splits"][s] = {"n": len(idx), "fraction": len(idx) / len(labels),
                            "per_class": [counts[s][c] for c in range(len(CLASSES))]}
save_json(summary, RES_DIR / "dataset_summary.json")
''')

code(r'''
# One example image per class
fig, axes = plt.subplots(2, 5, figsize=(12, 5))
for c, ax in enumerate(axes.flat):
    i = (labels == c).nonzero()[0].item()
    ax.imshow(images[i].permute(1, 2, 0).numpy())
    ax.set_title(CLASSES[c], fontsize=9)
    ax.axis("off")
fig.suptitle("EuroSAT samples at 64x64")
fig.tight_layout()
fig.savefig(FIG_DIR / "dataset_samples.png", dpi=150)
plt.show()
''')

code(r'''
loaders, CLASSES = get_dataloaders(batch_size=BATCH_SIZE)
{k: len(v.dataset) for k, v in loaders.items()}
''')

# ---------------------------------------------------------------- Task 2
md(r"""
---
## 2. Custom architecture design  *(Task 2 - 20 marks)*

Both models share the **same skeleton**:

```
4 x [ conv block -> MaxPool 2x2 ]   with channels 3 -> 32 -> 64 -> 128 -> 256
GlobalAveragePool -> FC 64 -> ReLU -> Dropout(0.3) -> FC 10 (logits)
```

and differ **only** in the conv block:

| | Model A (standard) | Model B (lightweight) |
|---|---|---|
| Block | Conv 3x3 -> BN -> ReLU | DWConv 3x3 -> BN -> ReLU -> PWConv 1x1 -> BN -> ReLU |

Keeping everything else identical is deliberate: any difference in parameters, MACs,
latency or accuracy between A and B is attributable to the depthwise separable
factorisation and to nothing else.

**Design choices and their hardware justification**

* **ReLU everywhere.** `max(0, x)` is a single comparison - no exponential, no division,
  no lookup table. Sigmoid, tanh, Swish and GELU all need `exp()`, which is slow or
  software-emulated on microcontrollers and DSPs without an FPU. ReLU also maps exactly
  onto int8 arithmetic (zero stays zero, no re-scaling), can be fused into the conv/BN
  kernel, and its gradient does not saturate. ReLU6 would be a good alternative since its
  bounded range makes int8 quantisation easier.
* **No softmax at inference.** Softmax is used only inside the cross-entropy loss during
  training. On the device the prediction is `argmax` of the logits - the same answer at
  no extra cost.
* **Convolutions have no bias**, because the following BatchNorm supplies its own shift.
* **BatchNorm is free at inference**: it folds into the preceding convolution's weights
  and bias once training is finished.
* **Global average pooling instead of Flatten.** Flatten would give a 4*4*256 = 4,096-wide
  vector, and a 64-unit FC on top of that alone costs 262k weights - Model B would blow
  its 100k budget in the classifier. GAP needs only 256 inputs.
* **Max-pooling** is comparisons only - no multiplications - and it shrinks the activation
  memory (RAM) the next layer needs, which on a microcontroller is usually the binding
  constraint rather than flash.
* **Channel widths 32-64-128-256** are multiples of 8/16, which map cleanly onto SIMD
  lanes and NPU tile sizes.
""")

code(r'''
CHANNELS = (32, 64, 128, 256)
FC_UNITS = 64
DROPOUT  = 0.3


class ConvBNReLU(nn.Sequential):
    """Standard 3x3 convolution. Bias omitted because BatchNorm adds its own shift."""

    def __init__(self, cin, cout):
        super().__init__(
            nn.Conv2d(cin, cout, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(cout),
            nn.ReLU(inplace=True),
        )


class DSConvBNReLU(nn.Sequential):
    """Depthwise separable convolution (MobileNetV1 style).

    depthwise : one 3x3 filter per input channel (groups=cin) -> 9*cin weights
    pointwise : 1x1 convolution mixing channels              -> cin*cout weights
    versus a standard 3x3 conv with 9*cin*cout weights.
    """

    def __init__(self, cin, cout):
        super().__init__(
            nn.Conv2d(cin, cin, kernel_size=3, padding=1, groups=cin, bias=False),
            nn.BatchNorm2d(cin),
            nn.ReLU(inplace=True),
            nn.Conv2d(cin, cout, kernel_size=1, bias=False),
            nn.BatchNorm2d(cout),
            nn.ReLU(inplace=True),
        )


class CustomCNN(nn.Module):
    def __init__(self, block, num_classes=NUM_CLASSES, channels=CHANNELS,
                 fc_units=FC_UNITS, dropout=DROPOUT):
        super().__init__()
        layers, cin = [], 3
        for cout in channels:
            # spatial resolution 64 -> 32 -> 16 -> 8 -> 4
            layers += [block(cin, cout), nn.MaxPool2d(2)]
            cin = cout
        self.features = nn.Sequential(*layers)
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(cin, fc_units),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(fc_units, num_classes),   # logits; softmax only inside the loss
        )

    def forward(self, x):
        return self.classifier(self.features(x))


def model_a(num_classes=NUM_CLASSES):
    return CustomCNN(ConvBNReLU, num_classes)


def model_b(num_classes=NUM_CLASSES):
    return CustomCNN(DSConvBNReLU, num_classes)


def mobilenet_v2(num_classes=NUM_CLASSES, pretrained=True):
    weights = tvm.MobileNet_V2_Weights.IMAGENET1K_V1 if pretrained else None
    m = tvm.mobilenet_v2(weights=weights)
    m.classifier[1] = nn.Linear(m.last_channel, num_classes)
    return m


def shufflenet_v2(num_classes=NUM_CLASSES, pretrained=True):
    weights = tvm.ShuffleNet_V2_X1_0_Weights.IMAGENET1K_V1 if pretrained else None
    m = tvm.shufflenet_v2_x1_0(weights=weights)
    m.fc = nn.Linear(m.fc.in_features, num_classes)
    return m


def head_parameters(name, model):
    """Parameters of the newly added classifier head (trained with a higher LR)."""
    return model.classifier.parameters() if name == "mobilenet_v2" else model.fc.parameters()


BUILDERS = {"model_a": model_a, "model_b": model_b,
            "mobilenet_v2": mobilenet_v2, "shufflenet_v2": shufflenet_v2}
''')

md("""
### Counting parameters, MACs, size on disk and CPU latency

MACs are counted with forward hooks:

* `Conv2d`: `H_out * W_out * C_out * (C_in / groups) * k_h * k_w`
* `Linear`: `in_features * out_features`

BatchNorm, ReLU and pooling are ignored - they are negligible, and BatchNorm disappears
into the convolution at inference time.
""")

code(r'''
def count_params(model, trainable_only=True):
    return sum(p.numel() for p in model.parameters() if p.requires_grad or not trainable_only)


@torch.no_grad()
def count_macs(model, input_size=IMG_SIZE):
    """Multiply-accumulate operations for one image, counted with forward hooks."""
    macs = 0

    def conv_hook(m, _inp, out):
        nonlocal macs
        kh, kw = m.kernel_size
        macs += (out.shape[2] * out.shape[3] * m.out_channels
                 * (m.in_channels // m.groups) * kh * kw)

    def linear_hook(m, _inp, _out):
        nonlocal macs
        macs += m.in_features * m.out_features

    hooks = []
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            hooks.append(m.register_forward_hook(conv_hook))
        elif isinstance(m, nn.Linear):
            hooks.append(m.register_forward_hook(linear_hook))

    was_training = model.training
    model.eval()
    device = next(model.parameters()).device
    model(torch.zeros(1, 3, input_size, input_size, device=device))
    model.train(was_training)
    for h in hooks:
        h.remove()
    return macs


def size_on_disk(model, path):
    """Save the float32 state_dict and return its file size in bytes."""
    torch.save(model.state_dict(), path)
    return os.path.getsize(path)


@torch.no_grad()
def cpu_latency_ms(model, input_size=IMG_SIZE, runs=100, warmup=10):
    """Median single-image (batch = 1) CPU inference latency - our edge-device proxy."""
    model = model.to("cpu").eval()
    x = torch.randn(1, 3, input_size, input_size)
    for _ in range(warmup):
        model(x)
    times = []
    for _ in range(runs):
        t0 = time.perf_counter()
        model(x)
        times.append((time.perf_counter() - t0) * 1000)
    return float(np.median(times))


def layer_table(model, input_size=IMG_SIZE):
    """(description, output shape, params, MACs) for every leaf module."""
    rows = []

    def hook(m, _inp, out):
        params = sum(p.numel() for p in m.parameters(recurse=False))
        if isinstance(m, nn.Conv2d):
            kh, kw = m.kernel_size
            macs = (out.shape[2] * out.shape[3] * m.out_channels
                    * (m.in_channels // m.groups) * kh * kw)
            kind = "DWConv" if m.groups > 1 else ("PWConv" if kh == 1 else "Conv")
            desc = f"{kind} {kh}x{kw}, {m.in_channels}->{m.out_channels}"
        elif isinstance(m, nn.Linear):
            macs = m.in_features * m.out_features
            desc = f"FC {m.in_features}->{m.out_features}"
        else:
            macs, desc = 0, type(m).__name__
        rows.append((desc, tuple(out.shape[1:]), params, macs))

    hooks = [m.register_forward_hook(hook) for m in model.modules()
             if len(list(m.children())) == 0]
    model.eval()
    with torch.no_grad():
        model(torch.zeros(1, 3, input_size, input_size))
    for h in hooks:
        h.remove()
    return rows
''')

md(r"""
### Hand calculation vs PyTorch

We worked the parameter counts out by hand first and the cell below checks them against
PyTorch - if they ever disagree it fails loudly. It also enforces Model B's
100,000-parameter budget.

With `k` = kernel size, `Cin`/`Cout` = channels and `HxW` = the conv's output size:

* standard conv - params `k^2 * Cin * Cout`, MACs `H * W * k^2 * Cin * Cout`
* depthwise conv - params `k^2 * Cin`, MACs `H * W * k^2 * Cin`
* pointwise conv - params `Cin * Cout`, MACs `H * W * Cin * Cout`
* FC - params `in * out + out`, MACs `in * out`
* BatchNorm - 2 trainable parameters per channel (gamma, beta)

**Model A:** 928 + 18,560 + 73,984 + 295,424 + 16,448 + 650 = **405,994** parameters,
~60.18 M MACs.
**Model B:** 193 + 2,528 + 9,152 + 34,688 + 16,448 + 650 = **63,659** parameters
(< 100,000, within budget), ~7.33 M MACs.

For a single layer the depthwise separable convolution costs `1/Cout + 1/k^2` of a standard
one. With `k = 3` and `Cout = 256` that is `1/256 + 1/9 ~ 0.115`, i.e. about 8.7x cheaper.
""")

code(r'''
EXPECTED_PARAMS = {"model_a": 405_994, "model_b": 63_659}   # hand-calculated
PARAM_BUDGET_B  = 100_000

model_summary = {}
for name, builder in (("model_a", model_a), ("model_b", model_b)):
    model = builder()
    print(f"\n=== {name} ===")
    print(f"{'layer':<28}{'output':>18}{'params':>12}{'MACs':>14}")
    for desc, shape, p, m in layer_table(model):
        print(f"{desc:<28}{str(shape):>18}{p:>12,}{m:>14,}")
    params, macs = count_params(model), count_macs(model)
    print(f"{'TOTAL':<46}{params:>12,}{macs:>14,}")
    print(f"float32 weights ~ {params * 4 / 1024:.1f} KB, int8 ~ {params / 1024:.1f} KB")

    assert params == EXPECTED_PARAMS[name], \
        f"{name}: PyTorch says {params}, hand calculation says {EXPECTED_PARAMS[name]}"
    model_summary[name] = {"params": params, "macs": macs}

assert model_summary["model_b"]["params"] <= PARAM_BUDGET_B, "Model B exceeds 100k parameters!"
ratio_p = model_summary["model_a"]["params"] / model_summary["model_b"]["params"]
ratio_m = model_summary["model_a"]["macs"] / model_summary["model_b"]["macs"]
print(f"\nModel B uses {ratio_p:.2f}x fewer parameters and {ratio_m:.2f}x fewer MACs than Model A.")
save_json(model_summary, RES_DIR / "model_summary.json")
''')

# ---------------------------------------------------------------- engine
md("""
---
## Training engine

The training loop, metrics and plotting helpers used by sections 3-5. `fit()` keeps the
weights from the best validation epoch, and times **training only** (validation is excluded
from the per-epoch time so the comparison is about the model, not the evaluation).
""")

code(r'''
def build_optimizer(name, params, lr, momentum=0.0, weight_decay=0.0):
    """name: 'adam' | 'adamw' | 'sgd' (momentum=0 gives plain SGD)."""
    if name == "adam":
        return torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)
    if name == "adamw":
        return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    if name == "sgd":
        return torch.optim.SGD(params, lr=lr, momentum=momentum, weight_decay=weight_decay)
    raise ValueError(f"unknown optimizer {name}")


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss, correct, n = 0.0, 0, 0
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * y.size(0)
        correct += (logits.argmax(1) == y).sum().item()
        n += y.size(0)
    return total_loss / n, correct / n


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    """Returns (mean loss, accuracy, y_true, y_pred)."""
    model.eval()
    total_loss, n = 0.0, 0
    y_true, y_pred = [], []
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        logits = model(x)
        total_loss += criterion(logits, y).item() * y.size(0)
        n += y.size(0)
        # argmax of the logits == argmax of the softmax, so no exp() is needed at inference
        y_pred.append(logits.argmax(1).cpu())
        y_true.append(y.cpu())
    y_true, y_pred = torch.cat(y_true).numpy(), torch.cat(y_pred).numpy()
    return total_loss / n, float((y_true == y_pred).mean()), y_true, y_pred


def fit(model, loaders, optimizer, epochs, device, scheduler=None, tag=""):
    """Train for `epochs`, keeping the weights with the best validation accuracy."""
    criterion = nn.CrossEntropyLoss()
    hist = {k: [] for k in ("train_loss", "train_acc", "val_loss", "val_acc", "epoch_time")}
    best_acc, best_state = -1.0, None

    for epoch in range(1, epochs + 1):
        sync(device)
        t0 = time.perf_counter()
        tr_loss, tr_acc = train_one_epoch(model, loaders["train"], criterion, optimizer, device)
        sync(device)
        ep_time = time.perf_counter() - t0          # training only, no validation

        va_loss, va_acc, _, _ = evaluate(model, loaders["val"], criterion, device)
        if scheduler is not None:
            scheduler.step()

        for k, v in zip(hist, (tr_loss, tr_acc, va_loss, va_acc, ep_time)):
            hist[k].append(v)
        if va_acc > best_acc:
            best_acc = va_acc
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}

        print(f"[{tag}] epoch {epoch:3d}/{epochs} | train loss {tr_loss:.4f} acc {tr_acc:.4f} | "
              f"val loss {va_loss:.4f} acc {va_acc:.4f} | {ep_time:.1f}s")

    model.load_state_dict(best_state)
    hist["best_val_acc"] = best_acc
    return hist


def test_metrics(model, loader, device, class_names):
    _, _, y_true, y_pred = evaluate(model, loader, nn.CrossEntropyLoss(), device)
    labels_ = list(range(len(class_names)))
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision_macro": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "recall_macro": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "precision_per_class": precision_score(y_true, y_pred, average=None, labels=labels_,
                                               zero_division=0).tolist(),
        "recall_per_class": recall_score(y_true, y_pred, average=None, labels=labels_,
                                         zero_division=0).tolist(),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels_).tolist(),
        "report": classification_report(y_true, y_pred, labels=labels_,
                                        target_names=class_names, digits=4, zero_division=0),
    }


def plot_curves(histories, path, keys=("train_loss", "val_loss"), title="Loss"):
    """histories: {label: history}. One subplot per key."""
    fig, axes = plt.subplots(1, len(keys), figsize=(6 * len(keys), 4))
    axes = np.atleast_1d(axes)
    for ax, key in zip(axes, keys):
        for label, h in histories.items():
            ax.plot(range(1, len(h[key]) + 1), h[key], label=label)
        ax.set_xlabel("Epoch")
        ax.set_ylabel(key.replace("_", " "))
        ax.set_title(f"{title}: {key.replace('_', ' ')}")
        ax.grid(alpha=0.3)
        ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.show()


def plot_confusion(cm, classes, path, title):
    cm = np.array(cm)
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(classes)), classes, rotation=45, ha="right")
    ax.set_yticks(range(len(classes)), classes)
    thresh = cm.max() / 2
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black", fontsize=8)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True"); ax.set_title(title)
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.show()


def run_experiment(name, model, loaders, classes, optimizer, epochs, device,
                   scheduler=None, input_size=IMG_SIZE, extra=None):
    """Train + test one model and store everything the report needs."""
    hist = fit(model, loaders, optimizer, epochs, device, scheduler, tag=name)
    metrics = test_metrics(model, loaders["test"], device, classes)

    size_bytes = size_on_disk(model, CKPT_DIR / f"{name}.pt")
    result = {
        "name": name, "classes": classes, "input_size": input_size,
        "params_trainable": count_params(model),
        "params_total": count_params(model, trainable_only=False),
        "macs": count_macs(model, input_size),
        "size_bytes": size_bytes, "size_kb": size_bytes / 1024,
        "size_mb": size_bytes / 1024 ** 2,
        "epochs": epochs,
        "mean_epoch_time_s": float(np.mean(hist["epoch_time"])),
        "device": str(device), "history": hist, "test": metrics,
        **(extra or {}),
    }
    save_json(result, RES_DIR / f"{name}.json")
    plot_curves({name: hist}, FIG_DIR / f"{name}_loss.png", title=name)
    plot_confusion(metrics["confusion_matrix"], classes, FIG_DIR / f"{name}_confusion.png",
                   f"{name} - test confusion matrix (acc {metrics['accuracy']:.4f})")
    print(metrics["report"])
    return result
''')

# ---------------------------------------------------------------- Task 3
md(r"""
---
## 3. Optimizer selection & tuning  *(Task 3 - 15 marks)*

The study is run on **Model B**, the model we actually care about deploying. To keep the
comparison fair every run starts from the *same* initial weights and sees the mini-batches
in the same order - the only thing that changes is the optimizer:

* **Adam at 1e-2 / 1e-3 / 1e-4** - justifies the learning rate we picked;
* **plain SGD (lr 1e-2)** - the baseline the assignment asks for;
* **SGD + momentum 0.5 / 0.9 / 0.99 (lr 1e-2)** - isolates what the momentum term does.

That is 7 training runs, so this is the slowest section of the notebook.
""")

code(r'''
# (label, optimizer, learning rate, momentum)
OPTIM_CONFIGS = [
    ("Adam lr=1e-2",        "adam", 1e-2, 0.0),
    ("Adam lr=1e-3",        "adam", 1e-3, 0.0),
    ("Adam lr=1e-4",        "adam", 1e-4, 0.0),
    ("SGD lr=1e-2",         "sgd",  1e-2, 0.0),
    ("SGD+M(0.5) lr=1e-2",  "sgd",  1e-2, 0.5),
    ("SGD+M(0.9) lr=1e-2",  "sgd",  1e-2, 0.9),
    ("SGD+M(0.99) lr=1e-2", "sgd",  1e-2, 0.99),
]
TARGET_ACC = 0.85
OPTIM_MODEL = "model_b"


def epochs_to_reach(accs, target):
    """First epoch at which validation accuracy reaches `target` (None if never)."""
    hits = [i + 1 for i, a in enumerate(accs) if a >= target]
    return hits[0] if hits else None


set_seed()
init_model = BUILDERS[OPTIM_MODEL]()     # every run starts from these exact weights

optim_histories, optim_summary = {}, {}
for label, opt_name, lr, momentum in OPTIM_CONFIGS:
    set_seed()                           # identical mini-batch order for every run
    m = copy.deepcopy(init_model).to(DEVICE)
    opt = build_optimizer(opt_name, m.parameters(), lr, momentum)
    hist = fit(m, loaders, opt, EPOCHS_OPTIM, DEVICE, tag=label)
    test = test_metrics(m, loaders["test"], DEVICE, CLASSES)

    optim_histories[label] = hist
    optim_summary[label] = {
        "optimizer": opt_name, "lr": lr, "momentum": momentum,
        "final_train_loss": hist["train_loss"][-1],
        "final_val_loss": hist["val_loss"][-1],
        "best_val_acc": hist["best_val_acc"],
        "val_loss_std_last5": float(np.std(hist["val_loss"][-5:])),   # oscillation measure
        f"epochs_to_{TARGET_ACC:.2f}_val_acc": epochs_to_reach(hist["val_acc"], TARGET_ACC),
        "test_acc": test["accuracy"],
        "history": hist,
    }

save_json(optim_summary, RES_DIR / f"optimizer_study_{OPTIM_MODEL}.json")

print(f"\n{'run':<22}{'best val acc':>14}{'test acc':>10}{'final val loss':>16}{'epochs->target':>16}")
for label, s in optim_summary.items():
    reach = s[f"epochs_to_{TARGET_ACC:.2f}_val_acc"]
    print(f"{label:<22}{s['best_val_acc']:>14.4f}{s['test_acc']:>10.4f}"
          f"{s['final_val_loss']:>16.4f}{str(reach):>16}")
''')

code(r'''
keys = ("train_loss", "val_loss", "val_acc")

# Figure 1: our choice vs plain SGD vs SGD+momentum (the comparison the assignment asks for)
main_runs = ["Adam lr=1e-3", "SGD lr=1e-2", "SGD+M(0.9) lr=1e-2"]
plot_curves({k: optim_histories[k] for k in main_runs},
            FIG_DIR / f"optim_compare_{OPTIM_MODEL}.png", keys, "Optimizers")

# Figure 2: effect of the momentum coefficient
plot_curves({k: v for k, v in optim_histories.items() if k.startswith("SGD")},
            FIG_DIR / f"optim_momentum_{OPTIM_MODEL}.png", keys, "Momentum")

# Figure 3: Adam learning-rate sweep
plot_curves({k: v for k, v in optim_histories.items() if k.startswith("Adam")},
            FIG_DIR / f"optim_adam_lr_{OPTIM_MODEL}.png", keys, "Adam LR")
''')

md(r"""
### What the study showed (20-epoch run)

| Run | Best val acc | Test acc | Final val loss | Val-loss sigma (last 5) | Epochs to 85% val acc |
|---|---:|---:|---:|---:|---:|
| Adam lr=1e-2 | 0.9249 | 0.9365 | 0.2341 | 0.0439 | 5 |
| **Adam lr=1e-3** | 0.9190 | 0.9259 | 0.2385 | **0.0124** | 8 |
| Adam lr=1e-4 | 0.8141 | 0.8249 | 0.5274 | 0.0108 | never |
| SGD lr=1e-2 | 0.8417 | 0.8568 | 0.4266 | 0.4056 | never |
| SGD+M(0.5) lr=1e-2 | 0.8731 | 0.8719 | 0.5935 | 0.0851 | 15 |
| SGD+M(0.9) lr=1e-2 | 0.9151 | 0.9210 | 0.5292 | 0.1067 | 9 |
| SGD+M(0.99) lr=1e-2 | 0.9077 | 0.9240 | 0.2945 | 0.0183 | 9 |

**Our choice: Adam, lr = 1e-3, weight decay 1e-4, with cosine decay.** Adam's per-parameter
adaptive step sizes matter here because Model B's layers are wildly uneven - a depthwise
layer has 9 weights per channel while the pointwise layer next to it has hundreds, so one
global step size suits neither. lr = 1e-2 reached 85% three epochs sooner but its
validation loss was 3.5x noisier (sigma = 0.044 vs 0.012); 1e-4 was stable but had not
converged after 20 epochs. 1e-3 is the point where convergence speed and stability meet,
and adding cosine decay lets the final epochs settle into the minimum.

**Effect of momentum.** Plain SGD is the worst run of the seven: at lr = 1e-2 the raw
gradient step is too small to make progress through the shallow, noisy regions, it never
reaches 85%, and its validation loss oscillates badly (sigma = 0.41). Momentum accumulates
an exponentially-weighted average of past gradients, `v <- beta*v + g`, which amplifies the
effective step by roughly `1/(1-beta)` along consistently-signed directions while averaging
away the mini-batch noise in the rest:

* **beta = 0.5** (~2x effective step): clearly better than plain SGD, 87.3% best val acc,
  but still slow - 15 epochs to 85%.
* **beta = 0.9** (~10x): the sweet spot. 91.5% best val acc, 85% reached in 9 epochs,
  essentially matching Adam.
* **beta = 0.99** (~100x): no faster to the target and slightly *lower* best val acc
  (90.8%), though its late-training loss was the smoothest of the SGD runs (sigma = 0.018).
  The long averaging window makes the update slow to respond to curvature changes, so it
  overshoots narrow valleys and then needs several epochs to turn around - the classic
  momentum trade-off between acceleration and responsiveness.

In short, momentum closes most of the gap between SGD and Adam (85.7% -> 92.1% test accuracy
at beta = 0.9) but needs the coefficient to be tuned, whereas Adam got there with its
default settings. On an edge device this choice costs nothing at inference - the optimizer
only exists during training.
""")

# ---------------------------------------------------------------- Task 4
md(r"""
---
## 4. Training & evaluation of the custom models  *(Task 4 - 25 marks)*

Both models are trained with the winner from section 3 - Adam, lr 1e-3, weight decay 1e-4,
cosine decay - for 30 epochs (the assignment requires at least 20). Same data, same seed,
same schedule for both, so any difference comes from the convolution type.
""")

code(r'''
LR, WEIGHT_DECAY = 1e-3, 1e-4

custom_results, custom_histories = {}, {}
for name in ("model_a", "model_b"):
    set_seed()
    model = BUILDERS[name](len(CLASSES)).to(DEVICE)
    opt = build_optimizer("adam", model.parameters(), LR, weight_decay=WEIGHT_DECAY)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS_CUSTOM)
    res = run_experiment(name, model, loaders, CLASSES, opt, EPOCHS_CUSTOM, DEVICE, sched,
                         extra={"optimizer": f"Adam lr={LR} wd={WEIGHT_DECAY} cosine"})
    custom_results[name] = res
    custom_histories[name] = res["history"]

plot_curves(custom_histories, FIG_DIR / "custom_models_curves.png",
            ("train_loss", "val_loss", "val_acc"), "Model A vs Model B")
''')

md(r"""
### Results (30-epoch run)

| Model | Params | MACs (M) | Size on disk (KB) | Train time / epoch (s) | CPU latency (ms) | Test acc | Precision | Recall |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Model A | 405,994 | 60.18 | 1598.7 | 5.25 | 4.87 | 0.9694 | 0.9690 | 0.9686 |
| Model B | 63,659 | 7.33 | 270.9 | 5.41 | 3.94 | 0.9496 | 0.9486 | 0.9484 |

**What the depthwise separable factorisation bought and cost.** Model B is **6.4x smaller**
in parameters, **8.2x cheaper** in MACs and **5.9x smaller** on disk, for **1.98 percentage
points** of test accuracy (96.94% -> 94.96%). On an edge budget that is a good trade: 271 KB
of float32 weights, or about 62 KB once quantised to int8, fits a microcontroller with
256 KB of flash, whereas Model A's 1.6 MB does not.

**The timing result is the interesting one.** Model B's training time per epoch did *not*
drop with its MAC count - it was marginally **slower** (5.41 s vs 5.25 s). Depthwise
convolutions are memory-bandwidth bound, not compute bound: each 3x3 depthwise filter does
9 MACs per loaded activation, where a dense conv with 256 output channels reuses every
loaded value hundreds of times. A GPU with thousands of idle ALUs gains nothing from fewer
MACs when the bottleneck is fetching activations, and the extra kernel launches (two
convolutions and two BatchNorms per block instead of one each) add overhead. The 8x MAC
reduction only shows up where arithmetic is genuinely the constraint - the single-threaded
CPU latency, our edge proxy, did improve (4.87 ms -> 3.94 ms, a 1.24x speed-up). The lesson
is that **MACs are a proxy for energy and memory, not a promise of wall-clock speed-up**,
and the benefit depends entirely on the target hardware.

**Where the errors are.** Both models fail in the same, sensible places. The largest
confusion for both is PermanentCrop -> HerbaceousVegetation (11 images for A, 18 for B):
from directly overhead, an orchard and rough grassland are both green textured fields and
differ mainly in the regularity of their planting, a fine-grained cue that is already
marginal at 64x64. River -> AnnualCrop (10 / 11) is similar - a narrow waterway crossing
farmland occupies only a few pixels of a 64x64 patch. Model B additionally confuses
Highway with River (9 images): both are long thin ribbons crossing the scene, and
distinguishing them needs texture and colour detail that the thinner network has less
capacity to encode. PermanentCrop has the worst recall in both models (A 0.931, B 0.891),
and the whole accuracy gap between A and B is concentrated in these ambiguous vegetation
and linear-feature classes; the easy, visually distinctive classes (Forest, SeaLake,
Residential, Industrial) stay above 0.97 recall even in the 64k-parameter model.
""")

# ---------------------------------------------------------------- Task 5
md(r"""
---
## 5. Fine-tuning lightweight SOTA networks  *(Task 5 - 20 marks)*

We picked **MobileNetV2** and **ShuffleNetV2 x1.0**: both were designed for phones and
embedded targets, both are parameter-efficient by construction (inverted residuals with
depthwise separable convolutions; channel split + shuffle respectively), and both ship with
ImageNet weights in torchvision.

**How we fine-tune.** Replace the 1000-class head with a 10-class linear layer, then train
the whole network with two learning rates - a gentle `1e-4` for the pre-trained backbone
(we want to nudge the ImageNet features, not erase them) and `1e-3`, ten times larger, for
the randomly-initialised head. AdamW, weight decay 1e-4, cosine decay, 20 epochs.

Both use the **same 64x64 splits** as our custom models, so the comparison is exact - even
though these networks were designed for 224x224 input and are therefore handicapped a
little by the small images.
""")

code(r'''
SOTA_LR = 1e-4            # backbone; the head uses 10x this
SOTA_INPUT = IMG_SIZE     # same 64x64 input as our custom models

sota_results = {}
for name in ("mobilenet_v2", "shufflenet_v2"):
    set_seed()
    model = BUILDERS[name](len(CLASSES)).to(DEVICE)
    head_ids = {id(p) for p in head_parameters(name, model)}
    backbone = [p for p in model.parameters() if id(p) not in head_ids]
    head     = [p for p in model.parameters() if id(p) in head_ids]

    opt = torch.optim.AdamW([{"params": backbone, "lr": SOTA_LR},
                             {"params": head,     "lr": SOTA_LR * 10}], weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS_SOTA)
    sota_results[name] = run_experiment(
        name, model, loaders, CLASSES, opt, EPOCHS_SOTA, DEVICE, sched,
        input_size=SOTA_INPUT,
        extra={"optimizer": f"AdamW backbone lr={SOTA_LR}, head lr={SOTA_LR * 10}, cosine"})
''')

md(r"""
### Results (20-epoch fine-tuning)

| Model | Params | MACs (M) | Size (MB) | Train time / epoch (s) | CPU latency (ms) | Test acc | Precision | Recall |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| ShuffleNetV2 x1.0 | 1,263,854 | 11.76 | 4.99 | 11.45 | 7.22 | 0.9793 | 0.9785 | 0.9786 |
| MobileNetV2 | 2,236,682 | 24.46 | 8.77 | 10.11 | 7.04 | 0.9854 | 0.9850 | 0.9846 |

Both reach their accuracy in 20 epochs rather than 30 - ImageNet features for edges,
textures and colour statistics transfer directly to satellite imagery, so fine-tuning only
has to re-weight them. MobileNetV2 is the more accurate of the two (98.54%) and
ShuffleNetV2 the more efficient (1.3 M parameters, 11.8 M MACs for 97.93%). Their remaining
errors are the same vegetation ambiguities our models hit - MobileNetV2's largest confusions
are AnnualCrop -> PermanentCrop and Pasture -> AnnualCrop, 7 images each - which suggests
those cases are close to the limit of what a 64x64 patch supports, not a capacity problem.
""")

# ---------------------------------------------------------------- Task 6
md(r"""
---
## 6. Final comparison & trade-off discussion  *(Task 6 - 20 marks)*

The cell below collects every result, measures single-image inference latency on **one CPU
thread** (our stand-in for a small edge device) and builds the comparison tables and the
accuracy-vs-size plot.
""")

code(r'''
def latency_of(r):
    base = r["name"] if r["name"] in BUILDERS else r["name"].rsplit("_", 1)[0]
    kwargs = {"pretrained": False} if base in ("mobilenet_v2", "shufflenet_v2") else {}
    m = BUILDERS[base](len(r["classes"]), **kwargs)
    m.load_state_dict(torch.load(CKPT_DIR / f"{r['name']}.pt", map_location="cpu"))
    return cpu_latency_ms(m, r["input_size"])


all_results = {**custom_results, **sota_results}
torch.set_num_threads(1)           # single core ~ a small edge CPU, and stable timings
for r in all_results.values():
    r["latency_ms"] = latency_of(r)

lines = ["## Model A vs Model B", "",
         "| Model | Params | MACs (M) | Size on disk (KB) | Train time / epoch (s) | "
         "CPU latency (ms) | Test acc | Precision | Recall |",
         "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
for name in ("model_a", "model_b"):
    r, t = all_results[name], all_results[name]["test"]
    lines.append(f"| {name} | {r['params_trainable']:,} | {r['macs'] / 1e6:.2f} | "
                 f"{r['size_kb']:.1f} | {r['mean_epoch_time_s']:.2f} | {r['latency_ms']:.2f} | "
                 f"{t['accuracy']:.4f} | {t['precision_macro']:.4f} | {t['recall_macro']:.4f} |")

lines += ["", "## Model B vs lightweight SOTA", "",
          "| Model | Params | MACs (M) | Size (MB) | Input | Train time / epoch (s) | "
          "CPU latency (ms) | Test acc | Precision | Recall |",
          "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
for name, r in sorted(all_results.items(), key=lambda kv: kv[1]["params_trainable"]):
    if name == "model_a":
        continue
    t = r["test"]
    lines.append(f"| {name} | {r['params_trainable']:,} | {r['macs'] / 1e6:.2f} | "
                 f"{r['size_mb']:.3f} | {r['input_size']} | {r['mean_epoch_time_s']:.2f} | "
                 f"{r['latency_ms']:.2f} | {t['accuracy']:.4f} | {t['precision_macro']:.4f} | "
                 f"{t['recall_macro']:.4f} |")

md_table = "\n".join(lines) + "\n"
with open(RES_DIR / "comparison.md", "w") as f:
    f.write(md_table)
print(md_table)
''')

code(r'''
# Accuracy vs memory footprint (bubble area ~ MACs)
fig, ax = plt.subplots(figsize=(7, 5))
for name, r in all_results.items():
    ax.scatter(r["size_mb"], r["test"]["accuracy"] * 100,
               s=max(r["macs"] / 1e6, 1) * 4, alpha=0.6)
    ax.annotate(f"{name}\n{r['macs'] / 1e6:.0f}M MACs",
                (r["size_mb"], r["test"]["accuracy"] * 100),
                textcoords="offset points", xytext=(8, -4), fontsize=8)
ax.set_xscale("log")
ax.set_xlabel("Model size on disk (MB, log scale)")
ax.set_ylabel("Test accuracy (%)")
ax.set_title("Accuracy vs memory footprint (bubble area ~ MACs)")
ax.grid(alpha=0.3, which="both")
fig.tight_layout()
fig.savefig(FIG_DIR / "accuracy_vs_size.png", dpi=150)
plt.show()
''')

md(r"""
### Discussion

| Model | Params | Size | MACs (M) | CPU latency (ms) | Test acc |
|---|---:|---:|---:|---:|---:|
| **Model B (ours)** | **63,659** | **265 KB** | **7.33** | **3.94** | 0.9496 |
| ShuffleNetV2 x1.0 | 1,263,854 | 4.99 MB | 11.76 | 7.22 | 0.9793 |
| MobileNetV2 | 2,236,682 | 8.77 MB | 24.46 | 7.04 | 0.9854 |

**Accuracy.** MobileNetV2 beats Model B by 3.58 points and ShuffleNetV2 by 2.97. Almost all
of that gap is ImageNet pre-training rather than architecture: MobileNetV2 is itself built
from depthwise separable convolutions, so the operations are the same - it simply starts
from features learned on a million photographs, while Model B learns everything from 18,900
satellite patches. Depth and width help too (MobileNetV2 is 53 layers against our 8), but
the transfer-learning head start is the dominant term, and it is also why the pre-trained
models converge in 20 epochs instead of 30.

**Memory footprint.** This is where the custom model wins outright, and not by a little:
Model B is **35x smaller than MobileNetV2** and **19x smaller than ShuffleNetV2**. The
distinction that matters on a microcontroller is categorical rather than incremental. An
STM32-class device with 256 KB-1 MB of flash can hold int8 Model B (~62 KB) with room to
spare for the application; int8 MobileNetV2 (~2.2 MB) does not fit at all, at any accuracy.
Activation RAM follows the same pattern - Model B's largest intermediate tensor is
64x64x32 = 128 KB in int8, already demanding for such a device, and MobileNetV2's is
several times that. So the honest comparison is often not "95% versus 98.5%" but "95%
versus nothing runs."

**Computational cost.** Model B needs 3.3x fewer MACs than MobileNetV2 and 1.6x fewer than
ShuffleNetV2, and was 1.8x faster in single-threaded CPU inference (3.94 ms vs ~7 ms). Fewer
MACs and a smaller working set also mean less energy per inference, which on a battery- or
solar-powered sensor directly sets the duty cycle. Note again the caveat from section 4:
the MAC advantage translated into a real speed-up on the CPU but not on the GPU, because
depthwise convolutions are memory-bound - on any specific target the numbers must be
measured, not predicted from MACs.

**Advantages of building a custom sub-100k model.** Full control over the footprint - we
could pick the budget first and design to it, which is exactly how embedded work proceeds;
no dependency on pre-trained weights or their licences; an architecture matched to the real
input resolution (MobileNetV2 downsamples by 32, so at 64x64 its last feature map is a
degenerate 2x2, wasting most of its depth); and a model small and simple enough to audit,
quantise and deploy with a minimal runtime.

**Limitations.** We paid ~3.6 accuracy points, needed 30 epochs instead of 20, and spent
considerably more engineering effort - the architecture, the parameter budget and the
optimizer all had to be worked out by hand, whereas fine-tuning MobileNetV2 is a dozen
lines of torchvision. A sub-100k model trained on 18,900 images is also more exposed to
distribution shift: it has far less redundant capacity, so a new sensor, season or
geography is likelier to degrade it than a model whose features were learned on ImageNet.

**Recommendation.** The choice follows the memory budget, not the accuracy table. Under
~1 MB of flash - microcontrollers, deeply embedded sensors - Model B is the only option
here that fits, and 94.96% is a usable classifier. With a few megabytes to spare and a
Linux-class edge device (Raspberry Pi, Jetson), fine-tuned MobileNetV2 is the better buy:
3.58 points of accuracy for 8.5 MB is cheap when the memory exists, and the engineering
cost is far lower. The middle ground - ShuffleNetV2, or an int8-quantised MobileNetV2 at
~2.2 MB - is worth measuring whenever the target sits between those two extremes. The
broader conclusion is that on edge hardware, accuracy is a constrained optimisation: the
memory and energy budget is fixed first by the device, and the best model is the most
accurate one that fits inside it.
""")

md(r"""
---
## Reproducibility notes

* Seed 42 is set before every model build and every training run; `data/splits.json` pins
  the train/validation/test indices, so all four models are evaluated on identical images.
* Results reported above were produced on an NVIDIA RTX 3060 Ti (CUDA), PyTorch 2.x, with
  `QUICK = False`. Training times per epoch are GPU numbers; CPU latency is measured with
  `torch.set_num_threads(1)` on the same machine.
* Minor run-to-run variation is expected from cuDNN's non-deterministic kernel selection.
* All figures are written to `outputs/figures/`, metrics to `outputs/results/` and weights
  to `outputs/checkpoints/`.
""")

nb = nbf.v4.new_notebook()
nb.cells = [nbf.v4.new_markdown_cell(s) if k == "md" else nbf.v4.new_code_cell(s)
            for k, s in C]
nb.metadata = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python", "version": "3.11"},
}
out = Path(r"E:\Pattern\submission\PatternX_A03_EN3150.ipynb")
nbf.write(nb, str(out))
print("wrote", out, "-", len(nb.cells), "cells")
