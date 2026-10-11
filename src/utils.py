"""Small helpers we kept reaching for: seeding, picking a device, measuring
model complexity (params / MACs / size / latency) and JSON in-out."""
import json
import os
import random
import time

import numpy as np
import torch
import torch.nn as nn

from .paths import CKPT_DIR, FIGURES_DIR, RESULTS_DIR  # noqa: F401  (re-exported for the scripts)


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


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


def count_params(model, trainable_only=True):
    return sum(p.numel() for p in model.parameters() if p.requires_grad or not trainable_only)


@torch.no_grad()
def count_macs(model, input_size=64):
    """Multiply-accumulate operations for one image, counted with forward hooks.

    Conv2d : H_out * W_out * C_out * (C_in / groups) * k_h * k_w
    Linear : in_features * out_features
    BN / ReLU / pooling are ignored (negligible, and BN folds into the conv at inference).
    """
    macs = 0

    def conv_hook(m, _inp, out):
        nonlocal macs
        kh, kw = m.kernel_size
        macs += out.shape[2] * out.shape[3] * m.out_channels * (m.in_channels // m.groups) * kh * kw

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
def cpu_latency_ms(model, input_size=64, runs=100, warmup=10):
    """Median single-image (batch = 1) CPU inference latency - an edge-device proxy."""
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


def save_json(obj, path):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def load_json(path):
    with open(path) as f:
        return json.load(f)
