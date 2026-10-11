"""Training loop, evaluation metrics, plotting and the shared experiment runner."""
import os
import time

import matplotlib

matplotlib.use("Agg")  # headless: figures are only saved to disk
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             precision_score, recall_score)

from .utils import (CKPT_DIR, FIGURES_DIR, RESULTS_DIR, count_macs, count_params,
                    save_json, size_on_disk, sync)


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
    """Train for `epochs`, keep the weights with the best validation accuracy.

    Returns a history dict with per-epoch losses/accuracies and training times.
    """
    criterion = nn.CrossEntropyLoss()
    hist = {k: [] for k in ("train_loss", "train_acc", "val_loss", "val_acc", "epoch_time")}
    best_acc, best_state = -1.0, None

    for epoch in range(1, epochs + 1):
        sync(device)
        t0 = time.perf_counter()
        tr_loss, tr_acc = train_one_epoch(model, loaders["train"], criterion, optimizer, device)
        sync(device)
        ep_time = time.perf_counter() - t0  # training time only (no validation)

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
    labels = list(range(len(class_names)))
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision_macro": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "recall_macro": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "precision_per_class": precision_score(y_true, y_pred, average=None, labels=labels,
                                               zero_division=0).tolist(),
        "recall_per_class": recall_score(y_true, y_pred, average=None, labels=labels,
                                         zero_division=0).tolist(),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
        "report": classification_report(y_true, y_pred, labels=labels, target_names=class_names,
                                        digits=4, zero_division=0),
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
    plt.close(fig)


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
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def run_experiment(name, model, loaders, classes, optimizer, epochs, device,
                   scheduler=None, input_size=64, extra=None):
    """Train + test one model and store everything needed for the report.

    Leaves behind (under outputs/): results/<name>.json, checkpoints/<name>.pt,
    figures/<name>_loss.png and figures/<name>_confusion.png.
    """
    hist = fit(model, loaders, optimizer, epochs, device, scheduler, tag=name)
    metrics = test_metrics(model, loaders["test"], device, classes)

    ckpt = os.path.join(CKPT_DIR, f"{name}.pt")
    size_bytes = size_on_disk(model, ckpt)
    result = {
        "name": name,
        "classes": classes,
        "input_size": input_size,
        "params_trainable": count_params(model),
        "params_total": count_params(model, trainable_only=False),
        "macs": count_macs(model, input_size),
        "size_bytes": size_bytes,
        "size_kb": size_bytes / 1024,
        "size_mb": size_bytes / 1024 ** 2,
        "epochs": epochs,
        "mean_epoch_time_s": float(np.mean(hist["epoch_time"])),
        "device": str(device),
        "history": hist,
        "test": metrics,
        **(extra or {}),
    }
    save_json(result, os.path.join(RESULTS_DIR, f"{name}.json"))
    plot_curves({name: hist}, os.path.join(FIGURES_DIR, f"{name}_loss.png"), title=name)
    plot_confusion(metrics["confusion_matrix"], classes,
                   os.path.join(FIGURES_DIR, f"{name}_confusion.png"),
                   f"{name} - test confusion matrix (acc {metrics['accuracy']:.4f})")
    print(metrics["report"])
    return result
