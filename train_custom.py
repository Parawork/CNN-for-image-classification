"""Task 4 - Train and evaluate the custom networks (Model A and Model B).

Both models use the optimizer chosen in optimizer_study.py
(Adam, lr = 1e-3, cosine LR decay) and identical data / epochs.

    python train_custom.py               # both models, 30 epochs
    python train_custom.py --models model_b --epochs 20
"""
import argparse
import os

import torch

from src.data import get_dataloaders
from src.engine import build_optimizer, plot_curves, run_experiment
from src.models import BUILDERS
from src.utils import FIGURES_DIR, get_device, set_seed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["model_a", "model_b"])
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    args = ap.parse_args()

    device = get_device()
    loaders, classes = get_dataloaders(batch_size=args.batch_size)

    histories = {}
    for name in args.models:
        set_seed(42)
        model = BUILDERS[name](len(classes)).to(device)
        opt = build_optimizer("adam", model.parameters(), args.lr, weight_decay=args.weight_decay)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
        res = run_experiment(name, model, loaders, classes, opt, args.epochs, device, sched,
                             extra={"optimizer": f"Adam lr={args.lr} wd={args.weight_decay} cosine"})
        histories[name] = res["history"]

    if len(histories) > 1:
        plot_curves(histories, os.path.join(FIGURES_DIR, "custom_models_curves.png"),
                    ("train_loss", "val_loss", "val_acc"), "Model A vs Model B")


if __name__ == "__main__":
    main()
