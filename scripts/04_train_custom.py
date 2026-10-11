"""Step 4 - Train our own models  (assignment Task 4)

Trains Model A and Model B with the winner from step 3 (Adam, lr 1e-3, with a
cosine decay so it settles nicely at the end). Same data, same epochs, same
seed for both - so whatever difference shows up comes from the conv type.

Each model leaves behind its loss curves, confusion matrix, test metrics and
saved weights (that file's size is our "size on disk").

    python scripts/04_train_custom.py                          # both models, 30 epochs
    python scripts/04_train_custom.py --models model_b --epochs 20
"""
import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

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
