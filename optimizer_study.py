"""Task 3 - Optimizer selection & tuning.

Trains the same network from the SAME initial weights with:
  * Adam at three learning rates (to justify the chosen LR)
  * plain SGD
  * SGD with momentum 0.5 / 0.9 / 0.99 (to show the effect of the momentum term)
and plots training/validation curves for all runs.

    python optimizer_study.py                 # Model B, 20 epochs
    python optimizer_study.py --model model_a --epochs 25
"""
import argparse
import copy
import os

import numpy as np

from src.data import get_dataloaders
from src.engine import build_optimizer, fit, plot_curves, test_metrics
from src.models import BUILDERS
from src.utils import FIGURES_DIR, RESULTS_DIR, get_device, save_json, set_seed

# (label, optimizer, learning rate, momentum)
CONFIGS = [
    ("Adam lr=1e-2", "adam", 1e-2, 0.0),
    ("Adam lr=1e-3", "adam", 1e-3, 0.0),
    ("Adam lr=1e-4", "adam", 1e-4, 0.0),
    ("SGD lr=1e-2", "sgd", 1e-2, 0.0),
    ("SGD+M(0.5) lr=1e-2", "sgd", 1e-2, 0.5),
    ("SGD+M(0.9) lr=1e-2", "sgd", 1e-2, 0.9),
    ("SGD+M(0.99) lr=1e-2", "sgd", 1e-2, 0.99),
]


def epochs_to_reach(accs, target):
    """First epoch at which validation accuracy reaches `target` (None if never)."""
    hits = [i + 1 for i, a in enumerate(accs) if a >= target]
    return hits[0] if hits else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="model_b", choices=["model_a", "model_b"])
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--target-acc", type=float, default=0.85)
    args = ap.parse_args()

    device = get_device()
    loaders, classes = get_dataloaders(batch_size=args.batch_size)

    set_seed(42)
    init_model = BUILDERS[args.model]()  # every run starts from these exact weights

    histories, summary = {}, {}
    for label, opt_name, lr, momentum in CONFIGS:
        set_seed(42)  # identical mini-batch order for every run
        model = copy.deepcopy(init_model).to(device)
        opt = build_optimizer(opt_name, model.parameters(), lr, momentum)
        hist = fit(model, loaders, opt, args.epochs, device, tag=label)
        test = test_metrics(model, loaders["test"], device, classes)

        histories[label] = hist
        summary[label] = {
            "optimizer": opt_name, "lr": lr, "momentum": momentum,
            "final_train_loss": hist["train_loss"][-1],
            "final_val_loss": hist["val_loss"][-1],
            "best_val_acc": hist["best_val_acc"],
            "val_loss_std_last5": float(np.std(hist["val_loss"][-5:])),  # oscillation measure
            f"epochs_to_{args.target_acc:.2f}_val_acc": epochs_to_reach(hist["val_acc"], args.target_acc),
            "test_acc": test["accuracy"],
            "history": hist,
        }

    save_json(summary, os.path.join(RESULTS_DIR, f"optimizer_study_{args.model}.json"))

    # Figure 1: chosen optimizer vs SGD vs SGD+momentum (the required comparison)
    main_runs = ["Adam lr=1e-3", "SGD lr=1e-2", "SGD+M(0.9) lr=1e-2"]
    keys = ("train_loss", "val_loss", "val_acc")
    plot_curves({k: histories[k] for k in main_runs},
                os.path.join(FIGURES_DIR, f"optim_compare_{args.model}.png"), keys, "Optimizers")
    # Figure 2: effect of momentum
    plot_curves({k: v for k, v in histories.items() if k.startswith("SGD")},
                os.path.join(FIGURES_DIR, f"optim_momentum_{args.model}.png"), keys, "Momentum")
    # Figure 3: Adam learning-rate sweep
    plot_curves({k: v for k, v in histories.items() if k.startswith("Adam")},
                os.path.join(FIGURES_DIR, f"optim_adam_lr_{args.model}.png"), keys, "Adam LR")

    print(f"\n{'run':<22}{'best val acc':>14}{'test acc':>10}{'final val loss':>16}{'epochs->target':>16}")
    for label, s in summary.items():
        reach = s[f"epochs_to_{args.target_acc:.2f}_val_acc"]
        print(f"{label:<22}{s['best_val_acc']:>14.4f}{s['test_acc']:>10.4f}"
              f"{s['final_val_loss']:>16.4f}{str(reach):>16}")


if __name__ == "__main__":
    main()
