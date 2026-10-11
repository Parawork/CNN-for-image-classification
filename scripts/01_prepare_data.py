"""Step 1 - Data preparation  (assignment Task 1)

Run this first. It downloads EuroSAT, caches it at 64x64, fixes the 70/15/15
split and prints a per-class table - handy for filling in the dataset sheet.
It also saves one sample image per class so we have something to show in the report.

    python scripts/01_prepare_data.py
"""
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # so `import src` works from anywhere

from src.data import IMG_SIZE, load_eurosat, make_splits
from src.utils import FIGURES_DIR, RESULTS_DIR, save_json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    images, labels, classes = load_eurosat()
    splits = make_splits(labels)

    print(f"Dataset   : EuroSAT RGB, {len(labels)} images, {len(classes)} classes")
    print(f"Image size: {IMG_SIZE}x{IMG_SIZE}x3")
    summary = {"classes": classes, "total": len(labels), "image_size": IMG_SIZE, "splits": {}}

    header = f"{'class':<22}" + "".join(f"{s:>8}" for s in splits)
    print(header + "\n" + "-" * len(header))
    counts = {s: Counter(labels[idx].tolist()) for s, idx in splits.items()}
    for c, name in enumerate(classes):
        print(f"{name:<22}" + "".join(f"{counts[s][c]:>8}" for s in splits))
    print("-" * len(header))
    print(f"{'TOTAL':<22}" + "".join(f"{len(idx):>8}" for idx in splits.values()))

    for s, idx in splits.items():
        summary["splits"][s] = {"n": len(idx), "fraction": len(idx) / len(labels),
                                "per_class": [counts[s][c] for c in range(len(classes))]}
    save_json(summary, f"{RESULTS_DIR}/dataset_summary.json")

    # One example image per class for the report
    fig, axes = plt.subplots(2, 5, figsize=(12, 5))
    for c, ax in enumerate(axes.flat):
        i = (labels == c).nonzero()[0].item()
        ax.imshow(images[i].permute(1, 2, 0).numpy())
        ax.set_title(classes[c], fontsize=9)
        ax.axis("off")
    fig.suptitle("EuroSAT samples at 64x64")
    fig.tight_layout()
    fig.savefig(f"{FIGURES_DIR}/dataset_samples.png", dpi=150)


if __name__ == "__main__":
    main()
