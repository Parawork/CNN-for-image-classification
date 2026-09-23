"""Task 1 - Data preparation.

Downloads EuroSAT, downsizes to 64x64, creates the fixed 70/15/15 stratified split
and prints the per-class counts for the report / dataset sheet.

    python prepare_data.py
"""
from collections import Counter

from src.data import IMG_SIZE, load_eurosat, make_splits
from src.utils import FIGURES_DIR, save_json, RESULTS_DIR

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
