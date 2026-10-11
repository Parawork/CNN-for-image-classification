"""Step 6 - Put it all side by side  (assignment Tasks 4 & 6)

Collects the results from steps 4 and 5, times how long each model takes to
classify one image on a single CPU core (our stand-in for a small edge device),
and writes the comparison tables to outputs/results/comparison.md - ready to
copy into the report. Also draws the accuracy-vs-size trade-off plot.

    python scripts/06_compare.py
"""
import glob
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from src.models import BUILDERS
from src.utils import CKPT_DIR, FIGURES_DIR, RESULTS_DIR, cpu_latency_ms, load_json


def load_results():
    results = {}
    for path in glob.glob(os.path.join(RESULTS_DIR, "*.json")):
        r = load_json(path)
        if "test" in r and "params_trainable" in r:  # skip summaries / optimizer study
            results[r["name"]] = r
    return results


def latency(r):
    base = r["name"] if r["name"] in BUILDERS else r["name"].rsplit("_", 1)[0]
    kwargs = {"pretrained": False} if base in ("mobilenet_v2", "shufflenet_v2") else {}
    model = BUILDERS[base](len(r["classes"]), **kwargs)
    model.load_state_dict(torch.load(os.path.join(CKPT_DIR, f"{r['name']}.pt"), map_location="cpu"))
    return cpu_latency_ms(model, r["input_size"])


def main():
    results = load_results()
    if not results:
        raise SystemExit("Nothing to compare yet - run steps 04 and 05 first "
                         "(or just `python run_all.py`).")
    torch.set_num_threads(1)  # single core ~ a small edge CPU, and stable timings
    for r in results.values():
        r["latency_ms"] = latency(r)

    lines = []
    custom = [results[k] for k in ("model_a", "model_b") if k in results]
    if custom:
        lines += ["## Model A vs Model B", "",
                  "| Model | Params | MACs (M) | Size on disk (KB) | Train time / epoch (s) | "
                  "CPU latency (ms) | Test acc | Precision | Recall |",
                  "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
        for r in custom:
            t = r["test"]
            lines.append(f"| {r['name']} | {r['params_trainable']:,} | {r['macs'] / 1e6:.2f} | "
                         f"{r['size_kb']:.1f} | {r['mean_epoch_time_s']:.2f} | {r['latency_ms']:.2f} | "
                         f"{t['accuracy']:.4f} | {t['precision_macro']:.4f} | {t['recall_macro']:.4f} |")
        lines.append("")

    lines += ["## Model B vs lightweight SOTA", "",
              "| Model | Params | MACs (M) | Size (MB) | Input | Train time / epoch (s) | "
              "CPU latency (ms) | Test acc | Precision | Recall |",
              "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for name, r in sorted(results.items(), key=lambda kv: kv[1]["params_trainable"]):
        if name == "model_a":
            continue
        t = r["test"]
        lines.append(f"| {name} | {r['params_trainable']:,} | {r['macs'] / 1e6:.2f} | {r['size_mb']:.3f} | "
                     f"{r['input_size']} | {r['mean_epoch_time_s']:.2f} | {r['latency_ms']:.2f} | "
                     f"{t['accuracy']:.4f} | {t['precision_macro']:.4f} | {t['recall_macro']:.4f} |")

    md = "\n".join(lines) + "\n"
    with open(os.path.join(RESULTS_DIR, "comparison.md"), "w") as f:
        f.write(md)
    print(md)

    # Accuracy vs memory footprint (bubble area ~ MACs)
    fig, ax = plt.subplots(figsize=(7, 5))
    for name, r in results.items():
        ax.scatter(r["size_mb"], r["test"]["accuracy"] * 100, s=max(r["macs"] / 1e6, 1) * 4, alpha=0.6)
        ax.annotate(f"{name}\n{r['macs'] / 1e6:.0f}M MACs", (r["size_mb"], r["test"]["accuracy"] * 100),
                    textcoords="offset points", xytext=(8, -4), fontsize=8)
    ax.set_xscale("log")
    ax.set_xlabel("Model size on disk (MB, log scale)")
    ax.set_ylabel("Test accuracy (%)")
    ax.set_title("Accuracy vs memory footprint (bubble area ~ MACs)")
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, "accuracy_vs_size.png"), dpi=150)


if __name__ == "__main__":
    main()
