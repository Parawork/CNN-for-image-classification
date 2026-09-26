"""Task 2 - Architecture details.

Prints a layer-by-layer table (output shape, parameters, MACs) for Model A and
Model B, checks the totals against the hand calculation in docs/architecture.md
and asserts the 100k-parameter budget for Model B.

    python model_summary.py
"""
import torch
import torch.nn as nn

from src.models import model_a, model_b
from src.utils import RESULTS_DIR, count_macs, count_params, save_json

# Hand-calculated totals (see docs/architecture.md)
EXPECTED_PARAMS = {"model_a": 405_994, "model_b": 63_659}
PARAM_BUDGET_B = 100_000


def layer_table(model, input_size=64):
    """Collect (layer, output shape, params, MACs) for every leaf module."""
    rows = []

    def hook(m, _inp, out):
        params = sum(p.numel() for p in m.parameters(recurse=False))
        if isinstance(m, nn.Conv2d):
            kh, kw = m.kernel_size
            macs = out.shape[2] * out.shape[3] * m.out_channels * (m.in_channels // m.groups) * kh * kw
            kind = "DWConv" if m.groups > 1 else ("PWConv" if kh == 1 else "Conv")
            desc = f"{kind} {kh}x{kw}, {m.in_channels}->{m.out_channels}"
        elif isinstance(m, nn.Linear):
            macs = m.in_features * m.out_features
            desc = f"FC {m.in_features}->{m.out_features}"
        else:
            macs = 0
            desc = type(m).__name__
        rows.append((desc, tuple(out.shape[1:]), params, macs))

    hooks = [m.register_forward_hook(hook) for m in model.modules() if len(list(m.children())) == 0]
    model.eval()
    with torch.no_grad():
        model(torch.zeros(1, 3, input_size, input_size))
    for h in hooks:
        h.remove()
    return rows


def main():
    summary = {}
    for name, builder in (("model_a", model_a), ("model_b", model_b)):
        model = builder()
        rows = layer_table(model)
        print(f"\n=== {name} ===")
        print(f"{'layer':<28}{'output':>18}{'params':>12}{'MACs':>14}")
        for desc, shape, p, m in rows:
            print(f"{desc:<28}{str(shape):>18}{p:>12,}{m:>14,}")
        params, macs = count_params(model), count_macs(model)
        print(f"{'TOTAL':<46}{params:>12,}{macs:>14,}")
        print(f"float32 weights ~ {params * 4 / 1024:.1f} KB, int8 ~ {params / 1024:.1f} KB")

        assert params == EXPECTED_PARAMS[name], \
            f"{name}: PyTorch says {params}, hand calculation says {EXPECTED_PARAMS[name]}"
        summary[name] = {"params": params, "macs": macs}

    assert summary["model_b"]["params"] <= PARAM_BUDGET_B, "Model B exceeds 100k parameters!"
    ratio_p = summary["model_a"]["params"] / summary["model_b"]["params"]
    ratio_m = summary["model_a"]["macs"] / summary["model_b"]["macs"]
    print(f"\nModel B uses {ratio_p:.2f}x fewer parameters and {ratio_m:.2f}x fewer MACs than Model A.")
    save_json(summary, f"{RESULTS_DIR}/model_summary.json")


if __name__ == "__main__":
    main()
