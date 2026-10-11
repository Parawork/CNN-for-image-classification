"""Step 5 - Fine-tune the pre-trained lightweight models  (assignment Task 5)

We picked MobileNetV2 and ShuffleNetV2 (x1.0) - both were designed for phones,
and both ship with ImageNet weights in torchvision.

How we fine-tune: swap the 1000-class head for a 10-class one, then train the
whole network with two learning rates - a gentle one for the pretrained
backbone (we want to nudge those ImageNet features, not wipe them out) and a
10x bigger one for the brand-new head, which starts from scratch.

    python scripts/05_finetune_sota.py                       # 64x64, same input as our models
    python scripts/05_finetune_sota.py --input-size 128      # optional experiment: upsampled input
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from src.data import get_dataloaders
from src.engine import run_experiment
from src.models import BUILDERS, head_parameters
from src.utils import get_device, set_seed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["mobilenet_v2", "shufflenet_v2"])
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-4, help="backbone LR (head uses 10x)")
    ap.add_argument("--input-size", type=int, default=64)
    args = ap.parse_args()

    device = get_device()
    loaders, classes = get_dataloaders(batch_size=args.batch_size, input_size=args.input_size)

    for name in args.models:
        set_seed(42)
        model = BUILDERS[name](len(classes)).to(device)
        head_ids = {id(p) for p in head_parameters(name, model)}
        backbone = [p for p in model.parameters() if id(p) not in head_ids]
        head = [p for p in model.parameters() if id(p) in head_ids]

        opt = torch.optim.AdamW([{"params": backbone, "lr": args.lr},
                                 {"params": head, "lr": args.lr * 10}], weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
        tag = name if args.input_size == 64 else f"{name}_{args.input_size}"
        run_experiment(tag, model, loaders, classes, opt, args.epochs, device, sched,
                       input_size=args.input_size,
                       extra={"optimizer": f"AdamW backbone lr={args.lr}, head lr={args.lr * 10}, cosine"})


if __name__ == "__main__":
    main()
