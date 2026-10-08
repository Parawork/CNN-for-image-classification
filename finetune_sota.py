"""Task 5 - Fine-tune two lightweight pre-trained networks on the same splits.

MobileNetV2 and ShuffleNetV2 (x1.0), ImageNet weights from torchvision.
Strategy: replace the classifier, then fine-tune the WHOLE network with
discriminative learning rates - a small LR for the pre-trained backbone (so the
ImageNet features are adapted, not destroyed) and a 10x larger LR for the new head.

    python finetune_sota.py                       # 64x64 input (same as the custom models)
    python finetune_sota.py --input-size 128      # optional: upsampled input
"""
import argparse

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
