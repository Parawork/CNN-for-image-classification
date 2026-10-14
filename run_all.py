"""One command to run the whole assignment, start to finish.

    python run_all.py                 # everything, full epochs (~1-1.5 h on an RTX 3060 Ti)
    python run_all.py --quick         # 2 epochs everywhere - a smoke test to check nothing is broken
    python run_all.py --from 4        # pick up again from step 4 (e.g. after tweaking a model)
    python run_all.py --only 3        # just one step
    python run_all.py --clean         # wipe outputs/ (figures, results, checkpoints) and stop
    python run_all.py --clean --quick # fresh smoke test

Each step is an ordinary script in scripts/, so you can always run one by hand
if you want to pass it extra options (`python scripts/04_train_custom.py --help`).
"""
import argparse
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUTPUTS = ROOT / "outputs"

# (step number, script, what it does, takes --epochs?)
STEPS = [
    (1, "01_prepare_data.py", "Download EuroSAT, resize to 64x64, make the 70/15/15 split", False),
    (2, "02_model_summary.py", "Layer tables + check our hand-counted parameters", False),
    (3, "03_optimizer_study.py", "Adam vs SGD vs SGD+momentum (7 runs - the slow one)", True),
    (4, "04_train_custom.py", "Train and test Model A and Model B", True),
    (5, "05_finetune_sota.py", "Fine-tune MobileNetV2 and ShuffleNetV2", True),
    (6, "06_compare.py", "Comparison tables + accuracy-vs-size plot", False),
]


def clean():
    """Delete everything the runs produce. The downloaded dataset in data/ is kept."""
    for sub in ("figures", "results", "checkpoints"):
        target = OUTPUTS / sub
        if target.exists():
            shutil.rmtree(target)
            print(f"  removed {target.relative_to(ROOT)}")
        target.mkdir(parents=True, exist_ok=True)
    print("Outputs cleaned (data/ left alone, so no re-download needed).")


def fmt(seconds):
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    return f"{h}h {m:02d}m {s:02d}s" if h else f"{m}m {s:02d}s"


def main():
    ap = argparse.ArgumentParser(description="Run the EN3150 A03 pipeline.")
    ap.add_argument("--quick", action="store_true", help="2 epochs per training step (smoke test)")
    ap.add_argument("--from", dest="start", type=int, default=1, metavar="N", help="start at step N")
    ap.add_argument("--only", type=int, metavar="N", help="run only step N")
    ap.add_argument("--clean", action="store_true", help="delete outputs/ first")
    args = ap.parse_args()

    if args.clean:
        clean()
        # `--clean` on its own just tidies up; combined with other flags it cleans then runs
        if len(sys.argv) == 2:
            return

    todo = [s for s in STEPS if (s[0] == args.only if args.only else s[0] >= args.start)]
    if args.quick:
        print("Quick mode: training steps run for 2 epochs. Numbers will be poor - that's expected.\n")

    t_total = time.perf_counter()
    for num, script, desc, has_epochs in todo:
        print("=" * 78)
        print(f" Step {num}/6 - {desc}")
        print("=" * 78, flush=True)

        cmd = [sys.executable, str(ROOT / "scripts" / script)]
        if args.quick and has_epochs:
            cmd += ["--epochs", "2"]

        t0 = time.perf_counter()
        ret = subprocess.run(cmd, cwd=ROOT).returncode
        if ret != 0:
            print(f"\nStep {num} failed (exit code {ret}). Fix it, then resume with:"
                  f"\n    python run_all.py --from {num}")
            sys.exit(ret)
        print(f"\nStep {num} done in {fmt(time.perf_counter() - t0)}\n")

    print(f"All done in {fmt(time.perf_counter() - t_total)}. "
          f"Figures are in outputs/figures, tables in outputs/results/comparison.md")


if __name__ == "__main__":
    main()
