"""Every folder the project reads from or writes to, in one place.

Paths are worked out from this file's location rather than the current working
directory, so the scripts behave the same whether you launch them from the repo
root, from scripts/, or from your IDE's run button.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = ROOT / "data"                     # raw download + cached 64x64 tensors (git-ignored)
OUTPUTS_DIR = ROOT / "outputs"
FIGURES_DIR = OUTPUTS_DIR / "figures"        # plots that go straight into the report
RESULTS_DIR = OUTPUTS_DIR / "results"        # JSON metrics + comparison tables
CKPT_DIR = OUTPUTS_DIR / "checkpoints"       # trained weights (git-ignored, but sizes are reported)

for _d in (DATA_DIR, FIGURES_DIR, RESULTS_DIR, CKPT_DIR):
    _d.mkdir(parents=True, exist_ok=True)
