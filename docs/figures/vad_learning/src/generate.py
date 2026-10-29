"""Regenerate all 14 learning figures without importing the VAD training stack.

Run from the repository root:
    python3 docs/figures/vad_learning/src/generate.py
"""
import os
from pathlib import Path
import sys
import tempfile

# Keep font caches and Python bytecode out of the documentation assets.
os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "vad-learning-mpl"))
sys.dont_write_bytecode = True

import core_figures
import data_figures
import model_figures
import training_figures


def main():
    for module in (core_figures, model_figures, data_figures, training_figures):
        module.generate()
    output_dir = Path(__file__).resolve().parent.parent
    outputs = sorted(output_dir.glob("[0-9][0-9]_*.png"))
    for output in outputs:
        print(output.name)
    print(f"Generated {len(outputs)} PNG figures in {output_dir}")


if __name__ == "__main__":
    main()
