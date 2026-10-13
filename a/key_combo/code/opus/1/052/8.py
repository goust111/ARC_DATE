import sys
from pathlib import Path

import numpy as np

_BASE = Path(__file__).resolve().parents[1]
if str(_BASE) not in sys.path:
    sys.path.insert(0, str(_BASE))

from arc_codegen_051_057_common import generate_for, main_for


def main(grid: np.ndarray) -> np.ndarray:
    return main_for(52, 8, grid)


def generate_input() -> np.ndarray:
    return generate_for(52, 8)
