import importlib.util
from pathlib import Path

import numpy as np

TASK_ID = 7
_spec = importlib.util.spec_from_file_location(f"arc057_common_{TASK_ID}", Path(__file__).with_name("_common.py"))
_common = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_common)


def main(grid: np.ndarray) -> np.ndarray:
    return _common.main_for(TASK_ID, grid)


def generate_input() -> np.ndarray:
    return _common.generate_for(TASK_ID)
