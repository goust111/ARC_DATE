import importlib.util
from pathlib import Path

import numpy as np


_HERE = Path(__file__)
FOLDER = int(_HERE.parent.name)
TASK_ID = int(_HERE.stem)
_COMMON = _HERE.parent.parent / "_generic_common.py"
_SPEC = importlib.util.spec_from_file_location(f"opus3_common_{FOLDER}_{TASK_ID}", _COMMON)
_MOD = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MOD)


def main(grid: np.ndarray) -> np.ndarray:
    return _MOD.main_for(FOLDER, TASK_ID, grid)


def generate_input() -> np.ndarray:
    return _MOD.generate_for(FOLDER, TASK_ID)
