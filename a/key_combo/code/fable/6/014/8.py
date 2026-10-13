from __future__ import annotations

from pathlib import Path
import random
import sys

import numpy as np


_HELPER_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_HELPER_DIR) not in sys.path:
    sys.path.append(str(_HELPER_DIR))

from arc_utils import Color


BG = int(Color.BLACK)
PALETTE = [int(color) for color in Color.NOT_BLACK]
HEIGHT_BOUNDS = (8, 20)
WIDTH_BOUNDS = (6, 18)
MAX_SEGMENT_LEN = 9
MAX_GENERATION_ATTEMPTS = 400


def _find_bead_run(column: np.ndarray) -> tuple[int, int] | None:
    rows = np.flatnonzero(column != BG)
    if rows.size == 0:
        return None
    splits = np.where(np.diff(rows) != 1)[0] + 1
    runs = np.split(rows, splits)
    run = max(runs, key=len)
    return int(run[0]), int(run[-1])


def _reverse_around_pivot(segment: np.ndarray, pivot_color: int) -> np.ndarray:
    pivot_rows = np.flatnonzero(segment == int(pivot_color))
    if pivot_rows.size != 1:
        return segment.copy()

    pivot = int(pivot_rows[0])
    out = segment.copy()
    out[:pivot] = segment[:pivot][::-1]
    out[pivot + 1 :] = segment[pivot + 1 :][::-1]
    return out


def main(input_grid: np.ndarray) -> np.ndarray:
    grid = np.asarray(input_grid, dtype=int)
    assert grid.ndim == 2, "expected a 2D grid"

    out = grid.copy()
    for col in np.flatnonzero(grid[0] != BG):
        control_color = int(grid[0, col])
        bead_run = _find_bead_run(grid[1:, col])
        if bead_run is None:
            continue

        r0, r1 = bead_run
        r0 += 1
        r1 += 1
        segment = grid[r0 : r1 + 1, col]
        out[r0 : r1 + 1, col] = _reverse_around_pivot(segment, control_color)
    return out


def _column_will_change(segment: np.ndarray, pivot: int) -> bool:
    upper = segment[:pivot]
    lower = segment[pivot + 1 :]
    return not np.array_equal(upper, upper[::-1]) or not np.array_equal(lower, lower[::-1])


def _sample_controlled_segment(max_len: int) -> tuple[int, np.ndarray]:
    control_color = random.choice(PALETTE)
    other_colors = [color for color in PALETTE if color != control_color]

    for _ in range(120):
        length = random.randint(4, max_len)
        pivot = random.randint(1, length - 2)
        segment = np.array([random.choice(other_colors) for _ in range(length)], dtype=int)
        segment[pivot] = control_color
        if _column_will_change(segment, pivot):
            return control_color, segment
    raise RuntimeError("failed to sample a controlled bead segment")


def _sample_plain_segment(max_len: int) -> np.ndarray:
    length = random.randint(1, max_len)
    return np.array([random.choice(PALETTE) for _ in range(length)], dtype=int)


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        height = random.randint(*HEIGHT_BOUNDS)
        width = random.randint(*WIDTH_BOUNDS)
        grid = np.full((height, width), BG, dtype=int)

        num_bead_columns = random.randint(3, min(width, 8))
        bead_columns = random.sample(range(width), num_bead_columns)
        num_controls = random.randint(1, num_bead_columns)
        control_columns = set(random.sample(bead_columns, num_controls))

        changed_controls = 0
        valid = True
        for col in bead_columns:
            max_start = height - 2
            if max_start < 1:
                valid = False
                break
            start = random.randint(1, max_start)
            remaining = height - start

            if col in control_columns:
                if remaining < 4:
                    valid = False
                    break
                control_color, segment = _sample_controlled_segment(min(MAX_SEGMENT_LEN, remaining))
                grid[0, col] = control_color
                changed_controls += 1
            else:
                segment = _sample_plain_segment(min(MAX_SEGMENT_LEN, remaining))

            grid[start : start + len(segment), col] = segment

        if valid and changed_controls > 0:
            return grid

    raise RuntimeError("failed to generate a valid input")
