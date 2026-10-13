from __future__ import annotations

import random

import numpy as np


BG = 0
BLUE = 1
RED = 2
YELLOW = 4
MAX_ATTEMPTS = 400


def _extract_seeds(grid: np.ndarray) -> tuple[list[int], list[int]]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    h, w = g.shape
    assert h >= 3 and w >= 3, "grid is too small for the edge sweeps"
    assert int(g[h - 1, 0]) == BG, "bottom-left corner must stay black"
    assert not np.any(g[:-1, 1:] != BG), "only edge seed pixels may be colored"
    assert np.all(np.isin(g, [BG, BLUE, RED])), "input may only contain black, blue, and red"
    assert np.all((g[:-1, 0] == BG) | (g[:-1, 0] == BLUE)), "left-edge seeds must be blue"
    assert np.all((g[h - 1, 1:] == BG) | (g[h - 1, 1:] == RED)), "bottom-edge seeds must be red"

    blue_rows = np.flatnonzero(g[:-1, 0] == BLUE).astype(int).tolist()
    red_cols = (np.flatnonzero(g[h - 1, 1:] == RED) + 1).astype(int).tolist()
    assert blue_rows, "at least one blue seed is required"
    assert red_cols, "at least one red seed is required"
    return blue_rows, red_cols


def _trace_masks(shape: tuple[int, int], blue_rows: list[int], red_cols: list[int]) -> tuple[np.ndarray, np.ndarray]:
    h, w = map(int, shape)
    blue_mask = np.zeros((h, w), dtype=bool)
    red_mask = np.zeros((h, w), dtype=bool)

    for r0 in blue_rows:
        steps = min(h - 1 - r0, w - 1)
        offset = np.arange(steps + 1, dtype=int)
        blue_mask[r0 + offset, offset] = True

    for c0 in red_cols:
        steps = min(h - 1, w - 1 - c0)
        offset = np.arange(steps + 1, dtype=int)
        red_mask[h - 1 - offset, c0 + offset] = True

    return blue_mask, red_mask


def _render_output(shape: tuple[int, int], blue_rows: list[int], red_cols: list[int]) -> np.ndarray:
    blue_mask, red_mask = _trace_masks(shape, blue_rows, red_cols)
    out = np.full(shape, BG, dtype=int)
    out[blue_mask] = BLUE
    out[red_mask] = RED
    out[blue_mask & red_mask] = YELLOW
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Sweep sparse left-edge and bottom-edge seeds into crossing diagonal traces with a third overlap color.

    Input structure:
    - A black grid contains blue seed pixels only on the left edge and red seed pixels only on the bottom edge.
    - The bottom-left corner is black, so the two seed lines do not start on the same cell.
    - No other colored pixels appear in the input.

    Rule:
    - Translate every blue left-edge seed repeatedly by (+1 row, +1 col), clipping any off-grid cells, and keep every visited cell.
    - Translate every red bottom-edge seed repeatedly by (-1 row, +1 col), again keeping every visited in-bounds cell.
    - Color blue-only trace cells blue, red-only trace cells red, overlap cells yellow, and leave uncovered cells black.

    Key insight:
    - The input stores only two sparse boundary masks; the full output is recovered by sweeping each seed family along its fixed diagonal direction and then combining the two coverage maps.
    """
    g = np.asarray(grid, dtype=int)
    blue_rows, red_cols = _extract_seeds(g)
    return _render_output(g.shape, blue_rows, red_cols)


def _sample_sparse_positions(low: int, high: int, count: int, min_gap: int = 2) -> list[int] | None:
    values = list(range(low, high + 1))
    if count <= 0 or len(values) < count:
        return None

    for _ in range(60):
        picked = sorted(random.sample(values, count))
        if all(b - a >= min_gap for a, b in zip(picked, picked[1:])):
            return picked

    return None


def _build_input(height: int, width: int, blue_rows: list[int], red_cols: list[int]) -> np.ndarray:
    grid = np.full((height, width), BG, dtype=int)
    grid[blue_rows, 0] = BLUE
    grid[height - 1, red_cols] = RED
    grid[height - 1, 0] = BG
    return grid


def _good_output(output: np.ndarray, mode: str) -> bool:
    blue_only = int(np.count_nonzero(output == BLUE))
    red_only = int(np.count_nonzero(output == RED))
    yellow = int(np.count_nonzero(output == YELLOW))
    colored = blue_only + red_only + yellow

    if yellow == 0 or blue_only == 0 or red_only == 0:
        return False
    if colored > int(output.size * 0.6):
        return False
    if yellow * 3 > colored:
        return False

    if mode == "single":
        return 1 <= yellow <= 2 and blue_only >= 5 and red_only >= 5
    return 3 <= yellow <= 7 and blue_only >= 6 and red_only >= 6


def _attempt_instance() -> np.ndarray | None:
    mode = random.choice(["single", "single", "multi", "multi", "multi"])
    if mode == "single":
        height = random.randint(12, 18)
        width = random.randint(12, 18)
        blue_count = random.randint(2, 3)
        red_count = random.randint(2, 3)
    else:
        height = random.randint(13, 20)
        width = random.randint(13, 20)
        blue_count = random.randint(2, 4)
        red_count = random.randint(2, 4)

    blue_rows = _sample_sparse_positions(0, height - 2, blue_count, min_gap=random.choice([1, 2, 2]))
    red_cols = _sample_sparse_positions(1, width - 1, red_count, min_gap=random.choice([1, 2, 2]))
    if blue_rows is None or red_cols is None:
        return None

    grid = _build_input(height, width, blue_rows, red_cols)
    output = main(grid)
    if not _good_output(output, mode):
        return None

    if not np.array_equal(output, _render_output(grid.shape, blue_rows, red_cols)):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        grid = _attempt_instance()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {MAX_ATTEMPTS} attempts")
