from __future__ import annotations

import itertools
import random

import numpy as np


GRID_H = 12
PANEL_W = 12
SEP_COL = PANEL_W
GRID_W = PANEL_W * 2 + 1

BLACK = 0
BLUE = 1
RED = 2
YELLOW = 4
GREY = 5

VALID_COLUMN_SETS = tuple(
    cols
    for size in range(3, 6)
    for cols in itertools.combinations(range(PANEL_W), size)
    if all(b - a >= 2 for a, b in zip(cols, cols[1:]))
)


def _column_runs(mask_col: np.ndarray) -> list[tuple[int, int]]:
    runs: list[tuple[int, int]] = []
    r = 0
    while r < len(mask_col):
        while r < len(mask_col) and not bool(mask_col[r]):
            r += 1
        if r == len(mask_col):
            break
        start = r
        while r < len(mask_col) and bool(mask_col[r]):
            r += 1
        runs.append((start, r - start))
    return runs


def _find_separator_col(grid: np.ndarray) -> int:
    candidates = [c for c in range(grid.shape[1]) if np.all(grid[:, c] == GREY)]
    if not candidates:
        raise AssertionError("expected a full grey separator column")
    mid = grid.shape[1] // 2
    return min(candidates, key=lambda c: (abs(c - mid), c))


def _recolor_panel(panel: np.ndarray, top_color: int) -> np.ndarray:
    out = np.asarray(panel, dtype=int).copy()
    for c in range(out.shape[1]):
        for start, height in _column_runs(out[:, c] == top_color):
            lower_start = start + (height + 1) // 2
            out[lower_start : start + height, c] = YELLOW
    return out


def _overlay_panels(left_panel: np.ndarray, right_panel: np.ndarray) -> np.ndarray:
    out = np.asarray(left_panel, dtype=int).copy()
    mask = (out == BLACK) & (np.asarray(right_panel, dtype=int) == YELLOW)
    out[mask] = YELLOW
    return out


def _extract_bar_specs(panel: np.ndarray, color: int) -> dict[int, tuple[int, int]]:
    panel = np.asarray(panel, dtype=int)
    if not np.all(np.isin(panel, (BLACK, color))):
        raise AssertionError("panel contains unexpected colors")
    specs: dict[int, tuple[int, int]] = {}
    for c in range(panel.shape[1]):
        runs = _column_runs(panel[:, c] == color)
        if not runs:
            continue
        if len(runs) != 1:
            raise AssertionError("each active column must contain exactly one vertical bar")
        specs[c] = runs[0]
    return specs


def _validate_structure(grid: np.ndarray) -> tuple[dict[int, tuple[int, int]], dict[int, tuple[int, int]]]:
    g = np.asarray(grid, dtype=int)
    if g.shape != (GRID_H, GRID_W):
        raise AssertionError(f"expected a {GRID_H}x{GRID_W} grid")
    if not np.all(g[:, SEP_COL] == GREY):
        raise AssertionError("separator column must be solid grey")

    left = g[:, :PANEL_W]
    right = g[:, SEP_COL + 1 :]
    left_specs = _extract_bar_specs(left, RED)
    right_specs = _extract_bar_specs(right, BLUE)

    for specs in (left_specs, right_specs):
        if not 3 <= len(specs) <= 5:
            raise AssertionError("each panel must contain 3 to 5 bars")
        cols = sorted(specs)
        if any(b - a < 2 for a, b in zip(cols, cols[1:])):
            raise AssertionError("bars inside one panel must keep at least one empty column between them")
        for start, height in specs.values():
            if not (4 <= height <= 10):
                raise AssertionError("bar heights must stay within 4..10")
            if not (0 <= start <= GRID_H - height):
                raise AssertionError("bar must stay in bounds")

    paired = set(left_specs) & set(right_specs)
    if len(paired) < 2:
        raise AssertionError("need at least two paired columns")
    if not (set(left_specs) - set(right_specs)):
        raise AssertionError("need at least one left-only column")
    if not (set(right_specs) - set(left_specs)):
        raise AssertionError("need at least one right-only column")
    if not any(left_specs[c] == right_specs[c] for c in paired):
        raise AssertionError("need an identical paired bar to demonstrate full blocking")
    if not any(right_specs[c][0] == left_specs[c][0] and right_specs[c][1] > left_specs[c][1] for c in paired):
        raise AssertionError("need a paired column where the right bar is taller")

    return left_specs, right_specs


def _validate_semantics(grid: np.ndarray) -> None:
    left_specs, right_specs = _validate_structure(grid)
    sep = _find_separator_col(grid)
    left = grid[:, :sep]
    right = grid[:, sep + 1 :]
    left_recolored = _recolor_panel(left, RED)
    right_recolored = _recolor_panel(right, BLUE)
    out = main(grid)
    expected = _overlay_panels(left_recolored, right_recolored)

    if not np.array_equal(out, expected):
        raise AssertionError("main() does not match the stated transformation")

    overlay_mask = (left_recolored == BLACK) & (right_recolored == YELLOW)
    if not np.any(overlay_mask):
        raise AssertionError("need at least one visible overlay pixel")
    paired = set(left_specs) & set(right_specs)
    if not any(np.any(overlay_mask[:, c]) for c in paired):
        raise AssertionError("a paired column must contribute new overlay pixels")
    if not any(left_specs[c] == right_specs[c] and not np.any(overlay_mask[:, c]) for c in paired):
        raise AssertionError("an identical pair must block overlay completely")
    if not np.any((right_recolored == BLUE) & (left_recolored == BLACK) & (out == BLACK)):
        raise AssertionError("need visible evidence that blue top halves do not copy")
    if np.array_equal(out, left_recolored):
        raise AssertionError("right panel must affect the output")


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor each panel's vertical bars by half, then copy only yellow right-panel pixels into empty left-panel cells.

    Input structure:
    - The grid is split into a left 12x12 panel and a right 12x12 panel by one full grey separator column.
    - The left panel contains red one-cell-wide vertical bars on black; the right panel contains blue vertical bars on black.
    - Each bar is a single contiguous vertical segment inside one column.

    Rule:
    - In each left red bar, keep the upper ceil(height/2) cells red and recolor the lower floor(height/2) cells yellow.
    - In each right blue bar, keep the upper ceil(height/2) cells blue and recolor the lower floor(height/2) cells yellow.
    - Start from the recolored left panel; for every position where the left panel is black and the recolored right panel is yellow, copy that yellow cell into the left panel.
    - Do not copy blue cells, do not overwrite any non-black left cell, and drop the separator so the output is only the transformed left panel.

    Key insight:
    - The half-split recoloring turns the right panel's lower halves into a conditional overlay mask, while the left panel's occupied cells block that mask column by column.
    """
    g = np.asarray(grid, dtype=int)
    if g.ndim != 2:
        raise AssertionError("expected a 2D grid")

    sep = _find_separator_col(g)
    left = g[:, :sep]
    right = g[:, sep + 1 :]
    left_recolored = _recolor_panel(left, RED)
    right_recolored = _recolor_panel(right, BLUE)
    return _overlay_panels(left_recolored, right_recolored)


def _sample_columns() -> tuple[tuple[int, ...], tuple[int, ...]]:
    for _ in range(400):
        left_cols = random.choice(VALID_COLUMN_SETS)
        right_cols = random.choice(VALID_COLUMN_SETS)
        paired = set(left_cols) & set(right_cols)
        if len(paired) < 2:
            continue
        if not (set(left_cols) - set(right_cols)):
            continue
        if not (set(right_cols) - set(left_cols)):
            continue
        return left_cols, right_cols
    raise RuntimeError("failed to sample compatible column sets")


def _random_bar(max_height: int = 10) -> tuple[int, int]:
    height = random.randint(4, max_height)
    start = random.randint(0, GRID_H - height)
    return start, height


def _build_specs(
    left_cols: tuple[int, ...], right_cols: tuple[int, ...]
) -> tuple[dict[int, tuple[int, int]], dict[int, tuple[int, int]]]:
    left_specs: dict[int, tuple[int, int]] = {}
    right_specs: dict[int, tuple[int, int]] = {}
    paired = sorted(set(left_cols) & set(right_cols))
    same_col, taller_col = random.sample(paired, 2)

    same_height = random.randint(4, 10)
    same_start = random.randint(0, GRID_H - same_height)
    left_specs[same_col] = (same_start, same_height)
    right_specs[same_col] = (same_start, same_height)

    left_height = random.randint(4, 8)
    right_height = random.randint(left_height + 1, min(10, left_height + 2))
    taller_start = random.randint(0, GRID_H - right_height)
    left_specs[taller_col] = (taller_start, left_height)
    right_specs[taller_col] = (taller_start, right_height)

    for c in sorted(set(left_cols) - {same_col, taller_col}):
        left_specs[c] = _random_bar()
    for c in sorted(set(right_cols) - {same_col, taller_col}):
        right_specs[c] = _random_bar()

    for c in sorted((set(left_cols) & set(right_cols)) - {same_col, taller_col}):
        mode = random.choice(("same", "left_taller", "right_taller"))
        if mode == "same":
            height = random.randint(4, 10)
            start = random.randint(0, GRID_H - height)
            left_specs[c] = (start, height)
            right_specs[c] = (start, height)
        elif mode == "left_taller":
            right_height = random.randint(4, 8)
            left_height = random.randint(right_height + 1, min(10, right_height + 2))
            start = random.randint(0, GRID_H - left_height)
            left_specs[c] = (start, left_height)
            right_specs[c] = (start, right_height)
        else:
            left_height = random.randint(4, 8)
            right_height = random.randint(left_height + 1, min(10, left_height + 2))
            start = random.randint(0, GRID_H - right_height)
            left_specs[c] = (start, left_height)
            right_specs[c] = (start, right_height)

    return left_specs, right_specs


def _render_grid(
    left_specs: dict[int, tuple[int, int]], right_specs: dict[int, tuple[int, int]]
) -> np.ndarray:
    grid = np.zeros((GRID_H, GRID_W), dtype=int)
    grid[:, SEP_COL] = GREY
    for c, (start, height) in left_specs.items():
        grid[start : start + height, c] = RED
    for c, (start, height) in right_specs.items():
        grid[start : start + height, SEP_COL + 1 + c] = BLUE
    return grid


def generate_input() -> np.ndarray:
    for _ in range(400):
        left_cols, right_cols = _sample_columns()
        left_specs, right_specs = _build_specs(left_cols, right_cols)
        grid = _render_grid(left_specs, right_specs)
        try:
            _validate_semantics(grid)
        except AssertionError:
            continue
        return grid
    raise RuntimeError("failed to generate a valid equal-split overlay puzzle")
