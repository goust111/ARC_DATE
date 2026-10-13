import random
import sys
from pathlib import Path

import numpy as np

_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_UTILS_DIR) not in sys.path:
    sys.path.append(str(_UTILS_DIR))

from arc_utils import Color, components_mask


GRID_H = 12
PANEL_W = 12
GRID_W = PANEL_W * 2
OUTPUT_H = GRID_H - 1

LEFT_FG = int(Color.BLUE)
RIGHT_FG = int(Color.RED)
LEFT_CTRL = int(Color.GREEN)
RIGHT_CTRL = int(Color.YELLOW)
OUT_MARK = int(Color.GREY)
BG = int(Color.BLACK)

LEFT_CELL_RANGE = [(r, c) for r in range(1, GRID_H) for c in range(PANEL_W)]

GEN_RETRIES = 400
PANEL_RETRIES = 120


def _toggle_panel(panel: np.ndarray, control_color: int, fg_color: int) -> np.ndarray:
    out = np.array(panel, dtype=int, copy=True)
    control_cols = [int(c) for c in np.flatnonzero(out[0] == control_color)]
    for c in control_cols:
        col = out[1:, c]
        assert np.all((col == BG) | (col == fg_color)), "ray column contains unexpected colors"
        out[1:, c] = np.where(col == fg_color, BG, fg_color)
    return out


def _output_from_panels(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    out = np.zeros((left.shape[0] - 1, left.shape[1]), dtype=int)
    out[(left[1:] == BG) & (right[1:] == BG)] = OUT_MARK
    return out


def _solver_from_grid(grid: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    g = np.asarray(grid, dtype=int)
    assert g.shape == (GRID_H, GRID_W), f"expected {(GRID_H, GRID_W)}, got {g.shape}"

    left = np.array(g[:, :PANEL_W], copy=True)
    right = np.array(g[:, PANEL_W:], copy=True)

    assert set(map(int, np.unique(left))).issubset({BG, LEFT_FG, LEFT_CTRL})
    assert set(map(int, np.unique(right))).issubset({BG, RIGHT_FG, RIGHT_CTRL})

    left_controls = [int(c) for c in np.flatnonzero(left[0] == LEFT_CTRL)]
    right_controls = [int(c) for c in np.flatnonzero(right[0] == RIGHT_CTRL)]
    assert 2 <= len(left_controls) <= 3
    assert 2 <= len(right_controls) <= 3
    assert set(left_controls).isdisjoint(right_controls)
    assert not np.any(left[0] == LEFT_FG)
    assert not np.any(right[0] == RIGHT_FG)

    left_after = _toggle_panel(left, LEFT_CTRL, LEFT_FG)
    right_after = _toggle_panel(right, RIGHT_CTRL, RIGHT_FG)
    return left_after, right_after, _output_from_panels(left_after, right_after)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Mark the aligned cells that remain empty in both panels after their top-row control rays toggle occupancy down each marked column.

    Input structure:
    - The grid is 12x24 with black background and is split into two aligned 12x12 panels.
    - The left panel contains sparse blue pixels plus 2 or 3 green control pixels on the top row.
    - The right panel contains sparse red pixels plus 2 or 3 yellow control pixels on the top row.

    Rule:
    - In the left panel, each green top-row control toggles every cell below it in the same column: blue becomes black and black becomes blue, while the green control cell stays green.
    - In the right panel, each yellow top-row control toggles every cell below it in the same column: red becomes black and black becomes red, while the yellow control cell stays yellow.
    - After both panels are updated, align them by row and panel-local column, discard the top control row, and output an 11x12 grid where a cell is grey only when both panels are black at that aligned position; all other cells are black.

    Key insight:
    - The answer is the occupancy complement of the two post-ray panels, so the sparse shapes alone are insufficient: the control columns must first invert each panel's occupied cells before the shared empty cells can be identified.
    """
    _left_after, _right_after, out = _solver_from_grid(grid)
    return out


def _empty_mask(panel: np.ndarray) -> np.ndarray:
    return panel[1:] == BG


def _mark(mask: np.ndarray) -> np.ndarray:
    out = np.zeros(mask.shape, dtype=int)
    out[mask] = OUT_MARK
    return out


def _sample_scatter(control_cols: set[int], fg_color: int) -> np.ndarray | None:
    for _ in range(PANEL_RETRIES):
        target = random.randint(8, 15)
        cells: set[tuple[int, int]] = set()

        for c in control_cols:
            rows = random.sample(range(1, GRID_H), k=random.randint(1, min(3, OUTPUT_H)))
            for r in rows:
                cells.add((r, c))

        while len(cells) < target:
            cells.add(random.choice(LEFT_CELL_RANGE))

        if len(cells) > target:
            cells = set(random.sample(list(cells), target))
            if any(not any(cell[1] == c for cell in cells) for c in control_cols):
                continue

        rows = {r for r, _c in cells}
        cols = {c for _r, c in cells}
        control_hits = sum(1 for _r, c in cells if c in control_cols)
        off_control_hits = target - control_hits
        if len(rows) < 4 or len(cols) < 4:
            continue
        if off_control_hits < 4:
            continue

        mask = np.zeros((OUTPUT_H, PANEL_W), dtype=bool)
        for r, c in cells:
            mask[r - 1, c] = True
        if len(components_mask(mask, connectivity=4)) < 3:
            continue

        rr, cc = np.where(mask)
        height = int(rr.max() - rr.min() + 1)
        width = int(cc.max() - cc.min() + 1)
        if height * width <= target + 4:
            continue

        panel = np.zeros((GRID_H, PANEL_W), dtype=int)
        panel[0, list(control_cols)] = LEFT_CTRL if fg_color == LEFT_FG else RIGHT_CTRL
        for r, c in cells:
            panel[r, c] = fg_color
        return panel
    return None


def _apply_subset_controls(panel: np.ndarray, control_color: int, fg_color: int, active_cols: set[int]) -> np.ndarray:
    out = np.array(panel, dtype=int, copy=True)
    for c in active_cols:
        col = out[1:, c]
        out[1:, c] = np.where(col == fg_color, BG, fg_color)
    return out


def _semantic_checks(grid: np.ndarray) -> bool:
    try:
        left = np.array(grid[:, :PANEL_W], copy=True)
        right = np.array(grid[:, PANEL_W:], copy=True)
        left_after, right_after, out = _solver_from_grid(grid)
    except Exception:
        return False

    if np.count_nonzero(out == OUT_MARK) < 3:
        return False
    if np.count_nonzero(out == OUT_MARK) > 70:
        return False
    if not np.any(out == 0):
        return False

    raw = _output_from_panels(left, right)
    if np.array_equal(raw, out):
        return False

    left_only = _output_from_panels(left_after, right)
    right_only = _output_from_panels(left, right_after)
    if np.array_equal(left_only, out) or np.array_equal(right_only, out):
        return False

    if np.array_equal(_mark(_empty_mask(left_after)), out):
        return False
    if np.array_equal(_mark(_empty_mask(right_after)), out):
        return False

    left_controls = set(map(int, np.flatnonzero(left[0] == LEFT_CTRL)))
    right_controls = set(map(int, np.flatnonzero(right[0] == RIGHT_CTRL)))
    if len(left_controls) != len(np.flatnonzero(left[0] == LEFT_CTRL)):
        return False
    if len(right_controls) != len(np.flatnonzero(right[0] == RIGHT_CTRL)):
        return False

    if not np.any(left[1:] != left_after[1:]) or not np.any(right[1:] != right_after[1:]):
        return False

    for c in left_controls:
        partial = _apply_subset_controls(left, LEFT_CTRL, LEFT_FG, left_controls - {c})
        if np.array_equal(_output_from_panels(partial, right_after), out):
            return False
    for c in right_controls:
        partial = _apply_subset_controls(right, RIGHT_CTRL, RIGHT_FG, right_controls - {c})
        if np.array_equal(_output_from_panels(left_after, partial), out):
            return False

    changed = out != raw
    if np.count_nonzero(changed) < 6:
        return False

    rows_with_change = np.flatnonzero(np.any(changed, axis=1))
    cols_with_change = np.flatnonzero(np.any(changed, axis=0))
    if len(rows_with_change) < 3 or len(cols_with_change) < 3:
        return False

    return True


def _build_attempt() -> np.ndarray | None:
    left_count = random.choice((2, 2, 3))
    right_count = random.choice((2, 2, 3))
    left_controls = set(random.sample(range(PANEL_W), k=left_count))
    right_pool = [c for c in range(PANEL_W) if c not in left_controls]
    right_controls = set(random.sample(right_pool, k=right_count))

    left_panel = _sample_scatter(left_controls, LEFT_FG)
    right_panel = _sample_scatter(right_controls, RIGHT_FG)
    if left_panel is None or right_panel is None:
        return None

    grid = np.zeros((GRID_H, GRID_W), dtype=int)
    grid[:, :PANEL_W] = left_panel
    grid[:, PANEL_W:] = right_panel

    return grid if _semantic_checks(grid) else None


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _build_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
