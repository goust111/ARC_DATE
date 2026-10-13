from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_ARC_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_ARC_UTILS_DIR) not in sys.path:
    sys.path.append(str(_ARC_UTILS_DIR))

from arc_utils import Color, get_bg, in_bounds


MOVER = Color.RED
TRIGGER = Color.GREY
LEFT_FILL = Color.GREEN
RIGHT_FILL = Color.BLUE
BACKGROUND = Color.BLACK
PANEL_SHAPE = (3, 6)
PANEL_SLOTS = ((0, 0), (0, 2), (2, 0), (2, 2))
MAX_GENERATION_ATTEMPTS = 400
_GEN_RNG = np.random.default_rng()


def _render_output(
    shape: tuple[int, int], trigger_row: int, red_coords: list[tuple[int, int]]
) -> np.ndarray:
    height, width = map(int, shape)
    out = np.full((height, width), BACKGROUND, dtype=int)
    out[int(trigger_row), :] = TRIGGER

    left_count = 0
    right_count = 0
    moved_coords: list[tuple[int, int]] = []
    for row, col in red_coords:
        moved = (int(row) + 1, int(col))
        moved_coords.append(moved)
        out[moved] = MOVER

    for row, col in moved_coords:
        if row != int(trigger_row) - 1:
            continue
        if col < width // 2:
            left_count += 1
        else:
            right_count += 1

        for dc in (-1, 1):
            rr, cc = row - 1, col + dc
            while in_bounds(out.shape, rr, cc):
                out[rr, cc] = MOVER
                rr -= 1
                cc += dc

    panel = out[-PANEL_SHAPE[0] :, -PANEL_SHAPE[1] :]
    panel[:, :] = BACKGROUND
    for rr, cc in PANEL_SLOTS[:left_count]:
        panel[rr, cc] = LEFT_FILL
    for rr, cc in PANEL_SLOTS[:right_count]:
        panel[rr, cc + 3] = RIGHT_FILL
    return out


def _find_trigger_row(grid: np.ndarray) -> int:
    rows = [r for r in range(grid.shape[0]) if np.all(grid[r, :] == TRIGGER)]
    assert len(rows) == 1, "expected exactly one full-width grey trigger row"
    return int(rows[0])


def _parse_red_pixels(grid: np.ndarray, trigger_row: int) -> list[tuple[int, int]]:
    coords = np.argwhere(grid == MOVER)
    assert 4 <= len(coords) <= 8, "expected 4 to 8 mover pixels"
    cols = [int(c) for _r, c in coords]
    assert len(set(cols)) == len(cols), "expected distinct mover columns"

    red_coords: list[tuple[int, int]] = []
    for row, col in coords:
        row = int(row)
        col = int(col)
        assert row <= int(trigger_row) - 2, "mover pixels must start at least two rows above the trigger"
        red_coords.append((row, col))
    return red_coords


def _semantic_stats(
    shape: tuple[int, int], trigger_row: int, red_coords: list[tuple[int, int]]
) -> tuple[int, int, int, int]:
    _, width = map(int, shape)
    left_total = sum(1 for _r, c in red_coords if c < width // 2)
    right_total = len(red_coords) - left_total
    left_trigger = sum(1 for r, c in red_coords if r == trigger_row - 2 and c < width // 2)
    right_trigger = sum(1 for r, c in red_coords if r == trigger_row - 2 and c >= width // 2)
    return left_total, right_total, left_trigger, right_trigger


def main(grid: np.ndarray) -> np.ndarray:
    """
    Move the colored pixels downward, splash diagonally from trigger-adjacent landings, and encode left/right splash counts into a double tally panel.

    Input structure:
    - A black background contains one full-width grey trigger row and several same-color single-pixel movers above it.
    - The mover pixels occupy distinct columns, and the bottom-right 3x6 region below the trigger row is an empty black panel zone.
    - The trigger row spans an even number of columns, so its width splits into equal left and right halves.

    Rule:
    - Erase every mover pixel and redraw it one row lower in the same column.
    - Any moved pixel that lands directly above the grey trigger row emits same-color rays toward the upper-left and upper-right diagonals until leaving the grid.
    - Count triggering pixels whose columns lie in the left half of the trigger row and in the right half separately.
    - In the bottom-right 3x6 region, treat the left 3x3 block and right 3x3 block as tally panels; fill corner slots in the order top-left, top-right, bottom-left, bottom-right with green on the left for the left count and blue on the right for the right count.

    Key insight:
    - The splash is both a visible geometric effect and a counted event stream: downward motion decides which pixels trigger, the trigger column decides the left/right class, and those counts drive the two output panels.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    height, width = g.shape
    assert height >= 8 and width >= 6, "grid is too small for the trigger and panel layout"
    assert width % 2 == 0, "trigger row width must be even"
    assert max(g.shape) <= 30, "ARC grids must stay within 30x30"
    assert get_bg(g) == BACKGROUND, "expected black background"
    assert set(map(int, np.unique(g))) <= {BACKGROUND, MOVER, TRIGGER}, "unexpected input colors"

    trigger_row = _find_trigger_row(g)
    assert trigger_row <= height - 4, "expected at least three rows below the trigger row"
    assert np.all(g[-PANEL_SHAPE[0] :, -PANEL_SHAPE[1] :] == BACKGROUND), "expected an empty bottom-right panel zone"

    red_coords = _parse_red_pixels(g, trigger_row)
    left_total, right_total, left_trigger, right_trigger = _semantic_stats(g.shape, trigger_row, red_coords)
    assert 1 <= left_trigger + right_trigger <= 4, "expected 1 to 4 triggering pixels"
    assert left_trigger + right_trigger < left_total + right_total, "expected at least one non-trigger mover"

    return _render_output(g.shape, trigger_row, red_coords)


def _sample_trigger_split(width: int) -> tuple[int, int]:
    half = width // 2
    mode = str(_GEN_RNG.choice(np.array(["left_only", "right_only", "both"], dtype=object)))
    if mode == "left_only":
        return int(_GEN_RNG.integers(1, min(4, half) + 1)), 0
    if mode == "right_only":
        return 0, int(_GEN_RNG.integers(1, min(4, half) + 1))

    total = int(_GEN_RNG.integers(2, 5))
    left = int(_GEN_RNG.integers(1, total))
    right = total - left
    return left, right


def _sample_distinct_columns(candidates: list[int], count: int) -> list[int] | None:
    if count > len(candidates):
        return None
    picked = _GEN_RNG.choice(np.array(candidates, dtype=int), size=count, replace=False)
    return [int(v) for v in picked]


def _gen_attempt() -> np.ndarray | None:
    height = int(_GEN_RNG.integers(12, 17))
    width = int(_GEN_RNG.choice(np.array([10, 12, 14, 16], dtype=int)))
    trigger_row = int(_GEN_RNG.integers(7, height - 3))

    left_trigger, right_trigger = _sample_trigger_split(width)
    trigger_total = left_trigger + right_trigger
    mover_total = int(_GEN_RNG.integers(max(4, trigger_total + 1), 9))
    non_trigger_total = mover_total - trigger_total

    left_cols = list(range(0, width // 2))
    right_cols = list(range(width // 2, width))
    left_trigger_cols = _sample_distinct_columns(left_cols, left_trigger)
    right_trigger_cols = _sample_distinct_columns(right_cols, right_trigger)
    if left_trigger_cols is None or right_trigger_cols is None:
        return None

    used_cols = set(left_trigger_cols + right_trigger_cols)
    remaining_cols = [c for c in range(width) if c not in used_cols]
    extra_cols = _sample_distinct_columns(remaining_cols, non_trigger_total)
    if extra_cols is None:
        return None

    red_coords: list[tuple[int, int]] = []
    for col in left_trigger_cols + right_trigger_cols:
        red_coords.append((trigger_row - 2, int(col)))

    if non_trigger_total > 0:
        near_miss_col = int(extra_cols[0])
        red_coords.append((trigger_row - 3, near_miss_col))
        for col in extra_cols[1:]:
            row = int(_GEN_RNG.integers(max(0, trigger_row - 6), trigger_row - 2))
            row = min(row, trigger_row - 3)
            red_coords.append((row, int(col)))

    _GEN_RNG.shuffle(red_coords)
    grid = np.full((height, width), BACKGROUND, dtype=int)
    grid[trigger_row, :] = TRIGGER
    for row, col in red_coords:
        grid[int(row), int(col)] = MOVER

    if not _validate_generated_case(
        grid=grid,
        trigger_row=trigger_row,
        red_coords=red_coords,
        left_trigger=left_trigger,
        right_trigger=right_trigger,
    ):
        return None
    return grid


def _validate_generated_case(
    *,
    grid: np.ndarray,
    trigger_row: int,
    red_coords: list[tuple[int, int]],
    left_trigger: int,
    right_trigger: int,
) -> bool:
    if grid.ndim != 2 or max(grid.shape) > 30:
        return False
    if not np.all(grid[-PANEL_SHAPE[0] :, -PANEL_SHAPE[1] :] == BACKGROUND):
        return False

    left_total, right_total, actual_left_trigger, actual_right_trigger = _semantic_stats(
        grid.shape, trigger_row, red_coords
    )
    if (actual_left_trigger, actual_right_trigger) != (left_trigger, right_trigger):
        return False
    if actual_left_trigger + actual_right_trigger not in range(1, 5):
        return False
    if actual_left_trigger + actual_right_trigger >= len(red_coords):
        return False
    if (left_total, right_total) == (actual_left_trigger, actual_right_trigger):
        return False

    expected = _render_output(grid.shape, trigger_row, red_coords)
    try:
        actual = main(grid)
    except Exception:
        return False
    if not np.array_equal(actual, expected):
        return False
    if np.array_equal(grid, actual):
        return False
    return True


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input for puzzle 8")
