from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np

_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_UTILS_DIR) not in sys.path:
    sys.path.append(str(_UTILS_DIR))

from arc_utils import Color


DIR_TO_EDGE = {
    "left": (0, -1),
    "right": (0, 1),
    "up": (-1, 0),
    "down": (1, 0),
}

OPPOSITE_DIR = {
    "left": "right",
    "right": "left",
    "up": "down",
    "down": "up",
}

GEN_ATTEMPTS = 400
HEIGHT_RANGE = (10, 16)
WIDTH_RANGE = (10, 16)


def _marker_info(grid: np.ndarray) -> tuple[str, tuple[int, int]]:
    points = np.argwhere(grid == Color.TEAL)
    assert len(points) == 1, "expected exactly one teal edge marker"
    r, c = map(int, points[0])
    h, w = map(int, grid.shape)
    assert (r in {0, h - 1}) ^ (c in {0, w - 1}), "marker must be on one edge only"
    assert 0 < r < h - 1 or 0 < c < w - 1, "marker must not be on a corner"
    if c == 0:
        return "left", (r, c)
    if c == w - 1:
        return "right", (r, c)
    if r == 0:
        return "up", (r, c)
    return "down", (r, c)


def _detect_bars(grid: np.ndarray) -> tuple[list[int], list[int]]:
    g = np.asarray(grid, dtype=int)
    h, w = map(int, g.shape)
    h_rows = [r for r in range(h) if not np.any(g[r] == Color.BLACK) and not np.any(g[r] == Color.TEAL)]
    v_cols = [c for c in range(w) if not np.any(g[:, c] == Color.BLACK) and not np.any(g[:, c] == Color.TEAL)]

    assert 2 <= len(h_rows) <= 3, f"expected 2-3 horizontal bars, got {h_rows}"
    assert 2 <= len(v_cols) <= 3, f"expected 2-3 vertical bars, got {v_cols}"

    h_set = set(h_rows)
    v_set = set(v_cols)
    for r in range(h):
        for c in range(w):
            value = int(g[r, c])
            if value == Color.TEAL:
                continue
            if r in h_set and c in v_set:
                assert value == Color.RED, "input crossings must show the vertical bar on top"
            elif r in h_set:
                assert value == Color.BLUE, "horizontal bars must be blue away from crossings"
            elif c in v_set:
                assert value == Color.RED, "vertical bars must be red"
            else:
                assert value == Color.BLACK, "off-bar cells must be background"
    return h_rows, v_cols


def _swap_crossings(grid: np.ndarray, h_rows: list[int], v_cols: list[int]) -> np.ndarray:
    out = np.array(grid, copy=True)
    for r in h_rows:
        for c in v_cols:
            out[r, c] = Color.BLUE
    return out


def _paint_trail(trail: np.ndarray, fixed_row: int, fixed_col: int, old: int, new: int, horizontal: bool) -> None:
    if old == new:
        return
    step = -1 if old > new else 1
    for pos in range(old, new, step):
        if horizontal:
            trail[fixed_row, pos] = True
        else:
            trail[pos, fixed_col] = True


def _available_indices(length: int, fixed: int | None) -> list[int]:
    return [idx for idx in range(length) if idx != fixed]


def _compact_horizontal(grid: np.ndarray, to_left: bool, marker_pos: tuple[int, int]) -> np.ndarray:
    h, w = map(int, grid.shape)
    mr, mc = marker_pos
    settled = np.full_like(grid, Color.BLACK)
    trail = np.zeros((h, w), dtype=bool)
    settled[mr, mc] = Color.TEAL

    for r in range(h):
        fixed = mc if r == mr else None
        positions = [c for c in range(w) if int(grid[r, c]) not in {Color.BLACK, Color.TEAL}]
        colors = [int(grid[r, c]) for c in positions]
        available = _available_indices(w, fixed)
        targets = available[: len(positions)] if to_left else available[-len(positions) :]
        for old, new, color in zip(positions, targets, colors):
            settled[r, new] = color
            _paint_trail(trail, r, -1, old, new, True)

    out = np.full_like(grid, Color.BLACK)
    out[trail] = Color.MAROON
    out[settled != Color.BLACK] = settled[settled != Color.BLACK]
    return out


def _compact_vertical(grid: np.ndarray, to_top: bool, marker_pos: tuple[int, int]) -> np.ndarray:
    h, w = map(int, grid.shape)
    mr, mc = marker_pos
    settled = np.full_like(grid, Color.BLACK)
    trail = np.zeros((h, w), dtype=bool)
    settled[mr, mc] = Color.TEAL

    for c in range(w):
        fixed = mr if c == mc else None
        positions = [r for r in range(h) if int(grid[r, c]) not in {Color.BLACK, Color.TEAL}]
        colors = [int(grid[r, c]) for r in positions]
        available = _available_indices(h, fixed)
        targets = available[: len(positions)] if to_top else available[-len(positions) :]
        for old, new, color in zip(positions, targets, colors):
            settled[new, c] = color
            _paint_trail(trail, -1, c, old, new, False)

    out = np.full_like(grid, Color.BLACK)
    out[trail] = Color.MAROON
    out[settled != Color.BLACK] = settled[settled != Color.BLACK]
    return out


def _compact(grid: np.ndarray, direction: str, marker_pos: tuple[int, int]) -> np.ndarray:
    if direction == "left":
        return _compact_horizontal(grid, True, marker_pos)
    if direction == "right":
        return _compact_horizontal(grid, False, marker_pos)
    if direction == "up":
        return _compact_vertical(grid, True, marker_pos)
    if direction == "down":
        return _compact_vertical(grid, False, marker_pos)
    raise ValueError(direction)


def _transform(grid: np.ndarray, *, direction_override: str | None = None) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    direction, marker_pos = _marker_info(g)
    h_rows, v_cols = _detect_bars(g)
    swapped = _swap_crossings(g, h_rows, v_cols)
    return _compact(swapped, direction_override or direction, marker_pos)


def main(grid: np.ndarray) -> np.ndarray:
    """Swap the original row-column crossings, then compress toward the teal-marked edge while leaving maroon sweep trails.

    Input structure:
    - Black background surrounds 2-3 one-pixel-thick blue horizontal bars and 2-3 one-pixel-thick red vertical bars.
    - Every horizontal bar fills its entire row, every vertical bar fills its entire column, and their crossings visibly use the vertical-bar red.
    - Exactly one teal marker sits on a non-corner border cell and indicates the compression direction.

    Rule:
    - Detect the fully filled horizontal rows and vertical columns.
    - Recolor every original crossing from red to blue before any movement.
    - Compress all movable non-background pixels toward the marker edge, line by line: use stable row compaction for left/right and stable column compaction for up/down, while the teal marker stays fixed.
    - For each moved pixel, paint maroon on every traversed cell from its original location up to but excluding its final location, then show the compressed colors and the unchanged teal marker.

    Key insight:
    - The intersections are defined only by the pre-compression geometry, but the visible result depends on how those already-swapped crossing pixels participate in the later stable compaction and leave sweep trails.
    """
    return _transform(grid)


def _build_grid(
    h: int,
    w: int,
    h_rows: list[int],
    v_cols: list[int],
    direction: str,
    marker_index: int,
) -> np.ndarray:
    grid = np.full((h, w), Color.BLACK, dtype=int)
    for r in h_rows:
        grid[r, :] = Color.BLUE
    for c in v_cols:
        grid[:, c] = Color.RED

    if direction == "left":
        grid[marker_index, 0] = Color.TEAL
    elif direction == "right":
        grid[marker_index, w - 1] = Color.TEAL
    elif direction == "up":
        grid[0, marker_index] = Color.TEAL
    else:
        grid[h - 1, marker_index] = Color.TEAL
    return grid


def _sample_positions(length: int, count: int) -> list[int]:
    choices = list(range(1, length - 1))
    return sorted(random.sample(choices, count))


def _generate_attempt() -> np.ndarray | None:
    h = random.randint(*HEIGHT_RANGE)
    w = random.randint(*WIDTH_RANGE)
    n_h = random.randint(2, 3)
    n_v = random.randint(2, 3)
    direction = random.choice(list(DIR_TO_EDGE))

    h_rows = _sample_positions(h, n_h)
    v_cols = _sample_positions(w, n_v)

    if direction in {"left", "right"}:
        marker_candidates = [r for r in range(1, h - 1) if r not in h_rows]
    else:
        marker_candidates = [c for c in range(1, w - 1) if c not in v_cols]
    if not marker_candidates:
        return None

    marker_index = random.choice(marker_candidates)
    grid = _build_grid(h, w, h_rows, v_cols, direction, marker_index)

    try:
        out = _transform(grid)
    except AssertionError:
        return None

    swapped_only = _swap_crossings(grid, h_rows, v_cols)
    wrong_no_swap = _compact(grid, direction, _marker_info(grid)[1])
    wrong_opposite_dir = _transform(grid, direction_override=OPPOSITE_DIR[direction])

    if not np.any(out == Color.MAROON):
        return None
    if np.array_equal(out, swapped_only):
        return None
    if np.array_equal(out, wrong_no_swap):
        return None
    if np.array_equal(out, wrong_opposite_dir):
        return None
    if np.array_equal(out, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid crossing-compression instance")
