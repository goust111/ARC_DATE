from __future__ import annotations

import random
from importlib.machinery import SourceFileLoader
from pathlib import Path

import numpy as np


def _fallback_bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


try:
    _ARC_UTILS = SourceFileLoader(
        "_arc_utils_new_task_033_3",
        str(Path("/home/lixinyu/snap/new/solvers/arc_utils.py")),
    ).load_module()
    bbox = _ARC_UTILS.bbox
except Exception:
    bbox = _fallback_bbox


BG = 0
OUT_COLOR = 2
DIVIDER_COLOR = 3
LEGEND_BORDER_COLOR = 5
INPUT_COLORS = (1, 4, 6, 7, 8, 9)
MAX_GENERATION_ATTEMPTS = 300


def _find_divider_row(grid: np.ndarray) -> int:
    rows = np.flatnonzero(np.all(grid == DIVIDER_COLOR, axis=1))
    assert len(rows) == 1, "expected exactly one full-width divider row"
    return int(rows[0])


def _parse_legend_and_masks(grid: np.ndarray) -> tuple[set[int], np.ndarray, np.ndarray]:
    divider_row = _find_divider_row(grid)
    mask_h = int(grid.shape[0] - divider_row - 1)
    assert mask_h >= 1, "bottom mask must be non-empty"

    top = grid[divider_row - mask_h : divider_row, :]
    bottom = grid[divider_row + 1 : divider_row + 1 + mask_h, :]
    assert top.shape == bottom.shape, "top and bottom masks must match"

    legend_zone_h = int(divider_row - mask_h)
    assert legend_zone_h in (2, 3), "legend zone must contain the legend and its lower border"
    legend_zone = grid[:legend_zone_h, :]

    gray_mask = legend_zone == LEGEND_BORDER_COLOR
    gray_box = bbox(gray_mask)
    assert gray_box is not None, "legend border must exist"
    gr0, gc0, gr1, gc1 = gray_box
    assert (gr0, gc0) == (0, 0), "legend border must touch the upper-left corner"
    legend_h = int(gr1)
    legend_w = int(gc1)
    assert (legend_h, legend_w) in ((1, 2), (2, 1)), "legend interior must be 1x2 or 2x1"

    expected_gray = np.zeros_like(legend_zone, dtype=bool)
    expected_gray[:legend_h, legend_w] = True
    expected_gray[legend_h, :legend_w] = True
    assert np.array_equal(gray_mask, expected_gray), "gray border must be the expected L shape"

    legend = legend_zone[:legend_h, :legend_w]
    active_colors = {int(v) for v in legend.ravel() if int(v) != BG}
    assert 1 <= len(active_colors) <= 2, "legend must define one or two active colors"

    return active_colors, top, bottom


def _active_intersection(top: np.ndarray, bottom: np.ndarray, active_colors: set[int]) -> np.ndarray:
    active = np.array(sorted(active_colors), dtype=int)
    return np.isin(top, active) & np.isin(bottom, active)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Use the legend-defined active colors to compute the boolean intersection of two aligned masks.

    Input structure:
    - A black background contains a small upper-left legend whose 1x2 or 2x1 interior lists one or two active colors.
    - The legend is outlined only by a gray L-shaped border on its right side and bottom side.
    - A full-width green divider row splits the remaining space into equal-size top and bottom masks filled with sparse colored pixels.

    Rule:
    - Read the non-black colors inside the legend interior as the active color set.
    - Treat every non-active pixel in the top and bottom masks as background.
    - Remove the divider row and align the two masks cell by cell.
    - Output red exactly at positions where both aligned masks contain an active-color pixel; output black everywhere else.

    Key insight:
    - The legend changes which colors count as foreground before the boolean operation, so the intersection depends on set membership rather than on the raw mask colors.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    active_colors, top, bottom = _parse_legend_and_masks(g)
    overlap = _active_intersection(top, bottom, active_colors)

    out = np.full(top.shape, BG, dtype=int)
    out[overlap] = OUT_COLOR
    return out


def _take_coord(pool: list[tuple[int, int]]) -> tuple[int, int]:
    assert pool, "ran out of free coordinates"
    return pool.pop()


def _choice(seq):
    return random.choice(tuple(seq))


def _paint_masks(
    mask_h: int,
    mask_w: int,
    active_colors: list[int],
    inactive_colors: list[int],
) -> tuple[np.ndarray, np.ndarray]:
    top = np.full((mask_h, mask_w), BG, dtype=int)
    bottom = np.full((mask_h, mask_w), BG, dtype=int)

    coords = [(r, c) for r in range(mask_h) for c in range(mask_w)]
    random.shuffle(coords)

    def place(top_color: int, bottom_color: int) -> None:
        r, c = _take_coord(coords)
        top[r, c] = int(top_color)
        bottom[r, c] = int(bottom_color)

    for color in active_colors:
        place(color, BG)
        place(BG, color)

    for color in inactive_colors:
        place(color, BG)
        place(BG, color)

    if len(active_colors) == 2:
        place(active_colors[0], active_colors[1])
        if random.random() < 0.7:
            place(active_colors[1], active_colors[0])
    else:
        place(active_colors[0], active_colors[0])
        if random.random() < 0.5:
            place(active_colors[0], active_colors[0])

    place(active_colors[0], inactive_colors[0])
    if random.random() < 0.5:
        place(inactive_colors[0], active_colors[-1])

    extra_cases = [
        lambda: (_choice(active_colors), BG),
        lambda: (BG, _choice(active_colors)),
        lambda: (_choice(inactive_colors), BG),
        lambda: (BG, _choice(inactive_colors)),
        lambda: (_choice(active_colors), _choice(active_colors)),
        lambda: (_choice(active_colors), _choice(inactive_colors)),
        lambda: (_choice(inactive_colors), _choice(active_colors)),
        lambda: (_choice(inactive_colors), _choice(inactive_colors)),
    ]
    extra_count = random.randint(0, min(5, len(coords)))
    for _ in range(extra_count):
        top_color, bottom_color = random.choice(extra_cases)()
        place(top_color, bottom_color)

    return top, bottom


def _build_legend(legend_h: int, legend_w: int, active_colors: list[int]) -> np.ndarray:
    legend = np.full((legend_h, legend_w), BG, dtype=int)
    cells = [(r, c) for r in range(legend_h) for c in range(legend_w)]
    random.shuffle(cells)
    for color, (r, c) in zip(active_colors, cells):
        legend[r, c] = int(color)
    return legend


def _expected_output(top: np.ndarray, bottom: np.ndarray, active_colors: list[int]) -> np.ndarray:
    out = np.full(top.shape, BG, dtype=int)
    overlap = np.isin(top, active_colors) & np.isin(bottom, active_colors)
    out[overlap] = OUT_COLOR
    return out


def _is_valid_candidate(
    grid: np.ndarray,
    top: np.ndarray,
    bottom: np.ndarray,
    active_colors: list[int],
    inactive_colors: list[int],
) -> bool:
    expected = _expected_output(top, bottom, active_colors)
    if not np.array_equal(main(grid), expected):
        return False

    top_colors = {int(v) for v in top.ravel() if int(v) != BG}
    bottom_colors = {int(v) for v in bottom.ravel() if int(v) != BG}
    active_set = set(map(int, active_colors))
    inactive_set = set(map(int, inactive_colors))

    if not (2 <= len(top_colors) <= 3 and 2 <= len(bottom_colors) <= 3):
        return False
    if not active_set.issubset(top_colors) or not active_set.issubset(bottom_colors):
        return False
    if not (top_colors & inactive_set) or not (bottom_colors & inactive_set):
        return False

    active_overlap = np.isin(top, active_colors) & np.isin(bottom, active_colors)
    if not bool(active_overlap.any()):
        return False

    false_overlap = (top != BG) & (bottom != BG) & ~active_overlap
    if not bool(false_overlap.any()):
        return False

    top_only_active = np.isin(top, active_colors) & (bottom == BG)
    bottom_only_active = (top == BG) & np.isin(bottom, active_colors)
    if not bool(top_only_active.any()) or not bool(bottom_only_active.any()):
        return False

    if len(active_colors) == 2:
        different_active_overlap = np.isin(top, active_colors) & np.isin(bottom, active_colors) & (top != bottom)
        if not bool(different_active_overlap.any()):
            return False

    return True


def _gen_attempt() -> np.ndarray | None:
    legend_h, legend_w = random.choice(((1, 2), (2, 1)))
    mask_h = random.randint(4, 8)
    mask_w = random.randint(6, 11)

    active_count = random.choice((1, 2))
    active_colors = random.sample(INPUT_COLORS, active_count)

    remaining = [color for color in INPUT_COLORS if color not in active_colors]
    inactive_count = 1 if active_count == 2 else random.choice((1, 2))
    inactive_colors = random.sample(remaining, inactive_count)

    top, bottom = _paint_masks(mask_h, mask_w, active_colors, inactive_colors)

    divider_row = legend_h + 1 + mask_h
    grid_h = divider_row + 1 + mask_h
    grid = np.full((grid_h, mask_w), BG, dtype=int)

    legend = _build_legend(legend_h, legend_w, active_colors)
    grid[:legend_h, :legend_w] = legend
    grid[:legend_h, legend_w] = LEGEND_BORDER_COLOR
    grid[legend_h, :legend_w] = LEGEND_BORDER_COLOR

    top_start = divider_row - mask_h
    grid[top_start:divider_row, :] = top
    grid[divider_row, :] = DIVIDER_COLOR
    grid[divider_row + 1 :, :] = bottom

    if not _is_valid_candidate(grid, top, bottom, active_colors, inactive_colors):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid selective-boolean input")
