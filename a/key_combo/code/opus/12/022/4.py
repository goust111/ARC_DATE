from __future__ import annotations

import random
from typing import Any

import numpy as np


RED = 2
GREY = 5
PIXEL_COLORS = (1, 3, 4, 7)

GRID_HEIGHT_RANGE = (13, 18)
GRID_WIDTH_RANGE = (12, 18)
PIXEL_COUNT_RANGE = (4, 6)
RECT_HEIGHTS = (2, 3)
RECT_BAR_GAP_RANGE = (3, 5)
GENERATION_ATTEMPTS = 400


def _full_red_rows(grid: np.ndarray) -> list[int]:
    return [r for r in range(grid.shape[0]) if np.all(grid[r, :] == RED)]


def _grey_bbox(grid: np.ndarray) -> tuple[int, int, int, int]:
    coords = np.argwhere(grid == GREY)
    assert coords.size > 0, "expected one grey rectangle"
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    patch = grid[r0 : r1 + 1, c0 : c1 + 1]
    assert patch.shape[1] == 2, "grey rectangle must be two columns wide"
    assert patch.shape[0] in RECT_HEIGHTS, "grey rectangle height must be 2 or 3"
    assert np.all(patch == GREY), "grey rectangle must be solid"
    return int(r0), int(c0), int(r1), int(c1)


def _loose_pixels(grid: np.ndarray) -> list[tuple[int, int, int]]:
    mask = (grid != 0) & (grid != RED) & (grid != GREY)
    coords = np.argwhere(mask)
    return [(int(r), int(c), int(grid[r, c])) for r, c in coords]


def _settle_on_bar(
    pixels: list[tuple[int, int, int]],
    bar_row: int,
) -> dict[int, list[dict[str, int]]]:
    by_col: dict[int, list[tuple[int, int]]] = {}
    for r, c, color in pixels:
        by_col.setdefault(c, []).append((r, color))

    landed: dict[int, list[dict[str, int]]] = {}
    for c, entries in by_col.items():
        entries.sort(key=lambda item: item[0])
        count = len(entries)
        top_row = bar_row - count
        assert top_row >= 0, "gravity stack must fit above the bar"
        landed[c] = [
            {"row": top_row + idx, "color": color}
            for idx, (_r, color) in enumerate(entries)
        ]
    return landed


def _first_obstacle_row(
    height: int,
    rect_top: int | None,
    rect_bottom: int | None,
    rect_c0: int,
    rect_c1: int,
    col: int,
) -> int:
    if rect_top is not None and rect_bottom is not None and rect_c0 <= col <= rect_c1:
        return rect_top
    return height


def _simulate(grid: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    h, w = g.shape

    red_rows = _full_red_rows(g)
    assert len(red_rows) == 1, "expected exactly one full-width red bar"
    bar_row = red_rows[0]

    rect_r0, rect_c0, rect_r1, rect_c1 = _grey_bbox(g)
    assert rect_r1 < bar_row, "grey rectangle must start above the red bar"
    rect_h = rect_r1 - rect_r0 + 1

    loose = _loose_pixels(g)
    assert len(loose) >= 3, "expected several loose pixels"
    assert all(r < bar_row for r, _c, _color in loose), "loose pixels must start above the bar"

    landed = _settle_on_bar(loose, bar_row)

    out = np.zeros_like(g)

    for c in range(w):
        if c < rect_c0:
            nc = c - 1
            if 0 <= nc < w:
                out[bar_row, nc] = RED
        elif c > rect_c1:
            nc = c + 1
            if 0 <= nc < w:
                out[bar_row, nc] = RED

    rect_top = h - rect_h
    rect_bottom = h - 1
    out[rect_top : rect_bottom + 1, rect_c0 : rect_c1 + 1] = GREY

    stable_positions: list[dict[str, int]] = []
    falling_positions: list[dict[str, int]] = []

    for c, stack in landed.items():
        supported = out[bar_row, c] == RED
        if supported:
            for item in stack:
                stable_positions.append({"row": item["row"], "col": c, "color": item["color"]})
            continue

        obstacle = _first_obstacle_row(h, rect_top, rect_bottom, rect_c0, rect_c1, c)
        count = len(stack)
        top_row = obstacle - count
        assert top_row >= 0, "falling stack must fit above its obstacle"
        for idx, item in enumerate(stack):
            falling_positions.append(
                {
                    "row": top_row + idx,
                    "col": c,
                    "color": item["color"],
                }
            )

    for item in stable_positions:
        out[item["row"], item["col"]] = item["color"]

    if falling_positions:
        ordered = sorted(falling_positions, key=lambda item: (item["col"], item["row"]))
        colors = [item["color"] for item in ordered]
        rotated = colors[1:] + colors[:1]
        for item, new_color in zip(ordered, rotated):
            out[item["row"], item["col"]] = new_color
    else:
        ordered = []

    info = {
        "bar_row": bar_row,
        "rect_cols": (rect_c0, rect_c1),
        "rect_height": rect_h,
        "rect_top": rect_top,
        "landed": landed,
        "stable_positions": stable_positions,
        "falling_positions": ordered,
    }
    return out, info


def main(grid: np.ndarray) -> np.ndarray:
    """
    Let loose pixels drop onto a red bar, then reopen the bar with a falling grey block so only the trapped pixels drop again and cycle colors.

    Input structure:
    - The grid is black except for one full-width red horizontal bar, one solid grey rectangle above it,
      and several isolated colored pixels above the bar.
    - The grey rectangle is exactly two columns wide and two or three rows tall.
    - Loose-pixel colors come from a small visible palette and act as individual movable particles.

    Rule:
    - First drop every loose pixel straight downward until it rests on the red bar or on another landed pixel
      in the same column.
    - Then move the grey rectangle straight downward until it is flush with the bottom edge.
      While it passes through the red bar, remove the bar cells in the rectangle's columns,
      shift the left surviving bar segment one cell left, and shift the right surviving segment one cell right.
    - Any landed pixel that no longer has red support beneath it falls again until it reaches the bottom
      or the top of the moved grey rectangle.
    - Recolor only these twice-falling pixels by sorting them from left to right and cyclically giving each one
      the next pixel's original color, with the rightmost pixel taking the leftmost color.
    - Pixels that remain supported keep both their positions and their colors.

    Key insight:
    - The grey rectangle creates a later trapdoor: the relevant pixels are the ones that first land normally,
      then lose their support when the bar splits and therefore participate in the final color cycle.
    """
    out, _info = _simulate(grid)
    return out


def _choose_bar_row(height: int, rect_h: int) -> int:
    low = max(height // 2, 5)
    high = height - rect_h - 2
    assert low <= high
    return random.randint(low, high)


def _outside_columns(
    width: int,
    rect_c0: int,
    rect_c1: int,
    outside_count: int,
) -> list[int] | None:
    left = list(range(0, rect_c0 - 1))
    right = list(range(rect_c1 + 2, width))
    if len(left) + len(right) < outside_count:
        return None

    side_modes: list[tuple[int, int]] = []
    for left_count in range(outside_count + 1):
        right_count = outside_count - left_count
        if left_count <= len(left) and right_count <= len(right):
            side_modes.append((left_count, right_count))

    assert side_modes
    if len(left) > 0 and len(right) > 0:
        balanced = [mode for mode in side_modes if mode[0] > 0 and mode[1] > 0]
        if balanced and random.random() < 0.65:
            side_modes = balanced

    left_count, right_count = random.choice(side_modes)
    return random.sample(left, left_count) + random.sample(right, right_count)


def _path_rows(rect_r1: int, bar_row: int) -> list[int]:
    return list(range(rect_r1 + 1, bar_row - 1))


def _pick_row(
    low: int,
    high: int,
    col: int,
    occupied: list[tuple[int, int]],
) -> int | None:
    candidates = []
    for row in range(low, high + 1):
        if all(not (orow == row and abs(ocol - col) == 1) for orow, ocol in occupied):
            candidates.append(row)
    if not candidates:
        return None
    return random.choice(candidates)


def _semantic_checks(grid: np.ndarray, info: dict[str, Any], out: np.ndarray) -> bool:
    rect_c0, rect_c1 = info["rect_cols"]
    landed = info["landed"]
    falling = info["falling_positions"]
    stable = info["stable_positions"]

    if set(landed) != set(c for c in landed):
        return False
    if not ({rect_c0, rect_c1} <= set(landed)):
        return False
    if len(falling) != sum(len(landed[c]) for c in (rect_c0, rect_c1)):
        return False
    if {item["col"] for item in falling} != {rect_c0, rect_c1}:
        return False
    if not stable:
        return False
    if any(item["col"] in {rect_c0, rect_c1} for item in stable):
        return False
    if any(out[info["bar_row"], c] != RED for c in {item["col"] for item in stable}):
        return False
    if len(falling) >= 2:
        original = [item["color"] for item in falling]
        recolored = [int(out[item["row"], item["col"]]) for item in falling]
        if original == recolored:
            return False
    return True


def _attempt() -> np.ndarray | None:
    height = random.randint(*GRID_HEIGHT_RANGE)
    width = random.randint(*GRID_WIDTH_RANGE)
    rect_h = random.choice(RECT_HEIGHTS)
    bar_row = _choose_bar_row(height, rect_h)

    gap = random.randint(*RECT_BAR_GAP_RANGE)
    rect_r1 = bar_row - gap
    rect_r0 = rect_r1 - rect_h + 1
    if rect_r0 < 0:
        return None

    rect_c0 = random.randint(2, width - 4)
    rect_c1 = rect_c0 + 1

    pixel_count = random.randint(*PIXEL_COUNT_RANGE)
    outside_count = pixel_count - 2
    outside_cols = _outside_columns(width, rect_c0, rect_c1, outside_count)
    if outside_cols is None:
        return None

    path_rows = _path_rows(rect_r1, bar_row)
    if len(path_rows) < 1:
        return None

    grid = np.zeros((height, width), dtype=int)
    grid[bar_row, :] = RED
    grid[rect_r0 : rect_r1 + 1, rect_c0 : rect_c1 + 1] = GREY

    path_colors = random.sample(PIXEL_COLORS, 2)
    occupied_pixels: list[tuple[int, int]] = []
    for col, color in zip((rect_c0, rect_c1), path_colors):
        row = _pick_row(path_rows[0], path_rows[-1], col, occupied_pixels)
        if row is None:
            return None
        grid[row, col] = color
        occupied_pixels.append((row, col))

    for col in outside_cols:
        row = _pick_row(0, bar_row - 2, col, occupied_pixels)
        if row is None:
            return None
        grid[row, col] = random.choice(PIXEL_COLORS)
        occupied_pixels.append((row, col))

    try:
        out, info = _simulate(grid)
    except AssertionError:
        return None

    if grid.shape != out.shape:
        return None
    if not _semantic_checks(grid, info, out):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_ATTEMPTS):
        grid = _attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
