from __future__ import annotations

import random
from typing import Iterable

import numpy as np


BACKGROUND = 0
GREEN = 3
YELLOW = 4
NON_BLACK = tuple(range(1, 10))
GEN_ATTEMPTS = 400


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _colors_present(grid: np.ndarray, exclude: Iterable[int] = ()) -> list[int]:
    excluded = set(int(x) for x in exclude)
    return sorted(int(v) for v in np.unique(grid) if int(v) not in excluded)


def _outer_and_inner_colors(grid: np.ndarray, bg: int) -> tuple[int, int, tuple[int, int, int, int]]:
    colors = _colors_present(grid, exclude={bg})
    assert len(colors) == 2, "expected exactly two foreground colors"

    for color in colors:
        box = _bbox(grid == color)
        if box is None:
            continue
        r0, c0, r1, c1 = box
        if r1 - r0 < 2 or c1 - c0 < 2:
            continue
        edge_mask = np.zeros_like(grid, dtype=bool)
        edge_mask[r0, c0 : c1 + 1] = True
        edge_mask[r1, c0 : c1 + 1] = True
        edge_mask[r0 : r1 + 1, c0] = True
        edge_mask[r0 : r1 + 1, c1] = True
        if not np.all(grid[edge_mask] == color):
            continue
        inner = grid[r0 + 1 : r1, c0 + 1 : c1]
        if inner.size == 0:
            continue
        other = [v for v in colors if v != color]
        assert len(other) == 1
        if not np.all(inner == other[0]):
            continue
        return int(color), int(other[0]), box

    raise AssertionError("failed to detect the bordered rectangle")


def _marker_axes(
    grid: np.ndarray, inner_color: int, box: tuple[int, int, int, int]
) -> tuple[list[int], list[int], list[tuple[int, int]]]:
    r0, c0, r1, c1 = box
    rows: set[int] = set()
    cols: set[int] = set()
    markers: list[tuple[int, int]] = []
    h, w = grid.shape

    if r0 - 1 >= 0:
        for c in range(c0 + 1, c1):
            if int(grid[r0 - 1, c]) == inner_color:
                cols.add(c)
                markers.append((r0 - 1, c))
    if r1 + 1 < h:
        for c in range(c0 + 1, c1):
            if int(grid[r1 + 1, c]) == inner_color:
                cols.add(c)
                markers.append((r1 + 1, c))
    if c0 - 1 >= 0:
        for r in range(r0 + 1, r1):
            if int(grid[r, c0 - 1]) == inner_color:
                rows.add(r)
                markers.append((r, c0 - 1))
    if c1 + 1 < w:
        for r in range(r0 + 1, r1):
            if int(grid[r, c1 + 1]) == inner_color:
                rows.add(r)
                markers.append((r, c1 + 1))

    assert len(rows) >= 2 and len(cols) >= 2, "expected at least a 3x3 partition"
    return sorted(rows), sorted(cols), markers


def _segments(start: int, end: int, cuts: list[int]) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    cur = start
    for cut in cuts:
        spans.append((cur, cut - 1))
        cur = cut + 1
    spans.append((cur, end))
    return spans


def _largest_true_rectangle(mask: np.ndarray) -> tuple[int, int, int, int]:
    h, w = mask.shape
    best_rect: tuple[int, int, int, int] | None = None
    best_area = 0
    best_count = 0

    for top in range(h):
        col_ok = np.ones(w, dtype=bool)
        for bottom in range(top, h):
            col_ok &= mask[bottom]
            height = bottom - top + 1
            if height < 2:
                continue

            run_start = None
            for c in range(w + 1):
                active = c < w and bool(col_ok[c])
                if active and run_start is None:
                    run_start = c
                if active:
                    continue
                if run_start is None:
                    continue
                width = c - run_start
                if width >= 2:
                    rect = (top, run_start, bottom, c - 1)
                    area = height * width
                    if area > best_area:
                        best_area = area
                        best_rect = rect
                        best_count = 1
                    elif area == best_area:
                        assert best_rect is not None
                        if rect != best_rect:
                            best_count += 1
                run_start = None

    assert best_rect is not None and best_area > 0, "expected a valid inner-color rectangle"
    assert best_count == 1, "largest inner-color rectangle must be unique"
    return best_rect


def _fill_color(outer_color: int, inner_color: int) -> int:
    return YELLOW if GREEN in (outer_color, inner_color) else GREEN


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor parity-selected subregions inside a two-color rectangle, then fill the largest remaining inner-color rectangle.

    Input structure:
    - A black background contains one axis-aligned rectangle with a one-cell outer border and a solid inner fill.
    - The rectangle uses exactly two foreground colors: outer color A on the border and inner color B everywhere inside.
    - Single-cell B leaf markers sit just outside the top/bottom edges to name columns and just outside the left/right edges to name rows.

    Rule:
    - Detect the rectangle from the complete A-colored border and read the marked interior rows/columns from the outside B leaf markers.
    - Inside the border, those marked rows and columns partition the B interior into subregions; recolor every marked line and every subregion whose row-index parity matches its column-index parity to A.
    - Among the remaining all-B rectangles of height and width at least 2, find the unique maximum-area one and fill it with green, or with yellow if A or B is already green; erase the outside leaf markers to black.

    Key insight:
    - The leaves do not extend rays outward here; they define a row/column grid whose parity pattern decides which interior cells flip, and that parity flip determines which B rectangle becomes the unique largest target.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg = BACKGROUND
    outer_color, inner_color, box = _outer_and_inner_colors(g, bg)
    r0, c0, r1, c1 = box
    marked_rows, marked_cols, markers = _marker_axes(g, inner_color, box)

    out = g.copy()
    for r, c in markers:
        out[r, c] = bg

    out[r0 + 1 : r1, c0 + 1 : c1] = inner_color
    for r in marked_rows:
        out[r, c0 + 1 : c1] = outer_color
    for c in marked_cols:
        out[r0 + 1 : r1, c] = outer_color

    row_segments = _segments(r0 + 1, r1 - 1, marked_rows)
    col_segments = _segments(c0 + 1, c1 - 1, marked_cols)
    for i, (rs, re) in enumerate(row_segments):
        if rs > re:
            continue
        for j, (cs, ce) in enumerate(col_segments):
            if cs > ce:
                continue
            if (i % 2) == (j % 2):
                out[rs : re + 1, cs : ce + 1] = outer_color

    inner_mask = out[r0 + 1 : r1, c0 + 1 : c1] == inner_color
    top, left, bottom, right = _largest_true_rectangle(inner_mask)
    fill = _fill_color(outer_color, inner_color)
    out[r0 + 1 + top : r0 + 2 + bottom, c0 + 1 + left : c0 + 2 + right] = fill
    return out


def _choose_colors() -> tuple[int, int]:
    if random.random() < 0.4:
        shared = random.choice([0, 1])
        other = random.choice([c for c in NON_BLACK if c not in (GREEN, YELLOW)])
        return (GREEN, other) if shared == 0 else (other, GREEN)
    outer, inner = random.sample([c for c in NON_BLACK if c != GREEN], 2)
    return int(outer), int(inner)


def _random_sizes(count: int, *, allow_thin: bool, max_size: int) -> list[int]:
    sizes = []
    thin_budget = 1 if allow_thin and random.random() < 0.45 else 0
    for _ in range(count):
        if thin_budget and random.random() < 0.5:
            sizes.append(1)
            thin_budget -= 1
        else:
            sizes.append(random.randint(2, max_size))
    return sizes


def _marker_positions(start: int, sizes: list[int]) -> list[int]:
    pos: list[int] = []
    cur = start
    for size in sizes[:-1]:
        cur += size
        pos.append(cur)
        cur += 1
    return pos


def _segment_analysis(row_sizes: list[int], col_sizes: list[int]) -> dict[str, object] | None:
    eligible_cross = []
    eligible_noncross = []
    all_cells = []

    for i, height in enumerate(row_sizes):
        for j, width in enumerate(col_sizes):
            area = height * width
            cell = {
                "i": i,
                "j": j,
                "height": height,
                "width": width,
                "area": area,
                "cross": (i % 2) == (j % 2),
            }
            if height >= 2 and width >= 2:
                all_cells.append(cell)
                if cell["cross"]:
                    eligible_cross.append(cell)
                else:
                    eligible_noncross.append(cell)

    if len(eligible_cross) < 2 or len(eligible_noncross) < 2:
        return None

    best_cross_area = max(cell["area"] for cell in eligible_cross)
    best_cross = [cell for cell in eligible_cross if cell["area"] == best_cross_area]
    best_noncross_area = max(cell["area"] for cell in eligible_noncross)
    best_noncross = [cell for cell in eligible_noncross if cell["area"] == best_noncross_area]
    best_all_area = max(cell["area"] for cell in all_cells)
    best_all = [cell for cell in all_cells if cell["area"] == best_all_area]

    if len(best_cross) != 1 or len(best_noncross) != 1 or len(best_all) != 1:
        return None
    if not best_all[0]["cross"]:
        return None
    if best_cross_area <= best_noncross_area:
        return None

    target = best_noncross[0]
    if target["i"] in (0, len(row_sizes) - 1) and target["j"] in (0, len(col_sizes) - 1):
        return None

    return {
        "target_index": (int(target["i"]), int(target["j"])),
        "target_area": int(target["area"]),
        "largest_cross_area": int(best_cross_area),
    }


def _build_input_from_layout(
    row_sizes: list[int],
    col_sizes: list[int],
    outer_color: int,
    inner_color: int,
) -> np.ndarray:
    row_markers = len(row_sizes) - 1
    col_markers = len(col_sizes) - 1
    inner_h = sum(row_sizes) + row_markers
    inner_w = sum(col_sizes) + col_markers
    rect_h = inner_h + 2
    rect_w = inner_w + 2

    top = random.randint(1, 3)
    bottom = random.randint(1, 3)
    left = random.randint(1, 3)
    right = random.randint(1, 3)
    grid = np.zeros((top + rect_h + bottom, left + rect_w + right), dtype=int)

    r0 = top
    c0 = left
    r1 = r0 + rect_h - 1
    c1 = c0 + rect_w - 1

    grid[r0 : r1 + 1, c0 : c1 + 1] = outer_color
    grid[r0 + 1 : r1, c0 + 1 : c1] = inner_color

    row_positions = _marker_positions(r0 + 1, row_sizes)
    col_positions = _marker_positions(c0 + 1, col_sizes)

    for r in row_positions:
        side = random.choice(["left", "right"])
        grid[r, c0 - 1 if side == "left" else c1 + 1] = inner_color
    for c in col_positions:
        side = random.choice(["top", "bottom"])
        grid[r0 - 1 if side == "top" else r1 + 1, c] = inner_color

    return grid


def _validate_input(grid: np.ndarray) -> bool:
    try:
        outer_color, inner_color, box = _outer_and_inner_colors(grid, BACKGROUND)
        marked_rows, marked_cols, markers = _marker_axes(grid, inner_color, box)
    except AssertionError:
        return False

    r0, c0, r1, c1 = box
    if any(grid[r, c] != inner_color for r, c in markers):
        return False
    if np.any(grid[:r0, c0 : c1 + 1] == outer_color) or np.any(grid[r1 + 1 :, c0 : c1 + 1] == outer_color):
        return False
    if np.any(grid[r0 : r1 + 1, :c0] == outer_color) or np.any(grid[r0 : r1 + 1, c1 + 1 :] == outer_color):
        return False

    row_segments = _segments(r0 + 1, r1 - 1, marked_rows)
    col_segments = _segments(c0 + 1, c1 - 1, marked_cols)
    row_sizes = [re - rs + 1 for rs, re in row_segments]
    col_sizes = [ce - cs + 1 for cs, ce in col_segments]
    analysis = _segment_analysis(row_sizes, col_sizes)
    if analysis is None:
        return False

    outside_inner = {
        (int(r), int(c))
        for r, c in np.argwhere(grid == inner_color)
        if not (r0 + 1 <= int(r) <= r1 - 1 and c0 + 1 <= int(c) <= c1 - 1)
    }
    if outside_inner != set(markers):
        return False

    out = main(grid)
    fill = _fill_color(outer_color, inner_color)
    fill_mask = out == fill
    fill_box = _bbox(fill_mask)
    if fill_box is None:
        return False
    fr0, fc0, fr1, fc1 = fill_box
    if not np.all(out[fr0 : fr1 + 1, fc0 : fc1 + 1] == fill):
        return False

    target_i, target_j = analysis["target_index"]
    ers, ere = row_segments[target_i]
    ecs, ece = col_segments[target_j]
    expected_fill_box = (ers, ecs, ere, ece)
    if fill_box != expected_fill_box:
        return False
    if int(fill_mask.sum()) != (ere - ers + 1) * (ece - ecs + 1):
        return False
    if any(int(out[r, c]) != BACKGROUND for r, c in markers):
        return False

    return not np.any(out[:r0, :] == fill) and not np.any(out[r1 + 1 :, :] == fill)


def _attempt_generate() -> np.ndarray | None:
    row_segments = random.choice([3, 4])
    col_segments = random.choice([3, 4])
    row_sizes = _random_sizes(row_segments, allow_thin=True, max_size=4)
    col_sizes = _random_sizes(col_segments, allow_thin=True, max_size=5)

    analysis = _segment_analysis(row_sizes, col_sizes)
    if analysis is None:
        return None

    row_markers = row_segments - 1
    col_markers = col_segments - 1
    inner_h = sum(row_sizes) + row_markers
    inner_w = sum(col_sizes) + col_markers
    rect_h = inner_h + 2
    rect_w = inner_w + 2
    if rect_h > 22 or rect_w > 24:
        return None

    outer_color, inner_color = _choose_colors()
    grid = _build_input_from_layout(row_sizes, col_sizes, outer_color, inner_color)
    if max(grid.shape) > 30:
        return None
    if not _validate_input(grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _attempt_generate()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid puzzle instance")
