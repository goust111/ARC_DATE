from __future__ import annotations

import random

import numpy as np


AZURE = 8
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5
MAROON = 9

HORIZONTAL_SIDES = ("left", "right")
VERTICAL_SIDES = ("up", "down")
MAX_GENERATION_ATTEMPTS = 300


def _dominant_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[int(np.argmax(counts))])


def _palette_info(grid: np.ndarray, bg: int) -> tuple[int, tuple[int, int], tuple[int, int]]:
    non_empty_rows = np.flatnonzero(np.any(grid != bg, axis=1))
    assert non_empty_rows.size > 0, "expected at least one non-background row"
    palette_row = int(non_empty_rows[-1])

    cols = np.flatnonzero(grid[palette_row] != bg)
    assert cols.size == 4, "palette row must contain four color markers"
    assert int(cols[1] - cols[0]) == 2, "palette first group must have one-cell spacing"
    assert int(cols[2] - cols[1]) == 3, "palette groups must be separated by two background cells"
    assert int(cols[3] - cols[2]) == 2, "palette second group must have one-cell spacing"

    colors = [int(grid[palette_row, c]) for c in cols]
    return palette_row, (colors[0], colors[1]), (colors[2], colors[3])


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                comp.append((cr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr = cr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _c in coords]
    cols = [c for _r, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _window_side(grid: np.ndarray, top: int, left: int, bbox: tuple[int, int, int, int]) -> str:
    r0, c0, r1, c1 = bbox
    sides: list[str] = []
    if left - 1 >= c0 and np.all(grid[top : top + 2, left - 1] == MAROON):
        sides.append("left")
    if left + 2 <= c1 and np.all(grid[top : top + 2, left + 2] == MAROON):
        sides.append("right")
    if top - 1 >= r0 and np.all(grid[top - 1, left : left + 2] == MAROON):
        sides.append("up")
    if top + 2 <= r1 and np.all(grid[top + 2, left : left + 2] == MAROON):
        sides.append("down")
    assert len(sides) == 1, "each window must have exactly one sill"
    return sides[0]


def _find_windows(
    grid: np.ndarray, bg: int, bbox: tuple[int, int, int, int]
) -> list[tuple[int, int, str]]:
    r0, c0, r1, c1 = bbox
    windows: list[tuple[int, int, str]] = []
    for top in range(r0 + 1, r1):
        for left in range(c0 + 1, c1):
            if top + 1 > r1 or left + 1 > c1:
                continue
            if not np.all(grid[top : top + 2, left : left + 2] == bg):
                continue
            side = _window_side(grid, top, left, bbox)
            windows.append((top, left, side))
    assert windows, "each building must contain at least one window"
    return windows


def main(grid: np.ndarray) -> np.ndarray:
    """
    Color each 2x2 window hole by sill orientation and floor level.

    Input structure:
    - Azure background.
    - One or more filled grey building rectangles.
    - Each building contains 2x2 background window holes.
    - Every window has one adjacent maroon 1x2 / 2x1 sill that indicates its facing direction.
    - The bottom palette row has two spaced color pairs: horizontal-class colors first, vertical-class colors second.

    Rule:
    - Left/right sills use the first palette pair, up/down sills use the second pair.
    - Within each building, upper-half window rows use the first color of the pair and lower-half rows use the second.
    - When the number of window rows is odd, the middle row counts as upper-half.
    """
    g = np.asarray(grid, dtype=int)
    bg = AZURE
    palette_row, horizontal_pair, vertical_pair = _palette_info(g, bg)

    building_mask = g[:palette_row] != bg
    comps = _components(building_mask)
    assert comps, "expected at least one building"

    out = g.copy()
    for comp in comps:
        bbox = _bbox(comp)
        windows = _find_windows(g, bg, bbox)
        row_levels = sorted({top for top, _left, _side in windows})
        row_to_index = {row: idx for idx, row in enumerate(row_levels)}
        upper_count = (len(row_levels) + 1) // 2

        for top, left, side in windows:
            is_upper = row_to_index[top] < upper_count
            if side in HORIZONTAL_SIDES:
                color = horizontal_pair[0] if is_upper else horizontal_pair[1]
            else:
                color = vertical_pair[0] if is_upper else vertical_pair[1]
            out[top : top + 2, left : left + 2] = color
    return out


def _random_side(kind: str) -> str:
    if kind == "horizontal":
        return random.choice(HORIZONTAL_SIDES)
    return random.choice(VERTICAL_SIDES)


def _assign_window_directions(nrows: int, ncols: int, *, anchor: bool) -> list[list[str]]:
    classes: list[list[str | None]] = [[None for _ in range(ncols)] for _ in range(nrows)]
    if anchor:
        upper_count = (nrows + 1) // 2
        upper_cells = [(r, c) for r in range(upper_count) for c in range(ncols)]
        lower_cells = [(r, c) for r in range(upper_count, nrows) for c in range(ncols)]
        for cells in (upper_cells, lower_cells):
            horizontal_cell, vertical_cell = random.sample(cells, 2)
            classes[horizontal_cell[0]][horizontal_cell[1]] = "horizontal"
            classes[vertical_cell[0]][vertical_cell[1]] = "vertical"

    for r in range(nrows):
        for c in range(ncols):
            if classes[r][c] is None:
                classes[r][c] = random.choice(("horizontal", "vertical"))

    return [[_random_side(classes[r][c]) for c in range(ncols)] for r in range(nrows)]


def _build_building(anchor: bool) -> dict:
    nrows = random.choice((2, 3))
    ncols = 2 if anchor else random.choice((1, 2))

    min_height = 4 * nrows
    height = random.randint(min_height, 15)
    extra_h = height - min_height
    top_extra = random.randint(0, extra_h)
    top_pad = 1 + top_extra
    bottom_pad = 1 + (extra_h - top_extra)

    base_width = 4 * ncols
    width = random.randint(max(5, base_width), 8)
    extra_w = width - base_width
    left_extra = random.randint(0, extra_w)
    left_pad = 1 + left_extra
    right_pad = 1 + (extra_w - left_extra)

    hole_rows = [top_pad + 4 * r for r in range(nrows)]
    hole_cols = [left_pad + 4 * c for c in range(ncols)]
    directions = _assign_window_directions(nrows, ncols, anchor=anchor)

    assert hole_rows[-1] + 2 <= height - bottom_pad
    assert hole_cols[-1] + 2 <= width - right_pad

    return {
        "nrows": nrows,
        "ncols": ncols,
        "height": height,
        "width": width,
        "hole_rows": hole_rows,
        "hole_cols": hole_cols,
        "directions": directions,
    }


def _paint_sill(grid: np.ndarray, top: int, left: int, side: str) -> None:
    if side == "left":
        grid[top : top + 2, left - 1] = MAROON
    elif side == "right":
        grid[top : top + 2, left + 2] = MAROON
    elif side == "up":
        grid[top - 1, left : left + 2] = MAROON
    else:
        grid[top + 2, left : left + 2] = MAROON


def _window_color(side: str, row_index: int, nrows: int) -> int:
    is_upper = row_index < (nrows + 1) // 2
    if side in HORIZONTAL_SIDES:
        return RED if is_upper else YELLOW
    return BLUE if is_upper else GREEN


def _render_pair(buildings: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    shuffled = buildings[:]
    random.shuffle(shuffled)

    x = random.randint(1, 3)
    for idx, building in enumerate(shuffled):
        building["left"] = x
        building["top"] = random.randint(0, 3)
        x += int(building["width"])
        if idx + 1 < len(shuffled):
            x += random.randint(2, 4)
    width = max(x + random.randint(1, 3), 10)

    max_bottom = max(int(building["top"]) + int(building["height"]) for building in shuffled)
    palette_gap = random.randint(1, 2)
    height = max_bottom + palette_gap + 1
    palette_row = height - 1

    inp = np.full((height, width), AZURE, dtype=int)
    expected = inp.copy()

    for building in shuffled:
        top = int(building["top"])
        left = int(building["left"])
        bottom = top + int(building["height"])
        right = left + int(building["width"])
        inp[top:bottom, left:right] = GREY
        expected[top:bottom, left:right] = GREY

        for row_index, rel_top in enumerate(building["hole_rows"]):
            for col_index, rel_left in enumerate(building["hole_cols"]):
                window_top = top + int(rel_top)
                window_left = left + int(rel_left)
                side = str(building["directions"][row_index][col_index])
                inp[window_top : window_top + 2, window_left : window_left + 2] = AZURE
                expected[window_top : window_top + 2, window_left : window_left + 2] = _window_color(
                    side, row_index, int(building["nrows"])
                )
                _paint_sill(inp, window_top, window_left, side)
                _paint_sill(expected, window_top, window_left, side)

    palette_start = random.randint(1, width - 9) if width >= 10 else 0
    palette_cols = [palette_start, palette_start + 2, palette_start + 5, palette_start + 7]
    palette_colors = [RED, YELLOW, BLUE, GREEN]
    for col, color in zip(palette_cols, palette_colors):
        inp[palette_row, col] = color
        expected[palette_row, col] = color

    return inp, expected


def _gen_attempt() -> np.ndarray | None:
    building_count = random.choice((1, 2, 3))
    buildings = [_build_building(anchor=True)]
    for _ in range(building_count - 1):
        buildings.append(_build_building(anchor=False))

    inp, expected = _render_pair(buildings)
    if np.array_equal(main(inp), expected):
        return inp
    return None


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid puzzle-6 input")
