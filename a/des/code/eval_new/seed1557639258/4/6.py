from __future__ import annotations

import math
import random
from collections import Counter

import numpy as np


BG = 0
COLOR_POOL = tuple(range(1, 10))
MAX_GRID = 30
MAX_RECT = 5
GEN_RETRIES = 300
PACK_RETRIES = 80


def _corner_run(grid: np.ndarray, row0: int, row1: int, from_left: bool) -> list[int]:
    cols: list[int] = []
    width = int(grid.shape[1])
    if from_left:
        c = 0
        while c < width and int(grid[row0, c]) != BG and int(grid[row1, c]) != BG:
            cols.append(int(c))
            c += 1
    else:
        c = width - 1
        while c >= 0 and int(grid[row0, c]) != BG and int(grid[row1, c]) != BG:
            cols.append(int(c))
            c -= 1
        cols.reverse()
    return cols


def _find_palette(grid: np.ndarray) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    g = np.asarray(grid, dtype=int)
    height, width = map(int, g.shape)
    assert height >= 2 and width >= 1

    candidates: list[tuple[int, int, list[tuple[int, int]], list[tuple[int, int]]]] = []
    scans = [
        (0, 1, True),
        (0, 1, False),
        (height - 2, height - 1, True),
        (height - 2, height - 1, False),
    ]

    for row0, row1, from_left in scans:
        cols = _corner_run(g, int(row0), int(row1), bool(from_left))
        if not cols:
            continue
        pairs = [(int(g[row0, c]), int(g[row1, c])) for c in cols]
        cells = [(int(row0), int(c)) for c in cols] + [(int(row1), int(c)) for c in cols]
        candidates.append((len(cols), 1 if from_left else 0, pairs, cells))

    assert candidates, "palette strip not found"
    candidates.sort(key=lambda item: (int(item[0]), int(item[1])), reverse=True)
    _ncols, _priority, pairs, cells = candidates[0]
    return pairs, cells


def _visible_color_areas(
    border_colors: list[int],
    center_color: int,
    height: int,
    width: int,
) -> dict[int, int]:
    areas: Counter[int] = Counter()
    cur_h = int(height)
    cur_w = int(width)
    for color in border_colors:
        inner_h = int(cur_h - 2)
        inner_w = int(cur_w - 2)
        ring = int(cur_h * cur_w - inner_h * inner_w)
        areas[int(color)] += ring
        cur_h = inner_h
        cur_w = inner_w
    areas[int(center_color)] += int(cur_h * cur_w)
    return {int(color): int(area) for color, area in areas.items() if int(area) > 0}


def _infer_dimensions(actual_areas: dict[int, int], border_colors: list[int], center_color: int) -> tuple[int, int]:
    total_area = int(sum(int(v) for v in actual_areas.values()))
    layers = int(len(border_colors))
    best: tuple[int, int, int] | None = None

    for height in range(1, int(math.isqrt(total_area)) + 1):
        if total_area % int(height) != 0:
            continue
        width = int(total_area // height)
        if height < 2 * layers + 1 or width < 2 * layers + 1:
            continue
        if _visible_color_areas(border_colors, int(center_color), int(height), int(width)) != actual_areas:
            continue
        cand = (int(width - height), int(height), int(width))
        if best is None or cand < best:
            best = cand

    assert best is not None, "no rectangle size matches the counted areas"
    return int(best[1]), int(best[2])


def _draw_nested(border_colors: list[int], center_color: int, height: int, width: int) -> np.ndarray:
    out = np.zeros((int(height), int(width)), dtype=int)
    layers = int(len(border_colors))

    for offset, color in enumerate(border_colors):
        r0 = int(offset)
        c0 = int(offset)
        r1 = int(height - 1 - offset)
        c1 = int(width - 1 - offset)
        out[r0, c0 : c1 + 1] = int(color)
        out[r1, c0 : c1 + 1] = int(color)
        out[r0 : r1 + 1, c0] = int(color)
        out[r0 : r1 + 1, c1] = int(color)

    out[layers : height - layers, layers : width - layers] = int(center_color)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Assemble scattered area-coded fragments into a complete set of concentric rectangles.

    Input structure:
    - bg = black.
    - Several solid monochrome rectangular fragments are scattered away from the border.
    - One corner contains a 2×N palette strip made of vertical color pairs, read left to right.
    - In each palette column the top pixel is a border color and the bottom pixel is the color immediately inside it.

    Rule:
    - Detect the corner palette and remove its pixels from the area counts.
    - Count the remaining pixels of every color; those counts are the visible areas of the final nested figure.
    - Use the palette's top colors as the border-ring order from outside to inside, and use the last bottom color as the innermost fill color.
    - Find the closest-to-square factor pair of the total visible area whose successive one-cell rings reproduce the counted color areas, then redraw that concentric rectangle.

    Key insight:
    - The palette gives the nesting order, while the fragment areas encode the absolute rectangle size, so combining order with color counts reconstructs the full geometry.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    palette_pairs, palette_cells = _find_palette(g)
    border_colors = [int(top) for top, _bottom in palette_pairs]
    center_color = int(palette_pairs[-1][1])

    color_areas: Counter[int] = Counter(int(v) for v in g.flatten() if int(v) != BG)
    for r, c in palette_cells:
        color = int(g[r, c])
        color_areas[color] -= 1
        assert color_areas[color] >= 0
        if color_areas[color] == 0:
            del color_areas[color]

    actual_areas = {int(color): int(area) for color, area in color_areas.items() if int(area) > 0}
    height, width = _infer_dimensions(actual_areas, border_colors, center_color)
    return _draw_nested(border_colors, center_color, height, width)


# --------------- generator ---------------

AREA_TO_DIMS: dict[int, list[tuple[int, int]]] = {}
for _h in range(1, MAX_RECT + 1):
    for _w in range(1, MAX_RECT + 1):
        AREA_TO_DIMS.setdefault(int(_h * _w), []).append((int(_h), int(_w)))
ALLOWED_AREAS = sorted(AREA_TO_DIMS)


def _nearest_square_pair(area: int) -> tuple[int, int]:
    best_h = 1
    for height in range(1, int(math.isqrt(area)) + 1):
        if area % int(height) == 0:
            best_h = int(height)
    return int(best_h), int(area // best_h)


def _sample_output_dims(layers: int) -> tuple[int, int]:
    min_side = int(2 * layers + 1)
    for _ in range(400):
        area = random.randint(int(min_side * min_side), 20 * 20)
        height, width = _nearest_square_pair(int(area))
        if height < min_side or width < min_side:
            continue
        if width > 20:
            continue
        return int(height), int(width)
    raise RuntimeError("failed to sample output dimensions")


def _split_area(total: int) -> list[tuple[int, int]]:
    remaining = int(total)
    pieces: list[tuple[int, int]] = []
    while remaining > 0:
        candidates = [area for area in ALLOWED_AREAS if int(area) <= remaining]
        if remaining in AREA_TO_DIMS and random.random() < 0.45:
            area = int(remaining)
        else:
            weights = [int(area * area * area) for area in candidates]
            area = int(random.choices(candidates, weights=weights, k=1)[0])
        pieces.append(tuple(random.choice(AREA_TO_DIMS[area])))
        remaining -= int(area)
    random.shuffle(pieces)
    return pieces


def _pack_rectangles(
    pieces: list[tuple[int, int, int]],
) -> tuple[int, int, list[tuple[int, int, int, int, int]]] | None:
    max_inner_h = 26
    widths = list(range(12, 29))
    random.shuffle(widths)

    order = pieces[:]
    random.shuffle(order)
    order.sort(key=lambda item: (-max(int(item[1]), int(item[2])), -(int(item[1]) * int(item[2]))))

    for inner_w in widths:
        x = 0
        y = 0
        row_h = 0
        used_w = 0
        placements: list[tuple[int, int, int, int, int]] = []
        ok = True
        for color, height, width in order:
            if x > 0 and x + int(width) > int(inner_w):
                y += int(row_h) + 1
                x = 0
                row_h = 0
            if y + int(height) > max_inner_h:
                ok = False
                break
            placements.append((int(color), int(y), int(x), int(height), int(width)))
            used_w = max(int(used_w), int(x + width))
            x += int(width) + 1
            row_h = max(int(row_h), int(height))
        if not ok:
            continue
        used_h = int(y + row_h)
        return int(used_h), int(used_w), placements
    return None


def _place_palette(grid: np.ndarray, pairs: list[tuple[int, int]], corner: str) -> None:
    height, width = map(int, grid.shape)
    ncols = int(len(pairs))
    if corner == "top_left":
        row0, row1 = 0, 1
        start = 0
    elif corner == "top_right":
        row0, row1 = 0, 1
        start = int(width - ncols)
    elif corner == "bottom_left":
        row0, row1 = int(height - 2), int(height - 1)
        start = 0
    elif corner == "bottom_right":
        row0, row1 = int(height - 2), int(height - 1)
        start = int(width - ncols)
    else:
        raise AssertionError(corner)

    for idx, (top, bottom) in enumerate(pairs):
        c = int(start + idx)
        grid[row0, c] = int(top)
        grid[row1, c] = int(bottom)


def _fragment_pieces(color_areas: dict[int, int]) -> list[tuple[int, int, int]]:
    pieces: list[tuple[int, int, int]] = []
    for color, area in color_areas.items():
        for height, width in _split_area(int(area)):
            pieces.append((int(color), int(height), int(width)))
    return pieces


def _paint_fragments(
    grid: np.ndarray,
    placements: list[tuple[int, int, int, int, int]],
    palette_on_top: bool,
) -> None:
    used_h = max(int(y + height) for _color, y, _x, height, _width in placements)
    used_w = max(int(x + width) for _color, _y, x, _height, width in placements)
    full_h, full_w = map(int, grid.shape)

    extra_y = int(full_h - 4 - used_h)
    extra_x = int(full_w - 2 - used_w)
    assert extra_y >= 0 and extra_x >= 0

    x_shift = random.randint(0, int(extra_x))
    y_shift = random.randint(0, int(extra_y))
    base_row = int(3 + y_shift) if palette_on_top else int(1 + y_shift)
    base_col = int(1 + x_shift)

    for color, y, x, height, width in placements:
        r0 = int(base_row + y)
        c0 = int(base_col + x)
        grid[r0 : r0 + height, c0 : c0 + width] = int(color)


def _gen_attempt() -> np.ndarray | None:
    layers = random.randint(2, 5)
    visible_colors = random.sample(COLOR_POOL, layers + 1)
    border_colors = [int(color) for color in visible_colors[:-1]]
    center_color = int(visible_colors[-1])
    palette_pairs = [(int(visible_colors[i]), int(visible_colors[i + 1])) for i in range(layers)]

    out_h, out_w = _sample_output_dims(layers)
    expected = _draw_nested(border_colors, center_color, out_h, out_w)
    color_areas = _visible_color_areas(border_colors, center_color, out_h, out_w)
    pieces = _fragment_pieces(color_areas)

    for _ in range(PACK_RETRIES):
        packed = _pack_rectangles(pieces)
        if packed is None:
            return None
        used_h, used_w, placements = packed

        full_h_min = int(used_h + 4)
        full_w_min = int(max(used_w + 2, len(palette_pairs) + 1))
        if full_h_min > MAX_GRID or full_w_min > MAX_GRID:
            continue

        full_h = random.randint(int(full_h_min), MAX_GRID)
        full_w = random.randint(int(full_w_min), MAX_GRID)
        corner = random.choice(("top_left", "top_right", "bottom_left", "bottom_right"))

        grid = np.zeros((int(full_h), int(full_w)), dtype=int)
        _place_palette(grid, palette_pairs, corner)
        _paint_fragments(grid, placements, palette_on_top=corner.startswith("top"))

        if np.array_equal(main(grid), expected):
            return grid

    return None


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
