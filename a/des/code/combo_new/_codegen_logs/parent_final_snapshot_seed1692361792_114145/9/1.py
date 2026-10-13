from __future__ import annotations

import random

import numpy as np


BG = 0
FRAME = 5
COLOR_POOL = [1, 2, 3, 4, 6, 7, 8, 9]
GEN_RETRIES = 400


def _neighbors4(r: int, c: int) -> list[tuple[int, int]]:
    return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
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
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _border_cells(r0: int, c0: int, r1: int, c1: int) -> list[tuple[int, int]]:
    cells: list[tuple[int, int]] = []
    for c in range(c0, c1 + 1):
        cells.append((r0, c))
        cells.append((r1, c))
    for r in range(r0 + 1, r1):
        cells.append((r, c0))
        cells.append((r, c1))
    return cells


def _rect_info_from_component(comp: list[tuple[int, int]]) -> dict[str, object] | None:
    rows = [r for r, _ in comp]
    cols = [c for _, c in comp]
    r0, r1 = min(rows), max(rows)
    c0, c1 = min(cols), max(cols)
    if r1 - r0 < 2 or c1 - c0 < 2:
        return None
    border = set(_border_cells(r0, c0, r1, c1))
    if set(comp) != border:
        return None
    return {
        "bbox": (r0, c0, r1, c1),
        "border": sorted(border),
        "perimeter": len(border),
    }


def _parse_palette(row: np.ndarray) -> dict[int, int]:
    palette: dict[int, int] = {}
    c = 0
    while c < row.shape[0]:
        color = int(row[c])
        if color == BG:
            c += 1
            continue
        start = c
        while c < row.shape[0] and int(row[c]) == color:
            c += 1
        palette[color] = c - start
    return palette


def _extract_rectangles(grid: np.ndarray) -> tuple[int, list[dict[str, object]]]:
    upper = grid[:-1]
    candidates: list[tuple[int, int, int, list[dict[str, object]]]] = []
    for color in sorted(int(v) for v in np.unique(upper) if int(v) != BG):
        comps = _components(upper == color)
        rects: list[dict[str, object]] = []
        total = 0
        ok = True
        for comp in comps:
            info = _rect_info_from_component(comp)
            if info is None:
                ok = False
                break
            rects.append(info)
            total += int(info["perimeter"])
        if ok and rects:
            candidates.append((total, len(rects), color, rects))
    if not candidates:
        raise ValueError("failed to find rectangular frame components")

    _, _, frame_color, rects = max(candidates)
    rects = sorted(rects, key=lambda info: tuple(info["bbox"]))  # type: ignore[arg-type]
    for info in rects:
        r0, c0, r1, c1 = info["bbox"]  # type: ignore[misc]
        interior = grid[r0 + 1 : r1, c0 + 1 : c1]
        marker_cells = np.argwhere((interior != BG) & (interior != frame_color))
        if len(marker_cells) != 1:
            raise ValueError("each rectangle must contain exactly one marker")
        mr, mc = map(int, marker_cells[0])
        info["marker"] = (r0 + 1 + mr, c0 + 1 + mc)
    return frame_color, rects


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor rectangle borders by matching interior-marker colors to palette lengths and then to frame perimeters.

    Input structure:
    - Black background with a palette on the bottom row: contiguous non-black color segments separated by black cells.
    - Above the palette are one or more non-touching hollow rectangular frame components, all drawn in one frame color.
    - Each frame contains exactly one interior marker pixel whose color appears in the palette row.

    Rule:
    - Read each marker color C and look up the length L of the palette segment with color C.
    - Among the rectangular frames, find the one whose border pixel count equals L.
    - Recolor that target frame's entire border to C.
    - Erase all interior markers back to black and leave the palette row unchanged.

    Key insight:
    - The marker does not color its own frame directly; it first selects a palette segment by color, then uses that segment length to choose the destination frame by perimeter.
    """
    g = np.asarray(grid, dtype=int)
    palette = _parse_palette(g[-1])
    frame_color, rects = _extract_rectangles(g)
    perim_to_rect = {int(info["perimeter"]): info for info in rects}

    out = g.copy()
    for info in rects:
        mr, mc = info["marker"]  # type: ignore[misc]
        marker_color = int(g[mr, mc])
        out[mr, mc] = BG
        target_perim = palette.get(marker_color)
        if target_perim is None or target_perim not in perim_to_rect:
            continue
        for r, c in perim_to_rect[target_perim]["border"]:  # type: ignore[index]
            out[r, c] = marker_color

    return out


PERIMETER_TO_DIMS: dict[int, list[tuple[int, int]]] = {}
for _h in range(3, 9):
    for _w in range(3, 9):
        _p = 2 * (_h + _w) - 4
        if _p <= 18:
            PERIMETER_TO_DIMS.setdefault(_p, []).append((_h, _w))

PERIMETER_PAIRS = [
    (p0, p1)
    for p0 in sorted(PERIMETER_TO_DIMS)
    for p1 in sorted(PERIMETER_TO_DIMS)
    if p0 < p1 and p0 + p1 <= 28
]


def _draw_border(grid: np.ndarray, top: int, left: int, height: int, width: int, color: int) -> None:
    grid[top, left : left + width] = color
    grid[top + height - 1, left : left + width] = color
    grid[top : top + height, left] = color
    grid[top : top + height, left + width - 1] = color


def _paint_palette(grid: np.ndarray, target_colors: list[int], perimeters: list[int]) -> None:
    order = [0, 1]
    random.shuffle(order)
    used = perimeters[order[0]] + perimeters[order[1]] + 1
    extra = grid.shape[1] - used
    left_gap = random.randint(0, extra)
    middle_extra = random.randint(0, extra - left_gap)
    cursor = left_gap
    for pos, idx in enumerate(order):
        length = perimeters[idx]
        grid[-1, cursor : cursor + length] = target_colors[idx]
        cursor += length
        if pos == 0:
            cursor += 1 + middle_extra


def _place_rectangles(top_height: int, width: int, sizes: list[tuple[int, int]]) -> list[tuple[int, int]] | None:
    occupied = np.zeros((top_height, width), dtype=bool)
    order = list(range(len(sizes)))
    random.shuffle(order)
    positions: list[tuple[int, int] | None] = [None] * len(sizes)

    for idx in order:
        height, rect_width = sizes[idx]
        candidates: list[tuple[int, int]] = []
        for top in range(1, top_height - height):
            for left in range(1, width - rect_width):
                r0 = max(0, top - 1)
                r1 = min(top_height, top + height + 1)
                c0 = max(0, left - 1)
                c1 = min(width, left + rect_width + 1)
                if occupied[r0:r1, c0:c1].any():
                    continue
                candidates.append((top, left))
        if not candidates:
            return None
        top, left = random.choice(candidates)
        positions[idx] = (top, left)
        occupied[top : top + height, left : left + rect_width] = True

    return [(top, left) for top, left in positions if top is not None and left is not None]


def _marker_assignment(target_colors: list[int]) -> list[int]:
    mode = random.choice(["identity", "swap", "dup_first", "dup_second"])
    if mode == "identity":
        return [target_colors[0], target_colors[1]]
    if mode == "swap":
        return [target_colors[1], target_colors[0]]
    if mode == "dup_first":
        return [target_colors[0], target_colors[0]]
    return [target_colors[1], target_colors[1]]


def _expected_output(
    grid: np.ndarray,
    rectangles: list[tuple[int, int, int, int]],
    perimeters: list[int],
    target_colors: list[int],
    marker_positions: list[tuple[int, int]],
    marker_colors: list[int],
) -> np.ndarray:
    out = grid.copy()
    color_to_perimeter = {target_colors[i]: perimeters[i] for i in range(2)}
    perimeter_to_rect = {perimeters[i]: rectangles[i] for i in range(2)}

    for mr, mc in marker_positions:
        out[mr, mc] = BG
    for color in marker_colors:
        top, left, height, width = perimeter_to_rect[color_to_perimeter[color]]
        _draw_border(out, top, left, height, width, color)
    return out


def _gen_attempt() -> np.ndarray | None:
    p0, p1 = random.choice(PERIMETER_PAIRS)
    perimeters = [p0, p1]
    sizes = [random.choice(PERIMETER_TO_DIMS[p0]), random.choice(PERIMETER_TO_DIMS[p1])]

    min_width = max(perimeters[0] + perimeters[1] + 1, max(width for _, width in sizes) + 4, 16)
    if min_width > 30:
        return None

    placement: list[tuple[int, int]] | None = None
    width = 0
    top_height = 0
    for _ in range(80):
        width = random.randint(min_width, 30)
        top_min = max(max(height for height, _ in sizes) + 3, 8)
        top_height = random.randint(top_min, min(20, 28))
        placement = _place_rectangles(top_height, width, sizes)
        if placement is not None:
            break
    if placement is None:
        return None

    grid = np.zeros((top_height + 2, width), dtype=int)
    target_colors = random.sample(COLOR_POOL, 2)
    marker_colors = _marker_assignment(target_colors)
    rectangles: list[tuple[int, int, int, int]] = []
    marker_positions: list[tuple[int, int]] = []

    for idx, ((height, rect_width), (top, left)) in enumerate(zip(sizes, placement)):
        rectangles.append((top, left, height, rect_width))
        _draw_border(grid, top, left, height, rect_width, FRAME)
        marker_row = random.randint(top + 1, top + height - 2)
        marker_col = random.randint(left + 1, left + rect_width - 2)
        grid[marker_row, marker_col] = marker_colors[idx]
        marker_positions.append((marker_row, marker_col))

    _paint_palette(grid, target_colors, perimeters)
    expected = _expected_output(grid, rectangles, perimeters, target_colors, marker_positions, marker_colors)
    return grid if np.array_equal(main(grid), expected) else None


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
