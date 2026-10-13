from __future__ import annotations

import random

import numpy as np


MAROON = 9
GREY = 5
DIR4 = ((1, 0), (-1, 0), (0, 1), (0, -1))

GRID_H_RANGE = (10, 20)
GRID_W_RANGE = (10, 20)
MAX_RECT_DIM = 7
GEN_RETRIES = 400


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    parts: list[list[tuple[int, int]]] = []
    for r, c in np.argwhere(mask):
        r = int(r)
        c = int(c)
        if seen[r, c]:
            continue
        stack = [(r, c)]
        seen[r, c] = True
        cells: list[tuple[int, int]] = []
        while stack:
            cr, cc = stack.pop()
            cells.append((cr, cc))
            for dr, dc in DIR4:
                nr = cr + dr
                nc = cc + dc
                if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                    seen[nr, nc] = True
                    stack.append((nr, nc))
        parts.append(cells)
    return parts


def _rectangles_for_colors(grid: np.ndarray, colors: list[int]) -> list[tuple[int, int, int, int, int]] | None:
    rectangles: list[tuple[int, int, int, int, int]] = []
    for color in colors:
        for cells in _components(grid == int(color)):
            rows = [r for r, _c in cells]
            cols = [c for _r, c in cells]
            r0 = min(rows)
            r1 = max(rows)
            c0 = min(cols)
            c1 = max(cols)
            if len(cells) != (r1 - r0 + 1) * (c1 - c0 + 1):
                return None
            if not np.all(grid[r0 : r1 + 1, c0 : c1 + 1] == int(color)):
                return None
            rectangles.append((int(color), r0, c0, r1, c1))
    rectangles.sort(key=lambda item: (item[1], item[2], item[0]))
    return rectangles


def _infer_checker_colors(grid: np.ndarray) -> tuple[int, int]:
    g = np.asarray(grid, dtype=int)
    rows, cols = np.indices(g.shape)
    even_mask = ((rows + cols) & 1) == 0
    odd_mask = ~even_mask
    colors = [int(c) for c in np.unique(g)]

    even_candidates: list[int] = []
    odd_candidates: list[int] = []
    for color in colors:
        parity_values = set(map(int, np.unique(((rows + cols) & 1)[g == color])))
        if parity_values == {0}:
            even_candidates.append(color)
        elif parity_values == {1}:
            odd_candidates.append(color)
    if len(even_candidates) == 1 and len(odd_candidates) == 1:
        return even_candidates[0], odd_candidates[0]

    best: tuple[tuple[int, int, int], int, int] | None = None
    for even_color in colors:
        if np.any((g == even_color) & odd_mask):
            continue
        for odd_color in colors:
            if odd_color == even_color:
                continue
            if np.any((g == odd_color) & even_mask):
                continue
            stamp_colors = [color for color in colors if color not in {even_color, odd_color}]
            rectangles = _rectangles_for_colors(g, stamp_colors)
            if rectangles is None or not rectangles:
                continue
            stamp_mask = np.isin(g, stamp_colors)
            score = (
                int(len(rectangles) <= 5),
                int(len(stamp_colors) == 2),
                int(np.size(g) - np.count_nonzero(stamp_mask)),
            )
            if best is None or score > best[0]:
                best = (score, int(even_color), int(odd_color))

    assert best is not None, "failed to infer checkerboard colors"
    return best[1], best[2]


def _checkerboard(shape: tuple[int, int], even_color: int, odd_color: int) -> np.ndarray:
    rows, cols = np.indices(shape)
    return np.where(((rows + cols) & 1) == 0, int(even_color), int(odd_color)).astype(int)


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    even_color, odd_color = _infer_checker_colors(g)
    stamp_colors = [int(c) for c in np.unique(g) if int(c) not in {even_color, odd_color}]
    rectangles = _rectangles_for_colors(g, stamp_colors)
    assert rectangles is not None and rectangles, "expected one or more solid stamp rectangles"

    out = _checkerboard(g.shape, even_color, odd_color)
    for _color, r0, c0, r1, c1 in rectangles:
        fill = MAROON if ((r0 + c0) & 1) == 0 else GREY
        out[r0 : r1 + 1, c0 : c1 + 1] = fill
    return out


def _positions_for_dim(used: np.ndarray, height: int, width: int, parity: int) -> list[tuple[int, int]]:
    h, w = used.shape
    positions: list[tuple[int, int]] = []
    for r0 in range(h - height + 1):
        for c0 in range(w - width + 1):
            if ((r0 + c0) & 1) != parity:
                continue
            rr0 = max(0, r0 - 1)
            rr1 = min(h, r0 + height + 1)
            cc0 = max(0, c0 - 1)
            cc1 = min(w, c0 + width + 1)
            if used[rr0:rr1, cc0:cc1].any():
                continue
            positions.append((r0, c0))
    return positions


def _available_dims(used: np.ndarray, parity: int) -> list[tuple[int, int, list[tuple[int, int]]]]:
    h, w = used.shape
    max_area = max(8, (h * w) // 6)
    dims: list[tuple[int, int, list[tuple[int, int]]]] = []
    for height in range(1, min(MAX_RECT_DIM, h) + 1):
        for width in range(1, min(MAX_RECT_DIM, w) + 1):
            area = height * width
            if area < 2 or area > max_area:
                continue
            positions = _positions_for_dim(used, height, width, parity)
            if positions:
                dims.append((height, width, positions))
    return dims


def _expected_output(
    shape: tuple[int, int],
    even_color: int,
    odd_color: int,
    rectangles: list[tuple[int, int, int, int, int]],
) -> np.ndarray:
    out = _checkerboard(shape, even_color, odd_color)
    for _color, r0, c0, r1, c1 in rectangles:
        out[r0 : r1 + 1, c0 : c1 + 1] = MAROON if ((r0 + c0) & 1) == 0 else GREY
    return out


def _gen_attempt() -> np.ndarray | None:
    height = random.randint(*GRID_H_RANGE)
    width = random.randint(*GRID_W_RANGE)

    palette = [c for c in range(10) if c not in {MAROON, GREY}]
    random.shuffle(palette)
    even_color, odd_color, primary_color, secondary_color = palette[:4]

    n_primary = random.randint(2, 4)
    total_rectangles = n_primary + 1
    colors = [primary_color] * n_primary + [secondary_color]
    parities = [0, 1] + [random.randint(0, 1) for _ in range(total_rectangles - 2)]
    paired = list(zip(colors, parities, strict=True))
    random.shuffle(paired)

    used = np.zeros((height, width), dtype=bool)
    rectangles: list[tuple[int, int, int, int, int]] = []
    for color, parity in paired:
        dims = _available_dims(used, parity)
        if not dims:
            return None
        rect_h, rect_w, positions = random.choice(dims)
        r0, c0 = random.choice(positions)
        r1 = r0 + rect_h - 1
        c1 = c0 + rect_w - 1
        used[r0 : r1 + 1, c0 : c1 + 1] = True
        rectangles.append((int(color), r0, c0, r1, c1))

    inp = _checkerboard((height, width), even_color, odd_color)
    for color, r0, c0, r1, c1 in rectangles:
        inp[r0 : r1 + 1, c0 : c1 + 1] = int(color)

    if np.count_nonzero(inp == even_color) < 4 or np.count_nonzero(inp == odd_color) < 4:
        return None
    if len({(r0 + c0) & 1 for _color, r0, c0, _r1, _c1 in rectangles}) != 2:
        return None
    covered_area = sum((r1 - r0 + 1) * (c1 - c0 + 1) for _color, r0, c0, r1, c1 in rectangles)
    if covered_area > (height * width) // 2:
        return None

    expected = _expected_output((height, width), even_color, odd_color, rectangles)
    try:
        if np.array_equal(main(inp), expected):
            return inp
    except AssertionError:
        return None
    return None


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid checkerboard stamp puzzle")
