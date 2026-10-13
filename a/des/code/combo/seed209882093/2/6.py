from __future__ import annotations

import random
from collections import Counter
from itertools import combinations

import numpy as np


BG = 0
EVEN_COLOR = 8
ODD_COLOR = 3
FRAGMENT_COLORS = (1, 2, 4, 5, 6, 7, 9)

BOARD_H_RANGE = (8, 15)
BOARD_W_RANGE = (9, 17)
MARGIN_RANGE = (3, 8)
MAX_HOLES_BY_AREA = ((90, 3), (130, 4), (10**9, 5))

MAX_GENERATION_ATTEMPTS = 400
SHAPE_ATTEMPTS = 80
HOLE_PLACEMENT_ATTEMPTS = 160
FRAGMENT_PLACEMENT_ATTEMPTS = 220

DIR4 = ((-1, 0), (1, 0), (0, -1), (0, 1))


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not bool(mask[r, c]) or bool(seen[r, c]):
                continue
            stack = [(int(r), int(c))]
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                comp.append((int(rr), int(cc)))
                for dr, dc in DIR4:
                    nr = int(rr + dr)
                    nc = int(cc + dc)
                    if 0 <= nr < h and 0 <= nc < w and bool(mask[nr, nc]) and not bool(seen[nr, nc]):
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _normalize(coords: list[tuple[int, int]] | set[tuple[int, int]]) -> tuple[tuple[int, int], ...]:
    rows = [int(r) for r, _ in coords]
    cols = [int(c) for _, c in coords]
    r0 = min(rows)
    c0 = min(cols)
    return tuple(sorted((int(r - r0), int(c - c0)) for r, c in coords))


def _shape_size(shape: tuple[tuple[int, int], ...]) -> tuple[int, int]:
    return max(r for r, _ in shape) + 1, max(c for _, c in shape) + 1


def _make_checkerboard(height: int, width: int, even_color: int, odd_color: int) -> np.ndarray:
    rows, cols = np.indices((int(height), int(width)))
    return np.where(((rows + cols) & 1) == 0, int(even_color), int(odd_color)).astype(int)


def _dilate(cells: set[tuple[int, int]], height: int, width: int) -> set[tuple[int, int]]:
    out: set[tuple[int, int]] = set()
    for r, c in cells:
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                nr = int(r + dr)
                nc = int(c + dc)
                if 0 <= nr < height and 0 <= nc < width:
                    out.add((nr, nc))
    return out


def _find_board(
    g: np.ndarray,
) -> tuple[tuple[int, int, int, int], int, int]:
    colors = [int(v) for v in np.unique(g) if int(v) != BG]
    assert len(colors) >= 2, "expected at least two non-background colors"

    best_key: tuple[int, int, int] | None = None
    best: tuple[tuple[int, int, int, int], int, int] | None = None

    for color_a, color_b in combinations(colors, 2):
        mask = (g == int(color_a)) | (g == int(color_b))
        if not bool(mask.any()):
            continue
        coords = np.argwhere(mask)
        r0, c0 = map(int, coords.min(axis=0))
        r1, c1 = map(int, coords.max(axis=0))
        area = int((r1 - r0 + 1) * (c1 - c0 + 1))

        score_ab = 0
        score_ba = 0
        for rr, cc in coords:
            r = int(rr)
            c = int(cc)
            parity = ((r - r0) + (c - c0)) & 1
            val = int(g[r, c])
            if (parity == 0 and val == color_a) or (parity == 1 and val == color_b):
                score_ab += 1
            if (parity == 0 and val == color_b) or (parity == 1 and val == color_a):
                score_ba += 1

        if score_ab >= score_ba:
            even_color, odd_color = int(color_a), int(color_b)
            score = int(score_ab)
        else:
            even_color, odd_color = int(color_b), int(color_a)
            score = int(score_ba)

        holes = int(area - score)
        key = (score, -holes, -area)
        if best_key is None or key > best_key:
            best_key = key
            best = ((r0, c0, r1, c1), even_color, odd_color)

    assert best is not None, "failed to detect the checkerboard"
    return best


def main(grid: np.ndarray) -> np.ndarray:
    """
    Restore the perforated checkerboard and erase the matching outside fragments.

    The two dominant structured colors form a large alternating rectangle. Black cells inside
    that rectangle are holes, and every outside fragment has the same shape as one hole.
    Fragments disappear in the output; the rectangle is filled back to a perfect checkerboard.
    """
    g = np.asarray(grid, dtype=int)
    (r0, c0, r1, c1), even_color, odd_color = _find_board(g)

    board_mask = np.zeros(g.shape, dtype=bool)
    board_mask[r0 : r1 + 1, c0 : c1 + 1] = True

    hole_shapes = Counter(_normalize(comp) for comp in _components(board_mask & (g == BG)))
    fragment_shapes: Counter[tuple[tuple[int, int], ...]] = Counter()
    for comp in _components((~board_mask) & (g != BG)):
        colors = {int(g[r, c]) for r, c in comp}
        assert len(colors) == 1, "each outside fragment must be monochrome"
        fragment_color = next(iter(colors))
        assert fragment_color not in {int(even_color), int(odd_color)}, "fragment color overlaps board colors"
        fragment_shapes[_normalize(comp)] += 1

    assert hole_shapes == fragment_shapes, "hole/fragment shapes must match exactly"

    out = np.zeros_like(g)
    out[r0 : r1 + 1, c0 : c1 + 1] = _make_checkerboard(
        r1 - r0 + 1, c1 - c0 + 1, even_color, odd_color
    )
    return out


def _random_shape(area: int) -> tuple[tuple[int, int], ...]:
    cells = {(0, 0)}
    while len(cells) < int(area):
        frontier = {
            (int(r + dr), int(c + dc))
            for r, c in cells
            for dr, dc in DIR4
            if (int(r + dr), int(c + dc)) not in cells
        }
        cells.add(random.choice(tuple(frontier)))
    return _normalize(cells)


def _choose_shapes(count: int) -> list[tuple[tuple[int, int], ...]] | None:
    shapes: list[tuple[tuple[int, int], ...]] = []
    seen: set[tuple[tuple[int, int], ...]] = set()
    for _ in range(SHAPE_ATTEMPTS):
        if len(shapes) == int(count):
            return shapes
        shape = _random_shape(random.randint(2, 5))
        if shape in seen:
            continue
        seen.add(shape)
        shapes.append(shape)
    return None


def _place_holes(
    board_h: int,
    board_w: int,
    shapes: list[tuple[tuple[int, int], ...]],
) -> list[dict] | None:
    blocked: set[tuple[int, int]] = set()
    placements: list[dict] = []
    ordered = sorted(shapes, key=lambda s: (_shape_size(s)[0] * _shape_size(s)[1], len(s)), reverse=True)

    for shape in ordered:
        shape_h, shape_w = _shape_size(shape)
        row_max = int(board_h - shape_h - 1)
        col_max = int(board_w - shape_w - 1)
        if row_max < 1 or col_max < 1:
            return None

        placed = None
        for _ in range(HOLE_PLACEMENT_ATTEMPTS):
            top = random.randint(1, row_max)
            left = random.randint(1, col_max)
            cells = {(int(top + r), int(left + c)) for r, c in shape}
            if cells & blocked:
                continue
            placed = {"shape": shape, "cells": cells}
            blocked |= _dilate(cells, board_h, board_w)
            break

        if placed is None:
            return None
        placements.append(placed)

    return placements


def _board_guard(
    top: int,
    left: int,
    height: int,
    width: int,
    grid_h: int,
    grid_w: int,
) -> set[tuple[int, int]]:
    rect = {
        (int(r), int(c))
        for r in range(int(top), int(top + height))
        for c in range(int(left), int(left + width))
    }
    return _dilate(rect, grid_h, grid_w)


def _place_fragments(
    grid_h: int,
    grid_w: int,
    board_bbox: tuple[int, int, int, int],
    shapes: list[tuple[tuple[int, int], ...]],
    colors: list[int],
) -> list[dict] | None:
    r0, c0, r1, c1 = board_bbox
    blocked = _board_guard(r0, c0, r1 - r0 + 1, c1 - c0 + 1, grid_h, grid_w)
    placements: list[dict] = []
    items = list(zip(shapes, colors))
    items.sort(key=lambda item: (_shape_size(item[0])[0] * _shape_size(item[0])[1], len(item[0])), reverse=True)

    for shape, color in items:
        shape_h, shape_w = _shape_size(shape)
        placed = None
        for _ in range(FRAGMENT_PLACEMENT_ATTEMPTS):
            top = random.randint(0, int(grid_h - shape_h))
            left = random.randint(0, int(grid_w - shape_w))
            cells = {(int(top + r), int(left + c)) for r, c in shape}
            if cells & blocked:
                continue
            placed = {"shape": shape, "color": int(color), "cells": cells}
            blocked |= _dilate(cells, grid_h, grid_w)
            break
        if placed is None:
            return None
        placements.append(placed)

    return placements


def _render_input(
    grid_h: int,
    grid_w: int,
    board_top: int,
    board_left: int,
    board_h: int,
    board_w: int,
    holes: list[dict],
    fragments: list[dict],
) -> np.ndarray:
    grid = np.zeros((int(grid_h), int(grid_w)), dtype=int)
    grid[board_top : board_top + board_h, board_left : board_left + board_w] = _make_checkerboard(
        board_h, board_w, EVEN_COLOR, ODD_COLOR
    )

    for hole in holes:
        for r, c in hole["cells"]:
            grid[int(board_top + r), int(board_left + c)] = BG

    for fragment in fragments:
        color = int(fragment["color"])
        for r, c in fragment["cells"]:
            grid[int(r), int(c)] = color

    return grid


def _max_holes(board_h: int, board_w: int) -> int:
    area = int(board_h * board_w)
    for limit, max_holes in MAX_HOLES_BY_AREA:
        if area < limit:
            return min(int(max_holes), len(FRAGMENT_COLORS))
    raise AssertionError("unreachable")


def _generate_attempt() -> np.ndarray | None:
    board_h = random.randint(*BOARD_H_RANGE)
    board_w = random.randint(*BOARD_W_RANGE)
    top_margin = random.randint(*MARGIN_RANGE)
    bottom_margin = random.randint(*MARGIN_RANGE)
    left_margin = random.randint(*MARGIN_RANGE)
    right_margin = random.randint(*MARGIN_RANGE)

    grid_h = int(board_h + top_margin + bottom_margin)
    grid_w = int(board_w + left_margin + right_margin)
    board_top = int(top_margin)
    board_left = int(left_margin)

    hole_count = random.randint(2, _max_holes(board_h, board_w))
    shapes = _choose_shapes(hole_count)
    if shapes is None:
        return None

    holes = _place_holes(board_h, board_w, shapes)
    if holes is None:
        return None

    fragment_colors = random.sample(FRAGMENT_COLORS, hole_count)
    fragments = _place_fragments(
        grid_h,
        grid_w,
        (board_top, board_left, board_top + board_h - 1, board_left + board_w - 1),
        [hole["shape"] for hole in holes],
        fragment_colors,
    )
    if fragments is None:
        return None

    inp = _render_input(grid_h, grid_w, board_top, board_left, board_h, board_w, holes, fragments)
    expected = np.zeros_like(inp)
    expected[board_top : board_top + board_h, board_left : board_left + board_w] = _make_checkerboard(
        board_h, board_w, EVEN_COLOR, ODD_COLOR
    )
    if not np.array_equal(main(inp), expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 6")
