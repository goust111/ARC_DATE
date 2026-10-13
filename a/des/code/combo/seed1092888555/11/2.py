import random
from collections import Counter

import numpy as np


BG = 5
RED = 2
LEGEND_LENGTHS = (2, 3, 4, 5, 6, 7)
PALETTE = (1, 3, 4, 6, 7, 8, 9)
GEN_ATTEMPTS = 300


def _components_of_color(grid: np.ndarray, color: int) -> list[dict]:
    h, w = grid.shape
    mask = grid == color
    seen = np.zeros((h, w), dtype=bool)
    comps: list[dict] = []

    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            cells = []
            while stack:
                rr, cc = stack.pop()
                cells.append((rr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            rows = [p[0] for p in cells]
            cols = [p[1] for p in cells]
            r0, r1 = min(rows), max(rows)
            c0, c1 = min(cols), max(cols)
            comps.append(
                {
                    "color": int(color),
                    "cells": cells,
                    "size": len(cells),
                    "bbox": (r0, c0, r1, c1),
                }
            )
    return comps


def _is_straight_line(comp: dict) -> bool:
    r0, c0, r1, c1 = comp["bbox"]
    return (r0 == r1 or c0 == c1) and comp["size"] == max(r1 - r0 + 1, c1 - c0 + 1)


def _line_length(comp: dict) -> int:
    r0, c0, r1, c1 = comp["bbox"]
    return max(r1 - r0 + 1, c1 - c0 + 1)


def _is_horizontal(comp: dict) -> bool:
    r0, _, r1, _ = comp["bbox"]
    return r0 == r1


def _is_filled_square(comp: dict) -> bool:
    r0, c0, r1, c1 = comp["bbox"]
    side_r = r1 - r0 + 1
    side_c = c1 - c0 + 1
    return side_r == side_c and comp["size"] == side_r * side_c


def _infer_bg(grid: np.ndarray) -> int:
    colors, counts = np.unique(grid, return_counts=True)
    return int(colors[np.argmax(counts)])


def _infer_red(grid: np.ndarray, bg: int) -> int:
    candidates = []
    for color in np.unique(grid):
        color = int(color)
        if color == bg:
            continue
        comps = _components_of_color(grid, color)
        horiz_lengths = {
            _line_length(comp)
            for comp in comps
            if _is_straight_line(comp) and _is_horizontal(comp)
        }
        if all(length in horiz_lengths for length in LEGEND_LENGTHS):
            candidates.append(color)
    assert len(candidates) == 1, "expected unique red line color"
    return candidates[0]


def _extract_mapping(grid: np.ndarray, bg: int, red: int) -> tuple[dict[int, int], set[tuple[int, int]]]:
    red_comps = [comp for comp in _components_of_color(grid, red) if _is_straight_line(comp)]
    square_comps = []
    for color in np.unique(grid):
        color = int(color)
        if color in (bg, red):
            continue
        square_comps.extend([comp for comp in _components_of_color(grid, color) if _is_filled_square(comp)])

    mapping: dict[int, int] = {}
    legend_cells: set[tuple[int, int]] = set()

    for comp in red_comps:
        if not _is_horizontal(comp):
            continue
        length = _line_length(comp)
        if length not in LEGEND_LENGTHS:
            continue
        r0, c0, _r1, c1 = comp["bbox"]
        candidates = []
        for square in square_comps:
            sr0, sc0, sr1, _sc1 = square["bbox"]
            if not (sr0 <= r0 <= sr1):
                continue
            if sc0 <= c1:
                continue
            gap = grid[r0, c1 + 1 : sc0]
            if np.any(gap != bg):
                continue
            candidates.append((sc0 - c1, square))
        if len(candidates) != 1:
            continue
        square = candidates[0][1]
        mapping[length] = square["color"]
        legend_cells.update(comp["cells"])
        legend_cells.update(square["cells"])

    assert set(mapping) == set(LEGEND_LENGTHS), "legend must define lengths 2..7"
    return mapping, legend_cells


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    bg = _infer_bg(g)
    red = _infer_red(g, bg)
    mapping, legend_cells = _extract_mapping(g, bg, red)

    out = g.copy()
    for r, c in legend_cells:
        out[r, c] = bg

    for comp in _components_of_color(g, red):
        if not _is_straight_line(comp):
            continue
        if any(cell in legend_cells for cell in comp["cells"]):
            continue
        color = mapping[_line_length(comp)]
        for r, c in comp["cells"]:
            out[r, c] = color

    return out


def _legend_layout(height: int, width: int, square_size: int, corner: str, order: list[int]) -> tuple[dict[int, int], tuple[int, int, int, int]]:
    row_step = square_size + 1
    legend_h = row_step * len(order) - 1
    legend_w = max(order) + 2 + square_size

    if "top" in corner:
        top = 0
    else:
        top = height - legend_h
    if "left" in corner:
        left = 0
    else:
        left = width - legend_w

    return {length: top + i * row_step for i, length in enumerate(order)}, (top, left, legend_h, legend_w)


def _try_place_line(blocked: np.ndarray, length: int, orientation: str) -> list[tuple[int, int]] | None:
    h, w = blocked.shape
    for _ in range(80):
        if orientation == "h":
            r = random.randrange(h)
            c = random.randrange(w - length + 1)
            cells = [(r, c + dc) for dc in range(length)]
        else:
            r = random.randrange(h - length + 1)
            c = random.randrange(w)
            cells = [(r + dr, c) for dr in range(length)]
        if all(not blocked[rr, cc] for rr, cc in cells):
            return cells
    return None


def _block_with_margin(blocked: np.ndarray, cells: list[tuple[int, int]]) -> None:
    h, w = blocked.shape
    for r, c in cells:
        for rr in range(max(0, r - 1), min(h, r + 2)):
            for cc in range(max(0, c - 1), min(w, c + 2)):
                blocked[rr, cc] = True


def _gen_attempt() -> np.ndarray | None:
    square_size = random.choice((1, 2))
    height = random.randint(20, 28)
    width = random.randint(20, 30)
    order = list(LEGEND_LENGTHS)
    random.shuffle(order)
    corner = random.choice(("top_left", "top_right", "bottom_left", "bottom_right"))
    row_map, legend_box = _legend_layout(height, width, square_size, corner, order)
    top, left, legend_h, legend_w = legend_box

    if top < 0 or left < 0:
        return None

    grid = np.full((height, width), BG, dtype=int)
    blocked = np.zeros((height, width), dtype=bool)
    mapping_colors = random.sample(PALETTE, len(LEGEND_LENGTHS))
    mapping = dict(zip(LEGEND_LENGTHS, mapping_colors))

    for length in order:
        row = row_map[length]
        bar_c0 = left
        bar_c1 = left + length
        sq_c0 = left + length + 2
        sq_c1 = sq_c0 + square_size
        if sq_c1 > width or row + square_size > height:
            return None
        grid[row, bar_c0:bar_c1] = RED
        grid[row : row + square_size, sq_c0:sq_c1] = mapping[length]
        for rr in range(max(0, row - 1), min(height, row + square_size + 1)):
            for cc in range(max(0, left - 1), min(width, left + legend_w + 1)):
                blocked[rr, cc] = True

    n_lines = random.randint(5, 9)
    lengths = [random.choice(LEGEND_LENGTHS) for _ in range(n_lines)]
    if len(set(lengths)) < 3:
        extra = random.sample(LEGEND_LENGTHS, 3)
        lengths[:3] = extra
        random.shuffle(lengths)

    orientations = [random.choice(("h", "v")) for _ in range(n_lines)]
    if all(o == "h" for o in orientations):
        orientations[0] = "v"
    if all(o == "v" for o in orientations):
        orientations[0] = "h"

    for length, orientation in zip(lengths, orientations):
        cells = _try_place_line(blocked, length, orientation)
        if cells is None:
            return None
        for r, c in cells:
            grid[r, c] = RED
        _block_with_margin(blocked, cells)

    out = main(grid)
    non_bg_out = set(map(int, np.unique(out))) - {BG}
    if RED in non_bg_out:
        return None
    if any(color not in mapping.values() for color in non_bg_out):
        return None
    if np.any(out[top : top + legend_h, left : left + legend_w] != BG):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
