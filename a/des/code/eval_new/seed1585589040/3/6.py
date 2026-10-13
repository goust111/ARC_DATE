from __future__ import annotations

import random

import numpy as np


BG = 0
YELLOW = 4
MAROON = 9
SHAPE_COLORS = (1, 2, 3, 5, 6, 7, 8)

DIR_TO_DELTA: dict[str, tuple[int, int]] = {
    "up": (-1, 0),
    "right": (0, 1),
    "down": (1, 0),
    "left": (0, -1),
}

# The 3-pixel arrow is a 2x2 L triomino. Its missing corner gives the global
# facing in clockwise order.
MISSING_TO_DIR: dict[tuple[int, int], str] = {
    (0, 0): "up",
    (0, 1): "right",
    (1, 1): "down",
    (1, 0): "left",
}
DIR_TO_MISSING = {direction: missing for missing, direction in MISSING_TO_DIR.items()}

GRID_H_RANGE = (20, 30)
GRID_W_RANGE = (20, 30)
SHAPE_COUNT_RANGE = (2, 4)
HOLE_OPTIONS = (0, 1, 2, 3)
MAX_PROJECTION = max(HOLE_OPTIONS) + 1
GEN_ATTEMPTS = 400
PLACE_ATTEMPTS = 160


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
                comp.append((int(cr), int(cc)))
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _bbox(cells: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]
    return min(rows), min(cols), max(rows), max(cols)


def _arrow_direction(grid: np.ndarray) -> str:
    candidates: list[str] = []
    for comp in _components(grid == YELLOW):
        if len(comp) != 3:
            continue
        r0, c0, r1, c1 = _bbox(comp)
        if (r1 - r0 + 1, c1 - c0 + 1) != (2, 2):
            continue
        local = {(r - r0, c - c0) for r, c in comp}
        missing = [(r, c) for r in range(2) for c in range(2) if (r, c) not in local]
        if len(missing) == 1 and missing[0] in MISSING_TO_DIR:
            candidates.append(MISSING_TO_DIR[missing[0]])
    assert len(candidates) == 1, "expected one yellow 3-pixel L-shaped arrow"
    return candidates[0]


def _boundary_pixels(cells: list[tuple[int, int]], shape: tuple[int, int]) -> set[tuple[int, int]]:
    h, w = shape
    cell_set = set(cells)
    boundary: set[tuple[int, int]] = set()
    for r, c in cell_set:
        for nr, nc in _neighbors4(r, c):
            if not (0 <= nr < h and 0 <= nc < w) or (nr, nc) not in cell_set:
                boundary.add((r, c))
                break
    return boundary


def _hole_count(grid: np.ndarray, cells: list[tuple[int, int]]) -> int:
    r0, c0, r1, c1 = _bbox(cells)
    sub = grid[r0 : r1 + 1, c0 : c1 + 1]
    bg_mask = sub == BG
    holes = 0
    for comp in _components(bg_mask):
        if all(0 < r < sub.shape[0] - 1 and 0 < c < sub.shape[1] - 1 for r, c in comp):
            holes += 1
    return holes


def main(grid: np.ndarray) -> np.ndarray:
    """
    Project each closed shape's boundary along the yellow arrow direction, using enclosed-hole count as the projection length.

    Input structure:
    - The background is black.
    - One corner contains a yellow 3-pixel L-shaped arrow occupying a 2x2 box with one missing corner; that missing corner encodes one global cardinal direction.
    - The other colored objects are separate monochrome closed shapes. Their visible cells form the shape border, and some shapes may also include internal divider walls.
    - Black 4-connected regions fully surrounded by one shape count as that shape's internal holes.

    Rule:
    - Decode the yellow arrow to one of up, right, down, or left.
    - For each non-yellow shape, count its enclosed black hole regions and set its projection length to holes + 1.
    - From every boundary pixel of that shape, march step by step in the global direction and paint up to that many cells in the shape color.
    - Stop one ray before it would leave the grid or before it would enter the boundary pixel of a different shape. If two different shapes project onto the same cell, recolor that cell maroon.
    - Keep the original shapes and the yellow arrow unchanged.

    Key insight:
    - The yellow corner-arrow gives one global projection direction, while each individual shape turns its own enclosed-hole count into the distance of its projection.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    direction = _arrow_direction(g)
    dr, dc = DIR_TO_DELTA[direction]

    shapes: list[dict[str, object]] = []
    border_owner: dict[tuple[int, int], int] = {}
    for color0 in np.unique(g):
        color = int(color0)
        if color in (BG, YELLOW):
            continue
        for comp in _components(g == color):
            boundary = _boundary_pixels(comp, g.shape)
            shape_id = len(shapes)
            shapes.append(
                {
                    "id": shape_id,
                    "color": color,
                    "cells": comp,
                    "boundary": boundary,
                    "holes": _hole_count(g, comp),
                }
            )
            for cell in boundary:
                border_owner[cell] = shape_id

    out = g.copy()
    projected_by: dict[tuple[int, int], set[int]] = {}

    for shape in shapes:
        shape_id = int(shape["id"])
        length = int(shape["holes"]) + 1
        for r, c in shape["boundary"]:
            for step in range(1, length + 1):
                rr = r + dr * step
                cc = c + dc * step
                if not (0 <= rr < h and 0 <= cc < w):
                    break
                if int(g[rr, cc]) == YELLOW:
                    break
                blocker = border_owner.get((rr, cc))
                if blocker is not None and blocker != shape_id:
                    break
                projected_by.setdefault((rr, cc), set()).add(shape_id)

    for (r, c), owners in projected_by.items():
        if int(g[r, c]) == YELLOW:
            continue
        if len(owners) >= 2:
            out[r, c] = MAROON
            continue
        owner = next(iter(owners))
        out[r, c] = int(shapes[owner]["color"])

    return out


def _mask_holes(mask: np.ndarray) -> int:
    return _hole_count(np.where(mask, 1, 0), [(int(r), int(c)) for r, c in np.argwhere(mask)])


def _solid_mask() -> np.ndarray:
    h = random.randint(3, 6)
    w = random.randint(3, 6)
    return np.ones((h, w), dtype=bool)


def _framed_mask(holes: int) -> np.ndarray:
    assert holes >= 1
    orientation = random.choice(("vertical", "horizontal"))
    if orientation == "vertical":
        chamber_widths = [random.randint(1, 3) for _ in range(holes)]
        h = random.randint(4, 7)
        w = 2 + sum(chamber_widths) + (holes - 1)
        mask = np.zeros((h, w), dtype=bool)
        mask[0, :] = True
        mask[-1, :] = True
        mask[:, 0] = True
        mask[:, -1] = True
        divider_col = 1 + chamber_widths[0]
        for width in chamber_widths[1:]:
            mask[:, divider_col] = True
            divider_col += 1 + width
        return mask

    chamber_heights = [random.randint(1, 3) for _ in range(holes)]
    h = 2 + sum(chamber_heights) + (holes - 1)
    w = random.randint(4, 7)
    mask = np.zeros((h, w), dtype=bool)
    mask[0, :] = True
    mask[-1, :] = True
    mask[:, 0] = True
    mask[:, -1] = True
    divider_row = 1 + chamber_heights[0]
    for height in chamber_heights[1:]:
        mask[divider_row, :] = True
        divider_row += 1 + height
    return mask


def _shape_mask(holes: int) -> np.ndarray:
    mask = _solid_mask() if holes == 0 else _framed_mask(holes)
    assert _mask_holes(mask) == holes
    return mask


def _arrow_cells(shape: tuple[int, int], direction: str) -> list[tuple[int, int]]:
    h, w = shape
    missing = DIR_TO_MISSING[direction]
    r0 = h - 2
    c0 = w - 2
    cells = []
    for dr in range(2):
        for dc in range(2):
            if (dr, dc) != missing:
                cells.append((r0 + dr, c0 + dc))
    return cells


def _reserve_zone(reserved: np.ndarray, r0: int, c0: int, h: int, w: int) -> None:
    rr0 = max(0, r0 - 1)
    cc0 = max(0, c0 - 1)
    rr1 = min(reserved.shape[0], r0 + h + 1)
    cc1 = min(reserved.shape[1], c0 + w + 1)
    reserved[rr0:rr1, cc0:cc1] = True


def _place_shape(
    grid: np.ndarray,
    reserved: np.ndarray,
    mask: np.ndarray,
    color: int,
    direction: str,
) -> bool:
    gh, gw = grid.shape
    mh, mw = mask.shape

    top_min = 1 + (MAX_PROJECTION if direction == "up" else 0)
    left_min = 1 + (MAX_PROJECTION if direction == "left" else 0)
    top_max = gh - mh - 3 - (MAX_PROJECTION if direction == "down" else 0)
    left_max = gw - mw - 3 - (MAX_PROJECTION if direction == "right" else 0)
    if top_max < top_min or left_max < left_min:
        return False

    for _ in range(PLACE_ATTEMPTS):
        r0 = random.randint(top_min, top_max)
        c0 = random.randint(left_min, left_max)
        if reserved[r0 : r0 + mh, c0 : c0 + mw].any():
            continue
        for rr, cc in np.argwhere(mask):
            grid[r0 + int(rr), c0 + int(cc)] = color
        _reserve_zone(reserved, r0, c0, mh, mw)
        return True
    return False


def _build_scene() -> np.ndarray | None:
    h = random.randint(*GRID_H_RANGE)
    w = random.randint(*GRID_W_RANGE)
    direction = random.choice(tuple(DIR_TO_DELTA))
    shape_count = random.randint(*SHAPE_COUNT_RANGE)

    hole_counts = [random.choice(HOLE_OPTIONS) for _ in range(shape_count)]
    if len(set(hole_counts)) < 2 or max(hole_counts) == 0:
        return None

    masks = [_shape_mask(holes) for holes in hole_counts]
    colors = random.sample(SHAPE_COLORS, shape_count)
    order = sorted(range(shape_count), key=lambda idx: masks[idx].size, reverse=True)

    grid = np.full((h, w), BG, dtype=int)
    reserved = np.zeros((h, w), dtype=bool)
    _reserve_zone(reserved, h - 2, w - 2, 2, 2)

    for r, c in _arrow_cells((h, w), direction):
        grid[r, c] = YELLOW

    for idx in order:
        if not _place_shape(grid, reserved, masks[idx], colors[idx], direction):
            return None

    out = main(grid)
    if np.array_equal(out, grid):
        return None
    if np.count_nonzero(out != grid) < 4:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _build_scene()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input grid")
