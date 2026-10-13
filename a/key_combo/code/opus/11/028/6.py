from __future__ import annotations

import random

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
YELLOW = 4

GRID_SHAPES = ((4, 4), (4, 5), (5, 4), (5, 5))
RECT_SIZES = (3, 4)
GEN_ATTEMPTS = 400


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int]:
    coords = np.argwhere(mask)
    assert coords.size > 0, "expected a foreground rectangle"
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _rectangle_border_mask(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    assert g.ndim == 2 and h > 0 and w > 0
    assert h * h <= 30 and w * w <= 30, "expanded output must stay within ARC limits"
    assert set(map(int, np.unique(g))).issubset({BLACK, BLUE, RED}), "unexpected input colors"

    fg = g != BLACK
    assert np.any(fg), "expected one foreground rectangle"
    r0, c0, r1, c1 = _bbox(fg)
    rect = g[r0 : r1 + 1, c0 : c1 + 1]
    rect_h, rect_w = rect.shape
    assert rect_h >= 3 and rect_w >= 3, "rectangle must have a non-empty interior"
    assert int(np.count_nonzero(fg)) == rect_h * rect_w, "foreground must be one solid rectangle"

    local_border = np.zeros((rect_h, rect_w), dtype=bool)
    local_border[0, :] = True
    local_border[-1, :] = True
    local_border[:, 0] = True
    local_border[:, -1] = True

    assert np.all(rect[local_border] == RED), "rectangle border must be red"
    assert np.all(rect[~local_border] == BLUE), "rectangle interior must be blue"

    border_mask = np.zeros_like(g, dtype=bool)
    border_mask[r0 : r1 + 1, c0 : c1 + 1] = local_border
    assert np.array_equal(border_mask, g == RED), "red pixels must coincide with the rectangle border"
    assert np.any(~fg), "keep some black background so the stamp copies both object and empty space"
    return border_mask


def _stamp_blocks(grid: np.ndarray, control_mask: np.ndarray, empty_color: int) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    out = np.full((h * h, w * w), int(empty_color), dtype=int)
    for r, c in np.argwhere(control_mask):
        out[r * h : (r + 1) * h, c * w : (c + 1) * w] = g
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Use the rectangle border as a recursive stamp-placement mask over a yellow block expansion.

    Input structure:
    - The input is a small black-background grid whose only foreground object is one solid axis-aligned rectangle.
    - The rectangle has a red one-pixel border and a blue strict interior.
    - The rectangle may be offset, edge-touching, or surrounded by extra black background cells.

    Rule:
    - Identify the rectangle and separate its red border cells from its blue interior cells.
    - Build an output of size H^2 x W^2, partitioned into HxW blocks of size HxW.
    - For every border position in the input, stamp the entire input grid into the corresponding output block.
    - Every non-border position contributes an all-yellow block.

    Key insight:
    - The border/interior split is not the final coloring target; the red border becomes the control template that decides where recursive copies of the whole input appear.
    """
    g = np.asarray(grid, dtype=int)
    control_mask = _rectangle_border_mask(g)
    return _stamp_blocks(g, control_mask, YELLOW)


def _all_layouts() -> dict[int, list[tuple[int, int, int, int, int]]]:
    layouts = {0: [], 1: [], 2: []}
    for h, w in GRID_SHAPES:
        for size in RECT_SIZES:
            if size > h or size > w or (size == h and size == w):
                continue
            for r0 in range(h - size + 1):
                for c0 in range(w - size + 1):
                    touches = int(r0 == 0) + int(c0 == 0) + int(r0 + size == h) + int(c0 + size == w)
                    group = 0 if touches == 0 else 1 if touches == 1 else 2
                    layouts[group].append((h, w, size, r0, c0))
    return layouts


LAYOUTS_BY_GROUP = _all_layouts()


def _build_input(h: int, w: int, size: int, r0: int, c0: int) -> np.ndarray:
    grid = np.zeros((h, w), dtype=int)
    grid[r0 : r0 + size, c0 : c0 + size] = BLUE
    grid[r0, c0 : c0 + size] = RED
    grid[r0 + size - 1, c0 : c0 + size] = RED
    grid[r0 : r0 + size, c0] = RED
    grid[r0 : r0 + size, c0 + size - 1] = RED
    return grid


def _check_output_blocks(inp: np.ndarray, out: np.ndarray, control_mask: np.ndarray) -> bool:
    h, w = inp.shape
    for r in range(h):
        for c in range(w):
            block = out[r * h : (r + 1) * h, c * w : (c + 1) * w]
            if control_mask[r, c]:
                if not np.array_equal(block, inp):
                    return False
            elif not np.all(block == YELLOW):
                return False
    return True


def _attempt() -> np.ndarray | None:
    group = random.choice([0, 1, 2])
    h, w, size, r0, c0 = random.choice(LAYOUTS_BY_GROUP[group])
    inp = _build_input(h, w, size, r0, c0)

    try:
        control_mask = _rectangle_border_mask(inp)
        out = main(inp)
    except AssertionError:
        return None

    if not _check_output_blocks(inp, out, control_mask):
        return None

    alt_all_fg = _stamp_blocks(inp, inp != BLACK, YELLOW)
    if np.array_equal(out, alt_all_fg):
        return None

    alt_black_fill = _stamp_blocks(inp, control_mask, BLACK)
    if np.array_equal(out, alt_black_fill):
        return None

    red_count = int(np.count_nonzero(control_mask))
    blue_count = int(np.count_nonzero(inp == BLUE))
    if red_count <= blue_count:
        return None

    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        candidate = _attempt()
        if candidate is not None:
            return candidate
    raise RuntimeError("failed to generate a valid input for puzzle 6")
