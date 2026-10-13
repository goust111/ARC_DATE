from __future__ import annotations

import numpy as np

from arc_utils import bbox_coords, colors_present, get_bg


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recenter each four-pixel color mask in a shared compact canvas.

    Input structure:
    - The grid has one dominant background color.
    - Every non-background color appears as exactly four pixels.
    - Each color's four-pixel mask has an odd-compatible row and column center
      when compared with the widest/tallest foreground mask.

    Rule:
    - For each non-background color, take its four pixels relative to that color's bbox.
    - Make the output background-colored with size equal to the maximum foreground
      bbox height and width.
    - Paste every relative color mask centered in that common output canvas.

    Key insight:
    - The large input only separates the colored masks spatially; the desired image
      is their center-aligned overlay after removing the surrounding background.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    _, counts = np.unique(g, return_counts=True)
    assert np.count_nonzero(counts == counts.max()) == 1, "expected a unique dominant background color"
    bg = get_bg(g)
    specs = []
    for color in colors_present(g, exclude={bg}):
        coords = np.argwhere(g == int(color))
        assert coords.shape == (4, 2), "each foreground color must occur exactly four times"
        r0, c0, r1, c1 = bbox_coords(coords)
        specs.append((int(color), coords, r0, c0, r1 - r0, c1 - c0))

    assert specs, "expected at least one foreground color"
    max_row_span = max(row_span for _color, _coords, _r0, _c0, row_span, _col_span in specs)
    max_col_span = max(col_span for _color, _coords, _r0, _c0, _row_span, col_span in specs)

    out = np.full((max_row_span + 1, max_col_span + 1), bg, dtype=int)
    occupied = np.zeros(out.shape, dtype=bool)

    for color, coords, r0, c0, row_span, col_span in specs:
        row_extra = max_row_span - row_span
        col_extra = max_col_span - col_span
        assert row_extra % 2 == 0 and col_extra % 2 == 0, "masks must center on the output grid"
        row_offset = row_extra // 2
        col_offset = col_extra // 2

        for r, c in coords:
            rr = int(r) - r0 + row_offset
            cc = int(c) - c0 + col_offset
            assert not occupied[rr, cc], "centered foreground masks must not overlap"
            occupied[rr, cc] = True
            out[rr, cc] = color

    return out

# --------------- generator (o5x) ---------------
import itertools
import random
from arc_utils import paste, random_free_location

# Structure choices and retry budget.
STRUCTURE_CHOICES = ((3, 2), (5, 2), (5, 3), (5, 4), (7, 2), (7, 3), (7, 4))
MAX_GENERATION_ATTEMPTS = 300

# Mask geometry.
MASK_DIM_MIN, MASK_DIM_STEP = 3, 2
PATTERN_CORNERS, PATTERN_DIAMOND = "corners", "diamond"
FOREGROUND_PIXELS_PER_COLOR = 4

# Input canvas and palette bounds.
INPUT_ROW_RANGE, INPUT_COL_RANGE = range(10, 19), range(8, 20)
BACKGROUND_FRACTION_MIN, BACKGROUND_FRACTION_MAX = 0.88, 0.95
PALETTE_SIZE = 10


def _gen_mask_cells(h, w, pattern):
    if pattern == PATTERN_CORNERS:
        return ((0, 0), (0, w - 1), (h - 1, 0), (h - 1, w - 1))
    return ((0, w // 2), (h // 2, 0), (h // 2, w - 1), (h - 1, w // 2))


def _gen_make_spec(S, h, w, pattern):
    rel = _gen_mask_cells(h, w, pattern)
    ro = (S - h) // 2
    co = (S - w) // 2
    centered = frozenset((r + ro, c + co) for r, c in rel)
    return {"h": h, "w": w, "pattern": pattern, "rel": rel, "cells": centered}


def _gen_choose_specs(S, n_colors):
    dims = list(range(MASK_DIM_MIN, S + 1, MASK_DIM_STEP))
    corners = [_gen_make_spec(S, h, w, PATTERN_CORNERS) for h in dims for w in dims]
    diamonds = [_gen_make_spec(S, d, d, PATTERN_DIAMOND) for d in dims]

    valid = []
    for diamond in diamonds:
        for corner_group in itertools.combinations(corners, n_colors - 1):
            chosen = (diamond,) + corner_group
            occupied = set()
            ok = True
            for spec in chosen:
                if occupied & spec["cells"]:
                    ok = False
                    break
                occupied.update(spec["cells"])
            if not ok:
                continue
            if max(spec["h"] for spec in chosen) != S:
                continue
            if max(spec["w"] for spec in chosen) != S:
                continue
            valid.append(chosen)

    if not valid:
        return None
    chosen = list(random.choice(valid))
    random.shuffle(chosen)
    return chosen


def _gen_grid_shape(S, n_colors):
    pairs = []
    for rows in INPUT_ROW_RANGE:
        for cols in INPUT_COL_RANGE:
            if rows < S or cols < S:
                continue
            area = rows * cols
            bg_frac = (area - FOREGROUND_PIXELS_PER_COLOR * n_colors) / area
            if BACKGROUND_FRACTION_MIN <= bg_frac <= BACKGROUND_FRACTION_MAX:
                pairs.append((rows, cols))
    return random.choice(pairs)


def _gen_assign_colors(specs, n_colors):
    bg = random.randrange(PALETTE_SIZE)
    fg_colors = random.sample([c for c in range(PALETTE_SIZE) if c != bg], n_colors)
    return bg, list(zip(fg_colors, specs))


def _gen_centered_output(S, bg, assigned):
    expected = np.full((S, S), bg, dtype=int)
    occupied = set()
    for color, spec in assigned:
        for r, c in spec["cells"]:
            if (r, c) in occupied:
                return None
            occupied.add((r, c))
            expected[r, c] = color
    return expected


def _gen_scattered_input(S, n_colors, bg, assigned):
    rows, cols = _gen_grid_shape(S, n_colors)
    grid = np.full((rows, cols), bg, dtype=int)
    random.shuffle(assigned)

    for color, spec in assigned:
        patch = np.full((spec["h"], spec["w"]), bg, dtype=int)
        for r, c in spec["rel"]:
            patch[r, c] = color
        try:
            r0, c0 = random_free_location(grid, patch, transparent=bg)
        except ValueError:
            return None
        paste(grid, patch, r0, c0, transparent=bg)
    return grid


def _gen_attempt() -> np.ndarray | None:
    S, n_colors = random.choice(STRUCTURE_CHOICES)
    specs = _gen_choose_specs(S, n_colors)
    if specs is None:
        return None

    bg, assigned = _gen_assign_colors(specs, n_colors)
    expected = _gen_centered_output(S, bg, assigned)
    if expected is None:
        return None

    grid = _gen_scattered_input(S, n_colors, bg, assigned)
    if grid is None:
        return None
    try:
        actual = main(grid)
    except AssertionError:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid c8cbb738 input")
