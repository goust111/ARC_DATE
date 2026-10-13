from __future__ import annotations

import random

import numpy as np


MARKER_COLOR = 7
BACKGROUND_COLOR = 0
PATTERN_COLORS = tuple(color for color in range(1, 10) if color != MARKER_COLOR)
PATTERN_SIZE_OPTIONS = ((3, 3), (3, 4), (4, 3), (4, 4))
MARGIN_RANGE = (2, 4)
REPEAT_RANGE = (2, 4)
GENERATION_ATTEMPTS = 300
PATTERN_ATTEMPTS = 200


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _invert_pattern(pattern: np.ndarray, foreground: int) -> np.ndarray:
    return np.where(pattern == BACKGROUND_COLOR, int(foreground), BACKGROUND_COLOR)


def _tile_checkerboard(pattern: np.ndarray, foreground: int, v_repeats: int, h_repeats: int) -> np.ndarray:
    tile_h, tile_w = map(int, pattern.shape)
    inverted = _invert_pattern(pattern, foreground)
    out = np.zeros((tile_h * int(v_repeats), tile_w * int(h_repeats)), dtype=int)
    for tile_r in range(int(v_repeats)):
        for tile_c in range(int(h_repeats)):
            tile = pattern if (tile_r + tile_c) % 2 == 0 else inverted
            r0 = tile_r * tile_h
            c0 = tile_c * tile_w
            out[r0 : r0 + tile_h, c0 : c0 + tile_w] = tile
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Read boundary marker counts as tiling dimensions and checkerboard-invert alternating copies of the center pattern.

    Input structure:
    - The grid uses black as background, one small central binary pattern made of black and one non-black foreground color, and orange boundary markers.
    - Orange markers appear only on the outer border and never inside the pattern region.
    - The pattern is the minimal bounding box containing every non-black, non-marker pixel, including any black holes inside that box.

    Rule:
    - Count orange markers on the top and bottom edges together to get the vertical repeat count V.
    - Count orange markers on the left and right edges together to get the horizontal repeat count H.
    - Extract the central binary pattern by cropping the minimal bounding box of the non-black, non-marker pixels.
    - Tile that crop into a V-by-H array, using the original pattern when (i+j) is even and its black/foreground inversion when (i+j) is odd.

    Key insight:
    - The boundary markers do not indicate where to draw; they encode the output's row and column repeat counts, while the tile parity controls where inversion is applied.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    v_repeats = int(np.count_nonzero(g[0] == MARKER_COLOR) + np.count_nonzero(g[-1] == MARKER_COLOR))
    h_repeats = int(np.count_nonzero(g[:, 0] == MARKER_COLOR) + np.count_nonzero(g[:, -1] == MARKER_COLOR))
    assert 1 < v_repeats <= 4 and 1 < h_repeats <= 4, "marker counts must define valid tiling dimensions"

    pattern_mask = (g != BACKGROUND_COLOR) & (g != MARKER_COLOR)
    pattern_box = _bbox(pattern_mask)
    assert pattern_box is not None, "expected a central foreground pattern"
    r0, c0, r1, c1 = pattern_box
    pattern = g[r0 : r1 + 1, c0 : c1 + 1].copy()

    foregrounds = sorted(set(map(int, pattern[pattern != BACKGROUND_COLOR].flatten())))
    assert len(foregrounds) == 1, "expected exactly one pattern foreground color"
    foreground = int(foregrounds[0])

    return _tile_checkerboard(pattern, foreground, v_repeats, h_repeats)


def _sample_pattern_mask(height: int, width: int) -> np.ndarray | None:
    total = int(height) * int(width)
    for _ in range(PATTERN_ATTEMPTS):
        density = random.uniform(0.35, 0.65)
        mask = np.random.random((int(height), int(width))) < density
        count = int(mask.sum())
        if not (2 <= count <= total - 2):
            continue
        if not mask.any(axis=0).all() or not mask.any(axis=1).all():
            continue
        if not (~mask).any(axis=0).all() or not (~mask).any(axis=1).all():
            continue
        return mask
    return None


def _sample_edge_split(total: int, capacity_per_edge: int) -> tuple[int, int]:
    options = [(a, total - a) for a in range(total + 1) if a <= capacity_per_edge and total - a <= capacity_per_edge]
    assert options, "expected at least one feasible edge split"
    weighted: list[tuple[int, int]] = []
    for split in options:
        a, b = split
        weight = 3 if 0 < a < total else 1
        weighted.extend([split] * weight)
    return random.choice(weighted)


def _place_boundary_markers(grid: np.ndarray, v_repeats: int, h_repeats: int) -> None:
    height, width = map(int, grid.shape)

    top_count, bottom_count = _sample_edge_split(int(v_repeats), width - 2)
    left_count, right_count = _sample_edge_split(int(h_repeats), height - 2)

    if top_count:
        top_cols = random.sample(range(1, width - 1), top_count)
        grid[0, top_cols] = MARKER_COLOR
    if bottom_count:
        bottom_cols = random.sample(range(1, width - 1), bottom_count)
        grid[height - 1, bottom_cols] = MARKER_COLOR
    if left_count:
        left_rows = random.sample(range(1, height - 1), left_count)
        grid[left_rows, 0] = MARKER_COLOR
    if right_count:
        right_rows = random.sample(range(1, height - 1), right_count)
        grid[right_rows, width - 1] = MARKER_COLOR


def _gen_attempt() -> np.ndarray | None:
    pattern_h, pattern_w = random.choice(PATTERN_SIZE_OPTIONS)
    pattern_mask = _sample_pattern_mask(pattern_h, pattern_w)
    if pattern_mask is None:
        return None

    foreground = random.choice(PATTERN_COLORS)
    pattern = np.zeros((pattern_h, pattern_w), dtype=int)
    pattern[pattern_mask] = int(foreground)

    v_repeats = random.randint(*REPEAT_RANGE)
    h_repeats = random.randint(*REPEAT_RANGE)

    margin_r = random.randint(*MARGIN_RANGE)
    margin_c = random.randint(*MARGIN_RANGE)
    height = pattern_h + 2 * margin_r
    width = pattern_w + 2 * margin_c
    if max(height, width) > 30:
        return None

    expected = _tile_checkerboard(pattern, foreground, v_repeats, h_repeats)
    if max(expected.shape) > 30:
        return None

    grid = np.zeros((height, width), dtype=int)
    r0 = margin_r
    c0 = margin_c
    grid[r0 : r0 + pattern_h, c0 : c0 + pattern_w] = pattern
    _place_boundary_markers(grid, v_repeats, h_repeats)

    if np.any(grid[1:-1, 1:-1] == MARKER_COLOR):
        return None
    if not np.array_equal(main(grid), expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 6")
