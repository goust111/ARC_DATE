from __future__ import annotations

import random

import numpy as np


BACKGROUND = 0
PALETTE = tuple(range(1, 10))
GRID_SIZES = (11, 13, 15, 17, 19)
PATTERN_SIZE = 3
MIN_RAY_EXTENSION = 3
MIN_DISTINCT_COLORS = 2
MAX_DISTINCT_COLORS = 4
MAX_GENERATION_ATTEMPTS = 400


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _find_center_pattern(grid: np.ndarray) -> tuple[int, int, np.ndarray]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    assert set(map(int, np.unique(g))).issubset(set(range(10))), "expected ARC colors"

    mask = g != BACKGROUND
    box = _bbox(mask)
    assert box is not None, "expected one central 3x3 pattern"
    r0, c0, r1, c1 = box
    assert (r1 - r0 + 1, c1 - c0 + 1) == (PATTERN_SIZE, PATTERN_SIZE), "expected a 3x3 pattern"
    assert np.all(mask[r0 : r1 + 1, c0 : c1 + 1]), "pattern must fill its 3x3 bbox"

    center_r = g.shape[0] // 2
    center_c = g.shape[1] // 2
    assert (r0, c0) == (center_r - 1, center_c - 1), "pattern must be centered"

    pattern = g[r0 : r1 + 1, c0 : c1 + 1].copy()
    distinct = {int(x) for x in pattern.flat}
    assert MIN_DISTINCT_COLORS <= len(distinct) <= MAX_DISTINCT_COLORS, "pattern must use 2 to 4 colors"
    corners = {
        int(pattern[0, 0]),
        int(pattern[0, 2]),
        int(pattern[2, 0]),
        int(pattern[2, 2]),
    }
    assert len(corners) >= 2, "corner colors must not all match"
    return r0, c0, pattern


def _edge_pad(pattern: np.ndarray) -> np.ndarray:
    return np.pad(np.asarray(pattern, dtype=int), ((1, 1), (1, 1)), mode="edge")


def _corner_specs(r0: int, c0: int, padded: np.ndarray) -> list[tuple[tuple[int, int], tuple[int, int], int]]:
    return [
        ((r0 - 1, c0 - 1), (-1, -1), int(padded[0, 0])),
        ((r0 - 1, c0 + 3), (-1, 1), int(padded[0, 4])),
        ((r0 + 3, c0 - 1), (1, -1), int(padded[4, 0])),
        ((r0 + 3, c0 + 3), (1, 1), int(padded[4, 4])),
    ]


def _apply_rule(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    r0, c0, pattern = _find_center_pattern(g)
    padded = _edge_pad(pattern)

    out = g.copy()
    out[r0 - 1 : r0 + 4, c0 - 1 : c0 + 4] = padded

    H, W = out.shape
    for (sr, sc), (dr, dc), color in _corner_specs(r0, c0, padded):
        r = sr + dr
        c = sc + dc
        while 0 <= r < H and 0 <= c < W:
            out[r, c] = color
            r += dr
            c += dc
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Expand the centered 3x3 pattern by edge padding, then emit four outward diagonal rays from the padded corners.

    Input structure:
    - The grid has a black background.
    - Exactly one centered 3x3 block is fully filled with non-black colors.
    - The 3x3 block uses 2 to 4 non-black colors, and its four corner pixels are not all the same color.

    Rule:
    - Copy the 3x3 block outward by one cell on every side using edge padding, producing a centered 5x5 block.
    - Keep that 5x5 padded block in the output.
    - Use the four corner pixels of the 5x5 block as ray emitters.
    - From the top-left, top-right, bottom-left, and bottom-right padded corners, paint diagonally outward until the grid boundary, using each corner's own color.

    Key insight:
    - The rays are not anchored to separate pre-existing objects; the edge-padding step creates the 5x5 corner emitters, and those corner colors determine the diagonal rays.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    assert max(g.shape) <= 30, "ARC grids must fit within 30x30"
    assert int(np.count_nonzero(g == BACKGROUND)) > 0, "expected black background"
    return _apply_rule(g)


def _sample_pattern() -> np.ndarray | None:
    distinct_count = random.randint(MIN_DISTINCT_COLORS, MAX_DISTINCT_COLORS)
    palette = random.sample(PALETTE, distinct_count)
    cells = palette[:] + [random.choice(palette) for _ in range(PATTERN_SIZE * PATTERN_SIZE - distinct_count)]
    random.shuffle(cells)
    pattern = np.array(cells, dtype=int).reshape(PATTERN_SIZE, PATTERN_SIZE)

    if len(set(map(int, pattern.flat))) != distinct_count:
        return None

    corners = [
        int(pattern[0, 0]),
        int(pattern[0, 2]),
        int(pattern[2, 0]),
        int(pattern[2, 2]),
    ]
    if len(set(corners)) < 2:
        return None

    nonuniform_sides = sum(
        len({int(x) for x in side}) > 1
        for side in (
            pattern[0, :],
            pattern[2, :],
            pattern[:, 0],
            pattern[:, 2],
        )
    )
    if nonuniform_sides < 2:
        return None

    if np.array_equal(pattern[0, :], pattern[2, :]) and np.array_equal(pattern[:, 0], pattern[:, 2]):
        return None

    return pattern


def _make_input() -> np.ndarray | None:
    size = random.choice(GRID_SIZES)
    grid = np.zeros((size, size), dtype=int)
    center = size // 2
    r0 = center - 1
    c0 = center - 1

    pattern = _sample_pattern()
    if pattern is None:
        return None

    padded_r0 = r0 - 1
    padded_c0 = c0 - 1
    if min(padded_r0, padded_c0, size - 1 - (padded_r0 + 4), size - 1 - (padded_c0 + 4)) < MIN_RAY_EXTENSION:
        return None

    grid[r0 : r0 + PATTERN_SIZE, c0 : c0 + PATTERN_SIZE] = pattern
    return grid


def _validate_semantics(inp: np.ndarray, out: np.ndarray) -> None:
    r0, c0, pattern = _find_center_pattern(inp)
    padded = _edge_pad(pattern)

    assert inp.shape == out.shape, "transform must preserve grid size"
    assert np.array_equal(out[r0 - 1 : r0 + 4, c0 - 1 : c0 + 4], padded), "expected centered 5x5 edge pad"

    expected = np.zeros_like(inp)
    expected[r0 - 1 : r0 + 4, c0 - 1 : c0 + 4] = padded
    H, W = inp.shape
    for (sr, sc), (dr, dc), color in _corner_specs(r0, c0, padded):
        ray_len = 0
        r = sr + dr
        c = sc + dc
        while 0 <= r < H and 0 <= c < W:
            expected[r, c] = color
            ray_len += 1
            r += dr
            c += dc
        assert ray_len >= MIN_RAY_EXTENSION, "each ray must extend at least three cells"

    assert np.array_equal(out, expected), "output must equal the semantic construction"


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        inp = _make_input()
        if inp is None:
            continue
        out = main(inp)
        _validate_semantics(inp, out)
        return inp
    raise RuntimeError("failed to generate a valid input for puzzle 8")
