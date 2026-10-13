from __future__ import annotations

import random
from collections import Counter

import numpy as np


BG = 0
BLUE = 1
RED = 2
GREEN = 3

MAX_GENERATION_ATTEMPTS = 300
REPEAT_CHOICES = {
    3: (2, 3, 4),
    4: (2, 3),
    5: (2,),
}


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _shortest_period(bits: np.ndarray) -> int:
    n = int(bits.size)
    for p in range(1, n + 1):
        if n % p != 0:
            continue
        base = bits[:p]
        if np.array_equal(np.tile(base, n // p), bits):
            return p
    raise AssertionError("periodic vector expected")


def _vector_to_output(vector: np.ndarray) -> np.ndarray:
    mask = np.outer(vector.astype(bool), vector.astype(bool))
    fractal = np.kron(mask, mask)
    return np.where(fractal, RED, BG).astype(int)


def _extract_vector(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    assert max(g.shape) <= 30, "ARC grids must fit within 30x30"

    bg = _get_bg(g)
    assert bg == BG, "background must be black"

    red_box = _bbox(g == RED)
    assert red_box is not None, "expected a red frame"
    r0, c0, r1, c1 = red_box
    outer_h = r1 - r0 + 1
    outer_w = c1 - c0 + 1
    assert outer_h >= 5 and outer_w >= 8, "frame too small"

    expected_red = np.zeros_like(g, dtype=bool)
    expected_red[r0, c0 : c1 + 1] = True
    expected_red[r1, c0 : c1 + 1] = True
    expected_red[r0 : r1 + 1, c0] = True
    expected_red[r0 : r1 + 1, c1] = True
    assert np.array_equal(g == RED, expected_red), "red pixels must be one rectangular border"

    inner = g[r0 + 1 : r1, c0 + 1 : c1]
    assert inner.shape[0] >= 3 and inner.shape[1] >= 3, "frame must have an interior"

    blue_rows = [idx for idx in range(inner.shape[0]) if np.all(inner[idx] == BLUE)]
    assert len(blue_rows) == 2, "expected exactly two horizontal blue marker rows"
    top_bar, bottom_bar = blue_rows
    assert top_bar + 1 < bottom_bar, "marker rows must leave a middle region"

    expected_blue = np.zeros_like(g, dtype=bool)
    for row in blue_rows:
        expected_blue[r0 + 1 + row, c0 + 1 : c1] = True
    assert np.array_equal(g == BLUE, expected_blue), "blue pixels must be exactly the two full marker rows"

    if top_bar > 0:
        assert np.all(inner[:top_bar] == BG), "rows above the top marker must be blank"
    if bottom_bar + 1 < inner.shape[0]:
        assert np.all(inner[bottom_bar + 1 :] == BG), "rows below the bottom marker must be blank"

    middle = inner[top_bar + 1 : bottom_bar]
    assert middle.shape[0] >= 1

    bits = np.zeros(inner.shape[1], dtype=int)
    for col in range(inner.shape[1]):
        column = middle[:, col]
        if np.all(column == GREEN):
            bits[col] = 1
        else:
            assert np.all(column == BG), "middle columns must be uniformly green or black"

    green_mask = np.zeros_like(g, dtype=bool)
    for col, bit in enumerate(bits):
        if bit:
            green_mask[r0 + 2 + top_bar : r0 + 1 + bottom_bar, c0 + 1 + col] = True
    assert np.array_equal(g == GREEN, green_mask), "green pixels must fill complete middle columns only"

    period = _shortest_period(bits)
    assert period in (3, 4, 5), "period must be 3, 4, or 5"
    vector = bits[:period].copy()
    assert vector.sum() >= 2 and vector.sum() <= period - 1, "period must contain at least two green and one black column"
    assert _shortest_period(vector) == period, "extracted period must be minimal"
    return vector


def main(grid: np.ndarray) -> np.ndarray:
    """
    Decode the periodic column mask inside the framed stripe gadget and turn it into a self-similar Kronecker fractal seed.

    Input structure:
    - The grid has black background, one red one-cell-thick rectangular frame, two full-width blue marker rows inside that frame, and green stripe columns only in the region between those blue rows.
    - Outside the red frame everything is black, and inside the frame every non-marker row outside the middle stripe region is also black.
    - Each middle-region column is either entirely green or entirely black, and the full column sequence is a repetition of a shorter binary period of length 3, 4, or 5.

    Rule:
    - Locate the red rectangular border and the two blue horizontal marker rows.
    - Read the middle-region columns from left to right, mapping full green columns to 1 and full black columns to 0, then extract the shortest repeating period vector V.
    - Form the square binary mask M by the self-outer-product rule M[r, c] = V[r] AND V[c].
    - Output the Kronecker self-product of M with itself as a P^2 x P^2 grid, using red for active cells and black for inactive cells.

    Key insight:
    - The only real payload is the shortest 1D periodic stripe generator hidden between the blue bars; both the intermediate 2D mask and the final self-similar image are deterministically derived from that same vector.
    """
    vector = _extract_vector(grid)
    return _vector_to_output(vector)


def _valid_base_patterns(period: int) -> list[tuple[int, ...]]:
    patterns: list[tuple[int, ...]] = []
    for mask in range(1 << period):
        bits = np.array([(mask >> shift) & 1 for shift in range(period)], dtype=int)
        if bits.sum() < 2 or bits.sum() > period - 1:
            continue
        if _shortest_period(bits) != period:
            continue
        patterns.append(tuple(int(x) for x in bits))
    return patterns


_PATTERNS_BY_PERIOD = {p: _valid_base_patterns(p) for p in (3, 4, 5)}
_PATTERN_POOLS = {p: [] for p in (3, 4, 5)}
_PERIOD_POOL: list[int] = []


def _take_pattern() -> np.ndarray:
    global _PERIOD_POOL

    if not _PERIOD_POOL:
        _PERIOD_POOL = [3, 4, 5]
        random.shuffle(_PERIOD_POOL)
    period = _PERIOD_POOL.pop()

    if not _PATTERN_POOLS[period]:
        _PATTERN_POOLS[period] = list(_PATTERNS_BY_PERIOD[period])
        random.shuffle(_PATTERN_POOLS[period])

    return np.array(_PATTERN_POOLS[period].pop(), dtype=int)


def _bar_pairs(inner_h: int) -> list[tuple[int, int]]:
    pairs = []
    for top in range(inner_h):
        for bottom in range(top + 2, inner_h):
            pairs.append((top, bottom))
    return pairs


def _gen_attempt() -> np.ndarray | None:
    vector = _take_pattern()
    period = int(vector.size)

    repeat = random.choice(REPEAT_CHOICES[period])
    full_bits = np.tile(vector, repeat)
    inner_w = int(full_bits.size)

    inner_h = random.choice((4, 5, 6))
    pairs = _bar_pairs(inner_h)
    if not pairs:
        return None
    top_bar, bottom_bar = random.choice(pairs)

    outer_h = inner_h + 2
    outer_w = inner_w + 2

    pad_top = random.randint(1, 6)
    pad_bottom = random.randint(1, 6)
    pad_left = random.randint(1, 6)
    pad_right = random.randint(1, 6)
    H = outer_h + pad_top + pad_bottom
    W = outer_w + pad_left + pad_right
    if max(H, W) > 30:
        return None

    grid = np.full((H, W), BG, dtype=int)
    r0 = pad_top
    c0 = pad_left
    r1 = r0 + outer_h - 1
    c1 = c0 + outer_w - 1

    grid[r0, c0 : c1 + 1] = RED
    grid[r1, c0 : c1 + 1] = RED
    grid[r0 : r1 + 1, c0] = RED
    grid[r0 : r1 + 1, c1] = RED

    inner_r0 = r0 + 1
    inner_c0 = c0 + 1
    grid[inner_r0 + top_bar, inner_c0 : inner_c0 + inner_w] = BLUE
    grid[inner_r0 + bottom_bar, inner_c0 : inner_c0 + inner_w] = BLUE

    if bottom_bar - top_bar > 1:
        middle_rows = slice(inner_r0 + top_bar + 1, inner_r0 + bottom_bar)
        for col, bit in enumerate(full_bits):
            if bit:
                grid[middle_rows, inner_c0 + col] = GREEN

    expected = _vector_to_output(vector)
    solved = main(grid)
    if not np.array_equal(solved, expected):
        return None

    extracted = _extract_vector(grid)
    if not np.array_equal(extracted, vector):
        return None

    if solved.shape != (period * period, period * period):
        return None
    if not (np.any(solved == RED) and np.any(solved == BG)):
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid sample for puzzle 7")
