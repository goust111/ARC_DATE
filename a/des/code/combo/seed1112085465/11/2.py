from __future__ import annotations

import math
import random
from collections import Counter

import numpy as np


BG = 8  # azure
PALETTE = (1, 2, 3, 4, 5, 6, 7, 9)

GRID_H_RANGE = (12, 24)
GRID_W_RANGE = (12, 24)
PAIR_COUNT_RANGE = (2, 5)
PERIOD_CHOICES = (1, 2, 3)
REDUCED_VECTORS = (
    (1, 1),
    (1, 2),
    (2, 1),
    (1, 3),
    (3, 1),
    (2, 3),
    (3, 2),
)
MARGIN = 1
PLACEMENT_ATTEMPTS = 120
MAX_GENERATION_ATTEMPTS = 400


def _dominant_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _seed_pairs(grid: np.ndarray, bg: int) -> list[tuple[int, tuple[int, int], tuple[int, int]]]:
    order: list[int] = []
    by_color: dict[int, list[tuple[int, int]]] = {}
    h, w = grid.shape

    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if color == bg:
                continue
            if color not in by_color:
                by_color[color] = []
                order.append(color)
            by_color[color].append((r, c))

    pairs = []
    for color in order:
        coords = sorted(by_color[color])
        assert len(coords) == 2, "each non-background color must appear exactly twice"
        p1, p2 = coords
        assert p1[0] != p2[0] and p1[1] != p2[1], "seed pixels must not share a row or column"
        pairs.append((color, p1, p2))
    return pairs


def _stripe_mask(p1: tuple[int, int], p2: tuple[int, int]) -> tuple[tuple[int, int, int, int], np.ndarray]:
    r1, c1 = p1
    r2, c2 = p2
    dr = r2 - r1
    dc = c2 - c1
    assert dr != 0 and dc != 0

    period = math.gcd(abs(dr), abs(dc))
    r0, r3 = sorted((r1, r2))
    c0, c3 = sorted((c1, c2))

    rr, cc = np.indices((r3 - r0 + 1, c3 - c0 + 1))
    rr += r0
    cc += c0

    if dc > 0:
        invariant = rr - cc
        anchor = r1 - c1
    else:
        invariant = rr + cc
        anchor = r1 + c1

    mask = ((invariant - anchor) % period) == 0
    return (r0, c0, r3, c3), mask


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill each seed-defined rectangle with diagonal stripes whose direction and period
    are determined by the relative offset between the two same-color seed pixels.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg = _dominant_color(g)
    out = g.copy()

    for color, p1, p2 in _seed_pairs(g, bg):
        (r0, c0, r1, c1), mask = _stripe_mask(p1, p2)
        patch = out[r0 : r1 + 1, c0 : c1 + 1]
        patch[mask] = int(color)

    return out


ALL_BRANCHES = tuple((period, sign) for period in PERIOD_CHOICES for sign in (-1, 1))


def _boxes_overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return not (ar1 < br0 or br1 < ar0 or ac1 < bc0 or bc1 < ac0)


def _choose_branches(rng: random.Random, pair_count: int) -> list[tuple[int, int]]:
    while True:
        branches = rng.sample(ALL_BRANCHES, pair_count)
        signs = {sign for _period, sign in branches}
        periods = {period for period, _sign in branches}
        if len(signs) == 2 and len(periods) >= 2:
            return branches


def _sample_bbox(
    rng: random.Random,
    h: int,
    w: int,
    period: int,
    sign: int,
    occupied: list[tuple[int, int, int, int]],
) -> tuple[tuple[int, int], tuple[int, int]] | None:
    candidates = list(REDUCED_VECTORS)
    rng.shuffle(candidates)

    for base_dr, base_dc in candidates:
        dr = period * base_dr
        dc_abs = period * base_dc
        box_h = dr + 1
        box_w = dc_abs + 1
        if box_h + 2 * MARGIN > h or box_w + 2 * MARGIN > w:
            continue

        for _ in range(PLACEMENT_ATTEMPTS):
            r0 = rng.randint(MARGIN, h - box_h - MARGIN)
            c0 = rng.randint(MARGIN, w - box_w - MARGIN)
            bbox = (r0, c0, r0 + dr, c0 + dc_abs)
            padded = (bbox[0] - 1, bbox[1] - 1, bbox[2] + 1, bbox[3] + 1)
            if any(_boxes_overlap(padded, other) for other in occupied):
                continue

            occupied.append(padded)
            if sign > 0:
                p1 = (r0, c0)
                p2 = (r0 + dr, c0 + dc_abs)
            else:
                p1 = (r0, c0 + dc_abs)
                p2 = (r0 + dr, c0)
            return p1, p2
    return None


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    h = rng.randint(*GRID_H_RANGE)
    w = rng.randint(*GRID_W_RANGE)
    pair_count = rng.randint(*PAIR_COUNT_RANGE)

    branches = _choose_branches(rng, pair_count)
    rng.shuffle(branches)
    colors = rng.sample(PALETTE, pair_count)

    occupied: list[tuple[int, int, int, int]] = []
    pairs: list[tuple[int, tuple[int, int], tuple[int, int]]] = []

    for color, (period, sign) in zip(colors, branches):
        placed = _sample_bbox(rng, h, w, period, sign, occupied)
        if placed is None:
            return None
        p1, p2 = placed
        pairs.append((color, p1, p2))

    grid = np.full((h, w), BG, dtype=int)
    for color, p1, p2 in pairs:
        grid[p1] = int(color)
        grid[p2] = int(color)

    counts = Counter(map(int, grid.ravel()))
    assert counts[BG] == h * w - 2 * pair_count
    assert all(counts[color] == 2 for color, _p1, _p2 in pairs)

    out = main(grid)
    if np.array_equal(out, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {MAX_GENERATION_ATTEMPTS} attempts")
