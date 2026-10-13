from __future__ import annotations

import random

import numpy as np


BLACK = 0
GREY = 5
PALETTE = [1, 2, 3, 4, 6, 7, 8, 9]
MAX_RETRIES = 400


def _chebyshev(a: tuple[int, int], b: tuple[int, int]) -> int:
    return max(abs(a[0] - b[0]), abs(a[1] - b[1]))


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= r < shape[0] and 0 <= c < shape[1]


def _draw_square_outline(out: np.ndarray, center: tuple[int, int], radius: int, color: int) -> None:
    cr, cc = center
    top = cr - radius
    bottom = cr + radius
    left = cc - radius
    right = cc + radius

    for c in range(left, right + 1):
        if _in_bounds(out.shape, top, c):
            out[top, c] = color
        if _in_bounds(out.shape, bottom, c):
            out[bottom, c] = color
    for r in range(top + 1, bottom):
        if _in_bounds(out.shape, r, left):
            out[r, left] = color
        if _in_bounds(out.shape, r, right):
            out[r, right] = color


def _parse_seeds(grid: np.ndarray) -> tuple[tuple[int, int], int]:
    seeds = [tuple(map(int, rc)) for rc in np.argwhere(grid == GREY)]
    assert len(seeds) == 3, "expected exactly three grey seed pixels"

    for i, center in enumerate(seeds):
        others = [seeds[j] for j in range(3) if j != i]
        if 2 * center[0] != others[0][0] + others[1][0]:
            continue
        if 2 * center[1] != others[0][1] + others[1][1]:
            continue
        dr1 = others[0][0] - center[0]
        dc1 = others[0][1] - center[1]
        dr2 = others[1][0] - center[0]
        dc2 = others[1][1] - center[1]
        s1 = max(abs(dr1), abs(dc1))
        s2 = max(abs(dr2), abs(dc2))
        if s1 == 0 or s1 != s2:
            continue
        if (dr1, dc1) != (-dr2, -dc2):
            continue
        if abs(dr1) != s1 or abs(dc1) != s1:
            continue
        return center, s1

    raise AssertionError("grey seeds must form an equally spaced 45-degree triple")


def _parse_strip(grid: np.ndarray, center: tuple[int, int]) -> list[int]:
    mask = (grid != BLACK) & (grid != GREY)
    coords = np.argwhere(mask)
    assert coords.size > 0, "expected one horizontal color strip"
    rows = {int(r) for r, _ in coords}
    assert len(rows) == 1, "strip must be horizontal"
    row = rows.pop()
    cols = sorted(int(c) for _, c in coords)
    assert cols == list(range(cols[0], cols[-1] + 1)), "strip must occupy one contiguous run"

    colors = [int(v) for v in grid[row, cols[0] : cols[-1] + 1]]
    left_end = (row, cols[0])
    right_end = (row, cols[-1])
    d_left = _chebyshev(center, left_end)
    d_right = _chebyshev(center, right_end)
    assert d_left != d_right, "strip must have a unique nearer end"
    if d_left < d_right:
        return colors
    return colors[::-1]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Decode a horizontal color strip into concentric square outlines centered by three grey diagonal seeds.

    Input structure:
    - The grid has a black background, exactly three grey pixels, and one separate horizontal strip of non-black non-grey colors.
    - The three grey pixels lie on one 45-degree diagonal with equal spacing, so their middle pixel is the geometric center.
    - The color strip is away from the seed trio and may repeat colors.

    Rule:
    - Find the middle grey seed and let s be the Chebyshev distance from it to either outer grey seed.
    - Read the strip starting from the end closer to the center; its i-th color controls the i-th ring.
    - Erase the strip to black, then draw the outline of the square with Chebyshev radius i*s around the center in that color, clipping anything that leaves the grid. Keep the center grey seed unchanged; the two outer grey seeds are covered by the first ring.

    Key insight:
    - The three grey pixels encode geometry while the strip encodes color order; the nearer strip end is the innermost ring and the seed spacing is the ring step.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    center, step = _parse_seeds(g)
    strip_colors = _parse_strip(g, center)

    out = g.copy()
    out[(out != BLACK) & (out != GREY)] = BLACK
    for i, color in enumerate(strip_colors, start=1):
        _draw_square_outline(out, center, i * step, color)
    out[center] = GREY
    return out


def _sample_colors(k: int) -> list[int]:
    if k >= 3 and random.random() < 0.35:
        base = random.sample(PALETTE, random.randint(2, min(4, len(PALETTE))))
        colors = [random.choice(base) for _ in range(k)]
        if len(set(colors)) == k:
            colors[random.randrange(k)] = random.choice(base)
        return colors
    return random.sample(PALETTE, k)


def _seed_offsets(step: int, diag_kind: int) -> list[tuple[int, int]]:
    if diag_kind == 1:
        return [(-step, -step), (0, 0), (step, step)]
    return [(-step, step), (0, 0), (step, -step)]


def _segment_is_far(
    row: int,
    c0: int,
    k: int,
    seeds: list[tuple[int, int]],
    bbox: tuple[int, int, int, int],
    pad: int,
) -> bool:
    r0, c_left, r1, c_right = bbox
    for c in range(c0, c0 + k):
        if any(_chebyshev((row, c), seed) <= 2 for seed in seeds):
            return False
        if r0 - pad <= row <= r1 + pad and c_left - pad <= c <= c_right + pad:
            return False
    return True


def _choose_center(h: int, w: int, step: int, max_radius: int, clip_mode: bool) -> tuple[int, int] | None:
    seeds_margin = step
    candidates = []
    for r in range(seeds_margin, h - seeds_margin):
        for c in range(seeds_margin, w - seeds_margin):
            edge_dist = min(r, h - 1 - r, c, w - 1 - c)
            if clip_mode and edge_dist < max_radius:
                candidates.append((r, c))
            if (not clip_mode) and edge_dist >= max_radius:
                candidates.append((r, c))
    if not candidates:
        return None
    return random.choice(candidates)


def _place_strip(
    h: int,
    w: int,
    k: int,
    center: tuple[int, int],
    seeds: list[tuple[int, int]],
    bbox: tuple[int, int, int, int],
    nearer_left: bool,
) -> tuple[int, int] | None:
    candidates = []
    for row in range(h):
        for c0 in range(w - k + 1):
            if not _segment_is_far(row, c0, k, seeds, bbox, pad=1):
                continue
            left = (row, c0)
            right = (row, c0 + k - 1)
            d_left = _chebyshev(center, left)
            d_right = _chebyshev(center, right)
            if d_left == d_right:
                continue
            if (d_left < d_right) == nearer_left:
                candidates.append((row, c0))
    if not candidates:
        return None
    return random.choice(candidates)


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(14, 30)
    w = random.randint(14, 30)
    step = random.randint(1, 4)
    k = random.randint(2, min(6, len(PALETTE)))
    max_radius = step * k
    if min(h, w) < 2 * step + 1:
        return None

    clip_mode = random.random() < 0.5
    center = _choose_center(h, w, step, max_radius, clip_mode)
    if center is None:
        return None

    diag_kind = random.choice([1, -1])
    seeds = [(center[0] + dr, center[1] + dc) for dr, dc in _seed_offsets(step, diag_kind)]
    if not all(_in_bounds((h, w), r, c) for r, c in seeds):
        return None

    bbox = (
        max(0, center[0] - max_radius),
        max(0, center[1] - max_radius),
        min(h - 1, center[0] + max_radius),
        min(w - 1, center[1] + max_radius),
    )
    nearer_left = random.choice([True, False])
    strip_pos = _place_strip(h, w, k, center, seeds, bbox, nearer_left)
    if strip_pos is None:
        return None

    grid = np.zeros((h, w), dtype=int)
    for r, c in seeds:
        grid[r, c] = GREY

    colors = _sample_colors(k)
    row, c0 = strip_pos
    grid[row, c0 : c0 + k] = np.array(colors, dtype=int)

    solved = main(grid)
    if solved.shape != grid.shape:
        return None
    if np.array_equal(solved, grid):
        return None
    if int((solved != BLACK).sum()) <= int((grid != BLACK).sum()):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate valid input after {MAX_RETRIES} attempts")
