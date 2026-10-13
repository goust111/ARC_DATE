from __future__ import annotations

import numpy as np


BLACK = 0
GRAY = 5
SEED_COLORS = (1, 2, 3, 4, 6, 7, 8, 9)
GEN_MAX_ATTEMPTS = 600
_RNG = np.random.default_rng()


def _extract_seeds(grid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    coords = np.argwhere(grid != BLACK)
    if coords.size == 0:
        raise ValueError("expected at least one non-black seed pixel")
    colors = grid[coords[:, 0], coords[:, 1]].astype(int)
    return coords.astype(int), colors


def _distance_stack(grid: np.ndarray, metric: str = "chebyshev") -> tuple[np.ndarray, np.ndarray]:
    g = np.asarray(grid, dtype=int)
    seeds, colors = _extract_seeds(g)
    rows, cols = np.indices(g.shape)
    dr = np.abs(rows[:, :, None] - seeds[None, None, :, 0])
    dc = np.abs(cols[:, :, None] - seeds[None, None, :, 1])
    if metric == "chebyshev":
        dists = np.maximum(dr, dc)
    elif metric == "manhattan":
        dists = dr + dc
    else:
        raise ValueError(f"unknown metric: {metric}")
    return dists.astype(int), colors


def _solve_with_metric(grid: np.ndarray, metric: str = "chebyshev") -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    dists, colors = _distance_stack(g, metric=metric)
    min_d = dists.min(axis=2)
    winners = dists == min_d[:, :, None]
    tie = winners.sum(axis=2) > 1
    nearest = dists.argmin(axis=2)

    out = np.full_like(g, BLACK)
    paint = (~tie) & ((min_d % 2) == 0)
    out[paint] = colors[nearest[paint]]
    out[tie] = GRAY
    out[g != BLACK] = g[g != BLACK]
    return out


def _solve_slow(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    seeds, colors = _extract_seeds(g)
    out = np.full_like(g, BLACK)
    h, w = g.shape

    for r in range(h):
        for c in range(w):
            best = None
            winners: list[int] = []
            for idx, (sr, sc) in enumerate(seeds):
                dist = max(abs(r - int(sr)), abs(c - int(sc)))
                if best is None or dist < best:
                    best = dist
                    winners = [idx]
                elif dist == best:
                    winners.append(idx)

            if len(winners) > 1:
                out[r, c] = GRAY
            elif best % 2 == 0:
                out[r, c] = int(colors[winners[0]])

    out[g != BLACK] = g[g != BLACK]
    return out


def _solve_ignore_parity(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    dists, colors = _distance_stack(g, metric="chebyshev")
    min_d = dists.min(axis=2)
    winners = dists == min_d[:, :, None]
    tie = winners.sum(axis=2) > 1
    nearest = dists.argmin(axis=2)

    out = np.full_like(g, BLACK)
    unique = ~tie
    out[unique] = colors[nearest[unique]]
    out[tie] = GRAY
    out[g != BLACK] = g[g != BLACK]
    return out


def _solve_ignore_ties(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    dists, colors = _distance_stack(g, metric="chebyshev")
    min_d = dists.min(axis=2)
    nearest = dists.argmin(axis=2)

    out = np.full_like(g, BLACK)
    paint = (min_d % 2) == 0
    out[paint] = colors[nearest[paint]]
    out[g != BLACK] = g[g != BLACK]
    return out


def _solve_overlay_even_rings(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    dists, colors = _distance_stack(g, metric="chebyshev")
    even = (dists % 2) == 0
    count = even.sum(axis=2)
    first = even.argmax(axis=2)

    out = np.full_like(g, BLACK)
    one = count == 1
    many = count > 1
    out[one] = colors[first[one]]
    out[many] = GRAY
    out[g != BLACK] = g[g != BLACK]
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """Expand each seed into even Chebyshev square-wave territory, with equidistant fronts turning grey.

    Input structure
    - A black background contains several isolated single-pixel colored seeds.
    - Seed colors are distinct and there are no other foreground objects.

    Rule
    - For every background cell, compute its Chebyshev distance to every seed and keep the minimum.
    - If exactly one seed attains that minimum and the minimum distance is even, color the cell with that seed.
    - If exactly one seed attains that minimum and the minimum distance is odd, leave the cell black.
    - If two or more seeds tie for the minimum distance, color the cell grey regardless of parity.
    - Seed pixels themselves stay unchanged.

    Key insight
    - The visible pattern is not just concentric square rings: each ring only appears inside its nearest-seed Voronoi region, and equal-distance collision fronts override everything with grey.
    """
    return _solve_with_metric(grid, metric="chebyshev")


def _random_coord(
    rng: np.random.Generator,
    h: int,
    w: int,
    used: list[tuple[int, int]],
    min_sep: int,
    *,
    row_forbidden: set[int] | None = None,
    col_forbidden: set[int] | None = None,
) -> tuple[int, int] | None:
    row_forbidden = set() if row_forbidden is None else set(row_forbidden)
    col_forbidden = set() if col_forbidden is None else set(col_forbidden)
    for _ in range(300):
        r = int(rng.integers(0, h))
        c = int(rng.integers(0, w))
        if r in row_forbidden or c in col_forbidden:
            continue
        if all(max(abs(r - rr), abs(c - cc)) >= min_sep for rr, cc in used):
            return r, c
    return None


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    h = int(rng.integers(14, 27))
    w = int(rng.integers(14, 27))
    min_sep = 4 if min(h, w) >= 18 else 3
    n_seeds = int(rng.integers(3, min(6, len(SEED_COLORS)) + 1))
    coords: list[tuple[int, int]] = []

    if rng.random() < 0.5:
        if w < 8:
            return None
        row = int(rng.integers(2, h - 2))
        gap = int(rng.integers(4, min(9, w - 2)))
        c1 = int(rng.integers(1, w - gap))
        coords.extend([(row, c1), (row, c1 + gap)])
        third = _random_coord(rng, h, w, coords, min_sep, row_forbidden={row})
    else:
        if h < 8:
            return None
        col = int(rng.integers(2, w - 2))
        gap = int(rng.integers(4, min(9, h - 2)))
        r1 = int(rng.integers(1, h - gap))
        coords.extend([(r1, col), (r1 + gap, col)])
        third = _random_coord(rng, h, w, coords, min_sep, col_forbidden={col})

    if third is None:
        return None
    coords.append(third)

    while len(coords) < n_seeds:
        extra = _random_coord(rng, h, w, coords, min_sep)
        if extra is None:
            return None
        coords.append(extra)

    colors = rng.choice(np.array(SEED_COLORS, dtype=int), size=len(coords), replace=False)
    grid = np.full((h, w), BLACK, dtype=int)
    for color, (r, c) in zip(colors, coords):
        grid[r, c] = int(color)

    return grid if _case_is_valid(grid) else None


def _case_is_valid(grid: np.ndarray) -> bool:
    g = np.asarray(grid, dtype=int)
    if g.ndim != 2 or max(g.shape) > 30:
        return False

    seed_coords, seed_colors = _extract_seeds(g)
    if seed_coords.shape[0] < 3 or np.unique(seed_colors).size != seed_coords.shape[0]:
        return False

    out = main(g)
    if np.array_equal(out, g):
        return False
    if not np.array_equal(out, _solve_slow(g)):
        return False

    dists, _ = _distance_stack(g, metric="chebyshev")
    min_d = dists.min(axis=2)
    winners = dists == min_d[:, :, None]
    tie = winners.sum(axis=2) > 1
    bg = g == BLACK

    added_color = bg & (~tie) & ((min_d % 2) == 0)
    odd_territory = bg & (~tie) & ((min_d % 2) == 1)
    gray_front = bg & tie

    if not added_color.any():
        return False
    if not odd_territory.any():
        return False
    if not gray_front.any():
        return False

    propagated = np.unique(out[added_color])
    if propagated.size < 2:
        return False

    gray_parity = min_d[gray_front] % 2
    if not gray_parity.size or not gray_parity.any() or not (~gray_parity.astype(bool)).any():
        return False

    wrong_outputs = (
        _solve_with_metric(g, metric="manhattan"),
        _solve_ignore_parity(g),
        _solve_ignore_ties(g),
        _solve_overlay_even_rings(g),
    )
    if any(np.array_equal(out, wrong) for wrong in wrong_outputs):
        return False

    return True


def generate_input() -> np.ndarray:
    for _ in range(GEN_MAX_ATTEMPTS):
        grid = _gen_attempt(_RNG)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid square-wave territory input")
