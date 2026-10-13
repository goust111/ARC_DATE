from __future__ import annotations

import numpy as np

from arc_utils import Color
import random


_CROSS_OFFSETS = ((0, 0), (-1, 0), (1, 0), (0, -1), (0, 1))


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor every black 5-cell plus whose center and four orthogonal neighbors are black to green.

    Input structure:
    - The grid contains only black and teal pixels.
    - Some black pixels are centers of a full black orthogonal cross.
    - Teal pixels are fixed blockers and are never recolored.

    Rule:
    - Find every black pixel whose north, south, west, and east neighbors are also black.
    - For each such center, recolor the center and its four orthogonal neighbors green.
    - Overlapping pluses simply merge into one green mask.

    Key insight:
    - The longer marked shapes are not separate templates; they are unions of adjacent black pluses.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    assert set(map(int, np.unique(g))) == {Color.BLACK, Color.TEAL}, "expected only black and teal pixels"

    height, width = map(int, g.shape)
    assert height >= 3 and width >= 3, "grid must be large enough to contain a plus"

    centers: list[tuple[int, int]] = []
    for r in range(1, height - 1):
        for c in range(1, width - 1):
            if all(g[r + dr, c + dc] == Color.BLACK for dr, dc in _CROSS_OFFSETS):
                centers.append((r, c))
    assert centers, "expected at least one black plus center"

    out = g.copy()
    for r, c in centers:
        for dr, dc in _CROSS_OFFSETS:
            out[r + dr, c + dc] = Color.GREEN
    return out

# --------------- generator (o5x) ---------------
# Canvas geometry.
_GEN_SIZE = 12
_GEN_INTERIOR_MIN, _GEN_INTERIOR_MAX = 1, _GEN_SIZE - 2
_GEN_PLUS_OFFSETS = ((0, 0), (-1, 0), (1, 0), (0, -1), (0, 1))

# Center grouping.
_GEN_CENTER_COUNT_MIN, _GEN_CENTER_COUNT_MAX = 2, 5
_GEN_GROUP_COUNT_MIN, _GEN_GROUP_COUNT_MAX = 1, 3
_GEN_GROUP_SIZE_MIN, _GEN_GROUP_SIZE_MAX = 1, 2
_GEN_PAIR_L1_DISTANCES = (1, 2)
_GEN_INTER_GROUP_MIN_L1 = 5

# Random scatter density.
_GEN_ACTIVE_COUNT_MIN, _GEN_ACTIVE_COUNT_MAX = 58, 74

# Rejection budget.
_GEN_MAX_ATTEMPTS = 300

_GEN_INTERIOR_CELLS = [
    (r, c)
    for r in range(_GEN_INTERIOR_MIN, _GEN_INTERIOR_MAX + 1)
    for c in range(_GEN_INTERIOR_MIN, _GEN_INTERIOR_MAX + 1)
]


def _gen_plus_cells(centers) -> set[tuple[int, int]]:
    cells = set()
    for r, c in centers:
        for dr, dc in _GEN_PLUS_OFFSETS:
            cells.add((r + dr, c + dc))
    return cells


def _gen_detect_centers(active: np.ndarray) -> set[tuple[int, int]]:
    """Return all interior pixels whose 5-cell plus is active."""
    cross = active[1:-1, 1:-1] & active[:-2, 1:-1] & active[2:, 1:-1]
    cross &= active[1:-1, :-2] & active[1:-1, 2:]
    rows, cols = np.nonzero(cross)
    return {(int(r) + 1, int(c) + 1) for r, c in zip(rows, cols)}


def _gen_group_variants(size: int) -> list[tuple[tuple[int, int], ...]]:
    """Enumerate single-center groups or valid adjacent two-center groups."""
    if size == _GEN_GROUP_SIZE_MIN:
        return [(coord,) for coord in _GEN_INTERIOR_CELLS]

    variants = []
    for r, c in _GEN_INTERIOR_CELLS:
        for dr in range(-_GEN_GROUP_SIZE_MAX, _GEN_GROUP_SIZE_MAX + 1):
            for dc in range(-_GEN_GROUP_SIZE_MAX, _GEN_GROUP_SIZE_MAX + 1):
                if abs(dr) + abs(dc) not in _GEN_PAIR_L1_DISTANCES:
                    continue
                r2, c2 = r + dr, c + dc
                if not (_GEN_INTERIOR_MIN <= r2 <= _GEN_INTERIOR_MAX and _GEN_INTERIOR_MIN <= c2 <= _GEN_INTERIOR_MAX):
                    continue
                pair = tuple(sorted(((r, c), (r2, c2))))
                active = np.zeros((_GEN_SIZE, _GEN_SIZE), dtype=bool)
                for rr, cc in _gen_plus_cells(list(pair)):
                    active[rr, cc] = True
                if _gen_detect_centers(active) == set(pair):
                    variants.append(pair)
    return sorted(set(variants))


_GEN_GROUP_VARIANTS_BY_SIZE = {size: _gen_group_variants(size) for size in (_GEN_GROUP_SIZE_MIN, _GEN_GROUP_SIZE_MAX)}


def _gen_choose_centers() -> list[tuple[int, int]] | None:
    """Sample center groups while preserving intended count and spacing."""
    n_centers = random.randint(_GEN_CENTER_COUNT_MIN, _GEN_CENTER_COUNT_MAX)
    group_counts = [
        g for g in range(_GEN_GROUP_COUNT_MIN, _GEN_GROUP_COUNT_MAX + 1)
        if g <= n_centers <= _GEN_GROUP_SIZE_MAX * g
    ]
    n_groups = random.choice(group_counts)
    sizes = [_GEN_GROUP_SIZE_MAX] * (n_centers - n_groups) + [_GEN_GROUP_SIZE_MIN] * (_GEN_GROUP_SIZE_MAX * n_groups - n_centers)
    random.shuffle(sizes)

    centers: list[tuple[int, int]] = []
    for size in sizes:
        candidates = []
        for variant in _GEN_GROUP_VARIANTS_BY_SIZE[size]:
            if all(
                abs(r - er) + abs(c - ec) >= _GEN_INTER_GROUP_MIN_L1
                for r, c in variant
                for er, ec in centers
            ):
                candidates.append(variant)
        if not candidates:
            return None
        chosen = random.choice(candidates)
        centers.extend(chosen)
    return centers


def _gen_can_add_pixel(active: np.ndarray, intended: set[tuple[int, int]], r: int, c: int) -> bool:
    active[r, c] = True
    try:
        for cr, cc in ((r, c), (r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
            if not (_GEN_INTERIOR_MIN <= cr <= _GEN_INTERIOR_MAX and _GEN_INTERIOR_MIN <= cc <= _GEN_INTERIOR_MAX):
                continue
            if (cr, cc) in intended:
                continue
            if all(active[cr + dr, cc + dc] for dr, dc in _GEN_PLUS_OFFSETS):
                return False
        return True
    finally:
        active[r, c] = False


def _gen_scatter_black(active, out, intended, green_cells) -> bool:
    target_active = random.randint(_GEN_ACTIVE_COUNT_MIN, _GEN_ACTIVE_COUNT_MAX)
    need_black = target_active - len(green_cells)
    cells = [(r, c) for r in range(_GEN_SIZE) for c in range(_GEN_SIZE) if not active[r, c]]
    random.shuffle(cells)
    for r, c in cells:
        if need_black <= 0:
            break
        if _gen_can_add_pixel(active, intended, r, c):
            active[r, c] = True
            out[r, c] = Color.BLACK
            need_black -= 1
    return need_black <= 0


def _gen_attempt() -> np.ndarray | None:
    centers = _gen_choose_centers()
    if centers is None:
        return None

    intended = set(centers)
    green_cells = _gen_plus_cells(centers)
    active = np.zeros((_GEN_SIZE, _GEN_SIZE), dtype=bool)
    out = np.full((_GEN_SIZE, _GEN_SIZE), Color.TEAL, dtype=int)
    for r, c in green_cells:
        active[r, c] = True
        out[r, c] = Color.GREEN
    if _gen_detect_centers(active) != intended:
        return None

    if not _gen_scatter_black(active, out, intended, green_cells):
        return None

    inp = out.copy()
    inp[inp == Color.GREEN] = Color.BLACK
    if not np.array_equal(main(inp), out):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(_GEN_MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid 7e02026e input")
