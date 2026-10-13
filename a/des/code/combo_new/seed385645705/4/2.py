from __future__ import annotations

import random

import numpy as np


BLACK = 0
YELLOW = 4
MARKER_COLORS = [1, 2, 3, 5, 6, 7, 8, 9]


def _ordered_pair(coords: np.ndarray) -> tuple[tuple[int, int], tuple[int, int]]:
    pts = [tuple(map(int, rc)) for rc in coords]
    pts.sort(key=lambda p: (p[0], p[1]))
    a, b = pts
    assert a[0] != b[0] and a[1] != b[1], "paired markers must not share a row or column"
    return a, b


def _path_cells(a: tuple[int, int], b: tuple[int, int]) -> list[tuple[int, int]]:
    ar, ac = a
    br, bc = b
    dc = 1 if bc > ac else -1
    dr = 1 if br > ar else -1

    cells: list[tuple[int, int]] = []
    for c in range(ac + dc, bc + dc, dc):
        cells.append((ar, c))
    for r in range(ar + dr, br, dr):
        cells.append((r, bc))
    return cells


def _path_parts(a: tuple[int, int], b: tuple[int, int]) -> tuple[set[tuple[int, int]], set[tuple[int, int]]]:
    ar, ac = a
    br, bc = b
    dc = 1 if bc > ac else -1
    dr = 1 if br > ar else -1

    horizontal = {(ar, c) for c in range(ac + dc, bc + dc, dc)}
    vertical = {(r, bc) for r in range(ar + dr, br, dr)}
    return horizontal, vertical


def _marker_pairs(g: np.ndarray) -> dict[int, tuple[tuple[int, int], tuple[int, int]]]:
    assert g.ndim == 2
    assert int(g.min()) >= 0 and int(g.max()) <= 9
    assert not np.any(g == YELLOW), "yellow is reserved for output conflict pixels"

    pairs: dict[int, tuple[tuple[int, int], tuple[int, int]]] = {}
    colors = [int(v) for v in np.unique(g) if int(v) != BLACK]
    assert 3 <= len(colors) <= 5, "expected 3-5 marker pairs"
    for color in colors:
        coords = np.argwhere(g == color)
        assert coords.shape[0] == 2, "each marker color must occur exactly twice"
        pairs[color] = _ordered_pair(coords)
    return pairs


def main(grid: np.ndarray) -> np.ndarray:
    """
    Connect each same-color marker pair by a top-first L path and mark crossings yellow.

    Input structure:
    - The background is black.
    - There are 3-5 non-yellow foreground colors, and each color appears as
      exactly two single-pixel markers.
    - The two markers of any color are never in the same row or same column.

    Rule:
    - For each color, treat the upper marker as A and the lower marker as B.
    - Draw the L path from A horizontally to B's column, then vertically to B,
      excluding the two endpoint marker pixels themselves.
    - A cell covered by paths of two or more different colors becomes yellow;
      otherwise a covered cell takes the single path color.

    Key insight:
    - The pairwise L paths are simple, but their simultaneous overlay creates a
      second rule: different-color path intersections are conflict cells.
    """
    g = np.asarray(grid, dtype=int)
    pairs = _marker_pairs(g)

    coverage: dict[tuple[int, int], set[int]] = {}
    for color, (a, b) in pairs.items():
        for cell in _path_cells(a, b):
            coverage.setdefault(cell, set()).add(int(color))

    out = g.copy()
    for (r, c), colors in coverage.items():
        if len(colors) >= 2:
            out[r, c] = YELLOW
        else:
            out[r, c] = next(iter(colors))
    return out


# --------------- generator ---------------

SIZE_MIN = 9
SIZE_MAX = 22
PAIR_COUNT_RANGE = (3, 5)
MAX_ATTEMPTS = 600


def _far_from_existing(cell: tuple[int, int], used: set[tuple[int, int]]) -> bool:
    r, c = cell
    return all(max(abs(r - rr), abs(c - cc)) > 1 for rr, cc in used)


def _pair_long_enough(pair: tuple[tuple[int, int], tuple[int, int]]) -> bool:
    (r0, c0), (r1, c1) = pair
    return abs(r1 - r0) + abs(c1 - c0) >= 4


def _render(shape: tuple[int, int], pairs: list[tuple[tuple[int, int], tuple[int, int]]], colors: list[int]) -> np.ndarray:
    grid = np.full(shape, BLACK, dtype=int)
    for color, (a, b) in zip(colors, pairs):
        grid[a] = color
        grid[b] = color
    return grid


def _forced_crossing_pair(height: int, width: int) -> list[tuple[tuple[int, int], tuple[int, int]]] | None:
    for _ in range(100):
        cross_r = random.randint(2, height - 3)
        cross_c = random.randint(2, width - 3)

        start2_r = random.randint(0, cross_r - 1)
        end2_r = random.randint(cross_r + 1, height - 1)
        start2_c = random.choice([c for c in range(width) if c != cross_c])

        end1_r = random.randint(cross_r + 1, height - 1)
        span_options = [
            (start_c, end_c)
            for start_c in range(width)
            for end_c in range(width)
            if start_c != end_c
            and start_c != cross_c
            and min(start_c, end_c) <= cross_c <= max(start_c, end_c)
        ]
        if not span_options:
            continue
        start1_c, end1_c = random.choice(span_options)

        pair1 = ((cross_r, start1_c), (end1_r, end1_c))
        pair2 = ((start2_r, start2_c), (end2_r, cross_c))
        if _pair_long_enough(pair1) and _pair_long_enough(pair2):
            return [pair1, pair2]
    return None


def _random_pair(height: int, width: int, used: set[tuple[int, int]]) -> tuple[tuple[int, int], tuple[int, int]] | None:
    for _ in range(200):
        r0, r1 = sorted(random.sample(range(height), 2))
        c0, c1 = random.sample(range(width), 2)
        pair = ((r0, c0), (r1, c1))
        if not _pair_long_enough(pair):
            continue
        if _far_from_existing(pair[0], used) and _far_from_existing(pair[1], used):
            return pair
    return None


def _coverage_details(
    pairs: list[tuple[tuple[int, int], tuple[int, int]]],
    colors: list[int],
) -> tuple[dict[tuple[int, int], set[int]], dict[tuple[int, int], set[int]], dict[tuple[int, int], set[int]]]:
    coverage: dict[tuple[int, int], set[int]] = {}
    horizontal: dict[tuple[int, int], set[int]] = {}
    vertical: dict[tuple[int, int], set[int]] = {}

    for color, (a, b) in zip(colors, pairs):
        h_cells, v_cells = _path_parts(a, b)
        for cell in h_cells | v_cells:
            coverage.setdefault(cell, set()).add(color)
        for cell in h_cells:
            horizontal.setdefault(cell, set()).add(color)
        for cell in v_cells:
            vertical.setdefault(cell, set()).add(color)
    return coverage, horizontal, vertical


def _valid_candidate(grid: np.ndarray, pairs: list[tuple[tuple[int, int], tuple[int, int]]], colors: list[int]) -> bool:
    marker_cells = {cell for pair in pairs for cell in pair}
    if len(marker_cells) != 2 * len(pairs):
        return False
    for cell in marker_cells:
        others = marker_cells - {cell}
        if not _far_from_existing(cell, others):
            return False

    coverage, horizontal, vertical = _coverage_details(pairs, colors)
    if marker_cells & set(coverage):
        return False
    if any(len(cs) > 1 for cs in horizontal.values()):
        return False
    if any(len(cs) > 1 for cs in vertical.values()):
        return False

    conflicts = [cell for cell, cs in coverage.items() if len(cs) >= 2]
    if not (1 <= len(conflicts) <= len(pairs) + 4):
        return False
    if len(coverage) <= len(conflicts):
        return False

    try:
        out = main(grid)
    except AssertionError:
        return False
    return bool(np.any(out == YELLOW))


def _gen_attempt() -> np.ndarray | None:
    height = random.randint(SIZE_MIN, SIZE_MAX)
    width = random.randint(SIZE_MIN, SIZE_MAX)
    n_pairs = random.randint(*PAIR_COUNT_RANGE)

    pairs = _forced_crossing_pair(height, width)
    if pairs is None:
        return None

    used = {cell for pair in pairs for cell in pair}
    if len(used) != 4:
        return None
    if any(not _far_from_existing(cell, used - {cell}) for cell in used):
        return None

    while len(pairs) < n_pairs:
        pair = _random_pair(height, width, used)
        if pair is None:
            return None
        pairs.append(pair)
        used.update(pair)

    colors = random.sample(MARKER_COLORS, n_pairs)
    grid = _render((height, width), pairs, colors)
    if not _valid_candidate(grid, pairs, colors):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid L-path conflict input")
