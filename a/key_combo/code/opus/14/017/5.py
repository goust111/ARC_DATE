from __future__ import annotations

import random

import numpy as np


BLACK = 0
GREEN = 3
YELLOW = 4
TEAL = 8

DIR4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
CROSS5 = ((0, 0),) + DIR4

GRID_H_RANGE = (16, 21)
GRID_W_RANGE = (16, 21)
CENTER_COUNT_CHOICES = (2, 3, 4)
CENTER_MARGIN = 4
MIN_CENTER_SEPARATION = 5
MAX_CLOSE_PAIR_DISTANCE = 8
DISTRACTOR_COUNT_RANGE = (2, 4)
MAX_GENERATION_ATTEMPTS = 400


def _chebyshev(a: tuple[int, int], b: tuple[int, int]) -> int:
    return max(abs(a[0] - b[0]), abs(a[1] - b[1]))


def _find_centers(grid: np.ndarray) -> list[tuple[int, int]]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    centers: list[tuple[int, int]] = []
    for r in range(1, h - 1):
        for c in range(1, w - 1):
            if g[r, c] != BLACK:
                continue
            if all(g[r + dr, c + dc] == BLACK for dr, dc in DIR4):
                centers.append((r, c))
    return centers


def _transform(grid: np.ndarray, centers: list[tuple[int, int]]) -> np.ndarray:
    out = np.array(grid, dtype=int, copy=True)
    for r, c in centers:
        for dr, dc in CROSS5:
            out[r + dr, c + dc] = GREEN

    if not centers:
        return out

    rows, cols = np.indices(out.shape)
    min_dist = np.full(out.shape, 10**9, dtype=int)
    for r, c in centers:
        dist = np.maximum(np.abs(rows - r), np.abs(cols - c))
        min_dist = np.minimum(min_dist, dist)

    ring_mask = (out == BLACK) & (min_dist >= 2) & (min_dist % 2 == 0)
    out[ring_mask] = YELLOW
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Detect every complete black plus and paint even Chebyshev-distance rings from the detected centers.

    Input structure:
    - The grid is 2D and uses only black background cells and teal blocker cells.
    - Some black cells are full 5-cell orthogonal plus centers: the center and its four cardinal neighbors are black.
    - There may be several such centers, along with other black distractor pixels that do not complete the template.

    Rule:
    - Scan the grid for every black cell whose north, south, west, and east neighbors are also black.
    - Recolor each detected center and its four orthogonal neighbors to green.
    - For every remaining black cell, compute its minimum Chebyshev distance to all detected centers.
    - Recolor the cell yellow iff that minimum distance is even and at least 2; keep teal blockers and green cross pixels unchanged.

    Key insight:
    - The green crosses reveal which local black templates are true centers, and those discovered centers jointly define one nearest-center ring field rather than separate independent fills.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    assert set(map(int, np.unique(g))) <= {BLACK, TEAL}, "expected only black and teal in the input"
    return _transform(g, _find_centers(g))


def _choose_centers(h: int, w: int, count: int) -> list[tuple[int, int]] | None:
    candidates = [
        (r, c)
        for r in range(CENTER_MARGIN, h - CENTER_MARGIN)
        for c in range(CENTER_MARGIN, w - CENTER_MARGIN)
    ]
    for _ in range(300):
        random.shuffle(candidates)
        centers: list[tuple[int, int]] = []
        for pos in candidates:
            if all(_chebyshev(pos, other) >= MIN_CENTER_SEPARATION for other in centers):
                centers.append(pos)
                if len(centers) == count:
                    break
        if len(centers) != count:
            continue
        pair_dists = [_chebyshev(a, b) for i, a in enumerate(centers) for b in centers[i + 1 :]]
        if pair_dists and min(pair_dists) <= MAX_CLOSE_PAIR_DISTANCE:
            return centers
    return None


def _add_distractors(
    h: int,
    w: int,
    centers: list[tuple[int, int]],
    mandatory_black: set[tuple[int, int]],
    forced_teal: set[tuple[int, int]],
) -> None:
    target = random.randint(*DISTRACTOR_COUNT_RANGE)
    added = 0
    for _ in range(500):
        if added >= target:
            return
        r = random.randint(2, h - 3)
        c = random.randint(2, w - 3)
        if any(_chebyshev((r, c), center) <= 2 for center in centers):
            continue
        if (r, c) in mandatory_black or (r, c) in forced_teal:
            continue

        missing_dir = random.choice(DIR4)
        cells = {(r, c)}
        blockers = set()
        for dr, dc in DIR4:
            rr, cc = r + dr, c + dc
            if (dr, dc) == missing_dir:
                blockers.add((rr, cc))
            else:
                cells.add((rr, cc))
                blockers.add((rr + dr, cc + dc))

        if any(not (0 <= rr < h and 0 <= cc < w) for rr, cc in cells | blockers):
            continue
        if any(cell in forced_teal for cell in cells):
            continue
        if any(cell in mandatory_black for cell in blockers):
            continue

        mandatory_black |= cells
        forced_teal |= blockers
        added += 1


def _candidate_weight(
    pos: tuple[int, int],
    centers: list[tuple[int, int]],
    span_r0: int,
    span_r1: int,
    span_c0: int,
    span_c1: int,
    h: int,
    w: int,
) -> float:
    r, c = pos
    dist = min(_chebyshev(pos, center) for center in centers)
    if dist <= 4:
        weight = 0.70
    elif dist <= 6:
        weight = 0.45
    elif dist <= 8:
        weight = 0.25
    else:
        weight = 0.08

    if span_r0 <= r <= span_r1 and span_c0 <= c <= span_c1:
        weight += 0.18
    if r in (0, h - 1) or c in (0, w - 1):
        weight += 0.05
    return min(weight, 0.95)


def _distance_grid(shape: tuple[int, int], centers: list[tuple[int, int]]) -> np.ndarray:
    rows, cols = np.indices(shape)
    min_dist = np.full(shape, 10**9, dtype=int)
    for r, c in centers:
        min_dist = np.minimum(min_dist, np.maximum(np.abs(rows - r), np.abs(cols - c)))
    return min_dist


def _validate_generated(grid: np.ndarray, expected_centers: list[tuple[int, int]]) -> bool:
    g = np.asarray(grid, dtype=int)
    if g.ndim != 2 or max(g.shape) > 30:
        return False
    if set(map(int, np.unique(g))) - {BLACK, TEAL}:
        return False

    actual_centers = _find_centers(g)
    if actual_centers != sorted(actual_centers):
        actual_centers = sorted(actual_centers)
    expected_centers = sorted(expected_centers)
    if actual_centers != expected_centers:
        return False
    if not (2 <= len(actual_centers) <= 4):
        return False

    pair_dists = [_chebyshev(a, b) for i, a in enumerate(actual_centers) for b in actual_centers[i + 1 :]]
    if pair_dists and (min(pair_dists) < 4 or min(pair_dists) > MAX_CLOSE_PAIR_DISTANCE):
        return False

    input_black = int(np.count_nonzero(g == BLACK))
    if not (0.18 * g.size <= input_black <= 0.55 * g.size):
        return False

    out = main(g)
    green_count = int(np.count_nonzero(out == GREEN))
    if green_count != 5 * len(actual_centers):
        return False

    min_dist = _distance_grid(g.shape, actual_centers)
    yellow_mask = out == YELLOW
    yellow_count = int(np.count_nonzero(yellow_mask))
    if yellow_count < 18 or yellow_count > int(0.45 * g.size):
        return False

    yellow_distances = {int(min_dist[r, c]) for r, c in np.argwhere(yellow_mask)}
    if not {2, 4}.issubset(yellow_distances):
        return False

    odd_black = int(np.count_nonzero((out == BLACK) & (min_dist % 2 == 1)))
    if odd_black < 12:
        return False

    near_plus = 0
    for r in range(1, g.shape[0] - 1):
        for c in range(1, g.shape[1] - 1):
            if g[r, c] != BLACK or (r, c) in actual_centers:
                continue
            black_neighbors = sum(g[r + dr, c + dc] == BLACK for dr, dc in DIR4)
            if black_neighbors == 3:
                near_plus += 1
    if near_plus < 2:
        return False

    tie_yellow = 0
    for r, c in np.argwhere(yellow_mask):
        dists = [_chebyshev((int(r), int(c)), center) for center in actual_centers]
        if dists.count(min(dists)) >= 2:
            tie_yellow += 1
    if tie_yellow < 2:
        return False

    return True


def _generate_attempt() -> np.ndarray | None:
    h = random.randint(*GRID_H_RANGE)
    w = random.randint(*GRID_W_RANGE)
    center_count = random.choice(CENTER_COUNT_CHOICES)
    centers = _choose_centers(h, w, center_count)
    if centers is None:
        return None

    grid = np.full((h, w), TEAL, dtype=int)
    mandatory_black: set[tuple[int, int]] = set()
    forced_teal: set[tuple[int, int]] = set()

    for r, c in centers:
        for dr, dc in CROSS5:
            mandatory_black.add((r + dr, c + dc))

    _add_distractors(h, w, centers, mandatory_black, forced_teal)

    for r, c in mandatory_black:
        grid[r, c] = BLACK

    r_values = [r for r, _ in centers]
    c_values = [c for _, c in centers]
    span_r0, span_r1 = min(r_values), max(r_values)
    span_c0, span_c1 = min(c_values), max(c_values)

    candidates: list[tuple[float, tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            pos = (r, c)
            if pos in mandatory_black or pos in forced_teal:
                continue
            if random.random() < _candidate_weight(pos, centers, span_r0, span_r1, span_c0, span_c1, h, w):
                candidates.append((random.random(), pos))

    candidates.sort()
    target_centers = sorted(centers)
    for _, (r, c) in candidates:
        grid[r, c] = BLACK
        if sorted(_find_centers(grid)) != target_centers:
            grid[r, c] = TEAL

    if not _validate_generated(grid, target_centers):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for ARC puzzle 5")
