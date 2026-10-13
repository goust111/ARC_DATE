from __future__ import annotations

import random
from collections import deque

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5
ORANGE = 7

ORTH_DIRS = [(-1, 0), (1, 0), (0, -1), (0, 1)]
DIAG_DIRS = [(-1, -1), (-1, 1), (1, -1), (1, 1)]
OUTER_RING = [(*d, BLUE) for d in ORTH_DIRS] + [(*d, RED) for d in DIAG_DIRS]

GRID_MIN = 20
GRID_MAX = 28
COUNT_MIN = 3
COUNT_MAX = 5
CENTER_GAP = 7
GEN_ATTEMPTS = 400


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = map(int, shape)
    return 0 <= int(r) < h and 0 <= int(c) < w


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _components4(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                comp.append((rr, cc))
                for dr, dc in ORTH_DIRS:
                    nr, nc = rr + dr, cc + dc
                    if _in_bounds(mask.shape, nr, nc) and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _center_color_from_count(blocked: int) -> int:
    if blocked == 0:
        return RED
    if blocked <= 2:
        return GREEN
    if blocked <= 4:
        return YELLOW
    return ORANGE


def _extract_crosses(grid: np.ndarray) -> list[tuple[int, int]]:
    mask = (grid == BLUE) | (grid == RED)
    comps = _components4(mask)
    assert 3 <= len(comps) <= 5, "expected 3 to 5 independent two-color crosses"

    centers: list[tuple[int, int]] = []
    for comp in comps:
        assert len(comp) == 5, "each cross must contain exactly five colored cells"
        r0, c0, r1, c1 = _bbox(comp)
        assert (r1 - r0 + 1, c1 - c0 + 1) == (3, 3), "each cross must occupy a 3x3 bbox"
        cr, cc = r0 + 1, c0 + 1

        for rr in range(r0, r1 + 1):
            for cc2 in range(c0, c1 + 1):
                expected = BLACK
                if (rr, cc2) == (cr, cc):
                    expected = RED
                elif rr == cr or cc2 == cc:
                    expected = BLUE
                assert int(grid[rr, cc2]) == expected, "cross pattern must be a blue plus with a red center"

        centers.append((cr, cc))

    return sorted(centers)


def _blocked_counts_and_output(grid: np.ndarray) -> tuple[list[int], np.ndarray]:
    g = np.asarray(grid, dtype=int)
    out = g.copy()
    centers = _extract_crosses(g)

    for cr, cc in centers:
        for dr, dc in DIAG_DIRS:
            rr, cc2 = cr + dr, cc + dc
            assert _in_bounds(g.shape, rr, cc2), "expanded diagonal must stay in bounds"
            assert int(g[rr, cc2]) != GREY, "obstacles may not overlap the cross's 3x3 footprint"
            out[rr, cc2] = RED

    counts: list[int] = []
    for cr, cc in centers:
        blocked = 0
        for dr, dc, color in OUTER_RING:
            rr, cc2 = cr + 2 * dr, cc + 2 * dc
            assert _in_bounds(g.shape, rr, cc2), "outer expansion targets must stay in bounds"
            if int(g[rr, cc2]) == GREY:
                blocked += 1
            else:
                out[rr, cc2] = color
        counts.append(blocked)
        out[cr, cc] = _center_color_from_count(blocked)
    return counts, out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Expand each two-color cross, stop blocked outer rays at grey obstacles, and recolor the center by the number of stopped directions.

    Input structure:
    - The background is black.
    - The foreground contains 3 to 5 separate 3x3 crosses whose four arms are blue and whose center is red.
    - Grey single-pixel obstacles may occupy some of the eight outer target cells around a cross, but never its original 3x3 footprint.

    Rule:
    - For every cross, first complete the 0962bcdd-style 5x5 motif: add red diagonal cells at distance 1 from the center, and try to add one outer cell in each of the 8 ray directions at distance 2.
    - If an outer target cell is grey, that ray is stopped and that target cell is left unchanged; otherwise paint it blue on orthogonal rays and red on diagonal rays.
    - Count the stopped outer rays for each cross and recolor only that cross center: 0 stays red, 1 or 2 becomes green, 3 or 4 becomes yellow, and 5 or more becomes orange.

    Key insight:
    - The obstacles have two roles at once: they clip the final expanded outline and, through the number of clipped rays, encode the center's final color.
    """
    _counts, out = _blocked_counts_and_output(np.asarray(grid, dtype=int))
    return out


def _candidate_centers(h: int, w: int) -> list[tuple[int, int]]:
    cells = [(r, c) for r in range(2, h - 2) for c in range(2, w - 2)]
    random.shuffle(cells)
    return cells


def _place_centers(h: int, w: int, n_crosses: int) -> list[tuple[int, int]] | None:
    centers: list[tuple[int, int]] = []
    for r, c in _candidate_centers(h, w):
        if all(max(abs(r - rr), abs(c - cc)) >= CENTER_GAP for rr, cc in centers):
            centers.append((r, c))
            if len(centers) == n_crosses:
                return centers
    return None


def _sample_block_counts(n_crosses: int) -> list[int]:
    counts = [0, random.choice([1, 2]), random.choice([3, 4])]
    while len(counts) < n_crosses:
        counts.append(random.randint(0, 4))
    random.shuffle(counts)
    if len(set(counts)) == 1:
        counts[-1] = (counts[-1] + 1) % 5
    return counts


def _outer_targets(center: tuple[int, int]) -> list[tuple[int, int]]:
    cr, cc = center
    return [(cr + 2 * dr, cc + 2 * dc) for dr, dc, _ in OUTER_RING]


def _build_input_case() -> tuple[np.ndarray, list[int]]:
    h = random.randint(GRID_MIN, GRID_MAX)
    w = random.randint(GRID_MIN, GRID_MAX)
    n_crosses = random.randint(COUNT_MIN, COUNT_MAX)
    centers = _place_centers(h, w, n_crosses)
    if centers is None:
        raise ValueError("failed to place enough crosses")
    centers = sorted(centers)

    grid = np.zeros((h, w), dtype=int)
    for cr, cc in centers:
        grid[cr, cc] = RED
        for dr, dc in ORTH_DIRS:
            grid[cr + dr, cc + dc] = BLUE

    counts = _sample_block_counts(n_crosses)
    for center, blocked in zip(centers, counts):
        targets = _outer_targets(center)
        for rr, cc in random.sample(targets, blocked):
            grid[rr, cc] = GREY

    return grid, counts


def _validate_generated_case(grid: np.ndarray, intended_counts: list[int]) -> bool:
    centers = _extract_crosses(grid)
    if len(centers) != len(intended_counts):
        return False

    allowed_greys = {coord for center in centers for coord in _outer_targets(center)}
    grey_coords = np.argwhere(grid == GREY)
    for r, c in grey_coords:
        if (int(r), int(c)) not in allowed_greys:
            return False
        for dr, dc in ORTH_DIRS:
            nr, nc = int(r + dr), int(c + dc)
            if _in_bounds(grid.shape, nr, nc) and int(grid[nr, nc]) == GREY:
                return False

    oracle_counts, oracle_out = _blocked_counts_and_output(grid)
    if oracle_counts != [int(x) for x in intended_counts]:
        return False
    if any(not (0 <= count <= 4) for count in oracle_counts):
        return False
    if len(set(oracle_counts)) == 1:
        return False

    actual = main(grid)
    if not np.array_equal(actual, oracle_out):
        return False

    center_colors = sorted(int(actual[cr, cc]) for cr, cc in centers)
    expected_center_colors = sorted(_center_color_from_count(count) for count in oracle_counts)
    return center_colors == expected_center_colors


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        try:
            grid, counts = _build_input_case()
        except ValueError:
            continue
        if _validate_generated_case(grid, counts):
            return grid
    raise RuntimeError(f"generate_input failed after {GEN_ATTEMPTS} attempts")
