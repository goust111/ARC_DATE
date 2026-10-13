from __future__ import annotations

import random

import numpy as np


BG = 0
BLUE = 1
RED = 2
PINK = 6
ORANGE = 7
TEAL = 8

GRID_SIDE_RANGE = (9, 13)
MARKER_COUNT_RANGE = (4, 6)
GENERATION_RETRIES = 400


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= r < shape[0] and 0 <= c < shape[1]


def _stamp_cells(r: int, c: int) -> list[tuple[int, int, int]]:
    if (r + c) % 2 == 0:
        return [(r, c - 1, ORANGE), (r, c, BLUE), (r, c + 1, PINK)]
    return [(r - 1, c, RED), (r, c, BLUE), (r + 1, c, TEAL)]


def _marker_footprint(r: int, c: int) -> frozenset[tuple[int, int]]:
    return frozenset((rr, cc) for rr, cc, _ in _stamp_cells(r, c))


def _markers_from_grid(grid: np.ndarray) -> list[tuple[int, int]]:
    return [tuple(map(int, cell)) for cell in np.argwhere(grid == BLUE)]


def _validate_marker_layout(shape: tuple[int, int], markers: list[tuple[int, int]]) -> None:
    seen_centers: set[tuple[int, int]] = set()
    occupied: set[tuple[int, int]] = set()
    for r, c in markers:
        assert _in_bounds(shape, r, c)
        assert 0 < r < shape[0] - 1 and 0 < c < shape[1] - 1, "markers must stay off the border"
        assert (r, c) not in seen_centers, "marker centers must be unique"
        seen_centers.add((r, c))
        footprint = _marker_footprint(r, c)
        assert occupied.isdisjoint(footprint), "stamps must not overlap"
        occupied |= footprint

    for i, (r1, c1) in enumerate(markers):
        for r2, c2 in markers[i + 1 :]:
            assert max(abs(r1 - r2), abs(c1 - c2)) >= 3, "markers must have Chebyshev distance >= 3"


def _expected_output(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    out = g.copy()
    for r, c in _markers_from_grid(g):
        for rr, cc, color in _stamp_cells(r, c):
            assert _in_bounds(g.shape, rr, cc), "generator promises unclipped stamps"
            if (rr, cc) == (r, c):
                assert int(out[rr, cc]) == BLUE
            else:
                assert int(out[rr, cc]) == BG
                out[rr, cc] = color
    return out


def _semantic_guards(markers: list[tuple[int, int]]) -> None:
    sums = [(r + c) % 2 for r, c in markers]
    assert sums.count(0) >= 2 and sums.count(1) >= 2, "both parity branches must be visible"
    assert len({r % 2 for r, _ in markers}) == 2, "row parities should vary"
    assert len({c % 2 for _, c in markers}) == 2, "column parities should vary"

    row_parity_to_branch = {0: set(), 1: set()}
    col_parity_to_branch = {0: set(), 1: set()}
    for r, c in markers:
        branch = (r + c) % 2
        row_parity_to_branch[r % 2].add(branch)
        col_parity_to_branch[c % 2].add(branch)

    assert any(len(branches) == 2 for branches in row_parity_to_branch.values()), "must rule out row-only orientation shortcuts"
    assert any(len(branches) == 2 for branches in col_parity_to_branch.values()), "must rule out column-only orientation shortcuts"


def main(grid: np.ndarray) -> np.ndarray:
    """
    Expand each blue marker into a parity-gated three-cell stamp with direction-colored endpoints.

    Input structure:
    - A black background contains only isolated blue marker pixels.
    - Each marker lies strictly inside the grid border, so a full three-cell stamp fits in bounds.
    - Marker neighborhoods are separated enough that the resulting horizontal or vertical stamps do not overlap.

    Rule:
    - For each blue marker at coordinate (r, c), compute (r + c) mod 2.
    - If the sum is even, paint a horizontal bar: orange at (r, c-1), keep blue at (r, c), and pink at (r, c+1).
    - If the sum is odd, paint a vertical bar: red at (r-1, c), keep blue at (r, c), and teal at (r+1, c).
    - Leave every other background cell unchanged.

    Key insight:
    - Row-plus-column parity is the hidden gate that selects the stamp axis, while compass direction from the marker fixes which endpoint color goes where.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and max(g.shape) <= 30
    assert set(map(int, np.unique(g))).issubset({BG, BLUE}), "input should contain only black background and blue markers"

    markers = _markers_from_grid(g)
    assert markers, "expected at least one blue marker"
    _validate_marker_layout(g.shape, markers)

    return _expected_output(g)


def _pick_mixed_parity_pair(size: int, rng: random.Random) -> tuple[int, int] | None:
    interior = list(range(1, size - 1))
    pairs: list[tuple[int, int]] = []
    for a in interior:
        for b in interior:
            if a % 2 == b % 2:
                continue
            if abs(a - b) < 3:
                continue
            pairs.append((a, b))
    if not pairs:
        return None
    return rng.choice(pairs)


def _can_add_marker(markers: list[tuple[int, int]], candidate: tuple[int, int], shape: tuple[int, int]) -> bool:
    r, c = candidate
    if not (0 < r < shape[0] - 1 and 0 < c < shape[1] - 1):
        return False
    footprint = _marker_footprint(r, c)
    occupied = set()
    for rr, cc in markers:
        if max(abs(rr - r), abs(cc - c)) < 3:
            return False
        occupied |= _marker_footprint(rr, cc)
    return occupied.isdisjoint(footprint)


def _core_markers(shape: tuple[int, int], rng: random.Random) -> list[tuple[int, int]] | None:
    rows = _pick_mixed_parity_pair(shape[0], rng)
    cols = _pick_mixed_parity_pair(shape[1], rng)
    if rows is None or cols is None:
        return None

    r_a, r_b = rows
    c_a, c_b = cols
    markers = [(r_a, c_a), (r_a, c_b), (r_b, c_a), (r_b, c_b)]
    rng.shuffle(markers)
    if any(not _can_add_marker(markers[:i], marker, shape) for i, marker in enumerate(markers)):
        return None
    return markers


def _fill_extra_markers(
    markers: list[tuple[int, int]],
    target_count: int,
    shape: tuple[int, int],
    rng: random.Random,
) -> list[tuple[int, int]] | None:
    if len(markers) > target_count:
        return None

    candidates = [
        (r, c)
        for r in range(1, shape[0] - 1)
        for c in range(1, shape[1] - 1)
    ]
    rng.shuffle(candidates)
    placed = list(markers)
    for candidate in candidates:
        if len(placed) == target_count:
            return placed
        if _can_add_marker(placed, candidate, shape):
            placed.append(candidate)
    if len(placed) == target_count:
        return placed
    return None


def _attempt_generate(rng: random.Random) -> np.ndarray | None:
    height = rng.randint(*GRID_SIDE_RANGE)
    width = rng.randint(*GRID_SIDE_RANGE)
    target_count = rng.randint(*MARKER_COUNT_RANGE)

    markers = _core_markers((height, width), rng)
    if markers is None:
        return None
    markers = _fill_extra_markers(markers, target_count, (height, width), rng)
    if markers is None:
        return None

    _validate_marker_layout((height, width), markers)
    _semantic_guards(markers)

    grid = np.full((height, width), BG, dtype=int)
    for r, c in markers:
        grid[r, c] = BLUE

    out = _expected_output(grid)
    used_output_colors = set(map(int, np.unique(out)))
    assert used_output_colors == {BG, BLUE, RED, PINK, ORANGE, TEAL}, "both stamp orientations must appear in output"
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GENERATION_RETRIES):
        grid = _attempt_generate(rng)
        if grid is None:
            continue
        try:
            expected = _expected_output(grid)
            assert np.array_equal(main(grid), expected)
            return grid
        except Exception:
            continue
    raise RuntimeError("failed to generate a valid input for the parity-gated stamp task")
