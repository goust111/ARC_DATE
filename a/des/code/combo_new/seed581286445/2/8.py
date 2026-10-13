"""ARC-style solver and generator for puzzle 8."""

from __future__ import annotations

import random
from typing import Optional

import numpy as np


BG = 0
COLOR_POOL = tuple(range(1, 10))
MODE_OPTIONS = ("no_clip", "top_clip", "bottom_clip")
HEIGHT_RANGE = (8, 30)
WIDTH_RANGE = (6, 30)
MIN_DISTANCE = 2
MAX_GENERATION_ATTEMPTS = 300


def _tree_cells(height: int, width: int, row: int, distance: int) -> set[tuple[int, int]]:
    """Return all output cells painted by one tree."""
    cells = {(row, col) for col in range(distance)}
    branch_col = distance - 1
    for rr in range(max(0, row - distance), row):
        cells.add((rr, branch_col))
    for rr in range(row + 1, min(height, row + distance + 1)):
        cells.add((rr, branch_col))
    return cells


def _distance_bounds(mode: str, row: int, height: int, width: int) -> Optional[tuple[int, int]]:
    """Return inclusive distance bounds for a row/mode pair."""
    if mode == "no_clip":
        upper = min(width - 1, row, height - 1 - row)
        lower = MIN_DISTANCE
    elif mode == "top_clip":
        lower = max(MIN_DISTANCE, row + 1)
        upper = min(width - 1, height - 1 - row)
    elif mode == "bottom_clip":
        lower = max(MIN_DISTANCE, height - row)
        upper = min(width - 1, row)
    else:
        raise ValueError(f"unknown mode: {mode}")
    if lower > upper:
        return None
    return lower, upper


def _choose_modes(count: int) -> list[str]:
    """Sample branch-clipping modes with balanced coverage across examples."""
    if count <= 0:
        return []
    pool = list(MODE_OPTIONS)
    while len(pool) < count:
        pool.extend(MODE_OPTIONS)
    random.shuffle(pool)
    return pool[:count]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Grow T-shaped trees from left-edge seed pixels using same-row distance markers.

    Input structure:
    - The background is black.
    - The leftmost column contains several isolated non-black seed pixels.
    - For each seed color, the same row contains exactly one matching-color marker to the right.
    - Apart from those seeds and markers, the grid is empty.

    Rule:
    - For each seed at row r and column 0, find the matching marker at column d in the same row.
    - Draw a horizontal branch of that color from columns 0 through d-1 on row r; the marker cell itself is removed.
    - At the branch endpoint (r, d-1), draw an upward and a downward vertical branch, each of length d.
    - If a vertical branch would leave the grid, truncate it at the grid boundary.

    Key insight:
    - The marker is not part of the output object; its horizontal offset from the seed encodes one shared growth length for both the horizontal trunk and the two vertical branches.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    height, width = g.shape
    assert width >= 2, "grid must allow a seed column and a marker column"

    out = np.full_like(g, BG)

    seed_rows = np.flatnonzero(g[:, 0] != BG)
    assert seed_rows.size > 0, "expected at least one seed in the left column"

    for row in map(int, seed_rows):
        color = int(g[row, 0])
        marker_cols = [int(col) for col in np.flatnonzero(g[row] == color) if int(col) != 0]
        assert len(marker_cols) == 1, "each seed row must contain exactly one matching marker"
        distance = marker_cols[0]
        assert distance >= 1, "marker must lie to the right of the seed"

        out[row, :distance] = color
        branch_col = distance - 1
        top = max(0, row - distance)
        bottom = min(height - 1, row + distance)
        out[top:row, branch_col] = color
        out[row + 1 : bottom + 1, branch_col] = color

    return out


def _gen_attempt() -> Optional[np.ndarray]:
    height = random.randint(*HEIGHT_RANGE)
    width = random.randint(*WIDTH_RANGE)

    max_trees = min(5, 9, (height + 1) // 2)
    if max_trees <= 0:
        return None

    counts = list(range(1, max_trees + 1))
    weights = [max_trees + 1 - count for count in counts]
    tree_count = random.choices(counts, weights=weights, k=1)[0]

    colors = random.sample(COLOR_POOL, tree_count)
    modes = _choose_modes(tree_count)
    random.shuffle(modes)

    occupied_output: set[tuple[int, int]] = set()
    reserved_markers: set[tuple[int, int]] = set()
    used_rows: set[int] = set()
    placements: list[tuple[int, int, int]] = []

    order = sorted(
        range(tree_count),
        key=lambda idx: {"top_clip": 0, "bottom_clip": 0, "no_clip": 1}[modes[idx]],
    )

    for idx in order:
        mode = modes[idx]
        candidates: list[tuple[int, int, set[tuple[int, int]]]] = []
        for row in range(height):
            if any(abs(row - other) <= 1 for other in used_rows):
                continue
            bounds = _distance_bounds(mode, row, height, width)
            if bounds is None:
                continue
            lower, upper = bounds
            distances = list(range(lower, upper + 1))
            random.shuffle(distances)
            for distance in distances:
                shape = _tree_cells(height, width, row, distance)
                marker = (row, distance)
                if marker in occupied_output or marker in reserved_markers:
                    continue
                if shape & occupied_output:
                    continue
                if shape & reserved_markers:
                    continue
                candidates.append((row, distance, shape))

        if not candidates:
            return None

        row, distance, shape = random.choice(candidates)
        placements.append((idx, row, distance))
        used_rows.add(row)
        occupied_output |= shape
        reserved_markers.add((row, distance))

    grid = np.full((height, width), BG, dtype=int)
    for idx, row, distance in placements:
        color = colors[idx]
        grid[row, 0] = color
        grid[row, distance] = color

    if np.count_nonzero(grid[:, 0] != BG) != tree_count:
        return None

    # Re-validate against the deterministic solver assumptions.
    for row in used_rows:
        color = int(grid[row, 0])
        marker_cols = [int(col) for col in np.flatnonzero(grid[row] == color) if int(col) != 0]
        if len(marker_cols) != 1:
            return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
