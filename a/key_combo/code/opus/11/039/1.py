from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np


BG = 0
WALL = 2
MARKER_COLORS = (1, 3, 4, 7)
MARKER_COLOR_SET = set(MARKER_COLORS)
GEN_ATTEMPTS = 300


@dataclass(frozen=True)
class Marker:
    row: int
    col: int
    color: int


def _parse_grid(grid: np.ndarray) -> tuple[np.ndarray, list[Marker]]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    height, width = map(int, g.shape)
    assert height >= 4 and width >= 5, "grid is too small"
    assert np.all(g[-1, :] == WALL), "expected a full red bottom wall"

    markers: list[Marker] = []
    for r in range(height - 1):
        for c in range(width):
            color = int(g[r, c])
            if color == BG:
                continue
            assert color in MARKER_COLOR_SET, "unexpected foreground color above the wall"
            markers.append(Marker(r, c, color))

    assert 3 <= len(markers) <= 4, "expected 3 or 4 isolated markers"
    assert len({marker.color for marker in markers}) == len(markers), "marker colors must be distinct"
    assert len({marker.row for marker in markers}) == len(markers), "markers must occupy distinct rows"
    assert len({marker.col for marker in markers}) == len(markers), "markers must occupy distinct columns"
    return g, markers


def _settle_markers(markers: list[Marker], shape: tuple[int, int]) -> list[Marker]:
    height, width = map(int, shape)
    occupied = np.zeros((height, width), dtype=bool)
    occupied[-1, :] = True

    ordered = sorted(markers, key=lambda marker: (height - 1 - marker.row, marker.col))
    settled: list[Marker] = []

    for marker in ordered:
        row = int(marker.row)
        while row + 1 < height and not occupied[row + 1, marker.col]:
            row += 1
        settled_marker = Marker(row, int(marker.col), int(marker.color))
        occupied[settled_marker.row, settled_marker.col] = True
        settled.append(settled_marker)

    return settled


def _expand_cross_rays(shape: tuple[int, int], settled: list[Marker]) -> np.ndarray:
    height, width = map(int, shape)
    out = np.full((height, width), BG, dtype=int)
    out[-1, :] = WALL

    marker_cells = {(marker.row, marker.col) for marker in settled}
    for marker in settled:
        out[marker.row, marker.col] = marker.color

    for marker in settled:
        r = int(marker.row)
        c = int(marker.col)
        color = int(marker.color)

        for cc in range(width):
            if (r, cc) in marker_cells or out[r, cc] == WALL:
                continue
            out[r, cc] = color

        for rr in range(height):
            if (rr, c) in marker_cells or out[rr, c] == WALL:
                continue
            out[rr, c] = color

    return out


def _direct_formula_output(grid: np.ndarray) -> np.ndarray:
    """Independent oracle for the constrained generator domain."""
    g, markers = _parse_grid(grid)
    height, width = map(int, g.shape)
    ordered = sorted(markers, key=lambda marker: marker.row, reverse=True)
    landing_row = height - 2
    last_color = int(min(markers, key=lambda marker: marker.row).color)

    out = np.full_like(g, BG)
    out[-1, :] = WALL
    marker_cols = {marker.col for marker in markers}

    for marker in markers:
        out[landing_row, marker.col] = marker.color
        out[:landing_row, marker.col] = marker.color

    for c in range(width):
        if c not in marker_cols:
            out[landing_row, c] = last_color

    assert ordered[-1].color == last_color
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Drop the scattered markers onto the red floor, then expand each landed marker into a row-and-column cross.

    Input structure:
    - The background is black.
    - The entire bottom row is a solid red wall.
    - Above that wall are 3 or 4 isolated one-cell markers with distinct colors chosen from blue, green, yellow, and orange, with no shared row or column.

    Rule:
    - Process markers from nearest to the red wall to farthest from it, sliding each marker straight downward until the next step would hit the wall or an already settled marker.
    - After all markers settle, expand them in that same settlement order: each marker paints its whole row and whole column as a cross ray.
    - Later expansions overwrite earlier ray colors on overlaps, but the red wall and every settled marker cell keep their original colors.

    Key insight:
    - The falling phase determines both the final ray anchors and the priority order for overlap resolution, so the output depends on reading height order before drawing the crosses.
    """
    g, markers = _parse_grid(grid)
    settled = _settle_markers(markers, g.shape)
    return _expand_cross_rays(g.shape, settled)


def _strictly_increasing_levels(height: int, fracs: tuple[float, ...]) -> list[int]:
    max_row = height - 2
    levels = [int(round(frac * max_row)) for frac in fracs]
    levels[0] = max(0, levels[0])
    levels[-1] = min(max_row, levels[-1])

    for i in range(1, len(levels)):
        levels[i] = max(levels[i], levels[i - 1] + 1)
    for i in range(len(levels) - 2, -1, -1):
        levels[i] = min(levels[i], levels[i + 1] - 1)

    assert levels[0] >= 0 and levels[-1] <= max_row
    assert len(set(levels)) == len(levels)
    return levels


def _row_profile(height: int, count: int, rng: np.random.Generator) -> list[int]:
    if count == 3:
        profiles = (
            (0.00, 0.45, 1.00),
            (0.00, 0.62, 1.00),
            (0.10, 0.55, 1.00),
        )
    else:
        profiles = (
            (0.00, 0.33, 0.67, 1.00),
            (0.00, 0.20, 0.74, 1.00),
            (0.08, 0.42, 0.78, 1.00),
        )
    return _strictly_increasing_levels(height, profiles[int(rng.integers(0, len(profiles)))])


def _sample_spaced_columns(width: int, count: int, rng: np.random.Generator) -> list[int]:
    min_gap = 2
    base_width = width - (count - 1) * (min_gap - 1)
    base = np.sort(rng.choice(base_width, size=count, replace=False))
    cols = [int(base[i] + i * (min_gap - 1)) for i in range(count)]
    assert all(cols[i + 1] - cols[i] >= min_gap for i in range(count - 1))
    return cols


def _is_confusing_order(values: list[int], order: np.ndarray) -> bool:
    asc = list(np.argsort(values))
    desc = asc[::-1]
    order_list = list(map(int, order))
    return order_list in (asc, desc)


def _build_case(
    height: int,
    width: int,
    colors: list[int],
    cols: list[int],
    settle_order: np.ndarray,
    rows: list[int],
) -> np.ndarray:
    grid = np.zeros((height, width), dtype=int)
    grid[-1, :] = WALL
    for marker_id, color in enumerate(colors):
        grid[int(rows[marker_id]), int(cols[marker_id])] = int(color)
    return grid


def _swap_last_order_case(grid: np.ndarray) -> np.ndarray:
    g, markers = _parse_grid(grid)
    order = sorted(range(len(markers)), key=lambda idx: markers[idx].row)
    swapped = list(markers)
    a = order[0]
    b = order[1]
    row_a = swapped[a].row
    row_b = swapped[b].row
    swapped[a] = Marker(row_b, swapped[a].col, swapped[a].color)
    swapped[b] = Marker(row_a, swapped[b].col, swapped[b].color)

    out = np.zeros_like(g)
    out[-1, :] = WALL
    for marker in swapped:
        out[marker.row, marker.col] = marker.color
    return out


def _expand_without_gravity(grid: np.ndarray) -> np.ndarray:
    g, markers = _parse_grid(grid)
    out = np.full_like(g, BG)
    out[-1, :] = WALL
    marker_cells = {(marker.row, marker.col) for marker in markers}
    ordered = sorted(markers, key=lambda marker: marker.row, reverse=True)

    for marker in markers:
        out[marker.row, marker.col] = marker.color

    for marker in ordered:
        for cc in range(g.shape[1]):
            if (marker.row, cc) in marker_cells or out[marker.row, cc] == WALL:
                continue
            out[marker.row, cc] = marker.color
        for rr in range(g.shape[0]):
            if (rr, marker.col) in marker_cells or out[rr, marker.col] == WALL:
                continue
            out[rr, marker.col] = marker.color
    return out


def _validate_case(grid: np.ndarray) -> None:
    g, markers = _parse_grid(grid)
    out = main(g)
    oracle = _direct_formula_output(g)
    assert np.array_equal(out, oracle), "main must match the independent oracle"

    landing_row = g.shape[0] - 2
    assert all(_settle_markers(markers, g.shape)[i].row == landing_row for i in range(len(markers)))
    assert np.count_nonzero(g[:-1, :] != BG) == len(markers), "input should contain markers only"

    no_gravity = _expand_without_gravity(g)
    assert not np.array_equal(out, no_gravity), "gravity must change the result"

    swapped = _swap_last_order_case(g)
    swapped_out = main(swapped)
    assert not np.array_equal(out, swapped_out), "changing the last-settled marker must change the output"

    non_marker_cols = [c for c in range(g.shape[1]) if c not in {marker.col for marker in markers}]
    assert non_marker_cols, "need at least one non-marker column for row-ray evidence"
    shared_row_colors = {int(out[landing_row, c]) for c in non_marker_cols}
    assert len(shared_row_colors) == 1, "shared landing row should show the winning late color"


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()

    for _ in range(GEN_ATTEMPTS):
        count = int(rng.choice([3, 4]))
        height = int(rng.integers(10, 19))
        width = int(rng.integers(max(9, 2 * count + 1), 16))

        colors = list(map(int, rng.choice(MARKER_COLORS, size=count, replace=False)))
        cols = _sample_spaced_columns(width, count, rng)
        settle_order = rng.permutation(count)
        row_levels = _row_profile(height, count, rng)

        if _is_confusing_order(cols, settle_order):
            continue
        if _is_confusing_order(colors, settle_order):
            continue

        rows = [0] * count
        for rank, marker_id in enumerate(settle_order):
            rows[int(marker_id)] = int(row_levels[-1 - rank])

        grid = _build_case(height, width, colors, cols, settle_order, rows)

        try:
            _validate_case(grid)
        except AssertionError:
            continue
        return grid

    raise RuntimeError("failed to generate a valid gravity-cross instance")


def _variant_signature(grid: np.ndarray) -> str:
    return hashlib.md5(grid.tobytes() + str(grid.shape).encode()).hexdigest()
