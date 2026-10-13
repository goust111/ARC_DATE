from __future__ import annotations

from collections import Counter
import random

import numpy as np


BLACK = 0
GREY = 5
PALETTE = [1, 2, 3, 4, 6, 7, 8, 9]
MAX_ATTEMPTS = 300


def _compressed(values: list[int]) -> list[int]:
    out: list[int] = []
    for value in values:
        if not out or out[-1] != value:
            out.append(value)
    return out


def _strict_row_majority(row: np.ndarray) -> int:
    values, counts = np.unique(row, return_counts=True)
    index = int(np.argmax(counts))
    color = int(values[index])
    count = int(counts[index])
    assert count > row.size // 2, "each upper row must have a strict majority color"
    assert int(np.count_nonzero(counts == count)) == 1, "row majority must be unique"
    return color


def _find_separator_row(grid: np.ndarray) -> int:
    candidates = [r for r in range(grid.shape[0]) if np.all(grid[r, :] == GREY)]
    assert len(candidates) == 1, "expected exactly one full grey separator row"
    sep = candidates[0]
    assert 0 < sep < grid.shape[0] - 1, "separator must split the grid into two non-empty parts"
    return sep


def _find_cavity(grid: np.ndarray, start_row: int) -> tuple[int, int, int, int]:
    h, w = grid.shape
    candidates: list[tuple[int, int, int, int]] = []
    lower = grid[start_row:, :]
    rows, cols = np.where(lower == BLACK)
    for rr, cc in zip(rows, cols):
        r = int(start_row + rr)
        c = int(cc)
        if r == 0 or c == 0 or r == h - 1 or c == w - 1:
            continue
        if grid[r - 1, c] != GREY or grid[r, c - 1] != GREY:
            continue

        c1 = c
        while c1 + 1 < w and grid[r, c1 + 1] == BLACK:
            c1 += 1
        r1 = r
        while r1 + 1 < h and grid[r1 + 1, c] == BLACK:
            r1 += 1

        if not np.all(grid[r : r1 + 1, c : c1 + 1] == BLACK):
            continue
        if np.any(grid[r - 1, c : c1 + 1] != GREY):
            continue
        if np.any(grid[r1 + 1, c : c1 + 1] != GREY):
            continue
        if np.any(grid[r : r1 + 1, c - 1] != GREY):
            continue
        if np.any(grid[r : r1 + 1, c1 + 1] != GREY):
            continue
        candidates.append((r, c, r1, c1))

    uniq = sorted(set(candidates))
    assert len(uniq) == 1, "expected exactly one enclosed rectangular cavity"
    return uniq[0]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Compress the upper jagged color bands into a vertical seed and tile that seed across the lower cavity.

    Input structure:
    - A black background grid is split by one full grey horizontal separator row into an upper band area and a lower cavity area.
    - The upper area is fully covered by 3-4 non-black, non-grey horizontal color bands with jagged boundaries, so each row has one strict majority color.
    - The lower area contains grey walls forming one rectangular black cavity whose height matches the number of compressed upper bands.

    Rule:
    - Read the upper area row by row, take each row's majority color, and compress consecutive equal majorities into a top-to-bottom color sequence.
    - Treat that sequence as an N x 1 seed tile.
    - Fill the lower cavity by repeating this seed horizontally from the cavity's top-left corner, so every cavity row becomes the corresponding sequence color.
    - Keep the upper area, the separator row, and all grey wall cells unchanged.

    Key insight:
    - The short color sequence is not drawn explicitly in the input; it must be inferred by compressing row-majority bands, and that inferred sequence is exactly the tiling seed for the cavity.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    sep = _find_separator_row(g)
    upper = g[:sep, :]
    assert upper.size > 0
    assert not np.any(upper == BLACK), "upper band area should have no black background"
    assert not np.any(upper == GREY), "upper band area should have no grey pixels"

    row_majorities = [_strict_row_majority(upper[r, :]) for r in range(upper.shape[0])]
    sequence = _compressed(row_majorities)
    assert 3 <= len(sequence) <= 4, "expected 3-4 compressed color bands"

    r0, c0, r1, c1 = _find_cavity(g, sep + 1)
    cavity_h = r1 - r0 + 1
    assert cavity_h == len(sequence), "cavity height must equal compressed band count"

    out = g.copy()
    for i, color in enumerate(sequence):
        out[r0 + i, c0 : c1 + 1] = color
    return out


def _partition_total(total: int, parts: int, minimum: int, maximum: int) -> list[int] | None:
    values = [minimum] * parts
    remaining = total - minimum * parts
    if remaining < 0:
        return None
    while remaining > 0:
        choices = [i for i, value in enumerate(values) if value < maximum]
        if not choices:
            return None
        index = random.choice(choices)
        values[index] += 1
        remaining -= 1
    random.shuffle(values)
    return values


def _generate_boundaries(thicknesses: list[int], width: int) -> np.ndarray:
    n_boundaries = len(thicknesses) - 1
    if n_boundaries == 0:
        return np.zeros((0, width), dtype=int)

    boundaries = np.zeros((n_boundaries, width), dtype=int)
    cumulative = np.cumsum(thicknesses[:-1])
    for i, nominal in enumerate(cumulative):
        limit = min(2, thicknesses[i] - 1, thicknesses[i + 1] - 1)
        offsets = [0]
        for _ in range(1, width):
            prev = offsets[-1]
            step_choices = [-1, 0, 1]
            if limit >= 2 and random.random() < 0.20:
                step_choices.extend([-2, 2])
            step = random.choice(step_choices)
            offsets.append(max(-limit, min(limit, prev + step)))
        boundaries[i, :] = nominal + np.array(offsets, dtype=int)

    for col in range(width):
        lower = 1
        for i in range(n_boundaries):
            upper = sum(thicknesses[: i + 2]) - 1
            value = int(boundaries[i, col])
            value = max(lower, min(upper, value))
            boundaries[i, col] = value
            lower = value + 1
    return boundaries


def _render_upper_band_area(colors: list[int], thicknesses: list[int], width: int) -> np.ndarray:
    height = sum(thicknesses)
    boundaries = _generate_boundaries(thicknesses, width)
    grid = np.zeros((height, width), dtype=int)
    for c in range(width):
        starts = [0] + [int(x) for x in boundaries[:, c]]
        ends = [int(x) for x in boundaries[:, c]] + [height]
        for band_index, color in enumerate(colors):
            grid[starts[band_index] : ends[band_index], c] = color
    return grid


def _valid_upper_area(grid: np.ndarray, colors: list[int]) -> bool:
    row_majorities = [_strict_row_majority(grid[r, :]) for r in range(grid.shape[0])]
    return _compressed(row_majorities) == colors


def _choose_dimensions() -> tuple[int, int, int, int, int] | None:
    band_count = random.choice([3, 4])
    upper_h = random.randint(7, 12)
    thicknesses = _partition_total(upper_h, band_count, 2, 5)
    if thicknesses is None:
        return None

    width_min = max(10, 2 * band_count + 4)
    width = random.randint(width_min, 18)

    cavity_h = band_count
    cavity_w_max = min(width - 4, 14)
    if cavity_w_max < 2 * band_count:
        return None
    cavity_w = random.randint(2 * band_count, cavity_w_max)

    return band_count, upper_h, width, cavity_h, cavity_w


def _build_example() -> np.ndarray | None:
    dims = _choose_dimensions()
    if dims is None:
        return None
    band_count, upper_h, width, cavity_h, cavity_w = dims

    colors = random.sample(PALETTE, band_count)
    thicknesses = _partition_total(upper_h, band_count, 2, 5)
    if thicknesses is None:
        return None

    upper = _render_upper_band_area(colors, thicknesses, width)
    try:
        if not _valid_upper_area(upper, colors):
            return None
    except AssertionError:
        return None

    lower_h = random.randint(cavity_h + 4, min(14, 30 - upper_h - 1))
    total_h = upper_h + 1 + lower_h
    if total_h > 30:
        return None

    grid = np.full((total_h, width), BLACK, dtype=int)
    grid[:upper_h, :] = upper
    grid[upper_h, :] = GREY

    outer_h = cavity_h + 2
    outer_w = cavity_w + 2
    if lower_h < outer_h + 2 or width < outer_w + 2:
        return None

    outer_top_local = random.randint(1, lower_h - outer_h - 1)
    outer_left = random.randint(1, width - outer_w - 1)
    outer_top = upper_h + 1 + outer_top_local

    grid[outer_top, outer_left : outer_left + outer_w] = GREY
    grid[outer_top + outer_h - 1, outer_left : outer_left + outer_w] = GREY
    grid[outer_top : outer_top + outer_h, outer_left] = GREY
    grid[outer_top : outer_top + outer_h, outer_left + outer_w - 1] = GREY

    try:
        out = main(grid)
    except AssertionError:
        return None

    cavity = _find_cavity(grid, upper_h + 1)
    if cavity[2] - cavity[0] + 1 != band_count:
        return None

    sep = _find_separator_row(grid)
    if sep != upper_h:
        return None

    if np.array_equal(out, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        grid = _build_example()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 4")
