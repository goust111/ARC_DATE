from __future__ import annotations

import random
from collections import Counter

import numpy as np


PALETTE = tuple(range(10))
GEN_ATTEMPTS = 400


def _runs(indices: list[int]) -> list[tuple[int, int]]:
    if not indices:
        return []

    out: list[tuple[int, int]] = []
    start = prev = int(indices[0])
    for value in indices[1:]:
        value = int(value)
        if value == prev + 1:
            prev = value
        else:
            out.append((start, prev))
            start = prev = value
    out.append((start, prev))
    return out


def _tile_intervals(length: int, separator_runs: list[tuple[int, int]]) -> list[tuple[int, int]]:
    intervals: list[tuple[int, int]] = []
    cursor = 0
    for start, end in separator_runs:
        assert start >= cursor, "separator runs must not overlap"
        if start > cursor:
            intervals.append((cursor, start - 1))
        cursor = end + 1
    if cursor < length:
        intervals.append((cursor, length - 1))

    assert intervals, "at least one tile interval is required"
    sizes = {end - start + 1 for start, end in intervals}
    assert len(sizes) == 1, "all tiles on an axis must have equal size"
    return intervals


def _find_lattice(grid: np.ndarray) -> tuple[int, int, list[tuple[int, int]], list[tuple[int, int]], set[int], set[int]]:
    height, width = map(int, grid.shape)
    candidates = []

    for color in map(int, np.unique(grid)):
        full_rows = [r for r in range(height) if np.all(grid[r, :] == color)]
        full_cols = [c for c in range(width) if np.all(grid[:, c] == color)]
        if not full_rows or not full_cols:
            continue

        row_runs = _runs(full_rows)
        col_runs = _runs(full_cols)
        row_widths = {end - start + 1 for start, end in row_runs}
        col_widths = {end - start + 1 for start, end in col_runs}
        if len(row_widths) != 1 or len(col_widths) != 1:
            continue

        row_intervals = _tile_intervals(height, row_runs)
        col_intervals = _tile_intervals(width, col_runs)
        if len(row_intervals) < 3 or len(col_intervals) < 3:
            continue
        if len(row_intervals) % 2 == 0 or len(col_intervals) % 2 == 0:
            continue

        sep_rows = set(full_rows)
        sep_cols = set(full_cols)
        sep_mask = np.zeros(grid.shape, dtype=bool)
        sep_mask[full_rows, :] = True
        sep_mask[:, full_cols] = True
        if not np.array_equal(grid == color, sep_mask):
            continue

        interior_values = grid[~sep_mask]
        assert interior_values.size > 0
        counts = Counter(map(int, interior_values.tolist()))
        background = counts.most_common(1)[0][0]
        if int(background) == color:
            continue

        candidates.append((color, int(background), row_intervals, col_intervals, sep_rows, sep_cols))

    assert len(candidates) == 1, "expected one unambiguous separator lattice"
    return candidates[0]


def _markers(
    grid: np.ndarray,
    background: int,
    separator: int,
    row_intervals: list[tuple[int, int]],
    col_intervals: list[tuple[int, int]],
) -> list[tuple[int, int, int, int, int]]:
    markers: list[tuple[int, int, int, int, int]] = []
    for tile_r, (r0, r1) in enumerate(row_intervals):
        for tile_c, (c0, c1) in enumerate(col_intervals):
            tile = grid[r0 : r1 + 1, c0 : c1 + 1]
            assert not np.any(tile == separator), "separator color cannot appear inside a tile"
            coords = np.argwhere(tile != background)
            assert len(coords) <= 1, "each input tile may contain at most one marker"
            if len(coords) == 0:
                continue
            dr, dc = map(int, coords[0])
            color = int(tile[dr, dc])
            markers.append((tile_r, tile_c, dr, dc, color))

    assert markers, "at least one marker is required"
    return markers


def main(grid: np.ndarray) -> np.ndarray:
    """
    Complete marker pixels by 180-degree point symmetry in tile coordinates.

    Input structure:
    - A single separator color forms full horizontal and vertical lines that split
      the grid into an odd-by-odd lattice of equal-size tiles.
    - Tile interiors share one background color.
    - Some tile interiors contain exactly one non-background marker pixel, with
      marker offsets inside tiles varying from tile to tile.

    Rule:
    - Keep the grid shape, background, separator lines, and existing markers.
    - For each marker, reflect only its tile row and tile column through the center
      tile of the lattice.
    - Paint the same marker color in the reflected tile at the same local row and
      column offset inside that tile.

    Key insight:
    - The 180-degree symmetry acts on tile indices, not on raw pixel coordinates;
      the local offset inside each tile is copied unchanged rather than rotated.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    separator, background, row_intervals, col_intervals, _sep_rows, _sep_cols = _find_lattice(g)
    marks = _markers(g, background, separator, row_intervals, col_intervals)

    out = g.copy()
    row_count = len(row_intervals)
    col_count = len(col_intervals)

    for tile_r, tile_c, dr, dc, color in marks:
        reflected_r = row_count - 1 - tile_r
        reflected_c = col_count - 1 - tile_c
        target_r = row_intervals[reflected_r][0] + dr
        target_c = col_intervals[reflected_c][0] + dc

        current = int(out[target_r, target_c])
        assert current in (background, color), "conflicting marker at reflected tile offset"
        out[target_r, target_c] = color

    return out


def _render_lattice(
    tile_rows: int,
    tile_cols: int,
    tile_h: int,
    tile_w: int,
    background: int,
    separator: int,
) -> tuple[np.ndarray, list[tuple[int, int]], list[tuple[int, int]]]:
    height = tile_rows * tile_h + tile_rows - 1
    width = tile_cols * tile_w + tile_cols - 1
    grid = np.full((height, width), int(background), dtype=int)

    for boundary in range(1, tile_rows):
        row = boundary * tile_h + boundary - 1
        grid[row, :] = int(separator)
    for boundary in range(1, tile_cols):
        col = boundary * tile_w + boundary - 1
        grid[:, col] = int(separator)

    row_intervals = []
    for tile_r in range(tile_rows):
        start = tile_r * (tile_h + 1)
        row_intervals.append((start, start + tile_h - 1))

    col_intervals = []
    for tile_c in range(tile_cols):
        start = tile_c * (tile_w + 1)
        col_intervals.append((start, start + tile_w - 1))

    return grid, row_intervals, col_intervals


def _pixel_level_completion(grid: np.ndarray) -> np.ndarray:
    separator, background, row_intervals, col_intervals, _sep_rows, _sep_cols = _find_lattice(grid)
    marks = _markers(grid, background, separator, row_intervals, col_intervals)
    out = grid.copy()
    height, width = map(int, grid.shape)

    for tile_r, tile_c, dr, dc, color in marks:
        r = row_intervals[tile_r][0] + dr
        c = col_intervals[tile_c][0] + dc
        rr = height - 1 - r
        cc = width - 1 - c
        current = int(out[rr, cc])
        if current in (background, color):
            out[rr, cc] = color
    return out


def _choose_dimensions(rng: random.Random) -> tuple[int, int, int, int]:
    choices: list[tuple[int, int, int, int]] = []
    for tile_rows in (3, 5, 7):
        for tile_cols in (3, 5, 7):
            for tile_h in range(3, 7):
                for tile_w in range(3, 7):
                    height = tile_rows * tile_h + tile_rows - 1
                    width = tile_cols * tile_w + tile_cols - 1
                    if height <= 30 and width <= 30:
                        choices.append((tile_rows, tile_cols, tile_h, tile_w))
    return rng.choice(choices)


def _orbit_pairs(tile_rows: int, tile_cols: int) -> list[tuple[tuple[int, int], tuple[int, int]]]:
    pairs = []
    seen: set[tuple[int, int]] = set()
    for tile_r in range(tile_rows):
        for tile_c in range(tile_cols):
            here = (tile_r, tile_c)
            there = (tile_rows - 1 - tile_r, tile_cols - 1 - tile_c)
            if here == there or here in seen or there in seen:
                continue
            pairs.append((here, there))
            seen.add(here)
            seen.add(there)
    return pairs


def _non_pixel_rotating_offsets(tile_h: int, tile_w: int) -> list[tuple[int, int]]:
    return [
        (dr, dc)
        for dr in range(tile_h)
        for dc in range(tile_w)
        if (dr, dc) != (tile_h - 1 - dr, tile_w - 1 - dc)
    ]


def _place_marker(
    grid: np.ndarray,
    row_intervals: list[tuple[int, int]],
    col_intervals: list[tuple[int, int]],
    tile: tuple[int, int],
    offset: tuple[int, int],
    color: int,
) -> None:
    tile_r, tile_c = tile
    dr, dc = offset
    r = row_intervals[tile_r][0] + dr
    c = col_intervals[tile_c][0] + dc
    grid[r, c] = int(color)


def _expected_completion(grid: np.ndarray) -> np.ndarray:
    return main(grid)


def _gen_attempt() -> np.ndarray | None:
    rng = random.Random()
    tile_rows, tile_cols, tile_h, tile_w = _choose_dimensions(rng)
    colors = list(PALETTE)
    rng.shuffle(colors)
    background, separator = colors[0], colors[1]
    marker_colors = colors[2:]

    grid, row_intervals, col_intervals = _render_lattice(
        tile_rows, tile_cols, tile_h, tile_w, background, separator
    )

    pairs = _orbit_pairs(tile_rows, tile_cols)
    rng.shuffle(pairs)

    pair_limit = min(len(pairs), rng.randint(2, 6))
    selected_pairs = pairs[:pair_limit]
    incomplete_count = rng.randint(1, max(1, pair_limit))
    incomplete_indices = set(rng.sample(range(pair_limit), incomplete_count))
    offsets_for_anti_shortcut = _non_pixel_rotating_offsets(tile_h, tile_w)
    if not offsets_for_anti_shortcut:
        return None

    for idx, (a, b) in enumerate(selected_pairs):
        color = rng.choice(marker_colors)
        if idx in incomplete_indices:
            source = a if rng.random() < 0.5 else b
            offset = rng.choice(offsets_for_anti_shortcut)
            _place_marker(grid, row_intervals, col_intervals, source, offset, color)
        else:
            offset = (rng.randrange(tile_h), rng.randrange(tile_w))
            _place_marker(grid, row_intervals, col_intervals, a, offset, color)
            _place_marker(grid, row_intervals, col_intervals, b, offset, color)

    if rng.random() < 0.35:
        center = (tile_rows // 2, tile_cols // 2)
        occupied = grid[
            row_intervals[center[0]][0] : row_intervals[center[0]][1] + 1,
            col_intervals[center[1]][0] : col_intervals[center[1]][1] + 1,
        ]
        if np.all(occupied == background):
            offset = (rng.randrange(tile_h), rng.randrange(tile_w))
            _place_marker(grid, row_intervals, col_intervals, center, offset, rng.choice(marker_colors))

    try:
        expected = _expected_completion(grid)
        pixel_rotated = _pixel_level_completion(grid)
    except AssertionError:
        return None

    if np.array_equal(grid, expected):
        return None
    if np.array_equal(pixel_rotated, expected):
        return None
    if not np.array_equal(main(grid), expected):
        return None
    if max(grid.shape) > 30 or max(expected.shape) > 30:
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid tile-coordinate point-symmetry input")
