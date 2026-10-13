from __future__ import annotations

from itertools import permutations
from typing import Sequence

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5
PINK = 6
ORANGE = 7
TEAL = 8

SEGMENT_COLORS = (RED, GREEN, YELLOW, PINK, ORANGE)
MARKER_COUNT_OPTIONS = (3, 4)
GRID_WIDTH_RANGE = (16, 25)
TOP_EXTRA_ROWS_RANGE = (3, 7)
MAX_ATTEMPTS = 400


def _find_divider_row(grid: np.ndarray) -> int:
    divider_rows = [r for r in range(grid.shape[0]) if np.all(grid[r] == GREY)]
    assert len(divider_rows) == 1, "expected exactly one full grey separator row"
    return int(divider_rows[0])


def _extract_tiles(grid: np.ndarray, divider_row: int) -> list[np.ndarray]:
    tiles: list[np.ndarray] = []
    for r in range(divider_row):
        row = grid[r]
        blue_cols = np.flatnonzero(row == BLUE)
        teal_cols = np.flatnonzero(row == TEAL)
        if blue_cols.size == 0 and teal_cols.size == 0:
            assert np.all(row == BLACK), "unmarked upper rows must be black"
            continue

        assert blue_cols.size == 1 and teal_cols.size == 1, "each marked row must have one blue and one teal marker"
        c0 = int(blue_cols[0])
        c1 = int(teal_cols[0])
        assert c0 < c1 - 1, "markers must bracket a non-empty segment"
        assert np.all(row[:c0] == BLACK) and np.all(row[c1 + 1 :] == BLACK), "outside markers must stay black"
        segment = row[c0 + 1 : c1]
        assert segment.size >= 2, "segments must have length at least two"
        assert np.all(np.isin(segment, SEGMENT_COLORS)), "segments must be colored and contain no marker colors"
        tiles.append(segment.copy())

    assert tiles, "expected at least one marked row"
    return tiles


def _find_cavities(grid: np.ndarray, divider_row: int) -> list[tuple[int, int, int]]:
    cavities: list[tuple[int, int, int]] = []
    h, w = grid.shape
    for r in range(divider_row + 1, h - 1):
        c = 0
        while c < w:
            if grid[r, c] != BLACK:
                c += 1
                continue
            start = c
            while c < w and grid[r, c] == BLACK:
                c += 1
            end = c - 1
            if (
                start > 0
                and end < w - 1
                and grid[r, start - 1] == GREY
                and grid[r, end + 1] == GREY
                and np.all(grid[r - 1, start : end + 1] == GREY)
                and np.all(grid[r + 1, start : end + 1] == GREY)
            ):
                cavities.append((r, start, end))
    assert cavities, "expected at least one cavity"
    return cavities


def _repeat_row(tile: Sequence[int], width: int) -> np.ndarray:
    tile_arr = np.asarray(tile, dtype=int)
    reps = (int(width) + tile_arr.size - 1) // tile_arr.size
    return np.tile(tile_arr, reps)[:width]


def main(grid: np.ndarray) -> np.ndarray:
    """Extract bracketed upper-row segments and use them as repeat tiles for lower one-row cavities.

    Input structure
    - A single full-width grey row separates an upper definition area from a lower cavity area.
    - In the upper area, each marked row has exactly one blue pixel left of one teal pixel, with a colored segment strictly between them; all other upper rows are black.
    - In the lower area, grey walls form several one-row-high rectangular cavities whose interiors are black.

    Rule
    - Read the marked upper rows from top to bottom and extract the pixels strictly between the blue and teal markers.
    - Read the lower cavities from top to bottom.
    - Assign the k-th extracted segment to the k-th cavity, then repeat that segment horizontally from the cavity's left edge, truncating the last copy at the right edge.
    - Leave the upper area, the grey separator row, and all grey walls unchanged.

    Key insight
    - The upper half does not directly become the output: each bracketed row segment is first reinterpreted as a one-dimensional tile, then applied by top-to-bottom order to a different region of the grid.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and max(g.shape) <= 30

    divider_row = _find_divider_row(g)
    tiles = _extract_tiles(g, divider_row)
    cavities = _find_cavities(g, divider_row)
    assert len(tiles) == len(cavities), "marked-row count must equal cavity count"

    out = g.copy()
    for tile, (r, c0, c1) in zip(tiles, cavities):
        out[r, c0 : c1 + 1] = _repeat_row(tile, c1 - c0 + 1)
    return out


def _tile_is_primitive(tile: Sequence[int]) -> bool:
    n = len(tile)
    for p in range(1, n):
        if n % p == 0 and all(tile[i] == tile[i % p] for i in range(n)):
            return False
    return True


def _make_tile(rng: np.random.Generator, length: int) -> np.ndarray:
    for _ in range(100):
        color_count = int(rng.integers(2, min(4, length + 1)))
        palette = rng.choice(SEGMENT_COLORS, size=color_count, replace=False)
        tile = rng.choice(palette, size=length, replace=True).astype(int)
        if len(set(map(int, tile))) < 2:
            continue
        if tile[0] == tile[-1] and len(tile) > 2 and rng.random() < 0.4:
            continue
        if _tile_is_primitive(tile):
            return tile
    raise AssertionError("failed to generate a primitive tile")


def _filled_rows_unique(tiles: list[np.ndarray], widths: list[int]) -> bool:
    rows = [tuple(map(int, _repeat_row(tile, width))) for tile, width in zip(tiles, widths)]
    return len(set(rows)) == len(rows)


def _mapping_is_unique(tiles: list[np.ndarray], widths: list[int]) -> bool:
    target = [tuple(map(int, _repeat_row(tile, width))) for tile, width in zip(tiles, widths)]
    indices = tuple(range(len(tiles)))
    for perm in permutations(indices):
        if perm == indices:
            continue
        candidate = [tuple(map(int, _repeat_row(tiles[perm[i]], widths[i]))) for i in indices]
        if candidate == target:
            return False
    return True


def _nontrivial_size_order(tile_lengths: list[int], widths: list[int]) -> bool:
    tile_rank = tuple(int(i) for i in np.argsort(tile_lengths))
    width_rank = tuple(int(i) for i in np.argsort(widths))
    return tile_rank != width_rank and tile_rank != width_rank[::-1]


def _choose_tiles_and_widths(rng: np.random.Generator, count: int) -> tuple[list[np.ndarray], list[int]] | None:
    base_lengths = rng.choice(np.array([2, 3, 4, 5]), size=count, replace=False)
    order = rng.permutation(count)
    lengths = [int(base_lengths[i]) for i in order]

    tiles: list[np.ndarray] = []
    seen_tiles: set[tuple[int, ...]] = set()
    for length in lengths:
        for _ in range(100):
            tile = _make_tile(rng, length)
            key = tuple(map(int, tile))
            if key not in seen_tiles:
                seen_tiles.add(key)
                tiles.append(tile)
                break
        else:
            return None

    for _ in range(300):
        widths = rng.choice(np.arange(5, 13), size=count, replace=False).astype(int).tolist()
        if any(widths[i] <= tiles[i].size for i in range(count)):
            continue
        divisible = [widths[i] % tiles[i].size == 0 for i in range(count)]
        if not any(divisible) or all(divisible):
            continue
        if not _nontrivial_size_order([tile.size for tile in tiles], widths):
            continue
        if not _filled_rows_unique(tiles, widths):
            continue
        if not _mapping_is_unique(tiles, widths):
            continue
        return tiles, widths
    return None


def _marker_rows(rng: np.random.Generator, top_h: int, count: int) -> list[int]:
    rows = sorted(rng.choice(np.arange(top_h), size=count, replace=False).astype(int).tolist())
    if top_h - count >= 1:
        return rows
    raise AssertionError("top area must contain at least one unmarked row")


def _blue_columns(rng: np.random.Generator, width: int, tiles: list[np.ndarray]) -> list[int]:
    cols = []
    for tile in tiles:
        max_blue = width - tile.size - 2
        cols.append(int(rng.integers(1, max_blue + 1)))
    if len(set(cols)) == 1 and len(cols) > 1:
        idx = int(rng.integers(0, len(cols)))
        tile = tiles[idx]
        max_blue = width - tile.size - 2
        alternatives = [c for c in range(1, max_blue + 1) if c != cols[idx]]
        if alternatives:
            cols[idx] = int(rng.choice(alternatives))
    return cols


def _build_planned_output(inp: np.ndarray, cavity_specs: list[tuple[int, int, int]], tiles: list[np.ndarray]) -> np.ndarray:
    out = inp.copy()
    for tile, (r, c0, width) in zip(tiles, cavity_specs):
        out[r, c0 : c0 + width] = _repeat_row(tile, width)
    return out


def _semantic_checks(inp: np.ndarray, expected: np.ndarray) -> bool:
    try:
        produced = main(inp)
    except Exception:
        return False
    if not np.array_equal(produced, expected):
        return False

    divider = _find_divider_row(inp)
    tiles = _extract_tiles(inp, divider)
    cavities = _find_cavities(inp, divider)
    if len(tiles) != len(cavities) or len(tiles) not in MARKER_COUNT_OPTIONS:
        return False
    if len(set(int(c1 - c0 + 1) for _, c0, c1 in cavities)) != len(cavities):
        return False

    widths = [int(c1 - c0 + 1) for _, c0, c1 in cavities]
    tile_lengths = [int(tile.size) for tile in tiles]
    if any(width <= tile_len for width, tile_len in zip(widths, tile_lengths)):
        return False
    if not any(width % tile_len == 0 for width, tile_len in zip(widths, tile_lengths)):
        return False
    if not any(width % tile_len != 0 for width, tile_len in zip(widths, tile_lengths)):
        return False
    if not _nontrivial_size_order(tile_lengths, widths):
        return False
    if not _filled_rows_unique(tiles, widths):
        return False
    if not _mapping_is_unique(tiles, widths):
        return False

    marker_blue_cols = [int(np.flatnonzero(inp[r] == BLUE)[0]) for r in range(divider) if np.any(inp[r] == BLUE)]
    if len(marker_blue_cols) > 1 and len(set(marker_blue_cols)) == 1:
        return False

    for r in range(divider):
        row = inp[r]
        if np.any(row == BLUE):
            blue = int(np.flatnonzero(row == BLUE)[0])
            teal = int(np.flatnonzero(row == TEAL)[0])
            if not np.all(row[blue + 1 : teal] != BLACK):
                return False
        elif not np.all(row == BLACK):
            return False

    return True


def _attempt_generate(rng: np.random.Generator) -> np.ndarray | None:
    count = int(rng.choice(MARKER_COUNT_OPTIONS))
    tile_and_widths = _choose_tiles_and_widths(rng, count)
    if tile_and_widths is None:
        return None
    tiles, cavity_widths = tile_and_widths

    grid_w = int(rng.integers(*GRID_WIDTH_RANGE))
    if grid_w < max(max(cavity_widths) + 6, max(tile.size for tile in tiles) + 4):
        return None

    top_h = count + int(rng.integers(*TOP_EXTRA_ROWS_RANGE))
    marker_rows = _marker_rows(rng, top_h, count)
    blue_cols = _blue_columns(rng, grid_w, tiles)

    cavity_specs_local: list[tuple[int, int, int]] = []
    y = int(rng.integers(0, 2))
    base_left_max = grid_w - max(cavity_widths) - 3
    if base_left_max < 2:
        return None
    base_left = int(rng.integers(2, base_left_max + 1))
    for width in cavity_widths:
        jitter_choices = []
        for delta in (-1, 0, 1):
            left = base_left + delta
            if 2 <= left <= grid_w - width - 3:
                jitter_choices.append(left)
        left = int(rng.choice(jitter_choices))
        cavity_specs_local.append((y + 1, left, width))
        y += 3 + int(rng.integers(0, 2))
    bottom_h = y + int(rng.integers(0, 2))

    total_h = top_h + 1 + bottom_h
    if total_h > 30:
        return None

    grid = np.zeros((total_h, grid_w), dtype=int)
    divider = top_h
    grid[divider, :] = GREY

    for row, blue_col, tile in zip(marker_rows, blue_cols, tiles):
        teal_col = blue_col + tile.size + 1
        grid[row, blue_col] = BLUE
        grid[row, blue_col + 1 : teal_col] = tile
        grid[row, teal_col] = TEAL

    cavity_specs_global: list[tuple[int, int, int]] = []
    for local_mid, left, width in cavity_specs_local:
        mid = divider + 1 + local_mid
        top = mid - 1
        bottom = mid + 1
        grid[top, left - 1 : left + width + 1] = GREY
        grid[mid, left - 1] = GREY
        grid[mid, left + width] = GREY
        grid[bottom, left - 1 : left + width + 1] = GREY
        cavity_specs_global.append((mid, left, width))

    expected = _build_planned_output(grid, cavity_specs_global, tiles)
    if not _semantic_checks(grid, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(MAX_ATTEMPTS):
        grid = _attempt_generate(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid puzzle instance")
