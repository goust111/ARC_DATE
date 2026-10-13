from __future__ import annotations

import random
from collections import Counter

import numpy as np


BLACK = 0
GREEN = 3
TEAL = 8

MARKER_COLORS = tuple(color for color in range(1, 10) if color not in (GREEN, TEAL))

HEIGHT_RANGE = (7, 12)
WIDTH_RANGE = (6, 10)
MARKER_COUNT_CHOICES = (2, 3)
MAX_GENERATION_ATTEMPTS = 600


def _sorted_markers(grid: np.ndarray) -> list[tuple[int, int, int]]:
    marker_coords = np.argwhere((grid != BLACK) & (grid != GREEN))
    markers = [(int(r), int(c), int(grid[r, c])) for r, c in marker_coords]
    assert 2 <= len(markers) <= 3, "expected 2 or 3 singleton markers"

    rows = [row for row, _col, _color in markers]
    colors = [color for _row, _col, color in markers]
    assert len(rows) == len(set(rows)), "marker rows must be unique"
    assert len(colors) == len(set(colors)), "marker colors must be unique"
    assert all(Counter(colors)[color] == 1 for color in colors), "markers must be singletons"
    markers.sort(key=lambda item: item[0])
    return markers


def _band_ranges(marker_rows: list[int], height: int, *, use_floor: bool = True) -> list[tuple[int, int]]:
    assert marker_rows == sorted(marker_rows)
    boundaries = []
    for row_a, row_b in zip(marker_rows, marker_rows[1:]):
        offset = 0 if use_floor else 1
        boundaries.append((row_a + row_b + offset) // 2)

    bands = []
    start = 0
    for boundary in boundaries:
        bands.append((start, boundary))
        start = boundary + 1
    bands.append((start, height - 1))
    return bands


def _band_column_mask(grid: np.ndarray, band: tuple[int, int]) -> np.ndarray:
    start, end = band
    return np.any(grid[start : end + 1] == GREEN, axis=0)


def _paint_tile(grid: np.ndarray, bands: list[tuple[int, int]]) -> np.ndarray:
    tile = np.array(grid, dtype=int, copy=True)
    for start, end in bands:
        selected_cols = _band_column_mask(grid, (start, end))
        band_view = tile[start : end + 1]
        band_view[(band_view == BLACK) & selected_cols[np.newaxis, :]] = TEAL
    return tile


def _global_column_tile(grid: np.ndarray) -> np.ndarray:
    tile = np.array(grid, dtype=int, copy=True)
    selected_cols = np.any(grid == GREEN, axis=0)
    tile[(tile == BLACK) & selected_cols[np.newaxis, :]] = TEAL
    return tile


def main(grid: np.ndarray) -> np.ndarray:
    """
    Split the grid into marker-defined horizontal bands, recolor background cells in band-local green-bearing columns, then tile the result 2x2.

    Input structure:
    - The input is a black-background grid containing 2 or 3 singleton colored markers.
    - Marker colors are distinct, markers lie on distinct rows, and sparse green pixels are the only other foreground.
    - The marker rows define horizontal-band ownership; marker columns do not define the bands.

    Rule:
    - Sort the markers by row and split the grid at the floor midpoint between consecutive marker rows.
    - In each band independently, find every column containing at least one green pixel inside that band's row range.
    - Recolor black cells in those selected columns to teal within that band only, keep markers and green pixels unchanged, then tile the completed grid twice vertically and twice horizontally.

    Key insight:
    - Column selection is local to each marker-defined band, so the same column can be teal in one band and remain black in another before the final 2x2 tiling.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"

    markers = _sorted_markers(g)
    marker_rows = [row for row, _col, _color in markers]
    bands = _band_ranges(marker_rows, int(g.shape[0]), use_floor=True)
    tile = _paint_tile(g, bands)
    return np.tile(tile, (2, 2))


def _interior_fill_rows(
    band: tuple[int, int], height: int, marker_rows: set[int]
) -> list[int]:
    start, end = band
    return [row for row in range(start, end + 1) if 1 <= row <= height - 2 and row not in marker_rows]


def _band_sets_from_profiles(profiles: list[tuple[bool, ...]]) -> list[set[int]]:
    band_count = len(profiles[0])
    sets = [set() for _ in range(band_count)]
    for col, profile in enumerate(profiles):
        for band_index, present in enumerate(profile):
            if present:
                sets[band_index].add(col)
    return sets


def _column_profiles(grid: np.ndarray, bands: list[tuple[int, int]]) -> list[tuple[bool, ...]]:
    profiles = []
    for col in range(grid.shape[1]):
        profiles.append(tuple(bool(np.any(grid[start : end + 1, col] == GREEN)) for start, end in bands))
    return profiles


def _semantic_checks(grid: np.ndarray) -> bool:
    height, width = map(int, grid.shape)
    if max(height, width) > 15:
        return False

    markers = _sorted_markers(grid)
    marker_rows = [row for row, _col, _color in markers]
    if any(row in (0, height - 1) for row in marker_rows):
        return False

    green_coords = np.argwhere(grid == GREEN)
    if green_coords.size == 0:
        return False
    if any(int(r) in (0, height - 1) for r, _c in green_coords):
        return False

    floor_bands = _band_ranges(marker_rows, height, use_floor=True)
    fillable_rows = [_interior_fill_rows(band, height, set(marker_rows)) for band in floor_bands]
    if any(not rows for rows in fillable_rows):
        return False

    profiles = _column_profiles(grid, floor_bands)
    if not any(0 < sum(profile) < len(profile) for profile in profiles):
        return False
    if not any(sum(profile) == 1 for profile in profiles):
        return False

    band_sets = _band_sets_from_profiles(profiles)
    if any(len(cols) == 0 or len(cols) == width for cols in band_sets):
        return False
    if len({tuple(sorted(cols)) for cols in band_sets}) == 1:
        return False

    floor_tile = _paint_tile(grid, floor_bands)
    global_tile = _global_column_tile(grid)
    if np.array_equal(floor_tile, global_tile):
        return False

    ceil_bands = _band_ranges(marker_rows, height, use_floor=False)
    if ceil_bands == floor_bands:
        return False
    ceil_tile = _paint_tile(grid, ceil_bands)
    if np.array_equal(floor_tile, ceil_tile):
        return False

    predicted = main(grid)
    expected = np.tile(floor_tile, (2, 2))
    return bool(np.array_equal(predicted, expected))


def _sample_marker_rows(height: int, marker_count: int) -> list[int] | None:
    interior_rows = list(range(1, height - 1))
    for _ in range(200):
        rows = sorted(random.sample(interior_rows, marker_count))
        gaps = [rows[i + 1] - rows[i] for i in range(marker_count - 1)]
        if min(gaps) < 2:
            continue
        if not any(gap >= 3 and gap % 2 == 1 for gap in gaps):
            continue
        bands = _band_ranges(rows, height, use_floor=True)
        if all(_interior_fill_rows(band, height, set(rows)) for band in bands):
            return rows
    return None


def _seed_band_columns(
    width: int, band_count: int, sensitive_band: int
) -> tuple[list[set[int]], int]:
    selected = [set() for _ in range(band_count)]
    all_cols = list(range(width))
    sensitive_col = random.choice(all_cols)
    selected[sensitive_band].add(sensitive_col)

    remaining_cols = [col for col in all_cols if col != sensitive_col]
    if remaining_cols and random.random() < 0.7:
        shared_col = random.choice(remaining_cols)
        shared_bands = random.sample(range(band_count), random.randint(1, band_count))
        for band_index in shared_bands:
            selected[band_index].add(shared_col)

    for band_index in range(band_count):
        if selected[band_index]:
            continue
        choices = [col for col in all_cols if col not in selected[band_index]]
        selected[band_index].add(random.choice(choices))

    max_cols = width - 1
    for band_index in range(band_count):
        extras = [col for col in all_cols if col not in selected[band_index]]
        random.shuffle(extras)
        for col in extras:
            if len(selected[band_index]) >= max_cols:
                break
            if random.random() < 0.35:
                selected[band_index].add(col)
    return selected, sensitive_col


def _build_candidate() -> np.ndarray | None:
    height = random.randint(*HEIGHT_RANGE)
    width = random.randint(*WIDTH_RANGE)
    marker_count = random.choice(MARKER_COUNT_CHOICES)

    marker_rows = _sample_marker_rows(height, marker_count)
    if marker_rows is None:
        return None
    floor_bands = _band_ranges(marker_rows, height, use_floor=True)

    odd_gap_indices = [
        index
        for index, (row_a, row_b) in enumerate(zip(marker_rows, marker_rows[1:]))
        if (row_b - row_a) >= 3 and (row_b - row_a) % 2 == 1
    ]
    if not odd_gap_indices:
        return None

    sensitive_gap = random.choice(odd_gap_indices)
    sensitive_band = sensitive_gap + 1
    sensitive_row = (marker_rows[sensitive_gap] + marker_rows[sensitive_gap + 1]) // 2 + 1

    selected_cols, sensitive_col = _seed_band_columns(width, marker_count, sensitive_band)
    if any(len(cols) == 0 or len(cols) >= width for cols in selected_cols):
        return None
    if not all(sensitive_col not in selected_cols[band] for band in range(marker_count) if band != sensitive_band):
        return None

    marker_colors = random.sample(list(MARKER_COLORS), marker_count)
    marker_col_choices = list(range(1, width - 1)) if width >= marker_count + 2 else list(range(width))
    if len(marker_col_choices) < marker_count:
        marker_col_choices = list(range(width))
    marker_cols = random.sample(marker_col_choices, marker_count)

    grid = np.zeros((height, width), dtype=int)
    for row, col, color in zip(marker_rows, marker_cols, marker_colors):
        grid[row, col] = color

    fill_rows_by_band = [
        _interior_fill_rows(band, height, set(marker_rows)) for band in floor_bands
    ]
    if sensitive_row not in fill_rows_by_band[sensitive_band]:
        return None

    grid[sensitive_row, sensitive_col] = GREEN
    for band_index, cols in enumerate(selected_cols):
        rows = fill_rows_by_band[band_index]
        for col in sorted(cols):
            if band_index == sensitive_band and col == sensitive_col:
                continue
            row = random.choice(rows)
            if grid[row, col] != BLACK:
                alt_rows = [candidate for candidate in rows if grid[candidate, col] == BLACK]
                if not alt_rows:
                    return None
                row = random.choice(alt_rows)
            grid[row, col] = GREEN

            if len(rows) >= 2 and random.random() < 0.3:
                extra_rows = [candidate for candidate in rows if grid[candidate, col] == BLACK]
                if extra_rows:
                    grid[random.choice(extra_rows), col] = GREEN

    if np.any((grid != BLACK) & (grid != GREEN) & (grid == TEAL)):
        return None
    return grid if _semantic_checks(grid) else None


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        candidate = _build_candidate()
        if candidate is not None:
            return candidate
    raise RuntimeError(f"generate_input failed after {MAX_GENERATION_ATTEMPTS} attempts")
