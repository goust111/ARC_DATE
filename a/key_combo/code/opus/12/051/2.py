from __future__ import annotations

import random
from itertools import combinations

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
ORANGE = 7
TEAL = 8

GEN_ATTEMPTS = 500
ROW_BAND_OPTIONS = (3, 4)
COL_BAND_OPTIONS = (3, 4)
BAND_SIZE_MIN = 2
BAND_SIZE_MAX = 6


def _bands_from_dividers(length: int, dividers: list[int]) -> list[tuple[int, int]]:
    bands: list[tuple[int, int]] = []
    prev = -1
    for divider in dividers + [length]:
        start = prev + 1
        end = divider - 1
        if start <= end:
            bands.append((start, end))
        prev = divider
    return bands


def _band_index(pos: int, bands: list[tuple[int, int]]) -> int:
    for idx, (start, end) in enumerate(bands):
        if start <= pos <= end:
            return idx
    raise AssertionError(f"position {pos} is not inside any band")


def _divider_rows(grid: np.ndarray) -> list[int]:
    return [int(r) for r in range(grid.shape[0]) if np.all(grid[r, :] == RED)]


def _divider_cols(grid: np.ndarray) -> list[int]:
    return [int(c) for c in range(grid.shape[1]) if np.all(grid[:, c] == RED)]


def _parse_grid(grid: np.ndarray) -> dict:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    height, width = g.shape
    assert height <= 30 and width <= 30, "ARC grids must fit within 30x30"

    row_dividers = _divider_rows(g)
    col_dividers = _divider_cols(g)
    assert row_dividers and col_dividers, "expected both horizontal and vertical red dividers"
    assert 0 not in row_dividers and height - 1 not in row_dividers, "outer border cannot be a divider row"
    assert 0 not in col_dividers and width - 1 not in col_dividers, "outer border cannot be a divider column"

    row_bands = _bands_from_dividers(height, row_dividers)
    col_bands = _bands_from_dividers(width, col_dividers)
    assert len(row_bands) >= 2 and len(col_bands) >= 3, "expected a genuine rectangular partition"

    blue_cols = [int(c) for c in np.flatnonzero(g[0, :] == BLUE)]
    green_rows = [int(r) for r in np.flatnonzero(g[:, width - 1] == GREEN)]
    assert 2 <= len(blue_cols) <= 4, "expected 2-4 blue top markers"
    assert 2 <= len(green_rows) <= 3, "expected 2-3 green right-edge markers"
    assert all(c != width - 1 for c in blue_cols), "blue markers must stay off the top-right corner"

    selected_col_bands = sorted({_band_index(c, col_bands) for c in blue_cols})
    selected_row_bands = sorted({_band_index(r, row_bands) for r in green_rows})
    assert len(selected_col_bands) == len(blue_cols), "blue markers must target distinct column bands"
    assert len(selected_row_bands) == len(green_rows), "green markers must target distinct row bands"

    cell_boxes: dict[tuple[int, int], tuple[int, int, int, int]] = {}
    areas: dict[tuple[int, int], int] = {}
    for rb, (r0, r1) in enumerate(row_bands):
        for cb, (c0, c1) in enumerate(col_bands):
            cell_boxes[(rb, cb)] = (r0, r1, c0, c1)
            cell = g[r0 : r1 + 1, c0 : c1 + 1]
            allowed = {BLACK}
            if rb == 0:
                allowed.add(BLUE)
            if cb == len(col_bands) - 1:
                allowed.add(GREEN)
            assert set(map(int, np.unique(cell))) <= allowed, "cells may contain only black plus edge markers"
            areas[(rb, cb)] = int(np.count_nonzero(cell == BLACK))

    selected_cells = [(rb, cb) for rb in selected_row_bands for cb in selected_col_bands]
    assert len(selected_cells) >= 3, "Cartesian product must select at least 3 cells"

    return {
        "grid": g,
        "row_bands": row_bands,
        "col_bands": col_bands,
        "selected_rows": selected_row_bands,
        "selected_cols": selected_col_bands,
        "selected_cells": selected_cells,
        "areas": areas,
        "cell_boxes": cell_boxes,
    }


def _selected_extrema(parsed: dict) -> tuple[tuple[int, int], tuple[int, int]]:
    selected = parsed["selected_cells"]
    areas = parsed["areas"]
    selected_areas = [areas[cell] for cell in selected]
    min_area = min(selected_areas)
    max_area = max(selected_areas)
    assert selected_areas.count(min_area) == 1, "selected minimum area must be unique"
    assert selected_areas.count(max_area) == 1, "selected maximum area must be unique"
    min_cell = next(cell for cell in selected if areas[cell] == min_area)
    max_cell = next(cell for cell in selected if areas[cell] == max_area)
    return min_cell, max_cell


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor the selected partition cells so the unique smallest chosen cell becomes teal and the unique largest becomes orange.

    Input structure:
    - Black rectangular cells are partitioned by complete one-pixel red divider rows and columns.
    - Blue markers appear on the top row and indicate selected column bands.
    - Green markers appear on the rightmost column and indicate selected row bands.

    Rule:
    - Determine the row bands and column bands cut out by the red divider lines.
    - Take the Cartesian product of the blue-selected column bands and green-selected row bands.
    - Count black pixels inside each selected cell only; fill the unique smallest selected cell teal,
      the unique largest selected cell orange, and every other selected cell yellow.
    - Leave all unselected cells black, and preserve the red dividers and edge markers.

    Key insight:
    - The edge markers do not point to single intersections; they select row and column partitions, and the
      area comparison happens only inside that selected Cartesian subset rather than across the whole grid.
    """
    parsed = _parse_grid(grid)
    min_cell, max_cell = _selected_extrema(parsed)

    out = parsed["grid"].copy()
    for cell in parsed["selected_cells"]:
        color = YELLOW
        if cell == min_cell:
            color = TEAL
        elif cell == max_cell:
            color = ORANGE
        r0, r1, c0, c1 = parsed["cell_boxes"][cell]
        region = out[r0 : r1 + 1, c0 : c1 + 1]
        region[region == BLACK] = color
    return out


def _cell_areas_from_sizes(
    row_sizes: list[int],
    col_sizes: list[int],
    selected_rows: tuple[int, ...],
    selected_cols: tuple[int, ...],
) -> dict[tuple[int, int], int]:
    areas: dict[tuple[int, int], int] = {}
    last_col = len(col_sizes) - 1
    for rb, row_size in enumerate(row_sizes):
        for cb, col_size in enumerate(col_sizes):
            area = row_size * col_size
            if rb == 0 and cb in selected_cols:
                area -= 1
            if cb == last_col and rb in selected_rows:
                area -= 1
            areas[(rb, cb)] = area
    return areas


def _candidate_selections(row_sizes: list[int], col_sizes: list[int]) -> list[dict]:
    rb = len(row_sizes)
    cb = len(col_sizes)
    candidates: list[dict] = []
    all_cells = [(r, c) for r in range(rb) for c in range(cb)]
    for row_count in range(2, min(3, rb) + 1):
        for selected_rows in combinations(range(rb), row_count):
            for col_count in range(2, min(4, cb) + 1):
                for selected_cols in combinations(range(cb), col_count):
                    selected_cells = [(r, c) for r in selected_rows for c in selected_cols]
                    if len(selected_cells) < 3:
                        continue
                    areas = _cell_areas_from_sizes(row_sizes, col_sizes, selected_rows, selected_cols)
                    selected_areas = [areas[cell] for cell in selected_cells]
                    min_area = min(selected_areas)
                    max_area = max(selected_areas)
                    if selected_areas.count(min_area) != 1 or selected_areas.count(max_area) != 1:
                        continue
                    global_areas = [areas[cell] for cell in all_cells]
                    if len(set(global_areas)) < 3:
                        continue
                    unselected_cells = [cell for cell in all_cells if cell not in selected_cells]
                    unselected_areas = [areas[cell] for cell in unselected_cells]
                    if not unselected_areas:
                        continue
                    if min(unselected_areas) >= min_area:
                        continue
                    if max(unselected_areas) <= max_area:
                        continue
                    if len(set(selected_areas)) < 3:
                        continue
                    candidates.append(
                        {
                            "rows": selected_rows,
                            "cols": selected_cols,
                            "selected_cells": selected_cells,
                            "areas": areas,
                        }
                    )
    return candidates


def _random_band_sizes(count: int) -> list[int]:
    while True:
        sizes = [random.randint(BAND_SIZE_MIN, BAND_SIZE_MAX) for _ in range(count)]
        if len(set(sizes)) >= 2:
            return sizes


def _weighted_candidate_choice(candidates: list[dict], n_cols: int) -> dict:
    prefer_top = random.choice([False, True])
    prefer_last_col = random.choice([False, True])
    best_score = None
    best: list[dict] = []
    last_col = n_cols - 1
    for candidate in candidates:
        score = 0
        if (0 in candidate["rows"]) == prefer_top:
            score += 1
        if (last_col in candidate["cols"]) == prefer_last_col:
            score += 1
        if best_score is None or score > best_score:
            best_score = score
            best = [candidate]
        elif score == best_score:
            best.append(candidate)
    return random.choice(best)


def _build_grid(
    row_sizes: list[int],
    col_sizes: list[int],
    selected_rows: tuple[int, ...],
    selected_cols: tuple[int, ...],
) -> np.ndarray:
    height = sum(row_sizes) + len(row_sizes) - 1
    width = sum(col_sizes) + len(col_sizes) - 1
    grid = np.full((height, width), BLACK, dtype=int)

    row_bands: list[tuple[int, int]] = []
    cur = 0
    for idx, size in enumerate(row_sizes):
        row_bands.append((cur, cur + size - 1))
        cur += size
        if idx != len(row_sizes) - 1:
            grid[cur, :] = RED
            cur += 1

    col_bands: list[tuple[int, int]] = []
    cur = 0
    for idx, size in enumerate(col_sizes):
        col_bands.append((cur, cur + size - 1))
        cur += size
        if idx != len(col_sizes) - 1:
            grid[:, cur] = RED
            cur += 1

    for cb in selected_cols:
        c0, c1 = col_bands[cb]
        marker_col = random.randint(c0, c1)
        grid[0, marker_col] = BLUE

    for rb in selected_rows:
        r0, r1 = row_bands[rb]
        marker_row = random.randint(r0, r1)
        grid[marker_row, width - 1] = GREEN

    return grid


def _validate_generated_instance(grid: np.ndarray) -> None:
    parsed = _parse_grid(grid)
    areas = parsed["areas"]
    assert len(areas) >= 6, "expected at least 6 partition cells"
    assert len(set(areas.values())) >= 3, "expected at least 3 distinct cell areas"

    min_cell, max_cell = _selected_extrema(parsed)
    selected = parsed["selected_cells"]
    selected_areas = [areas[cell] for cell in selected]
    unselected = [cell for cell in areas if cell not in selected]
    unselected_areas = [areas[cell] for cell in unselected]
    assert min(unselected_areas) < min(selected_areas), "global minimum must lie outside the selected subset"
    assert max(unselected_areas) > max(selected_areas), "global maximum must lie outside the selected subset"

    out = main(grid)
    unique_out = set(map(int, np.unique(out)))
    assert {RED, BLUE, GREEN, YELLOW, ORANGE, TEAL}.issubset(unique_out), "output must realize all target colors"
    assert np.count_nonzero(out == TEAL) == areas[min_cell], "teal fill must match the unique minimum cell area"
    assert np.count_nonzero(out == ORANGE) == areas[max_cell], "orange fill must match the unique maximum cell area"
    assert not np.array_equal(out, grid), "output must be non-trivial"


def _gen_attempt() -> np.ndarray | None:
    row_band_count = random.choice(ROW_BAND_OPTIONS)
    col_band_count = random.choice(COL_BAND_OPTIONS)
    row_sizes = _random_band_sizes(row_band_count)
    col_sizes = _random_band_sizes(col_band_count)

    if sum(row_sizes) + row_band_count - 1 > 30 or sum(col_sizes) + col_band_count - 1 > 30:
        return None

    candidates = _candidate_selections(row_sizes, col_sizes)
    if not candidates:
        return None

    chosen = _weighted_candidate_choice(candidates, col_band_count)
    grid = _build_grid(row_sizes, col_sizes, chosen["rows"], chosen["cols"])

    try:
        _validate_generated_instance(grid)
    except Exception:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid puzzle instance")
