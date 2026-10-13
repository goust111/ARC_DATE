from __future__ import annotations

import random
from collections import deque

import numpy as np


BG = 0
WALL = 5
MARKER = 9
PALETTE_POOL = [1, 2, 3, 4, 6, 7, 8, 9]

# --------------- generator configuration ---------------
COARSE_ROWS_RANGE = (2, 4)
COARSE_COLS_RANGE = (2, 4)
CELL_HEIGHT_RANGE = (4, 5)
CELL_WIDTH_RANGE = (4, 5)
REGION_COUNT_CAP = 8
LEGEND_LEN_CAP = 8
MARKER_SAMPLE_TRIES = 24
GEN_RETRIES = 300


def _neighbors4(r: int, c: int) -> list[tuple[int, int]]:
    return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not bool(mask[r, c]) or bool(seen[r, c]):
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                comp.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and bool(mask[nr, nc]) and not bool(seen[nr, nc]):
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill each enclosed region with the legend color indexed by its marker count.

    Input structure:
    - Background is black.
    - Row 0 contains a left-aligned horizontal legend of distinct colored pixels.
    - From row 2 downward, one-pixel grey walls partition the space into closed regions whose interiors contain black cells and maroon markers.

    Rule:
    - Read the legend colors from left to right.
    - For every enclosed 4-connected non-grey region below the legend, count its maroon marker pixels.
    - Use count N to select legend column N-1, then recolor the whole region, including the markers, with that color.
    - Keep the legend, the separator row, and all grey walls unchanged.

    Key insight: The maroon pixels are counters, and the top color strip is an index table that converts each region's count into its fill color.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.shape[0] >= 3

    legend_len = 0
    while legend_len < g.shape[1] and int(g[0, legend_len]) != BG:
        legend_len += 1
    assert 1 <= legend_len <= LEGEND_LEN_CAP
    legend = [int(v) for v in g[0, :legend_len]]

    out = g.copy()
    field = g[2:, :]
    passable = field != WALL

    for comp in _components(passable):
        if any(r in (0, field.shape[0] - 1) or c in (0, field.shape[1] - 1) for r, c in comp):
            continue
        marker_count = sum(int(field[r, c]) == MARKER for r, c in comp)
        assert 1 <= marker_count <= legend_len
        fill_color = int(legend[marker_count - 1])
        for r, c in comp:
            out[r + 2, c] = fill_color

    return out


def _cell_neighbors(cell: tuple[int, int], rows: int, cols: int) -> list[tuple[int, int]]:
    r, c = cell
    out: list[tuple[int, int]] = []
    if r > 0:
        out.append((r - 1, c))
    if r + 1 < rows:
        out.append((r + 1, c))
    if c > 0:
        out.append((r, c - 1))
    if c + 1 < cols:
        out.append((r, c + 1))
    return out


def _grow_partition(rng: random.Random, rows: int, cols: int, region_count: int) -> np.ndarray | None:
    cells = [(r, c) for r in range(rows) for c in range(cols)]
    seeds = rng.sample(cells, region_count)
    labels = np.full((rows, cols), -1, dtype=int)
    region_cells: list[list[tuple[int, int]]] = [[] for _ in range(region_count)]

    unassigned = set(cells)
    for label, cell in enumerate(seeds):
        labels[cell] = label
        region_cells[label].append(cell)
        unassigned.remove(cell)

    while unassigned:
        growable: list[int] = []
        candidate_map: dict[int, list[tuple[int, int]]] = {}
        for label in range(region_count):
            candidates = {
                nb
                for cell in region_cells[label]
                for nb in _cell_neighbors(cell, rows, cols)
                if nb in unassigned
            }
            if candidates:
                growable.append(label)
                candidate_map[label] = list(candidates)
        if not growable:
            return None
        label = rng.choice(growable)
        cell = rng.choice(candidate_map[label])
        labels[cell] = label
        region_cells[label].append(cell)
        unassigned.remove(cell)

    return labels


def _best_marker_sample(
    rng: random.Random,
    coords: list[tuple[int, int]],
    count: int,
) -> list[tuple[int, int]]:
    if count == 1:
        return [rng.choice(coords)]

    best = rng.sample(coords, count)
    best_score = -1
    for _ in range(MARKER_SAMPLE_TRIES):
        sample = rng.sample(coords, count)
        score = min(
            abs(r1 - r2) + abs(c1 - c2)
            for i, (r1, c1) in enumerate(sample)
            for r2, c2 in sample[i + 1 :]
        )
        if score > best_score:
            best = sample
            best_score = score
    return best


def _render_partition(
    labels: np.ndarray,
    row_heights: list[int],
    col_widths: list[int],
) -> tuple[np.ndarray, np.ndarray, dict[int, list[tuple[int, int]]]]:
    rows, cols = labels.shape
    main_h = sum(row_heights) + rows + 1
    main_w = sum(col_widths) + cols + 1
    field = np.full((main_h, main_w), WALL, dtype=int)
    region_map = np.full((main_h, main_w), -1, dtype=int)
    marker_slots: dict[int, list[tuple[int, int]]] = {}

    row_ranges: list[tuple[int, int]] = []
    col_ranges: list[tuple[int, int]] = []

    cur = 1
    for height in row_heights:
        row_ranges.append((cur, cur + height - 1))
        cur += height + 1

    cur = 1
    for width in col_widths:
        col_ranges.append((cur, cur + width - 1))
        cur += width + 1

    for r in range(rows):
        r0, r1 = row_ranges[r]
        for c in range(cols):
            c0, c1 = col_ranges[c]
            label = int(labels[r, c])
            field[r0 : r1 + 1, c0 : c1 + 1] = BG
            region_map[r0 : r1 + 1, c0 : c1 + 1] = label
            marker_slots.setdefault(label, [])
            for rr in range(r0, r1 + 1):
                for cc in range(c0, c1 + 1):
                    marker_slots[label].append((rr, cc))

    for r in range(rows):
        for c in range(cols):
            label = int(labels[r, c])
            r0, r1 = row_ranges[r]
            c0, c1 = col_ranges[c]
            if r + 1 < rows and int(labels[r + 1, c]) == label:
                wall_r = r1 + 1
                field[wall_r, c0 : c1 + 1] = BG
                region_map[wall_r, c0 : c1 + 1] = label
            if c + 1 < cols and int(labels[r, c + 1]) == label:
                wall_c = c1 + 1
                field[r0 : r1 + 1, wall_c] = BG
                region_map[r0 : r1 + 1, wall_c] = label

    return field, region_map, marker_slots


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    coarse_rows = rng.randint(*COARSE_ROWS_RANGE)
    coarse_cols = rng.randint(*COARSE_COLS_RANGE)
    row_heights = [rng.randint(*CELL_HEIGHT_RANGE) for _ in range(coarse_rows)]
    col_widths = [rng.randint(*CELL_WIDTH_RANGE) for _ in range(coarse_cols)]

    main_h = sum(row_heights) + coarse_rows + 1
    main_w = sum(col_widths) + coarse_cols + 1
    if main_h + 2 > 30 or main_w > 30:
        return None

    max_regions = min(REGION_COUNT_CAP, coarse_rows * coarse_cols)
    region_count = rng.randint(2, max_regions)
    legend_len = rng.randint(region_count, LEGEND_LEN_CAP)

    labels = _grow_partition(rng, coarse_rows, coarse_cols, region_count)
    if labels is None:
        return None

    field, region_map, marker_slots = _render_partition(labels, row_heights, col_widths)
    palette = rng.sample(PALETTE_POOL, legend_len)
    region_counts = rng.sample(list(range(1, legend_len + 1)), region_count)

    for label in range(region_count):
        coords = marker_slots[label]
        count = region_counts[label]
        if len(coords) < count:
            return None
        for r, c in _best_marker_sample(rng, coords, count):
            field[r, c] = MARKER

    grid = np.full((main_h + 2, main_w), BG, dtype=int)
    grid[0, :legend_len] = palette
    grid[2:, :] = field

    solved = main(grid)
    for label in range(region_count):
        expected_color = int(palette[region_counts[label] - 1])
        pixels = region_map == label
        if not bool(np.all(solved[2:, :][pixels] == expected_color)):
            return None
    if not bool(np.array_equal(solved[2:, :][field == WALL], field[field == WALL])):
        return None

    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("generate_input: exhausted attempts")
