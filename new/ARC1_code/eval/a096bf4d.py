from collections import Counter, defaultdict

import numpy as np
import random


def _runs(active: list[bool]) -> list[tuple[int, int]]:
    runs: list[tuple[int, int]] = []
    start = None
    for i, value in enumerate(active):
        if value and start is None:
            start = i
        elif not value and start is not None:
            runs.append((start, i))
            start = None
    if start is not None:
        runs.append((start, len(active)))
    return runs


def _unique_dominant(values: list[int]) -> int:
    counts = Counter(values)
    ranked = counts.most_common()
    assert ranked, "expected at least one interior structural value"
    assert len(ranked) == 1 or ranked[0][1] > ranked[1][1], "expected a unique interior fill color"
    return int(ranked[0][0])


def _lattice_runs(g: np.ndarray, separator: int) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    height, width = g.shape
    assert np.all(g[0, :] == separator) and np.all(g[-1, :] == separator), "expected separator rows on the boundary"
    assert np.all(g[:, 0] == separator) and np.all(g[:, -1] == separator), "expected separator columns on the boundary"

    row_runs = _runs([not np.all(g[r, :] == separator) for r in range(height)])
    col_runs = _runs([not np.all(g[:, c] == separator) for c in range(width)])
    assert row_runs and col_runs, "expected a nonempty tile lattice"
    assert all(end - start == 4 for start, end in row_runs), "expected 4-cell-high tile rows"
    assert all(end - start == 4 for start, end in col_runs), "expected 4-cell-wide tile columns"
    return row_runs, col_runs


def _structural_colors(
    g: np.ndarray,
    row_runs: list[tuple[int, int]],
    col_runs: list[tuple[int, int]],
) -> tuple[int, int]:
    border_values: set[int] = set()
    interior_values: list[int] = []
    for r0, r1 in row_runs:
        for c0, c1 in col_runs:
            tile = g[r0:r1, c0:c1]
            assert tile.shape == (4, 4), "expected 4x4 tiles"
            border = np.concatenate([tile[0, :], tile[3, :], tile[1:3, 0], tile[1:3, 3]])
            border_values.update(map(int, border))
            interior_values.extend(map(int, tile[1:3, 1:3].reshape(-1)))

    assert len(border_values) == 1, "expected one repeated frame color"
    frame = next(iter(border_values))
    fill = _unique_dominant([value for value in interior_values if value != frame])
    assert fill != frame, "frame and fill colors must be distinct"
    return int(frame), int(fill)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Complete same-color marker segments inside a regular lattice of framed tiles.

    Input structure
    - The grid is partitioned by separator rows and columns into 4x4 framed tiles.
    - All tile borders have one frame color, and each 2x2 tile interior has a dominant fill color.
    - Other interior colors are endpoint markers, with their exact 2x2 subcell position preserved.

    Rule
    - Group endpoint markers by marker color and by their relative interior subcell.
    - When two or more endpoints in a group share a tile row, fill every intervening tile at that same subcell with the marker color.
    - When two or more endpoints in a group share a tile column, fill every intervening tile at that same subcell with the marker color.

    Key insight
    - The colored endpoints are not connected in pixel space; they are connected on the coarser tile lattice, and the chosen interior subcell identifies which lane to paint.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    separator = int(g[0, 0])
    row_runs, col_runs = _lattice_runs(g, separator)
    frame, fill = _structural_colors(g, row_runs, col_runs)
    structural = {separator, frame, fill}

    groups: dict[tuple[int, int, int], list[tuple[int, int]]] = defaultdict(list)
    for tile_r, (r0, _) in enumerate(row_runs):
        for tile_c, (c0, _) in enumerate(col_runs):
            for dr in (1, 2):
                for dc in (1, 2):
                    color = int(g[r0 + dr, c0 + dc])
                    if color not in structural:
                        groups[(color, dr, dc)].append((tile_r, tile_c))
    assert groups, "expected at least one marker group"

    out = g.copy()
    for (color, dr, dc), points in groups.items():
        assert len(points) >= 2, "expected every marker group to define at least one segment"
        covered: set[tuple[int, int]] = set()
        by_row: dict[int, list[int]] = defaultdict(list)
        by_col: dict[int, list[int]] = defaultdict(list)
        for tile_r, tile_c in points:
            by_row[tile_r].append(tile_c)
            by_col[tile_c].append(tile_r)

        for tile_r, tile_cols in by_row.items():
            if len(tile_cols) < 2:
                continue
            for tile_c in range(min(tile_cols), max(tile_cols) + 1):
                r = row_runs[tile_r][0] + dr
                c = col_runs[tile_c][0] + dc
                assert int(out[r, c]) in structural or int(out[r, c]) == color, "conflicting marker segment"
                out[r, c] = color
                covered.add((tile_r, tile_c))

        for tile_c, tile_rows in by_col.items():
            if len(tile_rows) < 2:
                continue
            for tile_r in range(min(tile_rows), max(tile_rows) + 1):
                r = row_runs[tile_r][0] + dr
                c = col_runs[tile_c][0] + dc
                assert int(out[r, c]) in structural or int(out[r, c]) == color, "conflicting marker segment"
                out[r, c] = color
                covered.add((tile_r, tile_c))

        assert set(points) <= covered, "expected every marker endpoint to lie on a completed segment"

    return out

# --------------- generator (o5x) ---------------

# Tile lattice geometry.
TILE_COUNT_RANGE = (3, 5)
TILE_STRIDE, TILE_OFFSET, TILE_SIZE, INTERIOR_SIZE = 5, 1, 4, 2

# Marker group and segment bounds.
GROUP_COUNT_RANGE = (2, 4)
SEGMENT_LENGTH_MIN = 3
MAX_GENERATION_ATTEMPTS = 300

# Color and base interior sampling.
NONZERO_COLORS = list(range(1, 10))
OPTIONAL_FRAME_SUBCELL_CHOICES = 2
INTERIOR_LANES_0 = tuple((dr, dc) for dr in range(INTERIOR_SIZE) for dc in range(INTERIOR_SIZE))


def _gen_base_grid(tile_rows, tile_cols, frame, interior):
    """Render the structural lattice before marker segments are painted."""
    grid = np.zeros((TILE_STRIDE * tile_rows + TILE_OFFSET, TILE_STRIDE * tile_cols + TILE_OFFSET), dtype=int)
    for tile_r in range(tile_rows):
        for tile_c in range(tile_cols):
            r0, c0 = _gen_tile_origin(tile_r, tile_c)
            grid[r0 : r0 + TILE_SIZE, c0 : c0 + TILE_SIZE] = frame
            grid[r0 + TILE_OFFSET : r0 + TILE_OFFSET + INTERIOR_SIZE, c0 + TILE_OFFSET : c0 + TILE_OFFSET + INTERIOR_SIZE] = interior
    return grid


def _gen_tile_origin(tile_r, tile_c):
    return TILE_OFFSET + TILE_STRIDE * tile_r, TILE_OFFSET + TILE_STRIDE * tile_c


def _gen_pixel(tile_r, tile_c, dr, dc):
    r0, c0 = _gen_tile_origin(tile_r, tile_c)
    return r0 + dr, c0 + dc


def _gen_candidate(endpoint_count, segment_class, cells, endpoints):
    return {"endpoint_count": endpoint_count, "class": segment_class, "cells": cells, "endpoints": endpoints}


def _gen_axis_ends(center, length, limit):
    ends = []
    if center - length + 1 >= 0:
        ends.append(center - length + 1)
    if center + length - 1 < limit:
        ends.append(center + length - 1)
    return ends


def _gen_segment_candidates(tile_rows, tile_cols, eligible_lanes):
    """Build all non-conflicting segment shapes available before occupancy filtering."""
    candidates = []
    for dr, dc in eligible_lanes:
        for tile_r in range(tile_rows):
            for length in range(SEGMENT_LENGTH_MIN, tile_cols + 1):
                for start_c in range(tile_cols - length + 1):
                    end_c = start_c + length - 1
                    cells = tuple((tile_r, tile_c, dr, dc) for tile_c in range(start_c, end_c + 1))
                    endpoints = ((tile_r, start_c, dr, dc), (tile_r, end_c, dr, dc))
                    candidates.append(_gen_candidate(2, ("H", length), cells, endpoints))

        for tile_c in range(tile_cols):
            for length in range(SEGMENT_LENGTH_MIN, tile_rows + 1):
                for start_r in range(tile_rows - length + 1):
                    end_r = start_r + length - 1
                    cells = tuple((tile_r, tile_c, dr, dc) for tile_r in range(start_r, end_r + 1))
                    endpoints = ((start_r, tile_c, dr, dc), (end_r, tile_c, dr, dc))
                    candidates.append(_gen_candidate(2, ("V", length), cells, endpoints))

        for corner_r in range(tile_rows):
            for corner_c in range(tile_cols):
                for h_len in range(SEGMENT_LENGTH_MIN, tile_cols + 1):
                    h_ends = _gen_axis_ends(corner_c, h_len, tile_cols)
                    for v_len in range(SEGMENT_LENGTH_MIN, tile_rows + 1):
                        v_ends = _gen_axis_ends(corner_r, v_len, tile_rows)
                        for end_c in h_ends:
                            for end_r in v_ends:
                                c0, c1 = sorted((corner_c, end_c))
                                r0, r1 = sorted((corner_r, end_r))
                                cells = {(corner_r, tile_c, dr, dc) for tile_c in range(c0, c1 + 1)}
                                cells.update((tile_r, corner_c, dr, dc) for tile_r in range(r0, r1 + 1))
                                endpoints = (
                                    (corner_r, corner_c, dr, dc),
                                    (corner_r, end_c, dr, dc),
                                    (end_r, corner_c, dr, dc),
                                )
                                candidates.append(_gen_candidate(3, ("L", h_len, v_len), tuple(sorted(cells)), endpoints))
    return candidates


def _gen_pick_candidate(candidates, occupied):
    """Sample one currently unoccupied segment with the original class-balanced policy."""
    valid = [cand for cand in candidates if not any(cell in occupied for cell in cand["cells"])]
    if not valid:
        return None

    endpoint_counts = sorted({cand["endpoint_count"] for cand in valid})
    endpoint_count = random.choice(endpoint_counts)
    same_count = [cand for cand in valid if cand["endpoint_count"] == endpoint_count]

    by_class = {}
    for cand in same_count:
        by_class.setdefault(cand["class"], []).append(cand)
    segment_class = random.choice(list(by_class))
    return random.choice(by_class[segment_class])


def _gen_setup():
    """Sample dimensions, colors, base interior pattern, and marker-eligible lanes."""
    tile_rows = random.randint(*TILE_COUNT_RANGE)
    tile_cols = random.randint(*TILE_COUNT_RANGE)
    group_count = random.randint(*GROUP_COUNT_RANGE)

    colors = list(NONZERO_COLORS)
    random.shuffle(colors)
    frame = colors.pop()
    fill = colors.pop()
    marker_colors = [colors.pop() for _ in range(group_count)]

    interior = np.full((INTERIOR_SIZE, INTERIOR_SIZE), fill, dtype=int)
    if random.randrange(OPTIONAL_FRAME_SUBCELL_CHOICES):
        interior[random.randrange(INTERIOR_SIZE), random.randrange(INTERIOR_SIZE)] = frame
    eligible_lanes = [
        (dr + TILE_OFFSET, dc + TILE_OFFSET)
        for dr, dc in INTERIOR_LANES_0
        if int(interior[dr, dc]) == fill
    ]
    return tile_rows, tile_cols, frame, interior, marker_colors, eligible_lanes


def _gen_pick_groups(marker_colors, candidates):
    """Assign each marker color to a segment while preventing cell overlap."""
    occupied = set()
    groups = []
    for color in marker_colors:
        cand = _gen_pick_candidate(candidates, occupied)
        if cand is None:
            return None
        groups.append((color, cand))
        occupied.update(cand["cells"])
    return groups


def _gen_render_output(tile_rows, tile_cols, frame, interior, groups):
    out = _gen_base_grid(tile_rows, tile_cols, frame, interior)
    for color, cand in groups:
        for tile_r, tile_c, dr, dc in cand["cells"]:
            r, c = _gen_pixel(tile_r, tile_c, dr, dc)
            out[r, c] = color
    return out


def _gen_make_input(out, interior, groups):
    """Erase non-endpoint marker cells from the completed output grid."""
    inp = out.copy()
    for _color, cand in groups:
        endpoints = set(cand["endpoints"])
        for tile_r, tile_c, dr, dc in cand["cells"]:
            if (tile_r, tile_c, dr, dc) not in endpoints:
                r, c = _gen_pixel(tile_r, tile_c, dr, dc)
                inp[r, c] = int(interior[dr - TILE_OFFSET, dc - TILE_OFFSET])
    return inp


def _gen_attempt():
    tile_rows, tile_cols, frame, interior, marker_colors, eligible_lanes = _gen_setup()
    candidates = _gen_segment_candidates(tile_rows, tile_cols, eligible_lanes)
    groups = _gen_pick_groups(marker_colors, candidates)
    if groups is None:
        return None

    out = _gen_render_output(tile_rows, tile_cols, frame, interior, groups)
    inp = _gen_make_input(out, interior, groups)
    if not np.array_equal(main(inp), out):
        return None
    return inp


def generate_input():
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid a096bf4d input")
