from __future__ import annotations

from collections import Counter
import random

import numpy as np


BG = 0
PATH = 5
COLOR_PRIORITY = [1, 2, 3, 4, 6, 7, 8, 9]  # Blue, Red, Green, Yellow, Magenta, Orange, Azure, Maroon
PRIORITY_RANK = {color: rank for rank, color in enumerate(COLOR_PRIORITY)}
DIRS = [(-1, 0), (1, 0), (0, -1), (0, 1)]

# --- generator configuration ---
GRID_HEIGHT_RANGE = (12, 26)
GRID_WIDTH_RANGE = (12, 26)
CROSS_CASES = ("horizontal", "vertical", "tie")
EXTRA_PATH_COUNT_RANGE = (0, 2)
MAX_GENERATION_ATTEMPTS = 400
PLACEMENT_RETRIES = 120
EDGE_SPLIT_PROB = 0.35
SAME_ENDPOINT_COLOR_PROB = 0.35


def _path_paint_color(color_a: int, color_b: int) -> int:
    assert color_a in PRIORITY_RANK and color_b in PRIORITY_RANK, "unexpected endpoint color"
    return color_a if PRIORITY_RANK[color_a] <= PRIORITY_RANK[color_b] else color_b


def _in_bounds(shape: tuple[int, int], row: int, col: int) -> bool:
    return 0 <= row < shape[0] and 0 <= col < shape[1]


def _trace_straight_path(
    grid: np.ndarray,
    start: tuple[int, int],
    direction: tuple[int, int],
) -> list[tuple[int, int]]:
    """Trace one straight path from a colored endpoint until the opposite endpoint."""
    h, w = grid.shape
    dr, dc = direction
    r, c = start
    cells = [start]

    while True:
        r += dr
        c += dc
        assert 0 <= r < h and 0 <= c < w, "path left the grid before reaching an endpoint"
        value = int(grid[r, c])
        assert value != BG, "path is broken by background"
        cells.append((r, c))
        if value != PATH:
            return cells


def _extract_paths(grid: np.ndarray) -> list[dict]:
    endpoint_mask = (grid != BG) & (grid != PATH)
    endpoints = [tuple(map(int, rc)) for rc in np.argwhere(endpoint_mask)]
    endpoint_set = set(endpoints)
    assert endpoints, "expected at least one endpoint"

    seen_pairs: set[tuple[tuple[int, int], tuple[int, int]]] = set()
    paths: list[dict] = []

    for start in endpoints:
        directions = []
        sr, sc = start
        for dr, dc in DIRS:
            rr, cc = sr + dr, sc + dc
            if _in_bounds(grid.shape, rr, cc) and int(grid[rr, cc]) != BG:
                directions.append((dr, dc))
        assert len(directions) == 1, f"endpoint {start} must have exactly one incident path direction"

        cells = _trace_straight_path(grid, start, directions[0])
        end = cells[-1]
        assert end in endpoint_set and end != start, "trace must terminate at a different colored endpoint"

        key = tuple(sorted((start, end)))
        if key in seen_pairs:
            continue
        seen_pairs.add(key)

        color_a = int(grid[start])
        color_b = int(grid[end])
        paths.append(
            {
                "endpoints": (start, end),
                "cells": cells,
                "length": len(cells),
                "paint": _path_paint_color(color_a, color_b),
            }
        )

    usage = Counter()
    for path in paths:
        usage[path["endpoints"][0]] += 1
        usage[path["endpoints"][1]] += 1
    assert usage and set(usage) == endpoint_set, "every colored cell must be used as one endpoint"
    assert all(count == 1 for count in usage.values()), "each endpoint must belong to exactly one path"

    covered_grey: set[tuple[int, int]] = set()
    for path in paths:
        covered_grey.update(cell for cell in path["cells"] if int(grid[cell]) == PATH)
    expected_grey = {tuple(map(int, rc)) for rc in np.argwhere(grid == PATH)}
    assert covered_grey == expected_grey, "all grey pixels must belong to traced paths"

    return paths


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor each straight path by its endpoint-priority color while letting longer paths win at crossings.

    Input structure:
    - The background is black, and thin grey pixels form straight horizontal and vertical paths.
    - Every path has exactly two colored endpoint markers, one at each end, and these endpoint pixels are not grey.
    - Some horizontal and vertical paths cross on shared grey pixels; those crossing pixels belong to both paths.

    Rule:
    - Recover each full path by starting from a colored endpoint and tracing straight through grey pixels until the other colored endpoint.
    - Recolor each non-crossing path pixel to the earlier endpoint color in the fixed visible order [Blue, Red, Green, Yellow, Magenta, Orange, Azure, Maroon].
    - For any shared crossing pixel, compare the total lengths of the paths that pass through it: the unique longer path paints the crossing, while a tie leaves the crossing grey.
    - Keep every endpoint marker pixel unchanged.

    Key insight:
    - The output colors are not read from the grey paths directly: you must first reconstruct path ownership from the endpoint markers, then apply the endpoint-priority recolor, and only then resolve crossings by path length.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    paths = _extract_paths(g)
    out = g.copy()

    endpoint_cells = {endpoint for path in paths for endpoint in path["endpoints"]}
    owners: dict[tuple[int, int], list[int]] = {}
    for idx, path in enumerate(paths):
        for cell in path["cells"]:
            owners.setdefault(cell, []).append(idx)

    for cell, path_ids in owners.items():
        if cell in endpoint_cells:
            continue
        if len(path_ids) == 1:
            out[cell] = paths[path_ids[0]]["paint"]
            continue

        max_len = max(paths[path_id]["length"] for path_id in path_ids)
        winners = [path_id for path_id in path_ids if paths[path_id]["length"] == max_len]
        out[cell] = PATH if len(winners) != 1 else paths[winners[0]]["paint"]

    return out


def _make_horizontal_spec(row: int, col0: int, col1: int) -> dict:
    assert col0 < col1
    cells = [(row, col) for col in range(col0, col1 + 1)]
    return {"cells": cells, "endpoints": ((row, col0), (row, col1))}


def _make_vertical_spec(col: int, row0: int, row1: int) -> dict:
    assert row0 < row1
    cells = [(row, col) for row in range(row0, row1 + 1)]
    return {"cells": cells, "endpoints": ((row0, col), (row1, col))}


def _valid_splits(total_length: int, max_first: int, max_second: int) -> list[tuple[int, int]]:
    interior = total_length - 1
    return [
        (first, interior - first)
        for first in range(1, max_first + 1)
        if 1 <= interior - first <= max_second
    ]


def _choose_split(total_length: int, max_first: int, max_second: int) -> tuple[int, int] | None:
    options = _valid_splits(total_length, max_first, max_second)
    if not options:
        return None
    edge_options = [option for option in options if 1 in option]
    if edge_options and random.random() < EDGE_SPLIT_PROB:
        return random.choice(edge_options)
    return random.choice(options)


def _choose_cross_totals(
    h_max: int,
    v_max: int,
    case: str,
) -> tuple[int, int] | None:
    h_options = [length for length in range(3, h_max + 1) if _valid_splits(length, h_max, h_max)]
    v_options = [length for length in range(3, v_max + 1) if _valid_splits(length, v_max, v_max)]
    if not h_options or not v_options:
        return None

    if case == "tie":
        common = sorted(set(h_options) & set(v_options))
        if not common:
            return None
        length = random.choice(common)
        return length, length

    pairs = []
    for h_len in h_options:
        for v_len in v_options:
            if case == "horizontal" and h_len >= v_len + 2:
                pairs.append((h_len, v_len))
            elif case == "vertical" and v_len >= h_len + 2:
                pairs.append((h_len, v_len))
    return None if not pairs else random.choice(pairs)


def _place_cross_pair(height: int, width: int, case: str) -> list[dict] | None:
    if height < 7 or width < 7:
        return None

    cross_row = random.randint(2, height - 3)
    cross_col = random.randint(2, width - 3)

    max_left = cross_col
    max_right = width - 1 - cross_col
    max_up = cross_row
    max_down = height - 1 - cross_row

    h_total_max = max_left + max_right + 1
    v_total_max = max_up + max_down + 1
    totals = _choose_cross_totals(h_total_max, v_total_max, case)
    if totals is None:
        return None
    h_total, v_total = totals

    left_right = _choose_split(h_total, max_left, max_right)
    up_down = _choose_split(v_total, max_up, max_down)
    if left_right is None or up_down is None:
        return None
    left_len, right_len = left_right
    up_len, down_len = up_down

    horizontal = _make_horizontal_spec(cross_row, cross_col - left_len, cross_col + right_len)
    vertical = _make_vertical_spec(cross_col, cross_row - up_len, cross_row + down_len)
    return [horizontal, vertical]


def _cells_are_clear(
    cells: list[tuple[int, int]],
    occupied: set[tuple[int, int]],
    shape: tuple[int, int],
) -> bool:
    for row, col in cells:
        if (row, col) in occupied:
            return False
        for rr in range(max(0, row - 1), min(shape[0], row + 2)):
            for cc in range(max(0, col - 1), min(shape[1], col + 2)):
                if (rr, cc) in occupied:
                    return False
    return True


def _place_extra_path(
    height: int,
    width: int,
    occupied: set[tuple[int, int]],
) -> dict | None:
    for _ in range(PLACEMENT_RETRIES):
        orientation = random.choice(("h", "v"))
        if orientation == "h":
            max_length = min(10, width)
            if max_length < 3:
                return None
            length = random.randint(3, max_length)
            row = random.randint(0, height - 1)
            col0 = random.randint(0, width - length)
            spec = _make_horizontal_spec(row, col0, col0 + length - 1)
        else:
            max_length = min(10, height)
            if max_length < 3:
                return None
            length = random.randint(3, max_length)
            col = random.randint(0, width - 1)
            row0 = random.randint(0, height - length)
            spec = _make_vertical_spec(col, row0, row0 + length - 1)

        if _cells_are_clear(spec["cells"], occupied, (height, width)):
            return spec
    return None


def _assign_endpoint_colors(specs: list[dict]) -> None:
    available = random.sample(COLOR_PRIORITY, len(COLOR_PRIORITY))
    for spec in specs:
        if random.random() < SAME_ENDPOINT_COLOR_PROB or len(available) == 1:
            color = available.pop()
            spec["colors"] = (color, color)
        else:
            color_a = available.pop()
            color_b = available.pop()
            if random.random() < 0.5:
                spec["colors"] = (color_a, color_b)
            else:
                spec["colors"] = (color_b, color_a)


def _render_input(shape: tuple[int, int], specs: list[dict]) -> np.ndarray:
    grid = np.full(shape, BG, dtype=int)
    for spec in specs:
        for row, col in spec["cells"]:
            grid[row, col] = PATH
    for spec in specs:
        (row_a, col_a), (row_b, col_b) = spec["endpoints"]
        color_a, color_b = spec["colors"]
        grid[row_a, col_a] = color_a
        grid[row_b, col_b] = color_b
    return grid


def _render_expected(shape: tuple[int, int], specs: list[dict]) -> np.ndarray:
    inp = _render_input(shape, specs)
    out = inp.copy()

    endpoint_cells = {endpoint for spec in specs for endpoint in spec["endpoints"]}
    owners: dict[tuple[int, int], list[int]] = {}
    paints = [_path_paint_color(*spec["colors"]) for spec in specs]
    lengths = [len(spec["cells"]) for spec in specs]

    for idx, spec in enumerate(specs):
        for cell in spec["cells"]:
            owners.setdefault(cell, []).append(idx)

    for cell, path_ids in owners.items():
        assert cell not in endpoint_cells or len(path_ids) == 1, "overlaps cannot happen on endpoint cells"
        if cell in endpoint_cells:
            continue
        if len(path_ids) == 1:
            out[cell] = paints[path_ids[0]]
            continue
        max_len = max(lengths[path_id] for path_id in path_ids)
        winners = [path_id for path_id in path_ids if lengths[path_id] == max_len]
        out[cell] = PATH if len(winners) != 1 else paints[winners[0]]

    return out


def _gen_attempt() -> np.ndarray | None:
    height = random.randint(*GRID_HEIGHT_RANGE)
    width = random.randint(*GRID_WIDTH_RANGE)

    cross_specs = _place_cross_pair(height, width, random.choice(CROSS_CASES))
    if cross_specs is None:
        return None

    specs = cross_specs[:]
    occupied = {cell for spec in specs for cell in spec["cells"]}

    extra_count = random.randint(*EXTRA_PATH_COUNT_RANGE)
    for _ in range(extra_count):
        extra = _place_extra_path(height, width, occupied)
        if extra is None:
            return None
        specs.append(extra)
        occupied.update(extra["cells"])

    _assign_endpoint_colors(specs)
    inp = _render_input((height, width), specs)
    expected = _render_expected((height, width), specs)

    try:
        if np.array_equal(main(inp), expected):
            return inp
    except AssertionError:
        return None
    return None


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for task 6")
