from __future__ import annotations

import random
from collections import defaultdict

import numpy as np


BG_COLOR = 0
START_COLOR = 9
PATH_PALETTE = (1, 2, 3, 4, 5, 6, 7, 8)
DIRS4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
GENERATION_ATTEMPTS = 400
PLACEMENT_ATTEMPTS = 250


def _dominant_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _in_bounds(cell: tuple[int, int], shape: tuple[int, int]) -> bool:
    r, c = cell
    h, w = shape
    return 0 <= r < h and 0 <= c < w


def _neighbors4(cell: tuple[int, int], shape: tuple[int, int]) -> list[tuple[int, int]]:
    r, c = cell
    out: list[tuple[int, int]] = []
    for dr, dc in DIRS4:
        rr = r + dr
        cc = c + dc
        if 0 <= rr < shape[0] and 0 <= cc < shape[1]:
            out.append((rr, cc))
    return out


def _manhattan(a: tuple[int, int], b: tuple[int, int]) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def _components_of_color(grid: np.ndarray, color: int) -> list[list[tuple[int, int]]]:
    mask = grid == color
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r, c in np.argwhere(mask):
        r = int(r)
        c = int(c)
        if seen[r, c]:
            continue
        seen[r, c] = True
        stack = [(r, c)]
        comp: list[tuple[int, int]] = []
        while stack:
            cell = stack.pop()
            comp.append(cell)
            for nxt in _neighbors4(cell, grid.shape):
                if mask[nxt] and not seen[nxt]:
                    seen[nxt] = True
                    stack.append(nxt)
        comps.append(comp)
    return comps


def _parse_l_path(coords: list[tuple[int, int]], shape: tuple[int, int]) -> dict:
    cell_set = set(coords)
    assert len(cell_set) >= 3, "path must contain at least one bend and two endpoints"

    neighbors_by_cell: dict[tuple[int, int], list[tuple[int, int]]] = {}
    for cell in cell_set:
        neighbors = [nbr for nbr in _neighbors4(cell, shape) if nbr in cell_set]
        degree = len(neighbors)
        assert degree in (1, 2), "L path must be a single thin polyline"
        neighbors_by_cell[cell] = neighbors

    endpoints = [cell for cell, neighbors in neighbors_by_cell.items() if len(neighbors) == 1]
    assert len(endpoints) == 2, "L path must have exactly two endpoints"

    corner_candidates: list[tuple[tuple[int, int], list[tuple[int, int]]]] = []
    for cell, neighbors in neighbors_by_cell.items():
        if len(neighbors) != 2:
            continue
        dirs = [(nbr[0] - cell[0], nbr[1] - cell[1]) for nbr in neighbors]
        if dirs[0][0] == -dirs[1][0] and dirs[0][1] == -dirs[1][1]:
            continue
        if dirs[0][0] != dirs[1][0] and dirs[0][1] != dirs[1][1]:
            corner_candidates.append((cell, dirs))

    assert len(corner_candidates) == 1, "expected one right-angle bend"
    corner, dirs = corner_candidates[0]
    cr, cc = corner
    assert all(r == cr or c == cc for r, c in cell_set), "component must lie on one row and one column"

    indicator = (cr + dirs[0][0] + dirs[1][0], cc + dirs[0][1] + dirs[1][1])
    assert _in_bounds(indicator, shape), "indicator pixel must be inside the grid"
    assert indicator not in cell_set, "indicator position must be empty in the path component"

    return {
        "cells": cell_set,
        "corner": corner,
        "endpoints": tuple(endpoints),
        "indicator_pos": indicator,
    }


def _extract_paths(grid: np.ndarray, bg: int) -> list[dict]:
    paths: list[dict] = []
    for color in sorted(int(x) for x in np.unique(grid) if int(x) not in {bg, START_COLOR}):
        for comp in _components_of_color(grid, color):
            if len(comp) == 1:
                continue
            parsed = _parse_l_path(comp, grid.shape)
            indicator_color = int(grid[parsed["indicator_pos"]])
            assert indicator_color not in {bg, START_COLOR, color}, "indicator must be a distinct foreground color"
            parsed["color"] = color
            parsed["indicator_color"] = indicator_color
            paths.append(parsed)
    return paths


def main(grid: np.ndarray) -> np.ndarray:
    """
    Keep only the route selected by the maroon start marker and the indicator-color chain.
    """
    g = np.asarray(grid, dtype=int)
    bg = BG_COLOR if np.any(g == BG_COLOR) else _dominant_color(g)

    start_pixels = np.argwhere(g == START_COLOR)
    assert len(start_pixels) == 1, "expected a unique maroon start marker"
    start = (int(start_pixels[0, 0]), int(start_pixels[0, 1]))

    paths = _extract_paths(g, bg)
    assert paths, "expected at least one L path"

    paths_by_color: dict[int, list[int]] = defaultdict(list)
    first_candidates: list[tuple[int, tuple[int, int]]] = []
    for idx, path in enumerate(paths):
        paths_by_color[path["color"]].append(idx)
        touching = [ep for ep in path["endpoints"] if _manhattan(ep, start) == 1]
        if touching:
            assert len(touching) == 1, "start should touch exactly one endpoint of the first path"
            first_candidates.append((idx, touching[0]))

    assert len(first_candidates) == 1, "start marker must identify one first path"

    route_ids: list[int] = []
    current_id, entry = first_candidates[0]
    visited: set[int] = set()

    while True:
        route_ids.append(current_id)
        visited.add(current_id)
        path = paths[current_id]
        exit_endpoint = path["endpoints"][0] if path["endpoints"][1] == entry else path["endpoints"][1]
        next_color = path["indicator_color"]

        candidates = [idx for idx in paths_by_color.get(next_color, []) if idx not in visited]
        if not candidates:
            break

        adjacent: list[tuple[int, tuple[int, int]]] = []
        for idx in candidates:
            for endpoint in paths[idx]["endpoints"]:
                if _manhattan(endpoint, exit_endpoint) == 1:
                    adjacent.append((idx, endpoint))

        if not adjacent:
            break

        assert len(adjacent) == 1, "indicator must resolve to one adjacent next path"
        current_id, entry = adjacent[0]

    out = np.full_like(g, bg)
    out[start] = START_COLOR
    for idx in route_ids:
        color = paths[idx]["color"]
        for cell in paths[idx]["cells"]:
            out[cell] = color
    return out


def _build_segment(
    color: int,
    corner: tuple[int, int],
    vdir: int,
    hdir: int,
    vlen: int,
    hlen: int,
) -> dict:
    cr, cc = corner
    vertical = [(cr + step * vdir, cc) for step in range(vlen)]
    horizontal = [(cr, cc + step * hdir) for step in range(hlen)]
    cells = set(vertical)
    cells.update(horizontal)
    return {
        "color": color,
        "corner": corner,
        "vdir": vdir,
        "hdir": hdir,
        "vlen": vlen,
        "hlen": hlen,
        "cells": cells,
        "indicator_pos": (cr + vdir, cc + hdir),
        "endpoints": (vertical[-1], horizontal[-1]),
    }


def _outward_from_endpoint(segment: dict, endpoint_index: int) -> tuple[int, int]:
    if endpoint_index == 0:
        r, c = segment["endpoints"][0]
        return (r + segment["vdir"], c)
    r, c = segment["endpoints"][1]
    return (r, c + segment["hdir"])


def _segment_footprint(segment: dict) -> set[tuple[int, int]]:
    footprint = set(segment["cells"])
    footprint.add(segment["indicator_pos"])
    return footprint


def _can_place_cells(
    cells: set[tuple[int, int]] | list[tuple[int, int]] | tuple[tuple[int, int], ...],
    occupied: set[tuple[int, int]],
    shape: tuple[int, int],
    allowed_pairs: set[frozenset[tuple[int, int]]] | None = None,
) -> bool:
    allowed_pairs = allowed_pairs or set()
    cell_list = list(cells)
    cell_set = set(cell_list)
    if len(cell_set) != len(cell_list):
        return False
    for cell in cell_set:
        if not _in_bounds(cell, shape):
            return False
    if cell_set & occupied:
        return False
    for cell in cell_set:
        for nbr in _neighbors4(cell, shape):
            if nbr in occupied and frozenset((cell, nbr)) not in allowed_pairs:
                return False
    return True


def _sample_free_segment(color: int, shape: tuple[int, int]) -> dict:
    h, w = shape
    vdir = random.choice((-1, 1))
    hdir = random.choice((-1, 1))
    vlen = random.randint(2, min(5, h))
    hlen = random.randint(2, min(5, w))
    if vdir == 1:
        row = random.randint(0, h - vlen)
    else:
        row = random.randint(vlen - 1, h - 1)
    if hdir == 1:
        col = random.randint(0, w - hlen)
    else:
        col = random.randint(hlen - 1, w - 1)
    return _build_segment(color, (row, col), vdir, hdir, vlen, hlen)


def _max_lengths_for_entry(
    entry: tuple[int, int],
    entry_index: int,
    vdir: int,
    hdir: int,
    shape: tuple[int, int],
) -> tuple[int, int]:
    r, c = entry
    h, w = shape
    if entry_index == 0:
        max_v = r + 1 if vdir == 1 else h - r
        max_h = w - c if hdir == 1 else c + 1
    else:
        max_v = h - r if vdir == 1 else r + 1
        max_h = c + 1 if hdir == 1 else w - c
    return max_v, max_h


def _build_segment_from_entry(
    color: int,
    entry: tuple[int, int],
    entry_index: int,
    vdir: int,
    hdir: int,
    vlen: int,
    hlen: int,
) -> dict:
    er, ec = entry
    if entry_index == 0:
        corner = (er - (vlen - 1) * vdir, ec)
    else:
        corner = (er, ec - (hlen - 1) * hdir)
    segment = _build_segment(color, corner, vdir, hdir, vlen, hlen)
    segment["entry"] = segment["endpoints"][entry_index]
    segment["exit"] = segment["endpoints"][1 - entry_index]
    segment["entry_index"] = entry_index
    return segment


def _place_first_route_segment(
    color: int,
    occupied: set[tuple[int, int]],
    shape: tuple[int, int],
) -> tuple[dict, tuple[int, int]] | None:
    for _ in range(PLACEMENT_ATTEMPTS):
        segment = _sample_free_segment(color, shape)
        footprint = _segment_footprint(segment)
        if not _can_place_cells(footprint, occupied, shape):
            continue
        endpoint_indices = [0, 1]
        random.shuffle(endpoint_indices)
        for endpoint_index in endpoint_indices:
            start = _outward_from_endpoint(segment, endpoint_index)
            temp_occupied = occupied | footprint
            if not _can_place_cells(
                [start],
                temp_occupied,
                shape,
                {frozenset((start, segment["endpoints"][endpoint_index]))},
            ):
                continue
            segment["entry_index"] = endpoint_index
            segment["entry"] = segment["endpoints"][endpoint_index]
            segment["exit"] = segment["endpoints"][1 - endpoint_index]
            return segment, start
    return None


def _place_linked_segment(
    color: int,
    prev_exit: tuple[int, int],
    occupied: set[tuple[int, int]],
    shape: tuple[int, int],
) -> dict | None:
    neighbor_dirs = list(DIRS4)
    random.shuffle(neighbor_dirs)
    for dr, dc in neighbor_dirs:
        entry = (prev_exit[0] + dr, prev_exit[1] + dc)
        if not _in_bounds(entry, shape) or entry in occupied:
            continue
        for _ in range(PLACEMENT_ATTEMPTS):
            entry_index = random.randrange(2)
            vdir = random.choice((-1, 1))
            hdir = random.choice((-1, 1))
            max_v, max_h = _max_lengths_for_entry(entry, entry_index, vdir, hdir, shape)
            if max_v < 2 or max_h < 2:
                continue
            vlen = random.randint(2, min(5, max_v))
            hlen = random.randint(2, min(5, max_h))
            segment = _build_segment_from_entry(color, entry, entry_index, vdir, hdir, vlen, hlen)
            if segment["entry"] != entry:
                continue
            if not _can_place_cells(
                _segment_footprint(segment),
                occupied,
                shape,
                {frozenset((entry, prev_exit))},
            ):
                continue
            return segment
    return None


def _place_distractor_segment(
    color: int,
    occupied: set[tuple[int, int]],
    shape: tuple[int, int],
) -> dict | None:
    for _ in range(PLACEMENT_ATTEMPTS):
        segment = _sample_free_segment(color, shape)
        if _can_place_cells(_segment_footprint(segment), occupied, shape):
            return segment
    return None


def _render_scene(
    shape: tuple[int, int],
    route_segments: list[dict],
    distractor_segments: list[dict],
    start: tuple[int, int],
) -> np.ndarray:
    grid = np.full(shape, BG_COLOR, dtype=int)
    for segment in route_segments + distractor_segments:
        for cell in segment["cells"]:
            grid[cell] = segment["color"]
        grid[segment["indicator_pos"]] = segment["indicator_color"]
    grid[start] = START_COLOR
    return grid


def _expected_output(shape: tuple[int, int], route_segments: list[dict], start: tuple[int, int]) -> np.ndarray:
    out = np.full(shape, BG_COLOR, dtype=int)
    out[start] = START_COLOR
    for segment in route_segments:
        for cell in segment["cells"]:
            out[cell] = segment["color"]
    return out


def _gen_attempt() -> np.ndarray | None:
    total_paths = random.randint(4, 7)
    route_len = random.randint(2, min(5, total_paths - 1))
    side_min = 13 + total_paths
    shape = (
        random.randint(side_min, side_min + 6),
        random.randint(side_min, side_min + 6),
    )

    path_colors = random.sample(list(PATH_PALETTE), total_paths)
    route_colors = path_colors[:route_len]
    distractor_colors = path_colors[route_len:]
    spare_colors = [color for color in PATH_PALETTE if color not in path_colors]
    terminal_color = random.choice(spare_colors)

    occupied: set[tuple[int, int]] = set()
    first = _place_first_route_segment(route_colors[0], occupied, shape)
    if first is None:
        return None
    first_segment, start = first
    route_segments = [first_segment]
    occupied.update(_segment_footprint(first_segment))
    occupied.add(start)

    prev_exit = first_segment["exit"]
    for color in route_colors[1:]:
        segment = _place_linked_segment(color, prev_exit, occupied, shape)
        if segment is None:
            return None
        route_segments.append(segment)
        occupied.update(_segment_footprint(segment))
        prev_exit = segment["exit"]

    distractor_segments: list[dict] = []
    for color in distractor_colors:
        segment = _place_distractor_segment(color, occupied, shape)
        if segment is None:
            return None
        distractor_segments.append(segment)
        occupied.update(_segment_footprint(segment))

    for idx, segment in enumerate(route_segments):
        if idx + 1 < len(route_segments):
            segment["indicator_color"] = route_segments[idx + 1]["color"]
        else:
            segment["indicator_color"] = terminal_color

    indicator_choices = list(PATH_PALETTE)
    for segment in distractor_segments:
        choices = [color for color in indicator_choices if color != segment["color"]]
        segment["indicator_color"] = random.choice(choices)

    grid = _render_scene(shape, route_segments, distractor_segments, start)
    expected = _expected_output(shape, route_segments, start)

    try:
        actual = main(grid)
    except AssertionError:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid route-following ARC input")
