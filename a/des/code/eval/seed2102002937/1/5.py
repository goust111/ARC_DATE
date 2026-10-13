from __future__ import annotations

import random

import numpy as np


BG = 0
BASELINE_COLOR = 5
MARKER_COLOR = 9
PATH_COLOR_POOL = [1, 2, 3, 4, 6, 7, 8]

DIRS = [(0, 1), (1, 0), (0, -1), (-1, 0)]
MAX_GENERATION_ATTEMPTS = 400


def _mode_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _neighbors4(r: int, c: int) -> list[tuple[int, int]]:
    return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


def _manhattan(a: tuple[int, int], b: tuple[int, int]) -> int:
    return abs(int(a[0]) - int(b[0])) + abs(int(a[1]) - int(b[1]))


def _find_components(grid: np.ndarray, bg: int, blocked_rows: set[int] | None = None) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    seen = np.zeros((h, w), dtype=bool)
    blocked_rows = blocked_rows or set()
    components: list[dict] = []

    for r in range(h):
        for c in range(w):
            if r in blocked_rows or seen[r, c]:
                continue
            color = int(g[r, c])
            if color == bg:
                continue

            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                coords.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if not (0 <= nr < h and 0 <= nc < w):
                        continue
                    if nr in blocked_rows or seen[nr, nc]:
                        continue
                    if int(g[nr, nc]) != color:
                        continue
                    seen[nr, nc] = True
                    stack.append((nr, nc))

            arr = np.asarray(coords, dtype=int)
            r0, c0 = arr.min(axis=0)
            r1, c1 = arr.max(axis=0)
            components.append(
                {
                    "color": color,
                    "coords": arr,
                    "area": int(arr.shape[0]),
                    "bbox": (int(r0), int(c0), int(r1), int(c1)),
                }
            )

    return components


def _full_width_baseline(grid: np.ndarray, bg: int) -> tuple[int, int]:
    g = np.asarray(grid, dtype=int)
    rows: list[tuple[int, int]] = []
    for r in range(g.shape[0]):
        row = g[r]
        if int(row[0]) != bg and bool(np.all(row == row[0])):
            rows.append((r, int(row[0])))
    assert len(rows) == 1, f"expected one full-width non-background row, got {rows}"
    return rows[0]


def _segment_endpoints(component: dict) -> list[tuple[int, int]]:
    coords = np.asarray(component["coords"], dtype=int)
    rows = np.unique(coords[:, 0])
    cols = np.unique(coords[:, 1])

    if rows.size == 1:
        row = int(rows[0])
        c0 = int(cols.min())
        c1 = int(cols.max())
        assert component["area"] == c1 - c0 + 1
        assert bool(np.all(np.sort(coords[:, 1]) == np.arange(c0, c1 + 1)))
        if c0 == c1:
            return [(row, c0)]
        return [(row, c0), (row, c1)]

    if cols.size == 1:
        col = int(cols[0])
        r0 = int(rows.min())
        r1 = int(rows.max())
        assert component["area"] == r1 - r0 + 1
        assert bool(np.all(np.sort(coords[:, 0]) == np.arange(r0, r1 + 1)))
        if r0 == r1:
            return [(r0, col)]
        return [(r0, col), (r1, col)]

    raise AssertionError(f"component is not a straight segment: {component['bbox']}")


def _select_marker(components: list[dict]) -> dict:
    singletons = [comp for comp in components if int(comp["area"]) == 1]
    color9 = [comp for comp in singletons if int(comp["color"]) == MARKER_COLOR]
    if len(color9) == 1:
        return color9[0]
    if len(singletons) == 1:
        return singletons[0]

    for candidate in singletons:
        cell = tuple(map(int, candidate["coords"][0]))
        touches = 0
        for other in components:
            if other is candidate:
                continue
            try:
                endpoints = _segment_endpoints(other)
            except AssertionError:
                continue
            if any(_manhattan(cell, endpoint) == 1 for endpoint in endpoints):
                touches += 1
        if touches == 1:
            return candidate

    raise AssertionError("failed to identify the marker component")


def _segment_graph(segments: list[dict]) -> list[set[int]]:
    adjacency = [set() for _ in segments]
    endpoints = [set(_segment_endpoints(segment)) for segment in segments]
    for i in range(len(segments)):
        for j in range(i + 1, len(segments)):
            if any(_manhattan(a, b) == 1 for a in endpoints[i] for b in endpoints[j]):
                adjacency[i].add(j)
                adjacency[j].add(i)
    return adjacency


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reorder the path segments into one horizontal strip using the chain order
    defined by endpoint adjacencies and the maroon start marker.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    h, w = g.shape
    bg = _mode_color(g)

    baseline_row, baseline_color = _full_width_baseline(g, bg)
    assert baseline_row > 0, "need one row above the baseline"

    components = _find_components(g, bg, blocked_rows={baseline_row})
    assert components, "expected marker and path segments"

    marker = _select_marker(components)
    marker_cell = tuple(map(int, marker["coords"][0]))
    segments = [comp for comp in components if comp is not marker]
    assert segments, "expected at least one path segment"

    for segment in segments:
        _segment_endpoints(segment)

    adjacency = _segment_graph(segments)
    if len(segments) == 1:
        assert len(adjacency[0]) == 0
    else:
        degrees = [len(neighbors) for neighbors in adjacency]
        assert degrees.count(1) == 2, f"expected a single chain, got degrees={degrees}"
        assert degrees.count(2) == len(segments) - 2, f"expected internal degree-2 segments, got {degrees}"

    start_candidates = [
        i for i, segment in enumerate(segments)
        if any(_manhattan(marker_cell, endpoint) == 1 for endpoint in _segment_endpoints(segment))
    ]
    assert len(start_candidates) == 1, f"expected one start segment, got {start_candidates}"
    start = start_candidates[0]
    if len(segments) > 1:
        assert len(adjacency[start]) == 1, "marker must point to a chain endpoint"

    order: list[int] = []
    prev = -1
    cur = start
    while cur != -1:
        assert cur not in order, "chain must not cycle"
        order.append(cur)
        next_nodes = [neighbor for neighbor in adjacency[cur] if neighbor != prev]
        if not next_nodes:
            cur = -1
        else:
            assert len(next_nodes) == 1, f"chain branch detected at segment {cur}"
            prev, cur = cur, next_nodes[0]
    assert len(order) == len(segments), "chain traversal did not visit every segment"

    out = np.full((h, w), bg, dtype=int)
    out[baseline_row, :] = baseline_color

    draw_row = baseline_row - 1
    cursor = 0
    for idx in order:
        color = int(segments[idx]["color"])
        length = int(segments[idx]["area"])
        assert cursor + length <= w, "flattened chain does not fit on the baseline row"
        out[draw_row, cursor : cursor + length] = color
        cursor += length

    return out


def _dir_index(direction: tuple[int, int]) -> int:
    return DIRS.index(direction)


def _candidate_next_dirs(direction: tuple[int, int]) -> list[tuple[int, int]]:
    idx = _dir_index(direction)
    return [DIRS[idx], DIRS[(idx - 1) % 4], DIRS[(idx + 1) % 4]]


def _can_place_segment(
    start: tuple[int, int],
    direction: tuple[int, int],
    length: int,
    occupied: set[tuple[int, int]],
    anchor: tuple[int, int] | None,
) -> list[tuple[int, int]] | None:
    cells: list[tuple[int, int]] = []
    for step in range(length):
        r = int(start[0] + direction[0] * step)
        c = int(start[1] + direction[1] * step)
        cell = (r, c)
        if cell in occupied:
            return None

        old_neighbors = {nbr for nbr in _neighbors4(r, c) if nbr in occupied}
        allowed = {anchor} if step == 0 and anchor is not None else set()
        if old_neighbors != allowed:
            return None
        cells.append(cell)
    return cells


def _valid_marker_positions(
    start_cell: tuple[int, int],
    next_cell: tuple[int, int] | None,
    occupied: set[tuple[int, int]],
) -> list[tuple[int, int]]:
    valid: list[tuple[int, int]] = []
    blocked = {next_cell} if next_cell is not None else set()
    for candidate in _neighbors4(*start_cell):
        if candidate in occupied or candidate in blocked:
            continue
        touches = {nbr for nbr in _neighbors4(*candidate) if nbr in occupied}
        if touches == {start_cell}:
            valid.append(candidate)
    return valid


def _build_chain(
    lengths: list[int],
    colors: list[int],
) -> tuple[list[dict], tuple[int, int]] | None:
    first_dir = random.choice(DIRS)
    first_cells = [(first_dir[0] * step, first_dir[1] * step) for step in range(lengths[0])]
    segments = [{"color": int(colors[0]), "cells": first_cells}]
    occupied = set(first_cells)
    current_dir = first_dir
    current_end = first_cells[-1]

    for idx in range(1, len(lengths)):
        dirs = _candidate_next_dirs(current_dir)
        random.shuffle(dirs)
        placed = False
        for direction in dirs:
            start = (current_end[0] + direction[0], current_end[1] + direction[1])
            cells = _can_place_segment(start, direction, lengths[idx], occupied, current_end)
            if cells is None:
                continue
            segments.append({"color": int(colors[idx]), "cells": cells})
            occupied.update(cells)
            current_dir = direction
            current_end = cells[-1]
            placed = True
            break
        if not placed:
            return None

    if all(
        (segments[i]["cells"][1][0] - segments[i]["cells"][0][0], segments[i]["cells"][1][1] - segments[i]["cells"][0][1])
        == (segments[i - 1]["cells"][1][0] - segments[i - 1]["cells"][0][0], segments[i - 1]["cells"][1][1] - segments[i - 1]["cells"][0][1])
        for i in range(1, len(segments))
    ):
        return None

    start_cell = tuple(map(int, segments[0]["cells"][0]))
    next_cell = tuple(map(int, segments[0]["cells"][1])) if len(segments[0]["cells"]) >= 2 else None
    marker_positions = _valid_marker_positions(start_cell, next_cell, occupied)
    if not marker_positions:
        return None
    marker = random.choice(marker_positions)

    end_cell = tuple(map(int, segments[-1]["cells"][-1]))
    prev_cell = tuple(map(int, segments[-1]["cells"][-2])) if len(segments[-1]["cells"]) >= 2 else None
    end_neighbors = {nbr for nbr in _neighbors4(*end_cell) if nbr in occupied}
    if prev_cell is not None and end_neighbors != {prev_cell}:
        return None

    return segments, marker


def _coords_bounds(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [coord[0] for coord in coords]
    cols = [coord[1] for coord in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _paint_segment(grid: np.ndarray, cells: list[tuple[int, int]], color: int, shift_r: int, shift_c: int) -> None:
    for r, c in cells:
        grid[r + shift_r, c + shift_c] = int(color)


def _expected_output(shape: tuple[int, int], segments: list[dict]) -> np.ndarray:
    h, w = shape
    out = np.full((h, w), BG, dtype=int)
    out[h - 1, :] = BASELINE_COLOR
    cursor = 0
    for segment in segments:
        length = len(segment["cells"])
        out[h - 2, cursor : cursor + length] = int(segment["color"])
        cursor += length
    return out


def _gen_attempt() -> np.ndarray | None:
    segment_count = random.randint(3, 5)
    lengths = random.sample(range(2, 8), segment_count)
    random.shuffle(lengths)
    colors = random.sample(PATH_COLOR_POOL, segment_count)

    built = _build_chain(lengths, colors)
    if built is None:
        return None
    segments, marker = built

    all_coords = [marker]
    for segment in segments:
        all_coords.extend(segment["cells"])
    min_r, min_c, max_r, max_c = _coords_bounds(all_coords)

    top_margin = random.randint(1, 4)
    left_margin = random.randint(1, 4)
    right_margin = random.randint(1, 4)
    bottom_gap = random.randint(2, 4)

    total_length = sum(lengths)
    bbox_h = max_r - min_r + 1
    bbox_w = max_c - min_c + 1
    height = top_margin + bbox_h + bottom_gap + 1
    width = max(left_margin + bbox_w + right_margin, total_length + random.randint(0, 4))

    if not (10 <= height <= 30 and 10 <= width <= 30):
        return None

    shift_r = top_margin - min_r
    shift_c = left_margin - min_c
    grid = np.full((height, width), BG, dtype=int)
    grid[height - 1, :] = BASELINE_COLOR

    mr, mc = marker
    grid[mr + shift_r, mc + shift_c] = MARKER_COLOR
    for segment in segments:
        _paint_segment(grid, segment["cells"], int(segment["color"]), shift_r, shift_c)

    expected = _expected_output(grid.shape, segments)
    try:
        if np.array_equal(main(grid), expected):
            return grid
    except Exception:
        return None
    return None


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
