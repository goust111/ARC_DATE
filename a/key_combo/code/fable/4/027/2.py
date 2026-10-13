from __future__ import annotations

from collections import Counter

import numpy as np


BACKGROUND = 0
PATH_COLOR = 4
TRACK_COLOR = 5
STATION_COLORS = (1, 2, 3, 6, 7, 8, 9)
MAX_GENERATION_ATTEMPTS = 300
COLOR_ASSIGNMENT_ATTEMPTS = 200
PITCH_CHOICES = (4, 5)
STRIP_CASES = (
    "top_left_h",
    "top_right_h",
    "bottom_left_h",
    "bottom_right_h",
    "top_left_v",
    "top_right_v",
    "bottom_left_v",
    "bottom_right_v",
)


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def _components_from_mask(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    components: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                coords.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            components.append(coords)
    return components


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _normalize_path(path: tuple[int, ...] | list[int]) -> tuple[int, ...]:
    path_tuple = tuple(path)
    rev = path_tuple[::-1]
    return path_tuple if path_tuple <= rev else rev


def _is_straight_strip(grid: np.ndarray, coords: list[tuple[int, int]], track_color: int) -> bool:
    if any(int(grid[r, c]) == track_color for r, c in coords):
        return False
    r0, c0, r1, c1 = _bbox(coords)
    if r0 == r1:
        return len(coords) == c1 - c0 + 1
    if c0 == c1:
        return len(coords) == r1 - r0 + 1
    return False


def _read_strip_sequence(grid: np.ndarray, coords: list[tuple[int, int]]) -> tuple[int, ...]:
    r0, c0, r1, c1 = _bbox(coords)
    if r0 == r1:
        ordered = sorted(coords, key=lambda rc: rc[1])
    else:
        ordered = sorted(coords, key=lambda rc: rc[0])
    return tuple(int(grid[r, c]) for r, c in ordered)


def _split_graph_and_strip(grid: np.ndarray):
    bg = _get_bg(grid)
    components = _components_from_mask(grid != bg)
    strip_candidates = [comp for comp in components if _is_straight_strip(grid, comp, TRACK_COLOR)]
    assert len(strip_candidates) == 1, "expected one straight instruction strip"
    strip_coords = strip_candidates[0]
    graph_components = [comp for comp in components if comp is not strip_coords]
    assert len(graph_components) == 1, "expected one connected station-track graph"
    graph_coords = graph_components[0]

    graph_mask = np.zeros_like(grid, dtype=bool)
    strip_mask = np.zeros_like(grid, dtype=bool)
    for r, c in graph_coords:
        graph_mask[r, c] = True
    for r, c in strip_coords:
        strip_mask[r, c] = True
    return bg, graph_mask, strip_mask, strip_coords


def _detect_stations(grid: np.ndarray, graph_mask: np.ndarray, bg: int):
    station_mask = graph_mask & (grid != bg) & (grid != TRACK_COLOR)
    components = _components_from_mask(station_mask)
    stations = []
    station_of_cell: dict[tuple[int, int], int] = {}
    for idx, comp in enumerate(components):
        r0, c0, r1, c1 = _bbox(comp)
        assert len(comp) == 4 and r1 - r0 == 1 and c1 - c0 == 1, "expected solid 2x2 stations"
        color = int(grid[r0, c0])
        cells = {(r, c) for r, c in comp}
        assert all(int(grid[r, c]) == color for r, c in cells), "station must be monochrome"
        assert cells == {(r0, c0), (r0, c0 + 1), (r0 + 1, c0), (r0 + 1, c0 + 1)}, "station must fill its 2x2 box"
        stations.append({"color": color, "bbox": (r0, c0, r1, c1), "cells": cells})
        for cell in cells:
            station_of_cell[cell] = idx
    assert len(stations) >= 2, "expected at least two stations"
    return stations, station_of_cell


def _build_station_graph(grid: np.ndarray, graph_mask: np.ndarray, stations, station_of_cell):
    track_components = _components_from_mask(graph_mask & (grid == TRACK_COLOR))
    adjacency = {i: set() for i in range(len(stations))}
    edge_cells: dict[tuple[int, int], list[tuple[int, int]]] = {}

    for comp in track_components:
        touched = set()
        for r, c in comp:
            for nr, nc in _neighbors4(r, c):
                idx = station_of_cell.get((nr, nc))
                if idx is not None:
                    touched.add(idx)
        assert len(touched) == 2, "each track component must connect exactly two stations"
        a, b = sorted(touched)
        key = (a, b)
        assert key not in edge_cells, "parallel edges are not expected"
        adjacency[a].add(b)
        adjacency[b].add(a)
        edge_cells[key] = [(int(r), int(c)) for r, c in comp]

    assert edge_cells, "expected at least one track component"
    return adjacency, edge_cells


def _matching_paths(adjacency, station_colors: list[int], sequence: tuple[int, ...]) -> set[tuple[int, ...]]:
    matches: set[tuple[int, ...]] = set()
    target_len = len(sequence)

    def dfs(node: int, depth: int, path: list[int], seen: set[int]) -> None:
        if station_colors[node] != sequence[depth]:
            return
        if depth == target_len - 1:
            matches.add(_normalize_path(path))
            return
        next_color = sequence[depth + 1]
        for nb in adjacency[node]:
            if nb in seen or station_colors[nb] != next_color:
                continue
            seen.add(nb)
            path.append(nb)
            dfs(nb, depth + 1, path, seen)
            path.pop()
            seen.remove(nb)

    for node, color in enumerate(station_colors):
        if color != sequence[0]:
            continue
        dfs(node, 0, [node], {node})
    return matches


def _find_unique_matching_path(adjacency, station_colors: list[int], sequence: tuple[int, ...]) -> tuple[int, ...]:
    matches = _matching_paths(adjacency, station_colors, sequence)
    if sequence[::-1] != sequence:
        matches |= _matching_paths(adjacency, station_colors, sequence[::-1])
    assert len(matches) == 1, "instruction strip must identify a unique simple path"
    return next(iter(matches))


def main(grid: np.ndarray) -> np.ndarray:
    """
    Find the unique station-color itinerary and recolor its connecting track segments yellow.

    Input structure:
    - The background is the most common color.
    - One connected graph contains solid 2x2 colored station squares linked by one-cell-wide grey track components.
    - A separate straight color strip in a corner lists station colors in travel order.

    Rule:
    - Read the corner strip from one end to the other as a color sequence.
    - In the station graph, find the unique simple path whose visited station colors match that sequence, treating forward and reverse as the same path.
    - Recolor every grey track cell on that path to yellow, leaving stations, the strip, and all other cells unchanged.

    Key insight:
    - The detached strip is an itinerary over station colors, so the solution requires graph-path matching rather than direct color lookup.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg, graph_mask, _strip_mask, strip_coords = _split_graph_and_strip(g)
    stations, station_of_cell = _detect_stations(g, graph_mask, bg)
    adjacency, edge_cells = _build_station_graph(g, graph_mask, stations, station_of_cell)

    sequence = _read_strip_sequence(g, strip_coords)
    station_colors = [station["color"] for station in stations]
    path = _find_unique_matching_path(adjacency, station_colors, sequence)

    out = g.copy()
    for a, b in zip(path, path[1:]):
        for r, c in edge_cells[tuple(sorted((a, b)))]:
            out[r, c] = PATH_COLOR
    return out


def _lattice_neighbors(cell: tuple[int, int], rows: int, cols: int) -> list[tuple[int, int]]:
    r, c = cell
    out = []
    for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        nr, nc = r + dr, c + dc
        if 0 <= nr < rows and 0 <= nc < cols:
            out.append((nr, nc))
    return out


def _sample_connected_cells(rng: np.random.Generator, rows: int, cols: int, node_count: int):
    start = (int(rng.integers(rows)), int(rng.integers(cols)))
    cells = [start]
    chosen = {start}
    tree_edges: list[tuple[tuple[int, int], tuple[int, int]]] = []

    while len(cells) < node_count:
        frontier = []
        for cell in cells:
            for nb in _lattice_neighbors(cell, rows, cols):
                if nb not in chosen:
                    frontier.append((nb, cell))
        if not frontier:
            return None
        idx = int(rng.integers(len(frontier)))
        new_cell, parent = frontier[idx]
        if new_cell in chosen:
            continue
        chosen.add(new_cell)
        cells.append(new_cell)
        tree_edges.append((new_cell, parent))

    return cells, tree_edges


def _make_abstract_graph(rng: np.random.Generator):
    lattice_rows = int(rng.integers(2, 5))
    lattice_cols = int(rng.integers(3, 6))
    max_nodes = min(7, lattice_rows * lattice_cols)
    node_count = int(rng.integers(4, max_nodes + 1))
    sampled = _sample_connected_cells(rng, lattice_rows, lattice_cols, node_count)
    if sampled is None:
        return None
    cells, tree_edges = sampled

    node_of = {cell: i for i, cell in enumerate(cells)}
    edges = {tuple(sorted((node_of[a], node_of[b]))) for a, b in tree_edges}
    all_adjacent = []
    for cell, i in node_of.items():
        for nb in _lattice_neighbors(cell, lattice_rows, lattice_cols):
            j = node_of.get(nb)
            if j is not None and i < j:
                all_adjacent.append((i, j))

    rng.shuffle(all_adjacent)
    extra_budget = int(rng.integers(0, 3))
    for edge in all_adjacent:
        if edge in edges:
            continue
        if extra_budget <= 0:
            break
        if rng.random() < 0.35:
            edges.add(edge)
            extra_budget -= 1

    adjacency = {i: set() for i in range(node_count)}
    for a, b in edges:
        adjacency[a].add(b)
        adjacency[b].add(a)

    return {
        "lattice_rows": lattice_rows,
        "lattice_cols": lattice_cols,
        "cells": cells,
        "adjacency": adjacency,
        "edges": sorted(edges),
    }


def _enumerate_paths(adjacency, min_nodes: int, max_nodes: int) -> list[tuple[int, ...]]:
    paths: set[tuple[int, ...]] = set()
    node_count = len(adjacency)
    for start in range(node_count):
        stack = [(start, [start], {start})]
        while stack:
            node, path, seen = stack.pop()
            if len(path) >= min_nodes:
                paths.add(_normalize_path(path))
            if len(path) == max_nodes:
                continue
            for nb in adjacency[node]:
                if nb in seen:
                    continue
                stack.append((nb, path + [nb], seen | {nb}))
    return sorted(paths, key=lambda p: (len(p), p))


def _unique_sequence_candidates(adjacency, node_colors: list[int]) -> list[tuple[tuple[int, ...], tuple[int, ...]]]:
    max_nodes = min(6, len(node_colors))
    candidates = []
    for path in _enumerate_paths(adjacency, min_nodes=3, max_nodes=max_nodes):
        sequence = tuple(node_colors[i] for i in path)
        if len(set(sequence)) < 2:
            continue
        matches = _matching_paths(adjacency, node_colors, sequence)
        if sequence[::-1] != sequence:
            matches |= _matching_paths(adjacency, node_colors, sequence[::-1])
        if matches == {_normalize_path(path)}:
            candidates.append((_normalize_path(path), sequence))
    return candidates


def _choose_colors_and_path(rng: np.random.Generator, adjacency):
    node_count = len(adjacency)
    for _ in range(COLOR_ASSIGNMENT_ATTEMPTS):
        palette_size = int(rng.integers(3, min(5, len(STATION_COLORS)) + 1))
        palette = [int(x) for x in rng.choice(STATION_COLORS, size=palette_size, replace=False)]
        node_colors = [int(palette[int(rng.integers(palette_size))]) for _ in range(node_count)]
        if len(set(node_colors)) < 3:
            continue
        if len(set(node_colors)) == node_count:
            continue
        candidates = _unique_sequence_candidates(adjacency, node_colors)
        if not candidates:
            continue
        lengths = [len(path) for path, _ in candidates]
        max_len = max(lengths)
        preferred = [item for item in candidates if len(item[0]) == max_len]
        path, sequence = preferred[int(rng.integers(len(preferred)))]
        return node_colors, path, sequence
    return None


def _layout_canvas(graph_h: int, graph_w: int, strip_len: int, case: str):
    if case.endswith("_h"):
        top = case.startswith("top")
        left = "_left_" in case
        graph_top = 4 if top else 1
        graph_left = 1
        strip_r = 1 if top else graph_top + graph_h + 2
        height = strip_r + 2 if not top else graph_top + graph_h + 1
        width = max(graph_left + graph_w + 1, strip_len + 2)
        strip_c = 1 if left else width - 1 - strip_len
    else:
        left = "_left_" in case
        top = case.startswith("top")
        graph_top = 1
        graph_left = 4 if left else 1
        strip_c = 1 if left else graph_left + graph_w + 2
        width = strip_c + 2 if not left else graph_left + graph_w + 1
        height = max(graph_top + graph_h + 1, strip_len + 2)
        strip_r = 1 if top else height - 1 - strip_len
    return height, width, graph_top, graph_left, strip_r, strip_c


def _draw_graph(grid: np.ndarray, graph_info, node_colors: list[int], pitch: int, graph_top: int, graph_left: int, rng: np.random.Generator):
    station_pos = {}
    for idx, (lr, lc) in enumerate(graph_info["cells"]):
        r0 = graph_top + lr * pitch
        c0 = graph_left + lc * pitch
        grid[r0 : r0 + 2, c0 : c0 + 2] = node_colors[idx]
        station_pos[idx] = (r0, c0)

    edge_cells = {}
    for a, b in graph_info["edges"]:
        r1, c1 = station_pos[a]
        r2, c2 = station_pos[b]
        cells = []
        if r1 == r2:
            row = r1 + int(rng.integers(0, 2))
            start = min(c1, c2) + 2
            stop = max(c1, c2)
            for c in range(start, stop):
                grid[row, c] = TRACK_COLOR
                cells.append((row, c))
        else:
            col = c1 + int(rng.integers(0, 2))
            start = min(r1, r2) + 2
            stop = max(r1, r2)
            for r in range(start, stop):
                grid[r, col] = TRACK_COLOR
                cells.append((r, col))
        edge_cells[(a, b)] = cells
    return station_pos, edge_cells


def _draw_strip(grid: np.ndarray, sequence: tuple[int, ...], case: str, strip_r: int, strip_c: int) -> None:
    if case.endswith("_h"):
        for i, color in enumerate(sequence):
            grid[strip_r, strip_c + i] = color
    else:
        for i, color in enumerate(sequence):
            grid[strip_r + i, strip_c] = color


def _build_expected_output(grid: np.ndarray, path: tuple[int, ...], edge_cells) -> np.ndarray:
    out = grid.copy()
    for a, b in zip(path, path[1:]):
        key = tuple(sorted((a, b)))
        for r, c in edge_cells[key]:
            out[r, c] = PATH_COLOR
    return out


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    graph_info = _make_abstract_graph(rng)
    if graph_info is None:
        return None

    chosen = _choose_colors_and_path(rng, graph_info["adjacency"])
    if chosen is None:
        return None
    node_colors, path, sequence = chosen

    if rng.random() < 0.5:
        sequence = sequence[::-1]

    pitch = int(PITCH_CHOICES[int(rng.integers(len(PITCH_CHOICES)))])
    graph_h = (graph_info["lattice_rows"] - 1) * pitch + 2
    graph_w = (graph_info["lattice_cols"] - 1) * pitch + 2
    case = STRIP_CASES[int(rng.integers(len(STRIP_CASES)))]
    height, width, graph_top, graph_left, strip_r, strip_c = _layout_canvas(graph_h, graph_w, len(sequence), case)
    if height > 30 or width > 30:
        return None

    grid = np.full((height, width), BACKGROUND, dtype=int)
    _, edge_cells = _draw_graph(grid, graph_info, node_colors, pitch, graph_top, graph_left, rng)
    _draw_strip(grid, sequence, case, strip_r, strip_c)

    expected = _build_expected_output(grid, path, edge_cells)
    try:
        actual = main(grid)
    except AssertionError:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid itinerary-path puzzle")
