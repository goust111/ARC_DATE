from __future__ import annotations

from collections import defaultdict
import random

import numpy as np


MAROON = 9
DIRS = [(-1, 0), (1, 0), (0, -1), (0, 1)]
GEN_ATTEMPTS = 400


def _background_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[int(np.argmax(counts))])


def _priority_legend(grid: np.ndarray, bg: int) -> tuple[list[int], dict[int, tuple[int, int]], set[tuple[int, int]]]:
    colors: list[int] = []
    by_color: dict[int, tuple[int, int]] = {}
    legend_cells: set[tuple[int, int]] = set()
    for r in range(grid.shape[0]):
        color = int(grid[r, 0])
        if color == bg:
            continue
        assert color not in by_color, "legend colors must be unique"
        colors.append(color)
        by_color[color] = (r, 0)
        legend_cells.add((r, 0))

    assert colors, "expected a non-empty left-column legend"
    assert MAROON not in by_color, "input paths must not already use maroon"
    return colors, by_color, legend_cells


def _visible_cells(grid: np.ndarray, color: int, legend_cell: tuple[int, int]) -> set[tuple[int, int]]:
    coords = {tuple(map(int, coord)) for coord in np.argwhere(grid == int(color))}
    coords.discard(legend_cell)
    assert coords, "each legend color must also appear on a path"
    return coords


def _infer_hidden_crossings(
    grid: np.ndarray,
    bg: int,
    color: int,
    visible: set[tuple[int, int]],
) -> set[tuple[int, int]]:
    height, width = grid.shape
    hidden: set[tuple[int, int]] = set()
    for r, c in visible:
        for dr, dc in ((1, 0), (0, 1)):
            r1, c1 = r + dr, c + dc
            r2, c2 = r + 2 * dr, c + 2 * dc
            if not (0 <= r2 < height and 0 <= c2 < width):
                continue
            if (r2, c2) not in visible:
                continue
            middle = int(grid[r1, c1])
            if middle in (bg, color):
                continue
            hidden.add((r1, c1))

    for r, c in hidden:
        neighbors = [(r + dr, c + dc) for dr, dc in DIRS if (r + dr, c + dc) in visible]
        assert len(neighbors) == 2, "hidden crossing must reconnect exactly two visible path cells"
        (r0, c0), (r1, c1) = neighbors
        assert r0 == r1 or c0 == c1, "hidden crossing must be traversed straight through"
    return hidden


def _path_graph(nodes: set[tuple[int, int]]) -> dict[tuple[int, int], set[tuple[int, int]]]:
    graph = {node: set() for node in nodes}
    for r, c in nodes:
        for dr, dc in DIRS:
            nbr = (r + dr, c + dc)
            if nbr in graph:
                graph[(r, c)].add(nbr)
    return graph


def _trace_path(nodes: set[tuple[int, int]]) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    graph = _path_graph(nodes)
    degrees = {node: len(neighbors) for node, neighbors in graph.items()}
    assert all(degree in (1, 2) for degree in degrees.values()), "path graph must be a simple chain"
    endpoints = sorted(node for node, degree in degrees.items() if degree == 1)
    assert len(endpoints) == 2, "path must have exactly two endpoints"

    order = [endpoints[0]]
    prev: tuple[int, int] | None = None
    cur = endpoints[0]
    while cur != endpoints[1]:
        next_nodes = [nbr for nbr in graph[cur] if nbr != prev]
        assert len(next_nodes) == 1, "simple path tracing became ambiguous"
        prev, cur = cur, next_nodes[0]
        order.append(cur)

    assert len(order) == len(nodes), "path reconstruction left disconnected fragments"
    return order, endpoints


def _reconstruct_paths(
    grid: np.ndarray,
    bg: int,
    priority_colors: list[int],
    legend_by_color: dict[int, tuple[int, int]],
) -> tuple[dict[int, list[tuple[int, int]]], dict[int, list[tuple[int, int]]]]:
    paths: dict[int, list[tuple[int, int]]] = {}
    endpoints: dict[int, list[tuple[int, int]]] = {}
    for color in priority_colors:
        visible = _visible_cells(grid, color, legend_by_color[color])
        hidden = _infer_hidden_crossings(grid, bg, color, visible)
        order, ends = _trace_path(visible | hidden)
        paths[color] = order
        endpoints[color] = ends
    return paths, endpoints


def main(grid: np.ndarray) -> np.ndarray:
    """
    Repaint path crossings according to the left-column priority legend and mark all path endpoints in maroon.

    Input structure:
    - The background is the most common color.
    - Each non-background color draws exactly one orthogonal single-pixel path.
    - The leftmost column contains one 1x1 legend pixel for every path color, ordered
      from highest priority at the top to lowest at the bottom.
    - At crossings only one path color is visible; the lower-priority path is hidden there.

    Rule:
    - Reconstruct each full path, including hidden crossing cells where a straight segment is
      interrupted by another color.
    - At every crossing, show the higher-priority path color from the legend.
    - Recolor the two endpoints of every path to maroon.
    - Leave the legend unchanged.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.shape[1] >= 2

    bg = _background_color(g)
    priority_colors, legend_by_color, legend_cells = _priority_legend(g, bg)
    non_bg = {int(color) for color in np.unique(g) if int(color) != bg}
    assert set(priority_colors) == non_bg, "legend must list every non-background path color exactly once"

    priority_rank = {color: rank for rank, color in enumerate(priority_colors)}
    paths, path_endpoints = _reconstruct_paths(g, bg, priority_colors, legend_by_color)

    claims: dict[tuple[int, int], list[int]] = defaultdict(list)
    for color, order in paths.items():
        for cell in order:
            claims[cell].append(color)

    out = g.copy()
    for cell, colors in claims.items():
        if cell in legend_cells:
            continue
        assert 1 <= len(colors) <= 2, "crossings must involve one or two paths"
        if len(colors) == 1:
            out[cell] = int(colors[0])
            continue
        top_color = min(colors, key=lambda color: priority_rank[color])
        out[cell] = int(top_color)

    for ends in path_endpoints.values():
        for cell in ends:
            if cell in legend_cells:
                continue
            out[cell] = MAROON

    for cell in legend_cells:
        out[cell] = g[cell]
    return out


def _sample_spaced_positions(n: int, lo: int, hi: int, min_gap: int) -> list[int] | None:
    span = hi - lo + 1
    needed = n + (n - 1) * min_gap
    if span < needed:
        return None
    compressed = span - (n - 1) * min_gap
    picks = sorted(random.sample(range(compressed), n))
    return [lo + picks[i] + i * min_gap for i in range(n)]


def _segment_cells(a: tuple[int, int], b: tuple[int, int]) -> list[tuple[int, int]]:
    r0, c0 = a
    r1, c1 = b
    assert r0 == r1 or c0 == c1
    if r0 == r1:
        return [(r0, c) for c in range(min(c0, c1), max(c0, c1) + 1)]
    return [(r, c0) for r in range(min(r0, r1), max(r0, r1) + 1)]


def _polyline_cells(points: list[tuple[int, int]]) -> set[tuple[int, int]]:
    cells: set[tuple[int, int]] = set()
    for start, end in zip(points, points[1:]):
        cells.update(_segment_cells(start, end))
    return cells


def _all_crossings(paths: dict[int, set[tuple[int, int]]]) -> dict[tuple[int, int], list[int]]:
    claims: dict[tuple[int, int], list[int]] = defaultdict(list)
    for color, cells in paths.items():
        for cell in cells:
            claims[cell].append(color)
    return {cell: colors for cell, colors in claims.items() if len(colors) > 1}


def _bg_is_unique_dominant(grid: np.ndarray, bg: int) -> bool:
    values, counts = np.unique(grid, return_counts=True)
    pairs = sorted(((int(count), int(value)) for value, count in zip(values, counts)), reverse=True)
    return pairs[0][1] == int(bg) and (len(pairs) == 1 or pairs[0][0] > pairs[1][0])


def _legend_rows(height: int, n_colors: int) -> list[int]:
    return sorted(random.sample(range(height), n_colors))


def _render_input(
    shape: tuple[int, int],
    bg: int,
    priority_colors: list[int],
    legend_rows: list[int],
    path_cells: dict[int, set[tuple[int, int]]],
) -> np.ndarray:
    grid = np.full(shape, int(bg), dtype=int)
    priority_rank = {color: rank for rank, color in enumerate(priority_colors)}
    claims: dict[tuple[int, int], list[int]] = defaultdict(list)
    for color, cells in path_cells.items():
        for cell in cells:
            claims[cell].append(color)

    for cell, colors in claims.items():
        assert 1 <= len(colors) <= 2, "generator only supports pairwise crossings"
        if len(colors) == 1:
            grid[cell] = int(colors[0])
            continue
        top = min(colors, key=lambda color: priority_rank[color])
        bottom = max(colors, key=lambda color: priority_rank[color])
        grid[cell] = int(top if random.random() < 0.5 else bottom)

    for row, color in zip(legend_rows, priority_colors):
        grid[row, 0] = int(color)
    return grid


def _sample_row_pattern(center_row: int, height: int) -> tuple[int, int] | None:
    above = list(range(0, center_row))
    below = list(range(center_row + 1, height))
    if not above or not below:
        return None

    pattern = random.choice(("split", "split_rev", "both_above", "both_below"))
    if pattern == "split":
        return random.choice(above), random.choice(below)
    if pattern == "split_rev":
        return random.choice(below), random.choice(above)
    if pattern == "both_above":
        return random.choice(above), random.choice(above)
    return random.choice(below), random.choice(below)


def _sample_col_pattern(center_col: int, width: int) -> tuple[int, int] | None:
    left = list(range(2, center_col))
    right = list(range(center_col + 1, width))
    if not left or not right:
        return None

    pattern = random.choice(("split", "split_rev", "both_left", "both_right"))
    if pattern == "split":
        return random.choice(left), random.choice(right)
    if pattern == "split_rev":
        return random.choice(right), random.choice(left)
    if pattern == "both_left":
        return random.choice(left), random.choice(left)
    return random.choice(right), random.choice(right)


def _horizontal_bent_scene() -> tuple[np.ndarray, dict[int, set[tuple[int, int]]]] | None:
    height = random.randint(12, 20)
    width = random.randint(16, 24)
    n_vertical = random.randint(2, 4)

    vertical_cols = _sample_spaced_positions(n_vertical, 4, width - 4, 2)
    if vertical_cols is None:
        return None

    left_choices = list(range(2, min(vertical_cols) - 1))
    right_choices = list(range(max(vertical_cols) + 2, width))
    center_rows = list(range(2, height - 2))
    if not left_choices or not right_choices or not center_rows:
        return None

    main_row = random.choice(center_rows)
    endpoint_rows = _sample_row_pattern(main_row, height)
    if endpoint_rows is None:
        return None

    c_left = random.choice(left_choices)
    c_right = random.choice(right_choices)
    r_left, r_right = endpoint_rows

    path_colors_needed = n_vertical + 1
    bg = random.choice([color for color in range(10) if color != MAROON])
    palette = [color for color in range(10) if color not in {bg, MAROON}]
    priority_colors = random.sample(palette, path_colors_needed)
    assigned = priority_colors[:]
    random.shuffle(assigned)

    bent_color = assigned[0]
    path_cells: dict[int, set[tuple[int, int]]] = {
        bent_color: _polyline_cells([(r_left, c_left), (main_row, c_left), (main_row, c_right), (r_right, c_right)])
    }
    for color, col in zip(assigned[1:], vertical_cols):
        path_cells[color] = {(r, col) for r in range(height)}

    legend_rows = _legend_rows(height, path_colors_needed)
    grid = _render_input((height, width), bg, priority_colors, legend_rows, path_cells)
    return grid, path_cells


def _vertical_bent_scene() -> tuple[np.ndarray, dict[int, set[tuple[int, int]]]] | None:
    height = random.randint(16, 24)
    width = random.randint(12, 20)
    n_horizontal = random.randint(2, 4)

    horizontal_rows = _sample_spaced_positions(n_horizontal, 2, height - 3, 2)
    if horizontal_rows is None:
        return None

    top_choices = list(range(0, min(horizontal_rows) - 1))
    bottom_choices = list(range(max(horizontal_rows) + 2, height))
    center_cols = list(range(4, width - 2))
    if not top_choices or not bottom_choices or not center_cols:
        return None

    main_col = random.choice(center_cols)
    endpoint_cols = _sample_col_pattern(main_col, width)
    if endpoint_cols is None:
        return None

    r_top = random.choice(top_choices)
    r_bottom = random.choice(bottom_choices)
    c_top, c_bottom = endpoint_cols

    path_colors_needed = n_horizontal + 1
    bg = random.choice([color for color in range(10) if color != MAROON])
    palette = [color for color in range(10) if color not in {bg, MAROON}]
    priority_colors = random.sample(palette, path_colors_needed)
    assigned = priority_colors[:]
    random.shuffle(assigned)

    bent_color = assigned[0]
    path_cells: dict[int, set[tuple[int, int]]] = {
        bent_color: _polyline_cells([(r_top, c_top), (r_top, main_col), (r_bottom, main_col), (r_bottom, c_bottom)])
    }
    for color, row in zip(assigned[1:], horizontal_rows):
        path_cells[color] = {(row, c) for c in range(2, width)}

    legend_rows = _legend_rows(height, path_colors_needed)
    grid = _render_input((height, width), bg, priority_colors, legend_rows, path_cells)
    return grid, path_cells


def _gen_attempt() -> np.ndarray | None:
    scene = _horizontal_bent_scene() if random.random() < 0.5 else _vertical_bent_scene()
    if scene is None:
        return None
    grid, path_cells = scene

    if not _bg_is_unique_dominant(grid, _background_color(grid)):
        return None

    crossings = _all_crossings(path_cells)
    if len(crossings) < 2:
        return None

    try:
        out = main(grid)
    except AssertionError:
        return None

    if out.shape != grid.shape:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid puzzle input")
