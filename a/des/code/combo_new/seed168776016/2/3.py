from __future__ import annotations

from collections import defaultdict, deque
import random

import numpy as np


BG = 0
H_COMBOS = ("ul", "dl", "ur", "dr")
V_COMBOS = ("lt", "rt", "lb", "rb")
MIN_PATHS = 3
MAX_PATHS = 6
MIN_GAP = 2
MAX_SIDE = 30
GEN_ATTEMPTS = 300


def _crossing_edges(grid: np.ndarray) -> tuple[set[tuple[int, int]], set[int]]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    edges: set[tuple[int, int]] = set()
    colors = {int(v) for v in np.unique(g) if int(v) != BG}
    touched: set[int] = set()

    for r in range(1, h - 1):
        for c in range(1, w - 1):
            top = int(g[r, c])
            if top == BG:
                continue

            left = int(g[r, c - 1])
            right = int(g[r, c + 1])
            up = int(g[r - 1, c])
            down = int(g[r + 1, c])

            if left == top and right == top and up == down and up not in {BG, top}:
                edges.add((top, up))
                touched.update((top, up))

            if up == top and down == top and left == right and left not in {BG, top}:
                edges.add((top, left))
                touched.update((top, left))

    return edges, touched | colors


def _unique_top_color(colors: set[int], edges: set[tuple[int, int]]) -> int:
    above: dict[int, set[int]] = {color: set() for color in colors}
    indegree = {color: 0 for color in colors}
    for top, bottom in edges:
        if bottom not in above[top]:
            above[top].add(bottom)
            indegree[bottom] += 1

    queue = deque(sorted(color for color in colors if indegree[color] == 0))
    order: list[int] = []
    while queue:
        if len(queue) != 1:
            raise AssertionError("crossing relations must determine a unique global order")
        color = queue.popleft()
        order.append(color)
        for lower in sorted(above[color]):
            indegree[lower] -= 1
            if indegree[lower] == 0:
                queue.append(lower)

    if len(order) != len(colors):
        raise AssertionError("crossing relations must be acyclic and include every path")
    return int(order[0])


def main(grid: np.ndarray) -> np.ndarray:
    """
    Infer the global over-under order of orthogonal colored paths from their crossings and keep only the topmost path.

    Input structure:
    - The background is black.
    - Each non-black color forms one single-pixel-wide orthogonal path made of
      horizontal and vertical segments.
    - Whenever two paths cross, the visible crossing pixel belongs to the path
      on top while the lower path continues straight through underneath.

    Rule:
    - Detect crossings by looking for a straight visible segment of one color
      whose crossing cell sits between two opposite neighbors of another color.
    - Read each such local pattern as a depth comparison: the visible color is
      above the hidden color, then combine all comparisons into the unique
      global depth order of the paths.
    - Keep only the pixels of the globally topmost path and clear every other
      pixel to black.

    Key insight:
    - A hidden lower path leaves a color-matched gap across the crossing cell,
      so local over-under clues can be composed into a global ranking whose
      maximum is the only path that remains visible everywhere.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    assert max(g.shape) <= MAX_SIDE

    edges, colors = _crossing_edges(g)
    top_color = _unique_top_color(colors, edges)
    return np.where(g == top_color, top_color, BG).astype(int)


def _sample_spaced_positions(count: int, lo: int, hi: int, gap: int) -> list[int] | None:
    if count == 0:
        return []
    span = hi - lo + 1
    compressed = span - (count - 1) * (gap - 1)
    if compressed < count:
        return None
    picks = sorted(random.sample(range(compressed), count))
    return [lo + picks[i] + i * (gap - 1) for i in range(count)]


def _path_orientations(path_count: int) -> list[str]:
    first = random.choice(("h", "v"))
    return [first if i % 2 == 0 else ("v" if first == "h" else "h") for i in range(path_count)]


def _build_horizontal_path(
    rows: int,
    col_left: int,
    col_right: int,
    row: int,
    combo: str,
) -> set[tuple[int, int]]:
    cells = {(row, c) for c in range(col_left, col_right + 1)}
    tail_col = col_left if combo in {"ul", "dl"} else col_right
    if combo in {"ul", "ur"}:
        cells.update((r, tail_col) for r in range(0, row + 1))
    else:
        cells.update((r, tail_col) for r in range(row, rows))
    return cells


def _build_vertical_path(
    cols: int,
    row_top: int,
    row_bottom: int,
    col: int,
    combo: str,
) -> set[tuple[int, int]]:
    cells = {(r, col) for r in range(row_top, row_bottom + 1)}
    tail_row = row_top if combo in {"lt", "rt"} else row_bottom
    if combo in {"lt", "lb"}:
        cells.update((tail_row, c) for c in range(0, col + 1))
    else:
        cells.update((tail_row, c) for c in range(col, cols))
    return cells


def _all_paths_cross_centrally(paths: list[dict], h_rows: list[int], v_cols: list[int]) -> bool:
    central = {(r, c) for r in h_rows for c in v_cols}
    for path in paths:
        cells = path["cells"]
        if path["orient"] == "h":
            if not central.intersection(cells):
                return False
        else:
            if not central.intersection(cells):
                return False
    return True


def _render_paths(rows: int, cols: int, paths: list[dict]) -> np.ndarray:
    occupancy: dict[tuple[int, int], list[int]] = defaultdict(list)
    depth_rank = {path["id"]: depth for depth, path in enumerate(paths)}
    grid = np.full((rows, cols), BG, dtype=int)

    for path in paths:
        for cell in path["cells"]:
            occupancy[cell].append(path["id"])

    for (r, c), owners in occupancy.items():
        if len(owners) > 2:
            raise AssertionError("at most two paths may overlap at one cell")
        top_id = min(owners, key=lambda pid: depth_rank[pid])
        top_color = next(path["color"] for path in paths if path["id"] == top_id)
        grid[r, c] = int(top_color)

    return grid


def _make_paths(rows: int, cols: int, orientations: list[str], colors: list[int]) -> list[dict] | None:
    n_h = orientations.count("h")
    n_v = orientations.count("v")

    h_rows = _sample_spaced_positions(n_h, 2, rows - 3, MIN_GAP)
    v_cols = _sample_spaced_positions(n_v, 2, cols - 3, MIN_GAP)
    if h_rows is None or v_cols is None:
        return None

    row_top = min(h_rows) - 1
    row_bottom = max(h_rows) + 1
    col_left = min(v_cols) - 1
    col_right = max(v_cols) + 1

    h_combos = random.sample(H_COMBOS, n_h)
    v_combos = random.sample(V_COMBOS, n_v)

    h_iter = iter(zip(h_rows, h_combos))
    v_iter = iter(zip(v_cols, v_combos))
    paths: list[dict] = []
    for idx, (orient, color) in enumerate(zip(orientations, colors)):
        if orient == "h":
            row, combo = next(h_iter)
            cells = _build_horizontal_path(rows, col_left, col_right, row, combo)
        else:
            col, combo = next(v_iter)
            cells = _build_vertical_path(cols, row_top, row_bottom, col, combo)
        paths.append(
            {
                "id": idx,
                "color": int(color),
                "orient": orient,
                "combo": combo,
                "cells": cells,
            }
        )

    occupancy: dict[tuple[int, int], int] = defaultdict(int)
    for path in paths:
        for cell in path["cells"]:
            occupancy[cell] += 1
    if any(count > 2 for count in occupancy.values()):
        return None
    if not any(count == 2 for count in occupancy.values()):
        return None
    if not _all_paths_cross_centrally(paths, h_rows, v_cols):
        return None
    return paths


def _expected_output(grid: np.ndarray, top_color: int) -> np.ndarray:
    return np.where(grid == int(top_color), int(top_color), BG).astype(int)


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        path_count = random.randint(MIN_PATHS, MAX_PATHS)
        orientations = _path_orientations(path_count)
        n_h = orientations.count("h")
        n_v = orientations.count("v")

        min_rows = max(9, 2 * n_h + 3)
        min_cols = max(9, 2 * n_v + 3)
        rows = random.randint(min_rows, MAX_SIDE)
        cols = random.randint(min_cols, MAX_SIDE)
        colors = random.sample(range(1, 10), path_count)

        paths = _make_paths(rows, cols, orientations, colors)
        if paths is None:
            continue

        grid = _render_paths(rows, cols, paths)
        expected = _expected_output(grid, paths[0]["color"])
        try:
            actual = main(grid)
        except AssertionError:
            continue
        if np.array_equal(actual, expected):
            return grid

    raise RuntimeError("failed to generate a valid path-order input")
