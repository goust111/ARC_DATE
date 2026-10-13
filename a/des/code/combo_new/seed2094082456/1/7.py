import random
from collections import defaultdict

import numpy as np


BG = 0
COLOR_POOL = tuple(range(1, 10))
PATH_COUNT_VALUES = (2, 3, 4)
MAX_ATTEMPTS = 400


def _bg_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _sample_spaced(low: int, high: int, count: int, gap: int) -> list[int] | None:
    values = list(range(low, high + 1))
    if count == 0:
        return []
    if len(values) < count:
        return None
    for _ in range(120):
        picked = sorted(random.sample(values, count))
        if all(b - a >= gap for a, b in zip(picked, picked[1:])):
            return picked
    return None


def _sample_left_endpoint(max_col: int) -> int:
    assert max_col >= 0
    if max_col == 0 or random.random() < 0.5:
        return 0
    return random.randint(1, max_col)


def _sample_right_endpoint(min_col: int, max_col: int) -> int:
    assert min_col <= max_col
    if min_col == max_col or random.random() < 0.5:
        return max_col
    return random.randint(min_col, max_col - 1)


def _sample_top_endpoint(max_row: int) -> int:
    assert max_row >= 0
    if max_row == 0 or random.random() < 0.5:
        return 0
    return random.randint(1, max_row)


def _sample_bottom_endpoint(min_row: int, max_row: int) -> int:
    assert min_row <= max_row
    if min_row == max_row or random.random() < 0.5:
        return max_row
    return random.randint(min_row, max_row - 1)


def _find_legend(grid: np.ndarray, bg: int) -> tuple[int, list[int], list[int]]:
    h, w = grid.shape
    non_bg_cols = np.flatnonzero(np.any(grid != bg, axis=0))
    assert non_bg_cols.size > 0, "expected a non-background legend column"
    legend_c = int(non_bg_cols[-1])
    assert legend_c == w - 1, "legend must occupy the rightmost column"
    if non_bg_cols.size >= 2:
        assert legend_c - int(non_bg_cols[-2]) >= 2, "legend must be separated from the paths"

    legend_rows = [int(r) for r in np.flatnonzero(grid[:, legend_c] != bg)]
    assert legend_rows, "legend column must contain at least one color"
    assert legend_rows[-1] == h - 1, "legend must sit in the bottom-right corner"
    assert legend_rows == list(range(legend_rows[0], h)), "legend rows must be contiguous"

    legend_colors = [int(grid[r, legend_c]) for r in legend_rows]
    assert len(set(legend_colors)) == len(legend_colors), "legend colors must be distinct"
    return legend_c, legend_rows, legend_colors


def _top_orientation(grid: np.ndarray, color: int, row: int, col: int) -> str | None:
    h, w = grid.shape
    horizontal = 0 < col < w - 1 and int(grid[row, col - 1]) == color and int(grid[row, col + 1]) == color
    vertical = 0 < row < h - 1 and int(grid[row - 1, col]) == color and int(grid[row + 1, col]) == color
    assert not (horizontal and vertical), "a visible path pixel may not belong to two straight axes"
    if horizontal:
        return "h"
    if vertical:
        return "v"
    return None


def _hidden_candidate(
    grid: np.ndarray,
    row: int,
    col: int,
    dr: int,
    dc: int,
    top_color: int,
    path_colors: set[int],
    bg: int,
) -> tuple[int, list[tuple[int, int]]] | None:
    h, w = grid.shape

    def value_at(step: int) -> int | None:
        rr = row + dr * step
        cc = col + dc * step
        if 0 <= rr < h and 0 <= cc < w:
            return int(grid[rr, cc])
        return None

    near_neg = value_at(-1)
    near_pos = value_at(1)
    far_neg = value_at(-2)
    far_pos = value_at(2)

    candidates: list[tuple[int, list[tuple[int, int]]]] = []

    if near_neg is not None and near_neg == near_pos and near_neg in path_colors and near_neg != top_color:
        candidates.append((int(near_neg), []))

    if (
        near_neg == bg
        and near_pos == bg
        and far_neg is not None
        and far_neg == far_pos
        and far_neg in path_colors
        and far_neg != top_color
    ):
        candidates.append(
            (
                int(far_neg),
                [(row - dr, col - dc), (row + dr, col + dc)],
            )
        )

    if not candidates:
        return None
    assert len(candidates) == 1, "each crossing may hide at most one lower path"
    return candidates[0]


def _crossing_relations(
    grid: np.ndarray,
    legend_c: int,
    path_colors: set[int],
    bg: int,
) -> tuple[set[tuple[int, int]], dict[tuple[int, int], int]]:
    h, _w = grid.shape
    edges: set[tuple[int, int]] = set()
    fills: dict[tuple[int, int], int] = {}

    for row in range(h):
        for col in range(legend_c):
            top_color = int(grid[row, col])
            if top_color not in path_colors:
                continue

            orient = _top_orientation(grid, top_color, row, col)
            if orient is None:
                continue

            if orient == "h":
                hidden = _hidden_candidate(grid, row, col, 1, 0, top_color, path_colors, bg)
            else:
                hidden = _hidden_candidate(grid, row, col, 0, 1, top_color, path_colors, bg)

            if hidden is None:
                continue

            lower_color, gap_cells = hidden
            edges.add((top_color, lower_color))
            for rr, cc in gap_cells:
                assert 0 <= rr < h and 0 <= cc < legend_c
                prev = fills.get((rr, cc))
                assert prev in (None, lower_color), "gap cells may not belong to multiple lower paths"
                fills[(rr, cc)] = int(lower_color)

    return edges, fills


def _unique_topological_order(colors: list[int], edges: set[tuple[int, int]]) -> list[int]:
    followers = {color: set() for color in colors}
    indegree = {color: 0 for color in colors}
    for upper, lower in edges:
        assert upper in followers and lower in followers and upper != lower
        if lower not in followers[upper]:
            followers[upper].add(lower)
            indegree[lower] += 1

    ready = [color for color in colors if indegree[color] == 0]
    order: list[int] = []
    while ready:
        assert len(ready) == 1, "crossing relations must determine a unique layer order"
        color = ready.pop()
        order.append(color)
        for nxt in sorted(followers[color]):
            indegree[nxt] -= 1
            if indegree[nxt] == 0:
                ready.append(nxt)

    assert len(order) == len(colors), "layer graph must be acyclic"
    return order


def main(grid: np.ndarray) -> np.ndarray:
    """
    Infer the over-under order of crossed paths and recolor them by the bottom-right legend.

    Input structure:
    - The background is black.
    - Several one-pixel-wide colored orthogonal paths occupy the main field, with one unique color per path.
    - When two paths cross, the upper path stays continuous through the crossing pixel, while the lower path is interrupted on the two immediately adjacent cells along its own axis.
    - The rightmost column contains a bottom-right vertical legend whose distinct colors, read top to bottom, encode the repainting for layer 1, layer 2, and so on.

    Rule:
    - Ignore the legend while scanning the main field and detect every crossing by finding a straight visible path cell whose perpendicular axis shows a hidden same-color continuation.
    - Each crossing yields one relation: the continuous visible color is above the interrupted color.
    - Topologically sort those relations to obtain the unique global order from top layer to bottom layer.
    - Recolor every path with the legend color for its layer, keep the visible upper crossing pixels, and fill the lower path's adjacent black gap cells with that lower layer's new color. Keep the legend unchanged.

    Key insight:
    - The path breaks are not damage; they are an explicit occlusion code. A continuous strand at the crossing is higher, and the interrupted strand becomes recoverable once those local above-below clues are chained into one global depth order.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg = BG if np.any(g == BG) else _bg_color(g)
    legend_c, _legend_rows, legend_colors = _find_legend(g, bg)

    path_colors = [int(color) for color in np.unique(g[:, :legend_c]) if int(color) != bg]
    assert path_colors, "expected at least one path color"
    assert len(path_colors) == len(legend_colors), "legend height must equal the number of paths"

    edges, fills = _crossing_relations(g, legend_c, set(path_colors), bg)
    order = _unique_topological_order(path_colors, edges)

    repaint = {color: int(legend_colors[idx]) for idx, color in enumerate(order)}
    out = g.copy()
    for color, new_color in repaint.items():
        out[g == color] = new_color

    for (row, col), color in fills.items():
        assert int(g[row, col]) == bg, "filled under-pass cells must be background in the input"
        out[row, col] = repaint[color]

    return out


def _build_paths(
    height: int,
    play_width: int,
    orientations: list[str],
    rows: list[int],
    cols: list[int],
    colors: list[int],
) -> list[dict[str, object]]:
    h_idx = 0
    v_idx = 0
    vmin, vmax = min(cols), max(cols)
    hmin, hmax = min(rows), max(rows)

    paths: list[dict[str, object]] = []
    for layer, (orient, color) in enumerate(zip(orientations, colors, strict=True)):
        if orient == "h":
            row = int(rows[h_idx])
            h_idx += 1
            start = _sample_left_endpoint(vmin - 2)
            end = _sample_right_endpoint(vmax + 2, play_width - 1)
            cells = {(row, col) for col in range(start, end + 1)}
            paths.append(
                {
                    "color": int(color),
                    "layer": int(layer),
                    "orient": "h",
                    "row": row,
                    "start": int(start),
                    "end": int(end),
                    "cells": cells,
                }
            )
        else:
            col = int(cols[v_idx])
            v_idx += 1
            start = _sample_top_endpoint(hmin - 2)
            end = _sample_bottom_endpoint(hmax + 2, height - 1)
            cells = {(row, col) for row in range(start, end + 1)}
            paths.append(
                {
                    "color": int(color),
                    "layer": int(layer),
                    "orient": "v",
                    "col": col,
                    "start": int(start),
                    "end": int(end),
                    "cells": cells,
                }
            )

    return paths


def _render_scene(
    height: int,
    width: int,
    paths: list[dict[str, object]],
    legend_colors: list[int],
) -> tuple[np.ndarray, np.ndarray]:
    legend_c = width - 1
    play_width = width - 2
    grid = np.full((height, width), BG, dtype=int)
    expected = np.full((height, width), BG, dtype=int)

    visible = {int(path["color"]): set(path["cells"]) for path in paths}
    fills: dict[tuple[int, int], int] = {}

    horizontals = [path for path in paths if path["orient"] == "h"]
    verticals = [path for path in paths if path["orient"] == "v"]

    for h_path in horizontals:
        row = int(h_path["row"])
        for v_path in verticals:
            col = int(v_path["col"])
            assert (row, col) in h_path["cells"] and (row, col) in v_path["cells"]

            if int(h_path["layer"]) < int(v_path["layer"]):
                top = h_path
                lower = v_path
                gap_cells = [(row - 1, col), (row + 1, col)]
            else:
                top = v_path
                lower = h_path
                gap_cells = [(row, col - 1), (row, col + 1)]

            for cell in gap_cells:
                assert cell in lower["cells"], "under-pass gaps must stay on the lower path"
                visible[int(lower["color"])].discard(cell)
                fills[cell] = int(lower["color"])

            _ = top

    # Paint from bottom layer to top layer so upper crossings remain visible.
    for path in sorted(paths, key=lambda item: int(item["layer"]), reverse=True):
        color = int(path["color"])
        for row, col in visible[color]:
            grid[row, col] = color

    repaint = {int(path["color"]): int(legend_colors[int(path["layer"])]) for path in paths}
    expected[:, :] = grid
    for color, new_color in repaint.items():
        expected[grid == color] = new_color
    for (row, col), color in fills.items():
        expected[row, col] = repaint[color]

    legend_start = height - len(legend_colors)
    for offset, color in enumerate(legend_colors):
        row = legend_start + offset
        grid[row, legend_c] = int(color)
        expected[row, legend_c] = int(color)

    assert not np.any(grid[:, play_width] != BG), "separator column must stay empty"
    return grid, expected


def _gen_attempt() -> np.ndarray | None:
    path_count = random.choice(PATH_COUNT_VALUES)
    start_horizontal = random.random() < 0.5
    orientations = [
        "h" if ((idx % 2 == 0) == start_horizontal) else "v"
        for idx in range(path_count)
    ]

    h_count = orientations.count("h")
    v_count = orientations.count("v")

    height = random.randint(12, 28)
    width = random.randint(16, 30)
    play_width = width - 2
    if play_width < 8:
        return None

    row_choices = _sample_spaced(3, height - 4, h_count, 4)
    col_choices = _sample_spaced(3, play_width - 4, v_count, 4)
    if row_choices is None or col_choices is None:
        return None

    if min(row_choices) < 2 or min(col_choices) < 2:
        return None

    colors = random.sample(COLOR_POOL, 2 * path_count)
    path_colors = colors[:path_count]
    legend_colors = colors[path_count:]

    paths = _build_paths(height, play_width, orientations, row_choices, col_choices, path_colors)
    if any(len(path["cells"]) < 5 for path in paths):
        return None

    grid, expected = _render_scene(height, width, paths, legend_colors)

    # All paths must participate in the layer reasoning.
    if len({int(grid[row, col]) for row, col in np.argwhere(grid[:, : width - 1] != BG)}) < path_count:
        return None

    try:
        actual = main(grid)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {MAX_ATTEMPTS} attempts")
