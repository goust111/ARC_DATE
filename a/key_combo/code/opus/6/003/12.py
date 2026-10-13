from __future__ import annotations

from collections import deque
import random

import numpy as np


BG = 0
CORRIDOR = 5
PATH_CORRIDOR = 6
BAR_COLORS = (1, 2, 3, 4, 7, 8, 9)
HEIGHT_VALUES = tuple(range(2, 9))
BAR_COUNT_RANGE = (5, 7)
GAP_RANGE = (1, 3)
GRID_H_RANGE = (12, 17)
MARGIN_RANGE = (1, 2)
MAX_GENERATION_ATTEMPTS = 500


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _extract_bars(grid: np.ndarray, bg: int) -> list[dict[str, int]]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bars: list[dict[str, int]] = []
    for c in range(w):
        rows = np.flatnonzero(
            (g[:, c] != int(bg)) & (g[:, c] != CORRIDOR) & (g[:, c] != PATH_CORRIDOR)
        )
        if rows.size == 0:
            continue
        colors = np.unique(g[rows, c])
        assert colors.size == 1
        top = int(rows[0])
        assert np.array_equal(rows, np.arange(top, h))
        bars.append(
            {
                "col": int(c),
                "top": top,
                "height": int(h - top),
                "color": int(colors[0]),
            }
        )

    bars.sort(key=lambda bar: bar["col"])
    assert BAR_COUNT_RANGE[0] <= len(bars) <= BAR_COUNT_RANGE[1]
    assert len({bar["col"] for bar in bars}) == len(bars)
    assert len({bar["color"] for bar in bars}) == len(bars)
    assert len({bar["height"] for bar in bars}) == len(bars)
    return bars


def _extract_edges(grid: np.ndarray, bars: list[dict[str, int]], bg: int) -> dict[tuple[int, int], dict[str, object]]:
    g = np.asarray(grid, dtype=int)
    h, _w = g.shape
    edges: dict[tuple[int, int], dict[str, object]] = {}

    for i in range(len(bars) - 1):
        left = bars[i]
        right = bars[i + 1]
        left_col = int(left["col"])
        right_col = int(right["col"])
        assert right_col - left_col >= 2

        gap = g[:, left_col + 1 : right_col]
        assert gap.shape[1] >= 1

        grey_rows = np.flatnonzero(np.any(gap == CORRIDOR, axis=1))
        if grey_rows.size == 0:
            continue

        assert grey_rows.size == 1
        row = int(grey_rows[0])
        assert int(np.count_nonzero(gap == CORRIDOR)) == gap.shape[1]
        assert np.all(gap[row] == CORRIDOR)
        assert np.all(gap[np.arange(h) != row] != CORRIDOR)
        assert row >= max(int(left["top"]), int(right["top"]))

        edges[(i, i + 1)] = {
            "row": row,
            "cells": [(row, c) for c in range(left_col + 1, right_col)],
        }

    return edges


def _find_unique_extrema(bars: list[dict[str, int]]) -> tuple[int, int]:
    heights = [int(bar["height"]) for bar in bars]
    highest = max(heights)
    lowest = min(heights)
    assert heights.count(highest) == 1
    assert heights.count(lowest) == 1
    return heights.index(highest), heights.index(lowest)


def _build_adjacency(num_nodes: int, edges: dict[tuple[int, int], dict[str, object]]) -> list[list[int]]:
    adj = [[] for _ in range(num_nodes)]
    for (a, b) in edges:
        adj[int(a)].append(int(b))
        adj[int(b)].append(int(a))
    for nbrs in adj:
        nbrs.sort()
    return adj


def _unique_shortest_path(adj: list[list[int]], start: int, goal: int) -> list[int]:
    dist = [-1] * len(adj)
    ways = [0] * len(adj)
    prev: list[int | None] = [None] * len(adj)
    q: deque[int] = deque([int(start)])
    dist[int(start)] = 0
    ways[int(start)] = 1

    while q:
        u = int(q.popleft())
        for v in adj[u]:
            if dist[v] < 0:
                dist[v] = dist[u] + 1
                ways[v] = ways[u]
                prev[v] = u
                q.append(v)
            elif dist[v] == dist[u] + 1:
                ways[v] = min(2, ways[v] + ways[u])

    assert dist[int(goal)] >= 0
    assert ways[int(goal)] == 1

    path: list[int] = []
    cur: int | None = int(goal)
    while cur is not None:
        path.append(int(cur))
        cur = prev[int(cur)]
    path.reverse()
    return path


def _analyze_grid(grid: np.ndarray) -> dict[str, object]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    bg = _get_bg(g)
    bars = _extract_bars(g, bg)
    edges = _extract_edges(g, bars, bg)
    start, goal = _find_unique_extrema(bars)
    adj = _build_adjacency(len(bars), edges)
    node_path = _unique_shortest_path(adj, start, goal)
    edge_path = [tuple(sorted((node_path[i], node_path[i + 1]))) for i in range(len(node_path) - 1)]

    return {
        "bg": bg,
        "bars": bars,
        "edges": edges,
        "start": int(start),
        "goal": int(goal),
        "node_path": node_path,
        "edge_path": edge_path,
    }


def _expected_bar_state(analysis: dict[str, object]) -> list[dict[str, int]]:
    bars = [dict(bar) for bar in analysis["bars"]]
    path = list(analysis["node_path"])
    path_colors = [int(bars[idx]["color"]) for idx in path]
    path_heights = [int(bars[idx]["height"]) for idx in path]

    rotated_colors = [path_colors[-1], *path_colors[:-1]]
    rotated_heights = [*path_heights[1:], path_heights[0]]

    for idx, color, height in zip(path, rotated_colors, rotated_heights, strict=True):
        bars[int(idx)]["color"] = int(color)
        bars[int(idx)]["height"] = int(height)
    return bars


def main(grid: np.ndarray) -> np.ndarray:
    """
    Repaint the unique bar-graph shortest path chosen by the tallest and shortest bars.

    Input structure:
    - The background is black, with 5 to 7 bottom-aligned one-cell-wide vertical bars in distinct colors.
    - Each bar has a unique height, and some adjacent bar pairs are connected by a one-row grey corridor across the gap between their columns.
    - The tallest bar and the shortest bar are unique and serve as the path endpoints.

    Rule:
    - Treat each vertical bar as a graph node and each valid grey corridor between adjacent bars as an undirected edge.
    - Find the unique shortest path, measured by corridor count, from the tallest bar to the shortest bar.
    - Along that path, rotate bar colors one step to the right and rotate bar heights one step to the left; bars outside the path stay unchanged.
    - Recolor only the corridor segments on the chosen path to magenta, and keep every other non-path corridor and background cell unchanged.

    Key insight:
    - The visible bar heights determine the graph endpoints, while the grey horizontal links determine which subset of bars participates in the local color-height cyclic rewrite.
    """
    g = np.asarray(grid, dtype=int)
    h, _w = g.shape
    analysis = _analyze_grid(g)
    bars = analysis["bars"]
    edges = analysis["edges"]
    edge_path = set(analysis["edge_path"])
    final_bars = _expected_bar_state(analysis)

    out = g.copy()
    for bar in bars:
        out[:, int(bar["col"])] = int(analysis["bg"])

    for bar in final_bars:
        col = int(bar["col"])
        height = int(bar["height"])
        color = int(bar["color"])
        out[h - height : h, col] = color

    for edge_key in edge_path:
        edge = edges[edge_key]
        row = int(edge["row"])
        left_col = int(bars[edge_key[0]]["col"])
        right_col = int(bars[edge_key[1]]["col"])
        out[row, left_col + 1 : right_col] = PATH_CORRIDOR

    return out


def _render_case(
    height: int,
    width: int,
    bars: list[dict[str, int]],
    edge_rows: dict[int, int],
) -> np.ndarray:
    grid = np.full((height, width), BG, dtype=int)
    for bar in bars:
        col = int(bar["col"])
        bar_height = int(bar["height"])
        grid[height - bar_height : height, col] = int(bar["color"])
    for i, row in edge_rows.items():
        left_col = int(bars[i]["col"])
        right_col = int(bars[i + 1]["col"])
        grid[int(row), left_col + 1 : right_col] = CORRIDOR
    return grid


def _validate_output(inp: np.ndarray, out: np.ndarray) -> None:
    in_info = _analyze_grid(inp)
    out_bars = _extract_bars(out, _get_bg(out))
    expected_bars = _expected_bar_state(in_info)

    assert len(out_bars) == len(expected_bars)
    for got, exp in zip(out_bars, expected_bars, strict=True):
        assert int(got["col"]) == int(exp["col"])
        assert int(got["height"]) == int(exp["height"])
        assert int(got["color"]) == int(exp["color"])

    bars = in_info["bars"]
    edge_path = set(in_info["edge_path"])
    for edge_key, edge in in_info["edges"].items():
        row = int(edge["row"])
        left_col = int(bars[edge_key[0]]["col"])
        right_col = int(bars[edge_key[1]]["col"])
        expected_color = PATH_CORRIDOR if edge_key in edge_path else CORRIDOR
        assert np.all(out[row, left_col + 1 : right_col] == expected_color)

    protected = np.ones(inp.shape, dtype=bool)
    for bar in bars:
        protected[:, int(bar["col"])] = False
    for edge in in_info["edges"].values():
        for r, c in edge["cells"]:
            protected[int(r), int(c)] = False
    assert np.array_equal(inp[protected], out[protected])
    assert not np.array_equal(inp, out)


def _choose_outside_edge_states(rng: random.Random, outside: list[int]) -> dict[int, bool]:
    states: dict[int, bool] = {}
    if not outside:
        return states
    if len(outside) == 1:
        states[outside[0]] = bool(rng.getrandbits(1))
        return states

    forced = outside[:]
    rng.shuffle(forced)
    states[forced[0]] = True
    states[forced[1]] = False
    for idx in forced[2:]:
        states[idx] = bool(rng.getrandbits(1))
    return states


def _generate_candidate(rng: random.Random) -> np.ndarray | None:
    bar_count = rng.randint(BAR_COUNT_RANGE[0], BAR_COUNT_RANGE[1])
    path_edge_count = rng.randint(2, min(4, bar_count - 2))
    path_node_count = path_edge_count + 1

    lo = rng.randint(0, bar_count - path_node_count)
    hi = lo + path_node_count - 1
    if rng.random() < 0.5:
        start_idx, goal_idx = lo, hi
    else:
        start_idx, goal_idx = hi, lo

    colors = rng.sample(list(BAR_COLORS), bar_count)
    interior_heights = rng.sample([v for v in HEIGHT_VALUES if v not in (2, 8)], bar_count - 2)
    heights = [None] * bar_count
    heights[start_idx] = 8
    heights[goal_idx] = 2
    fill_iter = iter(interior_heights)
    for i in range(bar_count):
        if heights[i] is None:
            heights[i] = int(next(fill_iter))

    gaps = [rng.randint(GAP_RANGE[0], GAP_RANGE[1]) for _ in range(bar_count - 1)]
    left_margin = rng.randint(MARGIN_RANGE[0], MARGIN_RANGE[1])
    right_margin = rng.randint(MARGIN_RANGE[0], MARGIN_RANGE[1])
    cols = [left_margin]
    for gap in gaps:
        cols.append(cols[-1] + 1 + int(gap))
    width = cols[-1] + 1 + right_margin
    height = rng.randint(GRID_H_RANGE[0], GRID_H_RANGE[1])
    if width > 30 or height > 30:
        return None

    bars = [
        {
            "col": int(cols[i]),
            "height": int(heights[i]),
            "color": int(colors[i]),
        }
        for i in range(bar_count)
    ]

    path_range = range(min(start_idx, goal_idx), max(start_idx, goal_idx))
    outside = [i for i in range(bar_count - 1) if i not in path_range]
    edge_present = {i: True for i in path_range}
    edge_present.update(_choose_outside_edge_states(rng, outside))

    edge_rows: dict[int, int] = {}
    for i in range(bar_count - 1):
        if not edge_present.get(i, False):
            continue
        common_top = height - min(int(heights[i]), int(heights[i + 1]))
        candidates = list(range(common_top, height))
        assert candidates
        edge_rows[i] = rng.choice(candidates)

    path_rows = [edge_rows[i] for i in path_range]
    if len(path_rows) >= 2 and len(set(path_rows)) == 1:
        for i in path_range:
            common_top = height - min(int(heights[i]), int(heights[i + 1]))
            alternatives = [r for r in range(common_top, height) if r != edge_rows[i]]
            if alternatives:
                edge_rows[i] = rng.choice(alternatives)
                break

    grid = _render_case(height, width, bars, edge_rows)
    info = _analyze_grid(grid)
    expected_path = list(range(start_idx, goal_idx + (1 if goal_idx >= start_idx else -1), 1 if goal_idx >= start_idx else -1))
    if info["start"] != start_idx or info["goal"] != goal_idx:
        return None
    if info["node_path"] != expected_path:
        return None
    if len(set(expected_path)) == bar_count:
        return None
    if outside:
        present_outside = [i for i in outside if i in edge_rows]
        absent_outside = [i for i in outside if i not in edge_rows]
        if len(outside) >= 2 and (not present_outside or not absent_outside):
            return None

    out = main(grid)
    _validate_output(grid, out)
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _generate_candidate(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
