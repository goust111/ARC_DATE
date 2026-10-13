from __future__ import annotations

from collections import Counter
from typing import Iterable

import numpy as np


BG = 5
HORIZONTAL = "h"
VERTICAL = "v"
PALETTE = (1, 2, 3, 4, 6, 7, 8, 9)
RETRY_LIMIT = 400


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def _spaced_positions(length: int, count: int, rng: np.random.Generator) -> list[int]:
    base = sorted(rng.choice(length - count + 1, size=count, replace=False).tolist())
    return [base[i] + i for i in range(count)]


def _orientation_sequence(h_count: int, v_count: int, rng: np.random.Generator) -> list[str]:
    if h_count == v_count:
        start = str(rng.choice([HORIZONTAL, VERTICAL]))
    else:
        start = HORIZONTAL if h_count > v_count else VERTICAL
    other = VERTICAL if start == HORIZONTAL else HORIZONTAL
    seq = [start if i % 2 == 0 else other for i in range(h_count + v_count)]
    assert seq.count(HORIZONTAL) == h_count
    assert seq.count(VERTICAL) == v_count
    return seq


def _bar_colors(values: Iterable[int], bg: int) -> int:
    colors = {int(v) for v in values}
    assert len(colors) == 1
    color = next(iter(colors))
    assert color != bg
    return color


def _detect_structure(grid: np.ndarray) -> tuple[int, list[dict], list[dict]]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    h, w = g.shape
    bg = BG
    assert np.any(g == bg)

    h_rows = [r for r in range(h) if not np.any(g[r] == bg)]
    v_cols = [c for c in range(w) if not np.any(g[:, c] == bg)]
    assert 3 <= len(h_rows) <= 4
    assert 3 <= len(v_cols) <= 4
    assert len(h_rows) < h and len(v_cols) < w

    non_v_cols = [c for c in range(w) if c not in v_cols]
    non_h_rows = [r for r in range(h) if r not in h_rows]
    assert non_v_cols and non_h_rows

    h_bars = []
    v_bars = []
    used_colors: set[int] = set()

    for idx, r in enumerate(h_rows):
        color = _bar_colors(g[r, non_v_cols], bg)
        assert color not in used_colors
        used_colors.add(color)
        h_bars.append({"node": (HORIZONTAL, idx), "pos": int(r), "color": color})

    for idx, c in enumerate(v_cols):
        color = _bar_colors(g[non_h_rows, c], bg)
        assert color not in used_colors
        used_colors.add(color)
        v_bars.append({"node": (VERTICAL, idx), "pos": int(c), "color": color})

    h_colors = {bar["color"] for bar in h_bars}
    v_colors = {bar["color"] for bar in v_bars}
    for r in range(h):
        allowed = h_colors | (v_colors if r in h_rows else {bg} | v_colors)
        assert set(map(int, g[r])) <= allowed
    for c in range(w):
        allowed = v_colors | (h_colors if c in v_cols else {bg} | h_colors)
        assert set(map(int, g[:, c])) <= allowed

    return bg, h_bars, v_bars


def _infer_unique_order(grid: np.ndarray, h_bars: list[dict], v_bars: list[dict]) -> list[tuple[str, int]]:
    g = np.asarray(grid, dtype=int)
    nodes = [bar["node"] for bar in h_bars + v_bars]
    outgoing = {node: set() for node in nodes}
    indegree = {node: 0 for node in nodes}

    h_map = {bar["node"]: bar for bar in h_bars}
    v_map = {bar["node"]: bar for bar in v_bars}
    horizontal_visible = 0
    vertical_visible = 0

    for h_node, h_bar in h_map.items():
        for v_node, v_bar in v_map.items():
            visible = int(g[h_bar["pos"], v_bar["pos"]])
            if visible == h_bar["color"]:
                lower, upper = v_node, h_node
                horizontal_visible += 1
            elif visible == v_bar["color"]:
                lower, upper = h_node, v_node
                vertical_visible += 1
            else:
                raise AssertionError("crossing color does not match either bar")
            if upper not in outgoing[lower]:
                outgoing[lower].add(upper)
                indegree[upper] += 1

    assert horizontal_visible > 0 and vertical_visible > 0

    order: list[tuple[str, int]] = []
    remaining = set(nodes)
    while remaining:
        zeroes = sorted(node for node in remaining if indegree[node] == 0)
        assert len(zeroes) == 1, "crossing constraints must imply a unique total order"
        node = zeroes[0]
        remaining.remove(node)
        order.append(node)
        for nxt in outgoing[node]:
            indegree[nxt] -= 1

    assert len(order) == len(nodes)
    return order


def _output_from_rows(h_bars: list[dict], v_count: int) -> np.ndarray:
    ordered_rows = sorted(h_bars, key=lambda bar: int(bar["pos"]))
    out = np.zeros((len(ordered_rows), v_count), dtype=int)
    for r, bar in enumerate(ordered_rows):
        out[r, :] = int(bar["color"])
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """Extract the compact horizontal-color matrix from overlaid orthogonal full-span bars.

    Input structure:
    - The grid has a grey background with 3-4 horizontal full-width bars and 3-4 vertical full-height bars, each 1 cell thick.
    - Same-direction bars do not overlap, and every bar uses a distinct foreground color.
    - At each horizontal/vertical crossing, the visible pixel shows the color of the higher-layer bar.

    Rule:
    - Detect the horizontal-bar rows and vertical-bar columns from the full-span non-background strips.
    - Recover each bar's true color from its non-crossing cells, then use the crossing colors to infer a unique global layer order and verify which bar is above at every crossing.
    - Build an output of shape (#horizontal bars) x (#vertical bars), ordered by bar position (top-to-bottom, left-to-right), and fill each output row with that horizontal bar's color.

    Key insight:
    - The crossing pixels do not directly give the target matrix: some crossings show vertical colors that hide the row color, so the bars and their layer relations must be disentangled before the compact grid is formed.
    """
    g = np.asarray(grid, dtype=int)
    _bg, h_bars, v_bars = _detect_structure(g)
    _infer_unique_order(g, h_bars, v_bars)
    return _output_from_rows(h_bars, len(v_bars))


def _build_instance(rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray, list[tuple[str, int]]]:
    h_count = int(rng.integers(3, 5))
    v_count = int(rng.integers(3, 5))
    height = int(2 * h_count - 1 + rng.integers(0, 4))
    width = int(2 * v_count - 1 + rng.integers(0, 4))

    h_rows = _spaced_positions(height, h_count, rng)
    v_cols = _spaced_positions(width, v_count, rng)

    colors = rng.choice(np.array(PALETTE, dtype=int), size=h_count + v_count, replace=False).tolist()
    rng.shuffle(colors)
    h_colors = colors[:h_count]
    v_colors = colors[h_count:]

    h_bars = [{"node": (HORIZONTAL, i), "pos": h_rows[i], "color": int(h_colors[i])} for i in range(h_count)]
    v_bars = [{"node": (VERTICAL, j), "pos": v_cols[j], "color": int(v_colors[j])} for j in range(v_count)]

    h_order = [bar["node"] for bar in h_bars]
    v_order = [bar["node"] for bar in v_bars]
    rng.shuffle(h_order)
    rng.shuffle(v_order)

    orient_seq = _orientation_sequence(h_count, v_count, rng)
    order: list[tuple[str, int]] = []
    h_ptr = 0
    v_ptr = 0
    for orient in orient_seq:
        if orient == HORIZONTAL:
            order.append(h_order[h_ptr])
            h_ptr += 1
        else:
            order.append(v_order[v_ptr])
            v_ptr += 1

    bar_map = {bar["node"]: bar for bar in h_bars + v_bars}
    grid = np.full((height, width), BG, dtype=int)
    for node in order:
        bar = bar_map[node]
        if node[0] == HORIZONTAL:
            grid[int(bar["pos"]), :] = int(bar["color"])
        else:
            grid[:, int(bar["pos"])] = int(bar["color"])

    expected = _output_from_rows(h_bars, v_count)
    return grid, expected, order


def _validate_instance(grid: np.ndarray, expected: np.ndarray, order: list[tuple[str, int]]) -> bool:
    bg, h_bars, v_bars = _detect_structure(grid)
    if bg != BG:
        return False
    inferred = _infer_unique_order(grid, h_bars, v_bars)
    if inferred != order:
        return False
    if not np.array_equal(_output_from_rows(h_bars, len(v_bars)), expected):
        return False
    if not np.array_equal(main(grid), expected):
        return False

    crossings = np.array([[int(grid[h["pos"], v["pos"]]) for v in v_bars] for h in h_bars], dtype=int)
    h_colors = np.array([int(h["color"]) for h in h_bars], dtype=int)[:, None]
    v_colors = np.array([int(v["color"]) for v in v_bars], dtype=int)[None, :]
    if not np.any(crossings == h_colors):
        return False
    if not np.any(crossings == v_colors):
        return False
    if len({tuple(row.tolist()) for row in expected}) != expected.shape[0]:
        return False
    return True


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(RETRY_LIMIT):
        grid, expected, order = _build_instance(rng)
        if _validate_instance(grid, expected, order):
            return grid
    raise RuntimeError(f"failed to generate a valid input after {RETRY_LIMIT} attempts")
