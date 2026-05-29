# Combo variant of 7b0280bc (index 10)
# combo_spec: D3/0+D15/0
# Name: 图例驱动渐变色路径

from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve()
PARENT = HERE.parent.parent
if str(PARENT) not in sys.path:
    sys.path.append(str(PARENT))

from arc_utils import Color, get_bg
from _shared import load_variant, parse_base_shortest_path


_D15 = load_variant("D15", 0)
LEGEND_COLOR_POOL = [0, 1, 2, 4, 5, 6, 7, 8, 9]


def _extract_legend_strip(grid: np.ndarray, bg: int) -> tuple[np.ndarray, list[int]]:
    g = np.asarray(grid, dtype=int)
    colors: list[int] = []
    c = 0
    while c < g.shape[1] and int(g[0, c]) != int(bg):
        colors.append(int(g[0, c]))
        c += 1
    assert len(colors) >= 2
    g_graph = g.copy()
    g_graph[0, : len(colors)] = int(bg)
    return g_graph, colors


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor the shortest path using the corner legend's ordered edge palette.

    Input structure:
    - The graph stays the base shortest-path graph.
    - A horizontal legend strip is placed at the top-left corner on row 0.
    - Legend cell i gives the color for path edge i+1 from the visible start terminal.

    Rule:
    - Read the legend sequence.
    - Find the shortest terminal-to-terminal path, oriented from the top-left terminal.
    - Recolor path stations green and path edges by legend order.
    """
    g = np.asarray(grid, dtype=int)
    bg = int(get_bg(g))
    g_graph, legend_colors = _extract_legend_strip(g, bg)
    info = parse_base_shortest_path(g_graph, oriented_start=True)
    assert len(info["edge_path"]) <= len(legend_colors)

    out = g.copy()
    for nid in info["node_path"][1:-1]:
        for r, c in info["id2pts"][int(nid)]:
            out[int(r), int(c)] = int(Color.GREEN)
    for i, eid in enumerate(info["edge_path"]):
        edge_color = int(legend_colors[int(i)])
        for r, c in info["track_comp_pts"][int(eid)]:
            out[int(r), int(c)] = edge_color
    return out


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(400):
        base = np.asarray(_D15.generate_input(), dtype=int)
        bg = int(get_bg(base))
        g_graph, _track_paint, _node_paint, _orientation, _legend_cells = _D15._extract_legend(base, bg)
        info = parse_base_shortest_path(g_graph, oriented_start=True)

        extra = int(rng.choice([0, 1]))
        legend_len = max(3, len(info["edge_path"]) + extra)
        used = {bg, int(Color.GREEN)} | {int(c) for c in np.unique(g_graph) if int(c) != int(bg)}
        available = [c for c in LEGEND_COLOR_POOL if int(c) not in used]
        if len(available) < legend_len:
            continue
        rng.shuffle(available)
        legend_colors = [int(c) for c in available[:legend_len]]

        out = np.asarray(g_graph, dtype=int).copy()
        out[0, :legend_len] = np.array(legend_colors, dtype=int)
        try:
            _ = main(out)
        except Exception:
            continue
        return out
    raise RuntimeError("generate_input: failed to build a valid legend-gradient sample")
