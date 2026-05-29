# Combo variant of 7b0280bc (index 17)
# combo_spec: D12/1+D14/0
# Name: 自由布局蛇形轨道图

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
from _shared import parse_base_shortest_path


NODE_SIZE = 3
COLOR_POOL = [0, 1, 2, 4, 6, 7, 8, 9]


def _draw_node(grid: np.ndarray, r0: int, c0: int, color: int) -> None:
    grid[r0 : r0 + NODE_SIZE, c0 : c0 + NODE_SIZE] = int(color)


def _draw_segment(grid: np.ndarray, p0: tuple[int, int], p1: tuple[int, int], color: int) -> None:
    r0, c0 = map(int, p0)
    r1, c1 = map(int, p1)
    if r0 == r1:
        c_lo, c_hi = sorted((int(c0), int(c1)))
        grid[int(r0), c_lo : c_hi + 1] = int(color)
    else:
        r_lo, r_hi = sorted((int(r0), int(r1)))
        grid[r_lo : r_hi + 1, int(c0)] = int(color)


def _draw_polyline(grid: np.ndarray, points: list[tuple[int, int]], color: int) -> None:
    for a, b in zip(points, points[1:]):
        _draw_segment(grid, a, b, int(color))


def main(grid: np.ndarray) -> np.ndarray:
    """
    Solve the ordinary shortest-path task on a freely placed, snake-routed graph.
    """
    info = parse_base_shortest_path(np.asarray(grid, dtype=int), oriented_start=False)
    out = info["grid"].copy()
    for nid in info["node_path"][1:-1]:
        for r, c in info["id2pts"][int(nid)]:
            out[int(r), int(c)] = int(Color.GREEN)
    for eid in info["edge_path"]:
        for r, c in info["track_comp_pts"][int(eid)]:
            out[int(r), int(c)] = int(Color.GREY)
    return out


def _build_instance(rng: random.Random) -> np.ndarray:
    bg, terminal_color, station_color, track_color = rng.sample(COLOR_POOL, 4)
    g = np.full((25, 25), int(bg), dtype=int)

    nodes = {
        "start": (18, 2),
        "s1": (11, 8),
        "s2": (4, 15),
        "s3": (13, 18),
        "goal": (19, 11),
    }
    for name, (r0, c0) in nodes.items():
        color = terminal_color if name in {"start", "goal"} else station_color
        _draw_node(g, int(r0), int(c0), int(color))

    # Main unique shortest path: start -> s1 -> s2 -> s3 -> goal
    _draw_polyline(g, [(17, 3), (15, 3), (15, 6), (10, 6), (10, 9)], track_color)
    _draw_polyline(g, [(12, 11), (9, 11), (9, 13), (3, 13), (3, 16)], track_color)
    _draw_polyline(g, [(7, 16), (7, 17), (12, 17), (14, 17)], track_color)
    _draw_polyline(g, [(16, 19), (16, 15), (18, 15), (18, 12)], track_color)

    if rng.random() < 0.5:
        g = np.fliplr(g)
    if rng.random() < 0.5:
        g = np.flipud(g)
    return np.asarray(g, dtype=int)


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(200):
        g = _build_instance(rng)
        try:
            _ = main(g)
        except Exception:
            continue
        return g
    raise RuntimeError("generate_input: failed to build a free-layout snake sample")
