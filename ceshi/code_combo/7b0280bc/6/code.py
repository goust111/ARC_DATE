# Combo variant of 7b0280bc (index 6)
# combo_spec: D1/0+D9/0
# Name: 树中最长路径（直径）

from __future__ import annotations

# Design analysis
# ---------------
# `D9/0` changes only the topology: the graph becomes a tree.
# On a tree, between any fixed pair of terminals there is exactly one simple path, so the
# `D1/0` "longest simple path between the terminals" collapses to that unique path.
# To make the tree behavior visible, the generator starts from `D13/2`'s tree-like same-color
# layouts, rejects any sample that still contains a cycle, then recolors the unique farthest
# leaf pair as the two explicit terminals. After that, the existing `D1/0` solver is valid.

import random
import sys
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve()
PARENT = HERE.parent.parent
if str(PARENT) not in sys.path:
    sys.path.append(str(PARENT))

from arc_utils import get_bg
from _shared import load_variant


_D1 = load_variant("D1", 0)
_D13 = load_variant("D13", 2)
COLOR_POOL = [0, 1, 2, 4, 6, 7, 8, 9]


def _make_tree_terminal_colored(grid: np.ndarray, rng: random.Random) -> np.ndarray | None:
    g, id2pts, _track_comp_pts, adj = _D13._build_graph_from_grid(np.asarray(grid, dtype=int))
    node_ids = sorted(id2pts)
    edge_count = sum(len(v) for v in adj.values()) // 2
    if edge_count != len(node_ids) - 1:
        return None

    t0, t1 = _D13._infer_terminal_pair(adj)
    bg = int(get_bg(g))
    non_bg = sorted(int(c) for c in np.unique(g) if int(c) != bg)
    if len(non_bg) != 2:
        return None
    node_color = int(max(non_bg, key=lambda c: int((g == int(c)).sum())))
    track_color = int([c for c in non_bg if int(c) != node_color][0])

    available = [c for c in COLOR_POOL if c not in {bg, track_color}]
    rng.shuffle(available)
    terminal_color, station_color = available[:2]

    out = g.copy()
    for nid, pts in id2pts.items():
        fill = terminal_color if int(nid) in {int(t0), int(t1)} else station_color
        for r, c in pts:
            out[int(r), int(c)] = int(fill)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor the unique terminal-to-terminal path in a tree.

    Input structure:
    - Same visual grammar as the source task, but the underlying node-track graph is a tree.

    Rule:
    - The two terminal colors still define the endpoints.
    - Because the graph is a tree, there is exactly one simple path between them.
    - That unique path is also the longest simple terminal-to-terminal path, so recolor it.

    Key insight: once the topology is a tree, "shortest" and "longest simple" between the
    fixed terminal pair coincide because there is only one candidate path.
    """
    return np.asarray(_D1.main(np.asarray(grid, dtype=int)), dtype=int)


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(400):
        base = np.asarray(_D13.generate_input(), dtype=int)
        converted = _make_tree_terminal_colored(base, rng)
        if converted is None:
            continue
        try:
            _ = main(converted)
        except Exception:
            continue
        return converted
    raise RuntimeError("generate_input: failed to build a valid tree sample")
