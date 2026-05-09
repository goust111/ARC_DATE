# Combo variant of 7b0280bc (index 18)
# combo_spec: D1/2+D8/1
# Name: 删除后避障寻路

from __future__ import annotations

# Design analysis
# ---------------
# `D8/1` already adds a preprocessing stage: delete the single longest track component, then
# solve the shortest path on the reduced graph. `D1/2` adds blocked rails: a second edge color
# marks components that should remain visible but be ignored by BFS.
# The combo therefore proceeds in a strict sequence:
#   1. identify and erase the longest visible rail component;
#   2. among the remaining rails, treat one edge color as blocked;
#   3. run shortest-path search only on the passable rail color.
# Generator-wise, the easiest stable construction is to start from a valid `D8/1` sample and
# recolor one non-critical residual edge to the blocked color. That preserves the D8 route while
# making the extra D1 filtering visibly active.

import random
import sys
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve()
PARENT = HERE.parent.parent
ROOT = Path("/home/lixinyu/snap/new")
for p in (PARENT, ROOT):
    if str(p) not in sys.path:
        sys.path.append(str(p))

from arc_utils import colors_present, components_mask, get_bg
from _shared import load_variant


_D1 = load_variant("D1", 2)
_D8 = load_variant("D8", 1)


def _erase_longest_nonnode_component(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int).copy()
    bg = int(get_bg(g))

    square_cands = _D1.find_solid_squares(g, bg=bg, min_size=2, connectivity=8)
    assert square_cands
    size_counts = _D1.Counter(int(d["size"]) for d in square_cands)
    n_size = int(max(size_counts.items(), key=lambda kv: (int(kv[1]), int(kv[0])))[0])
    nodes_raw = [d for d in square_cands if int(d["size"]) == int(n_size)]

    node_colors = sorted({int(d["color"]) for d in nodes_raw})
    non_node_colors = [c for c in colors_present(g, exclude={bg}) if int(c) not in node_colors]
    assert non_node_colors

    best_pts: list[tuple[int, int]] | None = None
    best_len = -1
    for col in non_node_colors:
        for coords in components_mask(g == int(col), connectivity=8):
            pts = [(int(r), int(c)) for r, c in coords]
            if len(pts) > best_len:
                best_len = len(pts)
                best_pts = pts
    assert best_pts is not None
    for r, c in best_pts:
        g[int(r), int(c)] = int(bg)
    return g


def main(grid: np.ndarray) -> np.ndarray:
    """
    Delete the longest rail component first, then solve the blocked-edge shortest-path task.

    Input structure:
    - Same node/track graph vocabulary as the source task.
    - Two non-node rail colors may appear: the more prevalent one is passable, the other is blocked.

    Rule:
    - Find the single longest visible rail component and erase it to background.
    - On the reduced graph, ignore the blocked-color rail components.
    - Recolor the shortest remaining passable path.

    Key insight: The graph is filtered twice in order: first by physical deletion, then by
    logical impassability.
    """
    reduced = _erase_longest_nonnode_component(np.asarray(grid, dtype=int))
    return _D1.main(reduced)


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(300):
        base = np.asarray(_D8.generate_input(), dtype=int)
        parsed = _D8._parse_graph(base)
        longest_id = int(max(parsed["edge_infos"], key=lambda e: int(e["length"]))["id"])

        # Recover the post-deletion chosen path from the D8 output so we can avoid blocking it.
        d8_out = _D8.main(base)
        grey = int(_D8.Color.GREY)
        protected = {
            int(info["id"])
            for info in parsed["edge_infos"]
            if all(int(d8_out[int(r), int(c)]) == grey for r, c in info["pts"])
        }

        candidate_edges = [
            info
            for info in parsed["edge_infos"]
            if int(info["id"]) != longest_id and int(info["id"]) not in protected
        ]
        if not candidate_edges:
            continue

        used = set(map(int, np.unique(base)))
        bg = int(get_bg(base))
        blocked_candidates = [c for c in range(10) if c not in used and c != bg]
        if not blocked_candidates:
            continue
        blocked_color = int(rng.choice(blocked_candidates))

        edge_to_block = min(candidate_edges, key=lambda info: int(info["length"]))
        trial = base.copy()
        for r, c in edge_to_block["pts"]:
            trial[int(r), int(c)] = blocked_color

        try:
            _ = main(trial)
        except Exception:
            continue
        return trial
    raise RuntimeError("generate_input: failed to build a valid delete-then-block sample")
