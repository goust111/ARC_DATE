# Combo variant of 7b0280bc (index 8)
# combo_spec: D3/3+D6/3
# Name: 全路径元素类型化涂色

from __future__ import annotations

# Design analysis
# ---------------
# `D6/3` already turns the path into a typed node sequence by distinguishing source, goal,
# and intermediate stations in the output. `D3/3` then makes edge color depend on endpoint
# types instead of using a uniform path-track color.
# This combo therefore keeps the `D6/3` input grammar and shortest-path logic, and changes
# only the final rendering:
# - source node -> blue
# - goal node -> red
# - intermediate stations -> green
# - source/goal-to-station edges -> yellow
# - station-to-station edges -> green
# - direct source-to-goal edge (if ever present) -> grey

import sys
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve()
PARENT = HERE.parent.parent
if str(PARENT) not in sys.path:
    sys.path.append(str(PARENT))

from _shared import load_variant, parse_source_goal_shortest_path


_D6 = load_variant("D6", 3)

EDGE_COLOR_BY_PAIR = {
    frozenset({"source", "goal"}): int(_D6.TRACK_OUTPUT_COLOR),
    frozenset({"source", "station"}): int(_D6.Color.YELLOW),
    frozenset({"goal", "station"}): int(_D6.Color.YELLOW),
    frozenset({"station"}): int(_D6.Color.GREEN),
}


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor the source-goal shortest path with typed node colors and endpoint-type edge colors.

    Input structure:
    - Same as D6/3: one source square, one goal square, station squares, and track segments.

    Rule:
    - Use the unique shortest path from source to goal.
    - Paint path nodes by role: source blue, goal red, intermediate stations green.
    - Paint each path edge by its endpoint type pair:
      source-goal direct -> grey, station-station -> green, terminal-station -> yellow.

    Key insight: The output is no longer a flat "path highlight"; it is a fully typed path
    where both nodes and edges reveal their structural role.
    """
    parsed = parse_source_goal_shortest_path(np.asarray(grid, dtype=int))
    out = parsed["grid"].copy()

    for idx, nid in enumerate(parsed["node_path"]):
        kind = parsed["id2kind"][int(nid)]
        if kind == "source":
            fill = int(_D6.SOURCE_OUTPUT_COLOR)
        elif kind == "goal":
            fill = int(_D6.GOAL_OUTPUT_COLOR)
        else:
            fill = int(_D6.STATION_OUTPUT_COLOR)
        for r, c in parsed["id2pts"][int(nid)]:
            out[int(r), int(c)] = fill

    for eid in parsed["edge_path"]:
        a, b = parsed["edge_endpoints"][int(eid)]
        pair = frozenset({parsed["id2kind"][int(a)], parsed["id2kind"][int(b)]})
        edge_color = EDGE_COLOR_BY_PAIR[pair]
        for r, c in parsed["track_comp_pts"][int(eid)]:
            out[int(r), int(c)] = int(edge_color)

    return out


def generate_input() -> np.ndarray:
    return np.asarray(_D6.generate_input(), dtype=int)
