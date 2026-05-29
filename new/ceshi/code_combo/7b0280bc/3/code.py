# Combo variant of 7b0280bc (index 3)
# combo_spec: D3/0+D6/1
# Name: 渐变色序号双重编码路径

from __future__ import annotations

# Design analysis
# ---------------
# This combo is purely output-side. The graph, terminals, and shortest-path semantics remain
# the standard base task; only the chosen path is rendered with two synchronized order cues:
# - D3/0 colors path edges by distance from the start terminal.
# - D6/1 marks each intermediate station with a green dot tally equal to its ordinal position.
# To keep both signals legible, the generator simply reuses D6/1's input generator, which
# already fixes 3x3 stations and 3-edge shortest paths.

import sys
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve()
PARENT = HERE.parent.parent
if str(PARENT) not in sys.path:
    sys.path.append(str(PARENT))

from _shared import load_variant, parse_base_shortest_path


_D3 = load_variant("D3", 0)
_D6 = load_variant("D6", 1)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Encode the base shortest path with both edge-order gradient and station-order dot markers.

    Input structure:
    - Same square-node + track graph as the source task.

    Rule:
    - Use the ordinary unique shortest path between the two terminals.
    - Orient that path from the topmost/leftmost terminal, matching D3/0.
    - Color path edges grey/green/blue by edge order.
    - Mark intermediate stations with 1-dot, 2-dot, ... green tallies.

    Key insight: The same ordered path is visible twice: as a track color sequence and as
    local dot counts inside the traversed stations.
    """
    parsed = parse_base_shortest_path(np.asarray(grid, dtype=int), oriented_start=True)
    out = parsed["grid"].copy()

    ordinal = 1
    term_set = set(map(int, parsed["term_ids"]))
    for nid in parsed["node_path"]:
        if int(nid) in term_set:
            continue
        _D6._paint_station_marker(
            out,
            parsed["id2bbox"][int(nid)],
            parsed["id2size"][int(nid)],
            ordinal,
        )
        ordinal += 1

    for edge_idx, eid in enumerate(parsed["edge_path"]):
        edge_color = int(_D3._edge_color_by_distance(edge_idx))
        for r, c in parsed["track_comp_pts"][int(eid)]:
            out[int(r), int(c)] = edge_color

    return out


def generate_input() -> np.ndarray:
    return np.asarray(_D6.generate_input(), dtype=int)
