# Combo variant of 7b0280bc (index 20)
# combo_spec: D3/0+D5/0+D6/1
# Name: 定长路径渐变色+计数双重编码

from __future__ import annotations

# Design analysis
# ---------------
# This combo uses the same D5/0 prefix-length input encoding as combo 11, but it applies
# both output-side order encodings at once:
# - D3/0: the traversed prefix edges are recolored by ordered gradient.
# - D6/1: the reached prefix stations are marked by ordered dot tallies.
# Because D5/0 restricts K to {1,2,3}, the D3 gradient palette is sufficient and the two
# encodings agree on the same left-to-right / source-to-destination path order.

import sys
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve()
PARENT = HERE.parent.parent
if str(PARENT) not in sys.path:
    sys.path.append(str(PARENT))

from _shared import load_variant, parse_d5_prefix_problem


_D3 = load_variant("D3", 0)
_D5 = load_variant("D5", 0)
_D6 = load_variant("D6", 1)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Follow the D5 prefix-length rule, then encode the chosen prefix with both edge and node order.

    Input structure:
    - Same as D5/0: one source terminal, one encoded terminal, station squares, and graph tracks.

    Rule:
    - Let K be the number of grey pixels inside the encoded terminal.
    - Take the first K edges of the shortest source-to-encoded-terminal path.
    - Recolor edge 1/2/3 of that prefix with the D3 gradient colors.
    - Mark reached stations with D6-style ordinal green dots.

    Key insight: The output redundantly encodes the same ordered prefix in two visible ways:
    track color sequence and station dot counts.
    """
    parsed = parse_d5_prefix_problem(np.asarray(grid, dtype=int))
    out = parsed["grid"].copy()

    stop_len = int(parsed["stop_len"])
    prefix_nodes = parsed["node_path"][1 : stop_len + 1]
    prefix_edges = parsed["edge_path"][:stop_len]

    ordinal = 1
    for nid in prefix_nodes:
        if parsed["id2kind"][int(nid)] != "station":
            continue
        _D6._paint_station_marker(
            out,
            parsed["id2bbox"][int(nid)],
            parsed["id2size"][int(nid)],
            ordinal,
        )
        ordinal += 1

    for edge_idx, eid in enumerate(prefix_edges):
        edge_color = int(_D3._edge_color_by_distance(edge_idx))
        for r, c in parsed["track_comp_pts"][int(eid)]:
            out[int(r), int(c)] = edge_color

    return out


def generate_input() -> np.ndarray:
    return np.asarray(_D5.generate_input(), dtype=int)
