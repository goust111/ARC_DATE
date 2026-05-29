# Combo variant of 7b0280bc (index 11)
# combo_spec: D5/0+D6/1
# Name: 距离编码+计数标记验证路径

from __future__ import annotations

# Design analysis
# ---------------
# This combo keeps the D5/0 input semantics: one terminal contains K grey pixels, and K
# means "propagate exactly K edges from the source terminal". The D6/1 change then alters
# only how the reached intermediate stations are rendered: instead of full-green squares,
# each reached station keeps its original fill and receives a green dot tally equal to its
# ordinal position along that K-edge prefix. The input grammar therefore comes entirely from
# D5/0; only the station output encoding changes.

import sys
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve()
PARENT = HERE.parent.parent
if str(PARENT) not in sys.path:
    sys.path.append(str(PARENT))

from _shared import load_variant, parse_d5_prefix_problem


_D5 = load_variant("D5", 0)
_D6 = load_variant("D6", 1)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Follow the D5 prefix-length rule, but encode reached stations with ordinal dot markers.

    Input structure:
    - Same as D5/0: a source terminal, an encoded terminal with K grey pixels, stations,
      and track segments forming a graph.

    Rule:
    - Recover the shortest source-to-encoded-terminal path.
    - Keep only the first K edges/stations of that path active, where K is the encoded
      grey-pixel count.
    - Recolor those K path edges to grey.
    - For each reached station, draw the D6/1 green dot marker for its ordinal position.

    Key insight: The terminal does not just identify a destination; it encodes how far the
    path should propagate, and the output then verifies that propagation with ordered dots.
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

    for eid in prefix_edges:
        for r, c in parsed["track_comp_pts"][int(eid)]:
            out[int(r), int(c)] = int(_D5.Color.GREY)

    return out


def generate_input() -> np.ndarray:
    return np.asarray(_D5.generate_input(), dtype=int)
