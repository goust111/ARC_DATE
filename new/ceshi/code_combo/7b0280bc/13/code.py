# Combo variant of 7b0280bc (index 13)
# combo_spec: D1/1+D12/2+D15/1
# Name: 边缘编码目标权重的加权寻路

from __future__ import annotations

import importlib.util
import random
from pathlib import Path

import numpy as np

from arc_utils import Color, get_bg


HERE = Path(__file__).resolve()
SIBLING_1 = HERE.parent.parent / "1" / "code.py"
SPEC_1 = importlib.util.spec_from_file_location("combo_7b0280bc_1", SIBLING_1)
assert SPEC_1 is not None and SPEC_1.loader is not None
_COMBO1 = importlib.util.module_from_spec(SPEC_1)
SPEC_1.loader.exec_module(_COMBO1)

INSTR_COLOR_POOL = list(range(10))


def _extract_target_strip(grid: np.ndarray, bg: int) -> tuple[np.ndarray, int]:
    g = np.asarray(grid, dtype=int)
    count = 0
    c = 0
    while c < g.shape[1] and int(g[0, c]) != int(bg):
        count += 1
        c += 1
    assert count >= 1
    g_graph = g.copy()
    g_graph[0, :count] = int(bg)
    return g_graph, int(count)


def _all_simple_paths(graph: dict) -> list[tuple[list[int], list[int], int]]:
    start, goal = map(int, graph["term_ids"])
    out: list[tuple[list[int], list[int], int]] = []

    def dfs(u: int, seen: set[int], node_path: list[int], edge_path: list[int], total: int) -> None:
        if int(u) == int(goal):
            out.append((list(node_path), list(edge_path), int(total)))
            return
        for v, eid, weight in graph["adj"].get(int(u), []):
            if int(v) in seen:
                continue
            seen.add(int(v))
            node_path.append(int(v))
            edge_path.append(int(eid))
            dfs(int(v), seen, node_path, edge_path, int(total + int(weight)))
            edge_path.pop()
            node_path.pop()
            seen.remove(int(v))

    dfs(int(start), {int(start)}, [int(start)], [], 0)
    assert out
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor the path whose total width-weight matches the border target.

    Rule:
    - The top-edge strip length W encodes the desired total path weight.
    - Edge weights are still visible track widths.
    - Choose an exact-W path if one exists; otherwise choose the closest total weight.
    """
    g = np.asarray(grid, dtype=int)
    bg = int(get_bg(g))
    g_graph, target_weight = _extract_target_strip(g, bg)
    graph = _COMBO1._extract_graph(g_graph)
    candidates = _all_simple_paths(graph)
    candidates.sort(key=lambda item: (abs(int(item[2]) - int(target_weight)), int(item[2]), len(item[1])))
    node_path, edge_path, _total = candidates[0]

    out = g.copy()
    for nid in node_path[1:-1]:
        for r, c in graph["id2pts"][int(nid)]:
            out[int(r), int(c)] = int(Color.GREEN)
    for eid in edge_path:
        for r, c in graph["track_comp_pts"][int(eid)]:
            out[int(r), int(c)] = int(Color.GREY)
    return out


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(200):
        base = np.asarray(_COMBO1.generate_input(), dtype=int)
        graph = _COMBO1._extract_graph(base)
        candidates = sorted({int(total) for _nodes, _edges, total in _all_simple_paths(graph)})
        if len(candidates) < 2:
            continue
        bg = int(get_bg(base))
        used = {bg} | {int(c) for c in np.unique(base) if int(c) != int(bg)}
        instr_choices = [c for c in INSTR_COLOR_POOL if int(c) not in used]
        if not instr_choices:
            continue
        target_weight = int(rng.choice(candidates))
        if target_weight >= base.shape[1]:
            continue
        out = base.copy()
        out[0, :target_weight] = int(instr_choices[0])
        try:
            _ = main(out)
        except Exception:
            continue
        return out
    raise RuntimeError("generate_input: failed to build a target-weight sample")
