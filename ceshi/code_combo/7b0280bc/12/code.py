# Combo variant of 7b0280bc (index 12)
# combo_spec: D4/2+D5/2+D10/2
# Name: 空心方块内嵌标记颜色匹配

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve()
PARENT = HERE.parent.parent
if str(PARENT) not in sys.path:
    sys.path.append(str(PARENT))

from arc_utils import Color, get_bg
from _shared import load_variant


_D5 = load_variant("D5", 2)


def _frame_pixels(r0: int, c0: int, size: int = 3) -> list[tuple[int, int]]:
    pts: list[tuple[int, int]] = []
    r1 = int(r0 + size - 1)
    c1 = int(c0 + size - 1)
    for c in range(int(c0), int(c1) + 1):
        pts.append((int(r0), int(c)))
        pts.append((int(r1), int(c)))
    for r in range(int(r0) + 1, int(r1)):
        pts.append((int(r), int(c0)))
        pts.append((int(r), int(c1)))
    return pts


def _find_hollow_nodes(grid: np.ndarray, bg: int) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    used = np.zeros_like(g, dtype=bool)
    nodes: list[dict] = []
    for r0 in range(h - 2):
        for c0 in range(w - 2):
            if bool(used[r0 : r0 + 3, c0 : c0 + 3].any()):
                continue
            patch = g[r0 : r0 + 3, c0 : c0 + 3]
            border = [
                int(patch[0, 0]), int(patch[0, 1]), int(patch[0, 2]),
                int(patch[1, 0]), int(patch[1, 2]),
                int(patch[2, 0]), int(patch[2, 1]), int(patch[2, 2]),
            ]
            if any(int(v) == int(bg) for v in border):
                continue
            if len(set(border)) != 1:
                continue
            body = int(border[0])
            center = int(patch[1, 1])
            if int(center) == int(body):
                continue
            nodes.append(
                {
                    "bbox": (int(r0), int(c0), int(r0 + 2), int(c0 + 2)),
                    "body_color": int(body),
                    "center_color": int(center),
                }
            )
            used[r0 : r0 + 3, c0 : c0 + 3] = True
    return nodes


def _solidify_hollow_input(grid: np.ndarray, nodes: list[dict], bg: int) -> tuple[np.ndarray, np.ndarray]:
    g = np.asarray(grid, dtype=int)
    solid = g.copy()
    node_mask = np.zeros_like(g, dtype=bool)
    for nd in nodes:
        r0, c0, r1, c1 = map(int, nd["bbox"])
        solid[r0 : r1 + 1, c0 : c1 + 1] = int(nd["body_color"])
        center = int(nd["center_color"])
        if int(center) != int(bg):
            solid[r0 + 1, c0 + 1] = int(center)
        node_mask[r0 : r1 + 1, c0 : c1 + 1] = True
    return solid, node_mask


def main(grid: np.ndarray) -> np.ndarray:
    """
    Solve D5/2's marker-matching path rule on hollow 3x3 node frames.

    Rule:
    - Hollow frames are nodes.
    - The center pixel marker logic is unchanged from D5/2.
    - Output preserves hollowness: only frame borders recolor on path stations.
    """
    g = np.asarray(grid, dtype=int)
    bg = int(get_bg(g))
    nodes = _find_hollow_nodes(g, bg)
    assert len(nodes) >= 4

    solid, node_mask = _solidify_hollow_input(g, nodes, bg)
    solved = np.asarray(_D5.main(solid), dtype=int)

    out = g.copy()
    for nd in nodes:
        r0, c0, _r1, _c1 = map(int, nd["bbox"])
        patch = solved[r0 : r0 + 3, c0 : c0 + 3]
        if np.any(patch == int(Color.GREEN)):
            for r, c in _frame_pixels(r0, c0, 3):
                out[int(r), int(c)] = int(Color.GREEN)
    edge_mask = (~node_mask) & (solved == int(Color.GREY))
    out[edge_mask] = int(Color.GREY)
    return out


def generate_input() -> np.ndarray:
    for _ in range(300):
        base = np.asarray(_D5.generate_input(), dtype=int)
        info = _D5._build_instance_graph(base)
        out = base.copy()
        bg = int(info["bg"])
        for nd in info["nodes"]:
            r0, c0, r1, c1 = map(int, nd["bbox"])
            out[r0 : r1 + 1, c0 : c1 + 1] = int(bg)
            for r, c in _frame_pixels(r0, c0, int(info["N"])):
                out[int(r), int(c)] = int(nd["body_color"])
            marker = nd["marker_color"]
            if marker is not None:
                out[r0 + 1, c0 + 1] = int(marker)
        try:
            _ = main(out)
        except Exception:
            continue
        return out
    raise RuntimeError("generate_input: failed to build a hollow marker-matching sample")
