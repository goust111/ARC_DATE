from pathlib import Path
import importlib.util
import random

import numpy as np


def _load_common():
    path = Path(__file__).with_name("_common.py")
    spec = importlib.util.spec_from_file_location("_arc051_common", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


U = _load_common()


def _segments(g, colors):
    inner = np.zeros_like(g, dtype=bool)
    inner[1:-1, 1:-1] = True
    segs = []
    for col in colors:
        for comp in U.components((g == col) & inner):
            r0, c0, r1, c1 = U.bbox(comp)
            length = len(comp)
            orient = "h" if r0 == r1 else "v"
            segs.append({"color": col, "cells": comp, "bbox": (r0, c0, r1, c1),
                         "length": length, "orient": orient})
    return segs


def main(grid: np.ndarray) -> np.ndarray:
    """Stack interior segments toward the border side with the same color.

    Input structure:
    A black interior is surrounded by four colored border sides, with black
    corners. Interior one-cell line segments use those border colors.

    Rule:
    The color of a segment selects its gravity side. Within each side group,
    sort segments by length, breaking ties by initial nearness to that side.
    Clear the original segments and place each group in stacked rows or columns
    adjacent to its matching border, using the fixed placement order top, right,
    bottom, left so later groups win crossings.

    Key insight:
    Border color is an instruction channel: it groups objects and assigns a
    direction before the ordinary length sort is applied.
    """
    g = np.array(grid, copy=True)
    out = g.copy()
    h, w = g.shape
    side_of = {
        int(g[0, 1]): "top",
        int(g[1, w - 1]): "right",
        int(g[h - 1, 1]): "bottom",
        int(g[1, 0]): "left",
    }
    colors = list(side_of)
    segs = _segments(g, colors)
    for s in segs:
        for r, c in s["cells"]:
            out[r, c] = 0
    groups = {side: [] for side in ("top", "right", "bottom", "left")}
    for s in segs:
        side = side_of[s["color"]]
        r0, c0, r1, c1 = s["bbox"]
        dist = {"top": r0 - 1, "bottom": h - 2 - r1, "left": c0 - 1, "right": w - 2 - c1}[side]
        groups[side].append((s["length"], dist, s))
    for side in ("top", "right", "bottom", "left"):
        for i, (_, _, s) in enumerate(sorted(groups[side], key=lambda x: (x[0], x[1]))):
            length = s["length"]
            col = s["color"]
            if side == "top":
                r = 1 + i
                c0 = 1 + ((w - 2) - length) // 2
                out[r, c0:c0 + length] = col
            elif side == "bottom":
                r = h - 2 - i
                c0 = 1 + ((w - 2) - length) // 2
                out[r, c0:c0 + length] = col
            elif side == "left":
                c = 1 + i
                r0 = 1 + ((h - 2) - length) // 2
                out[r0:r0 + length, c] = col
            else:
                c = w - 2 - i
                r0 = 1 + ((h - 2) - length) // 2
                out[r0:r0 + length, c] = col
    return out


def generate_input() -> np.ndarray:
    h = w = random.randint(12, 15)
    g = np.zeros((h, w), dtype=int)
    cols = [1, 2, 4, 7]
    random.shuffle(cols)
    top, right, bottom, left = cols
    g[0, 1:-1] = top
    g[1:-1, w - 1] = right
    g[h - 1, 1:-1] = bottom
    g[1:-1, 0] = left
    counts = {top: 1, right: 1, bottom: 1, left: 1}
    for col in random.sample(cols, random.randint(2, 4)):
        counts[col] += 1
    occupied = np.zeros_like(g, dtype=bool)
    occupied[[0, h - 1], :] = True
    occupied[:, [0, w - 1]] = True
    for col, n in counts.items():
        side = {top: "top", bottom: "bottom", left: "left", right: "right"}[col]
        for _ in range(n):
            length = random.randint(2, 5)
            for _try in range(200):
                if side in ("top", "bottom"):
                    r = random.randint(2, h - 3)
                    c = random.randint(2, w - length - 2)
                    cells = [(r, x) for x in range(c, c + length)]
                else:
                    r = random.randint(2, h - length - 2)
                    c = random.randint(2, w - 3)
                    cells = [(x, c) for x in range(r, r + length)]
                if all(not occupied[x, y] for x, y in cells):
                    for x, y in cells:
                        g[x, y] = col
                        occupied[x, y] = True
                    break
    return g
