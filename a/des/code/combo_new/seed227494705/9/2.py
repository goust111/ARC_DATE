import random

import numpy as np


BG = 0


def _marker_order(g):
    colors = []
    for r in range(g.shape[0]):
        color = int(g[r, 0])
        if color != BG:
            colors.append(color)
    return colors


def main(grid: np.ndarray) -> np.ndarray:
    """
    Rebuild occluded rectangles using the edge marker stack as layer order.

    Input structure:
    - Black background with visible fragments of several colored rectangles.
    - A left-edge vertical marker stack lists the rectangle colors from bottom layer to top layer.
    - Each rectangle has enough visible pixels to reveal its full bounding box.

    Rule:
    - Read the left-edge markers from top to bottom; this is the draw order from bottom to top.
    - Remove marker pixels, infer every listed color's rectangle from its visible bounding box, and redraw full rectangles.
    - Later listed colors overwrite earlier colors.

    Key insight:
    - The marker order is inverted relative to visual depth: the top marker is the bottom rectangle.
    """
    g = np.asarray(grid, dtype=int)
    order = _marker_order(g)
    out = np.zeros_like(g)
    area_mask = np.ones(g.shape, dtype=bool)
    area_mask[:, 0] = False
    rects = {}
    for color in order:
        pts = np.argwhere((g == color) & area_mask)
        if pts.size == 0:
            continue
        r0, c0 = pts.min(axis=0)
        r1, c1 = pts.max(axis=0)
        rects[color] = (int(r0), int(c0), int(r1), int(c1))
    for color in order:
        if color in rects:
            r0, c0, r1, c1 = rects[color]
            out[r0 : r1 + 1, c0 : c1 + 1] = color
    return out


def _visible_bbox(rect, later):
    r0, c0, r1, c1 = rect
    mask = np.ones((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    for lr0, lc0, lr1, lc1 in later:
        ir0, ic0 = max(r0, lr0), max(c0, lc0)
        ir1, ic1 = min(r1, lr1), min(c1, lc1)
        if ir0 <= ir1 and ic0 <= ic1:
            mask[ir0 - r0 : ir1 - r0 + 1, ic0 - c0 : ic1 - c0 + 1] = False
    if not mask.any():
        return None
    pts = np.argwhere(mask)
    br0, bc0 = pts.min(axis=0)
    br1, bc1 = pts.max(axis=0)
    return (r0 + int(br0), c0 + int(bc0), r0 + int(br1), c0 + int(bc1))


def generate_input() -> np.ndarray:
    rng = random.Random()
    colors = [1, 2, 3, 4, 6, 7, 8, 9]
    for _ in range(500):
        h = rng.randint(14, 26)
        w = rng.randint(16, 30)
        n = rng.randint(2, 5)
        order = rng.sample(colors, n)
        rects = []
        ok = True
        for i in range(n):
            placed = False
            for _try in range(100):
                rh = rng.randint(3, 7)
                rw = rng.randint(3, 8)
                r0 = rng.randint(1, h - rh - 1)
                c0 = rng.randint(3, w - rw - 1)
                rect = (r0, c0, r0 + rh - 1, c0 + rw - 1)
                if _visible_bbox(rect, rects[i + 1 :]) is not None:
                    rects.append(rect)
                    placed = True
                    break
            if not placed:
                ok = False
                break
        if not ok:
            continue
        for idx, rect in enumerate(rects):
            if _visible_bbox(rect, rects[idx + 1 :]) != rect:
                ok = False
                break
        if not ok:
            continue
        g = np.zeros((h, w), dtype=int)
        for i, color in enumerate(order):
            g[1 + i, 0] = color
        for color, (r0, c0, r1, c1) in zip(order, rects):
            g[r0 : r1 + 1, c0 : c1 + 1] = color
        return g
    raise RuntimeError("generate_input failed")
