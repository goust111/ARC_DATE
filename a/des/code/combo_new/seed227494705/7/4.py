import random

import numpy as np


BG = 0


def _priority_tower(g: np.ndarray):
    h, w = g.shape
    best = None
    for c in range(w):
        rows = [r for r in range(h) if int(g[r, c]) != BG]
        if len(rows) < 2:
            continue
        rows.sort()
        runs = []
        start = prev = rows[0]
        for r in rows[1:]:
            if r == prev + 1:
                prev = r
            else:
                runs.append((start, prev))
                start = prev = r
        runs.append((start, prev))
        for r0, r1 in runs:
            colors = [int(g[r, c]) for r in range(r0, r1 + 1)]
            if len(colors) >= 2 and len(set(colors)) == len(colors):
                if best is None or c < best[0]:
                    best = (c, r0, r1, colors)
    if best is None:
        raise AssertionError("priority tower not found")
    return best


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recompose overlapping rectangles according to the visible priority tower.

    Input structure:
    - Black background with a vertical tower of distinct colors on the left.
    - The main area contains axis-aligned monochrome rectangles using those colors.
    - Rectangles may overlap in the input because they were drawn in an arbitrary order.

    Rule:
    - Read priority from the tower from top to bottom.
    - Infer each rectangle's full extent from the bounding box of its visible pixels in the main area.
    - Redraw all rectangles from lowest to highest priority so overlaps belong to the highest tower color.

    Key insight:
    - The input drawing order is a distractor; the tower gives the layer order for the final render.
    """
    g = np.asarray(grid, dtype=int)
    tower_c, tower_r0, tower_r1, priority = _priority_tower(g)
    out = np.zeros_like(g)
    out[tower_r0 : tower_r1 + 1, tower_c] = priority
    main_mask = np.ones(g.shape, dtype=bool)
    main_mask[:, : tower_c + 2] = False
    rects = {}
    for color in priority:
        pts = np.argwhere((g == color) & main_mask)
        if pts.size == 0:
            continue
        r0, c0 = pts.min(axis=0)
        r1, c1 = pts.max(axis=0)
        rects[color] = (int(r0), int(c0), int(r1), int(c1))
    for color in reversed(priority):
        if color not in rects:
            continue
        r0, c0, r1, c1 = rects[color]
        out[r0 : r1 + 1, c0 : c1 + 1] = color
    return out


def _bbox_visible_after(rect, later_rects):
    r0, c0, r1, c1 = rect
    mask = np.ones((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    for lr0, lc0, lr1, lc1 in later_rects:
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
        h = rng.randint(12, 24)
        w = rng.randint(16, 30)
        n = rng.randint(3, 5)
        priority = rng.sample(colors, n)
        g = np.zeros((h, w), dtype=int)
        tower_r0 = rng.randint(1, h - n - 1)
        tower_c = rng.choice([0, 1])
        for i, color in enumerate(priority):
            g[tower_r0 + i, tower_c] = color
        rects = []
        ok = True
        for _i in range(n):
            placed = False
            for _try in range(100):
                rh = rng.randint(3, 7)
                rw = rng.randint(3, 8)
                r0 = rng.randint(1, h - rh - 1)
                c0 = rng.randint(4, w - rw - 1)
                rect = (r0, c0, r0 + rh - 1, c0 + rw - 1)
                if _bbox_visible_after(rect, rects) is not None:
                    rects.append(rect)
                    placed = True
                    break
            if not placed:
                ok = False
                break
        if not ok:
            continue
        draw_order = list(range(n))
        rng.shuffle(draw_order)
        visible_ok = True
        for idx, color in zip(range(n), priority):
            later = [rects[j] for j in draw_order[draw_order.index(idx) + 1 :]]
            vis = _bbox_visible_after(rects[idx], later)
            if vis != rects[idx]:
                visible_ok = False
                break
        if not visible_ok:
            continue
        for idx in draw_order:
            r0, c0, r1, c1 = rects[idx]
            g[r0 : r1 + 1, c0 : c1 + 1] = priority[idx]
        return g
    raise RuntimeError("generate_input failed")
