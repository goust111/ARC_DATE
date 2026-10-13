import random
from collections import deque

import numpy as np


BG = 0


def _components(mask):
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            pts = []
            while q:
                cr, cc = q.popleft()
                pts.append((cr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            yield pts


def _hole_bbox(g, bbox, color):
    r0, c0, r1, c1 = bbox
    sub = g[r0 : r1 + 1, c0 : c1 + 1]
    holes = []
    for pts in _components(sub == BG):
        if any(r in (0, sub.shape[0] - 1) or c in (0, sub.shape[1] - 1) for r, c in pts):
            continue
        rows = [p[0] for p in pts]
        cols = [p[1] for p in pts]
        holes.append((min(rows), min(cols), max(rows), max(cols)))
    if not holes:
        return None
    return max(holes, key=lambda b: (b[2] - b[0] + 1) * (b[3] - b[1] + 1))


def main(grid: np.ndarray) -> np.ndarray:
    """
    Resize holed rectangles to their hole dimensions and center-stack them.

    Input structure:
    - Black background with separated single-color rectangular objects.
    - Each object is a filled rectangle with one black rectangular hole inside it.
    - Hole widths and heights encode target solid rectangle sizes.

    Rule:
    - Measure each object's internal black hole width and height.
    - Replace the object by a solid rectangle of that hole size and the same color.
    - Make the output just large enough for the widest and tallest target, then center-stack targets from largest area to smallest.

    Key insight:
    - The missing hole is a size label, not damage to preserve.
    """
    g = np.asarray(grid, dtype=int)
    items = []
    for color in [int(x) for x in np.unique(g) if int(x) != BG]:
        pts = np.argwhere(g == color)
        if pts.size == 0:
            continue
        r0, c0 = pts.min(axis=0)
        r1, c1 = pts.max(axis=0)
        hb = _hole_bbox(g, (int(r0), int(c0), int(r1), int(c1)), color)
        if hb is None:
            continue
        hr0, hc0, hr1, hc1 = hb
        target_h = int(hr1 - hr0 + 1)
        target_w = int(hc1 - hc0 + 1)
        items.append((target_h * target_w, target_h, target_w, color))
    if not items:
        return np.zeros((1, 1), dtype=int)
    out_h = max(item[1] for item in items)
    out_w = max(item[2] for item in items)
    out = np.zeros((out_h, out_w), dtype=int)
    for _area, th, tw, color in sorted(items, key=lambda x: -x[0]):
        r0 = (out_h - th) // 2
        c0 = (out_w - tw) // 2
        out[r0 : r0 + th, c0 : c0 + tw] = color
    return out


def generate_input() -> np.ndarray:
    rng = random.Random()
    colors = [1, 2, 3, 4, 6, 7, 8, 9]
    for _ in range(500):
        h = rng.randint(16, 30)
        w = rng.randint(16, 30)
        g = np.zeros((h, w), dtype=int)
        occupied = set()
        dims_used = set()
        ok = True
        for color in rng.sample(colors, rng.randint(2, 5)):
            placed = False
            for _try in range(200):
                hh = rng.randint(1, 3)
                hw = rng.randint(1, 4)
                if (hh, hw) in dims_used:
                    continue
                oh = rng.randint(hh + 2, hh + 5)
                ow = rng.randint(hw + 2, hw + 5)
                r0 = rng.randint(1, h - oh - 1)
                c0 = rng.randint(1, w - ow - 1)
                footprint = {(r, c) for r in range(r0 - 1, r0 + oh + 1) for c in range(c0 - 1, c0 + ow + 1)}
                if footprint & occupied:
                    continue
                occupied |= footprint
                dims_used.add((hh, hw))
                g[r0 : r0 + oh, c0 : c0 + ow] = color
                hr = rng.randint(r0 + 1, r0 + oh - hh - 1)
                hc = rng.randint(c0 + 1, c0 + ow - hw - 1)
                g[hr : hr + hh, hc : hc + hw] = BG
                placed = True
                break
            if not placed:
                ok = False
                break
        if ok:
            return g
    raise RuntimeError("generate_input failed")
