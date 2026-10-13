import random

import numpy as np


def main(grid: np.ndarray) -> np.ndarray:
    colors = sorted(int(c) for c in np.unique(grid) if c != 0)
    masks = []
    for color in colors:
        pts = np.argwhere(grid == color)
        r1, c1 = pts.min(axis=0)
        r2, c2 = pts.max(axis=0)
        coords = [(int(r - r1), int(c - c1)) for r, c in pts]
        masks.append((color, int(r2 - r1 + 1), int(c2 - c1 + 1), coords))
    if not masks:
        return np.zeros((1, 1), dtype=int)
    H = max(m[1] for m in masks)
    W = max(m[2] for m in masks)
    out = np.zeros((H, W), dtype=int)
    for color, mh, mw, coords in masks:
        roff = (H - mh) // 2
        coff = (W - mw) // 2
        for r, c in coords:
            out[roff + r, coff + c] = color
    rm = (H - 1) / 2
    cm = (W - 1) / 2
    filled = out.copy()
    for r in range(H):
        for c in range(W):
            if out[r, c] != 0:
                continue
            orbit = set()
            for rr in (r, int(round(2 * rm - r))):
                for cc in (c, int(round(2 * cm - c))):
                    if 0 <= rr < H and 0 <= cc < W:
                        orbit.add((rr, cc))
            vals = {int(out[rr, cc]) for rr, cc in orbit if out[rr, cc] != 0}
            if len(vals) == 1:
                filled[r, c] = vals.pop()
    return filled


def generate_input() -> np.ndarray:
    h = random.randint(16, 22)
    w = random.randint(16, 22)
    grid = np.zeros((h, w), dtype=int)
    H = random.choice([5, 6, 7])
    W = random.choice([5, 6, 7])
    colors = random.sample([1, 2, 3, 4, 6, 7, 8, 9], random.randint(3, 5))
    desired = {}
    center_r = (H - 1) / 2
    center_c = (W - 1) / 2
    candidates = [(r, c) for r in range(H) for c in range(W)
                  if abs(r - center_r) + abs(c - center_c) > 1]
    random.shuffle(candidates)
    for color in colors:
        while candidates:
            r, c = candidates.pop()
            orbit = {(r, c), (H - 1 - r, c), (r, W - 1 - c), (H - 1 - r, W - 1 - c)}
            if all(p not in desired for p in orbit):
                for p in orbit:
                    desired[p] = color
                break
    holes = set(random.sample(list(desired), min(random.randint(2, 4), len(desired))))
    per_color = {c: [] for c in colors}
    for pos, color in desired.items():
        if pos not in holes:
            per_color[color].append(pos)
    anchors = []
    for color, pts in per_color.items():
        if not pts:
            continue
        rs = [p[0] for p in pts]
        cs = [p[1] for p in pts]
        mh, mw = max(rs) - min(rs) + 1, max(cs) - min(cs) + 1
        for _ in range(200):
            ar = random.randint(0, h - mh)
            ac = random.randint(0, w - mw)
            cells = [(ar + r - min(rs), ac + c - min(cs)) for r, c in pts]
            if all(grid[r, c] == 0 for r, c in cells):
                for r, c in cells:
                    grid[r, c] = color
                anchors.append((ar, ac))
                break
    return grid
