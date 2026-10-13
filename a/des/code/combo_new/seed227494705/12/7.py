import random
from collections import Counter
import numpy as np


def _components(mask):
    h, w = mask.shape
    seen = np.zeros((h, w), bool)
    comps = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = [(r, c)]
            seen[r, c] = True
            pts = []
            while q:
                x, y = q.pop()
                pts.append((x, y))
                for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < h and 0 <= ny < w and mask[nx, ny] and not seen[nx, ny]:
                        seen[nx, ny] = True
                        q.append((nx, ny))
            comps.append(pts)
    return comps


def _bbox(pts):
    rs = [p[0] for p in pts]
    cs = [p[1] for p in pts]
    return min(rs), max(rs), min(cs), max(cs)


def main(grid: np.ndarray) -> np.ndarray:
    """Merge same-shape colored objects into a maroon target frame using hole counts as priority.

    Input structure: azure background with several same-bbox-shape colored
    objects, grey hole pixels within those objects, and a maroon rectangular
    target frame whose interior matches the object bbox size.
    Rule: crop the target frame, align every object to its interior, and for
    each relative pixel choose the non-grey object color with the fewest grey
    holes, breaking ties by the source object's topmost position.
    Key insight: grey holes both reveal the common underlying object shape and
    rank the competing colors by completeness.
    """
    bg, grey, maroon = 8, 5, 9
    frame = max(_components(grid == maroon), key=len)
    fr1, fr2, fc1, fc2 = _bbox(frame)
    out = grid[fr1:fr2 + 1, fc1:fc2 + 1].copy()
    out[1:-1, 1:-1] = bg
    mask = (grid != bg) & (grid != maroon)
    mask[fr1:fr2 + 1, fc1:fc2 + 1] = False
    objects = []
    for comp in _components(mask):
        vals = [int(grid[p]) for p in comp if grid[p] != grey]
        if not vals:
            continue
        color = Counter(vals).most_common(1)[0][0]
        r1, r2, c1, c2 = _bbox(comp)
        holes = sum(grid[p] == grey for p in comp)
        cells = {(r - r1, c - c1): int(grid[r, c]) for r, c in comp}
        objects.append({"color": color, "holes": holes, "top": r1, "cells": cells})
    ih, iw = fr2 - fr1 - 1, fc2 - fc1 - 1
    for r in range(ih):
        for c in range(iw):
            choices = []
            for obj in objects:
                val = obj["cells"].get((r, c))
                if val is not None and val != grey:
                    choices.append((obj["holes"], obj["top"], obj["color"]))
            if choices:
                out[r + 1, c + 1] = min(choices)[2]
    return out


def generate_input() -> np.ndarray:
    bg, grey, maroon = 8, 5, 9
    grid = np.full((24, 24), bg, dtype=int)
    shape = [(0, 1), (0, 2), (1, 0), (1, 1), (1, 2), (2, 1), (3, 1), (3, 2)]
    ih, iw = 4, 3
    fr1, fc1 = 1, 1
    fr2, fc2 = fr1 + ih + 1, fc1 + iw + 1
    grid[fr1:fr2 + 1, fc1] = maroon
    grid[fr1:fr2 + 1, fc2] = maroon
    grid[fr1, fc1:fc2 + 1] = maroon
    grid[fr2, fc1:fc2 + 1] = maroon
    colors = random.sample([1, 2, 3, 4, 6, 7], random.randint(3, 5))
    origins = [(2, 12), (8, 15), (14, 4), (16, 14), (9, 7)]
    random.shuffle(origins)
    for i, color in enumerate(colors):
        r, c = origins[i]
        holes = set(random.sample(shape, random.randint(0, 3)))
        for a, b in shape:
            grid[r + a, c + b] = grey if (a, b) in holes else color
    return grid
