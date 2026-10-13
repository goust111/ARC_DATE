from __future__ import annotations
import random
import numpy as np
from arc_utils import find_components

BG, GREY, ORANGE = 0, 5, 7


def _regions(g):
    rows = np.where(np.all(g == GREY, axis=1))[0]
    cols = np.where(np.all(g == GREY, axis=0))[0]
    rs = [-1] + rows.tolist() + [g.shape[0]]
    cs = [-1] + cols.tolist() + [g.shape[1]]
    regs=[]
    for i in range(len(rs)-1):
        for j in range(len(cs)-1):
            r0,r1 = rs[i]+1, rs[i+1]
            c0,c1 = cs[j]+1, cs[j+1]
            regs.append((r0,r1,c0,c1))
    return regs


def _marker(g, reg):
    r0,r1,c0,c1 = reg
    patch = g[r0:r1,c0:c1]
    vals = [int(v) for v in np.unique(patch) if v not in {BG,GREY,ORANGE}]
    return vals[0] if vals else None


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor an orange path only when its two endpoints lie in regions with the same corner marker color.

    Input structure:
    - Grey partition lines split the black grid into regions that each contain one colored corner marker.
    - Orange non-branching paths cross the partitions and have exactly two endpoints.

    Rule:
    - Locate each path and its endpoints.
    - Determine the partition region containing each endpoint.
    - If the two endpoint regions share the same marker color, recolor the entire path to that color.
    - Otherwise keep the path orange.

    Key insight:
    - Region membership of endpoints, not the path geometry itself, determines recoloring.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()
    regs = _regions(g)
    paths = find_components(g, bg=BG, colors=[ORANGE], connectivity=4, monochromatic=True)
    for path in paths:
        coords = [(int(r),int(c)) for r,c in path['coords']]
        deg1=[]
        cells=set(coords)
        for r,c in coords:
            deg = sum((r+dr,c+dc) in cells for dr,dc in ((1,0),(-1,0),(0,1),(0,-1)))
            if deg == 1: deg1.append((r,c))
        if len(deg1) != 2: continue
        colors=[]
        for r,c in deg1:
            for reg in regs:
                r0,r1,c0,c1 = reg
                if r0 <= r < r1 and c0 <= c < c1:
                    colors.append(_marker(g, reg)); break
        if len(colors)==2 and colors[0] == colors[1] and colors[0] is not None:
            for r,c in coords: out[r,c] = colors[0]
    return out


def generate_input() -> np.ndarray:
    size = random.choice([13,15,17])
    mid = size // 2
    g = np.zeros((size,size), dtype=int)
    g[mid,:] = GREY; g[:,mid] = GREY
    palette = random.sample([1,2,3,4,6,7,8,9], 4)
    if random.choice([True, False]):
        palette[2] = palette[0]
    corners = [(0,0),(0,size-1),(size-1,0),(size-1,size-1)]
    for color,(r,c) in zip(palette,corners, strict=False):
        g[r,c] = color
    col = random.randint(1, mid-1)
    row = random.randint(mid+1, size-2)
    g[1:row+1, col] = ORANGE
    g[row, col:mid] = ORANGE
    return g
