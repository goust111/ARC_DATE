from __future__ import annotations
import random
import numpy as np
from arc_utils import find_components

BG = 0


def main(grid: np.ndarray) -> np.ndarray:
    """
    Keep the largest rectangle of each color in place and stack the smaller same-color rectangles onto its center with black borders.

    Input structure:
    - Black background with color clusters made of disjoint solid rectangles.
    - All rectangles in a cluster share one color and differ in area.

    Rule:
    - Group rectangles by color and keep the largest rectangle at its original location.
    - Sort the remaining rectangles by descending area and center each on the largest one.
    - When placing a stacked rectangle, draw only a one-pixel black border around its footprint while leaving its interior the cluster color.
    - Erase the original non-largest rectangles.

    Key insight:
    - Area ranking determines stack order; visible layering comes only from the black borders of nested placements.
    """
    g = np.asarray(grid, dtype=int)
    out = np.zeros_like(g)
    by={}
    for comp in find_components(g, bg=BG, connectivity=4, monochromatic=True): by.setdefault(int(comp['color']), []).append(comp)
    for color, comps in by.items():
        comps.sort(key=lambda c: int(c['area']), reverse=True)
        base = comps[0]
        r0,c0,r1,c1 = map(int, base['bbox'])
        out[r0:r1+1, c0:c1+1] = color
        cr = (r0+r1)//2; cc = (c0+c1)//2
        for comp in comps[1:]:
            h = comp['bbox'][2]-comp['bbox'][0]+1; w = comp['bbox'][3]-comp['bbox'][1]+1
            sr = cr - (h-1)//2; sc = cc - (w-1)//2
            out[sr:sr+h, sc:sc+w] = color
            out[sr,sc:sc+w] = BG; out[sr+h-1,sc:sc+w] = BG; out[sr:sr+h,sc] = BG; out[sr:sr+h,sc+w-1] = BG
    return out


def generate_input() -> np.ndarray:
    g = np.zeros((22,22), dtype=int)
    for color in random.sample([1,2,3,4,6,7,8,9], random.randint(1,3)):
        for _ in range(random.randint(3,4)):
            h = random.randint(2,5); w = random.randint(2,5)
            r = random.randint(1,16); c = random.randint(1,16)
            g[r:r+h, c:c+w] = color
    return g
