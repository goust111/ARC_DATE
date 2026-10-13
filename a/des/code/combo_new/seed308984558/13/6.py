from __future__ import annotations
import random
import numpy as np
from arc_utils import find_components

BG, GREY = 0, 5


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fire a same-color ray from the matching grid edge through each grey rectangle until the opposite wall.

    Input structure:
    - Black background with several closed grey rectangles.
    - Each rectangle contains one colored marker pixel closer to one side than the others.

    Rule:
    - Determine which side the marker is closest to.
    - Start a ray of the marker color from the corresponding outer grid edge on the marker row or column.
    - Pass through the first grey wall unchanged, fill the background corridor inside the rectangle, and stop at the opposite wall.
    - Keep the marker and grey walls unchanged.

    Key insight:
    - Marker offset identifies the incident side, which in turn fixes both the ray origin and direction.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()
    for comp in find_components(g, bg=BG, colors=[GREY], connectivity=4, monochromatic=True):
        r0,c0,r1,c1 = map(int, comp['bbox'])
        inner = g[r0+1:r1, c0+1:c1]
        marks = np.argwhere((inner != BG) & (inner != GREY))
        if len(marks) != 1: continue
        sr,sc = map(int, marks[0]); color = int(inner[sr,sc]); sr += r0+1; sc += c0+1
        dists = {'left': sc-c0, 'right': c1-sc, 'up': sr-r0, 'down': r1-sr}
        side = min(dists, key=dists.get)
        if side == 'left': rr,cc,dr,dc = sr,0,0,1
        elif side == 'right': rr,cc,dr,dc = sr,g.shape[1]-1,0,-1
        elif side == 'up': rr,cc,dr,dc = 0,sc,1,0
        else: rr,cc,dr,dc = g.shape[0]-1,sc,-1,0
        entered = False
        while 0 <= rr < g.shape[0] and 0 <= cc < g.shape[1]:
            if g[rr,cc] == GREY:
                if entered: break
                entered = True
            elif entered and g[rr,cc] == BG:
                out[rr,cc] = color
            rr += dr; cc += dc
    return out


def generate_input() -> np.ndarray:
    g = np.zeros((22,22), dtype=int)
    for i,color in enumerate(random.sample([1,2,3,4,6,7,8,9], random.randint(1,3))):
        r0,c0 = 3+i*6, 4+i*4; h=w=5
        g[r0,c0:c0+w] = GREY; g[r0+h-1,c0:c0+w] = GREY; g[r0:r0+h,c0] = GREY; g[r0:r0+h,c0+w-1] = GREY
        side = random.choice(['left','right','up','down'])
        if side == 'left': g[r0+2,c0+1] = color
        elif side == 'right': g[r0+2,c0+w-2] = color
        elif side == 'up': g[r0+1,c0+2] = color
        else: g[r0+h-2,c0+2] = color
    return g
