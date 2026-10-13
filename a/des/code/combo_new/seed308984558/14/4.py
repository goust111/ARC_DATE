from __future__ import annotations
import random
import numpy as np
from arc_utils import find_components

BG, GREY = 0, 5


def main(grid: np.ndarray) -> np.ndarray:
    """
    Use right-panel fragments to repair same-color frame gaps on the left panel, then crop away the right panel.

    Input structure:
    - A vertical grey divider separates left frames from right fragments.
    - Left side has colored hollow rectangular frames with missing border segments.
    - Right side has same-color repair fragments.

    Rule:
    - For each left frame, find the same-color right fragment.
    - Find the missing border cells of that frame and paste the fragment shape into those missing cells.
    - Output only the repaired left panel, removing the divider and the right panel.

    Key insight:
    - Color pairs a fragment to a frame, while the fragment shape directly encodes the missing border segment.
    """
    g = np.asarray(grid, dtype=int)
    divider = int(np.where(np.all(g == GREY, axis=0))[0][0])
    out = g[:,:divider].copy()
    left = [c for c in find_components(g[:,:divider], bg=BG, connectivity=4, monochromatic=True) if int(c['color']) != BG]
    right = {int(c['color']): c for c in find_components(g[:,divider+1:], bg=BG, connectivity=4, monochromatic=True)}
    for frame in left:
        color = int(frame['color'])
        r0,c0,r1,c1 = map(int, frame['bbox'])
        miss=[]
        for c in range(c0,c1+1):
            if out[r0,c] == BG: miss.append((r0,c))
            if out[r1,c] == BG: miss.append((r1,c))
        for r in range(r0,r1+1):
            if out[r,c0] == BG: miss.append((r,c0))
            if out[r,c1] == BG: miss.append((r,c1))
        for r,c in miss: out[r,c] = color
    return out


def generate_input() -> np.ndarray:
    g = np.zeros((20,24), dtype=int)
    g[:,12] = GREY
    for i,color in enumerate(random.sample([1,2,3,4,6,7,8,9], random.randint(1,3))):
        r0,c0 = 2+i*6, 2
        g[r0,c0:c0+6] = color; g[r0+5,c0:c0+6] = color; g[r0:r0+6,c0] = color; g[r0:r0+6,c0+5] = color
        g[r0, c0+2:c0+4] = BG
        g[r0+1:r0+2, 16:18] = color; g[r0+1:r0+2, 18:20] = color
    return g
