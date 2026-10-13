from __future__ import annotations
import random
import numpy as np
from arc_utils import find_components, crop_bbox

BG = 8


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reflect each lower same-color shadow across its upper partner by reversing the shadow offset vector.

    Input structure:
    - Azure background with same-color component pairs of identical shape.
    - For each pair, the upper component is the object and the lower one is its shadow.

    Rule:
    - Group components by color and identify the upper component as the object.
    - Compute the top-left offset from object bbox to shadow bbox.
    - Erase the old shadow and redraw it at the opposite offset from the object.
    - Keep the object unchanged.

    Key insight:
    - The shadow moves by vector negation relative to the fixed upper object.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()
    by={}
    for comp in find_components(g, bg=BG, connectivity=4, monochromatic=True): by.setdefault(int(comp['color']), []).append(comp)
    for color,pair in by.items():
        if len(pair) < 2:
            continue
        pair.sort(key=lambda c: (c['bbox'][0], c['bbox'][1]))
        pair = pair[:2]
        obj, shadow = pair
        for r,c in shadow['coords']: out[int(r),int(c)] = BG
        dr = shadow['bbox'][0]-obj['bbox'][0]; dc = shadow['bbox'][1]-obj['bbox'][1]
        patch = crop_bbox(g, obj['bbox'])
        r0 = obj['bbox'][0]-dr; c0 = obj['bbox'][1]-dc
        rr0 = max(0, r0); cc0 = max(0, c0)
        rr1 = min(out.shape[0], r0 + patch.shape[0]); cc1 = min(out.shape[1], c0 + patch.shape[1])
        pr0 = rr0 - r0; pc0 = cc0 - c0
        sub = patch[pr0 : pr0 + (rr1 - rr0), pc0 : pc0 + (cc1 - cc0)]
        out[rr0:rr1, cc0:cc1][sub != BG] = color
    return out


def generate_input() -> np.ndarray:
    g = np.full((20,20), BG, dtype=int)
    shape = np.array([[1,1,0],[1,1,1]], dtype=bool)
    used = []
    for color in random.sample([1,2,3,4,6,7,9], random.randint(1,3)):
        r,c = random.randint(1,5), random.randint(4,9)
        dr,dc = random.randint(3,5), random.randint(-2,2)
        box1 = (r, c, r+1, c+2)
        box2 = (r+dr, c+dc, r+dr+1, c+dc+2)
        if box2[0] < 0 or box2[1] < 0 or box2[2] >= 20 or box2[3] >= 20:
            continue
        if any(not (box2[2] < a0-1 or box2[0] > a2+1 or box2[3] < a1-1 or box2[1] > a3+1 or box1[2] < a0-1 or box1[0] > a2+1 or box1[3] < a1-1 or box1[1] > a3+1) for a0,a1,a2,a3 in used):
            continue
        used.extend([box1, box2])
        g[r:r+2,c:c+3][shape] = color
        g[r+dr:r+dr+2, c+dc:c+dc+3][shape] = color
    return g
