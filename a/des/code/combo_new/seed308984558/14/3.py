from __future__ import annotations
import random
import numpy as np
from arc_utils import find_components, crop_bbox

BG, LOCK = 0, 9


def _rotations(mask: np.ndarray):
    cur = mask
    for _ in range(4):
        yield cur
        cur = np.rot90(cur)


def _hole(g, bbox):
    r0,c0,r1,c1 = bbox
    box = g[r0:r1+1, c0:c1+1]
    coords = np.argwhere(box == BG)
    rs,cs = coords[:,0], coords[:,1]
    rr0,cc0,rr1,cc1 = rs.min(), cs.min(), rs.max(), cs.max()
    hole = np.zeros((rr1-rr0+1, cc1-cc0+1), dtype=bool)
    for r,c in coords: hole[r-rr0,c-cc0] = True
    return hole, (r0+rr0,c0+cc0)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Match each lock-hole shape to a rotated key shape and paint the hole with the key color.

    Input structure:
    - Black background with maroon lock rectangles containing one notch-like black hole.
    - Separate small monochrome keys appear elsewhere on the grid.

    Rule:
    - Extract each lock's hole shape and each key's foreground shape.
    - Match a key to a lock if some rotation of the key equals the hole shape.
    - Fill the lock hole with the matched key color.
    - Erase the original key locations while preserving maroon lock bodies.

    Key insight:
    - Keys are matched by exact rotated shape complement, not by color or proximity.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()
    locks=[]; keys=[]
    for comp in find_components(g, bg=BG, connectivity=4, monochromatic=True):
        if int(comp['color']) == LOCK: locks.append(comp)
        else: keys.append(comp)
    key_shapes=[]
    for key in keys:
        patch = crop_bbox(g, key['bbox']) != BG
        key_shapes.append((patch, int(key['color']), key))
        for r,c in key['coords']: out[int(r),int(c)] = BG
    for lock in locks:
        hole, origin = _hole(g, tuple(map(int,lock['bbox'])))
        for patch,color,_ in key_shapes:
            if any(rot.shape == hole.shape and np.array_equal(rot, hole) for rot in _rotations(patch)):
                r0,c0 = origin
                out[r0:r0+hole.shape[0], c0:c0+hole.shape[1]][hole] = color
                break
    return out


def generate_input() -> np.ndarray:
    g = np.zeros((22,22), dtype=int)
    shapes = [np.array([[1,1],[1,0]], dtype=bool), np.array([[1,1,1]], dtype=bool), np.array([[1],[1],[1]], dtype=bool)]
    for i,shape in enumerate(shapes):
        rr,cc = 2+i*6, 2
        color = random.choice([1,2,3,4,6,7,8])
        g[rr:rr+shape.shape[0], cc:cc+shape.shape[1]][shape] = color
        lr,lc = 2+i*6, 10
        g[lr:lr+5, lc:lc+6] = LOCK
        g[lr+2:lr+2+shape.shape[0], lc+2:lc+2+shape.shape[1]][shape] = BG
    return g
