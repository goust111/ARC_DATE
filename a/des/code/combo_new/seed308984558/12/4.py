from __future__ import annotations
import random
import numpy as np
from arc_utils import find_components

BG = 8


def _norm(coords):
    rs = [int(r) for r,c in coords]; cs = [int(c) for r,c in coords]
    pts = {(int(r)-min(rs), int(c)-min(cs)) for r,c in coords}
    return tuple(sorted(pts))


def _holes(g, bbox):
    r0,c0,r1,c1 = bbox
    box = g[r0:r1+1, c0:c1+1]
    seen = np.zeros(box.shape, dtype=bool)
    out=[]
    for r in range(box.shape[0]):
        for c in range(box.shape[1]):
            if seen[r,c] or box[r,c] != BG:
                continue
            stack=[(r,c)]; seen[r,c]=True; comp=[]; touch=False
            while stack:
                rr,cc = stack.pop(); comp.append((rr,cc))
                if rr in {0,box.shape[0]-1} or cc in {0,box.shape[1]-1}: touch=True
                for dr,dc in ((1,0),(-1,0),(0,1),(0,-1)):
                    nr,nc = rr+dr, cc+dc
                    if 0 <= nr < box.shape[0] and 0 <= nc < box.shape[1] and not seen[nr,nc] and box[nr,nc] == BG:
                        seen[nr,nc]=True; stack.append((nr,nc))
            if not touch: out.append(comp)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor each solid object by the color of the holed object whose cavity matches its outer shape.

    Input structure:
    - Azure background with several monochrome objects.
    - Some objects contain exactly one enclosed azure hole.
    - Other objects are solid.

    Rule:
    - Normalize every enclosed hole shape from holed objects.
    - Normalize every solid object's outer foreground shape.
    - When a solid outer shape matches a hole shape, recolor that solid object to the color of the holed source object.
    - Leave holed objects unchanged.

    Key insight:
    - Matching crosses object categories: the key is the cavity shape, not the exterior silhouette of the holed object.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()
    solids = []
    mapping = {}
    for comp in find_components(g, bg=BG, connectivity=4, monochromatic=True):
        bbox = tuple(map(int, comp['bbox']))
        holes = _holes(g, bbox)
        if holes:
            mapping[_norm(max(holes, key=len))] = int(comp['color'])
        else:
            solids.append(comp)
    for comp in solids:
        key = _norm(comp['coords'])
        if key in mapping:
            for r,c in comp['coords']:
                out[int(r), int(c)] = mapping[key]
    return out


def generate_input() -> np.ndarray:
    g = np.full((20,24), BG, dtype=int)
    shapes = [np.array([[1,1],[1,0]], dtype=bool), np.array([[1,1,1]], dtype=bool), np.array([[1],[1],[1]], dtype=bool)]
    colors = random.sample([1,2,3,4,6,7,9], len(shapes)*2)
    for i,shape in enumerate(shapes):
        r0 = 2 + i*6; c0 = 2
        g[r0:r0+shape.shape[0], c0:c0+shape.shape[1]][shape] = colors[i]
        outer_h, outer_w = shape.shape[0]+4, shape.shape[1]+4
        rr,cc = r0, 10+i*4
        g[rr:rr+outer_h, cc:cc+outer_w] = colors[i+len(shapes)]
        sr = rr + 2; sc = cc + 2
        g[sr:sr+shape.shape[0], sc:sc+shape.shape[1]][shape] = BG
    return g
