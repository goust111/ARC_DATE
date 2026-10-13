from __future__ import annotations
import random
import numpy as np
from arc_utils import find_components

BG, GREY = 0, 5
NEXT = {2:9, 1:8, 3:4, 4:7, 7:2, 6:1, 9:2, 8:1}


def main(grid: np.ndarray) -> np.ndarray:
    """
    Propagate the innermost block color outward through nested grey frames, transforming color at each layer.

    Input structure:
    - Black background with separate groups of nested grey rectangular frames.
    - The innermost frame contains a small solid colored block.

    Rule:
    - Order frames in each group from inner to outer by containment.
    - Fill the black area inside the innermost frame with the block color.
    - For each next outer gap, transform the previous color through the fixed palette cycle and fill that gap.
    - Keep grey frame borders and the original inner block unchanged.

    Key insight:
    - The nesting depth determines how many times the color-transition rule is applied.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()
    frames = [tuple(map(int,c['bbox'])) for c in find_components(g, bg=BG, colors=[GREY], connectivity=4, monochromatic=True)]
    frames.sort(key=lambda b: (b[2]-b[0])*(b[3]-b[1]))
    used=set()
    for inner in frames:
        if inner in used:
            continue
        r0,c0,r1,c1 = inner
        inner_box = g[r0+1:r1, c0+1:c1]
        colors = [int(v) for v in np.unique(inner_box) if v not in {BG,GREY}]
        if not colors:
            continue
        color = colors[0]
        chain = [inner]
        cur = inner
        while True:
            outers = [b for b in frames if b not in used and b[0] < cur[0] and b[1] < cur[1] and b[2] > cur[2] and b[3] > cur[3]]
            if not outers:
                break
            cur = min(outers, key=lambda b: (b[2]-b[0])*(b[3]-b[1]))
            chain.append(cur)
        prev = None
        cur_color = color
        for box in chain:
            br0,bc0,br1,bc1 = box
            if prev is None:
                region = out[br0+1:br1, bc0+1:bc1]
                region[region == BG] = cur_color
            else:
                pr0,pc0,pr1,pc1 = prev
                region = out[br0+1:br1, bc0+1:bc1]
                mask = np.ones(region.shape, dtype=bool)
                mask[pr0-br0:pr1-br0+1, pc0-bc0:pc1-bc0+1] = False
                region[(region == BG) & mask] = cur_color
            prev = box
            cur_color = NEXT.get(cur_color, cur_color)
        for box in chain:
            used.add(box)
    return out


def generate_input() -> np.ndarray:
    g = np.zeros((22,22), dtype=int)
    for _ in range(random.randint(1,2)):
        layers = random.randint(2,4)
        r0 = random.randint(1,6); c0 = random.randint(1,6)
        h = random.randint(5,7); w = random.randint(5,7)
        for i in range(layers):
            rr,cc = r0-i*2, c0-i*2
            hh,ww = h+i*4, w+i*4
            g[rr,cc:cc+ww] = GREY; g[rr+hh-1,cc:cc+ww] = GREY; g[rr:rr+hh,cc] = GREY; g[rr:rr+hh,cc+ww-1] = GREY
        color = random.choice([1,2,3,4,6,7,8,9])
        g[r0+2:r0+h-2, c0+2:c0+w-2] = color
    return g
