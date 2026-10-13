from __future__ import annotations
import random
import numpy as np
from arc_utils import find_components

BG, FENCE, ORANGE, YELLOW = 0, 9, 7, 4
PATTERNS = {
    'up': {(1,0),(0,1),(1,2)},
    'down': {(0,0),(1,1),(0,2)},
    'left': {(0,1),(1,0),(2,1)},
    'right': {(0,0),(1,1),(2,0)},
}
PRIORITY = {'top':0,'left':1,'right':2,'bottom':3}


def _arrow_info(comp: dict, g: np.ndarray):
    coords = comp['coords']
    r0,c0,r1,c1 = map(int, comp['bbox'])
    shape = {(int(r-r0), int(c-c0)) for r,c in coords}
    color = int(g[int(coords[0][0]), int(coords[0][1])])
    for d,p in PATTERNS.items():
        if shape == p:
            if d == 'up': tip = (r0, c0+1); step=(-1,0)
            elif d == 'down': tip = (r0+1, c0+1); step=(1,0)
            elif d == 'left': tip = (r0+1, c0); step=(0,-1)
            else: tip = (r0+1, c0+1); step=(0,1)
            return color, tip, step
    return color, (r0,c0), (0,1)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Shoot colored ribbons from V-shaped arrows, turning yellow at orange stations and filling reached fences.

    Input structure:
    - Black background with maroon rectangular fences, colored 3-pixel arrows, and orange station pixels.
    - Fence interiors are black.

    Rule:
    - Extend each arrow as a straight ribbon from its tip in the arrow direction.
    - The ribbon stops before a fence wall, the grid edge, or a previously painted ribbon.
    - After the ribbon first passes an orange station, its color becomes yellow from that cell onward.
    - If the final ribbon cell is 4-neighbor adjacent to a fence wall, fill that fence interior with the ribbon's final color, resolving multiple hits by top > left > right > bottom.

    Key insight:
    - The final fill color depends on the traveled ribbon state at impact, not just on the arrow's original color.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()
    arrows = [c for c in find_components(g, bg=BG, connectivity=8, monochromatic=True) if int(c['area']) == 3 and int(c['color']) not in {ORANGE,FENCE}]
    fences = [c for c in find_components(g, bg=BG, colors=[FENCE], connectivity=4, monochromatic=True) if int(c['area']) >= 12]
    fills = {}
    for comp in arrows:
        color, (r,c), (dr,dc) = _arrow_info(comp, g)
        cur = color
        last = None
        rr,cc = r,c
        while 0 <= rr < g.shape[0] and 0 <= cc < g.shape[1] and g[rr,cc] != FENCE and out[rr,cc] in {BG, ORANGE, color, YELLOW}:
            if g[rr,cc] == ORANGE:
                cur = YELLOW
            if out[rr,cc] in {BG, ORANGE}:
                out[rr,cc] = cur
            last = (rr,cc)
            nr,nc = rr+dr, cc+dc
            if not (0 <= nr < g.shape[0] and 0 <= nc < g.shape[1]) or g[nr,nc] == FENCE or (out[nr,nc] not in {BG, ORANGE} and g[nr,nc] != g[rr,cc]):
                break
            rr,cc = nr,nc
        if last is None:
            continue
        lr,lc = last
        for fence in fences:
            r0,c0,r1,c1 = map(int, fence['bbox'])
            if r0-1 <= lr <= r1+1 and c0-1 <= lc <= c1+1:
                side = None
                if lr == r0-1 and c0 <= lc <= c1: side = 'top'
                elif lr == r1+1 and c0 <= lc <= c1: side = 'bottom'
                elif lc == c0-1 and r0 <= lr <= r1: side = 'left'
                elif lc == c1+1 and r0 <= lr <= r1: side = 'right'
                if side is not None:
                    best = fills.get((r0,c0,r1,c1))
                    cand = (PRIORITY[side], cur)
                    if best is None or cand[0] < best[0]:
                        fills[(r0,c0,r1,c1)] = cand
    for bbox, (_, color) in fills.items():
        r0,c0,r1,c1 = bbox
        inner = out[r0+1:r1, c0+1:c1]
        inner[inner == BG] = color
    return out


def generate_input() -> np.ndarray:
    g = np.zeros((20,24), dtype=int)
    for idx in range(random.randint(1,2)):
        r0 = 3 + idx*8; c0 = 10 + idx*3; h=5; w=6
        r1,c1 = r0+h-1, c0+w-1
        g[r0,c0:c1+1] = FENCE; g[r1,c0:c1+1] = FENCE; g[r0:r1+1,c0] = FENCE; g[r0:r1+1,c1] = FENCE
        side = random.choice(['top','left','right','bottom'])
        color = random.choice([2,3,6,8])
        if side == 'left':
            base = (r0+1, c0-3); pts=[(0,1),(1,0),(2,1)]
        elif side == 'right':
            base = (r0+1, c1+1); pts=[(0,0),(1,1),(2,0)]
        elif side == 'top':
            base = (r0-2, c0+1); pts=[(1,0),(0,1),(1,2)]
        else:
            base = (r1+1, c0+1); pts=[(0,0),(1,1),(0,2)]
        for dr,dc in pts:
            g[base[0]+dr, base[1]+dc] = color
        if random.choice([True, False]):
            if side in {'left','right'}:
                station_r = r0+2; station_c = c0-1 if side=='left' else c1+1
            else:
                station_r = r0-1 if side=='top' else r1+1; station_c = c0+2
            if 0 <= station_r < g.shape[0] and 0 <= station_c < g.shape[1]:
                g[station_r, station_c] = ORANGE
    return g
