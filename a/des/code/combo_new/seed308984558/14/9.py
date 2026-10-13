from __future__ import annotations
import random
import numpy as np
from arc_utils import find_components, crop_bbox

BG, GREY = 0, 5


def main(grid: np.ndarray) -> np.ndarray:
    """
    Replace each colored object by the grey template whose area matches it, keeping the object's color and center.

    Input structure:
    - Black background with colored solid objects and grey template objects.
    - Grey template areas are unique; colored objects match templates only by area.

    Rule:
    - Compute component areas for colored objects and grey templates.
    - Map each area to the corresponding grey template mask.
    - Erase original colored objects and all grey templates.
    - Repaint each colored object's matched template shape in that object's color, centered on the object's bbox center.

    Key insight:
    - Area is the only pairing key; color and original shape are irrelevant to template selection.
    """
    g = np.asarray(grid, dtype=int)
    out = np.zeros_like(g)
    comps = find_components(g, bg=BG, connectivity=4, monochromatic=True)
    tmpls={}
    objs=[]
    for comp in comps:
        if int(comp['color']) == GREY: tmpls[int(comp['area'])] = crop_bbox(g, comp['bbox']) != BG
        else: objs.append(comp)
    for obj in objs:
        area = int(obj['area']); color = int(obj['color'])
        if area not in tmpls: continue
        mask = tmpls[area]
        cr = (obj['bbox'][0] + obj['bbox'][2]) / 2.0; cc = (obj['bbox'][1] + obj['bbox'][3]) / 2.0
        r0 = int(round(cr - (mask.shape[0]-1)/2.0)); c0 = int(round(cc - (mask.shape[1]-1)/2.0))
        out[r0:r0+mask.shape[0], c0:c0+mask.shape[1]][mask] = color
    return out


def generate_input() -> np.ndarray:
    g = np.zeros((22,24), dtype=int)
    shapes = [np.array([[1,1],[1,0]], dtype=bool), np.array([[1,1,1]], dtype=bool), np.array([[1],[1],[1],[1]], dtype=bool), np.array([[1,1],[1,1]], dtype=bool)]
    for i,shape in enumerate(random.sample(shapes, random.randint(2,4))):
        tr,tc = 1+i*5, 1
        g[tr:tr+shape.shape[0], tc:tc+shape.shape[1]][shape] = GREY
        color = random.choice([1,2,3,4,6,7,8,9])
        rr,cc = 10+i*2, 12+i*2
        g[rr:rr+shape.shape[0], cc:cc+shape.shape[1]][shape] = color
    return g
