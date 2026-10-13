from __future__ import annotations
import random
import numpy as np
from arc_utils import find_components, crop_bbox

BG, GREY = 0, 5


def _tile_mask(mask: np.ndarray, h: int, w: int, color: int) -> np.ndarray:
    out = np.zeros((h,w), dtype=int)
    mh,mw = mask.shape
    for r in range(h):
        for c in range(w):
            if mask[r % mh, c % mw]: out[r,c] = color
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Tile each fence interior with the external template whose color matches the fence's seed color.

    Input structure:
    - Black background with grey rectangular fences containing same-color seed pixels.
    - Outside the fences are monochrome template shapes, one template color per fence seed color.

    Rule:
    - Read the seed color inside each fence.
    - Find the monochrome external template of that color and crop it to its tight bbox mask.
    - Tile that mask periodically from the fence interior's top-left corner, painting only template foreground cells.
    - Leave fences and external templates unchanged; the seed is overwritten by the tiling.

    Key insight:
    - Color links a fence to its template, while spatial tiling repeats the template shape across the full interior.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()
    comps = find_components(g, bg=BG, connectivity=4, monochromatic=True)
    templates = {}
    fences = []
    for comp in comps:
        r0,c0,r1,c1 = map(int, comp['bbox'])
        color = int(comp['color'])
        if color == GREY and r1-r0 >= 3 and c1-c0 >= 3:
            fences.append(comp)
        elif color != GREY:
            templates[color] = crop_bbox(g, comp['bbox']) != BG
    for fence in fences:
        r0,c0,r1,c1 = map(int, fence['bbox'])
        inner = g[r0+1:r1, c0+1:c1]
        seeds = [int(v) for v in np.unique(inner) if v != BG]
        if not seeds or seeds[0] not in templates: continue
        tile = _tile_mask(templates[seeds[0]], r1-r0-1, c1-c0-1, seeds[0])
        out[r0+1:r1, c0+1:c1][tile != 0] = seeds[0]
    return out


def generate_input() -> np.ndarray:
    g = np.zeros((24,24), dtype=int)
    colors = random.sample([1,2,3,4,6,7,8,9], random.randint(1,3))
    for i,color in enumerate(colors):
        tr,tc = 1+i*4, 1
        shape = np.array([[1,1,0],[1,0,0],[1,1,1]], dtype=bool)
        g[tr:tr+3, tc:tc+3][shape] = color
        r0,c0 = 7+i*5, 10
        g[r0,c0:c0+6] = GREY; g[r0+5,c0:c0+6] = GREY; g[r0:r0+6,c0] = GREY; g[r0:r0+6,c0+5] = GREY
        g[r0+2,c0+2] = color
    return g
