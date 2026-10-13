from __future__ import annotations
import random
import numpy as np
from arc_utils import find_components, crop_bbox

BG, GREY = 0, 5


def main(grid: np.ndarray) -> np.ndarray:
    """
    Stack same-shaped colored objects inside the grey frame according to the left-edge legend priority.

    Input structure:
    - Black background with a left-edge vertical color legend, several same-shaped objects, and one grey frame.
    - Objects may contain black holes.

    Rule:
    - Read legend colors from top to bottom; colors lower in the legend are higher priority.
    - Translate each object shape to the grey frame interior with aligned top-left bboxes.
    - Overlay them in legend order so higher-priority colors cover lower-priority colors, but black holes stay transparent.
    - Erase original object locations and keep the legend and frame.

    Key insight:
    - All objects share one aligned shape, so only the legend order determines visible colors in the final stack.
    """
    g = np.asarray(grid, dtype=int)
    out = np.zeros_like(g)
    legend = [int(g[r,0]) for r in range(g.shape[0]) if g[r,0] not in {BG,GREY}]
    frame = max([c for c in find_components(g, bg=BG, colors=[GREY], connectivity=4, monochromatic=True)], key=lambda c: int(c['area']))
    fr0,fc0,fr1,fc1 = map(int, frame['bbox'])
    objs = {int(c['color']): crop_bbox(g, tuple(map(int,c['bbox']))) for c in find_components(g, bg=BG, connectivity=4, monochromatic=True) if int(c['color']) not in {BG,GREY} and tuple(map(int,c['bbox'])) != tuple(map(int,frame['bbox']))}
    for color in legend:
        if color not in objs: continue
        patch = objs[color]
        view = out[fr0+1:fr1, fc0+1:fc1]
        ph,pw = min(view.shape[0], patch.shape[0]), min(view.shape[1], patch.shape[1])
        mask = patch[:ph,:pw] != BG
        view[:ph,:pw][mask] = color
    out[g[:,0] != BG, 0] = g[g[:,0] != BG, 0]
    out[fr0,fc0:fc1+1] = GREY; out[fr1,fc0:fc1+1] = GREY; out[fr0:fr1+1,fc0] = GREY; out[fr0:fr1+1,fc1] = GREY
    return out


def generate_input() -> np.ndarray:
    g = np.zeros((20,20), dtype=int)
    legend = random.sample([1,2,3,4,6,7,8,9], random.randint(3,4))
    for i,c in enumerate(legend): g[1+i*2,0] = c
    g[6,12:17] = GREY; g[12,12:17] = GREY; g[6:13,12] = GREY; g[6:13,16] = GREY
    shape = np.array([[1,1,1],[1,0,1],[1,1,1]], dtype=bool)
    for idx,c in enumerate(legend):
        r0 = 2+idx*4; c0 = 4
        g[r0:r0+3,c0:c0+3][shape] = c
    return g
