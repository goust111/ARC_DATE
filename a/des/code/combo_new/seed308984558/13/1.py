from __future__ import annotations
import random
import numpy as np
from arc_utils import find_components, crop_bbox

BG, RED, GREEN = 0, 2, 3


def _scale_mask(patch: np.ndarray, oh: int, ow: int) -> np.ndarray:
    mask = patch != BG
    h,w = patch.shape
    out = np.zeros((oh,ow), dtype=int)
    for r in range(oh):
        for c in range(ow):
            sr = min(h-1, int(r * h / oh))
            sc = min(w-1, int(c * w / ow))
            if mask[sr,sc]: out[r,c] = int(patch[mask][0])
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Select the sprite lying beyond each frame's gap direction, scale it into the frame, and recolor the frame by sprite-area parity.

    Input structure:
    - Background-dominant grid with colored rectangular frames that each have one multi-cell gap on one side.
    - Outside the frames sit separate monochrome sprites, one relevant sprite per frame direction.

    Rule:
    - Detect each frame and identify which side contains the gap.
    - Choose the unique sprite located beyond that side of the frame.
    - Scale the sprite's bbox content to the frame interior and paint it with the sprite color.
    - Recolor the frame border red for odd sprite area and green for even sprite area.

    Key insight:
    - Gap direction selects a sprite first; only then does sprite area decide the new border color.
    """
    g = np.asarray(grid, dtype=int)
    out = np.full_like(g, BG)
    comps = find_components(g, bg=BG, connectivity=4, monochromatic=True)
    frames=[]; sprites=[]
    for comp in comps:
        r0,c0,r1,c1 = map(int, comp['bbox'])
        color = int(comp['color'])
        area = int(comp['area'])
        per = 2*((r1-r0+1)+(c1-c0+1))-4
        if area <= per and r1-r0 >= 3 and c1-c0 >= 3:
            frames.append(comp)
        else:
            sprites.append(comp)
    for frame in frames:
        r0,c0,r1,c1 = map(int, frame['bbox'])
        color = int(frame['color'])
        top = np.sum(g[r0,c0:c1+1] == color); bottom = np.sum(g[r1,c0:c1+1] == color)
        left = np.sum(g[r0:r1+1,c0] == color); right = np.sum(g[r0:r1+1,c1] == color)
        if top < c1-c0+1: side='up'
        elif bottom < c1-c0+1: side='down'
        elif left < r1-r0+1: side='left'
        else: side='right'
        def pred(s):
            br0,bc0,br1,bc1 = map(int, s['bbox'])
            if side=='up': return br1 < r0
            if side=='down': return br0 > r1
            if side=='left': return bc1 < c0
            return bc0 > c1
        candidates = [s for s in sprites if pred(s)]
        if not candidates:
            continue
        sprite = min(candidates, key=lambda s: abs(((s['bbox'][0]+s['bbox'][2])//2)-((r0+r1)//2))+abs(((s['bbox'][1]+s['bbox'][3])//2)-((c0+c1)//2)))
        patch = crop_bbox(g, tuple(map(int, sprite['bbox'])))
        scaled = _scale_mask(patch, r1-r0-1, c1-c0-1)
        out[r0+1:r1, c0+1:c1][scaled != 0] = int(sprite['color'])
        border = RED if int(sprite['area']) % 2 else GREEN
        out[r0,c0:c1+1] = border; out[r1,c0:c1+1] = border; out[r0:r1+1,c0] = border; out[r0:r1+1,c1] = border
    return out


def generate_input() -> np.ndarray:
    g = np.zeros((24,24), dtype=int)
    sprites=[]
    for i in range(random.randint(1,3)):
        color = random.choice([1,3,4,6,7,8,9])
        side = random.choice(['up','down','left','right'])
        r0,c0 = 6+i*4, 6+i*3
        h,w = 6,6
        g[r0,c0:c0+w] = color; g[r0+h-1,c0:c0+w] = color; g[r0:r0+h,c0] = color; g[r0:r0+h,c0+w-1] = color
        if side=='up': g[r0,c0+2:c0+4] = 0; sr,sc = r0-4, c0+1
        elif side=='down': g[r0+h-1,c0+2:c0+4] = 0; sr,sc = r0+h+1, c0+1
        elif side=='left': g[r0+2:r0+4,c0] = 0; sr,sc = r0+1, c0-4
        else: g[r0+2:r0+4,c0+w-1] = 0; sr,sc = r0+1, c0+w+1
        patch = np.array([[color,0],[color,color]], dtype=int)
        g[sr:sr+2, sc:sc+2][patch != 0] = color
    return g
