from __future__ import annotations
import random
import numpy as np
from arc_utils import find_components

BG = 8
ARROWS = {'left': np.array([[1,0],[1,1],[1,0]], dtype=bool), 'right': np.array([[0,1],[1,1],[0,1]], dtype=bool), 'up': np.array([[0,1,0],[1,1,1]], dtype=bool), 'down': np.array([[1,1,1],[0,1,0]], dtype=bool)}
DIRS = {'left':(0,-1), 'right':(0,1), 'up':(-1,0), 'down':(1,0)}


def _solid_rect(mask: np.ndarray) -> bool:
    if not mask.any(): return False
    rows, cols = np.where(mask)
    return (rows.max()-rows.min()+1) * (cols.max()-cols.min()+1) == int(mask.sum())


def main(grid: np.ndarray) -> np.ndarray:
    """
    Slide each color group of fragments together along its arrow direction until the union becomes a solid rectangle.

    Input structure:
    - Azure background with several same-color fragments per group and one same-color arrow per group.

    Rule:
    - Detect the arrow direction for each color.
    - Move every fragment of that color one step at a time in that shared direction.
    - Stop at the first step where the translated union forms a solid rectangle.
    - Erase the arrows and keep only the fragments in their final assembled positions.

    Key insight:
    - The stop condition depends on the whole color-group union becoming rectangular, not on any single fragment touching another.
    """
    g = np.asarray(grid, dtype=int)
    out = np.full_like(g, BG)
    comps = find_components(g, bg=BG, connectivity=4, monochromatic=True)
    by = {}
    for c in comps: by.setdefault(int(c['color']), []).append(c)
    for color, items in by.items():
        arrow = min(items, key=lambda c: int(c['area']))
        fragments = [c for c in items if c is not arrow]
        ar = crop = None
        shape = np.zeros((arrow['bbox'][2]-arrow['bbox'][0]+1, arrow['bbox'][3]-arrow['bbox'][1]+1), dtype=bool)
        for r,c in arrow['coords']: shape[int(r)-arrow['bbox'][0], int(c)-arrow['bbox'][1]] = True
        direction = next((d for d,m in ARROWS.items() if m.shape == shape.shape and np.array_equal(m, shape)), 'right')
        dr,dc = DIRS[direction]
        coords = [np.array(c['coords'], dtype=int) for c in fragments]
        for _ in range(20):
            if any(
                np.any(pts[:, 0] < 0)
                or np.any(pts[:, 0] >= g.shape[0])
                or np.any(pts[:, 1] < 0)
                or np.any(pts[:, 1] >= g.shape[1])
                for pts in coords
            ):
                break
            mask = np.zeros(g.shape, dtype=bool)
            for pts in coords: mask[pts[:,0], pts[:,1]] = True
            if _solid_rect(mask): break
            coords = [pts + np.array([dr,dc]) for pts in coords]
        for pts in coords:
            keep = (pts[:,0] >= 0) & (pts[:,0] < g.shape[0]) & (pts[:,1] >= 0) & (pts[:,1] < g.shape[1])
            out[pts[keep,0], pts[keep,1]] = color
    return out


def generate_input() -> np.ndarray:
    g = np.full((20,20), BG, dtype=int)
    for color in random.sample([1,2,3,4,6,7,9], random.randint(1,3)):
        direction = random.choice(list(DIRS))
        if direction == 'right':
            g[2:4,2:4] = color; g[2:4,6:8] = color; g[2:5,10:12] = color
            m = ARROWS['right']; g[10:13,2:4][m] = color
        elif direction == 'left':
            g[10:12,10:12] = color; g[10:12,14:16] = color; m = ARROWS['left']; g[2:5,16:18][m] = color
        elif direction == 'down':
            g[2:4,10:12] = color; g[5:7,10:12] = color; m = ARROWS['down']; g[2:4,2:5][m] = color
        else:
            g[12:14,4:6] = color; g[15:17,4:6] = color; m = ARROWS['up']; g[16:18,10:13][m] = color
    return g
