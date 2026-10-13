from __future__ import annotations
import random
import numpy as np
from arc_utils import find_components, crop_bbox

BG = 8


def _dir_from_hole(square: np.ndarray) -> str:
    hole = np.argwhere(square == BG)
    r0,c0 = hole.min(axis=0)
    r1,c1 = hole.max(axis=0)
    if r1-r0 > c1-c0:
        return 'down' if r0 < square.shape[0]//2 else 'up'
    return 'right' if c0 < square.shape[1]//2 else 'left'


def main(grid: np.ndarray) -> np.ndarray:
    """
    Group equal-sized colored squares by the direction encoded by their internal L-shaped hole and output solid squares in directional rows or columns.

    Input structure:
    - Azure background with several same-sized solid colored squares.
    - Each square contains one internal azure L-hole.

    Rule:
    - Read the hole orientation to classify each square as up, down, left, or right.
    - Replace every square by its solid version.
    - Pack up-squares across the top row, down-squares across the bottom row, left-squares down the left column, and right-squares down the right column.
    - Preserve scan order within each direction group.

    Key insight:
    - The hidden hole is a directional pointer that decides output placement; it disappears in the output.
    """
    g = np.asarray(grid, dtype=int)
    comps = find_components(g, bg=BG, connectivity=4, monochromatic=True)
    squares=[]
    size=0
    for comp in comps:
        bbox = tuple(map(int, comp['bbox']))
        patch = crop_bbox(g, bbox)
        size = patch.shape[0]
        squares.append((_dir_from_hole(patch), int(comp['color']), patch.shape[0]))
    groups = {k: [] for k in ['up','down','left','right']}
    for d,color,s in squares:
        groups[d].append((color,s))
    top = len(groups['up']); bottom = len(groups['down']); left = len(groups['left']); right = len(groups['right'])
    rows = (1 if top else 0) + max(left,right,0) + (1 if bottom else 0)
    cols = max(top, bottom, 1 if left or right else 0, (1 if left else 0)+(1 if right else 0))
    rows = max(rows,1); cols = max(cols,1)
    out = np.full((rows*size, cols*size), BG, dtype=int)
    for i,(color,_) in enumerate(groups['up']): out[0:size, i*size:(i+1)*size] = color
    for i,(color,_) in enumerate(groups['down']): out[-size:, i*size:(i+1)*size] = color
    start = 1 if top else 0
    for i,(color,_) in enumerate(groups['left']): out[(start+i)*size:(start+i+1)*size, 0:size] = color
    for i,(color,_) in enumerate(groups['right']): out[(start+i)*size:(start+i+1)*size, -size:] = color
    return out


def generate_input() -> np.ndarray:
    size = 5
    g = np.full((20,20), BG, dtype=int)
    dirs = random.sample(['up','down','left','right'], random.randint(2,4))
    spots = [(1,1),(1,10),(10,1),(10,10)]
    for d,(r,c) in zip(dirs, spots, strict=False):
        color = random.choice([1,2,3,4,6,7,9])
        g[r:r+size, c:c+size] = color
        if d == 'up': hole = [(2,2),(3,2),(3,3)]
        elif d == 'down': hole = [(1,2),(1,3),(2,2)]
        elif d == 'left': hole = [(2,2),(2,3),(3,3)]
        else: hole = [(2,1),(2,2),(3,1)]
        for dr,dc in hole: g[r+dr, c+dc] = BG
    return g
