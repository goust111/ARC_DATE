from __future__ import annotations
import random
import numpy as np
from arc_utils import find_components

BG, GREY = 0, 5
CORNER_TO_K = {(0,0):0, (0,2):3, (2,2):2, (2,0):1}


def main(grid: np.ndarray) -> np.ndarray:
    """
    Match each colored L triomino to its same-color marked grey template and rotate it by the corner code.

    Input structure:
    - Black background with grey 3x3 template squares.
    - Each template has one colored corner marker.
    - Separate monochrome L triomino fragments share colors with templates.

    Rule:
    - Pair every fragment with the template that has the same corner-marker color.
    - Rotate the fragment according to which corner is marked.
    - Place the rotated triomino in the template's centered 2x2 area and restore the marked corner to grey.
    - Erase the fragment's original location to black.

    Key insight:
    - Color performs the pairing, but the marked corner determines the rotation.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()
    templates = {}
    for r in range(g.shape[0]-2):
        for c in range(g.shape[1]-2):
            block = g[r:r+3, c:c+3]
            greyish = (block == GREY) | ((block != BG) & (block != GREY))
            if not np.all(greyish):
                continue
            corners = [(0,0),(0,2),(2,2),(2,0)]
            marks = [(rr,cc) for rr,cc in corners if block[rr,cc] not in {BG,GREY}]
            if len(marks) == 1 and np.sum(block == GREY) >= 8:
                rr,cc = marks[0]
                templates[int(block[rr,cc])] = (r,c,rr,cc)
                out[r+rr, c+cc] = GREY
    for comp in find_components(g, bg=BG, connectivity=4, monochromatic=True):
        color = int(comp['color'])
        if color in {BG, GREY} or int(comp['area']) != 3 or color not in templates:
            continue
        coords = comp['coords']
        r0,c0,r1,c1 = map(int, comp['bbox'])
        patch = np.zeros((r1-r0+1, c1-c0+1), dtype=int)
        for rr,cc in coords:
            patch[int(rr)-r0, int(cc)-c0] = color
            out[int(rr), int(cc)] = BG
        tr,tc,mr,mc = templates[color]
        rotated = np.rot90(patch, CORNER_TO_K[(mr,mc)])
        dest = out[tr+1:tr+3, tc+1:tc+3]
        sh = min(2, rotated.shape[0]); sw = min(2, rotated.shape[1])
        dest[:sh,:sw][rotated[:sh,:sw] != 0] = color
    return out


def generate_input() -> np.ndarray:
    g = np.zeros((18,18), dtype=int)
    colors = random.sample([1,2,3,4,6,7,8,9], random.randint(2,4))
    positions = [(1,1),(1,8),(8,1),(8,8)]
    random.shuffle(positions)
    for color,(r,c) in zip(colors, positions, strict=False):
        g[r:r+3,c:c+3] = GREY
        corner = random.choice(list(CORNER_TO_K))
        g[r+corner[0], c+corner[1]] = color
        pr = random.randint(12,15)
        pc = random.randint(1,15)
        patch = np.array([[color,0],[color,color]], dtype=int)
        g[pr:pr+2, pc:pc+2][patch != 0] = color
    return g
