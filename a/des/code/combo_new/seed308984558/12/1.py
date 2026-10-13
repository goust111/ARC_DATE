from __future__ import annotations
import random
import numpy as np
from arc_utils import find_components

BG, FRAME = 0, 1


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill each blue frame with stripes whose orientation comes from the seed offset, drawing smaller interiors first.

    Input structure:
    - Black background with blue hollow rectangles.
    - Each rectangle contains exactly one colored seed pixel.
    - Rectangles may be different sizes.

    Rule:
    - For each rectangle, compare the seed's horizontal and vertical offset from the interior center.
    - If horizontal offset is larger, fill alternating horizontal stripes; otherwise fill alternating vertical stripes, with ties going to horizontal.
    - Use the seed color for stripe cells and black for the alternating gaps.
    - Process interiors in ascending area order while leaving the blue borders unchanged.

    Key insight:
    - The seed position determines stripe orientation, while rectangle size determines draw order.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()
    frames = []
    for comp in find_components(g, bg=BG, colors=[FRAME], connectivity=4, monochromatic=True):
        r0,c0,r1,c1 = map(int, comp['bbox'])
        if r1-r0 < 2 or c1-c0 < 2:
            continue
        frames.append((r0,c0,r1,c1))
    frames.sort(key=lambda b: (b[2]-b[0]-1)*(b[3]-b[1]-1))
    for r0,c0,r1,c1 in frames:
        inner = g[r0+1:r1, c0+1:c1]
        seeds = np.argwhere((inner != BG) & (inner != FRAME))
        if len(seeds) != 1:
            continue
        sr, sc = map(int, seeds[0])
        color = int(inner[sr, sc])
        cr = (inner.shape[0]-1)/2.0
        cc = (inner.shape[1]-1)/2.0
        horiz = abs(sc-cc) >= abs(sr-cr)
        for rr in range(inner.shape[0]):
            for cc2 in range(inner.shape[1]):
                if horiz:
                    out[r0+1+rr, c0+1+cc2] = color if rr % 2 == 0 else BG
                else:
                    out[r0+1+rr, c0+1+cc2] = color if cc2 % 2 == 0 else BG
    out[g == FRAME] = FRAME
    return out


def generate_input() -> np.ndarray:
    h = random.randint(14,24)
    w = random.randint(14,24)
    g = np.zeros((h,w), dtype=int)
    used = []
    for _ in range(random.randint(1,3)):
        for _ in range(100):
            rh = random.randint(5,8)
            rw = random.randint(5,8)
            r0 = random.randint(1,h-rh-2)
            c0 = random.randint(1,w-rw-2)
            box = (r0,c0,r0+rh-1,c0+rw-1)
            if any(not (box[2] < a0-1 or box[0] > a1+1 or box[3] < b0-1 or box[1] > b1+1) for a0,b0,a1,b1 in used):
                continue
            used.append(box)
            r1,c1 = box[2], box[3]
            g[r0,c0:c1+1] = FRAME; g[r1,c0:c1+1] = FRAME; g[r0:r1+1,c0] = FRAME; g[r0:r1+1,c1] = FRAME
            ir, ic = rh-2, rw-2
            color = random.choice([2,3,4,6,7,8,9])
            sr = random.randint(0, ir-1)
            sc = random.randint(0, ic-1)
            g[r0+1+sr, c0+1+sc] = color
            break
    return g
