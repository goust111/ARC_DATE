from __future__ import annotations
import random
import numpy as np
from arc_utils import find_components

BG, BLUE, GREEN, RED = 0, 1, 3, 2


def main(grid: np.ndarray) -> np.ndarray:
    """
    Pair blue fragments around each green marker, rebuild their rectangles, and stack the rebuilt rectangles at the red anchor.

    Input structure:
    - Black background with blue fragments, one green marker per fragment pair, and one red 2x2 anchor.
    - Each pair came from splitting a rectangle into two separated pieces.

    Rule:
    - For each green marker, take the two nearest blue fragments as one pair.
    - If the marker lies between their column ranges, rebuild a left-right rectangle; otherwise rebuild a top-bottom rectangle.
    - Convert each pair into its full solid rectangle and sort rebuilt rectangles by area descending.
    - Place them below one another starting from the red anchor's top-left corner, then clear the original fragments, markers, and anchor.

    Key insight:
    - The green marker resolves both pairing and assembly axis; the output keeps only the reconstructed rectangles.
    """
    g = np.asarray(grid, dtype=int)
    out = np.zeros_like(g)
    blues = [c for c in find_components(g, bg=BG, colors=[BLUE], connectivity=4, monochromatic=True)]
    greens = [tuple(map(int, c['coords'][0])) for c in find_components(g, bg=BG, colors=[GREEN], connectivity=4, monochromatic=True)]
    anchor = next(c for c in find_components(g, bg=BG, colors=[RED], connectivity=4, monochromatic=True) if tuple(map(int,c['bbox'][2:])) != tuple(map(int,c['bbox'][:2])))
    rects=[]
    for gr,gc in greens:
        pair = sorted(blues, key=lambda c: abs(((c['bbox'][0]+c['bbox'][2])//2)-gr)+abs(((c['bbox'][1]+c['bbox'][3])//2)-gc))[:2]
        a,b = pair
        if max(a['bbox'][1], b['bbox'][1]) >= gc >= min(a['bbox'][3], b['bbox'][3]):
            h = max(a['bbox'][2]-a['bbox'][0]+1, b['bbox'][2]-b['bbox'][0]+1)
            w = (a['bbox'][3]-a['bbox'][1]+1) + (b['bbox'][3]-b['bbox'][1]+1)
        else:
            h = (a['bbox'][2]-a['bbox'][0]+1) + (b['bbox'][2]-b['bbox'][0]+1)
            w = max(a['bbox'][3]-a['bbox'][1]+1, b['bbox'][3]-b['bbox'][1]+1)
        rects.append((h*w,h,w))
    rects.sort(reverse=True)
    r0,c0,_,_ = map(int, anchor['bbox'])
    cur = r0
    for _,h,w in rects:
        out[cur:cur+h, c0:c0+w] = BLUE
        cur += h
    return out


def generate_input() -> np.ndarray:
    g = np.zeros((22,22), dtype=int)
    g[1:3,1:3] = RED
    row = 5
    for _ in range(random.randint(1,3)):
        h = random.randint(2,4); w = random.randint(4,6)
        split = random.randint(1,w-1)
        g[row:row+h, 4:4+split] = BLUE
        g[row:row+h, 8+split:8+w] = BLUE
        g[row+h//2, 7] = GREEN
        row += h + 3
    return g
