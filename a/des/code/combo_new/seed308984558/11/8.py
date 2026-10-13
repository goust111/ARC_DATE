from __future__ import annotations

import random

import numpy as np
from arc_utils import find_components


BG = 0


def main(grid: np.ndarray) -> np.ndarray:
    """
    Expand border-touching 2x2 rectangles according to how many outer edges they touch.

    Input structure:
    - Black background with several monochrome 2x2 rectangles.
    - A rectangle may lie in the interior, on one border, or in a corner touching two borders.

    Rule:
    - Remove interior rectangles that touch no border.
    - Stretch one-border rectangles along the direction parallel to the touched border, doubling that span.
    - Enlarge corner rectangles to 4x4 anchored at the touched corner.
    - Keep the canvas size unchanged.

    Key insight:
    - The transform depends only on border-contact count and which borders are touched, not on the rectangle color itself.
    """
    g = np.asarray(grid, dtype=int)
    out = np.zeros_like(g)
    h, w = g.shape
    for comp in find_components(g, bg=BG, connectivity=4, monochromatic=True):
        r0, c0, r1, c1 = map(int, comp["bbox"])
        color = int(comp["color"])
        touches = [r0 == 0, r1 == h - 1, c0 == 0, c1 == w - 1]
        count = sum(touches)
        if count == 0:
            continue
        if count == 2:
            nr0 = 0 if r0 == 0 else h - 4
            nc0 = 0 if c0 == 0 else w - 4
            out[nr0 : nr0 + 4, nc0 : nc0 + 4] = color
        elif r0 == 0 or r1 == h - 1:
            nc0 = max(0, min(w - 4, c0 - 1))
            out[r0 : r0 + 2, nc0 : nc0 + 4] = color
        else:
            nr0 = max(0, min(h - 4, r0 - 1))
            out[nr0 : nr0 + 4, c0 : c0 + 2] = color
    return out


def generate_input() -> np.ndarray:
    h = random.randint(12, 20)
    w = random.randint(12, 20)
    g = np.zeros((h, w), dtype=int)
    colors = random.sample([1, 2, 3, 4, 6, 7, 8, 9], random.randint(3, 6))
    spots = []
    for color in colors:
        kind = random.choice(["center", "edge", "corner"])
        if kind == "corner":
            r0 = 0 if random.choice([True, False]) else h - 2
            c0 = 0 if random.choice([True, False]) else w - 2
        elif kind == "edge":
            if random.choice([True, False]):
                r0 = 0 if random.choice([True, False]) else h - 2
                c0 = random.randint(2, w - 4)
            else:
                c0 = 0 if random.choice([True, False]) else w - 2
                r0 = random.randint(2, h - 4)
        else:
            r0 = random.randint(2, h - 4)
            c0 = random.randint(2, w - 4)
        box = (r0, c0, r0 + 1, c0 + 1)
        if any(not (box[2] < ur0 - 1 or box[0] > ur1 + 1 or box[3] < uc0 - 1 or box[1] > uc1 + 1) for ur0, uc0, ur1, uc1 in spots):
            continue
        spots.append(box)
        g[r0 : r0 + 2, c0 : c0 + 2] = color
    return g
