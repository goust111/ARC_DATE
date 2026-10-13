from pathlib import Path
import importlib.util
import random
from collections import Counter

import numpy as np


def _load_common():
    path = Path(__file__).with_name("_common.py")
    spec = importlib.util.spec_from_file_location("_arc057_common", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


U = _load_common()


DIRS = {"up": (-1, 0), "down": (1, 0), "left": (0, -1), "right": (0, 1)}


def _dominants(g, cr, cc):
    ranges = {
        0: (range(0, cr), range(0, cc)),
        1: (range(0, cr), range(cc + 1, g.shape[1])),
        2: (range(cr + 1, g.shape[0]), range(0, cc)),
        3: (range(cr + 1, g.shape[0]), range(cc + 1, g.shape[1])),
    }
    dom = {}
    for q, (rs, cs) in ranges.items():
        vals = [int(g[r, c]) for r in rs for c in cs if g[r, c] not in (0, 5, 9)]
        dom[q] = Counter(vals).most_common(1)[0][0] if vals else 0
    return dom


def _ray_start(cup):
    r0, c0, r1, c1 = cup["bbox"]
    side = cup["opening"]
    if side == "up":
        return r0 - 1, (c0 + c1) // 2
    if side == "down":
        return r1 + 1, (c0 + c1) // 2
    if side == "left":
        return (r0 + r1) // 2, c0 - 1
    return (r0 + r1) // 2, c1 + 1


def main(grid: np.ndarray) -> np.ndarray:
    """Denoise quadrant colors, fill grey cups, and recolor rays when they cross the separator.

    Input structure:
    A maroon full-row/full-column cross separates four noisy color quadrants.
    Each quadrant contains one grey U-shaped cup with black interior.

    Rule:
    Determine each quadrant's dominant non-black, non-grey, non-maroon color.
    Clear the noisy quadrant pixels, preserve the maroon cross and grey cup
    walls, fill each cup interior with its quadrant dominant color, then shoot a
    ray from the cup opening. Rays take the current quadrant's dominant color
    and switch color after crossing a maroon separator line; they stop before
    grey cup walls or at the grid edge.

    Key insight:
    The separator is both the region boundary used for denoising and the event
    that changes ray color during propagation.
    """
    g = np.array(grid, copy=True)
    cr = next(r for r in range(g.shape[0]) if np.all(g[r, :] == 9))
    cc = next(c for c in range(g.shape[1]) if np.all(g[:, c] == 9))
    dom = _dominants(g, cr, cc)
    out = np.zeros_like(g)
    out[cr, :] = 9
    out[:, cc] = 9
    out[g == 5] = 5
    cups = U.detect_cups(g, colors={5})
    for cup in cups:
        r0, c0, r1, c1 = cup["bbox"]
        q = U.quadrant_of((r0 + r1) // 2, (c0 + c1) // 2, cr, cc)
        if q is None:
            continue
        for r, c in U.interior_of_cup(cup):
            out[r, c] = dom[q]
        dr, dc = DIRS[cup["opening"]]
        r, c = _ray_start(cup)
        while U.in_bounds(g.shape, r, c):
            if g[r, c] == 5:
                break
            q2 = U.quadrant_of(r, c, cr, cc)
            col = dom[q2] if q2 is not None else dom[q]
            out[r, c] = col
            r += dr
            c += dc
    return out


def _fill_noise(g, r_range, c_range, main, colors):
    for r in r_range:
        for c in c_range:
            if g[r, c] == 0:
                g[r, c] = main if random.random() < 0.82 else random.choice([x for x in colors if x != main])


def generate_input() -> np.ndarray:
    g = np.zeros((15, 15), dtype=int)
    cr = cc = 7
    g[cr, :] = 9
    g[:, cc] = 9
    colors = [1, 2, 3, 4]
    random.shuffle(colors)
    _fill_noise(g, range(0, cr), range(0, cc), colors[0], colors)
    _fill_noise(g, range(0, cr), range(cc + 1, 15), colors[1], colors)
    _fill_noise(g, range(cr + 1, 15), range(0, cc), colors[2], colors)
    _fill_noise(g, range(cr + 1, 15), range(cc + 1, 15), colors[3], colors)
    cups = [
        (2, 2, 5, 5, "right"),
        (2, 9, 5, 12, "down"),
        (9, 2, 12, 5, "up"),
        (9, 9, 12, 12, "left"),
    ]
    for cup in cups:
        U.draw_cup(g, cup[0], cup[1], cup[2], cup[3], 5, cup[4])
        for r in range(cup[0] + 1, cup[2]):
            for c in range(cup[1] + 1, cup[3]):
                g[r, c] = 0
    g[cr, :] = 9
    g[:, cc] = 9
    return g
