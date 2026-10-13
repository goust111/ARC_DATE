from pathlib import Path
import importlib.util
import random

import numpy as np


def _load_common():
    path = Path(__file__).with_name("_common.py")
    spec = importlib.util.spec_from_file_location("_arc057_common", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


U = _load_common()


DIRS = {"up": (-1, 0), "down": (1, 0), "left": (0, -1), "right": (0, 1)}


def main(grid: np.ndarray) -> np.ndarray:
    """Push each cup's internal fragment out through the cup opening and erase the cups.

    Input structure:
    Grey U-shaped cups each contain one colored connected fragment separated
    from the walls by black space.

    Rule:
    The missing wall of a cup gives the fragment's movement direction. Process
    cups in reading order, translating each whole fragment until the next step
    would leave the grid or overlap an already placed fragment. Draw only the
    final fragment positions; erase all grey cup walls.

    Key insight:
    The container geometry is converted into a motion instruction for its
    nested object, and earlier moved fragments become obstacles.
    """
    g = np.array(grid, copy=True)
    out = np.zeros_like(g)
    cups = sorted(U.detect_cups(g, colors={5}), key=lambda x: x["bbox"][:2])
    occupied = set()
    for cup in cups:
        r0, c0, r1, c1 = cup["bbox"]
        inside_mask = np.zeros_like(g, dtype=bool)
        inside_mask[r0 + 1:r1, c0 + 1:c1] = True
        comps = U.color_components(g * inside_mask, ignore=(0, 5))
        if not comps:
            continue
        col, comp = max(comps, key=lambda x: len(x[1]))
        dr, dc = DIRS[cup["opening"]]
        cells = set(comp)
        while True:
            nxt = {(r + dr, c + dc) for r, c in cells}
            if any(not U.in_bounds(g.shape, r, c) or (r, c) in occupied for r, c in nxt):
                break
            cells = nxt
        for r, c in cells:
            out[r, c] = col
            occupied.add((r, c))
    return out


def generate_input() -> np.ndarray:
    g = np.zeros((16, 16), dtype=int)
    colors = random.sample([1, 2, 3, 4], 4)
    shapes = [
        [(0, 0), (1, 0), (1, 1)],
        [(0, 1), (1, 0), (1, 1)],
        [(0, 0), (0, 1), (1, 0)],
        [(0, 0), (0, 1), (1, 1), (2, 1)],
    ]
    specs = [
        (1, 1, 5, 5, "left", colors[0], random.choice(shapes)),
        (1, 9, 5, 13, "right", colors[1], random.choice(shapes)),
        (9, 2, 13, 6, "down", colors[2], random.choice(shapes)),
        (9, 10, 13, 14, "up", colors[3], random.choice(shapes)),
    ]
    random.shuffle(specs)
    for r0, c0, r1, c1, opening, col, pts in specs[:random.randint(3, 4)]:
        U.draw_cup(g, r0, c0, r1, c1, 5, opening)
        ar, ac = r0 + 2, c0 + 2
        for dr, dc in pts:
            g[ar + dr, ac + dc] = col
    return g
