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


def _cup_at(cups, r, c):
    for cup in cups:
        r0, c0, r1, c1 = cup["bbox"]
        if r0 < r < r1 and c0 < c < c1:
            return cup
    return None


def main(grid: np.ndarray) -> np.ndarray:
    """Draw boundary rays and fill only cups whose open side is pierced.

    Input structure:
    Colored U-shaped rectangular cup outlines sit on black background. Orange
    markers on the top or left boundary emit inward rays.

    Rule:
    Left-edge markers emit rightward rays and top-edge markers emit downward
    rays. A ray draws orange on black cells, passes through a cup interior only
    when it enters through that cup's open side, and then stops before the
    opposite wall. After all rays are drawn, any cup interior containing orange
    has its remaining black interior cells filled with the cup color. Original
    boundary markers are erased.

    Key insight:
    Cup opening direction gates whether a ray is allowed to enter; filling is a
    second step triggered by the orange trace inside the cup.
    """
    g = np.array(grid, copy=True)
    out = g.copy()
    cups = U.detect_cups(g, colors={1, 2, 3})
    markers = list(zip(*np.where(g == 7)))
    for r, c in markers:
        if c == 0:
            dr, dc, required = 0, 1, "left"
        elif r == 0:
            dr, dc, required = 1, 0, "up"
        else:
            continue
        x, y = r + dr, c + dc
        while U.in_bounds(g.shape, x, y):
            cup = _cup_at(cups, x, y)
            if cup is not None and cup["opening"] != required:
                break
            if g[x, y] not in (0, 7):
                break
            out[x, y] = 7
            x += dr
            y += dc
    for cup in cups:
        cells = U.interior_of_cup(cup)
        if any(out[r, c] == 7 for r, c in cells):
            for r, c in cells:
                if out[r, c] == 0:
                    out[r, c] = cup["color"]
    for r, c in markers:
        if r in (0, g.shape[0] - 1) or c in (0, g.shape[1] - 1):
            out[r, c] = 0
    return out


def generate_input() -> np.ndarray:
    g = np.zeros((15, 15), dtype=int)
    cups = [
        (2, 4, 6, 8, random.choice([1, 2, 3]), "left"),
        (8, 2, 12, 6, random.choice([1, 2, 3]), "up"),
        (6, 10, 11, 13, random.choice([1, 2, 3]), random.choice(["right", "down"])),
    ]
    for cup in cups:
        U.draw_cup(g, *cup)
    g[4, 0] = 7
    g[0, 4] = 7
    if random.random() < 0.5:
        g[0, 12] = 7
    return g
