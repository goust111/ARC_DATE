from pathlib import Path
import importlib.util
import random
from collections import defaultdict

import numpy as np


def _load_common():
    path = Path(__file__).with_name("_common.py")
    spec = importlib.util.spec_from_file_location("_arc057_common", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


U = _load_common()


def _emitters(g):
    out = []
    seen = set()
    for comp in U.components(g == 7):
        if len(comp) != 3:
            continue
        r0, c0, r1, c1 = U.bbox(comp)
        if r1 - r0 != 1 or c1 - c0 != 1:
            continue
        full = {(r, c) for r in range(r0, r1 + 1) for c in range(c0, c1 + 1)}
        missing = list(full - set(comp))
        if not missing:
            continue
        sr, sc = missing[0]
        col = int(g[sr, sc])
        if col not in (0, 7):
            dirs = {(r0, c0): (-1, -1), (r0, c1): (-1, 1), (r1, c0): (1, -1), (r1, c1): (1, 1)}
            out.append({"source": (sr, sc), "color": col, "dir": dirs[(sr, sc)], "l": comp})
            seen.update(comp)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """Use orange L markers to fire diagonal rays that recolor touched blue objects.

    Input structure:
    Each emitter is a colored source pixel occupying the missing corner of an
    adjacent orange 2x2 L marker. Blue objects sit elsewhere on the black grid.

    Rule:
    The missing L corner encodes a diagonal ray direction. Draw each ray in the
    source color until just before any non-black cell or the grid edge. A blue
    object touched 4-neighbor by ray cells is recolored to the touching ray color,
    with touch count then source row as tie-break. Orange L pixels are erased;
    sources and rays remain.

    Key insight:
    The marker's geometry encodes direction, while contact between the generated
    ray and blue objects determines recoloring.
    """
    g = np.array(grid, copy=True)
    out = g.copy()
    emitters = _emitters(g)
    ray_cells = []
    occupied = g != 0
    for em in emitters:
        r, c = em["source"]
        dr, dc = em["dir"]
        x, y = r + dr, c + dc
        while U.in_bounds(g.shape, x, y):
            if occupied[x, y] and g[x, y] != 7:
                break
            if g[x, y] == 7:
                break
            out[x, y] = em["color"]
            ray_cells.append((x, y, em))
            x += dr
            y += dc
    for comp in U.components(g == 1):
        votes = defaultdict(int)
        for r, c in comp:
            for nr, nc in U.n4(r, c):
                for x, y, em in ray_cells:
                    if (nr, nc) == (x, y):
                        votes[(em["color"], em["source"][0])] += 1
        if votes:
            color, _ = sorted(votes.items(), key=lambda kv: (-kv[1], kv[0][1]))[0][0]
            for r, c in comp:
                out[r, c] = color
    for em in emitters:
        for r, c in em["l"]:
            out[r, c] = 0
    return out


def _put_l(g, corner, missing_corner, color):
    r, c = corner
    pts = {(r, c), (r, c + 1), (r + 1, c), (r + 1, c + 1)}
    miss = {"nw": (r, c), "ne": (r, c + 1), "sw": (r + 1, c), "se": (r + 1, c + 1)}[missing_corner]
    for p in pts - {miss}:
        g[p] = 7
    g[miss] = color


def generate_input() -> np.ndarray:
    g = np.zeros((14, 14), dtype=int)
    colors = random.sample([2, 3, 4, 6], 3)
    emitters = [
        ((9, 2), "ne", colors[0]),
        ((9, 10), "nw", colors[1]),
        ((2, 2), "se", colors[2]),
        ((2, 10), "sw", random.choice([2, 3, 4, 6])),
    ]
    random.shuffle(emitters)
    for args in emitters[:random.randint(2, 4)]:
        _put_l(g, *args)
    blue_shapes = [
        [(6, 6), (6, 7), (7, 6), (7, 7)],
        [(4, 9), (5, 9), (3, 10)],
        [(4, 4), (5, 4), (5, 5)],
        [(9, 6), (9, 7), (10, 7)],
    ]
    random.shuffle(blue_shapes)
    for shape in blue_shapes[:random.randint(2, 4)]:
        for r, c in shape:
            if g[r, c] == 0:
                g[r, c] = 1
    return g
