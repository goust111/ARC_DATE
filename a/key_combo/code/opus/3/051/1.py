from pathlib import Path
import importlib.util
import random

import numpy as np


def _load_common():
    path = Path(__file__).with_name("_common.py")
    spec = importlib.util.spec_from_file_location("_arc051_common", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


U = _load_common()


def main(grid: np.ndarray) -> np.ndarray:
    """Fill closed red-wall regions with their local diagonal cycle and mark repaired walls green.

    Input structure:
    Black background contains red one-cell walls. Some walls form closed regions,
    and each closed region contains visible colored samples from a diagonal cycle.

    Rule:
    Treat red as walls, flood-fill all non-wall cells reachable from the canvas
    border, and process every remaining enclosed component independently. Infer
    a mapping from (row + col) modulo the number of sampled colors to color, fill
    only black cells inside that component, and recolor adjacent red wall pixels
    green. Exterior and open-region contents stay unchanged.

    Key insight:
    The red walls classify regions first; only enclosed regions get pattern
    repair, and the wall recoloring records that classification.
    """
    g = np.array(grid, copy=True)
    out = g.copy()
    ext = U.flood_exterior_not_wall(g, wall=2)
    enclosed = (g != 2) & (~ext)
    for comp in U.components(enclosed):
        samples = [(r, c, int(g[r, c])) for r, c in comp if g[r, c] not in (0, 2)]
        if not samples:
            continue
        colors = sorted({v for _, _, v in samples})
        k = len(colors)
        mapping = {}
        for r, c, v in samples:
            mapping[(r + c) % k] = v
        for i, col in enumerate(colors):
            mapping.setdefault(i, col)
        for r, c in comp:
            if g[r, c] == 0:
                out[r, c] = mapping[(r + c) % k]
            for nr, nc in U.n8(r, c):
                if U.in_bounds(g.shape, nr, nc) and g[nr, nc] == 2:
                    out[nr, nc] = 3
    return out


def generate_input() -> np.ndarray:
    h = w = random.randint(15, 19)
    g = np.zeros((h, w), dtype=int)
    rects = [(2, 2, 6, 6), (8, w - 8, 13, w - 3)]
    if h > 17:
        rects[1] = (9, w - 8, 15, w - 3)
    colors = [1, 4, 7]
    for idx, (r0, c0, r1, c1) in enumerate(rects):
        U.draw_rect_border(g, r0, c0, r1, c1, 2)
        perm = colors[:]
        random.shuffle(perm)
        if idx == 1:
            perm = perm[1:] + perm[:1]
        need = set(range(3))
        cells = [(r, c) for r in range(r0 + 1, r1) for c in range(c0 + 1, c1)]
        random.shuffle(cells)
        for r, c in cells:
            residue = (r + c) % 3
            if residue in need:
                g[r, c] = perm[residue]
                need.remove(residue)
            if not need:
                break
        for r, c in random.sample(cells, random.randint(1, 3)):
            g[r, c] = perm[(r + c) % 3]
    # Open distractor wall connected to the exterior.
    r0, c0 = 3, w - 5
    g[r0:r0 + 6, c0] = 2
    g[r0 + 5, c0:c0 + 4] = 2
    for r in range(r0 + 1, r0 + 5):
        if random.random() < 0.45:
            g[r, c0 + 2] = random.choice(colors)
    return g
