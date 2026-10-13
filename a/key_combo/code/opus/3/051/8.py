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


def _legend(g):
    m = {}
    bottom = g.shape[0] - 1
    r = 0
    while r + 1 < bottom:
        if g[r, 0] and g[r + 1, 0]:
            m[int(g[r, 0])] = int(g[r + 1, 0])
            r += 3
        else:
            r += 1
    return m


def main(grid: np.ndarray) -> np.ndarray:
    """Sort objects by the area of their legend-colored one-cell halo.

    Input structure:
    The left edge maps object colors to halo colors. A grey stripe forms the
    bottom baseline, and several colored objects sit in the main area.

    Rule:
    For each object color in the legend, build an object-plus-halo crop by
    expanding the object's bounding box by one cell and painting black crop
    cells with the halo color. Sort these crops by bounding-box area, clear the
    original objects, and stack the crops centered above the grey stripe from
    smallest at the bottom upward.

    Key insight:
    Halo expansion is used both as visible output and as the sorting key.
    """
    g = np.array(grid, copy=True)
    out = g.copy()
    h, w = g.shape
    mapping = _legend(g)
    main_mask = np.zeros_like(g, dtype=bool)
    main_mask[:h - 1, 2:] = True
    items = []
    for col in mapping:
        for comp in U.components((g == col) & main_mask):
            r0, c0, r1, c1 = U.bbox(comp)
            er0, ec0 = max(0, r0 - 1), max(2, c0 - 1)
            er1, ec1 = min(h - 2, r1 + 1), min(w - 1, c1 + 1)
            crop = np.full((er1 - er0 + 1, ec1 - ec0 + 1), mapping[col], dtype=int)
            for r, c in comp:
                crop[r - er0, c - ec0] = col
            items.append(((crop.shape[0] * crop.shape[1], r0, c0), crop))
            for r, c in comp:
                out[r, c] = 0
    row = h - 2
    for _, crop in sorted(items, key=lambda x: x[0]):
        ch, cw = crop.shape
        r0 = row - ch + 1
        c0 = max(2, (w - cw) // 2)
        if r0 >= 0:
            out[r0:r0 + ch, c0:c0 + cw] = crop
        row = r0 - 2
    return out


def generate_input() -> np.ndarray:
    h, w = 18, 14
    g = np.zeros((h, w), dtype=int)
    pairs = [(1, 4), (2, 7), (3, 6)]
    random.shuffle(pairs)
    for i, (a, b) in enumerate(pairs[:random.randint(2, 3)]):
        g[3 * i, 0] = a
        g[3 * i + 1, 0] = b
    g[-1, :] = 5
    used = np.zeros_like(g, dtype=bool)
    used[-1, :] = True
    for col, _ in pairs[:3]:
        for _try in range(200):
            r = random.randint(1, h - 7)
            c = random.randint(3, w - 5)
            cells = [(r, c), (r + 1, c), (r, c + 1)]
            if random.random() < 0.5:
                cells.append((r + 1, c + 2))
            if all(not used[x, y] for x, y in cells):
                for x, y in cells:
                    g[x, y] = col
                    used[x, y] = True
                break
    return g
