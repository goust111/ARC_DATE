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


def _guides(g):
    h, w = g.shape
    hs = []
    vs = []
    for r in range(1, h - 1):
        if g[r, 0] != 0 and g[r, 0] == g[r, w - 1]:
            hs.append((r, int(g[r, 0])))
    for c in range(1, w - 1):
        if g[0, c] != 0 and g[0, c] == g[h - 1, c]:
            vs.append((c, int(g[0, c])))
    return hs, vs


def _template(g):
    h, w = g.shape
    border = np.zeros_like(g, dtype=bool)
    border[0, :] = border[-1, :] = border[:, 0] = border[:, -1] = True
    mask = (g != 0) & (~border)
    # Exclude full guide lines from the mask.
    hs, vs = _guides(g)
    for r, _ in hs:
        mask[r, :] = False
    for c, _ in vs:
        mask[:, c] = False
    comp = max(U.components(mask), key=len)
    r0, c0, r1, c1 = U.bbox(comp)
    crop = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=int)
    for r, c in comp:
        crop[r - r0, c - c0] = int(g[r, c])
    indicator_vals = [int(v) for v in np.unique(crop) if v not in (0, 7)]
    indicator = indicator_vals[0] if indicator_vals else 7
    return crop, indicator, (r0, c0, r1, c1)


def _stamp(out, tile, center):
    cr, cc = center
    th, tw = tile.shape
    r0 = cr - th // 2
    c0 = cc - tw // 2
    for r in range(th):
        for c in range(tw):
            if tile[r, c] and U.in_bounds(out.shape, r0 + r, c0 + c):
                out[r0 + r, c0 + c] = tile[r, c]


def main(grid: np.ndarray) -> np.ndarray:
    """Stamp a mirror-gated template at every colored guide-line intersection.

    Input structure:
    Matching same-color markers on opposite borders define horizontal and
    vertical guide lines. A small corner template is mostly orange with one
    colored indicator pixel.

    Rule:
    Draw all border-defined guide lines. At every horizontal/vertical
    intersection, copy the template; mirror left-right when the horizontal line
    color matches the indicator, and mirror up-down when the vertical line color
    matches it. Stamp the transformed copy centered on the intersection, then
    erase the original template.

    Key insight:
    The intersection gives location, while its two line colors independently
    gate the two possible mirror axes.
    """
    g = np.array(grid, copy=True)
    out = g.copy()
    hs, vs = _guides(g)
    for r, col in hs:
        out[r, :] = col
    for c, col in vs:
        out[:, c] = col
    tile, indicator, tb = _template(g)
    for hr, hc in hs:
        for vc, vc_col in vs:
            t = tile.copy()
            if hc == indicator:
                t = np.fliplr(t)
            if vc_col == indicator:
                t = np.flipud(t)
            _stamp(out, t, (hr, vc))
    r0, c0, r1, c1 = tb
    for r in range(r0, r1 + 1):
        for c in range(c0, c1 + 1):
            line_col = 0
            for rr, col in hs:
                if rr == r:
                    line_col = col
            for cc, col in vs:
                if cc == c:
                    line_col = col
            out[r, c] = line_col
    return out


def generate_input() -> np.ndarray:
    g = np.zeros((16, 16), dtype=int)
    rows = sorted(random.sample([5, 6, 10, 11], 2))
    cols = sorted(random.sample([6, 7, 11, 12], 2))
    hs = [(rows[0], random.choice([1, 2, 3])), (rows[1], random.choice([1, 2, 3]))]
    vs = [(cols[0], random.choice([1, 2, 3])), (cols[1], random.choice([1, 2, 3]))]
    for r, col in hs:
        g[r, 0] = g[r, -1] = col
    for c, col in vs:
        g[0, c] = g[-1, c] = col
    tiles = [
        np.array([[0, 7, 0], [7, 2, 7], [7, 0, 0]]),
        np.array([[7, 0, 0], [7, 3, 7], [0, 7, 0]]),
        np.array([[0, 7, 7], [7, 2, 0], [0, 7, 0]]),
    ]
    tile = random.choice(tiles).copy()
    inds = [(r, c) for r in range(3) for c in range(3) if tile[r, c] in (2, 3)]
    if inds and random.random() < 0.5:
        r, c = inds[0]
        tile[r, c] = random.choice([1, 2, 3])
    g[1:4, 1:4] = tile
    return g
