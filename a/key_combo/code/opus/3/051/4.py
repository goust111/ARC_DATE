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


def _template(top):
    coords = np.argwhere((top != 0) & (top != 5))
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    block = top[r0:r1 + 1, c0:c1 + 1]
    th, tw = block.shape[0] // 2, block.shape[1] // 2
    t = np.zeros((th, tw), dtype=int)
    for r in range(th):
        for c in range(tw):
            vals = block[2 * r:2 * r + 2, 2 * c:2 * c + 2].ravel()
            vals = vals[vals != 0]
            t[r, c] = int(vals[0]) if len(vals) else 0
    return t


def main(grid: np.ndarray) -> np.ndarray:
    """Fill hollow magenta rectangles using either a template stamp or concentric layers.

    Input structure:
    A grey separator row lies below a 2x2 coarse template encoded as 2x2 color
    blocks. Below it are hollow magenta rectangles of different inner areas.

    Rule:
    Rectangles whose inner area can hold the coarse template receive a periodic
    tiling of the template, expanded back to 2x2 blocks and clipped to the
    interior. Smaller interiors are filled by alternating red and yellow
    Chebyshev-distance layers. Frames, separator, and template remain.

    Key insight:
    Interior area selects between two different filling mechanisms.
    """
    g = np.array(grid, copy=True)
    out = g.copy()
    sep = next(r for r in range(g.shape[0]) if np.all(g[r] == 5))
    t = _template(g[:sep])
    for r0, c0, r1, c1 in U.find_hollow_rects_by_color(g[sep + 1:], 6):
        r0 += sep + 1
        r1 += sep + 1
        ih, iw = r1 - r0 - 1, c1 - c0 - 1
        if ih <= 0 or iw <= 0:
            continue
        if ih * iw >= t.size:
            for r in range(ih):
                for c in range(iw):
                    out[r0 + 1 + r, c0 + 1 + c] = t[(r // 2) % t.shape[0], (c // 2) % t.shape[1]]
        else:
            for r in range(ih):
                for c in range(iw):
                    d = min(r, c, ih - 1 - r, iw - 1 - c) + 1
                    out[r0 + 1 + r, c0 + 1 + c] = 2 if d % 2 else 4
    return out


def generate_input() -> np.ndarray:
    h, w = random.choice([(18, 14), (19, 15), (18, 16)])
    g = np.zeros((h, w), dtype=int)
    pats = [
        np.array([[1, 3], [3, 1]]),
        np.array([[3, 1], [1, 3]]),
        np.array([[1, 1], [3, 3]]),
        np.array([[3, 1], [3, 1]]),
    ]
    pat = random.choice(pats)
    for r in range(2):
        for c in range(2):
            g[2 * r:2 * r + 2, 2 * c:2 * c + 2] = pat[r, c]
    sep = 4
    g[sep, :] = 5
    candidates = [
        (6, 1, 10, 5),
        (6, 8, 9, 11),
        (12, 2, min(h - 2, 16), 7),
        (12, 9, min(h - 2, 14), min(w - 2, 12)),
        (7, max(1, w - 5), 9, w - 2),
    ]
    rects = candidates[:3]
    if random.random() < 0.7:
        rects.append(random.choice(candidates[3:]))
    for rect in rects:
        U.draw_rect_border(g, *rect, 6)
    return g
