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
    """Recolor quadrant objects by the legend and transparently overlay the quadrants.

    Input structure:
    A grey cross splits an odd-sized grid into four equal quadrants. The top-left
    2x2 cells encode two source-to-target color mappings, and each quadrant
    contains one colored object, sometimes with black holes.

    Rule:
    Read the two legend pairs, erase the legend and separator, recolor each
    quadrant locally, then align the four quadrants by local coordinates and
    overlay non-black pixels in the order lower-left, upper-right, upper-left,
    lower-right. Black pixels act as transparent holes or background.

    Key insight:
    The cross changes the reference frame: all four local panels collapse onto
    one shared canvas after legend-driven recoloring.
    """
    g = np.array(grid, copy=True)
    h, w = g.shape
    sr = next(r for r in range(h) if np.all(g[r, :] == 5))
    sc = next(c for c in range(w) if np.all(g[:, c] == 5))
    mapping = {int(g[0, 0]): int(g[0, 1]), int(g[1, 0]): int(g[1, 1])}
    quads = [
        g[sr + 1:, :sc],
        g[:sr, sc + 1:],
        g[:sr, :sc],
        g[sr + 1:, sc + 1:],
    ]
    out = np.zeros_like(quads[0])
    for q in quads:
        layer = q.copy()
        for src, dst in mapping.items():
            layer[q == src] = dst
        if layer.shape == out.shape:
            out[layer != 0] = layer[layer != 0]
    return out


def _ring(q, r, c, color):
    q[r:r + 4, c] = color
    q[r:r + 4, c + 3] = color
    q[r, c:c + 4] = color
    q[r + 3, c:c + 4] = color


def generate_input() -> np.ndarray:
    qh = qw = random.randint(6, 7)
    h = qh * 2 + 1
    w = qw * 2 + 1
    g = np.zeros((h, w), dtype=int)
    g[qh, :] = 5
    g[:, qw] = 5
    pairs = [(1, 3), (2, 7)]
    if random.random() < 0.5:
        pairs = [(4, 6), (1, 7)]
    for i, (src, dst) in enumerate(pairs):
        g[i, 0] = src
        g[i, 1] = dst
    starts = [(0, 0), (0, qw + 1), (qh + 1, 0), (qh + 1, qw + 1)]
    srcs = [p[0] for p in pairs]
    extra = [4, 6, 8]
    for rr, cc in starts:
        q = g[rr:rr + qh, cc:cc + qw]
        col = random.choice(srcs + extra)
        r = random.randint(1, qh - 4)
        c = random.randint(1, qw - 4)
        if random.random() < 0.65:
            _ring(q, r, c, col)
        else:
            q[r:r + 2, c:c + 3] = col
    return g
