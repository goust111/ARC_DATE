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
    t = np.zeros((block.shape[0] // 2, block.shape[1] // 2), dtype=int)
    for r in range(t.shape[0]):
        for c in range(t.shape[1]):
            vals = block[2 * r:2 * r + 2, 2 * c:2 * c + 2].ravel()
            vals = vals[vals != 0]
            t[r, c] = int(vals[0]) if len(vals) else 0
    return t


def _stamp(out, tile, r0, c0):
    for r in range(tile.shape[0]):
        for c in range(tile.shape[1]):
            rr, cc = r0 + 2 * r, c0 + 2 * c
            if rr + 1 < out.shape[0] and cc + 1 < out.shape[1]:
                out[rr:rr + 2, cc:cc + 2] = tile[r, c]


def main(grid: np.ndarray) -> np.ndarray:
    """Stamp a template variant at red markers according to marker-grid parity.

    Input structure:
    A grey separator sits below a 3x3 coarse template encoded as 2x2 blocks.
    The lower marker panel contains red pixels at coarse grid positions.

    Rule:
    Compress the top template. For each lower red marker, compute its coarse
    coordinate. Even parity stamps the original template expanded to 2x2 blocks;
    odd parity stamps a horizontally mirrored variant in which blue becomes
    orange while green stays green. Template and separator stay unchanged.

    Key insight:
    The marker's coarse-grid parity controls both geometry and palette of the
    stamp.
    """
    g = np.array(grid, copy=True)
    out = g.copy()
    sep = next(r for r in range(g.shape[0]) if np.all(g[r] == 5))
    t = _template(g[:sep])
    var = np.fliplr(t.copy())
    var[var == 1] = 7
    markers = list(zip(*np.where(g[sep + 1:] == 2)))
    for rr, c in markers:
        r = sep + 1 + rr
        R, C = (r - sep - 1) // 2, c // 2
        _stamp(out, t if (R + C) % 2 == 0 else var, r, c)
    return out


def generate_input() -> np.ndarray:
    h, w = 17, 16
    g = np.zeros((h, w), dtype=int)
    coarse = np.array([[1, 3, 1], [3, 1, 3], [1, 3, 3]])
    if random.random() < 0.5:
        coarse = np.fliplr(coarse)
    for r in range(3):
        for c in range(3):
            g[2 * r:2 * r + 2, 2 * c:2 * c + 2] = coarse[r, c]
    sep = 6
    g[sep, :] = 5
    positions = [(7, 0), (7, 8), (11, 2), (11, 9)]
    random.shuffle(positions)
    for r, c in positions[:random.randint(2, 4)]:
        g[r, c] = 2
    return g
