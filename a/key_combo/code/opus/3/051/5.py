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
    """Use a top mask to invert concentric colors inside lower blue rectangles.

    Input structure:
    A yellow/black binary mask sits above a full grey separator. The equally
    sized lower panel contains one or more solid blue rectangles on black.

    Rule:
    Convert each blue rectangle into red/green Chebyshev-distance layers. Then
    compare that lower non-black mask with the top yellow mask. Where exactly
    one mask is active and a rectangle cell exists, swap red and green; all
    other cells keep the concentric base color. Return only the lower panel.

    Key insight:
    The geometric layer pattern is an intermediate image, and the separate mask
    modulates it by xor.
    """
    g = np.array(grid, copy=True)
    sep = next(r for r in range(g.shape[0]) if np.all(g[r] == 5))
    mask = g[:sep] == 4
    lower = g[sep + 1:].copy()
    base = np.zeros_like(lower)
    for comp in U.components(lower == 1):
        r0, c0, r1, c1 = U.bbox(comp)
        for r, c in comp:
            d = min(r - r0, c - c0, r1 - r, c1 - c)
            base[r, c] = 2 if d % 2 == 0 else 3
    active = base != 0
    flip = np.logical_xor(mask[:base.shape[0], :base.shape[1]], active) & active
    out = base.copy()
    out[flip & (base == 2)] = 3
    out[flip & (base == 3)] = 2
    return out


def generate_input() -> np.ndarray:
    h, w = random.randint(7, 9), random.randint(10, 13)
    g = np.zeros((2 * h + 1, w), dtype=int)
    g[h, :] = 5
    for _ in range(random.randint(1, 3)):
        rh, rw = random.randint(3, 5), random.randint(3, 5)
        for _try in range(100):
            r = random.randint(h + 1, 2 * h - rh)
            c = random.randint(0, w - rw)
            if np.all(g[r:r + rh, c:c + rw] == 0):
                g[r:r + rh, c:c + rw] = 1
                break
    mask = np.random.choice([0, 4], size=(h, w), p=[0.7, 0.3])
    g[:h] = mask
    return g
