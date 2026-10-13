import random
import numpy as np


def _extract(mask):
    pts = np.argwhere(mask)
    if pts.size == 0:
        return np.zeros((0, 0), dtype=int)
    r0, c0 = pts.min(axis=0)
    r1, c1 = pts.max(axis=0)
    return mask[r0:r1 + 1, c0:c1 + 1].astype(int)


def _kron_inverted(mask, color):
    mh, mw = mask.shape
    out = np.zeros((mh * mh, mw * mw), dtype=int)
    colored = mask * color
    for r in range(mh):
        for c in range(mw):
            if mask[r, c] == 0:
                out[r * mh:(r + 1) * mh, c * mw:(c + 1) * mw] = colored
    return out


def main(grid: np.ndarray) -> np.ndarray:
    g = np.array(grid, copy=True)
    rows = [r for r in range(g.shape[0]) if np.all(g[r] == 2)]
    if not rows:
        return g
    split = rows[0]
    upper = _extract(g[:split] == 1)
    lower = _extract(g[split + 1:] == 1)
    ku = _kron_inverted(upper, 2)
    kl = _kron_inverted(lower, 1)
    width = max(ku.shape[1] if ku.size else 0, kl.shape[1] if kl.size else 0)
    out = np.zeros(((ku.shape[0] if ku.size else 0) + 1 + (kl.shape[0] if kl.size else 0), width), dtype=int)
    if ku.size:
        out[:ku.shape[0], :ku.shape[1]] = ku
    out[ku.shape[0], :] = 2
    if kl.size:
        out[ku.shape[0] + 1:, :kl.shape[1]] = kl
    return out


def generate_input() -> np.ndarray:
    h = w = 7
    g = np.zeros((h, w), dtype=int)
    split = random.choice([3, 4])
    g[split, :] = 2
    upper_h = split
    lower_h = h - split - 1
    for top, rows in ((0, upper_h), (split + 1, lower_h)):
        mh = random.randint(2, min(4, rows))
        mw = random.randint(2, 4)
        r0 = top + random.randint(0, rows - mh)
        c0 = random.randint(1, w - mw - 1)
        cells = [(r0 + r, c0 + c) for r in range(mh) for c in range(mw)]
        ones = random.sample(cells, random.randint(2, len(cells) - 2))
        for r, c in ones:
            g[r, c] = 1
    return g
