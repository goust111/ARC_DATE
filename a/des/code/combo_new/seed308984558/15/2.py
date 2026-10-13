import random

import numpy as np


def main(grid: np.ndarray) -> np.ndarray:
    """Trace each source-to-frame pipe and fill the frame interior with entry-directed stripes."""
    g = np.array(grid, dtype=int)
    out = g.copy()
    frames = []
    rs, cs = np.where(g == 8)
    if len(rs):
        r0, r1, c0, c1 = int(rs.min()), int(rs.max()), int(cs.min()), int(cs.max())
        frames.append((r0, r1, c0, c1))
    for color in sorted(set(map(int, g.ravel())) - {0, 5, 8}):
        rs, cs = np.where(g == color)
        if len(rs) != 4:
            continue
        r0, r1, c0, c1 = int(rs.min()), int(rs.max()), int(cs.min()), int(cs.max())
        if r1 - r0 != 1 or c1 - c0 != 1:
            continue
        for fr0, fr1, fc0, fc1 in frames:
            side = "left" if c1 < fc0 else "right" if c0 > fc1 else "top" if r1 < fr0 else "bottom"
            inner = out[fr0 + 1 : fr1, fc0 + 1 : fc1]
            if side == "left":
                inner[:, ::2] = color
            elif side == "right":
                inner[:, 1::2] = color
            elif side == "top":
                inner[::2, :] = color
            else:
                inner[1::2, :] = color
    return out


def generate_input() -> np.ndarray:
    g = np.zeros((random.randint(18, 24), random.randint(18, 24)), dtype=int)
    fr0, fc0 = random.randint(5, 7), random.randint(5, 7)
    fh, fw = random.randint(6, 8), random.randint(6, 8)
    fr1, fc1 = fr0 + fh - 1, fc0 + fw - 1
    g[fr0 : fr1 + 1, fc0] = 8
    g[fr0 : fr1 + 1, fc1] = 8
    g[fr0, fc0 : fc1 + 1] = 8
    g[fr1, fc0 : fc1 + 1] = 8
    colors = random.sample([1, 2, 3, 4, 6, 7, 9], random.randint(1, 3))
    sides = random.sample(["left", "right", "top", "bottom"], len(colors))
    for color, side in zip(colors, sides):
        if side == "left":
            if fc0 < 5:
                continue
            rr = random.randint(fr0 + 1, fr1 - 2)
            cc = random.randint(1, fc0 - 4)
            g[rr : rr + 2, cc : cc + 2] = color
            g[rr, cc + 2 : fc0] = 5
        elif side == "right":
            if fc1 > g.shape[1] - 6:
                continue
            rr = random.randint(fr0 + 1, fr1 - 2)
            cc = random.randint(fc1 + 2, g.shape[1] - 3)
            g[rr : rr + 2, cc : cc + 2] = color
            g[rr, fc1 + 1 : cc] = 5
        elif side == "top":
            if fr0 < 5:
                continue
            rr = random.randint(1, fr0 - 4)
            cc = random.randint(fc0 + 1, fc1 - 2)
            g[rr : rr + 2, cc : cc + 2] = color
            g[rr + 2 : fr0, cc] = 5
        else:
            if fr1 > g.shape[0] - 6:
                continue
            rr = random.randint(fr1 + 2, g.shape[0] - 3)
            cc = random.randint(fc0 + 1, fc1 - 2)
            g[rr : rr + 2, cc : cc + 2] = color
            g[fr1 + 1 : rr, cc] = 5
    return g
