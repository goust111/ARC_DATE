import random

import numpy as np


def _find_period_and_pattern(mask: np.ndarray):
    H, W = mask.shape
    # use rows that are non-empty (presence pattern appears)
    base = None
    for r in range(H):
        row = mask[r]
        if row.any():
            base = row.astype(int)
            break
    if base is None:
        return 1, np.zeros(W, dtype=int)
    base = base.astype(int)

    for p in range(1, W + 1):
        pat = base[:p]
        ok = True
        for r in range(H):
            row = mask[r].astype(int)
            for c, v in enumerate(row):
                if v != pat[c % p]:
                    ok = False
                    break
            if not ok:
                break
        if ok:
            return p, pat
    return W, base


def _complete(mask: np.ndarray, color: int):
    p, pat = _find_period_and_pattern(mask)
    H, W = mask.shape
    out = np.zeros((H, W), dtype=int)
    for r in range(H):
        for c in range(W):
            if pat[c % p]:
                out[r, c] = color
    return out


def main(grid: np.ndarray) -> np.ndarray:
    g = np.array(grid, dtype=int)
    H, W = g.shape
    sep = np.where(np.all(g == 5, axis=1))[0]
    if len(sep) == 0:
        return np.zeros((0, 0), dtype=int)
    m = int(sep[0])

    upper = g[:m]
    lower = g[m + 1 :]
    if upper.size == 0 or lower.size == 0:
        return np.zeros((max(0, H - 1), W), dtype=int)

    upper_color = 1
    lower_color = 2
    up_vals = [v for v in np.unique(upper) if v not in (0, 5)]
    lo_vals = [v for v in np.unique(lower) if v not in (0, 5)]
    if up_vals:
        upper_color = int(up_vals[0])
    if lo_vals:
        lower_color = int(lo_vals[0])

    upper_full = _complete(upper == upper_color, upper_color)
    lower_full = _complete(lower == lower_color, lower_color)

    out = np.zeros((H - 1, W), dtype=int)
    for r in range(H - 1):
        for c in range(W):
            a = r < upper_full.shape[0] and upper_full[r, c] == upper_color
            b = r < lower_full.shape[0] and lower_full[r, c] == lower_color
            if a and b:
                out[r, c] = 3
            elif a:
                out[r, c] = upper_color
            elif b:
                out[r, c] = lower_color

    return out


def generate_input() -> np.ndarray:
    H = 12
    W = 20
    g = np.zeros((H, W), dtype=int)

    sep = H // 2
    g[sep, :] = 5

    P1 = random.choice([3, 4, 5])
    P2 = random.choice([3, 4, 5, 6])
    while P2 == P1:
        P2 = random.choice([3, 4, 5, 6])

    pat1 = [random.randint(0, 1) for _ in range(P1)]
    if sum(pat1) == 0:
        pat1[0] = 1
    pat2 = [random.randint(0, 1) for _ in range(P2)]
    if sum(pat2) == 0:
        pat2[0] = 1

    up_w = random.randint(P1, W - 1)
    lo_w = random.randint(P2, W - 1)

    for r in range(sep):
        for c in range(W):
            if c < up_w and pat1[c % P1] == 1:
                g[r, c] = 1

    for r in range(H - sep - 1):
        rr = sep + 1 + r
        for c in range(W):
            if c < lo_w and pat2[c % P2] == 1:
                g[rr, c] = 2

    return g
