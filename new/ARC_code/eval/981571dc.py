from __future__ import annotations
from collections import Counter
import numpy as np

def main(grid: np.ndarray) -> np.ndarray:
    """
    Reconstruct a 4-way symmetric mosaic by filling missing (black) pixels.

    Input structure:
    - A colored grid exhibiting bilateral symmetry along both a vertical and a horizontal axis.
    - Some pixels are erased to color 0 (black), creating gaps in the pattern.

    Rule:
    - Infer the vertical and horizontal reflection axes from the visible (non-zero) pixels.
    - For each black pixel, recover its color from the mirrored position across either axis or both.
    - When multiple mirrors yield candidates, use the majority color.

    Key insight: The reflection axes are not given — they must be deduced from the surviving pixels by finding the axes that best preserve the existing symmetry.
    """
    g = np.array(grid, dtype=int).copy()
    H, W = g.shape
    miss = 0
    assert not (not bool((g == int(miss)).any()))

    nz = g != int(miss)

    # Find best vertical axis (a2 = 2*axis), minimizing mismatches on known cells.
    best_a2 = 0
    best_key = None
    for a2 in range(0, 2 * (W - 1) + 1):
        mism = 0
        pairs = 0
        for r in range(H):
            cols = np.where(nz[int(r)])[0]
            for c in cols:
                c2 = int(a2) - int(c)
                if 0 <= c2 < W and bool(nz[int(r), int(c2)]):
                    pairs += 1
                    if int(g[int(r), int(c)]) != int(g[int(r), int(c2)]):
                        mism += 1
        key = (int(mism), -int(pairs))
        if best_key is None or key < best_key:
            best_key = key
            best_a2 = int(a2)

    # Find best horizontal axis (b2 = 2*axis), minimizing mismatches on known cells.
    best_b2 = 0
    best_key = None
    rows = np.where(nz.any(axis=1))[0]
    for b2 in range(0, 2 * (H - 1) + 1):
        mism = 0
        pairs = 0
        for r in rows:
            r2 = int(b2) - int(r)
            if not (0 <= r2 < H):
                continue
            cols = np.where(nz[int(r)] & nz[int(r2)])[0]
            pairs += int(len(cols))
            if len(cols):
                mism += int(np.sum(g[int(r), cols] != g[int(r2), cols]))
        key = (int(mism), -int(pairs))
        if best_key is None or key < best_key:
            best_key = key
            best_b2 = int(b2)

    out = g.copy()

    def fill_once(cur: np.ndarray) -> bool:
        changed = False
        zeros = np.argwhere(cur == int(miss))
        for r, c in zeros:
            r = int(r)
            c = int(c)
            cand: list[int] = []

            c2 = int(best_a2) - int(c)
            if 0 <= c2 < W and int(cur[int(r), int(c2)]) != int(miss):
                cand.append(int(cur[int(r), int(c2)]))

            r2 = int(best_b2) - int(r)
            if 0 <= r2 < H and int(cur[int(r2), int(c)]) != int(miss):
                cand.append(int(cur[int(r2), int(c)]))

            if 0 <= r2 < H and 0 <= c2 < W and int(cur[int(r2), int(c2)]) != int(miss):
                cand.append(int(cur[int(r2), int(c2)]))

            if cand:
                cur[int(r), int(c)] = int(Counter(cand).most_common(1)[0][0])
                changed = True
        return bool(changed)

    # Usually one pass suffices; do a few in case missing regions touch their mirrors.
    for _ in range(5):
        if not fill_once(out):
            break

    return out


def generate_input() -> np.ndarray:
    miss = 0
    H = int(np.random.randint(10, 18))
    W = int(np.random.randint(10, 18))

    # Symmetry axes through the center: a2 = W-1, b2 = H-1.
    a2 = int(W - 1)
    b2 = int(H - 1)

    pal = list(range(1, 10))
    g = np.full((H, W), int(miss), dtype=int)

    # Fill by orbit representatives under (r,c)->(r,b2-r) and (r,c)->(r,a2-c).
    seen = set()
    for r in range(H):
        for c in range(W):
            if (int(r), int(c)) in seen:
                continue
            orbit = {
                (int(r), int(c)),
                (int(r), int(a2 - c)),
                (int(b2 - r), int(c)),
                (int(b2 - r), int(a2 - c)),
            }
            orbit = {(rr, cc) for (rr, cc) in orbit if 0 <= rr < H and 0 <= cc < W}
            col = int(pal[int(np.random.randint(0, len(pal)))])
            for rr, cc in orbit:
                g[int(rr), int(cc)] = int(col)
                seen.add((int(rr), int(cc)))

    # Punch missing holes.
    n_holes = int(np.random.randint(max(1, (H * W) // 10), max(2, (H * W) // 4)))
    idx = np.random.choice(H * W, size=n_holes, replace=False)
    for k in idx:
        r = int(k // W)
        c = int(k % W)
        g[int(r), int(c)] = int(miss)

    # Ensure at least one non-missing pixel remains.
    if not bool((g != int(miss)).any()):
        g[int(H // 2), int(W // 2)] = 1
    return g

