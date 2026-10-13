from __future__ import annotations
from collections import Counter
import numpy as np
from arc_utils import Color, components_mask, get_bg, label_mask

def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor a path-network by adjacent seed regions.

    Input structure:
    - bg = most common color.
    - Path color P forms a thin orthogonal network with a solid square hub (NxN).
    - Seed colors appear in colored regions (non-bg, non-P) adjacent to path branches.

    Rule:
    - Infer P as the non-bg color whose largest component is most network-like.
    - Find the largest solid k×k all-P square (the hub); exclude it from recoloring.
    - Recolor each path branch to the seed-region color it touches most.
    - Recolor the hub center pixel to the seed whose branches touch the hub most.

    Key insight: The hub stays uncolored; branches inherit color from adjacent seed regions by adjacency count.
    """
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))
    H, W = g.shape

    colors = sorted({int(x) for x in np.unique(g)} - {int(bg)})
    assert not (not colors)

    def comp_deg_score(mask: np.ndarray) -> float:
        comps = components_mask(mask, connectivity=4)
        if not comps:
            return -1e9
        best_pts = max(comps, key=lambda coords: int(coords.shape[0]))
        best_sz = int(best_pts.shape[0])

        m = np.zeros(mask.shape, dtype=bool)
        m[best_pts[:, 0], best_pts[:, 1]] = True

        deg2 = 0
        deg4 = 0
        for r, c in best_pts:
            r = int(r)
            c = int(c)
            d = 0
            if r > 0 and bool(m[r - 1, c]):
                d += 1
            if r + 1 < H and bool(m[r + 1, c]):
                d += 1
            if c > 0 and bool(m[r, c - 1]):
                d += 1
            if c + 1 < W and bool(m[r, c + 1]):
                d += 1
            if d == 2:
                deg2 += 1
            elif d == 4:
                deg4 += 1
        frac2 = deg2 / max(1, best_sz)
        frac4 = deg4 / max(1, best_sz)
        return float((frac2 - frac4) * best_sz)

    P = int(max(colors, key=lambda c: comp_deg_score(g == int(c))))

    # --- hub: largest solid k×k all-P square anywhere ---
    mP = g == int(P)
    ps = mP.astype(int).cumsum(axis=0).cumsum(axis=1)

    def rect_sum(r0: int, c0: int, r1: int, c1: int) -> int:
        s = int(ps[int(r1), int(c1)])
        if r0 > 0:
            s -= int(ps[int(r0 - 1), int(c1)])
        if c0 > 0:
            s -= int(ps[int(r1), int(c0 - 1)])
        if r0 > 0 and c0 > 0:
            s += int(ps[int(r0 - 1), int(c0 - 1)])
        return int(s)

    best = None  # (k, r0, c0)
    max_k = int(min(H, W))
    for k in range(2, max_k + 1):
        found = None
        for r0 in range(0, int(H - k + 1)):
            r1 = int(r0 + k - 1)
            for c0 in range(0, int(W - k + 1)):
                c1 = int(c0 + k - 1)
                if rect_sum(int(r0), int(c0), int(r1), int(c1)) == int(k * k):
                    found = (int(k), int(r0), int(c0))
        if found is not None:
            best = found

    assert not (best is None)

    k, r0, c0 = map(int, best)
    r1, c1 = int(r0 + k - 1), int(c0 + k - 1)
    hub = np.zeros((H, W), dtype=bool)
    hub[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1] = True

    # --- seed regions (non-bg, non-P) ---
    seed_mask = (g != int(bg)) & (g != int(P))
    seed_lab, seed_n = label_mask(seed_mask, connectivity=4)
    seed_major: dict[int, int] = {}
    all_seed_vals: list[int] = []
    for sid in range(1, int(seed_n) + 1):
        pts = np.argwhere(seed_lab == int(sid))
        vals = [int(g[int(r), int(c)]) for r, c in pts]
        maj = int(Counter(vals).most_common(1)[0][0])
        seed_major[int(sid)] = int(maj)
        all_seed_vals.extend(vals)

    out = g.copy()

    # --- recolor each path branch by the seed region it touches most ---
    branch_mask = (g == int(P)) & (~hub)
    branch_lab, branch_n = label_mask(branch_mask, connectivity=4)

    for bid in range(1, int(branch_n) + 1):
        pts = np.argwhere(branch_lab == int(bid))
        touches = Counter()
        for r, c in pts:
            r = int(r)
            c = int(c)
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                rr, cc = int(r + dr), int(c + dc)
                if 0 <= rr < H and 0 <= cc < W:
                    sid = int(seed_lab[rr, cc])
                    if sid != 0:
                        touches[int(sid)] += 1
        if not touches:
            continue
        sid = int(touches.most_common(1)[0][0])
        col = int(seed_major[int(sid)])
        for r, c in pts:
            out[int(r), int(c)] = int(col)

    # --- hub center pixel recolor ---
    touch_cols = Counter()
    if int(branch_n) > 0:
        for r in range(H):
            for c in range(W):
                if not bool(hub[int(r), int(c)]):
                    continue
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    rr, cc = int(r + dr), int(c + dc)
                    if 0 <= rr < H and 0 <= cc < W:
                        if int(branch_lab[rr, cc]) != 0:
                            touch_cols[int(out[rr, cc])] += 1

    cr = int(r0 + k // 2)
    cc = int(c0 + k // 2)

    if touch_cols:
        center_col = int(touch_cols.most_common(1)[0][0])
        out[int(cr), int(cc)] = int(center_col)
    elif all_seed_vals:
        center_col = int(Counter(all_seed_vals).most_common(1)[0][0])
        out[int(cr), int(cc)] = int(center_col)
    else:
        out[int(cr), int(cc)] = int(Color.BLACK)

    return out



