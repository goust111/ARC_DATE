from __future__ import annotations
import numpy as np
from arc_utils import components_mask, get_bg

def main(grid: np.ndarray) -> np.ndarray:
    """
    Remove diagonal staircase noise and complete broken straight lines.

    Input structure:
    - bg = background; one foreground color fg forming axis-aligned lines.
    - Some fg pixels form 45° diagonal chains (8-connected but 4-disconnected from lines).

    Rule:
    - Delete all diagonal-only fg pixels (those with no orthogonal fg neighbor, only diagonal ones).
    - Fill short gaps in horizontal/vertical lines where a deleted diagonal was nearby.
    - Prune remaining diagonal-adjacent leaf pixels near former diagonal sites.

    Key insight: Diagonal pixels are hints marking where a straight line is broken; their span sets the maximum gap size to fill.
    """
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))

    colors = {int(x) for x in np.unique(g)}
    colors.discard(int(bg))
    assert not (not colors)
    fg = int(next(iter(colors)))

    mask = g == int(fg)
    H, W = g.shape

    def deg4(m: np.ndarray, r: int, c: int) -> int:
        d = 0
        if r > 0 and bool(m[int(r - 1), int(c)]):
            d += 1
        if r + 1 < H and bool(m[int(r + 1), int(c)]):
            d += 1
        if c > 0 and bool(m[int(r), int(c - 1)]):
            d += 1
        if c + 1 < W and bool(m[int(r), int(c + 1)]):
            d += 1
        return int(d)

    def has_diag(m: np.ndarray, r: int, c: int) -> bool:
        if r > 0 and c > 0 and bool(m[int(r - 1), int(c - 1)]):
            return True
        if r > 0 and c + 1 < W and bool(m[int(r - 1), int(c + 1)]):
            return True
        if r + 1 < H and c > 0 and bool(m[int(r + 1), int(c - 1)]):
            return True
        if r + 1 < H and c + 1 < W and bool(m[int(r + 1), int(c + 1)]):
            return True
        return False

    # --- diagonal-only pixels (to remove) ---
    diag = np.zeros((H, W), dtype=bool)
    for r in range(H):
        for c in range(W):
            if bool(mask[int(r), int(c)]) and deg4(mask, int(r), int(c)) == 0 and has_diag(mask, int(r), int(c)):
                diag[int(r), int(c)] = True

    # Neighborhood of diagonal hints (Chebyshev radius 1).
    near = np.zeros((H, W), dtype=bool)
    for r in range(H):
        for c in range(W):
            if not bool(diag[int(r), int(c)]):
                continue
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    rr = int(r + dr)
                    cc = int(c + dc)
                    if 0 <= rr < H and 0 <= cc < W:
                        near[rr, cc] = True

    out = g.copy()
    out[diag] = int(bg)

    # Gap threshold: small default, optionally increased by diagonal-component span.
    MAXG = 3
    if bool(diag.any()):
        for coords in components_mask(diag, connectivity=8):
            if coords.size == 0:
                continue
            r0, c0 = coords.min(axis=0)
            r1, c1 = coords.max(axis=0)
            MAXG = max(int(MAXG), int(r1 - r0 + 1), int(c1 - c0 + 1))

    # Fill short horizontal gaps between segments when a diagonal hint is nearby.
    for r in range(H):
        cols = np.where(mask[int(r), :] & (~diag[int(r), :]))[0]
        if cols.size == 0:
            continue
        segs: list[tuple[int, int]] = []
        start = int(cols[0])
        prev = int(cols[0])
        for c in cols[1:]:
            c = int(c)
            if c == prev + 1:
                prev = c
            else:
                segs.append((int(start), int(prev)))
                start = prev = c
        segs.append((int(start), int(prev)))

        for (a1, b1), (a2, _b2) in zip(segs, segs[1:]):
            gap0 = int(b1 + 1)
            gap1 = int(a2 - 1)
            if gap1 < gap0:
                continue
            if int(gap1 - gap0 + 1) > int(MAXG):
                continue
            rr0 = max(0, int(r - 1))
            rr1 = min(int(H - 1), int(r + 1))
            cc0 = max(0, int(gap0 - 1))
            cc1 = min(int(W - 1), int(gap1 + 1))
            if bool(diag[rr0 : rr1 + 1, cc0 : cc1 + 1].any()):
                out[int(r), int(gap0) : int(gap1) + 1] = int(fg)

    # Fill short vertical gaps between segments when a diagonal hint is nearby.
    for c in range(W):
        rows = np.where(mask[:, int(c)] & (~diag[:, int(c)]))[0]
        if rows.size == 0:
            continue
        segs: list[tuple[int, int]] = []
        start = int(rows[0])
        prev = int(rows[0])
        for r in rows[1:]:
            r = int(r)
            if r == prev + 1:
                prev = r
            else:
                segs.append((int(start), int(prev)))
                start = prev = r
        segs.append((int(start), int(prev)))

        for (a1, b1), (a2, _b2) in zip(segs, segs[1:]):
            gap0 = int(b1 + 1)
            gap1 = int(a2 - 1)
            if gap1 < gap0:
                continue
            if int(gap1 - gap0 + 1) > int(MAXG):
                continue
            rr0 = max(0, int(gap0 - 1))
            rr1 = min(int(H - 1), int(gap1 + 1))
            cc0 = max(0, int(c - 1))
            cc1 = min(int(W - 1), int(c + 1))
            if bool(diag[rr0 : rr1 + 1, cc0 : cc1 + 1].any()):
                out[int(gap0) : int(gap1) + 1, int(c)] = int(fg)

    # Local pruning: remove diagonal-adjacent leaf pixels only near diagonal hints.
    m = out == int(fg)
    changed = True
    while changed:
        changed = False
        to_clear: list[tuple[int, int]] = []
        for r in range(H):
            for c in range(W):
                if not bool(m[int(r), int(c)]) or not bool(near[int(r), int(c)]):
                    continue
                if deg4(m, int(r), int(c)) != 1:
                    continue
                if not has_diag(m, int(r), int(c)):
                    continue
                to_clear.append((int(r), int(c)))
        if to_clear:
            changed = True
            for r, c in to_clear:
                m[int(r), int(c)] = False

    out2 = np.full((H, W), int(bg), dtype=int)
    out2[m] = int(fg)
    return out2



