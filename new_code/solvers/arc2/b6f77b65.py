from __future__ import annotations
from collections import Counter, defaultdict, deque
import numpy as np
from scipy.ndimage import label
from arc_utils import STRUCT4, STRUCT8, get_bg

def main(grid: np.ndarray) -> np.ndarray:
    """
    Simulate bar gravity after erasing colors flagged by top-row control pixels.

    Input structure:
    - bg = most common color.
    - Top-row non-bg pixels are fixed control pixels; their colors specify which pixels to erase elsewhere.
    - Remaining non-bg pixels form vertical and horizontal bar segments.

    Rule:
    - Erase all pixels matching any control color (except the controls themselves).
    - Classify remaining pixels: vertical bars (same-color vertical adjacency) vs horizontal bars (the rest).
    - Repeatedly drop the lowest movable bar: vertical bars fall unless blocked or side-supported by a stable horizontal bar; horizontal bars fall until both ends touch vertical bars.

    Key insight: Horizontal bars stabilize only when both endpoints abut vertical bars, creating mutual-support dynamics.
    """
    inp = grid
    g = np.array(inp, dtype=int).copy()
    H, W = g.shape
    bg = int(Counter(map(int, g.flatten())).most_common(1)[0][0])

    # control pixels: all non-bg on top row
    ctrl_pos = [(0, int(c)) for c in range(W) if int(g[0, c]) != int(bg)]
    ctrl_cols = set(int(g[r, c]) for r, c in ctrl_pos)

    fixed = np.zeros_like(g, dtype=bool)
    for r, c in ctrl_pos:
        fixed[int(r), int(c)] = True

    # delete all control colors from the structure (but keep the control pixels)
    for cc in ctrl_cols:
        g[(g == int(cc)) & (~fixed)] = int(bg)

    def vmask(cur):
        """Vertical-bar cells: same-color vertical adjacency (junctions included)."""
        v = np.zeros_like(cur, dtype=bool)
        v[1:, :] |= (cur[1:, :] != int(bg)) & (~fixed[1:, :]) & (cur[1:, :] == cur[:-1, :])
        v[:-1, :] |= (cur[:-1, :] != int(bg)) & (~fixed[:-1, :]) & (cur[:-1, :] == cur[1:, :])
        v &= ~fixed
        return v

    def vertical_segments(cur, v):
        """List vertical segments as (r0,r1,c,col)."""
        segs = []
        for c in range(W):
            r = 0
            while r < H:
                if not bool(v[r, c]):
                    r += 1
                    continue
                col = int(cur[r, c])
                r0 = int(r)
                r += 1
                while r < H and bool(v[r, c]) and int(cur[r, c]) == int(col):
                    r += 1
                r1 = int(r - 1)
                segs.append((r0, r1, int(c), int(col)))
        return segs

    def horizontal_segments(cur, v):
        """List horizontal segments and a cell->segment-id map."""
        hm = (cur != int(bg)) & (~v) & (~fixed)
        segs = []
        sid = np.full(cur.shape, -1, dtype=int)
        idx = 0
        for r in range(H):
            c = 0
            while c < W:
                if not bool(hm[r, c]):
                    c += 1
                    continue
                col = int(cur[r, c])
                c0 = int(c)
                c += 1
                while c < W and bool(hm[r, c]) and int(cur[r, c]) == int(col):
                    c += 1
                c1 = int(c - 1)
                segs.append((int(r), c0, c1, int(col)))
                for cc in range(c0, c1 + 1):
                    sid[int(r), int(cc)] = int(idx)
                idx += 1
        return segs, sid

    def horiz_stable(cur, v, seg):
        """Horizontal bar stable if both ends touch vertical-bar cells (colors may differ)."""
        r, c0, c1, _col = seg
        if c0 - 1 < 0 or c1 + 1 >= W:
            return False
        return (int(cur[r, c0 - 1]) != int(bg) and bool(v[r, c0 - 1])) and (
            int(cur[r, c1 + 1]) != int(bg) and bool(v[r, c1 + 1])
        )

    def can_fall_h(cur, seg):
        r, c0, c1, _col = seg
        if r == H - 1:
            return False
        for c in range(c0, c1 + 1):
            if bool(fixed[r + 1, c]) or int(cur[r + 1, c]) != int(bg):
                return False
        return True

    def vertical_supported(cur, v, hsegs, hsid, stable_h, seg):
        """Vertical is supported if below is occupied OR adjacent stable same-color horizontal at bottom."""
        r0, r1, c, col = seg
        if r1 == H - 1:
            return True
        if bool(fixed[r1 + 1, c]) or int(cur[r1 + 1, c]) != int(bg):
            return True
        rr = int(r1)
        for dc in (-1, 1):
            cc = int(c + dc)
            if 0 <= cc < W:
                sid = int(hsid[rr, cc])
                if sid != -1 and bool(stable_h[sid]) and int(cur[rr, cc]) == int(col):
                    return True
        return False

    def drop_dist_v(cur, v, hsegs, hsid, stable_h, seg):
        """How far a vertical bar will fall, possibly stopping when it becomes side-supported."""
        r0, r1, c, col = seg
        if vertical_supported(cur, v, hsegs, hsid, stable_h, seg):
            return 0
        d = 0
        rr0, rr1 = int(r0), int(r1)
        while True:
            if rr1 == H - 1:
                break
            if bool(fixed[rr1 + 1, c]) or int(cur[rr1 + 1, c]) != int(bg):
                break
            new_r1 = int(rr1 + 1)
            side_support = False
            for dc in (-1, 1):
                cc = int(c + dc)
                if 0 <= cc < W:
                    sid = int(hsid[new_r1, cc])
                    if sid != -1 and bool(stable_h[sid]) and int(cur[new_r1, cc]) == int(col):
                        side_support = True
                        break
            rr0 += 1
            rr1 += 1
            d += 1
            if side_support:
                break
        return int(d)

    # main loop: repeatedly move the lowest movable bar
    for _steps in range(10000):
        v = vmask(g)
        hsegs, hsid = horizontal_segments(g, v)
        stable_h = [bool(horiz_stable(g, v, s)) for s in hsegs]
        vsegs = vertical_segments(g, v)

        cand = []  # (y, typ, seg, d)
        for seg in vsegs:
            r0, r1, c, col = seg
            d = drop_dist_v(g, v, hsegs, hsid, stable_h, seg)
            if d > 0:
                cand.append((int(r1), "V", seg, int(d)))

        for seg in hsegs:
            r, c0, c1, col = seg
            if bool(horiz_stable(g, v, seg)):
                continue
            if can_fall_h(g, seg):
                cand.append((int(r), "H", seg, 1))

        if not cand:
            break

        # lowest-first: larger row index wins; tie -> vertical before horizontal; then leftmost
        cand.sort(
            key=lambda t: (
                -t[0],
                0 if t[1] == "V" else 1,
                t[2][2] if t[1] == "V" else t[2][1],
            )
        )
        _y, typ, seg, d = cand[0]

        if typ == "V":
            r0, r1, c, col = seg
            for r in range(r0, r1 + 1):
                g[int(r), int(c)] = int(bg)
            for r in range(r0 + d, r1 + d + 1):
                g[int(r), int(c)] = int(col)
        else:
            r, c0, c1, col = seg
            rr = int(r)
            while True:
                v = vmask(g)
                if bool(horiz_stable(g, v, (int(rr), int(c0), int(c1), int(col)))):
                    break
                if rr == H - 1:
                    break
                if any(
                    bool(fixed[rr + 1, int(c)]) or int(g[rr + 1, int(c)]) != int(bg)
                    for c in range(c0, c1 + 1)
                ):
                    break
                for c in range(c0, c1 + 1):
                    g[int(rr), int(c)] = int(bg)
                rr += 1
                for c in range(c0, c1 + 1):
                    g[int(rr), int(c)] = int(col)

    # restore control pixels
    for r, c in ctrl_pos:
        g[int(r), int(c)] = int(inp[int(r)][int(c)])

    return g


