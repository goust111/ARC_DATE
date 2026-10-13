from __future__ import annotations
from collections import Counter
import numpy as np
from arc_utils import get_bg

def main(grid: np.ndarray) -> np.ndarray:
    """
    Composite two pattern tiles into a fourth panel using a half-plane rule from cue pixels.

    Input structure:
    - Four 5×5 panels in one row; border-separated.
    - Panel 0: minority-color cues. Panels 1–2: patterns. Panel 3: blank.

    Rule:
    - Fit a line (row/col/diag/anti-diag) through panel 0's cue pixels.
    - Farthest off-line cue pixel picks the "inside" half-plane.
    - Inside → non-bg pixels from panel 1; outside → from panel 2.
    - For axis-aligned lines, clamp to cue span; boundary from panel 2.

    Key insight: Cue pixels encode both the dividing line and which half maps to which source tile.
    """
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))

    # Extract the 4 tiles.
    tiles = [g[1:6, 1 + 6 * j : 1 + 6 * j + 5].copy() for j in range(4)]
    t0, t1, t2, _t3 = tiles
    b = int(Counter(t1.flatten()).most_common(1)[0][0])

    cue_mask = t0 != int(Counter(t0.flatten()).most_common(1)[0][0])
    pts = [(int(r), int(c)) for r, c in np.argwhere(cue_mask)]
    assert not (not pts)

    # --- line detection (row/col/diag/anti by max count) ---
    best_kind, best_val, best_cnt = "row", 0, -1
    for r in range(5):
        cnt = sum(1 for rr, _cc in pts if int(rr) == int(r))
        if cnt > best_cnt:
            best_kind, best_val, best_cnt = "row", int(r), int(cnt)
    for c in range(5):
        cnt = sum(1 for _rr, cc in pts if int(cc) == int(c))
        if cnt > best_cnt:
            best_kind, best_val, best_cnt = "col", int(c), int(cnt)
    for d in range(-4, 5):
        cnt = sum(1 for rr, cc in pts if int(rr - cc) == int(d))
        if cnt > best_cnt:
            best_kind, best_val, best_cnt = "diag", int(d), int(cnt)
    for s in range(0, 9):
        cnt = sum(1 for rr, cc in pts if int(rr + cc) == int(s))
        if cnt > best_cnt:
            best_kind, best_val, best_cnt = "anti", int(s), int(cnt)

    def side_value(p: tuple[int, int]) -> int:
        r, c = int(p[0]), int(p[1])
        if best_kind == "row":
            return int(r - best_val)
        if best_kind == "col":
            return int(c - best_val)
        if best_kind == "diag":
            return int((r - c) - best_val)
        return int((r + c) - best_val)  # anti

    # Choose which side to keep using the farthest off-line point.
    offs = [int(side_value(p)) for p in pts if int(side_value(p)) != 0]
    if offs:
        v = max(offs, key=lambda x: abs(int(x)))
        sign = 1 if int(v) < 0 else -1
    else:
        sign = 1

    R = {(r, c) for r in range(5) for c in range(5) if int(sign * side_value((r, c))) <= 0}

    # For axis-aligned boundaries, restrict to the cue span in the perpendicular axis.
    rs = [int(r) for r, _c in pts]
    cs = [int(c) for _r, c in pts]
    if best_kind == "col":
        rlo, rhi = int(min(rs)), int(max(rs))
        R = {(r, c) for (r, c) in R if int(rlo) <= int(r) <= int(rhi)}
    if best_kind == "row":
        clo, chi = int(min(cs)), int(max(cs))
        R = {(r, c) for (r, c) in R if int(clo) <= int(c) <= int(chi)}

    # --- build tile4 ---
    out3 = np.full((5, 5), int(b), dtype=int)
    for r in range(5):
        for c in range(5):
            inR = (int(r), int(c)) in R
            on_line = int(side_value((r, c))) == 0
            if inR:
                if int(t1[int(r), int(c)]) != int(b):
                    out3[int(r), int(c)] = int(t1[int(r), int(c)])
                elif best_kind in ("row", "col") and on_line and int(t2[int(r), int(c)]) != int(b):
                    out3[int(r), int(c)] = int(t2[int(r), int(c)])
            else:
                if int(t2[int(r), int(c)]) != int(b):
                    out3[int(r), int(c)] = int(t2[int(r), int(c)])

    out = g.copy()
    out[1:6, 1 + 6 * 3 : 1 + 6 * 3 + 5] = out3
    return out

