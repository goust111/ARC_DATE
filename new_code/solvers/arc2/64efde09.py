from __future__ import annotations
from collections import Counter
import numpy as np
from arc_utils import STRUCT8, get_bg, label_mask

def main(grid: np.ndarray) -> np.ndarray:
    """
    Emit colored rays from bar-shaped emitters at positions dictated by their stripe pattern.

    Input structure:
    - bg = most common color.
    - 2xW or Hx2 bar emitters, each with a uniform "spine" row and a patterned "stripe" row.
    - Singleton pixels forming a 3-color palette.

    Rule:
    - Each bar emits 3 rays outward from its spine side, perpendicular to its long axis.
    - Port positions along the bar are derived from the stripe row's run-length structure.
    - Ray colors are the sorted palette triple, reordered by stripe geometry and ray direction.
    - Rays stop at the grid boundary or one cell before another emitter.

    Key insight: The stripe row's run-length signature encodes both the port offsets and the palette ordering.
    """
    g = np.array(grid, dtype=int)
    H, W = g.shape
    bg = int(get_bg(g))

    # STEP 1) Split foreground into (a) emitters (size>1 CCs) and (b) singleton palette pixels.
    lab, n = label_mask(g != int(bg), connectivity=8)
    emitter_mask = np.zeros((H, W), dtype=bool)
    palette: list[int] = []
    emitters: list[dict] = []
    for k in range(1, int(n) + 1):
        coords = np.argwhere(lab == int(k))
        if coords.size == 0:
            continue
        if int(coords.shape[0]) == 1:
            r, c = map(int, coords[0])
            palette.append(int(g[int(r), int(c)]))
            continue
        for r, c in coords:
            emitter_mask[int(r), int(c)] = True
        r0, c0 = coords.min(axis=0)
        r1, c1 = coords.max(axis=0)
        r0, c0, r1, c1 = map(int, (r0, c0, r1, c1))
        emitters.append({"r0": int(r0), "c0": int(c0), "r1": int(r1), "c1": int(c1)})

    assert not (not emitters or not palette)

    palette_sorted = sorted(set(map(int, palette)))
    if len(palette_sorted) < 3:
        palette_sorted = (palette_sorted + palette_sorted + palette_sorted)[:3]
    else:
        palette_sorted = palette_sorted[:3]

    out = g.copy()

    def _emit_ray(outv: np.ndarray, emaskv: np.ndarray, r: int, c: int, dr: int, dc: int, col: int) -> None:
        """Paint from (r,c) along (dr,dc) until boundary or emitter hit (in a given view)."""
        Hh, Ww = outv.shape
        rr, cc = int(r), int(c)
        while 0 <= rr < Hh and 0 <= cc < Ww:
            if bool(emaskv[int(rr), int(cc)]):
                break
            outv[int(rr), int(cc)] = int(col)
            rr += int(dr)
            cc += int(dc)

    def _dominance(line: np.ndarray) -> int:
        vals = set(map(int, line.tolist()))
        return max(int((line == v).sum()) for v in vals)

    def _runs(line: np.ndarray) -> list[tuple[int, int, int]]:
        """Run-length encoding: list of (value, start_idx, length)."""
        vals = list(map(int, line.tolist()))
        out_runs: list[tuple[int, int, int]] = []
        i = 0
        while i < len(vals):
            j = i + 1
            while j < len(vals) and int(vals[j]) == int(vals[i]):
                j += 1
            out_runs.append((int(vals[i]), int(i), int(j - i)))
            i = j
        return out_runs

    def _stripe_signature(stripe_row: np.ndarray) -> tuple[int, int, int, bool, int]:
        """
        Return (left_len, body_len, right_len, ends_match, body_start).
        Assumes stripe row is 3-run in this task family.
        """
        rs = _runs(stripe_row)
        left_len = int(rs[0][2])
        right_len = int(rs[-1][2])
        _v, body_start, body_len = max(rs, key=lambda t: (int(t[2]), -int(t[1])))
        ends_match = int(stripe_row[0]) == int(stripe_row[-1])
        return int(left_len), int(body_len), int(right_len), bool(ends_match), int(body_start)

    def _ports_from_stripe(stripe_row: np.ndarray) -> list[int]:
        """
        Uniform port rule: choose 3 port offsets from stripe run lengths.
        """
        L = int(stripe_row.shape[0])
        _left_len, body_len, right_len, ends_match, body_start = _stripe_signature(stripe_row)

        # Two stripe subtypes appear in this puzzle family:
        # - right_len == 2: “horizontal-type” stripe in the canonical 2×L bar
        # - right_len == 1: “vertical-type” stripe in the canonical 2×L bar
        if int(right_len) == 2:
            p3 = int(L - right_len)
            p1 = int(p3 - 2 * right_len)
            shift = 1 if (ends_match and int(body_len) == 2 * int(right_len)) else 0
            p2 = int(p3 - right_len - shift)
            return [int(p1), int(p2), int(p3)]

        # right_len == 1 in this family
        p1 = int(body_start - 1)
        p3 = int(body_start + 3)
        p2 = int(body_start + (2 if (ends_match and int(body_len) == 4) else 1))
        return [int(p1), int(p2), int(p3)]

    def _colors_from_stripe(stripe_row: np.ndarray, dr: int) -> list[int]:
        """
        Uniform palette order rule derived from stripe run lengths.
        Palette is treated as an ordered triple (ascending by value).
        """
        p0, p1, p2 = map(int, palette_sorted)  # ascending
        _left_len, body_len, right_len, ends_match, _body_start = _stripe_signature(stripe_row)

        if int(right_len) == 2:
            if int(dr) == -1:
                return [int(p2), int(p1), int(p0)]
            return [int(p1), int(p0), int(p2)] if (not ends_match) else [int(p0), int(p1), int(p2)]

        if int(body_len) >= 6:
            return [int(p0), int(p1), int(p2)]
        if ends_match:
            return [int(p2), int(p1), int(p0)]
        return [int(p2), int(p0), int(p1)]

    def _emit_bar(
        gridv: np.ndarray,
        outv: np.ndarray,
        emaskv: np.ndarray,
        bbox: tuple[int, int, int, int],
    ) -> None:
        """Emit 3 rays from a horizontal 2xL bar in the provided world-view."""
        r0, c0, r1, c1 = map(int, bbox)
        h = int(r1 - r0 + 1)
        w = int(c1 - c0 + 1)
        if int(h) != 2 or int(w) < 5:
            return

        row0 = gridv[int(r0), int(c0) : int(c1) + 1]
        row1 = gridv[int(r1), int(c0) : int(c1) + 1]
        dom0 = _dominance(row0)
        dom1 = _dominance(row1)
        spine_row = 0 if int(dom0) >= int(dom1) else 1
        dr = -1 if spine_row == 0 else 1  # spine side emits outward

        stripe_row = row1 if spine_row == 0 else row0  # non-spine row
        offs = _ports_from_stripe(stripe_row)
        port_cols = [int(c0) + int(o) for o in offs]
        cols = _colors_from_stripe(stripe_row, int(dr))

        start_r = int(r1 + 1) if int(dr) == 1 else int(r0 - 1)
        for pc, col in zip(port_cols, cols):
            if 0 <= int(start_r) < int(outv.shape[0]) and 0 <= int(pc) < int(outv.shape[1]):
                _emit_ray(outv, emaskv, int(start_r), int(pc), int(dr), 0, int(col))

    # STEP 2) For each emitter (in original CC order): emit rays.
    # To reuse the same "horizontal 2xL bar" logic for vertical emitters, we run the algorithm
    # on a transposed world-view (swap r/c) via `.T`.
    for e in emitters:
        r0, c0, r1, c1 = map(int, (e["r0"], e["c0"], e["r1"], e["c1"]))
        h = int(r1 - r0 + 1)
        w = int(c1 - c0 + 1)
        if int(h) == 2 and int(w) >= 5:
            _emit_bar(g, out, emitter_mask, (int(r0), int(c0), int(r1), int(c1)))
        elif int(w) == 2 and int(h) >= 5:
            _emit_bar(g.T, out.T, emitter_mask.T, (int(c0), int(r0), int(c1), int(r1)))

    return out



