# Combo variant of a6f40cea (index 18)
# combo_spec: D9/1+D16/2
# Name: 厚框比例投影
#
# Design analysis
# ---------------
# a) Concretization and trace
#    - The closed frame is now a 2px-thick rectangular annulus. If its outer bbox is
#      rows r0..r1 and cols c0..c1, then the true output crop starts at r0+2/c0+2.
#    - The second change is not just "subtract 2 instead of 1". Ring-touch coordinates
#      along a frame side are read in the outer frame coordinate system and projected
#      proportionally onto the smaller interior side length.
#    - Concrete width trace with interior W=7 and thick outer span outerW=11:
#      interior cols are 0..6, while top ring touch offsets along the frame side are 0..10.
#      A touch at outer offset 3 used to map by direct offset to 1 (=3-2), but under the fused
#      rule it maps to round(3 * 6 / 10) = 2. That one-cell disagreement is the intended signal.
#      The generator therefore picks ring touches via the inverse of this scaling so the produced
#      output really depends on proportional projection, not merely on thick-frame cropping.
# b) Complexity calibration
#    - Keep the source task's main freedoms: 2-3 outside components, optional U-pair, monochrome
#      or two-color alternating traces, optional distractors.
#    - The only semantic additions are thick-frame detection and sidewise scale projection. This
#      is enough to make the variant visually distinct without exploding the case split seen by a
#      human solver.
# c) Implicit assumptions audit
#    - `_find_framed_rect_uniform_fill()` can no longer rely on 1px perimeter cardinality; it
#      must verify an exact 2px annulus and a uniform inner rectangle.
#    - The old solver assumed every ring touch corresponded to "same coordinate minus border
#      offset". That assumption breaks once the outer side span is larger than the inner side span.
#      Mapping is now side-aware: top/bottom touches scale columns, left/right touches scale rows.
#    - The old generator placed ring touches at direct positions `t + inner_idx`. That would let a
#      thick-frame-only solver succeed on many cases. The new generator instead chooses inverse
#      scaled ring offsets and prefers offsets different from the old direct ones.
# d) Placement legality
#    - The frame bbox is sampled from interior size + 2*thickness, so all four border bands fit.
#    - Outside open-rect paths are still drawn only on background cells and are prevented from
#      overlapping or 4-touching each other by `_place_component()`.
#    - Inverse-projected ring offsets are restricted away from the outer corners, which avoids
#      ambiguous side assignment in the solver and keeps the open rectangles visually clean.
# e) Visual observability
#    - Across examples the input always shows a visibly thick closed frame, while output remains a
#      crop of the inner hole only.
#    - The extra twist is observable because corresponding touches land at compressed positions:
#      a top touch that looks "slightly inward" on the thick border can map farther inward inside.
#      Multiple examples with different frame sizes expose that this is scaling, not a fixed shift.
# f) Visual cleanliness
#    - Frame, ring-touch components, and distractors remain separated. The side-aware inverse
#      projection keeps touches on single clear side cells instead of corners, so there is no
#      visual ambiguity about which frame side generated which output endpoint.
# g) Inferability of change
#    - No new color semantics are introduced. Colors still only encode component identity and
#      alternating-path phase. The new reasoning burden is geometric: detect 2px thickness and
#      infer that side coordinates are scaled from the outer frame span to the inner span.

from __future__ import annotations

from collections import Counter, defaultdict
import random

import numpy as np
from scipy.ndimage import label

from arc_utils import STRUCT4, get_bg, neighbors4


_FRAME_THICKNESS = 2


def _scale_axis(offset: int, outer_len: int, inner_len: int) -> int:
    """Project an outer-side offset onto an inner-side offset by nearest scaled position."""
    if inner_len <= 1 or outer_len <= 1:
        return 0
    val = int(round(int(offset) * (int(inner_len) - 1) / (int(outer_len) - 1)))
    return max(0, min(int(inner_len) - 1, int(val)))


def _inverse_scaled_offsets(
    inner_idx: int,
    outer_len: int,
    inner_len: int,
    *,
    avoid_corners: bool = False,
) -> list[int]:
    """All outer offsets that scale to `inner_idx`."""
    out = []
    for off in range(int(outer_len)):
        if _scale_axis(int(off), int(outer_len), int(inner_len)) != int(inner_idx):
            continue
        if avoid_corners and (off == 0 or off == int(outer_len) - 1):
            continue
        out.append(int(off))
    return out


def _find_framed_rect_uniform_fill(
    g: np.ndarray,
    *,
    bg: int | None = None,
    structure=STRUCT4,
    thickness: int = _FRAME_THICKNESS,
) -> tuple[int, int, int, int, int, int] | None:
    """
    Detect the main 2px-thick rectangular frame with uniform interior fill.

    Returns (r0, c0, r1, c1, border_color, fill_color) for the largest such rectangle, else None.
    """
    if bg is None:
        bg = int(get_bg(g))
    else:
        bg = int(bg)

    t = int(thickness)
    best = None  # (area, r0, c0, r1, c1, bc, fill)
    for bc in sorted(set(map(int, g.flatten())) - {int(bg)}):
        lab, n = label(g == int(bc), structure=structure)
        for cid in range(1, int(n) + 1):
            pts = np.argwhere(lab == int(cid))
            if int(len(pts)) < 12:
                continue

            r0, c0 = pts.min(axis=0)
            r1, c1 = pts.max(axis=0)
            r0, c0, r1, c1 = map(int, (r0, c0, r1, c1))
            h = int(r1 - r0 + 1)
            w = int(c1 - c0 + 1)
            if h < 2 * t + 1 or w < 2 * t + 1:
                continue

            box = g[r0 : r1 + 1, c0 : c1 + 1]
            frame_mask = np.zeros((h, w), dtype=bool)
            frame_mask[:t, :] = True
            frame_mask[-t:, :] = True
            frame_mask[:, :t] = True
            frame_mask[:, -t:] = True
            if int(len(pts)) != int(frame_mask.sum()):
                continue
            if not (box[frame_mask] == int(bc)).all():
                continue

            inner = box[t : h - t, t : w - t]
            if inner.size == 0:
                continue
            fill = int(Counter(inner.flatten()).most_common(1)[0][0])
            if fill == int(bc):
                continue
            if not (inner == int(fill)).all():
                continue

            area = int(h * w)
            cand = (area, r0, c0, r1, c1, int(bc), int(fill))
            if best is None or cand[0] > best[0]:
                best = cand

    if best is None:
        return None
    _area, r0, c0, r1, c1, bc, fill = best
    return int(r0), int(c0), int(r1), int(c1), int(bc), int(fill)


def _map_ring_to_out(
    rr: int,
    cc: int,
    *,
    r0: int,
    c0: int,
    r1: int,
    c1: int,
    outH: int,
    outW: int,
    proportional: bool,
) -> tuple[int, int] | None:
    """Map a ring cell to an output coordinate using either direct or proportional projection."""
    if rr == int(r0) - 1 and int(c0) <= int(cc) <= int(c1):
        mc = (
            _scale_axis(int(cc) - int(c0), int(c1) - int(c0) + 1, int(outW))
            if proportional
            else max(0, min(int(outW) - 1, int(cc) - (int(c0) + _FRAME_THICKNESS)))
        )
        return 0, int(mc)
    if rr == int(r1) + 1 and int(c0) <= int(cc) <= int(c1):
        mc = (
            _scale_axis(int(cc) - int(c0), int(c1) - int(c0) + 1, int(outW))
            if proportional
            else max(0, min(int(outW) - 1, int(cc) - (int(c0) + _FRAME_THICKNESS)))
        )
        return int(outH) - 1, int(mc)
    if cc == int(c0) - 1 and int(r0) <= int(rr) <= int(r1):
        mr = (
            _scale_axis(int(rr) - int(r0), int(r1) - int(r0) + 1, int(outH))
            if proportional
            else max(0, min(int(outH) - 1, int(rr) - (int(r0) + _FRAME_THICKNESS)))
        )
        return int(mr), 0
    if cc == int(c1) + 1 and int(r0) <= int(rr) <= int(r1):
        mr = (
            _scale_axis(int(rr) - int(r0), int(r1) - int(r0) + 1, int(outH))
            if proportional
            else max(0, min(int(outH) - 1, int(rr) - (int(r0) + _FRAME_THICKNESS)))
        )
        return int(mr), int(outW) - 1
    return None


def _build_manhattan(
    p1: tuple[int, int], p2: tuple[int, int], cornerA: bool
) -> list[tuple[int, int]]:
    (ra, ca), (rb, cb) = p1, p2
    if ra == rb:
        step = 1 if cb >= ca else -1
        return [(int(ra), int(c)) for c in range(int(ca), int(cb) + step, step)]
    if ca == cb:
        step = 1 if rb >= ra else -1
        return [(int(r), int(ca)) for r in range(int(ra), int(rb) + step, step)]
    if cornerA:
        step = 1 if rb >= ra else -1
        path = [(int(r), int(ca)) for r in range(int(ra), int(rb) + step, step)]
        step = 1 if cb >= ca else -1
        path += [(int(rb), int(c)) for c in range(int(ca) + step, int(cb) + step, step)]
        return path
    step = 1 if cb >= ca else -1
    path = [(int(ra), int(c)) for c in range(int(ca), int(cb) + step, step)]
    step = 1 if rb >= ra else -1
    path += [(int(r), int(cb)) for r in range(int(ra) + step, int(rb) + step, step)]
    return path


def _inside_corner(
    p1: tuple[int, int], p2: tuple[int, int], cornerA: bool, outH: int, outW: int
) -> bool:
    (ra, ca), (rb, cb) = p1, p2
    if ra == rb or ca == cb:
        return True
    corner = (rb, ca) if cornerA else (ra, cb)
    rr, cc = corner
    return 0 < int(rr) < int(outH) - 1 and 0 < int(cc) < int(outW) - 1


def _solve(grid: np.ndarray, *, proportional: bool) -> np.ndarray:
    g = np.array(grid, dtype=int)
    H, W = g.shape
    bg = int(get_bg(g))
    t = int(_FRAME_THICKNESS)

    found = _find_framed_rect_uniform_fill(g, bg=int(bg), thickness=t)
    assert found is not None
    r0, c0, r1, c1, bc, fill = found
    outH = int(r1 - r0 + 1 - 2 * t)
    outW = int(c1 - c0 + 1 - 2 * t)
    out = np.full((outH, outW), int(fill), dtype=int)

    comp_mask = (g != int(bg)) & (g != int(fill)) & (g != int(bc))
    clab, cn = label(comp_mask, structure=STRUCT4)

    ring = set()
    for cc in range(c0, c1 + 1):
        for rr in (r0 - 1, r1 + 1):
            if 0 <= rr < H:
                ring.add((int(rr), int(cc)))
    for rr in range(r0, r1 + 1):
        for cc in (c0 - 1, c1 + 1):
            if 0 <= cc < W:
                ring.add((int(rr), int(cc)))

    comp_pts: dict[int, list[tuple[int, int, int]]] = defaultdict(list)
    for rr, cc in ring:
        cid = int(clab[rr, cc])
        if cid == 0:
            continue
        mapped = _map_ring_to_out(
            int(rr),
            int(cc),
            r0=int(r0),
            c0=int(c0),
            r1=int(r1),
            c1=int(c1),
            outH=int(outH),
            outW=int(outW),
            proportional=bool(proportional),
        )
        if mapped is None:
            continue
        mr, mc = mapped
        comp_pts[cid].append((int(mr), int(mc), int(g[rr, cc])))

    comp_cols: dict[int, list[int]] = {}
    for cid in range(1, int(cn) + 1):
        comp_cols[cid] = sorted(set(map(int, g[clab == int(cid)].flatten())))

    deferred: dict[int, set[tuple[int, int]]] = defaultdict(set)

    for cid, pts in comp_pts.items():
        if len(pts) < 2:
            continue
        cols = comp_cols.get(cid, [])
        if not cols:
            continue

        hist: dict[tuple[int, int], Counter] = defaultdict(Counter)
        for r, c, col in pts:
            hist[(int(r), int(c))][int(col)] += 1

        coords = list(hist.keys())

        if len(cols) == 1:
            colv = int(cols[0])
            rows = {r for r, _c in coords}
            cols2 = {c for _r, c in coords}
            if len(rows) == 1 or len(cols2) == 1:
                for p in coords:
                    deferred[colv].add((int(p[0]), int(p[1])))
                continue

            best = None
            for i in range(len(coords)):
                for j in range(i + 1, len(coords)):
                    d = abs(coords[i][0] - coords[j][0]) + abs(coords[i][1] - coords[j][1])
                    if best is None or d > best[0]:
                        best = (d, coords[i], coords[j])
            if best is None:
                continue
            _d, p1, p2 = best
            pathA = _build_manhattan(p1, p2, True)
            pathB = _build_manhattan(p1, p2, False)
            chosen = (
                pathB
                if (
                    _inside_corner(p1, p2, False, int(outH), int(outW))
                    and not _inside_corner(p1, p2, True, int(outH), int(outW))
                )
                else pathA
            )
            for rr, cc in chosen:
                out[int(rr), int(cc)] = int(colv)
            continue

        if len(cols) == 2:
            best = None
            for i in range(len(coords)):
                for j in range(i + 1, len(coords)):
                    d = abs(coords[i][0] - coords[j][0]) + abs(coords[i][1] - coords[j][1])
                    if best is None or d > best[0]:
                        best = (d, coords[i], coords[j])
            if best is None:
                continue
            _d, a, b = best
            ca, cb = map(int, cols)
            ha = int(hist[a].most_common(1)[0][0])
            hb = int(hist[b].most_common(1)[0][0])

            pathA = _build_manhattan(a, b, True)
            pathB = _build_manhattan(a, b, False)
            path = (
                pathB
                if (
                    _inside_corner(a, b, False, int(outH), int(outW))
                    and not _inside_corner(a, b, True, int(outH), int(outW))
                )
                else pathA
            )

            def paint_alt(path_pts: list[tuple[int, int]], start_color: int) -> None:
                cur = int(start_color)
                for rr, cc in path_pts:
                    out[int(rr), int(cc)] = int(cur)
                    cur = int(cb) if cur == int(ca) else int(ca)

            chosen = None
            for path_pts in (path, list(reversed(path))):
                for start_color in (ca, cb):
                    end_color = (
                        int(start_color)
                        if (len(path_pts) - 1) % 2 == 0
                        else (int(cb) if int(start_color) == int(ca) else int(ca))
                    )
                    if int(start_color) != int(ha):
                        continue
                    if int(end_color) != int(hb):
                        continue
                    chosen = (path_pts, int(start_color))
                    break
                if chosen is not None:
                    break

            assert chosen is not None
            path_pts, start_color = chosen
            paint_alt(path_pts, start_color)
            continue

    for col, pts in deferred.items():
        pts = list(pts)
        if not pts:
            continue
        tops = [p for p in pts if p[0] == 0]
        bots = [p for p in pts if p[0] == outH - 1]
        if tops and bots:
            for cc in sorted({c for _r, c in pts}):
                rs = [r for r, c in pts if c == cc]
                for rr in range(min(rs), max(rs) + 1):
                    out[int(rr), int(cc)] = int(col)
            continue
        lefts = [p for p in pts if p[1] == 0]
        rights = [p for p in pts if p[1] == outW - 1]
        if lefts and rights:
            for rr in sorted({r for r, _c in pts}):
                cs = [c for r, c in pts if r == rr]
                for cc in range(min(cs), max(cs) + 1):
                    out[int(rr), int(cc)] = int(col)
            continue
        if len(pts) == 1:
            rr, cc = pts[0]
            out[int(rr), int(cc)] = int(col)
            continue
        best = None
        for i in range(len(pts)):
            for j in range(i + 1, len(pts)):
                d = abs(pts[i][0] - pts[j][0]) + abs(pts[i][1] - pts[j][1])
                if best is None or d > best[0]:
                    best = (d, pts[i], pts[j])
        if best is None:
            continue
        _d, p1, p2 = best
        pathA = _build_manhattan(p1, p2, True)
        pathB = _build_manhattan(p1, p2, False)
        chosen = (
            pathB
            if (
                _inside_corner(p1, p2, False, int(outH), int(outW))
                and not _inside_corner(p1, p2, True, int(outH), int(outW))
            )
            else pathA
        )
        for rr, cc in chosen:
            out[int(rr), int(cc)] = int(col)

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recover the interior crop of a thick framed rectangle by proportionally projecting outside touches inward.

    Input structure:
    - A uniform background, one largest 2px-thick closed rectangular frame, and a uniformly filled inner hole.
    - Colored open-rectangle components lie outside the frame and touch the one-cell ring around its outer boundary.

    Rule:
    - Output only the frame interior, excluding both border layers, initialized with the interior fill color.
    - For each ring touch, map the touched side coordinate from the full outer frame span onto the shorter interior span by proportional scaling.
    - Monochrome components produce solid Manhattan paths or full stripes; two-color components produce alternating Manhattan paths whose phase is anchored by the touch colors.

    Key insight: The fused variant is solved by combining two geometric corrections at once: detect a 2px annulus, then project side coordinates by scale-to-fit rather than by direct offset.
    """
    return _solve(np.array(grid, dtype=int), proportional=True)


# --------------- generator (a6 / combo D9-thick-frame + D16-proportional) ---------------

_BORDER_COLOR = 3
_BG_CHOICES = [0, 1, 2, 4, 5, 6, 7, 8, 9]
_FILL_EQ_BG_PROB = 0.6
_INTERIOR_H_RANGE = (4, 13)
_INTERIOR_W_RANGE = (7, 14)
_GRID_SIZE_RANGE = (20, 30)
_GRID_INTERIOR_PAD = 10
_FRAME_MARGIN = 4
_FRAME_MARGIN_FAR = 6
_OUTER_OFFSET_RANGE = (2, 3)
_N_COMPONENTS_RANGE = (2, 3)
_U_PAIR_PROB = 0.25
_DUAL_COLOR_PROB = 0.45
_DISTRACTOR_COUNT_RANGE = (0, 2)
_DISTRACTOR_SIZE_RANGE = (1, 4)
_DISTRACTOR_TRIES = 40
_DISTRACTOR_CLEARANCE = 2
_GEN_RETRIES = 400


def _line(a: int, b: int) -> range:
    step = 1 if b >= a else -1
    return range(a, b + step, step)


def _vpath(r0: int, r1: int, c: int) -> list[tuple[int, int]]:
    return [(r, c) for r in _line(r0, r1)]


def _hpath(r: int, c0: int, c1: int) -> list[tuple[int, int]]:
    return [(r, c) for c in _line(c0, c1)]


def _join(*parts: list[tuple[int, int]]) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    for part in parts:
        for p in part:
            if not out or out[-1] != p:
                out.append(p)
    return out


def _paint_path(grid: np.ndarray, path: list[tuple[int, int]], colors: tuple[int, ...]) -> None:
    if not path:
        return
    rs, cs = zip(*path)
    if len(colors) == 1:
        grid[rs, cs] = colors[0]
    else:
        a, b = colors
        grid[rs, cs] = [a if i % 2 == 0 else b for i in range(len(path))]


def _can_place(grid: np.ndarray, coords: list[tuple[int, int]], bg: int, comp_mask: np.ndarray) -> bool:
    H, W = grid.shape
    s = set(coords)
    for r, c in s:
        if not (0 <= r < H and 0 <= c < W):
            return False
        if grid[r, c] != bg:
            return False
        for nr, nc in neighbors4(r, c):
            if (nr, nc) not in s and 0 <= nr < H and 0 <= nc < W and comp_mask[nr, nc]:
                return False
    return True


def _place_component(
    grid: np.ndarray,
    comp_mask: np.ndarray,
    path: list[tuple[int, int]],
    colors: tuple[int, ...],
    bg: int,
) -> bool:
    if len(set(path)) != len(path):
        return False
    if not _can_place(grid, path, bg, comp_mask):
        return False
    _paint_path(grid, path, colors)
    rs, cs = zip(*path)
    comp_mask[rs, cs] = True
    return True


def _draw_frame(grid: np.ndarray, r0: int, c0: int, r1: int, c1: int, bc: int, fill: int) -> None:
    t = int(_FRAME_THICKNESS)
    grid[r0 : r0 + t, c0 : c1 + 1] = bc
    grid[r1 - t + 1 : r1 + 1, c0 : c1 + 1] = bc
    grid[r0 : r1 + 1, c0 : c0 + t] = bc
    grid[r0 : r1 + 1, c1 - t + 1 : c1 + 1] = bc
    grid[r0 + t : r1 - t + 1, c0 + t : c1 - t + 1] = fill


def _non_direct_inverse_offsets(inner_idx: int, inner_len: int) -> list[int]:
    t = int(_FRAME_THICKNESS)
    outer_len = int(inner_len) + 2 * t
    direct = int(inner_idx) + t
    return [
        off
        for off in _inverse_scaled_offsets(int(inner_idx), int(outer_len), int(inner_len), avoid_corners=True)
        if int(off) != int(direct)
    ]


def _sample_scaled_index(inner_len: int, *, upper_half: bool) -> tuple[int, int]:
    """
    Pick an interior edge index plus an inverse-projected outer offset.

    The chosen outer offset is preferably different from the old direct offset so the sample
    genuinely depends on proportional projection.
    """
    if upper_half:
        raw = range(1, max(2, int(inner_len) // 2 + 1))
    else:
        raw = range(max(1, int(inner_len) // 2), int(inner_len) - 1)
    candidates = list(dict.fromkeys(int(x) for x in raw))
    random.shuffle(candidates)

    fallback = []
    t = int(_FRAME_THICKNESS)
    outer_len = int(inner_len) + 2 * t
    for idx in candidates:
        nd = _non_direct_inverse_offsets(int(idx), int(inner_len))
        if nd:
            return int(idx), int(random.choice(nd))
        all_inv = _inverse_scaled_offsets(int(idx), int(outer_len), int(inner_len), avoid_corners=True)
        if all_inv:
            fallback.append((int(idx), list(all_inv)))
    if fallback:
        idx, offs = random.choice(fallback)
        return int(idx), int(random.choice(offs))
    raise ValueError("no scalable interior index available")


def _ring_top(r0: int, c0: int, col_off: int) -> tuple[int, int]:
    return int(r0) - 1, int(c0) + int(col_off)


def _ring_bottom(r1: int, c0: int, col_off: int) -> tuple[int, int]:
    return int(r1) + 1, int(c0) + int(col_off)


def _ring_left(r0: int, c0: int, row_off: int) -> tuple[int, int]:
    return int(r0) + int(row_off), int(c0) - 1


def _ring_right(r0: int, c1: int, row_off: int) -> tuple[int, int]:
    return int(r0) + int(row_off), int(c1) + 1


def _l_specs(kind: str, H: int, W: int, r0: int, c0: int, r1: int, c1: int):
    """Build inner L-path coords and the matching proportional-projection outer coords."""
    is_top = kind[0] == "t"
    is_left = kind[1] == "l"

    r, row_off = _sample_scaled_index(int(H), upper_half=is_top)
    c, col_off = _sample_scaled_index(int(W), upper_half=is_left)

    lo, hi = _OUTER_OFFSET_RANGE
    R = (r0 - random.randint(lo, hi)) if is_top else (r1 + random.randint(lo, hi))
    C = (c0 - random.randint(lo, hi)) if is_left else (c1 + random.randint(lo, hi))

    ring_a = _ring_top(r0, c0, col_off) if is_top else _ring_bottom(r1, c0, col_off)
    ring_b = _ring_left(r0, c0, row_off) if is_left else _ring_right(r0, c1, row_off)

    edge_r = 0 if is_top else H - 1
    edge_c = 0 if is_left else W - 1
    inner = _join(_vpath(edge_r, r, c), _hpath(r, c, edge_c))
    outer = _join(
        _vpath(ring_a[0], R, ring_a[1]),
        _hpath(R, ring_a[1], C),
        _vpath(R, ring_b[0], C),
        _hpath(ring_b[0], C, ring_b[1]),
    )
    meta = {
        "kind": kind,
        "row_idx": int(r),
        "col_idx": int(c),
        "row_off": int(row_off),
        "col_off": int(col_off),
    }
    return inner, outer, meta


def _sample_two_scaled_indices(inner_len: int) -> tuple[tuple[int, int], tuple[int, int]]:
    mids = []
    for idx in range(1, int(inner_len) - 1):
        nd = _non_direct_inverse_offsets(int(idx), int(inner_len))
        if nd:
            mids.append((int(idx), list(nd)))
    if len(mids) >= 2:
        a, b = random.sample(mids, 2)
        return (int(a[0]), int(random.choice(a[1]))), (int(b[0]), int(random.choice(b[1])))

    full = []
    t = int(_FRAME_THICKNESS)
    outer_len = int(inner_len) + 2 * t
    for idx in range(1, int(inner_len) - 1):
        inv = _inverse_scaled_offsets(int(idx), int(outer_len), int(inner_len), avoid_corners=True)
        if inv:
            full.append((int(idx), list(inv)))
    if len(full) < 2:
        raise ValueError("need two scalable indices")
    a, b = random.sample(full, 2)
    return (int(a[0]), int(random.choice(a[1]))), (int(b[0]), int(random.choice(b[1])))


def _u_pair_specs(orientation: str, H: int, W: int, r0: int, c0: int, r1: int, c1: int):
    """Build inner stripes and the matching pair of proportional outer U-shapes."""
    lo, hi = _OUTER_OFFSET_RANGE
    if orientation == "vertical":
        (c_a, off_a), (c_b, off_b) = _sample_two_scaled_indices(int(W))
        if c_a == c_b:
            raise ValueError("vertical U-pair needs distinct columns")
        if c_a > c_b:
            c_a, c_b = c_b, c_a
            off_a, off_b = off_b, off_a
        top_r = r0 - random.randint(lo, hi)
        bot_r = r1 + random.randint(lo, hi)
        a_top = _ring_top(r0, c0, off_a)
        b_top = _ring_top(r0, c0, off_b)
        a_bot = _ring_bottom(r1, c0, off_a)
        b_bot = _ring_bottom(r1, c0, off_b)
        outer_a = _join(
            _vpath(a_top[0], top_r, a_top[1]),
            _hpath(top_r, a_top[1], b_top[1]),
            _vpath(top_r, b_top[0], b_top[1]),
        )
        outer_b = _join(
            _vpath(a_bot[0], bot_r, a_bot[1]),
            _hpath(bot_r, a_bot[1], b_bot[1]),
            _vpath(bot_r, b_bot[0], b_bot[1]),
        )
        inner = [[(r, c_a) for r in range(H)], [(r, c_b) for r in range(H)]]
    else:
        (r_a, off_a), (r_b, off_b) = _sample_two_scaled_indices(int(H))
        if r_a == r_b:
            raise ValueError("horizontal U-pair needs distinct rows")
        if r_a > r_b:
            r_a, r_b = r_b, r_a
            off_a, off_b = off_b, off_a
        left_c = c0 - random.randint(lo, hi)
        right_c = c1 + random.randint(lo, hi)
        a_left = _ring_left(r0, c0, off_a)
        b_left = _ring_left(r0, c0, off_b)
        a_right = _ring_right(r0, c1, off_a)
        b_right = _ring_right(r0, c1, off_b)
        outer_a = _join(
            _hpath(a_left[0], a_left[1], left_c),
            _vpath(a_left[0], b_left[0], left_c),
            _hpath(b_left[0], left_c, b_left[1]),
        )
        outer_b = _join(
            _hpath(a_right[0], a_right[1], right_c),
            _vpath(a_right[0], b_right[0], right_c),
            _hpath(b_right[0], right_c, b_right[1]),
        )
        inner = [[(r_a, c) for c in range(W)], [(r_b, c) for c in range(W)]]
    return inner, outer_a, outer_b


def _add_distractor(
    grid: np.ndarray,
    comp_mask: np.ndarray,
    bg: int,
    frame: tuple[int, int, int, int],
    color: int,
) -> None:
    H, W = grid.shape
    r0, c0, r1, c1 = frame
    gap = _DISTRACTOR_CLEARANCE
    sz_lo, sz_hi = _DISTRACTOR_SIZE_RANGE
    for _ in range(_DISTRACTOR_TRIES):
        size = random.randint(sz_lo, sz_hi)
        r, c = random.randint(1, H - 2), random.randint(1, W - 2)
        coords = [(r, c)]
        used = {(r, c)}
        for _ in range(size - 1):
            base = random.choice(coords)
            opts = [
                (nr, nc)
                for nr, nc in neighbors4(*base)
                if 1 <= nr < H - 1 and 1 <= nc < W - 1 and (nr, nc) not in used
            ]
            if not opts:
                break
            p = random.choice(opts)
            used.add(p)
            coords.append(p)
        if any(r0 - gap <= rr <= r1 + gap and c0 - gap <= cc <= c1 + gap for rr, cc in coords):
            continue
        if _place_component(grid, comp_mask, coords, (color,), bg):
            return


def _gen_attempt() -> np.ndarray | None:
    bg = random.choice(_BG_CHOICES)
    bc = _BORDER_COLOR
    fill_choices = [x for x in range(10) if x not in (bg, bc)]
    fill = bg if random.random() < _FILL_EQ_BG_PROB else random.choice(fill_choices)
    H = random.randint(*_INTERIOR_H_RANGE)
    W = random.randint(*_INTERIOR_W_RANGE)
    t = int(_FRAME_THICKNESS)
    outerH = H + 2 * t
    outerW = W + 2 * t

    rows = random.randint(max(_GRID_SIZE_RANGE[0], outerH + _GRID_INTERIOR_PAD), _GRID_SIZE_RANGE[1])
    cols = random.randint(max(_GRID_SIZE_RANGE[0], outerW + _GRID_INTERIOR_PAD), _GRID_SIZE_RANGE[1])
    r0 = random.randint(_FRAME_MARGIN, rows - outerH - _FRAME_MARGIN_FAR)
    c0 = random.randint(_FRAME_MARGIN, cols - outerW - _FRAME_MARGIN_FAR)
    r1 = r0 + outerH - 1
    c1 = c0 + outerW - 1

    grid = np.full((rows, cols), bg, dtype=int)
    _draw_frame(grid, r0, c0, r1, c1, bc, fill)

    expected = np.full((H, W), fill, dtype=int)
    comp_mask = np.zeros_like(grid, dtype=bool)
    palette = [x for x in range(10) if x not in (bg, bc, fill)]
    random.shuffle(palette)
    used_inner = np.zeros((H, W), dtype=bool)

    n_components = random.randint(*_N_COMPONENTS_RANGE)
    remaining = n_components
    color_i = 0

    if remaining >= 2 and random.random() < _U_PAIR_PROB:
        orientation = random.choice(["vertical", "horizontal"])
        if orientation == "horizontal" and H < 5:
            orientation = "vertical"
        if color_i >= len(palette):
            return None
        color = palette[color_i]
        color_i += 1
        try:
            inner_lines, outer_a, outer_b = _u_pair_specs(orientation, H, W, r0, c0, r1, c1)
        except ValueError:
            return None
        all_inner = [p for line in inner_lines for p in line]
        irs, ics = zip(*all_inner)
        if used_inner[irs, ics].any():
            return None
        if not _place_component(grid, comp_mask, outer_a, (color,), bg):
            return None
        if not _place_component(grid, comp_mask, outer_b, (color,), bg):
            return None
        expected[irs, ics] = color
        used_inner[irs, ics] = True
        remaining -= 2

    corners = random.sample(["tl", "tr", "bl", "br"], k=remaining)
    for kind in corners:
        if color_i >= len(palette):
            return None
        try:
            inner, outer, _meta = _l_specs(kind, H, W, r0, c0, r1, c1)
        except ValueError:
            return None
        irs, ics = zip(*inner)
        if used_inner[irs, ics].any():
            return None
        if random.random() < _DUAL_COLOR_PROB and color_i + 1 < len(palette):
            colors = (palette[color_i], palette[color_i + 1])
            color_i += 2
        else:
            colors = (palette[color_i],)
            color_i += 1
        if not _place_component(grid, comp_mask, outer, colors, bg):
            return None
        _paint_path(expected, inner, colors)
        used_inner[irs, ics] = True

    for _ in range(random.randint(*_DISTRACTOR_COUNT_RANGE)):
        if color_i >= len(palette):
            break
        _add_distractor(grid, comp_mask, bg, (r0, c0, r1, c1), palette[color_i])
        color_i += 1

    try:
        actual = main(grid)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    # Keep the fused rule observable: reject samples that a thick-frame-only
    # direct-offset solver would also solve.
    direct = _solve(grid, proportional=False)
    if np.array_equal(direct, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(_GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid a6f40cea combo input")
