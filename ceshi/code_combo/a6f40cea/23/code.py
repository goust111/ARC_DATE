# Combo variant of a6f40cea (index 23)
# combo_spec: D8/2+D19/1
# Name: 外形编码宽度与拐角
#
# Design analysis
# a) Concrete behavior and trace
#    - The base task projects one top/bottom touch and one left/right touch into the frame
#      interior, then draws a 1px L-path whose elbow is just the crossing of those two projected
#      endpoint axes.
#    - This combo keeps the same global scene, but each useful L-type source now encodes two
#      interior properties at once through its outside geometry:
#      1. the top/bottom stub length becomes the path width;
#      2. the outside frame corner position projects inward and fixes an interior detour corner.
#    - Example for a top-left source:
#      base: top touch at interior col 5 and left touch at interior row 4 -> draw
#            (0,5) -> (4,5) -> (4,0), width 1.
#      combo: if the outside top stub is 3 cells long, decode width=3 and therefore a 3-step
#             staircase. If the outside corner sits 3 cells above the frame and 4 cells left of it,
#             project that to interior pivot (2,3). The inside path becomes a 6-segment staircase
#             that passes through that pivot, then the same centerline is thickened to width 3.
#    - The visible difference is therefore not only "thicker than before", but also "the elbow is
#      no longer the trivial endpoint crossing": the outer shape must be read to recover both the
#      staircase complexity and the middle pivot.
# b) Complexity calibration
#    - Width is limited to 2-3 cells. This is visually obvious without creating massive interior
#      overlaps or making tiny interiors infeasible.
#    - The staircase depth is limited to 2-3 teeth per corner kind instead of allowing arbitrary
#      routings. That keeps the inference space small enough for 2-4 ARC examples.
#    - Every generated sample contains at least one L-type fused source, so the board cannot
#      collapse to the original thin-U-only behavior.
# c) Implicit-assumption review
#    - The original solver assumed each non-straight component was fully described by two projected
#      endpoints plus one of two possible single elbows. That fails here because the outside bbox is
#      now semantically meaningful.
#    - The original solver also assumed width was always 1, so painting only a centerline was
#      sufficient. The combo must instead paint a ribbon around each step of the route.
#    - Endpoint color anchoring for two-color components still works because the alternation is
#      still defined along a deterministic centerline order; width only thickens each step.
# d) Placement legality
#    - The generator remains output-first: it samples an interior thick detour, records the exact
#      occupied mask, then back-projects one outside open rectangle that encodes the same width and
#      pivot.
#    - Interior overlap checks use the full thick mask rather than only the centerline. Outside
#      placement still uses the original non-overlap / non-adjacency checks.
#    - U-pairs, when present, are only allowed if at least one L-source still remains, so the
#      fused behavior is always represented.
# e) Visual observability
#    - Both cues are human-visible geometry. Width comes from the visible stub length on the
#      top/bottom touched side, and the detour corner comes from the visible outside corner offset.
#      No color-id ordering or hidden metadata is used.
#    - Because the inside path is both thick and nontrivially bent, the combo output is clearly
#      different from both the base task and a width-only interpretation.
# f) Visual cleanliness
#    - Each L-source is constrained to its natural corner sector, so thick routes stay separated
#      instead of filling the whole interior.
#    - The staircase depth is capped at 3 so the extra bends stay readable rather than turning the
#      interior into a dense maze.
# g) Change inferability
#    - Colors keep their original meaning: monochrome sources paint solid routes, and two-color
#      sources alternate along the centerline with endpoint colors fixing the phase.
#    - The new reasoning burden is purely geometric and repeated across samples: read the outside
#      component's shape, then convert that shape into both a route width and a route bend.
# 9) New solver assumptions and generator guarantees
#    - Solver assumes every fused L-source has exactly one touch on a top/bottom side and one touch
#      on a left/right side. Generator enforces that by constructing only one ring endpoint per side
#      for L-sources.
#    - Solver assumes the top/bottom stub length is at least 2 so the projected pivot row stays
#      strictly inside. Generator enforces widths in {2,3} and samples interiors large enough.
#    - Solver assumes the projected pivot lies strictly between the two endpoint axes and that the
#      pre-pivot horizontal gap is wide enough to support the decoded number of staircase teeth.
#      Generator samples endpoint positions only after choosing a valid projected pivot.

from __future__ import annotations

from collections import Counter, defaultdict
import random

import numpy as np
from scipy.ndimage import label

from arc_utils import STRUCT4, get_bg, neighbors4


def _find_framed_rect_uniform_fill(
    g: np.ndarray,
    *,
    bg: int | None = None,
    structure=STRUCT4,
) -> tuple[int, int, int, int, int, int] | None:
    """
    Detect the main 1px-thick rectangular frame with uniform interior fill.

    Returns (r0, c0, r1, c1, border_color, fill_color) for the largest such rectangle, else None.
    """
    if bg is None:
        bg = int(get_bg(g))
    else:
        bg = int(bg)

    best = None
    for bc in sorted(set(map(int, g.flatten())) - {int(bg)}):
        lab, n = label(g == int(bc), structure=structure)
        for cid in range(1, int(n) + 1):
            pts = np.argwhere(lab == int(cid))
            if int(len(pts)) < 8:
                continue

            r0, c0 = pts.min(axis=0)
            r1, c1 = pts.max(axis=0)
            r0, c0, r1, c1 = map(int, (r0, c0, r1, c1))
            h = int(r1 - r0 + 1)
            w = int(c1 - c0 + 1)
            if h < 3 or w < 3:
                continue

            per = int(2 * (h + w) - 4)
            if int(len(pts)) != per:
                continue
            if not (g[r0, c0 : c1 + 1] == int(bc)).all():
                continue
            if not (g[r1, c0 : c1 + 1] == int(bc)).all():
                continue
            if not (g[r0 : r1 + 1, c0] == int(bc)).all():
                continue
            if not (g[r0 : r1 + 1, c1] == int(bc)).all():
                continue

            inner = g[r0 + 1 : r1, c0 + 1 : c1]
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


def _line(a: int, b: int) -> range:
    step = 1 if b >= a else -1
    return range(a, b + step, step)


def _vpath(r0: int, r1: int, c: int) -> list[tuple[int, int]]:
    return [(int(r), int(c)) for r in _line(r0, r1)]


def _hpath(r: int, c0: int, c1: int) -> list[tuple[int, int]]:
    return [(int(r), int(c)) for c in _line(c0, c1)]


def _join(*parts: list[tuple[int, int]]) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    for part in parts:
        for p in part:
            if not out or out[-1] != p:
                out.append((int(p[0]), int(p[1])))
    return out


def _split_positive(total: int, parts: int) -> list[int]:
    total = int(total)
    parts = int(parts)
    if parts <= 0:
        return []
    assert total >= parts
    base, rem = divmod(total, parts)
    return [int(base + (1 if i < rem else 0)) for i in range(parts)]


def _ring_cells(r0: int, c0: int, r1: int, c1: int, H: int, W: int) -> list[tuple[int, int]]:
    ring: list[tuple[int, int]] = []
    seen: set[tuple[int, int]] = set()
    for cc in range(c0, c1 + 1):
        for rr in (r0 - 1, r1 + 1):
            if 0 <= rr < H:
                p = (int(rr), int(cc))
                if p not in seen:
                    seen.add(p)
                    ring.append(p)
    for rr in range(r0, r1 + 1):
        for cc in (c0 - 1, c1 + 1):
            if 0 <= cc < W:
                p = (int(rr), int(cc))
                if p not in seen:
                    seen.add(p)
                    ring.append(p)
    return ring


def _classify_corner_component(
    coords: list[tuple[int, int]],
    outH: int,
    outW: int,
) -> tuple[str, tuple[int, int], tuple[int, int]] | None:
    tops = sorted([p for p in coords if p[0] == 0])
    bots = sorted([p for p in coords if p[0] == outH - 1])
    lefts = sorted([p for p in coords if p[1] == 0])
    rights = sorted([p for p in coords if p[1] == outW - 1])
    if tops and lefts:
        return "tl", tuple(map(int, tops[0])), tuple(map(int, lefts[0]))
    if tops and rights:
        return "tr", tuple(map(int, tops[0])), tuple(map(int, rights[0]))
    if bots and lefts:
        return "bl", tuple(map(int, bots[0])), tuple(map(int, lefts[0]))
    if bots and rights:
        return "br", tuple(map(int, bots[0])), tuple(map(int, rights[0]))
    return None


def _clamp_pivot(
    kind: str,
    pivot: tuple[int, int],
    p_a: tuple[int, int],
    p_b: tuple[int, int],
    outH: int,
    outW: int,
) -> tuple[int, int]:
    pr, pc = map(int, pivot)
    if kind == "tl":
        pr = max(1, min(pr, int(p_b[0]) - 1))
        pc = max(1, min(pc, int(p_a[1]) - 1))
    elif kind == "tr":
        pr = max(1, min(pr, int(p_b[0]) - 1))
        pc = min(outW - 2, max(pc, int(p_a[1]) + 1))
    elif kind == "bl":
        pr = min(outH - 2, max(pr, int(p_b[0]) + 1))
        pc = max(1, min(pc, int(p_a[1]) - 1))
    else:
        pr = min(outH - 2, max(pr, int(p_b[0]) + 1))
        pc = min(outW - 2, max(pc, int(p_a[1]) + 1))
    return int(pr), int(pc)


def _build_staircase_detour_path(
    kind: str,
    p_a: tuple[int, int],
    p_b: tuple[int, int],
    pivot: tuple[int, int],
    steps: int,
) -> list[tuple[int, int]]:
    steps = max(1, int(steps))
    pr, pc = map(int, pivot)
    if kind == "tl":
        _rt, ct = map(int, p_a)
        rl, _cl = map(int, p_b)
        row_pre = int(pr)
        row_post = int(rl - pr)
        col_pre = int(ct - pc)
        col_post = int(pc)
        sign_r, sign_c = 1, -1
    elif kind == "tr":
        _rt, ct = map(int, p_a)
        rr, _cr = map(int, p_b)
        row_pre = int(pr)
        row_post = int(rr - pr)
        col_pre = int(pc - ct)
        col_post = int(p_b[1] - pc)
        sign_r, sign_c = 1, 1
    elif kind == "bl":
        _rb, cb = map(int, p_a)
        rl, _cl = map(int, p_b)
        row_pre = int(p_a[0] - pr)
        row_post = int(pr - rl)
        col_pre = int(cb - pc)
        col_post = int(pc)
        sign_r, sign_c = -1, -1
    else:
        _rb, cb = map(int, p_a)
        rr, _cr = map(int, p_b)
        row_pre = int(p_a[0] - pr)
        row_post = int(pr - rr)
        col_pre = int(pc - cb)
        col_post = int(p_b[1] - pc)
        sign_r, sign_c = -1, 1

    if min(row_pre, row_post, col_pre, col_post) < 1:
        return _fallback_manhattan(p_a, p_b)

    steps = min(int(steps), int(row_pre + row_post), int(col_pre + col_post))
    pre_parts = max(0, int(steps) - 1)
    if row_pre < pre_parts or col_pre < pre_parts:
        return _fallback_manhattan(p_a, p_b)

    row_parts = _split_positive(row_pre, pre_parts) + [int(row_post)]
    col_parts = _split_positive(col_pre, pre_parts) + [int(col_post)]

    cur_r, cur_c = map(int, p_a)
    path = [(int(cur_r), int(cur_c))]
    for row_dist, col_dist in zip(row_parts, col_parts):
        for _ in range(int(row_dist)):
            cur_r += int(sign_r)
            path.append((int(cur_r), int(cur_c)))
        for _ in range(int(col_dist)):
            cur_c += int(sign_c)
            path.append((int(cur_r), int(cur_c)))
    assert (int(cur_r), int(cur_c)) == tuple(map(int, p_b))
    return path


def _point_axis(path_pts: list[tuple[int, int]], idx: int) -> str:
    prev_axis = None
    next_axis = None
    if idx > 0:
        pr, pc = path_pts[idx - 1]
        rr, cc = path_pts[idx]
        prev_axis = "v" if int(pr) != int(rr) else "h"
    if idx + 1 < len(path_pts):
        rr, cc = path_pts[idx]
        nr, nc = path_pts[idx + 1]
        next_axis = "v" if int(rr) != int(nr) else "h"
    if prev_axis is None:
        assert next_axis is not None
        return str(next_axis)
    if next_axis is None:
        return str(prev_axis)
    if prev_axis == next_axis:
        return str(prev_axis)
    return "b"


def _stamp_cells(rr: int, cc: int, axis: str, width: int, shape: tuple[int, int]) -> list[tuple[int, int]]:
    outH, outW = map(int, shape)
    width = int(width)
    lo = int((width - 1) // 2)
    hi = int(width // 2)
    if axis == "h":
        r0 = max(0, int(rr) - lo)
        r1 = min(outH - 1, int(rr) + hi)
        return [(int(r), int(cc)) for r in range(r0, r1 + 1)]
    if axis == "v":
        c0 = max(0, int(cc) - lo)
        c1 = min(outW - 1, int(cc) + hi)
        return [(int(rr), int(c)) for c in range(c0, c1 + 1)]
    r0 = max(0, int(rr) - lo)
    r1 = min(outH - 1, int(rr) + hi)
    c0 = max(0, int(cc) - lo)
    c1 = min(outW - 1, int(cc) + hi)
    return [(int(r), int(c)) for r in range(r0, r1 + 1) for c in range(c0, c1 + 1)]


def _path_mask(path_pts: list[tuple[int, int]], width: int, shape: tuple[int, int]) -> np.ndarray:
    outH, outW = map(int, shape)
    mask = np.zeros((outH, outW), dtype=bool)
    for idx, (rr, cc) in enumerate(path_pts):
        axis = _point_axis(path_pts, idx)
        cells = _stamp_cells(int(rr), int(cc), axis, int(width), (outH, outW))
        if cells:
            rs, cs = zip(*cells)
            mask[rs, cs] = True
    return mask


def _paint_thick_path(
    out: np.ndarray,
    path_pts: list[tuple[int, int]],
    width: int,
    colors: tuple[int, ...],
) -> None:
    if not path_pts:
        return
    outH, outW = out.shape
    if len(colors) == 1:
        seq = [int(colors[0])] * len(path_pts)
    else:
        a, b = map(int, colors)
        seq = [int(a if i % 2 == 0 else b) for i in range(len(path_pts))]
    for idx, (rr, cc) in enumerate(path_pts):
        axis = _point_axis(path_pts, idx)
        cells = _stamp_cells(int(rr), int(cc), axis, int(width), (outH, outW))
        if not cells:
            continue
        rs, cs = zip(*cells)
        out[rs, cs] = int(seq[idx])


def _fallback_manhattan(p1: tuple[int, int], p2: tuple[int, int]) -> list[tuple[int, int]]:
    (ra, ca), (rb, cb) = map(lambda p: (int(p[0]), int(p[1])), (p1, p2))
    if ra == rb:
        return [(int(ra), int(c)) for c in _line(ca, cb)]
    if ca == cb:
        return [(int(r), int(ca)) for r in _line(ra, rb)]
    return _join(_vpath(ra, rb, ca), _hpath(rb, ca, cb))


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recover thick detour paths inside the frame by decoding both endpoint touches and outside shape.

    Input structure:
    - One largest 1px closed rectangle with uniform fill, surrounded by a uniform background.
    - Colored components lie outside that frame and touch the one-cell ring around it.
    - Useful L-type components still touch one horizontal frame side and one vertical frame side.

    Rule:
    - Output only the frame interior, initialized with the fill color.
    - Project each ring touch inward to the corresponding boundary cell of the output crop.
    - Straight U-pairs still create solid thin full-span stripes.
    - For each L-type source, decode the path width from the top/bottom stub length, reuse that
      same visible width as the staircase tooth count, decode an interior pivot from the outside
      corner projection, then draw a thick multi-bend orthogonal staircase through that pivot.
    - Two-color sources alternate along that centerline, and each centerline step paints a width-w
      ribbon so the alternation stays aligned with the decoded route order.

    Key insight:
    The outside component is no longer just a pair of endpoints. Its vertical stand-off from the
    frame sets the path thickness, while its full outside corner location sets where the interior
    detour must bend.
    """
    g = np.array(grid, dtype=int)
    H, W = g.shape
    bg = int(get_bg(g))

    found = _find_framed_rect_uniform_fill(g, bg=int(bg))
    assert found is not None
    r0, c0, r1, c1, bc, fill = found
    outH = int(r1 - r0 - 1)
    outW = int(c1 - c0 - 1)
    out = np.full((outH, outW), int(fill), dtype=int)
    r_int0 = int(r0 + 1)
    c_int0 = int(c0 + 1)

    def map_to_out(rr: int, cc: int) -> tuple[int, int]:
        r = max(0, min(outH - 1, int(rr - r_int0)))
        c = max(0, min(outW - 1, int(cc - c_int0)))
        return int(r), int(c)

    comp_mask = (g != int(bg)) & (g != int(fill)) & (g != int(bc))
    clab, cn = label(comp_mask, structure=STRUCT4)

    ring = _ring_cells(r0, c0, r1, c1, H, W)

    comp_pts: dict[int, list[tuple[int, int, int]]] = defaultdict(list)
    for rr, cc in ring:
        cid = int(clab[rr, cc])
        if cid == 0:
            continue
        col = int(g[rr, cc])
        mr, mc = map_to_out(rr, cc)
        comp_pts[cid].append((int(mr), int(mc), int(col)))

    comp_cols: dict[int, list[int]] = {}
    comp_bbox: dict[int, tuple[int, int, int, int]] = {}
    for cid in range(1, int(cn) + 1):
        cells = np.argwhere(clab == int(cid))
        if cells.size == 0:
            continue
        comp_cols[cid] = sorted(set(map(int, g[clab == int(cid)].flatten())))
        cr0, cc0 = cells.min(axis=0)
        cr1, cc1 = cells.max(axis=0)
        comp_bbox[cid] = (int(cr0), int(cc0), int(cr1), int(cc1))

    deferred: dict[int, set[tuple[int, int]]] = defaultdict(set)

    for cid, pts in comp_pts.items():
        if len(pts) < 2:
            continue
        cols = comp_cols.get(int(cid), [])
        if not cols:
            continue

        hist: dict[tuple[int, int], Counter] = defaultdict(Counter)
        for rr, cc, col in pts:
            hist[(int(rr), int(cc))][int(col)] += 1
        coords = list(hist.keys())

        if len(cols) == 1:
            uniq = coords
            rows = {r for r, _c in uniq}
            cols2 = {c for _r, c in uniq}
            if len(rows) == 1 or len(cols2) == 1:
                for p in uniq:
                    deferred[int(cols[0])].add((int(p[0]), int(p[1])))
                continue

        classified = _classify_corner_component(coords, outH, outW)
        if classified is None:
            best = None
            for i in range(len(coords)):
                for j in range(i + 1, len(coords)):
                    d = abs(coords[i][0] - coords[j][0]) + abs(coords[i][1] - coords[j][1])
                    if best is None or d > best[0]:
                        best = (d, coords[i], coords[j])
            if best is None:
                continue
            _d, p1, p2 = best
            path = _fallback_manhattan(p1, p2)
            if len(cols) == 1:
                _paint_thick_path(out, path, 1, (int(cols[0]),))
            else:
                ca, cb = map(int, cols)
                _paint_thick_path(out, path, 1, (ca, cb))
            continue

        kind, p_a, p_b = classified
        cr0, cc0, cr1, cc1 = comp_bbox[int(cid)]
        if kind[0] == "t":
            width = int(r0 - cr0)
            pr = int(width - 1)
        else:
            width = int(cr1 - r1)
            pr = int(outH - width)
        if kind[1] == "l":
            pc = int(c0 - cc0 - 1)
        else:
            pc = int(outW - (cc1 - c1))
        width = max(1, int(width))
        pivot = _clamp_pivot(kind, (int(pr), int(pc)), p_a, p_b, outH, outW)
        path = _build_staircase_detour_path(kind, p_a, p_b, pivot, width)

        if len(cols) == 1:
            _paint_thick_path(out, path, width, (int(cols[0]),))
            continue

        ca, cb = map(int, cols)
        ha = int(hist[p_a].most_common(1)[0][0])
        hb = int(hist[p_b].most_common(1)[0][0])
        chosen = None
        for path_pts in (path, list(reversed(path))):
            for start_color in (ca, cb):
                end_color = start_color if ((len(path_pts) - 1) % 2 == 0) else (cb if start_color == ca else ca)
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
        alt_colors = (int(start_color), int(cb if start_color == ca else ca))
        _paint_thick_path(out, path_pts, width, alt_colors)

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
        path = _fallback_manhattan(p1, p2)
        _paint_thick_path(out, path, 1, (int(col),))

    return out


# --------------- generator (a6 / combo D8/2 + D19/1) ---------------

_BORDER_COLOR = 3
_BG_CHOICES = [0, 1, 2, 4, 5, 6, 7, 8, 9]
_FILL_EQ_BG_PROB = 0.6
_INTERIOR_H_RANGE = (7, 13)
_INTERIOR_W_RANGE = (9, 14)
_GRID_SIZE_RANGE = (22, 30)
_GRID_INTERIOR_PAD = 10
_FRAME_MARGIN = 4
_FRAME_MARGIN_FAR = 6
_N_COMPONENTS_RANGE = (2, 3)
_U_PAIR_PROB = 0.18
_DUAL_COLOR_PROB = 0.4
_WIDTH_RANGE = (2, 3)
_L_COMPONENT_TRIES = 80
_DISTRACTOR_COUNT_RANGE = (0, 2)
_DISTRACTOR_SIZE_RANGE = (1, 4)
_DISTRACTOR_TRIES = 40
_DISTRACTOR_CLEARANCE = 2
_GEN_RETRIES = 600


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
    if len(colors) == 1:
        rs, cs = zip(*path)
        grid[rs, cs] = int(colors[0])
    else:
        vals = [int(colors[i % 2]) for i in range(len(path))]
        rs, cs = zip(*path)
        grid[rs, cs] = vals
    comp_mask[rs, cs] = True
    return True


def _choose_width(H: int) -> int:
    lo, hi = _WIDTH_RANGE
    hi = min(int(hi), int(H - 2), 3)
    assert hi >= lo
    return int(random.randint(int(lo), int(hi)))


def _fused_l_specs(
    kind: str,
    H: int,
    W: int,
    r0: int,
    c0: int,
    r1: int,
    c1: int,
    width: int,
) -> tuple[list[tuple[int, int]], list[tuple[int, int]], np.ndarray] | None:
    width = int(width)
    if kind[0] == "t":
        pr = int(width - 1)
    else:
        pr = int(H - width)

    if kind == "tl":
        pc_hi = min(W - 3, max(1, W // 2))
        if pc_hi < 1:
            return None
        pc = int(random.randint(1, pc_hi))
        ct_lo = pc + width - 1
        ct_hi = min(W - 2, max(ct_lo, pc + max(2, W // 3)))
        rl_lo = pr + 1
        rl_hi = min(H - 2, max(rl_lo, pr + max(2, H // 3)))
        if ct_lo > ct_hi or rl_lo > rl_hi:
            return None
        ct = int(random.randint(ct_lo, ct_hi))
        rl = int(random.randint(rl_lo, rl_hi))
        p_a, p_b = (0, ct), (rl, 0)
        pivot = (pr, pc)
        outer = _join(
            _vpath(r0 - 1, r0 - width, c0 + 1 + ct),
            _hpath(r0 - width, c0 + 1 + ct, c0 - (pc + 1)),
            _vpath(r0 - width, r0 + 1 + rl, c0 - (pc + 1)),
            _hpath(r0 + 1 + rl, c0 - (pc + 1), c0 - 1),
        )
    elif kind == "tr":
        pc_lo = max(W // 2, 2)
        pc_hi = W - 2
        if pc_lo > pc_hi:
            return None
        pc = int(random.randint(pc_lo, pc_hi))
        ct_lo = max(1, pc - max(2, W // 3))
        ct_hi = pc - (width - 1)
        rr_lo = pr + 1
        rr_hi = min(H - 2, max(rr_lo, pr + max(2, H // 3)))
        if ct_lo > ct_hi or rr_lo > rr_hi:
            return None
        ct = int(random.randint(ct_lo, ct_hi))
        rr = int(random.randint(rr_lo, rr_hi))
        p_a, p_b = (0, ct), (rr, W - 1)
        pivot = (pr, pc)
        outer = _join(
            _vpath(r0 - 1, r0 - width, c0 + 1 + ct),
            _hpath(r0 - width, c0 + 1 + ct, c1 + (W - pc)),
            _vpath(r0 - width, r0 + 1 + rr, c1 + (W - pc)),
            _hpath(r0 + 1 + rr, c1 + (W - pc), c1 + 1),
        )
    elif kind == "bl":
        pc_hi = min(W - 3, max(1, W // 2))
        if pc_hi < 1 or pr <= 1:
            return None
        pc = int(random.randint(1, pc_hi))
        cb_lo = pc + width - 1
        cb_hi = min(W - 2, max(cb_lo, pc + max(2, W // 3)))
        rl_lo = max(1, pr - max(2, H // 3))
        rl_hi = pr - 1
        if cb_lo > cb_hi or rl_lo > rl_hi:
            return None
        cb = int(random.randint(cb_lo, cb_hi))
        rl = int(random.randint(rl_lo, rl_hi))
        p_a, p_b = (H - 1, cb), (rl, 0)
        pivot = (pr, pc)
        outer = _join(
            _vpath(r1 + 1, r1 + width, c0 + 1 + cb),
            _hpath(r1 + width, c0 + 1 + cb, c0 - (pc + 1)),
            _vpath(r1 + width, r0 + 1 + rl, c0 - (pc + 1)),
            _hpath(r0 + 1 + rl, c0 - (pc + 1), c0 - 1),
        )
    else:
        pc_lo = max(W // 2, 2)
        pc_hi = W - 2
        if pc_lo > pc_hi or pr <= 1:
            return None
        pc = int(random.randint(pc_lo, pc_hi))
        cb_lo = max(1, pc - max(2, W // 3))
        cb_hi = pc - (width - 1)
        rr_lo = max(1, pr - max(2, H // 3))
        rr_hi = pr - 1
        if cb_lo > cb_hi or rr_lo > rr_hi:
            return None
        cb = int(random.randint(cb_lo, cb_hi))
        rr = int(random.randint(rr_lo, rr_hi))
        p_a, p_b = (H - 1, cb), (rr, W - 1)
        pivot = (pr, pc)
        outer = _join(
            _vpath(r1 + 1, r1 + width, c0 + 1 + cb),
            _hpath(r1 + width, c0 + 1 + cb, c1 + (W - pc)),
            _vpath(r1 + width, r0 + 1 + rr, c1 + (W - pc)),
            _hpath(r0 + 1 + rr, c1 + (W - pc), c1 + 1),
        )

    center = _build_staircase_detour_path(kind, p_a, p_b, pivot, width)
    mask = _path_mask(center, width, (H, W))
    return center, outer, mask


def _u_pair_specs(orientation: str, H: int, W: int, r0: int, c0: int, r1: int, c1: int):
    if orientation == "vertical":
        c_a, c_b = sorted(random.sample(range(1, W - 1), 2))
        if c_b - c_a < 2:
            c_b = min(W - 2, c_a + 2)
        top_r = r0 - random.randint(2, 3)
        bot_r = r1 + random.randint(2, 3)
        a_top, b_top = (r0 - 1, c0 + 1 + c_a), (r0 - 1, c0 + 1 + c_b)
        a_bot, b_bot = (r1 + 1, c0 + 1 + c_a), (r1 + 1, c0 + 1 + c_b)
        outer_a = _join(_vpath(a_top[0], top_r, a_top[1]), _hpath(top_r, a_top[1], b_top[1]), _vpath(top_r, b_top[0], b_top[1]))
        outer_b = _join(_vpath(a_bot[0], bot_r, a_bot[1]), _hpath(bot_r, a_bot[1], b_bot[1]), _vpath(bot_r, b_bot[0], b_bot[1]))
        inner = [[(r, c_a) for r in range(H)], [(r, c_b) for r in range(H)]]
    else:
        r_a, r_b = sorted(random.sample(range(1, H - 1), 2))
        if r_b - r_a < 2:
            r_b = min(H - 2, r_a + 2)
        left_c = c0 - random.randint(2, 3)
        right_c = c1 + random.randint(2, 3)
        a_left, b_left = (r0 + 1 + r_a, c0 - 1), (r0 + 1 + r_b, c0 - 1)
        a_right, b_right = (r0 + 1 + r_a, c1 + 1), (r0 + 1 + r_b, c1 + 1)
        outer_a = _join(_hpath(a_left[0], a_left[1], left_c), _vpath(a_left[0], b_left[0], left_c), _hpath(b_left[0], left_c, b_left[1]))
        outer_b = _join(_hpath(a_right[0], a_right[1], right_c), _vpath(a_right[0], b_right[0], right_c), _hpath(b_right[0], right_c, b_right[1]))
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
        if _place_component(grid, comp_mask, coords, (int(color),), bg):
            return


def _gen_attempt() -> np.ndarray | None:
    bg = random.choice(_BG_CHOICES)
    bc = _BORDER_COLOR
    fill = bg if random.random() < _FILL_EQ_BG_PROB else random.choice([x for x in range(10) if x not in (bg, bc)])
    H = random.randint(*_INTERIOR_H_RANGE)
    W = random.randint(*_INTERIOR_W_RANGE)
    rows = random.randint(max(_GRID_SIZE_RANGE[0], H + _GRID_INTERIOR_PAD), _GRID_SIZE_RANGE[1])
    cols = random.randint(max(_GRID_SIZE_RANGE[0], W + _GRID_INTERIOR_PAD), _GRID_SIZE_RANGE[1])
    r0 = random.randint(_FRAME_MARGIN, rows - H - _FRAME_MARGIN_FAR)
    c0 = random.randint(_FRAME_MARGIN, cols - W - _FRAME_MARGIN_FAR)
    r1, c1 = r0 + H + 1, c0 + W + 1

    grid = np.full((rows, cols), bg, dtype=int)
    grid[r0, c0 : c1 + 1] = bc
    grid[r1, c0 : c1 + 1] = bc
    grid[r0 : r1 + 1, c0] = bc
    grid[r0 : r1 + 1, c1] = bc
    grid[r0 + 1 : r1, c0 + 1 : c1] = fill

    expected = np.full((H, W), fill, dtype=int)
    used_inner = np.zeros((H, W), dtype=bool)
    comp_mask = np.zeros_like(grid, dtype=bool)
    palette = [x for x in range(10) if x not in (bg, bc, fill)]
    random.shuffle(palette)
    color_i = 0

    n_components = random.randint(*_N_COMPONENTS_RANGE)
    remaining = n_components

    if remaining >= 3 and random.random() < _U_PAIR_PROB:
        orientation = random.choice(["vertical", "horizontal"])
        color = palette[color_i]
        color_i += 1
        inner_lines, outer_a, outer_b = _u_pair_specs(orientation, H, W, r0, c0, r1, c1)
        all_inner = [p for line in inner_lines for p in line]
        irs, ics = zip(*all_inner)
        if used_inner[irs, ics].any():
            return None
        if not _place_component(grid, comp_mask, outer_a, (int(color),), bg):
            return None
        if not _place_component(grid, comp_mask, outer_b, (int(color),), bg):
            return None
        expected[irs, ics] = int(color)
        used_inner[irs, ics] = True
        remaining -= 2

    corners = random.sample(["tl", "tr", "bl", "br"], k=remaining)
    for kind in corners:
        if color_i >= len(palette) - 1:
            return None
        chosen = None
        for _ in range(_L_COMPONENT_TRIES):
            width = _choose_width(H)
            spec = _fused_l_specs(kind, H, W, r0, c0, r1, c1, width)
            if spec is None:
                continue
            inner, outer, mask = spec
            if mask.any() and np.any(used_inner & mask):
                continue
            if len(set(outer)) != len(outer):
                continue
            if not _can_place(grid, outer, bg, comp_mask):
                continue
            chosen = (inner, outer, mask, width)
            break
        if chosen is None:
            return None

        inner, outer, mask, width = chosen
        if random.random() < _DUAL_COLOR_PROB:
            colors = (int(palette[color_i]), int(palette[color_i + 1]))
            color_i += 2
        else:
            colors = (int(palette[color_i]),)
            color_i += 1
        if not _place_component(grid, comp_mask, outer, colors, bg):
            return None
        _paint_thick_path(expected, inner, width, colors)
        used_inner |= mask

    for _ in range(random.randint(*_DISTRACTOR_COUNT_RANGE)):
        if color_i >= len(palette):
            break
        _add_distractor(grid, comp_mask, bg, (r0, c0, r1, c1), int(palette[color_i]))
        color_i += 1

    try:
        actual = main(grid)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(_GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid a6f40cea combo input")
