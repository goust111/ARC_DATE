# Combo variant of a6f40cea (index 34)
# combo_spec: D5/2+D9/0+D16/2
# Name: 双框含边比例投影
#
# Design analysis
# ---------------
# a) Concrete behavior trace:
#    The original task has a single frame, crops away that frame, and maps each
#    ring-touch by simple translation/clamping. The D9 reference turns that into
#    an adjacent double frame: outer border, inner border, then the drawable core.
#    D5 then changes the crop so the output keeps both border layers. D16 changes
#    the endpoint projection itself: the ring touches are no longer interpreted as
#    "same offset but shifted inward", but as coordinates on a larger outer side
#    that must be scaled onto the shorter innermost side.
#    Example: if the drawable width is 9, the outer frame width is 13, so the
#    usable top-side ring has 11 positions (`c0+1 .. c1-1`) while the innermost
#    top edge has only 9 cells. A touch at side-index 1 used to map directly to
#    the first drawable column; here it scales to column round(1 * 8 / 10) = 1,
#    i.e. the second drawable column. In output coordinates that means:
#      - old single-frame style: path endpoint would have been near the crop edge
#      - D9 only: endpoint lands on innermost edge after a 2-cell inward shift
#      - fused D5+D9+D16: endpoint still lands on that innermost edge, but at a
#        proportionally rescaled lateral position, while the 2 border rings stay
#        visible around it.
# b) Complexity calibration:
#    The variant keeps the original task's recognizable structure: 2-3 external
#    open-rect components, optional paired U traces, monochrome or alternating
#    colors, and distractors. The only added reasoning burden is identifying the
#    double frame and using side-wise proportional projection instead of direct
#    translation. This is visually obvious without exploding the number of cases.
# c) Implicit assumptions broken by the fusion:
#    - The original frame finder assumed a uniform fill immediately inside one
#      border ring. That fails because the outer frame now contains another frame.
#    - The old crop shape dropped all frame pixels; D5 requires preserving both
#      border layers, so path cells must stay inside `[2:-2, 2:-2]`.
#    - The old `map_to_out()` was side-agnostic subtraction + clamp. D16 breaks
#      that: top/bottom touches must scale across width, left/right touches across
#      height, and the mapping depends on which side the ring touch lies on.
#    - The original ring scan included every cell adjacent to the frame. Here the
#      scaling model is defined on side positions excluding outer corners, so the
#      solver and generator both use those side bands (`c0+1..c1-1`, `r0+1..r1-1`).
#    - The generator can no longer derive ring touches by adding a fixed offset to
#      the intended inner endpoints; it must invert the scaling map by choosing a
#      source side-index whose scaled image is the desired innermost endpoint.
# d) Placement legality:
#    Generation remains output-first. It first chooses inner paths in the HxW
#    innermost region, then chooses ring-touch side-indices that scale back to the
#    intended endpoints, then builds legal external open rectangles from those ring
#    touches. `_place_component()` still enforces in-bounds, no overlap, and no
#    4-adjacency between distinct external components. Distractors stay away from
#    the frame. The frame colors are never reused by components.
# e) Visual observability:
#    A human solver sees a two-color nested rectangle in every input, and the
#    output clearly preserves that full double border. Across examples, some outer
#    stubs sit nearer to outer-frame corners than a direct one-to-one projection
#    would predict, yet the drawn inner endpoints remain consistent only under a
#    scaled mapping. That makes the D16 change inspectable rather than hidden.
# f) Visual cleanliness:
#    The frame occupies one clean central object; paths stay only in the innermost
#    region; the preserved borders separate "decoration" from "path". External
#    components remain sparse and non-touching, so the grid does not collapse into
#    overlapping cues.
# g) Color-change inferability:
#    No hidden color remapping is introduced. Colors still express roles directly:
#    background, outer border, inner border, fill, and component trace colors.
# h) Anti-degeneration policy:
#    The generator rejects any sample whose scaled solver output equals the output
#    of the same double-frame solver with the old direct projection rule. This is
#    a quantitative guard: at least one projected endpoint must actually move due
#    to proportional scaling, not merely due to border preservation.
# i) New solver assumptions audited against generation:
#    1. There is exactly one large adjacent double frame with uniform innermost
#       fill. The generator draws exactly one such object.
#    2. External meaningful ring touches lie on side bands excluding outer-corner
#       adjacency. The generator only places them there.
#    3. The scaling map is invertible enough for all chosen endpoints. Generation
#       chooses each ring touch from the exact preimage set of the intended inner
#       endpoint under the same scaling formula used by the solver.
#    4. Alternating two-color components still have deterministic endpoint phase.
#       The generator paints the outer path in alternating order, and the solver
#       still recovers the phase from the colors seen at the two ring touches.

from __future__ import annotations

from collections import Counter, defaultdict
import random

import numpy as np
from scipy.ndimage import label

from arc_utils import STRUCT4, get_bg, neighbors4


def _is_rect_border(g: np.ndarray, r0: int, c0: int, r1: int, c1: int, color: int) -> bool:
    """Return True iff the bbox perimeter is completely filled with `color`."""
    return (
        (g[r0, c0 : c1 + 1] == int(color)).all()
        and (g[r1, c0 : c1 + 1] == int(color)).all()
        and (g[r0 : r1 + 1, c0] == int(color)).all()
        and (g[r0 : r1 + 1, c1] == int(color)).all()
    )


def _find_double_framed_rect(
    g: np.ndarray,
    *,
    bg: int | None = None,
    structure=STRUCT4,
) -> tuple[int, int, int, int, int, int, int] | None:
    """
    Detect the main adjacent double frame with uniform innermost fill.

    Returns (r0, c0, r1, c1, outer_border_color, inner_border_color, fill_color)
    for the largest valid outer rectangle, else None.
    """
    if bg is None:
        bg = int(get_bg(g))
    else:
        bg = int(bg)

    best = None  # (area, r0, c0, r1, c1, outer_bc, inner_bc, fill)
    for outer_bc in sorted(set(map(int, g.flatten())) - {int(bg)}):
        lab, n = label(g == int(outer_bc), structure=structure)
        for cid in range(1, int(n) + 1):
            pts = np.argwhere(lab == int(cid))
            if int(len(pts)) < 12:
                continue

            r0, c0 = pts.min(axis=0)
            r1, c1 = pts.max(axis=0)
            r0, c0, r1, c1 = map(int, (r0, c0, r1, c1))
            h = int(r1 - r0 + 1)
            w = int(c1 - c0 + 1)
            if h < 8 or w < 8:
                continue

            per = int(2 * (h + w) - 4)
            if int(len(pts)) != per:
                continue
            if not _is_rect_border(g, r0, c0, r1, c1, int(outer_bc)):
                continue

            ir0, ic0, ir1, ic1 = int(r0 + 1), int(c0 + 1), int(r1 - 1), int(c1 - 1)
            if ir1 - ir0 + 1 < 4 or ic1 - ic0 + 1 < 4:
                continue
            inner_bc = int(g[ir0, ic0])
            if inner_bc in (int(bg), int(outer_bc)):
                continue
            if not _is_rect_border(g, ir0, ic0, ir1, ic1, int(inner_bc)):
                continue

            inner = g[r0 + 2 : r1 - 1, c0 + 2 : c1 - 1]
            if inner.size == 0:
                continue
            fill = int(Counter(inner.flatten()).most_common(1)[0][0])
            if fill in (int(outer_bc), int(inner_bc)):
                continue
            if not (inner == int(fill)).all():
                continue

            area = int(h * w)
            cand = (area, r0, c0, r1, c1, int(outer_bc), int(inner_bc), int(fill))
            if best is None or cand[0] > best[0]:
                best = cand

    if best is None:
        return None
    _area, r0, c0, r1, c1, outer_bc, inner_bc, fill = best
    return int(r0), int(c0), int(r1), int(c1), int(outer_bc), int(inner_bc), int(fill)


def _make_double_framed_output(
    height: int,
    width: int,
    outer_bc: int,
    inner_bc: int,
    fill: int,
) -> np.ndarray:
    """Create the cropped output that keeps both border layers."""
    out = np.full((int(height), int(width)), int(fill), dtype=int)
    out[0, :] = int(outer_bc)
    out[-1, :] = int(outer_bc)
    out[:, 0] = int(outer_bc)
    out[:, -1] = int(outer_bc)
    out[1, 1:-1] = int(inner_bc)
    out[-2, 1:-1] = int(inner_bc)
    out[1:-1, 1] = int(inner_bc)
    out[1:-1, -2] = int(inner_bc)
    return out


def _scale_index(src_idx: int, src_len: int, dst_len: int) -> int:
    """Half-up rounding for 0..src_len-1 -> 0..dst_len-1 proportional scaling."""
    src_idx = int(src_idx)
    src_len = int(src_len)
    dst_len = int(dst_len)
    if dst_len <= 1 or src_len <= 1:
        return 0
    num = 2 * int(src_idx) * int(dst_len - 1) + int(src_len - 1)
    den = 2 * int(src_len - 1)
    return int(num // den)


def _project_side_index(side_idx: int, inner_len: int, *, scaled: bool) -> int:
    """Map a side-band index onto the innermost edge index."""
    side_idx = int(side_idx)
    inner_len = int(inner_len)
    if scaled:
        return _scale_index(int(side_idx), int(inner_len + 2), int(inner_len))
    return max(0, min(int(inner_len - 1), int(side_idx - 1)))


def _map_ring_to_output(
    rr: int,
    cc: int,
    *,
    r0: int,
    c0: int,
    r1: int,
    c1: int,
    inner_h: int,
    inner_w: int,
    out_h: int,
    out_w: int,
    scaled: bool,
) -> tuple[int, int]:
    """Project one ring-touch cell onto the innermost boundary inside the cropped output."""
    if rr == int(r0 - 1) and int(c0) < int(cc) < int(c1):
        side_idx = int(cc - (c0 + 1))
        return 2, int(2 + _project_side_index(side_idx, inner_w, scaled=scaled))
    if rr == int(r1 + 1) and int(c0) < int(cc) < int(c1):
        side_idx = int(cc - (c0 + 1))
        return int(out_h - 3), int(2 + _project_side_index(side_idx, inner_w, scaled=scaled))
    if cc == int(c0 - 1) and int(r0) < int(rr) < int(r1):
        side_idx = int(rr - (r0 + 1))
        return int(2 + _project_side_index(side_idx, inner_h, scaled=scaled)), 2
    if cc == int(c1 + 1) and int(r0) < int(rr) < int(r1):
        side_idx = int(rr - (r0 + 1))
        return int(2 + _project_side_index(side_idx, inner_h, scaled=scaled)), int(out_w - 3)
    raise ValueError(f"ring touch {(rr, cc)} is not on a usable outer-frame side band")


def _build_manhattan(
    p1: tuple[int, int],
    p2: tuple[int, int],
    corner_a: bool,
) -> list[tuple[int, int]]:
    """Build a straight or L-shaped Manhattan path between two points."""
    (ra, ca), (rb, cb) = p1, p2
    if ra == rb:
        step = 1 if cb >= ca else -1
        return [(int(ra), int(c)) for c in range(int(ca), int(cb) + step, step)]
    if ca == cb:
        step = 1 if rb >= ra else -1
        return [(int(r), int(ca)) for r in range(int(ra), int(rb) + step, step)]
    if corner_a:
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


def _solve(grid: np.ndarray, *, scaled: bool) -> np.ndarray:
    """Shared solver core used both by `main()` and by generator anti-degeneration checks."""
    g = np.array(grid, dtype=int)
    H, W = g.shape
    bg = int(get_bg(g))

    found = _find_double_framed_rect(g, bg=int(bg))
    assert not (found is None)
    r0, c0, r1, c1, outer_bc, inner_bc, fill = found
    inner_h = int(r1 - r0 - 3)
    inner_w = int(c1 - c0 - 3)
    out_h = int(r1 - r0 + 1)
    out_w = int(c1 - c0 + 1)
    out = _make_double_framed_output(out_h, out_w, int(outer_bc), int(inner_bc), int(fill))

    def inside_corner(p1: tuple[int, int], p2: tuple[int, int], corner_a: bool) -> bool:
        (ra, ca), (rb, cb) = p1, p2
        if ra == rb or ca == cb:
            return True
        corner = (rb, ca) if corner_a else (ra, cb)
        rr, cc = corner
        return (2 < rr < out_h - 3) and (2 < cc < out_w - 3)

    comp_mask = (
        (g != int(bg))
        & (g != int(fill))
        & (g != int(outer_bc))
        & (g != int(inner_bc))
    )
    clab, cn = label(comp_mask, structure=STRUCT4)

    ring = set()
    for cc in range(c0 + 1, c1):
        for rr in (r0 - 1, r1 + 1):
            if 0 <= rr < H:
                ring.add((int(rr), int(cc)))
    for rr in range(r0 + 1, r1):
        for cc in (c0 - 1, c1 + 1):
            if 0 <= cc < W:
                ring.add((int(rr), int(cc)))

    comp_pts: dict[int, list[tuple[int, int, int]]] = defaultdict(list)
    for rr, cc in ring:
        cid = int(clab[rr, cc])
        if cid == 0:
            continue
        col = int(g[rr, cc])
        mr, mc = _map_ring_to_output(
            rr,
            cc,
            r0=int(r0),
            c0=int(c0),
            r1=int(r1),
            c1=int(c1),
            inner_h=int(inner_h),
            inner_w=int(inner_w),
            out_h=int(out_h),
            out_w=int(out_w),
            scaled=bool(scaled),
        )
        comp_pts[cid].append((mr, mc, col))

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
        for rr, cc, col in pts:
            hist[(int(rr), int(cc))][int(col)] += 1

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
            path_a = _build_manhattan(p1, p2, True)
            path_b = _build_manhattan(p1, p2, False)
            chosen = path_b if (inside_corner(p1, p2, False) and not inside_corner(p1, p2, True)) else path_a
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

            path_a = _build_manhattan(a, b, True)
            path_b = _build_manhattan(a, b, False)
            path = path_b if (inside_corner(a, b, False) and not inside_corner(a, b, True)) else path_a

            def paint_alt(path_pts: list[tuple[int, int]], start_color: int):
                cur = int(start_color)
                for rr, cc in path_pts:
                    out[int(rr), int(cc)] = int(cur)
                    cur = int(cb) if cur == int(ca) else int(ca)

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
            paint_alt(path_pts, start_color)
            continue

    for col, pts in deferred.items():
        pts = list(pts)
        if not pts:
            continue
        tops = [p for p in pts if p[0] == 2]
        bots = [p for p in pts if p[0] == out_h - 3]
        if tops and bots:
            for cc in sorted({c for _r, c in pts}):
                rs = [r for r, c in pts if c == cc]
                for rr in range(min(rs), max(rs) + 1):
                    out[int(rr), int(cc)] = int(col)
            continue
        lefts = [p for p in pts if p[1] == 2]
        rights = [p for p in pts if p[1] == out_w - 3]
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
        path_a = _build_manhattan(p1, p2, True)
        path_b = _build_manhattan(p1, p2, False)
        chosen = path_b if (inside_corner(p1, p2, False) and not inside_corner(p1, p2, True)) else path_a
        for rr, cc in chosen:
            out[int(rr), int(cc)] = int(col)

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reconstruct hidden Manhattan paths inside a nested double frame, keeping both border layers.

    Input structure:
    - A uniform background, one large adjacent double frame, and a uniformly filled
      innermost region bounded by the inner frame.
    - Colored open-rectangle components lie outside the frame and touch the usable
      side bands just beyond the outer border.

    Rule:
    - Output the full crop of the double frame, preserving both the outer and inner
      border colors while leaving the paths only in the innermost region.
    - Project each ring touch onto the innermost boundary by proportional scaling
      along that side, not by direct offset preservation.
    - Monochrome components paint solid Manhattan paths or full-span stripes; two-color
      components paint alternating Manhattan paths whose phase is set by endpoint colors.

    Key insight:
    - The external traces attach to the outer frame, but the drawable target lives
      two layers deeper; once the solver recognizes that gap, the remaining puzzle is
      to rescale side positions from the longer outer side bands onto the shorter
      innermost side and then reuse the original path logic.
    """
    return _solve(np.array(grid, dtype=int), scaled=True)


# --------------- generator (combo 34) ---------------

_OUTER_BORDER_COLOR = 3
_BG_CHOICES = [0, 1, 2, 4, 5, 6, 7, 8, 9]
_FILL_EQ_BG_PROB = 0.55
_INNER_H_RANGE = (4, 13)
_INNER_W_RANGE = (7, 14)
_GRID_SIZE_RANGE = (20, 30)
_GRID_INTERIOR_PAD = 14
_FRAME_MARGIN = 4
_FRAME_MARGIN_FAR = 6
_OUTER_OFFSET_RANGE = (2, 3)
_N_COMPONENTS_RANGE = (2, 3)
_U_PAIR_PROB = 0.25
_DUAL_COLOR_PROB = 0.4
_DISTRACTOR_COUNT_RANGE = (0, 1)
_DISTRACTOR_SIZE_RANGE = (1, 4)
_DISTRACTOR_TRIES = 40
_DISTRACTOR_CLEARANCE = 2
_NONCANONICAL_BIAS = 0.8
_GEN_RETRIES = 500


def _line(a: int, b: int) -> range:
    """Inclusive range from a to b."""
    step = 1 if b >= a else -1
    return range(a, b + step, step)


def _vpath(r0: int, r1: int, c: int) -> list[tuple[int, int]]:
    """Vertical segment from (r0, c) to (r1, c) inclusive."""
    return [(int(r), int(c)) for r in _line(r0, r1)]


def _hpath(r: int, c0: int, c1: int) -> list[tuple[int, int]]:
    """Horizontal segment from (r, c0) to (r, c1) inclusive."""
    return [(int(r), int(c)) for c in _line(c0, c1)]


def _join(*parts: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """Concatenate path segments, deduplicating shared endpoints."""
    out: list[tuple[int, int]] = []
    for part in parts:
        for p in part:
            if not out or out[-1] != p:
                out.append((int(p[0]), int(p[1])))
    return out


def _paint_path(grid: np.ndarray, path: list[tuple[int, int]], colors: tuple[int, ...]) -> None:
    """Paint a path with solid or alternating colors."""
    if not path:
        return
    rs, cs = zip(*path)
    if len(colors) == 1:
        grid[rs, cs] = int(colors[0])
    else:
        a, b = map(int, colors)
        vals = [a if i % 2 == 0 else b for i in range(len(path))]
        grid[rs, cs] = vals


def _shift_path(path: list[tuple[int, int]], dr: int, dc: int) -> list[tuple[int, int]]:
    """Translate a path by a fixed offset."""
    return [(int(r + dr), int(c + dc)) for r, c in path]


def _can_place(grid: np.ndarray, coords: list[tuple[int, int]], bg: int, comp_mask: np.ndarray) -> bool:
    """Check that all coords are in-bounds, on bg, and not 4-adjacent to existing components."""
    H, W = grid.shape
    s = set(coords)
    for r, c in s:
        if not (0 <= r < H and 0 <= c < W):
            return False
        if grid[r, c] != int(bg):
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
    """Validate, paint, and register a component. Returns True on success."""
    if len(set(path)) != len(path):
        return False
    if not _can_place(grid, path, bg, comp_mask):
        return False
    _paint_path(grid, path, colors)
    rs, cs = zip(*path)
    comp_mask[rs, cs] = True
    return True


def _paint_double_frame(
    grid: np.ndarray,
    r0: int,
    c0: int,
    r1: int,
    c1: int,
    outer_bc: int,
    inner_bc: int,
    fill: int,
) -> None:
    """Draw the adjacent double frame and its uniform innermost fill."""
    grid[r0, c0 : c1 + 1] = int(outer_bc)
    grid[r1, c0 : c1 + 1] = int(outer_bc)
    grid[r0 : r1 + 1, c0] = int(outer_bc)
    grid[r0 : r1 + 1, c1] = int(outer_bc)

    grid[r0 + 1, c0 + 1 : c1] = int(inner_bc)
    grid[r1 - 1, c0 + 1 : c1] = int(inner_bc)
    grid[r0 + 1 : r1, c0 + 1] = int(inner_bc)
    grid[r0 + 1 : r1, c1 - 1] = int(inner_bc)

    grid[r0 + 2 : r1 - 1, c0 + 2 : c1 - 1] = int(fill)


def _preimages_for_target(target_idx: int, inner_len: int) -> list[int]:
    """All side-band indices that scale to `target_idx` for a given innermost side length."""
    return [
        int(src_idx)
        for src_idx in range(int(inner_len + 2))
        if _project_side_index(int(src_idx), int(inner_len), scaled=True) == int(target_idx)
    ]


def _choose_source_index(target_idx: int, inner_len: int, *, prefer_shift: bool) -> tuple[int, bool]:
    """Choose one inverse-mapped side index, preferring a non-direct preimage when available."""
    options = _preimages_for_target(int(target_idx), int(inner_len))
    assert options, "every target endpoint must have at least one scaling preimage"
    canonical = int(target_idx + 1)
    shifted = [src for src in options if int(src) != int(canonical)]
    if prefer_shift and shifted:
        chosen = random.choice(shifted)
    elif shifted and random.random() < _NONCANONICAL_BIAS:
        chosen = random.choice(shifted)
    else:
        chosen = random.choice(options)
    return int(chosen), int(chosen) != int(canonical)


def _ring_coord_from_side(side: str, side_idx: int, r0: int, c0: int, r1: int, c1: int) -> tuple[int, int]:
    """Convert a side-band index back into an absolute ring-touch coordinate."""
    side_idx = int(side_idx)
    if side == "top":
        return int(r0 - 1), int(c0 + 1 + side_idx)
    if side == "bottom":
        return int(r1 + 1), int(c0 + 1 + side_idx)
    if side == "left":
        return int(r0 + 1 + side_idx), int(c0 - 1)
    if side == "right":
        return int(r0 + 1 + side_idx), int(c1 + 1)
    raise ValueError(f"unknown side {side}")


def _l_specs(
    kind: str,
    inner_h: int,
    inner_w: int,
    r0: int,
    c0: int,
    r1: int,
    c1: int,
) -> tuple[list[tuple[int, int]], list[tuple[int, int]], bool]:
    """Build inner L-path and one scaled-projection outer open rectangle."""
    is_top = kind[0] == "t"
    is_left = kind[1] == "l"

    r = random.randint(1, max(1, inner_h // 2)) if is_top else random.randint(max(1, inner_h // 2), inner_h - 2)
    c = random.randint(1, max(1, inner_w // 2)) if is_left else random.randint(max(1, inner_w // 2), inner_w - 2)

    edge_r = 0 if is_top else inner_h - 1
    edge_c = 0 if is_left else inner_w - 1
    inner = _join(_vpath(edge_r, r, c), _hpath(r, c, edge_c))

    shifted_any = False
    side_idx_a, shifted = _choose_source_index(int(c), int(inner_w), prefer_shift=True)
    shifted_any |= shifted
    side_idx_b, shifted = _choose_source_index(int(r), int(inner_h), prefer_shift=not shifted_any)
    shifted_any |= shifted

    side_a = "top" if is_top else "bottom"
    side_b = "left" if is_left else "right"
    ring_a = _ring_coord_from_side(side_a, side_idx_a, r0, c0, r1, c1)
    ring_b = _ring_coord_from_side(side_b, side_idx_b, r0, c0, r1, c1)

    lo, hi = _OUTER_OFFSET_RANGE
    R = (r0 - random.randint(lo, hi)) if is_top else (r1 + random.randint(lo, hi))
    C = (c0 - random.randint(lo, hi)) if is_left else (c1 + random.randint(lo, hi))
    outer = _join(
        _vpath(ring_a[0], R, ring_a[1]),
        _hpath(R, ring_a[1], C),
        _vpath(R, ring_b[0], C),
        _hpath(ring_b[0], C, ring_b[1]),
    )
    return inner, outer, bool(shifted_any)


def _u_pair_specs(
    orientation: str,
    inner_h: int,
    inner_w: int,
    r0: int,
    c0: int,
    r1: int,
    c1: int,
) -> tuple[list[list[tuple[int, int]]], list[tuple[int, int]], list[tuple[int, int]], bool]:
    """Build inner stripes and two scaled-projection outer U-shaped open rectangles."""
    shifted_any = False
    lo, hi = _OUTER_OFFSET_RANGE
    if orientation == "vertical":
        c_a, c_b = sorted(random.sample(range(1, inner_w - 1), 2))
        if c_b - c_a < 2:
            c_b = min(inner_w - 2, c_a + 2)
        idx_a, shifted = _choose_source_index(int(c_a), int(inner_w), prefer_shift=True)
        shifted_any |= shifted
        idx_b, shifted = _choose_source_index(int(c_b), int(inner_w), prefer_shift=not shifted_any)
        shifted_any |= shifted
        a_top = _ring_coord_from_side("top", idx_a, r0, c0, r1, c1)
        b_top = _ring_coord_from_side("top", idx_b, r0, c0, r1, c1)
        a_bot = _ring_coord_from_side("bottom", idx_a, r0, c0, r1, c1)
        b_bot = _ring_coord_from_side("bottom", idx_b, r0, c0, r1, c1)
        top_r = r0 - random.randint(lo, hi)
        bot_r = r1 + random.randint(lo, hi)
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
        inner = [[(int(r), int(c_a)) for r in range(inner_h)], [(int(r), int(c_b)) for r in range(inner_h)]]
        return inner, outer_a, outer_b, bool(shifted_any)

    r_a, r_b = sorted(random.sample(range(1, inner_h - 1), 2))
    if r_b - r_a < 2:
        r_b = min(inner_h - 2, r_a + 2)
    idx_a, shifted = _choose_source_index(int(r_a), int(inner_h), prefer_shift=True)
    shifted_any |= shifted
    idx_b, shifted = _choose_source_index(int(r_b), int(inner_h), prefer_shift=not shifted_any)
    shifted_any |= shifted
    a_left = _ring_coord_from_side("left", idx_a, r0, c0, r1, c1)
    b_left = _ring_coord_from_side("left", idx_b, r0, c0, r1, c1)
    a_right = _ring_coord_from_side("right", idx_a, r0, c0, r1, c1)
    b_right = _ring_coord_from_side("right", idx_b, r0, c0, r1, c1)
    left_c = c0 - random.randint(lo, hi)
    right_c = c1 + random.randint(lo, hi)
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
    inner = [[(int(r_a), int(c)) for c in range(inner_w)], [(int(r_b), int(c)) for c in range(inner_w)]]
    return inner, outer_a, outer_b, bool(shifted_any)


def _add_distractor(
    grid: np.ndarray,
    comp_mask: np.ndarray,
    bg: int,
    frame: tuple[int, int, int, int],
    color: int,
) -> None:
    """Try to place a small random blob away from the frame."""
    H, W = grid.shape
    r0, c0, r1, c1 = frame
    gap = _DISTRACTOR_CLEARANCE
    sz_lo, sz_hi = _DISTRACTOR_SIZE_RANGE
    for _ in range(_DISTRACTOR_TRIES):
        size = random.randint(sz_lo, sz_hi)
        r, c = random.randint(1, H - 2), random.randint(1, W - 2)
        coords = [(int(r), int(c))]
        used = {(int(r), int(c))}
        for _ in range(size - 1):
            base = random.choice(coords)
            opts = [
                (int(nr), int(nc))
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
        if _place_component(grid, comp_mask, coords, (int(color),), int(bg)):
            return


def _gen_attempt() -> np.ndarray | None:
    """Single output-first generation attempt. Returns a valid input grid or None."""
    bg = random.choice(_BG_CHOICES)
    outer_bc = _OUTER_BORDER_COLOR
    fill_candidates = [x for x in range(10) if x not in (bg, outer_bc)]
    fill = bg if random.random() < _FILL_EQ_BG_PROB else random.choice(fill_candidates)
    inner_bc = random.choice([x for x in range(10) if x not in (bg, outer_bc, fill)])

    inner_h = random.randint(*_INNER_H_RANGE)
    inner_w = random.randint(*_INNER_W_RANGE)
    rows = random.randint(max(_GRID_SIZE_RANGE[0], inner_h + _GRID_INTERIOR_PAD), _GRID_SIZE_RANGE[1])
    cols = random.randint(max(_GRID_SIZE_RANGE[0], inner_w + _GRID_INTERIOR_PAD), _GRID_SIZE_RANGE[1])
    r0 = random.randint(_FRAME_MARGIN, rows - (inner_h + 4) - _FRAME_MARGIN_FAR)
    c0 = random.randint(_FRAME_MARGIN, cols - (inner_w + 4) - _FRAME_MARGIN_FAR)
    r1, c1 = r0 + inner_h + 3, c0 + inner_w + 3

    grid = np.full((rows, cols), int(bg), dtype=int)
    _paint_double_frame(grid, r0, c0, r1, c1, int(outer_bc), int(inner_bc), int(fill))

    expected = _make_double_framed_output(inner_h + 4, inner_w + 4, int(outer_bc), int(inner_bc), int(fill))
    comp_mask = np.zeros_like(grid, dtype=bool)
    palette = [x for x in range(10) if x not in (bg, outer_bc, inner_bc, fill)]
    random.shuffle(palette)
    used_inner = np.zeros((inner_h, inner_w), dtype=bool)
    used_scaled_projection = False

    n_components = random.randint(*_N_COMPONENTS_RANGE)
    remaining = n_components
    color_i = 0

    if remaining >= 2 and random.random() < _U_PAIR_PROB:
        orientation = random.choice(["vertical", "horizontal"])
        if orientation == "horizontal" and inner_h < 5:
            orientation = "vertical"
        if color_i >= len(palette):
            return None
        color = int(palette[color_i])
        color_i += 1
        inner_lines, outer_a, outer_b, shifted = _u_pair_specs(orientation, inner_h, inner_w, r0, c0, r1, c1)
        all_inner = [p for line in inner_lines for p in line]
        irs, ics = zip(*all_inner)
        if used_inner[irs, ics].any():
            return None
        if not _place_component(grid, comp_mask, outer_a, (int(color),), int(bg)):
            return None
        if not _place_component(grid, comp_mask, outer_b, (int(color),), int(bg)):
            return None
        _paint_path(expected, _shift_path(all_inner, 2, 2), (int(color),))
        used_inner[irs, ics] = True
        used_scaled_projection |= bool(shifted)
        remaining -= 2

    corners = random.sample(["tl", "tr", "bl", "br"], k=remaining)
    for kind in corners:
        inner, outer, shifted = _l_specs(kind, inner_h, inner_w, r0, c0, r1, c1)
        irs, ics = zip(*inner)
        if used_inner[irs, ics].any():
            return None
        use_dual = random.random() < _DUAL_COLOR_PROB
        need = 2 if use_dual else 1
        if color_i + need > len(palette):
            return None
        colors = tuple(int(x) for x in (palette[color_i], palette[color_i + 1])) if use_dual else (int(palette[color_i]),)
        color_i += need
        if not _place_component(grid, comp_mask, outer, colors, int(bg)):
            return None
        _paint_path(expected, _shift_path(inner, 2, 2), colors)
        used_inner[irs, ics] = True
        used_scaled_projection |= bool(shifted)

    for _ in range(random.randint(*_DISTRACTOR_COUNT_RANGE)):
        if color_i >= len(palette):
            break
        _add_distractor(grid, comp_mask, int(bg), (r0, c0, r1, c1), int(palette[color_i]))
        color_i += 1

    if not used_scaled_projection:
        return None

    try:
        actual = main(grid)
        naive = _solve(grid, scaled=False)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    if np.array_equal(actual, naive):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(_GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid a6f40cea combo-34 input")
