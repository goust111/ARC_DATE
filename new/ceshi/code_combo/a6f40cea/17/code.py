# Combo variant of a6f40cea (index 17)
# combo_spec: D5/2+D9/0
# Name: 双层框架含边输出
#
# Design analysis for the combo variant
# -------------------------------------
# This fusion combines two orthogonal changes, but they must interact through one
# consistent "effective frame" choice rather than being implemented independently.
#
# a) Concrete behavior trace:
#    Original task:
#    - One 1px frame at bbox rows r0..r1, cols c0..c1.
#    - Output is only the interior: shape (r1-r0-1, c1-c0-1).
#    - A top ring touch at (r0-1, c0+1+c) maps to output (0, c).
#    D9 alone:
#    - Replace the single frame with a double frame: outer border at r0..r1 and
#      inner border at r0+1..r1-1.
#    - The drawable region moves inward by one more cell, so the same top touch
#      maps to the innermost area cell (0, c) after subtracting (r0+2, c0+2).
#    This combo:
#    - Keep the D9 double-frame input, but apply the D5 "keep the border" idea to
#      the effective frame, which is the inner frame.
#    - Therefore output is the inner-frame crop including its border:
#      shape (r1-r0-1, c1-c0-1) = (H+2, W+2), not the outer frame crop.
#    - The same top touch now maps to (1, c+1): row/col 0 are reserved for the
#      preserved inner border, and the path still lives inside the innermost area.
#    - Visually, the input shows two nested borders, while the output keeps only
#      the inner border around the reconstructed paths.
#
# b) Complexity calibration:
#    Preserving the outer frame as well would make the crop decision too easy,
#    because the solver could just keep everything it detected. Preserving only the
#    inner frame forces one structural choice that is visible in every sample:
#    which layer acts as the output boundary. The outer border remains necessary
#    because the external open-rect components still attach to its outside ring.
#
# c) Implicit assumptions broken by the original code:
#    - `_find_framed_rect_uniform_fill()` no longer applies because the region
#      inside the outer border is not uniformly filled; it contains another border.
#    - The old crop origin (`r0+1`, `c0+1`) meant "first interior cell". Here
#      `r0+1`, `c0+1` mean "inner border origin", while the drawable cells begin at
#      `r0+2`, `c0+2`.
#    - `map_to_out()` cannot clamp ring touches onto output edge 0/-1 because those
#      cells are now preserved border pixels of the inner frame.
#    - `inside_corner()` and deferred stripe aggregation must treat rows/cols
#      1..outH-2 and 1..outW-2 as the usable interior band.
#    - `comp_mask` must exclude both frame colors; otherwise either border could be
#      mislabeled as an external component.
#    - The generator's oracle output cannot be the bare innermost fill from D9; it
#      must include the inner border and shift all expected path cells by +1,+1.
#
# d) Placement legality:
#    - The generator draws the outer frame, the adjacent inner frame, and then the
#      uniform innermost fill. External components are still placed only on `bg`,
#      checked for in-bounds placement and 4-disconnection via `_can_place()`.
#    - Ring-touch coordinates are chosen away from frame corners (`1..W-2` /
#      `1..H-2` in innermost coordinates), which guarantees projection onto the
#      inner-frame interior without ambiguous corner clamping.
#    - Expected path cells are painted only after shifting into the inner-frame crop,
#      so the preserved border is never overwritten.
#
# e) Visual observability:
#    A human sees a two-color nested rectangle in the input but only one of those
#    two borders in the output. Across 2-4 examples, the repeated pattern is that
#    the surviving border is always the inset one, and the path sits one cell
#    inside it. No hidden ordering by numeric color id is used.
#
# f) Visual cleanliness:
#    The two frame colors are reserved for structural roles only. External traces
#    remain outside the frame, distractors stay away from it, and the output has a
#    single neat preserved border rather than layered clutter.
#
# g) Change inferability:
#    There is no free-form color remapping. Colors keep semantic roles:
#    outer frame, inner frame, fill, and path colors. The critical observable rule
#    is spatial: keep the inner border, discard the outer border, and reconstruct
#    paths inside the inner border.
#
# Post-change solver assumptions audited against the generator:
# - Assumption: exactly one adjacent double frame exists and the inner border is a
#   full rectangle one cell inside the outer border.
#   Guarantee: `_paint_double_frame()` constructs exactly that object once.
# - Assumption: the innermost area is uniformly filled before path reconstruction.
#   Guarantee: the generator fills it uniformly and never paints inside the frame.
# - Assumption: external components touch only the outer ring and never reuse
#   `outer_bc` or `inner_bc`.
#   Guarantee: component colors are sampled from a palette excluding bg, fill, and
#   both border colors.
# - Assumption: projected endpoint coordinates that should form deferred stripes land
#   on the inner drawable edge band (`row/col == 1 or out-2`).
#   Guarantee: generator-side inner paths always start/end on the innermost area's
#   perimeter before the +1 shift into the framed output.

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
    Detect the main adjacent double frame with a uniform innermost fill.

    Returns (r0, c0, r1, c1, outer_border_color, inner_border_color, fill_color)
    for the largest valid outer rectangle, else None.
    """
    if bg is None:
        bg = int(get_bg(g))
    else:
        bg = int(bg)

    best = None  # (area, r0,c0,r1,c1, outer_bc, inner_bc, fill)
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


def _make_framed_output(height: int, width: int, border_color: int, fill_color: int) -> np.ndarray:
    """Create a uniform-filled crop whose outer ring is the preserved frame border."""
    out = np.full((int(height), int(width)), int(fill_color), dtype=int)
    out[0, :] = int(border_color)
    out[-1, :] = int(border_color)
    out[:, 0] = int(border_color)
    out[:, -1] = int(border_color)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reconstruct the hidden paths of a double-framed puzzle and crop the inner frame including its border.

    Input structure:
    - A uniform background, one adjacent double rectangular frame, and colored external
      components touching the outside ring of the outer frame.
    - The outer and inner frame layers use different colors, and the region inside the
      inner layer is uniformly filled.

    Rule:
    - Keep the inner frame as the output boundary, including its border pixels.
    - Project each outer ring-touch position through both frame layers onto the band just
      inside the preserved inner border.
    - Monochrome components paint solid Manhattan segments; two-color components paint
      alternating Manhattan paths; paired U-shapes still expand into full-span stripes.

    Key insight:
    - The outer frame tells you where the traces attach, but the inner frame is the one
      that survives in the output. The geometry of the paths is the same as in the base
      task after projection, except every valid path cell is shifted one step inward
      because the preserved inner border occupies the crop edge.
    """
    g = np.array(grid, dtype=int)
    H, W = g.shape
    bg = int(get_bg(g))

    found = _find_double_framed_rect(g, bg=int(bg))
    assert found is not None
    r0, c0, r1, c1, outer_bc, inner_bc, fill = found
    outH = int(r1 - r0 - 1)
    outW = int(c1 - c0 - 1)
    out = _make_framed_output(outH, outW, int(inner_bc), int(fill))

    def map_to_out(rr: int, cc: int) -> tuple[int, int]:
        # Output origin is the inner-frame bbox; row/col 0 and -1 are preserved border cells.
        r = max(1, min(outH - 2, int(rr - (r0 + 1))))
        c = max(1, min(outW - 2, int(cc - (c0 + 1))))
        return int(r), int(c)

    def build_manhattan(
        p1: tuple[int, int], p2: tuple[int, int], corner_a: bool
    ) -> list[tuple[int, int]]:
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

    def inside_corner(p1: tuple[int, int], p2: tuple[int, int], corner_a: bool) -> bool:
        (ra, ca), (rb, cb) = p1, p2
        if ra == rb or ca == cb:
            return True
        corner = (rb, ca) if corner_a else (ra, cb)
        rr, cc = corner
        return (1 < rr < outH - 2) and (1 < cc < outW - 2)

    comp_mask = (
        (g != int(bg))
        & (g != int(fill))
        & (g != int(outer_bc))
        & (g != int(inner_bc))
    )
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
        col = int(g[rr, cc])
        mr, mc = map_to_out(rr, cc)
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
            path_a = build_manhattan(p1, p2, True)
            path_b = build_manhattan(p1, p2, False)
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

            path_a = build_manhattan(a, b, True)
            path_b = build_manhattan(a, b, False)
            path = path_b if (inside_corner(a, b, False) and not inside_corner(a, b, True)) else path_a

            def paint_alt(path_pts: list[tuple[int, int]], start_color: int) -> None:
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
        tops = [p for p in pts if p[0] == 1]
        bots = [p for p in pts if p[0] == outH - 2]
        if tops and bots:
            for cc in sorted({c for _r, c in pts}):
                rs = [r for r, c in pts if c == cc]
                for rr in range(min(rs), max(rs) + 1):
                    out[int(rr), int(cc)] = int(col)
            continue
        lefts = [p for p in pts if p[1] == 1]
        rights = [p for p in pts if p[1] == outW - 2]
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
        path_a = build_manhattan(p1, p2, True)
        path_b = build_manhattan(p1, p2, False)
        chosen = path_b if (inside_corner(p1, p2, False) and not inside_corner(p1, p2, True)) else path_a
        for rr, cc in chosen:
            out[int(rr), int(cc)] = int(col)

    return out


# --------------- generator (a6 combo D5/2 + D9/0) ---------------

_OUTER_BORDER_COLOR = 3
_BG_CHOICES = [0, 1, 2, 4, 5, 6, 7, 8, 9]
_FILL_EQ_BG_PROB = 0.6
_INTERIOR_H_RANGE = (4, 13)
_INTERIOR_W_RANGE = (7, 14)
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
_GEN_RETRIES = 400


def _line(a: int, b: int) -> range:
    """Inclusive range from a to b."""
    step = 1 if b >= a else -1
    return range(a, b + step, step)


def _vpath(r0: int, r1: int, c: int) -> list[tuple[int, int]]:
    """Vertical segment from (r0, c) to (r1, c) inclusive."""
    return [(r, c) for r in _line(r0, r1)]


def _hpath(r: int, c0: int, c1: int) -> list[tuple[int, int]]:
    """Horizontal segment from (r, c0) to (r, c1) inclusive."""
    return [(r, c) for c in _line(c0, c1)]


def _join(*parts: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """Concatenate path segments, deduplicating shared endpoints."""
    out: list[tuple[int, int]] = []
    for part in parts:
        for p in part:
            if not out or out[-1] != p:
                out.append(p)
    return out


def _paint_path(grid: np.ndarray, path: list[tuple[int, int]], colors: tuple[int, ...]) -> None:
    """Paint a path with solid or alternating colors."""
    if not path:
        return
    rs, cs = zip(*path)
    if len(colors) == 1:
        grid[rs, cs] = colors[0]
    else:
        a, b = colors
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
    grid[r0, c0 : c1 + 1] = outer_bc
    grid[r1, c0 : c1 + 1] = outer_bc
    grid[r0 : r1 + 1, c0] = outer_bc
    grid[r0 : r1 + 1, c1] = outer_bc

    grid[r0 + 1, c0 + 1 : c1] = inner_bc
    grid[r1 - 1, c0 + 1 : c1] = inner_bc
    grid[r0 + 1 : r1, c0 + 1] = inner_bc
    grid[r0 + 1 : r1, c1 - 1] = inner_bc

    grid[r0 + 2 : r1 - 1, c0 + 2 : c1 - 1] = fill


def _l_specs(kind: str, H: int, W: int, r0: int, c0: int, r1: int, c1: int):
    """Build inner coords and one outer open-rect for an L-component."""
    is_top = kind[0] == "t"
    is_left = kind[1] == "l"

    r = random.randint(1, max(1, H // 2)) if is_top else random.randint(max(1, H // 2), H - 2)
    c = random.randint(1, max(1, W // 2)) if is_left else random.randint(max(1, W // 2), W - 2)

    lo, hi = _OUTER_OFFSET_RANGE
    R = (r0 - random.randint(lo, hi)) if is_top else (r1 + random.randint(lo, hi))
    C = (c0 - random.randint(lo, hi)) if is_left else (c1 + random.randint(lo, hi))

    ring_a = (r0 - 1 if is_top else r1 + 1, c0 + 2 + c)
    ring_b = (r0 + 2 + r, c0 - 1 if is_left else c1 + 1)

    edge_r = 0 if is_top else H - 1
    edge_c = 0 if is_left else W - 1
    inner = _join(_vpath(edge_r, r, c), _hpath(r, c, edge_c))

    outer = _join(
        _vpath(ring_a[0], R, ring_a[1]),
        _hpath(R, ring_a[1], C),
        _vpath(R, ring_b[0], C),
        _hpath(ring_b[0], C, ring_b[1]),
    )
    return inner, outer


def _u_pair_specs(orientation: str, H: int, W: int, r0: int, c0: int, r1: int, c1: int):
    """Build inner stripes and two outer U-shaped open-rects."""
    lo, hi = _OUTER_OFFSET_RANGE
    if orientation == "vertical":
        c_a, c_b = sorted(random.sample(range(1, W - 1), 2))
        if c_b - c_a < 2:
            c_b = min(W - 2, c_a + 2)
        top_r = r0 - random.randint(lo, hi)
        bot_r = r1 + random.randint(lo, hi)
        a_top, b_top = (r0 - 1, c0 + 2 + c_a), (r0 - 1, c0 + 2 + c_b)
        a_bot, b_bot = (r1 + 1, c0 + 2 + c_a), (r1 + 1, c0 + 2 + c_b)
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
        r_a, r_b = sorted(random.sample(range(1, H - 1), 2))
        if r_b - r_a < 2:
            r_b = min(H - 2, r_a + 2)
        left_c = c0 - random.randint(lo, hi)
        right_c = c1 + random.randint(lo, hi)
        a_left, b_left = (r0 + 2 + r_a, c0 - 1), (r0 + 2 + r_b, c0 - 1)
        a_right, b_right = (r0 + 2 + r_a, c1 + 1), (r0 + 2 + r_b, c1 + 1)
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
    """Try to place a small random blob away from the frame."""
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
    """Single attempt to generate a valid input grid. Returns None on failure."""
    bg = random.choice(_BG_CHOICES)
    outer_bc = _OUTER_BORDER_COLOR
    fill_choices = [x for x in range(10) if x not in (bg, outer_bc)]
    fill = bg if random.random() < _FILL_EQ_BG_PROB else random.choice(fill_choices)
    inner_bc = random.choice([x for x in range(10) if x not in (bg, outer_bc, fill)])

    H = random.randint(*_INTERIOR_H_RANGE)
    W = random.randint(*_INTERIOR_W_RANGE)
    rows = random.randint(max(_GRID_SIZE_RANGE[0], H + _GRID_INTERIOR_PAD), _GRID_SIZE_RANGE[1])
    cols = random.randint(max(_GRID_SIZE_RANGE[0], W + _GRID_INTERIOR_PAD), _GRID_SIZE_RANGE[1])
    r0 = random.randint(_FRAME_MARGIN, rows - (H + 4) - _FRAME_MARGIN_FAR)
    c0 = random.randint(_FRAME_MARGIN, cols - (W + 4) - _FRAME_MARGIN_FAR)
    r1, c1 = r0 + H + 3, c0 + W + 3

    grid = np.full((rows, cols), bg, dtype=int)
    _paint_double_frame(grid, r0, c0, r1, c1, outer_bc, inner_bc, fill)

    expected = _make_framed_output(H + 2, W + 2, int(inner_bc), int(fill))
    comp_mask = np.zeros_like(grid, dtype=bool)
    palette = [x for x in range(10) if x not in (bg, outer_bc, inner_bc, fill)]
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
        inner_lines, outer_a, outer_b = _u_pair_specs(orientation, H, W, r0, c0, r1, c1)
        all_inner = [p for line in inner_lines for p in line]
        irs, ics = zip(*all_inner)
        if used_inner[irs, ics].any():
            return None
        if not _place_component(grid, comp_mask, outer_a, (color,), bg):
            return None
        if not _place_component(grid, comp_mask, outer_b, (color,), bg):
            return None
        expected[tuple(int(r + 1) for r in irs), tuple(int(c + 1) for c in ics)] = color
        used_inner[irs, ics] = True
        remaining -= 2

    corners = random.sample(["tl", "tr", "bl", "br"], k=remaining)
    for kind in corners:
        inner, outer = _l_specs(kind, H, W, r0, c0, r1, c1)
        irs, ics = zip(*inner)
        if used_inner[irs, ics].any():
            return None
        use_dual = random.random() < _DUAL_COLOR_PROB
        need = 2 if use_dual else 1
        if color_i + need > len(palette):
            return None
        colors = (palette[color_i], palette[color_i + 1]) if use_dual else (palette[color_i],)
        color_i += need
        if not _place_component(grid, comp_mask, outer, colors, bg):
            return None
        _paint_path(expected, _shift_path(inner, 1, 1), colors)
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
    return grid


def generate_input() -> np.ndarray:
    for _ in range(_GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid a6f40cea combo input")
