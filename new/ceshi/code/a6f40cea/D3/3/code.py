# Variant of a6f40cea
# Dimension: D3 — 源对象类型 (src)
# Change: 外部开放矩形框架（L形：2条完整边+2条截断stub；或U形：3条完整边），通过触碰闭合框边来标记端点位置... → 单边条形源

from __future__ import annotations

from collections import Counter, defaultdict
import random

import numpy as np
from scipy.ndimage import label

from arc_utils import STRUCT4, get_bg, neighbors4


# Design analysis for the D3 variant
# ----------------------------------
# a) Concrete behavior:
# - The large closed rectangle is unchanged: 1px border, uniform fill, cropped interior as output.
# - The old external "open rectangle" source is removed. Each source is now a single straight bar
#   placed on the ring just outside one frame side. The two bar endpoints are the only visible
#   endpoint cues, so a single bar can only specify a same-side interval.
# - Because a one-sided bar no longer exposes a corner/elbow, the recoverable inside geometry is
#   necessarily the straight Manhattan segment between the projected endpoints on that same side.
# - Full-span stripes are still possible: they are encoded by two monochrome bars of the same color
#   on opposite sides using the same interval. The solver already aggregates same-color deferred
#   endpoints into a complete stripe, so that logic is preserved rather than replaced.
# - Small trace example:
#   original: an outside L-shape touched top + left ring cells, so the solver chose an inside elbow
#   and painted an L-path.
#   variant: a top bar from columns 2..5 only touches the top ring, so the solver projects it to
#   interior points (0,2)..(0,5) and paints a straight top-edge segment. A matched top/bottom pair
#   of the same color over columns 2..5 yields vertical stripes through those columns.
#
# b) Complexity calibration:
# - Bars are simpler than open rectangles, so the generator keeps 2-3 logical groups and still mixes
#   three visible behaviors: single monochrome bars, single alternating-color bars, and opposite-side
#   monochrome pairs. This prevents the task from collapsing into one trivial case.
# - Color reuse is tightly controlled: only intentional opposite-side stripe pairs reuse a color.
#
# c) Implicit assumption review for main():
# - `_find_framed_rect_uniform_fill` depends only on the big closed frame and uniform fill, so it
#   remains valid.
# - `comp_mask = (g != bg) & (g != fill) & (g != bc)` assumes every informative outside source uses
#   colors distinct from bg/border/fill. The generator preserves that.
# - Ring sampling assumes relevant source cells lie on the one-cell ring outside the frame. Bars are
#   placed exactly there; distractors stay away.
# - The monochrome branch defers same-row / same-column touches and later merges them by color. This
#   is now a feature, not an accident: single bars become straight segments, opposite pairs become
#   full stripes.
# - The two-color branch assumes one connected component contains the full alternating trace whose
#   endpoint colors anchor the phase. Each alternating bar is generated as a single connected bar,
#   so the assumption still holds.
#
# d) Placement legality:
# - Bars are painted only on bg cells, must stay in bounds, and cannot overlap or become 4-adjacent
#   to another source component.
# - Opposite-side pair bars use the same interval so the solver reconstructs a clean stripe instead
#   of ragged partial columns/rows.
# - Output paths are checked with `used_inner` so different logical groups never overlap inside.
#
# e) Visual observability:
# - The source-type change is obvious: outside the frame there are only short horizontal/vertical bars,
#   never the old open rectangular outlines.
# - The mapping is inferable from 2-4 examples because bar span directly matches the output segment
#   span, and opposite matched bars visibly explain the through-going stripe case.
#
# f) Visual cleanliness:
# - Logical groups are non-overlapping inside the cropped output.
# - Distractors remain away from the frame ring so they do not masquerade as sources.
#
# g) Color inference:
# - No hidden ordering by color id is used.
# - Alternation is determined purely by visible endpoint colors on a single bar, matching the old
#   task's endpoint-phase principle.


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


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reconstruct the cropped interior by projecting one-sided external bars onto the frame.

    Input structure:
    - A background grid contains one closed 1px rectangular frame with uniform fill.
    - Informative sources are single horizontal or vertical bars placed on the ring outside one frame side.

    Rule:
    - Output only the frame interior, starting from the fill color.
    - Project each ring-touching bar cell into interior coordinates.
    - A lone monochrome bar becomes a solid straight segment on the corresponding interior edge.
    - A two-color bar becomes an alternating straight segment, and opposite monochrome bar pairs of one color become full stripes.

    Key insight:
    - The bar's span matters, not the missing outer rectangle: endpoint colors still fix alternating phase, and same-color opposite bars are merged by shared projected coordinates.
    """
    g = np.array(grid, dtype=int)
    H, W = g.shape
    bg = int(get_bg(g))

    found = _find_framed_rect_uniform_fill(g, bg=int(bg))
    assert not (found is None)
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

    def build_manhattan(p1: tuple[int, int], p2: tuple[int, int], cornerA: bool) -> list[tuple[int, int]]:
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

    def inside_corner(p1: tuple[int, int], p2: tuple[int, int], cornerA: bool) -> bool:
        (ra, ca), (rb, cb) = p1, p2
        if ra == rb or ca == cb:
            return True
        corner = (rb, ca) if cornerA else (ra, cb)
        rr, cc = corner
        return (0 < rr < outH - 1) and (0 < cc < outW - 1)

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
            uniq = coords
            rows = {r for r, _c in uniq}
            cols2 = {c for _r, c in uniq}
            if len(rows) == 1 or len(cols2) == 1:
                for p in uniq:
                    deferred[colv].add((int(p[0]), int(p[1])))
                continue

            best = None
            for i in range(len(uniq)):
                for j in range(i + 1, len(uniq)):
                    d = abs(uniq[i][0] - uniq[j][0]) + abs(uniq[i][1] - uniq[j][1])
                    if best is None or d > best[0]:
                        best = (d, uniq[i], uniq[j])
            if best is None:
                continue
            _d, p1, p2 = best
            pathA = build_manhattan(p1, p2, True)
            pathB = build_manhattan(p1, p2, False)
            chosen = pathB if (inside_corner(p1, p2, False) and not inside_corner(p1, p2, True)) else pathA
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

            pathA = build_manhattan(a, b, True)
            pathB = build_manhattan(a, b, False)
            path = pathB if (inside_corner(a, b, False) and not inside_corner(a, b, True)) else pathA

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
        pathA = build_manhattan(p1, p2, True)
        pathB = build_manhattan(p1, p2, False)
        chosen = pathB if (inside_corner(p1, p2, False) and not inside_corner(p1, p2, True)) else pathA
        for rr, cc in chosen:
            out[int(rr), int(cc)] = int(col)

    return out


_BORDER_COLOR = 3
_BG_CHOICES = [0, 1, 2, 4, 5, 6, 7, 8, 9]
_FILL_EQ_BG_PROB = 0.6
_INTERIOR_H_RANGE = (4, 13)
_INTERIOR_W_RANGE = (7, 14)
_GRID_SIZE_RANGE = (20, 30)
_GRID_INTERIOR_PAD = 10
_FRAME_MARGIN = 4
_FRAME_MARGIN_FAR = 6
_N_GROUPS_RANGE = (2, 3)
_PAIR_PROB = 0.35
_DUAL_COLOR_PROB = 0.45
_BAR_LEN_MIN = 2
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


def _bar_coords(side: str, start: int, end: int, r0: int, c0: int, r1: int, c1: int) -> list[tuple[int, int]]:
    if side == "top":
        return _hpath(r0 - 1, c0 + 1 + start, c0 + 1 + end)
    if side == "bottom":
        return _hpath(r1 + 1, c0 + 1 + start, c0 + 1 + end)
    if side == "left":
        return _vpath(r0 + 1 + start, r0 + 1 + end, c0 - 1)
    assert side == "right"
    return _vpath(r0 + 1 + start, r0 + 1 + end, c1 + 1)


def _edge_segment(side: str, start: int, end: int, H: int, W: int) -> list[tuple[int, int]]:
    if side == "top":
        return [(0, c) for c in range(start, end + 1)]
    if side == "bottom":
        return [(H - 1, c) for c in range(start, end + 1)]
    if side == "left":
        return [(r, 0) for r in range(start, end + 1)]
    assert side == "right"
    return [(r, W - 1) for r in range(start, end + 1)]


def _full_span_coords(pair_kind: str, start: int, end: int, H: int, W: int) -> list[tuple[int, int]]:
    if pair_kind == "vertical":
        return [(r, c) for c in range(start, end + 1) for r in range(H)]
    return [(r, c) for r in range(start, end + 1) for c in range(W)]


def _choose_interval(length: int, min_len: int = _BAR_LEN_MIN) -> tuple[int, int]:
    assert length >= min_len
    span = random.randint(min_len, length)
    start = random.randint(0, length - span)
    return int(start), int(start + span - 1)


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
    comp_mask = np.zeros_like(grid, dtype=bool)
    used_inner = np.zeros((H, W), dtype=bool)
    palette = [x for x in range(10) if x not in (bg, bc, fill)]
    random.shuffle(palette)
    color_i = 0

    n_groups = random.randint(*_N_GROUPS_RANGE)

    for _ in range(n_groups):
        if color_i >= len(palette):
            return None

        can_pair = H >= _BAR_LEN_MIN and W >= _BAR_LEN_MIN
        if can_pair and random.random() < _PAIR_PROB:
            pair_kind = random.choice(["vertical", "horizontal"])
            axis_len = W if pair_kind == "vertical" else H
            start, end = _choose_interval(axis_len)
            inner = _full_span_coords(pair_kind, start, end, H, W)
            irs, ics = zip(*inner)
            if used_inner[irs, ics].any():
                return None
            color = palette[color_i]
            color_i += 1
            if pair_kind == "vertical":
                bar_a = _bar_coords("top", start, end, r0, c0, r1, c1)
                bar_b = _bar_coords("bottom", start, end, r0, c0, r1, c1)
            else:
                bar_a = _bar_coords("left", start, end, r0, c0, r1, c1)
                bar_b = _bar_coords("right", start, end, r0, c0, r1, c1)
            if not _place_component(grid, comp_mask, bar_a, (color,), bg):
                return None
            if not _place_component(grid, comp_mask, bar_b, (color,), bg):
                return None
            expected[irs, ics] = color
            used_inner[irs, ics] = True
            continue

        side = random.choice(["top", "bottom", "left", "right"])
        axis_len = W if side in ("top", "bottom") else H
        if axis_len < _BAR_LEN_MIN:
            return None
        start, end = _choose_interval(axis_len)
        inner = _edge_segment(side, start, end, H, W)
        irs, ics = zip(*inner)
        if used_inner[irs, ics].any():
            return None
        if random.random() < _DUAL_COLOR_PROB and color_i + 1 < len(palette):
            colors = (palette[color_i], palette[color_i + 1])
            color_i += 2
        else:
            colors = (palette[color_i],)
            color_i += 1
        bar = _bar_coords(side, start, end, r0, c0, r1, c1)
        if not _place_component(grid, comp_mask, bar, colors, bg):
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
    return grid


def generate_input() -> np.ndarray:
    for _ in range(_GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid a6f40cea D3 variant input")
