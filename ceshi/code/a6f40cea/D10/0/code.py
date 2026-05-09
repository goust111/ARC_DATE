# Variant of a6f40cea
# Dimension: D10 — 双色交替相位锚定 (trajectory)
# Change: 交替相位由触点处的颜色决定——路径起点颜色必须匹配该端触点的颜色，且终点颜色匹配另一端触点颜色... → 相位由路径长度奇偶决定

from __future__ import annotations

from collections import Counter, defaultdict
import random

import numpy as np
from scipy.ndimage import label

from arc_utils import STRUCT4, get_bg, neighbors4

# Design analysis for D10
# ----------------------
# variant:
#   focus: only the 2-color L-components change. Monochrome L-components, paired U-components,
#     frame detection, cropping, distractors, and placement constraints stay structurally the same.
#   visible object behavior:
#     - Outside the frame, a 2-color open rectangle is still a 4-connected path touching two
#       adjacent sides of the frame ring.
#     - In the original task, the two ring-touch colors forced the phase of the interior
#       alternating path.
#     - In this variant, the unique far outer corner of the open rectangle becomes the palette
#       anchor. The interior phase then flips or preserves relative to that anchor depending on
#       the interior path cell-count parity.
#   concrete trace:
#     - Suppose a dual-color component uses colors A/B and its far outer corner is colored A.
#       If its interior L-path has 7 cells, the canonical top-or-bottom endpoint starts with A.
#       If the same geometry is stretched to 8 cells, that same endpoint starts with B instead.
#       The old rule would inspect the two ring-touch colors; the new rule inspects path length.
#   implicit assumptions audited:
#     - The old 2-color branch assumed endpoint colors are sufficient and searched both path
#       directions to satisfy them. That fails here because matching both ring endpoints is no
#       longer the rule.
#     - The new 2-color branch therefore needs a canonical interior direction:
#       top/bottom endpoint -> left/right endpoint. Generator guarantees dual-color components are
#       L-shapes with exactly one touch on each of those side families.
#     - The new branch also needs a visible anchor color not based on color-id ordering. Generator
#       guarantees that the far outer corner exists and is unique; solver recovers its color by the
#       component's extremal row/col consistent with the touched frame sides.
#   legality / cleanliness:
#     - A sample must contain at least one dual-color L-component; otherwise D10 would be invisible.
#     - U-pairs are kept monochrome, so the solver never needs to disambiguate 2-color U-shapes.
#     - The generator rejects dual-color L-components whose new output would accidentally match the
#       old endpoint-anchored output, ensuring the variant is materially exercised.
#   observability:
#     - Across examples, parity must vary, so some dual-color outputs preserve the outer-corner
#       anchor at the canonical start while others flip it.
#     - No rule depends on hidden color-id sorting. The anchor is the visible far outer corner, and
#       the controlling variable is the visible interior path length parity.


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

    best = None  # (area, r0,c0,r1,c1, bc, fill)
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
    Reconstruct interior paths by projecting external open-rectangle traces into a framed region.

    Input structure:
    - A background grid contains one closed 1px rectangular frame with uniform interior fill.
    - Colored components lie outside the frame and touch the ring immediately around its border.

    Rule:
    - Output the cropped frame interior, initialized with the frame's fill color.
    - Monochrome components still become solid Manhattan stripes or L-paths between projected endpoints.
    - Two-color components still map to alternating Manhattan L-paths, but they use the canonical
      top/bottom-to-left/right direction rather than endpoint-color matching.
    - For those two-color L-paths, the far outer corner supplies the anchor color, and the parity
      of the interior path cell count decides whether that anchor color stays at the canonical start
      or is flipped to the other color.

    Key insight: The alternating phase is no longer solved by matching the two ring-touch colors;
    instead, find the unique far outer corner of the component, take its color as the anchor, and
    use interior path-length parity to decide the start phase.
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

    def build_manhattan(
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

    def inside_corner(p1: tuple[int, int], p2: tuple[int, int], cornerA: bool) -> bool:
        (ra, ca), (rb, cb) = p1, p2
        if ra == rb or ca == cb:
            return True
        corner = (rb, ca) if cornerA else (ra, cb)
        rr, cc = corner
        return (0 < rr < outH - 1) and (0 < cc < outW - 1)

    def alt_values(colors: tuple[int, int], start_color: int, n: int) -> list[int]:
        a, b = map(int, colors)
        cur = int(start_color)
        vals: list[int] = []
        for _ in range(int(n)):
            vals.append(int(cur))
            cur = int(b) if cur == int(a) else int(a)
        return vals

    def classify_l_touchpoints(
        coords: list[tuple[int, int]],
    ) -> tuple[tuple[int, int], tuple[int, int]] | None:
        tb = [p for p in coords if p[0] in (0, outH - 1) and p[1] not in (0, outW - 1)]
        lr = [p for p in coords if p[1] in (0, outW - 1) and p[0] not in (0, outH - 1)]
        if len(tb) == 1 and len(lr) == 1:
            return tb[0], lr[0]
        tb = [p for p in coords if p[0] in (0, outH - 1)]
        lr = [p for p in coords if p[1] in (0, outW - 1)]
        if len(tb) == 1 and len(lr) == 1 and tb[0] != lr[0]:
            return tb[0], lr[0]
        return None

    def component_corner_color(
        cid: int, tb_pt: tuple[int, int], lr_pt: tuple[int, int]
    ) -> int | None:
        pts = np.argwhere(clab == int(cid))
        if pts.size == 0:
            return None
        rr = int(pts[:, 0].min()) if tb_pt[0] == 0 else int(pts[:, 0].max())
        cc = int(pts[:, 1].min()) if lr_pt[1] == 0 else int(pts[:, 1].max())
        if int(clab[rr, cc]) != int(cid):
            return None
        return int(g[rr, cc])

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
            touchpoints = classify_l_touchpoints(coords)
            if touchpoints is None:
                continue
            tb_pt, lr_pt = touchpoints

            pathA = build_manhattan(tb_pt, lr_pt, True)
            pathB = build_manhattan(tb_pt, lr_pt, False)
            if inside_corner(tb_pt, lr_pt, True):
                path = pathA
            elif inside_corner(tb_pt, lr_pt, False):
                path = pathB
            else:
                path = pathA

            anchor_color = component_corner_color(cid, tb_pt, lr_pt)
            if anchor_color is None or anchor_color not in cols:
                continue
            other_color = next(col for col in cols if int(col) != int(anchor_color))
            start_color = int(anchor_color) if (len(path) % 2 == 1) else int(other_color)
            vals = alt_values((int(anchor_color), int(other_color)), int(start_color), len(path))
            for (rr, cc), val in zip(path, vals):
                out[int(rr), int(cc)] = int(val)
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


# --------------- generator (a6 / D10 variant) ---------------

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
_GEN_RETRIES = 300


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


def _alt_values_from_start(
    path: list[tuple[int, int]], colors: tuple[int, int], start_color: int
) -> list[int]:
    a, b = map(int, colors)
    cur = int(start_color)
    vals: list[int] = []
    for _ in path:
        vals.append(int(cur))
        cur = int(b) if cur == int(a) else int(a)
    return vals


def _alt_values_from_corner(
    path: list[tuple[int, int]], colors: tuple[int, int], corner: tuple[int, int]
) -> list[int]:
    lead, other = map(int, colors)
    k = path.index(corner)
    vals: list[int] = []
    for i, _p in enumerate(path):
        vals.append(int(lead) if (abs(i - k) % 2 == 0) else int(other))
    return vals


def _paint_values(grid: np.ndarray, path: list[tuple[int, int]], vals: list[int]) -> None:
    if not path:
        return
    rs, cs = zip(*path)
    grid[rs, cs] = vals


def _paint_path(grid: np.ndarray, path: list[tuple[int, int]], colors: tuple[int, ...]) -> None:
    if not path:
        return
    if len(colors) == 1:
        rs, cs = zip(*path)
        grid[rs, cs] = colors[0]
        return
    vals = _alt_values_from_start(path, (int(colors[0]), int(colors[1])), int(colors[0]))
    _paint_values(grid, path, vals)


def _paint_path_from_corner(
    grid: np.ndarray,
    path: list[tuple[int, int]],
    colors: tuple[int, int],
    corner: tuple[int, int],
) -> None:
    vals = _alt_values_from_corner(path, colors, corner)
    _paint_values(grid, path, vals)


def _paint_dual_inner_variant(
    grid: np.ndarray, path: list[tuple[int, int]], colors: tuple[int, int]
) -> None:
    anchor, other = map(int, colors)
    start_color = int(anchor) if (len(path) % 2 == 1) else int(other)
    vals = _alt_values_from_start(path, (int(anchor), int(other)), int(start_color))
    _paint_values(grid, path, vals)


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
    *,
    corner_anchor: tuple[int, int] | None = None,
) -> bool:
    if len(set(path)) != len(path):
        return False
    if not _can_place(grid, path, bg, comp_mask):
        return False
    if len(colors) == 2 and corner_anchor is not None:
        _paint_path_from_corner(grid, path, (int(colors[0]), int(colors[1])), corner_anchor)
    else:
        _paint_path(grid, path, colors)
    rs, cs = zip(*path)
    comp_mask[rs, cs] = True
    return True


def _l_specs(kind: str, H: int, W: int, r0: int, c0: int, r1: int, c1: int):
    is_top = kind[0] == "t"
    is_left = kind[1] == "l"

    r = random.randint(1, max(1, H // 2)) if is_top else random.randint(max(1, H // 2), H - 2)
    c = random.randint(1, max(1, W // 2)) if is_left else random.randint(max(1, W // 2), W - 2)

    lo, hi = _OUTER_OFFSET_RANGE
    R = (r0 - random.randint(lo, hi)) if is_top else (r1 + random.randint(lo, hi))
    C = (c0 - random.randint(lo, hi)) if is_left else (c1 + random.randint(lo, hi))

    ring_a = (r0 - 1 if is_top else r1 + 1, c0 + 1 + c)
    ring_b = (r0 + 1 + r, c0 - 1 if is_left else c1 + 1)

    edge_r = 0 if is_top else H - 1
    edge_c = 0 if is_left else W - 1
    inner = _join(_vpath(edge_r, r, c), _hpath(r, c, edge_c))

    outer = _join(
        _vpath(ring_a[0], R, ring_a[1]),
        _hpath(R, ring_a[1], C),
        _vpath(R, ring_b[0], C),
        _hpath(ring_b[0], C, ring_b[1]),
    )
    return inner, outer, (int(R), int(C))


def _u_pair_specs(orientation: str, H: int, W: int, r0: int, c0: int, r1: int, c1: int):
    lo, hi = _OUTER_OFFSET_RANGE
    if orientation == "vertical":
        c_a, c_b = sorted(random.sample(range(1, W - 1), 2))
        if c_b - c_a < 2:
            c_b = min(W - 2, c_a + 2)
        top_r = r0 - random.randint(lo, hi)
        bot_r = r1 + random.randint(lo, hi)
        a_top, b_top = (r0 - 1, c0 + 1 + c_a), (r0 - 1, c0 + 1 + c_b)
        a_bot, b_bot = (r1 + 1, c0 + 1 + c_a), (r1 + 1, c0 + 1 + c_b)
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
        a_left, b_left = (r0 + 1 + r_a, c0 - 1), (r0 + 1 + r_b, c0 - 1)
        a_right, b_right = (r0 + 1 + r_a, c1 + 1), (r0 + 1 + r_b, c1 + 1)
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


def _dual_old_matches_new(
    inner: list[tuple[int, int]],
    outer: list[tuple[int, int]],
    colors: tuple[int, int],
    outer_corner: tuple[int, int],
) -> bool:
    new_vals = _alt_values_from_start(
        inner,
        colors,
        int(colors[0]) if (len(inner) % 2 == 1) else int(colors[1]),
    )
    new_map = {p: v for p, v in zip(inner, new_vals)}

    outer_vals = _alt_values_from_corner(outer, colors, outer_corner)
    ha = int(outer_vals[0])
    hb = int(outer_vals[-1])
    for path_pts in (inner, list(reversed(inner))):
        for start_color in colors:
            end_color = int(start_color) if (len(path_pts) % 2 == 1) else (
                int(colors[1]) if int(start_color) == int(colors[0]) else int(colors[0])
            )
            if int(start_color) != int(ha):
                continue
            if int(end_color) != int(hb):
                continue
            old_vals = _alt_values_from_start(path_pts, colors, int(start_color))
            old_map = {p: v for p, v in zip(path_pts, old_vals)}
            return old_map == new_map
    return False


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
    fill = bg if random.random() < _FILL_EQ_BG_PROB else random.choice(
        [x for x in range(10) if x not in (bg, bc)]
    )
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
    palette = [x for x in range(10) if x not in (bg, bc, fill)]
    random.shuffle(palette)
    used_inner = np.zeros((H, W), dtype=bool)

    n_components = random.randint(*_N_COMPONENTS_RANGE)
    remaining = n_components
    color_i = 0

    if remaining >= 3 and random.random() < _U_PAIR_PROB:
        orientation = random.choice(["vertical", "horizontal"])
        if orientation == "horizontal" and H < 5:
            orientation = "vertical"
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
        expected[irs, ics] = color
        used_inner[irs, ics] = True
        remaining -= 2

    corners = random.sample(["tl", "tr", "bl", "br"], k=remaining)
    need_dual = True
    for idx, kind in enumerate(corners):
        if color_i >= len(palette):
            return None
        inner, outer, outer_corner = _l_specs(kind, H, W, r0, c0, r1, c1)
        irs, ics = zip(*inner)
        if used_inner[irs, ics].any():
            return None

        must_dual = need_dual and idx == len(corners) - 1
        want_dual = need_dual or (random.random() < _DUAL_COLOR_PROB)
        make_dual = want_dual or must_dual

        if make_dual:
            if color_i >= len(palette) - 1:
                return None
            colors = (palette[color_i], palette[color_i + 1])
            color_i += 2
            if _dual_old_matches_new(inner, outer, colors, outer_corner):
                return None
            if not _place_component(
                grid,
                comp_mask,
                outer,
                colors,
                bg,
                corner_anchor=outer_corner,
            ):
                return None
            _paint_dual_inner_variant(expected, inner, colors)
            need_dual = False
        else:
            colors = (palette[color_i],)
            color_i += 1
            if not _place_component(grid, comp_mask, outer, colors, bg):
                return None
            _paint_path(expected, inner, colors)

        used_inner[irs, ics] = True

    if need_dual:
        return None

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
    raise RuntimeError("failed to generate a valid a6f40cea input")
