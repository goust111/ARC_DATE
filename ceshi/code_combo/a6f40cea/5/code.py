# Combo variant of a6f40cea (index 5)
# combo_spec: D2/3+D10/1
# Name: 反相拐角变色

from __future__ import annotations

from collections import Counter, defaultdict
import random

import numpy as np
from scipy.ndimage import label

from arc_utils import STRUCT4, get_bg, neighbors4

# Design analysis
# ---------------
# combo_analysis:
#   concretization_and_trace:
#     - The geometry stays the same as the base task: one closed 1px frame with a uniform fill,
#       plus 2-3 outside open-rectangle components that touch the frame ring and project inward.
#     - Monochrome behavior stays unchanged. Only 2-color L-components change semantics.
#     - Base task: if the two ring-touch endpoints have colors (A, B), the interior L-path
#       alternates A/B/A/B... with phase anchored directly by the endpoint colors.
#     - D2/3 alone: the interior no longer alternates cell-by-cell; it stays one color before
#       the elbow and switches once at the elbow.
#     - This combo adds D10/1 on top of that split rule: the first leg uses the opposite of the
#       nearby ring-touch color, and the second leg uses the opposite of the other touch color.
#       With endpoint hints (A, B), the interior becomes B on the first leg, then A at and after
#       the elbow. Example: if a top touch is 8 and a left touch is 4, the output paints 4 down
#       to the elbow and 8 from the elbow to the left boundary.
#     - The outside 2-color L-components are also painted as corner-split traces, not alternating
#       per cell. Their visible order stays (A then B), while the inside reconstructed order is
#       flipped to (B then A). That makes both halves of the combo observable.
#   complexity_calibration:
#     - The combo still uses only two colors per affected component and exactly one structural
#       switch point, so the rule is visible without introducing uncontrolled degrees of freedom.
#     - U-pairs remain monochrome; otherwise same-side stripe cases would blur together with the
#       fused color rule and reduce sample clarity.
#     - The generator forces at least one 2-color L per sample so the combo never degenerates
#       into the original task or into the plain D2 split-only variant.
#   implicit_assumptions_review:
#     - The original `len(cols) == 2` solver branch assumed parity-based alternation. That is no
#       longer valid; painting must depend on the chosen elbow, not on path length parity.
#     - The original `build_manhattan` returned only a path. The combo needs the elbow explicitly
#       so the color boundary can be placed at the turn.
#     - `hist[(r, c)]` still only needs to recover the dominant touch color at each mapped endpoint.
#       The new rule uses the opposite palette color of each endpoint, so the component must expose
#       two distinct endpoint colors. The generator guarantees that for every 2-color L.
#     - Reversing endpoint order in the solver is harmless: whichever endpoint is treated as "first"
#       gets the opposite color on its adjacent leg, so the final painted cells are the same.
#   placement_legality:
#     - All outside components still pass the original non-overlap and non-adjacency checks.
#     - `_l_specs` keeps both the inside elbow and the outside corner strictly structural and
#       reachable by a simple Manhattan path; the designated switch cell is always on the path.
#     - The optional U-pair is only placed when at least one L slot remains, because the combo's
#       distinguishing behavior exists only on 2-color L-components.
#   visual_observability:
#     - A solver can see that 2-color traces are no longer checkerboard-like; they are split into
#       two solid legs with a turn-based switch.
#     - Comparing input and output across a few samples also reveals the inversion: the leg next to
#       a touchpoint always takes the *other* component color inside the frame.
#     - No hidden ordering by color id is used. Everything is signaled by visible touchpoint colors,
#       the visible corner split, and the output elbow.
#   visual_cleanliness:
#     - The sparse original layout is preserved, and only L-components can be two-color, so the
#       frame interior does not become crowded with competing color rules.
#     - The corner cell belongs to the second leg consistently, which keeps the switch visually crisp.
#   change_inferability:
#     - The only state the human must infer is a binary relation between the two endpoint colors:
#       "inside starts with the opposite color, then flips at the elbow." That relation repeats
#       across all dual-color Ls in every generated sample.
#   post_change_solver_assumptions:
#     - Every generated 2-color component is a non-degenerate L whose mapped endpoints lie on
#       adjacent frame sides, so there is a unique meaningful elbow split.
#     - Every generated 2-color component has distinct dominant colors at its two ring-touch ends.
#     - Monochrome paths and U-pair aggregation still satisfy the original stripe assumptions.


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
    Recover the cropped frame interior by projecting exterior open-rect traces inward.

    Input structure:
    - One largest 1-cell-thick closed rectangular frame encloses a uniformly filled interior.
    - Separate 4-connected colored components lie outside the frame and touch the ring just beyond it.

    Rule:
    - Return only the frame interior, initialized with the frame's fill color.
    - Map each ring-touching component onto interior boundary coordinates and connect its endpoints with a Manhattan path.
    - Monochrome components still paint solid stripes or L-paths.
    - Two-color L-components no longer alternate by parity: they keep one solid color before the elbow, switch at the elbow, and the first leg uses the opposite of its adjacent touch color.

    Key insight:
    - For a 2-color L trace, the touchpoint colors tell you the palette, but each interior leg uses the other endpoint color, with the elbow acting as the switch boundary.
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

    def build_manhattan(
        p1: tuple[int, int], p2: tuple[int, int], corner_a: bool
    ) -> tuple[list[tuple[int, int]], tuple[int, int] | None]:
        (ra, ca), (rb, cb) = p1, p2
        if ra == rb:
            step = 1 if cb >= ca else -1
            return ([(int(ra), int(c)) for c in range(int(ca), int(cb) + step, step)], None)
        if ca == cb:
            step = 1 if rb >= ra else -1
            return ([(int(r), int(ca)) for r in range(int(ra), int(rb) + step, step)], None)
        if corner_a:
            corner = (int(rb), int(ca))
            step = 1 if rb >= ra else -1
            path = [(int(r), int(ca)) for r in range(int(ra), int(rb) + step, step)]
            step = 1 if cb >= ca else -1
            path += [(int(rb), int(c)) for c in range(int(ca) + step, int(cb) + step, step)]
            return path, corner
        corner = (int(ra), int(cb))
        step = 1 if cb >= ca else -1
        path = [(int(ra), int(c)) for c in range(int(ca), int(cb) + step, step)]
        step = 1 if rb >= ra else -1
        path += [(int(r), int(cb)) for r in range(int(ra) + step, int(rb) + step, step)]
        return path, corner

    def inside_corner(p1: tuple[int, int], p2: tuple[int, int], corner_a: bool) -> bool:
        (ra, ca), (rb, cb) = p1, p2
        if ra == rb or ca == cb:
            return True
        corner = (rb, ca) if corner_a else (ra, cb)
        rr, cc = corner
        return (0 < rr < outH - 1) and (0 < cc < outW - 1)

    def boundary_side(p: tuple[int, int]) -> str | None:
        r, c = p
        if r == 0:
            return "top"
        if r == outH - 1:
            return "bottom"
        if c == 0:
            return "left"
        if c == outW - 1:
            return "right"
        return None

    def other_color(anchor: int, color_a: int, color_b: int) -> int:
        return int(color_b) if int(anchor) == int(color_a) else int(color_a)

    def farthest_pair(points: list[tuple[int, int]]) -> tuple[tuple[int, int], tuple[int, int]] | None:
        best = None
        for i in range(len(points)):
            for j in range(i + 1, len(points)):
                d = abs(points[i][0] - points[j][0]) + abs(points[i][1] - points[j][1])
                if best is None or d > best[0]:
                    best = (d, points[i], points[j])
        if best is None:
            return None
        return best[1], best[2]

    def ordered_dual_l_endpoints(
        points: list[tuple[int, int]],
    ) -> tuple[tuple[int, int], tuple[int, int]] | None:
        pair = farthest_pair(points)
        if pair is None:
            return None
        a, b = pair
        side_a = boundary_side(a)
        side_b = boundary_side(b)
        if side_a in ("top", "bottom") and side_b in ("left", "right"):
            return a, b
        if side_b in ("top", "bottom") and side_a in ("left", "right"):
            return b, a
        return a, b

    def paint_corner_split(
        path_pts: list[tuple[int, int]],
        corner: tuple[int, int],
        first_color: int,
        second_color: int,
    ) -> None:
        assert corner in path_pts
        switch_idx = int(path_pts.index(corner))
        for idx, (rr, cc) in enumerate(path_pts):
            out[int(rr), int(cc)] = int(first_color if idx < switch_idx else second_color)

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
    for rr, cc in sorted(ring):
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

            pair = farthest_pair(uniq)
            if pair is None:
                continue
            p1, p2 = pair
            path_a, _corner_a = build_manhattan(p1, p2, True)
            path_b, _corner_b = build_manhattan(p1, p2, False)
            chosen = path_b if (inside_corner(p1, p2, False) and not inside_corner(p1, p2, True)) else path_a
            for rr, cc in chosen:
                out[int(rr), int(cc)] = int(colv)
            continue

        if len(cols) == 2:
            pair = ordered_dual_l_endpoints(coords)
            if pair is None:
                continue
            a, b = pair
            ca, cb = map(int, cols)
            ha = int(hist[a].most_common(1)[0][0])
            hb = int(hist[b].most_common(1)[0][0])
            if ha == hb:
                ha, hb = ca, cb

            path_a, corner_a = build_manhattan(a, b, True)
            path_b, corner_b = build_manhattan(a, b, False)
            if inside_corner(a, b, False) and not inside_corner(a, b, True):
                path, corner = path_b, corner_b
            else:
                path, corner = path_a, corner_a

            assert corner is not None
            paint_corner_split(
                path,
                corner,
                other_color(ha, ca, cb),
                other_color(hb, ca, cb),
            )
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
                if not cs:
                    continue
                for cc in range(min(cs), max(cs) + 1):
                    out[int(rr), int(cc)] = int(col)
            continue
        if len(pts) == 1:
            rr, cc = pts[0]
            out[int(rr), int(cc)] = int(col)
            continue
        pair = farthest_pair(pts)
        if pair is None:
            continue
        p1, p2 = pair
        path_a, _corner_a = build_manhattan(p1, p2, True)
        path_b, _corner_b = build_manhattan(p1, p2, False)
        chosen = path_b if (inside_corner(p1, p2, False) and not inside_corner(p1, p2, True)) else path_a
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
_OUTER_OFFSET_RANGE = (2, 3)
_N_COMPONENTS_RANGE = (2, 3)
_U_PAIR_PROB = 0.25
_DUAL_COLOR_PROB = 0.45
_MIN_DUAL_L_COMPONENTS = 1
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


def _paint_path(
    grid: np.ndarray,
    path: list[tuple[int, int]],
    colors: tuple[int, ...],
    switch_at: tuple[int, int] | None = None,
) -> None:
    """Paint a solid path or a two-color path that switches once at a designated corner."""
    if not path:
        return
    rs, cs = zip(*path)
    if len(colors) == 1:
        grid[rs, cs] = colors[0]
        return
    assert switch_at is not None and switch_at in path
    switch_idx = int(path.index(switch_at))
    vals = [colors[0] if i < switch_idx else colors[1] for i in range(len(path))]
    grid[rs, cs] = vals


def _invert_two_colors(colors: tuple[int, ...]) -> tuple[int, ...]:
    if len(colors) != 2:
        return colors
    return int(colors[1]), int(colors[0])


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
    switch_at: tuple[int, int] | None = None,
) -> bool:
    if len(set(path)) != len(path):
        return False
    if not _can_place(grid, path, bg, comp_mask):
        return False
    _paint_path(grid, path, colors, switch_at=switch_at)
    rs, cs = zip(*path)
    comp_mask[rs, cs] = True
    return True


def _l_specs(kind: str, H: int, W: int, r0: int, c0: int, r1: int, c1: int):
    """Build inner/outer paths plus their switch corners for one L-component."""
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
    inner_corner = (int(r), int(c))
    inner = _join(_vpath(edge_r, r, c), _hpath(r, c, edge_c))

    outer_corner = (int(R), int(C))
    outer = _join(
        _vpath(ring_a[0], R, ring_a[1]),
        _hpath(R, ring_a[1], C),
        _vpath(R, ring_b[0], C),
        _hpath(ring_b[0], C, ring_b[1]),
    )
    return inner, outer, inner_corner, outer_corner


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
    dual_l_count = 0

    if remaining >= 3 and random.random() < _U_PAIR_PROB:
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
        expected[irs, ics] = color
        used_inner[irs, ics] = True
        remaining -= 2

    if remaining < _MIN_DUAL_L_COMPONENTS:
        return None

    corners = random.sample(["tl", "tr", "bl", "br"], k=remaining)
    for idx, kind in enumerate(corners):
        inner, outer, inner_corner, outer_corner = _l_specs(kind, H, W, r0, c0, r1, c1)
        irs, ics = zip(*inner)
        if used_inner[irs, ics].any():
            return None

        l_left_including_this = len(corners) - idx
        must_make_dual = (
            dual_l_count < _MIN_DUAL_L_COMPONENTS
            and l_left_including_this == (_MIN_DUAL_L_COMPONENTS - dual_l_count)
        )
        can_make_dual = color_i < len(palette) - 1
        want_dual = must_make_dual or (can_make_dual and random.random() < _DUAL_COLOR_PROB)

        if want_dual:
            if not can_make_dual:
                return None
            colors = (palette[color_i], palette[color_i + 1])
            color_i += 2
            if not _place_component(grid, comp_mask, outer, colors, bg, switch_at=outer_corner):
                return None
            _paint_path(expected, inner, _invert_two_colors(colors), switch_at=inner_corner)
            dual_l_count += 1
        else:
            if color_i >= len(palette):
                return None
            colors = (palette[color_i],)
            color_i += 1
            if not _place_component(grid, comp_mask, outer, colors, bg):
                return None
            _paint_path(expected, inner, colors)

        used_inner[irs, ics] = True

    if dual_l_count < _MIN_DUAL_L_COMPONENTS:
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
    raise RuntimeError("failed to generate a valid a6f40cea combo input")
