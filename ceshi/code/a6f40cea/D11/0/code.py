# Variant of a6f40cea
# Dimension: D11 — 端点配对策略 (path)
# Change: 同一连通分量的所有触点中选择曼哈顿距离最远的一对作为路径端点... → 最近点配对

from __future__ import annotations

from collections import Counter, defaultdict
import random

import numpy as np
from scipy.ndimage import label

from arc_utils import STRUCT4, get_bg, neighbors4


# Design analysis for the D11 variant
# ----------------------------------
# spec_variant:
#   preserved_core:
#     - one large closed rectangular frame with uniform fill
#     - outside colored 4-connected components touching the frame ring
#     - output is still the cropped frame interior
#     - monochrome components still paint solid paths; 2-color components still alternate
#   d11_change:
#     - the endpoint selector now chooses the nearest valid pair of ring-touch points
#       inside one connected component instead of the farthest pair
#   why_generator_must_change:
#     - in the original generator, an L-component touched the ring in exactly two cells
#       (one on each side), so "nearest" and "farthest" were identical and D11 was invisible
#     - this variant upgrades each L-component into one connected outside scaffold with four
#       isolated ring touches: two on one frame side and two on the adjacent side
#     - among pairs that lie on different frame sides, the near-near pair yields a short L
#       near the corner while the far-far pair yields a much longer L, so nearest vs farthest
#       becomes visually obvious
#   concrete_trace:
#     - top-left component touches mapped ring cells {(0,1), (0,5), (2,0), (6,0)}
#     - original rule would pick (0,5) and (6,0), producing a long L
#     - new rule picks (0,1) and (2,0), producing a short L close to the corner
#     - same-side pairs like (0,1)-(0,5) are intentionally excluded in the solver because
#       otherwise adjacent touches on one side would dominate the nearest-distance search and
#       collapse the task into trivial boundary segments
#   complexity_calibration:
#     - L-components always have exactly four ring touches, not an arbitrary count
#     - touches are constrained so the nearest cross-side pair is unique and differs from the
#       farthest cross-side pair
#     - U-pairs and distractors remain, but U-pairs are only allowed when at least one L
#       component still remains, so most examples still expose the D11 behavior directly
#   hidden_assumptions_checked:
#     - original main() assumed L-components effectively had one touch per side; that is no
#       longer true, so pair selection is centralized and side-aware
#     - original expected-output construction painted the inner L chosen by construction; the
#       new generator must instead compute the same nearest pair that main() will compute
#     - 2-color phase can no longer be assumed from construction order alone, so the generator
#       reads the actual ring endpoint colors after painting the outside component
#   placement_legality:
#     - the new outer scaffold stays strictly outside the frame except for isolated ring-touch
#       cells, uses fixed offsets 2-3 from the frame, and is validated against existing
#       components by the same placement checks as before
#   observability_and_cleanliness:
#     - interior minima are raised slightly so each L can contain both near and far touchpoints
#       with a clear size gap
#     - each grid still contains at least one L-component whenever a U-pair is present
#     - colors are never used for ordering or ranking; all decisions depend only on geometry


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


def _build_manhattan(
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


def _inside_corner(
    p1: tuple[int, int], p2: tuple[int, int], corner_a: bool, out_h: int, out_w: int
) -> bool:
    (ra, ca), (rb, cb) = p1, p2
    if ra == rb or ca == cb:
        return True
    corner = (rb, ca) if corner_a else (ra, cb)
    rr, cc = corner
    return (0 < rr < out_h - 1) and (0 < cc < out_w - 1)


def _choose_manhattan_path(
    p1: tuple[int, int], p2: tuple[int, int], out_h: int, out_w: int
) -> list[tuple[int, int]]:
    path_a = _build_manhattan(p1, p2, True)
    path_b = _build_manhattan(p1, p2, False)
    if _inside_corner(p1, p2, False, out_h, out_w) and not _inside_corner(
        p1, p2, True, out_h, out_w
    ):
        return path_b
    return path_a


def _side_tags(p: tuple[int, int], out_h: int, out_w: int) -> tuple[str, ...]:
    r, c = p
    tags: list[str] = []
    if r == 0:
        tags.append("t")
    if r == out_h - 1:
        tags.append("b")
    if c == 0:
        tags.append("l")
    if c == out_w - 1:
        tags.append("r")
    return tuple(tags)


def _select_endpoint_pair(
    coords: list[tuple[int, int]],
    out_h: int,
    out_w: int,
    *,
    prefer: str,
) -> tuple[tuple[int, int], tuple[int, int]] | None:
    diff_side: list[tuple[int, tuple[int, int], tuple[int, int]]] = []
    any_pair: list[tuple[int, tuple[int, int], tuple[int, int]]] = []
    for i in range(len(coords)):
        for j in range(i + 1, len(coords)):
            p1 = (int(coords[i][0]), int(coords[i][1]))
            p2 = (int(coords[j][0]), int(coords[j][1]))
            d = abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])
            rec = (int(d), p1, p2)
            any_pair.append(rec)
            if _side_tags(p1, out_h, out_w) != _side_tags(p2, out_h, out_w):
                diff_side.append(rec)
    pool = diff_side if diff_side else any_pair
    if not pool:
        return None
    if prefer == "nearest":
        _d, p1, p2 = min(pool, key=lambda item: (item[0], item[1], item[2]))
    else:
        _d, p1, p2 = max(pool, key=lambda item: (item[0], tuple(-x for x in item[1]), tuple(-x for x in item[2])))
    return p1, p2


def _resolve_alt_choice(
    path: list[tuple[int, int]], colors: tuple[int, int], ha: int, hb: int
) -> tuple[list[tuple[int, int]], int]:
    ca, cb = map(int, colors)
    for path_pts in (path, list(reversed(path))):
        for start_color in (ca, cb):
            end_color = (
                start_color
                if ((len(path_pts) - 1) % 2 == 0)
                else (cb if start_color == ca else ca)
            )
            if int(start_color) != int(ha):
                continue
            if int(end_color) != int(hb):
                continue
            return path_pts, int(start_color)
    raise AssertionError("failed to determine alternating phase")


def _paint_ordered_path(
    grid: np.ndarray, path: list[tuple[int, int]], colors: tuple[int, ...]
) -> None:
    if not path:
        return
    rs, cs = zip(*path)
    if len(colors) == 1:
        grid[rs, cs] = colors[0]
        return
    a, b = colors
    vals = [a if i % 2 == 0 else b for i in range(len(path))]
    grid[rs, cs] = vals


def _paint_component_cells(
    grid: np.ndarray,
    coords: list[tuple[int, int]],
    colors: tuple[int, ...],
) -> None:
    if not coords:
        return
    if len(colors) == 1:
        rs, cs = zip(*coords)
        grid[rs, cs] = colors[0]
        return
    a, b = colors
    anchor = min(coords)
    vals = [a if ((r - anchor[0]) + (c - anchor[1])) % 2 == 0 else b for r, c in coords]
    rs, cs = zip(*coords)
    grid[rs, cs] = vals


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reconstruct interior Manhattan paths from multi-touch outside traces around a framed rectangle.

    Input structure:
    - A background grid contains one large 1px rectangular frame with a uniform interior fill.
    - Colored components live outside the frame and touch the frame ring on one or more cells.

    Rule:
    - Output the frame interior only, initially filled with the frame's fill color.
    - Map each ring-touch cell into the interior boundary coordinates.
    - For each component, choose the nearest valid pair of mapped touch points on different frame sides.
    - Monochrome components paint a solid Manhattan path; two-color components paint an alternating path.

    Key insight:
    - The new ambiguity is resolved by shortest cross-side pairing, while two-color endpoint hints still fix the alternating phase.
    """
    g = np.array(grid, dtype=int)
    H, W = g.shape
    bg = int(get_bg(g))

    found = _find_framed_rect_uniform_fill(g, bg=int(bg))
    assert found is not None
    r0, c0, r1, c1, bc, fill = found
    out_h = int(r1 - r0 - 1)
    out_w = int(c1 - c0 - 1)
    out = np.full((out_h, out_w), int(fill), dtype=int)
    r_int0 = int(r0 + 1)
    c_int0 = int(c0 + 1)

    def map_to_out(rr: int, cc: int) -> tuple[int, int]:
        mr = max(0, min(out_h - 1, int(rr - r_int0)))
        mc = max(0, min(out_w - 1, int(cc - c_int0)))
        return int(mr), int(mc)

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

            pair = _select_endpoint_pair(coords, out_h, out_w, prefer="nearest")
            if pair is None:
                continue
            p1, p2 = pair
            for rr, cc in _choose_manhattan_path(p1, p2, out_h, out_w):
                out[int(rr), int(cc)] = int(colv)
            continue

        if len(cols) == 2:
            pair = _select_endpoint_pair(coords, out_h, out_w, prefer="nearest")
            if pair is None:
                continue
            a, b = pair
            ca, cb = map(int, cols)
            ha = int(hist[a].most_common(1)[0][0])
            hb = int(hist[b].most_common(1)[0][0])
            path = _choose_manhattan_path(a, b, out_h, out_w)
            path_pts, start_color = _resolve_alt_choice(path, (ca, cb), ha, hb)
            cur = int(start_color)
            for rr, cc in path_pts:
                out[int(rr), int(cc)] = int(cur)
                cur = int(cb) if cur == int(ca) else int(ca)
            continue

    for col, pts in deferred.items():
        pts = list(pts)
        if not pts:
            continue
        tops = [p for p in pts if p[0] == 0]
        bots = [p for p in pts if p[0] == out_h - 1]
        if tops and bots:
            for cc in sorted({c for _r, c in pts}):
                rs = [r for r, c in pts if c == cc]
                for rr in range(min(rs), max(rs) + 1):
                    out[int(rr), int(cc)] = int(col)
            continue
        lefts = [p for p in pts if p[1] == 0]
        rights = [p for p in pts if p[1] == out_w - 1]
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
        pair = _select_endpoint_pair(pts, out_h, out_w, prefer="nearest")
        if pair is None:
            continue
        p1, p2 = pair
        for rr, cc in _choose_manhattan_path(p1, p2, out_h, out_w):
            out[int(rr), int(cc)] = int(col)

    return out


# --------------- generator (a6 D11 variant) ---------------
_BORDER_COLOR = 3
_BG_CHOICES = [0, 1, 2, 4, 5, 6, 7, 8, 9]
_FILL_EQ_BG_PROB = 0.6
_INTERIOR_H_RANGE = (6, 13)
_INTERIOR_W_RANGE = (8, 14)
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


def _can_place(
    grid: np.ndarray, coords: list[tuple[int, int]], bg: int, comp_mask: np.ndarray
) -> bool:
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
    coords: list[tuple[int, int]],
    colors: tuple[int, ...],
    bg: int,
) -> bool:
    coords = sorted(set(coords))
    if not coords:
        return False
    if not _can_place(grid, coords, bg, comp_mask):
        return False
    _paint_component_cells(grid, coords, colors)
    rs, cs = zip(*coords)
    comp_mask[rs, cs] = True
    return True


def _pick_near_far(v_span: int, h_span: int) -> tuple[int, int, int, int] | None:
    if v_span < 6 or h_span < 8:
        return None
    for _ in range(80):
        v_near = random.randint(1, min(2, v_span - 3))
        h_near = random.randint(1, min(2, h_span - 3))
        v_far = random.randint(v_near + 2, v_span - 2)
        h_far = random.randint(h_near + 2, h_span - 2)
        near_sum = v_near + h_near
        if (v_far - v_near) <= near_sum:
            continue
        if (h_far - h_near) <= near_sum:
            continue
        return v_near, v_far, h_near, h_far
    return None


def _make_multi_touch_l_specs(
    kind: str, H: int, W: int, r0: int, c0: int, r1: int, c1: int
) -> tuple[
    list[tuple[int, int]],
    dict[tuple[int, int], tuple[int, int]],
    tuple[tuple[int, int], tuple[int, int]],
] | None:
    picked = _pick_near_far(H, W)
    if picked is None:
        return None
    v_near, v_far, h_near, h_far = picked
    is_top = kind[0] == "t"
    is_left = kind[1] == "l"
    ring_row = r0 - 1 if is_top else r1 + 1
    ring_col = c0 - 1 if is_left else c1 + 1
    outer_row = (r0 - random.randint(*_OUTER_OFFSET_RANGE)) if is_top else (r1 + random.randint(*_OUTER_OFFSET_RANGE))
    outer_col = (c0 - random.randint(*_OUTER_OFFSET_RANGE)) if is_left else (c1 + random.randint(*_OUTER_OFFSET_RANGE))

    if is_left:
        h_cols = (c0 + 1 + h_near, c0 + 1 + h_far)
    else:
        h_cols = (c1 - 1 - h_near, c1 - 1 - h_far)
    if is_top:
        v_rows = (r0 + 1 + v_near, r0 + 1 + v_far)
    else:
        v_rows = (r1 - 1 - v_near, r1 - 1 - v_far)

    top_near = (ring_row, h_cols[0])
    top_far = (ring_row, h_cols[1])
    side_near = (v_rows[0], ring_col)
    side_far = (v_rows[1], ring_col)

    h_touch = sorted([top_near, top_far], key=lambda p: p[1])
    v_touch = sorted([side_near, side_far], key=lambda p: p[0])
    h_cols_sorted = sorted([p[1] for p in h_touch])
    v_rows_sorted = sorted([p[0] for p in v_touch])

    coords = set()
    coords.update(_hpath(outer_row, h_cols_sorted[0], h_cols_sorted[1]))
    coords.update(_vpath(ring_row, outer_row, h_touch[0][1]))
    coords.update(_vpath(ring_row, outer_row, h_touch[1][1]))
    coords.update(_vpath(v_rows_sorted[0], v_rows_sorted[1], outer_col))
    coords.update(_hpath(v_touch[0][0], ring_col, outer_col))
    coords.update(_hpath(v_touch[1][0], ring_col, outer_col))
    connector_h_end = h_cols_sorted[0] if is_left else h_cols_sorted[1]
    connector_v_end = v_rows_sorted[0] if is_top else v_rows_sorted[1]
    coords.update(_hpath(outer_row, outer_col, connector_h_end))
    coords.update(_vpath(outer_row, connector_v_end, outer_col))

    mapped: dict[tuple[int, int], tuple[int, int]] = {}
    if kind == "tl":
        mapped[top_near] = (0, h_near)
        mapped[top_far] = (0, h_far)
        mapped[side_near] = (v_near, 0)
        mapped[side_far] = (v_far, 0)
    elif kind == "tr":
        mapped[top_near] = (0, W - 1 - h_near)
        mapped[top_far] = (0, W - 1 - h_far)
        mapped[side_near] = (v_near, W - 1)
        mapped[side_far] = (v_far, W - 1)
    elif kind == "bl":
        mapped[top_near] = (H - 1, h_near)
        mapped[top_far] = (H - 1, h_far)
        mapped[side_near] = (H - 1 - v_near, 0)
        mapped[side_far] = (H - 1 - v_far, 0)
    else:
        mapped[top_near] = (H - 1, W - 1 - h_near)
        mapped[top_far] = (H - 1, W - 1 - h_far)
        mapped[side_near] = (H - 1 - v_near, W - 1)
        mapped[side_far] = (H - 1 - v_far, W - 1)

    touch_coords = [top_near, top_far, side_near, side_far]
    pair = _select_endpoint_pair(list(mapped.values()), H, W, prefer="nearest")
    if pair is None:
        return None
    return sorted(coords), mapped, pair


def _u_pair_specs(
    orientation: str, H: int, W: int, r0: int, c0: int, r1: int, c1: int
) -> tuple[list[list[tuple[int, int]]], list[tuple[int, int]], list[tuple[int, int]]]:
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
    fill = (
        bg
        if random.random() < _FILL_EQ_BG_PROB
        else random.choice([x for x in range(10) if x not in (bg, bc)])
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
    l_components = 0

    if remaining >= 3 and random.random() < _U_PAIR_PROB:
        orientation = random.choice(["vertical", "horizontal"])
        if orientation == "horizontal" and H < 6:
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
    for kind in corners:
        if color_i >= len(palette) - 1:
            return None
        spec = _make_multi_touch_l_specs(kind, H, W, r0, c0, r1, c1)
        if spec is None:
            return None
        outer_coords, mapped, pair = spec
        p1, p2 = pair
        path = _choose_manhattan_path(p1, p2, H, W)
        irs, ics = zip(*path)
        if used_inner[irs, ics].any():
            return None

        if random.random() < _DUAL_COLOR_PROB:
            colors = (palette[color_i], palette[color_i + 1])
            color_i += 2
        else:
            colors = (palette[color_i],)
            color_i += 1

        if not _place_component(grid, comp_mask, outer_coords, colors, bg):
            return None

        touch_by_mapped = {mp: gp for gp, mp in mapped.items()}
        if len(colors) == 1:
            expected[irs, ics] = colors[0]
        else:
            gp1 = touch_by_mapped[p1]
            gp2 = touch_by_mapped[p2]
            ha = int(grid[gp1])
            hb = int(grid[gp2])
            path_pts, start_color = _resolve_alt_choice(path, colors, ha, hb)
            cur = int(start_color)
            for rr, cc in path_pts:
                expected[int(rr), int(cc)] = int(cur)
                cur = int(colors[1]) if cur == int(colors[0]) else int(colors[0])
        used_inner[irs, ics] = True
        l_components += 1

    if l_components == 0:
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
    raise RuntimeError("failed to generate a valid a6f40cea D11 input")
