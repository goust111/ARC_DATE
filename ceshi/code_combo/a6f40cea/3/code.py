# Combo variant of a6f40cea (index 3)
# combo_spec: D1/4+D8/2
# Name: 外角决定Z形拐点

from __future__ import annotations

from collections import Counter, defaultdict
import random

import numpy as np
from scipy.ndimage import label

from arc_utils import STRUCT4, get_bg, neighbors4

# Design analysis
# a) Concrete behavior and trace
#    - The global scene stays close to the original task: one closed 1px frame with uniform fill,
#      2-3 colored components outside the frame, with no extra off-ring clutter in generated samples.
#    - What changes is the reconstruction for adjacent-side L-components. The old solver mapped the
#      two ring touches to one elbow L-path. Here the outside open rectangle carries two depths:
#      how far its top/bottom branch stands off from the frame, and how far its left/right branch
#      stands off. Those two depths are mirrored inward and become one turn row plus one turn
#      column. The recovered path therefore follows a Z-like 4-segment trace instead of a 2-segment L.
#    - Small trace:
#      old: top touch -> single interior elbow -> left touch.
#      new: if the same component sits 3 cells above the frame and 2 cells left of it, then inside
#      the path first drops to row 2, then moves to col 1, then aligns to the side endpoint, so
#      both outer offsets visibly affect the output.
# b) Complexity calibration
#    - Only a small depth alphabet is used: 2, 3, or 4 cells when feasible. That is enough to make
#      the geometry visibly different, but still compact enough for humans to infer from a few examples.
#    - Every generated puzzle contains 2-3 adjacent-side L-components. Legacy opposite-side U-pairs
#      remain supported by the solver, but the generator does not sample them, so the fused rule is
#      present on every component instead of being diluted by a second branch.
#    - When possible, the two outside depths are forced to differ so both projected turn coordinates
#      are visually observable and the path does not look like an arbitrary symmetric detour.
# c) Hidden assumptions from the original solver that no longer hold
#    - The original `build_manhattan(...)` assumed every adjacent-side component was fully described
#      by its two endpoints and one elbow choice. That fails here because the outer bbox depth now
#      carries extra structure and changes the interior path even when endpoints stay fixed.
#    - The original mono and dual-color branches both assumed "farthest pair + one elbow" was enough.
#      Here solver must inspect which frame sides are touched and recover the outside corner depth
#      before drawing the internal path.
#    - Endpoint-color anchoring for alternating traces still works, because the outside path is painted
#      from the top/bottom touch toward the left/right touch and the solver reconstructs the same order.
# d) Placement legality
#    - Generator chooses the two mirrored depths first, then samples endpoints strictly beyond those
#      turn coordinates. That guarantees all interior segments have positive length, the path stays
#      inside bounds, and the result cannot collapse back to a single-elbow L.
#    - Components are still validated with the original no-overlap / no-adjacency checks outside the
#      frame, and `used_inner` forbids interior path overlap.
# e) Visual observability
#    - The rule depends only on visible geometry: the open rectangle sticks out by one vertical depth
#      and one horizontal depth, and the output mirrors both. No color-id sorting or hidden metadata
#      is used.
#    - Compared with the source task, outputs now show a repeated Z-like detour pattern instead of a
#      single Manhattan elbow, so the behavioral difference is visible from a few examples.
# f) Visual cleanliness
#    - Endpoints are still biased toward their local corner sectors, so multiple components do not
#      tangle in the center.
#    - Removing generator-side distractors keeps the ring geometry as the only salient cue and avoids
#      accidental attention sinks outside the frame.
# g) Color inferability
#    - Colors keep the original semantics: monochrome traces paint solid paths and dual-color traces
#      alternate. The new rule changes only geometry, not the meaning of color.
# 9) New solver assumptions and generator guarantees
#    - Solver assumes every L-component touches exactly one top/bottom ring cell and one left/right
#      ring cell. Generator enforces that by reusing the same outside open-rectangle template.
#    - Solver assumes the component bbox extremum in the touched corner is the intended outer corner.
#      Generator makes every L-component a simple monotone open rectangle, so that extremum is unique.
#    - Solver assumes the mirrored turn row/col lie strictly between the two endpoint coordinates.
#      Generator enforces positive first, middle, and last segments before placing any component.
#    - Solver no longer falls back to a generic adjacent-side path when this geometry is absent; that
#      keeps the combo tied to the intended observable construction instead of silently solving extras.


def _find_framed_rect_uniform_fill(
    g: np.ndarray,
    *,
    bg: int | None = None,
    structure=STRUCT4,
) -> tuple[int, int, int, int, int, int] | None:
    """Detect the main 1px-thick rectangular frame with uniform interior fill."""
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
    return [(int(r), int(c)) for r in _line(int(r0), int(r1))]


def _hpath(r: int, c0: int, c1: int) -> list[tuple[int, int]]:
    return [(int(r), int(c)) for c in _line(int(c0), int(c1))]


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
        grid[rs, cs] = int(colors[0])
    else:
        a, b = map(int, colors)
        vals = [a if i % 2 == 0 else b for i in range(len(path))]
        grid[rs, cs] = vals


def _count_bends(path: list[tuple[int, int]]) -> int:
    bends = 0
    for i in range(1, len(path) - 1):
        dr1 = int(path[i][0] - path[i - 1][0])
        dc1 = int(path[i][1] - path[i - 1][1])
        dr2 = int(path[i + 1][0] - path[i][0])
        dc2 = int(path[i + 1][1] - path[i][1])
        if (dr1, dc1) != (dr2, dc2):
            bends += 1
    return int(bends)


def _turn_from_depth(kind: str, H: int, W: int, depth_r: int, depth_c: int) -> tuple[int, int]:
    is_top = kind[0] == "t"
    is_left = kind[1] == "l"
    turn_r = int(depth_r - 1) if is_top else int(H - depth_r)
    turn_c = int(depth_c - 1) if is_left else int(W - depth_c)
    return int(turn_r), int(turn_c)


def _build_projected_z_path(
    start_tb: tuple[int, int],
    end_lr: tuple[int, int],
    turn_r: int,
    turn_c: int,
) -> list[tuple[int, int]]:
    sr, sc = map(int, start_tb)
    er, ec = map(int, end_lr)
    return _join(
        _vpath(sr, int(turn_r), sc),
        _hpath(int(turn_r), sc, int(turn_c)),
        _vpath(int(turn_r), er, int(turn_c)),
        _hpath(er, int(turn_c), ec),
    )


def _select_farthest_pair(coords: list[tuple[int, int]]) -> tuple[tuple[int, int], tuple[int, int]] | None:
    best = None
    for i in range(len(coords)):
        for j in range(i + 1, len(coords)):
            d = abs(int(coords[i][0]) - int(coords[j][0])) + abs(int(coords[i][1]) - int(coords[j][1]))
            if best is None or d > best[0]:
                best = (d, coords[i], coords[j])
    if best is None:
        return None
    _d, a, b = best
    return (int(a[0]), int(a[1])), (int(b[0]), int(b[1]))


def _decode_l_geometry(
    coords: list[tuple[int, int]],
    ring_pts: list[tuple[int, int, int]],
    bbox: tuple[int, int, int, int],
    *,
    r0: int,
    c0: int,
    r1: int,
    c1: int,
    outH: int,
    outW: int,
) -> tuple[tuple[int, int], tuple[int, int], int, int] | None:
    has_top = any(int(rr) == int(r0 - 1) for rr, _cc, _col in ring_pts)
    has_bot = any(int(rr) == int(r1 + 1) for rr, _cc, _col in ring_pts)
    has_left = any(int(cc) == int(c0 - 1) for _rr, cc, _col in ring_pts)
    has_right = any(int(cc) == int(c1 + 1) for _rr, cc, _col in ring_pts)
    if (has_top == has_bot) or (has_left == has_right):
        return None

    top_bot = [tuple(map(int, p)) for p in coords if int(p[0]) in (0, int(outH - 1))]
    left_right = [tuple(map(int, p)) for p in coords if int(p[1]) in (0, int(outW - 1))]
    if len(top_bot) != 1 or len(left_right) != 1:
        return None
    start_tb = top_bot[0]
    end_lr = left_right[0]

    br0, bc0, br1, bc1 = map(int, bbox)
    if has_top and has_left:
        kind = "tl"
        corner_r, corner_c = br0, bc0
        depth_r = int(r0 - corner_r)
        depth_c = int(c0 - corner_c)
    elif has_top and has_right:
        kind = "tr"
        corner_r, corner_c = br0, bc1
        depth_r = int(r0 - corner_r)
        depth_c = int(corner_c - c1)
    elif has_bot and has_left:
        kind = "bl"
        corner_r, corner_c = br1, bc0
        depth_r = int(corner_r - r1)
        depth_c = int(c0 - corner_c)
    else:
        kind = "br"
        corner_r, corner_c = br1, bc1
        depth_r = int(corner_r - r1)
        depth_c = int(corner_c - c1)

    if depth_r <= 0 or depth_c <= 0:
        return None
    turn_r, turn_c = _turn_from_depth(kind, int(outH), int(outW), int(depth_r), int(depth_c))

    if not (0 <= turn_r < outH and 0 <= turn_c < outW):
        return None
    if not (min(int(start_tb[0]), int(end_lr[0])) < int(turn_r) < max(int(start_tb[0]), int(end_lr[0]))):
        return None
    if not (min(int(start_tb[1]), int(end_lr[1])) < int(turn_c) < max(int(start_tb[1]), int(end_lr[1]))):
        return None
    return tuple(map(int, start_tb)), tuple(map(int, end_lr)), int(turn_r), int(turn_c)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recover interior Z-like paths by mirroring the geometry of outside open rectangles.

    Input structure:
    - A uniform background, one closed 1px rectangular frame, and a uniform fill inside it.
    - Colored components outside the frame touch the ring around that frame.
    - Some components form opposite-side U-pairs, while adjacent-side components form open rectangles
      whose outside corner sits a visible row-depth and column-depth away from the frame.

    Rule:
    - Output only the frame interior, initialized with the frame's fill color.
    - Project each ring-touch point onto the interior boundary.
    - Monochrome U-pairs still become solid straight stripes across the crop.
    - For each adjacent-side component, decode the outside corner depths from the component bbox,
      mirror those depths inward to one turn row and one turn column, and draw the resulting Z-like
      4-segment path between the two projected endpoints.
    - Two-color components alternate along that recovered path, with endpoint ring colors fixing the phase.

    Key insight:
    The outer component now tells you more than just which two sides connect: its stand-off distance
    from the frame in the vertical and horizontal directions independently determines how the interior
    path detours before reaching the second side.
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
    comp_ring_actual: dict[int, list[tuple[int, int, int]]] = defaultdict(list)
    for rr, cc in ring:
        cid = int(clab[rr, cc])
        if cid == 0:
            continue
        col = int(g[rr, cc])
        mr, mc = map_to_out(rr, cc)
        comp_pts[cid].append((mr, mc, col))
        comp_ring_actual[cid].append((int(rr), int(cc), int(col)))

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
        for r, c, col in pts:
            hist[(int(r), int(c))][int(col)] += 1
        coords = list(hist.keys())

        rows = {int(r) for r, _c in coords}
        cols2 = {int(c) for _r, c in coords}
        if len(rows) == 1 or len(cols2) == 1:
            if len(cols) == 1:
                for p in coords:
                    deferred[int(cols[0])].add((int(p[0]), int(p[1])))
            continue

        geom = _decode_l_geometry(
            coords,
            comp_ring_actual.get(int(cid), []),
            comp_bbox[int(cid)],
            r0=int(r0),
            c0=int(c0),
            r1=int(r1),
            c1=int(c1),
            outH=int(outH),
            outW=int(outW),
        )
        if geom is None:
            continue
        start_tb, end_lr, turn_r, turn_c = geom

        path = _build_projected_z_path(start_tb, end_lr, int(turn_r), int(turn_c))
        assert len(set(path)) == len(path)
        assert _count_bends(path) == 3

        if len(cols) == 1:
            for rr, cc in path:
                out[int(rr), int(cc)] = int(cols[0])
            continue

        if len(cols) == 2:
            ca, cb = map(int, cols)
            ha = int(hist[tuple(map(int, start_tb))].most_common(1)[0][0])
            hb = int(hist[tuple(map(int, end_lr))].most_common(1)[0][0])

            chosen = None
            for start_color in (ca, cb):
                end_color = start_color if ((len(path) - 1) % 2 == 0) else (cb if start_color == ca else ca)
                if int(start_color) == int(ha) and int(end_color) == int(hb):
                    chosen = int(start_color)
                    break
            assert chosen is not None

            cur = int(chosen)
            for rr, cc in path:
                out[int(rr), int(cc)] = int(cur)
                cur = int(cb) if cur == int(ca) else int(ca)

    for col, pts in deferred.items():
        pts = list(pts)
        if not pts:
            continue
        tops = [p for p in pts if int(p[0]) == 0]
        bots = [p for p in pts if int(p[0]) == outH - 1]
        if tops and bots:
            for cc in sorted({int(c) for _r, c in pts}):
                rs = [int(r) for r, c in pts if int(c) == int(cc)]
                for rr in range(min(rs), max(rs) + 1):
                    out[int(rr), int(cc)] = int(col)
            continue
        lefts = [p for p in pts if int(p[1]) == 0]
        rights = [p for p in pts if int(p[1]) == outW - 1]
        if lefts and rights:
            for rr in sorted({int(r) for r, _c in pts}):
                cs = [int(c) for r, c in pts if int(r) == int(rr)]
                for cc in range(min(cs), max(cs) + 1):
                    out[int(rr), int(cc)] = int(col)
            continue
        if len(pts) == 1:
            rr, cc = pts[0]
            out[int(rr), int(cc)] = int(col)

    return out


# --------------- generator (combo variant) ---------------

_BORDER_COLOR = 3
_BG_CHOICES = [0, 1, 2, 4, 5, 6, 7, 8, 9]
_FILL_EQ_BG_PROB = 0.6
_INTERIOR_H_RANGE = (4, 13)
_INTERIOR_W_RANGE = (7, 14)
_GRID_SIZE_RANGE = (20, 30)
_GRID_INTERIOR_PAD = 10
_FRAME_MARGIN = 4
_FRAME_MARGIN_FAR = 6
_Z_DEPTH_RANGE = (2, 4)
_N_COMPONENTS_RANGE = (2, 3)
_U_PAIR_PROB = 0.0
_DUAL_COLOR_PROB = 0.45
_L_COMPONENT_TRIES = 100
_DISTRACTOR_COUNT_RANGE = (0, 0)
_DISTRACTOR_SIZE_RANGE = (1, 4)
_DISTRACTOR_TRIES = 40
_DISTRACTOR_CLEARANCE = 2
_GEN_RETRIES = 1000


def _can_place(grid: np.ndarray, coords: list[tuple[int, int]], bg: int, comp_mask: np.ndarray) -> bool:
    H, W = grid.shape
    s = set(coords)
    for r, c in s:
        if not (0 <= int(r) < int(H) and 0 <= int(c) < int(W)):
            return False
        if int(grid[int(r), int(c)]) != int(bg):
            return False
        for nr, nc in neighbors4(int(r), int(c)):
            if (int(nr), int(nc)) not in s and 0 <= int(nr) < int(H) and 0 <= int(nc) < int(W) and comp_mask[int(nr), int(nc)]:
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
    if not _can_place(grid, path, int(bg), comp_mask):
        return False
    _paint_path(grid, path, colors)
    rs, cs = zip(*path)
    comp_mask[rs, cs] = True
    return True


def _sample_with_preference(lo: int, hi: int, pref_lo: int, pref_hi: int) -> int | None:
    lo = int(lo)
    hi = int(hi)
    if lo > hi:
        return None
    a = max(int(lo), int(pref_lo))
    b = min(int(hi), int(pref_hi))
    if a <= b:
        return int(random.randint(a, b))
    return int(random.randint(lo, hi))


def _z_specs(
    kind: str,
    H: int,
    W: int,
    r0: int,
    c0: int,
    r1: int,
    c1: int,
    depth_r: int,
    depth_c: int,
) -> tuple[list[tuple[int, int]], list[tuple[int, int]]] | None:
    turn_r, turn_c = _turn_from_depth(kind, int(H), int(W), int(depth_r), int(depth_c))
    if not (0 < int(turn_r) < int(H - 1) and 0 < int(turn_c) < int(W - 1)):
        return None

    if kind == "tl":
        top_c = _sample_with_preference(int(turn_c + 1), int(W - 2), int(turn_c + 1), int(max(turn_c + 1, W // 2)))
        side_r = _sample_with_preference(int(turn_r + 1), int(H - 2), int(turn_r + 1), int(max(turn_r + 1, H // 2 + 1)))
        if top_c is None or side_r is None:
            return None
        start_tb = (0, int(top_c))
        end_lr = (int(side_r), 0)
        R = int(r0 - depth_r)
        C = int(c0 - depth_c)
    elif kind == "tr":
        top_c = _sample_with_preference(int(1), int(turn_c - 1), int(max(1, W // 2)), int(turn_c - 1))
        side_r = _sample_with_preference(int(turn_r + 1), int(H - 2), int(turn_r + 1), int(max(turn_r + 1, H // 2 + 1)))
        if top_c is None or side_r is None:
            return None
        start_tb = (0, int(top_c))
        end_lr = (int(side_r), int(W - 1))
        R = int(r0 - depth_r)
        C = int(c1 + depth_c)
    elif kind == "bl":
        bot_c = _sample_with_preference(int(turn_c + 1), int(W - 2), int(turn_c + 1), int(max(turn_c + 1, W // 2)))
        side_r = _sample_with_preference(int(1), int(turn_r - 1), int(max(1, H // 2 - 1)), int(turn_r - 1))
        if bot_c is None or side_r is None:
            return None
        start_tb = (int(H - 1), int(bot_c))
        end_lr = (int(side_r), 0)
        R = int(r1 + depth_r)
        C = int(c0 - depth_c)
    else:
        bot_c = _sample_with_preference(int(1), int(turn_c - 1), int(max(1, W // 2)), int(turn_c - 1))
        side_r = _sample_with_preference(int(1), int(turn_r - 1), int(max(1, H // 2 - 1)), int(turn_r - 1))
        if bot_c is None or side_r is None:
            return None
        start_tb = (int(H - 1), int(bot_c))
        end_lr = (int(side_r), int(W - 1))
        R = int(r1 + depth_r)
        C = int(c1 + depth_c)

    inner = _build_projected_z_path(start_tb, end_lr, int(turn_r), int(turn_c))
    if len(set(inner)) != len(inner) or _count_bends(inner) != 3:
        return None

    ring_a = (int(r0 - 1) if kind[0] == "t" else int(r1 + 1), int(c0 + 1 + start_tb[1]))
    ring_b = (int(r0 + 1 + end_lr[0]), int(c0 - 1) if kind[1] == "l" else int(c1 + 1))
    outer = _join(
        _vpath(ring_a[0], int(R), ring_a[1]),
        _hpath(int(R), ring_a[1], int(C)),
        _vpath(int(R), ring_b[0], int(C)),
        _hpath(ring_b[0], int(C), ring_b[1]),
    )
    if len(set(outer)) != len(outer):
        return None
    return inner, outer


def _u_pair_specs(orientation: str, H: int, W: int, r0: int, c0: int, r1: int, c1: int):
    lo, hi = 2, 3
    if orientation == "vertical":
        c_a, c_b = sorted(random.sample(range(1, W - 1), 2))
        if c_b - c_a < 2:
            c_b = min(W - 2, c_a + 2)
        top_r = int(r0 - random.randint(lo, hi))
        bot_r = int(r1 + random.randint(lo, hi))
        a_top, b_top = (int(r0 - 1), int(c0 + 1 + c_a)), (int(r0 - 1), int(c0 + 1 + c_b))
        a_bot, b_bot = (int(r1 + 1), int(c0 + 1 + c_a)), (int(r1 + 1), int(c0 + 1 + c_b))
        outer_a = _join(_vpath(a_top[0], top_r, a_top[1]), _hpath(top_r, a_top[1], b_top[1]), _vpath(top_r, b_top[0], b_top[1]))
        outer_b = _join(_vpath(a_bot[0], bot_r, a_bot[1]), _hpath(bot_r, a_bot[1], b_bot[1]), _vpath(bot_r, b_bot[0], b_bot[1]))
        inner = [[(int(r), int(c_a)) for r in range(H)], [(int(r), int(c_b)) for r in range(H)]]
    else:
        r_a, r_b = sorted(random.sample(range(1, H - 1), 2))
        if r_b - r_a < 2:
            r_b = min(H - 2, r_a + 2)
        left_c = int(c0 - random.randint(lo, hi))
        right_c = int(c1 + random.randint(lo, hi))
        a_left, b_left = (int(r0 + 1 + r_a), int(c0 - 1)), (int(r0 + 1 + r_b), int(c0 - 1))
        a_right, b_right = (int(r0 + 1 + r_a), int(c1 + 1)), (int(r0 + 1 + r_b), int(c1 + 1))
        outer_a = _join(_hpath(a_left[0], a_left[1], left_c), _vpath(a_left[0], b_left[0], left_c), _hpath(b_left[0], left_c, b_left[1]))
        outer_b = _join(_hpath(a_right[0], a_right[1], right_c), _vpath(a_right[0], b_right[0], right_c), _hpath(b_right[0], right_c, b_right[1]))
        inner = [[(int(r_a), int(c)) for c in range(W)], [(int(r_b), int(c)) for c in range(W)]]
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
        r = random.randint(1, H - 2)
        c = random.randint(1, W - 2)
        coords = [(int(r), int(c))]
        used = {(int(r), int(c))}
        for _ in range(size - 1):
            base = random.choice(coords)
            opts = [
                (int(nr), int(nc))
                for nr, nc in neighbors4(*base)
                if 1 <= int(nr) < int(H - 1) and 1 <= int(nc) < int(W - 1) and (int(nr), int(nc)) not in used
            ]
            if not opts:
                break
            p = random.choice(opts)
            used.add(p)
            coords.append(p)
        if any(int(r0 - gap) <= int(rr) <= int(r1 + gap) and int(c0 - gap) <= int(cc) <= int(c1 + gap) for rr, cc in coords):
            continue
        if _place_component(grid, comp_mask, coords, (int(color),), int(bg)):
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
    r1, c1 = int(r0 + H + 1), int(c0 + W + 1)

    grid = np.full((rows, cols), int(bg), dtype=int)
    grid[r0, c0 : c1 + 1] = int(bc)
    grid[r1, c0 : c1 + 1] = int(bc)
    grid[r0 : r1 + 1, c0] = int(bc)
    grid[r0 : r1 + 1, c1] = int(bc)
    grid[r0 + 1 : r1, c0 + 1 : c1] = int(fill)

    expected = np.full((H, W), int(fill), dtype=int)
    comp_mask = np.zeros_like(grid, dtype=bool)
    used_inner = np.zeros((H, W), dtype=bool)
    palette = [x for x in range(10) if x not in (bg, bc, fill)]
    random.shuffle(palette)
    color_i = 0

    n_components = random.randint(*_N_COMPONENTS_RANGE)
    remaining = int(n_components)

    if remaining >= 3 and random.random() < _U_PAIR_PROB:
        orientation = random.choice(["vertical", "horizontal"])
        if orientation == "horizontal" and H < 5:
            orientation = "vertical"
        color = int(palette[color_i])
        color_i += 1
        inner_lines, outer_a, outer_b = _u_pair_specs(orientation, H, W, r0, c0, r1, c1)
        all_inner = [p for line in inner_lines for p in line]
        irs, ics = zip(*all_inner)
        if used_inner[irs, ics].any():
            return None
        if not _place_component(grid, comp_mask, outer_a, (color,), int(bg)):
            return None
        if not _place_component(grid, comp_mask, outer_b, (color,), int(bg)):
            return None
        expected[irs, ics] = int(color)
        used_inner[irs, ics] = True
        remaining -= 2

    if remaining <= 0:
        return None

    corners = random.sample(["tl", "tr", "bl", "br"], k=remaining)
    l_count = 0
    for kind in corners:
        max_depth = min(_Z_DEPTH_RANGE[1], H - 2, W - 2)
        if max_depth < _Z_DEPTH_RANGE[0]:
            return None
        chosen = None
        for _ in range(_L_COMPONENT_TRIES):
            depth_r = random.randint(_Z_DEPTH_RANGE[0], max_depth)
            depth_c = random.randint(_Z_DEPTH_RANGE[0], max_depth)
            if max_depth > _Z_DEPTH_RANGE[0] and depth_r == depth_c:
                continue
            built = _z_specs(kind, H, W, r0, c0, r1, c1, depth_r, depth_c)
            if built is None:
                continue
            inner, outer = built
            irs, ics = zip(*inner)
            if used_inner[irs, ics].any():
                continue
            if not _can_place(grid, outer, int(bg), comp_mask):
                continue
            chosen = (inner, outer)
            break
        if chosen is None:
            return None

        inner, outer = chosen
        irs, ics = zip(*inner)
        dual_ok = int(color_i + 1) < len(palette)
        if dual_ok and random.random() < _DUAL_COLOR_PROB:
            colors = (int(palette[color_i]), int(palette[color_i + 1]))
            color_i += 2
        elif color_i < len(palette):
            colors = (int(palette[color_i]),)
            color_i += 1
        else:
            return None
        if not _place_component(grid, comp_mask, outer, colors, int(bg)):
            return None
        _paint_path(expected, inner, colors)
        used_inner[irs, ics] = True
        l_count += 1

    if l_count == 0:
        return None

    for _ in range(random.randint(*_DISTRACTOR_COUNT_RANGE)):
        if color_i >= len(palette):
            break
        _add_distractor(grid, comp_mask, int(bg), (r0, c0, r1, c1), int(palette[color_i]))
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
