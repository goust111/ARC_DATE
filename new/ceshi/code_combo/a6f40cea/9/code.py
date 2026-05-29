# Combo variant of a6f40cea (index 9)
# combo_spec: D4/2+D6/0
# Name: 碰撞停止并变色
#
# Design analysis for the fused variant
# ------------------------------------
# a) Concretization / behavior trace
# - We keep the original object vocabulary: one closed rectangular frame with a
#   uniform fill, plus open rectangular traces outside the frame that touch the
#   ring around it.
# - The fused change is applied inside the cropped frame interior. A monochrome
#   vertical U-pair still reconstructs into two full-height blocker stripes.
#   A separate L-component starts from its top/bottom touchpoint, travels inward
#   vertically, turns once toward its side touchpoint, and does not complete the
#   full connection. Instead it stops as soon as the next intended cell already
#   belongs to an earlier path. That occupied cell is recolored to a fixed third
#   color. Example trace:
#   original task: the L reaches the side edge and paints every cell on its
#   Manhattan route;
#   D4-only logic: the L stops one cell before the blocker stripe;
#   fused logic here: the L still stops at that blocker, but the blocker cell at
#   the contact point becomes the collision color, making the stopping event
#   visible rather than implicit.
#
# b) Complexity calibration
# - The generator always includes one U-pair and one L-component. This keeps the
#   scene small enough to read, while still forcing the solver to infer both the
#   blocker path and the collision-stop rule.
# - The collision color is fixed across samples and is excluded from background,
#   fill, and path palettes. A human can therefore recognize it as an event
#   marker rather than an arbitrary path color.
# - The L may be monochrome or alternating, so the original endpoint-color phase
#   cue is still relevant at the starting touchpoint, but the far touchpoint is
#   now directional evidence instead of a painted destination.
#
# c) Hidden-assumption audit against the original main()
# - Original code assumed every adjacent-side component should become a complete
#   Manhattan connection between two mapped touchpoints. That is false here: the
#   second touchpoint only determines the intended direction after the elbow.
# - Original direct writes into the output treated overlaps as harmless
#   overwrites. Here occupancy is semantic: once stripes are painted, they become
#   blockers, and a collision with them must recolor the occupied cell.
# - Original alternating-color logic constrained both endpoint colors. After the
#   destination rule changes, only the start touchpoint color remains reliable.
# - Grouping same-side monochrome touchpoints by color now carries a stronger
#   assumption: only the intentional U-pair shares a color. The generator
#   guarantees that, so deferred stripe reconstruction stays unambiguous.
#
# d) Placement legality
# - The frame keeps the original scale band, but the minimum interior size is
#   raised slightly so a full blocker stripe and a visibly truncated L can
#   coexist.
# - The blocker stripe column is chosen strictly between the L elbow column and
#   its target side, so the collision always happens on the horizontal leg and
#   never degenerates into a full connection.
# - Outer components are still validated for in-bounds placement, no overlap, and
#   no 4-neighbor contact with other components.
#
# e) Visual observability
# - Every generated sample contains a real stop event and exactly one collision
#   color pixel inside the frame. The output visibly differs from both the base
#   task and the D4-only variant because the blocker is punctuated at the exact
#   point where the truncated L arrives.
# - No hidden ordering by numeric color id is used. The only order dependence is
#   geometric and visible: full-span stripes are reconstructed first because they
#   originate from same-side paired touches.
#
# f) Visual cleanliness
# - The interior contains only two blocker stripes and one truncated L. This
#   avoids segment-sharing clutter while leaving enough empty fill area for the
#   stop to be legible.
# - Distractors remain outside the frame ring only, so they do not compete with
#   the central collision cue.
#
# g) Change inferability
# - Color changes are tightly constrained: all path colors come from a palette
#   that excludes the collision color, and the collision color appears only at a
#   path/path impact point. Across a few examples, the repeated structure
#   "truncated L ending against a stripe + lone third-color pixel at the contact"
#   is enough to recover the fused rule.

from __future__ import annotations

from collections import Counter, defaultdict
import random

import numpy as np
from scipy.ndimage import label

from arc_utils import STRUCT4, get_bg, neighbors4


_COLLISION_COLOR = 2


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
            cell = (int(p[0]), int(p[1]))
            if not out or out[-1] != cell:
                out.append(cell)
    return out


def _paint_path(grid: np.ndarray, path: list[tuple[int, int]], colors: tuple[int, ...]) -> None:
    if not path:
        return
    rs, cs = zip(*path)
    if len(colors) == 1:
        grid[rs, cs] = int(colors[0])
        return
    a, b = map(int, colors)
    grid[rs, cs] = [a if i % 2 == 0 else b for i in range(len(path))]


def _side_of_point(p: tuple[int, int], out_h: int, out_w: int) -> str:
    r, c = map(int, p)
    if r == 0:
        return "top"
    if r == out_h - 1:
        return "bottom"
    if c == 0:
        return "left"
    if c == out_w - 1:
        return "right"
    return "inner"


def _choose_farthest_pair(coords: list[tuple[int, int]]) -> tuple[tuple[int, int], tuple[int, int]] | None:
    best = None
    for i in range(len(coords)):
        for j in range(i + 1, len(coords)):
            a = coords[i]
            b = coords[j]
            d = abs(int(a[0]) - int(b[0])) + abs(int(a[1]) - int(b[1]))
            if best is None or d > best[0]:
                best = (d, a, b)
    if best is None:
        return None
    return (int(best[1][0]), int(best[1][1])), (int(best[2][0]), int(best[2][1]))


def _build_l_path(start: tuple[int, int], dest: tuple[int, int]) -> list[tuple[int, int]]:
    ra, ca = map(int, start)
    rb, cb = map(int, dest)
    return _join(_vpath(ra, rb, ca), _hpath(rb, ca, cb))


def _can_place(grid: np.ndarray, coords: list[tuple[int, int]], bg: int, comp_mask: np.ndarray) -> bool:
    h, w = map(int, grid.shape)
    s = set(coords)
    for r, c in s:
        if not (0 <= int(r) < h and 0 <= int(c) < w):
            return False
        if int(grid[int(r), int(c)]) != int(bg):
            return False
        for nr, nc in neighbors4(int(r), int(c)):
            if (int(nr), int(nc)) not in s and 0 <= int(nr) < h and 0 <= int(nc) < w and comp_mask[int(nr), int(nc)]:
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


def _vertical_u_pair_specs(
    h: int,
    w: int,
    r0: int,
    c0: int,
    r1: int,
    c1: int,
    c_a: int,
    c_b: int,
    *,
    offset_range: tuple[int, int],
) -> tuple[list[list[tuple[int, int]]], list[tuple[int, int]], list[tuple[int, int]]]:
    lo, hi = offset_range
    top_r = int(r0 - random.randint(lo, hi))
    bot_r = int(r1 + random.randint(lo, hi))
    a_top = (int(r0 - 1), int(c0 + 1 + c_a))
    b_top = (int(r0 - 1), int(c0 + 1 + c_b))
    a_bot = (int(r1 + 1), int(c0 + 1 + c_a))
    b_bot = (int(r1 + 1), int(c0 + 1 + c_b))
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
    inner = [[(int(r), int(c_a)) for r in range(h)], [(int(r), int(c_b)) for r in range(h)]]
    return inner, outer_a, outer_b


def _l_specs_fixed(
    kind: str,
    elbow_r: int,
    elbow_c: int,
    h: int,
    w: int,
    r0: int,
    c0: int,
    r1: int,
    c1: int,
    *,
    offset_range: tuple[int, int],
) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    is_top = kind[0] == "t"
    is_left = kind[1] == "l"
    lo, hi = offset_range
    outer_r = int(r0 - random.randint(lo, hi)) if is_top else int(r1 + random.randint(lo, hi))
    outer_c = int(c0 - random.randint(lo, hi)) if is_left else int(c1 + random.randint(lo, hi))
    ring_a = (int(r0 - 1 if is_top else r1 + 1), int(c0 + 1 + elbow_c))
    ring_b = (int(r0 + 1 + elbow_r), int(c0 - 1 if is_left else c1 + 1))
    edge_r = 0 if is_top else h - 1
    edge_c = 0 if is_left else w - 1
    inner = _join(_vpath(edge_r, elbow_r, elbow_c), _hpath(elbow_r, elbow_c, edge_c))
    outer = _join(
        _vpath(ring_a[0], outer_r, ring_a[1]),
        _hpath(outer_r, ring_a[1], outer_c),
        _vpath(outer_r, ring_b[0], outer_c),
        _hpath(ring_b[0], outer_c, ring_b[1]),
    )
    return inner, outer


def _start_color_for_job(job: dict[str, object]) -> int:
    colors = tuple(map(int, job["colors"]))
    if len(colors) == 1:
        return int(colors[0])
    start = tuple(job["start"])
    hist = job["hist"]
    start_color = int(hist[start].most_common(1)[0][0])
    if start_color not in colors:
        start_color = int(colors[0])
    return int(start_color)


def _stripe_paths_from_deferred(
    deferred: dict[int, set[tuple[int, int]]],
    out_h: int,
    out_w: int,
) -> list[tuple[list[tuple[int, int]], int]]:
    stripe_paths: list[tuple[list[tuple[int, int]], int]] = []
    for col, pts in deferred.items():
        cur = list(pts)
        if not cur:
            continue
        tops = [p for p in cur if int(p[0]) == 0]
        bots = [p for p in cur if int(p[0]) == out_h - 1]
        if tops and bots:
            for cc in sorted({int(c) for _r, c in cur}):
                rs = [int(r) for r, c in cur if int(c) == int(cc)]
                path = [(int(rr), int(cc)) for rr in range(min(rs), max(rs) + 1)]
                stripe_paths.append((path, int(col)))
            continue
        lefts = [p for p in cur if int(p[1]) == 0]
        rights = [p for p in cur if int(p[1]) == out_w - 1]
        if lefts and rights:
            for rr in sorted({int(r) for r, _c in cur}):
                cs = [int(c) for r, c in cur if int(r) == int(rr)]
                path = [(int(rr), int(cc)) for cc in range(min(cs), max(cs) + 1)]
                stripe_paths.append((path, int(col)))
            continue
        if len(cur) == 1:
            rr, cc = cur[0]
            stripe_paths.append(([(int(rr), int(cc))], int(col)))
            continue
        pair = _choose_farthest_pair(cur)
        if pair is None:
            continue
        p1, p2 = pair
        stripe_paths.append((_build_l_path(p1, p2), int(col)))
    return stripe_paths


def _collect_component_specs(
    g: np.ndarray,
    *,
    bg: int,
    bc: int,
    fill: int,
    r0: int,
    c0: int,
    r1: int,
    c1: int,
) -> tuple[dict[int, set[tuple[int, int]]], list[dict[str, object]]]:
    h, w = map(int, g.shape)
    out_h = int(r1 - r0 - 1)
    out_w = int(c1 - c0 - 1)
    r_int0 = int(r0 + 1)
    c_int0 = int(c0 + 1)

    def map_to_out(rr: int, cc: int) -> tuple[int, int]:
        r = max(0, min(out_h - 1, int(rr - r_int0)))
        c = max(0, min(out_w - 1, int(cc - c_int0)))
        return int(r), int(c)

    comp_mask = (g != int(bg)) & (g != int(fill)) & (g != int(bc))
    clab, cn = label(comp_mask, structure=STRUCT4)

    ring = set()
    for cc in range(c0, c1 + 1):
        for rr in (r0 - 1, r1 + 1):
            if 0 <= rr < h:
                ring.add((int(rr), int(cc)))
    for rr in range(r0, r1 + 1):
        for cc in (c0 - 1, c1 + 1):
            if 0 <= cc < w:
                ring.add((int(rr), int(cc)))

    comp_pts: dict[int, list[tuple[int, int, int]]] = defaultdict(list)
    for rr, cc in ring:
        cid = int(clab[int(rr), int(cc)])
        if cid == 0:
            continue
        mr, mc = map_to_out(int(rr), int(cc))
        comp_pts[cid].append((int(mr), int(mc), int(g[int(rr), int(cc)])))

    comp_cols: dict[int, list[int]] = {}
    for cid in range(1, int(cn) + 1):
        comp_cols[cid] = sorted(set(map(int, g[clab == int(cid)].flatten())))

    deferred: dict[int, set[tuple[int, int]]] = defaultdict(set)
    l_jobs: list[dict[str, object]] = []

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
        rows = {int(r) for r, _c in coords}
        cols1 = {int(c) for _r, c in coords}

        if len(cols) == 1 and (len(rows) == 1 or len(cols1) == 1):
            for p in coords:
                deferred[int(cols[0])].add((int(p[0]), int(p[1])))
            continue

        pair = _choose_farthest_pair(coords)
        if pair is None:
            continue
        a, b = pair
        side_a = _side_of_point(a, out_h, out_w)
        side_b = _side_of_point(b, out_h, out_w)
        if side_a in ("top", "bottom") and side_b in ("left", "right"):
            start, dest = a, b
        elif side_b in ("top", "bottom") and side_a in ("left", "right"):
            start, dest = b, a
        else:
            continue

        l_jobs.append(
            {
                "cid": int(cid),
                "path": _build_l_path(start, dest),
                "start": (int(start[0]), int(start[1])),
                "dest": (int(dest[0]), int(dest[1])),
                "colors": tuple(map(int, cols)),
                "hist": hist,
                "start_side": _side_of_point(start, out_h, out_w),
            }
        )

    return deferred, l_jobs


def _paint_collision_job(
    out: np.ndarray,
    job: dict[str, object],
    fill: int,
) -> tuple[int, tuple[int, int] | None]:
    path = list(job["path"])
    colors = tuple(map(int, job["colors"]))
    cur = int(_start_color_for_job(job))
    alt = len(colors) == 2
    other = None if not alt else int(colors[1] if int(cur) == int(colors[0]) else colors[0])

    painted = 0
    for rr, cc in path:
        if int(out[int(rr), int(cc)]) != int(fill):
            out[int(rr), int(cc)] = int(_COLLISION_COLOR)
            return int(painted), (int(rr), int(cc))
        out[int(rr), int(cc)] = int(cur)
        painted += 1
        if alt:
            assert other is not None
            cur, other = other, cur
    return int(painted), None


def _simulate_output(
    out_h: int,
    out_w: int,
    fill: int,
    stripe_paths: list[tuple[list[tuple[int, int]], int]],
    l_jobs: list[dict[str, object]],
) -> tuple[np.ndarray, dict[str, int]]:
    out = np.full((int(out_h), int(out_w)), int(fill), dtype=int)
    metrics = {
        "stripe_paths": 0,
        "l_jobs": 0,
        "stopped_l_jobs": 0,
        "collision_events": 0,
        "red_cells": 0,
    }

    for path, col in stripe_paths:
        if not path:
            continue
        metrics["stripe_paths"] += 1
        rs, cs = zip(*path)
        out[rs, cs] = int(col)

    l_jobs_sorted = sorted(
        l_jobs,
        key=lambda job: (
            0 if str(job["start_side"]) == "top" else 1,
            int(job["start"][0]),
            int(job["start"][1]),
            int(job["cid"]),
        ),
    )

    for job in l_jobs_sorted:
        metrics["l_jobs"] += 1
        _painted, hit = _paint_collision_job(out, job, int(fill))
        if hit is not None:
            metrics["stopped_l_jobs"] += 1
            metrics["collision_events"] += 1

    metrics["red_cells"] = int(np.count_nonzero(out == int(_COLLISION_COLOR)))
    return out, metrics


def analyze_fusion_behavior(grid: np.ndarray) -> dict[str, int]:
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))
    found = _find_framed_rect_uniform_fill(g, bg=int(bg))
    assert found is not None
    r0, c0, r1, c1, bc, fill = found
    out_h = int(r1 - r0 - 1)
    out_w = int(c1 - c0 - 1)
    deferred, l_jobs = _collect_component_specs(
        g,
        bg=int(bg),
        bc=int(bc),
        fill=int(fill),
        r0=int(r0),
        c0=int(c0),
        r1=int(r1),
        c1=int(c1),
    )
    stripe_paths = _stripe_paths_from_deferred(deferred, out_h, out_w)
    _out, metrics = _simulate_output(out_h, out_w, int(fill), stripe_paths, l_jobs)
    return metrics


# Rebuild the interior paths implied by the outer traces, but stop later paths on contact and mark the impact.
# Input structure:
# - One closed 1px rectangular frame encloses a uniformly filled interior.
# - Outside it, ring-touching open-rectangle traces encode either same-side stripes or adjacent-side L-paths.
# Rule:
# - Crop to the frame interior, draw blocker stripes first, then extend each L from its top/bottom touchpoint until it would enter an occupied cell; recolor that impact cell to the fixed collision color.
# Key insight:
# - The far touchpoint of an L now supplies direction rather than a painted endpoint, so the solver must combine reconstruction order, collision stopping, and collision recoloring.
def main(grid: np.ndarray) -> np.ndarray:
    """
    Rebuild the interior paths implied by the outer traces, but stop later paths on contact and mark the impact.

    Input structure:
    - One closed 1px rectangular frame encloses a uniformly filled interior.
    - Outside the frame, colored open-rectangle components touch the ring around that frame.
    - Same-side monochrome touches encode full-span blocker stripes, while adjacent-side touches encode inward L-paths.

    Rule:
    - Crop to the frame interior and initialize it with the frame's fill color.
    - First reconstruct the same-side monochrome stripes exactly as in the base family.
    - Then process each adjacent-side L-component from its top/bottom touchpoint inward, turning once toward the side touchpoint.
    - If the next intended L cell is already occupied by an earlier reconstructed path, stop there and recolor that occupied cell to a fixed third collision color. Otherwise paint normally, using a solid color or an alternating pair anchored by the start touchpoint color.

    Key insight:
    - The far touchpoint of an L no longer marks a painted endpoint. It only reveals which side the path was trying to reach; the actual endpoint is the first occupied cell encountered after stripes and earlier paths are laid down, and that impact cell is specially recolored.
    """
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))

    found = _find_framed_rect_uniform_fill(g, bg=int(bg))
    assert found is not None
    r0, c0, r1, c1, bc, fill = found
    out_h = int(r1 - r0 - 1)
    out_w = int(c1 - c0 - 1)

    deferred, l_jobs = _collect_component_specs(
        g,
        bg=int(bg),
        bc=int(bc),
        fill=int(fill),
        r0=int(r0),
        c0=int(c0),
        r1=int(r1),
        c1=int(c1),
    )
    stripe_paths = _stripe_paths_from_deferred(deferred, out_h, out_w)
    out, _metrics = _simulate_output(out_h, out_w, int(fill), stripe_paths, l_jobs)
    return out


_BORDER_COLOR = 3
_BG_CHOICES = [0, 1, 4, 5, 6, 7, 8, 9]
_FILL_EQ_BG_PROB = 0.6
_INTERIOR_H_RANGE = (5, 13)
_INTERIOR_W_RANGE = (8, 14)
_GRID_SIZE_RANGE = (20, 30)
_GRID_INTERIOR_PAD = 10
_FRAME_MARGIN = 4
_FRAME_MARGIN_FAR = 6
_OUTER_OFFSET_RANGE = (2, 3)
_L_OUTER_OFFSET_RANGE = (4, 5)
_DUAL_COLOR_PROB = 0.45
_DISTRACTOR_COUNT_RANGE = (0, 1)
_DISTRACTOR_SIZE_RANGE = (1, 4)
_DISTRACTOR_TRIES = 40
_DISTRACTOR_CLEARANCE = 2
_GEN_RETRIES = 500


def _choose_stripe_columns(w: int) -> tuple[int, int] | None:
    candidates = list(range(2, w - 2))
    if len(candidates) < 2:
        return None
    for _ in range(40):
        c_a, c_b = sorted(random.sample(candidates, 2))
        if c_b - c_a >= 2:
            return int(c_a), int(c_b)
    return None


def _choose_l_collision_spec(h: int, w: int, stripes: tuple[int, int]):
    options: list[tuple[str, int, list[int]]] = []
    for blocker in stripes:
        if int(blocker) <= w - 4:
            candidates = [
                int(c)
                for c in range(int(blocker + 2), int(w - 1))
                if all(abs(int(c) - int(s)) > 1 for s in stripes)
            ]
            if candidates:
                options.append(("tl", int(blocker), candidates))
                options.append(("bl", int(blocker), candidates))
        if int(blocker) >= 3:
            candidates = [
                int(c)
                for c in range(1, int(blocker - 1))
                if all(abs(int(c) - int(s)) > 1 for s in stripes)
            ]
            if candidates:
                options.append(("tr", int(blocker), candidates))
                options.append(("br", int(blocker), candidates))
    if not options:
        return None
    kind, blocker, candidates = random.choice(options)
    elbow_c = random.choice(candidates)
    elbow_r = random.randint(1, h - 2)
    return str(kind), int(blocker), int(elbow_r), int(elbow_c)


def _render_expected(
    h: int,
    w: int,
    fill: int,
    stripe_cols: tuple[int, int],
    stripe_color: int,
    l_path: list[tuple[int, int]],
    l_colors: tuple[int, ...],
) -> tuple[np.ndarray, dict[str, int]]:
    stripe_paths = [
        ([(int(r), int(stripe_cols[0])) for r in range(h)], int(stripe_color)),
        ([(int(r), int(stripe_cols[1])) for r in range(h)], int(stripe_color)),
    ]
    l_job = {
        "cid": 1,
        "path": list(l_path),
        "start": tuple(l_path[0]),
        "dest": tuple(l_path[-1]),
        "colors": tuple(map(int, l_colors)),
        "hist": defaultdict(Counter, {tuple(l_path[0]): Counter({int(l_colors[0]): 1})}),
        "start_side": "top" if int(l_path[0][0]) == 0 else "bottom",
    }
    return _simulate_output(int(h), int(w), int(fill), stripe_paths, [l_job])


def _add_distractor(
    grid: np.ndarray,
    comp_mask: np.ndarray,
    bg: int,
    frame: tuple[int, int, int, int],
    color: int,
) -> None:
    h, w = map(int, grid.shape)
    r0, c0, r1, c1 = frame
    gap = _DISTRACTOR_CLEARANCE
    sz_lo, sz_hi = _DISTRACTOR_SIZE_RANGE
    for _ in range(_DISTRACTOR_TRIES):
        size = random.randint(sz_lo, sz_hi)
        r = random.randint(1, h - 2)
        c = random.randint(1, w - 2)
        coords = [(int(r), int(c))]
        used = {(int(r), int(c))}
        for _step in range(size - 1):
            base = random.choice(coords)
            opts = [
                (int(nr), int(nc))
                for nr, nc in neighbors4(*base)
                if 1 <= int(nr) < h - 1 and 1 <= int(nc) < w - 1 and (int(nr), int(nc)) not in used
            ]
            if not opts:
                break
            p = random.choice(opts)
            used.add((int(p[0]), int(p[1])))
            coords.append((int(p[0]), int(p[1])))
        if any(r0 - gap <= int(rr) <= r1 + gap and c0 - gap <= int(cc) <= c1 + gap for rr, cc in coords):
            continue
        if _place_component(grid, comp_mask, coords, (int(color),), int(bg)):
            return


def _gen_attempt() -> np.ndarray | None:
    bg = random.choice(_BG_CHOICES)
    bc = _BORDER_COLOR
    fill_choices = [x for x in range(10) if x not in (bg, bc, _COLLISION_COLOR)]
    fill = int(bg) if random.random() < _FILL_EQ_BG_PROB else int(random.choice(fill_choices))

    h = random.randint(*_INTERIOR_H_RANGE)
    w = random.randint(*_INTERIOR_W_RANGE)
    rows = random.randint(max(_GRID_SIZE_RANGE[0], h + _GRID_INTERIOR_PAD), _GRID_SIZE_RANGE[1])
    cols = random.randint(max(_GRID_SIZE_RANGE[0], w + _GRID_INTERIOR_PAD), _GRID_SIZE_RANGE[1])
    r0 = random.randint(_FRAME_MARGIN, rows - h - _FRAME_MARGIN_FAR)
    c0 = random.randint(_FRAME_MARGIN, cols - w - _FRAME_MARGIN_FAR)
    r1, c1 = int(r0 + h + 1), int(c0 + w + 1)

    grid = np.full((rows, cols), int(bg), dtype=int)
    grid[r0, c0 : c1 + 1] = int(bc)
    grid[r1, c0 : c1 + 1] = int(bc)
    grid[r0 : r1 + 1, c0] = int(bc)
    grid[r0 : r1 + 1, c1] = int(bc)
    grid[r0 + 1 : r1, c0 + 1 : c1] = int(fill)

    comp_mask = np.zeros_like(grid, dtype=bool)
    palette = [x for x in range(10) if x not in (bg, bc, fill, _COLLISION_COLOR)]
    random.shuffle(palette)
    if len(palette) < 2:
        return None

    stripe_cols = _choose_stripe_columns(w)
    if stripe_cols is None:
        return None
    inner_lines, outer_top, outer_bot = _vertical_u_pair_specs(
        h,
        w,
        r0,
        c0,
        r1,
        c1,
        stripe_cols[0],
        stripe_cols[1],
        offset_range=_OUTER_OFFSET_RANGE,
    )
    stripe_color = int(palette[0])
    if not _place_component(grid, comp_mask, outer_top, (int(stripe_color),), int(bg)):
        return None
    if not _place_component(grid, comp_mask, outer_bot, (int(stripe_color),), int(bg)):
        return None

    l_spec = _choose_l_collision_spec(h, w, stripe_cols)
    if l_spec is None:
        return None
    kind, blocker, elbow_r, elbow_c = l_spec
    l_inner, l_outer = _l_specs_fixed(
        kind,
        elbow_r,
        elbow_c,
        h,
        w,
        r0,
        c0,
        r1,
        c1,
        offset_range=_L_OUTER_OFFSET_RANGE,
    )

    if len(palette) >= 3 and random.random() < _DUAL_COLOR_PROB:
        l_colors = (int(palette[1]), int(palette[2]))
        next_color_idx = 3
    else:
        l_colors = (int(palette[1]),)
        next_color_idx = 2

    if not _place_component(grid, comp_mask, l_outer, l_colors, int(bg)):
        return None

    expected, metrics = _render_expected(
        h,
        w,
        int(fill),
        stripe_cols,
        int(stripe_color),
        l_inner,
        l_colors,
    )
    if metrics["stripe_paths"] != 2:
        return None
    if metrics["l_jobs"] != 1:
        return None
    if metrics["stopped_l_jobs"] != 1 or metrics["collision_events"] != 1:
        return None
    if metrics["red_cells"] != 1:
        return None

    collision_cell = (int(elbow_r), int(blocker))
    if int(expected[collision_cell]) != int(_COLLISION_COLOR):
        return None
    if int(expected[l_inner[-1][0], l_inner[-1][1]]) == int(_COLLISION_COLOR):
        return None

    for _ in range(random.randint(*_DISTRACTOR_COUNT_RANGE)):
        if next_color_idx >= len(palette):
            break
        _add_distractor(grid, comp_mask, int(bg), (r0, c0, r1, c1), int(palette[next_color_idx]))
        next_color_idx += 1

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
