# Variant of a6f40cea
# Dimension: D4 — 终点/停止条件 (dst)
# Change: 路径连接同一开放矩形框的两个触碰点（即补全开放框的缺失部分）... → 碰撞停止
#
# Design analysis for this D4 variant
# -----------------------------------
# a) Concretization / behavior trace
# - We keep the original closed frame, outer open-rectangle components, and cropped
#   output of the frame interior.
# - The visible change is only in the interior completion rule for L-components:
#   instead of fully connecting the two projected touchpoints, the path starts from
#   the top/bottom touchpoint, goes inward vertically, turns once, and then stops
#   immediately before the first pre-existing painted obstacle cell.
# - To make "collision stop" observable, generation always includes one monochrome
#   vertical U-pair. As in the original task, those paired outer U-shapes create
#   full-span vertical stripes inside the frame. A separate L-component aims toward
#   one of those stripes and therefore truncates early. Example trace:
#   original rule: top touch -> elbow -> side touch, so the horizontal leg reaches
#   the frame side;
#   new rule: top touch -> elbow -> move toward side, but the next cell is already
#   occupied by a vertical stripe, so the L stops one cell before that stripe.
# - This keeps the original object vocabulary while changing only the destination
#   condition from "other touchpoint" to "first collision".
#
# b) Complexity calibration
# - The variant is not reduced to a trivial ray-casting task: the blocker is itself
#   inferred from another ring-touching object pair, so the solver must still parse
#   multiple outer components and their roles.
# - The generator uses exactly one U-pair plus one L-component. This is enough to
#   show interaction without making the scene cluttered or introducing ambiguous
#   multi-order dependencies between several colliding L-paths.
# - Two-color alternating L-components are preserved, but their phase is now anchored
#   only by the start touchpoint color, because the path no longer reaches the far
#   touchpoint.
#
# c) Hidden-assumption audit against the original main()
# - Original code assumed an L-component can always be turned into a complete
#   Manhattan path between two mapped endpoints. That fails here because the far
#   endpoint is no longer the stop cell.
# - Original code painted L-components before deferred U-pair stripes. Under the new
#   rule that would remove the intended obstacle, so the processing order must be
#   inverted: paint same-side monochrome stripes first, then collision-stopped Ls.
# - Original two-color logic constrained both endpoint colors. That assumption no
#   longer holds; only the start touchpoint anchors the alternating phase.
# - Original unconditional path painting was harmless because no later logic used the
#   path as an obstacle. In this variant, every painted non-fill cell becomes a hard
#   stop, so painting must stop before overwriting occupied cells.
#
# d) Placement legality
# - The generator chooses frame sizes close to the original ranges but raises the
#   minimum width/height slightly so a vertical blocker and a visibly truncated L can
#   coexist.
# - The L start column is chosen away from the blocker columns, and the blocker lies
#   strictly between the elbow column and the intended side edge, ensuring a true
#   collision rather than a degenerate full connection.
# - Outer components are still placed with the original no-overlap / no-adjacency
#   checks, so the input remains solver-friendly.
#
# e) Visual observability
# - The new behavior is visible from the output alone: a full vertical stripe spans
#   the interior, while the L-shaped completion from another object clearly terminates
#   flush against that stripe instead of reaching its second side touchpoint.
# - No invisible ordering by color id is used. Geometry determines start direction
#   and collision.
#
# f) Cleanliness
# - Only one blocker family (vertical stripes) is used, which keeps the interior
#   legible. Distractors remain optional and outside the frame only.
#
# g) Color inference
# - Color usage stays close to the original: the blocker stripes are monochrome; the
#   L can be monochrome or alternating with two colors. The alternating phase is
#   visible at the entering touchpoint and along the painted segment.

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


def _paint_path(grid: np.ndarray, path: list[tuple[int, int]], colors: tuple[int, ...]) -> None:
    if not path:
        return
    rs, cs = zip(*path)
    if len(colors) == 1:
        grid[rs, cs] = int(colors[0])
        return
    a, b = map(int, colors)
    grid[rs, cs] = [a if i % 2 == 0 else b for i in range(len(path))]


def _side_of_point(p: tuple[int, int], outH: int, outW: int) -> str:
    r, c = map(int, p)
    if r == 0:
        return "top"
    if r == outH - 1:
        return "bottom"
    if c == 0:
        return "left"
    if c == outW - 1:
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


def _aggregate_deferred_stripes(
    out: np.ndarray,
    deferred: dict[int, set[tuple[int, int]]],
) -> None:
    outH, outW = map(int, out.shape)
    for col, pts in deferred.items():
        cur = list(pts)
        if not cur:
            continue
        tops = [p for p in cur if int(p[0]) == 0]
        bots = [p for p in cur if int(p[0]) == outH - 1]
        if tops and bots:
            for cc in sorted({int(c) for _r, c in cur}):
                rs = [int(r) for r, c in cur if int(c) == int(cc)]
                for rr in range(min(rs), max(rs) + 1):
                    out[int(rr), int(cc)] = int(col)
            continue
        lefts = [p for p in cur if int(p[1]) == 0]
        rights = [p for p in cur if int(p[1]) == outW - 1]
        if lefts and rights:
            for rr in sorted({int(r) for r, _c in cur}):
                cs = [int(c) for r, c in cur if int(r) == int(rr)]
                for cc in range(min(cs), max(cs) + 1):
                    out[int(rr), int(cc)] = int(col)
            continue
        if len(cur) == 1:
            rr, cc = cur[0]
            out[int(rr), int(cc)] = int(col)
            continue
        best = _choose_farthest_pair(cur)
        if best is None:
            continue
        p1, p2 = best
        for rr, cc in _build_l_path(p1, p2):
            out[int(rr), int(cc)] = int(col)


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
    H, W = map(int, g.shape)
    outH = int(r1 - r0 - 1)
    outW = int(c1 - c0 - 1)
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
        side_a = _side_of_point(a, outH, outW)
        side_b = _side_of_point(b, outH, outW)

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
                "start_side": _side_of_point(start, outH, outW),
            }
        )

    return deferred, l_jobs


def _paint_collision_job(out: np.ndarray, job: dict[str, object], fill: int) -> int:
    path = list(job["path"])
    colors = tuple(job["colors"])
    start = tuple(job["start"])
    hist = job["hist"]

    painted = 0
    if len(colors) == 1:
        col = int(colors[0])
        for rr, cc in path:
            if int(out[int(rr), int(cc)]) != int(fill):
                break
            out[int(rr), int(cc)] = int(col)
            painted += 1
        return int(painted)

    a, b = map(int, colors)
    start_color = int(hist[start].most_common(1)[0][0])
    if start_color not in (a, b):
        start_color = int(a)
    cur = int(start_color)
    for rr, cc in path:
        if int(out[int(rr), int(cc)]) != int(fill):
            break
        out[int(rr), int(cc)] = int(cur)
        painted += 1
        cur = int(b) if int(cur) == int(a) else int(a)
    return int(painted)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reconstruct the frame interior by projecting the outer traces inward until they collide.

    Input structure:
    - One closed 1px rectangular frame with a uniform fill inside.
    - Colored open-rectangle components sit outside the frame and touch the ring around it.
    - Some monochrome same-side components pair up into full vertical or horizontal stripes.

    Rule:
    - Output only the frame interior, initialized with the fill color.
    - First convert paired same-side monochrome touches into full-span stripes, as in the original task family.
    - For each adjacent-side L-component, start from its top/bottom touchpoint, move inward, turn once toward the side touchpoint, and stop before the first already-painted obstacle cell.
    - Monochrome L-components stay solid; two-color ones alternate from the start touchpoint color.

    Key insight: The second touchpoint no longer supplies the destination cell; it only fixes the intended L direction, while the actual endpoint is the first collision with previously painted structure.
    """
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))

    found = _find_framed_rect_uniform_fill(g, bg=int(bg))
    assert found is not None
    r0, c0, r1, c1, bc, fill = found
    outH = int(r1 - r0 - 1)
    outW = int(c1 - c0 - 1)
    out = np.full((outH, outW), int(fill), dtype=int)

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

    _aggregate_deferred_stripes(out, deferred)

    l_jobs.sort(
        key=lambda job: (
            0 if str(job["start_side"]) == "top" else 1,
            int(job["start"][0]),
            int(job["start"][1]),
            int(job["cid"]),
        )
    )
    for job in l_jobs:
        _paint_collision_job(out, job, int(fill))

    return out


# --------------- generator (a6 D4 collision-stop) ---------------

_BORDER_COLOR = 3
_BG_CHOICES = [0, 1, 2, 4, 5, 6, 7, 8, 9]
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
_GEN_RETRIES = 400


def _can_place(grid: np.ndarray, coords: list[tuple[int, int]], bg: int, comp_mask: np.ndarray) -> bool:
    H, W = map(int, grid.shape)
    s = set(coords)
    for r, c in s:
        if not (0 <= int(r) < H and 0 <= int(c) < W):
            return False
        if int(grid[int(r), int(c)]) != int(bg):
            return False
        for nr, nc in neighbors4(int(r), int(c)):
            if (int(nr), int(nc)) not in s and 0 <= int(nr) < H and 0 <= int(nc) < W and comp_mask[int(nr), int(nc)]:
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
    H: int,
    W: int,
    r0: int,
    c0: int,
    r1: int,
    c1: int,
    c_a: int,
    c_b: int,
):
    lo, hi = _OUTER_OFFSET_RANGE
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
    inner = [[(int(r), int(c_a)) for r in range(H)], [(int(r), int(c_b)) for r in range(H)]]
    return inner, outer_a, outer_b


def _l_specs_fixed(
    kind: str,
    elbow_r: int,
    elbow_c: int,
    H: int,
    W: int,
    r0: int,
    c0: int,
    r1: int,
    c1: int,
):
    is_top = kind[0] == "t"
    is_left = kind[1] == "l"
    lo, hi = _L_OUTER_OFFSET_RANGE
    R = int(r0 - random.randint(lo, hi)) if is_top else int(r1 + random.randint(lo, hi))
    C = int(c0 - random.randint(lo, hi)) if is_left else int(c1 + random.randint(lo, hi))
    ring_a = (int(r0 - 1 if is_top else r1 + 1), int(c0 + 1 + elbow_c))
    ring_b = (int(r0 + 1 + elbow_r), int(c0 - 1 if is_left else c1 + 1))
    edge_r = 0 if is_top else H - 1
    edge_c = 0 if is_left else W - 1
    inner = _join(_vpath(edge_r, elbow_r, elbow_c), _hpath(elbow_r, elbow_c, edge_c))
    outer = _join(
        _vpath(ring_a[0], R, ring_a[1]),
        _hpath(R, ring_a[1], C),
        _vpath(R, ring_b[0], C),
        _hpath(ring_b[0], C, ring_b[1]),
    )
    return inner, outer


def _paint_until_collision(grid: np.ndarray, path: list[tuple[int, int]], colors: tuple[int, ...], fill: int) -> int:
    painted = 0
    if len(colors) == 1:
        col = int(colors[0])
        for rr, cc in path:
            if int(grid[int(rr), int(cc)]) != int(fill):
                break
            grid[int(rr), int(cc)] = int(col)
            painted += 1
        return int(painted)

    a, b = map(int, colors)
    cur = int(a)
    for rr, cc in path:
        if int(grid[int(rr), int(cc)]) != int(fill):
            break
        grid[int(rr), int(cc)] = int(cur)
        painted += 1
        cur = int(b) if int(cur) == int(a) else int(a)
    return int(painted)


def _add_distractor(
    grid: np.ndarray,
    comp_mask: np.ndarray,
    bg: int,
    frame: tuple[int, int, int, int],
    color: int,
) -> None:
    H, W = map(int, grid.shape)
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
                if 1 <= int(nr) < H - 1 and 1 <= int(nc) < W - 1 and (int(nr), int(nc)) not in used
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


def _choose_stripe_columns(W: int) -> tuple[int, int] | None:
    candidates = list(range(2, W - 2))
    if len(candidates) < 2:
        return None
    for _ in range(30):
        c_a, c_b = sorted(random.sample(candidates, 2))
        if c_b - c_a >= 2:
            return int(c_a), int(c_b)
    return None


def _choose_l_collision_spec(H: int, W: int, stripes: tuple[int, int]):
    options: list[tuple[str, int, list[int]]] = []
    for blocker in stripes:
        if int(blocker) <= W - 4:
            cand = [
                int(c)
                for c in range(int(blocker + 2), int(W - 1))
                if all(abs(int(c) - int(s)) > 1 for s in stripes)
            ]
            if cand:
                options.append(("tl", int(blocker), cand))
                options.append(("bl", int(blocker), cand))
        if int(blocker) >= 3:
            cand = [
                int(c)
                for c in range(1, int(blocker - 1))
                if all(abs(int(c) - int(s)) > 1 for s in stripes)
            ]
            if cand:
                options.append(("tr", int(blocker), cand))
                options.append(("br", int(blocker), cand))
    if not options:
        return None
    kind, blocker, cand = random.choice(options)
    elbow_c = random.choice(cand)
    elbow_r = random.randint(1, H - 2)
    return str(kind), int(blocker), int(elbow_r), int(elbow_c)


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
    palette = [x for x in range(10) if x not in (bg, bc, fill)]
    random.shuffle(palette)
    if len(palette) < 2:
        return None

    stripe_cols = _choose_stripe_columns(W)
    if stripe_cols is None:
        return None
    inner_lines, outer_top, outer_bot = _vertical_u_pair_specs(H, W, r0, c0, r1, c1, stripe_cols[0], stripe_cols[1])
    stripe_color = int(palette[0])
    if not _place_component(grid, comp_mask, outer_top, (stripe_color,), int(bg)):
        return None
    if not _place_component(grid, comp_mask, outer_bot, (stripe_color,), int(bg)):
        return None
    for line in inner_lines:
        rs, cs = zip(*line)
        expected[rs, cs] = int(stripe_color)

    spec = _choose_l_collision_spec(H, W, stripe_cols)
    if spec is None:
        return None
    kind, blocker, elbow_r, elbow_c = spec
    full_inner, outer_l = _l_specs_fixed(kind, elbow_r, elbow_c, H, W, r0, c0, r1, c1)

    if random.random() < _DUAL_COLOR_PROB and len(palette) >= 3:
        l_colors = (int(palette[1]), int(palette[2]))
        next_color_idx = 3
    else:
        l_colors = (int(palette[1]),)
        next_color_idx = 2

    if not _place_component(grid, comp_mask, outer_l, l_colors, int(bg)):
        return None

    painted = _paint_until_collision(expected, full_inner, l_colors, int(fill))
    if painted <= 0 or painted >= len(full_inner):
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
    raise RuntimeError("failed to generate a valid a6f40cea D4 input")
