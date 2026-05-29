# Variant of a6f40cea
# Dimension: D8 — 拐角选择规则 (path)
# Change: L形路径拐角选择使其严格位于内部（不在边缘上），优先选择cornerA方案... → 多拐角阶梯路径

from __future__ import annotations

from collections import Counter, defaultdict
import random

import numpy as np
from scipy.ndimage import label

from arc_utils import STRUCT4, get_bg, neighbors4

# Design analysis
# a) Concrete behavior and trace
#    - Keep the original global scene: one large closed rectangular frame, 2-3 colored
#      components outside it, and optional distractors that do not touch the frame ring.
#    - Only L-type components change. Their two ring-touch points still project to two
#      adjacent sides of the output interior, but the interior path is no longer a 2-segment
#      L. It becomes a monotone staircase with multiple interior bends.
#    - The visible coding attribute is the external open-rectangle depth: both partial stubs
#      extend equally far from the frame. Depth 2 means a 2-step staircase; depth 3 means a
#      3-step staircase. The solver decodes that depth from the component bbox relative to the
#      frame ring.
#    - Small trace:
#      old: top touch at col 5 and left touch at row 4 -> pick one elbow -> draw one L.
#      new: same projected endpoints, but if the outside open rectangle protrudes 3 cells away
#      from the frame on both touched sides, split the row/col distances into 3 positive parts
#      each and draw V-H-V-H-V-H (or H-V-H-V-H-V) as a regular staircase.
# b) Complexity calibration
#    - Only two staircase depths are used: 2 and 3. That is enough for a visible rule change,
#      while keeping inference manageable from 2-4 ARC examples.
#    - The depth value repeats within a sample whenever multiple L-components appear, because
#      each component carries its own visible outside depth. We avoid many depth values that
#      would weaken the mapping signal.
# c) Hidden assumptions in the original main() that no longer hold
#    - build_manhattan/inside_corner assumed every non-straight component had exactly one elbow.
#      That fails here because the path structure is determined by an encoded step count.
#    - The monochrome and two-color branches both assumed "farthest pair + one of two L-corners"
#      fully described the interior shape. Now the same endpoint pair can yield two different
#      outputs depending on outside depth, so main() must inspect component geometry, not only
#      ring endpoints.
#    - The two-color phase logic remains valid only because the staircase path length is still
#      |dr|+|dc|+1, so endpoint parity is unchanged relative to the old L-path case.
# d) Placement legality
#    - Generator chooses staircase step count before endpoint coordinates, then constrains each
#      endpoint distance to its adjacent corner to be at least that step count. This guarantees
#      each row/col partition is positive and all bends stay inside the interior.
#    - Each L-component still uses one unique corner kind among tl/tr/bl/br, reducing overlap.
#      The outside component is validated by the same non-overlap and non-adjacency checks.
# e) Visual observability
#    - The rule depends only on visible geometry: how far the outer open rectangle stands off
#      from the frame. No sorting by color id or other hidden metadata is used.
#    - Depth 2 and depth 3 produce visibly different numbers of staircase "teeth", so the new
#      behavior is recoverable by eye from a few examples.
# f) Visual cleanliness
#    - Endpoint coordinates are biased toward their local corner sectors, as in the original
#      generator, so staircases stay localized instead of forming a tangled central mesh.
#    - U-pairs and distractors remain unchanged; they occupy distinct visual roles and do not
#      share cells with staircase paths.
# g) Color inferability
#    - Colors keep the original semantics: monochrome components paint solid staircases and
#      two-color components alternate along the staircase, with endpoint ring colors fixing the
#      phase. No extra color mapping is introduced.
# 9) New solver assumptions and generator guarantees
#    - Solver assumes every non-U ring-touching L-component has exactly one top/bottom touch and
#      one left/right touch, and that the outside bbox depth on those two sides is equal or at
#      least consistent enough that min(depth_v, depth_h) recovers the intended step count.
#      Generator enforces equal stub depths for all staircase components.
#    - Solver assumes the decoded step count does not exceed the available row and column
#      distances between projected endpoints. Generator enforces that by sampling endpoints only
#      after choosing the step count.


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


def _paint_path(grid: np.ndarray, path: list[tuple[int, int]], colors: tuple[int, ...]) -> None:
    if not path:
        return
    rs, cs = zip(*path)
    if len(colors) == 1:
        grid[rs, cs] = colors[0]
    else:
        a, b = colors
        vals = [a if i % 2 == 0 else b for i in range(len(path))]
        grid[rs, cs] = vals


def _split_positive(total: int, parts: int) -> list[int]:
    total = int(total)
    parts = int(parts)
    assert parts >= 1
    assert total >= parts
    base, rem = divmod(total, parts)
    return [base + (1 if i < rem else 0) for i in range(parts)]


def _boundary_axis(p: tuple[int, int], H: int, W: int) -> str:
    r, c = map(int, p)
    on_tb = r in (0, H - 1)
    on_lr = c in (0, W - 1)
    if on_tb and not on_lr:
        return "r"
    if on_lr and not on_tb:
        return "c"
    if on_tb:
        return "r"
    return "c"


def _build_staircase_path(
    p1: tuple[int, int],
    p2: tuple[int, int],
    H: int,
    W: int,
    steps: int,
) -> list[tuple[int, int]]:
    (ra, ca), (rb, cb) = (tuple(map(int, p1)), tuple(map(int, p2)))
    if ra == rb:
        return [(ra, c) for c in _line(ca, cb)]
    if ca == cb:
        return [(r, ca) for r in _line(ra, rb)]

    dr = int(rb - ra)
    dc = int(cb - ca)
    steps = max(1, min(int(steps), abs(dr), abs(dc)))
    row_parts = _split_positive(abs(dr), steps)
    col_parts = _split_positive(abs(dc), steps)
    sign_r = 1 if dr > 0 else -1
    sign_c = 1 if dc > 0 else -1
    axis = _boundary_axis((ra, ca), int(H), int(W))

    cur_r, cur_c = int(ra), int(ca)
    path = [(cur_r, cur_c)]
    row_i = 0
    col_i = 0
    for _ in range(2 * steps):
        if axis == "r":
            dist = row_parts[row_i]
            row_i += 1
            for _ in range(dist):
                cur_r += sign_r
                path.append((int(cur_r), int(cur_c)))
            axis = "c"
        else:
            dist = col_parts[col_i]
            col_i += 1
            for _ in range(dist):
                cur_c += sign_c
                path.append((int(cur_r), int(cur_c)))
            axis = "r"

    assert (int(cur_r), int(cur_c)) == (int(rb), int(cb))
    return path


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recover interior staircase paths by projecting ring-touching outside traces inward.

    Input structure:
    - A uniform background, one closed 1px rectangular frame, and a uniform fill inside it.
    - Colored components sit outside the frame and touch the ring just beyond the border.

    Rule:
    - Output only the frame interior, initially filled with the frame's fill color.
    - Project each ring-touch point onto the interior boundary.
    - Monochrome U-pairs still become solid straight stripes, but L-type components now become
      multi-bend staircases whose step count is decoded from the outside open-rectangle depth.
    - Two-color components alternate along the decoded staircase using the endpoint ring colors
      to determine the phase.

    Key insight:
    The outside geometry carries two independent clues at once: endpoint locations say where the
    path starts and ends, while the common stub depth of the open rectangle says how many
    staircase steps must be drawn.
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

    def decode_steps(cid: int) -> int:
        ring_pts = comp_ring_actual.get(int(cid), [])
        if not ring_pts:
            return 1
        cr0, cc0, cr1, cc1 = comp_bbox[int(cid)]
        has_top = any(rr == r0 - 1 for rr, _cc, _col in ring_pts)
        has_bot = any(rr == r1 + 1 for rr, _cc, _col in ring_pts)
        has_left = any(cc == c0 - 1 for _rr, cc, _col in ring_pts)
        has_right = any(cc == c1 + 1 for _rr, cc, _col in ring_pts)
        depths: list[int] = []
        if has_top:
            depths.append(int((r0 - 1) - cr0))
        if has_bot:
            depths.append(int(cr1 - (r1 + 1)))
        if has_left:
            depths.append(int((c0 - 1) - cc0))
        if has_right:
            depths.append(int(cc1 - (c1 + 1)))
        depths = [d for d in depths if d > 0]
        if not depths:
            return 1
        return max(1, min(depths) + 1)

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
        steps = decode_steps(int(cid))

        def ordered_pair() -> tuple[tuple[int, int], tuple[int, int]] | None:
            top_bottom = [p for p in coords if p[0] in (0, outH - 1)]
            left_right = [p for p in coords if p[1] in (0, outW - 1)]
            if top_bottom and left_right:
                return tuple(map(int, top_bottom[0])), tuple(map(int, left_right[0]))
            return None

        if len(cols) == 1:
            colv = int(cols[0])
            uniq = coords
            rows = {r for r, _c in uniq}
            cols2 = {c for _r, c in uniq}
            if len(rows) == 1 or len(cols2) == 1:
                for p in uniq:
                    deferred[colv].add((int(p[0]), int(p[1])))
                continue

            ordered = ordered_pair()
            if ordered is None:
                best = None
                for i in range(len(uniq)):
                    for j in range(i + 1, len(uniq)):
                        d = abs(uniq[i][0] - uniq[j][0]) + abs(uniq[i][1] - uniq[j][1])
                        if best is None or d > best[0]:
                            best = (d, uniq[i], uniq[j])
                if best is None:
                    continue
                _d, p1, p2 = best
            else:
                p1, p2 = ordered
            path = _build_staircase_path(p1, p2, outH, outW, steps)
            for rr, cc in path:
                out[int(rr), int(cc)] = int(colv)
            continue

        if len(cols) == 2:
            ordered = ordered_pair()
            if ordered is None:
                best = None
                for i in range(len(coords)):
                    for j in range(i + 1, len(coords)):
                        d = abs(coords[i][0] - coords[j][0]) + abs(coords[i][1] - coords[j][1])
                        if best is None or d > best[0]:
                            best = (d, coords[i], coords[j])
                if best is None:
                    continue
                _d, a, b = best
            else:
                a, b = ordered
            ca, cb = map(int, cols)
            ha = int(hist[a].most_common(1)[0][0])
            hb = int(hist[b].most_common(1)[0][0])

            path = _build_staircase_path(a, b, outH, outW, steps)

            def paint_alt(path_pts: list[tuple[int, int]], start_color: int) -> None:
                cur = int(start_color)
                for rr, cc in path_pts:
                    out[int(rr), int(cc)] = int(cur)
                    cur = int(cb) if cur == int(ca) else int(ca)

            chosen = None
            for start_color in (ca, cb):
                end_color = start_color if ((len(path) - 1) % 2 == 0) else (cb if start_color == ca else ca)
                if int(start_color) != int(ha):
                    continue
                if int(end_color) != int(hb):
                    continue
                chosen = (path, int(start_color))
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
        path = _build_staircase_path(p1, p2, outH, outW, 1)
        for rr, cc in path:
            out[int(rr), int(cc)] = int(col)

    return out


# --------------- generator (a6 D8 variant) ---------------

_BORDER_COLOR = 3
_BG_CHOICES = [0, 1, 2, 4, 5, 6, 7, 8, 9]
_FILL_EQ_BG_PROB = 0.6
_INTERIOR_H_RANGE = (4, 13)
_INTERIOR_W_RANGE = (7, 14)
_GRID_SIZE_RANGE = (20, 30)
_GRID_INTERIOR_PAD = 10
_FRAME_MARGIN = 4
_FRAME_MARGIN_FAR = 6
_U_OUTER_OFFSET_RANGE = (2, 3)
_STAIR_STEPS_RANGE = (2, 3)
_N_COMPONENTS_RANGE = (2, 3)
_U_PAIR_PROB = 0.2
_DUAL_COLOR_PROB = 0.45
_L_COMPONENT_TRIES = 40
_DISTRACTOR_COUNT_RANGE = (0, 2)
_DISTRACTOR_SIZE_RANGE = (1, 4)
_DISTRACTOR_TRIES = 40
_DISTRACTOR_CLEARANCE = 2
_GEN_RETRIES = 400


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


def _choose_local_endpoint(kind: str, H: int, W: int, steps: int) -> tuple[int, int]:
    max_row = max(int(steps), min(int(H - 2), int(H // 3 + 1)))
    max_col = max(int(steps), min(int(W - 2), int(W // 3 + 1)))

    if kind == "tl":
        r = random.randint(int(steps), int(max_row))
        c = random.randint(int(steps), int(max_col))
        return int(r), int(c)
    if kind == "tr":
        r = random.randint(int(steps), int(max_row))
        dist_right = random.randint(int(steps), int(max_col))
        return int(r), int(W - 1 - dist_right)
    if kind == "bl":
        dist_up = random.randint(int(steps), int(max_row))
        c = random.randint(int(steps), int(max_col))
        return int(H - 1 - dist_up), int(c)
    dist_up = random.randint(int(steps), int(max_row))
    dist_right = random.randint(int(steps), int(max_col))
    return int(H - 1 - dist_up), int(W - 1 - dist_right)


def _stair_specs(kind: str, H: int, W: int, r0: int, c0: int, r1: int, c1: int, steps: int):
    is_top = kind[0] == "t"
    is_left = kind[1] == "l"
    row_mid, col_mid = _choose_local_endpoint(kind, H, W, steps)

    top_or_bottom = (0 if is_top else H - 1, col_mid)
    left_or_right = (row_mid, 0 if is_left else W - 1)
    inner = _build_staircase_path(top_or_bottom, left_or_right, H, W, steps)

    ring_a = (r0 - 1 if is_top else r1 + 1, c0 + 1 + col_mid)
    ring_b = (r0 + 1 + row_mid, c0 - 1 if is_left else c1 + 1)
    R = (r0 - steps) if is_top else (r1 + steps)
    C = (c0 - steps) if is_left else (c1 + steps)
    outer = _join(
        _vpath(ring_a[0], R, ring_a[1]),
        _hpath(R, ring_a[1], C),
        _vpath(R, ring_b[0], C),
        _hpath(ring_b[0], C, ring_b[1]),
    )
    return inner, outer


def _u_pair_specs(orientation: str, H: int, W: int, r0: int, c0: int, r1: int, c1: int):
    lo, hi = _U_OUTER_OFFSET_RANGE
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
    palette = [x for x in range(10) if x not in (bg, bc, fill)]
    random.shuffle(palette)
    used_inner = np.zeros((H, W), dtype=bool)

    n_components = random.randint(*_N_COMPONENTS_RANGE)
    remaining = n_components
    color_i = 0

    if remaining >= 2 and random.random() < _U_PAIR_PROB:
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
    for kind in corners:
        max_steps = min(_STAIR_STEPS_RANGE[1], H - 2, W - 2)
        if max_steps < _STAIR_STEPS_RANGE[0]:
            return None
        if color_i >= len(palette) - 1:
            return None
        chosen = None
        for _ in range(_L_COMPONENT_TRIES):
            steps = random.randint(_STAIR_STEPS_RANGE[0], max_steps)
            inner, outer = _stair_specs(kind, H, W, r0, c0, r1, c1, steps)
            irs, ics = zip(*inner)
            if used_inner[irs, ics].any():
                continue
            if len(set(outer)) != len(outer):
                continue
            if not _can_place(grid, outer, bg, comp_mask):
                continue
            chosen = (inner, outer)
            break
        if chosen is None:
            return None
        inner, outer = chosen
        irs, ics = zip(*inner)
        if random.random() < _DUAL_COLOR_PROB:
            colors = (palette[color_i], palette[color_i + 1])
            color_i += 2
        else:
            colors = (palette[color_i],)
            color_i += 1
        if not _place_component(grid, comp_mask, outer, colors, bg):
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
    raise RuntimeError("failed to generate a valid a6f40cea D8 variant input")
