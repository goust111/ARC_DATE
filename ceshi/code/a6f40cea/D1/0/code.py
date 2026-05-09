# Variant of a6f40cea
# Dimension: D1 — 路径形态 (path)
# Change: 两端点间的曼哈顿L形路径（一个拐角，由水平段+垂直段组成），拐角选择使其严格位于内部而非边缘... → 对角线路径连接
#
# Design analysis
# - Concrete behavior:
#   The framed rectangle and the external open-rectangle components stay the same.
#   What changes is the reconstruction inside the frame for adjacent-side components.
#   Example: if the ring touch points map to interior endpoints (0, 4) on the top edge
#   and (4, 0) on the left edge, the original task drew an L path such as
#   (0,4)->(4,4)->(4,0). This variant instead draws the direct 45-degree line
#   (0,4),(1,3),(2,2),(3,1),(4,0). The outer component is still orthogonal; only the
#   inferred interior connection rule changes.
# - Complexity calibration:
#   The new rule uses one visible, repeatable mode: endpoints on adjacent sides are placed
#   at equal offset from their shared corner, so a single 8-connected diagonal is forced.
#   U-pairs that create full-span stripes are preserved, but only when at least one
#   diagonal component also remains in the same puzzle; otherwise the D1 change would be
#   invisible in some samples.
# - Implicit assumptions checked against the original solver:
#   `build_manhattan(...)` and `inside_corner(...)` depended on a valid elbow choice.
#   That assumption is invalid here and is replaced by `_build_direct_path(...)`, which
#   assumes the chosen endpoints are 8-aligned (horizontal, vertical, or 45-degree).
#   The generator now guarantees that every diagonal component satisfies `abs(dr)==abs(dc)`.
#   Alternating-color logic still relies on endpoint ring colors anchoring the phase, so the
#   inner diagonal path is ordered consistently with the outer path from ring_a to ring_b.
#   Deferred same-row / same-column monochrome handling is left intact for U-pairs.
# - Placement legality:
#   The outer shapes remain 4-connected orthogonal traces, so component extraction via
#   4-connectivity still works. Inner diagonal cells are tracked with `used_inner` to forbid
#   overlap. Distinct corners are still sampled without replacement. U-pairs are only placed
#   when at least one diagonal L-type component remains.
# - Visual observability:
#   The diagonal depth is at least 2, so the output visibly contains a slanted 8-connected
#   segment rather than a one-step corner. Since each puzzle now always contains at least one
#   such component, the new rule is observable from a small set of examples.
# - Visual cleanliness:
#   No extra markings are added. The frame, outer components, and distractors keep their
#   original separation rules. The output remains uncluttered because only one reconstruction
#   mode changed.
# - Color reasoning:
#   No color-id-based ordering or hidden mapping is introduced. Colors still only encode
#   component identity and optional alternation, with endpoint colors providing the same
#   visible phase cue as in the original task.

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


def _aligned_8(p1: tuple[int, int], p2: tuple[int, int]) -> bool:
    dr = int(p2[0] - p1[0])
    dc = int(p2[1] - p1[1])
    return (dr == 0 and dc != 0) or (dc == 0 and dr != 0) or (dr != 0 and abs(dr) == abs(dc))


def _build_direct_path(p1: tuple[int, int], p2: tuple[int, int]) -> list[tuple[int, int]]:
    (r0, c0), (r1, c1) = p1, p2
    dr = int(r1 - r0)
    dc = int(c1 - c0)
    if dr == 0 and dc == 0:
        return [(int(r0), int(c0))]
    assert _aligned_8(p1, p2)
    sr = 0 if dr == 0 else (1 if dr > 0 else -1)
    sc = 0 if dc == 0 else (1 if dc > 0 else -1)
    steps = max(abs(dr), abs(dc))
    return [(int(r0 + sr * k), int(c0 + sc * k)) for k in range(steps + 1)]


def _select_farthest_pair(coords: list[tuple[int, int]]) -> tuple[tuple[int, int], tuple[int, int]] | None:
    best = None
    for i in range(len(coords)):
        for j in range(i + 1, len(coords)):
            if not _aligned_8(coords[i], coords[j]):
                continue
            dr = abs(int(coords[i][0]) - int(coords[j][0]))
            dc = abs(int(coords[i][1]) - int(coords[j][1]))
            score = max(dr, dc)
            if best is None or score > best[0]:
                best = (score, coords[i], coords[j])
    if best is None:
        return None
    _score, a, b = best
    return a, b


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recover the frame interior by projecting each outer trace onto an internal line pattern.

    Input structure:
    - A uniform background, one closed 1px rectangular frame, and a uniformly filled interior.
    - Colored 4-connected components outside the frame that touch the ring around that frame.

    Rule:
    - Output only the frame interior, initialized with the interior fill color.
    - Map each ring-touch point into interior coordinates.
    - Monochrome components that hit opposite sides still produce solid full-span stripes.
    - Adjacent-side components now connect their two mapped endpoints by one direct 45-degree
      diagonal (or a straight segment if the endpoints are already row/column aligned).
    - Two-color components paint that direct segment with alternating colors.

    Key insight:
    - The external shape still tells you which two frame-touch points belong together, but the
      correct reconstruction is the unique 8-connected straight line between those projected
      endpoints; for two-color traces, the endpoint ring colors still lock the alternation phase.
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
            rows = {r for r, _c in coords}
            cols2 = {c for _r, c in coords}
            if len(rows) == 1 or len(cols2) == 1:
                for p in coords:
                    deferred[colv].add((int(p[0]), int(p[1])))
                continue

            pair = _select_farthest_pair(coords)
            if pair is None:
                continue
            p1, p2 = pair
            for rr, cc in _build_direct_path(p1, p2):
                out[int(rr), int(cc)] = int(colv)
            continue

        if len(cols) == 2:
            pair = _select_farthest_pair(coords)
            if pair is None:
                continue
            a, b = pair
            path = _build_direct_path(a, b)

            ca, cb = map(int, cols)
            ha = int(hist[a].most_common(1)[0][0])
            hb = int(hist[b].most_common(1)[0][0])

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
        pair = _select_farthest_pair(pts)
        if pair is None:
            continue
        p1, p2 = pair
        for rr, cc in _build_direct_path(p1, p2):
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
_DISTRACTOR_COUNT_RANGE = (0, 2)
_DISTRACTOR_SIZE_RANGE = (1, 4)
_DISTRACTOR_TRIES = 40
_DISTRACTOR_CLEARANCE = 2
_COMPONENT_SHAPE_TRIES = 16
_GEN_RETRIES = 900


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


def _dpath(a: tuple[int, int], b: tuple[int, int]) -> list[tuple[int, int]]:
    return _build_direct_path(a, b)


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


def _diag_specs(kind: str, H: int, W: int, r0: int, c0: int, r1: int, c1: int):
    """Build one diagonal-producing adjacent-side component and its outer open rectangle."""
    max_depth = min(H - 2, W - 2)
    depth = random.randint(2, max_depth)

    if kind == "tl":
        inner_a = (0, depth)
        inner_b = (depth, 0)
        ring_a = (r0 - 1, c0 + 1 + depth)
        ring_b = (r0 + 1 + depth, c0 - 1)
        R = r0 - random.randint(*_OUTER_OFFSET_RANGE)
        C = c0 - random.randint(*_OUTER_OFFSET_RANGE)
    elif kind == "tr":
        inner_a = (0, W - 1 - depth)
        inner_b = (depth, W - 1)
        ring_a = (r0 - 1, c1 - depth)
        ring_b = (r0 + 1 + depth, c1 + 1)
        R = r0 - random.randint(*_OUTER_OFFSET_RANGE)
        C = c1 + random.randint(*_OUTER_OFFSET_RANGE)
    elif kind == "bl":
        inner_a = (H - 1, depth)
        inner_b = (H - 1 - depth, 0)
        ring_a = (r1 + 1, c0 + 1 + depth)
        ring_b = (r1 - 1 - depth, c0 - 1)
        R = r1 + random.randint(*_OUTER_OFFSET_RANGE)
        C = c0 - random.randint(*_OUTER_OFFSET_RANGE)
    else:
        assert kind == "br"
        inner_a = (H - 1, W - 1 - depth)
        inner_b = (H - 1 - depth, W - 1)
        ring_a = (r1 + 1, c1 - depth)
        ring_b = (r1 - 1 - depth, c1 + 1)
        R = r1 + random.randint(*_OUTER_OFFSET_RANGE)
        C = c1 + random.randint(*_OUTER_OFFSET_RANGE)

    inner = _dpath(inner_a, inner_b)
    outer = _join(
        _vpath(ring_a[0], R, ring_a[1]),
        _hpath(R, ring_a[1], C),
        _vpath(R, ring_b[0], C),
        _hpath(ring_b[0], C, ring_b[1]),
    )
    return inner, outer


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

    if remaining >= 3 and random.random() < _U_PAIR_PROB:
        orientation = random.choice(["vertical", "horizontal"])
        if orientation == "horizontal" and H < 5:
            orientation = "vertical"
        color = palette[color_i]
        color_i += 1
        placed_u = False
        for _ in range(_COMPONENT_SHAPE_TRIES):
            inner_lines, outer_a, outer_b = _u_pair_specs(orientation, H, W, r0, c0, r1, c1)
            all_inner = [p for line in inner_lines for p in line]
            irs, ics = zip(*all_inner)
            if used_inner[irs, ics].any():
                continue
            if not _place_component(grid, comp_mask, outer_a, (color,), bg):
                continue
            if not _place_component(grid, comp_mask, outer_b, (color,), bg):
                rs, cs = zip(*outer_a)
                grid[rs, cs] = bg
                comp_mask[rs, cs] = False
                continue
            expected[irs, ics] = color
            used_inner[irs, ics] = True
            remaining -= 2
            placed_u = True
            break
        if not placed_u:
            return None

    corners = random.sample(["tl", "tr", "bl", "br"], k=remaining)
    diagonal_count = 0
    for kind in corners:
        if color_i >= len(palette) - 1:
            return None
        if random.random() < _DUAL_COLOR_PROB:
            colors = (palette[color_i], palette[color_i + 1])
            color_i += 2
        else:
            colors = (palette[color_i],)
            color_i += 1
        placed = False
        for _ in range(_COMPONENT_SHAPE_TRIES):
            inner, outer = _diag_specs(kind, H, W, r0, c0, r1, c1)
            irs, ics = zip(*inner)
            if used_inner[irs, ics].any():
                continue
            if not _place_component(grid, comp_mask, outer, colors, bg):
                continue
            _paint_path(expected, inner, colors)
            used_inner[irs, ics] = True
            diagonal_count += 1
            placed = True
            break
        if not placed:
            return None

    assert diagonal_count >= 1

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
