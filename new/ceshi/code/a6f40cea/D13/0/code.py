# Variant of a6f40cea
# Dimension: D13 — U形配对机制 (path)
# Change: 可选的U形对（两个U形开放框在框架对侧，产生贯穿内部的全跨直线条纹）... → U形产生封闭矩形
#
# Design analysis for the D13 variant
# a) Concrete behavior and trace
#    - The closed frame is unchanged: one 1px rectangle with uniform fill.
#    - L-shaped outside components still map to one interior Manhattan L path.
#    - The changed case is a same-color pair of U-shaped open frames placed on opposite
#      sides of the main frame. In the old task, their two touch columns/rows became two
#      independent full-span stripes. In this variant, the four projected touch points are
#      interpreted together as the four corners of one rectangle outline.
#    - Example: a top U touches interior top at columns 2 and 6, and a bottom U of the
#      same color touches interior bottom at those same columns. Old logic produced two
#      solid vertical stripes at c=2 and c=6. New logic produces the full perimeter of
#      the rectangle with corners (0,2), (0,6), (H-1,2), (H-1,6).
# b) Complexity calibration
#    - U-pairs remain optional, but their probability is raised so the new behavior is
#      commonly visible across a few samples.
#    - The rectangle uses one color and is constrained to touch opposite sides, so it is
#      still easy to infer from 2-4 examples and does not introduce many branches.
# c) Hidden assumptions from the original solver
#    - Original deferred monochrome handling assumed that same-color touch points lying on
#      one row/column should later expand into stripes. That assumption is invalid now for
#      opposite-side U-pairs.
#    - The new solver still defers row/column-aligned monochrome components, but when the
#      combined deferred set contains opposite-side pairs with two distinct columns/rows,
#      it paints a rectangle outline instead of two stripes.
#    - This relies on generator uniqueness: a U-pair color is not reused by L-components,
#      avoiding ambiguous merges inside the deferred bucket.
# d) Placement legality
#    - Outer components stay outside the frame, non-overlapping, and non-4-adjacent.
#    - U-pairs are generated only when the interior is large enough to form a visible
#      rectangle with at least one-cell thickness-free gap between its opposite sides.
#    - The generator still validates every sample with main(input) == expected output.
# e) Visual observability
#    - The difference from the base task is directly visible: two opposite U shapes now
#      imply a closed border, not two disjoint stripes.
#    - The new rectangle touches opposite sides, matching the projected endpoints, so a
#      human can align the outside U pair with the inside perimeter.
# f) Visual cleanliness
#    - The generator keeps colors unique per semantic object and rejects overlaps in the
#      output interior, so the rectangle border is not buried under other paths.
# g) Color inference
#    - No color remapping rule is introduced. Colors still mean "copy this component's
#      trace inward", with the only changed question being the geometry of U-pairs.

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


def _rect_outline(r0: int, c0: int, r1: int, c1: int) -> list[tuple[int, int]]:
    return _join(
        _hpath(r0, c0, c1),
        _vpath(r0, r1, c1),
        _hpath(r1, c1, c0),
        _vpath(r1, r0, c0),
    )


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reconstruct interior paths from colored open frames around a closed rectangle.

    Input structure:
    - A uniform background, one 1px closed rectangular frame, and a solid fill inside it.
    - Colored components remain outside the frame and touch the ring just beyond its border.

    Rule:
    - Output the interior crop of the frame, initialized with the fill color.
    - Single L-like outside traces still project to solid Manhattan paths, and two-color traces
      still project to alternating Manhattan paths.
    - A same-color U-shape pair on opposite frame sides is handled jointly: its four projected
      touch points define one closed rectangular outline instead of two separate stripes.

    Key insight:
    - Monochrome ring-touch points that were formerly expanded independently must now be grouped
      by color across opposite sides; the shared bounding columns or rows reveal a rectangle pair.
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

    def build_manhattan(p1: tuple[int, int], p2: tuple[int, int], cornerA: bool) -> list[tuple[int, int]]:
        (ra, ca), (rb, cb) = p1, p2
        if ra == rb:
            return [(int(ra), int(c)) for c in _line(int(ca), int(cb))]
        if ca == cb:
            return [(int(r), int(ca)) for r in _line(int(ra), int(rb))]
        if cornerA:
            return _join(_vpath(ra, rb, ca), _hpath(rb, ca, cb))
        return _join(_hpath(ra, ca, cb), _vpath(ra, rb, cb))

    def inside_corner(p1: tuple[int, int], p2: tuple[int, int], cornerA: bool) -> bool:
        (ra, ca), (rb, cb) = p1, p2
        if ra == rb or ca == cb:
            return True
        rr, cc = (rb, ca) if cornerA else (ra, cb)
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
        mr, mc = map_to_out(rr, cc)
        comp_pts[cid].append((mr, mc, int(g[rr, cc])))

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
            ca, cb2 = map(int, cols)
            ha = int(hist[a].most_common(1)[0][0])
            hb = int(hist[b].most_common(1)[0][0])
            pathA = build_manhattan(a, b, True)
            pathB = build_manhattan(a, b, False)
            path = pathB if (inside_corner(a, b, False) and not inside_corner(a, b, True)) else pathA

            def paint_alt(path_pts: list[tuple[int, int]], start_color: int) -> None:
                cur = int(start_color)
                for rr, cc in path_pts:
                    out[int(rr), int(cc)] = int(cur)
                    cur = int(cb2) if cur == int(ca) else int(ca)

            chosen = None
            for path_pts in (path, list(reversed(path))):
                for start_color in (ca, cb2):
                    if int(start_color) != int(ha):
                        continue
                    end_color = start_color if ((len(path_pts) - 1) % 2 == 0) else (cb2 if start_color == ca else ca)
                    if int(end_color) != int(hb):
                        continue
                    chosen = (path_pts, int(start_color))
                    break
                if chosen is not None:
                    break

            assert chosen is not None
            path_pts, start_color = chosen
            paint_alt(path_pts, start_color)

    for col, pts0 in deferred.items():
        pts = list(pts0)
        if not pts:
            continue

        tops = [p for p in pts if p[0] == 0]
        bots = [p for p in pts if p[0] == outH - 1]
        if tops and bots:
            cols_tb = sorted({c for _r, c in tops + bots})
            if len(cols_tb) >= 2:
                for rr, cc in _rect_outline(0, cols_tb[0], outH - 1, cols_tb[-1]):
                    out[int(rr), int(cc)] = int(col)
                continue
            for rr in range(0, outH):
                out[int(rr), int(cols_tb[0])] = int(col)
            continue

        lefts = [p for p in pts if p[1] == 0]
        rights = [p for p in pts if p[1] == outW - 1]
        if lefts and rights:
            rows_lr = sorted({r for r, _c in lefts + rights})
            if len(rows_lr) >= 2:
                for rr, cc in _rect_outline(rows_lr[0], 0, rows_lr[-1], outW - 1):
                    out[int(rr), int(cc)] = int(col)
                continue
            for cc in range(0, outW):
                out[int(rows_lr[0]), int(cc)] = int(col)
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


# --------------- generator (a6 D13) ---------------

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
_U_PAIR_PROB = 0.5
_DUAL_COLOR_PROB = 0.45
_DISTRACTOR_COUNT_RANGE = (0, 2)
_DISTRACTOR_SIZE_RANGE = (1, 4)
_DISTRACTOR_TRIES = 40
_DISTRACTOR_CLEARANCE = 2
_GEN_RETRIES = 300


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
    return inner, outer


def _u_pair_specs(orientation: str, H: int, W: int, r0: int, c0: int, r1: int, c1: int):
    lo, hi = _OUTER_OFFSET_RANGE
    if orientation == "vertical":
        cols = list(range(1, W - 1))
        valid_pairs = [(a, b) for a in cols for b in cols if a < b and b - a >= 2]
        assert valid_pairs
        c_a, c_b = random.choice(valid_pairs)
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
        inner = _rect_outline(0, c_a, H - 1, c_b)
    else:
        rows = list(range(1, H - 1))
        valid_pairs = [(a, b) for a in rows for b in rows if a < b and b - a >= 2]
        assert valid_pairs
        r_a, r_b = random.choice(valid_pairs)
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
        inner = _rect_outline(r_a, 0, r_b, W - 1)
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
        inner_rect, outer_a, outer_b = _u_pair_specs(orientation, H, W, r0, c0, r1, c1)
        irs, ics = zip(*inner_rect)
        if used_inner[irs, ics].any():
            return None
        if not _place_component(grid, comp_mask, outer_a, (color,), bg):
            return None
        if not _place_component(grid, comp_mask, outer_b, (color,), bg):
            return None
        _paint_path(expected, inner_rect, (color,))
        used_inner[irs, ics] = True
        remaining -= 2

    corners = random.sample(["tl", "tr", "bl", "br"], k=remaining)
    for kind in corners:
        if color_i >= len(palette) - 1:
            return None
        inner, outer = _l_specs(kind, H, W, r0, c0, r1, c1)
        irs, ics = zip(*inner)
        if used_inner[irs, ics].any():
            return None
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
    raise RuntimeError("failed to generate a valid a6f40cea D13 input")
