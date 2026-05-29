# Combo variant of a6f40cea (index 21)
# combo_spec: D17/0+D18/0
# Name: 图例驱动颜色映射

from __future__ import annotations

from collections import Counter, defaultdict
import random

import numpy as np
from scipy.ndimage import label

from arc_utils import STRUCT4, get_bg, neighbors4

# ---------------------------------------------------------------------------
# Design analysis for the fused variant
#
# Concrete behavior trace:
# - Original task: an outer open rectangle may touch the frame with colors 2 and 4,
#   so the interior path is painted directly with 2/4.
# - Fused task: the input also contains a compact legend in the top-left corner,
#   e.g. rows [2, 7] and [4, 8]. The same outer geometry still projects to the
#   same interior Manhattan path, but the painted colors become 7/8 instead of 2/4.
#   For a monochrome source 5 with legend [5, 9], the inside stripe becomes 9.
#
# Complexity calibration:
# - The legend is a vertical stack of 2-cell rows in one reserved corner.
# - It contains 2-4 entries: enough repetition for humans to spot the lookup rule,
#   but small enough to avoid clutter.
# - We keep the original geometric task intact; only the color transfer step changes.
#
# Hidden assumptions reviewed against the original solver:
# - Frame detection still works because the legend is tiny, disconnected, and not a
#   1px complete rectangle.
# - Ring-touch grouping still works because legend cells never touch the frame ring.
# - The original deferred monochrome aggregation assumes that identical deferred
#   colors only need to merge for matched U-pairs. The generator preserves that by
#   keeping source colors unique across unrelated components; the legend mapping is
#   bijective on used source colors, so this remains true after recoloring.
# - Two-color phase recovery remains valid because endpoint hints are read on the
#   outer component first, then mapped through the legend before painting inside.
#
# Placement / legality:
# - The frame is pushed down enough to leave a reserved legend strip above it.
# - Distractors are kept away from both the frame ring and the legend box.
# - The legend rows are exactly two cells wide, so they are visually distinct from
#   the external open rectangles.
#
# Observability and non-degeneracy:
# - Every used outer color appears in the legend.
# - The mapping is always a derangement on the used source colors, so at least one
#   interior path cell differs from the original "copy colors directly" behavior on
#   every generated sample.
# ---------------------------------------------------------------------------


def _find_framed_rect_uniform_fill(
    g: np.ndarray,
    *,
    bg: int | None = None,
    structure=STRUCT4,
) -> tuple[int, int, int, int, int, int] | None:
    """Detect the largest 1px rectangular frame with uniform interior fill."""
    if bg is None:
        bg = int(get_bg(g))
    else:
        bg = int(bg)

    best = None  # (area, r0, c0, r1, c1, border_color, fill_color)
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


def _find_legend_mapping(g: np.ndarray, bg: int) -> tuple[dict[int, int], np.ndarray]:
    """Read the top-left legend: contiguous rows of exactly two non-bg cells."""
    h, w = g.shape
    assert w >= 2
    entries: list[tuple[int, int]] = []
    legend_mask = np.zeros_like(g, dtype=bool)
    for r in range(h):
        nz = np.flatnonzero(g[r, :] != int(bg))
        if nz.size == 0:
            if entries:
                break
            continue
        if int(nz[0]) != 0 or nz.size != 2 or int(nz[1]) != 1:
            if entries:
                break
            continue
        src = int(g[r, 0])
        dst = int(g[r, 1])
        assert src != int(bg) and dst != int(bg)
        entries.append((src, dst))
        legend_mask[r, 0] = True
        legend_mask[r, 1] = True

    assert len(entries) >= 2, "legend must contain at least two color pairs"
    mapping: dict[int, int] = {}
    for src, dst in entries:
        assert src not in mapping, "legend source colors must be unique"
        mapping[int(src)] = int(dst)
    return mapping, legend_mask


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
        grid[rs, cs] = int(colors[0])
    else:
        a, b = map(int, colors)
        vals = [a if i % 2 == 0 else b for i in range(len(path))]
        grid[rs, cs] = vals


def _can_place(grid: np.ndarray, coords: list[tuple[int, int]], bg: int, comp_mask: np.ndarray) -> bool:
    H, W = grid.shape
    s = set(coords)
    for r, c in s:
        if not (0 <= r < H and 0 <= c < W):
            return False
        if int(grid[r, c]) != int(bg):
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
    """Build inner L path plus its matching outer open-rectangle path."""
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
    """Build two opposite U-shapes outside and the full-span stripes they imply inside."""
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


def _derangement(vals: list[int]) -> list[int] | None:
    vals = list(map(int, vals))
    if len(vals) < 2:
        return None
    if len(vals) == 2:
        return [vals[1], vals[0]]
    for _ in range(100):
        perm = vals[:]
        random.shuffle(perm)
        if all(int(a) != int(b) for a, b in zip(vals, perm)):
            return perm
    return None


def _draw_legend(grid: np.ndarray, mapping: dict[int, int]) -> tuple[int, int, int, int]:
    rows = list(mapping.items())
    for r, (src, dst) in enumerate(rows):
        grid[r, 0] = int(src)
        grid[r, 1] = int(dst)
    return (0, 0, len(rows) - 1, 1)


def _bbox_touches_or_nears(rr: int, cc: int, box: tuple[int, int, int, int], gap: int) -> bool:
    r0, c0, r1, c1 = box
    return (r0 - gap) <= rr <= (r1 + gap) and (c0 - gap) <= cc <= (c1 + gap)


def _add_distractor(
    grid: np.ndarray,
    comp_mask: np.ndarray,
    bg: int,
    frame: tuple[int, int, int, int],
    color: int,
    legend_box: tuple[int, int, int, int],
) -> None:
    """Try to place a small blob away from the frame and legend."""
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
        if any(_bbox_touches_or_nears(rr, cc, (r0, c0, r1, c1), gap) for rr, cc in coords):
            continue
        if any(_bbox_touches_or_nears(rr, cc, legend_box, 1) for rr, cc in coords):
            continue
        if _place_component(grid, comp_mask, coords, (int(color),), bg):
            return


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reconstruct the framed interior paths, but recolor them through a corner legend.

    Input structure:
    - A bg canvas with one 1px closed rectangle whose border color differs from the
      bg and whose interior has a uniform fill.
    - Outside the frame are the same open-rectangle traces as in the base task.
    - The top-left corner contains a small legend: each 1x2 row is
      (outer_trace_color, interior_path_color).

    Rule:
    - Output only the interior of the closed frame, initialized with the frame fill.
    - Detect each ring-touching outside component and recover the same interior
      Manhattan stripe or L-path as in the original task.
    - Before painting, convert every source color through the legend lookup instead
      of copying the outside color directly.
    - For two-color traces, map both colors independently, then keep the same
      alternating structure and endpoint-anchored phase.

    Key insight: Geometry still determines where each path goes; the only new step is
    that colors are not read literally from the outside traces and must be translated
    via the corner legend first.
    """
    g = np.array(grid, dtype=int)
    H, W = g.shape
    bg = int(get_bg(g))

    found = _find_framed_rect_uniform_fill(g, bg=int(bg))
    assert found is not None
    r0, c0, r1, c1, bc, fill = found
    mapping, legend_mask = _find_legend_mapping(g, bg)

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
        p1: tuple[int, int],
        p2: tuple[int, int],
        cornerA: bool,
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

    comp_mask = (g != int(bg)) & (g != int(fill)) & (g != int(bc)) & (~legend_mask)
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
        src_cols = comp_cols.get(cid, [])
        if not src_cols:
            continue
        for col in src_cols:
            assert int(col) in mapping, f"missing legend entry for color {col}"

        hist: dict[tuple[int, int], Counter] = defaultdict(Counter)
        for r, c, col in pts:
            hist[(int(r), int(c))][int(col)] += 1
        coords = list(hist.keys())

        if len(src_cols) == 1:
            src = int(src_cols[0])
            dst = int(mapping[src])
            uniq = coords
            rows = {r for r, _c in uniq}
            cols2 = {c for _r, c in uniq}
            if len(rows) == 1 or len(cols2) == 1:
                for p in uniq:
                    deferred[dst].add((int(p[0]), int(p[1])))
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
                out[int(rr), int(cc)] = int(dst)
            continue

        if len(src_cols) == 2:
            best = None
            for i in range(len(coords)):
                for j in range(i + 1, len(coords)):
                    d = abs(coords[i][0] - coords[j][0]) + abs(coords[i][1] - coords[j][1])
                    if best is None or d > best[0]:
                        best = (d, coords[i], coords[j])
            if best is None:
                continue
            _d, a, b = best

            src_a, src_b = map(int, src_cols)
            dst_a, dst_b = int(mapping[src_a]), int(mapping[src_b])
            assert dst_a != dst_b
            ha = int(mapping[int(hist[a].most_common(1)[0][0])])
            hb = int(mapping[int(hist[b].most_common(1)[0][0])])

            pathA = build_manhattan(a, b, True)
            pathB = build_manhattan(a, b, False)
            path = pathB if (inside_corner(a, b, False) and not inside_corner(a, b, True)) else pathA

            def paint_alt(path_pts: list[tuple[int, int]], start_color: int):
                cur = int(start_color)
                for rr, cc in path_pts:
                    out[int(rr), int(cc)] = int(cur)
                    cur = int(dst_b) if cur == int(dst_a) else int(dst_a)

            chosen = None
            for path_pts in (path, list(reversed(path))):
                for start_color in (dst_a, dst_b):
                    end_color = start_color if ((len(path_pts) - 1) % 2 == 0) else (dst_b if start_color == dst_a else dst_a)
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


# --------------- generator ---------------

_BORDER_COLOR = 3
_BG_CHOICES = [0, 1, 2, 4, 5, 6, 7, 8, 9]
_FILL_EQ_BG_PROB = 0.6
_INTERIOR_H_RANGE = (4, 13)
_INTERIOR_W_RANGE = (7, 14)
_GRID_SIZE_RANGE = (20, 30)
_GRID_INTERIOR_PAD = 10
_FRAME_LEFT_MARGIN = 6
_FRAME_TOP_MARGIN = 8
_FRAME_MARGIN_FAR = 6
_OUTER_OFFSET_RANGE = (2, 3)
_N_COMPONENTS_RANGE = (2, 3)
_U_PAIR_PROB = 0.4
_DUAL_COLOR_PROB = 0.5
_MAX_DUAL_COMPONENTS = 1
_DISTRACTOR_COUNT_RANGE = (0, 1)
_DISTRACTOR_SIZE_RANGE = (1, 4)
_DISTRACTOR_TRIES = 40
_DISTRACTOR_CLEARANCE = 2
_GEN_RETRIES = 400


def _gen_attempt() -> np.ndarray | None:
    bg = random.choice(_BG_CHOICES)
    bc = _BORDER_COLOR
    fill = bg if random.random() < _FILL_EQ_BG_PROB else random.choice([x for x in range(10) if x not in (bg, bc)])

    H = random.randint(*_INTERIOR_H_RANGE)
    W = random.randint(*_INTERIOR_W_RANGE)
    rows_lo = max(_GRID_SIZE_RANGE[0], H + _FRAME_TOP_MARGIN + _FRAME_MARGIN_FAR)
    cols_lo = max(_GRID_SIZE_RANGE[0], W + _FRAME_LEFT_MARGIN + _FRAME_MARGIN_FAR)
    rows = random.randint(rows_lo, _GRID_SIZE_RANGE[1])
    cols = random.randint(cols_lo, _GRID_SIZE_RANGE[1])

    r0 = random.randint(_FRAME_TOP_MARGIN, rows - H - _FRAME_MARGIN_FAR)
    c0 = random.randint(_FRAME_LEFT_MARGIN, cols - W - _FRAME_MARGIN_FAR)
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

    records: list[tuple[str, object, tuple[int, ...]]] = []
    used_source_order: list[int] = []

    def register_source(col: int) -> None:
        if int(col) not in used_source_order:
            used_source_order.append(int(col))

    force_u_pair = random.random() < _U_PAIR_PROB
    n_components = 3 if force_u_pair else random.randint(*_N_COMPONENTS_RANGE)
    remaining = n_components

    if force_u_pair:
        orientation = random.choice(["vertical", "horizontal"])
        if orientation == "horizontal" and H < 5:
            orientation = "vertical"
        if color_i >= len(palette):
            return None
        color = int(palette[color_i])
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
        used_inner[irs, ics] = True
        records.append(("u", inner_lines, (color,)))
        register_source(color)
        remaining -= 2

    corners = random.sample(["tl", "tr", "bl", "br"], k=remaining)
    dual_budget = _MAX_DUAL_COMPONENTS if random.random() < _DUAL_COLOR_PROB else 0
    dual_indices = set(random.sample(range(remaining), k=min(dual_budget, remaining))) if remaining else set()

    for idx, kind in enumerate(corners):
        inner, outer = _l_specs(kind, H, W, r0, c0, r1, c1)
        irs, ics = zip(*inner)
        if used_inner[irs, ics].any():
            return None
        if idx in dual_indices:
            if color_i >= len(palette) - 1:
                return None
            colors = (int(palette[color_i]), int(palette[color_i + 1]))
            color_i += 2
        else:
            if color_i >= len(palette):
                return None
            colors = (int(palette[color_i]),)
            color_i += 1
        if not _place_component(grid, comp_mask, outer, colors, bg):
            return None
        used_inner[irs, ics] = True
        records.append(("l", inner, colors))
        for col in colors:
            register_source(col)

    if len(used_source_order) < 2:
        return None

    perm = _derangement(used_source_order)
    if perm is None:
        return None
    mapping = {int(src): int(dst) for src, dst in zip(used_source_order, perm)}

    legend_box = _draw_legend(grid, mapping)

    for kind, payload, src_colors in records:
        dst_colors = tuple(int(mapping[int(col)]) for col in src_colors)
        if kind == "u":
            inner_lines = payload
            assert isinstance(inner_lines, list)
            for line in inner_lines:
                _paint_path(expected, line, dst_colors)
        else:
            inner = payload
            assert isinstance(inner, list)
            _paint_path(expected, inner, dst_colors)

    distractor_colors = [c for c in palette[color_i:] if int(c) not in mapping]
    random.shuffle(distractor_colors)
    n_distractors = min(random.randint(*_DISTRACTOR_COUNT_RANGE), len(distractor_colors))
    for i in range(n_distractors):
        _add_distractor(grid, comp_mask, bg, (r0, c0, r1, c1), int(distractor_colors[i]), legend_box)

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
    raise RuntimeError("failed to generate a valid combo-variant a6f40cea input")
