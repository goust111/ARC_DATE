# Variant of a6f40cea
# Dimension: D10 — 双色交替相位锚定 (trajectory)
# Change: 交替相位由触点处的颜色决定——路径起点颜色必须匹配该端触点的颜色，且终点颜色匹配另一端触点颜色... → 相位由拐角位置决定

from __future__ import annotations

# Design analysis for the D10 variant
# - Concrete behavior:
#   The closed frame and the external open-rectangle traces stay the same. The only
#   changed object class is a two-color L component. Its output is still an
#   alternating Manhattan L path, but the alternation phase is no longer read from
#   the two ring-touch endpoint colors. Instead:
#   1. The external open rectangle's remote corner defines palette order. That
#      corner cell is visibly special in the input, so its color is taken as
#      "color A"; the other component color is "color B".
#   2. The internal elbow row decides which color starts the interior path:
#      elbow in the upper half => start with A; elbow in the lower half => start
#      with B. The path direction is geometric and fixed: from the top/bottom-side
#      endpoint to the left/right-side endpoint.
# - Small execution trace:
#   Original solver: read endpoint colors ha/hb from ring cells, then search over
#   path direction and start color until both ends match.
#   Variant solver: order the endpoints geometrically, recover the inner elbow,
#   read the external-corner color to get A/B, then choose the start color from
#   the elbow half. Endpoint colors are allowed to disagree with the start/end
#   colors because they are no longer the anchor.
# - Complexity calibration:
#   Using only upper-vs-lower half keeps the rule low-branching and human-visible.
#   To avoid ambiguous "which of the two colors is A?", the generator paints every
#   two-color outer trace with phase anchored at the remote outside corner. That
#   creates a reusable visible marker in each example.
# - Implicit assumption audit:
#   The original two-color branch assumed endpoint colors are trustworthy and that
#   either path direction could be chosen. That becomes wrong here, because both
#   directions might satisfy geometry while only one matches the new canonical
#   traversal. The variant therefore fixes a geometric order and never uses ring
#   colors for phase.
# - Placement legality:
#   Two-color components are only generated as L shapes, never as U pairs. This
#   guarantees exactly one top/bottom-side touch plus one left/right-side touch,
#   which makes the canonical direction and the remote outside corner recoverable.
#   Paths still stay non-overlapping, in-bounds, and 4-separated from distractors.
# - Visual observability:
#   Each generated puzzle contains at least one upper-half and one lower-half
#   two-color L component. That gives direct support for the new rule inside a
#   single example set: the same style of outer marker produces different interior
#   starts depending on the elbow height.
# - Visual cleanliness and color inference:
#   The number of two-color components stays small. The external-corner marker is
#   part of the existing trace, so no extra glyphs are added. Color roles are
#   inferred from visible structure, never from numeric color ids.

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


def _inside_corner(
    p1: tuple[int, int], p2: tuple[int, int], cornerA: bool, outH: int, outW: int
) -> bool:
    (ra, ca), (rb, cb) = p1, p2
    if ra == rb or ca == cb:
        return True
    corner = (rb, ca) if cornerA else (ra, cb)
    rr, cc = corner
    return (0 < rr < outH - 1) and (0 < cc < outW - 1)


def _paint_alternating_linear(
    grid: np.ndarray,
    path: list[tuple[int, int]],
    start_color: int,
    other_color: int,
) -> None:
    cur = int(start_color)
    nxt = int(other_color)
    for rr, cc in path:
        grid[int(rr), int(cc)] = int(cur)
        cur, nxt = nxt, cur


def _paint_path(
    grid: np.ndarray,
    path: list[tuple[int, int]],
    colors: tuple[int, ...],
    *,
    anchor: tuple[int, int] | None = None,
) -> None:
    if not path:
        return
    rs, cs = zip(*path)
    if len(colors) == 1:
        grid[rs, cs] = colors[0]
        return
    a, b = map(int, colors)
    if anchor is None:
        vals = [a if i % 2 == 0 else b for i in range(len(path))]
    else:
        idx = path.index(anchor)
        vals = [a if abs(i - idx) % 2 == 0 else b for i in range(len(path))]
    grid[rs, cs] = vals


def _choose_half_rows(H: int, upper: bool) -> list[int]:
    vals = [r for r in range(1, H - 1) if (2 * r < H) == upper]
    if vals:
        return vals
    return [1] if upper else [H - 2]


def _start_color_from_elbow(colors: tuple[int, int], elbow_r: int, H: int) -> int:
    return int(colors[0]) if 2 * int(elbow_r) < int(H) else int(colors[1])


def _other_color(colors: tuple[int, int], color: int) -> int:
    a, b = map(int, colors)
    return b if int(color) == a else a


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reconstruct the frame interior by projecting the external traces through the frame.

    Input structure:
    - A uniform-background grid contains one closed 1px frame with uniform fill.
    - Separate colored components live outside the frame and touch the ring around it.

    Rule:
    - Output only the interior of the closed frame, initially filled with the interior color.
    - Monochrome components still create solid stripes or L-shaped Manhattan links between their mapped touch points.
    - Two-color L components still create alternating Manhattan paths, but their phase is not read from the touch colors.
    - For a two-color L, the outside remote corner defines color A, and the inner elbow row decides whether the path starts with A (upper half) or with the other color (lower half).

    Key insight:
    The crucial cue is the relation between two visible corners: the outside corner names the palette order, while the inside elbow position chooses the alternating phase.
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
            pathA = _build_manhattan(p1, p2, True)
            pathB = _build_manhattan(p1, p2, False)
            chosen = pathB if (_inside_corner(p1, p2, False, outH, outW) and not _inside_corner(p1, p2, True, outH, outW)) else pathA
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
            _d, p1, p2 = best

            ordered = None
            for a, b in ((p1, p2), (p2, p1)):
                if a[0] in (0, outH - 1) and b[1] in (0, outW - 1):
                    ordered = (a, b)
                    break
            assert ordered is not None
            start_pt, end_pt = ordered
            assert _inside_corner(start_pt, end_pt, True, outH, outW)
            path = _build_manhattan(start_pt, end_pt, True)
            elbow_r = int(end_pt[0])

            comp_cells = np.argwhere(clab == int(cid))
            min_r, min_c = map(int, comp_cells.min(axis=0))
            max_r, max_c = map(int, comp_cells.max(axis=0))
            is_top = int(start_pt[0]) == 0
            is_left = int(end_pt[1]) == 0
            ext_r = min_r if is_top else max_r
            ext_c = min_c if is_left else max_c
            color_a = int(g[ext_r, ext_c])
            assert color_a in cols
            color_b = int(cols[0]) if int(cols[1]) == color_a else int(cols[1])
            start_color = color_a if 2 * elbow_r < outH else color_b
            other = color_b if start_color == color_a else color_a
            _paint_alternating_linear(out, path, start_color, other)
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
        pathA = _build_manhattan(p1, p2, True)
        pathB = _build_manhattan(p1, p2, False)
        chosen = pathB if (_inside_corner(p1, p2, False, outH, outW) and not _inside_corner(p1, p2, True, outH, outW)) else pathA
        for rr, cc in chosen:
            out[int(rr), int(cc)] = int(col)

    return out


# --- CFG: tuned to keep the D10 phase cue visible ---
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
_DUAL_COLOR_PROB = 0.35
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
    *,
    anchor: tuple[int, int] | None = None,
) -> bool:
    if len(set(path)) != len(path):
        return False
    if not _can_place(grid, path, bg, comp_mask):
        return False
    _paint_path(grid, path, colors, anchor=anchor)
    rs, cs = zip(*path)
    comp_mask[rs, cs] = True
    return True


def _l_specs(kind: str, H: int, W: int, r0: int, c0: int, r1: int, c1: int):
    is_top = kind[0] == "t"
    is_left = kind[1] == "l"

    r = random.choice(_choose_half_rows(H, is_top))
    if is_left:
        c = random.randint(1, max(1, W // 2))
    else:
        c = random.randint(max(1, W // 2), W - 2)

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
    return inner, outer, (r, c), (R, C)


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
    top_dual = random.choice(["tl", "tr"])
    bottom_dual = random.choice(["bl", "br"])
    forced_duals = [top_dual, bottom_dual]
    remaining_corners = [k for k in ["tl", "tr", "bl", "br"] if k not in forced_duals]
    corners = forced_duals + random.sample(remaining_corners, k=n_components - 2)
    random.shuffle(corners)

    color_i = 0
    for kind in corners:
        inner, outer, elbow, outer_corner = _l_specs(kind, H, W, r0, c0, r1, c1)
        irs, ics = zip(*inner)
        if used_inner[irs, ics].any():
            return None

        need_dual = kind in forced_duals
        use_dual = need_dual or (random.random() < _DUAL_COLOR_PROB)
        if use_dual:
            if color_i >= len(palette) - 1:
                return None
            colors = (int(palette[color_i]), int(palette[color_i + 1]))
            color_i += 2
            start_color = _start_color_from_elbow(colors, int(elbow[0]), H)
            other = _other_color(colors, start_color)
            if not _place_component(grid, comp_mask, outer, colors, bg, anchor=outer_corner):
                return None
            _paint_alternating_linear(expected, inner, start_color, other)
        else:
            if color_i >= len(palette):
                return None
            colors = (int(palette[color_i]),)
            color_i += 1
            if not _place_component(grid, comp_mask, outer, colors, bg):
                return None
            _paint_path(expected, inner, colors)

        used_inner[irs, ics] = True

    for _ in range(random.randint(*_DISTRACTOR_COUNT_RANGE)):
        if color_i >= len(palette):
            break
        _add_distractor(grid, comp_mask, bg, (r0, c0, r1, c1), int(palette[color_i]))
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
    raise RuntimeError("failed to generate a valid a6f40cea D10 input")
