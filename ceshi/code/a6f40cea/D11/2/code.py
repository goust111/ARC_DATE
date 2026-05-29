# Variant of a6f40cea
# Dimension: D11 — 端点配对策略 (path)
# Change: 同一连通分量的所有触点中选择曼哈顿距离最远的一对作为路径端点... → 按颜色配对

from __future__ import annotations

from collections import Counter, defaultdict
import random

import numpy as np
from scipy.ndimage import label

from arc_utils import STRUCT4, get_bg, neighbors4


"""
Design analysis for the D11 variant

a) Concrete behavior and trace
   - Original behavior: one outside connected component contributes several ring-touch cells, but the
     solver collapses them to mapped interior coordinates and then picks one farthest pair. That makes
     the component behave like "one path source" even if it visually touches the frame more than twice.
   - Variant behavior: a multi-touch component now uses color as the pairing signal. When a component
     exposes four ring touches with colors A, A, B, B, the output must contain two separate interior
     paths: one connecting the two A touches and one connecting the two B touches.
   - Small trace for a top-left component:
     mapped ring touches = {(0, 2, A), (2, 0, A), (0, 5, B), (4, 0, B)}.
     Original solver would choose only the farthest pair, typically B-top to B-left, and draw a single
     path. The variant groups by color first, so it draws A:(0,2)->(2,0) and B:(0,5)->(4,0).

b) Complexity calibration
   - The new rule is only visible if a component has more than two touches. Therefore the generator
     always includes at least one four-touch paired component.
   - To keep the puzzle legible, each paired component uses exactly two colors and exactly two ring
     touches per color. That avoids ambiguous many-to-many matching.
   - Additional components stay monochrome and two-touch so the task still looks like the original
     family instead of turning into a dense routing puzzle.

c) Hidden-assumption review of the original main()
   - The original monochrome branch assumes "all relevant coords for this component reduce to one pair"
     and explicitly searches the farthest pair. That fails once one component encodes two paths.
   - The original two-color branch assumes the two colors are an alternation palette for one path, not
     pair labels for multiple paths. Reusing it would incorrectly paint one alternating path instead of
     multiple monochrome paths.
   - The deferred aggregation for same-row / same-column touches remains valid, because a color group can
     still represent a straight stripe; it just needs to operate per color group rather than per whole
     component.

d) Placement legality
   - The paired component is generated as the union of two disjoint open-rectangle L branches plus a
     short bridge outside the frame so the whole structure is one connected component.
   - Inner paths inside a single corner use monotone ordered row/column choices, which prevents the two
     L paths from crossing each other.
   - Outer branches are validated against bounds, background occupancy, and 4-neighbor clearance from
     other components before painting.

e) Visual observability
   - The key cue is visible directly in the input: one connected outside object touches the frame four
     times, and the touch colors come in repeated pairs. In the output, those repeated colors become two
     separate paths. This is inferable from a few examples without relying on hidden ordering.

f) Visual cleanliness
   - Each paired component is confined to one frame corner, and at most two such components are used.
   - Simple components occupy distinct remaining corners, which keeps both input and output uncluttered.
   - Distractors remain small and away from the frame, so they do not interfere with the pairing cue.

g) Color-rule inferability
   - Colors now carry only one message: equal colors on ring touches should connect to each other.
   - The generator avoids extra decorative recolorings and does not use color-id sorting as a visible
     rule. Colors are sampled from a shuffled palette and only repeated to signal pairing.

h) Generator-to-solver contract
   - For each paired component, every used color appears on exactly two mapped touch coordinates.
   - The two matched coordinates for a color are either an L-pair in one corner or a straight stripe
     pair. No ambiguous odd-cardinality color groups are generated.
   - The solver still contains a conservative fallback for malformed inputs, but the generator does not
     rely on it.

i) New implicit assumptions introduced by the variant
   - The solver now assumes the dominant ring color at a mapped coordinate is the pairing label for that
     coordinate. The generator guarantees this by keeping each ring touch cell monochrome and isolated.
   - The solver paints each color group independently, so overlapping output paths would overwrite each
     other. The generator prevents this with `used_inner` bookkeeping and corner partitioning.
"""


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
    Recover interior paths from colored outside traces around one framed rectangle.

    Input structure:
    - A background grid contains one 1px rectangular frame with a uniform fill.
    - Outside the frame sit colored 4-connected traces touching the ring just beyond the border.
    - Some traces are multi-touch components where the same color appears on more than one touch point.

    Rule:
    - Output only the rectangle's interior, initially filled with the frame's fill color.
    - Project each ring-touch cell into the interior by clamping through the frame border.
    - Within one outside connected component, group mapped touch coordinates by their touch color.
    - For each color group, connect the same-colored coordinates by a solid Manhattan path or stripe.

    Key insight: color repetition is now the endpoint-matching signal, so one outside component can encode
    several separate interior paths when its ring touches split into repeated color pairs.
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

    comp_mask = (g != int(bg)) & (g != int(fill)) & (g != int(bc))
    clab, _cn = label(comp_mask, structure=STRUCT4)

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

    deferred: dict[int, set[tuple[int, int]]] = defaultdict(set)

    for cid, pts in comp_pts.items():
        if len(pts) < 2:
            continue

        hist: dict[tuple[int, int], Counter] = defaultdict(Counter)
        for r, c, col in pts:
            hist[(int(r), int(c))][int(col)] += 1

        per_color: dict[int, list[tuple[int, int]]] = defaultdict(list)
        for coord, counter in hist.items():
            dom = int(counter.most_common(1)[0][0])
            per_color[dom].append((int(coord[0]), int(coord[1])))

        for colv, coords in per_color.items():
            uniq = sorted(set(coords))
            if len(uniq) < 2:
                continue

            rows = {r for r, _c in uniq}
            cols2 = {c for _r, c in uniq}
            if len(rows) == 1 or len(cols2) == 1:
                for p in uniq:
                    deferred[int(colv)].add((int(p[0]), int(p[1])))
                continue

            if len(uniq) == 2:
                p1, p2 = uniq
            else:
                best = None
                for i in range(len(uniq)):
                    for j in range(i + 1, len(uniq)):
                        d = abs(uniq[i][0] - uniq[j][0]) + abs(uniq[i][1] - uniq[j][1])
                        if best is None or d > best[0]:
                            best = (d, uniq[i], uniq[j])
                assert best is not None
                _d, p1, p2 = best

            pathA = build_manhattan(p1, p2, True)
            pathB = build_manhattan(p1, p2, False)
            chosen = pathB if (inside_corner(p1, p2, False) and not inside_corner(p1, p2, True)) else pathA
            for rr, cc in chosen:
                out[int(rr), int(cc)] = int(colv)

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
        assert best is not None
        _d, p1, p2 = best
        pathA = build_manhattan(p1, p2, True)
        pathB = build_manhattan(p1, p2, False)
        chosen = pathB if (inside_corner(p1, p2, False) and not inside_corner(p1, p2, True)) else pathA
        for rr, cc in chosen:
            out[int(rr), int(cc)] = int(col)

    return out


# --------------- generator (a6 D11/2) ---------------

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
_DISTRACTOR_COUNT_RANGE = (0, 2)
_DISTRACTOR_SIZE_RANGE = (1, 4)
_DISTRACTOR_TRIES = 40
_DISTRACTOR_CLEARANCE = 2
_GEN_RETRIES = 400


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


def _place_colored_component(
    grid: np.ndarray,
    comp_mask: np.ndarray,
    colored_cells: dict[tuple[int, int], int],
    bg: int,
) -> bool:
    coords = list(colored_cells.keys())
    if not coords:
        return False
    if not _can_place(grid, coords, bg, comp_mask):
        return False
    for (r, c), col in colored_cells.items():
        grid[int(r), int(c)] = int(col)
    rs, cs = zip(*coords)
    comp_mask[rs, cs] = True
    return True


def _merge_colored_parts(parts: list[tuple[list[tuple[int, int]], int]]) -> dict[tuple[int, int], int] | None:
    merged: dict[tuple[int, int], int] = {}
    for path, col in parts:
        for p in path:
            if p not in merged:
                merged[p] = int(col)
    return merged


def _ordered_pair(values: range | list[int], increasing: bool) -> tuple[int, int] | None:
    vals = list(values)
    if len(vals) < 2:
        return None
    a, b = sorted(random.sample(vals, 2))
    return (int(a), int(b)) if increasing else (int(b), int(a))


def _branch_specs(
    kind: str,
    H: int,
    W: int,
    r0: int,
    c0: int,
    r1: int,
    c1: int,
    r: int,
    c: int,
    R: int,
    C: int,
) -> tuple[list[tuple[int, int]], list[tuple[int, int]], tuple[int, int]]:
    is_top = kind[0] == "t"
    is_left = kind[1] == "l"

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
    return inner, outer, (int(R), int(C))


def _single_l_specs(kind: str, H: int, W: int, r0: int, c0: int, r1: int, c1: int):
    lo, hi = _OUTER_OFFSET_RANGE
    is_top = kind[0] == "t"
    is_left = kind[1] == "l"
    r = random.randint(1, H - 2)
    c = random.randint(1, W - 2)
    R = (r0 - random.randint(lo, hi)) if is_top else (r1 + random.randint(lo, hi))
    C = (c0 - random.randint(lo, hi)) if is_left else (c1 + random.randint(lo, hi))
    inner, outer, _corner = _branch_specs(kind, H, W, r0, c0, r1, c1, r, c, R, C)
    return inner, outer


def _paired_l_specs(
    kind: str,
    H: int,
    W: int,
    r0: int,
    c0: int,
    r1: int,
    c1: int,
) -> tuple[list[list[tuple[int, int]]], dict[tuple[int, int], int], tuple[int, int]] | None:
    lo, hi = _OUTER_OFFSET_RANGE
    row_increasing = kind[0] == "t"
    col_increasing = kind[1] == "l"

    rows_pair = _ordered_pair(range(1, H - 1), row_increasing)
    cols_pair = _ordered_pair(range(1, W - 1), col_increasing)
    if rows_pair is None or cols_pair is None:
        return None
    ra, rb = rows_pair
    ca, cb = cols_pair

    if kind[0] == "t":
        outer_rows = range(r0 - hi, r0 - lo + 1)
    else:
        outer_rows = range(r1 + lo, r1 + hi + 1)
    if kind[1] == "l":
        outer_cols = range(c0 - hi, c0 - lo + 1)
    else:
        outer_cols = range(c1 + lo, c1 + hi + 1)

    Rs = _ordered_pair(outer_rows, row_increasing)
    Cs = _ordered_pair(outer_cols, col_increasing)
    if Rs is None or Cs is None:
        return None
    RA, RB = Rs
    CA, CB = Cs

    inner_a, outer_a, corner_a = _branch_specs(kind, H, W, r0, c0, r1, c1, ra, ca, RA, CA)
    inner_b, outer_b, corner_b = _branch_specs(kind, H, W, r0, c0, r1, c1, rb, cb, RB, CB)

    rb_row = int(corner_b[0])
    if CA < CB:
        bridge = [(rb_row, cc) for cc in range(CA + 1, CB)]
    else:
        bridge = [(rb_row, cc) for cc in range(CA - 1, CB, -1)]

    return [inner_a, inner_b], {"outer_a": outer_a, "outer_b": outer_b, "bridge": bridge}, (ra, ca)


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
    corners = random.sample(["tl", "tr", "bl", "br"], k=n_components)
    if len(palette) < n_components + 1:
        return None

    color_i = 0

    paired_kind = corners[0]
    if color_i + 1 >= len(palette):
        return None
    paired_colors = (palette[color_i], palette[color_i + 1])
    color_i += 2
    paired = _paired_l_specs(paired_kind, H, W, r0, c0, r1, c1)
    if paired is None:
        return None
    inner_paths, outer_bits, _anchor = paired
    all_inner = [p for path in inner_paths for p in path]
    if len(set(all_inner)) != len(all_inner):
        return None
    irs, ics = zip(*all_inner)
    if used_inner[irs, ics].any():
        return None

    merged = _merge_colored_parts(
        [
            (outer_bits["outer_a"], paired_colors[0]),
            (outer_bits["outer_b"], paired_colors[1]),
            (outer_bits["bridge"], paired_colors[0]),
        ]
    )
    if merged is None:
        return None
    if not _place_colored_component(grid, comp_mask, merged, bg):
        return None
    for inner_path, col in zip(inner_paths, paired_colors):
        _paint_path(expected, inner_path, (col,))
    used_inner[irs, ics] = True

    for kind in corners[1:]:
        if color_i >= len(palette):
            return None
        inner, outer = _single_l_specs(kind, H, W, r0, c0, r1, c1)
        irs, ics = zip(*inner)
        if used_inner[irs, ics].any():
            return None
        color = palette[color_i]
        color_i += 1
        if not _place_component(grid, comp_mask, outer, (color,), bg):
            return None
        _paint_path(expected, inner, (color,))
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
    raise RuntimeError("failed to generate a valid a6f40cea D11 variant input")
