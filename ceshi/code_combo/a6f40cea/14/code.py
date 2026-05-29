# Combo variant of a6f40cea (index 14)
# combo_spec: D6/2+D15/0
# Name: 棋盘格镂空路径

# Design analysis for the combo variant
# a) Concrete behavior and trace:
#    - Core geometry is preserved from a6f40cea: outer open-rectangle traces still project to
#      interior Manhattan L-paths or full-span stripes.
#    - D15 contributes a two-color checkerboard floor inside the closed frame.
#    - D6 contributes erase semantics: recovered paths are not painted with the trace colors;
#      instead, path cells are turned into one visible fill color.
#    - To make that erase target observable, the input interior uses a fused structure:
#      checkerboard everywhere except the four inner corners, which are all the fill color.
#      Example on a small panel:
#      source task: solid fill base -> paint colored L.
#      D15 only: checkerboard base -> paint colored L over it.
#      combo: checkerboard base + four fill anchors -> crop that panel and erase the L to the
#      anchor color. The output is therefore a checkerboard sheet with a blank path cut out.
# b) Complexity calibration:
#    - The fusion keeps the original 2-3 outer components, optional paired U-components,
#      optional two-color alternation, and optional distractors.
#    - Only one new interior template is introduced: strict checkerboard plus four fixed anchors.
#      This is visually richer than a flat sheet but still regular enough for humans to spot.
#    - Checker colors are always distinct from the erase color so every erased path cell changes
#      visibly; otherwise the combo would partially degenerate.
# c) Implicit assumption audit:
#    - Original `_find_framed_rect_uniform_fill()` fails because the interior is no longer uniform.
#      The solver now detects the largest complete 1px rectangular border regardless of interior.
#    - Original `out = full(fill)` fails because it would destroy the observed checkerboard.
#      The solver now copies the cropped interior and edits only recovered path cells.
#    - Original component mask excluded one fill color. The combo interior uses three semantic
#      colors (erase fill + two checker colors), so all three must be excluded from component
#      detection.
#    - D15's strict checkerboard detector also no longer applies directly because the four corners
#      are overwritten by fill anchors. The solver therefore validates "checkerboard except
#      corners, with four equal fill anchors" instead of a pure checkerboard.
# d) Placement legality:
#    - The generator remains output-first: build the interior panel first, then derive outer traces.
#    - Corner anchors are safe because generated L-paths never include interior corners and U-pair
#      stripes use rows/cols chosen strictly away from the corners.
#    - Outer components still use the original non-overlap and non-adjacency checks.
#    - Path cells are tracked with `used_inner` so erased regions do not overlap each other.
# e) Visual observability:
#    - A human sees the same outer trace language as the source task, but the framed interior is
#      already patterned before any transformation.
#    - The input itself exposes the erase target via the four matching corner cells, so the rule is
#      inferable from visible structure rather than hidden color ids.
#    - Across examples, the key consistent change is "recover path geometry, then punch it out of
#      the checkerboard to the corner-anchor color".
# f) Visual cleanliness:
#    - The interior uses only three colors total: two checker colors and one anchor/hole color.
#      That keeps the crop legible even with multiple erased paths.
#    - Distractors remain outside the frame, so the panel itself is not cluttered.
# g) Color inference:
#    - No ordering on numeric color ids is used. The checker parity is read directly from position,
#      and the erase target is the repeated corner color.
# 9) New solver assumptions audited against the generator:
#    - `main()` assumes exactly one largest complete rectangular border; `generate_input()` creates
#      exactly one and keeps all other components open.
#    - `main()` assumes the four inner corners share one fill color and all other interior cells
#      follow a two-color checkerboard by parity; the generator enforces that exact template.
#    - `main()` assumes outer trace colors avoid bg, border, fill, and checker colors so component
#      extraction is unambiguous; the generator chooses its palette accordingly.
#    - `main()` assumes ring-touching components lie fully outside the frame bbox; `_place_component`
#      plus the path construction preserve that invariant.

from __future__ import annotations

from collections import Counter, defaultdict
import random

import numpy as np
from scipy.ndimage import label

from arc_utils import STRUCT4, get_bg, neighbors4


def _find_framed_rect(
    g: np.ndarray,
    *,
    bg: int | None = None,
    structure=STRUCT4,
) -> tuple[int, int, int, int, int] | None:
    """
    Detect the main 1px-thick rectangular frame even when its interior is patterned.

    Returns (r0, c0, r1, c1, border_color) for the largest such rectangle, else None.
    """
    if bg is None:
        bg = int(get_bg(g))
    else:
        bg = int(bg)

    best = None  # (area, r0, c0, r1, c1, bc)
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

            area = int(h * w)
            cand = (area, r0, c0, r1, c1, int(bc))
            if best is None or cand[0] > best[0]:
                best = cand

    if best is None:
        return None
    _area, r0, c0, r1, c1, bc = best
    return int(r0), int(c0), int(r1), int(c1), int(bc)


def _make_checkerboard(h: int, w: int, even: int, odd: int) -> np.ndarray:
    """Construct a checkerboard whose (0,0) cell is `even`."""
    rr, cc = np.indices((int(h), int(w)))
    return np.where((rr + cc) % 2 == 0, int(even), int(odd)).astype(int)


def _apply_corner_fill_anchors(panel: np.ndarray, fill: int) -> np.ndarray:
    """Return a copy of `panel` with the four inner-corner anchor cells set to `fill`."""
    out = np.array(panel, dtype=int, copy=True)
    h, w = out.shape
    out[0, 0] = int(fill)
    out[0, w - 1] = int(fill)
    out[h - 1, 0] = int(fill)
    out[h - 1, w - 1] = int(fill)
    return out


def _infer_checker_hole_panel(inner: np.ndarray) -> tuple[int, int, int] | None:
    """
    Validate "checkerboard except four fill anchors" and return (fill, even_color, odd_color).
    """
    h, w = map(int, inner.shape)
    if h < 2 or w < 2:
        return None

    corners = [
        int(inner[0, 0]),
        int(inner[0, w - 1]),
        int(inner[h - 1, 0]),
        int(inner[h - 1, w - 1]),
    ]
    if len(set(corners)) != 1:
        return None
    fill = int(corners[0])

    even = None
    odd = None
    for r in range(h):
        for c in range(w):
            if (r, c) in {(0, 0), (0, w - 1), (h - 1, 0), (h - 1, w - 1)}:
                continue
            v = int(inner[r, c])
            if v == int(fill):
                return None
            if (r + c) % 2 == 0:
                if even is None:
                    even = int(v)
                elif int(v) != int(even):
                    return None
            else:
                if odd is None:
                    odd = int(v)
                elif int(v) != int(odd):
                    return None

    if even is None or odd is None or int(even) == int(odd):
        return None
    return int(fill), int(even), int(odd)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recover projected Manhattan paths inside a framed panel and punch them out of a checkerboard.

    Input structure:
    - One 1px rectangular frame encloses a two-color checkerboard interior whose four inner corners
      are all overwritten by the same fill color.
    - Colored open-rectangle traces lie outside the frame and touch the ring immediately around it.

    Rule:
    - Crop the observed frame interior as the starting output.
    - Reconstruct the same projected L-paths and full-span stripes from the outer traces as in the
      source task.
    - For every recovered path cell, erase that cell to the repeated corner-anchor color instead of
      painting it with the outer trace colors.

    Key insight: The outside traces still specify path geometry, while the repeated corner color
    reveals the hole color and the checkerboard must be preserved everywhere the path does not pass.
    """
    g = np.array(grid, dtype=int)
    H, W = g.shape
    bg = int(get_bg(g))

    found = _find_framed_rect(g, bg=int(bg))
    assert not (found is None)
    r0, c0, r1, c1, bc = found
    out = g[r0 + 1 : r1, c0 + 1 : c1].copy()
    outH, outW = out.shape
    r_int0 = int(r0 + 1)
    c_int0 = int(c0 + 1)

    panel_info = _infer_checker_hole_panel(out)
    assert not (panel_info is None)
    fill, base_even, base_odd = panel_info

    def map_to_out(rr: int, cc: int) -> tuple[int, int]:
        r = max(0, min(outH - 1, int(rr - r_int0)))
        c = max(0, min(outW - 1, int(cc - c_int0)))
        return int(r), int(c)

    def build_manhattan(
        p1: tuple[int, int],
        p2: tuple[int, int],
        corner_a: bool,
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

    def inside_corner(p1: tuple[int, int], p2: tuple[int, int], corner_a: bool) -> bool:
        (ra, ca), (rb, cb) = p1, p2
        if ra == rb or ca == cb:
            return True
        corner = (rb, ca) if corner_a else (ra, cb)
        rr, cc = corner
        return (0 < rr < outH - 1) and (0 < cc < outW - 1)

    def erase_path(path_pts: list[tuple[int, int]]) -> None:
        for rr, cc in path_pts:
            out[int(rr), int(cc)] = int(fill)

    excluded = {int(bg), int(bc), int(fill), int(base_even), int(base_odd)}
    comp_mask = np.ones_like(g, dtype=bool)
    for col in excluded:
        comp_mask &= g != int(col)
    comp_mask[r0 : r1 + 1, c0 : c1 + 1] = False
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

            best = None
            for i in range(len(coords)):
                for j in range(i + 1, len(coords)):
                    d = abs(coords[i][0] - coords[j][0]) + abs(coords[i][1] - coords[j][1])
                    if best is None or d > best[0]:
                        best = (d, coords[i], coords[j])
            if best is None:
                continue
            _d, p1, p2 = best
            path_a = build_manhattan(p1, p2, True)
            path_b = build_manhattan(p1, p2, False)
            chosen = path_b if (inside_corner(p1, p2, False) and not inside_corner(p1, p2, True)) else path_a
            erase_path(chosen)
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
            path_a = build_manhattan(a, b, True)
            path_b = build_manhattan(a, b, False)
            path = path_b if (inside_corner(a, b, False) and not inside_corner(a, b, True)) else path_a
            erase_path(path)
            continue

    for _col, pts in deferred.items():
        pts = list(pts)
        if not pts:
            continue
        tops = [p for p in pts if p[0] == 0]
        bots = [p for p in pts if p[0] == outH - 1]
        if tops and bots:
            for cc in sorted({c for _r, c in pts}):
                rs = [r for r, c in pts if c == cc]
                erase_path([(rr, int(cc)) for rr in range(min(rs), max(rs) + 1)])
            continue
        lefts = [p for p in pts if p[1] == 0]
        rights = [p for p in pts if p[1] == outW - 1]
        if lefts and rights:
            for rr in sorted({r for r, _c in pts}):
                cs = [c for r, c in pts if r == rr]
                erase_path([(int(rr), cc) for cc in range(min(cs), max(cs) + 1)])
            continue
        if len(pts) == 1:
            rr, cc = pts[0]
            out[int(rr), int(cc)] = int(fill)
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
        path_a = build_manhattan(p1, p2, True)
        path_b = build_manhattan(p1, p2, False)
        chosen = path_b if (inside_corner(p1, p2, False) and not inside_corner(p1, p2, True)) else path_a
        erase_path(chosen)

    return out


# --------------- generator (a6 combo D6/2 + D15/0) ---------------

# --- CFG: tunable generation parameters ---
_BORDER_COLOR = 3
_BG_CHOICES = [0, 1, 2, 4, 5, 6, 7, 8, 9]
_FILL_EQ_BG_PROB = 0.7
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
_GEN_RETRIES = 400


def _line(a: int, b: int) -> range:
    """Inclusive range from a to b."""
    step = 1 if b >= a else -1
    return range(a, b + step, step)


def _vpath(r0: int, r1: int, c: int) -> list[tuple[int, int]]:
    """Vertical segment from (r0, c) to (r1, c) inclusive."""
    return [(r, c) for r in _line(r0, r1)]


def _hpath(r: int, c0: int, c1: int) -> list[tuple[int, int]]:
    """Horizontal segment from (r, c0) to (r, c1) inclusive."""
    return [(r, c) for c in _line(c0, c1)]


def _join(*parts: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """Concatenate path segments, deduplicating shared endpoints."""
    out: list[tuple[int, int]] = []
    for part in parts:
        for p in part:
            if not out or out[-1] != p:
                out.append(p)
    return out


def _paint_path(grid: np.ndarray, path: list[tuple[int, int]], colors: tuple[int, ...]) -> None:
    """Paint a path with solid or alternating colors."""
    if not path:
        return
    rs, cs = zip(*path)
    if len(colors) == 1:
        grid[rs, cs] = colors[0]
    else:
        a, b = colors
        vals = [a if i % 2 == 0 else b for i in range(len(path))]
        grid[rs, cs] = vals


def _erase_path(grid: np.ndarray, path: list[tuple[int, int]], fill: int) -> None:
    """Set every path cell to the erase/fill color."""
    if not path:
        return
    rs, cs = zip(*path)
    grid[rs, cs] = int(fill)


def _can_place(grid: np.ndarray, coords: list[tuple[int, int]], bg: int, comp_mask: np.ndarray) -> bool:
    """Check that all coords are in-bounds, on bg, and not 4-adjacent to existing components."""
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
    """Validate, paint, and register a component. Returns True on success."""
    if len(set(path)) != len(path):
        return False
    if not _can_place(grid, path, bg, comp_mask):
        return False
    _paint_path(grid, path, colors)
    rs, cs = zip(*path)
    comp_mask[rs, cs] = True
    return True


def _pick_fill(bg: int, bc: int) -> int:
    """Choose the erase color; often equal to bg so the path reads as a hole."""
    if random.random() < _FILL_EQ_BG_PROB:
        return int(bg)
    return int(random.choice([x for x in range(10) if x not in (bg, bc)]))


def _pick_checker_colors(bg: int, bc: int, fill: int) -> tuple[int, int]:
    """Choose two checkerboard colors distinct from bg, border, and erase color."""
    usable = [x for x in range(10) if x not in (bg, bc, fill)]
    a, b = random.sample(usable, 2)
    if random.random() < 0.5:
        return int(a), int(b)
    return int(b), int(a)


def _l_specs(kind: str, H: int, W: int, r0: int, c0: int, r1: int, c1: int):
    """Build inner (interior L-path) and outer (open-rect) coords for one L-component."""
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
    """Build inner (two full-span stripes) and two outer U-shaped open-rects."""
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
    """Try to place a small random blob away from the frame."""
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
    """Single attempt to generate a valid input grid. Returns None on failure."""
    bg = random.choice(_BG_CHOICES)
    bc = _BORDER_COLOR
    fill = _pick_fill(int(bg), int(bc))
    base_even, base_odd = _pick_checker_colors(int(bg), int(bc), int(fill))
    H = random.randint(*_INTERIOR_H_RANGE)
    W = random.randint(*_INTERIOR_W_RANGE)
    rows = random.randint(max(_GRID_SIZE_RANGE[0], H + _GRID_INTERIOR_PAD), _GRID_SIZE_RANGE[1])
    cols = random.randint(max(_GRID_SIZE_RANGE[0], W + _GRID_INTERIOR_PAD), _GRID_SIZE_RANGE[1])
    r0 = random.randint(_FRAME_MARGIN, rows - H - _FRAME_MARGIN_FAR)
    c0 = random.randint(_FRAME_MARGIN, cols - W - _FRAME_MARGIN_FAR)
    r1, c1 = r0 + H + 1, c0 + W + 1

    base_panel = _make_checkerboard(H, W, int(base_even), int(base_odd))
    anchored_panel = _apply_corner_fill_anchors(base_panel, int(fill))

    grid = np.full((rows, cols), bg, dtype=int)
    grid[r0, c0 : c1 + 1] = bc
    grid[r1, c0 : c1 + 1] = bc
    grid[r0 : r1 + 1, c0] = bc
    grid[r0 : r1 + 1, c1] = bc
    grid[r0 + 1 : r1, c0 + 1 : c1] = anchored_panel

    expected = anchored_panel.copy()
    comp_mask = np.zeros_like(grid, dtype=bool)
    palette = [x for x in range(10) if x not in {int(bg), int(bc), int(fill), int(base_even), int(base_odd)}]
    random.shuffle(palette)
    used_inner = np.zeros((H, W), dtype=bool)

    n_components = random.randint(*_N_COMPONENTS_RANGE)
    remaining = n_components
    color_i = 0

    if remaining >= 2 and random.random() < _U_PAIR_PROB:
        orientation = random.choice(["vertical", "horizontal"])
        if orientation == "horizontal" and H < 5:
            orientation = "vertical"
        if color_i >= len(palette):
            return None
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
        _erase_path(expected, all_inner, int(fill))
        used_inner[irs, ics] = True
        remaining -= 2

    corners = random.sample(["tl", "tr", "bl", "br"], k=remaining)
    for kind in corners:
        if color_i >= len(palette):
            return None
        inner, outer = _l_specs(kind, H, W, r0, c0, r1, c1)
        irs, ics = zip(*inner)
        if used_inner[irs, ics].any():
            return None
        if random.random() < _DUAL_COLOR_PROB and color_i + 1 < len(palette):
            colors = (palette[color_i], palette[color_i + 1])
            color_i += 2
        else:
            colors = (palette[color_i],)
            color_i += 1
        if not _place_component(grid, comp_mask, outer, colors, bg):
            return None
        _erase_path(expected, inner, int(fill))
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
    raise RuntimeError("failed to generate a valid a6f40cea combo input")
