# Combo variant of a6f40cea (index 33)
# combo_spec: D5/0+D6/2+D15/0
# Name: 全网格棋盘格镂空

from __future__ import annotations

from collections import Counter, defaultdict
import random

import numpy as np
from scipy.ndimage import label

from arc_utils import STRUCT4, get_bg, neighbors4


# Design analysis for the fused variant
# design_analysis:
#   fused_rule:
#     - Keep the whole input canvas in the output instead of cropping to the frame interior.
#     - The frame interior is visibly prefilled by a 1x1 checkerboard, not a uniform fill.
#     - Recovered paths do not copy the outer component colors inward; they are carved out to a
#       single "hole" color that is exposed by four anchor cells at the inner corners.
#   behavior_trace:
#     - Original task: detect a uniformly filled frame, create a cropped fill-colored canvas, then
#       paint projected mono / alternating paths inside it.
#     - Here: detect the largest frame whose interior is "checkerboard + four equal corner anchors",
#       rebuild that checkerboard inside a full-grid copy, then set every recovered path cell to the
#       anchor color.
#     - Example: if an outside L touches the top ring at local (0,5) and the left ring at (3,0),
#       the geometry stage still reconstructs the same Manhattan L. The only change is write-back:
#       absolute cells inside the original frame become the hole color instead of becoming the
#       component color, while everything outside the frame stays unchanged.
#   complexity_calibration:
#     - The generator keeps the original 2-3 external components, optional U-pairs, optional
#       alternating colors, and optional distractors. The geometric reasoning load is unchanged.
#     - The checkerboard is intentionally only two colors, with one color tied to the global
#       background, so the base is visually rich but still legible. The hole color is a distinct
#       third interior color, keeping the erase effect obvious.
#   implicit_assumption_audit:
#     - Original `_find_framed_rect_uniform_fill()` assumed every interior cell matched one fill
#       value; that fails immediately on a checkerboard and must be replaced.
#     - Original `out = np.full(..., fill)` would destroy both the full-grid requirement and the
#       checkerboard base. The fused solver must reconstruct the base pattern in absolute coords.
#     - Original component painting depended on outer colors and, for 2-color components, endpoint
#       phase hints. In the erase variant, the output color no longer depends on those hints, so the
#       solver only needs them for grouping monochrome U-pairs by color.
#     - Because the erase target must be human-visible in the input, the generator adds four inner
#       corner anchors. The solver therefore assumes all four inner corners share one color and the
#       remaining interior follows a strict parity checkerboard.
#   placement_legality:
#     - Corner anchors are safe because generated L-paths never use interior corner cells and U
#       stripes use only rows / columns strictly away from the corners.
#     - External components still use the original no-overlap and no-adjacency placement checks and
#       remain outside the framed rectangle.
#     - A `used_inner` mask still keeps all inferred inner paths disjoint, so erasing one path never
#       overwrites another intended path.
#   observability_and_cleanliness:
#     - Across a few samples, a human sees the same outside open-rectangle cues, but now the output
#       preserves the whole scene and only the frame interior changes from "checkerboard + 4 corner
#       anchors" to "checkerboard with a carved hole path".
#     - The visible hole color is explicitly present in the input and is always distinct from the
#       non-anchor checker color, so the erase rule is inferable without using hidden color ids.
#     - Visual clutter is controlled by keeping the interior structure to three roles only:
#       checker-even, checker-odd, and hole anchors / holes.
#   new_main_assumptions_checked_against_generator:
#     - The target frame is the largest complete 1px rectangle whose interior matches the anchored
#       checkerboard template. `generate_input()` creates exactly one such frame.
#     - All four inner-corner cells share the hole color. The generator enforces this directly.
#     - Outer components never use the border color, either checker color, or the hole color, so
#       component extraction by color exclusion is sound.


def _make_checkerboard(h: int, w: int, even: int, odd: int) -> np.ndarray:
    """Construct a checkerboard whose local (0,0) parity is `even`."""
    rr, cc = np.indices((int(h), int(w)))
    return np.where((rr + cc) % 2 == 0, int(even), int(odd)).astype(int)


def _anchor_cells(h: int, w: int) -> list[tuple[int, int]]:
    """Return the four interior-corner coordinates in local frame coordinates."""
    return [(0, 0), (0, int(w - 1)), (int(h - 1), 0), (int(h - 1), int(w - 1))]


def _apply_corner_anchors_local(arr: np.ndarray, fill: int) -> None:
    """Set the four corner cells of an interior array to the hole color."""
    h, w = map(int, arr.shape)
    for rr, cc in _anchor_cells(h, w):
        arr[int(rr), int(cc)] = int(fill)


def _checkerboard_with_corner_fill(inner: np.ndarray) -> tuple[int, int, int] | None:
    """
    Validate an interior as checkerboard + four equal corner anchors.

    Returns (fill, even_color, odd_color), where `even_color` is the checker color for local
    parity (r+c)%2==0 before the corners were replaced by `fill`.
    """
    h, w = map(int, inner.shape)
    if h < 2 or w < 2:
        return None

    corners = set(_anchor_cells(h, w))
    fill = int(inner[0, 0])
    for rr, cc in corners:
        if int(inner[rr, cc]) != int(fill):
            return None

    even = None
    odd = None
    seen = set()
    for rr in range(h):
        for cc in range(w):
            if (int(rr), int(cc)) in corners:
                continue
            val = int(inner[rr, cc])
            seen.add(val)
            if (rr + cc) % 2 == 0:
                if even is None:
                    even = int(val)
                elif int(even) != int(val):
                    return None
            else:
                if odd is None:
                    odd = int(val)
                elif int(odd) != int(val):
                    return None

    if even is None or odd is None:
        return None
    if int(even) == int(odd):
        return None
    if int(fill) in {int(even), int(odd)}:
        return None
    if seen != {int(even), int(odd)}:
        return None
    return int(fill), int(even), int(odd)


def _find_framed_rect_checkerboard_anchors(
    g: np.ndarray,
    *,
    bg: int | None = None,
    structure=STRUCT4,
) -> tuple[int, int, int, int, int, int, int, int] | None:
    """
    Detect the main 1px-thick rectangular frame with an anchored checkerboard interior.

    Returns (r0, c0, r1, c1, border_color, fill_color, even_color, odd_color) for the largest
    valid rectangle, else None.
    """
    if bg is None:
        bg = int(get_bg(g))
    else:
        bg = int(bg)

    best = None  # (area, r0, c0, r1, c1, bc, fill, even, odd)
    for bc in dict.fromkeys(map(int, g.flatten())):
        if int(bc) == int(bg):
            continue
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
            fill_info = _checkerboard_with_corner_fill(inner)
            if fill_info is None:
                continue
            fill, even, odd = fill_info
            if int(bc) in {int(fill), int(even), int(odd)}:
                continue

            area = int(h * w)
            cand = (area, r0, c0, r1, c1, int(bc), int(fill), int(even), int(odd))
            if best is None or cand[0] > best[0]:
                best = cand

    if best is None:
        return None
    _area, r0, c0, r1, c1, bc, fill, even, odd = best
    return int(r0), int(c0), int(r1), int(c1), int(bc), int(fill), int(even), int(odd)


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


def _shift_path(path: list[tuple[int, int]], dr: int, dc: int) -> list[tuple[int, int]]:
    """Translate a local path to absolute coordinates."""
    return [(int(r + dr), int(c + dc)) for r, c in path]


def _paint_path(grid: np.ndarray, path: list[tuple[int, int]], colors: tuple[int, ...]) -> None:
    """Paint a path with solid or alternating colors."""
    if not path:
        return
    rs, cs = zip(*path)
    if len(colors) == 1:
        grid[rs, cs] = int(colors[0])
    else:
        a, b = map(int, colors)
        grid[rs, cs] = [a if i % 2 == 0 else b for i in range(len(path))]


def _erase_path(grid: np.ndarray, path: list[tuple[int, int]], fill: int) -> None:
    """Overwrite a path with the hole color."""
    if not path:
        return
    rs, cs = zip(*path)
    grid[rs, cs] = int(fill)


def _can_place(grid: np.ndarray, coords: list[tuple[int, int]], bg: int, comp_mask: np.ndarray) -> bool:
    """Check that all coords are in-bounds, on bg, and not 4-adjacent to existing components."""
    H, W = grid.shape
    s = set(coords)
    for r, c in s:
        if not (0 <= int(r) < int(H) and 0 <= int(c) < int(W)):
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
    """Build inner full-span stripes and two outer U-shaped open rectangles."""
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


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recover framed Manhattan paths as holes in a checkerboard panel while keeping the full canvas.

    Input structure:
    - A single 1px rectangular frame encloses a checkerboard interior whose four inner-corner cells
      all share a distinct hole color.
    - Outside the frame are 4-connected open-rectangle traces touching the ring just beyond the
      border, plus optional distractor blobs away from that ring.

    Rule:
    - Output has the same shape as the input and preserves everything outside the frame interior.
    - Reconstruct the same projected L-paths / full-span stripes encoded by the ring-touch points.
    - Inside the frame, restore the anchored checkerboard base and erase each recovered path to the
      hole color instead of painting it with the outer trace colors.

    Key insight:
    - The outer traces still determine geometry exactly as before; the fused change is that the
      target panel is a visible checkerboard on the original grid, and the path operation is
      subtractive, using the shared corner-anchor color as the carved-out value.
    """
    g = np.array(grid, dtype=int)
    H, W = g.shape
    bg = int(get_bg(g))

    found = _find_framed_rect_checkerboard_anchors(g, bg=int(bg))
    assert found is not None
    r0, c0, r1, c1, bc, fill, base_even, base_odd = found
    outH = int(r1 - r0 - 1)
    outW = int(c1 - c0 - 1)
    r_int0 = int(r0 + 1)
    c_int0 = int(c0 + 1)

    out = g.copy()
    base = _make_checkerboard(outH, outW, int(base_even), int(base_odd))
    _apply_corner_anchors_local(base, int(fill))
    out[r_int0:r1, c_int0:c1] = base

    def map_to_local(rr: int, cc: int) -> tuple[int, int]:
        r = max(0, min(outH - 1, int(rr - r_int0)))
        c = max(0, min(outW - 1, int(cc - c_int0)))
        return int(r), int(c)

    def erase_local(path_pts: list[tuple[int, int]]) -> None:
        for rr, cc in path_pts:
            out[int(r_int0 + rr), int(c_int0 + cc)] = int(fill)

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
        mr, mc = map_to_local(rr, cc)
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
        for rr, cc, col in pts:
            hist[(int(rr), int(cc))][int(col)] += 1
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
            erase_local(chosen)
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
            pathA = build_manhattan(p1, p2, True)
            pathB = build_manhattan(p1, p2, False)
            chosen = pathB if (inside_corner(p1, p2, False) and not inside_corner(p1, p2, True)) else pathA
            erase_local(chosen)
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
                erase_local([(rr, int(cc)) for rr in range(min(rs), max(rs) + 1)])
            continue
        lefts = [p for p in pts if p[1] == 0]
        rights = [p for p in pts if p[1] == outW - 1]
        if lefts and rights:
            for rr in sorted({r for r, _c in pts}):
                cs = [c for r, c in pts if r == rr]
                erase_local([(int(rr), cc) for cc in range(min(cs), max(cs) + 1)])
            continue
        if len(pts) == 1:
            rr, cc = pts[0]
            erase_local([(int(rr), int(cc))])
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
        erase_local(chosen)

    return out


# --- CFG: tunable generation parameters ---
_BORDER_COLOR = 3
_BG_CHOICES = [0, 1, 2, 4, 5, 6, 7, 8, 9]
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


def _pick_fill_and_checker(bg: int, bc: int) -> tuple[int, int, int]:
    """
    Choose a distinct hole color and a checkerboard that uses `bg` on one parity.

    Keeping one checker color equal to the global background preserves the original task's visual
    feel and leaves enough remaining colors for alternating external components.
    """
    usable = [x for x in range(10) if x not in (int(bg), int(bc))]
    fill = int(random.choice(usable))
    other = int(random.choice([x for x in usable if x != int(fill)]))
    if random.random() < 0.5:
        return int(fill), int(bg), int(other)
    return int(fill), int(other), int(bg)


def _gen_attempt() -> np.ndarray | None:
    """Single attempt to generate a valid input grid. Returns None on failure."""
    bg = int(random.choice(_BG_CHOICES))
    bc = int(_BORDER_COLOR)
    fill, base_even, base_odd = _pick_fill_and_checker(int(bg), int(bc))
    H = random.randint(*_INTERIOR_H_RANGE)
    W = random.randint(*_INTERIOR_W_RANGE)
    rows = random.randint(max(_GRID_SIZE_RANGE[0], H + _GRID_INTERIOR_PAD), _GRID_SIZE_RANGE[1])
    cols = random.randint(max(_GRID_SIZE_RANGE[0], W + _GRID_INTERIOR_PAD), _GRID_SIZE_RANGE[1])
    r0 = random.randint(_FRAME_MARGIN, rows - H - _FRAME_MARGIN_FAR)
    c0 = random.randint(_FRAME_MARGIN, cols - W - _FRAME_MARGIN_FAR)
    r1, c1 = r0 + H + 1, c0 + W + 1

    grid = np.full((rows, cols), int(bg), dtype=int)
    grid[r0, c0 : c1 + 1] = int(bc)
    grid[r1, c0 : c1 + 1] = int(bc)
    grid[r0 : r1 + 1, c0] = int(bc)
    grid[r0 : r1 + 1, c1] = int(bc)

    interior = _make_checkerboard(H, W, int(base_even), int(base_odd))
    _apply_corner_anchors_local(interior, int(fill))
    grid[r0 + 1 : r1, c0 + 1 : c1] = interior

    comp_mask = np.zeros_like(grid, dtype=bool)
    excluded = {int(bg), int(bc), int(fill), int(base_even), int(base_odd)}
    palette = [x for x in range(10) if x not in excluded]
    random.shuffle(palette)
    used_inner = np.zeros((H, W), dtype=bool)
    inner_specs: list[list[tuple[int, int]]] = []

    n_components = random.randint(*_N_COMPONENTS_RANGE)
    remaining = n_components
    color_i = 0

    if remaining >= 2 and random.random() < _U_PAIR_PROB:
        if color_i >= len(palette):
            return None
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
        inner_specs.extend(inner_lines)
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
            colors = (int(palette[color_i]), int(palette[color_i + 1]))
            color_i += 2
        else:
            colors = (int(palette[color_i]),)
            color_i += 1
        if not _place_component(grid, comp_mask, outer, colors, int(bg)):
            return None
        inner_specs.append(inner)
        used_inner[irs, ics] = True

    for _ in range(random.randint(*_DISTRACTOR_COUNT_RANGE)):
        if color_i >= len(palette):
            break
        _add_distractor(grid, comp_mask, int(bg), (r0, c0, r1, c1), int(palette[color_i]))
        color_i += 1

    expected = np.array(grid, copy=True)
    interior_out = _make_checkerboard(H, W, int(base_even), int(base_odd))
    _apply_corner_anchors_local(interior_out, int(fill))
    expected[r0 + 1 : r1, c0 + 1 : c1] = interior_out
    for inner in inner_specs:
        _erase_path(expected, _shift_path(inner, r0 + 1, c0 + 1), int(fill))

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
