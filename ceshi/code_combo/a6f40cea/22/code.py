# Combo variant of a6f40cea (index 22)
# combo_spec: D6/0+D7/1
# Name: 密集路径碰撞变色

"""
Design analysis for the dense-collision combo variant.

a) Concrete behavior and trace:
   - Base task: each outside open rectangle gives one inside Manhattan path and
     the solver can paint it immediately because overlaps are rare and visually
     unimportant.
   - This combo keeps the same geometric decoding, but now every sample uses
     one monochrome vertical U-pair (2 components). 4-component samples add two
     corner L-components, while 5-6 component samples add a horizontal U-pair
     and then one or two corner L-components. The recovered paths are
     intentionally packed so the stripes and Ls cross many times.
   - Example trace on a small interior:
       tl path paints a column down and then a row left,
       br path paints a column up and then a row right.
     In the base task, whichever path happened to be written last would own the
     shared cells. In this combo, every path is reconstructed first, visits are
     counted globally, and any cell visited by multiple paths becomes the fixed
     collision color instead of either path color.
   - The visual difference is therefore twofold at once: more components make
     overlaps common, and the overlap result is a third color rather than a
     hidden overwrite order.

b) Complexity calibration:
   - The new rule is visible in only one new way: multi-hit cells turn red.
   - The generator keeps the path family narrow: two stripe paths come from a
     same-color U-pair and two more paths come from L-style open rectangles.
   - The generator keeps the exterior routing narrow and template-based, so
     5-6 component examples stay readable instead of degenerating into tangled
     outside clutter.
   - Two-color alternating traces are kept but capped by palette budget; this
     preserves the original alternation clue without flooding the grid.

c) Implicit assumptions audited from the original main():
   - Direct `out[r, c] = color` writes assumed overlap was irrelevant. That is
     false now, so the solver first converts every component into a path map and
     only then composes them with collision detection.
   - Arbitrary component iteration order no longer matters because the combo
     output is defined by visit multiplicity, not overwrite order.
   - Deferred monochrome stripe handling from the original solver is preserved
     in the same collision-aware composition pass, and the generator now uses it
     explicitly for the top/bottom U-pair that reconstructs the two stripes.
   - Alternating-color phase inference still depends on visible ring-touch
     endpoint colors; the generator therefore preserves the original parity
     relationship between outer traces and inner paths.

d) Placement legality:
   - Outside components remain pairwise non-overlapping and non-4-adjacent by
     reusing the original component placement checks.
   - The generator fixes four exterior corridor rows/columns: the U-pair stays
     close to the frame, while the two L-components route one lane farther out.
     This prevents the outer traces from tangling even though the inside paths
     are deliberately collision-heavy.

e) Visual observability:
   - The collision color is fixed to red and is excluded from background, fill,
     and path palettes, so every red output pixel is an explicit "crossing"
     signal.
   - No invisible metadata is used for sorting or tie-breaking; every inferred
     fact comes from visible geometry and endpoint colors.
   - Samples always contain several collision cells, so the combo never silently
     degenerates into the base task.

f) Visual cleanliness:
   - The generator allows only point collisions by requiring distinct path rows
     and columns and rejecting any planned cell hit by 3 or more paths.
   - Distractors are removed; the grid is already visually busy because of the
     dense path field, so extra clutter would only hurt readability.

g) Change inferability:
   - Colors do not participate in any hidden mapping. The only new color event
     is: "if multiple reconstructed paths want the same inside cell, that cell
     becomes red."
   - Because several paths cross in every sample and non-collision segments keep
     their source colors, humans can infer both the base path reconstruction and
     the collision recolor from 2-4 examples.

New solver assumptions and generator guarantees:
   - Every intended primary component is a separate 4-connected component that
     touches the outer ring in either two adjacent-side positions (the L traces)
     or two same-side positions paired across top/bottom (the U traces).
   - The generator guarantees enough frame margin for the randomized outer
     corridors to fit legally.
   - The generator validates that the intended path plan has at least six
     collision cells and no triple-hit cells before accepting a sample.
"""

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
    """Detect the largest 1px rectangular frame with a uniform interior fill."""
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

            cand = (int(h * w), r0, c0, r1, c1, int(bc), int(fill))
            if best is None or cand[0] > best[0]:
                best = cand

    if best is None:
        return None
    _area, r0, c0, r1, c1, bc, fill = best
    return int(r0), int(c0), int(r1), int(c1), int(bc), int(fill)


def _build_manhattan(
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


def _inside_corner(
    p1: tuple[int, int],
    p2: tuple[int, int],
    corner_a: bool,
    out_h: int,
    out_w: int,
) -> bool:
    (ra, ca), (rb, cb) = p1, p2
    if ra == rb or ca == cb:
        return True
    corner = (rb, ca) if corner_a else (ra, cb)
    rr, cc = corner
    return 0 < int(rr) < int(out_h - 1) and 0 < int(cc) < int(out_w - 1)


def _choose_inside_path(
    p1: tuple[int, int],
    p2: tuple[int, int],
    out_h: int,
    out_w: int,
) -> list[tuple[int, int]]:
    path_a = _build_manhattan(p1, p2, True)
    path_b = _build_manhattan(p1, p2, False)
    if _inside_corner(p1, p2, False, out_h, out_w) and not _inside_corner(p1, p2, True, out_h, out_w):
        return path_b
    return path_a


def _path_color_map(path: list[tuple[int, int]], colors: tuple[int, ...]) -> dict[tuple[int, int], int]:
    """Return the intended per-cell colors for a solid or alternating path."""
    out: dict[tuple[int, int], int] = {}
    if not path:
        return out
    if len(colors) == 1:
        col = int(colors[0])
        for rr, cc in path:
            out[(int(rr), int(cc))] = int(col)
        return out

    ca, cb = map(int, colors)
    cur = int(ca)
    for rr, cc in path:
        out[(int(rr), int(cc))] = int(cur)
        cur = int(cb) if int(cur) == int(ca) else int(ca)
    return out


def _compose_with_collisions(
    out_h: int,
    out_w: int,
    fill: int,
    path_maps: list[dict[tuple[int, int], int]],
) -> tuple[np.ndarray, int]:
    """
    Merge all reconstructed paths into the cropped output.

    Any cell visited by 2 or more distinct paths is recolored to the fixed
    collision color instead of using path write order.
    """
    out = np.full((int(out_h), int(out_w)), int(fill), dtype=int)
    visits: dict[tuple[int, int], list[int]] = defaultdict(list)
    for path_map in path_maps:
        for coord, col in path_map.items():
            visits[(int(coord[0]), int(coord[1]))].append(int(col))

    collisions = 0
    for (rr, cc), cols in visits.items():
        if len(cols) >= 2:
            out[int(rr), int(cc)] = int(_COLLISION_COLOR)
            collisions += 1
        else:
            out[int(rr), int(cc)] = int(cols[0])
    return out, int(collisions)


def _extract_path_maps(
    g: np.ndarray,
    r0: int,
    c0: int,
    r1: int,
    c1: int,
    bg: int,
    bc: int,
    fill: int,
) -> tuple[list[dict[tuple[int, int], int]], int, int]:
    """
    Recover all inside paths from outside components as collision-ready path maps.

    The original same-side monochrome stripe fallback is kept, but every path is
    returned as a separate map so collision composition can be done globally.
    """
    H, W = g.shape
    out_h = int(r1 - r0 - 1)
    out_w = int(c1 - c0 - 1)
    r_int0 = int(r0 + 1)
    c_int0 = int(c0 + 1)

    def map_to_out(rr: int, cc: int) -> tuple[int, int]:
        mr = max(0, min(out_h - 1, int(rr - r_int0)))
        mc = max(0, min(out_w - 1, int(cc - c_int0)))
        return int(mr), int(mc)

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
        comp_pts[cid].append((int(mr), int(mc), int(g[rr, cc])))

    comp_cols: dict[int, list[int]] = {}
    for cid in range(1, int(cn) + 1):
        comp_cols[cid] = sorted(set(map(int, g[clab == int(cid)].flatten())))

    deferred: dict[int, set[tuple[int, int]]] = defaultdict(set)
    path_maps: list[dict[tuple[int, int], int]] = []

    for cid in range(1, int(cn) + 1):
        pts = comp_pts.get(cid, [])
        if len(pts) < 2:
            continue
        cols = comp_cols.get(cid, [])
        if not cols:
            continue

        hist: dict[tuple[int, int], Counter] = defaultdict(Counter)
        for rr, cc, col in pts:
            hist[(int(rr), int(cc))][int(col)] += 1
        coords = list(hist.keys())
        if len(coords) < 2:
            continue

        best = None
        for i in range(len(coords)):
            for j in range(i + 1, len(coords)):
                d = abs(coords[i][0] - coords[j][0]) + abs(coords[i][1] - coords[j][1])
                if best is None or d > best[0]:
                    best = (d, coords[i], coords[j])
        if best is None:
            continue
        _dist, a, b = best

        if len(cols) == 1:
            colv = int(cols[0])
            if a[0] == b[0] or a[1] == b[1]:
                for p in coords:
                    deferred[colv].add((int(p[0]), int(p[1])))
                continue
            path = _choose_inside_path(a, b, out_h, out_w)
            path_maps.append(_path_color_map(path, (int(colv),)))
            continue

        if len(cols) == 2:
            ca, cb = map(int, cols)
            ha = int(hist[a].most_common(1)[0][0])
            hb = int(hist[b].most_common(1)[0][0])
            base_path = _choose_inside_path(a, b, out_h, out_w)
            chosen = None
            for path_pts in (base_path, list(reversed(base_path))):
                for start_color in (ca, cb):
                    if len(path_pts) % 2 == 1:
                        end_color = int(start_color)
                    else:
                        end_color = int(cb) if int(start_color) == int(ca) else int(ca)
                    if int(start_color) == int(ha) and int(end_color) == int(hb):
                        chosen = (path_pts, int(start_color))
                        break
                if chosen is not None:
                    break
            assert chosen is not None

            path_pts, start_color = chosen
            second = int(cb) if int(start_color) == int(ca) else int(ca)
            path_maps.append(_path_color_map(path_pts, (int(start_color), int(second))))

    for col, pts in deferred.items():
        items = list(pts)
        if not items:
            continue
        cells: list[tuple[int, int]] = []
        tops = [p for p in items if p[0] == 0]
        bots = [p for p in items if p[0] == out_h - 1]
        if tops and bots:
            for cc in sorted({c for _r, c in items}):
                rs = [r for r, c in items if c == cc]
                for rr in range(min(rs), max(rs) + 1):
                    cells.append((int(rr), int(cc)))
            path_maps.append(_path_color_map(cells, (int(col),)))
            continue

        lefts = [p for p in items if p[1] == 0]
        rights = [p for p in items if p[1] == out_w - 1]
        if lefts and rights:
            for rr in sorted({r for r, _c in items}):
                cs = [c for r, c in items if r == rr]
                for cc in range(min(cs), max(cs) + 1):
                    cells.append((int(rr), int(cc)))
            path_maps.append(_path_color_map(cells, (int(col),)))
            continue

        if len(items) == 1:
            path_maps.append(_path_color_map([(int(items[0][0]), int(items[0][1]))], (int(col),)))
            continue

        best = None
        for i in range(len(items)):
            for j in range(i + 1, len(items)):
                d = abs(items[i][0] - items[j][0]) + abs(items[i][1] - items[j][1])
                if best is None or d > best[0]:
                    best = (d, items[i], items[j])
        if best is None:
            continue
        _dist, p1, p2 = best
        path = _choose_inside_path(p1, p2, out_h, out_w)
        path_maps.append(_path_color_map(path, (int(col),)))

    return path_maps, int(out_h), int(out_w)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Rebuild many outer-frame traces as inside Manhattan paths, then recolor every crossing.

    Input structure:
    - One closed 1px rectangular frame surrounds a uniformly filled interior.
    - Four to six colored open-rectangle traces sit outside the frame: a
      same-color top/bottom U-pair always implies two full-height stripes;
      some samples also include a same-color left/right U-pair that implies two
      full-width stripes; the remaining traces imply one or two L-paths.

    Rule:
    - Crop to the frame interior and reconstruct each component's implied L-path
      or stripe in that smaller canvas.
    - Monochrome components stay solid; two-color L-components alternate with
      the phase fixed by the visible ring-touch endpoint colors.
    - After all paths are reconstructed, any interior cell visited by multiple
      paths is changed to the fixed collision color instead of keeping a path color.

    Key insight:
    - The outer geometry still determines the same endpoints and alternation as
      in the base task, but the stripe pair(s) and the active L-paths are
      arranged so overlaps are guaranteed and visible. The correct output comes
      from global collision detection, not from whichever path would be painted
      last.
    """
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))

    found = _find_framed_rect_uniform_fill(g, bg=int(bg))
    assert found is not None
    r0, c0, r1, c1, bc, fill = found
    path_maps, out_h, out_w = _extract_path_maps(g, r0, c0, r1, c1, bg, bc, fill)
    out, _collisions = _compose_with_collisions(out_h, out_w, int(fill), path_maps)
    return out


# --------------- generator (a6f40cea combo 22) ---------------

_BORDER_COLOR = 3
_BG_CHOICES = [0, 1, 4, 5, 6, 7, 8, 9]
_FILL_EQ_BG_PROB = 0.7
_INTERIOR_H_RANGE = (10, 15)
_INTERIOR_W_RANGE = (12, 18)
_GRID_SIZE_RANGE = (28, 40)
_FRAME_MARGIN = 6
_FRAME_MARGIN_FAR = 10
_DUAL_COMPONENT_CAP = 2
_DUAL_REQUIRED_PROB = 0.7
_GEN_RETRIES = 900


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
                out.append(p)
    return out


def _paint_path(grid: np.ndarray, path: list[tuple[int, int]], colors: tuple[int, ...]) -> None:
    if not path:
        return
    rs, cs = zip(*path)
    if len(colors) == 1:
        grid[rs, cs] = int(colors[0])
        return
    a, b = map(int, colors)
    vals = [int(a) if i % 2 == 0 else int(b) for i in range(len(path))]
    grid[rs, cs] = vals


def _can_place(grid: np.ndarray, coords: list[tuple[int, int]], bg: int, comp_mask: np.ndarray) -> bool:
    H, W = grid.shape
    cells = set(coords)
    for r, c in cells:
        if not (0 <= r < H and 0 <= c < W):
            return False
        if int(grid[r, c]) != int(bg):
            return False
        for nr, nc in neighbors4(r, c):
            if (nr, nc) not in cells and 0 <= nr < H and 0 <= nc < W and comp_mask[nr, nc]:
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


def _component_endpoints(kind: str, row: int, col: int, H: int, W: int) -> tuple[tuple[int, int], tuple[int, int]]:
    if kind == "tl":
        return (0, int(col)), (int(row), 0)
    if kind == "tr":
        return (0, int(col)), (int(row), int(W - 1))
    if kind == "bl":
        return (int(H - 1), int(col)), (int(row), 0)
    if kind == "br":
        return (int(H - 1), int(col)), (int(row), int(W - 1))
    raise ValueError(kind)


def _inner_path_from_kind(
    kind: str,
    row: int,
    col: int,
    H: int,
    W: int,
) -> list[tuple[int, int]]:
    """Return the intended inside L-path for one component."""
    p1, p2 = _component_endpoints(kind, row, col, H, W)
    return _choose_inside_path(p1, p2, H, W)


def _outer_path_from_corner(
    kind: str,
    row: int,
    col: int,
    r0: int,
    c0: int,
    r1: int,
    c1: int,
    outer_r: int,
    outer_c: int,
) -> list[tuple[int, int]]:
    """Connect the two ring touches through one chosen exterior corner."""
    if kind == "tl":
        ring_a = (int(r0 - 1), int(c0 + 1 + col))
        ring_b = (int(r0 + 1 + row), int(c0 - 1))
    elif kind == "tr":
        ring_a = (int(r0 - 1), int(c0 + 1 + col))
        ring_b = (int(r0 + 1 + row), int(c1 + 1))
    elif kind == "bl":
        ring_a = (int(r1 + 1), int(c0 + 1 + col))
        ring_b = (int(r0 + 1 + row), int(c0 - 1))
    elif kind == "br":
        ring_a = (int(r1 + 1), int(c0 + 1 + col))
        ring_b = (int(r0 + 1 + row), int(c1 + 1))
    else:
        raise ValueError(kind)

    return _join(
        _vpath(ring_a[0], outer_r, ring_a[1]),
        _hpath(outer_r, ring_a[1], outer_c),
        _vpath(outer_r, ring_b[0], outer_c),
        _hpath(ring_b[0], outer_c, ring_b[1]),
    )


def _l_specs_fixed(
    kind: str,
    row: int,
    col: int,
    H: int,
    W: int,
    r0: int,
    c0: int,
    r1: int,
    c1: int,
) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    """One L-path inside plus one matching outside open rectangle."""
    inner = _inner_path_from_kind(kind, row, col, H, W)
    if kind == "tl":
        outer_r = int(r0 - 4)
        outer_c = int(c0 - 2)
    elif kind == "tr":
        outer_r = int(r0 - 4)
        outer_c = int(c1 + 2)
    elif kind == "bl":
        outer_r = int(r1 + 4)
        outer_c = int(c0 - 2)
    elif kind == "br":
        outer_r = int(r1 + 4)
        outer_c = int(c1 + 2)
    else:
        raise ValueError(kind)
    outer = _outer_path_from_corner(kind, row, col, r0, c0, r1, c1, outer_r, outer_c)
    return inner, outer


def _u_pair_specs_vertical(
    c_a: int,
    c_b: int,
    H: int,
    W: int,
    r0: int,
    c0: int,
    r1: int,
    c1: int,
) -> tuple[list[list[tuple[int, int]]], list[tuple[int, int]], list[tuple[int, int]]]:
    """Return two full-height interior stripes and the matching top/bottom U traces."""
    top_r = int(r0 - 2)
    bot_r = int(r1 + 2)
    a_top, b_top = (int(r0 - 1), int(c0 + 1 + c_a)), (int(r0 - 1), int(c0 + 1 + c_b))
    a_bot, b_bot = (int(r1 + 1), int(c0 + 1 + c_a)), (int(r1 + 1), int(c0 + 1 + c_b))
    outer_top = _join(
        _vpath(a_top[0], top_r, a_top[1]),
        _hpath(top_r, a_top[1], b_top[1]),
        _vpath(top_r, b_top[0], b_top[1]),
    )
    outer_bot = _join(
        _vpath(a_bot[0], bot_r, a_bot[1]),
        _hpath(bot_r, a_bot[1], b_bot[1]),
        _vpath(bot_r, b_bot[0], b_bot[1]),
    )
    inner = [[(int(r), int(c_a)) for r in range(H)], [(int(r), int(c_b)) for r in range(H)]]
    return inner, outer_top, outer_bot


def _u_pair_specs_horizontal(
    r_a: int,
    r_b: int,
    H: int,
    W: int,
    r0: int,
    c0: int,
    r1: int,
    c1: int,
) -> tuple[list[list[tuple[int, int]]], list[tuple[int, int]], list[tuple[int, int]]]:
    """Return two full-width interior stripes and the matching left/right U traces."""
    left_c = int(c0 - 2)
    right_c = int(c1 + 2)
    a_left, b_left = (int(r0 + 1 + r_a), int(c0 - 1)), (int(r0 + 1 + r_b), int(c0 - 1))
    a_right, b_right = (int(r0 + 1 + r_a), int(c1 + 1)), (int(r0 + 1 + r_b), int(c1 + 1))
    outer_left = _join(
        _hpath(a_left[0], a_left[1], left_c),
        _vpath(a_left[0], b_left[0], left_c),
        _hpath(b_left[0], left_c, b_left[1]),
    )
    outer_right = _join(
        _hpath(a_right[0], a_right[1], right_c),
        _vpath(a_right[0], b_right[0], right_c),
        _hpath(b_right[0], right_c, b_right[1]),
    )
    inner = [[(int(r_a), int(c)) for c in range(W)], [(int(r_b), int(c)) for c in range(W)]]
    return inner, outer_left, outer_right


def _center_spaced_positions(count: int, size: int) -> list[int] | None:
    """Pick positions in 1..size-2 with global gap >= 2 and roughly centered spread."""
    if count == 0:
        return []
    slack = int(size - 3 - 2 * (count - 1))
    if slack < 0:
        return None
    center_offset = slack // 2
    wiggle = min(1, center_offset, slack - center_offset)
    offset = center_offset
    if wiggle > 0:
        offset = center_offset + random.randint(-wiggle, wiggle)
    return [int(1 + offset + 2 * i) for i in range(count)]


def _choose_component_kinds(n_components: int) -> list[str]:
    kinds = ["tl", "tr", "bl", "br"]
    if int(n_components) == 5:
        kinds.append(random.choice(["tl", "tr", "bl", "br"]))
    elif int(n_components) == 6:
        kinds.append(random.choice(["tl", "br"]))
        kinds.append(random.choice(["tr", "bl"]))
    random.shuffle(kinds)
    return kinds


def _allocate_positions(kinds: list[str], H: int, W: int) -> tuple[dict[int, int], dict[int, int]] | None:
    top_ids = [idx for idx, kind in enumerate(kinds) if kind in ("tl", "tr")]
    bot_ids = [idx for idx, kind in enumerate(kinds) if kind in ("bl", "br")]
    left_ids = [idx for idx, kind in enumerate(kinds) if kind in ("tl", "bl")]
    right_ids = [idx for idx, kind in enumerate(kinds) if kind in ("tr", "br")]

    top_cols = _center_spaced_positions(len(top_ids), W)
    bot_cols = _center_spaced_positions(len(bot_ids), W)
    left_rows = _center_spaced_positions(len(left_ids), H)
    right_rows = _center_spaced_positions(len(right_ids), H)
    if top_cols is None or bot_cols is None or left_rows is None or right_rows is None:
        return None

    row_map: dict[int, int] = {}
    col_map: dict[int, int] = {}

    def assign_cols(ids_a: list[int], ids_b: list[int], cols: list[int], first_gets_large: bool) -> None:
        cols_sorted = sorted(map(int, cols))
        random.shuffle(ids_a)
        random.shuffle(ids_b)
        if first_gets_large:
            chosen_a = list(reversed(cols_sorted[-len(ids_a) :]))
            chosen_b = cols_sorted[: len(ids_b)]
        else:
            chosen_a = cols_sorted[: len(ids_a)]
            chosen_b = list(reversed(cols_sorted[-len(ids_b) :]))
        for idx, col in zip(ids_a, chosen_a):
            col_map[int(idx)] = int(col)
        for idx, col in zip(ids_b, chosen_b):
            col_map[int(idx)] = int(col)

    def assign_rows(ids_a: list[int], ids_b: list[int], rows: list[int], first_gets_large: bool) -> None:
        rows_sorted = sorted(map(int, rows))
        random.shuffle(ids_a)
        random.shuffle(ids_b)
        if first_gets_large:
            chosen_a = list(reversed(rows_sorted[-len(ids_a) :]))
            chosen_b = rows_sorted[: len(ids_b)]
        else:
            chosen_a = rows_sorted[: len(ids_a)]
            chosen_b = list(reversed(rows_sorted[-len(ids_b) :]))
        for idx, row in zip(ids_a, chosen_a):
            row_map[int(idx)] = int(row)
        for idx, row in zip(ids_b, chosen_b):
            row_map[int(idx)] = int(row)

    assign_cols([idx for idx in top_ids if kinds[idx] == "tl"], [idx for idx in top_ids if kinds[idx] == "tr"], top_cols, True)
    assign_cols([idx for idx in bot_ids if kinds[idx] == "bl"], [idx for idx in bot_ids if kinds[idx] == "br"], bot_cols, True)
    assign_rows([idx for idx in left_ids if kinds[idx] == "tl"], [idx for idx in left_ids if kinds[idx] == "bl"], left_rows, True)
    assign_rows([idx for idx in right_ids if kinds[idx] == "tr"], [idx for idx in right_ids if kinds[idx] == "br"], right_rows, True)
    return row_map, col_map


def _choose_component_colors(palette: list[int], n_components: int) -> list[tuple[int, ...]] | None:
    max_dual = min(
        int(_DUAL_COMPONENT_CAP),
        int(n_components),
        max(0, len(palette) - int(n_components)),
    )
    if max_dual > 0 and random.random() < _DUAL_REQUIRED_PROB:
        dual_count = random.randint(1, max_dual)
    else:
        dual_count = random.randint(0, max_dual)
    dual_ids = set(random.sample(range(int(n_components)), k=int(dual_count)))

    need = int(n_components + dual_count)
    if need > len(palette):
        return None

    out: list[tuple[int, ...]] = []
    color_i = 0
    for idx in range(int(n_components)):
        if idx in dual_ids:
            out.append((int(palette[color_i]), int(palette[color_i + 1])))
            color_i += 2
        else:
            out.append((int(palette[color_i]),))
            color_i += 1
    return out


def _choose_l_kinds(n_l_components: int) -> list[str] | None:
    if int(n_l_components) == 1:
        kinds = [random.choice(["tl", "tr", "bl", "br"])]
    elif int(n_l_components) == 2:
        kinds = list(random.choice((("tl", "br"), ("tr", "bl"))))
    else:
        return None
    random.shuffle(kinds)
    return kinds


def _extreme_slots(count: int, size: int, *, toward_large: bool) -> list[int] | None:
    if int(count) < 0 or int(count) > 2:
        return None
    vals = [int(size - 2 - 2 * i) if toward_large else 1 + 2 * i for i in range(int(count))]
    if any(v <= 0 or v >= int(size - 1) for v in vals):
        return None
    return vals


def _assign_l_positions(kinds: list[str], H: int, W: int) -> tuple[dict[str, int], dict[str, int]] | None:
    top_kinds = [kind for kind in kinds if kind in ("tl", "tr")]
    bot_kinds = [kind for kind in kinds if kind in ("bl", "br")]
    left_kinds = [kind for kind in kinds if kind in ("tl", "bl")]
    right_kinds = [kind for kind in kinds if kind in ("tr", "br")]

    top_rows = _extreme_slots(len(top_kinds), H, toward_large=True)
    bot_rows = _extreme_slots(len(bot_kinds), H, toward_large=False)
    left_cols = _extreme_slots(len(left_kinds), W, toward_large=True)
    right_cols = _extreme_slots(len(right_kinds), W, toward_large=False)
    if top_rows is None or bot_rows is None or left_cols is None or right_cols is None:
        return None

    row_map: dict[str, int] = {}
    col_map: dict[str, int] = {}
    random.shuffle(top_rows)
    random.shuffle(bot_rows)
    random.shuffle(left_cols)
    random.shuffle(right_cols)

    for kind, row in zip(top_kinds, top_rows):
        row_map[str(kind)] = int(row)
    for kind, row in zip(bot_kinds, bot_rows):
        row_map[str(kind)] = int(row)
    for kind, col in zip(left_kinds, left_cols):
        col_map[str(kind)] = int(col)
    for kind, col in zip(right_kinds, right_cols):
        col_map[str(kind)] = int(col)
    return row_map, col_map


def _choose_stripe_columns(col_map: dict[str, int]) -> tuple[int, int] | None:
    left_cols = [int(col) for kind, col in col_map.items() if kind in ("tl", "bl")]
    right_cols = [int(col) for kind, col in col_map.items() if kind in ("tr", "br")]
    if not left_cols or not right_cols:
        return None

    lo = int(max(right_cols) + 1)
    hi = int(min(left_cols) - 1)
    candidates = list(range(lo, hi + 1))
    if len(candidates) < 2:
        return None
    picks = sorted(random.sample(candidates, k=2))
    return int(picks[0]), int(picks[1])


def _choose_axis_pair(candidates: list[int]) -> tuple[int, int] | None:
    uniq = sorted({int(v) for v in candidates})
    if len(uniq) < 2:
        return None
    picks = sorted(random.sample(uniq, k=2))
    return int(picks[0]), int(picks[1])


def _choose_horizontal_rows(col_kind: str, row: int, H: int) -> tuple[int, int] | None:
    if col_kind in ("tl", "tr"):
        candidates = list(range(1, int(row)))
    else:
        candidates = list(range(int(row + 1), int(H - 1)))
    return _choose_axis_pair(candidates)


def _choose_vertical_cols(row_kind: str, col: int, W: int) -> tuple[int, int] | None:
    if row_kind in ("tl", "bl"):
        candidates = list(range(1, int(col)))
    else:
        candidates = list(range(int(col + 1), int(W - 1)))
    return _choose_axis_pair(candidates)


def _l_specs_far(
    kind: str,
    row: int,
    col: int,
    H: int,
    W: int,
    r0: int,
    c0: int,
    r1: int,
    c1: int,
) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    """One L-path inside plus a farther outside corridor that stays clear of U-pairs."""
    inner = _inner_path_from_kind(kind, row, col, H, W)
    outer_r = int(r0 - 4) if kind.startswith("t") else int(r1 + 4)
    outer_c = int(c0 - 4) if kind.endswith("l") else int(c1 + 4)
    outer = _outer_path_from_corner(kind, row, col, r0, c0, r1, c1, outer_r, outer_c)
    return inner, outer


def _path_plan_stats(
    path_specs: list[tuple[list[tuple[int, int]], tuple[int, ...]]],
) -> dict[str, int] | None:
    occ: Counter = Counter()
    for path, _colors in path_specs:
        if len(set(path)) != len(path):
            return None
        for cell in path:
            occ[(int(cell[0]), int(cell[1]))] += 1

    if not occ:
        return None

    collision_cells = sum(1 for v in occ.values() if v >= 2)
    collision_hits = sum(v for v in occ.values() if v >= 2)
    max_hits = max(occ.values())
    return {
        "collision_cells": int(collision_cells),
        "collision_hits": int(collision_hits),
        "max_hits": int(max_hits),
        "painted_cells": int(len(occ)),
    }


def _render_expected(
    H: int,
    W: int,
    fill: int,
    path_specs: list[tuple[list[tuple[int, int]], tuple[int, ...]]],
) -> tuple[np.ndarray, int]:
    path_maps = [_path_color_map(path, colors) for path, colors in path_specs]
    return _compose_with_collisions(int(H), int(W), int(fill), path_maps)


def _place_components_with_random_corridors(
    grid: np.ndarray,
    bg: int,
    components: list[dict[str, object]],
    r0: int,
    c0: int,
    r1: int,
    c1: int,
    base_comp_mask: np.ndarray | None = None,
) -> bool:
    """
    Try multiple random exterior corner choices until all components fit.

    The inside paths are fixed already; only the outer routing is randomized.
    This decouples the dense-collision interior design from the legality of the
    outside traces.
    """
    rows, cols = grid.shape
    for _outer_attempt in range(120):
        work = grid.copy()
        if base_comp_mask is None:
            comp_mask = np.zeros_like(work, dtype=bool)
        else:
            comp_mask = np.array(base_comp_mask, copy=True, dtype=bool)
        order = list(range(len(components)))
        random.shuffle(order)
        ok = True
        for idx in order:
            comp = components[idx]
            kind = str(comp["kind"])
            row = int(comp["row"])
            col = int(comp["col"])
            colors = tuple(map(int, comp["colors"]))
            placed = False
            for _choice_attempt in range(80):
                if kind.startswith("t"):
                    outer_r = random.randint(1, int(r0 - 2))
                else:
                    outer_r = random.randint(int(r1 + 2), int(rows - 2))
                if kind.endswith("l"):
                    outer_c = random.randint(1, int(c0 - 2))
                else:
                    outer_c = random.randint(int(c1 + 2), int(cols - 2))
                outer = _outer_path_from_corner(kind, row, col, r0, c0, r1, c1, outer_r, outer_c)
                if _place_component(work, comp_mask, outer, colors, bg):
                    placed = True
                    break
            if not placed:
                ok = False
                break
        if ok:
            grid[:, :] = work
            return True
    return False


def _gen_attempt() -> np.ndarray | None:
    bg = int(random.choice(_BG_CHOICES))
    bc = int(_BORDER_COLOR)
    fill_choices = [x for x in range(10) if x not in (bg, bc, _COLLISION_COLOR)]
    fill = int(bg if random.random() < _FILL_EQ_BG_PROB else random.choice(fill_choices))

    H = random.randint(*_INTERIOR_H_RANGE)
    W = random.randint(*_INTERIOR_W_RANGE)
    rows_min = max(_GRID_SIZE_RANGE[0], H + 2 + _FRAME_MARGIN + _FRAME_MARGIN_FAR)
    cols_min = max(_GRID_SIZE_RANGE[0], W + 2 + _FRAME_MARGIN + _FRAME_MARGIN_FAR)
    if rows_min > _GRID_SIZE_RANGE[1] or cols_min > _GRID_SIZE_RANGE[1]:
        return None
    rows = random.randint(rows_min, _GRID_SIZE_RANGE[1])
    cols = random.randint(cols_min, _GRID_SIZE_RANGE[1])
    r0 = random.randint(_FRAME_MARGIN, rows - (H + 2) - _FRAME_MARGIN_FAR)
    c0 = random.randint(_FRAME_MARGIN, cols - (W + 2) - _FRAME_MARGIN_FAR)
    r1, c1 = int(r0 + H + 1), int(c0 + W + 1)

    palette = [x for x in range(10) if x not in (bg, bc, fill, _COLLISION_COLOR)]
    random.shuffle(palette)
    n_components = random.choice([4, 5, 6])
    if int(n_components) == 4:
        n_l_components = 2
        use_horizontal_pair = False
    elif int(n_components) == 5:
        n_l_components = 1
        use_horizontal_pair = True
    else:
        n_l_components = 2
        use_horizontal_pair = True

    l_kinds = _choose_l_kinds(n_l_components)
    if l_kinds is None:
        return None
    pos_maps = _assign_l_positions(l_kinds, H, W)
    if pos_maps is None:
        return None
    row_map, col_map = pos_maps

    if int(n_l_components) == 2:
        stripe_cols = _choose_stripe_columns(col_map)
    else:
        only_kind = str(l_kinds[0])
        stripe_cols = _choose_vertical_cols(only_kind, int(col_map[only_kind]), W)
    if stripe_cols is None:
        return None
    c_left, c_right = stripe_cols

    row_pair: tuple[int, int] | None = None
    if use_horizontal_pair:
        if int(n_l_components) == 2:
            top_rows = [int(row_map[kind]) for kind in l_kinds if kind in ("tl", "tr")]
            bot_rows = [int(row_map[kind]) for kind in l_kinds if kind in ("bl", "br")]
            if not top_rows or not bot_rows:
                return None
            row_pair = _choose_axis_pair(list(range(int(max(bot_rows) + 1), int(min(top_rows)))))
        else:
            only_kind = str(l_kinds[0])
            row_pair = _choose_horizontal_rows(only_kind, int(row_map[only_kind]), H)
        if row_pair is None:
            return None

    u_color = int(palette[0])
    rem_palette = palette[1:]
    if use_horizontal_pair:
        if not rem_palette:
            return None
        h_color = int(rem_palette[0])
        l_palette = rem_palette[1:]
    else:
        h_color = None
        l_palette = rem_palette
    l_colors = _choose_component_colors(l_palette, len(l_kinds))
    if l_colors is None:
        return None

    grid = np.full((rows, cols), int(bg), dtype=int)
    grid[r0, c0 : c1 + 1] = int(bc)
    grid[r1, c0 : c1 + 1] = int(bc)
    grid[r0 : r1 + 1, c0] = int(bc)
    grid[r0 : r1 + 1, c1] = int(bc)
    grid[r0 + 1 : r1, c0 + 1 : c1] = int(fill)

    u_inner, outer_top_u, outer_bot_u = _u_pair_specs_vertical(c_left, c_right, H, W, r0, c0, r1, c1)
    h_inner: list[list[tuple[int, int]]] = []
    outer_left_u: list[tuple[int, int]] = []
    outer_right_u: list[tuple[int, int]] = []
    if use_horizontal_pair:
        assert row_pair is not None
        assert h_color is not None
        h_inner, outer_left_u, outer_right_u = _u_pair_specs_horizontal(row_pair[0], row_pair[1], H, W, r0, c0, r1, c1)

    path_specs: list[tuple[list[tuple[int, int]], tuple[int, ...]]] = []
    path_specs.append((list(u_inner[0]), (int(u_color),)))
    path_specs.append((list(u_inner[1]), (int(u_color),)))
    outer_components: list[tuple[list[tuple[int, int]], tuple[int, ...]]] = [
        (list(outer_top_u), (int(u_color),)),
        (list(outer_bot_u), (int(u_color),)),
    ]
    if use_horizontal_pair:
        assert h_color is not None
        path_specs.append((list(h_inner[0]), (int(h_color),)))
        path_specs.append((list(h_inner[1]), (int(h_color),)))
        outer_components.append((list(outer_left_u), (int(h_color),)))
        outer_components.append((list(outer_right_u), (int(h_color),)))

    for kind, colors in zip(l_kinds, l_colors):
        row = int(row_map[str(kind)])
        col = int(col_map[str(kind)])
        inner, outer = _l_specs_far(str(kind), row, col, H, W, r0, c0, r1, c1)
        color_tuple = tuple(map(int, colors))
        path_specs.append((list(inner), color_tuple))
        outer_components.append((list(outer), color_tuple))

    stats = _path_plan_stats(path_specs)
    if stats is None:
        return None
    if int(stats["max_hits"]) > 2:
        return None
    if int(stats["collision_cells"]) < 6:
        return None

    expected, collision_count = _render_expected(int(H), int(W), int(fill), path_specs)
    if int(collision_count) != int(stats["collision_cells"]):
        return None
    if int(np.count_nonzero(expected == int(_COLLISION_COLOR))) != int(stats["collision_cells"]):
        return None

    comp_mask = np.zeros_like(grid, dtype=bool)
    for outer, colors in outer_components:
        if not _place_component(grid, comp_mask, list(outer), colors, bg):
            return None

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
