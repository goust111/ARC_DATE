# Variant of a6f40cea
# Dimension: D6 — 路径上的变换效果 (transform)
# Change: 路径用开放框架的颜色（或交替颜色）直接涂色到填充色背景上，无额外变换... → 路径碰撞重着色

"""
Design analysis for the D6 collision-recolor variant.

variant_spec:
  source_task_solver:
    status: not found in the provided files after repository search
    fallback: implement the collision-detection mechanic directly from the variant brief
  core_behavior:
    original: every reconstructed interior path writes its own color straight into the cropped output
    variant: geometry reconstruction stays the same, but any interior cell touched by 2+ distinct paths is recolored to a fixed collision color
  collision_color:
    choice: 2
    rationale:
      - kept constant across samples so humans can see the new effect immediately
      - excluded from bg/fill/path palettes so every red output cell is an actual collision, never a base fill/path ambiguity
  concrete visual forms:
    opposite_L_pair:
      input: two open rectangles on opposite corner-pairs of the frame
      output: two L-paths share the same interior elbow, so only that elbow becomes the collision color
    optional_third_L:
      input: on some samples, a third open rectangle is added in a non-intersecting quadrant
      output: the extra L is reconstructed normally while the shared-elbow pair still supplies the lone collision cell
  behavior_trace:
    example:
      base_task:
        - path A paints cell x with blue
        - path B later paints the same cell with yellow
        - final cell is whichever path happened to be painted last
      variant:
        - path A and path B are both reconstructed first
        - cell x is marked as visited twice
        - final cell becomes red regardless of paint order
    difference: overlap is no longer an accidental overwrite; it is the intended visible transform
  complexity_calibration:
    kept:
      - same frame detector
      - same ring-touch parsing
      - same Manhattan/L reconstruction and alternating-phase logic
      - same distractor model
    changed:
      - generator now guarantees at least one true interior collision
      - generator restricts collisions to clean single-point intersections; no shared segments or triple overlaps
  implicit_assumptions_review:
    solver_lines_affected:
      - every direct assignment into out[...] assumed paths never overlapped; this is false in the variant
      - deferred monochrome stripes also used direct writes; they must join the same collision pass
      - grouping deferred monochrome ring points by color is still valid only because generator keeps ring-component colors unique, except for the intentional U-pair
    generator_lines_affected:
      - used_inner boolean overlap rejection would erase the variant entirely
      - expected-output painting must merge paths through a collision-aware compositor instead of a plain overwrite painter
  placement_legality:
    rules:
      - outside components remain non-overlapping and non-adjacent under the original placement checks
      - inside overlaps are allowed only at the planned collision cell
      - every collision cell is strictly inside the cropped interior, never on the border
  observability:
    human_cues:
      - red exists only as the "crossing result"
      - each sample contains at least one collision cell, so the variant never silently falls back to the original behavior
      - layouts vary between 2-component and 3-component cases, but both share the same recolor rule
  cleanliness:
    safeguards:
      - only single-cell intersections are generated
      - no path shares a full segment with another path
      - distractors still stay away from the frame ring, so they do not compete with the collision cue
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


def _path_color_map(path: list[tuple[int, int]], colors: tuple[int, ...]) -> dict[tuple[int, int], int]:
    """Return the per-cell colors for a solid or alternating path."""
    out: dict[tuple[int, int], int] = {}
    if not path:
        return out
    if len(colors) == 1:
        col = int(colors[0])
        for rr, cc in path:
            out[(int(rr), int(cc))] = col
        return out

    ca, cb = map(int, colors)
    cur = int(ca)
    for rr, cc in path:
        out[(int(rr), int(cc))] = int(cur)
        cur = int(cb) if cur == int(ca) else int(ca)
    return out


def _compose_with_collisions(
    out_h: int,
    out_w: int,
    fill: int,
    path_maps: list[dict[tuple[int, int], int]],
) -> tuple[np.ndarray, int]:
    """
    Merge reconstructed paths into the cropped output.

    This is the key D6 change: the original task could paint directly because paths never met.
    Here we must defer painting until every path is known, then recolor multi-hit cells.
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


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reconstruct the interior Manhattan paths implied by the outer traces, then recolor crossings.

    Input structure:
    - One large 1-cell rectangular frame surrounds a uniformly filled interior.
    - Two or three colored open-rectangle traces sit outside the frame and touch its ring.

    Rule:
    - Crop to the frame interior and rebuild each implied stripe or L-path in that smaller canvas.
    - Monochrome traces stay solid; two-color traces alternate with phase fixed by the ring-touch endpoint colors.
    - After all paths are rebuilt, any interior cell visited by multiple distinct paths is changed to the collision color.

    Key insight:
    - The outer geometry still determines each path exactly as in the base task; the new inference step is to treat overlaps as events to detect globally, not as write-order accidents.
    """
    g = np.array(grid, dtype=int)
    H, W = g.shape
    bg = int(get_bg(g))

    found = _find_framed_rect_uniform_fill(g, bg=int(bg))
    assert not (found is None)
    r0, c0, r1, c1, bc, fill = found
    out_h = int(r1 - r0 - 1)
    out_w = int(c1 - c0 - 1)
    r_int0 = int(r0 + 1)
    c_int0 = int(c0 + 1)

    def map_to_out(rr: int, cc: int) -> tuple[int, int]:
        r = max(0, min(out_h - 1, int(rr - r_int0)))
        c = max(0, min(out_w - 1, int(cc - c_int0)))
        return int(r), int(c)

    def build_manhattan(
        p1: tuple[int, int], p2: tuple[int, int], corner_a: bool
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
        return (0 < rr < out_h - 1) and (0 < cc < out_w - 1)

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
    path_maps: list[dict[tuple[int, int], int]] = []

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
            path_a = build_manhattan(p1, p2, True)
            path_b = build_manhattan(p1, p2, False)
            chosen = path_b if (inside_corner(p1, p2, False) and not inside_corner(p1, p2, True)) else path_a
            path_maps.append(_path_color_map(chosen, (int(colv),)))
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
            ca, cb = map(int, cols)
            ha = int(hist[a].most_common(1)[0][0])
            hb = int(hist[b].most_common(1)[0][0])

            path_a = build_manhattan(a, b, True)
            path_b = build_manhattan(a, b, False)
            path = path_b if (inside_corner(a, b, False) and not inside_corner(a, b, True)) else path_a

            chosen = None  # (path_pts, start_color)
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
            second = int(cb) if int(start_color) == int(ca) else int(ca)
            path_maps.append(_path_color_map(path_pts, (int(start_color), int(second))))
            continue

    for col, pts in deferred.items():
        pts = list(pts)
        if not pts:
            continue
        cells: list[tuple[int, int]] = []
        tops = [p for p in pts if p[0] == 0]
        bots = [p for p in pts if p[0] == out_h - 1]
        if tops and bots:
            for cc in sorted({c for _r, c in pts}):
                rs = [r for r, c in pts if c == cc]
                for rr in range(min(rs), max(rs) + 1):
                    cells.append((int(rr), int(cc)))
            path_maps.append(_path_color_map(cells, (int(col),)))
            continue

        lefts = [p for p in pts if p[1] == 0]
        rights = [p for p in pts if p[1] == out_w - 1]
        if lefts and rights:
            for rr in sorted({r for r, _c in pts}):
                cs = [c for r, c in pts if r == rr]
                for cc in range(min(cs), max(cs) + 1):
                    cells.append((int(rr), int(cc)))
            path_maps.append(_path_color_map(cells, (int(col),)))
            continue

        if len(pts) == 1:
            rr, cc = pts[0]
            path_maps.append(_path_color_map([(int(rr), int(cc))], (int(col),)))
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
        path_maps.append(_path_color_map(chosen, (int(col),)))

    out, _collisions = _compose_with_collisions(out_h, out_w, int(fill), path_maps)
    return out


# --------------- generator (a6f40cea D6) ---------------

_BORDER_COLOR = 3
_BG_CHOICES = [0, 1, 4, 5, 6, 7, 8, 9]
_FILL_EQ_BG_PROB = 0.6
_INTERIOR_H_RANGE = (4, 13)
_INTERIOR_W_RANGE = (7, 14)
_GRID_SIZE_RANGE = (20, 30)
_GRID_INTERIOR_PAD = 10
_FRAME_MARGIN = 4
_FRAME_MARGIN_FAR = 6
_OUTER_OFFSET_RANGE = (2, 3)
_DUAL_COLOR_PROB = 0.45
_EXTRA_L_PROB = 0.4
_DISTRACTOR_COUNT_RANGE = (0, 2)
_DISTRACTOR_SIZE_RANGE = (1, 4)
_DISTRACTOR_TRIES = 40
_DISTRACTOR_CLEARANCE = 2
_GEN_RETRIES = 500


def _line(a: int, b: int) -> range:
    """Inclusive range from a to b."""
    step = 1 if b >= a else -1
    return range(a, b + step, step)


def _vpath(r0: int, r1: int, c: int) -> list[tuple[int, int]]:
    """Vertical segment from (r0, c) to (r1, c) inclusive."""
    return [(r, c) for r in _line(r0, r1)]


def _hpath(r: int, c0: int, c1: int) -> list[tuple[int, int]]:
    """Horizontal segment from (r, c0) to (c1) inclusive."""
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
        grid[rs, cs] = int(colors[0])
    else:
        a, b = map(int, colors)
        vals = [a if i % 2 == 0 else b for i in range(len(path))]
        grid[rs, cs] = vals


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


def _l_specs(
    kind: str,
    H: int,
    W: int,
    r0: int,
    c0: int,
    r1: int,
    c1: int,
    *,
    elbow: tuple[int, int] | None = None,
) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    """Build inner (interior L-path) and outer (open-rect) coords for one L-component."""
    is_top = kind[0] == "t"
    is_left = kind[1] == "l"

    if elbow is None:
        r = random.randint(1, H - 2)
        c = random.randint(1, W - 2)
    else:
        r, c = map(int, elbow)

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


def _u_pair_specs(
    orientation: str,
    H: int,
    W: int,
    r0: int,
    c0: int,
    r1: int,
    c1: int,
    *,
    positions: tuple[int, int] | None = None,
) -> tuple[list[list[tuple[int, int]]], list[tuple[int, int]], list[tuple[int, int]], tuple[int, int]]:
    """Build inner (two full-span stripes) and two outer U-shaped open-rects."""
    lo, hi = _OUTER_OFFSET_RANGE
    if orientation == "vertical":
        if positions is None:
            c_a, c_b = sorted(random.sample(range(1, W - 1), 2))
            if c_b - c_a < 2:
                c_b = min(W - 2, c_a + 2)
        else:
            c_a, c_b = map(int, positions)
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
        return inner, outer_a, outer_b, (int(c_a), int(c_b))

    if positions is None:
        r_a, r_b = sorted(random.sample(range(1, H - 1), 2))
        if r_b - r_a < 2:
            r_b = min(H - 2, r_a + 2)
    else:
        r_a, r_b = map(int, positions)
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
    return inner, outer_a, outer_b, (int(r_a), int(r_b))


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
        for _step in range(size - 1):
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
        if _place_component(grid, comp_mask, coords, (int(color),), bg):
            return


def _take_path_colors(palette: list[int], color_i: int) -> tuple[tuple[int, ...] | None, int]:
    """Allocate either one color or an alternating pair from the remaining palette."""
    if color_i >= len(palette):
        return None, color_i
    if color_i + 1 < len(palette) and random.random() < _DUAL_COLOR_PROB:
        return (int(palette[color_i]), int(palette[color_i + 1])), color_i + 2
    return (int(palette[color_i]),), color_i + 1


def _validate_path_plan(path_specs: list[tuple[list[tuple[int, int]], tuple[int, ...]]]) -> int:
    """Return the number of collision cells; reject segment overlaps or triple hits upstream."""
    occ: Counter = Counter()
    for path, _colors in path_specs:
        if len(set(path)) != len(path):
            return -1
        for coord in path:
            occ[(int(coord[0]), int(coord[1]))] += 1
    if not occ:
        return -1
    if max(occ.values()) > 2:
        return -1
    return sum(1 for n in occ.values() if n >= 2)


def _render_expected(
    H: int,
    W: int,
    fill: int,
    path_specs: list[tuple[list[tuple[int, int]], tuple[int, ...]]],
) -> tuple[np.ndarray, int]:
    """Create the expected cropped output using the same collision compositor as the solver."""
    path_maps = [_path_color_map(path, colors) for path, colors in path_specs]
    return _compose_with_collisions(int(H), int(W), int(fill), path_maps)


def _gen_attempt() -> np.ndarray | None:
    """Single attempt to generate a valid input grid. Returns None on failure."""
    bg = random.choice(_BG_CHOICES)
    bc = _BORDER_COLOR
    fill_choices = [x for x in range(10) if x not in (bg, bc, _COLLISION_COLOR)]
    fill = bg if random.random() < _FILL_EQ_BG_PROB else random.choice(fill_choices)

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

    comp_mask = np.zeros_like(grid, dtype=bool)
    palette = [x for x in range(10) if x not in (bg, bc, fill, _COLLISION_COLOR)]
    random.shuffle(palette)
    color_i = 0
    path_specs: list[tuple[list[tuple[int, int]], tuple[int, ...]]] = []

    shared_elbow = (random.randint(1, H - 2), random.randint(1, W - 2))
    kinds = random.choice([("tl", "br"), ("tr", "bl")])
    for kind in kinds:
        l_colors, color_i = _take_path_colors(palette, color_i)
        if l_colors is None:
            return None
        inner_l, outer_l = _l_specs(kind, H, W, r0, c0, r1, c1, elbow=shared_elbow)
        if not _place_component(grid, comp_mask, outer_l, l_colors, bg):
            return None
        path_specs.append((inner_l, l_colors))

    if random.random() < _EXTRA_L_PROB:
        sr, sc = shared_elbow
        extra_candidates: list[tuple[str, tuple[int, int]]] = []
        if kinds == ("tl", "br"):
            if sr > 1 and sc < W - 2:
                extra_candidates.append(("tr", (random.randint(1, sr - 1), random.randint(sc + 1, W - 2))))
            if sr < H - 2 and sc > 1:
                extra_candidates.append(("bl", (random.randint(sr + 1, H - 2), random.randint(1, sc - 1))))
        else:
            if sr > 1 and sc > 1:
                extra_candidates.append(("tl", (random.randint(1, sr - 1), random.randint(1, sc - 1))))
            if sr < H - 2 and sc < W - 2:
                extra_candidates.append(("br", (random.randint(sr + 1, H - 2), random.randint(sc + 1, W - 2))))

        if extra_candidates:
            extra_kind, extra_elbow = random.choice(extra_candidates)
            l_colors, color_i = _take_path_colors(palette, color_i)
            if l_colors is None:
                return None
            inner_l, outer_l = _l_specs(extra_kind, H, W, r0, c0, r1, c1, elbow=extra_elbow)
            if not _place_component(grid, comp_mask, outer_l, l_colors, bg):
                return None
            path_specs.append((inner_l, l_colors))

    collision_cells = _validate_path_plan(path_specs)
    if collision_cells != 1:
        return None

    expected, collision_count = _render_expected(H, W, int(fill), path_specs)
    if collision_count != 1:
        return None
    if int(np.count_nonzero(expected == int(_COLLISION_COLOR))) != 1:
        return None

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
    raise RuntimeError("failed to generate a valid a6f40cea D6 input")
