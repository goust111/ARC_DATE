# Combo variant of a6f40cea (index 31)
# combo_spec: D4/2+D6/0+D14/2
# Name: 竞争生长碰撞变色
#
# Design analysis
# ---------------
# a) Concretization / behavior trace
# - We keep the original object vocabulary: one closed 1px frame with uniform fill,
#   plus 2-3 open rectangular traces outside the frame that touch its ring.
# - Each usable outer component still implies one interior Manhattan L-path. The
#   change is entirely in how those candidate paths grow after projection.
# - Small trace with two diagonal Ls:
#   tl-path wants to grow from the top edge downward, then left;
#   br-path wants to grow from the right edge leftward, then downward.
#   In the base task both full paths are painted atomically.
#   In this combo task both paths advance one pixel per round. When both next cells
#   are the same interior pixel in the same round, that pixel becomes the collision
#   color and both paths stop there. If a later round hits an already-painted path,
#   the touched pixel is recolored to the collision color and the newcomer stops.
# - The visible difference from the base task is therefore twofold:
#   truncated growth prefixes replace full completions, and collision cells are
#   marked by a third color instead of being explained away by overwrite order.
#
# b) Complexity calibration
# - The generator always creates one readable core collision between a tl-component
#   and a br-component. This guarantees that the combo behavior is visible in every
#   sample.
# - Optional third L-components live in the opposite corner family and are forced to
#   avoid the core paths. This adds variety without creating ambiguous multi-collision
#   tangles or requiring humans to disentangle many simultaneous interactions.
# - Only one dedicated collision color is used, and it is excluded from bg/fill/path
#   palettes, so every occurrence of that color is a structural event.
#
# c) Hidden-assumption audit against the original solver
# - Original direct writes `out[r,c] = color` assumed whole paths can be committed
#   independently. That fails once path length depends on shared round-by-round state.
# - Original endpoint-to-endpoint reconstruction assumed the second touchpoint is the
#   actual destination. In this combo it only defines the intended L geometry; the
#   realized endpoint may be a collision much earlier.
# - Original overlap behavior was accidental last-write-wins. Here overlaps must be
#   detected explicitly and converted into the fixed collision color.
# - Original two-color logic painted a whole path after phase resolution. The phase
#   inference still comes from the ring endpoints, but only a prefix may survive.
#
# d) Placement legality
# - Outside components are still placed with the original no-overlap / no-4-adjacency
#   checks.
# - Hidden interior overlaps are allowed only through the planned core collision; the
#   optional third component is rejected if its intended path intersects the core.
# - The chosen collision cell is strictly inside the cropped frame interior so the
#   special color never sits on the frame boundary.
#
# e) Visual observability
# - Humans can infer the rule from 2-4 examples because the same visual cue repeats:
#   two outer traces aim inward, the interior paths stop at a shared special-colored
#   pixel, and no hidden color-id ordering is used.
# - Orientation is derived from visible clockwise boundary order only. The solver does
#   not rely on palette rank, component label order, or any invisible metadata.
#
# f) Visual cleanliness
# - The core collision geometry is limited to one perpendicular meeting point, not a
#   shared segment or a dense mesh.
# - At most one extra L and one tiny distractor are added. The cropped outputs remain
#   legible and the collision color stays the dominant novelty.
#
# g) Color-change inferability
# - The collision color appears only when two growth fronts compete for space or a
#   late front touches an already-painted path. Because the same color never appears
#   elsewhere, the mapping "path-path contact -> third color" is recoverable from a
#   few samples.
#
# New solver assumptions and generator guarantees
# - Every generated main component is an adjacent-side L, so the farthest ring-touch
#   pair does identify the intended endpoints.
# - Two-color components are generated with parity-compatible outer traces, so the
#   oriented start/end ring colors uniquely determine the alternating phase.
# - At least one synchronous collision cell is present, and the competitive-growth
#   result is rejected if it collapses to the base task's full overlay.

from __future__ import annotations

from collections import Counter, defaultdict
import random

import numpy as np
from scipy.ndimage import label

from arc_utils import STRUCT4, get_bg, neighbors4


_BORDER_COLOR = 3
_COLLISION_COLOR = 2

_BG_CHOICES = [0, 1, 4, 5, 6, 7, 8, 9]
_FILL_EQ_BG_PROB = 0.6
_INTERIOR_H_RANGE = (6, 13)
_INTERIOR_W_RANGE = (8, 14)
_GRID_SIZE_RANGE = (20, 30)
_GRID_INTERIOR_PAD = 10
_FRAME_MARGIN = 4
_FRAME_MARGIN_FAR = 6
_OUTER_OFFSET_RANGE = (2, 3)
_N_COMPONENTS_RANGE = (2, 3)
_DUAL_COLOR_PROB = 0.45
_RETRO_COLLISION_PROB = 0.5
_DISTRACTOR_COUNT_RANGE = (0, 1)
_DISTRACTOR_SIZE_RANGE = (1, 4)
_DISTRACTOR_TRIES = 40
_DISTRACTOR_CLEARANCE = 2
_GEN_RETRIES = 500


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
    return [(int(r), int(c)) for r in _line(r0, r1)]


def _hpath(r: int, c0: int, c1: int) -> list[tuple[int, int]]:
    return [(int(r), int(c)) for c in _line(c0, c1)]


def _join(*parts: list[tuple[int, int]]) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    for part in parts:
        for p in part:
            if not out or out[-1] != p:
                out.append((int(p[0]), int(p[1])))
    return out


def _build_manhattan(
    p1: tuple[int, int], p2: tuple[int, int], corner_a: bool
) -> list[tuple[int, int]]:
    (ra, ca), (rb, cb) = p1, p2
    if ra == rb:
        return [(int(ra), int(c)) for c in _line(ca, cb)]
    if ca == cb:
        return [(int(r), int(ca)) for r in _line(ra, rb)]
    if corner_a:
        return _join(_vpath(ra, rb, ca), _hpath(rb, ca, cb))
    return _join(_hpath(ra, ca, cb), _vpath(ra, rb, cb))


def _inside_corner(
    p1: tuple[int, int], p2: tuple[int, int], corner_a: bool, out_h: int, out_w: int
) -> bool:
    (ra, ca), (rb, cb) = p1, p2
    if ra == rb or ca == cb:
        return True
    rr, cc = (rb, ca) if corner_a else (ra, cb)
    return 0 < int(rr) < out_h - 1 and 0 < int(cc) < out_w - 1


def _choose_inside_path(
    p1: tuple[int, int], p2: tuple[int, int], out_h: int, out_w: int
) -> list[tuple[int, int]]:
    path_a = _build_manhattan(p1, p2, True)
    path_b = _build_manhattan(p1, p2, False)
    if _inside_corner(p2, p1, False, out_h, out_w) and not _inside_corner(p2, p1, True, out_h, out_w):
        return path_b
    if _inside_corner(p1, p2, False, out_h, out_w) and not _inside_corner(p1, p2, True, out_h, out_w):
        return path_b
    return path_a


def _boundary_key(p: tuple[int, int], out_h: int, out_w: int) -> int:
    r, c = map(int, p)
    if r == 0:
        return int(c)
    if c == out_w - 1:
        return int((out_w - 1) + r)
    if r == out_h - 1:
        return int((out_w - 1) + (out_h - 1) + (out_w - 1 - c))
    assert c == 0
    return int(2 * (out_w - 1) + (out_h - 1) + (out_h - 1 - r))


def _orient_path_by_boundary(
    path: list[tuple[int, int]], out_h: int, out_w: int
) -> tuple[list[tuple[int, int]], tuple[int, int], tuple[int, int], int]:
    a = path[0]
    b = path[-1]
    ka = _boundary_key(a, out_h, out_w)
    kb = _boundary_key(b, out_h, out_w)
    if kb < ka:
        rev = list(reversed(path))
        return rev, rev[0], rev[-1], int(kb)
    return path, a, b, int(ka)


def _side_of_point(p: tuple[int, int], out_h: int, out_w: int) -> str:
    r, c = map(int, p)
    if r == 0:
        return "top"
    if c == out_w - 1:
        return "right"
    if r == out_h - 1:
        return "bottom"
    if c == 0:
        return "left"
    return "inner"


def _choose_farthest_pair(coords: list[tuple[int, int]]) -> tuple[tuple[int, int], tuple[int, int]] | None:
    best = None
    for i in range(len(coords)):
        for j in range(i + 1, len(coords)):
            a = coords[i]
            b = coords[j]
            d = abs(int(a[0]) - int(b[0])) + abs(int(a[1]) - int(b[1]))
            if best is None or d > best[0]:
                best = (int(d), a, b)
    if best is None:
        return None
    return (int(best[1][0]), int(best[1][1])), (int(best[2][0]), int(best[2][1]))


def _map_to_out(rr: int, cc: int, r_int0: int, c_int0: int, out_h: int, out_w: int) -> tuple[int, int]:
    r = max(0, min(out_h - 1, int(rr - r_int0)))
    c = max(0, min(out_w - 1, int(cc - c_int0)))
    return int(r), int(c)


class PaintCandidate:
    def __init__(
        self,
        path: list[tuple[int, int]],
        colors: tuple[int, ...],
        start_key: int,
        tie_key: int,
        tag: str,
    ) -> None:
        self.path = list(path)
        self.colors = tuple(map(int, colors))
        self.start_key = int(start_key)
        self.tie_key = int(tie_key)
        self.tag = str(tag)


def _color_at(colors: tuple[int, ...], idx: int) -> int:
    if len(colors) == 1:
        return int(colors[0])
    a, b = map(int, colors)
    return int(a if idx % 2 == 0 else b)


def _paint_path(grid: np.ndarray, path: list[tuple[int, int]], colors: tuple[int, ...]) -> None:
    if not path:
        return
    for idx, (rr, cc) in enumerate(path):
        grid[int(rr), int(cc)] = _color_at(tuple(map(int, colors)), int(idx))


def _extract_candidates_from_grid(
    g: np.ndarray,
    r0: int,
    c0: int,
    r1: int,
    c1: int,
    bc: int,
    fill: int,
) -> list[PaintCandidate]:
    bg = int(get_bg(g))
    h, w = map(int, g.shape)
    out_h = int(r1 - r0 - 1)
    out_w = int(c1 - c0 - 1)
    r_int0 = int(r0 + 1)
    c_int0 = int(c0 + 1)

    comp_mask = (g != int(bg)) & (g != int(fill)) & (g != int(bc))
    clab, cn = label(comp_mask, structure=STRUCT4)

    ring = set()
    for cc in range(c0, c1 + 1):
        for rr in (r0 - 1, r1 + 1):
            if 0 <= rr < h:
                ring.add((int(rr), int(cc)))
    for rr in range(r0, r1 + 1):
        for cc in (c0 - 1, c1 + 1):
            if 0 <= cc < w:
                ring.add((int(rr), int(cc)))

    comp_pts: dict[int, list[tuple[int, int, int]]] = defaultdict(list)
    for rr, cc in ring:
        cid = int(clab[int(rr), int(cc)])
        if cid == 0:
            continue
        mr, mc = _map_to_out(int(rr), int(cc), r_int0, c_int0, out_h, out_w)
        comp_pts[cid].append((int(mr), int(mc), int(g[int(rr), int(cc)])))

    comp_cols = {
        int(cid): sorted(set(map(int, g[clab == int(cid)].flatten()))) for cid in range(1, int(cn) + 1)
    }

    candidates: list[PaintCandidate] = []
    tie_key = 0
    for cid in sorted(comp_pts):
        pts = comp_pts[int(cid)]
        if len(pts) < 2:
            continue
        cols = comp_cols.get(int(cid), [])
        if len(cols) not in (1, 2):
            continue

        hist: dict[tuple[int, int], Counter] = defaultdict(Counter)
        for rr, cc, col in pts:
            hist[(int(rr), int(cc))][int(col)] += 1
        coords = list(hist.keys())
        pair = _choose_farthest_pair(coords)
        if pair is None:
            continue
        a, b = pair
        sa = _side_of_point(a, out_h, out_w)
        sb = _side_of_point(b, out_h, out_w)
        if sa == "inner" or sb == "inner":
            continue
        if not (
            (sa in ("top", "bottom") and sb in ("left", "right"))
            or (sb in ("top", "bottom") and sa in ("left", "right"))
        ):
            continue

        ordered_path, start_pt, end_pt, start_key = _orient_path_by_boundary(
            _choose_inside_path(a, b, out_h, out_w),
            out_h,
            out_w,
        )

        if len(cols) == 1:
            colors = (int(cols[0]),)
        else:
            ca, cb = map(int, cols)
            hs = int(hist[start_pt].most_common(1)[0][0])
            he = int(hist[end_pt].most_common(1)[0][0])
            colors = None
            for start_color in (ca, cb):
                other = int(cb if start_color == ca else ca)
                end_color = int(start_color if (len(ordered_path) - 1) % 2 == 0 else other)
                if int(start_color) == int(hs) and int(end_color) == int(he):
                    colors = (int(start_color), int(other))
                    break
            if colors is None:
                continue

        candidates.append(
            PaintCandidate(
                path=ordered_path,
                colors=colors,
                start_key=int(start_key),
                tie_key=int(tie_key),
                tag=f"cid-{int(cid)}",
            )
        )
        tie_key += 1

    return sorted(candidates, key=lambda cand: (cand.start_key, cand.tie_key, cand.tag))


def _simulate_growth(
    out_h: int,
    out_w: int,
    fill: int,
    candidates: list[PaintCandidate],
) -> tuple[np.ndarray, dict[str, int]]:
    out = np.full((int(out_h), int(out_w)), int(fill), dtype=int)
    order = sorted(range(len(candidates)), key=lambda i: (candidates[i].start_key, candidates[i].tie_key))
    idxs = [0] * len(candidates)
    active = [True] * len(candidates)
    rounds = 0
    stopped = 0
    finished = 0
    collision_cells = 0
    simultaneous_collision_cells = 0
    retro_collision_cells = 0

    while True:
        proposals: dict[tuple[int, int], list[int]] = defaultdict(list)
        for i in order:
            if not active[i]:
                continue
            cand = candidates[i]
            if idxs[i] >= len(cand.path):
                active[i] = False
                finished += 1
                continue
            rr, cc = cand.path[idxs[i]]
            proposals[(int(rr), int(cc))].append(int(i))

        if not proposals:
            break

        rounds += 1
        for cell in sorted(proposals):
            rr, cc = map(int, cell)
            cur = int(out[int(rr), int(cc)])
            hits = proposals[cell]

            if len(hits) >= 2:
                if cur != int(_COLLISION_COLOR):
                    collision_cells += 1
                    if cur == int(fill):
                        simultaneous_collision_cells += 1
                    else:
                        retro_collision_cells += 1
                out[int(rr), int(cc)] = int(_COLLISION_COLOR)
                for i in hits:
                    if active[i]:
                        active[i] = False
                        stopped += 1
                continue

            i = int(hits[0])
            cand = candidates[i]
            if cur == int(fill):
                out[int(rr), int(cc)] = _color_at(cand.colors, idxs[i])
                idxs[i] += 1
                if idxs[i] >= len(cand.path):
                    active[i] = False
                    finished += 1
            else:
                if cur != int(_COLLISION_COLOR):
                    out[int(rr), int(cc)] = int(_COLLISION_COLOR)
                    collision_cells += 1
                    retro_collision_cells += 1
                if active[i]:
                    active[i] = False
                    stopped += 1

    return out, {
        "rounds": int(rounds),
        "stopped": int(stopped),
        "finished": int(finished),
        "collision_cells": int(collision_cells),
        "simultaneous_collision_cells": int(simultaneous_collision_cells),
        "retro_collision_cells": int(retro_collision_cells),
        "candidates": int(len(candidates)),
    }


def _full_overlay(out_h: int, out_w: int, fill: int, candidates: list[PaintCandidate]) -> np.ndarray:
    out = np.full((int(out_h), int(out_w)), int(fill), dtype=int)
    for cand in sorted(candidates, key=lambda c: (c.start_key, c.tie_key)):
        for idx, (rr, cc) in enumerate(cand.path):
            out[int(rr), int(cc)] = _color_at(cand.colors, int(idx))
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reconstruct the frame interior by letting projected L-paths grow competitively and recolor collisions.

    Input structure:
    - A uniform background contains one complete 1px rectangular frame with a uniform fill.
    - Two or three colored open-rectangle traces sit outside the frame and touch the ring next to the border.
    - Each usable trace defines an adjacent-side Manhattan L inside the frame, with one or two path colors.

    Rule:
    - Crop to the frame interior and convert each outside trace into its ordered interior L-path.
    - Initialize the crop with the fill color, then advance every path one pixel per round.
    - If exactly one path proposes a fill-colored next cell, paint that cell with the path's color and let it continue next round.
    - If multiple paths propose the same next cell, or a path proposes a cell already occupied by another path, that contact cell becomes the collision color and the arriving path(s) stop.
    - Monochrome paths stay solid; two-color paths alternate from the boundary-anchored start color along the surviving prefix.

    Key insight:
    - The outer traces still determine the intended full L geometry, but the final interior is no longer a static completion problem: it is a shared growth simulation where contact events both truncate paths and create a distinct third-color boundary.
    """
    g = np.array(grid, dtype=int)
    found = _find_framed_rect_uniform_fill(g, bg=int(get_bg(g)))
    assert found is not None
    r0, c0, r1, c1, bc, fill = found
    out_h = int(r1 - r0 - 1)
    out_w = int(c1 - c0 - 1)
    candidates = _extract_candidates_from_grid(g, int(r0), int(c0), int(r1), int(c1), int(bc), int(fill))
    out, _metrics = _simulate_growth(out_h, out_w, int(fill), candidates)
    return out


def _can_place(grid: np.ndarray, coords: list[tuple[int, int]], bg: int, comp_mask: np.ndarray) -> bool:
    h, w = map(int, grid.shape)
    s = set(coords)
    for rr, cc in s:
        if not (0 <= int(rr) < h and 0 <= int(cc) < w):
            return False
        if int(grid[int(rr), int(cc)]) != int(bg):
            return False
        for nr, nc in neighbors4(int(rr), int(cc)):
            if (nr, nc) not in s and 0 <= int(nr) < h and 0 <= int(nc) < w and comp_mask[int(nr), int(nc)]:
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
    if not _can_place(grid, path, int(bg), comp_mask):
        return False
    _paint_path(grid, path, colors)
    rs, cs = zip(*path)
    comp_mask[rs, cs] = True
    return True


def _l_specs(
    kind: str,
    out_h: int,
    out_w: int,
    r0: int,
    c0: int,
    r1: int,
    c1: int,
    elbow_r: int,
    elbow_c: int,
) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    is_top = kind[0] == "t"
    is_left = kind[1] == "l"
    lo, hi = _OUTER_OFFSET_RANGE
    outer_r = (r0 - random.randint(lo, hi)) if is_top else (r1 + random.randint(lo, hi))
    outer_c = (c0 - random.randint(lo, hi)) if is_left else (c1 + random.randint(lo, hi))

    ring_a = (r0 - 1 if is_top else r1 + 1, c0 + 1 + elbow_c)
    ring_b = (r0 + 1 + elbow_r, c0 - 1 if is_left else c1 + 1)
    edge_r = 0 if is_top else out_h - 1
    edge_c = 0 if is_left else out_w - 1
    inner = _join(_vpath(edge_r, elbow_r, elbow_c), _hpath(elbow_r, elbow_c, edge_c))
    outer = _join(
        _vpath(ring_a[0], outer_r, ring_a[1]),
        _hpath(outer_r, ring_a[1], outer_c),
        _vpath(outer_r, ring_b[0], outer_c),
        _hpath(ring_b[0], outer_c, ring_b[1]),
    )
    return inner, outer


def _choose_alt_colors_for_paths(
    inner_path: list[tuple[int, int]],
    outer_path: list[tuple[int, int]],
    palette_pair: tuple[int, int],
) -> tuple[int, int] | None:
    a, b = map(int, palette_pair)
    for start_color in (a, b):
        other = int(b if start_color == a else a)
        inner_end = int(start_color if (len(inner_path) - 1) % 2 == 0 else other)
        outer_end = int(start_color if (len(outer_path) - 1) % 2 == 0 else other)
        if int(inner_end) == int(outer_end):
            return int(start_color), int(other)
    return None


def _build_component_candidate(
    kind: str,
    out_h: int,
    out_w: int,
    r0: int,
    c0: int,
    r1: int,
    c1: int,
    elbow_r: int,
    elbow_c: int,
    colors: tuple[int, ...],
    tie_key: int,
) -> tuple[PaintCandidate, list[tuple[int, int]], tuple[int, ...]] | None:
    inner, outer = _l_specs(kind, out_h, out_w, r0, c0, r1, c1, elbow_r, elbow_c)
    ordered_inner, _start, _end, start_key = _orient_path_by_boundary(inner, out_h, out_w)
    ordered_outer = outer if ordered_inner[0] == inner[0] else list(reversed(outer))

    placed_colors = tuple(map(int, colors))
    if len(placed_colors) == 2:
        chosen = _choose_alt_colors_for_paths(ordered_inner, ordered_outer, (int(placed_colors[0]), int(placed_colors[1])))
        if chosen is None:
            return None
        placed_colors = (int(chosen[0]), int(chosen[1]))

    return (
        PaintCandidate(
            path=ordered_inner,
            colors=placed_colors,
            start_key=int(start_key),
            tie_key=int(tie_key),
            tag=f"gen-{kind}-{tie_key}",
        ),
        ordered_outer,
        placed_colors,
    )


def _path_overlap(a: list[tuple[int, int]], b: list[tuple[int, int]]) -> int:
    return int(len(set(a) & set(b)))


def _core_pair_specs(
    out_h: int, out_w: int, mode: str
) -> tuple[list[tuple[str, int, int]], tuple[int, int], str] | None:
    max_hit_r = min(out_h - 3, out_w - 3)
    if max_hit_r < 1:
        return None

    if mode == "simultaneous":
        hit_r = random.randint(1, int(max_hit_r))
        hit_c = int(out_w - 1 - hit_r)
        if not (2 <= hit_c <= out_w - 2):
            return None
        if hit_r + 1 > out_h - 2:
            return None
        tl_r = random.randint(hit_r + 1, out_h - 2)
        br_c = random.randint(1, hit_c - 1)
        return [("tl", int(tl_r), int(hit_c)), ("br", int(hit_r), int(br_c))], (int(hit_r), int(hit_c)), mode

    if mode != "retro":
        return None

    for _ in range(80):
        hit_r = random.randint(1, out_h - 3)
        hit_c = random.randint(2, out_w - 2)
        d_tl = int(hit_r)
        d_br = int(out_w - 1 - hit_c)
        if d_tl == d_br:
            continue
        if min(out_h, out_w) >= 6 and abs(d_tl - d_br) < 2:
            continue
        tl_r = random.randint(hit_r + 1, out_h - 2)
        br_c = random.randint(1, hit_c - 1)
        return [("tl", int(tl_r), int(hit_c)), ("br", int(hit_r), int(br_c))], (int(hit_r), int(hit_c)), mode
    return None


def _safe_extra_elbow(kind: str, out_h: int, out_w: int) -> tuple[int, int]:
    if kind == "tr":
        return (
            random.randint(1, max(1, out_h // 3)),
            random.randint(max(1, (2 * out_w) // 3), out_w - 2),
        )
    return (
        random.randint(max(1, (2 * out_h) // 3), out_h - 2),
        random.randint(1, max(1, out_w // 3)),
    )


def _take_path_colors(palette: list[int], color_i: int) -> tuple[tuple[int, ...] | None, int]:
    if color_i >= len(palette):
        return None, color_i
    if color_i + 1 < len(palette) and random.random() < _DUAL_COLOR_PROB:
        return (int(palette[color_i]), int(palette[color_i + 1])), color_i + 2
    return (int(palette[color_i]),), color_i + 1


def _component_specs(
    out_h: int,
    out_w: int,
    r0: int,
    c0: int,
    r1: int,
    c1: int,
    palette: list[int],
) -> tuple[list[PaintCandidate], list[tuple[list[tuple[int, int]], tuple[int, ...]]], str] | None:
    core_mode = "retro" if random.random() < _RETRO_COLLISION_PROB else "simultaneous"
    core = _core_pair_specs(out_h, out_w, core_mode)
    if core is None:
        return None
    descs, collision_cell, core_mode = core

    core_paths = [
        _l_specs(kind, out_h, out_w, r0, c0, r1, c1, elbow_r, elbow_c)[0] for kind, elbow_r, elbow_c in descs
    ]

    target_components = random.randint(*_N_COMPONENTS_RANGE)
    while len(descs) < target_components:
        placed_extra = False
        for extra_kind in random.sample(["tr", "bl"], k=2):
            for _ in range(80):
                elbow_r, elbow_c = _safe_extra_elbow(extra_kind, out_h, out_w)
                inner_extra, _outer_extra = _l_specs(extra_kind, out_h, out_w, r0, c0, r1, c1, elbow_r, elbow_c)
                if collision_cell in inner_extra:
                    continue
                if all(_path_overlap(inner_extra, path) == 0 for path in core_paths):
                    descs.append((extra_kind, int(elbow_r), int(elbow_c)))
                    core_paths.append(inner_extra)
                    placed_extra = True
                    break
            if placed_extra:
                break
        if not placed_extra:
            return None

    candidates: list[PaintCandidate] = []
    outer_payloads: list[tuple[list[tuple[int, int]], tuple[int, ...]]] = []
    color_i = 0
    for tie_key, (kind, elbow_r, elbow_c) in enumerate(descs):
        raw_colors, color_i = _take_path_colors(palette, color_i)
        if raw_colors is None:
            return None
        built = _build_component_candidate(
            kind,
            out_h,
            out_w,
            r0,
            c0,
            r1,
            c1,
            int(elbow_r),
            int(elbow_c),
            raw_colors,
            int(tie_key),
        )
        if built is None:
            return None
        cand, outer_path, placed_colors = built
        candidates.append(cand)
        outer_payloads.append((outer_path, placed_colors))

    return candidates, outer_payloads, core_mode


def _add_distractor(
    grid: np.ndarray,
    comp_mask: np.ndarray,
    bg: int,
    frame: tuple[int, int, int, int],
    color: int,
) -> None:
    h, w = map(int, grid.shape)
    r0, c0, r1, c1 = frame
    gap = _DISTRACTOR_CLEARANCE
    for _ in range(_DISTRACTOR_TRIES):
        size = random.randint(*_DISTRACTOR_SIZE_RANGE)
        rr = random.randint(1, h - 2)
        cc = random.randint(1, w - 2)
        coords = [(int(rr), int(cc))]
        used = {(int(rr), int(cc))}
        for _step in range(size - 1):
            base = random.choice(coords)
            opts = [
                (int(nr), int(nc))
                for nr, nc in neighbors4(*base)
                if 1 <= int(nr) < h - 1 and 1 <= int(nc) < w - 1 and (int(nr), int(nc)) not in used
            ]
            if not opts:
                break
            nxt = random.choice(opts)
            used.add((int(nxt[0]), int(nxt[1])))
            coords.append((int(nxt[0]), int(nxt[1])))
        if any(r0 - gap <= int(r) <= r1 + gap and c0 - gap <= int(c) <= c1 + gap for r, c in coords):
            continue
        if _place_component(grid, comp_mask, coords, (int(color),), int(bg)):
            return


def _gen_attempt() -> np.ndarray | None:
    bg = random.choice(_BG_CHOICES)
    bc = _BORDER_COLOR
    fill_choices = [x for x in range(10) if x not in (bg, bc, _COLLISION_COLOR)]
    fill = bg if random.random() < _FILL_EQ_BG_PROB else random.choice(fill_choices)

    out_h = random.randint(*_INTERIOR_H_RANGE)
    out_w = random.randint(*_INTERIOR_W_RANGE)
    rows = random.randint(max(_GRID_SIZE_RANGE[0], out_h + _GRID_INTERIOR_PAD), _GRID_SIZE_RANGE[1])
    cols = random.randint(max(_GRID_SIZE_RANGE[0], out_w + _GRID_INTERIOR_PAD), _GRID_SIZE_RANGE[1])
    r0 = random.randint(_FRAME_MARGIN, rows - out_h - _FRAME_MARGIN_FAR)
    c0 = random.randint(_FRAME_MARGIN, cols - out_w - _FRAME_MARGIN_FAR)
    r1, c1 = int(r0 + out_h + 1), int(c0 + out_w + 1)

    grid = np.full((rows, cols), int(bg), dtype=int)
    grid[r0, c0 : c1 + 1] = int(bc)
    grid[r1, c0 : c1 + 1] = int(bc)
    grid[r0 : r1 + 1, c0] = int(bc)
    grid[r0 : r1 + 1, c1] = int(bc)
    grid[r0 + 1 : r1, c0 + 1 : c1] = int(fill)

    palette = [x for x in range(10) if x not in (bg, bc, fill, _COLLISION_COLOR)]
    random.shuffle(palette)
    built = _component_specs(out_h, out_w, r0, c0, r1, c1, palette)
    if built is None:
        return None
    candidates, outer_payloads, core_mode = built

    expected, metrics = _simulate_growth(out_h, out_w, int(fill), candidates)
    if metrics["collision_cells"] != 1:
        return None
    if core_mode == "simultaneous" and metrics["simultaneous_collision_cells"] != 1:
        return None
    if core_mode == "simultaneous" and metrics["retro_collision_cells"] != 0:
        return None
    if core_mode == "retro" and metrics["retro_collision_cells"] != 1:
        return None
    if core_mode == "retro" and metrics["simultaneous_collision_cells"] != 0:
        return None
    if metrics["stopped"] < 1:
        return None
    if np.array_equal(expected, _full_overlay(out_h, out_w, int(fill), candidates)):
        return None

    comp_mask = np.zeros_like(grid, dtype=bool)
    for outer_path, colors in outer_payloads:
        if not _place_component(grid, comp_mask, outer_path, colors, int(bg)):
            return None

    used_colors = sum(len(colors) for _path, colors in outer_payloads)
    for _ in range(random.randint(*_DISTRACTOR_COUNT_RANGE)):
        if used_colors >= len(palette):
            break
        _add_distractor(grid, comp_mask, int(bg), (r0, c0, r1, c1), int(palette[used_colors]))
        used_colors += 1

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
