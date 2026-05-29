# Combo variant of a6f40cea (index 10)
# combo_spec: D4/2+D14/2
# Name: 竞争生长路径
#
# Design analysis
# ---------------
# a) Concretization / behavior trace
# - The input still contains one closed 1px frame with uniform fill and 2-3 colored
#   open-rectangle components outside the frame. Each useful component touches the
#   ring around the frame at two places, exactly as in the base task family.
# - Base task behavior:
#   each component is projected inward and its whole Manhattan path is painted in one
#   shot, so the far touchpoint is always reached.
# - Fused behavior:
#   each component first yields a full *candidate* interior path, but candidates are
#   not committed atomically. All candidates advance one pixel per round from a
#   visible boundary start. If the next pixel is already occupied by another path,
#   that candidate stops forever. So D4 changes the stop condition from "reach the
#   other touchpoint" to "first collision", while D14 changes the drawing process
#   from one-shot painting to synchronous stepwise growth.
# - Small trace:
#   TL path starts on the top edge and wants to go down then left.
#   BR path starts on the right edge and wants to go left then down.
#   Their full Ls cross at one interior cell.
#   Original task: both whole Ls are drawn.
#   Fused task: in each round TL paints one more cell, BR paints one more cell.
#   TL reaches the crossing first, claims it, and BR later halts when that cell is
#   no longer fill-colored. The visible output is therefore a full TL prefix and a
#   truncated BR prefix.
#
# b) Complexity calibration
# - The generator always creates a real competitive collision, so the fusion is
#   visually different from the base task and from a no-collision D14 sample.
# - Candidate paths are restricted to a small family of corner-based L-shapes, with
#   an optional third L parked in a spare corner. This keeps the rule inferable from
#   2-4 examples instead of creating many unrelated collision layouts.
# - Solver support for monochrome same-side stripe pairs is preserved, but the
#   generator focuses on L-path competition because it makes both "collision stop"
#   and "round-based growth" easier to read.
#
# c) Hidden-assumption audit against the original main()
# - Original code assumed every recovered path could be painted completely. That no
#   longer holds because a later path can be truncated mid-way.
# - Original code assumed path order did not matter. Here order matters at the gran-
#   ularity of one step per round, so we need a visible, deterministic boundary order.
# - Original two-color logic assumed endpoint colors uniquely identify the final full
#   painted path. In the fused task the *candidate* path is still inferred from the
#   two endpoints, but only its surviving prefix is painted.
# - Original generator forbade any hidden interior overlap, which would erase the new
#   behavior. The fused generator instead allows candidate paths to intersect while
#   keeping only the external components non-overlapping.
#
# d) Placement legality
# - The frame size stays near the base ranges but uses slightly larger minimums so a
#   crossing pair and a visibly truncated loser can coexist.
# - Outer open-rectangle components still obey the original legality checks: they
#   stay on background cells, remain 4-disconnected from each other, and avoid the
#   frame itself.
# - Hidden interior candidate paths are allowed to overlap, but the generator only
#   accepts scenes where the step simulation is well-defined and produces at least
#   one nontrivial truncation.
#
# e) Visual observability
# - Start order is derived from the clockwise position of the earlier boundary
#   touchpoint: top row left->right, right side top->bottom, bottom row right->left,
#   left side bottom->top. This order is human-visible from the input geometry.
# - The competitive effect is visible in the output because one or more paths stop
#   flush against another path instead of reaching their second boundary side.
# - No hidden ordering by color id, component label, or scan order is used.
#
# f) Cleanliness
# - The generator limits itself to 2 or 3 main components and at most one distractor.
# - Colors are not reused across unrelated main components, avoiding ambiguous pairing.
# - Optional third paths are biased away from the main collision so the central event
#   remains legible.
#
# g) Color inference
# - Monochrome components remain solid-colored.
# - Two-color components alternate along their candidate path exactly as in the base
#   family, with endpoint ring colors fixing the phase of the full intended path.
# - Competitive stopping only shortens the painted prefix; it does not introduce any
#   extra recoloring rule that would be hard to infer from a few examples.

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


class PaintCandidate:
    def __init__(
        self,
        path: list[tuple[int, int]],
        colors: tuple[int, ...],
        start_key: int,
        tie_key: int,
        tag: str,
    ) -> None:
        self.path = path
        self.colors = colors
        self.start_key = int(start_key)
        self.tie_key = int(tie_key)
        self.tag = tag


def _color_at(colors: tuple[int, ...], idx: int) -> int:
    if len(colors) == 1:
        return int(colors[0])
    a, b = map(int, colors)
    return int(a if idx % 2 == 0 else b)


def _simulate_growth(
    out_h: int, out_w: int, fill: int, candidates: list[PaintCandidate]
) -> tuple[np.ndarray, dict[str, int]]:
    out = np.full((out_h, out_w), int(fill), dtype=int)
    order = sorted(range(len(candidates)), key=lambda i: (candidates[i].start_key, candidates[i].tie_key))
    idxs = [0] * len(candidates)
    stopped = [False] * len(candidates)
    blocked = 0
    painted = 0
    rounds = 0
    trunc_loss = 0

    while True:
        advanced = False
        rounds += 1
        for i in order:
            cand = candidates[i]
            if stopped[i] or idxs[i] >= len(cand.path):
                continue
            rr, cc = cand.path[idxs[i]]
            if int(out[int(rr), int(cc)]) != int(fill):
                stopped[i] = True
                blocked += 1
                trunc_loss += int(len(cand.path) - idxs[i])
                continue
            out[int(rr), int(cc)] = _color_at(cand.colors, idxs[i])
            idxs[i] += 1
            painted += 1
            advanced = True
        if not advanced:
            break

    finished = sum(1 for i, cand in enumerate(candidates) if idxs[i] == len(cand.path))
    return out, {
        "rounds": int(rounds - 1),
        "blocked": int(blocked),
        "finished": int(finished),
        "painted": int(painted),
        "planned": int(sum(len(c.path) for c in candidates)),
        "trunc_loss": int(trunc_loss),
    }


def _full_overlay(out_h: int, out_w: int, fill: int, candidates: list[PaintCandidate]) -> np.ndarray:
    out = np.full((out_h, out_w), int(fill), dtype=int)
    order = sorted(candidates, key=lambda cand: (cand.start_key, cand.tie_key))
    for cand in order:
        for idx, (rr, cc) in enumerate(cand.path):
            out[int(rr), int(cc)] = _color_at(cand.colors, idx)
    return out


def _map_to_out(rr: int, cc: int, r_int0: int, c_int0: int, out_h: int, out_w: int) -> tuple[int, int]:
    r = max(0, min(out_h - 1, int(rr - r_int0)))
    c = max(0, min(out_w - 1, int(cc - c_int0)))
    return int(r), int(c)


def _classify_line_half(
    coords: list[tuple[int, int]], out_h: int, out_w: int
) -> dict[str, object] | None:
    rows = sorted({int(r) for r, _c in coords})
    cols = sorted({int(c) for _r, c in coords})
    if len(rows) == 1 and rows[0] in (0, out_h - 1):
        return {
            "axis": "vertical",
            "side": "top" if rows[0] == 0 else "bottom",
            "positions": cols,
        }
    if len(cols) == 1 and cols[0] in (0, out_w - 1):
        return {
            "axis": "horizontal",
            "side": "left" if cols[0] == 0 else "right",
            "positions": rows,
        }
    return None


def _pair_line_candidates(
    line_halves: dict[int, list[dict[str, object]]],
    out_h: int,
    out_w: int,
    next_tie_key: int,
) -> tuple[list[PaintCandidate], int]:
    cands: list[PaintCandidate] = []
    tie = int(next_tie_key)
    for color, halves in line_halves.items():
        tops = [h for h in halves if h["axis"] == "vertical" and h["side"] == "top"]
        bots = [h for h in halves if h["axis"] == "vertical" and h["side"] == "bottom"]
        for top, bot in zip(tops, bots):
            for ca, cb in zip(top["positions"], bot["positions"]):
                cc = int(ca if ca == cb else round((int(ca) + int(cb)) / 2))
                path = [(int(r), int(cc)) for r in range(out_h)]
                cands.append(
                    PaintCandidate(
                        path=path,
                        colors=(int(color),),
                        start_key=_boundary_key(path[0], out_h, out_w),
                        tie_key=tie,
                        tag="u-vertical",
                    )
                )
                tie += 1

        lefts = [h for h in halves if h["axis"] == "horizontal" and h["side"] == "left"]
        rights = [h for h in halves if h["axis"] == "horizontal" and h["side"] == "right"]
        for left, right in zip(lefts, rights):
            for ra, rb in zip(left["positions"], right["positions"]):
                rr = int(ra if ra == rb else round((int(ra) + int(rb)) / 2))
                path = [(int(rr), int(c)) for c in range(out_w)]
                cands.append(
                    PaintCandidate(
                        path=path,
                        colors=(int(color),),
                        start_key=_boundary_key(path[0], out_h, out_w),
                        tie_key=tie,
                        tag="u-horizontal",
                    )
                )
                tie += 1
    return cands, tie


def _extract_candidates_from_grid(
    g: np.ndarray, r0: int, c0: int, r1: int, c1: int, bc: int, fill: int
) -> list[PaintCandidate]:
    bg = int(get_bg(g))
    h, w = g.shape
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
        mr, mc = _map_to_out(rr, cc, r_int0, c_int0, out_h, out_w)
        comp_pts[cid].append((int(mr), int(mc), int(g[int(rr), int(cc)])))

    comp_cols = {
        int(cid): sorted(set(map(int, g[clab == int(cid)].flatten()))) for cid in range(1, int(cn) + 1)
    }

    candidates: list[PaintCandidate] = []
    line_halves: dict[int, list[dict[str, object]]] = defaultdict(list)
    tie_key = 0

    for cid, pts in comp_pts.items():
        if len(pts) < 2:
            continue
        cols = comp_cols.get(int(cid), [])
        if not cols:
            continue

        hist: dict[tuple[int, int], Counter] = defaultdict(Counter)
        for rr, cc, col in pts:
            hist[(int(rr), int(cc))][int(col)] += 1
        coords = list(hist.keys())

        if len(cols) == 1:
            half = _classify_line_half(coords, out_h, out_w)
            if half is not None:
                line_halves[int(cols[0])].append(half)
                continue

        best = None
        for i in range(len(coords)):
            for j in range(i + 1, len(coords)):
                d = abs(coords[i][0] - coords[j][0]) + abs(coords[i][1] - coords[j][1])
                if best is None or d > best[0]:
                    best = (int(d), coords[i], coords[j])
        if best is None:
            continue
        _d, a, b = best
        ordered_path, start_pt, end_pt, start_key = _orient_path_by_boundary(
            _choose_inside_path(a, b, out_h, out_w), out_h, out_w
        )

        if len(cols) == 1:
            candidates.append(
                PaintCandidate(
                    path=ordered_path,
                    colors=(int(cols[0]),),
                    start_key=int(start_key),
                    tie_key=tie_key,
                    tag=f"l-mono-{cid}",
                )
            )
            tie_key += 1
            continue

        if len(cols) == 2:
            ca, cb = map(int, cols)
            hs = int(hist[start_pt].most_common(1)[0][0])
            he = int(hist[end_pt].most_common(1)[0][0])
            chosen_colors = None
            for start_color in (ca, cb):
                other = cb if start_color == ca else ca
                end_color = start_color if (len(ordered_path) - 1) % 2 == 0 else other
                if int(start_color) == int(hs) and int(end_color) == int(he):
                    chosen_colors = (int(start_color), int(other))
                    break
            if chosen_colors is None:
                continue
            candidates.append(
                PaintCandidate(
                    path=ordered_path,
                    colors=chosen_colors,
                    start_key=int(start_key),
                    tie_key=tie_key,
                    tag=f"l-alt-{cid}",
                )
            )
            tie_key += 1

    extra, _ = _pair_line_candidates(line_halves, out_h, out_w, tie_key)
    candidates.extend(extra)
    return candidates


def _growth_metrics_from_grid(grid: np.ndarray) -> dict[str, int]:
    g = np.array(grid, dtype=int)
    found = _find_framed_rect_uniform_fill(g, bg=int(get_bg(g)))
    assert found is not None
    r0, c0, r1, c1, bc, fill = found
    out_h = int(r1 - r0 - 1)
    out_w = int(c1 - c0 - 1)
    cands = _extract_candidates_from_grid(g, r0, c0, r1, c1, bc, fill)
    _out, metrics = _simulate_growth(out_h, out_w, int(fill), cands)
    return metrics


def main(grid: np.ndarray) -> np.ndarray:
    """
    Rebuild the frame interior by letting all projected paths grow competitively from the boundary.

    Input structure:
    - One complete 1px rectangular frame with a uniform fill inside.
    - Two or three colored open-rectangle components outside the frame, touching the ring next to it.
    - Components can be monochrome or 2-color alternating, matching the base task family.

    Rule:
    - Convert each useful outside component into the same interior Manhattan candidate path it would imply in the base task.
    - Orient every candidate by the visible clockwise-earlier boundary touchpoint.
    - Start from a fill-colored crop of the frame interior and advance every candidate one pixel per round.
    - If a candidate's next pixel is already occupied by another path, that candidate stops permanently.
    - Return the cropped interior after this synchronous competitive growth.

    Key insight:
    The outer geometry still tells you where each full intended path wants to go, but that far endpoint is only a
    target. The actual painted length is determined by competition: earlier visible entrants claim shared space first,
    and later ones are truncated exactly at their first collision.
    """
    g = np.array(grid, dtype=int)
    found = _find_framed_rect_uniform_fill(g, bg=int(get_bg(g)))
    assert found is not None
    r0, c0, r1, c1, bc, fill = found
    out_h = int(r1 - r0 - 1)
    out_w = int(c1 - c0 - 1)
    candidates = _extract_candidates_from_grid(g, r0, c0, r1, c1, bc, fill)
    out, _metrics = _simulate_growth(out_h, out_w, int(fill), candidates)
    return out


# --------------- generator ---------------

_BORDER_COLOR = 3
_BG_CHOICES = [0, 1, 2, 4, 5, 6, 7, 8, 9]
_FILL_EQ_BG_PROB = 0.6
_INTERIOR_H_RANGE = (6, 13)
_INTERIOR_W_RANGE = (8, 14)
_GRID_SIZE_RANGE = (20, 30)
_GRID_INTERIOR_PAD = 10
_FRAME_MARGIN = 4
_FRAME_MARGIN_FAR = 6
_OUTER_OFFSET_RANGE = (2, 3)
_N_COMPONENTS_RANGE = (2, 3)
_THREE_COMPONENT_PROB = 0.7
_DUAL_COLOR_PROB = 0.45
_DISTRACTOR_COUNT_RANGE = (0, 1)
_DISTRACTOR_SIZE_RANGE = (1, 4)
_DISTRACTOR_TRIES = 40
_DISTRACTOR_CLEARANCE = 2
_GEN_RETRIES = 500


def _paint_path(grid: np.ndarray, path: list[tuple[int, int]], colors: tuple[int, ...]) -> None:
    if not path:
        return
    for idx, (rr, cc) in enumerate(path):
        grid[int(rr), int(cc)] = _color_at(colors, idx)


def _can_place(grid: np.ndarray, coords: list[tuple[int, int]], bg: int, comp_mask: np.ndarray) -> bool:
    h, w = grid.shape
    s = set(coords)
    for rr, cc in s:
        if not (0 <= rr < h and 0 <= cc < w):
            return False
        if int(grid[int(rr), int(cc)]) != int(bg):
            return False
        for nr, nc in neighbors4(int(rr), int(cc)):
            if (nr, nc) not in s and 0 <= nr < h and 0 <= nc < w and comp_mask[int(nr), int(nc)]:
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
        other = b if start_color == a else a
        inner_end = start_color if (len(inner_path) - 1) % 2 == 0 else other
        outer_end = start_color if (len(outer_path) - 1) % 2 == 0 else other
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

    if len(colors) == 2:
        chosen = _choose_alt_colors_for_paths(ordered_inner, ordered_outer, (int(colors[0]), int(colors[1])))
        if chosen is None:
            return None
        colors = chosen

    return (
        PaintCandidate(
            path=ordered_inner,
            colors=tuple(map(int, colors)),
            start_key=int(start_key),
            tie_key=int(tie_key),
            tag=f"gen-{kind}",
        ),
        ordered_outer,
        tuple(map(int, colors)),
    )


def _core_pair_elbows(out_h: int, out_w: int) -> list[tuple[str, int, int]] | None:
    for _ in range(100):
        if random.random() < 0.5:
            tl_r = random.randint(max(2, out_h // 2), out_h - 2)
            tl_c = random.randint(2, out_w - 3)
            br_r = random.randint(1, min(out_h - 3, tl_r - 1))
            br_c = random.randint(1, min(out_w - 3, tl_c - 1))
            tl_idx = int(br_r)
            br_idx = int(out_w - 1 - tl_c)
            if tl_idx > br_idx:
                continue
            if tl_r - br_r < 1 or tl_c - br_c < 1:
                continue
            return [
                ("tl", int(tl_r), int(tl_c)),
                ("br", int(br_r), int(br_c)),
            ]

        tr_r = random.randint(max(2, out_h // 2), out_h - 2)
        tr_c = random.randint(1, out_w - 3)
        bl_r = random.randint(1, min(out_h - 3, tr_r - 1))
        bl_c = random.randint(tr_c + 1, out_w - 2)
        tr_idx = int(bl_r)
        bl_idx = int(tr_c)
        if tr_idx > bl_idx:
            continue
        if tr_r - bl_r < 1 or bl_c - tr_c < 1:
            continue
        return [
            ("tr", int(tr_r), int(tr_c)),
            ("bl", int(bl_r), int(bl_c)),
        ]
    return None


def _safe_extra_elbow(kind: str, out_h: int, out_w: int) -> tuple[int, int]:
    if kind == "tr":
        return random.randint(1, max(1, out_h // 3)), random.randint(max(1, (2 * out_w) // 3), out_w - 2)
    if kind == "bl":
        return random.randint(max(1, (2 * out_h) // 3), out_h - 2), random.randint(1, max(1, out_w // 3))
    if kind == "tl":
        return random.randint(1, max(1, out_h // 3)), random.randint(1, max(1, out_w // 3))
    return random.randint(max(1, (2 * out_h) // 3), out_h - 2), random.randint(max(1, (2 * out_w) // 3), out_w - 2)


def _path_overlap(a: list[tuple[int, int]], b: list[tuple[int, int]]) -> int:
    return int(len(set(a) & set(b)))


def _outer_path_fits(path: list[tuple[int, int]], occupied: set[tuple[int, int]]) -> bool:
    path_set = {(int(r), int(c)) for r, c in path}
    if len(path_set) != len(path):
        return False
    if path_set & occupied:
        return False
    for rr, cc in path_set:
        for nr, nc in neighbors4(int(rr), int(cc)):
            if (int(nr), int(nc)) in occupied and (int(nr), int(nc)) not in path_set:
                return False
    return True


def _competition_profile(candidates: list[PaintCandidate]) -> dict[str, int]:
    visits: dict[tuple[int, int], list[int]] = defaultdict(list)
    for cand in candidates:
        for idx, cell in enumerate(cand.path):
            visits[(int(cell[0]), int(cell[1]))].append(int(idx))

    shared = 0
    strict = 0
    ties = 0
    for idxs in visits.values():
        if len(idxs) < 2:
            continue
        shared += 1
        best = min(idxs)
        if sum(1 for idx in idxs if idx == best) > 1:
            ties += 1
        else:
            strict += 1
    return {
        "shared_cells": int(shared),
        "strict_cells": int(strict),
        "tie_cells": int(ties),
    }


def _component_specs(
    out_h: int,
    out_w: int,
    r0: int,
    c0: int,
    r1: int,
    c1: int,
    palette: list[int],
) -> tuple[list[PaintCandidate], list[tuple[list[tuple[int, int]], tuple[int, ...]]]] | None:
    core = _core_pair_elbows(out_h, out_w)
    if core is None:
        return None

    n_components = 3 if random.random() < _THREE_COMPONENT_PROB else 2

    candidates: list[PaintCandidate] = []
    outer_payloads: list[tuple[list[tuple[int, int]], tuple[int, ...]]] = []
    occupied_outer: set[tuple[int, int]] = set()
    color_i = 0

    def next_colors() -> tuple[int, ...] | None:
        nonlocal color_i
        if color_i >= len(palette):
            return None
        if random.random() < _DUAL_COLOR_PROB and color_i + 1 < len(palette):
            raw_colors: tuple[int, ...] = (int(palette[color_i]), int(palette[color_i + 1]))
            color_i += 2
        else:
            raw_colors = (int(palette[color_i]),)
            color_i += 1
        return raw_colors

    for tie_key, (kind, elbow_r, elbow_c) in enumerate(core):
        raw_colors = next_colors()
        if raw_colors is None:
            return None
        built = _build_component_candidate(
            kind, out_h, out_w, r0, c0, r1, c1, elbow_r, elbow_c, raw_colors, tie_key
        )
        if built is None:
            return None
        cand, outer_path, placed_colors = built
        if not _outer_path_fits(outer_path, occupied_outer):
            return None
        candidates.append(cand)
        outer_payloads.append((outer_path, placed_colors))
        occupied_outer.update((int(r), int(c)) for r, c in outer_path)

    profile = _competition_profile(candidates)
    if profile["strict_cells"] < 1 or profile["tie_cells"] > 0:
        return None

    if n_components == 3:
        raw_colors = next_colors()
        if raw_colors is None:
            return None
        remaining_kinds = [kind for kind in ("tl", "tr", "bl", "br") if kind not in {k for k, _, _ in core}]
        random.shuffle(remaining_kinds)
        existing_paths = [cand.path for cand in candidates]
        found_extra = False
        for extra_kind in remaining_kinds:
            for _ in range(120):
                er, ec = _safe_extra_elbow(extra_kind, out_h, out_w)
                built = _build_component_candidate(
                    extra_kind,
                    out_h,
                    out_w,
                    r0,
                    c0,
                    r1,
                    c1,
                    er,
                    ec,
                    raw_colors,
                    len(candidates),
                )
                if built is None:
                    continue
                cand, outer_path, placed_colors = built
                if any(_path_overlap(cand.path, path) > 1 for path in existing_paths):
                    continue
                if not _outer_path_fits(outer_path, occupied_outer):
                    continue
                profile = _competition_profile(candidates + [cand])
                if profile["tie_cells"] > 0 or profile["strict_cells"] < 1:
                    continue
                candidates.append(cand)
                outer_payloads.append((outer_path, placed_colors))
                occupied_outer.update((int(r), int(c)) for r, c in outer_path)
                found_extra = True
                break
            if found_extra:
                break
        if not found_extra:
            return None

    return candidates, outer_payloads


def _add_distractor(
    grid: np.ndarray, comp_mask: np.ndarray, bg: int, frame: tuple[int, int, int, int], color: int
) -> None:
    h, w = grid.shape
    r0, c0, r1, c1 = frame
    gap = _DISTRACTOR_CLEARANCE
    for _ in range(_DISTRACTOR_TRIES):
        size = random.randint(*_DISTRACTOR_SIZE_RANGE)
        rr = random.randint(1, h - 2)
        cc = random.randint(1, w - 2)
        coords = [(rr, cc)]
        used = {(rr, cc)}
        for _ in range(size - 1):
            base = random.choice(coords)
            opts = [
                (nr, nc)
                for nr, nc in neighbors4(*base)
                if 1 <= nr < h - 1 and 1 <= nc < w - 1 and (nr, nc) not in used
            ]
            if not opts:
                break
            nxt = random.choice(opts)
            used.add(nxt)
            coords.append(nxt)
        if any(r0 - gap <= r <= r1 + gap and c0 - gap <= c <= c1 + gap for r, c in coords):
            continue
        if _place_component(grid, comp_mask, coords, (int(color),), bg):
            return


def _gen_attempt() -> np.ndarray | None:
    bg = random.choice(_BG_CHOICES)
    bc = _BORDER_COLOR
    fill = bg if random.random() < _FILL_EQ_BG_PROB else random.choice([x for x in range(10) if x not in (bg, bc)])
    out_h = random.randint(*_INTERIOR_H_RANGE)
    out_w = random.randint(*_INTERIOR_W_RANGE)
    rows = random.randint(max(_GRID_SIZE_RANGE[0], out_h + _GRID_INTERIOR_PAD), _GRID_SIZE_RANGE[1])
    cols = random.randint(max(_GRID_SIZE_RANGE[0], out_w + _GRID_INTERIOR_PAD), _GRID_SIZE_RANGE[1])
    r0 = random.randint(_FRAME_MARGIN, rows - out_h - _FRAME_MARGIN_FAR)
    c0 = random.randint(_FRAME_MARGIN, cols - out_w - _FRAME_MARGIN_FAR)
    r1, c1 = r0 + out_h + 1, c0 + out_w + 1

    grid = np.full((rows, cols), int(bg), dtype=int)
    grid[r0, c0 : c1 + 1] = int(bc)
    grid[r1, c0 : c1 + 1] = int(bc)
    grid[r0 : r1 + 1, c0] = int(bc)
    grid[r0 : r1 + 1, c1] = int(bc)
    grid[r0 + 1 : r1, c0 + 1 : c1] = int(fill)

    palette = [x for x in range(10) if x not in (bg, bc, fill)]
    random.shuffle(palette)
    built = _component_specs(out_h, out_w, r0, c0, r1, c1, palette)
    if built is None:
        return None
    candidates, outer_payloads = built
    profile = _competition_profile(candidates)
    if profile["strict_cells"] < 1 or profile["tie_cells"] > 0:
        return None

    expected, metrics = _simulate_growth(out_h, out_w, int(fill), candidates)
    if metrics["blocked"] < 1:
        return None
    if metrics["trunc_loss"] < 2:
        return None
    if metrics["painted"] >= metrics["planned"]:
        return None
    if np.array_equal(expected, _full_overlay(out_h, out_w, int(fill), candidates)):
        return None

    comp_mask = np.zeros_like(grid, dtype=bool)
    for outer_path, colors in outer_payloads:
        if not _place_component(grid, comp_mask, outer_path, colors, int(bg)):
            return None

    color_i = sum(len(payload[1]) for payload in outer_payloads)
    for _ in range(random.randint(*_DISTRACTOR_COUNT_RANGE)):
        if color_i >= len(palette):
            break
        _add_distractor(grid, comp_mask, int(bg), (r0, c0, r1, c1), int(palette[color_i]))
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
