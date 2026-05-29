# Combo variant of a6f40cea (index 32)
# combo_spec: D4/2+D7/1+D14/2
# Name: 多路径竞争生长
#
# Design analysis for the fused variant
# -------------------------------------
# a) Concretization / behavior trace
# - Kept from the base task:
#   one closed 1px frame with uniform fill, plus colored open rectangular traces
#   outside the frame. Output is still the cropped frame interior.
# - Changed jointly by the combo:
#   each outside L-trace still defines one interior Manhattan L-path, but now
#   there are 4-6 such components and they do not paint atomically. Instead they
#   grow one cell per round from a visible boundary entry point. When a path's
#   next cell is already occupied by another path, it stops permanently.
# - Small example:
#   TL path wants to go down then left, TR path wants to go down then right,
#   BL path wants to go up then left, BR path wants to go left then down.
#   Original task: all four full Ls are painted independently.
#   Fused task: round 1 paints one boundary-adjacent cell for each path; later
#   rounds push them toward the center; at crossings, the earlier boundary entrant
#   claims the cell and the later one halts. Output therefore contains several
#   truncated prefixes instead of four full completions.
#
# b) Complexity calibration
# - The rule change is visible from geometry alone: many components compete inside
#   one shared interior, so collision truncation is frequent and cannot be ignored.
# - Freedom is kept controlled:
#   only L-components are used as primary objects; order comes only from clockwise
#   boundary position; colors are mostly monochrome with at most two alternating
#   traces. This keeps the task inferable from a few examples.
# - Dense interaction comes from quantity plus shared central lanes, not from many
#   unrelated object families or hidden tags.
#
# c) Hidden-assumption audit against the original main()
# - Original assumption: a component can always be completed as one full path.
#   Broken: the actual endpoint is now the first occupied next cell, not the far
#   ring touchpoint.
# - Original assumption: component iteration order is irrelevant.
#   Broken: growth order must be deterministic and visible, so paths are oriented
#   by the earlier boundary touchpoint and stepped in clockwise boundary order.
# - Original assumption: forbidding interior overlaps is harmless.
#   Broken: the generator must allow intended candidate-path overlap, otherwise the
#   combo degenerates back to the base task.
# - Original assumption: endpoint colors of two-color traces constrain both ends of
#   the fully painted path. Adjusted: they still constrain the intended full path,
#   but only the surviving prefix is ultimately drawn.
#
# d) Placement legality
# - Outer components remain pairwise non-overlapping and non-4-touching.
# - Duplicate corner families use separate offset layers outside the frame.
# - Side touch positions are spaced on each frame side, so components remain
#   recoverable as separate 4-connected traces.
# - Candidate-path overlap is allowed only inside the hidden cropped region, where
#   the competitive growth rule is meant to act.
#
# e) Visual observability
# - The ordering signal is human-visible: earlier boundary positions around the
#   frame enter first.
# - The collision rule is also human-visible: several paths visibly stop flush
#   against previously grown structure instead of reaching their second side.
# - No rule depends on color id magnitude or other non-visual metadata.
#
# f) Visual cleanliness
# - The scene uses one frame, 4-6 main traces, and at most one tiny distractor.
# - The generator biases all main paths through a central band so competition is
#   obvious, while the outside remains sparse and separated.
#
# g) Color inference
# - Most components are monochrome; at most two use alternating pairs.
# - Alternation is still structurally anchored by the two ring endpoint colors,
#   and the solver reconstructs the intended full-path phase before simulating the
#   competitive truncation.
#
# Post-implementation assumption audit
# - Solver assumption: every primary component touches the frame ring in exactly two
#   adjacent-side positions and uses one or two colors. Generator guarantees this.
# - Solver assumption: each recovered intended path has a unique visible earlier
#   boundary endpoint. Generator guarantees distinct side-touch placements.
# - Solver assumption: blocked growth is decided solely by already painted interior
#   cells. Generator guarantees that all intended competition happens inside the
#   frame, not via hidden extra obstacles.

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
    """Return the largest 1px rectangular frame with uniform interior fill."""
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
            cell = (int(p[0]), int(p[1]))
            if not out or out[-1] != cell:
                out.append(cell)
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


def _color_at(colors: tuple[int, ...], idx: int) -> int:
    if len(colors) == 1:
        return int(colors[0])
    a, b = map(int, colors)
    return int(a if idx % 2 == 0 else b)


def _paint_path(grid: np.ndarray, path: list[tuple[int, int]], colors: tuple[int, ...]) -> None:
    for idx, (rr, cc) in enumerate(path):
        grid[int(rr), int(cc)] = _color_at(colors, idx)


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


def _clockwise_ring(r0: int, c0: int, r1: int, c1: int, h: int, w: int) -> list[tuple[int, int]]:
    ring: list[tuple[int, int]] = []
    for cc in range(c0, c1 + 1):
        rr = r0 - 1
        if 0 <= rr < h:
            ring.append((int(rr), int(cc)))
    for rr in range(r0, r1 + 1):
        cc = c1 + 1
        if 0 <= cc < w:
            ring.append((int(rr), int(cc)))
    for cc in range(c1, c0 - 1, -1):
        rr = r1 + 1
        if 0 <= rr < h:
            ring.append((int(rr), int(cc)))
    for rr in range(r1, r0 - 1, -1):
        cc = c0 - 1
        if 0 <= cc < w:
            ring.append((int(rr), int(cc)))
    return ring


def _map_to_out(rr: int, cc: int, r_int0: int, c_int0: int, out_h: int, out_w: int) -> tuple[int, int]:
    r = max(0, min(out_h - 1, int(rr - r_int0)))
    c = max(0, min(out_w - 1, int(cc - c_int0)))
    return int(r), int(c)


def _ring_side(rr: int, cc: int, r0: int, c0: int, r1: int, c1: int) -> str | None:
    if int(rr) == int(r0 - 1) and c0 <= int(cc) <= c1:
        return "top"
    if int(cc) == int(c1 + 1) and r0 <= int(rr) <= r1:
        return "right"
    if int(rr) == int(r1 + 1) and c0 <= int(cc) <= c1:
        return "bottom"
    if int(cc) == int(c0 - 1) and r0 <= int(rr) <= r1:
        return "left"
    return None


def _trace_component_path(
    cells: set[tuple[int, int]],
    start: tuple[int, int],
    end: tuple[int, int],
) -> list[tuple[int, int]] | None:
    nbrs: dict[tuple[int, int], list[tuple[int, int]]] = {}
    for rr, cc in cells:
        nbrs[(int(rr), int(cc))] = [
            (int(nr), int(nc))
            for nr, nc in neighbors4(int(rr), int(cc))
            if (int(nr), int(nc)) in cells
        ]
    if int(len(nbrs.get(start, []))) > 1 or int(len(nbrs.get(end, []))) > 1:
        return None
    for cell, cell_nbrs in nbrs.items():
        if cell not in (start, end) and int(len(cell_nbrs)) != 2:
            return None

    path = [start]
    prev: tuple[int, int] | None = None
    cur = start
    while cur != end:
        opts = [nxt for nxt in nbrs[cur] if nxt != prev]
        if len(opts) != 1:
            return None
        nxt = opts[0]
        path.append(nxt)
        prev, cur = cur, nxt
        if len(path) > len(cells):
            return None
    if len(path) != len(cells):
        return None
    return path


def _recover_component_colors(g: np.ndarray, outer_path: list[tuple[int, int]]) -> tuple[int, ...] | None:
    seq = [int(g[int(rr), int(cc)]) for rr, cc in outer_path]
    uniq = sorted(set(seq))
    if len(uniq) == 1:
        return (int(uniq[0]),)
    if len(uniq) != 2:
        return None
    start = int(seq[0])
    other = next((int(col) for col in seq[1:] if int(col) != start), None)
    if other is None:
        return None
    colors = (int(start), int(other))
    if any(int(seq[idx]) != _color_at(colors, idx) for idx in range(len(seq))):
        return None
    return colors


def _extract_candidates_from_grid(
    g: np.ndarray, r0: int, c0: int, r1: int, c1: int, bc: int, fill: int
) -> list[PaintCandidate]:
    bg = int(get_bg(g))
    h, w = g.shape
    out_h = int(r1 - r0 - 1)
    out_w = int(c1 - c0 - 1)
    r_int0 = int(r0 + 1)
    c_int0 = int(c0 + 1)

    ring = _clockwise_ring(r0, c0, r1, c1, h, w)
    candidates: list[PaintCandidate] = []
    comp_mask = (g != int(bg)) & (g != int(bc)) & (g != int(fill))
    lab, n = label(comp_mask, structure=STRUCT4)
    for cid in range(1, int(n) + 1):
        cells = {
            (int(rr), int(cc))
            for rr, cc in np.argwhere(lab == int(cid))
        }
        hits: list[tuple[int, tuple[int, int], str, tuple[int, int]]] = []
        for order_key, (rr, cc) in enumerate(ring):
            if int(lab[int(rr), int(cc)]) != int(cid):
                continue
            side = _ring_side(int(rr), int(cc), r0, c0, r1, c1)
            if side is None:
                continue
            mapped = _map_to_out(int(rr), int(cc), r_int0, c_int0, out_h, out_w)
            hits.append((int(order_key), (int(rr), int(cc)), side, mapped))
        if len(hits) < 2:
            continue

        side_hits: dict[str, tuple[int, tuple[int, int], tuple[int, int]]] = {}
        for order_key, ring_cell, side, mapped in hits:
            prev = side_hits.get(side)
            if prev is None or int(order_key) < int(prev[0]):
                side_hits[side] = (int(order_key), ring_cell, mapped)
        sides = set(side_hits)
        if sides == {"top", "left"}:
            start_hit = side_hits["top"]
            end_hit = side_hits["left"]
        elif sides == {"top", "right"}:
            start_hit = side_hits["top"]
            end_hit = side_hits["right"]
        elif sides == {"bottom", "left"}:
            start_hit = side_hits["bottom"]
            end_hit = side_hits["left"]
        elif sides == {"bottom", "right"}:
            start_hit = side_hits["bottom"]
            end_hit = side_hits["right"]
        else:
            continue

        p1 = tuple(start_hit[2])
        p2 = tuple(end_hit[2])
        ordered_path, _start_pt, _end_pt, start_key = _orient_path_by_boundary(
            _choose_inside_path(p1, p2, out_h, out_w), out_h, out_w
        )
        outer_start = tuple(start_hit[1]) if tuple(ordered_path[0]) == p1 else tuple(end_hit[1])
        outer_end = tuple(end_hit[1]) if tuple(ordered_path[0]) == p1 else tuple(start_hit[1])
        outer_path = _trace_component_path(cells, outer_start, outer_end)
        if outer_path is None:
            continue
        colors = _recover_component_colors(g, outer_path)
        if colors is None:
            continue
        tie_key = int(start_hit[0] if tuple(ordered_path[0]) == p1 else end_hit[0])
        candidates.append(
            PaintCandidate(
                path=ordered_path,
                colors=colors,
                start_key=int(start_key),
                tie_key=int(tie_key),
                tag=f"comp-{cid}",
            )
        )

    return candidates


def _simulate_growth(
    out_h: int, out_w: int, fill: int, candidates: list[PaintCandidate]
) -> tuple[np.ndarray, dict[str, int]]:
    out = np.full((out_h, out_w), int(fill), dtype=int)
    order = sorted(range(len(candidates)), key=lambda i: (candidates[i].start_key, candidates[i].tie_key))
    idxs = [0] * len(candidates)
    stopped = [False] * len(candidates)
    blocked = 0
    finished = 0
    painted = 0
    rounds = 0

    while True:
        advanced = False
        rounds += 1
        for i in order:
            cand = candidates[i]
            if stopped[i]:
                continue
            if idxs[i] >= len(cand.path):
                stopped[i] = True
                finished += 1
                continue
            rr, cc = cand.path[idxs[i]]
            if int(out[int(rr), int(cc)]) != int(fill):
                stopped[i] = True
                blocked += 1
                continue
            out[int(rr), int(cc)] = _color_at(cand.colors, idxs[i])
            idxs[i] += 1
            painted += 1
            advanced = True
            if idxs[i] >= len(cand.path):
                stopped[i] = True
                finished += 1
        if not advanced:
            break

    overlap_counter = Counter()
    for cand in candidates:
        for cell in cand.path:
            overlap_counter[tuple(cell)] += 1
    overlap_cells = sum(1 for v in overlap_counter.values() if v >= 2)
    overlap_events = sum(v - 1 for v in overlap_counter.values() if v >= 2)
    return out, {
        "components": int(len(candidates)),
        "blocked": int(blocked),
        "finished": int(finished),
        "rounds": int(rounds - 1),
        "painted": int(painted),
        "planned": int(sum(len(c.path) for c in candidates)),
        "overlap_cells": int(overlap_cells),
        "overlap_events": int(overlap_events),
    }


def _full_overlay(out_h: int, out_w: int, fill: int, candidates: list[PaintCandidate]) -> np.ndarray:
    out = np.full((out_h, out_w), int(fill), dtype=int)
    for cand in sorted(candidates, key=lambda c: (c.start_key, c.tie_key)):
        for idx, (rr, cc) in enumerate(cand.path):
            out[int(rr), int(cc)] = _color_at(cand.colors, idx)
    return out


def _measure_competition(candidates: list[PaintCandidate], out_h: int, out_w: int, fill: int) -> dict[str, int]:
    grown, metrics = _simulate_growth(out_h, out_w, int(fill), candidates)
    overlay = _full_overlay(out_h, out_w, int(fill), candidates)
    metrics = dict(metrics)
    metrics["lost_cells"] = int(metrics["planned"] - metrics["painted"])
    metrics["changed_vs_overlay"] = int(not np.array_equal(grown, overlay))
    return metrics


def main(grid: np.ndarray) -> np.ndarray:
    """
    Rebuild the cropped frame interior by letting many projected L-paths grow competitively.

    Input structure:
    - One complete 1px rectangular frame with a uniform fill sits in a larger background.
    - Around that frame are 4-6 colored exterior traces, each touching the ring next to the
      frame at two positions on adjacent sides.
    - A trace can be monochrome or alternate between two colors along its visible outer path.

    Rule:
    - Convert every ring-touching trace into its intended interior Manhattan L-path, oriented from
      the earlier visible boundary touchpoint in clockwise perimeter order.
    - Initialize the output as the frame interior filled with the frame fill color.
    - Advance all intended paths round by round: each surviving path tries to paint exactly one
      next cell per round; if that next cell is already occupied, that path stops permanently.
    - Return only the cropped interior after this multi-path competition.

    Key insight:
    The outside traces still encode the full paths, but the final output keeps only the prefixes
    that win the race for shared cells; the deciding priority is the visible boundary entry order,
    not any hidden color-based convention.
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
_FILL_EQ_BG_PROB = 0.7
_INTERIOR_H_RANGE = (12, 18)
_INTERIOR_W_RANGE = (12, 18)
_GRID_SIZE_RANGE = (72, 100)
_FRAME_MARGIN = 16
_FRAME_MARGIN_FAR = 34
_COMPONENT_COUNT_RANGE = (4, 6)
_OUTER_OFFSET_BASE = 3
_OUTER_OFFSET_STEP = 8
_MAX_DUAL_COMPONENTS = 1
_DUAL_COLOR_PROB = 0.35
_DISTRACTOR_COUNT_RANGE = (0, 0)
_DISTRACTOR_SIZE_RANGE = (1, 3)
_DISTRACTOR_TRIES = 40
_DISTRACTOR_CLEARANCE = 3
_GEN_RETRIES = 1200
_LAYOUT_RETRIES = 10
_LAYER_SEARCH = 5


def _can_place(grid: np.ndarray, coords: list[tuple[int, int]], bg: int, comp_mask: np.ndarray) -> bool:
    h, w = grid.shape
    cells = set(coords)
    for rr, cc in cells:
        if not (0 <= int(rr) < h and 0 <= int(cc) < w):
            return False
        if int(grid[int(rr), int(cc)]) != int(bg):
            return False
        for nr, nc in neighbors4(int(rr), int(cc)):
            nbr = (int(nr), int(nc))
            if nbr in cells:
                continue
            if 0 <= int(nr) < h and 0 <= int(nc) < w and bool(comp_mask[int(nr), int(nc)]):
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


def _center_spaced_positions(count: int, size: int) -> list[int] | None:
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


def _component_endpoints(kind: str, row: int, col: int, out_h: int, out_w: int) -> tuple[tuple[int, int], tuple[int, int]]:
    if kind == "tl":
        return (0, int(col)), (int(row), 0)
    if kind == "tr":
        return (0, int(col)), (int(row), int(out_w - 1))
    if kind == "bl":
        return (int(out_h - 1), int(col)), (int(row), 0)
    if kind == "br":
        return (int(out_h - 1), int(col)), (int(row), int(out_w - 1))
    raise ValueError(kind)


def _l_specs(
    kind: str,
    row: int,
    col: int,
    layer: int,
    out_h: int,
    out_w: int,
    r0: int,
    c0: int,
    r1: int,
    c1: int,
) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    top_bus = {"tl": 0, "tr": 1}
    bottom_bus = {"bl": 0, "br": 1}
    left_bus = {"tl": 0, "bl": 1}
    right_bus = {"tr": 0, "br": 1}
    if kind == "tl":
        ring_a = (int(r0 - 1), int(c0 + 1 + col))
        ring_b = (int(r0 + 1 + row), int(c0 - 1))
        bus_r = int(r0 - (_OUTER_OFFSET_BASE + 4 * top_bus[kind] + layer * _OUTER_OFFSET_STEP))
        bus_c = int(c0 - (6 + 4 * left_bus[kind] + layer * _OUTER_OFFSET_STEP))
        outer = _join(
            _vpath(ring_a[0], bus_r, ring_a[1]),
            _hpath(bus_r, ring_a[1], bus_c),
            _vpath(bus_r, ring_b[0], bus_c),
            _hpath(ring_b[0], bus_c, ring_b[1]),
        )
    elif kind == "tr":
        ring_a = (int(r0 - 1), int(c0 + 1 + col))
        ring_b = (int(r0 + 1 + row), int(c1 + 1))
        bus_r = int(r0 - (_OUTER_OFFSET_BASE + 4 * top_bus[kind] + layer * _OUTER_OFFSET_STEP))
        bus_c = int(c1 + (6 + 4 * right_bus[kind] + layer * _OUTER_OFFSET_STEP))
        outer = _join(
            _vpath(ring_a[0], bus_r, ring_a[1]),
            _hpath(bus_r, ring_a[1], bus_c),
            _vpath(bus_r, ring_b[0], bus_c),
            _hpath(ring_b[0], bus_c, ring_b[1]),
        )
    elif kind == "bl":
        ring_a = (int(r1 + 1), int(c0 + 1 + col))
        ring_b = (int(r0 + 1 + row), int(c0 - 1))
        bus_r = int(r1 + (_OUTER_OFFSET_BASE + 4 * bottom_bus[kind] + layer * _OUTER_OFFSET_STEP))
        bus_c = int(c0 - (6 + 4 * left_bus[kind] + layer * _OUTER_OFFSET_STEP))
        outer = _join(
            _vpath(ring_a[0], bus_r, ring_a[1]),
            _hpath(bus_r, ring_a[1], bus_c),
            _vpath(bus_r, ring_b[0], bus_c),
            _hpath(ring_b[0], bus_c, ring_b[1]),
        )
    elif kind == "br":
        ring_a = (int(r1 + 1), int(c0 + 1 + col))
        ring_b = (int(r0 + 1 + row), int(c1 + 1))
        bus_r = int(r1 + (_OUTER_OFFSET_BASE + 4 * bottom_bus[kind] + layer * _OUTER_OFFSET_STEP))
        bus_c = int(c1 + (6 + 4 * right_bus[kind] + layer * _OUTER_OFFSET_STEP))
        outer = _join(
            _vpath(ring_a[0], bus_r, ring_a[1]),
            _hpath(bus_r, ring_a[1], bus_c),
            _vpath(bus_r, ring_b[0], bus_c),
            _hpath(ring_b[0], bus_c, ring_b[1]),
        )
    else:
        raise ValueError(kind)

    p1, p2 = _component_endpoints(kind, row, col, out_h, out_w)
    inner = _choose_inside_path(p1, p2, out_h, out_w)
    return inner, outer


def _choose_alt_colors_for_paths(
    inner_path: list[tuple[int, int]],
    outer_path: list[tuple[int, int]],
    palette_pair: tuple[int, int],
) -> tuple[int, int] | None:
    a, b = map(int, palette_pair)
    for start_color in (a, b):
        other = int(b) if int(start_color) == int(a) else int(a)
        inner_end = int(start_color) if (len(inner_path) - 1) % 2 == 0 else int(other)
        outer_end = int(start_color) if (len(outer_path) - 1) % 2 == 0 else int(other)
        if int(inner_end) == int(outer_end):
            return int(start_color), int(other)
    return None


def _build_component_candidate(
    kind: str,
    row: int,
    col: int,
    layer: int,
    out_h: int,
    out_w: int,
    r0: int,
    c0: int,
    r1: int,
    c1: int,
    colors: tuple[int, ...],
    tie_key: int,
) -> tuple[PaintCandidate, list[tuple[int, int]], tuple[int, ...]] | None:
    inner, outer = _l_specs(kind, row, col, layer, out_h, out_w, r0, c0, r1, c1)
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
            tag=f"{kind}-{layer}",
        ),
        ordered_outer,
        tuple(map(int, colors)),
    )


def _choose_kind_multiset(n_components: int) -> list[str]:
    anchor, rival = random.choice(
        [("tl", "br"), ("br", "tl"), ("tr", "bl"), ("bl", "tr")]
    )
    if n_components == 4:
        anchor_count = 1 if random.random() < 0.7 else 2
    else:
        anchor_count = max(2, n_components // 2)
    anchor_count = min(anchor_count, n_components - 1)
    kinds = [anchor] * int(anchor_count) + [rival] * int(n_components - anchor_count)
    random.shuffle(kinds)
    return kinds


def _pick_component_count() -> int:
    lo, hi = _COMPONENT_COUNT_RANGE
    if (lo, hi) != (4, 6):
        return random.randint(lo, hi)
    return int(random.choices([4, 5, 6], weights=[5, 4, 3], k=1)[0])


def _assign_side_positions(kinds: list[str], out_h: int, out_w: int) -> tuple[dict[int, int], dict[int, int]] | None:
    top_ids = [idx for idx, kind in enumerate(kinds) if kind in ("tl", "tr")]
    bot_ids = [idx for idx, kind in enumerate(kinds) if kind in ("bl", "br")]
    left_ids = [idx for idx, kind in enumerate(kinds) if kind in ("tl", "bl")]
    right_ids = [idx for idx, kind in enumerate(kinds) if kind in ("tr", "br")]

    top_cols = _center_spaced_positions(len(top_ids), out_w)
    bot_cols = _center_spaced_positions(len(bot_ids), out_w)
    left_rows = _center_spaced_positions(len(left_ids), out_h)
    right_rows = _center_spaced_positions(len(right_ids), out_h)
    if top_cols is None or bot_cols is None or left_rows is None or right_rows is None:
        return None

    random.shuffle(top_ids)
    random.shuffle(bot_ids)
    random.shuffle(left_ids)
    random.shuffle(right_ids)
    top_map = {idx: int(top_cols[pos]) for pos, idx in enumerate(top_ids)}
    bot_map = {idx: int(bot_cols[pos]) for pos, idx in enumerate(bot_ids)}
    left_map = {idx: int(left_rows[pos]) for pos, idx in enumerate(left_ids)}
    right_map = {idx: int(right_rows[pos]) for pos, idx in enumerate(right_ids)}

    row_map: dict[int, int] = {}
    col_map: dict[int, int] = {}
    for idx, kind in enumerate(kinds):
        row_map[idx] = int(left_map[idx] if kind in ("tl", "bl") else right_map[idx])
        col_map[idx] = int(top_map[idx] if kind in ("tl", "tr") else bot_map[idx])
    return row_map, col_map


def _build_layout_candidates(
    out_h: int,
    out_w: int,
    rows: int,
    cols: int,
    r0: int,
    c0: int,
    r1: int,
    c1: int,
    palette: list[int],
) -> tuple[list[PaintCandidate], list[tuple[list[tuple[int, int]], tuple[int, ...]]]] | None:
    for _attempt in range(_LAYOUT_RETRIES):
        n_components = _pick_component_count()
        if len(palette) < n_components:
            return None
        kinds = _choose_kind_multiset(n_components)
        assigned = _assign_side_positions(kinds, out_h, out_w)
        if assigned is None:
            continue
        row_map, col_map = assigned
        if any(
            not (1 <= int(row_map[idx]) <= out_h - 2 and 1 <= int(col_map[idx]) <= out_w - 2)
            for idx in range(n_components)
        ):
            continue

        dual_count = 0
        max_dual = min(int(_MAX_DUAL_COMPONENTS), max(0, len(palette) - n_components))
        if max_dual > 0 and random.random() < _DUAL_COLOR_PROB:
            dual_count = random.randint(1, max_dual)
        dual_ids = set(random.sample(range(n_components), dual_count)) if dual_count > 0 else set()

        temp_grid = np.zeros((rows, cols), dtype=int)
        temp_mask = np.zeros((rows, cols), dtype=bool)
        color_i = 0
        layer_floor: dict[str, int] = defaultdict(int)
        candidates: list[PaintCandidate] = []
        outer_payloads: list[tuple[list[tuple[int, int]], tuple[int, ...]]] = []
        ok = True
        for idx, kind in enumerate(kinds):
            color_need = 2 if idx in dual_ids else 1
            if color_i + color_need > len(palette):
                ok = False
                break
            raw_colors = tuple(int(palette[color_i + off]) for off in range(color_need))
            color_i += color_need

            placed = False
            for layer in range(int(layer_floor[kind]), int(layer_floor[kind]) + _LAYER_SEARCH):
                built = _build_component_candidate(
                    kind=kind,
                    row=int(row_map[idx]),
                    col=int(col_map[idx]),
                    layer=int(layer),
                    out_h=out_h,
                    out_w=out_w,
                    r0=r0,
                    c0=c0,
                    r1=r1,
                    c1=c1,
                    colors=raw_colors,
                    tie_key=idx,
                )
                if built is None:
                    continue
                cand, outer_path, placed_colors = built
                if not _place_component(temp_grid, temp_mask, outer_path, placed_colors, 0):
                    continue
                layer_floor[kind] = int(layer) + 1
                candidates.append(cand)
                outer_payloads.append((outer_path, placed_colors))
                placed = True
                break
            if not placed:
                ok = False
                break
        if ok and len(candidates) == n_components:
            return candidates, outer_payloads
    return None


def _add_distractor(
    grid: np.ndarray,
    comp_mask: np.ndarray,
    bg: int,
    frame: tuple[int, int, int, int],
    color: int,
) -> None:
    h, w = grid.shape
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
            pick = random.choice(opts)
            used.add((int(pick[0]), int(pick[1])))
            coords.append((int(pick[0]), int(pick[1])))
        if any(r0 - gap <= int(r) <= r1 + gap and c0 - gap <= int(c) <= c1 + gap for r, c in coords):
            continue
        if _place_component(grid, comp_mask, coords, (int(color),), int(bg)):
            return


def _competition_is_visible(metrics: dict[str, int]) -> bool:
    if int(metrics["components"]) < 4:
        return False
    if int(metrics["blocked"]) < 2:
        return False
    if int(metrics["overlap_cells"]) < 2:
        return False
    if int(metrics["lost_cells"]) < max(4, int(metrics["components"]) - 1):
        return False
    if int(metrics["changed_vs_overlay"]) != 1:
        return False
    return True


def _sample_metrics(grid: np.ndarray) -> dict[str, int]:
    found = _find_framed_rect_uniform_fill(grid, bg=int(get_bg(grid)))
    assert found is not None
    r0, c0, r1, c1, bc, fill = found
    out_h = int(r1 - r0 - 1)
    out_w = int(c1 - c0 - 1)
    candidates = _extract_candidates_from_grid(grid, r0, c0, r1, c1, bc, fill)
    return _measure_competition(candidates, out_h, out_w, int(fill))


def _gen_attempt() -> np.ndarray | None:
    bg = int(random.choice(_BG_CHOICES))
    bc = int(_BORDER_COLOR)
    fill_choices = [x for x in range(10) if x not in (bg, bc)]
    fill = int(bg if random.random() < _FILL_EQ_BG_PROB else random.choice(fill_choices))

    out_h = random.randint(*_INTERIOR_H_RANGE)
    out_w = random.randint(*_INTERIOR_W_RANGE)
    rows_min = max(_GRID_SIZE_RANGE[0], out_h + 2 + _FRAME_MARGIN + _FRAME_MARGIN_FAR)
    cols_min = max(_GRID_SIZE_RANGE[0], out_w + 2 + _FRAME_MARGIN + _FRAME_MARGIN_FAR)
    if rows_min > _GRID_SIZE_RANGE[1] or cols_min > _GRID_SIZE_RANGE[1]:
        return None
    rows = random.randint(rows_min, _GRID_SIZE_RANGE[1])
    cols = random.randint(cols_min, _GRID_SIZE_RANGE[1])
    r0 = random.randint(_FRAME_MARGIN, rows - (out_h + 2) - _FRAME_MARGIN_FAR)
    c0 = random.randint(_FRAME_MARGIN, cols - (out_w + 2) - _FRAME_MARGIN_FAR)
    r1, c1 = int(r0 + out_h + 1), int(c0 + out_w + 1)

    palette = [x for x in range(10) if x not in (bg, bc, fill)]
    random.shuffle(palette)
    built = _build_layout_candidates(out_h, out_w, rows, cols, r0, c0, r1, c1, palette)
    if built is None:
        return None
    candidates, outer_payloads = built
    metrics = _measure_competition(candidates, out_h, out_w, int(fill))
    if not _competition_is_visible(metrics):
        return None

    grid = np.full((rows, cols), int(bg), dtype=int)
    grid[r0, c0 : c1 + 1] = int(bc)
    grid[r1, c0 : c1 + 1] = int(bc)
    grid[r0 : r1 + 1, c0] = int(bc)
    grid[r0 : r1 + 1, c1] = int(bc)
    grid[r0 + 1 : r1, c0 + 1 : c1] = int(fill)

    comp_mask = np.zeros_like(grid, dtype=bool)
    for outer_path, colors in outer_payloads:
        if not _place_component(grid, comp_mask, outer_path, colors, int(bg)):
            return None

    used_colors = sum(len(colors) for _outer_path, colors in outer_payloads)
    extra_palette = palette[used_colors:]
    for extra_color in extra_palette[: random.randint(*_DISTRACTOR_COUNT_RANGE)]:
        _add_distractor(grid, comp_mask, int(bg), (r0, c0, r1, c1), int(extra_color))

    try:
        actual = main(grid)
    except Exception:
        return None
    expected, _sim_metrics = _simulate_growth(out_h, out_w, int(fill), candidates)
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(_GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid a6f40cea combo input")
