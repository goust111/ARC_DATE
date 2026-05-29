# Combo variant of a6f40cea (index 25)
# combo_spec: D4/2+D14/0
# Name: 链式碰撞路径
#
# Design analysis for this combo variant
# --------------------------------------
# a) Concretization / behavior trace
# - We keep the original vocabulary: one closed rectangular frame, colored components
#   outside the ring around that frame, and an output that is the cropped interior.
# - The fused change is:
#   1. The first two-touch L-component still yields one complete interior starter path.
#   2. Later one-touch components do not start from their own ring touches. Instead,
#      the endpoint reached by the previous segment becomes the next segment's start.
#   3. Each later segment aims toward that link's projected touchpoint, but it does
#      not necessarily reach it. It paints only until the next step would enter an
#      already painted older path cell; if no such collision happens, it may end on
#      the boundary target.
# - Concrete trace in a left-ending case:
#   starter: top touch -> interior elbow -> left edge, producing a full L.
#   link 1: start from the left endpoint, move vertically to a new row, then head
#   rightward, but stop one cell before the starter's vertical leg.
#   link 2: start from that new interior endpoint, move vertically again, then head
#   rightward, and again stop against the same older leg.
# - Compared with the original task, later paths are no longer independent objects
#   completed from their own two endpoints; compared with plain D14 chain reaction,
#   each link's visible length is clipped by a collision rule.
#
# b) Complexity calibration
# - The variant is intentionally narrow rather than fully free-form:
#   exactly one starter has two ring touches, and 2-3 later links are one-touch
#   monochrome stubs on the opposite frame side.
# - This is enough to make the fused behavior visible without creating ambiguous
#   multi-component matching. Humans can see that the outer stubs are all similar,
#   but the interior segments are chained and truncated by an older barrier.
# - Later links share a common geometric blocker (the starter's vertical leg), so
#   the collision rule repeats across links and becomes learnable from 2-4 examples.
#
# c) Hidden-assumption audit against the original main()
# - Original assumption: every useful component independently encodes both endpoints
#   of one finished interior path. Broken here because later links expose only one
#   ring touch and depend on the previous endpoint as their start.
# - Original assumption: once a Manhattan path is chosen, it is painted in full.
#   Broken here because later paths must stop before overwriting older painted cells.
# - Original assumption: choosing the path with an inside elbow is enough. Broken
#   here because for chained links one candidate path can be blocked immediately at
#   the first step, so path selection must prefer a route that can actually leave the
#   current endpoint.
# - New combo interaction: the solver now assumes exactly one multi-touch starter
#   exists, and that at each chain step there is a unique nearest viable link whose
#   chosen route can advance at least one new cell. The generator must guarantee that.
#
# d) Placement legality
# - The generator places one starter on the left or right, then places all later
#   one-touch stubs on the opposite side with row gaps of at least 2 cells so the
#   outer components stay 4-disconnected.
# - Interior dimensions and starter elbow rows are chosen so the starter's vertical
#   leg has enough span to host 2-3 distinct collision rows.
# - Every outer component still goes through the original no-overlap / no-adjacency
#   checks, and every sample is oracle-checked with main().
#
# e) Visual observability
# - The chain order is geometric rather than based on color ids: from the current
#   endpoint, the nearest remaining stub is the next one that can extend.
# - The collision behavior is visible because several later segments stop flush
#   against the same older starter leg instead of reaching their far-side targets.
# - The difference from the original task is visible in the output alone: multiple
#   colored segments hang off one earlier path and are shorter than the naïve full
#   boundary-to-boundary completions.
#
# f) Cleanliness
# - The output stays legible because only one starter barrier is used and later links
#   are aligned in a simple ladder-like chain. Distractors, if any, remain outside.
# - We avoid mixing U-pairs with chain links here; that would add another object
#   family without helping the fused concept.
#
# g) Color inference
# - The starter may be monochrome or two-color alternating, preserving a recognizable
#   part of the original family.
# - Later links are monochrome on purpose: their role is to reveal chain order and
#   collision stopping, not introduce additional alternation ambiguity with only one
#   visible ring touch.

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
    step = 1 if int(b) >= int(a) else -1
    return range(int(a), int(b) + step, step)


def _vpath(r0: int, r1: int, c: int) -> list[tuple[int, int]]:
    return [(int(r), int(c)) for r in _line(int(r0), int(r1))]


def _hpath(r: int, c0: int, c1: int) -> list[tuple[int, int]]:
    return [(int(r), int(c)) for c in _line(int(c0), int(c1))]


def _join(*parts: list[tuple[int, int]]) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    for part in parts:
        for p in part:
            pp = (int(p[0]), int(p[1]))
            if not out or out[-1] != pp:
                out.append(pp)
    return out


def _build_manhattan(p1: tuple[int, int], p2: tuple[int, int], corner_a: bool) -> list[tuple[int, int]]:
    (ra, ca), (rb, cb) = p1, p2
    if int(ra) == int(rb):
        return [(int(ra), int(c)) for c in _line(int(ca), int(cb))]
    if int(ca) == int(cb):
        return [(int(r), int(ca)) for r in _line(int(ra), int(rb))]
    if corner_a:
        return _join(_vpath(int(ra), int(rb), int(ca)), _hpath(int(rb), int(ca), int(cb)))
    return _join(_hpath(int(ra), int(ca), int(cb)), _vpath(int(ra), int(rb), int(cb)))


def _inside_corner(p1: tuple[int, int], p2: tuple[int, int], corner_a: bool, out_h: int, out_w: int) -> bool:
    (ra, ca), (rb, cb) = p1, p2
    if int(ra) == int(rb) or int(ca) == int(cb):
        return True
    corner = (int(rb), int(ca)) if corner_a else (int(ra), int(cb))
    rr, cc = corner
    return 0 < int(rr) < int(out_h) - 1 and 0 < int(cc) < int(out_w) - 1


def _clockwise_ring(r0: int, c0: int, r1: int, c1: int, h: int, w: int) -> list[tuple[int, int]]:
    ring: list[tuple[int, int]] = []
    for cc in range(int(c0), int(c1) + 1):
        rr = int(r0) - 1
        if 0 <= rr < int(h):
            ring.append((int(rr), int(cc)))
    for rr in range(int(r0), int(r1) + 1):
        cc = int(c1) + 1
        if 0 <= cc < int(w):
            ring.append((int(rr), int(cc)))
    for cc in range(int(c1), int(c0) - 1, -1):
        rr = int(r1) + 1
        if 0 <= rr < int(h):
            ring.append((int(rr), int(cc)))
    for rr in range(int(r1), int(r0) - 1, -1):
        cc = int(c0) - 1
        if 0 <= cc < int(w):
            ring.append((int(rr), int(cc)))
    return ring


def _paint_path(grid: np.ndarray, path: list[tuple[int, int]], colors: tuple[int, ...]) -> None:
    if not path:
        return
    rs, cs = zip(*path)
    if len(colors) == 1:
        grid[rs, cs] = int(colors[0])
        return
    a, b = map(int, colors)
    grid[rs, cs] = [int(a) if i % 2 == 0 else int(b) for i in range(len(path))]


def _paint_alt_with_hints(
    out: np.ndarray,
    path: list[tuple[int, int]],
    colors: tuple[int, int],
    start_hint: int,
    end_hint: int | None = None,
) -> None:
    a, b = map(int, colors)
    choice = None
    for start_color in (int(a), int(b)):
        if int(start_color) != int(start_hint):
            continue
        end_color = int(start_color) if (len(path) - 1) % 2 == 0 else (int(b) if int(start_color) == int(a) else int(a))
        if end_hint is not None and int(end_color) != int(end_hint):
            continue
        choice = int(start_color)
        break
    if choice is None:
        choice = int(start_hint) if int(start_hint) in (int(a), int(b)) else int(a)
    cur = int(choice)
    for rr, cc in path:
        out[int(rr), int(cc)] = int(cur)
        cur = int(b) if int(cur) == int(a) else int(a)


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


def _path_paintable_len(out: np.ndarray, path: list[tuple[int, int]], fill: int) -> int:
    painted = 0
    for rr, cc in path:
        if int(out[int(rr), int(cc)]) != int(fill):
            break
        painted += 1
    return int(painted)


def _choose_chain_path(
    start: tuple[int, int],
    target: tuple[int, int],
    out: np.ndarray,
    fill: int,
) -> tuple[list[tuple[int, int]], int] | None:
    out_h, out_w = map(int, out.shape)
    candidates: list[tuple[tuple[int, int], ...]] = []
    for corner_a in (True, False):
        path = tuple(_build_manhattan(start, target, corner_a))
        if path not in candidates:
            candidates.append(path)

    best = None
    for path_tuple in candidates:
        path = list(path_tuple)
        suffix = path[1:]
        progress = _path_paintable_len(out, suffix, int(fill))
        if progress <= 0:
            continue
        # Prefer routes that can leave the current endpoint and remain paintable longest.
        # Tie-break toward the more "inside" elbow to stay close to the original family.
        turn_inside = _inside_corner(start, target, path == list(_build_manhattan(start, target, True)), out_h, out_w)
        score = (int(progress), 1 if turn_inside else 0)
        if best is None or score > best[0]:
            best = (score, path, int(progress))
    if best is None:
        return None
    _score, path, progress = best
    return list(path), int(progress)


def _paint_chain_segment(
    out: np.ndarray,
    path: list[tuple[int, int]],
    color: int,
    fill: int,
) -> tuple[int, int] | None:
    painted: list[tuple[int, int]] = []
    for rr, cc in path[1:]:
        if int(out[int(rr), int(cc)]) != int(fill):
            break
        out[int(rr), int(cc)] = int(color)
        painted.append((int(rr), int(cc)))
    if not painted:
        return None
    return painted[-1]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recover the framed interior by drawing one starter path and then chaining collision-stopped links from its endpoint.

    Input structure:
    - One closed 1px rectangular frame with a uniform interior fill.
    - Exactly one outer starter component touches the ring in two places and behaves like the original L-shaped source.
    - Two or more later outer link components are monochrome one-touch stubs on the opposite side of the frame.

    Rule:
    - Output only the frame interior, initialized with the frame fill color.
    - Reconstruct the starter's full Manhattan path first, preserving solid or alternating color behavior.
    - Then repeatedly choose the nearest remaining viable one-touch link from the current endpoint, route a Manhattan path toward its mapped touchpoint, and paint only until the next cell would hit already-painted older path cells.
    - The newly reached endpoint becomes the next chain start; if a chosen route has no blocker before its target, it may finish on that boundary target.

    Key insight: later components do not contribute independent completed paths. They only reveal targets and colors; the true start of each later segment is the previous endpoint, and its visible length is determined by the first collision with earlier painted structure.
    """
    g = np.array(grid, dtype=int)
    h, w = g.shape
    bg = int(get_bg(g))

    found = _find_framed_rect_uniform_fill(g, bg=int(bg))
    assert found is not None
    r0, c0, r1, c1, bc, fill = found
    out_h = int(r1 - r0 - 1)
    out_w = int(c1 - c0 - 1)
    out = np.full((out_h, out_w), int(fill), dtype=int)
    r_int0 = int(r0 + 1)
    c_int0 = int(c0 + 1)

    def map_to_out(rr: int, cc: int) -> tuple[int, int]:
        mr = max(0, min(out_h - 1, int(rr - r_int0)))
        mc = max(0, min(out_w - 1, int(cc - c_int0)))
        return int(mr), int(mc)

    comp_mask = (g != int(bg)) & (g != int(fill)) & (g != int(bc))
    clab, cn = label(comp_mask, structure=STRUCT4)
    ring = _clockwise_ring(int(r0), int(c0), int(r1), int(c1), int(h), int(w))
    ring_order = {cell: i for i, cell in enumerate(ring)}

    starters: list[dict[str, object]] = []
    links: list[dict[str, object]] = []

    for cid in range(1, int(cn) + 1):
        pts = np.argwhere(clab == int(cid))
        if pts.size == 0:
            continue
        ring_hits = [(int(rr), int(cc)) for rr, cc in pts if (int(rr), int(cc)) in ring_order]
        if not ring_hits:
            continue

        hist: dict[tuple[int, int], Counter] = defaultdict(Counter)
        for rr, cc in ring_hits:
            hist[map_to_out(int(rr), int(cc))][int(g[int(rr), int(cc)])] += 1
        coords = list(hist.keys())
        colors = tuple(sorted(set(map(int, g[clab == int(cid)].flatten()))))
        first_order = min(int(ring_order[(int(rr), int(cc))]) for rr, cc in ring_hits)

        if len(coords) >= 2:
            pair = _choose_farthest_pair(coords)
            assert pair is not None
            a, b = pair
            path_a = _build_manhattan(a, b, True)
            path_b = _build_manhattan(a, b, False)
            path = path_b if (_inside_corner(a, b, False, out_h, out_w) and not _inside_corner(a, b, True, out_h, out_w)) else path_a
            starters.append(
                {
                    "cid": int(cid),
                    "order": int(first_order),
                    "path": list(path),
                    "start": (int(a[0]), int(a[1])),
                    "end": (int(b[0]), int(b[1])),
                    "colors": tuple(map(int, colors)),
                    "hist": hist,
                }
            )
        elif len(coords) == 1:
            coord = coords[0]
            color = int(hist[coord].most_common(1)[0][0])
            links.append(
                {
                    "cid": int(cid),
                    "order": int(first_order),
                    "coord": (int(coord[0]), int(coord[1])),
                    "color": int(color),
                }
            )

    assert starters
    link_side = None
    if links:
        link_cols = {int(link["coord"][1]) for link in links}
        link_rows = {int(link["coord"][0]) for link in links}
        if len(link_cols) == 1:
            only_col = next(iter(link_cols))
            if int(only_col) == 0:
                link_side = "left"
            elif int(only_col) == int(out_w) - 1:
                link_side = "right"
        elif len(link_rows) == 1:
            only_row = next(iter(link_rows))
            if int(only_row) == 0:
                link_side = "top"
            elif int(only_row) == int(out_h) - 1:
                link_side = "bottom"

    starter = min(starters, key=lambda item: (int(item["order"]), int(item["cid"])))
    starter_path = list(starter["path"])
    starter_colors = tuple(map(int, starter["colors"]))
    starter_hist = starter["hist"]
    start_pt = tuple(starter["start"])
    end_pt = tuple(starter["end"])

    if link_side is not None:
        desired_end = {
            "left": lambda pt: int(pt[1]) == int(out_w) - 1,
            "right": lambda pt: int(pt[1]) == 0,
            "top": lambda pt: int(pt[0]) == int(out_h) - 1,
            "bottom": lambda pt: int(pt[0]) == 0,
        }[str(link_side)]
        if desired_end(start_pt) and not desired_end(end_pt):
            starter_path = list(reversed(starter_path))
            start_pt, end_pt = end_pt, start_pt

    if len(starter_colors) == 1:
        for rr, cc in starter_path:
            out[int(rr), int(cc)] = int(starter_colors[0])
    else:
        start_hint = int(starter_hist[start_pt].most_common(1)[0][0])
        end_hint = int(starter_hist[end_pt].most_common(1)[0][0])
        _paint_alt_with_hints(out, starter_path, (int(starter_colors[0]), int(starter_colors[1])), int(start_hint), int(end_hint))

    current = tuple(starter_path[-1])
    remaining = list(links)
    while remaining:
        choices = []
        for link in remaining:
            target = tuple(link["coord"])
            picked = _choose_chain_path(current, target, out, int(fill))
            if picked is None:
                continue
            path, progress = picked
            dist = abs(int(target[0]) - int(current[0])) + abs(int(target[1]) - int(current[1]))
            choices.append((int(dist), int(link["order"]), -int(progress), link, path))
        assert choices
        choices.sort(key=lambda item: (item[0], item[1], item[2]))
        _dist, _order, _neg_progress, chosen_link, chosen_path = choices[0]
        nxt = _paint_chain_segment(out, chosen_path, int(chosen_link["color"]), int(fill))
        assert nxt is not None
        current = tuple(nxt)
        remaining = [link for link in remaining if int(link["cid"]) != int(chosen_link["cid"])]

    return out


_BORDER_COLOR = 3
_BG_CHOICES = [0, 1, 2, 4, 5, 6, 7, 8, 9]
_FILL_EQ_BG_PROB = 0.6
_INTERIOR_H_RANGE = (8, 13)
_INTERIOR_W_RANGE = (8, 14)
_GRID_SIZE_RANGE = (22, 32)
_GRID_INTERIOR_PAD = 10
_FRAME_MARGIN = 4
_FRAME_MARGIN_FAR = 6
_OUTER_OFFSET_RANGE = (2, 3)
_STARTER_DUAL_PROB = 0.45
_LINK_COUNT_RANGE = (2, 3)
_LINK_LEN_RANGE = (2, 4)
_DISTRACTOR_COUNT_RANGE = (0, 1)
_DISTRACTOR_SIZE_RANGE = (1, 3)
_DISTRACTOR_TRIES = 30
_DISTRACTOR_CLEARANCE = 2
_GEN_RETRIES = 800


def _can_place(grid: np.ndarray, coords: list[tuple[int, int]], bg: int, comp_mask: np.ndarray) -> bool:
    hh, ww = map(int, grid.shape)
    cells = set((int(r), int(c)) for r, c in coords)
    for rr, cc in cells:
        if not (0 <= int(rr) < hh and 0 <= int(cc) < ww):
            return False
        if int(grid[int(rr), int(cc)]) != int(bg):
            return False
        for nr, nc in neighbors4(int(rr), int(cc)):
            if (int(nr), int(nc)) not in cells and 0 <= int(nr) < hh and 0 <= int(nc) < ww and comp_mask[int(nr), int(nc)]:
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


def _starter_specs(
    kind: str,
    elbow_r: int,
    elbow_c: int,
    hh: int,
    ww: int,
    r0: int,
    c0: int,
    r1: int,
    c1: int,
) -> tuple[list[tuple[int, int]], list[tuple[int, int]], tuple[tuple[int, int], tuple[int, int]]]:
    is_top = str(kind)[0] == "t"
    is_left = str(kind)[1] == "l"
    ring_a = (int(r0 - 1 if is_top else r1 + 1), int(c0 + 1 + elbow_c))
    ring_b = (int(r0 + 1 + elbow_r), int(c0 - 1 if is_left else c1 + 1))
    rr = int(r0 - random.randint(*_OUTER_OFFSET_RANGE)) if is_top else int(r1 + random.randint(*_OUTER_OFFSET_RANGE))
    cc = int(c0 - random.randint(*_OUTER_OFFSET_RANGE)) if is_left else int(c1 + random.randint(*_OUTER_OFFSET_RANGE))
    outer = _join(
        _vpath(int(ring_a[0]), int(rr), int(ring_a[1])),
        _hpath(int(rr), int(ring_a[1]), int(cc)),
        _vpath(int(rr), int(ring_b[0]), int(cc)),
        _hpath(int(ring_b[0]), int(cc), int(ring_b[1])),
    )
    start = (0 if is_top else hh - 1, int(elbow_c))
    end = (int(elbow_r), 0 if is_left else ww - 1)
    inner = _build_manhattan(start, end, True)
    return list(inner), list(outer), ((int(ring_a[0]), int(ring_a[1])), (int(ring_b[0]), int(ring_b[1])))


def _link_stub(side: str, pos: int, length: int, r0: int, c0: int, r1: int, c1: int) -> tuple[tuple[int, int], list[tuple[int, int]]]:
    if str(side) == "top":
        ring_touch = (int(r0 - 1), int(c0 + 1 + pos))
        outer = _vpath(int(r0 - 1), int(r0 - length), int(c0 + 1 + pos))
    elif str(side) == "bottom":
        ring_touch = (int(r1 + 1), int(c0 + 1 + pos))
        outer = _vpath(int(r1 + 1), int(r1 + length), int(c0 + 1 + pos))
    elif str(side) == "left":
        ring_touch = (int(r0 + 1 + pos), int(c0 - 1))
        outer = _hpath(int(r0 + 1 + pos), int(c0 - 1), int(c0 - length))
    else:
        ring_touch = (int(r0 + 1 + pos), int(c1 + 1))
        outer = _hpath(int(r0 + 1 + pos), int(c1 + 1), int(c1 + length))
    return (int(ring_touch[0]), int(ring_touch[1])), list(outer)


def _add_distractor(grid: np.ndarray, comp_mask: np.ndarray, bg: int, frame: tuple[int, int, int, int], color: int) -> None:
    hh, ww = map(int, grid.shape)
    r0, c0, r1, c1 = frame
    gap = _DISTRACTOR_CLEARANCE
    sz_lo, sz_hi = _DISTRACTOR_SIZE_RANGE
    for _ in range(_DISTRACTOR_TRIES):
        size = random.randint(int(sz_lo), int(sz_hi))
        rr = random.randint(1, hh - 2)
        cc = random.randint(1, ww - 2)
        coords = [(int(rr), int(cc))]
        used = {(int(rr), int(cc))}
        for _step in range(size - 1):
            base = random.choice(coords)
            opts = [
                (int(nr), int(nc))
                for nr, nc in neighbors4(*base)
                if 1 <= int(nr) < hh - 1 and 1 <= int(nc) < ww - 1 and (int(nr), int(nc)) not in used
            ]
            if not opts:
                break
            pick = random.choice(opts)
            used.add(pick)
            coords.append(pick)
        if any(int(r0) - gap <= int(rp) <= int(r1) + gap and int(c0) - gap <= int(cp) <= int(c1) + gap for rp, cp in coords):
            continue
        if _place_component(grid, comp_mask, coords, (int(color),), int(bg)):
            return


def _starter_row_options(kind: str, hh: int, link_count: int) -> list[int]:
    if str(kind) in ("tl", "tr"):
        needed = 2 * int(link_count) - 1
        return [int(r) for r in range(int(needed) + 1, hh - 1)]
    needed = 2 * int(link_count) - 1
    return [int(r) for r in range(1, hh - 1 - int(needed))]


def _collision_rows(kind: str, elbow_r: int, link_count: int) -> list[int]:
    if str(kind) in ("tl", "tr"):
        rows = [int(elbow_r - (2 * i + 1)) for i in range(int(link_count))]
        return rows
    rows = [int(elbow_r + (2 * i + 1)) for i in range(int(link_count))]
    return rows


def _gen_attempt() -> np.ndarray | None:
    bg = int(random.choice(_BG_CHOICES))
    bc = int(_BORDER_COLOR)
    fill_choices = [x for x in range(10) if x not in (int(bg), int(bc))]
    fill = int(bg if random.random() < _FILL_EQ_BG_PROB else random.choice(fill_choices))
    hh = random.randint(*_INTERIOR_H_RANGE)
    ww = random.randint(*_INTERIOR_W_RANGE)
    rows = random.randint(max(_GRID_SIZE_RANGE[0], hh + _GRID_INTERIOR_PAD), _GRID_SIZE_RANGE[1])
    cols = random.randint(max(_GRID_SIZE_RANGE[0], ww + _GRID_INTERIOR_PAD), _GRID_SIZE_RANGE[1])
    r0 = random.randint(_FRAME_MARGIN, rows - hh - _FRAME_MARGIN_FAR)
    c0 = random.randint(_FRAME_MARGIN, cols - ww - _FRAME_MARGIN_FAR)
    r1, c1 = int(r0 + hh + 1), int(c0 + ww + 1)

    grid = np.full((rows, cols), int(bg), dtype=int)
    grid[r0, c0 : c1 + 1] = int(bc)
    grid[r1, c0 : c1 + 1] = int(bc)
    grid[r0 : r1 + 1, c0] = int(bc)
    grid[r0 : r1 + 1, c1] = int(bc)
    grid[r0 + 1 : r1, c0 + 1 : c1] = int(fill)

    expected = np.full((hh, ww), int(fill), dtype=int)
    comp_mask = np.zeros_like(grid, dtype=bool)
    palette = [x for x in range(10) if x not in (int(bg), int(bc), int(fill))]
    random.shuffle(palette)
    if len(palette) < 4:
        return None

    link_count = random.randint(*_LINK_COUNT_RANGE)
    kind = random.choice(["tl", "tr", "bl", "br"])
    side_end = "left" if str(kind)[1] == "l" else "right"

    row_opts = _starter_row_options(kind, hh, int(link_count))
    if not row_opts:
        return None
    elbow_r = int(random.choice(row_opts))
    elbow_c_lo = 2
    elbow_c_hi = ww - 3
    if elbow_c_lo > elbow_c_hi:
        return None
    elbow_c = int(random.randint(elbow_c_lo, elbow_c_hi))
    starter_inner, starter_outer, _starter_ring = _starter_specs(kind, elbow_r, elbow_c, hh, ww, r0, c0, r1, c1)

    if random.random() < _STARTER_DUAL_PROB:
        starter_colors = (int(palette[0]), int(palette[1]))
        color_i = 2
    else:
        starter_colors = (int(palette[0]),)
        color_i = 1
    if not _place_component(grid, comp_mask, starter_outer, starter_colors, int(bg)):
        return None
    _paint_path(expected, starter_inner, starter_colors)
    current = tuple(starter_inner[-1])

    rows_chain = _collision_rows(kind, int(elbow_r), int(link_count))
    if any(not (1 <= int(rr) <= hh - 2) for rr in rows_chain):
        return None

    if str(side_end) == "left":
        target_side = "right"
    else:
        target_side = "left"

    for rr in rows_chain:
        if color_i >= len(palette):
            return None
        link_color = int(palette[color_i])
        color_i += 1
        pos = int(rr)
        _ring_touch, outer = _link_stub(target_side, int(pos), random.randint(*_LINK_LEN_RANGE), r0, c0, r1, c1)
        if not _place_component(grid, comp_mask, outer, (int(link_color),), int(bg)):
            return None
        target = (int(rr), ww - 1) if str(target_side) == "right" else (int(rr), 0)
        picked = _choose_chain_path(current, target, expected, int(fill))
        if picked is None:
            return None
        path, progress = picked
        if int(progress) <= 0:
            return None
        if int(progress) >= len(path) - 1:
            return None
        nxt = _paint_chain_segment(expected, path, int(link_color), int(fill))
        if nxt is None:
            return None
        current = tuple(nxt)

    for extra_color in palette[color_i : color_i + random.randint(*_DISTRACTOR_COUNT_RANGE)]:
        _add_distractor(grid, comp_mask, int(bg), (r0, c0, r1, c1), int(extra_color))

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
