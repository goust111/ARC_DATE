# Combo variant of a6f40cea (index 36)
# combo_spec: D7/1+D12/2+D14/1
# Name: 多组件假源过滤排序
#
# Design analysis
# ---------------
# a) Concrete behavior and trace:
#    - Base task: each ring-touching outside open rectangle is treated independently and paints one
#      inside path. With only 2-3 components, paint order is mostly irrelevant.
#    - Combo task: this implementation expands to 4-6 touching components total, with 1-2 fake
#      sources mixed into 3-5 genuine ones. A fake
#      source is an ordinary L-shaped open rectangle whose far outer elbow carries one extra cell,
#      creating a visible filled 2x2 stamp. Those stamped components still touch the frame ring but
#      must be discarded before any path reconstruction.
#    - Each genuine component still determines one inside L-path. The ranking cue is no longer
#      arbitrary iteration order: genuine components are painted from smaller to larger Manhattan
#      distance between the visible outer source elbow and the frame center. Later, farther
#      components overwrite earlier, nearer ones.
#    - Small trace: suppose a top-left source has outer elbow at (6,7), a bottom-right source has
#      outer elbow at (15,16), and both reconstructed paths cross. Originally both would be painted
#      in whatever component order the solver happened to see. Here, if the first source is closer
#      to center, it paints first and the farther source overwrites the crossing later. A stamped
#      fake source with its own two ring touches is ignored entirely.
# b) Complexity calibration:
#    - I keep a single fake-source cue (presence of a 2x2 stamp) and a single ordering cue
#      (distance of the visible outer elbow to the center). That is enough to make the combo visually
#      different without exploding the hypothesis space.
#    - U-pairs are disabled in generation. They are legal in the original family, but mixing stripe
#      reconstruction with fake filtering and distance sorting would hide the intended combo signal in
#      only 2-4 ARC examples.
#    - The generator spans the full D7 count range 4-6 while capping fake sources at 2. That keeps
#      the density increase visible while leaving enough legal space for stamped decoys.
# c) Implicit assumption audit:
#    - Original assumption "every ring-touching component is actionable" is broken by D12/2.
#      The solver must classify components before projecting endpoints.
#    - Original assumption "component processing order does not matter" is broken by D7+D14 once
#      multiple real paths overlap. The solver must extract all genuine actions first, then sort them by
#      a visible geometric key before painting.
#    - Original generator assumption `used_inner` can forbid interior overlap is also broken: that
#      would make the new ordering invisible. The combo generator therefore allows hidden interior
#      path overlap and explicitly rejects samples where sorting would not change anything.
# d) Placement legality:
#    - Outside components are still 4-disconnected and may not overlap or touch each other. Side
#      touch positions are spaced along each frame side; duplicate corner families use different
#      outside layers so their open rectangles do not collide.
#    - Fake sources reuse the same L geometry and add exactly one extra elbow-adjacent cell, so they
#      remain easy to place and easy to recognize.
# e) Visual observability:
#    - Filtering depends only on a visible geometric stamp, never on color id.
#    - Sorting depends only on the visible outer elbow of each source, which is directly readable from
#      the component geometry and matches "distance from the source to the frame center".
#    - The generator enforces both `fake_delta > 0` and `order_delta > 0`, so training examples
#      cannot silently collapse back to the original task. If two genuine sources happen to be at
#      the same center distance, the visible clockwise order of first ring contact breaks the tie.
# f) Visual cleanliness:
#    - The outside remains sparse and non-overlapping. Clutter is controlled by removing unrelated
#      far-away distractor blobs and by using monochrome genuine traces in generation.
#    - The inside may contain overlaps, but only among 3-5 reconstructed genuine L-paths, which is the
#      intended dense effect of the combo rather than accidental noise.
# g) New solver assumptions and generator guarantees:
#    - Every intended real or fake source is one 4-connected outside component with exactly two ring
#      touches on adjacent frame sides.
#    - Genuine sources have no 2x2 block; fake sources do.
#    - Genuine sources are ranked by visible outer-elbow distance to frame center, with clockwise
#      first-contact order as a visible tie-breaker, and at least one overlap must produce a
#      different final result under reverse paint order.

from __future__ import annotations

from collections import Counter, defaultdict
from itertools import combinations
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
    """Return the largest 1px rectangular frame whose interior is uniformly filled."""
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


def _paint_path(grid: np.ndarray, path: list[tuple[int, int]], colors: tuple[int, ...]) -> None:
    if not path:
        return
    rs, cs = zip(*path)
    if len(colors) == 1:
        grid[rs, cs] = int(colors[0])
        return
    a, b = map(int, colors)
    vals = [int(a if i % 2 == 0 else b) for i in range(len(path))]
    grid[rs, cs] = vals


def _can_place(grid: np.ndarray, coords: list[tuple[int, int]], bg: int, comp_mask: np.ndarray) -> bool:
    h, w = grid.shape
    cells = set(coords)
    for r, c in cells:
        if not (0 <= r < h and 0 <= c < w):
            return False
        if int(grid[r, c]) != int(bg):
            return False
        for nr, nc in neighbors4(r, c):
            if (nr, nc) not in cells and 0 <= nr < h and 0 <= nc < w and comp_mask[nr, nc]:
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


def _has_full_2x2(mask: np.ndarray) -> bool:
    coords = np.argwhere(mask)
    if len(coords) < 4:
        return False
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    local = mask[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1]
    return bool(np.any(local[:-1, :-1] & local[1:, :-1] & local[:-1, 1:] & local[1:, 1:]))


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


def _build_manhattan(
    p1: tuple[int, int],
    p2: tuple[int, int],
    corner_a: bool,
) -> tuple[list[tuple[int, int]], tuple[int, int]]:
    (ra, ca), (rb, cb) = p1, p2
    if ra == rb:
        return [(int(ra), int(c)) for c in _line(int(ca), int(cb))], (int(ra), int(ca))
    if ca == cb:
        return [(int(r), int(ca)) for r in _line(int(ra), int(rb))], (int(ra), int(ca))
    if corner_a:
        return _join(_vpath(ra, rb, ca), _hpath(rb, ca, cb)), (int(rb), int(ca))
    return _join(_hpath(ra, ca, cb), _vpath(ra, rb, cb)), (int(ra), int(cb))


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
    rr, cc = ((rb, ca) if corner_a else (ra, cb))
    return 0 < int(rr) < out_h - 1 and 0 < int(cc) < out_w - 1


def _choose_inside_path(
    p1: tuple[int, int],
    p2: tuple[int, int],
    out_h: int,
    out_w: int,
) -> tuple[list[tuple[int, int]], tuple[int, int]]:
    path_a, corner_a = _build_manhattan(p1, p2, True)
    path_b, corner_b = _build_manhattan(p1, p2, False)
    if _inside_corner(p1, p2, False, out_h, out_w) and not _inside_corner(p1, p2, True, out_h, out_w):
        return path_b, corner_b
    return path_a, corner_a


def _side_of(p: tuple[int, int], out_h: int, out_w: int) -> str | None:
    r, c = map(int, p)
    if r == 0 and 0 < c < out_w - 1:
        return "top"
    if r == out_h - 1 and 0 < c < out_w - 1:
        return "bottom"
    if c == 0 and 0 < r < out_h - 1:
        return "left"
    if c == out_w - 1 and 0 < r < out_h - 1:
        return "right"
    return None


def _kind_from_sides(side_a: str | None, side_b: str | None) -> str | None:
    pair = {side_a, side_b}
    if pair == {"top", "left"}:
        return "tl"
    if pair == {"top", "right"}:
        return "tr"
    if pair == {"bottom", "left"}:
        return "bl"
    if pair == {"bottom", "right"}:
        return "br"
    return None


def _center_distance_from_corner(corner: tuple[int, int], out_h: int, out_w: int) -> int:
    rr, cc = map(int, corner)
    return int(abs(2 * rr - (out_h - 1)) + abs(2 * cc - (out_w - 1)))


def _outer_corner_from_component(kind: str, pts: np.ndarray) -> tuple[int, int]:
    r0, c0 = pts.min(axis=0)
    r1, c1 = pts.max(axis=0)
    if kind == "tl":
        return int(r0), int(c0)
    if kind == "tr":
        return int(r0), int(c1)
    if kind == "bl":
        return int(r1), int(c0)
    if kind == "br":
        return int(r1), int(c1)
    raise ValueError(kind)


def _frame_center_distance(corner: tuple[int, int], r0: int, c0: int, r1: int, c1: int) -> int:
    rr, cc = map(int, corner)
    return int(abs(2 * rr - (r0 + r1)) + abs(2 * cc - (c0 + c1)))


_KIND_SIDES = {
    "tl": ("top", "left"),
    "tr": ("top", "right"),
    "bl": ("bottom", "left"),
    "br": ("bottom", "right"),
}


def _choose_component_kinds(real_count: int, fake_count: int) -> tuple[list[str], list[str]]:
    base_kinds = ["tl", "tr", "bl", "br"]
    if real_count < 3 or fake_count < 1:
        raise ValueError("invalid component counts")

    real_options: list[list[str]] = []
    if real_count <= len(base_kinds):
        real_options = [list(opt) for opt in combinations(base_kinds, real_count)]
    elif real_count == len(base_kinds) + 1:
        for dup in base_kinds:
            real_options.append(list(base_kinds) + [dup])
    else:
        raise ValueError("unsupported real_count")

    fake_options = [list(opt) for opt in combinations(base_kinds, fake_count)]
    scored: list[tuple[tuple[int, int, int], list[str], list[str]]] = []
    for real_kinds in real_options:
        for fake_kinds in fake_options:
            loads = Counter()
            for kind in real_kinds + fake_kinds:
                for side in _KIND_SIDES[kind]:
                    loads[side] += 1
            score = (
                max(loads.values()),
                sum(v * v for v in loads.values()),
                sum(1 for v in loads.values() if v >= 3),
            )
            scored.append((score, list(real_kinds), list(fake_kinds)))

    best_score = min(score for score, _real, _fake in scored)
    candidates = [(real_kinds, fake_kinds) for score, real_kinds, fake_kinds in scored if score == best_score]
    real_kinds, fake_kinds = random.choice(candidates)

    real_load = Counter(real_kinds)
    fake_load = Counter(fake_kinds)
    real_kinds.sort(key=lambda kind: (-real_load[kind], kind))
    fake_kinds.sort(key=lambda kind: (-fake_load[kind], -real_load[kind], kind))
    return real_kinds, fake_kinds


def _paint_actions(out: np.ndarray, actions: list[dict[str, object]]) -> None:
    for action in actions:
        path = list(action["path"])
        if action["mode"] == "solid":
            color = int(action["color"])
            for rr, cc in path:
                out[int(rr), int(cc)] = int(color)
            continue
        ca, cb = map(int, action["colors"])
        cur = int(action["start"])
        for rr, cc in path:
            out[int(rr), int(cc)] = int(cur)
            cur = int(cb) if int(cur) == int(ca) else int(ca)


def _render_actions(
    out_h: int,
    out_w: int,
    fill: int,
    actions: list[dict[str, object]],
) -> np.ndarray:
    out = np.full((out_h, out_w), int(fill), dtype=int)
    _paint_actions(out, actions)
    return out


def _extract_actions(
    g: np.ndarray,
    r0: int,
    c0: int,
    r1: int,
    c1: int,
    bg: int,
    bc: int,
    fill: int,
    *,
    include_fake: bool,
) -> tuple[list[dict[str, object]], dict[str, int], int, int]:
    h, w = g.shape
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
    ring = _clockwise_ring(r0, c0, r1, c1, h, w)
    ring_set = set(ring)
    ring_order = {cell: i for i, cell in enumerate(ring)}

    actions: list[dict[str, object]] = []
    meta = {"touching": 0, "real": 0, "fake": 0}

    for cid in range(1, int(cn) + 1):
        cmask = clab == int(cid)
        pts = np.argwhere(cmask)
        ring_hits = [(int(rr), int(cc)) for rr, cc in pts if (int(rr), int(cc)) in ring_set]
        if len(ring_hits) < 2:
            continue
        meta["touching"] += 1
        is_fake = _has_full_2x2(cmask)
        if is_fake:
            meta["fake"] += 1
            if not include_fake:
                continue
        else:
            meta["real"] += 1

        hist: dict[tuple[int, int], Counter] = defaultdict(Counter)
        for rr, cc in ring_hits:
            hist[map_to_out(int(rr), int(cc))][int(g[int(rr), int(cc)])] += 1
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
        side_a = _side_of(a, out_h, out_w)
        side_b = _side_of(b, out_h, out_w)
        kind = _kind_from_sides(side_a, side_b)
        if kind is None:
            continue

        path, _elbow = _choose_inside_path(a, b, out_h, out_w)
        first_order = min(ring_order[(rr, cc)] for rr, cc in ring_hits)
        source_corner = _outer_corner_from_component(kind, pts)
        center_dist = _frame_center_distance(source_corner, r0, c0, r1, c1)
        cols = sorted(set(map(int, g[cmask].flatten())))
        if len(cols) == 1:
            actions.append(
                {
                    "cid": int(cid),
                    "fake": bool(is_fake),
                    "mode": "solid",
                    "path": path,
                    "color": int(cols[0]),
                    "dist": int(center_dist),
                    "order": int(first_order),
                    "source_corner": source_corner,
                }
            )
            continue
        if len(cols) != 2:
            continue

        ca, cb = map(int, cols)
        endpoint_hint = {
            a: int(hist[a].most_common(1)[0][0]),
            b: int(hist[b].most_common(1)[0][0]),
        }
        chosen = None
        for path_pts in (path, list(reversed(path))):
            want_start = endpoint_hint[path_pts[0]]
            want_end = endpoint_hint[path_pts[-1]]
            for start_color in (ca, cb):
                end_color = int(start_color if len(path_pts) % 2 == 1 else (cb if start_color == ca else ca))
                if int(start_color) == int(want_start) and int(end_color) == int(want_end):
                    chosen = (path_pts, int(start_color))
                    break
            if chosen is not None:
                break
        if chosen is None:
            continue
        actions.append(
            {
                "cid": int(cid),
                "fake": bool(is_fake),
                "mode": "alt",
                "path": chosen[0],
                "colors": (int(ca), int(cb)),
                "start": int(chosen[1]),
                "dist": int(center_dist),
                "order": int(first_order),
                "source_corner": source_corner,
            }
        )

    actions.sort(key=lambda item: (int(item["dist"]), int(item["order"]), int(item["cid"])))
    return actions, meta, out_h, out_w


def _behavior_metrics_from_actions(
    real_actions: list[dict[str, object]],
    all_actions: list[dict[str, object]],
    out_h: int,
    out_w: int,
    fill: int,
) -> dict[str, int]:
    cover = Counter()
    for action in real_actions:
        for cell in action["path"]:
            cover[tuple(cell)] += 1
    overlap_cells = sum(1 for v in cover.values() if v >= 2)
    overwrite_events = sum(v - 1 for v in cover.values() if v >= 2)

    base = _render_actions(out_h, out_w, fill, real_actions)
    rev_actions = sorted(real_actions, key=lambda item: (-int(item["dist"]), -int(item["order"]), -int(item["cid"])))
    rev = _render_actions(out_h, out_w, fill, rev_actions)
    all_out = _render_actions(out_h, out_w, fill, all_actions)

    return {
        "overlap_cells": int(overlap_cells),
        "overwrite_events": int(overwrite_events),
        "order_delta": int(np.sum(base != rev)),
        "fake_delta": int(np.sum(base != all_out)),
        "real_count": int(len(real_actions)),
        "fake_count": int(sum(1 for a in all_actions if a["fake"])),
    }


def _behavior_metrics(grid: np.ndarray) -> dict[str, int]:
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))
    found = _find_framed_rect_uniform_fill(g, bg=bg)
    assert found is not None
    r0, c0, r1, c1, bc, fill = found
    real_actions, meta, out_h, out_w = _extract_actions(g, r0, c0, r1, c1, bg, bc, fill, include_fake=False)
    all_actions, _meta2, _h2, _w2 = _extract_actions(g, r0, c0, r1, c1, bg, bc, fill, include_fake=True)
    metrics = _behavior_metrics_from_actions(real_actions, all_actions, out_h, out_w, fill)
    metrics.update(
        {
            "touching": int(meta["touching"]),
            "real_detected": int(meta["real"]),
            "fake_detected": int(meta["fake"]),
        }
    )
    return metrics


def main(grid: np.ndarray) -> np.ndarray:
    """
    Rebuild a dense framed interior by filtering fake sources, sorting genuine ones, and repainting them.

    Input structure:
    - One 1px closed rectangular frame with a uniform fill sits in a larger background.
    - Around that frame are 4-6 outside open-rectangle L traces touching the ring just beyond the
      border.
    - Genuine traces are thin 1-cell-wide L-shaped open frames; fake traces look similar but include
      a visible filled 2x2 stamp at the far outer elbow.

    Rule:
    - Crop only the frame interior and initialize it with the frame fill color.
    - Ignore every stamped fake source before reconstructing any path.
    - For each remaining source, map its two ring touches inward, recover the corresponding Manhattan
      L-path, then sort those genuine paths by the center distance of their visible outer source
      corner.
    - Paint from near to far; later, farther paths overwrite earlier ones where overlaps occur.

    Key insight:
    - The task is no longer "every touching trace draws a path". First separate genuine thin traces
      from stamped decoys, then rank genuine sources by how close their visible outer elbow is to the
      frame center before repainting the recovered paths.
    """
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))

    found = _find_framed_rect_uniform_fill(g, bg=bg)
    assert found is not None
    r0, c0, r1, c1, bc, fill = found

    actions, _meta, out_h, out_w = _extract_actions(g, r0, c0, r1, c1, bg, bc, fill, include_fake=False)
    out = np.full((out_h, out_w), int(fill), dtype=int)
    _paint_actions(out, actions)
    return out


# --------------- generator ---------------

_BORDER_COLOR = 3
_BG_CHOICES = [0, 1, 2, 4, 5, 6, 7, 8, 9]
_FILL_EQ_BG_PROB = 0.7
_INTERIOR_H_RANGE = (12, 18)
_INTERIOR_W_RANGE = (12, 18)
_GRID_SIZE_RANGE = (42, 58)
_FRAME_MARGIN = 10
_FRAME_MARGIN_FAR = 20
_TOTAL_COMPONENTS_RANGE = (4, 6)
_FAKE_COUNT_RANGE = (1, 2)
_OUTER_OFFSET_BASE = 2
_OUTER_OFFSET_STEP = 4
_DUAL_COMPONENT_CAP = 0
_DUAL_COLOR_PROB = 0.0
_COMPONENT_TRIES = 80
_GEN_RETRIES = 1000
_COMPONENT_MODES = ((4, 1), (5, 1), (5, 2), (6, 1))


def _pick_axis_position(size: int, used: list[int], preferred: int | None = None) -> int | None:
    candidates = [p for p in range(1, size - 1) if all(abs(p - u) >= 2 for u in used)]
    if not candidates:
        return None
    random.shuffle(candidates)
    if preferred is None:
        candidates.sort(key=lambda p: abs(2 * p - (size - 1)))
    else:
        candidates.sort(key=lambda p: (abs(p - preferred), abs(2 * p - (size - 1))))
    head = candidates[: min(len(candidates), 12)]
    return int(random.choice(head))


def _cross_levels(size: int) -> tuple[int, int]:
    far = max(1, size // 2 - 2)
    near = min(size - 2, size // 2 + 1)
    if near - far < 2:
        far = max(1, near - 2)
    return int(far), int(near)


def _l_specs(
    kind: str,
    row: int,
    col: int,
    layer: int,
    h: int,
    w: int,
    r0: int,
    c0: int,
    r1: int,
    c1: int,
    *,
    stamped: bool,
) -> tuple[list[tuple[int, int]], list[tuple[int, int]], tuple[int, int], tuple[int, int], tuple[int, int]]:
    row_offset = int(_OUTER_OFFSET_BASE + (0 if kind.endswith("l") else 3) + layer * _OUTER_OFFSET_STEP)
    col_offset = int(_OUTER_OFFSET_BASE + (0 if kind.startswith("t") else 3) + layer * _OUTER_OFFSET_STEP)

    if kind == "tl":
        ring_a = (int(r0 - 1), int(c0 + 1 + col))
        ring_b = (int(r0 + 1 + row), int(c0 - 1))
        outer_corner = (int(r0 - row_offset), int(c0 - col_offset))
        outer = _join(
            _vpath(ring_a[0], outer_corner[0], ring_a[1]),
            _hpath(outer_corner[0], ring_a[1], outer_corner[1]),
            _vpath(outer_corner[0], ring_b[0], outer_corner[1]),
            _hpath(ring_b[0], outer_corner[1], ring_b[1]),
        )
        stamp = (int(outer_corner[0] + 1), int(outer_corner[1] + 1))
        p1, p2 = (0, int(col)), (int(row), 0)
    elif kind == "tr":
        ring_a = (int(r0 - 1), int(c0 + 1 + col))
        ring_b = (int(r0 + 1 + row), int(c1 + 1))
        outer_corner = (int(r0 - row_offset), int(c1 + col_offset))
        outer = _join(
            _vpath(ring_a[0], outer_corner[0], ring_a[1]),
            _hpath(outer_corner[0], ring_a[1], outer_corner[1]),
            _vpath(outer_corner[0], ring_b[0], outer_corner[1]),
            _hpath(ring_b[0], outer_corner[1], ring_b[1]),
        )
        stamp = (int(outer_corner[0] + 1), int(outer_corner[1] - 1))
        p1, p2 = (0, int(col)), (int(row), int(w - 1))
    elif kind == "bl":
        ring_a = (int(r1 + 1), int(c0 + 1 + col))
        ring_b = (int(r0 + 1 + row), int(c0 - 1))
        outer_corner = (int(r1 + row_offset), int(c0 - col_offset))
        outer = _join(
            _vpath(ring_a[0], outer_corner[0], ring_a[1]),
            _hpath(outer_corner[0], ring_a[1], outer_corner[1]),
            _vpath(outer_corner[0], ring_b[0], outer_corner[1]),
            _hpath(ring_b[0], outer_corner[1], ring_b[1]),
        )
        stamp = (int(outer_corner[0] - 1), int(outer_corner[1] + 1))
        p1, p2 = (int(h - 1), int(col)), (int(row), 0)
    elif kind == "br":
        ring_a = (int(r1 + 1), int(c0 + 1 + col))
        ring_b = (int(r0 + 1 + row), int(c1 + 1))
        outer_corner = (int(r1 + row_offset), int(c1 + col_offset))
        outer = _join(
            _vpath(ring_a[0], outer_corner[0], ring_a[1]),
            _hpath(outer_corner[0], ring_a[1], outer_corner[1]),
            _vpath(outer_corner[0], ring_b[0], outer_corner[1]),
            _hpath(ring_b[0], outer_corner[1], ring_b[1]),
        )
        stamp = (int(outer_corner[0] - 1), int(outer_corner[1] - 1))
        p1, p2 = (int(h - 1), int(col)), (int(row), int(w - 1))
    else:
        raise ValueError(kind)

    inner, elbow = _choose_inside_path(p1, p2, h, w)
    if stamped:
        outer = list(outer) + [stamp]
    return inner, outer, ring_a, ring_b, elbow


def _gen_attempt(
    debug: Counter | None = None,
    *,
    total_count: int | None = None,
    fake_count: int | None = None,
) -> np.ndarray | None:
    grid: np.ndarray | None = None

    def fail(reason: str) -> None:
        if debug is not None:
            debug[reason] += 1

    bg = int(random.choice(_BG_CHOICES))
    bc = int(_BORDER_COLOR)
    fill_choices = [x for x in range(10) if x not in (bg, bc)]
    fill = int(bg if random.random() < _FILL_EQ_BG_PROB else random.choice(fill_choices))

    h = random.randint(*_INTERIOR_H_RANGE)
    w = random.randint(*_INTERIOR_W_RANGE)
    rows_min = max(_GRID_SIZE_RANGE[0], h + 2 + _FRAME_MARGIN + _FRAME_MARGIN_FAR)
    cols_min = max(_GRID_SIZE_RANGE[0], w + 2 + _FRAME_MARGIN + _FRAME_MARGIN_FAR)
    if rows_min > _GRID_SIZE_RANGE[1] or cols_min > _GRID_SIZE_RANGE[1]:
        fail("grid")
        return None

    rows = random.randint(rows_min, _GRID_SIZE_RANGE[1])
    cols = random.randint(cols_min, _GRID_SIZE_RANGE[1])
    r0 = random.randint(_FRAME_MARGIN, rows - (h + 2) - _FRAME_MARGIN_FAR)
    c0 = random.randint(_FRAME_MARGIN, cols - (w + 2) - _FRAME_MARGIN_FAR)
    r1, c1 = int(r0 + h + 1), int(c0 + w + 1)

    if total_count is None:
        total_count = random.randint(*_TOTAL_COMPONENTS_RANGE)
    total_count = int(total_count)
    fake_cap = min(_FAKE_COUNT_RANGE[1], total_count - 3)
    if fake_cap < _FAKE_COUNT_RANGE[0]:
        fail("real_count")
        return None
    if fake_count is None:
        fake_count = random.randint(_FAKE_COUNT_RANGE[0], fake_cap)
    fake_count = int(fake_count)
    if not (_FAKE_COUNT_RANGE[0] <= fake_count <= fake_cap):
        fail("fake_count")
        return None
    real_count = int(total_count - fake_count)

    grid = np.full((rows, cols), int(bg), dtype=int)
    grid[r0, c0 : c1 + 1] = int(bc)
    grid[r1, c0 : c1 + 1] = int(bc)
    grid[r0 : r1 + 1, c0] = int(bc)
    grid[r0 : r1 + 1, c1] = int(bc)
    grid[r0 + 1 : r1, c0 + 1 : c1] = int(fill)

    palette = [x for x in range(10) if x not in (bg, bc, fill)]
    random.shuffle(palette)
    dual_count = 0
    dual_real_ids: set[int] = set()
    color_need = total_count + dual_count
    if color_need > len(palette):
        fail("palette")
        return None

    ring_order = {cell: i for i, cell in enumerate(_clockwise_ring(r0, c0, r1, c1, rows, cols))}
    kind_layers: Counter[str] = Counter()
    used_side: dict[str, list[int]] = {"top": [], "bottom": [], "left": [], "right": []}
    comp_mask = np.zeros_like(grid, dtype=bool)
    placed_components: list[dict[str, object]] = []

    real_kinds, fake_kinds = _choose_component_kinds(real_count, fake_count)
    components = [{"kind": kind, "fake": False} for kind in real_kinds] + [
        {"kind": kind, "fake": True} for kind in fake_kinds
    ]

    color_i = 0
    row_far, row_near = _cross_levels(h)
    col_far, col_near = _cross_levels(w)
    preferred_map = {
        "tl": (row_near, col_near),
        "tr": (row_near, col_far),
        "bl": (row_far, col_near),
        "br": (row_near, col_near),
    }
    for idx, comp in enumerate(components):
        kind = str(comp["kind"])
        if idx in dual_real_ids:
            colors = (int(palette[color_i]), int(palette[color_i + 1]))
            color_i += 2
        else:
            colors = (int(palette[color_i]),)
            color_i += 1

        placed = False
        for _ in range(_COMPONENT_TRIES):
            row_side = "left" if kind in ("tl", "bl") else "right"
            col_side = "top" if kind in ("tl", "tr") else "bottom"
            pref_row, pref_col = preferred_map[kind]
            row = _pick_axis_position(h, used_side[row_side], pref_row)
            col = _pick_axis_position(w, used_side[col_side], pref_col)
            if row is None or col is None:
                continue
            layer = int(kind_layers[kind])
            inner, outer, ring_a, ring_b, elbow = _l_specs(
                kind, int(row), int(col), layer, h, w, r0, c0, r1, c1, stamped=bool(comp["fake"])
            )
            if not _can_place(grid, list(outer), bg, comp_mask):
                continue
            _paint_path(grid, list(outer), colors)
            rs, cs = zip(*outer)
            comp_mask[rs, cs] = True
            used_side[row_side].append(int(row))
            used_side[col_side].append(int(col))
            kind_layers[kind] += 1
            placed_components.append(
                {
                    "kind": kind,
                    "fake": bool(comp["fake"]),
                    "row": int(row),
                    "col": int(col),
                    "layer": int(layer),
                    "inner": inner,
                    "outer": outer,
                    "ring_a": ring_a,
                    "ring_b": ring_b,
                    "elbow": elbow,
                    "dist": int(_frame_center_distance(_outer_corner_from_component(kind, np.array(outer)), r0, c0, r1, c1)),
                    "order": int(min(ring_order[ring_a], ring_order[ring_b])),
                    "colors": colors,
                }
            )
            placed = True
            break
        if not placed:
            fail(f"place:{kind}:{kind_layers[kind]}:{'fake' if comp['fake'] else 'real'}")
            return None

    try:
        actual = main(grid)
    except Exception:
        fail("main_exc")
        return None

    found = _find_framed_rect_uniform_fill(grid, bg=bg)
    if found is None:
        fail("frame")
        return None
    rr0, cc0, rr1, cc1, bbc, ffill = found
    real_actions, meta, out_h, out_w = _extract_actions(grid, rr0, cc0, rr1, cc1, bg, bbc, ffill, include_fake=False)
    all_actions, _meta2, _hh, _ww = _extract_actions(grid, rr0, cc0, rr1, cc1, bg, bbc, ffill, include_fake=True)
    if meta["touching"] != total_count or meta["real"] != real_count or meta["fake"] != fake_count:
        fail("meta_count")
        return None
    if len(real_actions) != real_count or len(all_actions) != total_count:
        fail("action_count")
        return None
    if out_h != h or out_w != w:
        fail("shape")
        return None
    metrics = _behavior_metrics_from_actions(real_actions, all_actions, out_h, out_w, ffill)
    if len({int(action["dist"]) for action in real_actions}) < 2:
        fail("dist_levels")
        return None
    if metrics["overlap_cells"] < 1:
        fail("overlap")
        return None
    if metrics["order_delta"] <= 0 or metrics["fake_delta"] <= 0:
        fail(f"behavior:{metrics['order_delta']}:{metrics['fake_delta']}")
        return None
    recovered = _render_actions(out_h, out_w, ffill, real_actions)
    if not np.array_equal(recovered, actual):
        fail("recover_mismatch")
        return None
    return grid


def generate_input() -> np.ndarray:
    modes = list(_COMPONENT_MODES)
    random.shuffle(modes)
    primary_total, primary_fake = modes[0]
    primary_budget = max(1, _GEN_RETRIES // 2)

    for _ in range(primary_budget):
        grid = _gen_attempt(total_count=int(primary_total), fake_count=int(primary_fake))
        if grid is not None:
            return grid

    for total_count, fake_count in modes[1:]:
        for _ in range(max(1, (_GEN_RETRIES - primary_budget) // max(1, len(modes) - 1))):
            grid = _gen_attempt(total_count=int(total_count), fake_count=int(fake_count))
            if grid is not None:
                return grid
    raise RuntimeError("failed to generate a valid a6f40cea combo input")
