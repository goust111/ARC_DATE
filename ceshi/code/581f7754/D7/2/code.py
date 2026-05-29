# Variant of 581f7754
# Dimension: D7 — 网格形状：输出裁切方式 (grid_shape)
# Change: 输出网格与输入网格尺寸完全一致，不进行裁切或扩展
#   → 仅输出锚点所在边的条带

from __future__ import annotations

from collections import Counter
import random

import numpy as np

from arc_utils import find_components, get_bg, neighbors4, paste


# Design analysis for the D7 variant
# ----------------------------------
# spec_variant:
#   core_change:
#     from: "Solve on the full input canvas and return the full solved canvas."
#     to: "Solve on the full canvas first, then crop to the narrow strip attached to the anchor edge."
#   concretization:
#     input:
#       - Same object vocabulary as the original task: isolated border anchors plus interior sprites.
#       - To make a single strip unambiguous, every generated example uses anchors from one edge only:
#         top, bottom, left, or right.
#       - The sprites that will align to those anchors are already constrained to a narrow band next to
#         that same edge, because the original move rule preserves one coordinate:
#         top/bottom anchors only change column; left/right anchors only change row.
#     output:
#       - First compute the original full solved grid.
#       - Then keep only the full-width or full-height strip containing the solved sprites and the anchors:
#         top    -> rows [0 .. max_used_row]
#         bottom -> rows [min_used_row .. H-1]
#         left   -> cols [0 .. max_used_col]
#         right  -> cols [min_used_col .. W-1]
#       - The long dimension is preserved; only the dimension orthogonal to the chosen edge is cropped.
#   behavior_trace:
#     example:
#       - Original top-edge case: anchor at (0, 8), marker at (3, 2) => sprite slides horizontally to
#         marker column 8, but the output remains HxW.
#       - Variant top-edge case: the same full-grid slide still happens, but if all solved sprite pixels
#         lie in rows 0..5, the returned output is only rows 0..5 of the solved grid.
#       - Difference: movement logic is unchanged; only the reported field of view is narrowed.
#   complexity_calibration:
#     - Mixed-edge anchors would make the crop ambiguous or force a union of multiple strips, which would
#       be visually noisy and under-supported by 2-4 ARC examples. The generator therefore uses one edge
#       per sample, but still varies that edge across samples.
#     - Cropping by exact bounding box in both dimensions would turn the rule into generic bbox extraction.
#       Keeping full width/full height preserves the "edge strip" concept and matches the D7 description.
#   implicit_assumption_audit:
#     - Original anchor detection assumed the returned canvas keeps every border anchor visible; still true.
#     - Original translation code assumed a single full-size canvas; still used internally in `_solve_full`.
#     - New crop logic assumes the relevant anchors belong to one dominant edge. `main()` chooses the
#       majority edge if several exist; `generate_input()` guarantees all anchors use one edge, so the
#       assumption is satisfied for produced samples.
#     - Crop extent must be measured from the actual solved placements, not from input scatter positions.
#       The solver therefore records written target cells during pasting.
#   placement_legality:
#     - Output-first generation is retained: build a full solved canvas, then scatter along the inverse axis.
#     - Solved placements are restricted to a thin band near the chosen edge so the crop is genuinely small.
#     - Standard non-overlap / non-touching checks are preserved from the original generator.
#   visual_observability:
#     - The user can see that the output always keeps the edge with the anchors and discards the distant,
#       empty part of the grid.
#     - Edge choice varies among top/bottom/left/right, so the crop orientation is inferable from samples.
#   cleanliness:
#     - Only one active edge is present per sample, avoiding multi-strip clutter.
#     - Colors still serve only as anchor/marker matching cues; no hidden numeric ordering is introduced.


def main(grid: np.ndarray) -> np.ndarray:
    """
    Slide marker-tagged sprites to their matching border anchors, then keep only the anchor-side strip.

    Input structure:
    - Background is the most frequent color.
    - Interior 8-connected sprites contain a rarer marker color that matches an isolated border anchor.
    - All meaningful anchors in a generated instance lie on one border edge, which defines the output strip.

    Rule:
    - Match each sprite to the anchor with the same marker color and apply the original axis-specific slide.
    - After the full solved canvas is formed, crop away the far side of the grid and keep only the strip
      attached to the anchor edge that contains the solved objects.

    Key insight: The anchor edge still decides the slide axis, but now it also decides which side of the
    solved canvas remains visible in the final output.
    """
    solved, anchors_by_color, placements = _solve_full(grid)
    return _crop_solved_strip(solved, anchors_by_color, placements)


def _edge_name(r: int, c: int, h: int, w: int) -> str:
    if int(r) == 0:
        return "top"
    if int(r) == int(h - 1):
        return "bottom"
    if int(c) == 0:
        return "left"
    return "right"


def _detect_anchors(g: np.ndarray, bg: int) -> dict[int, tuple[int, int]]:
    h, w = g.shape

    def is_border(r: int, c: int) -> bool:
        return int(r) in (0, int(h - 1)) or int(c) in (0, int(w - 1))

    anchors_by_color: dict[int, tuple[int, int]] = {}
    for r in range(h):
        for c in range(w):
            if not is_border(r, c):
                continue
            v = int(g[r, c])
            if v == int(bg):
                continue
            ok = True
            for rr, cc in neighbors4(int(r), int(c)):
                if 0 <= rr < h and 0 <= cc < w and int(g[rr, cc]) != int(bg):
                    ok = False
                    break
            if ok and v not in anchors_by_color:
                anchors_by_color[int(v)] = (int(r), int(c))
    return anchors_by_color


def _component_marker_color(g: np.ndarray, coords: np.ndarray) -> int | None:
    vals = [int(g[int(r), int(c)]) for r, c in coords]
    cnt = Counter(vals)
    if len(cnt) <= 1:
        return None
    return int(min(cnt.items(), key=lambda kv: (int(kv[1]), int(kv[0])))[0])


def _marker_pixel(g: np.ndarray, coords: np.ndarray, marker: int) -> tuple[int, int] | None:
    pts = [(int(r), int(c)) for r, c in coords if int(g[int(r), int(c)]) == int(marker)]
    if not pts:
        return None
    if len(pts) == 1:
        return pts[0]
    comp_set = {(int(r), int(c)) for r, c in coords}
    for r, c in pts:
        for rr, cc in neighbors4(int(r), int(c)):
            if (int(rr), int(cc)) in comp_set and int(g[int(rr), int(cc)]) != int(marker):
                return (int(r), int(c))
    return pts[0]


def _solve_full(grid: np.ndarray) -> tuple[np.ndarray, dict[int, tuple[int, int]], list[dict]]:
    g = np.array(grid, dtype=int)
    h, w = g.shape
    bg = int(get_bg(g))
    anchors_by_color = _detect_anchors(g, bg)
    assert anchors_by_color

    anchor_cells = {(int(r), int(c)) for (r, c) in anchors_by_color.values()}
    comps = find_components(g, bg=bg, connectivity=8, monochromatic=False, return_format="dict")

    moves: list[tuple[np.ndarray, int, int, str]] = []
    for comp in comps:
        coords = comp["coords"]
        if coords.size == 0:
            continue

        marker_col = _component_marker_color(g, coords)
        if marker_col is None:
            if int(coords.shape[0]) != 1:
                continue
            r, c = map(int, coords[0])
            v = int(g[r, c])
            if v == int(bg) or v not in anchors_by_color or (r, c) in anchor_cells:
                continue
            marker_col = int(v)

        if int(marker_col) not in anchors_by_color:
            continue

        mp = _marker_pixel(g, coords, int(marker_col))
        if mp is None:
            continue

        ar, ac = anchors_by_color[int(marker_col)]
        edge = _edge_name(ar, ac, h, w)
        horizontal = edge in {"top", "bottom"}

        dr = 0
        dc = 0
        if horizontal:
            dc = int(ac - mp[1])
        else:
            dr = int(ar - mp[0])
        moves.append((coords, int(dr), int(dc), edge))

    out = g.copy()
    for coords, _dr, _dc, _edge in moves:
        out[coords[:, 0], coords[:, 1]] = int(bg)

    placements: list[dict] = []
    for coords, dr, dc, edge in moves:
        written: list[tuple[int, int]] = []
        for r, c in coords:
            rr = int(r) + int(dr)
            cc = int(c) + int(dc)
            if 0 <= rr < h and 0 <= cc < w and int(out[rr, cc]) == int(bg):
                out[rr, cc] = int(g[int(r), int(c)])
                written.append((int(rr), int(cc)))
        placements.append({"edge": edge, "cells": written})

    return out, anchors_by_color, placements


def _crop_solved_strip(
    solved: np.ndarray, anchors_by_color: dict[int, tuple[int, int]], placements: list[dict]
) -> np.ndarray:
    h, w = solved.shape
    edge_counts = Counter(_edge_name(r, c, h, w) for r, c in anchors_by_color.values())
    focus_edge = max(edge_counts.items(), key=lambda kv: int(kv[1]))[0]

    focus_cells: list[tuple[int, int]] = []
    for r, c in anchors_by_color.values():
        if _edge_name(r, c, h, w) == focus_edge:
            focus_cells.append((int(r), int(c)))
    for placement in placements:
        if placement["edge"] == focus_edge:
            focus_cells.extend((int(r), int(c)) for r, c in placement["cells"])

    assert focus_cells
    rows = [int(r) for r, _c in focus_cells]
    cols = [int(c) for _r, c in focus_cells]

    if focus_edge == "top":
        return solved[: max(rows) + 1, :].copy()
    if focus_edge == "bottom":
        return solved[min(rows) :, :].copy()
    if focus_edge == "left":
        return solved[:, : max(cols) + 1].copy()
    return solved[:, min(cols) :].copy()


# --------------- generator (D7 variant) ---------------

_COLORS = tuple(range(10))
_BG_COLOR_RANGE = (1, 8)
_GRID_HEIGHT_RANGE = (10, 22)
_GRID_WIDTH_RANGE = (8, 22)
_FOCUS_MIN_DIM = 12

_NEIGHBORS4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
_INNER_BORDER_GAP = 1
_ANCHOR_EDGE_MARGIN = 2
_ISOLATION_RADIUS = 1

_ANCHOR_COUNT_RANGE = (1, 3)
_SPRITE_COUNT_RANGE = (3, 5)
_BODY_AREA_RANGE = (4, 13)
_GEN_RETRY_LIMIT = 400
_SPRITE_BUILD_RETRIES = 30
_STRIP_THICKNESS_RANGE = (4, 8)
_EDGE_CHOICES = ("top", "bottom", "left", "right")
_MIN_REDUCTION = 2
_MAX_AREA_RATIO = 0.82

_BODY_HEIGHT_RANGE = (2, 5)
_BODY_WIDTH_RANGE = (3, 7)
_FRAME_HEIGHT_RANGE = (3, 5)
_FRAME_WIDTH_RANGE = (3, 7)
_FRAME_MIN_AREA = 8
_FRAME_MAX_GAP = 3
_FRAME_BODY_THRESHOLD = 0.28
_WIDE_BODY_THRESHOLD = 0.55
_SYMMETRIC_BODY_THRESHOLD = 0.82
_SYMMETRIC_TRANSPOSE_MIN_HEIGHT = 3
_SYMMETRIC_TRANSPOSE_PROB = 0.45
_MARKER_INTERIOR_PROB = 0.65
_FLIP_PROB = 0.5

_DISTRACTOR_PROB = 0.12
_DISTRACTOR_COUNT_RANGE = (1, 3)

_GROW_BODY_SEEDS = [
    {(0, 0), (0, 1), (0, 2), (1, 0)},
    {(0, 0), (0, 1), (0, 2), (1, 2)},
    {(0, 0), (1, 0), (1, 1), (1, 2)},
    {(0, 2), (1, 0), (1, 1), (1, 2)},
    {(0, 1), (1, 0), (1, 1), (1, 2)},
]


def _norm(cells):
    r0 = min(r for r, _c in cells)
    c0 = min(c for _r, c in cells)
    return {(r - r0, c - c0) for r, c in cells}


def _bbox(cells):
    return (
        min(r for r, _c in cells),
        min(c for _r, c in cells),
        max(r for r, _c in cells),
        max(c for _r, c in cells),
    )


def _flip(cells):
    cells = set(cells)
    if random.random() < _FLIP_PROB:
        mc = max(c for _r, c in cells)
        cells = {(r, mc - c) for r, c in cells}
    if random.random() < _FLIP_PROB:
        mr = max(r for r, _c in cells)
        cells = {(mr - r, c) for r, c in cells}
    return _norm(cells)


def _inclusive_range(bounds):
    return range(int(bounds[0]), int(bounds[1]) + 1)


def _frontier_cells(cells, height, width):
    frontier = []
    for r, c in cells:
        for dr, dc in _NEIGHBORS4:
            rr, cc = r + dr, c + dc
            if 0 <= rr < height and 0 <= cc < width and (rr, cc) not in cells:
                frontier.append((rr, cc))
    return frontier


def _grow_body(area):
    choices = [
        (h, w)
        for h in _inclusive_range(_BODY_HEIGHT_RANGE)
        for w in _inclusive_range(_BODY_WIDTH_RANGE)
        if h * w >= area
    ]
    h, w = random.choice(choices)
    cells = set(random.choice(_GROW_BODY_SEEDS))
    while len(cells) < area:
        frontier = _frontier_cells(cells, h, w)
        if not frontier:
            return _grow_body(area)
        cells.add(random.choice(frontier))
    return _flip(cells)


def _symmetric_body(area):
    choices = []
    for height in _inclusive_range(_BODY_HEIGHT_RANGE):
        extra = area - height
        if extra >= 2 and extra % 2 == 0 and extra // 2 <= height:
            choices.append((height, extra // 2))
    if not choices:
        return _grow_body(area)

    height, pair_count = random.choice(choices)
    cells = {(r, 1) for r in range(height)}
    rows = list(range(height))
    random.shuffle(rows)
    for r in rows[:pair_count]:
        cells.add((r, 0))
        cells.add((r, 2))

    if height >= _SYMMETRIC_TRANSPOSE_MIN_HEIGHT and random.random() < _SYMMETRIC_TRANSPOSE_PROB:
        cells = {(c, r) for r, c in cells}
    return _flip(cells)


def _wide_body(area):
    width = random.randint(_BODY_WIDTH_RANGE[0], min(_BODY_WIDTH_RANGE[1], area - 1))
    height = random.choice([h for h in _inclusive_range(_BODY_HEIGHT_RANGE) if h * width >= area])
    cells = {(0, c) for c in range(width)}
    cells.add((1, random.randrange(width)))
    while len(cells) < area:
        frontier = _frontier_cells(cells, height, width)
        cells.add(random.choice(frontier))
    return _flip(cells)


def _frame_body(area):
    choices = []
    for h in _inclusive_range(_FRAME_HEIGHT_RANGE):
        for w in _inclusive_range(_FRAME_WIDTH_RANGE):
            perim = 2 * h + 2 * w - 4
            removable = 2 * max(0, h - 2) + 2 * max(0, w - 2)
            if perim >= area and perim - area <= min(_FRAME_MAX_GAP, removable):
                choices.append((h, w))
    if not choices:
        return _grow_body(area)

    h, w = random.choice(choices)
    cells = {
        (r, c)
        for r in range(h)
        for c in range(w)
        if r in (0, h - 1) or c in (0, w - 1)
    }
    gap = len(cells) - area
    if gap:
        removable = [
            (r, c)
            for r, c in cells
            if not (r in (0, h - 1) and c in (0, w - 1))
        ]
        for cell in random.sample(removable, gap):
            cells.remove(cell)
    return _flip(cells)


def _make_body(area):
    x = random.random()
    if area >= _FRAME_MIN_AREA and x < _FRAME_BODY_THRESHOLD:
        return _frame_body(area)
    if x < _WIDE_BODY_THRESHOLD:
        return _wide_body(area)
    if x < _SYMMETRIC_BODY_THRESHOLD:
        return _symmetric_body(area)
    return _grow_body(area)


def _marker_cell(body):
    r0, c0, r1, c1 = _bbox(body)
    interior, exterior = set(), set()
    for r, c in body:
        for dr, dc in _NEIGHBORS4:
            p = (r + dr, c + dc)
            if p in body:
                continue
            if r0 <= p[0] <= r1 and c0 <= p[1] <= c1:
                interior.add(p)
            else:
                exterior.add(p)
    pool = list(interior if interior and random.random() < _MARKER_INTERIOR_PROB else (exterior | interior))
    return random.choice(pool)


def _make_sprite(area, body_color, marker_color, bg):
    body = _make_body(area)
    marker = _marker_cell(body)
    all_cells = set(body) | {marker}
    r0, c0, r1, c1 = _bbox(all_cells)
    patch = np.full((r1 - r0 + 1, c1 - c0 + 1), int(bg), dtype=int)
    for r, c in body:
        patch[r - r0, c - c0] = int(body_color)
    patch[marker[0] - r0, marker[1] - c0] = int(marker_color)
    return {"patch": patch, "offset": (marker[0] - r0, marker[1] - c0)}


def _top_left(sprite, marker_r, marker_c):
    off_r, off_c = sprite["offset"]
    return int(marker_r) - int(off_r), int(marker_c) - int(off_c)


def _can_place(canvas, sprite, marker_r, marker_c, bg):
    patch = sprite["patch"]
    top, left = _top_left(sprite, marker_r, marker_c)
    h, w = patch.shape
    grid_h, grid_w = canvas.shape
    if top < _INNER_BORDER_GAP or left < _INNER_BORDER_GAP:
        return False
    if top + h > grid_h - _INNER_BORDER_GAP or left + w > grid_w - _INNER_BORDER_GAP:
        return False
    mask = patch != int(bg)
    view = canvas[top : top + h, left : left + w]
    if bool((view[mask] != int(bg)).any()):
        return False
    return not _touches_foreground(canvas, mask, top, left, bg)


def _touches_foreground(canvas, mask, top, left, bg):
    h, w = mask.shape
    radius = _ISOLATION_RADIUS
    occupied = canvas[top - radius : top + h + radius, left - radius : left + w + radius] != int(bg)
    touch_zone = np.zeros((h + 2 * radius, w + 2 * radius), dtype=bool)
    span = 2 * radius + 1
    for dr in range(span):
        for dc in range(span):
            touch_zone[dr : dr + h, dc : dc + w] |= mask
    return bool((occupied & touch_zone).any())


def _place(canvas, sprite, marker_r, marker_c, bg):
    top, left = _top_left(sprite, marker_r, marker_c)
    paste(canvas, sprite["patch"], top, left, transparent=int(bg))


def _sprite_cells(sprite, marker_r, marker_c, bg):
    patch = sprite["patch"]
    top, left = _top_left(sprite, marker_r, marker_c)
    cells = []
    for rr, cc in np.argwhere(patch != int(bg)):
        cells.append((int(top + rr), int(left + cc)))
    return cells


def _anchor_ok(canvas, r, c, bg):
    h, w = canvas.shape
    if not (r in (0, h - 1) or c in (0, w - 1)):
        return False
    if r in (0, h - 1) and c in (0, w - 1):
        return False
    return _isolated_bg_cell(canvas, r, c, bg)


def _isolated_bg_cell(canvas, r, c, bg):
    h, w = canvas.shape
    radius = _ISOLATION_RADIUS
    r0, r1 = max(0, r - radius), min(h, r + radius + 1)
    c0, c1 = max(0, c - radius), min(w, c + radius + 1)
    return not bool((canvas[r0:r1, c0:c1] != int(bg)).any())


def _edge_anchor_candidates(canvas, bg, focus_edge):
    h, w = canvas.shape
    rows = list(range(_ANCHOR_EDGE_MARGIN, h - _ANCHOR_EDGE_MARGIN))
    rows = rows or list(range(_INNER_BORDER_GAP, h - _INNER_BORDER_GAP))
    cols = list(range(_ANCHOR_EDGE_MARGIN, w - _ANCHOR_EDGE_MARGIN))
    cols = cols or list(range(_INNER_BORDER_GAP, w - _INNER_BORDER_GAP))
    if focus_edge == "top":
        candidates = [(0, c) for c in cols]
    elif focus_edge == "bottom":
        candidates = [(h - 1, c) for c in cols]
    elif focus_edge == "left":
        candidates = [(r, 0) for r in rows]
    else:
        candidates = [(r, w - 1) for r in rows]
    random.shuffle(candidates)
    return [(r, c) for r, c in candidates if _anchor_ok(canvas, r, c, bg)]


def _put_anchor(canvas, r, c, color):
    paste(canvas, np.array([[int(color)]], dtype=int), int(r), int(c))


def _pick_strip_thickness(focus_edge, shape):
    dim = int(shape[0]) if focus_edge in {"top", "bottom"} else int(shape[1])
    lo = min(_STRIP_THICKNESS_RANGE[0], max(2, dim - 2))
    hi = min(_STRIP_THICKNESS_RANGE[1], dim - 2)
    if hi < lo:
        hi = lo
    return random.randint(lo, hi)


def _fits_focus_band(sprite, marker_r, marker_c, focus_edge, band_thickness, bg, shape):
    grid_h, grid_w = shape
    cells = _sprite_cells(sprite, marker_r, marker_c, bg)
    rows = [r for r, _c in cells]
    cols = [c for _r, c in cells]
    if focus_edge == "top":
        return min(rows) >= 0 and max(rows) < int(band_thickness)
    if focus_edge == "bottom":
        return min(rows) >= int(grid_h - band_thickness) and max(rows) < int(grid_h)
    if focus_edge == "left":
        return min(cols) >= 0 and max(cols) < int(band_thickness)
    return min(cols) >= int(grid_w - band_thickness) and max(cols) < int(grid_w)


def _aligned_positions(canvas, sprite, anchor, bg, focus_edge, band_thickness):
    grid_h, grid_w = canvas.shape
    if anchor["axis"] == "h":
        return [
            (r, anchor["c"])
            for r in range(_INNER_BORDER_GAP, grid_h - _INNER_BORDER_GAP)
            if _can_place(canvas, sprite, r, anchor["c"], bg)
            and _fits_focus_band(sprite, r, anchor["c"], focus_edge, band_thickness, bg, canvas.shape)
        ]
    return [
        (anchor["r"], c)
        for c in range(_INNER_BORDER_GAP, grid_w - _INNER_BORDER_GAP)
        if _can_place(canvas, sprite, anchor["r"], c, bg)
        and _fits_focus_band(sprite, anchor["r"], c, focus_edge, band_thickness, bg, canvas.shape)
    ]


def _scatter_positions(canvas, sprite, anchor, marker, bg):
    grid_h, grid_w = canvas.shape
    mr, mc = marker
    if anchor["axis"] == "h":
        return [
            (mr, c)
            for c in range(_INNER_BORDER_GAP, grid_w - _INNER_BORDER_GAP)
            if c != mc and _can_place(canvas, sprite, mr, c, bg)
        ]
    return [
        (r, mc)
        for r in range(_INNER_BORDER_GAP, grid_h - _INNER_BORDER_GAP)
        if r != mr and _can_place(canvas, sprite, r, mc, bg)
    ]


def _body_color(bg, marker, used):
    pool = [c for c in _COLORS if c not in {int(bg), int(marker)} and c not in used]
    if not pool:
        pool = [c for c in _COLORS if c not in {int(bg), int(marker)}]
    return random.choice(pool)


def _bg_is_mode(grid, bg):
    counts = Counter(map(int, grid.ravel()))
    return counts[int(bg)] > max((n for c, n in counts.items() if c != int(bg)), default=0)


def _crop_from_focus_cells(out, focus_edge, focus_cells):
    rows = [int(r) for r, _c in focus_cells]
    cols = [int(c) for _r, c in focus_cells]
    if focus_edge == "top":
        return out[: max(rows) + 1, :].copy()
    if focus_edge == "bottom":
        return out[min(rows) :, :].copy()
    if focus_edge == "left":
        return out[:, : max(cols) + 1].copy()
    return out[:, min(cols) :].copy()


def _variant_visible(inp, expected, focus_edge):
    in_h, in_w = inp.shape
    out_h, out_w = expected.shape
    if focus_edge in {"top", "bottom"}:
        if int(out_h) > int(in_h - _MIN_REDUCTION):
            return False
    else:
        if int(out_w) > int(in_w - _MIN_REDUCTION):
            return False
    return (int(out_h) * int(out_w)) / float(int(in_h) * int(in_w)) <= _MAX_AREA_RATIO


def _new_canvas_pair(focus_edge):
    h_lo, h_hi = _GRID_HEIGHT_RANGE
    w_lo, w_hi = _GRID_WIDTH_RANGE
    if focus_edge in {"top", "bottom"}:
        h_lo = max(h_lo, _FOCUS_MIN_DIM)
    else:
        w_lo = max(w_lo, _FOCUS_MIN_DIM)
    h = random.randint(h_lo, h_hi)
    w = random.randint(w_lo, w_hi)
    bg = random.randint(*_BG_COLOR_RANGE)
    out = np.full((h, w), int(bg), dtype=int)
    inp = np.full((h, w), int(bg), dtype=int)
    return bg, out, inp


def _place_anchors(out, inp, bg, focus_edge):
    anchor_colors = random.sample([c for c in _COLORS if c != bg], random.randint(*_ANCHOR_COUNT_RANGE))
    anchors = []
    for color in anchor_colors:
        candidates = _edge_anchor_candidates(out, bg, focus_edge)
        if not candidates:
            return None
        r, c = random.choice(candidates)
        axis = "h" if focus_edge in {"top", "bottom"} else "v"
        anchor = {"color": int(color), "r": int(r), "c": int(c), "axis": axis}
        anchors.append(anchor)
        _put_anchor(out, r, c, color)
        _put_anchor(inp, r, c, color)
    return anchor_colors, anchors


def _marker_color_sequence(anchor_colors):
    n_sprites = random.randint(*_SPRITE_COUNT_RANGE)
    marker_colors = anchor_colors[:]
    while len(marker_colors) < n_sprites:
        marker_colors.append(random.choice(anchor_colors))
    random.shuffle(marker_colors)
    return marker_colors


def _place_aligned_sprites(out, anchors, marker_colors, bg, focus_edge, band_thickness, focus_cells):
    by_color = {a["color"]: a for a in anchors}
    used_body = set()
    sprites = []
    for marker in marker_colors:
        anchor = by_color[int(marker)]
        placed = False
        for _ in range(_SPRITE_BUILD_RETRIES):
            body = _body_color(bg, marker, used_body)
            sprite = _make_sprite(random.randint(*_BODY_AREA_RANGE), body, marker, bg)
            positions = _aligned_positions(out, sprite, anchor, bg, focus_edge, band_thickness)
            if not positions:
                continue
            marker_pos = random.choice(positions)
            _place(out, sprite, marker_pos[0], marker_pos[1], bg)
            focus_cells.extend(_sprite_cells(sprite, marker_pos[0], marker_pos[1], bg))
            sprites.append((sprite, anchor, marker_pos))
            used_body.add(body)
            placed = True
            break
        if not placed:
            return None
    return sprites


def _scatter_sprites(inp, sprites, bg):
    random.shuffle(sprites)
    for sprite, anchor, marker_pos in sprites:
        positions = _scatter_positions(inp, sprite, anchor, marker_pos, bg)
        if not positions:
            return False
        in_pos = random.choice(positions)
        _place(inp, sprite, in_pos[0], in_pos[1], bg)
    return True


def _add_distractors(out, inp, anchors, bg, focus_edge, band_thickness, focus_cells):
    if random.random() >= _DISTRACTOR_PROB:
        return
    anchor = random.choice(anchors)
    sprite = {"patch": np.array([[anchor["color"]]], dtype=int), "offset": (0, 0)}
    for _ in range(random.randint(*_DISTRACTOR_COUNT_RANGE)):
        positions = _aligned_positions(out, sprite, anchor, bg, focus_edge, band_thickness)
        if not positions:
            return
        marker_pos = random.choice(positions)
        scatter = _scatter_positions(inp, sprite, anchor, marker_pos, bg)
        if not scatter:
            return
        _place(out, sprite, marker_pos[0], marker_pos[1], bg)
        _place(inp, sprite, *random.choice(scatter), bg)
        focus_cells.extend(_sprite_cells(sprite, marker_pos[0], marker_pos[1], bg))


def _generated_grid_is_valid(inp, expected, bg):
    if not _bg_is_mode(inp, bg):
        return False
    return bool(np.array_equal(main(inp), expected))


def _gen_attempt():
    focus_edge = random.choice(_EDGE_CHOICES)
    bg, out, inp = _new_canvas_pair(focus_edge)
    band_thickness = _pick_strip_thickness(focus_edge, out.shape)

    placed = _place_anchors(out, inp, bg, focus_edge)
    if placed is None:
        return None
    anchor_colors, anchors = placed
    focus_cells = [(int(a["r"]), int(a["c"])) for a in anchors]

    marker_colors = _marker_color_sequence(anchor_colors)
    sprites = _place_aligned_sprites(out, anchors, marker_colors, bg, focus_edge, band_thickness, focus_cells)
    if sprites is None:
        return None

    if not _scatter_sprites(inp, sprites, bg):
        return None

    _add_distractors(out, inp, anchors, bg, focus_edge, band_thickness, focus_cells)
    expected = _crop_from_focus_cells(out, focus_edge, focus_cells)
    if not _variant_visible(inp, expected, focus_edge):
        return None
    if not _generated_grid_is_valid(inp, expected, bg):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(_GEN_RETRY_LIMIT):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid 581f7754 D7 input")
