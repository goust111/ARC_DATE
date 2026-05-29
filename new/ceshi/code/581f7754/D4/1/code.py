# Variant of 581f7754
# Dimension: D4 — 目标位置：锚点类型与位置 (dst)
# Change: 边界上的孤立单像素（非角落），4-邻域均为背景色，1-3个不同颜色锚点... → 内部锚点（十字线交叉点）
#
# Design analysis
# ---------------
# variant_spec:
#   core_change:
#     old_dst:
#       A sprite matched a same-colored isolated border singleton.
#       The border side encoded the slide axis, and only one coordinate was constrained.
#     new_dst:
#       Each target is an interior crosshair:
#       - one full-span rail (same orientation for all anchors in one grid),
#       - one short perpendicular tick crossing the rail,
#       - a colored center pixel at the crossing.
#       The sprite's marker slides along that rail until it reaches the center pixel.
#   concrete_pixel_form:
#     rail_color:
#       One shared rail color is used for every long line in the grid.
#       This keeps the long tracks visually grouped and easy to parse.
#     anchor_center:
#       The center pixel color is unique per anchor and matches the moving sprite's marker color.
#     sprite_structure:
#       Still a 2-color sprite: one body color plus one minority marker color.
#       To keep the rail legible, only the marker is allowed to lie on the rail row/column;
#       the body is generated entirely on one side of the rail.
#   behavior_trace:
#     old_logic_example:
#       A top-edge anchor at column 5 forced horizontal motion so the marker column became 5.
#       Several sprites could share that anchor because only the column mattered.
#     new_logic_example:
#       A horizontal rail on row 6 with crosshair center at (6, 9) means the marker must end
#       exactly at (6, 9), sliding only left/right on row 6.
#       The old "many sprites per anchor" pattern would now collide at the same center cell,
#       so the generator switches to one sprite per crosshair.
#   complexity_calibration:
#     kept_from_original:
#       Grid size range, body size range, irregular body generation, output oracle check.
#     adapted_but_not_simplified_away:
#       There are still 1-3 anchors and 1-3 sprites, with repeated examples of the same
#       rail orientation within a grid.
#       The long rails are the new visually dominant clue, replacing the old border singleton cue.
#     necessary_restrictions:
#       Mixed horizontal and vertical full-span rails would intersect, so each grid uses one
#       shared orientation. The dataset still varies orientation across samples.
#   implicit_assumption_audit:
#     old_main_dependencies_that_break:
#       - "anchor must be on the border": false now.
#       - "anchor is an isolated singleton": false now, because the center is embedded in a crosshair.
#       - "all foreground components are candidate sprites": false now, because rails are permanent
#         foreground structure and must be removed before component extraction.
#       - "clearing a moved sprite always reveals bg": false now, because the marker sits on a rail
#         cell whose underlying value must be restored to the rail color.
#     new_main_assumptions:
#       - Rail cells are monochrome, long, and easy to detect from their nearly full-span row/column.
#       - Body colors never reuse the rail color, so removing rails does not damage sprites.
#       - Only the marker lies on the rail; the body occupies bg cells only.
#       The generator enforces all three.
#   placement_legality:
#     rails:
#       Chosen away from the border for the center point, with enough room for the short tick.
#       Because the long rails span the whole row/column, anchors of different orientations are
#       not mixed in one grid.
#     sprites:
#       Output is built first at the solved center position, then each sprite is scattered to a
#       different coordinate on the same rail. Placement checks allow the marker to overwrite the
#       rail cell but require every body cell to remain in bg.
#   visual_observability:
#     A human can see:
#       - long rails as the new reference object,
#       - crosshair centers as the destinations,
#       - same-colored marker pixels on the rails as the moving sprite handles.
#     The change is visible in 2-3 examples because there are no border singleton anchors at all,
#     and the long rails dominate the composition.
#   cleanliness:
#     No extra distractor singletons are added in this variant.
#     That avoids clutter near the rails and keeps the new crosshair cue unambiguous.

from __future__ import annotations

from collections import Counter
import random

import numpy as np

from arc_utils import find_components, get_bg, neighbors4, paste


def main(grid: np.ndarray) -> np.ndarray:
    """
    Slide multi-color sprites along interior crosshair rails until their marker pixels reach the matching centers.

    Input structure:
    - bg = most common color.
    - One or more interior crosshairs: a long monochrome rail plus a short perpendicular tick, with a colored center pixel.
    - Multi-color sprites whose minority-color marker matches one crosshair center color; only the marker sits on that rail.

    Rule:
    - Detect each crosshair by its long rail, then match sprites to the center pixel sharing the marker color.
    - A horizontal rail means horizontal sliding to the center column; a vertical rail means vertical sliding to the center row.
    - When a sprite vacates a rail cell, restore the underlying rail; clipped pixels still disappear at the boundary.

    Key insight: The long internal line, not the border, now encodes the motion axis and target location; the colored crossing pixel is the destination.
    """
    g = np.array(grid, dtype=int)
    H, W = g.shape
    bg = int(get_bg(g))

    def detect_tracks() -> list[dict]:
        tracks: list[dict] = []

        for rail_color in _COLORS:
            if int(rail_color) == int(bg):
                continue
            for r in range(1, H - 1):
                rail_count = sum(int(g[r, c]) == int(rail_color) for c in range(W))
                if rail_count < W - 2:
                    continue
                center_cols = [
                    c
                    for c in range(1, W - 1)
                    if int(g[r, c]) not in {int(bg), int(rail_color)}
                    and (
                        int(g[r - 1, c]) == int(rail_color)
                        or int(g[r + 1, c]) == int(rail_color)
                    )
                ]
                if len(center_cols) != 1:
                    continue
                c = int(center_cols[0])
                rail_cells = {(int(r), int(cc)) for cc in range(W)}
                for rr in range(r - 1, -1, -1):
                    if int(g[rr, c]) != int(rail_color):
                        break
                    rail_cells.add((int(rr), int(c)))
                for rr in range(r + 1, H):
                    if int(g[rr, c]) != int(rail_color):
                        break
                    rail_cells.add((int(rr), int(c)))
                tracks.append(
                    {
                        "axis": "h",
                        "r": int(r),
                        "c": int(c),
                        "center_color": int(g[r, c]),
                        "rail_color": int(rail_color),
                        "rail_cells": rail_cells,
                    }
                )

        if tracks:
            return tracks

        for rail_color in _COLORS:
            if int(rail_color) == int(bg):
                continue
            for c in range(1, W - 1):
                rail_count = sum(int(g[r, c]) == int(rail_color) for r in range(H))
                if rail_count < H - 2:
                    continue
                center_rows = [
                    r
                    for r in range(1, H - 1)
                    if int(g[r, c]) not in {int(bg), int(rail_color)}
                    and (
                        int(g[r, c - 1]) == int(rail_color)
                        or int(g[r, c + 1]) == int(rail_color)
                    )
                ]
                if len(center_rows) != 1:
                    continue
                r = int(center_rows[0])
                rail_cells = {(int(rr), int(c)) for rr in range(H)}
                for cc in range(c - 1, -1, -1):
                    if int(g[r, cc]) != int(rail_color):
                        break
                    rail_cells.add((int(r), int(cc)))
                for cc in range(c + 1, W):
                    if int(g[r, cc]) != int(rail_color):
                        break
                    rail_cells.add((int(r), int(cc)))
                tracks.append(
                    {
                        "axis": "v",
                        "r": int(r),
                        "c": int(c),
                        "center_color": int(g[r, c]),
                        "rail_color": int(rail_color),
                        "rail_cells": rail_cells,
                    }
                )
        return tracks

    tracks = detect_tracks()
    assert tracks

    tracks_by_color = {int(t["center_color"]): t for t in tracks}
    under = np.full_like(g, int(bg))
    erase_cells: set[tuple[int, int]] = set()
    center_cells: set[tuple[int, int]] = set()
    for track in tracks:
        for r, c in track["rail_cells"]:
            under[int(r), int(c)] = int(track["rail_color"])
            if int(g[int(r), int(c)]) == int(track["rail_color"]):
                erase_cells.add((int(r), int(c)))
        under[int(track["r"]), int(track["c"])] = int(track["center_color"])
        center_cells.add((int(track["r"]), int(track["c"])))

    work = g.copy()
    for r, c in erase_cells:
        work[int(r), int(c)] = int(bg)

    comps = find_components(work, bg=bg, connectivity=8, monochromatic=False, return_format="dict")

    def component_marker_color(coords: np.ndarray) -> int | None:
        vals = [int(work[int(r), int(c)]) for r, c in coords]
        cnt = Counter(vals)
        if len(cnt) <= 1:
            return None
        return int(min(cnt.items(), key=lambda kv: (int(kv[1]), int(kv[0])))[0])

    def marker_pixel(coords: np.ndarray, marker: int) -> tuple[int, int] | None:
        pts = [(int(r), int(c)) for r, c in coords if int(work[int(r), int(c)]) == int(marker)]
        if not pts:
            return None
        if len(pts) == 1:
            return pts[0]
        comp_set = {(int(r), int(c)) for r, c in coords}
        for r, c in pts:
            for rr, cc in neighbors4(int(r), int(c)):
                if (int(rr), int(cc)) in comp_set and int(work[int(rr), int(cc)]) != int(marker):
                    return (int(r), int(c))
        return pts[0]

    moves: list[tuple[np.ndarray, int, int]] = []
    for comp in comps:
        coords = comp["coords"]
        if coords.size == 0:
            continue

        marker_col = component_marker_color(coords)
        if marker_col is None:
            if int(coords.shape[0]) != 1:
                continue
            r, c = map(int, coords[0])
            v = int(work[r, c])
            if (r, c) in center_cells or v not in tracks_by_color:
                continue
            marker_col = int(v)

        track = tracks_by_color.get(int(marker_col))
        if track is None:
            continue

        mp = marker_pixel(coords, int(marker_col))
        if mp is None or mp == (int(track["r"]), int(track["c"])):
            continue

        dr = 0
        dc = 0
        if track["axis"] == "h":
            dc = int(track["c"] - mp[1])
        else:
            dr = int(track["r"] - mp[0])
        moves.append((coords, int(dr), int(dc)))

    out = g.copy()
    for coords, _dr, _dc in moves:
        for r, c in coords:
            out[int(r), int(c)] = int(under[int(r), int(c)])

    for coords, dr, dc in moves:
        for r, c in coords:
            rr = int(r) + int(dr)
            cc = int(c) + int(dc)
            if 0 <= rr < H and 0 <= cc < W and int(out[rr, cc]) == int(under[rr, cc]):
                out[rr, cc] = int(g[int(r), int(c)])

    return out


# --------------- generator (variant D4) ---------------

_COLORS = tuple(range(10))
_BG_COLOR_RANGE = (1, 8)
_GRID_HEIGHT_RANGE = (10, 22)
_GRID_WIDTH_RANGE = (8, 22)
_GEN_RETRY_LIMIT = 400

_ANCHOR_COUNT_RANGE = (1, 3)
_BODY_AREA_RANGE = (4, 13)
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
_FLIP_PROB = 0.5

_TRACK_MARGIN = 2
_TICK_LEN_CHOICES = (1, 2)
_TRACK_GAP = 3
_NEIGHBORS4 = ((1, 0), (-1, 0), (0, 1), (0, -1))

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


def _edge_cells(body, axis, side):
    if axis == "h":
        target_row = min(r for r, _c in body) if side == "below" else max(r for r, _c in body)
        return [(r, c) for r, c in body if r == target_row]
    target_col = min(c for _r, c in body) if side == "right" else max(c for _r, c in body)
    return [(r, c) for r, c in body if c == target_col]


def _make_sprite(area, body_color, marker_color, bg, axis, forced_side=None):
    body = _make_body(area)
    if forced_side is not None:
        side = str(forced_side)
    elif axis == "h":
        side = random.choice(["below", "above"])
    else:
        side = random.choice(["right", "left"])
    attach = random.choice(_edge_cells(body, axis, side))
    if axis == "h":
        marker = (attach[0] - 1, attach[1]) if side == "below" else (attach[0] + 1, attach[1])
    else:
        marker = (attach[0], attach[1] - 1) if side == "right" else (attach[0], attach[1] + 1)

    all_cells = set(body) | {marker}
    r0, c0, r1, c1 = _bbox(all_cells)
    patch = np.full((r1 - r0 + 1, c1 - c0 + 1), int(bg), dtype=int)
    for r, c in body:
        patch[int(r - r0), int(c - c0)] = int(body_color)
    patch[int(marker[0] - r0), int(marker[1] - c0)] = int(marker_color)
    return {
        "patch": patch,
        "offset": (int(marker[0] - r0), int(marker[1] - c0)),
        "marker_color": int(marker_color),
        "axis": axis,
    }


def _top_left(sprite, marker_r, marker_c):
    off_r, off_c = sprite["offset"]
    return int(marker_r) - int(off_r), int(marker_c) - int(off_c)


def _can_place_sprite(canvas, sprite, marker_r, marker_c, bg, allowed_marker_cells):
    patch = sprite["patch"]
    top, left = _top_left(sprite, marker_r, marker_c)
    h, w = patch.shape
    H, W = canvas.shape
    if top < 0 or left < 0 or top + h > H or left + w > W:
        return False
    off_r, off_c = sprite["offset"]
    for pr in range(h):
        for pc in range(w):
            val = int(patch[pr, pc])
            if val == int(bg):
                continue
            rr = int(top + pr)
            cc = int(left + pc)
            if (pr, pc) == (int(off_r), int(off_c)):
                if (rr, cc) not in allowed_marker_cells:
                    return False
            elif int(canvas[rr, cc]) != int(bg):
                return False
    return True


def _place_sprite(canvas, sprite, marker_r, marker_c, bg):
    top, left = _top_left(sprite, marker_r, marker_c)
    paste(canvas, sprite["patch"], top, left, transparent=int(bg))


def _spaced_positions(values, count, gap):
    vals = list(values)
    random.shuffle(vals)
    chosen = []
    for v in vals:
        if all(abs(int(v) - int(u)) >= int(gap) for u in chosen):
            chosen.append(int(v))
            if len(chosen) == int(count):
                return chosen
    return None


def _track_centers(H, W, count, axis):
    if axis == "h":
        rows = _spaced_positions(range(_TRACK_MARGIN, H - _TRACK_MARGIN), count, _TRACK_GAP)
        if rows is None:
            return None
        return [
            {"axis": "h", "r": int(r), "c": int(random.randint(_TRACK_MARGIN, W - _TRACK_MARGIN - 1))}
            for r in rows
        ]
    cols = _spaced_positions(range(_TRACK_MARGIN, W - _TRACK_MARGIN), count, _TRACK_GAP)
    if cols is None:
        return None
    return [
        {"axis": "v", "r": int(random.randint(_TRACK_MARGIN, H - _TRACK_MARGIN - 1)), "c": int(c)}
        for c in cols
    ]


def _draw_track(canvas, anchor):
    H, W = canvas.shape
    r = int(anchor["r"])
    c = int(anchor["c"])
    rail = int(anchor["rail_color"])
    center = int(anchor["color"])
    tick_len = int(anchor["tick_len"])
    tick_side = str(anchor["tick_side"])
    if anchor["axis"] == "h":
        canvas[r, :] = int(rail)
        if tick_side == "up":
            deltas = range(1, tick_len + 1)
            cells = [r - d for d in deltas]
        else:
            deltas = range(1, tick_len + 1)
            cells = [r + d for d in deltas]
        for rr in cells:
            if 0 <= rr < H:
                canvas[rr, c] = int(rail)
        canvas[r, c] = int(center)
        anchor["marker_cells"] = {(r, cc) for cc in range(W)}
    else:
        canvas[:, c] = int(rail)
        if tick_side == "left":
            deltas = range(1, tick_len + 1)
            cells = [c - d for d in deltas]
        else:
            deltas = range(1, tick_len + 1)
            cells = [c + d for d in deltas]
        for cc in cells:
            if 0 <= cc < W:
                canvas[r, cc] = int(rail)
        canvas[r, c] = int(center)
        anchor["marker_cells"] = {(rr, c) for rr in range(H)}


def _body_color(bg, marker_colors, rail_color, used):
    forbidden = {int(bg), int(rail_color), *[int(c) for c in marker_colors]}
    pool = [c for c in _COLORS if c not in forbidden and c not in used]
    if not pool:
        pool = [c for c in _COLORS if c not in forbidden]
    return random.choice(pool)


def _scatter_positions(canvas, sprite, anchor, bg):
    if anchor["axis"] == "h":
        coords = [
            (int(anchor["r"]), c)
            for c in range(canvas.shape[1])
            if abs(int(c) - int(anchor["c"])) >= 2
        ]
    else:
        coords = [
            (r, int(anchor["c"]))
            for r in range(canvas.shape[0])
            if abs(int(r) - int(anchor["r"])) >= 2
        ]
    random.shuffle(coords)
    return [
        (r, c)
        for r, c in coords
        if _can_place_sprite(canvas, sprite, r, c, bg, {(int(r), int(c))})
    ]


def _bg_is_mode(grid, bg):
    counts = Counter(map(int, grid.ravel()))
    return counts[int(bg)] > max((n for c, n in counts.items() if c != int(bg)), default=0)


def _new_canvas_pair():
    H = random.randint(*_GRID_HEIGHT_RANGE)
    W = random.randint(*_GRID_WIDTH_RANGE)
    bg = random.randint(*_BG_COLOR_RANGE)
    out = np.full((H, W), int(bg), dtype=int)
    inp = np.full((H, W), int(bg), dtype=int)
    return bg, out, inp


def _gen_attempt():
    bg, out, inp = _new_canvas_pair()
    H, W = out.shape
    axis = random.choice(["h", "v"])
    n_anchors = random.randint(*_ANCHOR_COUNT_RANGE)
    centers = _track_centers(H, W, n_anchors, axis)
    if centers is None:
        return None

    available = [c for c in _COLORS if c != int(bg)]
    rail_color = random.choice(available)
    color_pool = [c for c in available if c != int(rail_color)]
    if len(color_pool) < n_anchors:
        return None
    anchor_colors = random.sample(color_pool, n_anchors)

    anchors = []
    for base, color in zip(centers, anchor_colors):
        anchor = {
            "axis": base["axis"],
            "r": int(base["r"]),
            "c": int(base["c"]),
            "color": int(color),
            "rail_color": int(rail_color),
            "tick_len": int(random.choice(_TICK_LEN_CHOICES)),
        }
        if axis == "h":
            anchor["tick_side"] = random.choice(["up", "down"])
            anchor["body_side"] = "below" if anchor["tick_side"] == "up" else "above"
        else:
            anchor["tick_side"] = random.choice(["left", "right"])
            anchor["body_side"] = "right" if anchor["tick_side"] == "left" else "left"
        anchors.append(anchor)
        _draw_track(out, anchor)
        _draw_track(inp, anchor)

    used_body = set()
    random.shuffle(anchors)
    for anchor in anchors:
        placed = False
        for _ in range(80):
            body = _body_color(bg, anchor_colors, rail_color, used_body)
            sprite = _make_sprite(
                random.randint(*_BODY_AREA_RANGE),
                body,
                anchor["color"],
                bg,
                anchor["axis"],
                anchor["body_side"],
            )
            if not _can_place_sprite(out, sprite, anchor["r"], anchor["c"], bg, {(anchor["r"], anchor["c"])}):
                continue
            positions = _scatter_positions(inp, sprite, anchor, bg)
            if not positions:
                continue
            in_pos = random.choice(positions)
            _place_sprite(out, sprite, anchor["r"], anchor["c"], bg)
            _place_sprite(inp, sprite, in_pos[0], in_pos[1], bg)
            used_body.add(int(body))
            placed = True
            break
        if not placed:
            return None

    if not _bg_is_mode(inp, bg):
        return None
    if not np.array_equal(main(inp), out):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(_GEN_RETRY_LIMIT):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid 581f7754 D4 input")
