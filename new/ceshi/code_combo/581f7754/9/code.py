# Combo variant of 581f7754 (index 9)
# combo_spec: D4/1+D7/2
# Name: 内部十字线锚点+锚点边条带裁切
#
# Design analysis
# ---------------
# fusion_spec:
#   concrete_behavior:
#     - The old task used isolated border singletons as anchors and returned the full solved canvas.
#     - This combo replaces each anchor with an interior crosshair: one full-span rail plus a short
#       perpendicular tick, with a unique colored center pixel at the crossing.
#     - Sprites no longer align to a border coordinate; they slide only along the rail until their
#       minority-color marker lands exactly on the center pixel.
#     - After solving on the full canvas, the output is cropped to the narrow strip containing the
#       rails, their ticks, and the solved sprites. Horizontal rails keep full width but only a few
#       rows; vertical rails keep full height but only a few columns.
#   behavior_trace:
#     - Original example: a bottom-edge anchor at column 8 made a sprite slide horizontally until the
#       marker column became 8, and the full HxW grid stayed visible.
#     - Combo example: a horizontal rail on row 6 with a tick at column 9 and center color 4 makes
#       the sprite slide left/right on row 6 until marker=(6,9). The solved full canvas is then
#       reduced to rows around that rail, for example rows 4..9.
#     - Visible difference: border singletons disappear entirely; long interior rails become the
#       motion cue, and the final answer is a cropped band rather than the whole board.
#   complexity_calibration:
#     - Each grid still contains 1-3 anchors, but they all share one orientation to avoid ambiguous
#       mixed-axis crops and rail intersections.
#     - Every anchor gets exactly one sprite. Reusing a center color for multiple sprites would force
#       multiple bodies to collide on the same target cell, so the generator forbids that.
#     - The crop is not a tight bbox in both dimensions; it preserves the rail direction's long axis,
#       which keeps the "strip around the crosshair" rule visually distinct from generic cropping.
#   implicit_assumption_audit:
#     - Old assumption "all non-bg foreground components are sprites" is false now because rails are
#       permanent foreground structure. The solver first detects rails, records their underlay, and
#       erases rail-colored cells before component extraction.
#     - Old assumption "clearing a moved sprite reveals background" is false because marker cells sit
#       on rails. Clearing now restores the rail color under markers instead of plain bg.
#     - Old assumption "output shape equals input shape" is false. The solver therefore separates
#       "solve the full canvas" from "crop the focused strip".
#     - New assumption: one shared rail orientation and one shared rail color exist per instance, and
#       body colors never equal the rail color. The generator enforces all three.
#   placement_legality:
#     - Generation remains output-first: draw tracks and solved sprites on the full canvas, then move
#       each sprite away from the center along the same rail to create the input.
#     - Bodies are forced to one side of the rail so only the marker touches the rail. This keeps rail
#       detection clean and guarantees the solved sprites fit inside the crop strip.
#     - The strip band is chosen before track placement, and all rails plus solved sprites are kept
#       inside it, so the crop is genuinely smaller than the input but never clips intended output.
#   visual_observability:
#     - A human can see the long rail as a track, the short tick as a crosshair cue, and the matching
#       marker color as the handle that must reach the crossing.
#     - Across examples the orientation changes, but within one grid it is consistent, so the crop
#       direction and slide axis remain inferable from only a few samples.
#   cleanliness_and_inferability:
#     - No distractor singletons are added. With rails and cropping both active, extra singleton noise
#       would obscure which foreground is structural and which foreground moves.
#     - Colors are used only for local matching between center pixel and marker pixel; there is no
#       hidden ordering by color value.

from __future__ import annotations

from collections import Counter
import random

import numpy as np

from arc_utils import find_components, get_bg, neighbors4, paste


_COLORS = tuple(range(10))
_NEIGHBORS4 = ((1, 0), (-1, 0), (0, 1), (0, -1))

# Canvas and retry bounds.
_BG_COLOR_RANGE = (1, 8)
_GRID_HEIGHT_RANGE = (10, 22)
_GRID_WIDTH_RANGE = (8, 22)
_GEN_RETRY_LIMIT = 500
_FOCUS_MIN_DIM = 12

# Crosshair / crop strip configuration.
_ANCHOR_COUNT_RANGE = (1, 3)
_TRACK_MARGIN = 2
_TRACK_GAP = 3
_TICK_LEN = 1
_STRIP_THICKNESS_RANGE = (6, 10)
_MIN_REDUCTION = 2
_MAX_AREA_RATIO = 0.82
_BODY_AREA_MAX_BY_COUNT = {1: 13, 2: 10, 3: 7}
_AXIS_SEQUENCE = ("h", "v")
_AXIS_CURSOR = 0

# Sprite geometry.
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

_GROW_BODY_SEEDS = [
    {(0, 0), (0, 1), (0, 2), (1, 0)},
    {(0, 0), (0, 1), (0, 2), (1, 2)},
    {(0, 0), (1, 0), (1, 1), (1, 2)},
    {(0, 2), (1, 0), (1, 1), (1, 2)},
    {(0, 1), (1, 0), (1, 1), (1, 2)},
]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Slide marker-tagged sprites along interior crosshair rails to the matching centers, then crop to the rail strip.

    Input structure:
    - The background is the most common color.
    - One or more interior crosshairs appear as a full-span monochrome rail plus a short perpendicular tick, with a differently colored center pixel at the crossing.
    - Each moving sprite is a 2-color component whose minority marker color matches one center pixel; only that marker lies on the rail.

    Rule:
    - Detect the rails first, remove the rail-colored cells from component extraction, and match each sprite to the same-colored center pixel.
    - Horizontal rails cause horizontal sliding to the center column; vertical rails cause vertical sliding to the center row.
    - After reconstructing the full solved canvas, keep only the narrow strip that contains the crosshairs and solved sprites.

    Key insight: The long interior rail replaces the border as the motion guide, and the final answer keeps only the band around those rails instead of the whole grid.
    """
    solved, tracks, placements = _solve_full(grid)
    return _crop_solved_strip(solved, tracks, placements)


def _detect_tracks(g: np.ndarray, bg: int) -> list[dict]:
    h, w = g.shape
    tracks: list[dict] = []

    for rail_color in _COLORS:
        if int(rail_color) == int(bg):
            continue
        for r in range(1, h - 1):
            rail_count = sum(int(g[r, c]) == int(rail_color) for c in range(w))
            if rail_count < w - 2:
                continue
            center_cols = [
                c
                for c in range(1, w - 1)
                if int(g[r, c]) not in {int(bg), int(rail_color)}
                and (int(g[r - 1, c]) == int(rail_color) or int(g[r + 1, c]) == int(rail_color))
            ]
            if len(center_cols) != 1:
                continue
            c = int(center_cols[0])
            rail_cells = {(int(r), int(cc)) for cc in range(w)}
            for rr in range(r - 1, -1, -1):
                if int(g[rr, c]) != int(rail_color):
                    break
                rail_cells.add((int(rr), int(c)))
            for rr in range(r + 1, h):
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

        for c in range(1, w - 1):
            rail_count = sum(int(g[r, c]) == int(rail_color) for r in range(h))
            if rail_count < h - 2:
                continue
            center_rows = [
                r
                for r in range(1, h - 1)
                if int(g[r, c]) not in {int(bg), int(rail_color)}
                and (int(g[r, c - 1]) == int(rail_color) or int(g[r, c + 1]) == int(rail_color))
            ]
            if len(center_rows) != 1:
                continue
            r = int(center_rows[0])
            rail_cells = {(int(rr), int(c)) for rr in range(h)}
            for cc in range(c - 1, -1, -1):
                if int(g[r, cc]) != int(rail_color):
                    break
                rail_cells.add((int(r), int(cc)))
            for cc in range(c + 1, w):
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

        if tracks:
            return tracks
    return tracks


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


def _solve_full(grid: np.ndarray) -> tuple[np.ndarray, list[dict], list[dict]]:
    g = np.array(grid, dtype=int)
    h, w = g.shape
    bg = int(get_bg(g))
    tracks = _detect_tracks(g, bg)
    assert tracks

    tracks_by_color = {int(track["center_color"]): track for track in tracks}
    assert len(tracks_by_color) == len(tracks)

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
    moves: list[tuple[np.ndarray, int, int, str]] = []

    for comp in comps:
        coords = comp["coords"]
        if coords.size == 0:
            continue

        marker_col = _component_marker_color(work, coords)
        if marker_col is None:
            continue

        track = tracks_by_color.get(int(marker_col))
        if track is None:
            continue

        mp = _marker_pixel(work, coords, int(marker_col))
        if mp is None or mp == (int(track["r"]), int(track["c"])):
            continue

        dr = 0
        dc = 0
        if track["axis"] == "h":
            dc = int(track["c"] - mp[1])
        else:
            dr = int(track["r"] - mp[0])
        moves.append((coords, int(dr), int(dc), str(track["axis"])))

    out = g.copy()
    for coords, _dr, _dc, _axis in moves:
        for r, c in coords:
            out[int(r), int(c)] = int(under[int(r), int(c)])

    placements: list[dict] = []
    for coords, dr, dc, axis in moves:
        written: list[tuple[int, int]] = []
        for r, c in coords:
            rr = int(r) + int(dr)
            cc = int(c) + int(dc)
            if 0 <= rr < h and 0 <= cc < w and int(out[rr, cc]) == int(under[rr, cc]):
                out[rr, cc] = int(g[int(r), int(c)])
                written.append((int(rr), int(cc)))
        placements.append({"axis": axis, "cells": written})

    return out, tracks, placements


def _crop_from_focus_cells(solved: np.ndarray, axis: str, focus_cells: list[tuple[int, int]]) -> np.ndarray:
    rows = [int(r) for r, _c in focus_cells]
    cols = [int(c) for _r, c in focus_cells]
    if axis == "h":
        return solved[min(rows) : max(rows) + 1, :].copy()
    return solved[:, min(cols) : max(cols) + 1].copy()


def _crop_solved_strip(solved: np.ndarray, tracks: list[dict], placements: list[dict]) -> np.ndarray:
    axis_counts = Counter(str(track["axis"]) for track in tracks)
    focus_axis = max(axis_counts.items(), key=lambda kv: int(kv[1]))[0]

    focus_cells: list[tuple[int, int]] = []
    for track in tracks:
        if str(track["axis"]) != focus_axis:
            continue
        focus_cells.extend((int(r), int(c)) for r, c in track["rail_cells"])
        focus_cells.append((int(track["r"]), int(track["c"])))
    for placement in placements:
        if str(placement["axis"]) != focus_axis:
            continue
        focus_cells.extend((int(r), int(c)) for r, c in placement["cells"])

    assert focus_cells
    return _crop_from_focus_cells(solved, focus_axis, focus_cells)


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


def _make_sprite(area, body_color, marker_color, bg, axis, forced_side):
    body = _make_body(area)
    side = str(forced_side)
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
        "axis": str(axis),
    }


def _top_left(sprite, marker_r, marker_c):
    off_r, off_c = sprite["offset"]
    return int(marker_r) - int(off_r), int(marker_c) - int(off_c)


def _sprite_cells(sprite, marker_r, marker_c, bg):
    patch = sprite["patch"]
    top, left = _top_left(sprite, marker_r, marker_c)
    cells = []
    for pr, pc in np.argwhere(patch != int(bg)):
        cells.append((int(top + pr), int(left + pc)))
    return cells


def _can_place_sprite(canvas, sprite, marker_r, marker_c, bg, allowed_marker_cells):
    patch = sprite["patch"]
    top, left = _top_left(sprite, marker_r, marker_c)
    h, w = patch.shape
    grid_h, grid_w = canvas.shape
    if top < 0 or left < 0 or top + h > grid_h or left + w > grid_w:
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
    vals = sorted(int(v) for v in values)
    if count <= 0:
        return []
    if not vals:
        return None

    needed_span = int(gap) * (int(count) - 1)
    if vals[-1] - vals[0] < needed_span:
        return None

    chosen = []
    min_allowed = int(vals[0])
    max_allowed = int(vals[-1])
    for remaining in range(int(count), 0, -1):
        hi = int(max_allowed - int(gap) * (remaining - 1))
        candidates = [v for v in vals if int(min_allowed) <= int(v) <= int(hi)]
        if not candidates:
            return None
        pick = int(random.choice(candidates))
        chosen.append(int(pick))
        min_allowed = int(pick + gap)
    return chosen


def _min_band_thickness(anchor_count: int) -> int:
    return max(int(_STRIP_THICKNESS_RANGE[0]), 3 * int(anchor_count) + 1)


def _pick_band(axis: str, shape: tuple[int, int], anchor_count: int) -> tuple[int, int] | None:
    dim = int(shape[0]) if axis == "h" else int(shape[1])
    lo = min(_min_band_thickness(anchor_count), max(4, dim - 2))
    hi = min(_STRIP_THICKNESS_RANGE[1], dim - 2)
    if hi < lo:
        return None
    thickness = random.randint(lo, hi)
    max_start = dim - thickness - 1
    if max_start < 1:
        return None
    start = random.randint(1, max_start)
    return int(start), int(thickness)


def _track_centers(shape: tuple[int, int], count: int, axis: str, band_start: int, band_thickness: int):
    h, w = shape
    band_end = int(band_start + band_thickness - 1)
    if axis == "h":
        rows = _spaced_positions(range(int(band_start + 1), int(band_end)), count, _TRACK_GAP)
        if rows is None:
            return None
        return [
            {"axis": "h", "r": int(r), "c": int(random.randint(_TRACK_MARGIN, w - _TRACK_MARGIN - 1))}
            for r in rows
        ]
    cols = _spaced_positions(range(int(band_start + 1), int(band_end)), count, _TRACK_GAP)
    if cols is None:
        return None
    return [
        {"axis": "v", "r": int(random.randint(_TRACK_MARGIN, h - _TRACK_MARGIN - 1)), "c": int(c)}
        for c in cols
    ]


def _feasible_anchor_counts(dim: int) -> list[int]:
    feasible = []
    for count in range(_ANCHOR_COUNT_RANGE[0], _ANCHOR_COUNT_RANGE[1] + 1):
        min_thickness = _min_band_thickness(count)
        max_thickness = min(int(_STRIP_THICKNESS_RANGE[1]), int(dim - 2))
        if max_thickness >= min_thickness:
            feasible.append(int(count))
    return feasible


def _choose_anchor_count(axis: str, shape: tuple[int, int]) -> int | None:
    dim = int(shape[0]) if axis == "h" else int(shape[1])
    feasible = _feasible_anchor_counts(dim)
    if not feasible:
        return None
    weights = {1: 1, 2: 3, 3: 3}
    return int(random.choices(feasible, weights=[weights[int(c)] for c in feasible], k=1)[0])


def _body_area_bounds(anchor_count: int) -> tuple[int, int]:
    lo = int(_BODY_AREA_RANGE[0])
    hi = int(_BODY_AREA_MAX_BY_COUNT.get(int(anchor_count), _BODY_AREA_RANGE[1]))
    return lo, max(lo, hi)


def _draw_track(canvas, anchor):
    h, w = canvas.shape
    r = int(anchor["r"])
    c = int(anchor["c"])
    rail = int(anchor["rail_color"])
    center = int(anchor["color"])
    if anchor["axis"] == "h":
        canvas[r, :] = int(rail)
        rr = int(r - 1) if anchor["tick_side"] == "up" else int(r + 1)
        if 0 <= rr < h:
            canvas[rr, c] = int(rail)
        canvas[r, c] = int(center)
        anchor["marker_cells"] = {(int(r), int(cc)) for cc in range(w)}
        anchor["track_cells"] = set(anchor["marker_cells"])
        if 0 <= rr < h:
            anchor["track_cells"].add((int(rr), int(c)))
    else:
        canvas[:, c] = int(rail)
        cc = int(c - 1) if anchor["tick_side"] == "left" else int(c + 1)
        if 0 <= cc < w:
            canvas[r, cc] = int(rail)
        canvas[r, c] = int(center)
        anchor["marker_cells"] = {(int(rr), int(c)) for rr in range(h)}
        anchor["track_cells"] = set(anchor["marker_cells"])
        if 0 <= cc < w:
            anchor["track_cells"].add((int(r), int(cc)))


def _body_color(bg, marker_colors, rail_color, used):
    forbidden = {int(bg), int(rail_color), *[int(c) for c in marker_colors]}
    pool = [c for c in _COLORS if c not in forbidden and c not in used]
    if not pool:
        pool = [c for c in _COLORS if c not in forbidden]
    return random.choice(pool)


def _fits_focus_band(sprite, marker_r, marker_c, axis, band_start, band_thickness, bg, shape):
    cells = _sprite_cells(sprite, marker_r, marker_c, bg)
    band_end = int(band_start + band_thickness - 1)
    if axis == "h":
        rows = [r for r, _c in cells]
        return min(rows) >= int(band_start) and max(rows) <= int(band_end)
    cols = [c for _r, c in cells]
    return min(cols) >= int(band_start) and max(cols) <= int(band_end)


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


def _variant_visible(inp, expected, axis):
    in_h, in_w = inp.shape
    out_h, out_w = expected.shape
    if axis == "h":
        if int(out_h) > int(in_h - _MIN_REDUCTION):
            return False
    else:
        if int(out_w) > int(in_w - _MIN_REDUCTION):
            return False
    return (int(out_h) * int(out_w)) / float(int(in_h) * int(in_w)) <= _MAX_AREA_RATIO


def _new_canvas_pair(axis):
    h_lo, h_hi = _GRID_HEIGHT_RANGE
    w_lo, w_hi = _GRID_WIDTH_RANGE
    if axis == "h":
        h_lo = max(h_lo, _FOCUS_MIN_DIM)
    else:
        w_lo = max(w_lo, _FOCUS_MIN_DIM)
    h = random.randint(h_lo, h_hi)
    w = random.randint(w_lo, w_hi)
    bg = random.randint(*_BG_COLOR_RANGE)
    out = np.full((h, w), int(bg), dtype=int)
    inp = np.full((h, w), int(bg), dtype=int)
    return bg, out, inp


def _choose_anchor_sides(anchor, band_start, band_thickness):
    band_end = int(band_start + band_thickness - 1)
    if anchor["axis"] == "h":
        space_above = int(anchor["r"] - band_start)
        space_below = int(band_end - anchor["r"])
        if space_below >= space_above:
            anchor["tick_side"] = "up"
            anchor["body_side"] = "below"
        else:
            anchor["tick_side"] = "down"
            anchor["body_side"] = "above"
    else:
        space_left = int(anchor["c"] - band_start)
        space_right = int(band_end - anchor["c"])
        if space_right >= space_left:
            anchor["tick_side"] = "left"
            anchor["body_side"] = "right"
        else:
            anchor["tick_side"] = "right"
            anchor["body_side"] = "left"


def _next_axis() -> str:
    global _AXIS_CURSOR
    axis = str(_AXIS_SEQUENCE[int(_AXIS_CURSOR) % len(_AXIS_SEQUENCE)])
    _AXIS_CURSOR += 1
    return axis


def _gen_attempt(axis: str):
    bg, out, inp = _new_canvas_pair(axis)
    n_anchors = _choose_anchor_count(axis, out.shape)
    if n_anchors is None:
        return None
    band = _pick_band(axis, out.shape, n_anchors)
    if band is None:
        return None
    band_start, band_thickness = band
    centers = _track_centers(out.shape, n_anchors, axis, band_start, band_thickness)
    if centers is None:
        return None

    available = [c for c in _COLORS if c != int(bg)]
    rail_color = random.choice(available)
    marker_pool = [c for c in available if c != int(rail_color)]
    if len(marker_pool) < n_anchors:
        return None
    anchor_colors = random.sample(marker_pool, n_anchors)

    anchors = []
    for base, color in zip(centers, anchor_colors):
        anchor = {
            "axis": str(base["axis"]),
            "r": int(base["r"]),
            "c": int(base["c"]),
            "color": int(color),
            "rail_color": int(rail_color),
        }
        _choose_anchor_sides(anchor, band_start, band_thickness)
        anchors.append(anchor)
        _draw_track(out, anchor)
        _draw_track(inp, anchor)

    used_body = set()
    focus_cells: list[tuple[int, int]] = []
    sprites = []
    area_bounds = _body_area_bounds(n_anchors)
    for anchor in anchors:
        focus_cells.extend((int(r), int(c)) for r, c in anchor["track_cells"])
        focus_cells.append((int(anchor["r"]), int(anchor["c"])))

        placed = False
        for _ in range(120):
            body = _body_color(bg, anchor_colors, rail_color, used_body)
            sprite = _make_sprite(
                random.randint(*area_bounds),
                body,
                anchor["color"],
                bg,
                anchor["axis"],
                anchor["body_side"],
            )
            if not _fits_focus_band(
                sprite, anchor["r"], anchor["c"], axis, band_start, band_thickness, bg, out.shape
            ):
                continue
            if not _can_place_sprite(out, sprite, anchor["r"], anchor["c"], bg, {(anchor["r"], anchor["c"])}):
                continue
            scatter = _scatter_positions(inp, sprite, anchor, bg)
            if not scatter:
                continue
            in_pos = random.choice(scatter)
            _place_sprite(out, sprite, anchor["r"], anchor["c"], bg)
            _place_sprite(inp, sprite, in_pos[0], in_pos[1], bg)
            focus_cells.extend(_sprite_cells(sprite, anchor["r"], anchor["c"], bg))
            sprites.append((sprite, anchor, in_pos))
            used_body.add(int(body))
            placed = True
            break
        if not placed:
            return None

    expected = _crop_from_focus_cells(out, axis, focus_cells)
    if not _variant_visible(inp, expected, axis):
        return None
    if not _bg_is_mode(inp, bg):
        return None
    if not np.array_equal(main(inp), expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    axis = _next_axis()
    for _ in range(_GEN_RETRY_LIMIT):
        grid = _gen_attempt(axis)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid 581f7754 combo input")
