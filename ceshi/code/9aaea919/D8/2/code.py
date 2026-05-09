# Variant of 9aaea919
# Dimension: D8 — 多步处理与顺序 (multi_step)
# Change: 所有红色车道的精灵同时转移到最近的绿色车道，按精灵数叠加堆叠... → 链式转移

import numpy as np

from arc_utils import bbox_coords, components_mask, get_bg, paste


# Design analysis for the D8 variant
# ----------------------------------
# spec_variant:
#   preserved_layout:
#     grid: 30x30
#     lanes: 4 fixed 5-column lanes at cols 3-7 / 9-13 / 15-19 / 21-25
#     sprites: identical 3x5 plus-shapes on a period-4 vertical lattice
#     markers: red 1x5 bars for source lanes, one green 1x5 bar for the final lane
#   D8_change:
#     chain: red lane A -> nearest unmarked lane B -> green lane C
#     recolor_step1: A's copies arrive in B using B's lane color
#     recolor_step2: all sprites currently in B move again into C using C's lane color
#     in_place_trace:
#       - A ends as grey in its original positions
#       - B also ends as grey, including both its original sprites and the sprites received from A
#       - C gains the full B-total after step1, so C's gain is larger than in the original task
#   concrete_generator_choice:
#     roles: exactly 2 red lanes, 1 unmarked bridge lane, 1 green lane
#     patterns:
#       - [red, bridge, green, red]
#       - [red, green, bridge, red]
#     reason: there is exactly one unmarked lane, so "nearest unmarked lane" is visually and logically unambiguous
#   behavior_trace_example:
#     original:
#       red counts = 2+1, green count = 2 -> output green = 5, bridge unchanged
#     variant:
#       red counts = 2+1, bridge count = 2, green count = 2
#       step1: bridge becomes 5 (its 2 originals + 3 received), red lanes grey out
#       step2: green becomes 7 by receiving those 5 bridge sprites, bridge greys out at 5
#     visible_difference:
#       the formerly unmarked lane changes in the output and the green lane gains bridge_count + red_count, not only red_count
#   implicit-assumption_audit:
#     - The original solver only built state for green-marked lanes. That no longer works because the intermediate unmarked lane now changes and must be parsed explicitly.
#     - The original one-pass red->green loop assumed the destination never becomes a source later. That is false here: bridge lanes receive sprites first, then act as sources.
#     - Greying a lane can no longer happen only for red markers. Bridge lanes must be greyed after their own transfer step, and the generator must leave enough room for their temporary enlargement.
#     - "Nearest" cannot rely on color IDs or hidden ordering. The generator removes ties by using exactly one unmarked bridge lane.
#   placement_legality:
#     - Input counts satisfy green + bridge + sum(reds) <= 7 so the final green stack fits.
#     - The temporary bridge stack also fits because bridge + sum(reds) <= 7 follows from the previous bound.
#     - Added sprites always occupy empty period-4 slots above the existing stack, so no overlap occurs.
#   observability:
#     - Human-solvable evidence appears in three places at once: red lanes grey out, the unmarked bridge lane grows and then greys out, and the green lane gains even more sprites.
#     - The rule depends only on visible marker status and lane positions; no sorting by color id or hidden metadata is used.


def _find_marker_bars(g: np.ndarray, marker_color: int) -> list[list[tuple[int, int]]]:
    """Marker bars are thin 4-connected components of the given marker color."""
    cands = []
    for pts_arr in components_mask(g == marker_color, connectivity=4):
        r0, c0, r1, c1 = bbox_coords(pts_arr)
        hh = r1 - r0 + 1
        ww = c1 - c0 + 1
        if hh == 1 or ww == 1:
            pts = [(int(r), int(c)) for r, c in pts_arr]
            cands.append((0 if len(pts) == 5 else 1, -r1, hh * ww, len(pts), pts))
    cands.sort()
    return [t[4] for t in cands]


def _single_color_components_in_cols(g: np.ndarray, cols: range, bg: int) -> list[list[tuple[int, int]]]:
    h, w = g.shape
    colmask = np.zeros((h, w), dtype=bool)
    colmask[:, cols] = True
    m = (g != bg) & colmask
    comps = [[(int(r), int(c)) for r, c in pts] for pts in components_mask(m, connectivity=4)]
    out = []
    for pts in comps:
        colors_in = {int(g[r, c]) for r, c in pts}
        if len(colors_in) == 1:
            out.append(pts)
    return out


def _collect_lane_col_ranges(g: np.ndarray, bg: int) -> list[tuple[int, int]]:
    cols = set()
    for pts_arr in components_mask(g != bg, connectivity=4):
        r0, c0, r1, c1 = bbox_coords(pts_arr)
        if c1 - c0 + 1 == 5:
            cols.add((int(c0), int(c1)))
    return sorted(cols)


def _lane_state(
    g: np.ndarray,
    c0: int,
    c1: int,
    marker_pts: set[tuple[int, int]],
    role: str,
    bg: int,
) -> dict | None:
    cols = range(c0, c1 + 1)
    comps = _single_color_components_in_cols(g, cols, bg)
    comps = [[p for p in pts if p not in marker_pts] for pts in comps]
    comps = [pts for pts in comps if pts]
    if not comps:
        return None

    sprite_size = max(len(pts) for pts in comps)
    sprites = [pts for pts in comps if len(pts) == sprite_size]
    if not sprites:
        return None

    sprites_sorted = []
    for pts in sprites:
        rmin = min(r for r, _ in pts)
        sprites_sorted.append((rmin, pts))
    sprites_sorted.sort()

    tops = [rmin for rmin, _pts in sprites_sorted]
    if len(tops) >= 2:
        diffs = [b - a for a, b in zip(tops, tops[1:]) if b - a > 0]
        period = int(np.gcd.reduce(diffs)) if diffs else 4
    else:
        r0, _c0, r1, _c1 = bbox_coords(np.array(sprites_sorted[0][1]))
        period = (r1 - r0 + 1) + 1

    ref_top, ref_pts = sprites_sorted[0]
    template_offsets = sorted((int(r - ref_top), int(c - c0)) for r, c in ref_pts)
    sprite_h = max(dr for dr, _dc in template_offsets) + 1
    color = int(g[ref_pts[0][0], ref_pts[0][1]])

    return {
        "role": role,
        "c0": int(c0),
        "c1": int(c1),
        "center": (c0 + c1) / 2.0,
        "color": color,
        "period": int(period),
        "bottom0": int(max(tops)),
        "occupied": set(int(t) for t in tops),
        "template_offsets": template_offsets,
        "sprite_h": int(sprite_h),
        "sprites": [pts for _rmin, pts in sprites_sorted],
    }


def _grey_lane(out: np.ndarray, lane: dict) -> None:
    for top in sorted(lane["occupied"]):
        for dr, dc in lane["template_offsets"]:
            out[top + dr, lane["c0"] + dc] = 5


def _place_into_lane(
    out: np.ndarray,
    lane: dict,
    n_new: int,
    src_offsets: list[tuple[int, int]],
    dst_color: int,
    bg: int,
) -> None:
    h, w = out.shape
    placed = 0
    k = 1
    while placed < n_new:
        new_top = lane["bottom0"] - k * lane["period"]
        k += 1
        if new_top < 0:
            break
        if new_top + lane["sprite_h"] > h or new_top in lane["occupied"]:
            continue

        cells = [(new_top + dr, lane["c0"] + dc) for dr, dc in src_offsets]
        ok = True
        for rr, cc in cells:
            if not (0 <= rr < h and 0 <= cc < w):
                ok = False
                break
            if int(out[rr, cc]) != bg:
                ok = False
                break
        if not ok:
            continue

        for rr, cc in cells:
            out[rr, cc] = int(dst_color)
        lane["occupied"].add(int(new_top))
        placed += 1

    if placed != n_new:
        raise AssertionError("failed to place all transferred sprites")


def main(grid: np.ndarray) -> np.ndarray:
    """
    Relay sprites through an unmarked bridge lane before they reach the green lane.

    Input structure:
    - The background is the most common color, with four 5-column lanes holding evenly spaced monochrome plus-sprites.
    - Red marker bars indicate source lanes; one green marker bar indicates the final destination; one lane is left unmarked and acts as the bridge.

    Rule:
    - Every red lane first sends all of its sprites to the nearest unmarked lane, recoloring those copied sprites to the bridge lane's color while the original red sprites become grey.
    - After those first-hop copies are stacked into the bridge lane, the bridge lane sends its entire current stack to the green lane; the bridge stack then turns grey in place and the new green copies stack upward at the regular lane spacing.
    - All marker bars are removed at the end.

    Key insight:
    - The output is produced by two ordered transfers, not one: you must track where red sprites land first, then treat that enlarged unmarked lane as the source for the second hop into green.
    """
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)

    red_markers = _find_marker_bars(g, 2)
    green_markers = _find_marker_bars(g, 3)
    assert red_markers and green_markers, "expected both red and green marker bars"

    role_lookup: dict[tuple[int, int], tuple[str, set[tuple[int, int]]]] = {}
    for pts in red_markers:
        _r0, c0, _r1, c1 = bbox_coords(np.array(pts))
        role_lookup[(int(c0), int(c1))] = ("red", set(pts))
    for pts in green_markers:
        _r0, c0, _r1, c1 = bbox_coords(np.array(pts))
        role_lookup[(int(c0), int(c1))] = ("green", set(pts))

    lanes = []
    for c0, c1 in _collect_lane_col_ranges(g, bg):
        role, marker_pts = role_lookup.get((c0, c1), ("plain", set()))
        lane = _lane_state(g, c0, c1, marker_pts, role, bg)
        if lane is not None:
            lanes.append(lane)

    red_lanes = [lane for lane in lanes if lane["role"] == "red"]
    green_lanes = [lane for lane in lanes if lane["role"] == "green"]
    plain_lanes = [lane for lane in lanes if lane["role"] == "plain"]
    assert red_lanes and green_lanes and plain_lanes, "expected red, green, and bridge lanes"

    out = g.copy()
    active_bridge_ids = set()

    for src in red_lanes:
        _grey_lane(out, src)
        bridge = min(plain_lanes, key=lambda lane: abs(lane["center"] - src["center"]))
        active_bridge_ids.add(id(bridge))
        _place_into_lane(out, bridge, len(src["sprites"]), src["template_offsets"], bridge["color"], bg)

    for bridge in plain_lanes:
        if id(bridge) not in active_bridge_ids:
            continue
        _grey_lane(out, bridge)
        dst = min(green_lanes, key=lambda lane: abs(lane["center"] - bridge["center"]))
        _place_into_lane(out, dst, len(bridge["occupied"]), bridge["template_offsets"], dst["color"], bg)

    for pts in red_markers + green_markers:
        for r, c in pts:
            out[r, c] = bg

    return out


# --------------- generator ---------------

# --- CFG: canvas & layout ---
_H, _W = 30, 30
_NUM_LANES = 4
_LANE_STARTS = [3, 9, 15, 21]
_MARKER_ROW = 29
_SPRITE_H, _SPRITE_W = 3, 5

# --- CFG: vertical stacking ---
_BOTTOM_TOP = 25
_PERIOD = 4

# --- CFG: color palette ---
_BG_CHOICES = [0, 1, 4, 6, 7, 8, 9]
_GREY = 5

# --- CFG: counts & limits ---
_MIN_SPRITES = 1
_MAX_SPRITES = 3
_MAX_FINAL_GREEN = 7
_GEN_RETRIES = 300

# --- CFG: role patterns ---
_PATTERNS = [
    {"reds": (0, 3), "bridge": 1, "green": 2},  # R B G R
    {"reds": (0, 3), "bridge": 2, "green": 1},  # R G B R
]


def _sprite_patch(color: int, bg: int) -> np.ndarray:
    patch = np.full((_SPRITE_H, _SPRITE_W), bg, dtype=int)
    patch[0, 1:4] = color
    patch[1, :] = color
    patch[2, 1:4] = color
    return patch


def _lane_tops(count: int) -> list[int]:
    return [_BOTTOM_TOP - _PERIOD * k for k in range(count - 1, -1, -1)]


def _paint_lane(grid: np.ndarray, lane_idx: int, color: int, count: int, bg: int) -> None:
    patch = _sprite_patch(color, bg)
    c0 = _LANE_STARTS[lane_idx]
    for r0 in _lane_tops(count):
        paste(grid, patch, r0, c0, transparent=bg)


def _erase_lane_tops(grid: np.ndarray, lane_idx: int, total_count: int, erase_count: int, bg: int) -> None:
    c0 = _LANE_STARTS[lane_idx]
    for r0 in _lane_tops(total_count)[:erase_count]:
        grid[r0 : r0 + _SPRITE_H, c0 : c0 + _SPRITE_W] = bg


def _sample_counts(rng: np.random.Generator, pattern: dict) -> list[int]:
    while True:
        counts = rng.integers(_MIN_SPRITES, _MAX_SPRITES + 1, size=_NUM_LANES).tolist()
        red_total = sum(counts[i] for i in pattern["reds"])
        bridge_total = counts[pattern["bridge"]] + red_total
        green_total = counts[pattern["green"]] + bridge_total
        if bridge_total <= _MAX_FINAL_GREEN and green_total <= _MAX_FINAL_GREEN:
            return counts


def _gen_attempt() -> np.ndarray | None:
    rng = np.random.default_rng()

    bg = int(rng.choice(_BG_CHOICES))
    sprite_palette = [c for c in range(10) if c not in (bg, _GREY)]
    lane_colors = rng.choice(sprite_palette, size=_NUM_LANES, replace=False).tolist()

    pattern = _PATTERNS[int(rng.integers(len(_PATTERNS)))]
    reds = tuple(int(i) for i in pattern["reds"])
    bridge = int(pattern["bridge"])
    green = int(pattern["green"])
    red_set = set(reds)

    input_counts = _sample_counts(rng, pattern)
    red_total = sum(input_counts[i] for i in reds)
    bridge_total = input_counts[bridge] + red_total
    green_total = input_counts[green] + bridge_total

    output_counts = input_counts[:]
    output_counts[bridge] = bridge_total
    output_counts[green] = green_total

    out = np.full((_H, _W), bg, dtype=int)
    for lane_idx in range(_NUM_LANES):
        color = _GREY if lane_idx in red_set or lane_idx == bridge else lane_colors[lane_idx]
        _paint_lane(out, lane_idx, color, output_counts[lane_idx], bg)

    inp = out.copy()
    for lane_idx in reds:
        _paint_lane(inp, lane_idx, lane_colors[lane_idx], input_counts[lane_idx], bg)

    _erase_lane_tops(inp, bridge, bridge_total, red_total, bg)
    _paint_lane(inp, bridge, lane_colors[bridge], input_counts[bridge], bg)

    _erase_lane_tops(inp, green, green_total, bridge_total, bg)

    inp[_MARKER_ROW, _LANE_STARTS[green] : _LANE_STARTS[green] + _SPRITE_W] = 3
    for lane_idx in reds:
        c0 = _LANE_STARTS[lane_idx]
        inp[_MARKER_ROW, c0 : c0 + _SPRITE_W] = 2

    if not np.array_equal(main(inp), out):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(_GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {_GEN_RETRIES} attempts")
