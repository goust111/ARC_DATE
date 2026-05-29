# Variant of 9aaea919
# Dimension: D9 — 车道数量与布局 (meta)
# Change: 固定4条车道，列中心分别在5、11、17、23，间距6列... → 水平车道替代垂直车道
#
# Design analysis
# spec:
#   grid: 30x30
#   background: any color from {0-9} except 2, 3, 5
#   lanes:
#     count: 4 fixed horizontal lanes
#     row_ranges: [3..7], [9..13], [15..19], [21..25]
#     row_centers: 5, 11, 17, 23
#     spacing: 6 rows between lane centers
#   sprite:
#     shape: 90-degree rotation of the original plus/diamond
#       .X.
#       XXX
#       XXX
#       XXX
#       .X.
#     size: 5x3, 11 pixels
#     color: one monochrome color per lane, all 4 distinct, from {0-9} minus {bg, 5}
#     horizontal_period: 4 columns (3 sprite cols + 1 gap)
#     rightmost_sprite_left_col: 25
#     count_per_lane: 1-3 in input
#   markers:
#     shape: vertical 5x1 bar on col 29, aligned to the lane row range
#     red_count: 1-2 source markers in color 2
#     green_count: exactly 1 destination marker in color 3
#   transformation:
#     red lanes: their sprites turn grey (5) in place
#     green lane: gains N new sprites, where N is the total sprite count from all red lanes
#     placement: new sprites use the source sprite mask recolored to the green lane color,
#       and fill empty period-4 slots from right to left
#     markers: all marker bars are removed
#
# a) Concrete behavior trace:
#   Original task grouped objects by columns and searched for empty top rows above a bottom anchor.
#   This variant groups objects by rows and searches for empty left columns before a right anchor.
#   Example: a green lane with 2 sprites at cols 21 and 25 receiving 2 red sprites becomes
#   a 4-sprite row at cols 13, 17, 21, 25. The two added sprites are therefore the leftmost two.
#
# b) Complexity calibration:
#   The lane count, colors, and source/destination role structure remain unchanged, so the rule
#   is still inferable from 2-4 examples. Rotating both lane direction and sprite aspect ratio
#   makes the D9 change visually explicit instead of being a hidden bookkeeping change.
#
# c) Implicit-assumption audit:
#   Old code assumed: lane membership == fixed column slice; stack coordinate == top row;
#   marker alignment == bottom horizontal bar; nearest lane chosen by x-center; sprite period
#   inferred from row differences. All of those assumptions break under the rotated layout and
#   are replaced here by row slices, left-column occupancy, right-side vertical markers, y-center
#   matching, and period inference from column differences.
#
# d) Placement legality:
#   With width 3, period 4, and rightmost left col 25, up to 7 sprites fit exactly at
#   left cols 1,5,9,13,17,21,25. The marker lives at col 29, leaving a blank separator column.
#
# e) Visual observability:
#   The solver never uses color id ordering. Humans can see that red-marked rows donate sprites,
#   green-marked rows receive same-shaped sprites on their left side, and the source sprites grey out.
#
# f) Visual cleanliness:
#   Each lane occupies its own 5-row band, markers sit outside the sprite area, and there is no
#   overlap between lanes or with markers.
#
# g) Color-change inferability:
#   Recoloring is still local and structured: source rows become grey; received sprites adopt the
#   existing color of the single green-marked lane. No free-form color mapping is introduced.
#
# h) New solver assumptions checked against generator:
#   - Every marked lane contains at least one sprite component in its 5-row band.
#   - All sprite copies inside one lane share one exact shape and one color.
#   - Green-lane total after transfer never exceeds 7, so leftward filling remains in bounds.
#   The generator constructs inputs to guarantee these conditions and validates them with main().

import numpy as np

from arc_utils import bbox_coords, components_mask, get_bg, paste
from itertools import combinations


def _find_marker_bars(g: np.ndarray, marker_color: int) -> list[list[tuple[int, int]]]:
    """
    Marker bars are thin 4-connected components of the given marker color.
    """
    cands = []
    for pts_arr in components_mask(g == marker_color, connectivity=4):
        r0, c0, r1, c1 = bbox_coords(pts_arr)
        hh = r1 - r0 + 1
        ww = c1 - c0 + 1
        if hh == 1 or ww == 1:
            pts = [(int(r), int(c)) for r, c in pts_arr]
            cands.append((0 if len(pts) == 5 else 1, -c1, hh * ww, len(pts), pts))
    cands.sort()
    return [t[4] for t in cands]


def _single_color_components_in_rows(g: np.ndarray, rows: range, bg: int) -> list[list[tuple[int, int]]]:
    h, w = g.shape
    rowmask = np.zeros((h, w), dtype=bool)
    rowmask[rows, :] = True
    m = (g != bg) & rowmask
    comps = [[(int(r), int(c)) for r, c in pts] for pts in components_mask(m, connectivity=4)]
    out = []
    for pts in comps:
        colors_in = {int(g[r, c]) for r, c in pts}
        if len(colors_in) == 1:
            out.append(pts)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Transfer sprites from red-marked horizontal lanes into green-marked horizontal lanes.

    Input structure:
    - The background is the most common color. Red (2) and green (3) thin vertical bars sit at the right edge of lanes.
    - Each lane is a 5-row horizontal band containing repeated identical monochrome sprites laid out on an even left-right lattice.

    Rule:
    - Every red-marked lane loses all of its sprites in place, recoloring them to grey (5), and contributes that many copies to the nearest green lane by row center.
    - The destination lane receives same-shaped sprites recolored to its own lane color, filling empty lattice slots from right to left.
    - After the transfer, all marker bars disappear back to background.

    Key insight:
    - The task is the original count-conservation rule expressed in a 90-degree rotated layout: lane detection, nearest-lane matching, and stacking all move from the vertical axis to the horizontal axis.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = get_bg(g)

    red_markers = _find_marker_bars(g, 2)
    green_markers = _find_marker_bars(g, 3)
    assert red_markers and green_markers, "expected both red and green marker bars to be present"

    green_state = {}
    for gm in green_markers:
        dst_r0, _, dst_r1, _ = bbox_coords(np.array(gm))
        dst_rows = range(dst_r0, dst_r1 + 1)
        dst_set = set(gm)
        dst_comps = _single_color_components_in_rows(g, dst_rows, bg)
        dst_comps = [[p for p in pts if p not in dst_set] for pts in dst_comps]
        dst_comps = [pts for pts in dst_comps if pts]
        if not dst_comps:
            continue

        dst_sz = max(len(p) for p in dst_comps if len(p) != len(gm))
        dst_sprites = [p for p in dst_comps if len(p) == dst_sz]
        if not dst_sprites:
            continue

        dst_sprites_sorted = []
        for pts in dst_sprites:
            rmin = min(r for r, _ in pts)
            cmin = min(c for _, c in pts)
            dst_sprites_sorted.append((cmin, rmin, pts))
        dst_sprites_sorted.sort()
        dst_color = int(g[dst_sprites_sorted[0][2][0][0], dst_sprites_sorted[0][2][0][1]])

        lefts = sorted({cmin for cmin, _rmin, _pts in dst_sprites_sorted})
        if len(lefts) >= 2:
            diffs = [b - a for a, b in zip(lefts, lefts[1:]) if b - a > 0]
            period = int(np.gcd.reduce(diffs)) if diffs else 4
        else:
            r0, c0, r1, c1 = bbox_coords(np.array(dst_sprites_sorted[0][2]))
            period = (c1 - c0 + 1) + 1

        gr = (dst_r0 + dst_r1) / 2.0
        green_state[id(gm)] = {
            "marker": gm,
            "center": gr,
            "dst_color": dst_color,
            "period": period,
            "occupied": set(lefts),
            "right0": max(lefts),
        }

    assert green_state, "expected at least one green lane with valid sprites"

    out = g.copy()

    for rm in red_markers:
        src_r0, _, src_r1, _ = bbox_coords(np.array(rm))
        src_rows = range(src_r0, src_r1 + 1)
        src_set = set(rm)
        src_comps = _single_color_components_in_rows(g, src_rows, bg)
        src_comps = [[p for p in pts if p not in src_set] for pts in src_comps]
        src_comps = [pts for pts in src_comps if pts]
        if not src_comps:
            continue

        src_sz = max(len(p) for p in src_comps if len(p) != len(rm))
        src_sprites = [p for p in src_comps if len(p) == src_sz]
        if not src_sprites:
            continue

        for pts in src_sprites:
            for r, c in pts:
                out[r, c] = 5

        rc = (src_r0 + src_r1) / 2.0
        best = min(green_state.values(), key=lambda st: abs(st["center"] - rc))

        src_sprites_sorted = []
        for pts in src_sprites:
            rmin = min(r for r, _ in pts)
            cmin = min(c for _, c in pts)
            src_sprites_sorted.append((cmin, rmin, pts))
        src_sprites_sorted.sort()
        src_ref_c0, _src_ref_r0, src_ref_pts = src_sprites_sorted[0]

        dr_shift = int(round(best["center"] - rc))
        src_offsets = [(r + dr_shift, c - src_ref_c0) for r, c in src_ref_pts]
        sprite_w = max(dc for _rr, dc in src_offsets) + 1

        n_new = len(src_sprites)
        placed = 0
        k = 0
        while placed < n_new:
            new_left = best["right0"] - k * best["period"]
            k += 1
            if new_left < 0 or new_left + sprite_w > w:
                if new_left < 0:
                    break
                continue
            if new_left in best["occupied"]:
                continue
            ok = True
            for rr, dc in src_offsets:
                cc = new_left + dc
                if not (0 <= rr < h and 0 <= cc < w):
                    ok = False
                    break
                if int(out[rr, cc]) != bg:
                    ok = False
                    break
            if not ok:
                continue
            for rr, dc in src_offsets:
                out[rr, new_left + dc] = best["dst_color"]
            best["occupied"].add(new_left)
            placed += 1

    for rm in red_markers:
        for r, c in rm:
            out[r, c] = bg
    for gm in green_markers:
        for r, c in gm:
            out[r, c] = bg

    return out


# --------------- generator (O3) ---------------

# --- CFG: canvas & layout ---
_H, _W = 30, 30
_NUM_LANES = 4
_LANE_ROW_STARTS = [3, 9, 15, 21]   # row-0 of each lane's 5-high sprite slot
_MARKER_COL = 29                    # column for red/green 5x1 marker bars
_SPRITE_H, _SPRITE_W = 5, 3         # rotated plus/diamond sprite bbox

# --- CFG: horizontal stacking ---
_RIGHTMOST_LEFT = 25                # left col of the rightmost sprite
_PERIOD = 4                         # horizontal period (3 sprite cols + 1 gap)

# --- CFG: color palette ---
_BG_CHOICES = [0, 1, 4, 6, 7, 8, 9]
_GREY = 5

# --- CFG: counts & limits ---
_MIN_SPRITES = 1
_MAX_SPRITES = 3
_MAX_LANE_TOTAL = 7
_GEN_RETRIES = 300


def _sprite_patch(color: int, bg: int) -> np.ndarray:
    """Return the 5x3 rotated sprite patch with bg-transparent corners."""
    patch = np.full((_SPRITE_H, _SPRITE_W), bg, dtype=int)
    patch[0, 1] = color
    patch[1:4, :] = color
    patch[4, 1] = color
    return patch


def _lane_lefts(count: int) -> list[int]:
    """Return left-column positions for `count` sprites, ordered left-to-right."""
    return [_RIGHTMOST_LEFT - _PERIOD * k for k in range(count - 1, -1, -1)]


def _paint_lane(grid: np.ndarray, lane_idx: int, color: int, count: int, bg: int) -> None:
    """Stamp `count` sprites of `color` into the given horizontal lane."""
    patch = _sprite_patch(color, bg)
    r0 = _LANE_ROW_STARTS[lane_idx]
    for c0 in _lane_lefts(count):
        paste(grid, patch, r0, c0, transparent=bg)


def _feasible_roles(counts: list[int]) -> dict[int, list[tuple[tuple[int, ...], int]]]:
    """Return feasible (reds, green) assignments grouped by number of red lanes."""
    grouped: dict[int, list[tuple[tuple[int, ...], int]]] = {1: [], 2: []}
    for green in range(_NUM_LANES):
        others = [i for i in range(_NUM_LANES) if i != green]
        for red_count in (1, 2):
            for reds in combinations(others, red_count):
                if counts[green] + sum(counts[i] for i in reds) <= _MAX_LANE_TOTAL:
                    grouped[red_count].append((reds, green))
    return grouped


def _gen_attempt() -> np.ndarray | None:
    rng = np.random.default_rng()

    bg = int(rng.choice(_BG_CHOICES))
    sprite_palette = [c for c in range(10) if c not in (bg, _GREY)]
    lane_colors = rng.choice(sprite_palette, size=_NUM_LANES, replace=False).tolist()
    input_counts = rng.integers(_MIN_SPRITES, _MAX_SPRITES + 1, size=_NUM_LANES).tolist()

    feasible = _feasible_roles(input_counts)
    feasible_red_counts = [rc for rc in (1, 2) if feasible[rc]]
    if not feasible_red_counts:
        return None

    red_count = int(rng.choice(feasible_red_counts))
    reds, green = feasible[red_count][int(rng.integers(len(feasible[red_count])))]
    red_set = set(reds)
    gain = sum(input_counts[i] for i in reds)

    output_counts = input_counts[:]
    output_counts[green] += gain
    out = np.full((_H, _W), bg, dtype=int)
    for lane_idx in range(_NUM_LANES):
        color = _GREY if lane_idx in red_set else lane_colors[lane_idx]
        _paint_lane(out, lane_idx, color, output_counts[lane_idx], bg)

    inp = out.copy()
    for lane_idx in reds:
        _paint_lane(inp, lane_idx, lane_colors[lane_idx], input_counts[lane_idx], bg)

    green_r0 = _LANE_ROW_STARTS[green]
    for c0 in _lane_lefts(output_counts[green])[:gain]:
        inp[green_r0:green_r0 + _SPRITE_H, c0:c0 + _SPRITE_W] = bg
    inp[green_r0:green_r0 + _SPRITE_H, _MARKER_COL] = 3

    for lane_idx in reds:
        r0 = _LANE_ROW_STARTS[lane_idx]
        inp[r0:r0 + _SPRITE_H, _MARKER_COL] = 2

    if not np.array_equal(main(inp), out):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(_GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {_GEN_RETRIES} attempts")
