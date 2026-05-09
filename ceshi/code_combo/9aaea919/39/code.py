# Combo variant of 9aaea919 (index 39)
# combo_spec: D3/0+D7/1+D9/2
# Name: 水平车道+左侧堆叠+仅输出目标

import sys
from itertools import combinations
from pathlib import Path

import numpy as np

_SOLVER_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_SOLVER_DIR) not in sys.path:
    sys.path.append(str(_SOLVER_DIR))

from arc_utils import bbox_coords, components_mask, get_bg, paste


# Design analysis for the combo variant
# a) Concrete behavior and trace
#    - The original task uses four vertical 5-column lanes, bottom-row horizontal markers, and bottom-up
#      filling inside the green lane. This combo rotates the whole layout by 90 degrees, so lanes become
#      5-row horizontal bands and markers become vertical 5x1 bars at the far right.
#    - After rotation, the original D9 behavior would add new sprites from right to left. D3 then flips that
#      priority: new sprites must occupy the leftmost empty lattice slots first, while the pre-existing green
#      sprites remain in their original right-stacked positions.
#    - D7 changes the returned view: after solving the full 30x30 scene, keep only the destination lane's
#      5-row band across the full width, so the output is 5x30 instead of 30x30.
#    - Example: suppose the green lane is rows 9..13 and initially has sprites at left cols {21, 25}; two
#      red lanes contribute gain=2. The rotated-original rule would add at {13, 17}. The combo rule adds at
#      {1, 5} instead, leaving a large visible gap in the middle. The final answer then crops to rows 9..13.
# b) Complexity calibration
#    - Only three coupled freedoms change: orientation, destination fill priority, and output crop. Sprite
#      shape, lane count, role semantics, and color behavior stay fixed, so the puzzle remains inferable.
#    - The generator forbids fully filled destination lanes. If the target band became full, left-first and
#      right-first stacking would produce the same final occupancy and the D3 change would become invisible.
# c) Hidden-assumption audit of the original main()
#    - Column-based lane extraction no longer applies; every lane operation must switch to fixed row bands.
#    - Marker bars are no longer horizontal bottom bars, so marker sorting and target-crop bookkeeping must
#      use row spans and the shared right-edge column instead of column spans and the last row.
#    - The old `right0 - k * period` search from D9 assumes right-priority filling. This combo breaks that
#      assumption; the solver must reconstruct the full legal slot ladder and scan empty slots from left to
#      right.
#    - Returning the whole solved canvas is also invalid now. The green marker's row span must be remembered
#      before marker erasure so the final result can be cropped to the target band.
# d) Placement legality
#    - With a 5x3 rotated sprite, period 4, and rightmost legal left col 25, the legal lattice is
#      {1, 5, 9, 13, 17, 21, 25}; at most 7 sprites fit without overlap.
#    - Markers occupy only column 29, leaving column 28 as a separator, so marker components never merge with
#      sprites even if a lane's sprite color happens to be 2 or 3.
# e) Visual observability
#    - The changed rule is visible from pixels alone: red-tagged horizontal lanes grey out, the green-tagged
#      lane gains same-shaped copies on its far left side, and the answer is a cropped 5-row strip.
#    - No invisible ordering such as numeric color id is used; all matching is driven by marker position and
#      visible lane geometry.
# f) Visual cleanliness
#    - Each horizontal lane lives in its own 5-row band with 1-row gaps between bands, and the output keeps
#      exactly one band, so the transformed objects stay easy to parse.
# g) Color inferability
#    - Colors remain structured: source lanes become grey, inserted sprites reuse the destination lane color,
#      and all unmarked lanes stay unchanged off-crop. The crop changes view, not the color logic.
# h) New solver assumptions guaranteed by the generator
#    - There is exactly one green marker bar, making the crop target unique.
#    - Every marked lane contains at least one valid monochrome sprite component in its own 5-row band.
#    - Destination sprites all lie on one period-4 residue class, allowing reconstruction of the full slot
#      ladder from observed left columns.
#    - The generator deletes only the left-priority added sprites from the destination band, so those slots
#      are genuinely empty when `main()` scans left-to-right.


def _find_marker_bars(g: np.ndarray, marker_color: int) -> list[list[tuple[int, int]]]:
    """Marker bars are thin 4-connected components of the given marker color."""
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


def _lane_slot_cols(occupied_lefts: list[int], period: int, sprite_w: int, w: int) -> list[int]:
    """Reconstruct all legal lattice left columns for a lane and order them from left to right."""
    left0 = min(occupied_lefts)
    while left0 - period >= 0:
        left0 -= period
    cols = []
    c = left0
    while c + sprite_w <= w:
        cols.append(c)
        c += period
    return cols


def _solve_full_scene(grid: np.ndarray) -> tuple[np.ndarray, tuple[int, int]]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = get_bg(g)

    red_markers = _find_marker_bars(g, 2)
    green_markers = _find_marker_bars(g, 3)
    assert red_markers, "expected at least one red marker bar"
    assert len(green_markers) == 1, "expected exactly one green marker bar for the cropped target lane"

    green_marker = green_markers[0]
    dst_r0, _, dst_r1, _ = bbox_coords(np.array(green_marker))
    dst_rows = range(dst_r0, dst_r1 + 1)
    dst_set = set(green_marker)
    dst_comps = _single_color_components_in_rows(g, dst_rows, bg)
    dst_comps = [[p for p in pts if p not in dst_set] for pts in dst_comps]
    dst_comps = [pts for pts in dst_comps if pts]
    assert dst_comps, "expected non-marker content inside the destination lane"

    dst_sz = max(len(p) for p in dst_comps if len(p) != len(green_marker))
    dst_sprites = [p for p in dst_comps if len(p) == dst_sz]
    assert dst_sprites, "expected valid destination sprites"

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
        _r0, c0, _r1, c1 = bbox_coords(np.array(dst_sprites_sorted[0][2]))
        period = (c1 - c0 + 1) + 1

    sprite_w = max(c for _, c in dst_sprites_sorted[0][2]) - min(c for _, c in dst_sprites_sorted[0][2]) + 1
    dst_center = (dst_r0 + dst_r1) / 2.0
    dst_state = {
        "center": dst_center,
        "dst_color": dst_color,
        "occupied": set(lefts),
        "slots": _lane_slot_cols(lefts, period, sprite_w, w),
    }

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

        src_center = (src_r0 + src_r1) / 2.0
        best = dst_state

        src_sprites_sorted = []
        for pts in src_sprites:
            rmin = min(r for r, _ in pts)
            cmin = min(c for _, c in pts)
            src_sprites_sorted.append((cmin, rmin, pts))
        src_sprites_sorted.sort()
        src_ref_c0, _src_ref_r0, src_ref_pts = src_sprites_sorted[0]

        dr_shift = int(round(best["center"] - src_center))
        src_offsets = [(r + dr_shift, c - src_ref_c0) for r, c in src_ref_pts]

        n_new = len(src_sprites)
        placed = 0
        for new_left in best["slots"]:
            if placed >= n_new:
                break
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
    for r, c in green_marker:
        out[r, c] = bg

    return out, (dst_r0, dst_r1)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Move sprites from red-marked horizontal lanes into the green-marked horizontal lane, then keep only that lane.

    Input structure:
    - The background is the most common color. Four fixed horizontal lanes occupy 5-row bands, and thin red (2)
      or green (3) vertical bars on the right edge mark source and destination lanes.
    - Each lane contains repeated identical monochrome rotated-plus sprites arranged on a regular period-4
      horizontal lattice, with the destination lane initially holding only its original right-stacked sprites.

    Rule:
    - Every sprite in a red-marked lane turns grey (5), and the same number of copies is created in the nearest
      green-marked lane using that lane's own color.
    - The destination copies do not extend from the right-side block leftward; they fill the leftmost empty
      lattice slots first, proceeding from left to right across the lane.
    - After the full scene is solved and markers are removed, return only the destination lane's 5-row band, so
      the output is 5x30 rather than 30x30.

    Key insight:
    - This combo is the rotated lane-transfer rule plus a flipped fill priority and a cropped answer: infer the
      target lane from the green bar, rebuild that lane's legal slot ladder, place new sprites into the earliest
      empty left-side slots, and finally keep only the solved target band.
    """
    full_out, (r0, r1) = _solve_full_scene(grid)
    return full_out[r0 : r1 + 1, :]


# --------------- generator ---------------

# --- CFG: canvas & layout ---
_H, _W = 30, 30
_NUM_LANES = 4
_LANE_ROW_STARTS = [3, 9, 15, 21]
_MARKER_COL = 29
_SPRITE_H, _SPRITE_W = 5, 3

# --- CFG: horizontal lattice ---
_RIGHTMOST_LEFT = 25
_PERIOD = 4
_LEFTMOST_LEFT = _RIGHTMOST_LEFT % _PERIOD
_ALL_LEFTS = list(range(_LEFTMOST_LEFT, _RIGHTMOST_LEFT + 1, _PERIOD))

# --- CFG: color palette ---
_BG_CHOICES = [0, 1, 4, 6, 7, 8, 9]
_GREY = 5

# --- CFG: counts & limits ---
_MIN_SPRITES = 1
_MAX_SPRITES = 3
_MAX_LANE_TOTAL = len(_ALL_LEFTS)
_GEN_RETRIES = 300


def _sprite_patch(color: int, bg: int) -> np.ndarray:
    """Return the 5x3 rotated sprite patch with bg-transparent corners."""
    patch = np.full((_SPRITE_H, _SPRITE_W), bg, dtype=int)
    patch[0, 1] = color
    patch[1:4, :] = color
    patch[4, 1] = color
    return patch


def _lane_lefts(count: int) -> list[int]:
    """Return right-stacked left columns for `count` sprites, ordered left-to-right."""
    return [_RIGHTMOST_LEFT - _PERIOD * k for k in range(count - 1, -1, -1)]


def _leftfill_added_lefts(existing_count: int, gain: int) -> list[int]:
    """Return the combo's destination slots: leftmost empty columns first."""
    occupied = set(_lane_lefts(existing_count))
    empties = [c for c in _ALL_LEFTS if c not in occupied]
    return empties[:gain]


def _rightfill_added_lefts(existing_count: int, gain: int) -> list[int]:
    """Return the rotated-original destination slots: nearest empty columns on the left of the block."""
    occupied = set(_lane_lefts(existing_count))
    empties = [c for c in reversed(_ALL_LEFTS) if c not in occupied]
    picked = empties[:gain]
    return sorted(picked)


def _paint_lane_at_lefts(grid: np.ndarray, lane_idx: int, color: int, lefts: list[int], bg: int) -> None:
    patch = _sprite_patch(color, bg)
    r0 = _LANE_ROW_STARTS[lane_idx]
    for c0 in lefts:
        paste(grid, patch, r0, c0, transparent=bg)


def _paint_lane(grid: np.ndarray, lane_idx: int, color: int, count: int, bg: int) -> None:
    """Stamp `count` right-stacked sprites of `color` into the given horizontal lane."""
    _paint_lane_at_lefts(grid, lane_idx, color, _lane_lefts(count), bg)


def _feasible_roles(counts: list[int]) -> dict[int, list[tuple[tuple[int, ...], int]]]:
    """Return feasible (reds, green) assignments grouped by number of red lanes."""
    grouped: dict[int, list[tuple[tuple[int, ...], int]]] = {1: [], 2: []}
    for green in range(_NUM_LANES):
        others = [i for i in range(_NUM_LANES) if i != green]
        for red_count in (1, 2):
            for reds in combinations(others, red_count):
                total = counts[green] + sum(counts[i] for i in reds)
                if total < _MAX_LANE_TOTAL:
                    grouped[red_count].append((reds, green))
    return grouped


def _gen_attempt() -> np.ndarray | None:
    rng = np.random.default_rng()

    bg = int(rng.choice(_BG_CHOICES))
    # Keep marker colors exclusive to the red/green bars so lane semantics stay visually unambiguous.
    sprite_palette = [c for c in range(10) if c not in (bg, _GREY, 2, 3)]
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

    added_lefts = _leftfill_added_lefts(input_counts[green], gain)
    if len(added_lefts) != gain:
        return None
    if added_lefts == _rightfill_added_lefts(input_counts[green], gain):
        return None

    out_full = np.full((_H, _W), bg, dtype=int)
    for lane_idx in range(_NUM_LANES):
        color = _GREY if lane_idx in red_set else lane_colors[lane_idx]
        if lane_idx == green:
            lefts = sorted(_lane_lefts(input_counts[green]) + added_lefts)
            _paint_lane_at_lefts(out_full, lane_idx, color, lefts, bg)
        else:
            _paint_lane(out_full, lane_idx, color, input_counts[lane_idx], bg)

    inp = out_full.copy()
    for lane_idx in reds:
        _paint_lane(inp, lane_idx, lane_colors[lane_idx], input_counts[lane_idx], bg)

    green_r0 = _LANE_ROW_STARTS[green]
    for c0 in added_lefts:
        inp[green_r0 : green_r0 + _SPRITE_H, c0 : c0 + _SPRITE_W] = bg
    inp[green_r0 : green_r0 + _SPRITE_H, _MARKER_COL] = 3

    for lane_idx in reds:
        r0 = _LANE_ROW_STARTS[lane_idx]
        inp[r0 : r0 + _SPRITE_H, _MARKER_COL] = 2

    expected = out_full[green_r0 : green_r0 + _SPRITE_H, :]
    got = main(inp)
    if got.shape != expected.shape:
        return None
    if not np.array_equal(got, expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(_GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {_GEN_RETRIES} attempts")
