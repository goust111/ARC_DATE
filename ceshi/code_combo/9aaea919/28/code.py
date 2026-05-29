# Combo variant of 9aaea919 (index 28)
# combo_spec: D2/2+D3/0+D10/0
# Name: 多形状+顶部堆叠+可变周期

"""
Design analysis for the combo variant
=====================================

source_task:
- Base task: 9aaea919.
- Combined changes: D2/2 (different source shapes per lane), D3/0 (top-down destination fill),
  D10/0 (lane-specific vertical periods).

a) Concrete behavior and trace
------------------------------
- The grid stays 30x30 with four fixed 5-column lanes and 1x5 marker bars on row 29.
- Each lane repeats one monochrome 3x5 sprite, but the four lanes now use a permutation of:
  plus, T, L, square.
- Each lane also chooses its own vertical period from {4, 5}, while the bottom visible sprite
  still has top row 25.
- Red bars mark source lanes; one green bar marks the destination lane.
- Transfer still conserves counts, but the copied objects now preserve source shape, adopt the
  destination color, obey the destination period, and occupy the highest empty destination slots.

- Small trace:
  - Suppose a red lane at columns 3-7 contains two L sprites with period 4 at tops [21, 25].
  - Suppose the green lane at columns 15-19 contains two square sprites with period 5 at tops [20, 25].
  - Original seed behavior would grey the red lane and then look for new slots just above the bottom
    block; if it were still bottom-priority, the two inserted tops would be [10, 15] in that lane.
  - In this combo, the inserted shapes stay L-shaped, recolor to the green lane color, and occupy the
    highest empty period-5 slots [0, 5].
  - The visible difference is joint: the destination lane shows a mixed-shape stack, the gap remains
    near the middle instead of the top, and the step size follows the destination lane's own spacing.

b) Complexity calibration
-------------------------
- Shapes are limited to four reusable 3x5 templates. This is enough to make shape preservation obvious
  without introducing scale or rotation as extra hidden rules.
- Periods are limited to {4, 5}. Two values are easy to compare visually and still force the solver to
  infer cadence per lane instead of assuming one global step.
- All marked lanes are generated with at least two visible sprites so the period is directly supported
  by repeated evidence in the image.
- The destination lane is never filled to capacity after transfer, so top-down placement remains
  distinguishable from bottom-up placement.

c) Hidden assumptions audited from the original main()
------------------------------------------------------
- Marker detection as thin 4-connected bars remains valid.
- Restricting extraction to one lane's columns remains valid because lanes are still disjoint 5-column slots.
- The fragile assumption is "sprite type can be identified by max component size". With four shapes, pixel
  counts differ (11, 7, 7, 9), so the combo switches to normalized-shape signatures and keeps the repeated
  signature inside each lane.
- The old placement scan `bottom0 - k * period` no longer matches the rule: the combo must reconstruct the
  full destination slot ladder and scan from the smallest top row downward.
- The old global-period intuition is also invalid. Placement must continue the chosen destination lane's own
  cadence, not a task-wide default.
- Preserving source geometry is still safe if horizontal placement uses lane-local column offsets (`c - lane_c0`)
  instead of assuming a centered plus shape.

d) Placement legality
---------------------
- Every sprite fits inside the fixed 3x5 lane slot, so horizontal overlap between lanes is impossible.
- All four shapes have height 3. Combined with periods 4 or 5, distinct slot rows never overlap vertically.
- Capacity is computed from the destination period with the fixed bottom anchor at row 25. Role assignments
  are filtered so the transferred stack fits fully inside the canvas and still leaves at least one empty slot.
- Markers remain on row 29, one clear row below the lowest sprite row 27, so markers never connect to sprites.

e) Visual observability
-----------------------
- No invisible metadata such as numeric color ordering is used for matching or placement.
- The three changed factors are all directly visible:
  1. different lanes show different silhouettes,
  2. marked lanes visibly exhibit different vertical spacings,
  3. new destination sprites appear in the top empty slots rather than hugging the existing bottom block.
- The generator requires the marked lanes to include both period values and requires the destination output to
  differ from the bottom-up alternative, so the fusion does not collapse back into the seed task by accident.

f) Visual cleanliness
---------------------
- Objects remain arranged in four clean lanes with consistent 5-column slots.
- Mixed-shape destination stacks are still readable because copied shapes share the destination color but do not
  overlap, and the lane spacing stays unchanged.
- The generator keeps exactly one source lane marked in each sample. This avoids an extra hidden ordering rule
  when multiple different source shapes would compete for the same top-down destination slots.

g) Color-change inferability
----------------------------
- Lane colors remain distinct, as in the seed task.
- Recoloring stays structurally cued: the green-marked lane already displays the color that transferred sprites
  must adopt, while red-marked lanes visibly turn grey in place.
- Because copied shapes stay source-derived, humans can separate "shape comes from source" from "color comes from
  destination" using only the examples.

h) Generator strategy
---------------------
- Output-first construction becomes awkward because the destination lane can mix two shapes while still encoding
  top-down and variable-period evidence.
- The generator therefore uses input-first construction:
  1. choose four distinct lane colors, a permutation of the four shapes, and one period per lane;
  2. choose counts and marked roles subject to capacity and observability constraints;
  3. paint the input directly and place markers;
  4. validate by running `main(input)`.

i) New solver assumptions and generator guarantees
--------------------------------------------------
- Assumption: each lane repeats exactly one normalized sprite signature.
  Guarantee: each lane is painted from one fixed shape template.
- Assumption: each marked lane shows at least two visible sprites, so its period can be inferred from top-row gaps.
  Guarantee: generated counts are 2-3 in every lane.
- Assumption: destination legal slots are the rows in one residue class determined by the visible destination lane.
  Guarantee: every lane uses one consistent period with bottom top row fixed at 25.
- Assumption: top-down placement is meaningful because the destination output is not completely full.
  Guarantee: role selection enforces `existing + gain < capacity`.
"""

from __future__ import annotations

import sys
from itertools import combinations
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[4]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from arc_utils import bbox_coords, components_mask, get_bg, paste


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
        colors = {int(g[r, c]) for r, c in pts}
        if len(colors) == 1:
            out.append(pts)
    return out


def _component_signature(pts: list[tuple[int, int]]) -> tuple[int, int, tuple[tuple[int, int], ...]]:
    arr = np.array(pts, dtype=int)
    r0, c0, r1, c1 = bbox_coords(arr)
    norm = tuple(sorted((int(r - r0), int(c - c0)) for r, c in pts))
    return (r1 - r0 + 1, c1 - c0 + 1, norm)


def _extract_lane_sprites(
    g: np.ndarray, cols: range, marker_pts: list[tuple[int, int]], bg: int
) -> list[list[tuple[int, int]]]:
    marker_set = set(marker_pts)
    grouped: dict[tuple[int, int, tuple[tuple[int, int], ...]], list[list[tuple[int, int]]]] = {}
    for pts in _single_color_components_in_cols(g, cols, bg):
        kept = [p for p in pts if p not in marker_set]
        if not kept:
            continue
        sig = _component_signature(kept)
        grouped.setdefault(sig, []).append(kept)
    if not grouped:
        return []
    best_group = max(grouped.values(), key=lambda group: (len(group), len(group[0])))
    return sorted(best_group, key=lambda pts: (min(r for r, _ in pts), min(c for _, c in pts)))


def _infer_period(tops: list[int]) -> int:
    diffs = [b - a for a, b in zip(tops, tops[1:]) if b > a]
    assert diffs, "expected at least two visible sprites to infer the lane period"
    period = int(np.gcd.reduce(diffs))
    assert period > 0, "lane period must be positive"
    return period


def _lane_slot_rows(occupied_tops: list[int], period: int, sprite_h: int, h: int) -> list[int]:
    top0 = min(occupied_tops)
    while top0 - period >= 0:
        top0 -= period
    rows = []
    r = top0
    while r + sprite_h <= h:
        rows.append(r)
        r += period
    return rows


def _lane_state_from_marker(g: np.ndarray, marker_pts: list[tuple[int, int]], bg: int) -> dict | None:
    _r0, c0, _r1, c1 = bbox_coords(np.array(marker_pts, dtype=int))
    cols = range(c0, c1 + 1)
    sprites = _extract_lane_sprites(g, cols, marker_pts, bg)
    if len(sprites) < 2:
        return None

    tops = [min(r for r, _ in pts) for pts in sprites]
    period = _infer_period(tops)
    color = int(g[sprites[0][0][0], sprites[0][0][1]])
    sprite_h = max(r for r, _ in sprites[0]) - min(r for r, _ in sprites[0]) + 1
    return {
        "center": (c0 + c1) / 2.0,
        "c0": c0,
        "color": color,
        "period": period,
        "sprite_h": sprite_h,
        "occupied": set(tops),
        "slots": _lane_slot_rows(tops, period, sprite_h, g.shape[0]),
        "sprites": sprites,
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Move sprites from red-marked lanes into the green-marked lane, preserving source shape and using top-first placement.

    Input structure:
    - The background is the most common color. Thin red (2) and green (3) bars on the bottom row mark source and destination lanes.
    - Each 5-column lane contains repeated monochrome 3x5 sprites, but lanes may differ in both sprite shape and vertical period.

    Rule:
    - Every sprite in a red-marked lane turns grey (5), and the same number of sprites is recreated in the nearest green-marked lane.
    - Recreated sprites keep the source lane's silhouette, adopt the destination lane's color, and fill the highest empty slots of the destination lane's own lattice from top to bottom.
    - All marker bars are erased afterward.

    Key insight:
    - Three information streams must be fused: shape comes from the source lane, color and cadence come from the destination lane, and the destination scan order is top-first rather than bottom-first.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = get_bg(g)

    red_markers = sorted(_find_marker_bars(g, 2), key=lambda pts: bbox_coords(np.array(pts, dtype=int))[1])
    green_markers = sorted(_find_marker_bars(g, 3), key=lambda pts: bbox_coords(np.array(pts, dtype=int))[1])
    assert red_markers and green_markers, "expected both red and green marker bars to be present"

    green_states = []
    for gm in green_markers:
        st = _lane_state_from_marker(g, gm, bg)
        if st is not None:
            green_states.append(st)
    assert green_states, "expected at least one green lane with inferable period"

    out = g.copy()

    for rm in red_markers:
        src_state = _lane_state_from_marker(g, rm, bg)
        if src_state is None:
            continue

        for pts in src_state["sprites"]:
            for r, c in pts:
                out[r, c] = 5

        best = min(green_states, key=lambda st: abs(st["center"] - src_state["center"]))

        src_ref_pts = src_state["sprites"][0]
        src_ref_r0 = min(r for r, _ in src_ref_pts)
        src_offsets = [(r - src_ref_r0, c - src_state["c0"]) for r, c in src_ref_pts]

        placed = 0
        need = len(src_state["sprites"])
        for new_top in best["slots"]:
            if placed >= need:
                break
            if new_top in best["occupied"]:
                continue

            ok = True
            for dr, dc in src_offsets:
                rr = new_top + dr
                cc = best["c0"] + dc
                if not (0 <= rr < h and 0 <= cc < w):
                    ok = False
                    break
                if int(out[rr, cc]) != bg:
                    ok = False
                    break
            if not ok:
                continue

            for dr, dc in src_offsets:
                out[new_top + dr, best["c0"] + dc] = best["color"]
            best["occupied"].add(new_top)
            placed += 1

    for rm in red_markers:
        for r, c in rm:
            out[r, c] = bg
    for gm in green_markers:
        for r, c in gm:
            out[r, c] = bg

    return out


# --------------- generator ---------------

# --- CFG: canvas & layout ---
_H, _W = 30, 30
_NUM_LANES = 4
_LANE_STARTS = [3, 9, 15, 21]
_MARKER_ROW = 29
_SPRITE_H, _SPRITE_W = 3, 5

# --- CFG: vertical lattice ---
_BOTTOM_TOP = 25
_PERIOD_CHOICES = (4, 5)

# --- CFG: color palette ---
_BG_CHOICES = [0, 1, 4, 6, 7, 8, 9]
_GREY = 5

# --- CFG: counts & limits ---
_MIN_SPRITES = 2
_MAX_SPRITES = 3
_GEN_RETRIES = 400

# --- CFG: shape library ---
_SHAPE_MASKS: dict[str, tuple[str, str, str]] = {
    "plus": (".XXX.", "XXXXX", ".XXX."),
    "tee": ("XXXXX", "..X..", "..X.."),
    "ell": ("X....", "X....", "XXXXX"),
    "square": (".XXX.", ".XXX.", ".XXX."),
}
_SHAPE_KEYS = tuple(_SHAPE_MASKS)


def _sprite_patch(shape_key: str, color: int, bg: int) -> np.ndarray:
    patch = np.full((_SPRITE_H, _SPRITE_W), bg, dtype=int)
    for r, row in enumerate(_SHAPE_MASKS[shape_key]):
        for c, ch in enumerate(row):
            if ch == "X":
                patch[r, c] = color
    return patch


def _lane_capacity(period: int) -> int:
    return (_BOTTOM_TOP // period) + 1


def _lane_tops(period: int, count: int) -> list[int]:
    return [_BOTTOM_TOP - period * k for k in range(count - 1, -1, -1)]


def _paint_lane_at_tops(
    grid: np.ndarray, lane_idx: int, shape_key: str, color: int, tops: list[int], bg: int
) -> None:
    patch = _sprite_patch(shape_key, color, bg)
    c0 = _LANE_STARTS[lane_idx]
    for r0 in tops:
        paste(grid, patch, r0, c0, transparent=bg)


def _paint_lane(
    grid: np.ndarray, lane_idx: int, shape_key: str, color: int, count: int, bg: int, period: int
) -> None:
    _paint_lane_at_tops(grid, lane_idx, shape_key, color, _lane_tops(period, count), bg)


def _topdown_added_tops(existing_tops: list[int], period: int, gain: int) -> list[int]:
    slots = _lane_slot_rows(existing_tops, period, _SPRITE_H, _H)
    occupied = set(existing_tops)
    empties = [r for r in slots if r not in occupied]
    return empties[:gain]


def _bottomup_added_tops(existing_count: int, period: int, gain: int) -> list[int]:
    return _lane_tops(period, existing_count + gain)[:gain]


def _feasible_roles(counts: list[int], periods: list[int]) -> list[tuple[tuple[int, ...], int]]:
    feasible: list[tuple[tuple[int, ...], int]] = []
    for green in range(_NUM_LANES):
        others = [i for i in range(_NUM_LANES) if i != green]
        cap = _lane_capacity(periods[green])
        for reds in combinations(others, 1):
            total = counts[green] + sum(counts[i] for i in reds)
            marked_periods = {periods[green], *(periods[i] for i in reds)}
            if total >= cap:
                continue
            if len(marked_periods) < 2:
                continue
            added_topdown = _topdown_added_tops(
                _lane_tops(periods[green], counts[green]), periods[green], total - counts[green]
            )
            added_bottomup = _bottomup_added_tops(counts[green], periods[green], total - counts[green])
            if len(added_topdown) != total - counts[green]:
                continue
            if set(added_topdown) == set(added_bottomup):
                continue
            feasible.append((reds, green))
    return feasible


def _gen_attempt() -> np.ndarray | None:
    rng = np.random.default_rng()

    bg = int(rng.choice(_BG_CHOICES))
    sprite_palette = [c for c in range(10) if c not in (bg, _GREY)]
    lane_colors = rng.choice(sprite_palette, size=_NUM_LANES, replace=False).tolist()
    lane_shapes = rng.permutation(_SHAPE_KEYS).tolist()
    counts = rng.integers(_MIN_SPRITES, _MAX_SPRITES + 1, size=_NUM_LANES).tolist()
    periods = rng.choice(_PERIOD_CHOICES, size=_NUM_LANES, replace=True).tolist()
    if len(set(periods)) < 2:
        return None

    feasible = _feasible_roles(counts, periods)
    if not feasible:
        return None

    reds, green = feasible[int(rng.integers(len(feasible)))]

    inp = np.full((_H, _W), bg, dtype=int)
    for lane_idx in range(_NUM_LANES):
        _paint_lane(inp, lane_idx, lane_shapes[lane_idx], lane_colors[lane_idx], counts[lane_idx], bg, periods[lane_idx])

    green_c0 = _LANE_STARTS[green]
    inp[_MARKER_ROW, green_c0 : green_c0 + _SPRITE_W] = 3
    for lane_idx in reds:
        c0 = _LANE_STARTS[lane_idx]
        inp[_MARKER_ROW, c0 : c0 + _SPRITE_W] = 2

    try:
        main(inp)
    except Exception:
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(_GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {_GEN_RETRIES} attempts")
