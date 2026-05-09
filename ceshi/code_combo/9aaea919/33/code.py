# Combo variant of 9aaea919 (index 33)
# combo_spec: D2/2+D3/0+D5/1+D9/2
# Name: 水平旋转+多形状

"""
Design analysis for the combo variant
=====================================

source_task:
- Original task 9aaea919, plus the single-dimension references for D2/2, D3/0, and D9/2.
- D5/1 is applied semantically: markers move from "below the lane" to "at the lane side".
  After the D9 horizontal rotation, that side marker lives at the right edge of the row lane.

a) Concrete behavior and trace
------------------------------
- The original task used four vertical 5-column lanes, bottom horizontal bars, one shared plus shape,
  and bottom-up insertion in the green lane.
- The fused task rotates the whole layout into four horizontal 5-row lanes. Sprites now live in 5x3 slots
  placed on a period-4 horizontal lattice, and the markers are 5x1 vertical bars at the right edge.
- Each lane keeps one repeated shape, but the four lanes use four different rotated templates:
  plus, T, L, and square.
- Transfer semantics also rotate and invert together:
  - original vertical bottom-up stacking -> rotated horizontal right-to-left stacking;
  - D3 inversion flips that priority -> final fused rule fills leftmost empty slots first.
- Small trace:
  - Suppose the green lane initially has two native square sprites at left cols 21 and 25.
  - A top red lane contains two T sprites; a lower red lane contains one L sprite.
  - Original rotated-but-not-inverted behavior would add them at 17, 13, 9.
  - Fused behavior adds them at 1, 5, 9 instead.
  - The source lanes turn grey in place; the destination receives T, T, then L, all recolored to
    the green lane's color.

b) Complexity calibration
-------------------------
- The fusion changes exactly four visible axes and keeps everything else fixed: rotated geometry,
  side markers, insertion direction, and lane-specific shapes.
- Shape variety is capped at four templates, one per lane, so the rule stays learnable from a few samples
  instead of requiring open-ended object matching.
- The generator forbids fully filling the green lane. If the lane became full, left-to-right and
  right-to-left insertion would collapse to the same output and D3 would become visually invisible.

c) Hidden-assumption audit of the original solver
-------------------------------------------------
- Original assumption: lanes are found by fixed column slices. Broken by D9; replaced with fixed row slices.
- Original assumption: markers are bottom horizontal bars. Broken by D5+D9; replaced with right-edge vertical bars.
- Original assumption: all useful sprites share one pixel count. Broken by D2; plus=11, T/L=7, square=9.
  The solver therefore groups components by normalized shape signature instead of area.
- Original assumption: "stack upward from bottom" can be implemented by probing from one bottom anchor.
  Broken by D3 after rotation. The solver must reconstruct the whole horizontal slot ladder and scan
  empty left columns from small to large.
- Original assumption: shifting a template by lane center is enough. In the fused task, the solver instead
  preserves row-local offsets inside the 5-row lane band, which works for all four rotated shapes.

d) Placement legality
---------------------
- Each lane occupies one disjoint 5-row band: rows 3..7, 9..13, 15..19, 21..25.
- Each sprite fits in a 5x3 slot. Legal slot left columns are 1, 5, 9, 13, 17, 21, 25, leaving col 28 blank
  and col 29 free for the marker bar.
- Counts are chosen so `green_count + transferred_count < 7`, guaranteeing no overlap, no overflow,
  and a visible difference between left-fill and right-fill behavior.

e) Visual observability
-----------------------
- Humans can directly see all deciding cues: right-edge red/green markers, row-lane grouping,
  four distinct silhouettes, source lanes turning grey, and left-side growth in the destination lane.
- No invisible property such as numeric color order is used for routing or tie-breaking.
- Batch order for multiple red lanes is top-to-bottom source order, which is also visible in the grid.

f) Visual cleanliness
---------------------
- There is one blank separator column between the rightmost sprite body and the marker bar.
- Different lanes never overlap because their 5-row bands are disjoint.
- Mixed-shape destination lanes remain readable because all shapes share the same 5x3 slot system.

g) Change inferability
----------------------
- Color flow is still structured: source lanes become grey; transferred copies adopt the destination lane color.
- Shape flow is separate and equally visible: the destination does not copy its own native shape, it imports
  each red lane's silhouette.
- The fused task therefore exposes three orthogonal signals in few-shot examples:
  location marker -> source/destination role, shape -> source identity, color -> destination identity.

h) Generator strategy
---------------------
- The generator is input-first for clarity, then builds the expected output explicitly.
- It also builds a counterfactual "right-to-left" output and rejects any sample where that would match the
  fused output, preventing silent regression to the non-inverted rotated task.

i) New solver assumptions and generator guarantees
--------------------------------------------------
- Assumption: within one lane, all repeated non-marker components share one normalized signature.
  Guarantee: each lane paints exactly one shape template.
- Assumption: the destination lane's visible native sprites lie on one period-4 lattice.
  Guarantee: input lanes always start as contiguous right-stacked placements on that lattice.
- Assumption: transferred copies preserve lane-local row offsets inside the 5-row band.
  Guarantee: every shape template is defined in the same 5x3 aligned slot.
- Assumption: multiple red lanes are processed in visible top-to-bottom order.
  Guarantee: generator uses the same order when constructing the expected output.
"""

from __future__ import annotations

import sys
from itertools import combinations
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[4]
_ARC_UTILS_DIR = _ROOT / "arc2_solver_select"
if str(_ARC_UTILS_DIR) not in sys.path:
    sys.path.insert(0, str(_ARC_UTILS_DIR))

from arc_utils import bbox_coords, components_mask, get_bg, paste


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
            cands.append((0 if len(pts) == 5 else 1, r0, hh * ww, len(pts), pts))
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


def _component_signature(pts: list[tuple[int, int]]) -> tuple[int, int, tuple[tuple[int, int], ...]]:
    arr = np.array(pts, dtype=int)
    r0, c0, r1, c1 = bbox_coords(arr)
    norm = tuple(sorted((int(r - r0), int(c - c0)) for r, c in pts))
    return (r1 - r0 + 1, c1 - c0 + 1, norm)


def _sorted_sprites(pts_groups: list[list[tuple[int, int]]]) -> list[list[tuple[int, int]]]:
    return sorted(pts_groups, key=lambda pts: (min(c for _, c in pts), min(r for r, _ in pts)))


def _extract_lane_sprites(
    g: np.ndarray, rows: range, marker_pts: list[tuple[int, int]], bg: int
) -> list[list[tuple[int, int]]]:
    marker_set = set(marker_pts)
    grouped: dict[tuple[int, int, tuple[tuple[int, int], ...]], list[list[tuple[int, int]]]] = {}
    for pts in _single_color_components_in_rows(g, rows, bg):
        kept = [p for p in pts if p not in marker_set]
        if not kept:
            continue
        sig = _component_signature(kept)
        grouped.setdefault(sig, []).append(kept)
    if not grouped:
        return []
    best_group = max(grouped.values(), key=lambda group: (len(group), len(group[0])))
    return _sorted_sprites(best_group)


def _infer_period(sprites: list[list[tuple[int, int]]]) -> int:
    lefts = sorted(min(c for _, c in pts) for pts in sprites)
    if len(lefts) >= 2:
        diffs = [b - a for a, b in zip(lefts, lefts[1:]) if b - a > 0]
        if diffs:
            return int(np.gcd.reduce(diffs))
    _r0, c0, _r1, c1 = bbox_coords(np.array(sprites[0], dtype=int))
    return (c1 - c0 + 1) + 1


def _lane_slot_cols(occupied_lefts: list[int], period: int, sprite_w: int, w: int) -> list[int]:
    """Reconstruct all legal lattice columns for a lane and order them left-to-right."""
    left0 = min(occupied_lefts)
    while left0 - period >= 0:
        left0 -= period
    cols = []
    c = left0
    while c + sprite_w <= w:
        cols.append(c)
        c += period
    return cols


def main(grid: np.ndarray) -> np.ndarray:
    """
    Transfer sprites from red-marked horizontal lanes into the green-marked horizontal lane.

    Input structure:
    - The background is the most common color. Thin red (2) and green (3) vertical bars sit at the right edge of 5-row lanes.
    - The four lanes are horizontal bands containing repeated monochrome sprites on a period-4 left-right lattice, and each lane uses its own 5x3 shape.

    Rule:
    - Every red-marked lane turns all of its sprites grey (5) in place, then contributes the same number of copies to the nearest green-marked lane by row center.
    - Copied sprites keep their source silhouette, adopt the destination lane color, and fill the destination lane's empty lattice slots from left to right.
    - If multiple red lanes feed the same destination, their batches are inserted in top-to-bottom source order. All marker bars are then erased.

    Key insight:
    - The fused task decouples three things at once: side markers identify lane roles, the destination lane supplies the color, and each source lane supplies its own rotated shape while the destination growth direction is the left-first inverse of the usual rotated stacking.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = get_bg(g)

    red_markers = sorted(_find_marker_bars(g, 2), key=lambda pts: bbox_coords(np.array(pts, dtype=int))[0])
    green_markers = sorted(_find_marker_bars(g, 3), key=lambda pts: bbox_coords(np.array(pts, dtype=int))[0])
    assert red_markers and green_markers, "expected both red and green marker bars to be present"

    green_state = {}
    for gm in green_markers:
        dst_r0, _dst_c0, dst_r1, _dst_c1 = bbox_coords(np.array(gm, dtype=int))
        dst_rows = range(dst_r0, dst_r1 + 1)
        dst_sprites = _extract_lane_sprites(g, dst_rows, gm, bg)
        if not dst_sprites:
            continue

        lefts = sorted(min(c for _, c in pts) for pts in dst_sprites)
        dst_color = int(g[dst_sprites[0][0][0], dst_sprites[0][0][1]])
        _r0, c0, _r1, c1 = bbox_coords(np.array(dst_sprites[0], dtype=int))
        sprite_w = c1 - c0 + 1
        green_state[(dst_r0, dst_r1)] = {
            "center": (dst_r0 + dst_r1) / 2.0,
            "r0": dst_r0,
            "dst_color": dst_color,
            "occupied": set(lefts),
            "slots": _lane_slot_cols(lefts, _infer_period(dst_sprites), sprite_w, w),
        }

    assert green_state, "expected at least one green lane with valid sprites"

    out = g.copy()

    for rm in red_markers:
        src_r0, _src_c0, src_r1, _src_c1 = bbox_coords(np.array(rm, dtype=int))
        src_rows = range(src_r0, src_r1 + 1)
        src_sprites = _extract_lane_sprites(g, src_rows, rm, bg)
        if not src_sprites:
            continue

        for pts in src_sprites:
            for r, c in pts:
                out[r, c] = 5

        src_center = (src_r0 + src_r1) / 2.0
        best = min(green_state.values(), key=lambda st: abs(st["center"] - src_center))

        src_ref_pts = src_sprites[0]
        src_ref_c0 = min(c for _, c in src_ref_pts)
        src_offsets = [(r - src_r0, c - src_ref_c0) for r, c in src_ref_pts]

        placed = 0
        for new_left in best["slots"]:
            if placed >= len(src_sprites):
                break
            if new_left in best["occupied"]:
                continue

            ok = True
            for dr, dc in src_offsets:
                rr = best["r0"] + dr
                cc = new_left + dc
                if not (0 <= rr < h and 0 <= cc < w):
                    ok = False
                    break
                if int(out[rr, cc]) != bg:
                    ok = False
                    break
            if not ok:
                continue

            for dr, dc in src_offsets:
                out[best["r0"] + dr, new_left + dc] = best["dst_color"]
            best["occupied"].add(new_left)
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
_LANE_ROW_STARTS = [3, 9, 15, 21]
_MARKER_COL = 29
_SPRITE_H, _SPRITE_W = 5, 3

# --- CFG: horizontal lattice ---
_RIGHTMOST_LEFT = 25
_PERIOD = 4
_ALL_LEFTS = list(range(_RIGHTMOST_LEFT % _PERIOD, _RIGHTMOST_LEFT + 1, _PERIOD))

# --- CFG: color palette ---
_BG_CHOICES = [0, 1, 4, 6, 7, 8, 9]
_GREY = 5

# --- CFG: counts & limits ---
_MIN_SPRITES = 1
_MAX_SPRITES = 3
_MAX_LANE_TOTAL = len(_ALL_LEFTS)
_GEN_RETRIES = 300

# --- CFG: rotated shape library ---
_SHAPE_MASKS: dict[str, tuple[str, str, str, str, str]] = {
    "plus": (".X.", "XXX", "XXX", "XXX", ".X."),
    "tee": ("..X", "..X", "XXX", "..X", "..X"),
    "ell": ("XXX", "X..", "X..", "X..", "X.."),
    "square": ("...", "XXX", "XXX", "XXX", "..."),
}
_SHAPE_KEYS = tuple(_SHAPE_MASKS)


def _sprite_patch(shape_key: str, color: int, bg: int) -> np.ndarray:
    patch = np.full((_SPRITE_H, _SPRITE_W), bg, dtype=int)
    for r, row in enumerate(_SHAPE_MASKS[shape_key]):
        for c, ch in enumerate(row):
            if ch == "X":
                patch[r, c] = color
    return patch


def _lane_lefts(count: int) -> list[int]:
    """Return right-stacked left-column positions for `count` sprites, ordered left-to-right."""
    return [_RIGHTMOST_LEFT - _PERIOD * k for k in range(count - 1, -1, -1)]


def _paint_lane_at_lefts(
    grid: np.ndarray, lane_idx: int, shape_key: str, color: int, lefts: list[int], bg: int
) -> None:
    patch = _sprite_patch(shape_key, color, bg)
    r0 = _LANE_ROW_STARTS[lane_idx]
    for c0 in lefts:
        paste(grid, patch, r0, c0, transparent=bg)


def _paint_lane(grid: np.ndarray, lane_idx: int, shape_key: str, color: int, count: int, bg: int) -> None:
    _paint_lane_at_lefts(grid, lane_idx, shape_key, color, _lane_lefts(count), bg)


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


def _build_expected_output(
    bg: int,
    lane_colors: list[int],
    lane_shapes: list[str],
    input_counts: list[int],
    reds: tuple[int, ...],
    green: int,
    *,
    fill_left_to_right: bool,
) -> np.ndarray:
    red_set = set(reds)
    out = np.full((_H, _W), bg, dtype=int)

    for lane_idx in range(_NUM_LANES):
        if lane_idx == green:
            _paint_lane(out, lane_idx, lane_shapes[lane_idx], lane_colors[lane_idx], input_counts[lane_idx], bg)
        elif lane_idx in red_set:
            _paint_lane(out, lane_idx, lane_shapes[lane_idx], _GREY, input_counts[lane_idx], bg)
        else:
            _paint_lane(out, lane_idx, lane_shapes[lane_idx], lane_colors[lane_idx], input_counts[lane_idx], bg)

    occupied = set(_lane_lefts(input_counts[green]))
    empties = [c for c in _ALL_LEFTS if c not in occupied]
    if not fill_left_to_right:
        empties = list(reversed(empties))

    cursor = 0
    for lane_idx in sorted(reds):
        for _ in range(input_counts[lane_idx]):
            c0 = empties[cursor]
            cursor += 1
            patch = _sprite_patch(lane_shapes[lane_idx], lane_colors[green], bg)
            paste(out, patch, _LANE_ROW_STARTS[green], c0, transparent=bg)

    return out


def _gen_attempt() -> np.ndarray | None:
    rng = np.random.default_rng()

    bg = int(rng.choice(_BG_CHOICES))
    sprite_palette = [c for c in range(10) if c not in (bg, _GREY)]
    lane_colors = rng.choice(sprite_palette, size=_NUM_LANES, replace=False).tolist()
    lane_shapes = rng.permutation(_SHAPE_KEYS).tolist()
    input_counts = rng.integers(_MIN_SPRITES, _MAX_SPRITES + 1, size=_NUM_LANES).tolist()

    feasible = _feasible_roles(input_counts)
    feasible_red_counts = [rc for rc in (1, 2) if feasible[rc]]
    if not feasible_red_counts:
        return None

    red_count = int(rng.choice(feasible_red_counts))
    reds, green = feasible[red_count][int(rng.integers(len(feasible[red_count])))]

    inp = np.full((_H, _W), bg, dtype=int)
    for lane_idx in range(_NUM_LANES):
        _paint_lane(inp, lane_idx, lane_shapes[lane_idx], lane_colors[lane_idx], input_counts[lane_idx], bg)

    inp[_LANE_ROW_STARTS[green] : _LANE_ROW_STARTS[green] + _SPRITE_H, _MARKER_COL] = 3
    for lane_idx in reds:
        inp[_LANE_ROW_STARTS[lane_idx] : _LANE_ROW_STARTS[lane_idx] + _SPRITE_H, _MARKER_COL] = 2

    expected = _build_expected_output(
        bg,
        lane_colors,
        lane_shapes,
        input_counts,
        reds,
        green,
        fill_left_to_right=True,
    )
    counterfactual = _build_expected_output(
        bg,
        lane_colors,
        lane_shapes,
        input_counts,
        reds,
        green,
        fill_left_to_right=False,
    )
    if np.array_equal(expected, counterfactual):
        return None
    if not np.array_equal(main(inp), expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(_GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {_GEN_RETRIES} attempts")
