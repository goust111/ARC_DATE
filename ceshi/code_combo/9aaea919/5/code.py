# Combo variant of 9aaea919 (index 5)
# combo_spec: D2/2+D10/0
# Name: 多形状精灵+可变周期

"""
Design analysis for the D2/2 + D10/0 combo
==========================================

source_task:
- Original task 9aaea919 plus the provided single-dimension references for D2/2 and D10/0.

a) Concrete behavior and trace
------------------------------
- The board still has four fixed 5-column lanes and bottom-row marker bars: red marks source lanes, green marks the
  destination lane.
- The fused change has two simultaneous effects:
  1. each lane repeats its own monochrome 3x5 sprite shape from a library of four distinct silhouettes;
  2. each lane also has its own vertical period, chosen from a small set, and transferred sprites must follow the
     destination lane's period rather than the source lane's.
- Concrete trace:
  - Suppose lane A is red, uses an L-shape, period 4, and contains sprites with top rows [17, 21, 25].
  - Suppose lane C is green, uses a square shape, period 5, and contains sprites with top rows [20, 25].
  - In the output, lane A's three L-shapes turn grey in place.
  - Lane C keeps its own two square sprites at [20, 25], then gains three new sprites at the next empty target-lattice
    tops [15, 10, 5]. Those three new sprites are still L-shaped, but recolored to lane C's color.
- Relative to the seed task, two old simplifications disappear at once:
  - transferred shape no longer matches the destination lane's native shape;
  - transferred spacing no longer follows a global period-4 lattice.

b) Complexity calibration
-------------------------
- Shape freedom is limited to exactly four fixed 3x5 silhouettes: plus, T, L, and square. This is enough to make the
  shape-preservation rule visible without introducing scale, rotation, or reflection ambiguity.
- Period freedom is limited to {4, 5}. Two values are easy to compare visually and provide repeated evidence from
  stacked tops.
- All marked lanes are generated with 2-3 visible sprites, so period is inferable directly from repeated row gaps
  rather than from hidden metadata.
- At least one transferred source lane is non-plus, and the marked lanes collectively use both periods, so neither
  dimension collapses to the seed behavior.

c) Hidden assumptions audited from the original main()
------------------------------------------------------
- Original assumption: useful components can be identified by maximum area because every lane uses the same plus shape.
  Broken by D2/2: plus=11 pixels, square=9, T/L=7. The combo solver instead groups lane components by normalized shape
  signature and keeps the repeated signature.
- Original assumption: a single global period is enough. Broken by D10/0: each marked lane must expose its own cadence.
  The combo solver infers period from sprite top-row differences inside that lane.
- Original assumption: horizontal placement can shift absolute columns from source center to destination center.
  Fragile once silhouettes differ. The combo solver preserves slot-local column offsets inside the 5-column lane slot.
- Original assumption: occupancy can be tracked by top row only. This still holds because every allowed shape has
  height 3 and every legal period is at least 4, so same-lane slots never overlap vertically.

d) Placement legality
---------------------
- All shapes live inside the same 3x5 bounding box and are pasted only within a lane's fixed 5-column slot.
- Period choices {4, 5} both leave at least one blank row between adjacent sprite boxes in the same lane.
- Lane capacity is filtered by the destination period: period 4 fits at most 7 sprites, period 5 fits at most 6.
- Markers stay on row 29 while the bottommost sprite occupies rows 25-27, leaving a guaranteed separating blank row.

e) Visual observability
-----------------------
- The human-visible cues are explicit:
  - different lanes visibly repeat different silhouettes;
  - marked lanes visibly show both row-gap values 4 and 5;
  - transferred destination additions visibly keep source silhouettes while adopting destination color.
- No step depends on hidden ordering by numeric color id. Matching is driven by lane position, repeated shape, and
  repeated spacing.

f) Visual cleanliness
---------------------
- The four lane slots remain disjoint and evenly spaced.
- Shapes are monochrome and never overlap across lanes.
- Mixed-shape destination stacks remain readable because all pieces share one destination color but preserve distinct
  silhouettes and separate lattice slots.

g) Color-change inferability
----------------------------
- Each lane still has exactly one non-grey working color, and all lane colors are distinct.
- Recoloring remains structurally anchored: red-marked lanes become grey, and new sprites take the already visible
  color of the green-marked lane.
- Because shape and period now come from different visible sources than color, a solver must disentangle these cues
  using the marked lane structure rather than memorizing color identities.

h) Generator strategy adjustment
--------------------------------
- A naive output-first build is awkward here because the destination lane can contain its native shape plus one or two
  incoming source batches, all sharing the same color but following one target period.
- The combo generator therefore uses parameter-first construction:
  1. choose background, four distinct lane colors, a permutation of the four shapes, per-lane periods, and counts;
  2. choose marked lanes subject to destination capacity, marked-lane period diversity, and non-degenerate shape flow;
  3. paint the input directly from those lane parameters;
  4. build an expected output by explicitly simulating the rule from the chosen parameters;
  5. keep the sample only if `main(input)` exactly matches that expected output.

i) New solver assumptions and generator guarantees
--------------------------------------------------
- Assumption: each lane repeats one normalized non-marker sprite signature.
  Guarantee: the generator paints exactly one shape per lane.
- Assumption: each marked lane has at least two visible sprites, so its period is inferable from row differences.
  Guarantee: input counts are always 2-3.
- Assumption: the destination lane's bottom visible native sprite remains present in the input.
  Guarantee: input is painted directly; no destination-native sprite is removed before solving.
- Assumption: source batches are processed in left-to-right marker order.
  Guarantee: the generator builds expected outputs using the same visible lane ordering rule.
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
    """Return thin 4-connected marker bars of the requested color."""
    cands = []
    for pts_arr in components_mask(g == marker_color, connectivity=4):
        r0, c0, r1, c1 = bbox_coords(pts_arr)
        hh = r1 - r0 + 1
        ww = c1 - c0 + 1
        if hh == 1 or ww == 1:
            pts = [(int(r), int(c)) for r, c in pts_arr]
            cands.append((0 if len(pts) == 5 else 1, -r1, hh * ww, len(pts), pts))
    cands.sort()
    bars = [t[4] for t in cands]
    return sorted(bars, key=lambda pts: bbox_coords(np.array(pts, dtype=int))[1])


def _single_color_components_in_cols(g: np.ndarray, cols: range, bg: int) -> list[list[tuple[int, int]]]:
    h, w = g.shape
    colmask = np.zeros((h, w), dtype=bool)
    colmask[:, cols] = True
    mask = (g != bg) & colmask
    comps = [[(int(r), int(c)) for r, c in pts] for pts in components_mask(mask, connectivity=4)]
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


def _sorted_sprites(sprites: list[list[tuple[int, int]]]) -> list[list[tuple[int, int]]]:
    return sorted(sprites, key=lambda pts: (min(r for r, _ in pts), min(c for _, c in pts)))


def _extract_lane_sprites(
    g: np.ndarray, cols: range, marker_pts: list[tuple[int, int]], bg: int
) -> list[list[tuple[int, int]]]:
    """
    Extract the repeated sprite family inside one lane.

    The combo variant cannot rely on component area because lane shapes differ. We instead keep the repeated normalized
    signature after removing the marker bar cells.
    """
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
    return _sorted_sprites(best_group)


def _infer_period_from_sprites(sprites: list[list[tuple[int, int]]]) -> int | None:
    tops = [min(r for r, _ in pts) for pts in sprites]
    if len(tops) < 2:
        return None
    diffs = [b - a for a, b in zip(tops, tops[1:]) if b > a]
    if not diffs:
        return None
    period = int(np.gcd.reduce(diffs))
    return period if period > 0 else None


def _offsets_from_sprite(pts: list[tuple[int, int]], lane_c0: int) -> list[tuple[int, int]]:
    top = min(r for r, _ in pts)
    return sorted((r - top, c - lane_c0) for r, c in pts)


def _lane_state_from_marker(g: np.ndarray, marker_pts: list[tuple[int, int]], bg: int) -> dict | None:
    _r0, c0, _r1, c1 = bbox_coords(np.array(marker_pts, dtype=int))
    cols = range(c0, c1 + 1)
    sprites = _extract_lane_sprites(g, cols, marker_pts, bg)
    if len(sprites) < 2:
        return None
    period = _infer_period_from_sprites(sprites)
    if period is None:
        return None
    tops = [min(r for r, _ in pts) for pts in sprites]
    color = int(g[sprites[0][0][0], sprites[0][0][1]])
    return {
        "c0": c0,
        "c1": c1,
        "center": (c0 + c1) / 2.0,
        "color": color,
        "period": period,
        "bottom0": max(tops),
        "occupied": set(tops),
        "sprites": sprites,
    }


def _place_batch(
    out: np.ndarray,
    lane_state: dict,
    src_offsets: list[tuple[int, int]],
    count: int,
    color: int,
    bg: int,
    *,
    strict: bool,
) -> None:
    h, w = out.shape
    sprite_h = max(dr for dr, _dc in src_offsets) + 1
    placed = 0
    probe = 0
    while placed < count:
        new_top = lane_state["bottom0"] - probe * lane_state["period"]
        probe += 1
        if new_top < 0 or new_top + sprite_h > h:
            if new_top < 0:
                break
            continue
        if new_top in lane_state["occupied"]:
            continue

        ok = True
        for dr, dc in src_offsets:
            rr = new_top + dr
            cc = lane_state["c0"] + dc
            if not (0 <= rr < h and 0 <= cc < w):
                ok = False
                break
            if int(out[rr, cc]) != bg:
                ok = False
                break
        if not ok:
            continue

        for dr, dc in src_offsets:
            out[new_top + dr, lane_state["c0"] + dc] = color
        lane_state["occupied"].add(new_top)
        placed += 1

    if strict and placed != count:
        raise ValueError("failed to place the full transfer batch")


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor source lanes to grey and extend the destination lane using source silhouettes on the destination rhythm.

    Input structure:
    - The background is the most common color. Thin red (2) and green (3) 1x5 bars on the last row mark source and
      destination lanes.
    - Each of the four fixed 5-column lanes contains repeated monochrome sprites, but lanes may differ both in sprite
      shape and in vertical period.

    Rule:
    - Every sprite in a red-marked lane turns grey (5) in place.
    - The same number of sprites is recreated in the nearest green-marked lane, using the exact source-lane shape,
      recolored to the destination lane's color, and stacked bottom-up on the destination lane's own inferred period.
    - If multiple red lanes feed the same destination, their batches are placed in left-to-right source-lane order.
      All marker bars are then erased.

    Key insight:
    - The transferred batch combines cues from two different lanes: shape comes from the red source lane, while both
      color and vertical cadence come from the green destination lane. Solving therefore requires tracking repeated
      shape signatures and repeated row gaps independently.
    """
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)

    red_markers = _find_marker_bars(g, 2)
    green_markers = _find_marker_bars(g, 3)
    assert red_markers and green_markers, "expected both red and green marker bars to be present"

    green_states = []
    for gm in green_markers:
        st = _lane_state_from_marker(g, gm, bg)
        if st is not None:
            green_states.append(st)
    assert green_states, "expected a green lane with at least two visible sprites"

    out = g.copy()

    for rm in red_markers:
        src_state = _lane_state_from_marker(g, rm, bg)
        if src_state is None:
            continue

        for pts in src_state["sprites"]:
            for r, c in pts:
                out[r, c] = 5

        best = min(green_states, key=lambda st: abs(st["center"] - src_state["center"]))
        src_offsets = _offsets_from_sprite(src_state["sprites"][0], src_state["c0"])
        _place_batch(
            out,
            best,
            src_offsets,
            len(src_state["sprites"]),
            best["color"],
            bg,
            strict=False,
        )

    for marker in red_markers + green_markers:
        for r, c in marker:
            out[r, c] = bg
    return out


# --------------- generator (combo) ---------------

# --- CFG: canvas & layout ---
_H, _W = 30, 30
_NUM_LANES = 4
_LANE_STARTS = [3, 9, 15, 21]
_MARKER_ROW = 29
_SPRITE_H, _SPRITE_W = 3, 5

# --- CFG: vertical stacking ---
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


def _shape_offsets(shape_key: str) -> list[tuple[int, int]]:
    offs = []
    for r, row in enumerate(_SHAPE_MASKS[shape_key]):
        for c, ch in enumerate(row):
            if ch == "X":
                offs.append((r, c))
    return offs


def _lane_capacity(period: int) -> int:
    return (_BOTTOM_TOP // period) + 1


def _lane_tops(period: int, count: int) -> list[int]:
    return [_BOTTOM_TOP - period * k for k in range(count - 1, -1, -1)]


def _paint_lane(
    grid: np.ndarray, lane_idx: int, shape_key: str, color: int, count: int, bg: int, period: int
) -> None:
    patch = _sprite_patch(shape_key, color, bg)
    c0 = _LANE_STARTS[lane_idx]
    for r0 in _lane_tops(period, count):
        paste(grid, patch, r0, c0, transparent=bg)


def _expected_output(
    bg: int,
    lane_colors: list[int],
    lane_shapes: list[str],
    counts: list[int],
    periods: list[int],
    reds: tuple[int, ...],
    green: int,
) -> np.ndarray:
    out = np.full((_H, _W), bg, dtype=int)
    red_set = set(reds)

    for lane_idx in range(_NUM_LANES):
        color = _GREY if lane_idx in red_set else lane_colors[lane_idx]
        _paint_lane(out, lane_idx, lane_shapes[lane_idx], color, counts[lane_idx], bg, periods[lane_idx])

    dst_state = {
        "c0": _LANE_STARTS[green],
        "period": periods[green],
        "bottom0": _BOTTOM_TOP,
        "occupied": set(_lane_tops(periods[green], counts[green])),
    }
    for lane_idx in reds:
        _place_batch(
            out,
            dst_state,
            _shape_offsets(lane_shapes[lane_idx]),
            counts[lane_idx],
            lane_colors[green],
            bg,
            strict=True,
        )
    return out


def _gen_attempt() -> np.ndarray | None:
    rng = np.random.default_rng()

    bg = int(rng.choice(_BG_CHOICES))
    sprite_palette = [c for c in range(10) if c not in (bg, _GREY)]
    lane_colors = rng.choice(sprite_palette, size=_NUM_LANES, replace=False).tolist()
    lane_shapes = rng.permutation(_SHAPE_KEYS).tolist()
    green = int(rng.integers(_NUM_LANES))
    target_green_period = int(rng.choice(_PERIOD_CHOICES))
    red_count = int(rng.choice((1, 2)))
    others = [i for i in range(_NUM_LANES) if i != green]

    if red_count == 1:
        red_candidates = [i for i in others if lane_shapes[i] != "plus"]
        if not red_candidates:
            return None
        reds = (int(rng.choice(red_candidates)),)
    else:
        red_pairs = list(combinations(others, 2))
        reds = tuple(red_pairs[int(rng.integers(len(red_pairs)))])

    periods = rng.choice(_PERIOD_CHOICES, size=_NUM_LANES, replace=True).tolist()
    periods[green] = target_green_period
    other_period = 4 if target_green_period == 5 else 5
    if red_count == 1:
        periods[reds[0]] = other_period
    else:
        forced_other = reds[int(rng.integers(len(reds)))]
        periods[forced_other] = other_period
        if all(periods[r] == target_green_period for r in reds):
            periods[reds[0]] = other_period

    counts = rng.integers(_MIN_SPRITES, _MAX_SPRITES + 1, size=_NUM_LANES).tolist()
    if red_count == 2:
        if target_green_period == 5:
            for lane_idx in (green, *reds):
                counts[lane_idx] = 2
        else:
            marked_count_choices = [(2, 2, 2), (2, 2, 3), (2, 3, 2), (3, 2, 2)]
            green_count, red0_count, red1_count = marked_count_choices[int(rng.integers(len(marked_count_choices)))]
            counts[green] = green_count
            counts[reds[0]] = red0_count
            counts[reds[1]] = red1_count

    if counts[green] + sum(counts[i] for i in reds) > _lane_capacity(periods[green]):
        return None

    inp = np.full((_H, _W), bg, dtype=int)
    for lane_idx in range(_NUM_LANES):
        _paint_lane(inp, lane_idx, lane_shapes[lane_idx], lane_colors[lane_idx], counts[lane_idx], bg, periods[lane_idx])

    inp[_MARKER_ROW, _LANE_STARTS[green] : _LANE_STARTS[green] + _SPRITE_W] = 3
    for lane_idx in reds:
        c0 = _LANE_STARTS[lane_idx]
        inp[_MARKER_ROW, c0 : c0 + _SPRITE_W] = 2

    expected = _expected_output(bg, lane_colors, lane_shapes, counts, periods, reds, green)
    if not np.array_equal(main(inp), expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(_GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {_GEN_RETRIES} attempts")
