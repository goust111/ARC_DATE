# Variant of 9aaea919
# Dimension: D2 — 源对象类型 (src)
# Change: 十字/菱形精灵(3x5外接矩形，11像素)，每车道单色，均匀垂直排列... → 每车道不同形状精灵

"""
Design analysis for the D2 variant
==================================

source_task:
- None. No external reference solver is consulted.

a) Concrete behavior and trace
------------------------------
- The canvas, lane positions, marker bars, vertical lattice, and recolor-to-grey / recolor-to-destination
  behavior all stay the same as the original task.
- The changed part is the source object type: the four lanes now carry four distinct monochrome 3x5 sprites.
  A sample lane assignment is:
  - plus:
      .XXX.
      XXXXX
      .XXX.
  - T:
      XXXXX
      ..X..
      ..X..
  - L:
      X....
      X....
      XXXXX
  - square:
      .XXX.
      .XXX.
      .XXX.
- Each lane still repeats only one shape down its own vertical lattice, but different lanes use different shapes.
- A red marker still means "this lane is a source"; a green marker still means "this lane is the destination".
- When a source lane transfers, every copied sprite keeps the source lane's exact pixel template and only changes
  color to the destination lane color. The source copies left behind become grey.

- Small trace:
  - Original logic: if a red lane with 2 plus sprites transfers into a green lane with 2 plus sprites, solver can
    identify the template only by component size because every lane uses the same plus shape.
  - Variant logic: if a red lane with 2 L sprites transfers into a green lane with 2 square sprites, the output
    destination lane must contain 2 lower square sprites plus 2 higher L sprites, all recolored to the destination
    lane color. Shape identity and color identity now come from different lanes.
  - Difference: the solver can no longer infer "the sprite template" from the destination lane or from max area alone;
    it must carry the source lane's actual shape across the move.

b) Complexity calibration
-------------------------
- Four shapes are enough to make the D2 change obvious by eye, but still small enough that the rule remains learnable.
- All shapes live in the same 3x5 slot and share the same 4-row vertical period, so the task does not gain a second
  hidden transformation about scale, spacing, or orientation.
- Shapes are visually distinct and asymmetric enough that preserving shape through recoloring is observable in 2-3
  examples.

c) Hidden assumptions in the original main()
--------------------------------------------
- `_single_color_components_in_cols(...)` is still valid because each lane still contains only monochrome objects.
- The original `dst_sz = max(...)` / `src_sz = max(...)` assumed "all useful sprites have the same pixel count".
  That is unsafe now because plus=11 pixels, T/L=7, square=9.
- The original placement code reused the top-most source sprite as the template and shifted it horizontally into the
  destination lane. That remains valid if the shift preserves slot-local columns instead of assuming a centered plus.
- The original occupancy logic tracked only top rows. That remains valid because all new shapes still occupy exactly
  3 rows and the period stays 4.

- Therefore the variant replaces size-based sprite selection with signature-based grouping inside each lane:
  normalize each component by its bbox-local coordinates, group equal signatures, and use the repeated signature as
  that lane's sprite type.

d) Placement legality
---------------------
- Every sprite stays inside a fixed 5-column lane slot, so different lanes cannot overlap horizontally.
- Every sprite is 3 rows tall and the lattice period stays 4, so vertically adjacent slots do not overlap.
- The generator restricts `green_count + sum(red_counts) <= 7`, keeping the topmost possible inserted sprite at row 1.
- Markers remain on row 29, one blank row below the lowest sprite row 27, so markers never connect to sprites.

e) Visual observability
-----------------------
- No invisible property such as color id order is used for sorting or mapping.
- The only new information a solver must infer is visible: the copied objects in the green lane keep their source
  silhouettes while changing to the destination lane color.
- Shape assignment is permuted across lanes per sample so the task cannot collapse into "lane 0 always means plus".

f) Visual cleanliness
---------------------
- All objects still sit in the same four clean vertical lanes.
- Markers remain separate horizontal bars at the bottom.
- Mixed-shape destination stacks are readable because shapes never overlap and share one destination color.

g) Color-change inferability
----------------------------
- Colors remain lane-level and distinct, just as in the original task.
- The structural cue for recoloring is unchanged: the destination lane color is the color already present in the
  green-marked lane, and every transferred sprite adopts that color.
- Because shape now stays source-derived, humans can disentangle "shape comes from source lane" from
  "color comes from destination lane".

h) Generator strategy adjustment
--------------------------------
- The original generator used a simple output-first construction because the green lane stayed one uniform shape.
- In this variant, the output green lane can contain its native shape plus one or two different incoming source shapes.
  Building that output directly is no longer simpler than building a valid input.
- The generator therefore switches to input-first construction:
  1. choose four distinct lane colors and a permutation of the four shapes;
  2. choose counts and red/green roles subject to capacity;
  3. paint the input lanes and markers directly;
  4. validate with `main(input)`.

i) New assumptions introduced by the variant solver
---------------------------------------------------
- Assumption: inside one lane, the repeated non-marker components all share one normalized signature.
  Generator guarantee: each lane paints only one shape.
- Assumption: transferred shapes keep the same slot-local horizontal offsets within the 5-column lane.
  Generator guarantee: every shape is defined as a 3x5 patch aligned to the lane slot.
- Assumption: destination occupancy can still be indexed by top row only.
  Generator guarantee: all shapes have height 3 and all lane tops come from the same 4-row lattice.
"""

from __future__ import annotations

import sys
from itertools import combinations
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[5]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

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
        cols_in = {int(g[r, c]) for r, c in pts}
        if len(cols_in) == 1:
            out.append(pts)
    return out


def _component_signature(pts: list[tuple[int, int]]) -> tuple[int, int, tuple[tuple[int, int], ...]]:
    arr = np.array(pts, dtype=int)
    r0, c0, r1, c1 = bbox_coords(arr)
    norm = tuple(sorted((int(r - r0), int(c - c0)) for r, c in pts))
    return (r1 - r0 + 1, c1 - c0 + 1, norm)


def _sorted_sprites(pts_groups: list[list[tuple[int, int]]]) -> list[list[tuple[int, int]]]:
    return sorted(pts_groups, key=lambda pts: (min(r for r, _ in pts), min(c for _, c in pts)))


def _extract_lane_sprites(
    g: np.ndarray, cols: range, marker_pts: list[tuple[int, int]], bg: int
) -> list[list[tuple[int, int]]]:
    """
    Extract the repeated sprite components from one lane.

    In the original task, taking the maximum-area component size was enough because every lane used the same shape.
    Here we instead group by normalized shape signature and keep the repeated signature for that lane.
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


def _infer_period(sprites: list[list[tuple[int, int]]]) -> int:
    tops = sorted(min(r for r, _ in pts) for pts in sprites)
    if len(tops) >= 2:
        diffs = [b - a for a, b in zip(tops, tops[1:]) if b - a > 0]
        if diffs:
            return int(np.gcd.reduce(diffs))
    r0, _c0, r1, _c1 = bbox_coords(np.array(sprites[0], dtype=int))
    return (r1 - r0 + 1) + 1


def main(grid: np.ndarray) -> np.ndarray:
    """
    Move sprites from red-marked lanes into the green-marked lane while preserving source shapes.

    Input structure:
    - The background is the most common color. Thin red (2) and green (3) bars on the bottom row mark source and destination lanes.
    - The four lanes still use a regular 3x5-by-period-4 vertical lattice, but different lanes can carry different monochrome sprite shapes.

    Rule:
    - Every sprite in a red lane turns grey (5), and the same number of sprites is recreated in the nearest green lane.
    - Recreated sprites keep the exact source-lane silhouette, adopt the destination lane's color, and fill empty lattice slots from the bottom upward.
    - If multiple red lanes feed the same destination, their batches stack in left-to-right source-lane order. All marker bars are then erased.

    Key insight:
    - Color flow and shape flow are now decoupled: the destination lane contributes the color, but each transferred batch must reuse its own source template rather than the destination lane's native sprite.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = get_bg(g)

    red_markers = sorted(_find_marker_bars(g, 2), key=lambda pts: bbox_coords(np.array(pts, dtype=int))[1])
    green_markers = sorted(_find_marker_bars(g, 3), key=lambda pts: bbox_coords(np.array(pts, dtype=int))[1])
    assert red_markers and green_markers, "expected both red and green marker bars to be present"

    green_state = {}
    for gm in green_markers:
        _r0, dst_c0, _r1, dst_c1 = bbox_coords(np.array(gm, dtype=int))
        dst_cols = range(dst_c0, dst_c1 + 1)
        dst_sprites = _extract_lane_sprites(g, dst_cols, gm, bg)
        if not dst_sprites:
            continue

        tops = sorted(min(r for r, _ in pts) for pts in dst_sprites)
        dst_color = int(g[dst_sprites[0][0][0], dst_sprites[0][0][1]])
        green_state[(dst_c0, dst_c1)] = {
            "center": (dst_c0 + dst_c1) / 2.0,
            "c0": dst_c0,
            "dst_color": dst_color,
            "period": _infer_period(dst_sprites),
            "occupied": set(tops),
            "bottom0": max(tops),
        }

    assert green_state, "expected at least one green lane with valid sprites"

    out = g.copy()

    for rm in red_markers:
        _r0, src_c0, _r1, src_c1 = bbox_coords(np.array(rm, dtype=int))
        src_cols = range(src_c0, src_c1 + 1)
        src_sprites = _extract_lane_sprites(g, src_cols, rm, bg)
        if not src_sprites:
            continue

        for pts in src_sprites:
            for r, c in pts:
                out[r, c] = 5

        src_center = (src_c0 + src_c1) / 2.0
        best = min(green_state.values(), key=lambda st: abs(st["center"] - src_center))

        src_ref_pts = src_sprites[0]
        src_ref_r0 = min(r for r, _ in src_ref_pts)
        src_offsets = [(r - src_ref_r0, c - src_c0) for r, c in src_ref_pts]
        sprite_h = max(dr for dr, _dc in src_offsets) + 1

        placed = 0
        probe = 0
        while placed < len(src_sprites):
            new_top = best["bottom0"] - probe * best["period"]
            probe += 1
            if new_top < 0 or new_top + sprite_h > h:
                if new_top < 0:
                    break
                continue
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
                out[new_top + dr, best["c0"] + dc] = best["dst_color"]
            best["occupied"].add(new_top)
            placed += 1

    for rm in red_markers:
        for r, c in rm:
            out[r, c] = bg
    for gm in green_markers:
        for r, c in gm:
            out[r, c] = bg

    return out


# --------------- generator (D2 variant) ---------------

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
_MAX_LANE_TOTAL = 7
_GEN_RETRIES = 300

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
    rows = _SHAPE_MASKS[shape_key]
    for r, row in enumerate(rows):
        for c, ch in enumerate(row):
            if ch == "X":
                patch[r, c] = color
    return patch


def _lane_tops(count: int) -> list[int]:
    return [_BOTTOM_TOP - _PERIOD * k for k in range(count - 1, -1, -1)]


def _paint_lane(grid: np.ndarray, lane_idx: int, shape_key: str, color: int, count: int, bg: int) -> None:
    patch = _sprite_patch(shape_key, color, bg)
    c0 = _LANE_STARTS[lane_idx]
    for r0 in _lane_tops(count):
        paste(grid, patch, r0, c0, transparent=bg)


def _feasible_roles(
    counts: list[int], lane_shapes: list[str]
) -> dict[int, list[tuple[tuple[int, ...], int]]]:
    grouped: dict[int, list[tuple[tuple[int, ...], int]]] = {1: [], 2: []}
    for green in range(_NUM_LANES):
        others = [i for i in range(_NUM_LANES) if i != green]
        for red_count in (1, 2):
            for reds in combinations(others, red_count):
                if counts[green] + sum(counts[i] for i in reds) > _MAX_LANE_TOTAL:
                    continue
                if all(lane_shapes[i] == "plus" for i in reds):
                    continue
                grouped[red_count].append((reds, green))
    return grouped


def _gen_attempt() -> np.ndarray | None:
    rng = np.random.default_rng()

    bg = int(rng.choice(_BG_CHOICES))
    sprite_palette = [c for c in range(10) if c not in (bg, _GREY)]
    lane_colors = rng.choice(sprite_palette, size=_NUM_LANES, replace=False).tolist()
    lane_shapes = rng.permutation(_SHAPE_KEYS).tolist()
    input_counts = rng.integers(_MIN_SPRITES, _MAX_SPRITES + 1, size=_NUM_LANES).tolist()

    feasible = _feasible_roles(input_counts, lane_shapes)
    feasible_red_counts = [rc for rc in (1, 2) if feasible[rc]]
    if not feasible_red_counts:
        return None

    red_count = int(rng.choice(feasible_red_counts))
    reds, green = feasible[red_count][int(rng.integers(len(feasible[red_count])))]

    inp = np.full((_H, _W), bg, dtype=int)
    for lane_idx in range(_NUM_LANES):
        _paint_lane(inp, lane_idx, lane_shapes[lane_idx], lane_colors[lane_idx], input_counts[lane_idx], bg)

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
