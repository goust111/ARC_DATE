# Combo variant of 9aaea919 (index 4)
# combo_spec: D2/2+D3/0
# Name: 多形状精灵+顶部向下堆叠

from __future__ import annotations

import sys
from itertools import combinations
from pathlib import Path

import numpy as np

_SOLVER_DIR = Path(__file__).resolve().parents[4] / "solvers"
if str(_SOLVER_DIR) not in sys.path:
    sys.path.append(str(_SOLVER_DIR))

from arc_utils import bbox_coords, components_mask, get_bg, paste


"""
Design analysis for the combo variant
=====================================

a) Concrete behavior and trace
------------------------------
- The base geometry is unchanged: 30x30 canvas, four fixed 5-column lanes, 1x5 marker bars on row 29, and a
  period-4 vertical lattice whose legal sprite tops are rows 1, 5, 9, 13, 17, 21, 25.
- The fused change modifies two linked parts at once:
  1. each lane uses one of four distinct 3x5 sprite shapes (plus, T, L, square);
  2. transferred sprites fill the destination lane from the highest empty slot downward.
- Example trace:
  - lane 0 is red with two L sprites, lane 2 is green with two square sprites, and lane 3 is red with one T sprite.
  - Original task: all three transferred sprites would be pluses and would occupy rows 17, 13, 9 above the green
    lane's existing bottom block.
  - This combo: the destination already has square sprites at rows 21 and 25, while the incoming rows are the
    highest empty rows 1, 5, 9. Rows 1 and 5 get recolored L sprites from lane 0, and row 9 gets a recolored T
    sprite from lane 3. The red-lane originals turn grey in place.
  - Difference: the solver must preserve source silhouettes and also reverse the priority of destination slots.

b) Complexity calibration
-------------------------
- Four shapes are enough to make source-shape preservation visible, but they all share one 3x5 slot and one period-4
  lattice, so the task still has a single clean spatial grammar.
- The generator keeps lane colors distinct and shapes unique per lane; this avoids ambiguous matching while leaving
  repeated evidence inside each lane.
- Destination totals are restricted to stay below the full 7-slot capacity, so top-down stacking remains visually
  distinguishable from the original bottom-up rule.

c) Hidden-assumption review of the original main()
--------------------------------------------------
- The original `max(len(component))` heuristic no longer works because plus/T/L/square have different pixel counts.
  The combo solver replaces that with normalized shape-signature grouping inside each lane.
- The original placement code assumed "next slot = nearest empty row above the bottom block". That fails under the
  top-down variant, so the combo solver reconstructs the full slot ladder from the destination lane residue class and
  scans it from small row to large row.
- The original horizontal transfer used a center shift that happened to be safe for one common shape. With multiple
  shapes, the stable invariant is slot-local column offset inside the 5-column lane, so copied sprites preserve their
  original `(row offset, lane-local col offset)` template.
- Occupancy-by-top-row remains valid because every allowed shape is exactly 3 rows tall and all lanes share the same
  4-row period.

d) Placement legality
---------------------
- All shapes are defined inside one 3x5 patch, so copying by lane-local offsets cannot leave the destination lane.
- Legal top rows are clipped to positions where a 3-row sprite fits inside the 30-row canvas.
- The generator only chooses role/count combinations with `green_count + transferred_count < 7`, so every moved batch
  has enough empty top-down slots and never overlaps existing sprites.
- Marker bars stay on row 29 with an empty row 28 above them, keeping markers disconnected from lane sprites.

e) Visual observability
-----------------------
- The shape rule is directly visible because destination stacks mix the destination lane's native shape with one or
  two clearly different transferred silhouettes recolored to the destination lane color.
- The stacking-direction rule is directly visible because new sprites appear in the highest empty lane cells rather
  than attaching to the bottom cluster.
- No invisible feature such as raw color id or generation order is used to decide mapping between lanes.

f) Visual cleanliness
---------------------
- Objects remain partitioned into four vertical lanes with fixed spacing, so even mixed-shape destination stacks stay
  readable.
- Only one marker language is used: red bars for sources and one green bar for the destination.
- Source lanes still communicate transfer by a simple local change to grey, without adding extra clutter.

g) Change inferability
----------------------
- Recoloring is still easy to infer because every transferred sprite adopts the color already present in the
  green-marked lane, while its silhouette continues to match its source lane.
- Because each lane keeps a single repeated shape in the input, a human can identify the copied shape class from the
  source lane before checking the output.

h) Generator strategy
---------------------
- The generator uses output-first construction with an explicit oracle layout:
  1. choose distinct lane colors and a permutation of the four shapes;
  2. choose source/destination roles and counts with spare destination capacity;
  3. build the expected output by painting grey source lanes, native destination sprites, and top-down transferred
     batches in left-to-right source order;
  4. build the input from the original lane shapes/colors plus marker bars;
  5. require `main(input) == oracle_output`.

i) New assumptions introduced by the combo solver
-------------------------------------------------
- Each lane's non-marker components all share one repeated normalized signature; the generator guarantees exactly one
  shape per lane.
- Destination sprites lie on a single period-4 residue class; the generator guarantees that by keeping native
  destination sprites bottom-stacked on the standard lattice.
- Multiple red lanes are processed left-to-right because marker bars are sorted by column before transfer; the
  generator mirrors that same order when constructing the oracle output.
"""


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


def _sorted_sprites(sprites: list[list[tuple[int, int]]]) -> list[list[tuple[int, int]]]:
    return sorted(sprites, key=lambda pts: (min(r for r, _ in pts), min(c for _, c in pts)))


def _extract_lane_sprites(
    g: np.ndarray, cols: range, marker_pts: list[tuple[int, int]], bg: int
) -> list[list[tuple[int, int]]]:
    marker_set = set(marker_pts)
    grouped: dict[tuple[int, int, tuple[tuple[int, int], ...]], list[list[tuple[int, int]]]] = {}
    for pts in _single_color_components_in_cols(g, cols, bg):
        kept = [p for p in pts if p not in marker_set]
        if not kept:
            continue
        grouped.setdefault(_component_signature(kept), []).append(kept)
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


def _lane_slot_rows(occupied_tops: list[int], period: int, sprite_h: int, h: int) -> list[int]:
    """Reconstruct all legal lattice rows for a lane from top to bottom."""
    top0 = min(occupied_tops)
    while top0 - period >= 0:
        top0 -= period
    rows = []
    r = top0
    while r + sprite_h <= h:
        rows.append(r)
        r += period
    return rows


def main(grid: np.ndarray) -> np.ndarray:
    """
    Move marked source-lane sprites into the marked destination lane while preserving source shape and using top-first stacking.

    Input structure:
    - The background is the most common color. Thin red (2) and green (3) bars on the last row mark source and destination lanes.
    - Each of the four fixed lanes contains repeated monochrome sprites on the same 3x5, period-4 lattice, but different lanes can use different sprite shapes.

    Rule:
    - Every sprite in a red-marked lane turns grey (5), and the same number of copies is created in the nearest green-marked lane.
    - Each copied sprite keeps its source lane's exact silhouette, changes to the destination lane's color, and fills the highest empty destination slot first, then proceeds downward.
    - If multiple red lanes feed the same destination, their batches are inserted in left-to-right source order. All marker bars are then erased back to the background.

    Key insight:
    - The task now separates two kinds of evidence: source lanes tell you which shape to copy, while the green lane tells you which color to use and which top-down lattice slots are empty.
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
        sprite_h = max(r for r, _ in dst_sprites[0]) - min(r for r, _ in dst_sprites[0]) + 1
        period = _infer_period(dst_sprites)
        green_state[(dst_c0, dst_c1)] = {
            "center": (dst_c0 + dst_c1) / 2.0,
            "c0": dst_c0,
            "dst_color": dst_color,
            "occupied": set(tops),
            "slots": _lane_slot_rows(tops, period, sprite_h, h),
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

        placed = 0
        for new_top in best["slots"]:
            if placed >= len(src_sprites):
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


# --------------- generator ---------------

# --- CFG: canvas & layout ---
_H, _W = 30, 30
_NUM_LANES = 4
_LANE_STARTS = [3, 9, 15, 21]
_MARKER_ROW = 29
_SPRITE_H, _SPRITE_W = 3, 5

# --- CFG: vertical lattice ---
_BOTTOM_TOP = 25
_PERIOD = 4
_TOP_SLOT0 = _BOTTOM_TOP % _PERIOD
_ALL_TOPS = list(range(_TOP_SLOT0, _BOTTOM_TOP + 1, _PERIOD))

# --- CFG: color palette ---
_BG_CHOICES = [0, 1, 4, 6, 7, 8, 9]
_GREY = 5

# --- CFG: counts & limits ---
_MIN_SPRITES = 1
_MAX_SPRITES = 3
_MAX_LANE_TOTAL = len(_ALL_TOPS)
_GEN_RETRIES = 300
_GEN_CALL_COUNT = 0

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


def _lane_tops(count: int) -> list[int]:
    return [_BOTTOM_TOP - _PERIOD * k for k in range(count - 1, -1, -1)]


def _topdown_added_tops(existing_count: int, gain: int) -> list[int]:
    occupied = set(_lane_tops(existing_count))
    empties = [r for r in _ALL_TOPS if r not in occupied]
    return empties[:gain]


def _bottomup_added_tops(existing_count: int, gain: int) -> list[int]:
    occupied = set(_lane_tops(existing_count))
    empties = [r for r in _ALL_TOPS if r not in occupied]
    return list(reversed(empties))[:gain]


def _paint_lane_at_tops(
    grid: np.ndarray, lane_idx: int, shape_key: str, color: int, tops: list[int], bg: int
) -> None:
    patch = _sprite_patch(shape_key, color, bg)
    c0 = _LANE_STARTS[lane_idx]
    for r0 in tops:
        paste(grid, patch, r0, c0, transparent=bg)


def _paint_lane(grid: np.ndarray, lane_idx: int, shape_key: str, color: int, count: int, bg: int) -> None:
    _paint_lane_at_tops(grid, lane_idx, shape_key, color, _lane_tops(count), bg)


def _assigned_added_tops(
    reds: tuple[int, ...], counts: list[int], added_tops: list[int]
) -> dict[int, list[int]]:
    assigned: dict[int, list[int]] = {}
    cursor = 0
    for lane_idx in sorted(reds):
        take = counts[lane_idx]
        assigned[lane_idx] = added_tops[cursor : cursor + take]
        cursor += take
    assert cursor == len(added_tops)
    return assigned


def _feasible_roles(counts: list[int]) -> dict[int, list[tuple[tuple[int, ...], int]]]:
    grouped: dict[int, list[tuple[tuple[int, ...], int]]] = {1: [], 2: []}
    for green in range(_NUM_LANES):
        others = [i for i in range(_NUM_LANES) if i != green]
        for red_count in (1, 2):
            for reds in combinations(others, red_count):
                total = counts[green] + sum(counts[i] for i in reds)
                if total < _MAX_LANE_TOTAL:
                    grouped[red_count].append((reds, green))
    return grouped


def _preferred_role_targets(call_idx: int) -> tuple[int, int]:
    """Cycle branch targets so repeated generation covers both red-count branches and all green lanes."""
    target_red_count = 1 + ((call_idx // _NUM_LANES) % 2)
    target_green = call_idx % _NUM_LANES
    return target_red_count, target_green


def _oracle_output(
    bg: int,
    lane_colors: list[int],
    lane_shapes: list[str],
    input_counts: list[int],
    reds: tuple[int, ...],
    green: int,
) -> np.ndarray:
    gain = sum(input_counts[i] for i in reds)
    added_tops = _topdown_added_tops(input_counts[green], gain)
    assert len(added_tops) == gain

    out = np.full((_H, _W), bg, dtype=int)
    red_set = set(reds)

    for lane_idx in range(_NUM_LANES):
        if lane_idx == green:
            _paint_lane(out, lane_idx, lane_shapes[lane_idx], lane_colors[lane_idx], input_counts[lane_idx], bg)
        elif lane_idx in red_set:
            _paint_lane(out, lane_idx, lane_shapes[lane_idx], _GREY, input_counts[lane_idx], bg)
        else:
            _paint_lane(out, lane_idx, lane_shapes[lane_idx], lane_colors[lane_idx], input_counts[lane_idx], bg)

    for lane_idx, tops in _assigned_added_tops(reds, input_counts, added_tops).items():
        _paint_lane_at_tops(out, green, lane_shapes[lane_idx], lane_colors[green], tops, bg)

    return out


def _build_input(
    bg: int,
    lane_colors: list[int],
    lane_shapes: list[str],
    input_counts: list[int],
    reds: tuple[int, ...],
    green: int,
) -> np.ndarray:
    inp = np.full((_H, _W), bg, dtype=int)
    for lane_idx in range(_NUM_LANES):
        _paint_lane(inp, lane_idx, lane_shapes[lane_idx], lane_colors[lane_idx], input_counts[lane_idx], bg)

    green_c0 = _LANE_STARTS[green]
    inp[_MARKER_ROW, green_c0 : green_c0 + _SPRITE_W] = 3
    for lane_idx in reds:
        c0 = _LANE_STARTS[lane_idx]
        inp[_MARKER_ROW, c0 : c0 + _SPRITE_W] = 2
    return inp


def _gen_attempt(call_idx: int) -> np.ndarray | None:
    rng = np.random.default_rng()
    preferred_red_count, preferred_green = _preferred_role_targets(call_idx)
    fallback: tuple[list[int], list[tuple[tuple[int, ...], int]]] | None = None
    for _ in range(24):
        input_counts = rng.integers(_MIN_SPRITES, _MAX_SPRITES + 1, size=_NUM_LANES).tolist()
        feasible = _feasible_roles(input_counts)
        candidate_roles = [
            role for role in feasible.get(preferred_red_count, []) if role[1] == preferred_green
        ]
        if candidate_roles:
            break
        if fallback is None:
            preferred_roles = feasible.get(preferred_red_count, [])
            if preferred_roles:
                fallback = (input_counts, preferred_roles)
            else:
                all_roles = feasible[1] + feasible[2]
                if all_roles:
                    fallback = (input_counts, all_roles)
    else:
        if fallback is None:
            return None
        input_counts, candidate_roles = fallback

    reds, green = candidate_roles[int(rng.integers(len(candidate_roles)))]

    bg = int(rng.choice(_BG_CHOICES))
    sprite_palette = [c for c in range(10) if c not in (bg, _GREY)]
    lane_colors = rng.choice(sprite_palette, size=_NUM_LANES, replace=False).tolist()
    lane_shapes = rng.permutation(_SHAPE_KEYS).tolist()

    gain = sum(input_counts[i] for i in reds)
    topdown = _topdown_added_tops(input_counts[green], gain)
    bottomup = _bottomup_added_tops(input_counts[green], gain)
    if len(topdown) != gain or topdown == bottomup:
        return None

    inp = _build_input(bg, lane_colors, lane_shapes, input_counts, reds, green)
    expected = _oracle_output(bg, lane_colors, lane_shapes, input_counts, reds, green)

    try:
        actual = main(inp)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    global _GEN_CALL_COUNT
    for _ in range(_GEN_RETRIES):
        grid = _gen_attempt(_GEN_CALL_COUNT)
        if grid is not None:
            _GEN_CALL_COUNT += 1
            return grid
    raise RuntimeError(f"failed to generate a valid input after {_GEN_RETRIES} attempts")
