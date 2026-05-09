# Combo variant of 9aaea919 (index 20)
# combo_spec: D9/0+D10/0
# Name: 可变车道数+可变周期

# Design analysis
# ----------------
# fused_variant_spec:
#   a) concrete_behavior_and_trace:
#     - Keep the 30x30 canvas, 3x5 plus/diamond sprite, bottom anchor row 25, red source bars,
#       green target bar, and "source turns grey / target gains equal count" conservation rule.
#     - Change 1 (D9): the lane count is no longer fixed at four; each puzzle has 3..6 lanes, and
#       each lane is still a 5-column window whose start columns are derived from the current layout.
#     - Change 2 (D10): each lane has its own vertical period, chosen from {4, 5}. Transfers must
#       continue the destination lane's cadence instead of using a global spacing.
#     - Example trace:
#       num_lanes = 5, starts = [1, 7, 13, 18, 24], periods = [4, 5, 4, 5, 4].
#       Suppose lane 1 (period 5) is red with tops [20, 25], lane 3 (period 5) is also red with
#       tops [20, 25], and lane 2 (period 4) is green with tops [21, 25].
#       Original task assumptions would fail twice: it would expect fixed columns and would extend
#       the green lane on step 4 only accidentally. The fused rule instead reads the 1x5 markers to
#       locate the three active lanes, greys the 2 + 2 source sprites, and inserts four new sprites
#       into lane 2 at the empty target tops [17, 13, 9, 5], which continue the destination period
#       4 rather than the sources' period 5. The important difference is that horizontal layout and
#       vertical cadence are both inferred from the image itself.
#   b) complexity_calibration:
#     - Restrict lane counts to 3..6 and periods to {4, 5}. Both freedoms are visually obvious but
#       still low-branching enough to be learnable from 2-4 ARC examples.
#     - Marked lanes are required to have at least two visible sprites so their periods are directly
#       inferable from repeated top-row gaps; no hidden fallback on a hard-coded default is needed.
#     - The generator also forces the marked set (reds + green) to contain both period values, so
#       the "per-lane cadence" change is visible in the examples instead of hiding only in unmarked
#       lanes.
#   c) implicit_assumption_audit_of_seed_solver:
#     - Fixed lane centers/starts are invalid once lane count varies. The solver must derive each
#       active lane window from the marker's exact 1x5 bounding box.
#     - Accepting any thin marker component is unsafe when 6 lanes can touch: adjacent red bars would
#       merge into a 1x10 component. The fused solver accepts only exact 1x5 horizontal bars, and
#       the generator forbids selecting two touching red lanes simultaneously.
#     - Global period assumptions are invalid. Period is inferred per lane from repeated sprite tops.
#     - "Single visible target sprite is enough" is also invalid for period inference, so the
#       generator guarantees at least two visible sprites in every marked lane in the input.
#     - Target placement may not assume fixed capacity 7; capacity depends on the target period:
#       period 4 fits 7 sprites, period 5 fits 6.
#   d) placement_legality:
#     - Every lane occupies a disjoint 5-column strip. With 6 lanes the strips may touch edge-to-edge
#       but never overlap.
#     - Row 29 still holds markers; sprites end at row <= 27, leaving row 28 empty so markers stay
#       disconnected from lane sprites.
#     - Output is built first, then input is derived by restoring source colors and erasing only the
#       gained top sprites from the green lane. Oracle checking with main(inp) == out is retained.
#   e) visual_observability:
#     - Lane-count variation is directly visible: 3-lane boards have wide gaps, 6-lane boards look
#       compressed, and intermediate counts visibly differ.
#     - Period variation is also directly visible because the vertical gaps in marked lanes repeat and
#       the marked set is forced to include both period values across the active transfer lanes.
#     - No rule depends on numeric color ordering. Colors merely label lanes; geometry and marker
#       position reveal the transformation.
#   f) visual_cleanliness:
#     - Lane colors remain pairwise distinct, which avoids same-color touching sprites across adjacent
#       lane boundaries when lanes compress.
#     - Marker bars are only one row tall and aligned with lane windows, so the board stays readable
#       even when the 6-lane case is dense.
#   g) new_solver_assumptions_to_guarantee:
#     - Each red-marked lane and the green-marked lane has at least two visible sprites in the input.
#     - There is exactly one green lane, one or two red lanes, and any pair of red lanes that would
#       be edge-adjacent in the current layout is disallowed by the generator.
#     - The destination lane has enough capacity under its own period to absorb all transferred
#       sprites without leaving the board.

import numpy as np

from arc_utils import bbox_coords, components_mask, get_bg, paste
from itertools import combinations


def _find_marker_bars(g: np.ndarray, marker_color: int) -> list[list[tuple[int, int]]]:
    """Marker bars are exact horizontal 1x5 components of the given color."""
    bars = []
    for pts_arr in components_mask(g == marker_color, connectivity=4):
        r0, c0, r1, c1 = bbox_coords(pts_arr)
        if (r1 - r0 + 1, c1 - c0 + 1, len(pts_arr)) == (1, 5, 5):
            pts = [(int(r), int(c)) for r, c in pts_arr]
            bars.append((r0, c0, pts))
    bars.sort()
    return [pts for _r0, _c0, pts in bars]


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


def _sorted_lane_sprites(
    g: np.ndarray, cols: range, bg: int, marker_pts: list[tuple[int, int]]
) -> list[tuple[int, int, list[tuple[int, int]]]]:
    marker_set = set(marker_pts)
    comps = _single_color_components_in_cols(g, cols, bg)
    comps = [[p for p in pts if p not in marker_set] for pts in comps]
    comps = [pts for pts in comps if pts]
    if not comps:
        return []

    sprite_sizes = [len(pts) for pts in comps if len(pts) != len(marker_pts)]
    if not sprite_sizes:
        return []
    sprite_sz = max(sprite_sizes)
    sprites = [pts for pts in comps if len(pts) == sprite_sz]

    tagged = []
    for pts in sprites:
        rmin = min(r for r, _ in pts)
        cmin = min(c for _, c in pts)
        tagged.append((rmin, cmin, pts))
    tagged.sort()
    return tagged


def _infer_period(sprite_rows: list[int]) -> int:
    diffs = [b - a for a, b in zip(sprite_rows, sprite_rows[1:]) if b > a]
    assert diffs, "expected at least two visible sprites to infer period"
    period = int(np.gcd.reduce(diffs))
    assert period > 0, "lane period must be positive"
    return period


def _lane_state_from_marker(g: np.ndarray, marker_pts: list[tuple[int, int]], bg: int) -> dict | None:
    _, c0, _, c1 = bbox_coords(np.array(marker_pts))
    cols = range(c0, c1 + 1)
    sprites_sorted = _sorted_lane_sprites(g, cols, bg, marker_pts)
    if len(sprites_sorted) < 2:
        return None

    tops = [rmin for rmin, _cmin, _pts in sprites_sorted]
    period = _infer_period(tops)
    color = int(g[sprites_sorted[0][2][0][0], sprites_sorted[0][2][0][1]])
    center = (c0 + c1) / 2.0
    return {
        "center": center,
        "color": color,
        "period": period,
        "occupied": set(tops),
        "bottom0": max(tops),
        "sprites_sorted": sprites_sorted,
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Transfer every marked source stack into the marked destination stack while inferring both lane layout and lane cadence from the grid.

    Input structure:
    - The background is the most common color. There are 3-6 lane windows, each exactly 5 columns wide, but their horizontal positions are not fixed; red 1x5 bars and one green 1x5 bar on the last row identify the active source and destination lanes directly.
    - Each lane contains repeated monochrome plus/diamond sprites, and different lanes may use different vertical periods even though the bottom visible sprite remains anchored near the same row.

    Rule:
    - Every sprite in each red-marked lane is recolored grey in place.
    - The same number of sprites is recreated in the nearest green-marked lane, shifted into that lane's columns, recolored to the destination lane color, and inserted upward using the destination lane's own inferred period.
    - All marker bars are then removed back to the background.

    Key insight:
    - The markers no longer just label fixed slots: they reveal where the lane windows are, while the repeated sprite tops reveal each lane's cadence. The transfer still conserves sprite count, but both the horizontal layout and the vertical stacking rhythm must be inferred per puzzle.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = get_bg(g)

    red_markers = _find_marker_bars(g, 2)
    green_markers = _find_marker_bars(g, 3)
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

        src_sprites_sorted = src_state["sprites_sorted"]
        for _rmin, _cmin, pts in src_sprites_sorted:
            for r, c in pts:
                out[r, c] = 5

        rc = src_state["center"]
        best = min(green_states, key=lambda st: abs(st["center"] - rc))

        src_ref_r0, _src_ref_c0, src_ref_pts = src_sprites_sorted[0]
        dc_shift = int(round(best["center"] - rc))
        src_offsets = [(r - src_ref_r0, c + dc_shift) for r, c in src_ref_pts]
        sprite_h = max(dr for dr, _c in src_offsets) + 1

        placed = 0
        need = len(src_sprites_sorted)
        k = 0
        while placed < need:
            new_top = best["bottom0"] - k * best["period"]
            k += 1
            if new_top < 0 or new_top + sprite_h > h:
                if new_top < 0:
                    break
                continue
            if new_top in best["occupied"]:
                continue

            ok = True
            for dr, c in src_offsets:
                rr = new_top + dr
                if not (0 <= rr < h and 0 <= c < w):
                    ok = False
                    break
                if int(out[rr, c]) != bg:
                    ok = False
                    break
            if not ok:
                continue

            for dr, c in src_offsets:
                out[new_top + dr, c] = best["color"]
            best["occupied"].add(new_top)
            placed += 1

    for marker_group in (red_markers, green_markers):
        for pts in marker_group:
            for r, c in pts:
                out[r, c] = bg

    return out


# --------------- generator ---------------

# --- CFG: canvas & sprite geometry ---
_H, _W = 30, 30
_MARKER_ROW = 29
_SPRITE_H, _SPRITE_W = 3, 5
_BOTTOM_TOP = 25

# --- CFG: fused variation ---
_MIN_LANES = 3
_MAX_LANES = 6
_PERIOD_CHOICES = (4, 5)

# --- CFG: color palette ---
_BG_CHOICES = [0, 1, 4, 6, 7, 8, 9]
_GREY = 5

# --- CFG: counts & limits ---
_MIN_SPRITES = 2
_MAX_SPRITES = 3
_GEN_RETRIES = 600
_GEN_CALL_INDEX = 0


def _sprite_patch(color: int, bg: int) -> np.ndarray:
    """Return the 3x5 plus/diamond sprite patch with transparent corners."""
    patch = np.full((_SPRITE_H, _SPRITE_W), bg, dtype=int)
    patch[0, 1:4] = color
    patch[1, :] = color
    patch[2, 1:4] = color
    return patch


def _lane_layout(num_lanes: int) -> list[int]:
    """
    Return lane start columns for `num_lanes`.

    Extra width is distributed round-robin across interior gaps first, then the two outer margins,
    so lower lane counts spread apart while 6 lanes compress into touching windows.
    """
    assert _MIN_LANES <= num_lanes <= _MAX_LANES
    used = num_lanes * _SPRITE_W
    extra = _W - used
    assert extra >= 0

    gaps = [0] * (num_lanes - 1)
    left_margin = 0
    right_margin = 0
    slots: list[tuple[str, int | None]] = [("gap", i) for i in range(num_lanes - 1)] + [
        ("left", None),
        ("right", None),
    ]
    for k in range(extra):
        kind, idx = slots[k % len(slots)]
        if kind == "gap":
            gaps[int(idx)] += 1
        elif kind == "left":
            left_margin += 1
        else:
            right_margin += 1

    starts = [left_margin]
    for gap in gaps:
        starts.append(starts[-1] + _SPRITE_W + gap)
    assert starts[-1] + _SPRITE_W <= _W - right_margin
    return starts


def _lane_capacity(period: int) -> int:
    """Maximum number of sprites that fit when the bottom top-row is fixed at row 25."""
    return (_BOTTOM_TOP // period) + 1


def _lane_tops(period: int, count: int) -> list[int]:
    """Return sprite top rows for `count` sprites, ordered top-to-bottom."""
    return [_BOTTOM_TOP - period * k for k in range(count - 1, -1, -1)]


def _paint_lane(grid: np.ndarray, lane_start: int, color: int, count: int, bg: int, period: int) -> None:
    """Stamp `count` sprites of `color` into the lane starting at `lane_start`."""
    patch = _sprite_patch(color, bg)
    for r0 in _lane_tops(period, count):
        paste(grid, patch, r0, lane_start, transparent=bg)


def _touching_neighbor_pairs(starts: list[int]) -> set[tuple[int, int]]:
    """Return adjacent lane-index pairs whose 5-wide windows touch edge-to-edge."""
    pairs = set()
    for i in range(len(starts) - 1):
        if starts[i + 1] == starts[i] + _SPRITE_W:
            pairs.add((i, i + 1))
    return pairs


def _feasible_roles(
    counts: list[int], periods: list[int], starts: list[int]
) -> dict[int, list[tuple[tuple[int, ...], int]]]:
    """Return feasible (reds, green) assignments grouped by red-lane count."""
    touching = _touching_neighbor_pairs(starts)
    grouped: dict[int, list[tuple[tuple[int, ...], int]]] = {1: [], 2: []}
    num_lanes = len(counts)
    for green in range(num_lanes):
        others = [i for i in range(num_lanes) if i != green]
        cap = _lane_capacity(periods[green])
        for red_count in (1, 2):
            for reds in combinations(others, red_count):
                if counts[green] + sum(counts[i] for i in reds) > cap:
                    continue
                if red_count == 2 and tuple(sorted(reds)) in touching:
                    continue
                marked_periods = {periods[green], *(periods[i] for i in reds)}
                if len(marked_periods) < 2:
                    continue
                grouped[red_count].append((reds, green))
    return grouped


def _gen_attempt(num_lanes: int, preferred_red_count: int) -> np.ndarray | None:
    rng = np.random.default_rng()

    lane_starts = _lane_layout(num_lanes)

    bg = int(rng.choice(_BG_CHOICES))
    sprite_palette = [c for c in range(10) if c not in (bg, _GREY)]
    lane_colors = rng.choice(sprite_palette, size=num_lanes, replace=False).tolist()
    input_counts = rng.integers(_MIN_SPRITES, _MAX_SPRITES + 1, size=num_lanes).tolist()
    periods = rng.choice(_PERIOD_CHOICES, size=num_lanes, replace=True).tolist()

    if len(set(periods)) < 2:
        return None

    feasible = _feasible_roles(input_counts, periods, lane_starts)
    feasible_red_counts = []
    if feasible[preferred_red_count]:
        feasible_red_counts.append(preferred_red_count)
    feasible_red_counts.extend(rc for rc in (1, 2) if rc != preferred_red_count and feasible[rc])
    if not feasible_red_counts:
        return None

    red_count = feasible_red_counts[0]
    reds, green = feasible[red_count][int(rng.integers(len(feasible[red_count])))]
    red_set = set(reds)
    gain = sum(input_counts[i] for i in reds)

    output_counts = input_counts[:]
    output_counts[green] += gain

    out = np.full((_H, _W), bg, dtype=int)
    for lane_idx, lane_start in enumerate(lane_starts):
        color = _GREY if lane_idx in red_set else lane_colors[lane_idx]
        _paint_lane(out, lane_start, color, output_counts[lane_idx], bg, periods[lane_idx])

    inp = out.copy()
    for lane_idx in reds:
        _paint_lane(inp, lane_starts[lane_idx], lane_colors[lane_idx], input_counts[lane_idx], bg, periods[lane_idx])

    green_c0 = lane_starts[green]
    for r0 in _lane_tops(periods[green], output_counts[green])[:gain]:
        inp[r0 : r0 + _SPRITE_H, green_c0 : green_c0 + _SPRITE_W] = bg
    inp[_MARKER_ROW, green_c0 : green_c0 + _SPRITE_W] = 3

    for lane_idx in reds:
        c0 = lane_starts[lane_idx]
        inp[_MARKER_ROW, c0 : c0 + _SPRITE_W] = 2

    if not np.array_equal(main(inp), out):
        return None
    return inp


def generate_input() -> np.ndarray:
    global _GEN_CALL_INDEX

    num_lanes_cycle = list(range(_MIN_LANES, _MAX_LANES + 1))
    target_num_lanes = num_lanes_cycle[_GEN_CALL_INDEX % len(num_lanes_cycle)]
    preferred_red_count = 1 if (_GEN_CALL_INDEX % 2 == 0) else 2

    for _ in range(_GEN_RETRIES):
        grid = _gen_attempt(target_num_lanes, preferred_red_count)
        if grid is not None:
            _GEN_CALL_INDEX += 1
            return grid
    raise RuntimeError(f"failed to generate a valid input after {_GEN_RETRIES} attempts")
