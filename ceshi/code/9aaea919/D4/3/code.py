# Variant of 9aaea919
# Dimension: D4 — 颜色变换规则 (transform)
# Change: color_transfer — 转移的精灵重新着色为目标车道的原有精灵颜色；源车道精灵变灰(5)... → 源精灵变为目标车道色

import numpy as np

from arc_utils import bbox_coords, components_mask, get_bg, paste
from itertools import combinations


# Design analysis for the D4 variant
# ----------------------------------
# source_task:
#   - None. The only intended innovation is the color rule, so geometry and stacking stay
#     close to the original task.
#
# concretized_behavior:
#   - The canvas remains 30x30 with four fixed 5-column lanes.
#   - Each lane still contains repeated 3x5 plus/diamond sprites on a period-4 vertical lattice.
#   - Input colors are still all distinct per lane, which makes it visible that the source lanes
#     start with their own colors.
#   - Red 1x5 bars on the last row mark source lanes; one green 1x5 bar marks the destination lane.
#   - Output difference versus the original task:
#     source-lane sprites stay in place but no longer become grey; instead they are recolored to the
#     destination lane's sprite color, and the transferred copies in the green lane use that same color.
#
# behavior_trace:
#   - Example: lane A has 2 orange sprites and a red marker; lane B has 1 blue sprite and a green marker.
#   - Original rule:
#     lane A -> 2 grey sprites in place; lane B -> gains 2 new blue sprites stacked upward.
#   - Variant rule here:
#     lane A -> 2 blue sprites in place; lane B -> gains 2 new blue sprites stacked upward.
#   - The transfer count logic is unchanged; only the recolor target for source lanes changes.
#
# complexity_calibration:
#   - Keeping one green lane and 1-2 red lanes preserves the original relational structure.
#   - Input lane colors stay pairwise distinct and exclude grey, so the new recolor is always visible.
#   - The rule uses only one destination color per sample, repeated across both the source lanes and the
#     destination lane. That repetition gives enough evidence for humans to infer "unify to target color"
#     from a few examples.
#
# implicit_assumptions_audit:
#   - In the original solver, `out[r, c] = 5` was safe because every source lane always ended as grey.
#     That assumption is invalid now; source recolor must depend on the matched destination lane.
#   - The nearest-green matching logic still works because the destination color is read from the green
#     lane itself, and the generator continues to provide exactly one green marker.
#   - Placement emptiness checks still inspect destination columns only, so recoloring source lanes first
#     cannot accidentally block later placements.
#   - Component extraction still relies on a blank row between the lowest sprite and the marker bar.
#     The generator keeps row 28 empty by fixing the marker row at 29 and bottom sprite top at 25.
#
# placement_legality:
#   - Output-first generation is preserved.
#   - The generator only chooses role/count assignments where destination_count + transferred_count <= 7,
#     matching the available lattice slots and preventing overflow above row 0.
#   - Source recoloring never changes occupancy, only colors, so no new overlap case is introduced.
#
# visual_observability_and_cleanliness:
#   - The variant is visible because source lanes retain their shapes and positions while changing to the
#     same color as the green lane, rather than fading to a fixed neutral grey.
#   - Markers remain simple 1x5 bars isolated on the last row, so lane roles stay easy to parse.
#   - Output can contain several lanes with the same destination color, but this is orderly because lanes
#     are spatially separated and still aligned on the same clean lattice.
#
# color_inference:
#   - No sorting or branching depends on numeric color ids.
#   - The destination color is inferable from the existing sprites in the green-marked lane, which then
#     propagates to both the copied stack and the recolored source lanes.
#
# new_solver_assumptions_and_generator_guarantees:
#   - Every green-marked lane has at least one sprite so its destination color is observable.
#   - Every source lane has at least one sprite.
#   - Input lane colors are distinct, so a source lane's original color always differs from the
#     destination color in the same sample.


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


def main(grid: np.ndarray) -> np.ndarray:
    """
    Transfer sprite counts into the green-marked lane while unifying colors to that lane.

    Input structure:
    - The background is the most frequent color. Thin red (2) bars mark source lanes and thin green (3) bars mark destination lanes.
    - Each lane contains repeated monochrome plus sprites aligned to the same vertical lattice, with one lane color per lane in the input.

    Rule:
    - Each red-marked lane keeps its sprites in place, but those sprites are repainted to the matched green lane's sprite color instead of turning grey.
    - The same number of source-shaped sprites is also added into the nearest green lane, filling empty lattice slots upward from the bottom in that destination color.
    - All marker bars are removed after the recoloring and stacking step.

    Key insight:
    - The green lane supplies a shared target color. Red lanes contribute their sprite counts to that lane and also adopt the same color themselves, so the final scene shows a lane-wide color unification rather than a greyed-out source.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = get_bg(g)

    red_markers = _find_marker_bars(g, 2)
    green_markers = _find_marker_bars(g, 3)
    assert red_markers and green_markers, "expected both red and green marker bars to be present"

    # Build destination lane state per green marker.
    green_state = {}
    for gm in green_markers:
        _, dst_c0, _, dst_c1 = bbox_coords(np.array(gm))
        dst_cols = range(dst_c0, dst_c1 + 1)
        dst_set = set(gm)
        dst_comps = _single_color_components_in_cols(g, dst_cols, bg)
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
            dst_sprites_sorted.append((rmin, cmin, pts))
        dst_sprites_sorted.sort()
        dst_color = int(g[dst_sprites_sorted[0][2][0][0], dst_sprites_sorted[0][2][0][1]])

        tops = sorted({rmin for rmin, _cmin, _pts in dst_sprites_sorted})
        if len(tops) >= 2:
            diffs = [b - a for a, b in zip(tops, tops[1:]) if b - a > 0]
            period = int(np.gcd.reduce(diffs)) if diffs else 4
        else:
            r0, c0, r1, c1 = bbox_coords(np.array(dst_sprites_sorted[0][2]))
            period = (r1 - r0 + 1) + 1

        gc = (dst_c0 + dst_c1) / 2.0
        green_state[id(gm)] = {
            "marker": gm,
            "center": gc,
            "dst_color": dst_color,
            "period": period,
            "occupied": set(tops),
            "bottom0": max(tops),
        }

    assert green_state, "expected at least one green lane with valid sprites"

    out = g.copy()

    for rm in red_markers:
        _, src_c0, _, src_c1 = bbox_coords(np.array(rm))
        src_cols = range(src_c0, src_c1 + 1)
        src_set = set(rm)
        src_comps = _single_color_components_in_cols(g, src_cols, bg)
        src_comps = [[p for p in pts if p not in src_set] for pts in src_comps]
        src_comps = [pts for pts in src_comps if pts]
        if not src_comps:
            continue

        src_sz = max(len(p) for p in src_comps if len(p) != len(rm))
        src_sprites = [p for p in src_comps if len(p) == src_sz]
        if not src_sprites:
            continue

        # Nearest green lane by x-center.
        rc = (src_c0 + src_c1) / 2.0
        best = min(green_state.values(), key=lambda st: abs(st["center"] - rc))

        # Recolor all source sprites to the matched destination lane color.
        for pts in src_sprites:
            for r, c in pts:
                out[r, c] = best["dst_color"]

        # Source template (top-most sprite), shifted horizontally to the destination lane center.
        src_sprites_sorted = []
        for pts in src_sprites:
            rmin = min(r for r, _ in pts)
            cmin = min(c for _, c in pts)
            src_sprites_sorted.append((rmin, cmin, pts))
        src_sprites_sorted.sort()
        src_ref_r0, _src_ref_c0, src_ref_pts = src_sprites_sorted[0]

        dc_shift = int(round(best["center"] - rc))
        src_offsets = [(r - src_ref_r0, c + dc_shift) for r, c in src_ref_pts]
        sprite_h = max(dr for dr, _c in src_offsets) + 1

        # Place as many sprites as removed, filling empty slots from the bottom upward.
        n_new = len(src_sprites)
        placed = 0
        k = 0
        while placed < n_new:
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
                out[new_top + dr, c] = best["dst_color"]
            best["occupied"].add(new_top)
            placed += 1

    # Remove all marker bars.
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
_LANE_STARTS = [3, 9, 15, 21]       # col-0 of each lane's 5-wide sprite slot
_MARKER_ROW = 29                    # row for red/green 1x5 marker bars
_SPRITE_H, _SPRITE_W = 3, 5         # bounding box of the plus/diamond sprite

# --- CFG: vertical stacking ---
_BOTTOM_TOP = 25                    # top-row of the bottommost sprite
_PERIOD = 4                         # vertical period (3 sprite rows + 1 gap)

# --- CFG: color palette ---
_BG_CHOICES = [0, 1, 4, 6, 7, 8, 9]  # valid background colors (excludes 2, 3, 5)
_GREY = 5                             # kept excluded so the variant never collapses to the original grey output

# --- CFG: counts & limits ---
_MIN_SPRITES = 1
_MAX_SPRITES = 3
_MAX_LANE_TOTAL = 7
_GEN_RETRIES = 300


def _sprite_patch(color: int, bg: int) -> np.ndarray:
    """Return a 3x5 plus/diamond sprite patch with bg-transparent corners."""
    patch = np.full((_SPRITE_H, _SPRITE_W), bg, dtype=int)
    patch[0, 1:4] = color
    patch[1, :] = color
    patch[2, 1:4] = color
    return patch


def _lane_tops(count: int) -> list[int]:
    """Return top-row positions for `count` sprites, ordered top-to-bottom."""
    return [_BOTTOM_TOP - _PERIOD * k for k in range(count - 1, -1, -1)]


def _paint_lane(grid: np.ndarray, lane_idx: int, color: int, count: int, bg: int) -> None:
    """Stamp `count` sprites of `color` into the given lane."""
    patch = _sprite_patch(color, bg)
    c0 = _LANE_STARTS[lane_idx]
    for r0 in _lane_tops(count):
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

    # Choose palette. Distinct lane colors make the recolor-to-destination rule visible.
    bg = int(rng.choice(_BG_CHOICES))
    sprite_palette = [c for c in range(10) if c not in (bg, _GREY)]
    lane_colors = rng.choice(sprite_palette, size=_NUM_LANES, replace=False).tolist()
    input_counts = rng.integers(_MIN_SPRITES, _MAX_SPRITES + 1, size=_NUM_LANES).tolist()

    # Choose role assignment (red source lanes, green destination lane).
    feasible = _feasible_roles(input_counts)
    feasible_red_counts = [rc for rc in (1, 2) if feasible[rc]]
    if not feasible_red_counts:
        return None

    red_count = int(rng.choice(feasible_red_counts))
    reds, green = feasible[red_count][int(rng.integers(len(feasible[red_count])))]
    red_set = set(reds)
    gain = sum(input_counts[i] for i in reds)
    dst_color = lane_colors[green]

    # Build OUTPUT grid: source lanes and destination lane both show the destination color.
    output_counts = input_counts[:]
    output_counts[green] += gain
    out = np.full((_H, _W), bg, dtype=int)
    for lane_idx in range(_NUM_LANES):
        color = dst_color if lane_idx in red_set else lane_colors[lane_idx]
        _paint_lane(out, lane_idx, color, output_counts[lane_idx], bg)

    # Derive INPUT from output by restoring the original source-lane colors.
    inp = out.copy()
    for lane_idx in reds:
        _paint_lane(inp, lane_idx, lane_colors[lane_idx], input_counts[lane_idx], bg)

    # Erase the gained sprites in the green lane and add markers.
    green_c0 = _LANE_STARTS[green]
    for r0 in _lane_tops(output_counts[green])[:gain]:
        inp[r0:r0 + _SPRITE_H, green_c0:green_c0 + _SPRITE_W] = bg
    inp[_MARKER_ROW, green_c0:green_c0 + _SPRITE_W] = 3

    for lane_idx in reds:
        c0 = _LANE_STARTS[lane_idx]
        inp[_MARKER_ROW, c0:c0 + _SPRITE_W] = 2

    # Oracle check.
    if not np.array_equal(main(inp), out):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(_GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {_GEN_RETRIES} attempts")
