# Combo variant of 9aaea919 (index 12)
# combo_spec: D4/3+D14/0
# Name: 源精灵变目标色+标记变灰保留

import numpy as np

from itertools import combinations

from arc_utils import bbox_coords, components_mask, get_bg, paste


# Design analysis for the combo variant
# -------------------------------------
# concretized_behavior:
#   - Keep the original 30x30 lane layout from the spec: four fixed 5-column lanes, one sprite
#     color per lane, sprites on a period-4 vertical lattice, and 1x5 marker bars on row 29.
#   - The fusion changes two output-side signals together:
#     1. every source-lane sprite is recolored to the destination lane's sprite color;
#     2. every red/green marker bar remains visible, but all such bars become grey (5).
#   - The transferred copies still stack into the green-marked lane from the bottom upward.
#
# behavior_trace:
#   - Example: lane A has 2 orange sprites with a red bar; lane B has 1 blue sprite with a green bar.
#   - Original task:
#     lane A -> 2 grey sprites, lane B -> gains 2 blue sprites, both markers disappear.
#   - This combo:
#     lane A -> 2 blue sprites, lane B -> gains 2 blue sprites, and both lane-A and lane-B bars stay
#     on row 29 as grey 1x5 bars.
#   - The count transfer is unchanged; the visible differences are the unified target color in source
#     lanes and the preserved grey markers.
#
# complexity_calibration:
#   - The task still has exactly one destination lane and 1-2 source lanes, so the relational burden
#     stays close to the original puzzle instead of introducing extra matching branches.
#   - Lane colors stay pairwise distinct in the input and exclude grey, ensuring both combo effects are
#     visible: source recolor is noticeable, and grey markers do not blend into any lane sprites.
#   - The destination color repeats across the destination lane and all source lanes in the output,
#     providing enough repeated evidence for humans to infer a "unify to target color" rule.
#
# implicit_assumptions_audit:
#   - The original assignment `out[r, c] = 5` for source sprites is no longer valid; source recolor must
#     depend on the matched destination lane.
#   - The original marker-removal step assumed bars become background. That changes here: bars remain in
#     place, so the solver must recolor marker cells to grey instead of erasing them.
#   - Component extraction still assumes the marker row is separated from the bottom sprite by row 28.
#     The generator preserves `_BOTTOM_TOP = 25`, so sprite rows stop at 27 and never touch row 29.
#   - Placement checks still read destination cells from `out`. Grey source markers cannot block transfer
#     placement because they live outside the destination sprite rows.
#
# placement_legality:
#   - Output-first generation is still the safest construction strategy.
#   - The generator only chooses role/count assignments where destination_count + transferred_count <= 7,
#     matching the available lattice slots and preventing upward overflow.
#   - Grey markers are painted only on row 29, after sprites are placed, so they never overlap sprite
#     bodies or alter the legality of the lane packing.
#
# visual_observability_and_cleanliness:
#   - The output stays tidy because lanes remain spatially separated and the only repeated special color
#     is grey on the bottom marker row.
#   - The combo is easy to see in 2-3 examples: marked lanes keep a visible trace at the bottom, and
#     source lanes do not neutralize to grey but instead adopt the destination lane's color.
#
# color_inference:
#   - No branch depends on numeric color order.
#   - The target color is always observable from the pre-existing sprites inside the green-marked lane,
#     and the grey marker behavior is tied to marker position/shape rather than color-id arithmetic.
#
# new_solver_assumptions_and_generator_guarantees:
#   - Every green-marked lane contains at least one sprite so its destination color is readable.
#   - Every source lane contains at least one sprite so transfer count is non-zero and source recolor is
#     visibly exercised.
#   - The generator keeps source colors different from the destination color in the input, so the combo
#     cannot collapse into an unchanged source lane.


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


def _lane_sprites_from_marker(
    g: np.ndarray, marker_pts: list[tuple[int, int]], bg: int
) -> tuple[list[list[tuple[int, int]]], float]:
    """Return monochrome sprite components in the marker's lane plus the lane center."""
    _, c0, _, c1 = bbox_coords(np.array(marker_pts))
    lane_cols = range(c0, c1 + 1)
    marker_set = set(marker_pts)
    comps = _single_color_components_in_cols(g, lane_cols, bg)
    comps = [[p for p in pts if p not in marker_set] for pts in comps]
    comps = [pts for pts in comps if pts]
    if not comps:
        return [], (c0 + c1) / 2.0
    sprite_size = max(len(p) for p in comps if len(p) != len(marker_pts))
    sprites = [p for p in comps if len(p) == sprite_size]
    return sprites, (c0 + c1) / 2.0


def main(grid: np.ndarray) -> np.ndarray:
    """
    Transfer marked-lane sprite counts into the green lane while preserving the markers as grey traces.

    Input structure:
    - The background is the most common color. Thin red (2) bars identify source lanes and a thin green
      (3) bar identifies the destination lane.
    - Each lane contains repeated monochrome plus sprites aligned to the same vertical lattice, with one
      visible sprite color per lane in the input.

    Rule:
    - Each red-marked lane keeps its sprites in place, but those sprites are repainted to the matched
      destination lane's sprite color rather than turning grey.
    - The same number of source-shaped sprites is added into the nearest green-marked lane, filling empty
      lattice slots upward from the bottom using that destination color.
    - The red and green marker bars are not removed; they remain in place and are all recolored to grey.

    Key insight:
    - The green lane supplies one shared target color. Marked source lanes contribute their sprite counts
      to that lane and also adopt its color themselves, while the bottom bars survive as grey evidence of
      which lanes participated in the transfer.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = get_bg(g)

    red_markers = _find_marker_bars(g, 2)
    green_markers = _find_marker_bars(g, 3)
    assert red_markers and green_markers, "expected both red and green marker bars to be present"

    green_state = {}
    for gm in green_markers:
        dst_sprites, gc = _lane_sprites_from_marker(g, gm, bg)
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

        green_state[id(gm)] = {
            "center": gc,
            "dst_color": dst_color,
            "period": period,
            "occupied": set(tops),
            "bottom0": max(tops),
        }

    assert green_state, "expected at least one green lane with valid sprites"

    out = g.copy()

    for rm in red_markers:
        src_sprites, rc = _lane_sprites_from_marker(g, rm, bg)
        if not src_sprites:
            continue

        best = min(green_state.values(), key=lambda st: abs(st["center"] - rc))

        for pts in src_sprites:
            for r, c in pts:
                out[r, c] = best["dst_color"]

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

    for rm in red_markers:
        for r, c in rm:
            out[r, c] = 5
    for gm in green_markers:
        for r, c in gm:
            out[r, c] = 5

    return out


# --------------- generator (O3) ---------------

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
    dst_color = lane_colors[green]

    output_counts = input_counts[:]
    output_counts[green] += gain
    out = np.full((_H, _W), bg, dtype=int)
    for lane_idx in range(_NUM_LANES):
        color = dst_color if lane_idx in red_set else lane_colors[lane_idx]
        _paint_lane(out, lane_idx, color, output_counts[lane_idx], bg)

    marked_lanes = red_set | {green}
    for lane_idx in marked_lanes:
        c0 = _LANE_STARTS[lane_idx]
        out[_MARKER_ROW, c0:c0 + _SPRITE_W] = _GREY

    inp = out.copy()
    for lane_idx in reds:
        _paint_lane(inp, lane_idx, lane_colors[lane_idx], input_counts[lane_idx], bg)

    green_c0 = _LANE_STARTS[green]
    for r0 in _lane_tops(output_counts[green])[:gain]:
        inp[r0:r0 + _SPRITE_H, green_c0:green_c0 + _SPRITE_W] = bg
    inp[_MARKER_ROW, green_c0:green_c0 + _SPRITE_W] = 3

    for lane_idx in reds:
        c0 = _LANE_STARTS[lane_idx]
        inp[_MARKER_ROW, c0:c0 + _SPRITE_W] = 2

    if not np.array_equal(main(inp), out):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(_GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {_GEN_RETRIES} attempts")
