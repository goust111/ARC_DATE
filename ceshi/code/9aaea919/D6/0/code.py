# Variant of 9aaea919
# Dimension: D6 — 源车道残留处理 (trajectory)
# Change: 源车道精灵原位变灰(5)，形状保留但颜色变为灰色... → 源车道精灵完全擦除

import numpy as np

from arc_utils import bbox_coords, components_mask, get_bg, paste
from itertools import combinations


# Design analysis for the D6 variant (no source_task was provided, so this is based
# on the original solver/spec only).
#
# spec:
#   concretization:
#     - The canvas is still a 30x30 field with 4 fixed 5-column lanes.
#     - Each lane still holds 1-3 identical 3x5 plus/diamond sprites on a period-4
#       vertical lattice, and marker bars stay as horizontal 1x5 bars on row 29.
#     - Red bars still mean "source lane"; green bars still mean "destination lane".
#     - The changed visual outcome is only in the source lane: the output no longer
#       shows grey residue. Every source sprite cell becomes background, so the lane
#       turns completely blank after transfer.
#   behavior_trace:
#     - Example: if a red lane has sprites with top rows [21, 25] and the green lane
#       already has sprites at [17, 21, 25], the original task kept two grey sprites
#       at [21, 25] and added two destination-colored sprites at [9, 13].
#     - In this variant, the same input instead clears the source cells back to the
#       background at [21, 25], while still adding the same two new sprites at [9, 13].
#     - Therefore the transfer count is inferred from the missing source objects rather
#       than from grey traces.
#   complexity_calibration:
#     - The change is visually strong: a marked source lane becomes empty in output.
#     - No extra object types, sorting rules, or color mappings are added, so the task
#       does not become underconstrained or combinatorially noisy.
#     - Keeping fixed lane positions and the same sprite lattice preserves a repeated
#       structural cue that supports human inference from a few examples.
#   implicit_assumptions_review:
#     - `_find_marker_bars(...)` only assumes markers are thin red/green components;
#       D6 does not affect this.
#     - `_single_color_components_in_cols(...)` only reads the input; source sprites
#       are still present in the input, so component extraction is unchanged.
#     - The original source-lane mutation `out[r, c] = 5` depended on grey residue
#       being the intended output. In this variant that assumption is false, so those
#       assignments must become `bg`.
#     - Destination placement checks `out[rr, c] == bg`. Clearing a source lane earlier
#       is still harmless because source and destination lanes occupy disjoint columns.
#     - The output-first generator previously painted red lanes grey in `out`; that
#       would silently regenerate the old task, so `out` must now leave red lanes empty.
#   placement_legality:
#     - Lane columns remain disjoint and fixed, so clearing a source lane cannot create
#       overlaps with other lanes.
#     - The existing feasibility guard `counts[green] + transferred <= 7` still ensures
#       the added sprites fit inside the 30-row canvas on the same lattice.
#     - Input derivation is safe: starting from the blank-lane output, painting source
#       sprites back into red lanes reconstructs the required input without collision.
#   visual_observability:
#     - Across examples, the stable cue is: marked red lanes lose their sprites entirely
#       while marked green lanes gain the same number of sprites stacked upward.
#     - The new rule is observable because the same input would have produced visible
#       grey remnants in the original task; here those cells become pure background.
#   visual_cleanliness:
#     - No extra overlays are introduced. Output is actually cleaner than the original
#       because source lanes become empty rather than mixed with a special residue color.
#   color_inference:
#     - Recoloring only happens in the destination lane and still uses the preexisting
#       destination lane color. Source disappearance uses background, which is already
#       visible everywhere else and does not require learning a new color mapping.


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


def _lane_sprites(g: np.ndarray, marker_pts: list[tuple[int, int]], bg: int) -> tuple[int, int, list[list[tuple[int, int]]]]:
    """Extract same-sized sprite components from the lane aligned with one marker bar."""
    _, c0, _, c1 = bbox_coords(np.array(marker_pts))
    lane_cols = range(c0, c1 + 1)
    marker_set = set(marker_pts)
    comps = _single_color_components_in_cols(g, lane_cols, bg)
    comps = [[p for p in pts if p not in marker_set] for pts in comps]
    comps = [pts for pts in comps if pts]
    if not comps:
        return c0, c1, []

    sprite_sizes = [len(pts) for pts in comps if len(pts) != len(marker_pts)]
    if not sprite_sizes:
        return c0, c1, []

    sprite_sz = max(sprite_sizes)
    sprites = [pts for pts in comps if len(pts) == sprite_sz]
    return c0, c1, sprites


def main(grid: np.ndarray) -> np.ndarray:
    """
    Move sprites out of red-marked lanes into green-marked lanes, then erase their origin.

    Input structure:
    - The background is the dominant color, and thin red (2) / green (3) bars mark lane roles.
    - Each marked lane sits on the same fixed lane lattice and contains repeated identical
      monochrome plus sprites aligned to a regular vertical period.

    Rule:
    - Every sprite in a red-marked source lane is removed completely, turning back into
      background, while the same number of sprites is inserted into the nearest green lane.
    - Inserted sprites reuse the source shape but are recolored to the destination lane's
      sprite color and stacked upward into empty lattice slots from the bottom.
    - All marker bars are deleted in the final output.

    Key insight: The transfer still conserves sprite count, but the evidence in the source
    lane is absence rather than grey residue, so the solver must infer quantity from the
    source sprites visible only in the input.
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
        dst_c0, dst_c1, dst_sprites = _lane_sprites(g, gm, bg)
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
        src_c0, src_c1, src_sprites = _lane_sprites(g, rm, bg)
        if not src_sprites:
            continue

        # In the D6 variant the source lane is wiped clean instead of turning grey.
        for pts in src_sprites:
            for r, c in pts:
                out[r, c] = bg

        # Nearest green lane by x-center.
        rc = (src_c0 + src_c1) / 2.0
        best = min(green_state.values(), key=lambda st: abs(st["center"] - rc))

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
_MARKER_ROW = 29                     # row for red/green 1x5 marker bars
_SPRITE_H, _SPRITE_W = 3, 5         # bounding box of the plus/diamond sprite

# --- CFG: vertical stacking ---
_BOTTOM_TOP = 25                     # top-row of the bottommost sprite
_PERIOD = 4                          # vertical period (3 sprite rows + 1 gap)

# --- CFG: color palette ---
_BG_CHOICES = [0, 1, 4, 6, 7, 8, 9] # valid background colors (excludes 2, 3, 5)
_RESERVED_TRACE_COLOR = 5            # intentionally kept out of sprite palettes

# --- CFG: counts & limits ---
_MIN_SPRITES = 1                     # min sprites per lane in input
_MAX_SPRITES = 3                     # max sprites per lane in input
_MAX_LANE_TOTAL = 7                  # max sprites in green lane after transfer
_GEN_RETRIES = 300                   # retry budget for generate_input()


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

    # Choose palette
    bg = int(rng.choice(_BG_CHOICES))
    sprite_palette = [c for c in range(10) if c not in (bg, _RESERVED_TRACE_COLOR)]
    lane_colors = rng.choice(sprite_palette, size=_NUM_LANES, replace=False).tolist()
    input_counts = rng.integers(_MIN_SPRITES, _MAX_SPRITES + 1, size=_NUM_LANES).tolist()

    # Choose role assignment (red source lanes, green destination lane)
    feasible = _feasible_roles(input_counts)
    feasible_red_counts = [rc for rc in (1, 2) if feasible[rc]]
    if not feasible_red_counts:
        return None

    red_count = int(rng.choice(feasible_red_counts))
    reds, green = feasible[red_count][int(rng.integers(len(feasible[red_count])))]
    red_set = set(reds)
    gain = sum(input_counts[i] for i in reds)

    # Build OUTPUT grid: red lanes are erased, green lane has extra sprites.
    output_counts = input_counts[:]
    output_counts[green] += gain
    out = np.full((_H, _W), bg, dtype=int)
    for lane_idx in range(_NUM_LANES):
        if lane_idx in red_set:
            continue
        _paint_lane(out, lane_idx, lane_colors[lane_idx], output_counts[lane_idx], bg)

    # Derive INPUT from output by restoring source sprites and removing transferred ones.
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

    # Oracle check
    if not np.array_equal(main(inp), out):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(_GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {_GEN_RETRIES} attempts")
