# Variant of 9aaea919
# Dimension: D6 — 源车道残留处理 (trajectory)
# Change: 源车道精灵原位变灰(5)，形状保留但颜色变为灰色... → 源车道精灵缩小为单像素

import numpy as np

from arc_utils import bbox_coords, components_mask, get_bg, paste
from itertools import combinations


# Design analysis
# spec:
#   variant_core:
#     unchanged:
#       - 30x30 canvas, 4 fixed 5-column lanes, period-4 vertical sprite lattice.
#       - Red 1x5 bars still mark source lanes; one green 1x5 bar still marks the destination lane.
#       - Source sprite counts still transfer into the nearest green lane, and transferred sprites remain full 3x5 plus shapes.
#     changed:
#       - A source sprite no longer survives as a full 3x5 grey plus.
#       - Instead, its 11 pixels are erased and replaced by exactly one grey pixel at the sprite center.
#       - For the fixed 3x5 plus sprite, that center is the bbox midpoint: (top+1, lane_start+2).
#   behavior_trace:
#     example:
#       - Suppose a red lane has sprites with top rows 21 and 25.
#       - Original task output kept two 3x5 grey shapes occupying rows 21:23 and 25:27.
#       - This variant erases those 22 colored cells and leaves only grey pixels at rows 22 and 26 in the lane center column.
#       - The destination green lane still gains two full recolored plus sprites in the lowest free lattice slots.
#       - Markers are still removed afterward.
#   complexity_check:
#     - Using a single center pixel keeps the change visually obvious: grey residue count equals sprite count, not sprite area.
#     - The dots stay on the same lane centerline and on the same period-4 rows, so humans can relate them to the vanished pluses.
#     - The rule stays single-branch: every source sprite always collapses to its own center; there is no per-color or per-lane exception.
#   implicit_assumption_review:
#     - `_single_color_components_in_cols(...)` is still safe because solver input is unchanged: sources are still full monochrome pluses at input time.
#     - The original `out[r, c] = 5` loop depended on "residual shape is identical to the source shape"; that assumption is false now and must be replaced.
#     - Destination placement logic still works because transferred sprites remain full pluses and source residue dots never occupy destination columns.
#     - Computing the center from each component bbox assumes an odd-by-odd anchorable sprite; generator preserves the original 3x5 plus, so this holds.
#   placement_legality:
#     - Source dots land strictly inside each source sprite bbox, so they never touch the marker row or neighboring lanes.
#     - Green-lane capacity constraints stay identical to the original generator because only source residue geometry changed.
#   visual_observability:
#     - The variant is distinguishable from the base task on every non-empty red lane: old output shows 11 grey cells per source sprite, new output shows 1.
#     - No hidden ordering by color id is introduced; the only cue is visible geometry contraction.


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


def _component_center(pts: list[tuple[int, int]]) -> tuple[int, int]:
    r0, c0, r1, c1 = bbox_coords(np.array(pts))
    return (r0 + r1) // 2, (c0 + c1) // 2


def main(grid: np.ndarray) -> np.ndarray:
    """
    Move sprites out of red-marked lanes while source lanes collapse to single grey center pixels.

    Input structure:
    - The background is the most common color, with thin red (2) and green (3) bars identifying source and destination lanes.
    - Each lane contains repeated identical monochrome plus sprites arranged on a regular vertical lattice.

    Rule:
    - Every sprite in a red lane disappears except for one grey pixel at its own center, and the same number of full sprites is transferred to the nearest green lane.
    - Transferred sprites keep the source shape but are recolored to the destination lane color and stack upward into empty lattice slots.
    - All marker bars are erased in the final grid.

    Key insight: The green lane still receives one full sprite per removed source sprite, but the source lane now records only each sprite's center point, so the preserved information is count-and-position rather than full silhouette.
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

        # Collapse each source sprite to its center grey pixel.
        for pts in src_sprites:
            for r, c in pts:
                out[r, c] = bg
            cr, cc = _component_center(pts)
            out[cr, cc] = 5

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
_GREY = 5                            # color used for source-lane residue dots

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
    """Stamp `count` full sprites of `color` into the given lane."""
    patch = _sprite_patch(color, bg)
    c0 = _LANE_STARTS[lane_idx]
    for r0 in _lane_tops(count):
        paste(grid, patch, r0, c0, transparent=bg)


def _paint_residue_lane(grid: np.ndarray, lane_idx: int, count: int) -> None:
    """Paint one grey center pixel for each source sprite position in the lane."""
    center_c = _LANE_STARTS[lane_idx] + (_SPRITE_W // 2)
    center_dr = _SPRITE_H // 2
    for r0 in _lane_tops(count):
        grid[r0 + center_dr, center_c] = _GREY


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
    sprite_palette = [c for c in range(10) if c not in (bg, _GREY)]
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

    # Build OUTPUT grid: red lanes shrink to center dots, green lane has extra sprites.
    output_counts = input_counts[:]
    output_counts[green] += gain
    out = np.full((_H, _W), bg, dtype=int)
    for lane_idx in range(_NUM_LANES):
        if lane_idx in red_set:
            _paint_residue_lane(out, lane_idx, input_counts[lane_idx])
        else:
            _paint_lane(out, lane_idx, lane_colors[lane_idx], output_counts[lane_idx], bg)

    # Derive INPUT from output.
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
