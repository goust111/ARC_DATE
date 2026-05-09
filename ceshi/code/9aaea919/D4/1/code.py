# Variant of 9aaea919
# Dimension: D4 — 颜色变换规则 (transform)
# Change: color_transfer — 转移的精灵重新着色为目标车道的原有精灵颜色；源车道精灵变灰(5)... → 源精灵消失而非变灰

import numpy as np

from arc_utils import bbox_coords, components_mask, get_bg, paste
from itertools import combinations


# Design analysis for the D4 variant
# ----------------------------------
# variant_spec:
#   unchanged_layout:
#     grid: 30x30
#     lanes: 4 fixed 5-column lanes at starts [3, 9, 15, 21]
#     sprite: 3x5 plus/diamond, monochrome per lane, period 4, bottom top-row 25
#     markers:
#       red: 1-2 horizontal 1x5 bars on row 29, marking source lanes
#       green: exactly 1 horizontal 1x5 bar on row 29, marking destination lane
#   changed_transform:
#     before: red-lane sprites stayed visible as grey(5) while copies were added to the green lane
#     after: red-lane sprites are fully erased to background; only the destination lane gains sprites
#   concrete_trace:
#     example:
#       input: lane A has 2 red-marked blue sprites, lane C has 1 green-marked yellow sprite
#       old_output: lane A kept 2 grey sprites; lane C ended with 3 yellow sprites
#       new_output: lane A becomes empty background; lane C still ends with 3 yellow sprites
#     visual_difference:
#       the transferred count is now inferred from the source lane becoming blank, not from grey remnants
#   complexity_check:
#     - No new matching ambiguity is introduced because marker bars still identify sources and the destination.
#     - The nearest-green-lane rule, regular vertical lattice, and lane-local monochrome sprites are unchanged.
#     - The change is visible in every marked source lane because the full 11-pixel sprite disappears.
#   implicit_assumption_audit:
#     - main() previously relied on "writing 5 is always safe". In this variant that would be wrong because
#       grey is no longer part of the target semantics, so source pixels must be restored to bg instead.
#     - Destination placement checks use the partially-built output grid. This still works because source
#       erasure only affects source columns, while destination validation only probes destination columns.
#     - Component extraction assumes sprites remain separated from markers by row 28 being blank. Generator
#       must preserve that gap so bars do not merge with bottom sprites.
#     - Destination color is inferred from pre-existing destination sprites. Generator therefore must keep
#       at least one original sprite in the green lane, exactly as in the original task family.
#   placement_legality:
#     - Counts remain 1-3 per input lane; green lane total after transfer is capped at 7 so added sprites fit.
#     - Lanes are disjoint fixed columns, preventing overlap across lanes.
#     - New sprites are inserted only into empty lattice slots; source erasure never creates false free space
#       in the destination lane because columns do not overlap.
#   observability:
#     - Human-solvable examples still show the same lane markers and destination stacking.
#     - The new rule is observable because output removes all source-lane colored mass instead of recoloring it.
#     - Generator must avoid any grey in the intended output, otherwise samples would drift back toward the
#       original "grey residue" reading.
#   cleanliness:
#     - The scene stays sparse: one marker row, four aligned lanes, one repeated sprite shape.
#     - Erasure actually simplifies the output by reducing clutter rather than adding another color layer.


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
    Relocate sprites from red-marked lanes into green-marked lanes, deleting them at the source.

    Input structure:
    - The background is the most frequent color. Thin red (2) and green (3) bars on the last row mark lanes.
    - Every lane contains repeated copies of the same monochrome plus/diamond sprite on a regular vertical lattice.

    Rule:
    - Sprites in each red-marked lane are erased to background, and the same number of copies is transferred.
    - Each transferred copy uses the source shape but is recolored to the destination lane's own sprite color and stacked bottom-up into empty lattice slots of the nearest green-marked lane.
    - All marker bars are removed in the final output.

    Key insight: The count is preserved only through relocation — the source lane becomes blank, so the transfer must be inferred from disappearing source sprites and extra destination sprites rather than from grey leftovers.
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

        # Source sprites vanish completely in this variant.
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

    # Choose palette.
    bg = int(rng.choice(_BG_CHOICES))
    sprite_palette = [c for c in range(10) if c not in (bg, 5)]
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

    # Build OUTPUT grid: red lanes are empty, green lane has extra sprites.
    output_counts = input_counts[:]
    for lane_idx in reds:
        output_counts[lane_idx] = 0
    output_counts[green] += gain

    out = np.full((_H, _W), bg, dtype=int)
    for lane_idx in range(_NUM_LANES):
        count = output_counts[lane_idx]
        if count > 0:
            _paint_lane(out, lane_idx, lane_colors[lane_idx], count, bg)

    # Derive INPUT from output by restoring source lanes and hiding transferred destination sprites.
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

    # Oracle check: the generated sample must express the erase variant, not the grey one.
    if not np.array_equal(main(inp), out):
        return None
    if int((out == 5).sum()) != 0:
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(_GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {_GEN_RETRIES} attempts")
