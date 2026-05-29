# Variant of 9aaea919
# Dimension: D7 — 输出网格形态 (grid_shape)
# Change: same — 输出网格与输入网格尺寸完全相同(30x30)... → 紧凑裁剪

import numpy as np

from arc_utils import bbox_coords, components_mask, get_bg, paste
from itertools import combinations


# Design analysis for the D7 variant (tight crop), based on the original task
# and the crop-after-render baseline used by reference task 20270e3b.
#
# 1) Concrete behavior:
#    - Input stays 30x30 and keeps the original lane layout: four 5-column lanes,
#      sprites stacked every 4 rows, red bars marking source lanes, one green bar
#      marking the destination lane.
#    - The transfer itself is unchanged: source sprites turn grey, equal-count new
#      sprites are inserted into the destination lane from the bottom upward, and
#      all marker bars disappear.
#    - The only output-shape change happens after the full 30x30 transformation is
#      complete: crop to the minimal rectangle containing every non-background pixel.
#    - Example trace:
#      original full output rows/cols might occupy rows 9..27 and cols 3..25.
#      Old D7 returned the whole 30x30 canvas with empty border.
#      New D7 returns only rows 9..27, cols 3..25, i.e. a 19x23 grid.
#
# 2) Complexity calibration:
#    - This is intentionally a pure D7 change. No sprite geometry, color logic,
#      matching rule, or marker semantics are altered.
#    - The crop is still visually obvious because the original layout always leaves
#      blank margins: left cols 0..2, right cols 26..29, and bottom rows 28..29
#      become removable once markers vanish. Height also varies with the highest
#      occupied sprite row, so samples do not collapse to a single fixed crop size.
#
# 3) Implicit-assumption audit in main():
#    - All marker detection and lane parsing happen on the input grid, so they still
#      rely on the original 30x30 canvas and remain valid.
#    - Placement bounds continue to use the input height/width; this is correct
#      because cropping must occur strictly after every placement decision.
#    - The original final `return out` encoded the old D7=same assumption. That is
#      the only solver line that becomes semantically wrong under this variant.
#
# 4) Generator legality:
#    - Generator still constructs a legal full 30x30 terminal state first, because
#      source recoloring and destination stacking are easiest to validate there.
#    - Input is derived from that full terminal state exactly as before, then oracle
#      checking compares `main(input)` against the cropped full terminal state.
#    - Counts are still constrained so the destination lane never overflows the
#      30x30 canvas before cropping.
#
# 5) Observability / cleanliness:
#    - The new behavior is human-visible from shape alone: outputs lose the empty
#      frame and hug the transformed sprite block.
#    - No invisible ordering or color-id-based branching is introduced.


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


def _tight_crop_non_bg(g: np.ndarray, bg: int) -> np.ndarray:
    nz = np.argwhere(g != bg)
    if nz.size == 0:
        return g[:0, :0].copy()
    r0, c0, r1, c1 = bbox_coords(nz)
    return g[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1].copy()


def main(grid: np.ndarray) -> np.ndarray:
    """
    Transfer sprites from red-marked lanes to green-marked lanes, then tight-crop.

    Input structure:
    - The input canvas is the original 30x30 lane layout: repeated monochrome sprites
      sit inside four fixed vertical lanes, and thin red/green bars identify source
      and destination lanes.
    - Background is the most common color; marker bars are color 2 (source) and 3
      (destination).

    Rule:
    - Every sprite in a red-marked lane turns grey, and the same number of copies is
      inserted into the nearest green-marked lane using that lane's own sprite color.
    - Inserted sprites fill empty lattice positions from the bottom upward, and all
      marker bars are erased afterward.
    - After the full scene is updated, remove the empty border by cropping to the
      minimal rectangle that contains all non-background pixels.

    Key insight: The transport rule is unchanged from the original task; the new D7
    requirement only changes the final presentation by trimming away unused canvas.
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

        # Grey out all source sprites.
        for pts in src_sprites:
            for r, c in pts:
                out[r, c] = 5

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

    # Remove all marker bars before computing the crop.
    for rm in red_markers:
        for r, c in rm:
            out[r, c] = bg
    for gm in green_markers:
        for r, c in gm:
            out[r, c] = bg

    return _tight_crop_non_bg(out, bg)


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
_BG_CHOICES = [0, 1, 4, 6, 7, 8, 9] # valid background colors (excludes 2, 3, 5)
_GREY = 5                           # color used for greyed-out source sprites

# --- CFG: counts & limits ---
_MIN_SPRITES = 1                    # min sprites per lane in input
_MAX_SPRITES = 3                    # max sprites per lane in input
_MAX_LANE_TOTAL = 7                 # max sprites in green lane after transfer
_GEN_RETRIES = 300                  # retry budget for generate_input()


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


def _full_output_grid(
    bg: int,
    lane_colors: list[int],
    input_counts: list[int],
    reds: tuple[int, ...],
    green: int,
) -> np.ndarray:
    red_set = set(reds)
    gain = sum(input_counts[i] for i in reds)
    output_counts = input_counts[:]
    output_counts[green] += gain
    out = np.full((_H, _W), bg, dtype=int)
    for lane_idx in range(_NUM_LANES):
        color = _GREY if lane_idx in red_set else lane_colors[lane_idx]
        _paint_lane(out, lane_idx, color, output_counts[lane_idx], bg)
    return out


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
    gain = sum(input_counts[i] for i in reds)

    # Build the full 30x30 terminal scene first, then crop only for oracle output.
    full_out = _full_output_grid(bg, lane_colors, input_counts, reds, green)
    cropped_out = _tight_crop_non_bg(full_out, bg)

    # Derive INPUT from the full terminal scene.
    inp = full_out.copy()
    for lane_idx in reds:
        _paint_lane(inp, lane_idx, lane_colors[lane_idx], input_counts[lane_idx], bg)

    # Erase the gained sprites in the green lane and add markers.
    output_counts = input_counts[:]
    output_counts[green] += gain
    green_c0 = _LANE_STARTS[green]
    for r0 in _lane_tops(output_counts[green])[:gain]:
        inp[r0:r0 + _SPRITE_H, green_c0:green_c0 + _SPRITE_W] = bg
    inp[_MARKER_ROW, green_c0:green_c0 + _SPRITE_W] = 3

    for lane_idx in reds:
        c0 = _LANE_STARTS[lane_idx]
        inp[_MARKER_ROW, c0:c0 + _SPRITE_W] = 2

    # D7 observability guard: the crop must really shrink the canvas.
    if cropped_out.shape == (_H, _W):
        return None

    # Oracle check
    if not np.array_equal(main(inp), cropped_out):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(_GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {_GEN_RETRIES} attempts")
