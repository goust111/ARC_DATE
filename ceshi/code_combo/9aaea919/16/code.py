# Combo variant of 9aaea919 (index 16)
# combo_spec: D6/1+D7/0
# Name: 源精灵缩小为单像素+紧凑裁剪

import numpy as np

from arc_utils import bbox_coords, components_mask, get_bg, paste
from itertools import combinations


# Design analysis
# a) Concrete behavior and trace:
#    - The input remains the original 30x30 lane scene: four fixed 5-column lanes,
#      red 1x5 bars mark source lanes, and one green 1x5 bar marks the destination.
#    - Example: if a red lane has source sprites at top rows 21 and 25, the base task
#      would keep two full 3x5 grey pluses there. In this combo, each plus is erased
#      and replaced by one grey center pixel at rows 22 and 26, lane-center column.
#    - The destination lane still gains the same number of full recolored pluses in
#      the lowest free period-4 slots, exactly as in the original transfer rule.
#    - After all transfers and marker removal, the full 30x30 result is cropped to
#      the minimal rectangle that contains every remaining non-background pixel.
#    - So the fused difference is visible in two places at once: source lanes keep
#      sparse grey dots instead of grey shapes, and the final canvas hugs the data.
#
# b) Complexity calibration:
#    - D6 adds only one geometry change: each source sprite always contracts to one
#      center pixel. There is no color-conditional or lane-conditional branch.
#    - D7 adds only one presentation change: crop after the whole scene is rendered.
#    - These two dimensions stay easy to infer because every generated sample has at
#      least one red lane with at least one sprite, and cropping always removes the
#      outer empty border left by the fixed lane layout and deleted marker row.
#
# c) Implicit-assumption review:
#    - Marker parsing and source/destination component discovery still run on the
#      unchanged input, so `_single_color_components_in_cols(...)` can still assume
#      that source objects are full monochrome pluses rather than already-shrunk dots.
#    - The original source rewrite loop assumed "same footprint, new grey color";
#      that must be replaced with "clear full footprint, then set one center pixel".
#    - Destination placement must continue using the full 30x30 coordinates; cropping
#      cannot happen earlier or it would corrupt lane-relative slot calculations.
#    - Computing a component center from the bbox midpoint is valid because generator
#      keeps the original odd-sized 3x5 plus as the source object in every input.
#
# d) Placement legality:
#    - Residue dots always stay inside the old 3x5 source bbox, so they never touch
#      markers, neighboring lanes, or transferred sprites.
#    - Destination counts are still bounded by `_MAX_LANE_TOTAL = 7`, which keeps the
#      topmost transferred full sprite inside the 30x30 pre-crop canvas.
#    - Generator derives inputs from a legal full terminal state and verifies them
#      with the fused solver before accepting a sample.
#
# e) Visual observability:
#    - A human can see that each red-lane plus disappears except for one center dot;
#      the grey pixel count equals source sprite count rather than sprite area.
#    - The crop is equally visible because outputs are narrower and shorter than the
#      original 30x30 frame while still preserving all transformed content.
#    - No invisible use of color ids or hidden ordering is introduced.
#
# f) Visual cleanliness:
#    - Grey dots sit on the lane centerline with period-4 spacing, which matches the
#      original sprite lattice and keeps the residue readable instead of noisy.
#    - Cropping removes the empty frame, so the output emphasizes the transformed
#      object block without adding overlaps or mixed encodings in one region.
#
# g) Inference of changes:
#    - The destination recoloring rule is still cued structurally by the green-marked
#      lane receiving full pluses in its own color.
#    - The source-side change is purely geometric and repeated across multiple red
#      sprites within samples, so the "shrink to center pixel" rule is inferable from
#      2-4 examples without relying on arbitrary color remapping.


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


def _tight_crop_non_bg(g: np.ndarray, bg: int) -> np.ndarray:
    nz = np.argwhere(g != bg)
    if nz.size == 0:
        return g[:0, :0].copy()
    r0, c0, r1, c1 = bbox_coords(nz)
    return g[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1].copy()


def main(grid: np.ndarray) -> np.ndarray:
    """
    Move red-lane sprites into the nearest green lane, shrink the leftovers to dots, then tight-crop.

    Input structure:
    - The input is the original 30x30 four-lane layout: each lane contains repeated monochrome plus sprites on a period-4 vertical lattice.
    - Thin red bars (color 2) mark source lanes and a thin green bar (color 3) marks a destination lane; background is the most common color.

    Rule:
    - Every full sprite in a red-marked lane is removed from its footprint, replaced by one grey center pixel, and contributes one transferred full sprite to the nearest green lane.
    - Transferred sprites keep the original plus geometry, recolor to the destination lane color, and fill empty destination slots from the bottom upward.
    - After deleting all marker bars, crop the finished scene to the smallest rectangle containing every non-background pixel.

    Key insight: Count is still conserved across lanes, but the source lane now preserves only each sprite's center position while the final answer also discards the unused outer canvas.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = get_bg(g)

    red_markers = _find_marker_bars(g, 2)
    green_markers = _find_marker_bars(g, 3)
    assert red_markers and green_markers, "expected both red and green marker bars to be present"

    # New solver assumptions audited here:
    # - Every green marker lane still contains at least one full sprite in the input.
    # - Source lanes contain full 3x5 monochrome sprites, so the bbox midpoint is the
    #   intended center residue location after contraction.
    # - Cropping happens strictly after placement and marker deletion.
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

        # Contract each full source plus to one grey midpoint pixel.
        for pts in src_sprites:
            for r, c in pts:
                out[r, c] = bg
            cr, cc = _component_center(pts)
            out[cr, cc] = 5

        rc = (src_c0 + src_c1) / 2.0
        best = min(green_state.values(), key=lambda st: abs(st["center"] - rc))

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
            out[r, c] = bg
    for gm in green_markers:
        for r, c in gm:
            out[r, c] = bg

    return _tight_crop_non_bg(out, bg)


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
    """Return a 3x5 plus sprite with transparent corners."""
    patch = np.full((_SPRITE_H, _SPRITE_W), bg, dtype=int)
    patch[0, 1:4] = color
    patch[1, :] = color
    patch[2, 1:4] = color
    return patch


def _lane_tops(count: int) -> list[int]:
    """Return top rows for a lane's sprites from top to bottom."""
    return [_BOTTOM_TOP - _PERIOD * k for k in range(count - 1, -1, -1)]


def _paint_lane(grid: np.ndarray, lane_idx: int, color: int, count: int, bg: int) -> None:
    """Stamp `count` full sprites into one lane."""
    patch = _sprite_patch(color, bg)
    c0 = _LANE_STARTS[lane_idx]
    for r0 in _lane_tops(count):
        paste(grid, patch, r0, c0, transparent=bg)


def _paint_residue_lane(grid: np.ndarray, lane_idx: int, count: int) -> None:
    """Paint one grey center pixel for each contracted source sprite."""
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
        if lane_idx in red_set:
            _paint_residue_lane(out, lane_idx, input_counts[lane_idx])
        else:
            _paint_lane(out, lane_idx, lane_colors[lane_idx], output_counts[lane_idx], bg)
    return out


def _gen_attempt() -> np.ndarray | None:
    rng = np.random.default_rng()

    bg = int(rng.choice(_BG_CHOICES))
    # Reserve 2/3 exclusively for marker bars so lane-object colors never alias
    # the source/destination role cues a human is meant to infer.
    sprite_palette = [c for c in range(10) if c not in (bg, 2, 3, _GREY)]
    lane_colors = rng.choice(sprite_palette, size=_NUM_LANES, replace=False).tolist()
    input_counts = rng.integers(_MIN_SPRITES, _MAX_SPRITES + 1, size=_NUM_LANES).tolist()

    feasible = _feasible_roles(input_counts)
    if not (feasible[1] and feasible[2]):
        return None

    red_count = int(rng.choice([1, 2]))
    reds, green = feasible[red_count][int(rng.integers(len(feasible[red_count])))]
    gain = sum(input_counts[i] for i in reds)

    full_out = _full_output_grid(bg, lane_colors, input_counts, reds, green)
    cropped_out = _tight_crop_non_bg(full_out, bg)

    inp = full_out.copy()
    for lane_idx in reds:
        _paint_lane(inp, lane_idx, lane_colors[lane_idx], input_counts[lane_idx], bg)

    output_counts = input_counts[:]
    output_counts[green] += gain
    green_c0 = _LANE_STARTS[green]
    for r0 in _lane_tops(output_counts[green])[:gain]:
        inp[r0 : r0 + _SPRITE_H, green_c0 : green_c0 + _SPRITE_W] = bg
    inp[_MARKER_ROW, green_c0 : green_c0 + _SPRITE_W] = 3

    for lane_idx in reds:
        c0 = _LANE_STARTS[lane_idx]
        inp[_MARKER_ROW, c0 : c0 + _SPRITE_W] = 2

    if cropped_out.shape == (_H, _W):
        return None
    if not np.array_equal(main(inp), cropped_out):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(_GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {_GEN_RETRIES} attempts")
