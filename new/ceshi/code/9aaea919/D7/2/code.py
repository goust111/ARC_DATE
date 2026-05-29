# Variant of 9aaea919
# Dimension: D7 — 输出网格形态 (grid_shape)
# Change: same — 输出网格与输入网格尺寸完全相同(30x30)... → 按车道分离输出

import numpy as np

from arc_utils import bbox_coords, components_mask, get_bg, paste
from itertools import combinations


# Design analysis for the D7 variant
# ----------------------------------
# spec_variant:
#   input:
#     grid: 30x30, exactly the original lane scene
#     lanes:
#       count: 4
#       width: 5 columns each
#       order: preserved from left to right
#     markers:
#       red bars: 1-2 source lanes
#       green bars: exactly 1 target lane
#   core_behavior:
#     step_1: perform the original lane transfer on the full 30x30 canvas
#     step_2: remove all marker bars, exactly as before
#     step_3: crop each of the 4 finished lanes to its own tight non-bg bbox
#     step_4: concatenate the 4 lane crops horizontally with 1 bg column separators
#   packed_layout:
#     lane_order: original left-to-right lane order
#     separator: 1 column of bg between adjacent lane crops
#     outer_border: none
#     vertical_alignment: bottom-aligned, so all lanes still visually stack from the same baseline
#
# a) Behavior trace:
#   Original logic keeps the transformed result on the original 30x30 board. For example, if lane 1 is red,
#   lane 3 is green, and the green lane gains two new sprites, the old output is still 30x30 with all four
#   lanes sitting in their original x positions.
#   In this variant, that same transformed full board is only an intermediate state. Each lane is then cut
#   down to its occupied rows and the four strips are repacked side by side. The visual difference is large:
#   the lane transfer rule is unchanged, but the final answer no longer contains the empty 30x30 background.
#
# b) Complexity calibration:
#   Only one packing rule is used: crop-by-lane, preserve lane order, 1-column separators, bottom alignment.
#   This is visible across examples and avoids hidden branches. The lane geometry, sprite geometry, recoloring,
#   and transfer counts all stay unchanged, so the task is still recognizably the original task plus one clear
#   output-shape twist.
#
# c) Old assumptions that must change:
#   - `return out` in the old `main()` assumes the final answer is the same-size transformed canvas.
#     That no longer holds; we must pack lane crops after the full-canvas transfer.
#   - Any code that uses original coordinates is still valid only for the intermediate transfer stage.
#     Packing must happen afterward, using detected lane ranges and cropped row spans.
#   - Cropping each lane independently introduces a new alignment question; top-aligning would destroy the
#     shared bottom stacking cue, so the packed output must be bottom-aligned.
#
# d) Placement legality:
#   The generator still builds the same legal 30x30 input scene as the original task, so all sprite placement
#   and transfer legality constraints remain inherited from the original generator. Packing is safe because
#   each lane crop is a disjoint 5-column strip, and the packed canvas width is the sum of widths plus
#   separators.
#
# e) Visual observability:
#   The packed output is always 23 columns wide (4 lane strips of width 5 plus 3 separators) and at most
#   27 rows tall, so it can never be mistaken for the original 30x30 behavior. The left-to-right lane order
#   is repeated across all examples, and bottom alignment makes the transfer growth still easy to read.
#
# f) Visual cleanliness:
#   Each output lane is isolated in its own strip, so there is less clutter than in the original full board.
#   The 1-column separators prevent adjacent lanes from touching after packing.
#
# g) Color inference:
#   No new color mapping is introduced. Colors still mean exactly what they meant in the original task:
#   source-lane sprites turn grey, transferred sprites take the destination lane's color, and separators
#   are background-colored empty space.


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


def _lane_sprites(
    g: np.ndarray, lane_cols: range, bg: int, marker_pts: set[tuple[int, int]]
) -> list[list[tuple[int, int]]]:
    comps = _single_color_components_in_cols(g, lane_cols, bg)
    comps = [[p for p in pts if p not in marker_pts] for pts in comps]
    comps = [pts for pts in comps if pts]
    if not comps:
        return []
    sprite_sz = max(len(pts) for pts in comps)
    return [pts for pts in comps if len(pts) == sprite_sz]


def _detect_lane_ranges(
    g: np.ndarray, bg: int, marker_bars: list[list[tuple[int, int]]]
) -> list[tuple[int, int]]:
    marker_pts = {pt for bar in marker_bars for pt in bar}
    mask = g != bg
    for r, c in marker_pts:
        mask[r, c] = False

    lane_ranges = set()
    for pts_arr in components_mask(mask, connectivity=4):
        colors = {int(g[r, c]) for r, c in pts_arr}
        if len(colors) != 1:
            continue
        _r0, c0, _r1, c1 = bbox_coords(pts_arr)
        lane_ranges.add((int(c0), int(c1)))

    lanes = sorted(lane_ranges)
    assert len(lanes) == 4, "expected four sprite lanes"
    return lanes


def _transfer_full_grid(g: np.ndarray) -> tuple[np.ndarray, int, list[tuple[int, int]]]:
    h, w = g.shape
    bg = get_bg(g)

    red_markers = _find_marker_bars(g, 2)
    green_markers = _find_marker_bars(g, 3)
    assert red_markers and green_markers, "expected both red and green marker bars to be present"

    lane_ranges = _detect_lane_ranges(g, bg, red_markers + green_markers)

    # Build destination lane state per green marker.
    green_state = {}
    for gm in green_markers:
        _, dst_c0, _, dst_c1 = bbox_coords(np.array(gm))
        dst_cols = range(dst_c0, dst_c1 + 1)
        dst_set = set(gm)
        dst_sprites = _lane_sprites(g, dst_cols, bg, dst_set)
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
            r0, _c0, r1, _c1 = bbox_coords(np.array(dst_sprites_sorted[0][2]))
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
        src_sprites = _lane_sprites(g, src_cols, bg, src_set)
        if not src_sprites:
            continue

        for pts in src_sprites:
            for r, c in pts:
                out[r, c] = 5

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

    return out, bg, lane_ranges


def _pack_lanes(full_out: np.ndarray, bg: int, lane_ranges: list[tuple[int, int]]) -> np.ndarray:
    crops = []
    for c0, c1 in lane_ranges:
        lane_patch = full_out[:, c0 : c1 + 1]
        rows = np.where(np.any(lane_patch != bg, axis=1))[0]
        assert len(rows) > 0, "each lane should remain non-empty after transfer"
        r0 = int(rows.min())
        r1 = int(rows.max())
        crops.append(lane_patch[r0 : r1 + 1, :].copy())

    out_h = max(int(p.shape[0]) for p in crops)
    out_w = sum(int(p.shape[1]) for p in crops) + (len(crops) - 1)
    out = np.full((out_h, out_w), int(bg), dtype=int)

    cc = 0
    for patch in crops:
        ph, pw = map(int, patch.shape)
        rr = out_h - ph
        out[rr : rr + ph, cc : cc + pw] = patch
        cc += pw + 1
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Transfer sprites between marked lanes, then repack the finished lanes into a compact strip.

    Input structure:
    - bg is the most common color on a 30x30 grid with four vertical sprite lanes.
    - Red 1x5 bars mark source lanes, and one green 1x5 bar marks the receiving lane.
    - Each lane contains repeated identical monochrome plus sprites aligned to the same 5-column slot.

    Rule:
    - First apply the original lane transfer: source-lane sprites turn grey, and equally many sprites are
      added into the nearest green-marked lane from the bottom upward using the destination color.
    - Remove all marker bars.
    - Then crop each finished lane to its occupied rows and place the four lane crops side by side, keeping
      their left-to-right order and separating them with 1 background column.

    Key insight: The semantic transformation still happens on the original lane lattice, but the answer is not
    the full scene anymore; it is a packed readout of the four post-transfer lanes with a shared bottom edge.
    """
    g = np.asarray(grid, dtype=int)
    full_out, bg, lane_ranges = _transfer_full_grid(g)
    return _pack_lanes(full_out, bg, lane_ranges)


# --------------- generator (O3) ---------------

# --- CFG: canvas & layout ---
_H, _W = 30, 30
_NUM_LANES = 4
_LANE_STARTS = [3, 9, 15, 21]       # col-0 of each lane's 5-wide sprite slot
_MARKER_ROW = 29                    # row for red/green 1x5 marker bars
_SPRITE_H, _SPRITE_W = 3, 5         # bounding box of the plus/diamond sprite
_PACK_SEP = 1                       # separator width in the packed output

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


def _build_full_output(
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


def _pack_generator_output(full_out: np.ndarray, bg: int) -> np.ndarray:
    crops = []
    for c0 in _LANE_STARTS:
        patch = full_out[:, c0 : c0 + _SPRITE_W]
        rows = np.where(np.any(patch != bg, axis=1))[0]
        assert len(rows) > 0
        r0 = int(rows.min())
        r1 = int(rows.max())
        crops.append(patch[r0 : r1 + 1, :].copy())

    out_h = max(int(p.shape[0]) for p in crops)
    out_w = _NUM_LANES * _SPRITE_W + (_NUM_LANES - 1) * _PACK_SEP
    out = np.full((out_h, out_w), bg, dtype=int)

    cc = 0
    for patch in crops:
        ph, pw = map(int, patch.shape)
        rr = out_h - ph
        out[rr : rr + ph, cc : cc + pw] = patch
        cc += pw + _PACK_SEP
    return out


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
    gain = sum(input_counts[i] for i in reds)

    full_out = _build_full_output(bg, lane_colors, input_counts, reds, green)
    packed_out = _pack_generator_output(full_out, bg)

    inp = full_out.copy()
    for lane_idx in reds:
        _paint_lane(inp, lane_idx, lane_colors[lane_idx], input_counts[lane_idx], bg)

    green_c0 = _LANE_STARTS[green]
    output_green_count = input_counts[green] + gain
    for r0 in _lane_tops(output_green_count)[:gain]:
        inp[r0 : r0 + _SPRITE_H, green_c0 : green_c0 + _SPRITE_W] = bg
    inp[_MARKER_ROW, green_c0 : green_c0 + _SPRITE_W] = 3

    for lane_idx in reds:
        c0 = _LANE_STARTS[lane_idx]
        inp[_MARKER_ROW, c0 : c0 + _SPRITE_W] = 2

    if not np.array_equal(main(inp), packed_out):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(_GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {_GEN_RETRIES} attempts")
