# Combo variant of 9aaea919 (index 17)
# combo_spec: D7/2+D9/0
# Name: 按车道分离输出+可变车道数

import numpy as np

from arc_utils import bbox_coords, components_mask, get_bg, paste
from itertools import combinations


# Design analysis for the combo variant
# -------------------------------------
# fused_spec:
#   input:
#     grid: still 30x30
#     lanes:
#       count: variable, between 3 and 6
#       width: every lane still occupies an exact 5-column window
#       layout: lane starts are recomputed from the current lane count, so 3-lane scenes are sparse
#         while 6-lane scenes can have touching neighboring windows
#     markers:
#       red bars: 1 or 2 exact horizontal 1x5 bars on row 29
#       green bars: exactly 1 exact horizontal 1x5 bar on row 29
#       alignment: every marker is flush with one lane window
#   output:
#     stage_1: perform the same transfer semantics as the base task on the full 30x30 board
#     stage_2: erase all markers
#     stage_3: crop each finished lane to its occupied rows only
#     stage_4: pack those lane crops left-to-right with 1 background column separators, bottom-aligned
#
# a) Behavior trace:
#   Original task: there are always 4 lanes at fixed x positions, so the solver can hard-code the lane layout
#   and return a same-size 30x30 output after greying red lanes and extending the green lane upward.
#   Fused task: the semantic transfer is unchanged, but first the solver must infer how many 5-column lanes
#   exist in this particular input, because the lane count is now 3..6 and the horizontal gaps vary. After the
#   transfer, the answer is no longer the original board; each lane is cropped and the lane strips are repacked.
#   Example sketch: with 6 lanes at starts [0, 5, 10, 15, 20, 25], two adjacent windows may touch. The source
#   sprites still become grey and the destination lane still gains the same number of new sprites, but the final
#   output becomes a packed strip of width 35 rather than the original 30.
#
# b) Complexity calibration:
#   Only two coupled freedoms are introduced, both visibly supported by examples:
#   lane count changes the horizontal density, and packed output changes the final canvas shape. The transfer
#   semantics, sprite shape, and marker meanings stay fixed, so the puzzle is harder for structural reasons
#   without becoming a branchy rule soup.
#
# c) Implicit assumption audit:
#   - Old marker detection accepted any thin component. That fails once 6-lane layouts allow touching windows,
#     because two adjacent red bars would merge into a 1x10 blob. The combo solver therefore accepts only exact
#     1x5 horizontal bars, and the generator forbids touching red-red assignments.
#   - Old lane-range detection via boolean connected components fails for touching lanes, because adjacent sprites
#     of different colors still form one non-background component. The combo solver instead collects 5-column
#     windows from monochrome components per color, which remains stable even when lane windows touch.
#   - Packing introduces a new postcondition: every lane must remain non-empty so it has a crop. That still holds
#     because every input lane has at least one sprite and source lanes are greyed rather than erased.
#   - Transfer logic still assumes the destination lane already contains at least one original sprite so its
#     color and period can be inferred. The generator preserves the original lower bound of one sprite per lane.
#
# d) Placement legality:
#   Generator layout uses non-overlapping 5-column windows within the 30-column board. Six lanes may touch but
#   never overlap. Sprites still occupy rows ending at 27, markers stay on row 29, and row 28 stays empty, so
#   bars never connect to sprites. The green lane total is capped at 7 sprites so upward filling stays in-bounds.
#
# e) Visual observability:
#   Humans can see both variant dimensions directly. Inputs vary between sparse 3-lane scenes and dense 6-lane
#   scenes, so lane count clearly is not fixed. Outputs are packed strips whose width is 17, 23, 29, or 35
#   depending on lane count, making the repacking rule equally visible.
#
# f) Visual cleanliness:
#   The packed output isolates lanes into separate strips with one background separator column, which keeps the
#   result readable even when the original input used touching windows. Bottom alignment preserves the visual cue
#   that the destination lane is filled upward from the same baseline as the others.
#
# g) Change inferability:
#   No extra color mapping is added. Colors still encode the same structural roles as the base task: red bars
#   mark sources, the green bar marks the destination, source sprites become grey, and transferred sprites adopt
#   the destination lane color. The only new inference burden is spatial: detect variable lane windows, then read
#   the packed-lane output format.


# --- Shared CFG: canvas, sprite geometry, packing ---
_H, _W = 30, 30
_MARKER_ROW = 29
_SPRITE_H, _SPRITE_W = 3, 5
_SPRITE_SIZE = 11
_BOTTOM_TOP = 25
_PERIOD = 4
_PACK_SEP = 1

# --- Shared CFG: lane-count variant ---
_MIN_LANES = 3
_MAX_LANES = 6

# --- Shared CFG: color palette ---
_BG_CHOICES = [0, 1, 4, 6, 7, 8, 9]
_GREY = 5

# --- Shared CFG: counts & limits ---
_MIN_SPRITES = 1
_MAX_SPRITES = 3
_MAX_LANE_TOTAL = 7
_GEN_RETRIES = 400


def _find_marker_bars(g: np.ndarray, marker_color: int) -> list[list[tuple[int, int]]]:
    """
    Marker bars are exact horizontal 1x5 components of the given color.
    """
    bars = []
    for pts_arr in components_mask(g == marker_color, connectivity=4):
        r0, c0, r1, c1 = bbox_coords(pts_arr)
        if (r1 - r0 + 1, c1 - c0 + 1, len(pts_arr)) == (1, _SPRITE_W, _SPRITE_W):
            pts = [(int(r), int(c)) for r, c in pts_arr]
            bars.append((int(r0), int(c0), pts))
    bars.sort()
    return [pts for _r0, _c0, pts in bars]


def _single_color_components_in_cols(g: np.ndarray, cols: range, bg: int) -> list[list[tuple[int, int]]]:
    h, w = g.shape
    colmask = np.zeros((h, w), dtype=bool)
    colmask[:, cols] = True
    mask = (g != bg) & colmask
    comps = [[(int(r), int(c)) for r, c in pts] for pts in components_mask(mask, connectivity=4)]
    out = []
    for pts in comps:
        colors_in = {int(g[r, c]) for r, c in pts}
        if len(colors_in) == 1:
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
    sprite_sizes = [len(pts) for pts in comps if len(pts) != len(marker_pts)]
    if not sprite_sizes:
        return []
    sprite_size = max(sprite_sizes)
    return [pts for pts in comps if len(pts) == sprite_size]


def _lane_sprites_from_marker(
    g: np.ndarray, marker_pts: list[tuple[int, int]], bg: int
) -> tuple[list[list[tuple[int, int]]], float] | tuple[None, float]:
    _, c0, _, c1 = bbox_coords(np.array(marker_pts))
    lane_cols = range(c0, c1 + 1)
    marker_set = set(marker_pts)
    sprites = _lane_sprites(g, lane_cols, bg, marker_set)
    return (sprites if sprites else None), (c0 + c1) / 2.0


def _sorted_sprites(sprites: list[list[tuple[int, int]]]) -> list[tuple[int, int, list[tuple[int, int]]]]:
    tagged = []
    for pts in sprites:
        rmin = min(r for r, _ in pts)
        cmin = min(c for _, c in pts)
        tagged.append((rmin, cmin, pts))
    tagged.sort()
    return tagged


def _detect_lane_ranges(g: np.ndarray, bg: int) -> list[tuple[int, int]]:
    """
    Recover all 5-column lane windows from monochrome sprite/marker components.

    This avoids boolean connected-component merging when adjacent lane windows touch.
    """
    lane_ranges = set()
    for color in np.unique(g):
        color = int(color)
        if color == bg:
            continue
        for pts_arr in components_mask(g == color, connectivity=4):
            r0, c0, r1, c1 = bbox_coords(pts_arr)
            hh = r1 - r0 + 1
            ww = c1 - c0 + 1
            size = len(pts_arr)
            if ww != _SPRITE_W:
                continue
            if (hh, size) in ((1, _SPRITE_W), (_SPRITE_H, _SPRITE_SIZE)):
                lane_ranges.add((int(c0), int(c1)))

    lanes = sorted(lane_ranges)
    assert _MIN_LANES <= len(lanes) <= _MAX_LANES, "expected 3-6 lane windows"
    return lanes


def _transfer_full_grid(g: np.ndarray) -> tuple[np.ndarray, int, list[tuple[int, int]]]:
    h, w = g.shape
    bg = get_bg(g)

    red_markers = _find_marker_bars(g, 2)
    green_markers = _find_marker_bars(g, 3)
    assert red_markers and green_markers, "expected both red and green marker bars to be present"

    lane_ranges = _detect_lane_ranges(g, bg)

    green_state = []
    for gm in green_markers:
        dst_sprites, gc = _lane_sprites_from_marker(g, gm, bg)
        if not dst_sprites:
            continue

        dst_sprites_sorted = _sorted_sprites(dst_sprites)
        dst_color = int(g[dst_sprites_sorted[0][2][0][0], dst_sprites_sorted[0][2][0][1]])
        tops = sorted({rmin for rmin, _cmin, _pts in dst_sprites_sorted})
        if len(tops) >= 2:
            diffs = [b - a for a, b in zip(tops, tops[1:]) if b - a > 0]
            period = int(np.gcd.reduce(diffs)) if diffs else _PERIOD
        else:
            r0, _c0, r1, _c1 = bbox_coords(np.array(dst_sprites_sorted[0][2]))
            period = (r1 - r0 + 1) + 1

        green_state.append(
            {
                "center": gc,
                "dst_color": dst_color,
                "period": period,
                "occupied": set(tops),
                "bottom0": max(tops),
            }
        )

    assert green_state, "expected at least one green lane with valid sprites"

    out = g.copy()

    for rm in red_markers:
        src_sprites, rc = _lane_sprites_from_marker(g, rm, bg)
        if not src_sprites:
            continue

        for pts in src_sprites:
            for r, c in pts:
                out[r, c] = _GREY

        best = min(green_state, key=lambda st: abs(st["center"] - rc))

        src_sprites_sorted = _sorted_sprites(src_sprites)
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

    for marker_group in (red_markers, green_markers):
        for pts in marker_group:
            for r, c in pts:
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
    out_w = sum(int(p.shape[1]) for p in crops) + (len(crops) - 1) * _PACK_SEP
    out = np.full((out_h, out_w), int(bg), dtype=int)

    cc = 0
    for patch in crops:
        ph, pw = map(int, patch.shape)
        rr = out_h - ph
        out[rr : rr + ph, cc : cc + pw] = patch
        cc += pw + _PACK_SEP
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Transfer sprites between marked lanes, then repack every finished lane into a compact strip.

    Input structure:
    - The board is still 30x30, but the lane count is no longer fixed: there are 3-6 vertical 5-column lanes.
    - Red exact 1x5 bars identify source lanes and one green exact 1x5 bar identifies the receiving lane.
    - Each lane contains repeated identical monochrome plus/diamond sprites on the same vertical lattice.

    Rule:
    - On the original 30x30 layout, every red-marked lane turns grey in place and transfers the same number of
      sprites into the green-marked lane, stacking new destination-colored copies upward from the bottom.
    - Remove every marker bar.
    - Then crop each lane to its occupied rows and concatenate the lane crops from left to right with one
      background-colored separator column between neighbors, aligned on a shared bottom edge.

    Key insight:
    - The semantic move still happens lane-to-lane, but the solver must first infer the current lane windows from
      the variable layout, and the final answer is a packed per-lane readout rather than the original full canvas.
    """
    g = np.asarray(grid, dtype=int)
    full_out, bg, lane_ranges = _transfer_full_grid(g)
    return _pack_lanes(full_out, bg, lane_ranges)


# --------------- generator ---------------


def _sprite_patch(color: int, bg: int) -> np.ndarray:
    """Return the 3x5 plus/diamond sprite patch with transparent corners."""
    patch = np.full((_SPRITE_H, _SPRITE_W), bg, dtype=int)
    patch[0, 1:4] = color
    patch[1, :] = color
    patch[2, 1:4] = color
    return patch


def _lane_layout(num_lanes: int) -> tuple[list[int], list[int], tuple[int, int]]:
    """
    Return lane starts, interior gaps, and (left_margin, right_margin) for `num_lanes`.

    Extra columns are distributed round-robin across interior gaps first, then the two outer margins.
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
    return starts, gaps, (left_margin, right_margin)


def _lane_tops(count: int) -> list[int]:
    """Return top-row positions for `count` sprites, ordered top-to-bottom."""
    return [_BOTTOM_TOP - _PERIOD * k for k in range(count - 1, -1, -1)]


def _paint_lane(grid: np.ndarray, lane_start: int, color: int, count: int, bg: int) -> None:
    """Stamp `count` sprites of `color` into the lane whose window begins at `lane_start`."""
    patch = _sprite_patch(color, bg)
    for r0 in _lane_tops(count):
        paste(grid, patch, r0, lane_start, transparent=bg)


def _touching_neighbor_pairs(starts: list[int]) -> set[tuple[int, int]]:
    """Return adjacent lane-index pairs whose 5-column windows touch without a bg gap."""
    pairs = set()
    for i in range(len(starts) - 1):
        if starts[i + 1] == starts[i] + _SPRITE_W:
            pairs.add((i, i + 1))
    return pairs


def _feasible_roles(
    counts: list[int], starts: list[int]
) -> dict[int, list[tuple[tuple[int, ...], int]]]:
    """Return feasible (reds, green) assignments grouped by number of red lanes."""
    touching = _touching_neighbor_pairs(starts)
    grouped: dict[int, list[tuple[tuple[int, ...], int]]] = {1: [], 2: []}
    num_lanes = len(counts)
    for green in range(num_lanes):
        others = [i for i in range(num_lanes) if i != green]
        for red_count in (1, 2):
            for reds in combinations(others, red_count):
                if counts[green] + sum(counts[i] for i in reds) > _MAX_LANE_TOTAL:
                    continue
                if red_count == 2 and tuple(sorted(reds)) in touching:
                    continue
                grouped[red_count].append((reds, green))
    return grouped


def _build_full_output(
    bg: int,
    lane_starts: list[int],
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
    for lane_idx, lane_start in enumerate(lane_starts):
        color = _GREY if lane_idx in red_set else lane_colors[lane_idx]
        _paint_lane(out, lane_start, color, output_counts[lane_idx], bg)
    return out


def _pack_generator_output(full_out: np.ndarray, bg: int, lane_starts: list[int]) -> np.ndarray:
    crops = []
    for lane_start in lane_starts:
        patch = full_out[:, lane_start : lane_start + _SPRITE_W]
        rows = np.where(np.any(patch != bg, axis=1))[0]
        assert len(rows) > 0
        r0 = int(rows.min())
        r1 = int(rows.max())
        crops.append(patch[r0 : r1 + 1, :].copy())

    out_h = max(int(p.shape[0]) for p in crops)
    out_w = len(crops) * _SPRITE_W + (len(crops) - 1) * _PACK_SEP
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

    num_lanes = int(rng.integers(_MIN_LANES, _MAX_LANES + 1))
    lane_starts, _gaps, _margins = _lane_layout(num_lanes)

    bg = int(rng.choice(_BG_CHOICES))
    sprite_palette = [c for c in range(10) if c not in (bg, _GREY)]
    lane_colors = rng.choice(sprite_palette, size=num_lanes, replace=False).tolist()
    input_counts = rng.integers(_MIN_SPRITES, _MAX_SPRITES + 1, size=num_lanes).tolist()

    feasible = _feasible_roles(input_counts, lane_starts)
    feasible_red_counts = [rc for rc in (1, 2) if feasible[rc]]
    if not feasible_red_counts:
        return None

    red_count = int(rng.choice(feasible_red_counts))
    reds, green = feasible[red_count][int(rng.integers(len(feasible[red_count])))]
    gain = sum(input_counts[i] for i in reds)

    full_out = _build_full_output(bg, lane_starts, lane_colors, input_counts, reds, green)
    packed_out = _pack_generator_output(full_out, bg, lane_starts)

    inp = full_out.copy()
    for lane_idx in reds:
        _paint_lane(inp, lane_starts[lane_idx], lane_colors[lane_idx], input_counts[lane_idx], bg)

    output_green_count = input_counts[green] + gain
    green_c0 = lane_starts[green]
    for r0 in _lane_tops(output_green_count)[:gain]:
        inp[r0 : r0 + _SPRITE_H, green_c0 : green_c0 + _SPRITE_W] = bg
    inp[_MARKER_ROW, green_c0 : green_c0 + _SPRITE_W] = 3

    for lane_idx in reds:
        c0 = lane_starts[lane_idx]
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
