# Variant of 9aaea919
# Dimension: D17 — 未标记车道行为 (transform)
# Change: 未标记车道(无红/绿标记)在输出中完全不变... → 未标记车道精灵也变灰

import numpy as np

from arc_utils import bbox_coords, components_mask, get_bg, paste
from itertools import combinations


# Design analysis for D17
# a) Concrete behavior and trace:
#    The canvas still has four fixed 5-column lanes filled with repeated 3x5 plus sprites.
#    Red bars on row 29 still mark source lanes, and one green bar still marks the destination lane.
#    The old task produced: red lanes grey, green lane expanded, unmarked lanes unchanged.
#    The new variant changes only the post-state of unmarked lanes:
#    every lane that is not the green destination must end grey.
#    Example with lane roles [red, unmarked, green, unmarked] and counts [2,1,1,3]:
#    old output -> [grey, original-color, green+2, original-color];
#    new output -> [grey, grey, green+2, grey].
#    The transfer count still comes only from the red lanes; unmarked lanes do not donate sprites.
# b) Complexity calibration:
#    The geometry, sprite shape, bar encoding, and bottom-up stacking stay unchanged so the task
#    remains recognizably the same puzzle. The visible novelty comes from at least one unmarked lane
#    changing to grey in every example, which makes the “only the target lane keeps color” rule
#    observable without adding extra object types or ambiguous correspondences.
# c) Hidden assumptions from the original solver:
#    The original code greyed sprites only inside the red-marker loop, which silently encoded the old
#    assumption that unmarked lanes survive untouched. That write pattern is no longer valid.
#    The fix is to detect every lane first and grey all non-green lanes before performing transfers.
#    The original output-first generator also assumed only red lanes differed between input and output;
#    for D17 it must repaint all non-green lanes when reconstructing the input from the grey output.
# d) Placement legality:
#    Only red-lane counts are transferred, so the original feasibility bound
#    counts[green] + sum(counts[red]) <= 7 still guarantees the target stack fits the period-4 lattice.
#    Unmarked lanes keep their original counts and positions; they merely change color in the output,
#    so no new collision mode is introduced.
# e) Visual observability:
#    With 4 total lanes, 1 green lane, and only 1-2 red lanes, each generated sample necessarily has
#    1-2 unmarked lanes. Since each lane has 1-3 sprites and input colors exclude grey, the new greying
#    behavior is visible in every sample pair.
# f) Visual tidiness:
#    Lanes remain separated by one blank column on both sides, markers stay on the bottom row, and all
#    marker bars are erased in the output. This keeps the extra D17 signal clean instead of cluttering
#    the transfer region.
# g) Color-change inference:
#    The output now has a simple global color rule: green target lane keeps its own color, every other
#    lane becomes grey. This uses only visually explicit structure (presence/absence of the green target
#    bar) and avoids any hidden ordering or color-id dependent behavior.


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


def _lane_sprites_in_cols(
    g: np.ndarray,
    cols: range,
    bg: int,
    ignored: set[tuple[int, int]] | None = None,
) -> list[list[tuple[int, int]]]:
    comps = _single_color_components_in_cols(g, cols, bg)
    if ignored:
        comps = [[p for p in pts if p not in ignored] for pts in comps]
        comps = [pts for pts in comps if pts]
    if not comps:
        return []
    sprite_sz = max(len(pts) for pts in comps)
    return [pts for pts in comps if len(pts) == sprite_sz]


def _sort_sprites(sprites: list[list[tuple[int, int]]]) -> list[tuple[int, int, list[tuple[int, int]]]]:
    out = []
    for pts in sprites:
        rmin = min(r for r, _ in pts)
        cmin = min(c for _, c in pts)
        out.append((rmin, cmin, pts))
    out.sort()
    return out


def _find_lane_ranges(
    g: np.ndarray,
    bg: int,
    marker_cells: set[tuple[int, int]],
) -> list[tuple[int, int]]:
    mask = g != bg
    for r, c in marker_cells:
        mask[r, c] = False
    ranges = set()
    for pts_arr in components_mask(mask, connectivity=4):
        r0, c0, r1, c1 = bbox_coords(pts_arr)
        hh = r1 - r0 + 1
        ww = c1 - c0 + 1
        if hh > 1 and ww > 1:
            ranges.add((int(c0), int(c1)))
    return sorted(ranges)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Move red-lane sprite counts into the green lane, while bleaching every lane that is not the target.

    Input structure:
    - The background is the most common color; red (2) and green (3) horizontal bars on the bottom row mark source and destination lanes.
    - Each lane is a 5-column slot containing repeated identical monochrome plus sprites aligned on a shared vertical lattice.

    Rule:
    - Red-marked lanes still donate their sprite counts to the nearest green lane, which receives recolored copies stacked into empty slots from the bottom upward.
    - Every non-green lane ends grey in the output, including unmarked lanes; only the green destination lane keeps its original color.
    - All marker bars are removed at the end.

    Key insight:
    - Count transfer is still driven only by the red markers, but color survival is now a global rule: the green target is the sole colored lane, and every other lane is bleached.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = get_bg(g)

    red_markers = _find_marker_bars(g, 2)
    green_markers = _find_marker_bars(g, 3)
    assert red_markers and green_markers, "expected both red and green marker bars to be present"

    marker_cells = {pt for bar in red_markers + green_markers for pt in bar}
    lane_ranges = _find_lane_ranges(g, bg, marker_cells)
    assert lane_ranges, "expected at least one sprite lane"

    # Build destination lane state per green marker.
    green_state = {}
    target_ranges = set()
    for gm in green_markers:
        _, dst_c0, _, dst_c1 = bbox_coords(np.array(gm))
        target_ranges.add((dst_c0, dst_c1))
        dst_cols = range(dst_c0, dst_c1 + 1)
        dst_sprites = _lane_sprites_in_cols(g, dst_cols, bg, ignored=set(gm))
        if not dst_sprites:
            continue

        dst_sprites_sorted = _sort_sprites(dst_sprites)
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

    # D17 change: every lane except the green target is grey in the final image.
    for c0, c1 in lane_ranges:
        if (c0, c1) in target_ranges:
            continue
        lane_sprites = _lane_sprites_in_cols(g, range(c0, c1 + 1), bg)
        for pts in lane_sprites:
            for r, c in pts:
                out[r, c] = 5

    for rm in red_markers:
        _, src_c0, _, src_c1 = bbox_coords(np.array(rm))
        src_cols = range(src_c0, src_c1 + 1)
        src_sprites = _lane_sprites_in_cols(g, src_cols, bg, ignored=set(rm))
        if not src_sprites:
            continue

        # Nearest green lane by x-center.
        rc = (src_c0 + src_c1) / 2.0
        best = min(green_state.values(), key=lambda st: abs(st["center"] - rc))

        # Source template (top-most sprite), shifted horizontally to the destination lane center.
        src_sprites_sorted = _sort_sprites(src_sprites)
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
_SPRITE_H, _SPRITE_W = 3, 5          # bounding box of the plus/diamond sprite

# --- CFG: vertical stacking ---
_BOTTOM_TOP = 25                     # top-row of the bottommost sprite
_PERIOD = 4                          # vertical period (3 sprite rows + 1 gap)

# --- CFG: color palette ---
_BG_CHOICES = [0, 1, 4, 6, 7, 8, 9]  # valid background colors (excludes 2, 3, 5)
_GREY = 5                            # color used for greyed-out non-target sprites

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
    gain = sum(input_counts[i] for i in reds)

    # Build OUTPUT grid: only the green destination stays colored; every other lane is grey.
    output_counts = input_counts[:]
    output_counts[green] += gain
    out = np.full((_H, _W), bg, dtype=int)
    for lane_idx in range(_NUM_LANES):
        color = lane_colors[lane_idx] if lane_idx == green else _GREY
        _paint_lane(out, lane_idx, color, output_counts[lane_idx], bg)

    # Derive INPUT from output by restoring each non-green lane's original color.
    inp = out.copy()
    for lane_idx in range(_NUM_LANES):
        if lane_idx == green:
            continue
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
