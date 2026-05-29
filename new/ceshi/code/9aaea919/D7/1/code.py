# Variant of 9aaea919
# Dimension: D7 — 输出网格形态 (grid_shape)
# Change: same — 输出网格与输入网格尺寸完全相同(30x30)... → 仅输出目标车道

import sys
from itertools import combinations
from pathlib import Path

import numpy as np

_SOLVER_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_SOLVER_DIR) not in sys.path:
    sys.path.append(str(_SOLVER_DIR))

from arc_utils import bbox_coords, components_mask, get_bg, paste


# Design analysis for the D7 variant
# a) Concrete behavior:
#    The input remains the original 30x30 four-lane scene. Each lane still occupies a 5-column slot,
#    sprites still stack on a period-4 lattice, red bars still mark source lanes, and exactly one green
#    bar still marks the destination lane. The transformation inside the scene is unchanged: red-lane
#    sprites become grey and the same number of sprites are added into the green lane from the bottom up.
#    The only changed stage is the final view: instead of returning the whole 30x30 transformed scene,
#    we crop to the green lane's own 5 columns and keep the full 30-row height.
#    Example trace:
#    - Original logic: solve full scene -> erase markers -> return all 30 columns.
#    - Variant logic: solve full scene -> erase markers -> locate the green marker's column window from
#      the input -> return only those 5 columns.
#    - Difference: unmarked lanes and greyed source lanes are no longer present in the output, but they
#      still matter in the input because they determine how many sprites transfer into the kept lane.
# b) Complexity calibration:
#    The crop is visually strong because output width changes from 30 to 5. The green lane varies across
#    the four lane slots, so the kept view is not constant in absolute position within the input. We keep
#    one or two red lanes and one green lane, matching the original difficulty; no extra object types or
#    hidden bookkeeping are added.
# c) Old assumptions that must be audited:
#    - The original main() assumed returning the same-shaped canvas; that no longer holds, so the final
#      return must crop explicitly.
#    - The solver cannot infer the crop from the transformed output after marker erasure alone, so it must
#      remember the green marker's bbox from the input before cleaning markers.
#    - Any code that used all lane columns as part of the answer must now be treated as intermediate state.
# d) Placement legality:
#    Generator keeps the original lane starts, sprite size, period, and max total of 7 sprites in the
#    green lane after transfer, so placement stays non-overlapping and in-bounds. The cropped output is
#    always exactly one legal lane window of width 5.
# e) Visual observability:
#    Across examples, the green bar appears in different lanes, yet every output is a 30x5 slice showing
#    only the final target lane. A human can infer that the rest of the board is omitted rather than
#    destroyed, because the input still shows other lanes whose counts explain the number of added sprites.
# f) Visual cleanliness:
#    The output contains one lane only, so there is less clutter than the original task. Within that lane,
#    sprites remain aligned on the same lattice and markers are removed, preserving a clean appearance.
# g) Color inference:
#    No new color mapping is introduced. The kept lane retains its original destination color, and source
#    recoloring to grey only affects off-crop lanes, so the crop change is separable from color behavior.
# 9) New solver assumptions guaranteed by the generator:
#    - There is exactly one green marker bar, so the crop target is unique.
#    - That marker spans exactly one 5-column lane slot.
#    - The input still contains at least one valid sprite in the destination lane so its lattice can be
#      read before adding transferred sprites.


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


def _solve_full_scene(grid: np.ndarray) -> tuple[np.ndarray, tuple[int, int]]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = get_bg(g)

    red_markers = _find_marker_bars(g, 2)
    green_markers = _find_marker_bars(g, 3)
    assert red_markers, "expected at least one red marker bar"
    assert len(green_markers) == 1, "expected exactly one green marker bar for the cropped target lane"

    green_marker = green_markers[0]
    _, dst_c0, _, dst_c1 = bbox_coords(np.array(green_marker))
    dst_cols = range(dst_c0, dst_c1 + 1)
    dst_set = set(green_marker)
    dst_comps = _single_color_components_in_cols(g, dst_cols, bg)
    dst_comps = [[p for p in pts if p not in dst_set] for pts in dst_comps]
    dst_comps = [pts for pts in dst_comps if pts]
    assert dst_comps, "expected non-marker content inside the destination lane"

    dst_sz = max(len(p) for p in dst_comps if len(p) != len(green_marker))
    dst_sprites = [p for p in dst_comps if len(p) == dst_sz]
    assert dst_sprites, "expected valid destination sprites"

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

    dst_center = (dst_c0 + dst_c1) / 2.0
    dst_state = {
        "center": dst_center,
        "dst_color": dst_color,
        "period": period,
        "occupied": set(tops),
        "bottom0": max(tops),
    }

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

        for pts in src_sprites:
            for r, c in pts:
                out[r, c] = 5

        src_center = (src_c0 + src_c1) / 2.0
        src_sprites_sorted = []
        for pts in src_sprites:
            rmin = min(r for r, _ in pts)
            cmin = min(c for _, c in pts)
            src_sprites_sorted.append((rmin, cmin, pts))
        src_sprites_sorted.sort()
        src_ref_r0, _src_ref_c0, src_ref_pts = src_sprites_sorted[0]

        dc_shift = int(round(dst_state["center"] - src_center))
        src_offsets = [(r - src_ref_r0, c + dc_shift) for r, c in src_ref_pts]
        sprite_h = max(dr for dr, _c in src_offsets) + 1

        n_new = len(src_sprites)
        placed = 0
        k = 0
        while placed < n_new:
            new_top = dst_state["bottom0"] - k * dst_state["period"]
            k += 1
            if new_top < 0 or new_top + sprite_h > h:
                if new_top < 0:
                    break
                continue
            if new_top in dst_state["occupied"]:
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
                out[new_top + dr, c] = dst_state["dst_color"]
            dst_state["occupied"].add(new_top)
            placed += 1

    for rm in red_markers:
        for r, c in rm:
            out[r, c] = bg
    for r, c in green_marker:
        out[r, c] = bg

    return out, (dst_c0, dst_c1)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Move source-lane sprites into the green-marked lane, then return only that lane's window.

    Input structure:
    - The grid is still a 30x30 four-lane scene with repeated monochrome plus sprites on a vertical lattice.
    - Red bars mark source lanes, and exactly one green bar marks the destination lane whose 5-column slot will be kept.

    Rule:
    - Red-lane sprites become grey, and the same number of sprites are inserted into the green lane from bottom upward.
    - After marker removal, crop the solved scene to the destination lane's column span, so the output is 30x5 instead of 30x30.

    Key insight: The transfer rule is unchanged; the real novelty is that the answer is the destination lane's local view, not the whole board.
    """
    full_out, (c0, c1) = _solve_full_scene(grid)
    return full_out[:, c0 : c1 + 1]


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

    output_counts = input_counts[:]
    output_counts[green] += gain
    out_full = np.full((_H, _W), bg, dtype=int)
    for lane_idx in range(_NUM_LANES):
        color = _GREY if lane_idx in red_set else lane_colors[lane_idx]
        _paint_lane(out_full, lane_idx, color, output_counts[lane_idx], bg)

    inp = out_full.copy()
    for lane_idx in reds:
        _paint_lane(inp, lane_idx, lane_colors[lane_idx], input_counts[lane_idx], bg)

    green_c0 = _LANE_STARTS[green]
    for r0 in _lane_tops(output_counts[green])[:gain]:
        inp[r0 : r0 + _SPRITE_H, green_c0 : green_c0 + _SPRITE_W] = bg
    inp[_MARKER_ROW, green_c0 : green_c0 + _SPRITE_W] = 3

    for lane_idx in reds:
        c0 = _LANE_STARTS[lane_idx]
        inp[_MARKER_ROW, c0 : c0 + _SPRITE_W] = 2

    expected = out_full[:, green_c0 : green_c0 + _SPRITE_W]
    if main(inp).shape != expected.shape:
        return None
    if not np.array_equal(main(inp), expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(_GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {_GEN_RETRIES} attempts")
