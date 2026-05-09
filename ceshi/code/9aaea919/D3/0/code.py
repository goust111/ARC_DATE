# Variant of 9aaea919
# Dimension: D3 — 目标位置与堆叠规则 (dst)
# Change: stacking — 新精灵在目标车道底部已有精灵之上，按周期4(3精灵行+1间隙行)向上堆叠空槽位... → 顶部向下堆叠

import sys
from itertools import combinations
from pathlib import Path

import numpy as np

_SOLVER_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_SOLVER_DIR) not in sys.path:
    sys.path.append(str(_SOLVER_DIR))

from arc_utils import bbox_coords, components_mask, get_bg, paste


# Design analysis for the D3 variant
# a) Concrete behavior and trace
#    - The grid stays 30x30 with four fixed 5-column lanes and 1x5 marker bars on row 29.
#    - Every lane still uses the same 3x5 plus sprite and the same period-4 vertical lattice.
#    - Source lanes marked by red bars still turn grey in place. The green-marked lane still receives the
#      same number of recolored copies.
#    - Only the destination fill order changes: the input green lane keeps its original bottom-stacked sprites,
#      but the newly created sprites now occupy the highest empty lattice rows first.
#    - Example with destination tops {21, 25} and gain=2:
#      original behavior: add at 17, then 13 (bottom-up near the existing block);
#      variant behavior: add at 1, then 5 (top-down from the highest empty slots).
#      The lane keeps the same lattice and same total count, but the visible gap shifts from the top to the
#      middle, so the changed direction is apparent in input/output pairs.
# b) Complexity calibration
#    - The variant keeps exactly one changed degree of freedom: destination slot priority.
#    - Shapes, colors, lane geometry, marker semantics, and nearest-lane matching stay unchanged, preventing
#      the task from turning into a multi-rule puzzle.
#    - To keep the changed rule visible, generation forbids cases where the green lane would become completely
#      full, because a fully filled lane would look identical under top-down and bottom-up stacking.
# c) Hidden-assumption review of the original main()
#    - Detecting markers as thin components is still valid because the bars remain 1x5 and isolated.
#    - Extracting monochrome lane components is still valid because sprite shape/color are unchanged.
#    - Inferring the period from existing destination sprites is still valid because the lattice is unchanged.
#    - The old dependency that breaks is `bottom0 - k * period`: it assumes the next legal slot is always the
#      nearest empty row above the current occupied block. In the variant, legal slots must be reconstructed for
#      the full lane and scanned from the smallest row downward.
#    - The source-template horizontal shift remains safe because destination lane centers do not move.
# d) Placement legality
#    - Legal lattice rows are reconstructed from the observed destination residue class and period, then clipped
#      to rows where a 3-high sprite fits inside the 30-row canvas.
#    - The generator keeps initial green sprites bottom-stacked and adds new sprites only into unoccupied lattice
#      rows, so no overlap or out-of-bounds placement occurs.
# e) Visual observability
#    - The rule does not use invisible metadata such as color ids for ordering.
#    - Each sample leaves at least one empty destination slot after transfer, ensuring top-priority filling is
#      distinguishable from the original bottom-priority filling by eye.
# f) Visual cleanliness
#    - Lanes, markers, and sprites keep the original spacing, so the top-added sprites occupy otherwise empty
#      lane cells rather than mixing with other structures.
# g) Color inferability
#    - Color behavior is unchanged and remains strongly cued: red-marked lanes grey out, destination copies take
#      the destination lane color, and unmarked lanes stay fixed.
# 9) New assumptions introduced by the variant
#    - `main()` now assumes the destination sprites all lie on one period-4 residue class so the full slot list
#      can be reconstructed from existing tops; `generate_input()` guarantees this by keeping the original lane
#      lattice and only deleting/adding whole sprites on that lattice.
#    - `main()` scans slots from top to bottom and requires empty background under a new sprite; the generator
#      guarantees the top-down target slots are actually empty in the input.


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


def _lane_slot_rows(occupied_tops: list[int], period: int, sprite_h: int, h: int) -> list[int]:
    """Reconstruct all legal lattice top rows for a lane and order them from top to bottom."""
    top0 = min(occupied_tops)
    while top0 - period >= 0:
        top0 -= period
    rows = []
    r = top0
    while r + sprite_h <= h:
        rows.append(r)
        r += period
    return rows


def main(grid: np.ndarray) -> np.ndarray:
    """
    Move sprites out of red-tagged lanes and copy them into the green-tagged lane using top-first stacking.

    Input structure:
    - The background is the most common color; thin red (2) and green (3) bars on the last row tag source and destination lanes.
    - Each lane contains repeated monochrome plus sprites on a regular period-4 vertical lattice, with destination-lane sprites initially occupying only some of those slots.

    Rule:
    - Every sprite in a red-marked lane turns grey (5), while the same number of copies is created in the nearest green-marked lane using that lane's own color.
    - The destination copies do not grow upward from the bottom block; instead they fill the highest empty lattice slots first, proceeding downward through the lane.
    - After the transfer, all marker bars are erased back to the background.

    Key insight:
    - The destination lane already reveals the sprite shape, color, and lattice period; the only changed reasoning step is to rebuild the full slot ladder and prioritize empty rows from top to bottom rather than from bottom to top.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = get_bg(g)

    red_markers = _find_marker_bars(g, 2)
    green_markers = _find_marker_bars(g, 3)
    assert red_markers and green_markers, "expected both red and green marker bars to be present"

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
            r0, _c0, r1, _c1 = bbox_coords(np.array(dst_sprites_sorted[0][2]))
            period = (r1 - r0 + 1) + 1

        sprite_h = max(r for r, _ in dst_sprites_sorted[0][2]) - min(r for r, _ in dst_sprites_sorted[0][2]) + 1
        gc = (dst_c0 + dst_c1) / 2.0
        green_state[id(gm)] = {
            "center": gc,
            "dst_color": dst_color,
            "occupied": set(tops),
            "slots": _lane_slot_rows(tops, period, sprite_h, h),
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

        n_new = len(src_sprites)
        placed = 0
        for new_top in best["slots"]:
            if placed >= n_new:
                break
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

    return out


# --------------- generator ---------------

# --- CFG: canvas & layout ---
_H, _W = 30, 30
_NUM_LANES = 4
_LANE_STARTS = [3, 9, 15, 21]
_MARKER_ROW = 29
_SPRITE_H, _SPRITE_W = 3, 5

# --- CFG: vertical lattice ---
_BOTTOM_TOP = 25
_PERIOD = 4
_TOP_SLOT0 = _BOTTOM_TOP % _PERIOD
_ALL_TOPS = list(range(_TOP_SLOT0, _BOTTOM_TOP + 1, _PERIOD))

# --- CFG: color palette ---
_BG_CHOICES = [0, 1, 4, 6, 7, 8, 9]
_GREY = 5

# --- CFG: counts & limits ---
_MIN_SPRITES = 1
_MAX_SPRITES = 3
_MAX_LANE_TOTAL = len(_ALL_TOPS)
_GEN_RETRIES = 300


def _sprite_patch(color: int, bg: int) -> np.ndarray:
    """Return a 3x5 plus sprite patch with bg-transparent corners."""
    patch = np.full((_SPRITE_H, _SPRITE_W), bg, dtype=int)
    patch[0, 1:4] = color
    patch[1, :] = color
    patch[2, 1:4] = color
    return patch


def _lane_tops(count: int) -> list[int]:
    """Return the bottom-stacked top rows for `count` sprites, ordered top-to-bottom."""
    return [_BOTTOM_TOP - _PERIOD * k for k in range(count - 1, -1, -1)]


def _topdown_added_tops(existing_count: int, gain: int) -> list[int]:
    """Return the destination rows used by the variant: highest empty slots first."""
    occupied = set(_lane_tops(existing_count))
    empties = [r for r in _ALL_TOPS if r not in occupied]
    return empties[:gain]


def _paint_lane_at_tops(grid: np.ndarray, lane_idx: int, color: int, tops: list[int], bg: int) -> None:
    patch = _sprite_patch(color, bg)
    c0 = _LANE_STARTS[lane_idx]
    for r0 in tops:
        paste(grid, patch, r0, c0, transparent=bg)


def _paint_lane(grid: np.ndarray, lane_idx: int, color: int, count: int, bg: int) -> None:
    """Stamp `count` bottom-stacked sprites of `color` into the given lane."""
    _paint_lane_at_tops(grid, lane_idx, color, _lane_tops(count), bg)


def _feasible_roles(counts: list[int]) -> dict[int, list[tuple[tuple[int, ...], int]]]:
    """Return feasible (reds, green) assignments grouped by number of red lanes."""
    grouped: dict[int, list[tuple[tuple[int, ...], int]]] = {1: [], 2: []}
    for green in range(_NUM_LANES):
        others = [i for i in range(_NUM_LANES) if i != green]
        for red_count in (1, 2):
            for reds in combinations(others, red_count):
                total = counts[green] + sum(counts[i] for i in reds)
                if total < _MAX_LANE_TOTAL:
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
    added_tops = _topdown_added_tops(input_counts[green], gain)
    if len(added_tops) != gain:
        return None

    out = np.full((_H, _W), bg, dtype=int)
    for lane_idx in range(_NUM_LANES):
        color = _GREY if lane_idx in red_set else lane_colors[lane_idx]
        if lane_idx == green:
            tops = sorted(_lane_tops(input_counts[green]) + added_tops)
            _paint_lane_at_tops(out, lane_idx, color, tops, bg)
        else:
            _paint_lane(out, lane_idx, color, input_counts[lane_idx], bg)

    inp = out.copy()
    for lane_idx in reds:
        _paint_lane(inp, lane_idx, lane_colors[lane_idx], input_counts[lane_idx], bg)

    green_c0 = _LANE_STARTS[green]
    for r0 in added_tops:
        inp[r0 : r0 + _SPRITE_H, green_c0 : green_c0 + _SPRITE_W] = bg
    inp[_MARKER_ROW, green_c0 : green_c0 + _SPRITE_W] = 3

    for lane_idx in reds:
        c0 = _LANE_STARTS[lane_idx]
        inp[_MARKER_ROW, c0 : c0 + _SPRITE_W] = 2

    if not np.array_equal(main(inp), out):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(_GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {_GEN_RETRIES} attempts")
