# Combo variant of 9aaea919 (index 19)
# combo_spec: D3/0+D5/1+D9/2
# Name: 水平车道+侧边标记+方向翻转堆叠

import sys
from itertools import combinations
from pathlib import Path

import numpy as np

_SOLVER_DIR = Path(__file__).resolve().parents[4] / "solvers"
if str(_SOLVER_DIR) not in sys.path:
    sys.path.append(str(_SOLVER_DIR))

from arc_utils import bbox_coords, components_mask, get_bg, paste


# Design analysis
# a) Concrete behavior and trace:
#    - The original task used four vertical lanes with horizontal marker bars underneath,
#      and destination copies climbed upward from the bottom occupied block.
#    - This combo rotates the whole lane system by 90 degrees into four horizontal 5-row
#      bands, moves the markers to 5x1 vertical bars on the left side, and flips the fill
#      priority again so new copies occupy the leftmost empty slots first.
#    - Concrete trace: suppose the green lane is the band at rows 15..19 and currently has
#      two sprites at left cols {23, 27}. A red lane with two sprites donates gain=2.
#      Under the plain horizontal variant, those copies would fill nearest empty slots
#      {15, 19} from right to left; under this combo they fill {3, 7} from left to right.
#      The output therefore shows a visible gap between the new left block and the old
#      right block, making the direction flip observable.
#
# b) Complexity calibration:
#    - Only three coupled changes are introduced: lane orientation, marker placement, and
#      destination fill order. Sprite shape, lane count, lane spacing, source greying, and
#      nearest-lane matching stay structurally identical.
#    - The generator keeps destination lanes initially right-stacked, so the left-to-right
#      insertion rule creates a distinctive non-contiguous pattern instead of looking like a
#      trivial rigid rotation of the original task.
#    - To avoid degeneracy, generation forbids fully filling the destination lane. If every
#      slot were occupied after transfer, left-first and right-first stacking would end at
#      the same final occupancy set and the combo signal would vanish.
#
# c) Implicit-assumption review:
#    - Original code assumed lane membership came from fixed column slices; that breaks and
#      must become fixed row slices.
#    - Original marker parsing expected thin bars but not a specific orientation. That still
#      works, but marker lookup now targets left-side vertical 5x1 components instead of
#      bottom horizontal 1x5 components.
#    - Original destination bookkeeping used top rows and scanned upward from a bottom anchor.
#      In the rotated layout the stack coordinate becomes sprite left-column, and the combo
#      replaces right-to-left scanning with reconstruction of the full horizontal lattice and
#      a left-to-right scan over empty slots.
#    - Nearest-lane matching can no longer compare x-centers; it must compare row centers.
#
# d) Placement legality:
#    - Each horizontal lane occupies rows [3..7], [9..13], [15..19], [21..25], so bands do
#      not overlap.
#    - Markers live in column 0 only. Sprite slots start at columns 3, 7, 11, 15, 19, 23,
#      27 for a 5x3 sprite with period 4, leaving a clean two-column gap from the marker.
#    - The generator derives inputs from legal output states, erases only whole destination
#      sprites, and validates every accepted sample with `main()`.
#
# e) Visual observability:
#    - A human sees horizontal rows instead of vertical columns, red/green side tags instead
#      of bottom bars, greyed source rows, and new destination sprites appearing on the far
#      left rather than adjacent to the existing right block.
#    - No hidden use of numeric color ordering is introduced; all routing is based on visible
#      spatial structure and the unique green-marked destination row.
#
# f) Visual cleanliness:
#    - Each lane has a dedicated 5-row band and marker column, so objects do not overlap.
#    - The left-side markers are visually separate from the sprite field, and the left-first
#      destination copies occupy clean lattice slots rather than arbitrary coordinates.
#
# g) Change inferability:
#    - Recoloring remains easy to infer: red-marked lanes become grey and received sprites
#      adopt the color already present in the green-marked lane.
#    - The generator repeats the same left-first fill logic across many count combinations,
#      giving the solver multiple examples where the destination lane is split into a left
#      added block and a right original block.
#    - New solver assumptions introduced here are explicit: every marked lane has at least
#      one sprite, each lane's sprites share one monochrome shape, and the destination total
#      stays below the full seven-slot capacity so left-first occupancy remains distinguishable.


def _find_marker_bars(g: np.ndarray, marker_color: int) -> list[list[tuple[int, int]]]:
    """Return thin 4-connected marker bars of the requested color."""
    cands = []
    for pts_arr in components_mask(g == marker_color, connectivity=4):
        r0, c0, r1, c1 = bbox_coords(pts_arr)
        hh = r1 - r0 + 1
        ww = c1 - c0 + 1
        if hh == 1 or ww == 1:
            pts = [(int(r), int(c)) for r, c in pts_arr]
            cands.append((0 if len(pts) == 5 else 1, c0, r0, hh * ww, len(pts), pts))
    cands.sort()
    return [t[5] for t in cands]


def _single_color_components_in_rows(g: np.ndarray, rows: range, bg: int) -> list[list[tuple[int, int]]]:
    h, w = g.shape
    rowmask = np.zeros((h, w), dtype=bool)
    rowmask[rows, :] = True
    m = (g != bg) & rowmask
    comps = [[(int(r), int(c)) for r, c in pts] for pts in components_mask(m, connectivity=4)]
    out = []
    for pts in comps:
        colors_in = {int(g[r, c]) for r, c in pts}
        if len(colors_in) == 1:
            out.append(pts)
    return out


def _lane_slot_cols(occupied_lefts: list[int], period: int, sprite_w: int, w: int) -> list[int]:
    """Reconstruct all legal sprite left-columns for one lane, ordered left to right."""
    left0 = min(occupied_lefts)
    while left0 - period >= 0:
        left0 -= period
    cols = []
    c = left0
    while c + sprite_w <= w:
        cols.append(c)
        c += period
    return cols


def main(grid: np.ndarray) -> np.ndarray:
    """
    Transfer sprites from red-marked horizontal lanes into green-marked horizontal lanes using left-first filling.

    Input structure:
    - The background is the most common color. Four lanes are horizontal 5-row bands, and red (2) / green (3) markers are thin vertical 5x1 bars placed on the left side of marked lanes.
    - Each lane contains repeated identical monochrome sprites on a regular horizontal period-4 lattice, with existing destination sprites starting as a right-stacked block.

    Rule:
    - Every sprite in a red-marked lane turns grey (5), and the same number of copies is transferred to the nearest green-marked lane by row center.
    - Transferred sprites keep the same shape but recolor to the destination lane's color, filling the destination lane's empty lattice slots from left to right rather than from the right edge inward.
    - After all transfers, every marker bar is erased back to the background.

    Key insight:
    - This is the original count-conservation task after a 90-degree rotation plus a direction inversion: detect lanes by row bands, read markers on the side, reconstruct the destination slot lattice, and prioritize the leftmost empty slots first.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = get_bg(g)

    red_markers = _find_marker_bars(g, 2)
    green_markers = _find_marker_bars(g, 3)
    assert red_markers and green_markers, "expected both red and green marker bars to be present"

    # New assumptions audited against the generator:
    # - Every marked lane contains at least one full sprite in its row band.
    # - All sprites within a lane share one exact monochrome shape and lattice residue.
    # - The destination lane is not completely filled after transfer, so left-first filling
    #   remains visually distinct from the original right-first horizontal behavior.
    green_state = {}
    for gm in green_markers:
        dst_r0, _, dst_r1, _ = bbox_coords(np.array(gm))
        dst_rows = range(dst_r0, dst_r1 + 1)
        dst_set = set(gm)
        dst_comps = _single_color_components_in_rows(g, dst_rows, bg)
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
            dst_sprites_sorted.append((cmin, rmin, pts))
        dst_sprites_sorted.sort()
        dst_color = int(g[dst_sprites_sorted[0][2][0][0], dst_sprites_sorted[0][2][0][1]])

        lefts = sorted({cmin for cmin, _rmin, _pts in dst_sprites_sorted})
        if len(lefts) >= 2:
            diffs = [b - a for a, b in zip(lefts, lefts[1:]) if b - a > 0]
            period = int(np.gcd.reduce(diffs)) if diffs else 4
        else:
            _r0, c0, _r1, c1 = bbox_coords(np.array(dst_sprites_sorted[0][2]))
            period = (c1 - c0 + 1) + 1

        sprite_w = max(c for _r, c in dst_sprites_sorted[0][2]) - min(c for _r, c in dst_sprites_sorted[0][2]) + 1
        gr = (dst_r0 + dst_r1) / 2.0
        green_state[id(gm)] = {
            "center": gr,
            "dst_color": dst_color,
            "occupied": set(lefts),
            "slots": _lane_slot_cols(lefts, period, sprite_w, w),
        }

    assert green_state, "expected at least one green lane with valid sprites"

    out = g.copy()

    for rm in red_markers:
        src_r0, _, src_r1, _ = bbox_coords(np.array(rm))
        src_rows = range(src_r0, src_r1 + 1)
        src_set = set(rm)
        src_comps = _single_color_components_in_rows(g, src_rows, bg)
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

        rc = (src_r0 + src_r1) / 2.0
        best = min(green_state.values(), key=lambda st: abs(st["center"] - rc))

        src_sprites_sorted = []
        for pts in src_sprites:
            rmin = min(r for r, _ in pts)
            cmin = min(c for _, c in pts)
            src_sprites_sorted.append((cmin, rmin, pts))
        src_sprites_sorted.sort()
        src_ref_c0, _src_ref_r0, src_ref_pts = src_sprites_sorted[0]

        dr_shift = int(round(best["center"] - rc))
        src_offsets = [(r + dr_shift, c - src_ref_c0) for r, c in src_ref_pts]

        n_new = len(src_sprites)
        placed = 0
        for new_left in best["slots"]:
            if placed >= n_new:
                break
            if new_left in best["occupied"]:
                continue
            ok = True
            for rr, dc in src_offsets:
                cc = new_left + dc
                if not (0 <= rr < h and 0 <= cc < w):
                    ok = False
                    break
                if int(out[rr, cc]) != bg:
                    ok = False
                    break
            if not ok:
                continue
            for rr, dc in src_offsets:
                out[rr, new_left + dc] = best["dst_color"]
            best["occupied"].add(new_left)
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
_LANE_ROW_STARTS = [3, 9, 15, 21]
_MARKER_COL = 0
_SPRITE_H, _SPRITE_W = 5, 3

# --- CFG: horizontal lattice ---
_LEFTMOST_LEFT = 3
_RIGHTMOST_LEFT = 27
_PERIOD = 4
_ALL_LEFTS = list(range(_LEFTMOST_LEFT, _RIGHTMOST_LEFT + 1, _PERIOD))

# --- CFG: color palette ---
_BG_CHOICES = [0, 1, 4, 6, 7, 8, 9]
_GREY = 5
_RED_MARKER = 2
_GREEN_MARKER = 3

# --- CFG: counts & limits ---
_MIN_SPRITES = 1
_MAX_SPRITES = 3
_MAX_LANE_TOTAL = len(_ALL_LEFTS)
_GEN_RETRIES = 300


def _sprite_patch(color: int, bg: int) -> np.ndarray:
    """Return the 5x3 rotated plus/diamond sprite patch."""
    patch = np.full((_SPRITE_H, _SPRITE_W), bg, dtype=int)
    patch[0, 1] = color
    patch[1:4, :] = color
    patch[4, 1] = color
    return patch


def _lane_lefts(count: int) -> list[int]:
    """Return right-stacked sprite left-columns for `count` sprites, ordered left to right."""
    return [_RIGHTMOST_LEFT - _PERIOD * k for k in range(count - 1, -1, -1)]


def _leftright_added_lefts(existing_count: int, gain: int) -> list[int]:
    """Return the leftmost empty slots used by the combo's destination fill rule."""
    occupied = set(_lane_lefts(existing_count))
    empties = [c for c in _ALL_LEFTS if c not in occupied]
    return empties[:gain]


def _paint_lane_at_lefts(grid: np.ndarray, lane_idx: int, color: int, lefts: list[int], bg: int) -> None:
    patch = _sprite_patch(color, bg)
    r0 = _LANE_ROW_STARTS[lane_idx]
    for c0 in lefts:
        paste(grid, patch, r0, c0, transparent=bg)


def _paint_lane(grid: np.ndarray, lane_idx: int, color: int, count: int, bg: int) -> None:
    """Stamp `count` right-stacked sprites into the given horizontal lane."""
    _paint_lane_at_lefts(grid, lane_idx, color, _lane_lefts(count), bg)


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
    # Reserve 2/3 for side markers so the red/green tag colors stay unambiguous.
    sprite_palette = [c for c in range(10) if c not in (bg, _GREY, _RED_MARKER, _GREEN_MARKER)]
    lane_colors = rng.choice(sprite_palette, size=_NUM_LANES, replace=False).tolist()
    target_red_count = int(rng.choice([1, 2]))

    input_counts = None
    feasible = None
    for _ in range(24):
        candidate_counts = rng.integers(_MIN_SPRITES, _MAX_SPRITES + 1, size=_NUM_LANES).tolist()
        candidate_feasible = _feasible_roles(candidate_counts)
        if candidate_feasible[target_red_count]:
            input_counts = candidate_counts
            feasible = candidate_feasible
            break
    if feasible is None or input_counts is None:
        return None

    reds, green = feasible[target_red_count][int(rng.integers(len(feasible[target_red_count])))]
    red_set = set(reds)
    gain = sum(input_counts[i] for i in reds)
    added_lefts = _leftright_added_lefts(input_counts[green], gain)
    if len(added_lefts) != gain:
        return None

    out = np.full((_H, _W), bg, dtype=int)
    for lane_idx in range(_NUM_LANES):
        color = _GREY if lane_idx in red_set else lane_colors[lane_idx]
        if lane_idx == green:
            lefts = sorted(_lane_lefts(input_counts[green]) + added_lefts)
            _paint_lane_at_lefts(out, lane_idx, color, lefts, bg)
        else:
            _paint_lane(out, lane_idx, color, input_counts[lane_idx], bg)

    inp = out.copy()
    for lane_idx in reds:
        _paint_lane(inp, lane_idx, lane_colors[lane_idx], input_counts[lane_idx], bg)

    green_r0 = _LANE_ROW_STARTS[green]
    for c0 in added_lefts:
        inp[green_r0 : green_r0 + _SPRITE_H, c0 : c0 + _SPRITE_W] = bg
    inp[green_r0 : green_r0 + _SPRITE_H, _MARKER_COL] = _GREEN_MARKER

    for lane_idx in reds:
        r0 = _LANE_ROW_STARTS[lane_idx]
        inp[r0 : r0 + _SPRITE_H, _MARKER_COL] = _RED_MARKER

    if not np.array_equal(main(inp), out):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(_GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {_GEN_RETRIES} attempts")
