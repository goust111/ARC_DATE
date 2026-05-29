# Variant of 9aaea919
# Dimension: D4 — 颜色变换规则 (transform)
# Change: color_transfer — 转移的精灵重新着色为目标车道的原有精灵颜色；源车道精灵变灰(5)... → 保持源颜色不变

import numpy as np

from arc_utils import bbox_coords, components_mask, get_bg, paste
from itertools import combinations


# Design analysis for this D4 variant
# a) Concrete behavior and flow tracking
#    The canvas, lane layout, sprite geometry, marker bars, and bottom-up filling rule all stay unchanged.
#    Each lane still holds repeated 3x5 plus/diamond sprites on a period-4 vertical lattice, with red bars
#    marking source lanes and one green bar marking the destination lane. The changed behavior is only in the
#    transferred sprite color: after moving, each copied sprite keeps the original color of the red-marked lane
#    it came from. The source lane still turns grey. Example:
#    - input: left red lane has two maroon sprites, green lane has one blue sprite.
#    - original task output: green lane becomes three blue sprites.
#    - variant output: green lane keeps its bottom blue sprite, and the two newly filled slots above are maroon.
#    With two red lanes, the inserted colors appear in deterministic groups because the original solver already
#    processes red markers one lane at a time while filling the destination from the bottom upward.
# b) Complexity calibration
#    This is not simplified to a trivial recolor toggle. The green lane now has 2 colors when one red lane feeds
#    it, or 3 colors when two red lanes feed it. That preserves enough structure for humans to see that color is
#    inherited from source lanes rather than converted to the destination color. The number of possible colors is
#    still tightly controlled: at most three visible colors coexist in the destination lane.
# c) Hidden assumptions in the original main()
#    The D4-sensitive assumption in the original code was the final paint step:
#        out[new_top + dr, c] = best["dst_color"]
#    That line relied on "all transferred sprites adopt the green-lane color". In this variant that premise is
#    false, so the paint color must come from the current source lane instead. The rest of the solver logic still
#    holds: marker detection, lane extraction, period inference, nearest-green matching, greying the source lane,
#    and bottom-up slot search are all independent of the recolor rule.
# d) Placement legality
#    No new geometry is introduced. The generator still restricts each input lane to 1-3 sprites and keeps the
#    final destination total at most 7 sprites, so every stacked top row remains on the fixed lattice and stays
#    within rows 1..25. Insertions are simulated with the same occupied-slot bookkeeping as the solver, so there
#    is no overlap, no clipping, and no disagreement between generated outputs and solver behavior.
# e) Visual observability
#    The variant must be visibly different from the original recolor task. To guarantee that, lane colors remain
#    distinct and the green lane always has at least one pre-existing sprite. Therefore the output green lane
#    always contains both the destination's old color and at least one source color. The rule is readable from
#    examples because the red lanes turn grey, and the same pre-grey source colors reappear in the green lane.
# f) Visual cleanliness
#    The lane grid and sprite spacing are unchanged, so the extra complexity is only a clean color mixture inside
#    one vertical stack. There is no partial overlap, no mixed-color single component, and no additional marker or
#    annotation clutter. Each sprite remains monochrome and occupies a separate lattice slot.
# g) Color-rule inferability
#    The color behavior is supported by strong spatial cues: the only lanes that lose color are the red-marked
#    ones, and the lost colors are exactly the ones that show up in the destination lane. Because all lane colors
#    are distinct, there is no ambiguity between "kept source color" and "coincidentally matched destination
#    color". The generator also allows both one-source and two-source cases so the rule does not collapse to a
#    single narrow example family.
# Assumption audit for the new solver
#    - Red markers are effectively processed left-to-right; the generator sorts source lanes the same way so the
#      vertical order of preserved colors in the destination stack is deterministic.
#    - Each input lane is monochrome and uses the common plus sprite; the generator enforces this, so copying a
#      source template and recoloring by source lane is always well-defined.
#    - The green lane already contains at least one sprite; counts stay in 1..3, so the mixed-color outcome is
#      always visible and the destination lattice period can always be inferred from existing sprites.


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


def _lane_sprites_for_marker(
    g: np.ndarray, marker_pts: list[tuple[int, int]], bg: int
) -> dict[str, object]:
    """Extract sorted monochrome sprite components that belong to the lane under one marker bar."""
    _r0, c0, _r1, c1 = bbox_coords(np.array(marker_pts))
    lane_cols = range(c0, c1 + 1)
    marker_set = set(marker_pts)
    comps = _single_color_components_in_cols(g, lane_cols, bg)
    comps = [[p for p in pts if p not in marker_set] for pts in comps]
    comps = [pts for pts in comps if pts]

    sprite_entries = []
    if comps:
        sprite_sizes = [len(p) for p in comps if len(p) != len(marker_pts)]
        if sprite_sizes:
            sprite_sz = max(sprite_sizes)
            for pts in comps:
                if len(pts) != sprite_sz:
                    continue
                rmin = min(r for r, _ in pts)
                cmin = min(c for _, c in pts)
                color = int(g[pts[0][0], pts[0][1]])
                sprite_entries.append((rmin, cmin, color, pts))
            sprite_entries.sort()

    return {
        "c0": c0,
        "c1": c1,
        "center": (c0 + c1) / 2.0,
        "sprites": sprite_entries,
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Transfer sprites from red-marked lanes to green-marked lanes while preserving source colors.

    Input structure:
    - The background is the most common color. Thin red (2) and green (3) bars sit below 5-column lanes.
    - Every lane contains copies of the same monochrome plus sprite on a regular vertical lattice.

    Rule:
    - Sprites in each red-marked lane turn grey (5), and the same number of copies are added to the nearest green lane.
    - Added sprites still use the source lane's color instead of being recolored to the destination lane's original color.
    - New sprites occupy empty lane slots from the bottom upward, and all marker bars disappear.

    Key insight:
    - The transfer geometry is unchanged from the original task; only the color inheritance changes, so the destination lane can end up with mixed source/destination colors.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = get_bg(g)

    red_markers = _find_marker_bars(g, 2)
    green_markers = _find_marker_bars(g, 3)
    assert red_markers and green_markers, "expected both red and green marker bars to be present"

    green_state = {}
    for gm in green_markers:
        lane = _lane_sprites_for_marker(g, gm, bg)
        dst_sprites = lane["sprites"]
        if not dst_sprites:
            continue

        tops = sorted({rmin for rmin, _cmin, _color, _pts in dst_sprites})
        if len(tops) >= 2:
            diffs = [b - a for a, b in zip(tops, tops[1:]) if b - a > 0]
            period = int(np.gcd.reduce(diffs)) if diffs else 4
        else:
            r0, c0, r1, c1 = bbox_coords(np.array(dst_sprites[0][3]))
            period = (r1 - r0 + 1) + 1

        green_state[id(gm)] = {
            "marker": gm,
            "center": lane["center"],
            "period": period,
            "occupied": set(tops),
            "bottom0": max(tops),
        }

    assert green_state, "expected at least one green lane with valid sprites"

    out = g.copy()

    for rm in red_markers:
        lane = _lane_sprites_for_marker(g, rm, bg)
        src_sprites = lane["sprites"]
        if not src_sprites:
            continue

        for _rmin, _cmin, _src_color, pts in src_sprites:
            for r, c in pts:
                out[r, c] = 5

        rc = lane["center"]
        best = min(green_state.values(), key=lambda st: abs(st["center"] - rc))

        src_ref_r0, _src_ref_c0, src_color, src_ref_pts = src_sprites[0]
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
                out[new_top + dr, c] = src_color
            best["occupied"].add(new_top)
            placed += 1

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
    reds = tuple(sorted(reds))
    red_set = set(reds)

    out = np.full((_H, _W), bg, dtype=int)
    for lane_idx in range(_NUM_LANES):
        if lane_idx == green:
            _paint_lane(out, lane_idx, lane_colors[lane_idx], input_counts[lane_idx], bg)
        elif lane_idx in red_set:
            _paint_lane(out, lane_idx, _GREY, input_counts[lane_idx], bg)
        else:
            _paint_lane(out, lane_idx, lane_colors[lane_idx], input_counts[lane_idx], bg)

    green_c0 = _LANE_STARTS[green]
    occupied = set(_lane_tops(input_counts[green]))
    transferred = []
    for lane_idx in reds:
        patch = _sprite_patch(lane_colors[lane_idx], bg)
        placed = 0
        k = 0
        while placed < input_counts[lane_idx]:
            new_top = _BOTTOM_TOP - _PERIOD * k
            k += 1
            if new_top < 0 or new_top + _SPRITE_H > _H:
                if new_top < 0:
                    return None
                continue
            if new_top in occupied:
                continue
            paste(out, patch, new_top, green_c0, transparent=bg)
            occupied.add(new_top)
            transferred.append((lane_idx, new_top))
            placed += 1

    inp = out.copy()
    for lane_idx in reds:
        _paint_lane(inp, lane_idx, lane_colors[lane_idx], input_counts[lane_idx], bg)

    for _lane_idx, r0 in transferred:
        inp[r0 : r0 + _SPRITE_H, green_c0 : green_c0 + _SPRITE_W] = bg
    inp[_MARKER_ROW, green_c0 : green_c0 + _SPRITE_W] = 3

    for lane_idx in reds:
        c0 = _LANE_STARTS[lane_idx]
        inp[_MARKER_ROW, c0 : c0 + _SPRITE_W] = 2

    green_lane_colors = {
        int(out[r, c])
        for r in range(_H)
        for c in range(green_c0, green_c0 + _SPRITE_W)
        if int(out[r, c]) != bg
    }
    if green_lane_colors != {lane_colors[green], *(lane_colors[l] for l in reds)}:
        return None

    if not np.array_equal(main(inp), out):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(_GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {_GEN_RETRIES} attempts")
