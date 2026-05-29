# Combo variant of 9aaea919 (index 27)
# combo_spec: D4/0+D6/2+D17/0
# Name: 保色克隆+旁观变灰

import numpy as np

from arc_utils import bbox_coords, components_mask, get_bg, paste
from itertools import combinations


# Design analysis for combo D4/0 + D6/2 + D17/0
# a) Concrete behavior and flow tracking
#    The seed task has four fixed 5-column lanes filled with repeated 3x5 plus/diamond sprites, one green
#    marker bar for the destination, and one or two red marker bars for source lanes. Originally the red lanes
#    lost color by turning grey, the green lane gained equally many recolored copies, and unmarked lanes stayed
#    unchanged. This combo changes three linked points at once:
#    - D4/0: transferred sprites keep the source-lane color instead of adopting the target-lane color.
#    - D6/2: the source lanes are not bleached; they stay exactly as they appeared in the input.
#    - D17/0 is narrowed by the fusion description: only the unmarked spectator lanes turn grey, rather than
#      every non-target lane. This resolves the direct conflict between "keep source colored" and the raw D17
#      single-variant wording that would otherwise grey the red lanes too.
#    Concrete trace: if lane 0 is red with two blue sprites, lane 2 is green with one yellow sprite, and lane 3
#    is unmarked with three orange sprites, then the seed output would be [grey blue-lane, unchanged lane 1,
#    yellow lane with two extra yellow sprites, unchanged orange lane]. The combo output becomes [blue lane kept
#    in place, lane 1 grey if unmarked, yellow lane plus two extra blue sprites stacked into empty slots, orange
#    spectator lane grey]. The marker bars still disappear.
# b) Complexity calibration
#    The puzzle remains structurally simple enough for ARC: all lanes share one lattice, all sprites share one
#    shape, source counts still determine how many new sprites appear, and only one lane receives copies. The
#    visible effect is stronger than any single change alone: red-marked lanes remain colored, copied colors are
#    visible in the destination stack, and unmarked lanes visibly bleach. Because there are only four lanes and
#    1-2 sources, the number of role categories stays small and repeated across samples.
# c) Hidden-assumption audit of the original solver
#    The seed solver encoded three assumptions that no longer hold:
#    - It greyed source sprites directly inside the red-lane loop; this must be removed because D6/2 preserves
#      source pixels.
#    - It painted copied sprites using `best["dst_color"]`; this must switch to the current source color.
#    - It left unmarked lanes untouched; now the solver must detect all lanes first so it can grey only the
#      spectator lanes before placing copied sprites.
#    A new combined assumption is introduced: lane roles must be inferred from marker spans, while unmarked lanes
#    are found from sprite-column bounding boxes after marker pixels are ignored. The generator must therefore keep
#    every lane populated and aligned to the fixed 5-column layout so those column ranges remain recoverable.
# d) Placement legality
#    No new geometry is introduced. The generator still uses the original period-4 lattice, counts 1..3 per input
#    lane, and the same capacity bound `target_count + sum(source_counts) <= 7`, so every copied sprite fits in
#    rows 1..27 without overlap. Output-first construction explicitly simulates the same bottom-up slot search as
#    the solver, which keeps input/output consistency exact.
# e) Visual observability
#    From 2-4 examples, a human can read the rule because three independent cues line up: red-marked lanes keep
#    their own colors, those same colors reappear as new sprites in the green lane, and the unmarked lanes alone
#    turn grey. Distinct lane colors guarantee the destination lane is visibly mixed rather than accidentally
#    matching a source color.
# f) Visual cleanliness
#    The lane spacing and sprite lattice are unchanged, so nothing overlaps except the intended vertical stacking
#    inside the target lane. Each copied sprite occupies a whole empty slot, all marker bars are erased, and the
#    spectator greying affects only entire monochrome sprites, keeping the output easy to parse.
# g) Change inferability
#    The color changes remain tightly structured: only source colors can appear as new colors in the green lane,
#    and only lanes without any marker bleach to grey. There is no dependence on color IDs or hidden ordering
#    beyond a left-to-right source processing order, which is visible because copied source-color groups stack in
#    deterministic chunks from the bottom upward.
# Assumption audit for the new solver/generator pair
#    - Every lane contains at least one sprite in the input, so `_find_lane_ranges(...)` recovers all four lanes.
#    - The green lane contains at least one pre-existing sprite, so its vertical period and occupied slots can be
#      inferred before any copying happens.
#    - Source markers are processed left-to-right in both solver and generator, which makes the order of colored
#      copied groups in the target lane deterministic.
#    - Unmarked lanes always exist with four total lanes, one green lane, and only one or two red lanes, so the
#      spectator-grey part of the combo is visible in every generated sample.


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
    bars = [t[4] for t in cands]
    bars.sort(key=lambda pts: bbox_coords(np.array(pts))[1])
    return bars


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


def _sort_sprites(
    sprites: list[list[tuple[int, int]]],
    g: np.ndarray,
) -> list[tuple[int, int, int, list[tuple[int, int]]]]:
    out = []
    for pts in sprites:
        rmin = min(r for r, _ in pts)
        cmin = min(c for _, c in pts)
        color = int(g[pts[0][0], pts[0][1]])
        out.append((rmin, cmin, color, pts))
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
    Copy source-colored sprites into the green lane while greying only the unmarked spectator lanes.

    Input structure:
    - The most common color is the background. Thin red (2) and green (3) 1x5 bars on the last row mark source and destination lanes.
    - Each lane is a 5-column vertical slot filled with repeated identical monochrome plus sprites on a regular period-4 lattice.

    Rule:
    - Every red-marked lane keeps its own sprites in place and also contributes the same number of copied sprites to the nearest green lane by horizontal center.
    - Copied sprites keep the source lane's color, stack into empty target slots from the bottom upward, and coexist with the green lane's original color.
    - Lanes with no marker are greyed out, and all marker bars are removed.

    Key insight:
    - Red bars now mean "clone this colored stack into the green lane" rather than "move and bleach it", while the only lanes that fade are the unmarked spectators.
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

    red_ranges = set()
    for rm in red_markers:
        _r0, c0, _r1, c1 = bbox_coords(np.array(rm))
        red_ranges.add((int(c0), int(c1)))

    green_state = {}
    target_ranges = set()
    for gm in green_markers:
        _r0, dst_c0, _r1, dst_c1 = bbox_coords(np.array(gm))
        target_ranges.add((int(dst_c0), int(dst_c1)))
        dst_sprites = _lane_sprites_in_cols(g, range(dst_c0, dst_c1 + 1), bg, ignored=set(gm))
        if not dst_sprites:
            continue

        dst_sprites_sorted = _sort_sprites(dst_sprites, g)
        tops = sorted({rmin for rmin, _cmin, _color, _pts in dst_sprites_sorted})
        if len(tops) >= 2:
            diffs = [b - a for a, b in zip(tops, tops[1:]) if b - a > 0]
            period = int(np.gcd.reduce(diffs)) if diffs else 4
        else:
            r0, c0, r1, c1 = bbox_coords(np.array(dst_sprites_sorted[0][3]))
            period = (r1 - r0 + 1) + 1

        green_state[id(gm)] = {
            "center": (dst_c0 + dst_c1) / 2.0,
            "period": period,
            "occupied": set(tops),
            "bottom0": max(tops),
        }

    assert green_state, "expected at least one green lane with valid sprites"

    out = g.copy()

    for c0, c1 in lane_ranges:
        lane_range = (int(c0), int(c1))
        if lane_range in target_ranges or lane_range in red_ranges:
            continue
        lane_sprites = _lane_sprites_in_cols(g, range(c0, c1 + 1), bg)
        for pts in lane_sprites:
            for r, c in pts:
                out[r, c] = 5

    for rm in red_markers:
        _r0, src_c0, _r1, src_c1 = bbox_coords(np.array(rm))
        src_sprites = _lane_sprites_in_cols(g, range(src_c0, src_c1 + 1), bg, ignored=set(rm))
        if not src_sprites:
            continue

        rc = (src_c0 + src_c1) / 2.0
        best = min(green_state.values(), key=lambda st: abs(st["center"] - rc))

        src_sprites_sorted = _sort_sprites(src_sprites, g)
        src_ref_r0, _src_ref_c0, src_color, src_ref_pts = src_sprites_sorted[0]
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
_MARKER_COLORS = {2, 3}

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
    # Reserve marker hues for the bars so lane roles stay visually unambiguous.
    sprite_palette = [c for c in range(10) if c not in ({bg, _GREY} | _MARKER_COLORS)]
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
    gain = sum(input_counts[i] for i in reds)

    output_counts = input_counts[:]
    output_counts[green] += gain
    out = np.full((_H, _W), bg, dtype=int)

    for lane_idx in range(_NUM_LANES):
        if lane_idx == green:
            _paint_lane(out, lane_idx, lane_colors[lane_idx], input_counts[lane_idx], bg)
        elif lane_idx in red_set:
            _paint_lane(out, lane_idx, lane_colors[lane_idx], input_counts[lane_idx], bg)
        else:
            _paint_lane(out, lane_idx, _GREY, input_counts[lane_idx], bg)

    green_c0 = _LANE_STARTS[green]
    occupied = set(_lane_tops(input_counts[green]))
    transferred: list[tuple[int, int]] = []
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
    for lane_idx in range(_NUM_LANES):
        if lane_idx == green or lane_idx in red_set:
            continue
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

    unmarked = [i for i in range(_NUM_LANES) if i != green and i not in red_set]
    if not unmarked:
        return None
    for lane_idx in unmarked:
        c0 = _LANE_STARTS[lane_idx]
        lane_colors_out = {
            int(out[r, c])
            for r in range(_H)
            for c in range(c0, c0 + _SPRITE_W)
            if int(out[r, c]) != bg
        }
        if lane_colors_out != {_GREY}:
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
