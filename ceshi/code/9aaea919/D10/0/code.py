# Variant of 9aaea919
# Dimension: D10 — 精灵垂直周期与间距 (meta)
# Change: 垂直周期为4行(3行精灵高度+1行间隙)，最底部精灵顶行固定为第25行... → 可变周期

import numpy as np

from arc_utils import bbox_coords, components_mask, get_bg, paste
from itertools import combinations


# Design analysis for D10
# a) Concrete behavior:
#    - The canvas, lane columns, marker bars, and 3x5 plus sprite stay unchanged.
#    - The only structural change is vertical cadence: each lane now uses period 4 or 5,
#      while the bottom visible sprite still sits with top row 25.
#    - Example trace:
#      source lane tops [21, 25] (period 4), target lane tops [20, 25] (period 5).
#      Old fixed-period reading would keep stepping by 4; the variant must continue the
#      target cadence instead, so transferred sprites land at rows 15, 10, ... rather
#      than 17, 13, ...
# b) Complexity calibration:
#    - Period choices are limited to {4, 5}. This is visually distinct but not too broad.
#    - Every generated lane shows 2-3 visible sprites, so the repeated spacing needed to
#      infer the period is always present directly in the image.
# c) Implicit assumptions audited from the seed solver:
#    - The old fallback "single visible target sprite => period 4" is invalid here.
#    - Placement must not assume a global period shared by all lanes.
#    - Bottom anchoring should come from the visible bottom sprite of the chosen target
#      lane, not from a hard-coded transfer lattice.
# d) Placement legality:
#    - Lane capacity depends on the destination period: period 4 fits 7 sprites, period 5
#      fits 6 sprites, so role selection is filtered against that capacity.
#    - Sprites are only placed on empty cells and stay inside the 30x30 canvas.
# e) Visual observability:
#    - Every sample contains both period values somewhere on the board, and the marked
#      lanes are also required to include both values, so the D10 change is visible from
#      a few examples rather than hiding in unmarked background lanes.
# f) Visual cleanliness:
#    - Lanes remain disjoint 5-column slots separated by 1+ columns of background.
#    - Marker bars remain on the last row, leaving at least one empty row below sprites.
# g) Color inference:
#    - No ordering or recoloring depends on numeric color ids; colors only label lanes,
#      while geometry and marker positions expose the rule.
# h) New solver assumptions guaranteed by the generator:
#    - Every marked lane has at least two visible sprites, so its own period is directly
#      inferable from top-row differences.
#    - The input target lane always keeps its bottom segment intact; only gained top
#      sprites are removed when deriving the input from the output.


def _find_marker_bars(g: np.ndarray, marker_color: int) -> list[list[tuple[int, int]]]:
    """Marker bars are thin 4-connected components of the given marker color."""
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


def _sorted_lane_sprites(
    g: np.ndarray, cols: range, bg: int, marker_pts: list[tuple[int, int]]
) -> list[tuple[int, int, list[tuple[int, int]]]]:
    marker_set = set(marker_pts)
    comps = _single_color_components_in_cols(g, cols, bg)
    comps = [[p for p in pts if p not in marker_set] for pts in comps]
    comps = [pts for pts in comps if pts]
    if not comps:
        return []

    sprite_sizes = [len(pts) for pts in comps if len(pts) != len(marker_pts)]
    if not sprite_sizes:
        return []
    sprite_sz = max(sprite_sizes)
    sprites = [pts for pts in comps if len(pts) == sprite_sz]

    out = []
    for pts in sprites:
        rmin = min(r for r, _ in pts)
        cmin = min(c for _, c in pts)
        out.append((rmin, cmin, pts))
    out.sort()
    return out


def _infer_period(sprite_rows: list[int]) -> int:
    diffs = [b - a for a, b in zip(sprite_rows, sprite_rows[1:]) if b > a]
    assert len(diffs) >= 1, "expected at least two visible sprites to infer a lane period"
    period = int(np.gcd.reduce(diffs))
    assert period > 0, "lane period must be positive"
    return period


def _lane_state_from_marker(g: np.ndarray, marker_pts: list[tuple[int, int]], bg: int) -> dict | None:
    _, c0, _, c1 = bbox_coords(np.array(marker_pts))
    cols = range(c0, c1 + 1)
    sprites_sorted = _sorted_lane_sprites(g, cols, bg, marker_pts)
    if len(sprites_sorted) < 2:
        return None

    tops = [rmin for rmin, _cmin, _pts in sprites_sorted]
    period = _infer_period(tops)
    color = int(g[sprites_sorted[0][2][0][0], sprites_sorted[0][2][0][1]])
    center = (c0 + c1) / 2.0
    return {
        "marker": marker_pts,
        "center": center,
        "color": color,
        "period": period,
        "tops": tops,
        "occupied": set(tops),
        "bottom0": max(tops),
        "sprites_sorted": sprites_sorted,
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Move sprites out of red-marked lanes and extend the chosen green lane using that lane's own rhythm.

    Input structure:
    - The background is the most common color, and 1x5 red/green bars on the last row mark source and destination lanes.
    - Each 5-column lane contains repeated monochrome plus sprites; different lanes may use different vertical periods even though their bottom visible sprite is aligned near the same row.

    Rule:
    - Every sprite in a red-marked lane turns grey in place.
    - The same number of sprites is recreated in the nearest green-marked lane, recolored to that lane's sprite color and stacked upward following the destination lane's inferred period.
    - All marker bars are erased afterward.

    Key insight:
    - Counts still transfer exactly as before, but spacing is no longer global: infer each marked lane's cadence from its repeated sprite tops, then continue the target lane on that cadence instead of assuming a fixed step of 4.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = get_bg(g)

    red_markers = _find_marker_bars(g, 2)
    green_markers = _find_marker_bars(g, 3)
    assert red_markers and green_markers, "expected both red and green marker bars to be present"

    green_states = []
    for gm in green_markers:
        st = _lane_state_from_marker(g, gm, bg)
        if st is not None:
            green_states.append(st)
    assert green_states, "expected at least one green lane with inferable period"

    out = g.copy()

    for rm in red_markers:
        src_state = _lane_state_from_marker(g, rm, bg)
        if src_state is None:
            continue

        src_sprites_sorted = src_state["sprites_sorted"]

        for _rmin, _cmin, pts in src_sprites_sorted:
            for r, c in pts:
                out[r, c] = 5

        rc = src_state["center"]
        best = min(green_states, key=lambda st: abs(st["center"] - rc))

        src_ref_r0, _src_ref_c0, src_ref_pts = src_sprites_sorted[0]
        dc_shift = int(round(best["center"] - rc))
        src_offsets = [(r - src_ref_r0, c + dc_shift) for r, c in src_ref_pts]
        sprite_h = max(dr for dr, _c in src_offsets) + 1

        placed = 0
        need = len(src_sprites_sorted)
        k = 0
        while placed < need:
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
                out[new_top + dr, c] = best["color"]
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
_PERIOD_CHOICES = (4, 5)

# --- CFG: color palette ---
_BG_CHOICES = [0, 1, 4, 6, 7, 8, 9]
_GREY = 5

# --- CFG: counts & limits ---
_MIN_SPRITES = 2
_MAX_SPRITES = 3
_GEN_RETRIES = 300


def _sprite_patch(color: int, bg: int) -> np.ndarray:
    """Return a 3x5 plus/diamond sprite patch with bg-transparent corners."""
    patch = np.full((_SPRITE_H, _SPRITE_W), bg, dtype=int)
    patch[0, 1:4] = color
    patch[1, :] = color
    patch[2, 1:4] = color
    return patch


def _lane_capacity(period: int) -> int:
    """Maximum number of sprites that fit when the bottom top-row is fixed at row 25."""
    return (_BOTTOM_TOP // period) + 1


def _lane_tops(period: int, count: int) -> list[int]:
    """Return top-row positions for `count` sprites, ordered top-to-bottom."""
    return [_BOTTOM_TOP - period * k for k in range(count - 1, -1, -1)]


def _paint_lane(grid: np.ndarray, lane_idx: int, color: int, count: int, bg: int, period: int) -> None:
    """Stamp `count` sprites of `color` into the given lane using the chosen period."""
    patch = _sprite_patch(color, bg)
    c0 = _LANE_STARTS[lane_idx]
    for r0 in _lane_tops(period, count):
        paste(grid, patch, r0, c0, transparent=bg)


def _feasible_roles(counts: list[int], periods: list[int]) -> dict[int, list[tuple[tuple[int, ...], int]]]:
    """Return feasible (reds, green) assignments grouped by number of red lanes."""
    grouped: dict[int, list[tuple[tuple[int, ...], int]]] = {1: [], 2: []}
    for green in range(_NUM_LANES):
        others = [i for i in range(_NUM_LANES) if i != green]
        cap = _lane_capacity(periods[green])
        for red_count in (1, 2):
            for reds in combinations(others, red_count):
                marked_periods = {periods[green], *(periods[i] for i in reds)}
                total = counts[green] + sum(counts[i] for i in reds)
                if total <= cap and len(marked_periods) == 2:
                    grouped[red_count].append((reds, green))
    return grouped


def _gen_attempt() -> np.ndarray | None:
    rng = np.random.default_rng()

    bg = int(rng.choice(_BG_CHOICES))
    sprite_palette = [c for c in range(10) if c not in (bg, _GREY)]
    lane_colors = rng.choice(sprite_palette, size=_NUM_LANES, replace=False).tolist()
    input_counts = rng.integers(_MIN_SPRITES, _MAX_SPRITES + 1, size=_NUM_LANES).tolist()

    periods = rng.choice(_PERIOD_CHOICES, size=_NUM_LANES, replace=True).tolist()
    if len(set(periods)) < 2:
        return None

    feasible = _feasible_roles(input_counts, periods)
    feasible_red_counts = [rc for rc in (1, 2) if feasible[rc]]
    if not feasible_red_counts:
        return None

    red_count = int(rng.choice(feasible_red_counts))
    reds, green = feasible[red_count][int(rng.integers(len(feasible[red_count])))]
    red_set = set(reds)
    gain = sum(input_counts[i] for i in reds)

    output_counts = input_counts[:]
    output_counts[green] += gain

    out = np.full((_H, _W), bg, dtype=int)
    for lane_idx in range(_NUM_LANES):
        color = _GREY if lane_idx in red_set else lane_colors[lane_idx]
        _paint_lane(out, lane_idx, color, output_counts[lane_idx], bg, periods[lane_idx])

    inp = out.copy()
    for lane_idx in reds:
        _paint_lane(inp, lane_idx, lane_colors[lane_idx], input_counts[lane_idx], bg, periods[lane_idx])

    green_c0 = _LANE_STARTS[green]
    for r0 in _lane_tops(periods[green], output_counts[green])[:gain]:
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
