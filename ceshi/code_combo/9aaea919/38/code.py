# Combo variant of 9aaea919 (index 38)
# combo_spec: D10/0+D15/2
# Name: 可变周期+颜色随高度渐变

import numpy as np

from arc_utils import bbox_coords, components_mask, get_bg, paste


# Design analysis for the fused D10/0 + D15/2 variant
# ----------------------------------------------------
# a) Concrete behavior and trace
#    - The seed task already has 4 fixed 5-column lanes, 3x5 plus sprites, red source bars on row 29,
#      and one green destination bar on row 29.
#    - Fusion change 1: lanes no longer share one global period. Each lane uses period 4 or 5, while the
#      bottom visible sprite still has top row 25.
#    - Fusion change 2: non-grey sprite color is no longer constant per lane. Instead, slot 0..5 from
#      bottom to top uses one shared 6-step gradient across all non-grey lanes.
#    - Example trace:
#      red lane A shows tops [21, 25] (period 4); green lane B shows tops [15, 20, 25] (period 5).
#      If A transfers 2 sprites into B, the old monochrome reading would add two copies using one lane color.
#      The fused rule instead places them at tops 10 and 5 (continue B's period-5 ladder) and colors them
#      with the slot-3 and slot-4 gradient colors rather than reusing a single destination color.
#
# b) Complexity calibration
#    - Period choices are limited to {4, 5}; this is visually obvious from repeated spacings but not too broad.
#    - The gradient uses exactly 6 colors, matching the maximum total sprites we allow in any destination lane.
#      That keeps the palette finite and every possible transferred color directly visible in one legend lane.
#    - Marked lanes are generated with at least 2 visible sprites, so their own cadence is readable from the image.
#
# c) Implicit assumptions audited from the seed solver
#    - The seed line `dst_color = ...` is invalid because a destination lane is no longer monochrome.
#      Replacement: infer slot->color, then color each inserted sprite from its final height slot.
#    - Any logic that derives one global period from all sprite tops is invalid once period 4 and 5 can coexist.
#      Replacement: infer period per lane from that lane's own repeated top rows.
#    - The seed fallback "single visible destination sprite => use sprite height + 1" is too weak here.
#      Replacement: the generator guarantees every marked lane has at least two visible sprites.
#    - The transfer loop still assumes all sprites share one shape and that the target lane keeps its bottom
#      visible segment. The generator preserves both assumptions.
#
# d) Placement legality
#    - Destination totals are capped at 6 sprites; this fits period 5 exactly (tops 25,20,15,10,5,0) and also
#      fits period 4 (tops 25,21,17,13,9,5).
#    - All lanes stay in disjoint fixed 5-column slots, so horizontal shifts never overlap adjacent lanes.
#    - New sprites are only placed on background cells after the generator removes the gained top segment from
#      the chosen green lane.
#
# e) Visual observability
#    - Every sample includes one unmarked full-height legend lane showing the complete 6-step color ladder.
#      That reveals the slot->color rule without requiring hidden numeric color ordering.
#    - Marked lanes are constrained so that the acted-upon lanes include both period values somewhere among
#      red/green lanes, making the variable-period dimension visible in the transformation itself.
#
# f) Visual cleanliness
#    - Marker bars remain isolated on row 29 with row 28 empty, so they do not merge with the bottom sprites.
#    - Only whole sprites move conceptually; no partial overlays or multi-object collisions are introduced.
#    - The legend lane is left unmarked, so it reads as an explanatory reference rather than extra action noise.
#
# g) Color-change inferability
#    - Colors are not ranked by numeric id. The mapping is inferred from repeated height-position evidence:
#      whichever color appears at slot k in visible lanes is the color reused for any transferred sprite landing
#      in slot k.
#    - A single shared gradient across all non-grey lanes is deliberately simpler than arbitrary per-lane gradients;
#      it keeps the rule learnable from 2-4 ARC examples while still forcing the solver to bind color to height.
#
# New solver assumptions after fusion, and how the generator guarantees them
#    - Period inference assumes every marked lane has at least 2 visible sprites. `generate_input()` enforces this:
#      one-red cases sample 2-3 visible sprites, while two-red cases pin all marked lanes to 2 so the 6-slot
#      destination remains feasible.
#    - Slot-color inference assumes every slot that can appear after transfer is already visible in the input.
#      The unmarked legend lane always contains all 6 slots.
#    - Slot indexing assumes a shared bottom anchor at top row 25. The generator keeps that original layout rule.
#    - The nearest-green matching assumes exactly one green lane in practice. The generator still creates exactly
#      one green marker bar.


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
    mask = (g != bg) & colmask
    comps = [[(int(r), int(c)) for r, c in pts] for pts in components_mask(mask, connectivity=4)]
    out = []
    for pts in comps:
        colors = {int(g[r, c]) for r, c in pts}
        if len(colors) == 1:
            out.append(pts)
    return out


def _lane_sprites(
    g: np.ndarray,
    cols: range,
    bg: int,
    marker_pts: set[tuple[int, int]],
) -> list[dict[str, object]]:
    sprites = []
    for pts in _single_color_components_in_cols(g, cols, bg):
        pts2 = [p for p in pts if p not in marker_pts]
        if not pts2:
            continue
        r0 = min(r for r, _ in pts2)
        c0 = min(c for _, c in pts2)
        sprites.append(
            {
                "pts": pts2,
                "top": r0,
                "left": c0,
                "color": int(g[pts2[0][0], pts2[0][1]]),
            }
        )
    sprites.sort(key=lambda sp: (int(sp["top"]), int(sp["left"])))
    return sprites


def _infer_period_from_tops(tops: list[int]) -> int | None:
    if len(tops) < 2:
        return None
    diffs = [b - a for a, b in zip(tops, tops[1:]) if b > a]
    assert diffs, "expected at least one positive spacing to infer a lane period"
    period = int(np.gcd.reduce(diffs))
    assert period > 0, "lane period must be positive"
    return period


def _collect_lane_states(
    g: np.ndarray,
    bg: int,
    red_markers: list[list[tuple[int, int]]],
    green_markers: list[list[tuple[int, int]]],
) -> list[dict[str, object]]:
    marker_sets = {id(m): set(m) for m in red_markers + green_markers}
    lane_marker_pts = {c0: set() for c0 in _LANE_STARTS}
    lane_marker_color: dict[int, int | None] = {c0: None for c0 in _LANE_STARTS}

    for marker, marker_color in [(m, 2) for m in red_markers] + [(m, 3) for m in green_markers]:
        _r0, c0, _r1, _c1 = bbox_coords(np.array(marker))
        lane_marker_pts[c0].update(marker_sets[id(marker)])
        lane_marker_color[c0] = marker_color

    states = []
    for lane_idx, c0 in enumerate(_LANE_STARTS):
        cols = range(c0, c0 + _SPRITE_W)
        sprites = _lane_sprites(g, cols, bg, lane_marker_pts[c0])
        tops = [int(sp["top"]) for sp in sprites]
        states.append(
            {
                "lane_idx": lane_idx,
                "c0": c0,
                "center": c0 + (_SPRITE_W - 1) / 2.0,
                "marker_color": lane_marker_color[c0],
                "sprites": sprites,
                "tops": tops,
                "period": _infer_period_from_tops(tops),
            }
        )
    return states


def _infer_height_palette(lane_states: list[dict[str, object]]) -> tuple[int, dict[int, int]]:
    inferable = [st for st in lane_states if st["period"] is not None and st["sprites"]]
    assert inferable, "expected at least one lane with inferable cadence"

    bottom0 = max(int(sp["top"]) for st in inferable for sp in st["sprites"])
    slot_to_color: dict[int, int] = {}
    for st in inferable:
        period = int(st["period"])
        for sp in st["sprites"]:
            color = int(sp["color"])
            if color == _GREY:
                continue
            delta = bottom0 - int(sp["top"])
            assert delta % period == 0, "sprite top must align to its lane lattice"
            slot = delta // period
            if slot in slot_to_color:
                assert slot_to_color[slot] == color, "height-color mapping must be globally consistent"
            else:
                slot_to_color[slot] = color

    return bottom0, slot_to_color


def main(grid: np.ndarray) -> np.ndarray:
    """
    Move sprites out of red-marked lanes and extend the nearest green lane using that lane's own cadence and a shared height gradient.

    Input structure:
    - The background is the most common color, and row-29 bars in red (2) and green (3) mark source and destination lanes.
    - Each 5-column lane contains repeated monochrome plus sprites, but lanes may use different vertical periods and non-grey lanes color sprites by height slot rather than by lane identity.

    Rule:
    - Every sprite in a red-marked lane turns grey (5) in place.
    - The same number of sprites is recreated in the nearest green-marked lane, stacked upward in that destination lane's own period.
    - Each inserted sprite takes the color associated with its final height slot, and all marker bars are erased afterward.

    Key insight:
    - Two old shortcuts fail at once: the target lane is not monochrome, and the board no longer has one universal cadence. Solve by inferring each lane's period locally, recovering the shared slot->color ladder from visible lanes, and then continuing the chosen green lane on its own lattice.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = get_bg(g)

    red_markers = _find_marker_bars(g, 2)
    green_markers = _find_marker_bars(g, 3)
    assert red_markers and green_markers, "expected both red and green marker bars to be present"

    lane_states = _collect_lane_states(g, bg, red_markers, green_markers)
    bottom0, slot_to_color = _infer_height_palette(lane_states)

    green_states = []
    for st in lane_states:
        if st["marker_color"] != 3:
            continue
        assert st["period"] is not None, "expected the destination lane period to be inferable"
        assert st["sprites"], "expected the destination lane to contain visible sprites"
        green_states.append(
            {
                "center": float(st["center"]),
                "period": int(st["period"]),
                "occupied": {int(sp["top"]) for sp in st["sprites"]},
                "bottom0": int(bottom0),
            }
        )
    assert green_states, "expected at least one green lane with valid visible sprites"

    out = g.copy()

    for st in lane_states:
        if st["marker_color"] != 2:
            continue
        src_sprites = st["sprites"]
        if not src_sprites:
            continue

        for sp in src_sprites:
            for r, c in sp["pts"]:
                out[r, c] = _GREY

        rc = float(st["center"])
        best = min(green_states, key=lambda gst: abs(float(gst["center"]) - rc))

        src_ref_top = int(src_sprites[0]["top"])
        src_ref_pts = src_sprites[0]["pts"]
        dc_shift = int(round(float(best["center"]) - rc))
        src_offsets = [(r - src_ref_top, c + dc_shift) for r, c in src_ref_pts]
        sprite_h = max(dr for dr, _c in src_offsets) + 1

        need = len(src_sprites)
        placed = 0
        k = 0
        while placed < need:
            new_top = int(best["bottom0"]) - k * int(best["period"])
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

            slot = (int(best["bottom0"]) - new_top) // int(best["period"])
            assert slot in slot_to_color, "missing visible color for the destination slot"
            new_color = slot_to_color[slot]
            for dr, c in src_offsets:
                out[new_top + dr, c] = new_color
            best["occupied"].add(new_top)
            placed += 1

    for marker in red_markers + green_markers:
        for r, c in marker:
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
_MAX_LANE_TOTAL = 6
_LEGEND_COUNT = 6

# --- CFG: color palette ---
_BG_CHOICES = [1, 6]
_GREY = 5
_HEIGHT_COLORS_BOTTOM_UP = [9, 2, 7, 4, 8, 0]

# --- CFG: counts & limits ---
_MIN_ACTIVE_SPRITES = 2
_MAX_ACTIVE_SPRITES = 3
_GEN_RETRIES = 300


def _sprite_patch(color: int, bg: int) -> np.ndarray:
    """Return a 3x5 plus sprite patch with bg-transparent corners."""
    patch = np.full((_SPRITE_H, _SPRITE_W), bg, dtype=int)
    patch[0, 1:4] = color
    patch[1, :] = color
    patch[2, 1:4] = color
    return patch


def _lane_capacity(period: int) -> int:
    return (_BOTTOM_TOP // period) + 1


def _lane_tops(period: int, count: int) -> list[int]:
    return [_BOTTOM_TOP - period * k for k in range(count - 1, -1, -1)]


def _color_for_top(top: int, period: int) -> int:
    slot = (_BOTTOM_TOP - int(top)) // int(period)
    return _HEIGHT_COLORS_BOTTOM_UP[slot]


def _paint_gradient_lane(grid: np.ndarray, lane_idx: int, count: int, bg: int, period: int) -> None:
    c0 = _LANE_STARTS[lane_idx]
    for top in _lane_tops(period, count):
        paste(grid, _sprite_patch(_color_for_top(top, period), bg), top, c0, transparent=bg)


def _paint_solid_lane(grid: np.ndarray, lane_idx: int, color: int, count: int, bg: int, period: int) -> None:
    patch = _sprite_patch(color, bg)
    c0 = _LANE_STARTS[lane_idx]
    for top in _lane_tops(period, count):
        paste(grid, patch, top, c0, transparent=bg)


def _gen_attempt() -> np.ndarray | None:
    rng = np.random.default_rng()
    bg = int(rng.choice(_BG_CHOICES))

    legend_lane = int(rng.integers(_NUM_LANES))
    active = [i for i in range(_NUM_LANES) if i != legend_lane]
    green = active[int(rng.integers(len(active)))]
    others = [i for i in active if i != green]

    red_count = int(rng.choice([1, 2]))
    red_pick = rng.choice(others, size=red_count, replace=False)
    reds = tuple(int(x) for x in np.atleast_1d(red_pick).tolist())
    red_set = set(reds)

    periods = [int(rng.choice(_PERIOD_CHOICES)) for _ in range(_NUM_LANES)]
    green_period = periods[green]
    other_period = 9 - green_period
    if red_count == 1:
        periods[reds[0]] = other_period
    else:
        mismatch_mode = int(rng.integers(2))
        if mismatch_mode == 0:
            periods[reds[0]] = green_period
            periods[reds[1]] = other_period
        else:
            periods[reds[0]] = other_period
            periods[reds[1]] = other_period
    periods[legend_lane] = int(rng.choice(_PERIOD_CHOICES))

    counts = [int(rng.integers(_MIN_ACTIVE_SPRITES, _MAX_ACTIVE_SPRITES + 1)) for _ in range(_NUM_LANES)]
    counts[legend_lane] = _LEGEND_COUNT
    if red_count == 1:
        counts[green] = int(rng.integers(_MIN_ACTIVE_SPRITES, _MAX_ACTIVE_SPRITES + 1))
        counts[reds[0]] = int(rng.integers(_MIN_ACTIVE_SPRITES, _MAX_ACTIVE_SPRITES + 1))
    else:
        # Two red source lanes only fit the 6-slot destination if all three marked lanes start with 2 sprites.
        counts[green] = _MIN_ACTIVE_SPRITES
        for lane_idx in reds:
            counts[lane_idx] = _MIN_ACTIVE_SPRITES

    gain = sum(counts[i] for i in reds)
    cap = min(_MAX_LANE_TOTAL, _lane_capacity(periods[green]))
    if counts[green] + gain > cap:
        return None

    output_counts = counts[:]
    output_counts[green] += gain

    out = np.full((_H, _W), bg, dtype=int)
    for lane_idx in range(_NUM_LANES):
        if lane_idx in red_set:
            _paint_solid_lane(out, lane_idx, _GREY, output_counts[lane_idx], bg, periods[lane_idx])
        else:
            _paint_gradient_lane(out, lane_idx, output_counts[lane_idx], bg, periods[lane_idx])

    inp = out.copy()
    for lane_idx in reds:
        _paint_gradient_lane(inp, lane_idx, counts[lane_idx], bg, periods[lane_idx])

    green_c0 = _LANE_STARTS[green]
    for top in _lane_tops(periods[green], output_counts[green])[:gain]:
        inp[top : top + _SPRITE_H, green_c0 : green_c0 + _SPRITE_W] = bg
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
