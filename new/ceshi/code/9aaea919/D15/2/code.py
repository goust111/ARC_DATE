# Variant of 9aaea919
# Dimension: D15 — 精灵颜色约束 (meta)
# Change: 4条车道各自使用不同的单色(从{0-9}中排除背景色和灰色5后选取4种不重复色)... → 精灵颜色随高度变化

import numpy as np

from arc_utils import bbox_coords, components_mask, get_bg, paste
from itertools import combinations


# Design analysis for the D15 variant
# -----------------------------------
# spec_variant:
#   unchanged_structure:
#     grid: 30x30
#     lanes: 4 fixed 5-column lanes at cols 3-7 / 9-13 / 15-19 / 21-25
#     sprite_shape: 3x5 plus sprite with transparent corners
#     markers: red 1x5 bar marks source lane, green 1x5 bar marks destination lane
#     motion_rule: source sprites grey out; the same number of sprites stack bottom-up into the nearest green lane
#   core_change:
#     old_color_rule: every lane used one monochrome sprite color
#     new_color_rule: sprite color is determined by vertical slot, not by lane identity
#     visible_mapping: slot 0..4 from bottom to top uses a fixed 5-step gradient shared by all non-grey lanes
#   adaptation_for_observability:
#     keep one unmarked "legend" lane filled with all 5 slots in the input
#     cap destination total to 5 sprites, so every output color already appears somewhere in the input
#
# a) Concrete behavior trace
#    Original: if a green lane already had 2 yellow sprites and receives 2 more, both new sprites were also yellow.
#    Variant: if the lane has bottom two slots occupied, then the inserted sprites go into higher slots and must take
#    the colors assigned to those higher slots. The transfer still preserves count and shape, but color now depends on
#    final height.
#
# b) Complexity calibration
#    A fully arbitrary per-lane palette would be unreadable from 2-4 ARC examples. A shared height palette plus a
#    complete unmarked legend lane is enough structure for humans to infer the rule, while still forcing the solver to
#    reason about height instead of copying a single destination color.
#
# c) Hidden assumptions in the original solver
#    - The original `dst_color = ...` assumes one constant color per destination lane. That fails here.
#    - The original component finder is still valid because each individual sprite remains monochrome; only different
#      heights use different colors.
#    - Filling logic based on bottom/top lattice slots is still valid, but each placed sprite must query a
#      slot-specific color.
#
# d) Placement legality
#    - A destination lane is limited to 5 slots, matching the visible legend lane.
#    - The generator only chooses role/count assignments whose transferred total fits inside those 5 slots.
#    - Markers stay on row 29, separated from the bottom sprite by one blank row, so components remain distinct.
#
# e/f/g) Visual readability and color inference
#    - The unmarked legend lane shows the full gradient every time, so newly appearing higher sprites do not require
#      guessing unseen colors.
#    - Colors are not sorted by numeric id during solving. The solver infers the slot->color map from observed sprites.
#    - Only five gradient colors are used, keeping the scene clean and the mapping learnable.
#
# Solver assumptions audited against the generator
#    - `_infer_height_palette(...)` assumes at least one lane exposes every slot that may appear in the output:
#      `generate_input()` guarantees this via the fixed 5-sprite unmarked legend lane and by capping totals at 5.
#    - Destination placement assumes sprites lie on one common period-4 lattice with a shared bottom anchor:
#      the generator still uses the original lane geometry unchanged.
#    - Marker filtering assumes at most one marker bar per lane and a one-row gap above row 29:
#      the generator preserves the original bar placement exactly.


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


def _infer_height_palette(all_lane_sprites: list[list[dict[str, object]]]) -> tuple[int, int, dict[int, int]]:
    all_tops = sorted({int(sp["top"]) for lane in all_lane_sprites for sp in lane})
    assert all_tops, "expected at least one visible sprite"

    if len(all_tops) >= 2:
        diffs = [b - a for a, b in zip(all_tops, all_tops[1:]) if b - a > 0]
        period = int(np.gcd.reduce(diffs)) if diffs else 4
    else:
        pts = np.array(all_lane_sprites[0][0]["pts"])
        r0, _c0, r1, _c1 = bbox_coords(pts)
        period = (r1 - r0 + 1) + 1

    bottom0 = max(all_tops)
    slot_to_color: dict[int, int] = {}
    for lane in all_lane_sprites:
        for sp in lane:
            top = int(sp["top"])
            delta = bottom0 - top
            assert delta % period == 0, "sprite top must align to the lane lattice"
            slot = delta // period
            color = int(sp["color"])
            if slot in slot_to_color:
                assert slot_to_color[slot] == color, "height-color mapping must be consistent"
            else:
                slot_to_color[slot] = color

    return bottom0, period, slot_to_color


def main(grid: np.ndarray) -> np.ndarray:
    """
    Move sprites from red-marked lanes into the green-marked lane, using a height-based color ladder.

    Input structure:
    - The background is the most common color; row-29 bars in red (2) and green (3) mark source and destination lanes.
    - Every visible sprite is the same plus/diamond shape on a regular vertical lattice, but non-grey lanes color sprites by height.

    Rule:
    - Each red lane turns grey (5) in place, while its sprite count transfers to the nearest green lane.
    - Inserted sprites still stack upward into empty lattice slots, but each new sprite takes the color assigned to its final height slot.
    - Marker bars are erased after the transfer.

    Key insight:
    - Color is no longer tied to lane identity. The input contains a visible slot->color ladder, so the solver must recover the
      lane lattice and reuse the color belonging to each destination height.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = get_bg(g)

    red_markers = _find_marker_bars(g, 2)
    green_markers = _find_marker_bars(g, 3)
    assert red_markers and green_markers, "expected both red and green marker bars to be present"

    marker_sets = {id(m): set(m) for m in red_markers + green_markers}
    lane_marker_pts = {c0: set() for c0 in _LANE_STARTS}
    for marker in red_markers + green_markers:
        _mr0, mc0, _mr1, _mc1 = bbox_coords(np.array(marker))
        lane_marker_pts[mc0].update(marker_sets[id(marker)])

    lane_data = {}
    all_lane_sprites = []
    for marker in red_markers + green_markers:
        _r0, c0, _r1, c1 = bbox_coords(np.array(marker))
        cols = range(c0, c1 + 1)
        sprites = _lane_sprites(g, cols, bg, marker_sets[id(marker)])
        lane_data[id(marker)] = {
            "marker": marker,
            "center": (c0 + c1) / 2.0,
            "cols": cols,
            "sprites": sprites,
        }
        if sprites:
            all_lane_sprites.append(sprites)

    for c0 in _LANE_STARTS:
        cols = range(c0, c0 + _SPRITE_W)
        sprites = _lane_sprites(g, cols, bg, lane_marker_pts[c0])
        if sprites:
            all_lane_sprites.append(sprites)

    bottom0, period, slot_to_color = _infer_height_palette(all_lane_sprites)

    green_state = {}
    for gm in green_markers:
        st = lane_data[id(gm)]
        sprites = st["sprites"]
        assert sprites, "expected the destination lane to contain visible sprites"
        tops = {int(sp["top"]) for sp in sprites}
        green_state[id(gm)] = {
            "center": st["center"],
            "occupied": set(tops),
            "bottom0": bottom0,
            "period": period,
        }

    out = g.copy()

    for rm in red_markers:
        st = lane_data[id(rm)]
        src_sprites = st["sprites"]
        if not src_sprites:
            continue

        for sp in src_sprites:
            for r, c in sp["pts"]:
                out[r, c] = 5

        rc = float(st["center"])
        best = min(green_state.values(), key=lambda gst: abs(float(gst["center"]) - rc))

        src_ref_top = int(src_sprites[0]["top"])
        src_ref_pts = src_sprites[0]["pts"]
        dc_shift = int(round(float(best["center"]) - rc))
        src_offsets = [(r - src_ref_top, c + dc_shift) for r, c in src_ref_pts]
        sprite_h = max(dr for dr, _c in src_offsets) + 1

        n_new = len(src_sprites)
        placed = 0
        k = 0
        while placed < n_new:
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
_MAX_LANE_TOTAL = 5
_LEGEND_COUNT = 5

# --- CFG: color palette ---
_BG_CHOICES = [0, 1, 6]
_GREY = 5
_HEIGHT_COLORS_BOTTOM_UP = [9, 2, 7, 4, 8]

# --- CFG: counts & limits ---
_MIN_SPRITES = 1
_MAX_SPRITES = 3
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


def _color_for_top(r0: int) -> int:
    slot = (_BOTTOM_TOP - int(r0)) // _PERIOD
    return _HEIGHT_COLORS_BOTTOM_UP[slot]


def _paint_gradient_lane(grid: np.ndarray, lane_idx: int, count: int, bg: int) -> None:
    c0 = _LANE_STARTS[lane_idx]
    for r0 in _lane_tops(count):
        paste(grid, _sprite_patch(_color_for_top(r0), bg), r0, c0, transparent=bg)


def _paint_solid_lane(grid: np.ndarray, lane_idx: int, color: int, count: int, bg: int) -> None:
    patch = _sprite_patch(color, bg)
    c0 = _LANE_STARTS[lane_idx]
    for r0 in _lane_tops(count):
        paste(grid, patch, r0, c0, transparent=bg)


def _feasible_roles(counts: list[int], legend_lane: int) -> dict[int, list[tuple[tuple[int, ...], int]]]:
    grouped: dict[int, list[tuple[tuple[int, ...], int]]] = {1: [], 2: []}
    active = [i for i in range(_NUM_LANES) if i != legend_lane]
    for green in active:
        others = [i for i in active if i != green]
        for red_count in (1, 2):
            for reds in combinations(others, red_count):
                if counts[green] + sum(counts[i] for i in reds) <= _MAX_LANE_TOTAL:
                    grouped[red_count].append((reds, green))
    return grouped


def _gen_attempt() -> np.ndarray | None:
    rng = np.random.default_rng()
    bg = int(rng.choice(_BG_CHOICES))

    legend_lane = int(rng.integers(_NUM_LANES))
    input_counts = rng.integers(_MIN_SPRITES, _MAX_SPRITES + 1, size=_NUM_LANES).tolist()
    input_counts[legend_lane] = _LEGEND_COUNT

    feasible = _feasible_roles(input_counts, legend_lane)
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
        if lane_idx in red_set:
            _paint_solid_lane(out, lane_idx, _GREY, output_counts[lane_idx], bg)
        else:
            _paint_gradient_lane(out, lane_idx, output_counts[lane_idx], bg)

    inp = out.copy()
    for lane_idx in reds:
        _paint_gradient_lane(inp, lane_idx, input_counts[lane_idx], bg)

    green_c0 = _LANE_STARTS[green]
    for r0 in _lane_tops(output_counts[green])[:gain]:
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
