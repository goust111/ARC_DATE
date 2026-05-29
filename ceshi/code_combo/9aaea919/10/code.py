# Combo variant of 9aaea919 (index 10)
# combo_spec: D4/0+D17/0
# Name: 保持源色+旁观车道变灰

import numpy as np

from arc_utils import bbox_coords, components_mask, get_bg, paste
from itertools import combinations


# Design analysis for combo D4/0 + D17/0
# a) Concrete behavior and flow tracking
#    The geometry stays the same as the base task: four fixed 5-column lanes, repeated 3x5 plus sprites,
#    red 1x5 bars for sources, and one green 1x5 bar for the destination. The fusion changes two color
#    consequences at once. Example with lane roles [red, unmarked, green, red], colors
#    [maroon, yellow, blue, teal], and counts [2, 1, 1, 1]:
#    - original task output: [grey, yellow, blue+3 copied as blue, grey]
#    - D4-only output:       [grey, yellow, blue + {maroon, maroon, teal}]
#    - fused output here:    [grey, grey, blue + {maroon, maroon, teal}, grey]
#    So the red lanes still donate counts and become grey, but the target lane keeps each transferred
#    sprite in its source color, and every bystander lane that was not marked at all also turns grey.
# b) Complexity calibration
#    The puzzle still has only one destination lane and at most two source lanes, so the target stack
#    shows at most three visible colors: its own original color plus one or two source colors. That is
#    enough to make "color inheritance" visible without introducing many arbitrary color mappings. At the
#    same time, the unmarked-lane greying adds a global rule that is easy to see because 4 lanes with
#    1-2 red markers always leave at least one bystander lane.
# c) Hidden assumptions and combined changes
#    The base solver had two D4/D17-sensitive assumptions:
#    - It painted transferred sprites with the destination lane color.
#    - It greyed only red-marked lanes, leaving unmarked lanes untouched.
#    Both are false in the combo. The safe order is:
#    1. detect all lanes and grey every non-target lane in the output buffer,
#    2. process red lanes only for count transfer,
#    3. paint inserted copies with the current source lane color.
#    This avoids a conflict where D17 would grey a lane after D4 had already inserted preserved-color
#    copies somewhere else; only the target lane is exempt from global greying.
# d) Placement legality
#    No new objects are introduced. The generator still uses 1-3 input sprites per lane and enforces
#    counts[green] + sum(counts[red]) <= 7, so the destination stack stays on the period-4 lattice
#    within rows 1..25. Bystander lanes keep their original counts and positions; only their colors
#    change in the output, so D17 adds no new overlap mode.
# e) Visual observability
#    The generator keeps all four lane colors distinct and excludes grey from the input palette. Because
#    the green lane always starts with at least one sprite, every sample shows both signals:
#    - the target lane contains its old color plus at least one preserved source color,
#    - at least one unmarked lane changes from a vivid input color to grey in the output.
#    These are visible from 2-4 examples without relying on any hidden numeric ordering of color ids.
# f) Visual cleanliness
#    The only mixed-color region is the destination lane, where colors stay separated by full sprite
#    slots on the same vertical lattice. All non-target lanes become uniformly grey, which actually
#    reduces clutter. Marker bars are removed in the output, so no extra annotation competes with the
#    transferred stack.
# g) Change inferability and solver/generator contract
#    Humans can infer a structured rule: the green-marked lane is the only lane that stays colorful,
#    and its new upper sprites reuse the exact colors lost from the red lanes. The solver now assumes:
#    - every lane contains at least one sprite, so all four lane ranges can be recovered from components;
#    - each lane is monochrome in the input, so a single source color per transferred group is well-defined;
#    - the green lane has at least one pre-existing sprite, so its lattice period is inferable.
#    The generator explicitly guarantees all three conditions.


def _find_marker_bars(g: np.ndarray, marker_color: int) -> list[list[tuple[int, int]]]:
    """Return thin 4-connected marker components of the requested color."""
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


def _sort_markers_left_to_right(markers: list[list[tuple[int, int]]]) -> list[list[tuple[int, int]]]:
    return sorted(markers, key=lambda pts: bbox_coords(np.array(pts))[1])


def _single_color_components_in_cols(g: np.ndarray, cols: range, bg: int) -> list[list[tuple[int, int]]]:
    h, w = g.shape
    colmask = np.zeros((h, w), dtype=bool)
    colmask[:, cols] = True
    mask = (g != bg) & colmask
    comps = [[(int(r), int(c)) for r, c in pts] for pts in components_mask(mask, connectivity=4)]
    out = []
    for pts in comps:
        if len({int(g[r, c]) for r, c in pts}) == 1:
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


def _lane_info_for_marker(
    g: np.ndarray,
    marker_pts: list[tuple[int, int]],
    bg: int,
) -> dict[str, object]:
    """Return lane geometry plus sorted sprite entries for the lane under a marker."""
    _r0, c0, _r1, c1 = bbox_coords(np.array(marker_pts))
    sprites = _lane_sprites_in_cols(g, range(c0, c1 + 1), bg, ignored=set(marker_pts))
    entries = []
    for pts in sprites:
        rmin = min(r for r, _ in pts)
        cmin = min(c for _, c in pts)
        color = int(g[pts[0][0], pts[0][1]])
        entries.append((rmin, cmin, color, pts))
    entries.sort()
    return {
        "c0": c0,
        "c1": c1,
        "center": (c0 + c1) / 2.0,
        "sprites": entries,
    }


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
    Move red-lane sprite counts into the green lane, while keeping transfer colors and greying all bystanders.

    Input structure:
    - The background is the most common color. Thin red (2) and green (3) 1x5 bars sit beneath 5-column sprite lanes.
    - Each lane contains repeated identical monochrome plus sprites on a shared vertical period-4 lattice.

    Rule:
    - Every lane except the green-marked destination ends grey in the output, so both red source lanes and unmarked bystander lanes lose their original colors.
    - Only red-marked lanes donate sprite counts to the nearest green lane; the inserted copies stack upward into empty slots but keep the source lane color instead of adopting the target color.
    - All marker bars are erased at the end.

    Key insight:
    - The green marker identifies the only lane allowed to stay colorful; its old sprites keep the destination color, while its newly added upper sprites expose exactly which colors were taken from the red-marked lanes.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = get_bg(g)

    red_markers = _sort_markers_left_to_right(_find_marker_bars(g, 2))
    green_markers = _sort_markers_left_to_right(_find_marker_bars(g, 3))
    assert red_markers and green_markers, "expected both red and green marker bars to be present"

    marker_cells = {pt for bar in red_markers + green_markers for pt in bar}
    lane_ranges = _find_lane_ranges(g, bg, marker_cells)
    assert lane_ranges, "expected at least one sprite lane"

    green_state = {}
    target_ranges = set()
    for gm in green_markers:
        lane = _lane_info_for_marker(g, gm, bg)
        dst_sprites = lane["sprites"]
        if not dst_sprites:
            continue

        target_ranges.add((lane["c0"], lane["c1"]))
        tops = sorted({rmin for rmin, _cmin, _color, _pts in dst_sprites})
        if len(tops) >= 2:
            diffs = [b - a for a, b in zip(tops, tops[1:]) if b - a > 0]
            period = int(np.gcd.reduce(diffs)) if diffs else 4
        else:
            r0, c0, r1, c1 = bbox_coords(np.array(dst_sprites[0][3]))
            period = (r1 - r0 + 1) + 1

        green_state[id(gm)] = {
            "center": lane["center"],
            "period": period,
            "occupied": set(tops),
            "bottom0": max(tops),
        }

    assert green_state, "expected at least one green lane with valid sprites"

    out = g.copy()

    for c0, c1 in lane_ranges:
        if (c0, c1) in target_ranges:
            continue
        lane_sprites = _lane_sprites_in_cols(g, range(c0, c1 + 1), bg)
        for pts in lane_sprites:
            for r, c in pts:
                out[r, c] = 5

    for rm in red_markers:
        lane = _lane_info_for_marker(g, rm, bg)
        src_sprites = lane["sprites"]
        if not src_sprites:
            continue

        rc = lane["center"]
        best = min(green_state.values(), key=lambda st: abs(st["center"] - rc))

        src_ref_r0, _src_ref_c0, src_color, src_ref_pts = src_sprites[0]
        dc_shift = int(round(best["center"] - rc))
        src_offsets = [(r - src_ref_r0, c + dc_shift) for r, c in src_ref_pts]
        sprite_h = max(dr for dr, _c in src_offsets) + 1

        placed = 0
        k = 0
        while placed < len(src_sprites):
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

    for bar in red_markers + green_markers:
        for r, c in bar:
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
_BRANCH_CYCLE = (1, 2)
_next_branch_idx = 0


def _sprite_patch(color: int, bg: int) -> np.ndarray:
    """Return the 3x5 plus/diamond sprite with transparent corners."""
    patch = np.full((_SPRITE_H, _SPRITE_W), bg, dtype=int)
    patch[0, 1:4] = color
    patch[1, :] = color
    patch[2, 1:4] = color
    return patch


def _lane_tops(count: int) -> list[int]:
    """Return sprite top rows for a lane, ordered top-to-bottom."""
    return [_BOTTOM_TOP - _PERIOD * k for k in range(count - 1, -1, -1)]


def _paint_lane(grid: np.ndarray, lane_idx: int, color: int, count: int, bg: int) -> None:
    patch = _sprite_patch(color, bg)
    c0 = _LANE_STARTS[lane_idx]
    for r0 in _lane_tops(count):
        paste(grid, patch, r0, c0, transparent=bg)


def _feasible_roles(counts: list[int]) -> dict[int, list[tuple[tuple[int, ...], int]]]:
    grouped: dict[int, list[tuple[tuple[int, ...], int]]] = {1: [], 2: []}
    for green in range(_NUM_LANES):
        others = [i for i in range(_NUM_LANES) if i != green]
        for red_count in (1, 2):
            for reds in combinations(others, red_count):
                if counts[green] + sum(counts[i] for i in reds) <= _MAX_LANE_TOTAL:
                    grouped[red_count].append((tuple(sorted(reds)), green))
    return grouped


def _gen_attempt(preferred_red_count: int | None = None, strict_preference: bool = False) -> np.ndarray | None:
    rng = np.random.default_rng()

    bg = int(rng.choice(_BG_CHOICES))
    sprite_palette = [c for c in range(10) if c not in (bg, _GREY)]
    lane_colors = rng.choice(sprite_palette, size=_NUM_LANES, replace=False).tolist()
    input_counts = rng.integers(_MIN_SPRITES, _MAX_SPRITES + 1, size=_NUM_LANES).tolist()

    feasible = _feasible_roles(input_counts)
    feasible_red_counts = [rc for rc in (1, 2) if feasible[rc]]
    if not feasible_red_counts:
        return None

    if preferred_red_count in feasible_red_counts:
        red_count = preferred_red_count
    elif strict_preference:
        return None
    else:
        red_count = int(rng.choice(feasible_red_counts))
    reds, green = feasible[red_count][int(rng.integers(len(feasible[red_count])))]
    red_set = set(reds)
    unmarked = [lane_idx for lane_idx in range(_NUM_LANES) if lane_idx not in red_set and lane_idx != green]
    if not unmarked:
        return None

    out = np.full((_H, _W), bg, dtype=int)
    for lane_idx in range(_NUM_LANES):
        base_color = lane_colors[lane_idx] if lane_idx == green else _GREY
        _paint_lane(out, lane_idx, base_color, input_counts[lane_idx], bg)

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
        if lane_idx == green:
            continue
        _paint_lane(inp, lane_idx, lane_colors[lane_idx], input_counts[lane_idx], bg)

    for _lane_idx, r0 in transferred:
        inp[r0 : r0 + _SPRITE_H, green_c0 : green_c0 + _SPRITE_W] = bg
    inp[_MARKER_ROW, green_c0 : green_c0 + _SPRITE_W] = 3

    for lane_idx in reds:
        c0 = _LANE_STARTS[lane_idx]
        inp[_MARKER_ROW, c0 : c0 + _SPRITE_W] = 2

    target_colors = {
        int(out[r, c])
        for r in range(_H)
        for c in range(green_c0, green_c0 + _SPRITE_W)
        if int(out[r, c]) != bg
    }
    expected_target_colors = {lane_colors[green], *(lane_colors[lane_idx] for lane_idx in reds)}
    if target_colors != expected_target_colors:
        return None

    for lane_idx in unmarked:
        c0 = _LANE_STARTS[lane_idx]
        lane_out_colors = {
            int(out[r, c])
            for r in range(_H)
            for c in range(c0, c0 + _SPRITE_W)
            if int(out[r, c]) != bg
        }
        if lane_out_colors != {_GREY}:
            return None

    if not np.array_equal(main(inp), out):
        return None
    return inp


def generate_input() -> np.ndarray:
    global _next_branch_idx
    preferred_red_count = _BRANCH_CYCLE[_next_branch_idx]
    for _ in range(_GEN_RETRIES):
        grid = _gen_attempt(preferred_red_count=preferred_red_count, strict_preference=True)
        if grid is not None:
            _next_branch_idx = (_next_branch_idx + 1) % len(_BRANCH_CYCLE)
            return grid
    for _ in range(_GEN_RETRIES):
        grid = _gen_attempt(preferred_red_count=preferred_red_count, strict_preference=False)
        if grid is not None:
            _next_branch_idx = (_next_branch_idx + 1) % len(_BRANCH_CYCLE)
            return grid
    raise RuntimeError(f"failed to generate a valid input after {_GEN_RETRIES} attempts")
