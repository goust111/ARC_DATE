# Combo variant of 9aaea919 (index 1)
# combo_spec: D1/0+D5/2
# Name: 颜色匹配+单像素标记

import numpy as np

from itertools import combinations

from arc_utils import bbox_coords, components_mask, get_bg, paste


# Design analysis
# combo_variant_spec:
#   summary:
#     - Preserve the original four fixed lanes and the same sprite-transfer/grey-out rule.
#     - Replace bottom 1x5 red/green bars with 1x1 edge cues whose colors match lane sprite colors.
#     - Fuse D1/0 and D5/2 by making the role cue both color-based and tiny: a single pixel on the left edge
#       means "the lane with this color is a source", and a single pixel on the right edge means
#       "the lane with this color is the destination".
#   a_concretization_and_behavior_trace:
#     original:
#       - Input had 1-2 red bars and 1 green bar on row 29 directly under marked lanes.
#       - Solver could read lane roles from bar color + bar position with almost no intermediate inference.
#     fused:
#       - Input has 1-2 source cue pixels on col 0 and exactly 1 target cue pixel on col 29.
#       - Each cue pixel color equals one lane's sprite color; lane colors remain pairwise distinct.
#       - Example: lane colors [8, 2, 4, 7], left cues {row 11: 7, row 28: 4}, right cue {row 17: 8}.
#         Then color 7 and 4 identify the two source lanes, while color 8 identifies the target lane.
#       - Transfer itself is unchanged: source sprites become grey, and the same number of copies are
#         stacked upward into the target lane in the target lane color.
#     key_difference:
#       - The old task exposed roles directly through special marker colors.
#       - The fused task requires a two-step read: edge side gives the role, cue color identifies the lane.
#       - D5/2 is realized by shrinking every cue to one pixel, so the same logical signal is visually subtler.
#   b_complexity_calibration:
#     - The cue vocabulary stays small: only 1-2 left cues and 1 right cue, all 1x1.
#     - Lane colors are forced distinct, so each cue color maps to exactly one lane and no ambiguity appears.
#     - To prevent regression toward the original red/green semantics, the generator forbids
#       destination cue color 3 and source cue color 2. This keeps the cue colors structurally relevant.
#   c_implicit_assumption_audit_on_original_main:
#     - `_find_marker_bars(...)` assumed colors 2/3 formed thin bars; this is invalid and removed.
#     - Using a marker bbox to recover lane columns is no longer possible; the fused solver must rely on
#       the fixed lane slots from the original spec.
#     - Stripping marker pixels out of lane components is no longer needed because cues live on the far edges.
#     - "Nearest green lane" was an artifact of marker-local geometry; the fused rule has exactly one
#       right-edge cue, so the destination lane should be identified explicitly by color matching.
#     - Marker cleanup now means clearing edge cue pixels, not clearing row-29 bars.
#   d_placement_legality:
#     - Lanes stay in columns [3..7], [9..13], [15..19], [21..25], so edge cues at cols 0 and 29 never touch sprites.
#     - Cue rows are chosen from a small fixed set and do not repeat across left/right, avoiding accidental cue clusters.
#     - The output-first generator still enforces `destination_count + transferred_count <= 7`, so all newly
#       stacked sprites remain within the 30x30 grid.
#   e_visual_observability:
#     - Across 2-4 examples, a human can notice that only the edge pixels disappear, while lane motion follows
#       the colors of those edge pixels rather than their numeric ids.
#     - Because source cue colors are never 2 and the destination cue color is never 3, the observer is nudged
#       away from the old "red means source / green means target" shortcut.
#   f_visual_cleanliness:
#     - The main scene remains the same clean four-lane layout.
#     - Cues occupy isolated border columns, so they are easy to spot yet do not overlap the lane bodies.
#   g_change_inferability:
#     - The only new color relation is a direct equality: cue color == corresponding lane color.
#     - This relation is structurally supported by fixed lane partitions and by source/target side roles,
#       making the color mapping inferable from a few examples.
#   post_change_main_assumptions_checked_against_generator:
#     - main assumes every lane contains at least one monochrome sprite component; generator guarantees 1-3 sprites.
#     - main assumes lane colors are all distinct; generator samples them without replacement.
#     - main assumes left edge has 1-2 cue pixels, right edge has exactly 1, and every cue color matches one lane.
#       generator constructs cues exactly that way.
#     - main assumes source lanes differ from the destination lane; generator chooses disjoint role assignments.


_H, _W = 30, 30
_NUM_LANES = 4
_LANE_STARTS = [3, 9, 15, 21]
_SPRITE_H, _SPRITE_W = 3, 5
_BOTTOM_TOP = 25
_PERIOD = 4
_BG_CHOICES = [0, 1, 4, 6, 7, 8, 9]
_GREY = 5
_MIN_SPRITES = 1
_MAX_SPRITES = 3
_MAX_LANE_TOTAL = 7
_GEN_RETRIES = 300
_LEFT_CUE_COL = 0
_RIGHT_CUE_COL = _W - 1
_CUE_ROWS = [4, 10, 16, 22, 28]


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


def _lane_state(g: np.ndarray, lane_idx: int, bg: int) -> dict | None:
    c0 = _LANE_STARTS[lane_idx]
    comps = _single_color_components_in_cols(g, range(c0, c0 + _SPRITE_W), bg)
    if not comps:
        return None

    sprite_sz = max(len(pts) for pts in comps)
    sprites = [pts for pts in comps if len(pts) == sprite_sz]
    if not sprites:
        return None

    sprites_sorted = []
    for pts in sprites:
        rmin = min(r for r, _ in pts)
        cmin = min(c for _, c in pts)
        sprites_sorted.append((rmin, cmin, pts))
    sprites_sorted.sort()

    color = int(g[sprites_sorted[0][2][0][0], sprites_sorted[0][2][0][1]])
    tops = sorted({rmin for rmin, _cmin, _pts in sprites_sorted})
    if len(tops) >= 2:
        diffs = [b - a for a, b in zip(tops, tops[1:]) if b - a > 0]
        period = int(np.gcd.reduce(diffs)) if diffs else _PERIOD
    else:
        r0, _c0, r1, _c1 = bbox_coords(np.array(sprites_sorted[0][2]))
        period = (r1 - r0 + 1) + 1

    center = c0 + (_SPRITE_W - 1) / 2.0
    return {
        "lane_idx": lane_idx,
        "center": center,
        "color": color,
        "period": period,
        "bottom0": max(tops),
        "occupied": set(tops),
        "sprites": [pts for _rmin, _cmin, pts in sprites_sorted],
    }


def _edge_cues(g: np.ndarray, bg: int) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    h, w = g.shape
    left = [(int(r), int(g[r, _LEFT_CUE_COL])) for r in range(h) if int(g[r, _LEFT_CUE_COL]) != bg]
    right = [(int(r), int(g[r, w - 1])) for r in range(h) if int(g[r, w - 1]) != bg]
    return left, right


def main(grid: np.ndarray) -> np.ndarray:
    """
    Move sprites from color-matched source lanes into a color-matched target lane using tiny edge cues.

    Input structure:
    - The background is the dominant color. Four fixed vertical lanes contain repeated monochrome plus sprites on the usual period-4 stack.
    - Role markers are no longer bottom bars. Instead, 1-2 single cue pixels appear on the left edge and 1 single cue pixel appears on the right edge.
    - Each cue pixel's color equals the sprite color of exactly one lane.

    Rule:
    - Interpret left-edge cue colors as source-lane selectors and the right-edge cue color as the destination-lane selector.
    - Match cue colors to the uniquely colored lanes, turn every selected source sprite grey (5), and add the same number of copies into the selected destination lane.
    - New destination sprites use the destination lane color, occupy empty lattice slots from the bottom upward, and all cue pixels are erased.

    Key insight:
    - The side of a 1x1 cue tells you the role, while the cue's color tells you which lane plays that role.
    """
    g = np.asarray(grid, dtype=int)
    h, _w = g.shape
    bg = get_bg(g)

    left_cues, right_cues = _edge_cues(g, bg)
    left_colors = [color for _row, color in left_cues]
    right_colors = [color for _row, color in right_cues]
    assert 1 <= len(left_colors) <= 2, "expected 1-2 left-edge source cues"
    assert len(right_colors) == 1, "expected exactly one right-edge target cue"
    assert len(left_colors) == len(set(left_colors)), "expected distinct source cue colors"

    lane_states = []
    for lane_idx in range(_NUM_LANES):
        st = _lane_state(g, lane_idx, bg)
        assert st is not None, "expected every lane to contain valid monochrome sprites"
        lane_states.append(st)

    lanes_by_color = {}
    for st in lane_states:
        assert st["color"] not in lanes_by_color, "expected lane colors to be unique"
        lanes_by_color[st["color"]] = st

    dst_color = right_colors[0]
    assert dst_color in lanes_by_color, "target cue color must match a lane"
    dst = lanes_by_color[dst_color]

    src_states = []
    for color in left_colors:
        assert color in lanes_by_color, "source cue color must match a lane"
        src_states.append(lanes_by_color[color])
    assert all(src["lane_idx"] != dst["lane_idx"] for src in src_states), "source and target lanes must differ"

    out = g.copy()

    for src in src_states:
        src_sprites = src["sprites"]
        for pts in src_sprites:
            for r, c in pts:
                out[r, c] = _GREY

        rc = src["center"]
        src_ref_pts = src_sprites[0]
        src_ref_r0 = min(r for r, _ in src_ref_pts)
        dc_shift = int(round(dst["center"] - rc))
        src_offsets = [(r - src_ref_r0, c + dc_shift) for r, c in src_ref_pts]
        sprite_h = max(dr for dr, _c in src_offsets) + 1

        placed = 0
        k = 0
        while placed < len(src_sprites):
            new_top = dst["bottom0"] - k * dst["period"]
            k += 1
            if new_top < 0 or new_top + sprite_h > h:
                if new_top < 0:
                    break
                continue
            if new_top in dst["occupied"]:
                continue

            ok = True
            for dr, c in src_offsets:
                rr = new_top + dr
                if not (0 <= rr < h and 0 <= c < _W):
                    ok = False
                    break
                if int(out[rr, c]) != bg:
                    ok = False
                    break
            if not ok:
                continue

            for dr, c in src_offsets:
                out[new_top + dr, c] = dst["color"]
            dst["occupied"].add(new_top)
            placed += 1

    for r, _color in left_cues:
        out[r, _LEFT_CUE_COL] = bg
    for r, _color in right_cues:
        out[r, _RIGHT_CUE_COL] = bg

    return out


def _sprite_patch(color: int, bg: int) -> np.ndarray:
    patch = np.full((_SPRITE_H, _SPRITE_W), bg, dtype=int)
    patch[0, 1:4] = color
    patch[1, :] = color
    patch[2, 1:4] = color
    return patch


def _lane_tops(count: int) -> list[int]:
    return [_BOTTOM_TOP - _PERIOD * k for k in range(count - 1, -1, -1)]


def _paint_lane(grid: np.ndarray, lane_idx: int, color: int, count: int, bg: int) -> None:
    patch = _sprite_patch(color, bg)
    c0 = _LANE_STARTS[lane_idx]
    for r0 in _lane_tops(count):
        paste(grid, patch, r0, c0, transparent=bg)


def _feasible_roles(counts: list[int], lane_colors: list[int]) -> dict[int, list[tuple[tuple[int, ...], int]]]:
    feasible = {1: [], 2: []}
    for green in range(_NUM_LANES):
        if lane_colors[green] == 3:
            continue
        others = [i for i in range(_NUM_LANES) if i != green and lane_colors[i] != 2]
        for red_count in (1, 2):
            for reds in combinations(others, red_count):
                if counts[green] + sum(counts[i] for i in reds) <= _MAX_LANE_TOTAL:
                    feasible[red_count].append((tuple(int(i) for i in reds), int(green)))
    return feasible


def _place_cues(
    inp: np.ndarray,
    lane_colors: list[int],
    reds: tuple[int, ...],
    green: int,
    rng: np.random.Generator,
) -> None:
    left_rows = rng.choice(_CUE_ROWS, size=len(reds), replace=False).tolist()
    remaining_rows = [r for r in _CUE_ROWS if r not in left_rows]
    right_row = int(rng.choice(remaining_rows))

    for row, lane_idx in zip(left_rows, reds):
        inp[int(row), _LEFT_CUE_COL] = lane_colors[lane_idx]
    inp[right_row, _RIGHT_CUE_COL] = lane_colors[green]


def _gen_attempt() -> np.ndarray | None:
    rng = np.random.default_rng()

    bg = int(rng.choice(_BG_CHOICES))
    sprite_palette = [c for c in range(10) if c not in (bg, _GREY)]
    lane_colors = rng.choice(sprite_palette, size=_NUM_LANES, replace=False).tolist()
    input_counts = rng.integers(_MIN_SPRITES, _MAX_SPRITES + 1, size=_NUM_LANES).tolist()

    feasible = _feasible_roles(input_counts, lane_colors)
    branch_sizes = [red_count for red_count, roles in feasible.items() if roles]
    if not branch_sizes:
        return None

    red_count = int(rng.choice(branch_sizes))
    roles = feasible[red_count]
    reds, green = roles[int(rng.integers(len(roles)))]
    red_set = set(reds)
    gain = sum(input_counts[i] for i in reds)

    output_counts = input_counts[:]
    output_counts[green] += gain
    out = np.full((_H, _W), bg, dtype=int)
    for lane_idx in range(_NUM_LANES):
        color = _GREY if lane_idx in red_set else lane_colors[lane_idx]
        _paint_lane(out, lane_idx, color, output_counts[lane_idx], bg)

    inp = out.copy()
    for lane_idx in reds:
        _paint_lane(inp, lane_idx, lane_colors[lane_idx], input_counts[lane_idx], bg)

    green_c0 = _LANE_STARTS[green]
    for r0 in _lane_tops(output_counts[green])[:gain]:
        inp[r0 : r0 + _SPRITE_H, green_c0 : green_c0 + _SPRITE_W] = bg

    _place_cues(inp, lane_colors, reds, green, rng)

    if not np.array_equal(main(inp), out):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(_GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {_GEN_RETRIES} attempts")
