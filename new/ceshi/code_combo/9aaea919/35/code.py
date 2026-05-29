# Combo variant of 9aaea919 (index 35)
# combo_spec: D1/0+D15/0+D18/1
# Name: 颜色匹配+同色车道+隐形车道

import numpy as np

from arc_utils import bbox_coords, components_mask, get_bg, paste


# Design analysis for the D1/0 + D15/0 + D18/1 combo
# ---------------------------------------------------
# spec:
#   grid: unchanged 30x30 canvas with 4 fixed 5-column lanes and the original 3x5
#     plus/diamond sprite on a period-4 vertical lattice.
#   fusion summary:
#     - D1 removes bottom red/green bars and replaces them with edge cue pixels:
#       left edge = source role, right edge = target role.
#     - D15 allows two lanes to share a sprite color.
#     - D18 makes one unmarked lane invisible by setting its sprite color equal to bg.
#   fused resolution:
#     - The hidden lane is always the only unmarked lane, so invisibility never blocks a
#       role-carrying lane.
#     - The two source lanes are forced to share one visible color.
#     - Two same-colored left-edge cues mark that shared source color; together they
#       select both source lanes.
#     - The target lane has a different visible color and is selected by one right-edge cue.
#
# a) Concrete behavior and trace:
#   Example conceptual lane colors/counts:
#     lane 0 = hidden/unmarked, color bg, count 2
#     lane 1 = source, color 8, count 1
#     lane 2 = source, color 8, count 3
#     lane 3 = target, color 4, count 2
#   Input view:
#     - lane 0 looks empty because bg-colored sprites vanish.
#     - lanes 1 and 2 visibly share color 8.
#     - the left edge shows two cue pixels of color 8 at unrelated rows.
#     - the right edge shows one cue pixel of color 4.
#   Output view:
#     - both color-8 lanes turn grey.
#     - lane 3 gains 1+3 new sprites of color 4, stacked upward.
#     - all edge cues disappear.
#   Difference from the base task:
#     - source/target are no longer given by fixed semantic colors 2/3.
#     - color matching no longer maps one cue to one lane; one source color can select a
#       duplicated pair of lanes.
#     - one lane slot stays present geometrically but is visually blank because its
#       sprite color equals the background.
#
# b) Complexity calibration:
#   Every generated sample forces all three phenomena at once:
#     - exactly one hidden lane,
#     - exactly one visible duplicated-color pair,
#     - two left cues with the duplicated source color.
#   This avoids degenerate samples where D15 or D18 are absent, while still keeping only
#   two visible role colors on the board so the pattern remains learnable from a few
#   examples.
#
# c) Implicit-assumption audit:
#   Original assumptions that break:
#     - `_find_marker_bars(...)` and marker-colored lane lookup are invalid because there
#       are no red/green bars anymore.
#     - "every lane is visible via `g != bg`" is false because one lane is hidden.
#     - "cue color maps to exactly one lane" is false because the two source lanes share
#       one color by construction.
#   New solver-side assumptions:
#     - only columns in `_LANE_STARTS` can contain lane sprites;
#       `generate_input()` still uses the fixed lane scaffold.
#     - all non-bg pixels on col 0 / col 29 are cues; the generator keeps the border clean.
#     - the target cue color matches exactly one visible lane; the generator assigns a
#       unique target color.
#     - each source cue color matches one or more visible lanes, and the union of those
#       matches is the full source set; the generator makes both source lanes share one
#       color and duplicates that cue on the left edge.
#
# d) Placement legality:
#   Lane geometry, sprite size, and period stay unchanged. The hidden lane is painted with
#   `transparent=bg`, so its conceptual sprites occupy no visible cells and cannot collide.
#   Edge cues live in col 0 / col 29, while lane pixels stay in cols 3..25, so cues never
#   touch the lane bodies. `_MAX_LANE_TOTAL` still bounds the post-transfer target stack.
#
# e) Visual observability:
#   Across examples a human can see:
#     - one of the four regular lane slots is always blank,
#     - the two non-target visible lanes share a color,
#     - the left edge repeats that shared color twice,
#     - both same-colored lanes grey out together,
#     - the right-cued color marks the receiving lane.
#   No numeric color ordering or hidden metadata is required.
#
# f) Visual cleanliness:
#   Only three small edge pixels are added, all on the outer border. The middle of the
#   board remains the same tidy four-lane scaffold, with only one blank lane slot as the
#   extra D18 signal.
#
# g) Change inferability:
#   The only color relations to infer are structural and repeated:
#     - left-edge cue color = source-lane color, possibly shared by two lanes,
#     - right-edge cue color = unique target-lane color,
#     - source sprites become grey,
#     - transferred sprites keep the target lane color.
#   The hidden lane is the only lane whose "color" equals bg, so its disappearance is
#   explainable from the fixed lane scaffold.
#
# main()/generator contract review:
#   - `main()` expects exactly one visible target lane matched by the right cue; the
#     generator enforces a unique visible target color.
#   - `main()` allows a source cue color to map to multiple lanes; the generator uses that
#     to make both source lanes selectable together.
#   - `main()` ignores invisible lanes because they have no visible components; the
#     generator guarantees the hidden lane is never targeted by a cue.


_H, _W = 30, 30
_NUM_LANES = 4
_LANE_STARTS = [3, 9, 15, 21]
_SPRITE_H, _SPRITE_W = 3, 5
_BOTTOM_TOP = 25
_PERIOD = 4
_BG_CHOICES = [0, 1, 4, 6, 7, 8, 9]
_GREY = 5
_GEN_RETRIES = 300
_MIN_SPRITES = 1
_MAX_SPRITES = 3
_MAX_LANE_TOTAL = 7
_LEFT_CUE_COL = 0
_RIGHT_CUE_COL = _W - 1
_CUE_ROWS = [4, 10, 16, 22, 28]
_BLOCKED_VISIBLE_COLORS = {2, 3, _GREY}


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
    if sprite_sz <= 1:
        return None
    sprites = [pts for pts in comps if len(pts) == sprite_sz]
    if not sprites:
        return None

    sprites_sorted = []
    for pts in sprites:
        rmin = min(r for r, _ in pts)
        cmin = min(c for _, c in pts)
        sprites_sorted.append((rmin, cmin, pts))
    sprites_sorted.sort()

    tops = sorted({rmin for rmin, _cmin, _pts in sprites_sorted})
    if len(tops) >= 2:
        diffs = [b - a for a, b in zip(tops, tops[1:]) if b - a > 0]
        period = int(np.gcd.reduce(diffs)) if diffs else _PERIOD
    else:
        r0, _c0, r1, _c1 = bbox_coords(np.array(sprites_sorted[0][2]))
        period = (r1 - r0 + 1) + 1

    color = int(g[sprites_sorted[0][2][0][0], sprites_sorted[0][2][0][1]])
    center = c0 + (_SPRITE_W - 1) / 2.0
    return {
        "lane_idx": lane_idx,
        "center": center,
        "color": color,
        "period": period,
        "occupied": set(tops),
        "bottom0": max(tops),
        "sprites": [pts for _rmin, _cmin, pts in sprites_sorted],
    }


def _edge_cues(g: np.ndarray, bg: int) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    h, w = g.shape
    left = [(int(r), int(g[r, _LEFT_CUE_COL])) for r in range(h) if int(g[r, _LEFT_CUE_COL]) != bg]
    right = [(int(r), int(g[r, w - 1])) for r in range(h) if int(g[r, w - 1]) != bg]
    return left, right


def main(grid: np.ndarray) -> np.ndarray:
    """
    Move cue-selected source sprites into a cue-selected target lane, even when two source lanes share a color and a fourth lane is invisible.

    Input structure:
    - The board keeps the same four fixed vertical lane slots with repeated monochrome plus sprites on a period-4 lattice.
    - Bottom bars are gone. Instead, exactly two same-colored cue pixels on the left edge mark the shared source color, and one cue pixel on the right edge marks the target color.
    - One unmarked lane may be invisible because its sprite color equals the background, and two visible lanes may share the same source color.

    Rule:
    - Match the right-edge cue color to the unique visible target lane.
    - Match the shared left-edge cue color to every visible lane that currently uses that color; all matched lanes are sources, so their sprites turn grey (5).
    - Add the same total number of source sprites into the target lane, filling empty lattice slots upward from the bottom and recoloring the inserted sprites to the target lane color. All edge cues are removed.

    Key insight:
    - Cue side gives the role, but cue color can now name a set of lanes rather than exactly one lane: duplicated visible colors mean one left cue color may simultaneously select both source lanes, while the hidden background-colored lane contributes no visible match at all.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = get_bg(g)

    left_cues, right_cues = _edge_cues(g, bg)
    left_colors = [color for _row, color in left_cues]
    right_colors = [color for _row, color in right_cues]
    assert len(left_colors) == 2, "expected exactly two left-edge source cues"
    assert len(set(left_colors)) == 1, "left-edge source cues must share one color"
    assert len(right_colors) == 1, "expected exactly one right-edge target cue"
    assert len(set(right_colors)) == 1, "right-edge cues must select one target color"

    visible_lanes = []
    for lane_idx in range(_NUM_LANES):
        st = _lane_state(g, lane_idx, bg)
        if st is not None:
            visible_lanes.append(st)
    assert len(visible_lanes) == 3, "expected exactly three visible lanes and one hidden lane"

    lanes_by_color: dict[int, list[dict]] = {}
    for st in visible_lanes:
        lanes_by_color.setdefault(st["color"], []).append(st)

    dst_matches = lanes_by_color.get(right_colors[0], [])
    assert len(dst_matches) == 1, "target cue color must match exactly one visible lane"
    dst = dst_matches[0]

    src_by_lane: dict[int, dict] = {}
    for color in left_colors:
        src_matches = lanes_by_color.get(color, [])
        assert src_matches, "source cue color must match at least one visible lane"
        for st in src_matches:
            assert st["lane_idx"] != dst["lane_idx"], "source and target lanes must differ"
            src_by_lane[st["lane_idx"]] = st
    src_states = [src_by_lane[idx] for idx in sorted(src_by_lane)]
    assert src_states, "expected at least one source lane"

    out = g.copy()

    for src in src_states:
        for pts in src["sprites"]:
            for r, c in pts:
                out[r, c] = _GREY

        src_ref_pts = src["sprites"][0]
        src_ref_r0 = min(r for r, _ in src_ref_pts)
        dc_shift = int(round(dst["center"] - src["center"]))
        src_offsets = [(r - src_ref_r0, c + dc_shift) for r, c in src_ref_pts]
        sprite_h = max(dr for dr, _c in src_offsets) + 1

        placed = 0
        n_new = len(src["sprites"])
        k = 0
        while placed < n_new:
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
                if not (0 <= rr < h and 0 <= c < w):
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


def _feasible_role_assignments(counts: list[int]) -> list[tuple[int, tuple[int, int], int]]:
    out = []
    for hidden in range(_NUM_LANES):
        visible = [i for i in range(_NUM_LANES) if i != hidden]
        for target in visible:
            sources = tuple(i for i in visible if i != target)
            if len(sources) != 2:
                continue
            if counts[target] + counts[sources[0]] + counts[sources[1]] <= _MAX_LANE_TOTAL:
                out.append((hidden, sources, target))
    return out


def _sample_lane_colors(
    rng: np.random.Generator, bg: int, hidden: int, sources: tuple[int, int], target: int
) -> list[int]:
    del hidden  # hidden lane color is always bg
    palette = [c for c in range(10) if c not in _BLOCKED_VISIBLE_COLORS and c != bg]
    source_color, target_color = [int(x) for x in rng.choice(palette, size=2, replace=False)]

    lane_colors = [bg] * _NUM_LANES
    lane_colors[sources[0]] = source_color
    lane_colors[sources[1]] = source_color
    lane_colors[target] = target_color
    return lane_colors


def _place_cues(
    inp: np.ndarray,
    source_color: int,
    target_color: int,
    rng: np.random.Generator,
) -> None:
    rows = rng.choice(_CUE_ROWS, size=3, replace=False).tolist()
    left_rows = sorted(int(r) for r in rows[:2])
    right_row = int(rows[2])

    for row in left_rows:
        inp[row, _LEFT_CUE_COL] = source_color
    inp[right_row, _RIGHT_CUE_COL] = target_color


def _gen_attempt() -> np.ndarray | None:
    rng = np.random.default_rng()

    bg = int(rng.choice(_BG_CHOICES))
    input_counts = rng.integers(_MIN_SPRITES, _MAX_SPRITES + 1, size=_NUM_LANES).tolist()

    feasible = _feasible_role_assignments(input_counts)
    if not feasible:
        return None
    hidden, sources, target = feasible[int(rng.integers(len(feasible)))]
    source_set = set(sources)
    lane_colors = _sample_lane_colors(rng, bg, hidden, sources, target)
    gain = sum(input_counts[i] for i in sources)

    output_counts = input_counts[:]
    output_counts[target] += gain
    out = np.full((_H, _W), bg, dtype=int)
    for lane_idx in range(_NUM_LANES):
        color = _GREY if lane_idx in source_set else lane_colors[lane_idx]
        _paint_lane(out, lane_idx, color, output_counts[lane_idx], bg)

    inp = out.copy()
    for lane_idx in sources:
        _paint_lane(inp, lane_idx, lane_colors[lane_idx], input_counts[lane_idx], bg)

    target_c0 = _LANE_STARTS[target]
    for r0 in _lane_tops(output_counts[target])[:gain]:
        inp[r0 : r0 + _SPRITE_H, target_c0 : target_c0 + _SPRITE_W] = bg

    _place_cues(inp, lane_colors[sources[0]], lane_colors[target], rng)

    visible = [i for i in range(_NUM_LANES) if i != hidden]
    visible_colors = [lane_colors[i] for i in visible]
    if lane_colors[hidden] != bg:
        return None
    if any(lane_colors[i] == bg for i in visible):
        return None
    if lane_colors[sources[0]] != lane_colors[sources[1]]:
        return None
    if lane_colors[target] == lane_colors[sources[0]]:
        return None
    if len(set(visible_colors)) != 2:
        return None
    if sum(1 for row in range(_H) if int(inp[row, _LEFT_CUE_COL]) != bg) != 2:
        return None
    if len({int(inp[row, _LEFT_CUE_COL]) for row in range(_H) if int(inp[row, _LEFT_CUE_COL]) != bg}) != 1:
        return None
    if sum(1 for row in range(_H) if int(inp[row, _RIGHT_CUE_COL]) != bg) != 1:
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
