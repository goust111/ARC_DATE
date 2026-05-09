# Combo variant of 9aaea919 (index 24)
# combo_spec: D1/1+D13/2
# Name: 数量决定角色+数量排序匹配
#
# Design analysis (YAML-ish, condensed for implementation use):
# design:
#   a_specificization_and_trace:
#     - The canvas remains the original 30x30 four-lane display with fixed
#       5-column lanes at cols 3-7, 9-13, 15-19, 21-25.
#     - Marker bars disappear entirely. Every visible clue now comes from the
#       number of 3x5 plus/diamond sprites stacked in each lane.
#     - Fusion behavior on a concrete example:
#       input counts by lane = [2, 1, 4, 3]
#       global count ranking => sources are the two smaller lanes [1, 2], targets
#       are the two larger lanes [3, 4].
#       source ranking for pairing = 2 then 1; target ranking = 3 then 4.
#       therefore lane(count 2) transfers into lane(count 3), and lane(count 1)
#       transfers into lane(count 4). Both source lanes turn grey in place.
#     - Old base task needed explicit red/green bars plus nearest-destination
#       matching. The combo removes marker colors and replaces both stages with
#       one unified count pipeline: count to infer roles, then count again to
#       infer which source feeds which target.
#   b_complexity_calibration:
#     - Using only one source and one target would collapse D13/2; there would be
#       no cross-lane sorting to observe. The fused scene therefore uses all four
#       lanes: two sources and two targets.
#     - To keep the role split visible, generated count patterns always satisfy
#       second-smallest < third-smallest. This prevents an ambiguous middle tie.
#     - The input count range is expanded slightly beyond the original 1-3 family
#       to include count 4. That extra level makes the two-vs-two ranking legible
#       without changing the lane geometry or causing overcrowding.
#     - The generator rotates among three visibly distinct count families:
#       one strict ordering, one source-side tie, and one target-side tie.
#       This keeps the mechanism legible while still exposing the left-to-right
#       disambiguation branch required by the fused rule.
#   c_hidden_assumption_review_of_old_main:
#     - `_find_marker_bars(...)` and all marker removal logic are invalid here
#       because colors 2 and 3 may be ordinary sprite colors again.
#     - The old solver inferred lane extents from marker bounding boxes. Without
#       markers, the fused solver must recover lanes from the fixed geometry in
#       the original task spec.
#     - The D1 single-source/single-target assumption is also invalid now. The
#       fused rule needs two source lanes and two target lanes, derived from one
#       global ranking over all lane counts.
#     - The old nearest-lane decision was local per source. Under D13/2 fusion,
#       pair ownership must be fixed globally before placement, otherwise the
#       count-ranked matching can be corrupted by processing order.
#   d_placement_legality:
#     - Every lane still holds one monochrome sprite stack on the original
#       period-4 lattice, so components never overlap across lanes.
#     - Generated patterns are filtered so each target lane plus its matched
#       source gain stays <= 7 sprites, which fits safely in the 30x30 canvas.
#     - Input construction removes exactly the newly gained top slots from each
#       target lane, leaving legal empty positions for the solver to refill.
#   e_visual_observability:
#     - All sorting keys are visible sprite counts; no hidden use of raw color
#       ids, component enumeration order, or other non-human signals is allowed.
#     - The generator rejects layouts where count-ranked pairing collapses to the
#       original nearest-target heuristic or to a plain left-to-right one-to-one
#       pairing, so D13/2 has a visible effect in the pixels.
#     - Because the source/target split is also count-driven, the solver can be
#       inferred from comparing lane heights alone across 2-4 examples.
#   f_visual_tidy:
#     - No markers or overlays are added, so the scene stays a clean four-column
#       stack display.
#     - The fixed six-column spacing between lane centers keeps the recoloring
#       and height changes readable even when two lanes become grey in the output.
#   g_color_change_inference:
#     - Colors are not part of the matching logic.
#     - The only color rule remains structurally simple: source lanes become grey
#       and transferred sprites adopt the pre-existing color of their paired
#       target lane.
#   new_main_assumptions_and_generator_guarantees:
#     - All four lanes contain at least one valid sprite in the input.
#     - The role split is always exactly two sources and two targets, obtained by
#       sorting lane counts ascending with left-to-right tie-break.
#     - The second-smallest count is strictly less than the third-smallest count,
#       so the source/target boundary is visible.
#     - Generated samples cover strict ordering, source-side ties, and target-side
#       ties, so the left-to-right tie-break is observable rather than implicit.
#     - Count-ranked pairing differs from nearest-target pairing and from plain
#       left-to-right one-to-one pairing on every sample emitted by
#       `generate_input()`.

from itertools import permutations

import numpy as np

from arc_utils import bbox_coords, components_mask, get_bg, paste


# --- CFG: canvas & layout ---
_H, _W = 30, 30
_NUM_LANES = 4
_NUM_SOURCES = 2
_NUM_TARGETS = 2
_LANE_STARTS = [3, 9, 15, 21]
_LANE_CENTERS = [c0 + 2 for c0 in _LANE_STARTS]
_SPRITE_H, _SPRITE_W = 3, 5

# --- CFG: vertical stacking ---
_BOTTOM_TOP = 25
_PERIOD = 4

# --- CFG: color palette ---
_BG_CHOICES = [0, 1, 4, 6, 7, 8, 9]
_GREY = 5

# --- CFG: counts & limits ---
_COUNT_PATTERNS = [
    (1, 2, 3, 4),
    (1, 1, 3, 4),
    (1, 2, 4, 4),
]
_MAX_LANE_TOTAL = 7
_GEN_RETRIES = 400


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


def _extract_lane_state(g: np.ndarray, lane_idx: int, bg: int) -> dict[str, object]:
    c0 = _LANE_STARTS[lane_idx]
    cols = range(c0, c0 + _SPRITE_W)
    comps = _single_color_components_in_cols(g, cols, bg)
    assert comps, "expected every lane to contain at least one sprite"

    sprite_size = max(len(pts) for pts in comps)
    sprites = [pts for pts in comps if len(pts) == sprite_size]
    assert sprites, "expected sprite-sized components in every lane"

    sprites_sorted = []
    for pts in sprites:
        r0, c0b, _r1, _c1 = bbox_coords(np.array(pts))
        sprites_sorted.append((r0, c0b, pts))
    sprites_sorted.sort()

    tops = [r0 for r0, _c0, _pts in sprites_sorted]
    if len(tops) >= 2:
        diffs = [b - a for a, b in zip(tops, tops[1:]) if b - a > 0]
        period = int(np.gcd.reduce(diffs)) if diffs else _PERIOD
    else:
        period = _PERIOD

    ref_r0, _ref_c0, ref_pts = sprites_sorted[0]
    rr, cc = ref_pts[0]
    return {
        "lane_idx": lane_idx,
        "center": _LANE_CENTERS[lane_idx],
        "sprites": sprites_sorted,
        "count": len(sprites_sorted),
        "color": int(g[rr, cc]),
        "period": period,
        "occupied": set(tops),
        "bottom0": max(tops),
        "ref_r0": ref_r0,
        "ref_pts": ref_pts,
    }


def _partition_roles(counts: list[int]) -> tuple[list[int], list[int]]:
    ordered = sorted(range(_NUM_LANES), key=lambda idx: (counts[idx], _LANE_CENTERS[idx]))
    return ordered[:_NUM_SOURCES], ordered[_NUM_SOURCES:]


def _ranked_pairs(source_idxs: list[int], target_idxs: list[int], counts: list[int]) -> list[tuple[int, int]]:
    src_order = sorted(source_idxs, key=lambda idx: (-counts[idx], _LANE_CENTERS[idx]))
    dst_order = sorted(target_idxs, key=lambda idx: (counts[idx], _LANE_CENTERS[idx]))
    return list(zip(src_order, dst_order))


def _nearest_like_pairs(source_idxs: list[int], target_idxs: list[int]) -> list[tuple[int, int]]:
    pairs = []
    for src_idx in source_idxs:
        dst_idx = min(
            target_idxs,
            key=lambda idx: (abs(_LANE_CENTERS[idx] - _LANE_CENTERS[src_idx]), _LANE_CENTERS[idx]),
        )
        pairs.append((src_idx, dst_idx))
    return sorted(pairs)


def _left_to_right_pairs(source_idxs: list[int], target_idxs: list[int]) -> list[tuple[int, int]]:
    src_order = sorted(source_idxs, key=lambda idx: _LANE_CENTERS[idx])
    dst_order = sorted(target_idxs, key=lambda idx: _LANE_CENTERS[idx])
    return list(zip(src_order, dst_order))


def _pattern_layouts(pattern: tuple[int, int, int, int]) -> list[tuple[int, int, int, int]]:
    layouts = []
    for counts in sorted(set(permutations(pattern))):
        count_list = list(counts)
        source_idxs, target_idxs = _partition_roles(count_list)
        if count_list[source_idxs[-1]] >= count_list[target_idxs[0]]:
            continue
        pairs = _ranked_pairs(source_idxs, target_idxs, count_list)
        if sorted(pairs) == _nearest_like_pairs(source_idxs, target_idxs):
            continue
        if sorted(pairs) == sorted(_left_to_right_pairs(source_idxs, target_idxs)):
            continue
        layouts.append(counts)
    return layouts


_VALID_LAYOUTS_BY_PATTERN = {pattern: _pattern_layouts(pattern) for pattern in _COUNT_PATTERNS}
assert all(_VALID_LAYOUTS_BY_PATTERN.values()), "every count pattern must keep the fusion observable"


def _choose_input_counts(rng: np.random.Generator) -> list[int]:
    pattern = _COUNT_PATTERNS[int(rng.integers(len(_COUNT_PATTERNS)))]
    layouts = _VALID_LAYOUTS_BY_PATTERN[pattern]
    return list(layouts[int(rng.integers(len(layouts)))])


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reassign the two smaller sprite stacks into the two larger ones using count-ranked matching.

    Input structure:
    - The background is the most common color, and there are four fixed vertical lanes with no marker bars.
    - Each lane contains repeated copies of the same monochrome 3x5 plus/diamond sprite on the original period-4 lattice.

    Rule:
    - Count the sprites in all four lanes. The two lanes with fewer sprites are the sources; the two with more sprites are the targets, using left-to-right tie-breaks.
    - Rank the source lanes from more sprites to fewer, and rank the target lanes from fewer sprites to more. Pair by rank.
    - Each source lane turns grey (5), and its sprite count is recreated in the paired target lane using the target lane's existing color, stacked into empty slots from the bottom upward.

    Key insight: One visible quantity, sprite count, controls the whole transformation twice: first to decide which lanes donate or receive, then again to decide which donor matches which receiver.
    """
    g = np.asarray(grid, dtype=int)
    h, _w = g.shape
    bg = get_bg(g)

    lane_states = [_extract_lane_state(g, lane_idx, bg) for lane_idx in range(_NUM_LANES)]
    counts = [int(st["count"]) for st in lane_states]
    source_idxs, target_idxs = _partition_roles(counts)
    assert counts[source_idxs[-1]] < counts[target_idxs[0]], "expected a visible source/target count boundary"

    pairs = _ranked_pairs(source_idxs, target_idxs, counts)
    out = g.copy()

    for src_idx in source_idxs:
        for _r0, _c0, pts in lane_states[src_idx]["sprites"]:
            for r, c in pts:
                out[r, c] = _GREY

    for src_idx, dst_idx in pairs:
        src = lane_states[src_idx]
        dst = lane_states[dst_idx]
        dc_shift = int(round(int(dst["center"]) - int(src["center"])))
        src_offsets = [(r - int(src["ref_r0"]), c + dc_shift) for r, c in src["ref_pts"]]
        sprite_h = max(dr for dr, _c in src_offsets) + 1

        n_new = int(src["count"])
        placed = 0
        k = 0
        while placed < n_new:
            new_top = int(dst["bottom0"]) - k * int(dst["period"])
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
                out[new_top + dr, c] = int(dst["color"])
            dst["occupied"].add(new_top)
            placed += 1

        assert placed == n_new, "expected enough empty slots in each target lane"

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


def _build_output(input_counts: list[int], lane_colors: list[int], bg: int) -> tuple[np.ndarray, list[int], list[int], list[tuple[int, int]], list[int]]:
    source_idxs, target_idxs = _partition_roles(input_counts)
    pairs = _ranked_pairs(source_idxs, target_idxs, input_counts)

    output_counts = input_counts[:]
    for src_idx, dst_idx in pairs:
        output_counts[dst_idx] += input_counts[src_idx]
        if output_counts[dst_idx] > _MAX_LANE_TOTAL:
            raise AssertionError("target lane would overflow")

    out = np.full((_H, _W), bg, dtype=int)
    source_set = set(source_idxs)
    for lane_idx in range(_NUM_LANES):
        color = _GREY if lane_idx in source_set else lane_colors[lane_idx]
        _paint_lane(out, lane_idx, color, output_counts[lane_idx], bg)
    return out, source_idxs, target_idxs, pairs, output_counts


def _gen_attempt() -> np.ndarray | None:
    rng = np.random.default_rng()

    bg = int(rng.choice(_BG_CHOICES))
    sprite_palette = [c for c in range(10) if c not in (bg, _GREY)]
    lane_colors = rng.choice(sprite_palette, size=_NUM_LANES, replace=False).tolist()

    input_counts = _choose_input_counts(rng)

    try:
        out, source_idxs, target_idxs, pairs, output_counts = _build_output(input_counts, lane_colors, bg)
    except AssertionError:
        return None

    inp = out.copy()
    for lane_idx in source_idxs:
        _paint_lane(inp, lane_idx, lane_colors[lane_idx], input_counts[lane_idx], bg)

    gains = {dst_idx: input_counts[src_idx] for src_idx, dst_idx in pairs}
    for lane_idx in target_idxs:
        gain = gains[lane_idx]
        c0 = _LANE_STARTS[lane_idx]
        for r0 in _lane_tops(output_counts[lane_idx])[:gain]:
            inp[r0 : r0 + _SPRITE_H, c0 : c0 + _SPRITE_W] = bg

    if not np.array_equal(main(inp), out):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(_GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {_GEN_RETRIES} attempts")
