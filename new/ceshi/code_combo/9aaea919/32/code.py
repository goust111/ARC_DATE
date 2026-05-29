# Combo variant of 9aaea919 (index 32)
# combo_spec: D1/1+D11/2+D12/0+D13/2
# Name: 全数量驱动系统
#
# Design analysis
# ===============
# a_specificization_and_trace:
#   - The canvas keeps the original 30x30 geometry: four fixed 5-column lanes
#     at cols 3-7, 9-13, 15-19, 21-25; sprites keep the same 3x5 plus/diamond
#     footprint and period-4 bottom stacking lattice.
#   - D1/1 removes marker bars entirely. Colors 2 and 3 are no longer
#     structural, so every role decision must come from visible stack counts.
#   - The fused role pipeline is global and count-sorted:
#       1. Count all four lanes and sort them from fewer -> more, breaking ties
#          left-to-right.
#       2. The lower pair becomes the source side; the higher pair becomes the
#          target side. This is the smallest count-driven reconciliation that
#          keeps D11's "difference between two sources" and D12's "one donor to
#          many targets" simultaneously visible.
#       3. Inside the lower pair, only the fuller lane actually donates. Its
#          transferable amount is the count difference to the smaller lane.
#       4. Inside the higher pair, the target with fewer sprites receives the
#          larger half of that difference and the fuller target receives the
#          smaller half; ties again resolve left-to-right.
#   - Concrete trace with counts [6, 1, 5, 4]:
#       sorted counts = [1, 4, 5, 6]
#       lower pair = (1, 4) -> donor is the 4-stack, diff = 3
#       higher pair = (5, 6) -> gains = (2, 1) by target count order
#       output: only the top 3 sprites of the 4-stack turn grey; the 5-stack
#       grows by 2 and the 6-stack grows by 1.
#   - Tie trace with counts [4, 1, 5, 5]:
#       sorted counts = [1, 4, 5, 5]
#       the donor diff is still 3, but the two targets are tied, so the left
#       target receives the larger half (2) and the right target receives 1.
#   - Relative to the base task, every changed step now depends on counts:
#     roles come from global ranking, transfer quantity comes from a difference,
#     the move is 1:many, and the larger share is routed by target count order
#     rather than by marker color or spatial nearness.
# b_complexity_calibration:
#   - A naive "minimum lane -> maximum lanes" interpretation conflicts with the
#     subtraction variant because the minimum lane cannot define a positive
#     source-target difference. The low-pair/high-pair decomposition resolves
#     that conflict while keeping the whole task count-driven.
#   - The generator uses four small count families:
#       * (1, 4, 5, 6): strict ordering, odd split, distinct targets;
#       * (1, 4, 5, 5): target-side tie, so target tie-break is exercised;
#       * (1, 4, 4, 5): source/target-boundary tie, so global rank tie-break is
#         exercised;
#       * (1, 3, 4, 5): even difference, so the no-remainder split branch also
#         appears.
#   - Most layouts still force target count order to disagree with left-to-right
#     position when the target counts are distinct, so D13 stays visible rather
#     than collapsing to a positional heuristic.
# c_hidden_assumption_review_of_old_main:
#   - `_find_marker_bars(...)` and every marker-removal step are invalid because
#     there are no structural bars in the fusion.
#   - The original solver inferred lane extents from marker bboxes. The fused
#     solver must instead use the fixed lane geometry from the task spec.
#   - The old loop "each marked red lane fully greys and transfers to one green
#     lane" breaks in three ways: there is no marker-defined source set, only
#     part of one lane greys, and the transfer fans out to two targets.
#   - The old local nearest-target match is also invalid. Target ownership and
#     target order must be fixed globally from counts before any placement.
# d_placement_legality:
#   - All edits stay inside the original lane columns and on the original
#     period-4 lattice, so components never overlap across lanes.
#   - Every emitted family is filtered so both final target totals stay within
#     the 7-slot height limit.
#   - Input reconstruction simply recolors the donor's grey slice back to its
#     lane color and erases the top gained slots from each target, so the
#     generator never needs ambiguous overlap handling.
# e_visual_observability:
#   - Humans can infer every control variable from stack heights alone: no
#     sorting or branching depends on raw color ids or invisible metadata.
#   - The lower-pair vs higher-pair partition is visible because every sample
#     has a positive lower-pair difference and at least one target gain.
#   - The subtraction step is visible because only the donor lane's top excess
#     turns grey, while the smallest lane in the lower pair stays unchanged.
#   - The emitted family mix covers distinct-target ordering, target-side ties,
#     and source/target-boundary ties, so the tie-break branches are visible in
#     the training distribution instead of existing only in code.
# f_visual_tidy:
#   - The scene stays a clean four-column stack display with no overlays.
#   - Only one lane becomes partially grey, one lower lane stays unchanged, and
#     two upper lanes grow. This yields a readable three-role output without
#     crowding or color mixing inside any single sprite.
# g_color_change_inference:
#   - Count alone determines roles; colors never participate in sorting.
#   - Grey has a single structural meaning: donor excess. Newly placed sprites
#     adopt the pre-existing color of their respective target lane, exactly as
#     in the original task family.
# fusion_metric_and_non_degeneracy_check:
#   - For a generated sample, record:
#       sorted_counts in {(1, 4, 5, 6), (1, 4, 5, 5), (1, 4, 4, 5), (1, 3, 4, 5)}
#       donor_diff in {2, 3}
#       ordered_target_gains in {(1, 1), (2, 1)}
#   - For families with distinct target counts, the generator only keeps
#     permutations where target-count order is opposite to left-to-right order,
#     so the fused rule and a simpler positional rule provably disagree on the
#     same input.
# new_main_assumptions_and_generator_guarantees:
#   - Every lane has at least one valid monochrome sprite component.
#   - Every sample has a unique baseline lane and a strictly fuller donor lane,
#     so the donor difference is always positive.
#   - The emitted family mix covers strict ordering, target-side ties,
#     source/target-boundary ties, and even-vs-odd split amounts.
#   - When target counts are distinct, the smaller target is always placed on
#     the opposite horizontal side from the larger target, guaranteeing that
#     target count order matters and is not equivalent to a left-to-right
#     heuristic.

import sys
from itertools import permutations

import numpy as np

try:
    from arc_utils import bbox_coords, components_mask, get_bg, paste
except ModuleNotFoundError:
    sys.path.append("/home/lixinyu/snap/new/solvers")
    from arc_utils import bbox_coords, components_mask, get_bg, paste


# --- CFG: canvas & layout ---
_H, _W = 30, 30
_NUM_LANES = 4
_LANE_STARTS = [3, 9, 15, 21]
_LANE_CENTERS = [c0 + 2 for c0 in _LANE_STARTS]
_SPRITE_H, _SPRITE_W = 3, 5

# --- CFG: vertical stacking ---
_BOTTOM_TOP = 25
_PERIOD = 4

# --- CFG: color palette ---
_BG_CHOICES = [0, 1, 4, 6, 7, 8, 9]
_GREY = 5

# --- CFG: count families ---
_COUNT_PATTERNS = [
    (1, 4, 5, 6),
    (1, 4, 5, 5),
    (1, 4, 4, 5),
    (1, 3, 4, 5),
]
_GEN_RETRIES = 300


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


def _sorted_sprites(sprites: list[list[tuple[int, int]]]) -> list[tuple[int, int, list[tuple[int, int]]]]:
    ranked = []
    for pts in sprites:
        r0, c0, _r1, _c1 = bbox_coords(np.array(pts))
        ranked.append((r0, c0, pts))
    ranked.sort()
    return ranked


def _extract_lane_state(g: np.ndarray, lane_idx: int, bg: int) -> dict[str, object]:
    c0 = _LANE_STARTS[lane_idx]
    cols = range(c0, c0 + _SPRITE_W)
    comps = _single_color_components_in_cols(g, cols, bg)
    assert comps, "expected each lane to contain visible sprites"

    sprite_size = max(len(pts) for pts in comps)
    sprites = [pts for pts in comps if len(pts) == sprite_size]
    assert sprites, "expected sprite-sized components in every lane"

    sprites_sorted = _sorted_sprites(sprites)
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
        "color": int(g[rr, cc]),
        "count": len(sprites_sorted),
        "period": period,
        "occupied": set(tops),
        "bottom0": max(tops),
        "sprites_sorted": sprites_sorted,
        "ref_r0": ref_r0,
        "ref_pts": ref_pts,
    }


def _sorted_role_indices(counts: list[int]) -> tuple[int, int, list[int]]:
    ordered = sorted(range(_NUM_LANES), key=lambda idx: (counts[idx], _LANE_CENTERS[idx]))
    baseline_idx = ordered[0]
    donor_idx = ordered[1]
    target_idxs = ordered[2:]
    assert counts[baseline_idx] < counts[donor_idx], "expected a positive lower-pair difference"
    assert counts[donor_idx] <= counts[target_idxs[0]] <= counts[target_idxs[1]]
    return baseline_idx, donor_idx, target_idxs


def _split_transfer(n: int) -> tuple[int, int]:
    return (n + 1) // 2, n // 2


def _grey_top_sprites(
    out: np.ndarray,
    sprites_sorted: list[tuple[int, int, list[tuple[int, int]]]],
    n_top: int,
) -> None:
    for _r0, _c0, pts in sprites_sorted[:n_top]:
        for r, c in pts:
            out[r, c] = _GREY


def _place_cloned_sprites(
    out: np.ndarray,
    lane_state: dict[str, object],
    src_center: float,
    src_ref_r0: int,
    src_ref_pts: list[tuple[int, int]],
    n_new: int,
    bg: int,
) -> None:
    if n_new <= 0:
        return

    h, w = out.shape
    dc_shift = int(round(float(lane_state["center"]) - src_center))
    src_offsets = [(r - src_ref_r0, c + dc_shift) for r, c in src_ref_pts]
    sprite_h = max(dr for dr, _c in src_offsets) + 1

    placed = 0
    k = 0
    while placed < n_new:
        new_top = int(lane_state["bottom0"]) - k * int(lane_state["period"])
        k += 1
        if new_top < 0 or new_top + sprite_h > h:
            if new_top < 0:
                break
            continue
        if new_top in lane_state["occupied"]:
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
            out[new_top + dr, c] = int(lane_state["color"])
        lane_state["occupied"].add(new_top)
        placed += 1

    assert placed == n_new, "expected enough room for all transferred sprites"


def main(grid: np.ndarray) -> np.ndarray:
    """
    Use count ranking to derive a low-pair donor and distribute its excess across the high-pair targets.

    Input structure:
    - The background is the most common color. There are four fixed vertical lanes and no marker bars at all.
    - Each lane contains repeated copies of one monochrome 3x5 plus/diamond sprite on the original period-4 lattice.

    Rule:
    - Count sprites in all lanes and sort the lanes from fewer to more, using left-to-right tie-breaks.
    - The lower two lanes form the source side. Only the fuller lane in that low pair donates, and it donates only its excess over the smaller low lane: the top difference-sized slice turns grey.
    - The higher two lanes form the target side. Order them from fewer sprites to more sprites, again using left-to-right tie-breaks; if the donor excess is N, the smaller target receives ceil(N/2) new copies and the larger target receives floor(N/2). Inserted copies keep the donor shape, adopt each target lane's own color, and stack into empty slots from the bottom upward.

    Key insight:
    - A single visible quantity, sprite count, controls the entire pipeline: it partitions lanes into low and high pairs, determines the transferable excess inside the low pair, and then orders the two targets that split that excess.
    """
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)

    lane_states = [_extract_lane_state(g, lane_idx, bg) for lane_idx in range(_NUM_LANES)]
    counts = [int(st["count"]) for st in lane_states]
    baseline_idx, donor_idx, target_idxs = _sorted_role_indices(counts)

    donor = lane_states[donor_idx]
    baseline_count = int(lane_states[baseline_idx]["count"])
    donor_count = int(donor["count"])
    diff = donor_count - baseline_count
    assert diff > 0, "expected donor to have a positive excess over the baseline lane"

    out = g.copy()
    _grey_top_sprites(out, donor["sprites_sorted"], diff)

    src_ref_r0 = int(donor["ref_r0"])
    src_ref_pts = donor["ref_pts"]
    left_gain, right_gain = _split_transfer(diff)
    target_states = [lane_states[i] for i in target_idxs]
    _place_cloned_sprites(out, target_states[0], float(donor["center"]), src_ref_r0, src_ref_pts, left_gain, bg)
    _place_cloned_sprites(out, target_states[1], float(donor["center"]), src_ref_r0, src_ref_pts, right_gain, bg)

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


def _paint_tops(grid: np.ndarray, lane_idx: int, color: int, tops: list[int], bg: int) -> None:
    patch = _sprite_patch(color, bg)
    c0 = _LANE_STARTS[lane_idx]
    for r0 in tops:
        paste(grid, patch, r0, c0, transparent=bg)


def _pattern_layouts(pattern: tuple[int, int, int, int]) -> list[tuple[int, int, int, int]]:
    layouts = []
    for counts in sorted(set(permutations(pattern))):
        count_list = list(counts)
        baseline_idx, donor_idx, target_idxs = _sorted_role_indices(count_list)
        diff = count_list[donor_idx] - count_list[baseline_idx]
        left_gain, right_gain = _split_transfer(diff)
        low_target_idx, high_target_idx = target_idxs
        if count_list[low_target_idx] + left_gain > 7 or count_list[high_target_idx] + right_gain > 7:
            continue
        if count_list[low_target_idx] < count_list[high_target_idx]:
            if _LANE_CENTERS[low_target_idx] < _LANE_CENTERS[high_target_idx]:
                continue
        layouts.append(counts)
    return layouts


_VALID_LAYOUTS_BY_PATTERN = {pattern: _pattern_layouts(pattern) for pattern in _COUNT_PATTERNS}
assert all(_VALID_LAYOUTS_BY_PATTERN.values()), "expected every count family to yield valid layouts"


def _choose_input_counts(rng: np.random.Generator) -> list[int]:
    pattern = _COUNT_PATTERNS[int(rng.integers(len(_COUNT_PATTERNS)))]
    layouts = _VALID_LAYOUTS_BY_PATTERN[pattern]
    return list(layouts[int(rng.integers(len(layouts)))])


def _build_output(
    input_counts: list[int],
    lane_colors: list[int],
    bg: int,
) -> tuple[np.ndarray, int, int, list[int], tuple[int, int]]:
    baseline_idx, donor_idx, target_idxs = _sorted_role_indices(input_counts)
    diff = input_counts[donor_idx] - input_counts[baseline_idx]
    left_gain, right_gain = _split_transfer(diff)

    output_counts = input_counts[:]
    output_counts[target_idxs[0]] += left_gain
    output_counts[target_idxs[1]] += right_gain
    assert output_counts[target_idxs[0]] <= 7 and output_counts[target_idxs[1]] <= 7

    out = np.full((_H, _W), bg, dtype=int)
    for lane_idx in range(_NUM_LANES):
        _paint_lane(out, lane_idx, lane_colors[lane_idx], output_counts[lane_idx], bg)

    donor_tops = _lane_tops(input_counts[donor_idx])[:diff]
    _paint_tops(out, donor_idx, _GREY, donor_tops, bg)
    return out, baseline_idx, donor_idx, target_idxs, (left_gain, right_gain)


def _gen_attempt() -> np.ndarray | None:
    rng = np.random.default_rng()

    bg = int(rng.choice(_BG_CHOICES))
    sprite_palette = [c for c in range(10) if c not in (bg, _GREY)]
    lane_colors = rng.choice(sprite_palette, size=_NUM_LANES, replace=False).tolist()
    input_counts = _choose_input_counts(rng)

    out, baseline_idx, donor_idx, target_idxs, gains = _build_output(input_counts, lane_colors, bg)

    inp = out.copy()
    donor_diff = input_counts[donor_idx] - input_counts[baseline_idx]
    assert donor_diff == sum(gains)
    donor_tops = _lane_tops(input_counts[donor_idx])[:donor_diff]
    _paint_tops(inp, donor_idx, lane_colors[donor_idx], donor_tops, bg)

    for target_idx, gain in zip(target_idxs, gains):
        c0 = _LANE_STARTS[target_idx]
        output_count = input_counts[target_idx] + gain
        for r0 in _lane_tops(output_count)[:gain]:
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
