# Combo variant of 9aaea919 (index 2)
# combo_spec: D1/1+D12/0
# Name: 数量决定角色+一源多目标
#
# Design analysis
# ===============
# a_specificization_and_trace:
#   - The canvas remains 30x30 with four fixed 5-column lanes at columns
#     3-7, 9-13, 15-19, 21-25. Sprites keep the original 3x5 plus/diamond
#     footprint and the same period-4 vertical lattice ending at top row 25.
#   - The D1 change removes marker bars entirely, so the last row is just
#     background and colors 2/3 may now be ordinary lane colors.
#   - The D12 change still uses one source and two targets, but roles are now
#     inferred from counts instead of marker colors:
#       input counts [3, 3, 4, 4]
#       old rule: impossible without red/green bars.
#       fused rule: the leftmost minimum lane (lane 0) is the source; the two
#       maximum lanes (lanes 2 and 3) are the targets. The source turns grey,
#       lane 2 gets ceil(3/2)=2 new sprites, lane 3 gets floor(3/2)=1, both
#       stacked upward in their own lanes.
#   - The visible difference from the original task is twofold: roles come from
#     stack heights, and the transferred count is split across two tallest lanes
#     instead of merging into one marked destination.
# b_complexity_calibration:
#   - The generator uses a small count family with exactly two maximum lanes and
#     source counts restricted to 2 or 3:
#       [2,2,3,3], [2,3,4,4], [3,3,4,4], [3,4,5,5]
#   - This keeps the one-to-many split always visible because both targets
#     receive at least one sprite, and it shows both even and odd splits.
#   - Half of the patterns have a minimum tie, making the "leftmost minimum"
#     disambiguation inferable, but the family stays compact enough that humans
#     can compare examples by eye instead of memorizing many cases.
# c_hidden_assumption_review_of_old_main:
#   - `_find_marker_bars(...)` and all marker removal logic no longer apply.
#     In the fusion, colors 2 and 3 are not structural.
#   - The original destination lookup via marker bounding boxes is invalid.
#     Lane geometry must come from the fixed task layout.
#   - The original many:1 logic chose one nearest green lane. That assumption
#     conflicts with D12/0: the source count must be split over two targets.
#   - D1/1 by itself picked a single fullest lane; the fusion instead requires
#     exactly two fullest lanes so the one-to-many mechanism is exercised rather
#     than collapsing back to single-target transfer.
# d_placement_legality:
#   - With target counts drawn from the patterns above, each target ends with at
#     most 7 sprites, which fits the original lattice (top row 1 for 7 sprites).
#   - Every sprite stays inside its lane's 5 columns, and all placements are on
#     the same period-4 slots, so no overlap or boundary issues arise.
# e_visual_observability:
#   - Source/target roles depend only on visible stack heights, never on hidden
#     color ids or other non-pixel properties.
#   - Every generated sample has exactly two tallest lanes in the input, and the
#     output shows both of them growing. This makes the 1:many behavior visible.
#   - Odd-source cases visibly give one extra new sprite to the left target,
#     which supports inferring the ceil/floor split.
# f_visual_tidy:
#   - No extra overlays are introduced. The scene remains four clean vertical
#     stacks, so the main evidence is just lane height changes and greying.
#   - Because both targets are existing lanes, the output remains uncluttered:
#     no lane mixes colors and no shapes overlap across lane boundaries.
# g_color_change_inference:
#   - Colors do not participate in role selection.
#   - The only recolor rule is structural and easy to see: the counted source
#     lane becomes grey, and transferred copies adopt each target lane's own
#     existing color.
# new_main_assumptions_and_generator_guarantees:
#   - Every lane contains at least one monochrome sprite component.
#   - Exactly two lanes share the maximum input count, and at least one other
#     lane has a smaller count, so source and target roles are well-defined.
#   - The source count is always 2 or 3, ensuring both targets actually gain
#     sprites and the fusion does not degenerate to a one-target visible change.

import sys

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
_SPRITE_H, _SPRITE_W = 3, 5

# --- CFG: vertical stacking ---
_BOTTOM_TOP = 25
_PERIOD = 4

# --- CFG: color palette ---
_BG_CHOICES = [0, 1, 4, 6, 7, 8, 9]
_GREY = 5

# --- CFG: count patterns ---
_COUNT_PATTERNS = [
    (2, 2, 3, 3),
    (2, 3, 4, 4),
    (3, 3, 4, 4),
    (3, 4, 5, 5),
]
_MAX_LANE_TOTAL = 7
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


def _extract_lane_state(g: np.ndarray, lane_idx: int, bg: int) -> dict:
    c0 = _LANE_STARTS[lane_idx]
    cols = range(c0, c0 + _SPRITE_W)
    comps = _single_color_components_in_cols(g, cols, bg)
    assert comps, "expected each lane to contain visible sprites"

    sprite_size = max(len(pts) for pts in comps)
    sprites = [pts for pts in comps if len(pts) == sprite_size]
    assert sprites, "expected at least one sprite-sized component per lane"

    sprites_sorted = _sorted_sprites(sprites)
    tops = [r0 for r0, _c0, _pts in sprites_sorted]
    if len(tops) >= 2:
        diffs = [b - a for a, b in zip(tops, tops[1:]) if b - a > 0]
        period = int(np.gcd.reduce(diffs)) if diffs else _PERIOD
    else:
        period = _PERIOD

    sample_r, sample_c = sprites_sorted[0][2][0]
    return {
        "lane_idx": lane_idx,
        "center": c0 + (_SPRITE_W - 1) / 2.0,
        "color": int(g[sample_r, sample_c]),
        "count": len(sprites_sorted),
        "period": period,
        "occupied": set(tops),
        "bottom0": max(tops),
        "sprites_sorted": sprites_sorted,
    }


def _role_indices_from_counts(counts: list[int]) -> tuple[int, list[int]]:
    src_idx = min(range(len(counts)), key=lambda i: (counts[i], i))
    max_count = max(counts)
    dst_idxs = [i for i, count in enumerate(counts) if count == max_count]
    assert len(dst_idxs) == 2, "expected exactly two fullest lanes"
    assert src_idx not in dst_idxs, "expected source to be strictly smaller than the targets"
    return src_idx, dst_idxs


def _split_transfer(n_sprites: int) -> tuple[int, int]:
    return (n_sprites + 1) // 2, n_sprites // 2


def _place_cloned_sprites(
    out: np.ndarray,
    lane_state: dict,
    src_center: float,
    src_ref_r0: int,
    src_ref_pts: list[tuple[int, int]],
    n_new: int,
    bg: int,
) -> None:
    if n_new <= 0:
        return

    h, w = out.shape
    dc_shift = int(round(lane_state["center"] - src_center))
    src_offsets = [(r - src_ref_r0, c + dc_shift) for r, c in src_ref_pts]
    sprite_h = max(dr for dr, _c in src_offsets) + 1

    placed = 0
    k = 0
    while placed < n_new:
        new_top = lane_state["bottom0"] - k * lane_state["period"]
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
            out[new_top + dr, c] = lane_state["color"]
        lane_state["occupied"].add(new_top)
        placed += 1

    assert placed == n_new, "expected enough room to place all transferred sprites"


def main(grid: np.ndarray) -> np.ndarray:
    """
    Transfer sprites from the sparsest lane into the two fullest lanes, recoloring and splitting the count.

    Input structure:
    - The background is the most common color. There are four fixed vertical lanes and no marker bars at all.
    - Each lane contains repeated identical monochrome plus sprites stacked on the same period-4 lattice.

    Rule:
    - Count sprites in all four lanes. The leftmost lane with the fewest sprites is the single source.
    - The two lanes with the most sprites are the targets, ordered left-to-right. All source sprites turn grey (5).
    - If the source contains N sprites, the left target receives ceil(N/2) new copies and the right target receives floor(N/2). Each copy keeps the source shape, shifts into the target lane, adopts that target lane's own color, and fills empty lattice slots from the bottom upward.

    Key insight:
    - The puzzle now combines two visible inferences: stack heights determine roles, and the conserved source count is split across both tallest lanes rather than merged into one destination.
    """
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)

    lane_states = [_extract_lane_state(g, lane_idx, bg) for lane_idx in range(_NUM_LANES)]
    counts = [st["count"] for st in lane_states]
    src_idx, dst_idxs = _role_indices_from_counts(counts)

    src_state = lane_states[src_idx]
    dst_states = [lane_states[i] for i in dst_idxs]
    dst_states.sort(key=lambda st: st["center"])

    out = g.copy()

    src_sprites = [pts for _r0, _c0, pts in src_state["sprites_sorted"]]
    for pts in src_sprites:
        for r, c in pts:
            out[r, c] = _GREY

    src_ref_r0, _src_ref_c0, src_ref_pts = src_state["sprites_sorted"][0]
    left_gain, right_gain = _split_transfer(len(src_sprites))
    _place_cloned_sprites(out, dst_states[0], src_state["center"], src_ref_r0, src_ref_pts, left_gain, bg)
    _place_cloned_sprites(out, dst_states[1], src_state["center"], src_ref_r0, src_ref_pts, right_gain, bg)

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


def _build_output(input_counts: list[int], lane_colors: list[int], bg: int) -> tuple[np.ndarray, int, list[int], list[int]]:
    src_idx, dst_idxs = _role_indices_from_counts(input_counts)
    left_idx, right_idx = sorted(dst_idxs)
    left_gain, right_gain = _split_transfer(input_counts[src_idx])

    output_counts = input_counts[:]
    output_counts[left_idx] += left_gain
    output_counts[right_idx] += right_gain
    assert output_counts[left_idx] <= _MAX_LANE_TOTAL
    assert output_counts[right_idx] <= _MAX_LANE_TOTAL

    out = np.full((_H, _W), bg, dtype=int)
    for lane_idx in range(_NUM_LANES):
        color = _GREY if lane_idx == src_idx else lane_colors[lane_idx]
        _paint_lane(out, lane_idx, color, output_counts[lane_idx], bg)

    return out, src_idx, [left_idx, right_idx], output_counts


def _gen_attempt() -> np.ndarray | None:
    rng = np.random.default_rng()

    bg = int(rng.choice(_BG_CHOICES))
    sprite_palette = [c for c in range(10) if c not in (bg, _GREY)]
    lane_colors = rng.choice(sprite_palette, size=_NUM_LANES, replace=False).tolist()

    pattern = list(_COUNT_PATTERNS[int(rng.integers(len(_COUNT_PATTERNS)))])
    input_counts = [pattern[i] for i in rng.permutation(_NUM_LANES)]

    out, src_idx, dst_idxs, output_counts = _build_output(input_counts, lane_colors, bg)

    inp = out.copy()
    _paint_lane(inp, src_idx, lane_colors[src_idx], input_counts[src_idx], bg)

    left_idx, right_idx = dst_idxs
    left_gain, right_gain = _split_transfer(input_counts[src_idx])
    for lane_idx, gain in ((left_idx, left_gain), (right_idx, right_gain)):
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
