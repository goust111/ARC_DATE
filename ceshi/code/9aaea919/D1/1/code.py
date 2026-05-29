# Variant of 9aaea919
# Dimension: D1 — 匹配条件机制 (match_cond)
# Change: marker_pixels — 红色(2)标记条标识源车道，绿色(3)标记条标识目标车道，通过标记颜色直接指定源和目标... → 精灵数量匹配决定角色
#
# Design analysis (YAML-ish, condensed for implementation use):
# design:
#   a_specificization_and_trace:
#     - The canvas stays 30x30 with the same four fixed 5-column lanes:
#       3-7, 9-13, 15-19, 21-25.
#     - Each sprite keeps the original 3x5 plus/diamond footprint and stacks on
#       the same period-4 lattice ending at top row 25.
#     - The key visual change is that the input has no bottom marker bars at all.
#       The last row is pure background, so role inference must come from counts.
#     - Variant trace example:
#       input lane counts [2, 1, 3, 2]
#       old rule: without red/green bars the original solver cannot decide roles.
#       new rule: lane 1 is source (leftmost minimum count 1), lane 2 is target
#       (leftmost maximum count 3). Output greys lane 1 and adds one new sprite
#       to lane 2 in the highest empty lattice slot above its current stack.
#   b_complexity_calibration:
#     - The change is visible but localized: geometry, recoloring, and stacking
#       all stay the same; only the matching condition becomes count-based.
#     - The generator intentionally uses a small family of count patterns that
#       cover unique extremes and tie cases:
#       [1,2,2,3], [1,1,2,3], [1,2,3,3], [1,1,3,3], [2,2,3,3].
#     - This is enough variation to make the left-to-right tie-break observable
#       without adding unrelated mechanisms or too many free variables.
#   c_hidden_assumption_review_of_old_main:
#     - `_find_marker_bars(...)` assumed colors 2 and 3 were structural markers.
#       In the variant those colors may be ordinary sprite colors, so all marker
#       parsing/removal must disappear.
#     - The old destination-state builder recovered lane columns from marker
#       bounding boxes. Without markers, the solver must instead rely on the
#       fixed lane geometry from the original task spec.
#     - The old transfer loop supported 1-2 independently marked source lanes.
#       The new role logic yields one global source lane and one global target
#       lane chosen by visible sprite multiplicity.
#   d_placement_legality:
#     - Input counts are always chosen so min(counts) < max(counts), guaranteeing
#       distinct source and target lanes.
#     - With input counts drawn from the patterns above, the output target count
#       is at most 6, so the original 30x30 lattice still has enough space.
#     - Because sprites stay inside their original lane columns and use the same
#       period-4 slots, no overlap or boundary issues are introduced.
#   e_visual_observability:
#     - Humans can compare lane heights directly across 2-4 examples.
#     - Some generated samples contain ties at the minimum and/or maximum count,
#       so the "choose the leftmost tied lane" rule is inferable from pixels.
#   f_visual_tidy:
#     - No new overlays or decorations are added.
#     - The scene remains a clean four-lane stack display; only one lane turns
#       grey and one lane becomes taller.
#   g_color_change_inference:
#     - Colors are never used for sorting or matching.
#     - The only color rule remains the old one: source sprites become grey and
#       transferred sprites adopt the target lane's existing color.
#   new_main_assumptions_and_generator_guarantees:
#     - Every lane contains at least one monochrome sprite component in the input.
#     - All sprites share the original 3x5 footprint and fixed lane positions.
#     - `generate_input()` guarantees a distinct sparsest lane and fullest lane,
#       so `main()` never faces the degenerate all-counts-equal case.

import numpy as np

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

# --- CFG: counts & limits ---
_COUNT_PATTERNS = [
    (1, 2, 2, 3),  # unique min / unique max
    (1, 1, 2, 3),  # min tie only
    (1, 2, 3, 3),  # max tie only
    (1, 1, 3, 3),  # both min and max tie
    (2, 2, 3, 3),  # both tie with a larger transfer
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


def _extract_lane_state(g: np.ndarray, lane_idx: int, bg: int) -> dict:
    c0 = _LANE_STARTS[lane_idx]
    cols = range(c0, c0 + _SPRITE_W)
    comps = _single_color_components_in_cols(g, cols, bg)
    if not comps:
        return {
            "lane_idx": lane_idx,
            "center": c0 + (_SPRITE_W - 1) / 2.0,
            "sprites": [],
            "count": 0,
            "color": None,
            "period": _PERIOD,
            "occupied": set(),
            "bottom0": _BOTTOM_TOP,
        }

    sprite_size = max(len(pts) for pts in comps)
    sprites = [pts for pts in comps if len(pts) == sprite_size]

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

    color = int(g[sprites_sorted[0][2][0][0], sprites_sorted[0][2][0][1]])
    return {
        "lane_idx": lane_idx,
        "center": c0 + (_SPRITE_W - 1) / 2.0,
        "sprites": sprites_sorted,
        "count": len(sprites_sorted),
        "color": color,
        "period": period,
        "occupied": set(tops),
        "bottom0": max(tops),
    }


def _role_indices_from_counts(counts: list[int]) -> tuple[int, int]:
    src_idx = min(range(len(counts)), key=lambda i: counts[i])
    dst_idx = max(range(len(counts)), key=lambda i: counts[i])
    return src_idx, dst_idx


def main(grid: np.ndarray) -> np.ndarray:
    """
    Transfer sprites from the sparsest lane into the fullest lane, recoloring and stacking.

    Input structure:
    - bg = most common color. There are four fixed vertical lanes and no marker bars.
    - Each lane contains repeated identical monochrome sprites on the same evenly spaced lattice.

    Rule:
    - Count the sprites in every lane. The leftmost lane with the fewest sprites is the source.
    - The leftmost lane with the most sprites is the target. Source sprites turn grey (5).
    - The source count is recreated in the target lane using the target color, filling empty lattice slots from the bottom upward.

    Key insight: The move is no longer announced by special marker colors; the solver must infer source and target roles by comparing visible stack heights across lanes.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = get_bg(g)

    lane_states = [_extract_lane_state(g, lane_idx, bg) for lane_idx in range(_NUM_LANES)]
    counts = [st["count"] for st in lane_states]
    src_idx, dst_idx = _role_indices_from_counts(counts)
    assert src_idx != dst_idx, "expected distinct sparsest and fullest lanes"

    src = lane_states[src_idx]
    dst = lane_states[dst_idx]
    assert src["sprites"] and dst["sprites"], "expected sprites in both source and target lanes"

    out = g.copy()

    for _r0, _c0, pts in src["sprites"]:
        for r, c in pts:
            out[r, c] = _GREY

    src_ref_r0, _src_ref_c0, src_ref_pts = src["sprites"][0]
    dc_shift = int(round(dst["center"] - src["center"]))
    src_offsets = [(r - src_ref_r0, c + dc_shift) for r, c in src_ref_pts]
    sprite_h = max(dr for dr, _c in src_offsets) + 1

    n_new = src["count"]
    placed = 0
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


def _build_output(input_counts: list[int], lane_colors: list[int], bg: int) -> tuple[np.ndarray, int, int, list[int]]:
    src_idx, dst_idx = _role_indices_from_counts(input_counts)
    output_counts = input_counts[:]
    output_counts[dst_idx] += input_counts[src_idx]

    out = np.full((_H, _W), bg, dtype=int)
    for lane_idx in range(_NUM_LANES):
        color = _GREY if lane_idx == src_idx else lane_colors[lane_idx]
        _paint_lane(out, lane_idx, color, output_counts[lane_idx], bg)
    return out, src_idx, dst_idx, output_counts


def _gen_attempt() -> np.ndarray | None:
    rng = np.random.default_rng()

    bg = int(rng.choice(_BG_CHOICES))
    sprite_palette = [c for c in range(10) if c not in (bg, _GREY)]
    lane_colors = rng.choice(sprite_palette, size=_NUM_LANES, replace=False).tolist()

    pattern = list(_COUNT_PATTERNS[int(rng.integers(len(_COUNT_PATTERNS)))])
    input_counts = [pattern[i] for i in rng.permutation(_NUM_LANES)]
    src_idx, dst_idx = _role_indices_from_counts(input_counts)
    if src_idx == dst_idx:
        return None

    out, src_idx, dst_idx, output_counts = _build_output(input_counts, lane_colors, bg)

    inp = out.copy()
    _paint_lane(inp, src_idx, lane_colors[src_idx], input_counts[src_idx], bg)

    dst_c0 = _LANE_STARTS[dst_idx]
    gain = input_counts[src_idx]
    for r0 in _lane_tops(output_counts[dst_idx])[:gain]:
        inp[r0 : r0 + _SPRITE_H, dst_c0 : dst_c0 + _SPRITE_W] = bg

    if not np.array_equal(main(inp), out):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(_GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {_GEN_RETRIES} attempts")
