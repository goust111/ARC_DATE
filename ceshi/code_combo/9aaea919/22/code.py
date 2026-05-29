# Combo variant of 9aaea919 (index 22)
# combo_spec: D11/2+D12/0
# Name: 差值转移+一源多目标

import sys
from itertools import combinations

import numpy as np

try:
    from arc_utils import bbox_coords, components_mask, get_bg, paste
except ModuleNotFoundError:
    sys.path.append("/home/lixinyu/snap/new/solvers")
    from arc_utils import bbox_coords, components_mask, get_bg, paste


# Design analysis for the combo variant
# a) Concrete behavior and trace:
#    - The 30x30 canvas, four fixed 5-column lanes, period-4 bottom stacking, and 3x5 plus sprites stay unchanged.
#    - The fused rule uses exactly 1 red source lane and exactly 2 green target lanes.
#    - Example: source has 3 sprites; the left and right green lanes have 2 and 1 sprites.
#      Base task: one green lane would absorb all 3 sprites.
#      D12-only: the left/right greens would receive (2, 1) by position, regardless of their own heights.
#      Combo task: the source lane still greys out completely, but each green lane compares itself to the source:
#      gains are (3-2, 3-1) = (1, 2), so both green lanes end at height 3.
#    - The visual effect is "two targets are independently topped up until they match the red source height".
# b) Complexity calibration:
#    - Fixing the structure to 1 red + 2 green keeps the puzzle readable while still forcing multi-target reasoning.
#    - The generator uses a 3-sprite source and smaller green lanes, so both targets always grow and the diff rule
#      is visible. It also forbids the left/right count pattern that would accidentally match the old even split.
# c) Implicit assumption audit:
#    - The original `best = min(...)` single-destination selection is invalid; the combo must process both greens.
#    - The original transfer amount = source count is invalid; each destination now needs its own gain computed from
#      `source_count - target_count`.
#    - The original conservation assumption no longer holds. One source can seed multiple targets, so the source shape
#      is cloned into each target independently after the source turns grey.
#    - Ordering must be spatial, not color-based. The only ordering used is the green lanes' x-position.
# d) Placement legality:
#    - Every lane still paints only inside its own 5-column slot, using the same period-4 lattice as the base task.
#    - The generator enforces `target_count < source_count`, so every gain is positive and output lane height becomes
#      exactly the source height, which is at most 3 and safely within the available 7 slots.
#    - Because the added sprites occupy the highest missing lattice slots, erasing those same top slots reconstructs
#      the input without overlap or ambiguity.
# e) Visual observability:
#    - Input shows one red marker and two green markers, so the solver must already consider one-to-many transfer.
#    - Across examples, both green lanes end with the same number of sprites as the red lane had, even when the two
#      green inputs start at different heights. That exposes the "difference to source" rule directly in the pixels.
# f) Visual cleanliness:
#    - No extra symbols are introduced. The only visible changes are source greying, two green lane growth events,
#      and marker removal, so the lane structure remains uncluttered.
# g) Color inference:
#    - Grey still means "source lane after transfer"; new target sprites still use the target lane's own color.
#      The combo changes counts and routing, not the palette semantics.
# h) New main() assumptions and how generate_input() guarantees them:
#    - There is exactly 1 red marker and exactly 2 green markers.
#    - Every marked lane has at least one detectable sprite component in its own columns.
#    - The source lane has exactly 3 sprites; each green lane has 1 or 2 sprites and the left green never has fewer
#      than the right green. This prevents the sample from collapsing to the old even-split behavior.
# i) Fusion-behavior metric used after implementation:
#    - For each sample, compare combo gains `(source-left_count, source-right_count)` against the old split gains
#      `((source+1)//2, source//2)`. The generator is only accepted if these tuples differ, proving the dataset is
#      not silently degenerating back to D12/0 on the same inputs.


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
        colors_in = {int(g[r, c]) for r, c in pts}
        if len(colors_in) == 1:
            out.append(pts)
    return out


def _sorted_sprites(sprites: list[list[tuple[int, int]]]) -> list[tuple[int, int, list[tuple[int, int]]]]:
    ranked = []
    for pts in sprites:
        rmin = min(r for r, _ in pts)
        cmin = min(c for _, c in pts)
        ranked.append((rmin, cmin, pts))
    ranked.sort()
    return ranked


def _lane_state_from_marker(g: np.ndarray, marker: list[tuple[int, int]], bg: int) -> dict[str, object]:
    """Extract sprite layout, color, and stacking metadata for one marked lane."""
    _, c0, _, c1 = bbox_coords(np.array(marker))
    lane_cols = range(c0, c1 + 1)
    marker_set = set(marker)
    lane_comps = _single_color_components_in_cols(g, lane_cols, bg)
    lane_comps = [[p for p in pts if p not in marker_set] for pts in lane_comps]
    lane_comps = [pts for pts in lane_comps if pts]
    assert lane_comps, "expected marked lane to contain sprites"

    sprite_sizes = [len(pts) for pts in lane_comps if len(pts) != len(marker)]
    assert sprite_sizes, "expected sprite-sized components in marked lane"
    sprite_size = max(sprite_sizes)
    sprites = [pts for pts in lane_comps if len(pts) == sprite_size]
    assert sprites, "expected at least one valid sprite in marked lane"

    sprites_sorted = _sorted_sprites(sprites)
    sample_r, sample_c = sprites_sorted[0][2][0]
    tops = [rmin for rmin, _cmin, _pts in sprites_sorted]
    if len(tops) >= 2:
        diffs = [b - a for a, b in zip(tops, tops[1:]) if b - a > 0]
        period = int(np.gcd.reduce(diffs)) if diffs else 4
    else:
        r0, _c0, r1, _c1 = bbox_coords(np.array(sprites_sorted[0][2]))
        period = (r1 - r0 + 1) + 1

    return {
        "center": (c0 + c1) / 2.0,
        "color": int(g[sample_r, sample_c]),
        "period": period,
        "occupied": set(tops),
        "bottom0": max(tops),
        "sprites_sorted": sprites_sorted,
        "count": len(sprites_sorted),
    }


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

    assert placed == n_new, "expected enough room to place all transferred sprites"


def main(grid: np.ndarray) -> np.ndarray:
    """
    Use one red source lane to top up two green target lanes by per-target differences.

    Input structure:
    - The background is the most common color. One thin red 1x5 bar marks the source lane and two thin green 1x5 bars mark the target lanes.
    - Each marked lane contains repeated identical monochrome plus sprites aligned to the same vertical lattice, and the green lanes already contain their own smaller stacks.

    Rule:
    - All sprites in the red-marked source lane turn grey.
    - Let S be the source sprite count. Order the two green lanes from left to right; for each green lane with current count T, add exactly S-T new sprites into that lane.
    - Each inserted sprite copies the source shape, is recolored to that target lane's own color, and is stacked into empty lattice slots from the bottom upward. Finally, remove all marker bars.

    Key insight:
    - The transferred amount is no longer a global total and no longer an even split. Each green lane independently compares its own height to the red source, so both targets are filled until they match the source stack.
    """
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)

    red_markers = _find_marker_bars(g, 2)
    green_markers = _find_marker_bars(g, 3)
    assert len(red_markers) == 1, "expected exactly one red source lane"
    assert len(green_markers) == 2, "expected exactly two green target lanes"

    src_state = _lane_state_from_marker(g, red_markers[0], bg)
    green_states = [_lane_state_from_marker(g, gm, bg) for gm in green_markers]
    green_states.sort(key=lambda st: float(st["center"]))

    out = g.copy()

    for _rmin, _cmin, pts in src_state["sprites_sorted"]:
        for r, c in pts:
            out[r, c] = 5

    src_ref_r0, _src_ref_c0, src_ref_pts = src_state["sprites_sorted"][0]
    source_count = int(src_state["count"])
    for green_state in green_states:
        gain = max(0, source_count - int(green_state["count"]))
        _place_cloned_sprites(out, green_state, float(src_state["center"]), src_ref_r0, src_ref_pts, gain, bg)

    for marker in red_markers + green_markers:
        for r, c in marker:
            out[r, c] = bg

    return out


# --------------- generator ---------------

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
_SOURCE_COUNT = 3
_TARGET_COUNT_PATTERNS = [(1, 1), (2, 1), (2, 2)]
_GEN_RETRIES = 300


def _sprite_patch(color: int, bg: int) -> np.ndarray:
    """Return a 3x5 plus sprite with bg-transparent corners."""
    patch = np.full((_SPRITE_H, _SPRITE_W), bg, dtype=int)
    patch[0, 1:4] = color
    patch[1, :] = color
    patch[2, 1:4] = color
    return patch


def _lane_tops(count: int) -> list[int]:
    """Return top-row positions for `count` sprites, ordered top-to-bottom."""
    return [_BOTTOM_TOP - _PERIOD * k for k in range(count - 1, -1, -1)]


def _paint_lane(grid: np.ndarray, lane_idx: int, color: int, count: int, bg: int) -> None:
    patch = _sprite_patch(color, bg)
    c0 = _LANE_STARTS[lane_idx]
    for r0 in _lane_tops(count):
        paste(grid, patch, r0, c0, transparent=bg)


def _role_choices() -> list[tuple[int, tuple[int, int], int]]:
    """Enumerate all 1-source / 2-target / 1-unmarked role assignments."""
    roles = []
    for red in range(_NUM_LANES):
        others = [i for i in range(_NUM_LANES) if i != red]
        for greens in combinations(others, 2):
            green_pair = tuple(sorted(greens))
            unmarked = next(i for i in others if i not in green_pair)
            roles.append((red, green_pair, unmarked))
    return roles


_ROLE_CHOICES = _role_choices()


def _gen_attempt() -> np.ndarray | None:
    rng = np.random.default_rng()

    bg = int(rng.choice(_BG_CHOICES))
    sprite_palette = [c for c in range(10) if c not in (bg, _GREY)]
    lane_colors = rng.choice(sprite_palette, size=_NUM_LANES, replace=False).tolist()
    red, greens, unmarked = _ROLE_CHOICES[int(rng.integers(len(_ROLE_CHOICES)))]
    left_green, right_green = greens
    left_target_count, right_target_count = _TARGET_COUNT_PATTERNS[int(rng.integers(len(_TARGET_COUNT_PATTERNS)))]

    input_counts = rng.integers(_MIN_SPRITES, _MAX_SPRITES + 1, size=_NUM_LANES).tolist()
    input_counts[red] = _SOURCE_COUNT
    input_counts[left_green] = left_target_count
    input_counts[right_green] = right_target_count
    input_counts[unmarked] = int(rng.integers(_MIN_SPRITES, _MAX_SPRITES + 1))

    output_counts = input_counts[:]
    output_counts[left_green] = input_counts[red]
    output_counts[right_green] = input_counts[red]

    out = np.full((_H, _W), bg, dtype=int)
    for lane_idx in range(_NUM_LANES):
        color = _GREY if lane_idx == red else lane_colors[lane_idx]
        _paint_lane(out, lane_idx, color, output_counts[lane_idx], bg)

    inp = out.copy()
    _paint_lane(inp, red, lane_colors[red], input_counts[red], bg)

    for green_lane in (left_green, right_green):
        gain = input_counts[red] - input_counts[green_lane]
        c0 = _LANE_STARTS[green_lane]
        for r0 in _lane_tops(output_counts[green_lane])[:gain]:
            inp[r0 : r0 + _SPRITE_H, c0 : c0 + _SPRITE_W] = bg
        inp[_MARKER_ROW, c0 : c0 + _SPRITE_W] = 3

    red_c0 = _LANE_STARTS[red]
    inp[_MARKER_ROW, red_c0 : red_c0 + _SPRITE_W] = 2

    combo_gains = (
        input_counts[red] - input_counts[left_green],
        input_counts[red] - input_counts[right_green],
    )
    old_split = ((input_counts[red] + 1) // 2, input_counts[red] // 2)
    assert combo_gains != old_split, "generator must not collapse to the D12-only even split"

    if not np.array_equal(main(inp), out):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(_GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {_GEN_RETRIES} attempts")
