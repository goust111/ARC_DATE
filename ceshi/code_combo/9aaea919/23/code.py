# Combo variant of 9aaea919 (index 23)
# combo_spec: D12/1+D13/2
# Name: 多源多目标+按数量匹配

import numpy as np

from arc_utils import bbox_coords, components_mask, get_bg, paste


"""
design_analysis:
  a_concretization_and_trace:
    - The canvas, lane geometry, sprite shape, marker-row placement, and grey recoloring all stay the same as the base task.
    - The fused change makes all four lanes participate: exactly two red source lanes and exactly two green target lanes.
    - Each red lane still transfers its full sprite count to exactly one green lane, but the pairing is no longer geometric.
    - Concrete example with lane centers 5, 11, 17, 23:
      red counts = [3, 1] on centers 5 and 11; green counts = [3, 1] on centers 17 and 23.
      Base many:1 rule would send both reds to center 17 because it is the nearest green for both.
      D12-only many:many nearest matching would split them as 5->17 and 11->23.
      The fused rule sorts reds by count high->low and greens by count low->high, so it produces 3-sprite red -> center 23
      and 1-sprite red -> center 17. This is visibly different from both earlier behaviors.
  b_complexity_calibration:
    - Two red and two green markers are mandatory; otherwise either the many:many change or the count-based pairing disappears.
    - Counts inside the red pair and inside the green pair must both be distinct, or the ranking would depend on hidden tie-breaks.
    - The generator also rejects cases where count-ranking accidentally matches the nearest one-to-one pairing, because those
      scenes would hide the D13 contribution.
  c_implicit_assumption_audit:
    - The base solver assumed a single destination could be chosen greedily per red lane; that fails under many:many.
    - It also assumed destination choice depends on x-distance; that fails under the count-ranking rule.
    - The fused solver must therefore:
      1. parse every marked lane into a lane-state object,
      2. compute all red/green pairings globally before placement,
      3. place transfers only after source greying is complete.
    - Placement still relies on unchanged assumptions that sprites are monochrome plus-shapes on a shared period-4 lattice.
  d_placement_legality:
    - Each lane starts with 1..3 sprites.
    - Each green lane receives from exactly one red lane, so its final count is at most 3 + 3 = 6, which fits the 30x30 stack.
    - Output-first generation removes exactly the transferred top slots from each green lane, leaving legal empty targets.
  e_visual_observability:
    - Ranking uses only visible sprite multiplicities, never color ids or any hidden metadata.
    - The generator forces layouts where both reds would compete for one green under the original many:1 nearest rule, so the
      multi-target change is exposed.
    - It separately forces the count-ranked one-to-one pairing to differ from nearest one-to-one pairing, so the sorting rule
      is also exposed from a few examples.
  f_visual_cleanliness:
    - No extra symbols are added: sprites remain inside their own 5-column lanes, markers remain on the last row, and row 28
      stays empty, so the scene remains easy to segment visually.
    - Using all four lanes as marked participants reduces ambiguity about which lanes matter.
  g_color_change_inference:
    - Color behavior is unchanged from the base task: red-marked source stacks become grey, while transferred stacks use the
      pre-existing color of the matched green lane. The markers and preserved lane colors make this recoloring rule legible.
  solver_assumptions_after_change:
    - There are exactly two red and two green marker bars.
    - Every marked lane contains at least one valid monochrome sprite stack.
    - Red counts are distinct and green counts are distinct, so count-ranking is visually determinate.
    - Every matched green lane has enough empty lattice slots for the transferred stack.
"""


def _find_marker_bars(g: np.ndarray, marker_color: int) -> list[list[tuple[int, int]]]:
    """Return thin 4-connected marker bars of the requested color."""
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
        colors = {int(g[r, c]) for r, c in pts}
        if len(colors) == 1:
            out.append(pts)
    return out


def _sorted_sprites(sprites: list[list[tuple[int, int]]]) -> list[tuple[int, int, list[tuple[int, int]]]]:
    items = []
    for pts in sprites:
        rmin = min(r for r, _ in pts)
        cmin = min(c for _, c in pts)
        items.append((rmin, cmin, pts))
    items.sort()
    return items


def _extract_lane_state(g: np.ndarray, marker: list[tuple[int, int]], bg: int) -> dict[str, object] | None:
    """Collect geometry, sprite count, color, and placement state for one marked lane."""
    _, c0, _, c1 = bbox_coords(np.array(marker))
    cols = range(c0, c1 + 1)
    marker_set = set(marker)

    comps = _single_color_components_in_cols(g, cols, bg)
    comps = [[p for p in pts if p not in marker_set] for pts in comps]
    comps = [pts for pts in comps if pts]
    if not comps:
        return None

    non_marker_sizes = [len(pts) for pts in comps if len(pts) != len(marker)]
    if not non_marker_sizes:
        return None
    sprite_size = max(non_marker_sizes)
    sprites = [pts for pts in comps if len(pts) == sprite_size]
    if not sprites:
        return None

    sprites_sorted = _sorted_sprites(sprites)
    tops = sorted({rmin for rmin, _cmin, _pts in sprites_sorted})
    if len(tops) >= 2:
        diffs = [b - a for a, b in zip(tops, tops[1:]) if b - a > 0]
        period = int(np.gcd.reduce(diffs)) if diffs else 4
    else:
        r0, _cc0, r1, _cc1 = bbox_coords(np.array(sprites_sorted[0][2]))
        period = (r1 - r0 + 1) + 1

    ref_top, _ref_left, ref_pts = sprites_sorted[0]
    ref_color = int(g[ref_pts[0][0], ref_pts[0][1]])
    sprite_h = max(r for r, _ in ref_pts) - ref_top + 1
    return {
        "marker": marker,
        "center": (c0 + c1) / 2.0,
        "color": ref_color,
        "count": len(sprites_sorted),
        "sprites": [pts for _r0, _c0, pts in sprites_sorted],
        "tops": tops,
        "period": period,
        "occupied": set(tops),
        "bottom0": max(tops),
        "ref_top": ref_top,
        "ref_pts": ref_pts,
        "sprite_h": sprite_h,
    }


def _ranked_pairs(red_states: list[dict[str, object]], green_states: list[dict[str, object]]) -> list[tuple[dict[str, object], dict[str, object]]]:
    """Pair reds by many->few with greens by few->many."""
    reds = sorted(red_states, key=lambda st: (-int(st["count"]), float(st["center"])))
    greens = sorted(green_states, key=lambda st: (int(st["count"]), float(st["center"])))
    return list(zip(reds, greens))


def _place_transfer(out: np.ndarray, src: dict[str, object], dst: dict[str, object], bg: int) -> None:
    """Place all transferred sprites from one source lane into one matched destination lane."""
    dc_shift = int(round(float(dst["center"]) - float(src["center"])))
    src_offsets = [(r - int(src["ref_top"]), c + dc_shift) for r, c in src["ref_pts"]]

    placed = 0
    needed = int(src["count"])
    step = 0
    while placed < needed:
        new_top = int(dst["bottom0"]) - step * int(dst["period"])
        step += 1
        if new_top < 0:
            break
        if new_top + int(src["sprite_h"]) > out.shape[0]:
            continue
        if new_top in dst["occupied"]:
            continue

        ok = True
        for dr, cc in src_offsets:
            rr = new_top + dr
            if not (0 <= rr < out.shape[0] and 0 <= cc < out.shape[1]):
                ok = False
                break
            if int(out[rr, cc]) != bg:
                ok = False
                break
        if not ok:
            continue

        for dr, cc in src_offsets:
            out[new_top + dr, cc] = int(dst["color"])
        dst["occupied"].add(new_top)
        placed += 1

    assert placed == needed, "destination lane lacked enough empty lattice slots"


def main(grid: np.ndarray) -> np.ndarray:
    """
    Match two red source lanes to two green target lanes by sprite-count ordering, then transfer their stacks.

    Input structure:
    - The dominant background fills a 30x30 board with four fixed 5-column lanes, and the last row contains exactly two red 1x5 bars plus two green 1x5 bars.
    - Every marked lane contains repeated monochrome plus sprites on the same vertical lattice, with one lane-color per lane.

    Rule:
    - Rank the red lanes from most sprites to fewest, and rank the green lanes from fewest sprites to most.
    - Pair lanes by those ranks; each red lane turns grey in place, and its whole sprite stack is recreated inside its paired green lane.
    - Recreated sprites keep the source shape, take the destination lane color, fill empty slots from the bottom upward, and all marker bars disappear.

    Key insight:
    - The solver must first compute a global one-to-one pairing from visible stack sizes, because neither the original single-destination rule nor nearest-lane matching captures the fused behavior.
    """
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)

    red_markers = _find_marker_bars(g, 2)
    green_markers = _find_marker_bars(g, 3)
    assert len(red_markers) == 2 and len(green_markers) == 2, "expected exactly two red and two green markers"

    red_states = [_extract_lane_state(g, rm, bg) for rm in red_markers]
    green_states = [_extract_lane_state(g, gm, bg) for gm in green_markers]
    assert all(st is not None for st in red_states + green_states), "every marked lane must contain a valid sprite stack"
    red_states = [st for st in red_states if st is not None]
    green_states = [st for st in green_states if st is not None]

    pairs = _ranked_pairs(red_states, green_states)
    out = g.copy()

    for src, _dst in pairs:
        for pts in src["sprites"]:
            for r, c in pts:
                out[r, c] = 5

    for src, dst in pairs:
        _place_transfer(out, src, dst, bg)

    for marker in red_markers + green_markers:
        for r, c in marker:
            out[r, c] = bg

    return out


# --------------- generator ---------------

# --- CFG: canvas & layout ---
_H, _W = 30, 30
_NUM_LANES = 4
_LANE_STARTS = [3, 9, 15, 21]
_LANE_CENTERS = [c0 + 2 for c0 in _LANE_STARTS]
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
_NUM_RED = 2
_NUM_GREEN = 2
_MAX_LANE_TOTAL = 6
_GEN_RETRIES = 400

# Chosen so the original many:1 nearest rule would send both reds to the same green lane.
_ROLE_LAYOUTS = [
    ((0, 1), (2, 3)),
    ((2, 3), (0, 1)),
]


def _sprite_patch(color: int, bg: int) -> np.ndarray:
    """Return the 3x5 plus/diamond sprite with transparent corners."""
    patch = np.full((_SPRITE_H, _SPRITE_W), bg, dtype=int)
    patch[0, 1:4] = color
    patch[1, :] = color
    patch[2, 1:4] = color
    return patch


def _lane_tops(count: int) -> list[int]:
    """Top rows of a bottom-aligned stack, ordered from top to bottom."""
    return [_BOTTOM_TOP - _PERIOD * k for k in range(count - 1, -1, -1)]


def _paint_lane(grid: np.ndarray, lane_idx: int, color: int, count: int, bg: int) -> None:
    """Stamp `count` sprites into one fixed lane."""
    patch = _sprite_patch(color, bg)
    c0 = _LANE_STARTS[lane_idx]
    for r0 in _lane_tops(count):
        paste(grid, patch, r0, c0, transparent=bg)


def _ranked_lane_pairs(reds: tuple[int, int], greens: tuple[int, int], counts: list[int]) -> list[tuple[int, int]]:
    red_order = sorted(reds, key=lambda idx: (-counts[idx], _LANE_CENTERS[idx]))
    green_order = sorted(greens, key=lambda idx: (counts[idx], _LANE_CENTERS[idx]))
    return list(zip(red_order, green_order))


def _nearest_one_to_one_pairs(reds: tuple[int, int], greens: tuple[int, int]) -> list[tuple[int, int]]:
    left_red, right_red = sorted(reds, key=lambda idx: _LANE_CENTERS[idx])
    left_green, right_green = sorted(greens, key=lambda idx: _LANE_CENTERS[idx])
    return [(left_red, left_green), (right_red, right_green)]


def _normalized_pairs(pairs: list[tuple[int, int]]) -> tuple[tuple[int, int], ...]:
    """Compare pairings by membership, not by the iteration order that produced them."""
    return tuple(sorted((int(ridx), int(gidx)) for ridx, gidx in pairs))


def _greedy_many_to_one_targets(reds: tuple[int, int], greens: tuple[int, int]) -> dict[int, int]:
    out = {}
    for ridx in reds:
        out[ridx] = min(
            greens,
            key=lambda gidx: (abs(_LANE_CENTERS[gidx] - _LANE_CENTERS[ridx]), _LANE_CENTERS[gidx]),
        )
    return out


def _gen_attempt() -> np.ndarray | None:
    rng = np.random.default_rng()

    bg = int(rng.choice(_BG_CHOICES))
    sprite_palette = [c for c in range(10) if c not in (bg, _GREY)]
    lane_colors = rng.choice(sprite_palette, size=_NUM_LANES, replace=False).tolist()

    reds, greens = _ROLE_LAYOUTS[int(rng.integers(len(_ROLE_LAYOUTS)))]
    counts = rng.integers(_MIN_SPRITES, _MAX_SPRITES + 1, size=_NUM_LANES).tolist()

    if len({counts[i] for i in reds}) != _NUM_RED:
        return None
    if len({counts[i] for i in greens}) != _NUM_GREEN:
        return None

    ranked_pairs = _ranked_lane_pairs(reds, greens, counts)
    nearest_pairs = _nearest_one_to_one_pairs(reds, greens)
    if _normalized_pairs(ranked_pairs) == _normalized_pairs(nearest_pairs):
        return None

    greedy_targets = _greedy_many_to_one_targets(reds, greens)
    if len(set(greedy_targets.values())) != 1:
        return None

    output_counts = counts[:]
    green_gain = {gidx: 0 for gidx in greens}
    for ridx, gidx in ranked_pairs:
        total = counts[gidx] + counts[ridx]
        if total > _MAX_LANE_TOTAL:
            return None
        output_counts[gidx] = total
        green_gain[gidx] = counts[ridx]

    out = np.full((_H, _W), bg, dtype=int)
    for lane_idx in range(_NUM_LANES):
        color = _GREY if lane_idx in reds else lane_colors[lane_idx]
        _paint_lane(out, lane_idx, color, output_counts[lane_idx], bg)

    inp = out.copy()
    for lane_idx in reds:
        _paint_lane(inp, lane_idx, lane_colors[lane_idx], counts[lane_idx], bg)

    for lane_idx in greens:
        gain = green_gain[lane_idx]
        c0 = _LANE_STARTS[lane_idx]
        for r0 in _lane_tops(output_counts[lane_idx])[:gain]:
            inp[r0 : r0 + _SPRITE_H, c0 : c0 + _SPRITE_W] = bg

    for lane_idx in reds:
        c0 = _LANE_STARTS[lane_idx]
        inp[_MARKER_ROW, c0 : c0 + _SPRITE_W] = 2
    for lane_idx in greens:
        c0 = _LANE_STARTS[lane_idx]
        inp[_MARKER_ROW, c0 : c0 + _SPRITE_W] = 3

    if not np.array_equal(main(inp), out):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(_GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {_GEN_RETRIES} attempts")
