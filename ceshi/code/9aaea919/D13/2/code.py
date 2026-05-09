# Variant of 9aaea919
# Dimension: D13 — 最近车道匹配策略 (match_cond)
# Change: 红色车道的精灵转移到离它最近(按列中心距离)的绿色车道... → 按精灵数量排序匹配

import numpy as np

from arc_utils import bbox_coords, components_mask, get_bg, paste
from itertools import combinations


"""
design_analysis:
  a_concretization_and_trace:
    scene:
      - The canvas stays 30x30 with four fixed 5-column lanes.
      - Each lane still contains repeated 3x5 plus/diamond sprites on a period-4 vertical lattice.
      - The minimal necessary adaptation for D13 is to mark two lanes red and two lanes green.
      - This keeps the original lane grammar while making "how to pair lanes" visible in the I/O.
    concrete_behavior:
      - Red lanes are sources; their sprites will become grey in the output.
      - Green lanes are destinations; each receives sprites from exactly one red lane.
      - Pairing no longer depends on geometric nearness.
      - Instead, red lanes are ranked by sprite count from many -> few, while green lanes are ranked by sprite count from few -> many.
      - The i-th ranked red lane transfers into the i-th ranked green lane.
    trace_example:
      - Suppose the four lane centers are 5, 11, 17, 23.
      - Red lanes at centers 5 and 11 contain 3 and 1 sprites.
      - Green lanes at centers 17 and 23 contain 3 and 1 sprites.
      - Old rule:
        red@5 -> green@17 and red@11 -> green@17 because both are nearest to center 17.
      - New rule:
        reds sorted = [count 3 at center 5, count 1 at center 11]
        greens sorted = [count 1 at center 23, count 3 at center 17]
        therefore 3-sprite red -> 1-sprite green@23, and 1-sprite red -> 3-sprite green@17.
      - The visible difference is that the large transfer lands in the lane with fewer existing sprites, even when that lane is farther away.
  b_complexity_calibration:
    - Keeping only one green lane would make the new matching rule unobservable, so two green lanes are required.
    - Allowing equal counts on the red side or green side would introduce hidden tie-break dependence.
    - The generator therefore enforces distinct counts among red lanes and distinct counts among green lanes.
    - The generator also rejects scenes where the new sorted pairing matches the original nearest-lane behavior.
  c_implicit_assumption_audit:
    - `_find_marker_bars` still works unchanged because markers remain thin 1x5 bars.
    - `_single_color_components_in_cols` still works because each lane remains monochrome and vertically separated.
    - The original `best = min(green_state.values(), key=abs(center-src_center))` relied on independent per-red greedy matching.
    - That assumption is invalid now; pairing must be computed globally before any placement.
    - The original loop updated one mutable destination state while deciding matches on the fly.
    - Under the new rule, destination ownership must be fixed first so each green lane receives exactly one red lane.
    - Placement logic still assumes all sprites share the same shape and lane widths stay aligned; those properties remain unchanged.
  d_placement_legality:
    - Each lane starts with 1-3 sprites.
    - With one-to-one pairing, each green lane gains at most 3 sprites, so total per green lane is at most 6.
    - The 30x30 grid with bottom top-row 25 and period 4 can fit 7 sprites, so 6 is safely inside bounds.
    - Input construction removes exactly the topmost gained slots from each green lane, leaving empty legal targets for the solver.
  e_visual_observability:
    - Sorting uses only visible sprite multiplicities, never color ids or hidden metadata.
    - Every generated sample has two red counts that differ and two green counts that differ.
    - Every generated sample is rejected if the sorted pairing coincides with nearest matching.
    - This makes the D13 change inferable from a few examples: the destination is driven by "fewest/most sprites", not distance.
  f_visual_cleanliness:
    - The lane spacing and sprite shape are unchanged, so objects remain separated and easy to parse.
    - Marker bars stay on the last row and never touch sprites because row 28 remains empty.
    - All four lanes are marked, which reduces ambiguity about which lanes participate in the transfer.
  g_color_change_inference:
    - Color behavior stays the same as the base task: source lanes turn grey, transferred sprites use the destination lane's color.
    - Because only the matching condition changes, color inference remains structurally supported by lane markers and unchanged sprite shapes.
  solver_assumptions_after_change:
    - Every marked lane has at least one valid sprite component. The generator guarantees 1-3 sprites in each lane.
    - The number of red markers equals the number of green markers. The generator fixes this at 2 and 2.
    - Counts are distinct within the red set and within the green set. The generator enforces this to avoid invisible tie-breaks.
    - For each paired green lane there is enough empty vertical capacity. The generator guarantees total <= 6.
"""


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


def _lane_info(g: np.ndarray, marker_pts: list[tuple[int, int]], bg: int) -> dict[str, object] | None:
    marker_arr = np.array(marker_pts)
    _, c0, _, c1 = bbox_coords(marker_arr)
    lane_cols = range(c0, c1 + 1)
    marker_set = set(marker_pts)

    comps = _single_color_components_in_cols(g, lane_cols, bg)
    comps = [[p for p in pts if p not in marker_set] for pts in comps]
    comps = [pts for pts in comps if pts]
    if not comps:
        return None

    sprite_sizes = [len(pts) for pts in comps if len(pts) != len(marker_pts)]
    if not sprite_sizes:
        return None
    sprite_sz = max(sprite_sizes)
    sprites = [pts for pts in comps if len(pts) == sprite_sz]
    if not sprites:
        return None

    sprites_sorted = []
    for pts in sprites:
        rmin = min(r for r, _ in pts)
        cmin = min(c for _, c in pts)
        sprites_sorted.append((rmin, cmin, pts))
    sprites_sorted.sort()

    ref_r0, _ref_c0, ref_pts = sprites_sorted[0]
    rr, cc = ref_pts[0]
    sprite_color = int(g[rr, cc])

    tops = sorted({rmin for rmin, _cmin, _pts in sprites_sorted})
    if len(tops) >= 2:
        diffs = [b - a for a, b in zip(tops, tops[1:]) if b - a > 0]
        period = int(np.gcd.reduce(diffs)) if diffs else 4
    else:
        r0, _c0, r1, _c1 = bbox_coords(np.array(ref_pts))
        period = (r1 - r0 + 1) + 1

    return {
        "marker": marker_pts,
        "center": (c0 + c1) / 2.0,
        "c0": c0,
        "c1": c1,
        "sprite_color": sprite_color,
        "sprites": sprites,
        "count": len(sprites),
        "period": period,
        "occupied": set(tops),
        "bottom0": max(tops),
        "ref_r0": ref_r0,
        "ref_pts": ref_pts,
    }


def _ranked_pairs(red_infos: list[dict[str, object]], green_infos: list[dict[str, object]]) -> list[tuple[dict[str, object], dict[str, object]]]:
    reds = sorted(red_infos, key=lambda st: (-int(st["count"]), float(st["center"])))
    greens = sorted(green_infos, key=lambda st: (int(st["count"]), float(st["center"])))
    return list(zip(reds, greens))


def main(grid: np.ndarray) -> np.ndarray:
    """
    Transfer sprites from red-marked lanes to green-marked lanes by count-ranked pairing.

    Input structure:
    - The background is the most common color, and thin red (2) / green (3) bars mark source and destination lanes.
    - Each marked lane contains repeated copies of one monochrome sprite shape placed on a regular vertical lattice.

    Rule:
    - Rank red lanes by sprite count from most to fewest, and rank green lanes by sprite count from fewest to most.
    - Pair lanes by rank; each red lane turns grey, and its sprite count is recreated in its paired green lane.
    - New sprites are stacked into empty slots from the bottom upward, recolored to the destination lane's existing sprite color, and all marker bars are removed.

    Key insight: The transfer amount is still conserved lane-by-lane, but the crucial step is a global sorting match across all marked lanes rather than a local nearest-lane choice.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = get_bg(g)

    red_markers = _find_marker_bars(g, 2)
    green_markers = _find_marker_bars(g, 3)
    assert red_markers and green_markers, "expected both red and green marker bars to be present"
    assert len(red_markers) == len(green_markers), "expected equal numbers of red and green lanes"

    red_infos = []
    for rm in red_markers:
        info = _lane_info(g, rm, bg)
        if info is None:
            raise AssertionError("expected every red lane to contain valid sprites")
        red_infos.append(info)

    green_infos = []
    for gm in green_markers:
        info = _lane_info(g, gm, bg)
        if info is None:
            raise AssertionError("expected every green lane to contain valid sprites")
        green_infos.append(info)

    pairs = _ranked_pairs(red_infos, green_infos)
    out = g.copy()

    for src, _dst in pairs:
        for pts in src["sprites"]:
            for r, c in pts:
                out[r, c] = 5

    for src, dst in pairs:
        dc_shift = int(round(float(dst["center"]) - float(src["center"])))
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
                if not (0 <= rr < h and 0 <= c < w):
                    ok = False
                    break
                if int(out[rr, c]) != bg:
                    ok = False
                    break
            if not ok:
                continue

            for dr, c in src_offsets:
                out[new_top + dr, c] = int(dst["sprite_color"])
            dst["occupied"].add(new_top)
            placed += 1

        assert placed == n_new, "expected enough empty slots in each matched green lane"

    for rm in red_markers:
        for r, c in rm:
            out[r, c] = bg
    for gm in green_markers:
        for r, c in gm:
            out[r, c] = bg

    return out


# --------------- generator (D13 variant) ---------------

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


def _paint_lane(grid: np.ndarray, lane_idx: int, color: int, count: int, bg: int) -> None:
    """Stamp `count` sprites of `color` into the given lane."""
    patch = _sprite_patch(color, bg)
    c0 = _LANE_STARTS[lane_idx]
    for r0 in _lane_tops(count):
        paste(grid, patch, r0, c0, transparent=bg)


def _sorted_lane_pairs(reds: tuple[int, ...], greens: tuple[int, ...], counts: list[int]) -> list[tuple[int, int]]:
    red_order = sorted(reds, key=lambda idx: (-counts[idx], _LANE_CENTERS[idx]))
    green_order = sorted(greens, key=lambda idx: (counts[idx], _LANE_CENTERS[idx]))
    return list(zip(red_order, green_order))


def _nearest_targets(reds: tuple[int, ...], greens: tuple[int, ...]) -> dict[int, int]:
    out = {}
    for ridx in reds:
        out[ridx] = min(greens, key=lambda gidx: (abs(_LANE_CENTERS[gidx] - _LANE_CENTERS[ridx]), _LANE_CENTERS[gidx]))
    return out


def _feasible_role_sets() -> list[tuple[tuple[int, ...], tuple[int, ...]]]:
    roles = []
    for reds in combinations(range(_NUM_LANES), _NUM_RED):
        greens = tuple(i for i in range(_NUM_LANES) if i not in reds)
        roles.append((reds, greens))
    return roles


def _gen_attempt() -> np.ndarray | None:
    rng = np.random.default_rng()

    bg = int(rng.choice(_BG_CHOICES))
    sprite_palette = [c for c in range(10) if c not in (bg, _GREY)]
    lane_colors = rng.choice(sprite_palette, size=_NUM_LANES, replace=False).tolist()

    roles = _feasible_role_sets()
    rng.shuffle(roles)

    for reds, greens in roles:
        counts = rng.integers(_MIN_SPRITES, _MAX_SPRITES + 1, size=_NUM_LANES).tolist()
        if len({counts[i] for i in reds}) != _NUM_RED:
            continue
        if len({counts[i] for i in greens}) != _NUM_GREEN:
            continue

        sorted_pairs = _sorted_lane_pairs(reds, greens, counts)
        nearest = _nearest_targets(reds, greens)
        if all(nearest[ridx] == gidx for ridx, gidx in sorted_pairs):
            continue

        output_counts = counts[:]
        green_gain = {gidx: 0 for gidx in greens}
        for ridx, gidx in sorted_pairs:
            if counts[gidx] + counts[ridx] > _MAX_LANE_TOTAL:
                break
            output_counts[gidx] += counts[ridx]
            green_gain[gidx] = counts[ridx]
        else:
            out = np.full((_H, _W), bg, dtype=int)
            red_set = set(reds)
            green_set = set(greens)
            for lane_idx in range(_NUM_LANES):
                color = _GREY if lane_idx in red_set else lane_colors[lane_idx]
                _paint_lane(out, lane_idx, color, output_counts[lane_idx], bg)

            inp = out.copy()
            for lane_idx in reds:
                _paint_lane(inp, lane_idx, lane_colors[lane_idx], counts[lane_idx], bg)

            for lane_idx in greens:
                gain = green_gain[lane_idx]
                c0 = _LANE_STARTS[lane_idx]
                for r0 in _lane_tops(output_counts[lane_idx])[:gain]:
                    inp[r0:r0 + _SPRITE_H, c0:c0 + _SPRITE_W] = bg

            for lane_idx in reds:
                c0 = _LANE_STARTS[lane_idx]
                inp[_MARKER_ROW, c0:c0 + _SPRITE_W] = 2
            for lane_idx in greens:
                c0 = _LANE_STARTS[lane_idx]
                inp[_MARKER_ROW, c0:c0 + _SPRITE_W] = 3

            if not np.array_equal(main(inp), out):
                return None
            return inp

    return None


def generate_input() -> np.ndarray:
    for _ in range(_GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {_GEN_RETRIES} attempts")
