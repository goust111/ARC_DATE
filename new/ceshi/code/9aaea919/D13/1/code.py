# Variant of 9aaea919
# Dimension: D13 — 最近车道匹配策略 (match_cond)
# Change: 红色车道的精灵转移到离它最近(按列中心距离)的绿色车道... → 固定方向匹配

# Design analysis for the D13 variant
# ----------------------------------
# spec_variant:
#   preserved_structure:
#     grid: 30x30
#     lanes: 4 fixed vertical lanes at columns 3-7 / 9-13 / 15-19 / 21-25
#     sprite: same 3x5 plus shape on a period-4 vertical lattice
#     markers: thin 1x5 bars on the last row, red=source, green=destination
#     recolor: source sprites still become grey(5), inserted sprites still take the destination lane color
#   adapted_for_visibility:
#     green_count: raise from exactly 1 to exactly 2
#     reason: with only one green lane, "nearest green" and "nearest green on the right with wrap"
#       collapse to the same behavior, so D13 would be visually unobservable.
#   concrete_behavior:
#     - Each red lane still donates all of its sprites.
#     - The destination is no longer the globally nearest green lane.
#     - Instead, scan to the right in x-order and choose the first green lane encountered there.
#     - If no green lane lies to the right, wrap around and choose the leftmost green lane.
#     - Green lanes therefore form a cyclic ordered set, not an unordered set of nearest candidates.
#   trace_example:
#     lanes: [0, 1, 2, 3]
#     green lanes: [0, 3]
#     red lane: 1
#     old rule: lane 1 is closer to green lane 0 than to green lane 3, so sprites move left.
#     new rule: lane 1 must ignore the closer left green and move to the nearest green strictly on its right,
#       namely lane 3, so the added stack appears on the far right.
#     visible_difference:
#       the output growth shifts to a different lane while all other cues stay the same.
#   complexity_check:
#     - Keeping only two green lanes avoids a messy many-way routing problem.
#     - Requiring at least one red lane whose right-match differs from nearest-match prevents degeneration back
#       to the original task.
#     - The sprite shape and lane spacing remain unchanged, so the task is still readable from a few examples.
#   implicit-assumption_audit:
#     - Source discovery code assumes each marked lane still contains at least one monochrome sprite component.
#       The generator preserves the original 1-3 sprites per lane.
#     - Destination state code assumes each green-marked lane already contains at least one sprite so that its
#       color, period, and bottom anchor can be inferred. The generator preserves that too.
#     - Placement code assumes all transferred sprites fit into empty lattice slots of the chosen green lane.
#       The generator now checks capacity after routing every red lane through the new directed matcher.
#     - Matching code now assumes green lanes have a stable left-to-right order by column center. Fixed lane
#       columns guarantee this, and the solver explicitly uses column centers rather than any color ids.
#   legality_and_cleanliness:
#     - All objects stay on the same non-overlapping fixed lane lattice.
#     - Marker bars remain isolated on row 29, away from the sprite stacks.
#     - Output-first generation erases only the gained top slots of each green lane, which preserves a tidy visual
#       separation between original and transferred stacks.
#   observability:
#     - The generator only accepts role assignments where at least one red lane routes differently under the
#       new rule than under the old nearest rule.
#     - This creates training pairs in which a human can see "it always goes right, even when a left green lane
#       is closer" or "it wraps from the last lane back to the leftmost green lane".

import numpy as np

from arc_utils import bbox_coords, components_mask, get_bg, paste
from itertools import combinations


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


def _match_right_green(src_center: float, green_states: list[dict]) -> dict:
    """Choose the nearest green lane strictly to the right; wrap to the leftmost green lane if needed."""
    right = [st for st in green_states if st["center"] > src_center]
    if right:
        return min(right, key=lambda st: st["center"] - src_center)
    return min(green_states, key=lambda st: st["center"])


def main(grid: np.ndarray) -> np.ndarray:
    """
    Route sprites out of red-marked lanes into green-marked lanes, then restack and recolor them.

    Input structure:
    - The background is the dominant color, and the final row contains thin marker bars: red bars mark donor lanes,
      green bars mark eligible receiver lanes.
    - Each lane is a fixed 5-column slot containing repeated copies of the same monochrome plus sprite on a regular
      vertical lattice.

    Rule:
    - Every sprite in a red lane turns grey in place, and the same number of sprites is inserted into a green lane.
    - The receiver is directional: use the nearest green lane strictly to the right of the red lane; if none exists,
      wrap around to the leftmost green lane.
    - Inserted sprites keep the source shape, shift into the destination lane, take the destination lane's color,
      stack upward from the bottom into empty period-aligned slots, and all marker bars disappear.

    Key insight:
    - The green markers must be read as an ordered cyclic set of targets, so horizontal direction matters more than
      raw nearest-distance matching.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = get_bg(g)

    red_markers = _find_marker_bars(g, 2)
    green_markers = _find_marker_bars(g, 3)
    assert red_markers and green_markers, "expected both red and green marker bars to be present"

    # Build destination lane state per green marker.
    green_states = []
    for gm in green_markers:
        _, dst_c0, _, dst_c1 = bbox_coords(np.array(gm))
        dst_cols = range(dst_c0, dst_c1 + 1)
        dst_set = set(gm)
        dst_comps = _single_color_components_in_cols(g, dst_cols, bg)
        dst_comps = [[p for p in pts if p not in dst_set] for pts in dst_comps]
        dst_comps = [pts for pts in dst_comps if pts]
        if not dst_comps:
            continue

        dst_sz = max(len(p) for p in dst_comps if len(p) != len(gm))
        dst_sprites = [p for p in dst_comps if len(p) == dst_sz]
        if not dst_sprites:
            continue

        dst_sprites_sorted = []
        for pts in dst_sprites:
            rmin = min(r for r, _ in pts)
            cmin = min(c for _, c in pts)
            dst_sprites_sorted.append((rmin, cmin, pts))
        dst_sprites_sorted.sort()
        dst_color = int(g[dst_sprites_sorted[0][2][0][0], dst_sprites_sorted[0][2][0][1]])

        tops = sorted({rmin for rmin, _cmin, _pts in dst_sprites_sorted})
        if len(tops) >= 2:
            diffs = [b - a for a, b in zip(tops, tops[1:]) if b - a > 0]
            period = int(np.gcd.reduce(diffs)) if diffs else 4
        else:
            r0, _c0, r1, _c1 = bbox_coords(np.array(dst_sprites_sorted[0][2]))
            period = (r1 - r0 + 1) + 1

        gc = (dst_c0 + dst_c1) / 2.0
        green_states.append(
            {
                "marker": gm,
                "center": gc,
                "dst_color": dst_color,
                "period": period,
                "occupied": set(tops),
                "bottom0": max(tops),
            }
        )

    assert green_states, "expected at least one green lane with valid sprites"

    out = g.copy()

    for rm in red_markers:
        _, src_c0, _, src_c1 = bbox_coords(np.array(rm))
        src_cols = range(src_c0, src_c1 + 1)
        src_set = set(rm)
        src_comps = _single_color_components_in_cols(g, src_cols, bg)
        src_comps = [[p for p in pts if p not in src_set] for pts in src_comps]
        src_comps = [pts for pts in src_comps if pts]
        if not src_comps:
            continue

        src_sz = max(len(p) for p in src_comps if len(p) != len(rm))
        src_sprites = [p for p in src_comps if len(p) == src_sz]
        if not src_sprites:
            continue

        # Grey out all source sprites.
        for pts in src_sprites:
            for r, c in pts:
                out[r, c] = 5

        src_center = (src_c0 + src_c1) / 2.0
        best = _match_right_green(src_center, green_states)

        # Source template (top-most sprite), shifted horizontally to the destination lane center.
        src_sprites_sorted = []
        for pts in src_sprites:
            rmin = min(r for r, _ in pts)
            cmin = min(c for _, c in pts)
            src_sprites_sorted.append((rmin, cmin, pts))
        src_sprites_sorted.sort()
        src_ref_r0, _src_ref_c0, src_ref_pts = src_sprites_sorted[0]

        dc_shift = int(round(best["center"] - src_center))
        src_offsets = [(r - src_ref_r0, c + dc_shift) for r, c in src_ref_pts]
        sprite_h = max(dr for dr, _c in src_offsets) + 1

        # Place as many sprites as removed, filling empty slots from the bottom upward.
        n_new = len(src_sprites)
        placed = 0
        k = 0
        while placed < n_new:
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
                out[new_top + dr, c] = best["dst_color"]
            best["occupied"].add(new_top)
            placed += 1

    # Remove all marker bars.
    for rm in red_markers:
        for r, c in rm:
            out[r, c] = bg
    for gm in green_markers:
        for r, c in gm:
            out[r, c] = bg

    return out


# --------------- generator (O3) ---------------

# --- CFG: canvas & layout ---
_H, _W = 30, 30
_NUM_LANES = 4
_LANE_STARTS = [3, 9, 15, 21]       # col-0 of each lane's 5-wide sprite slot
_LANE_CENTERS = [c0 + 2 for c0 in _LANE_STARTS]
_MARKER_ROW = 29                     # row for red/green 1x5 marker bars
_SPRITE_H, _SPRITE_W = 3, 5         # bounding box of the plus/diamond sprite

# --- CFG: vertical stacking ---
_BOTTOM_TOP = 25                     # top-row of the bottommost sprite
_PERIOD = 4                          # vertical period (3 sprite rows + 1 gap)

# --- CFG: color palette ---
_BG_CHOICES = [0, 1, 4, 6, 7, 8, 9] # valid background colors (excludes 2, 3, 5)
_GREY = 5                            # color used for greyed-out source sprites

# --- CFG: counts & limits ---
_MIN_SPRITES = 1                     # min sprites per lane in input
_MAX_SPRITES = 3                     # max sprites per lane in input
_MAX_LANE_TOTAL = 7                  # max sprites in a receiving green lane after transfer
_NUM_GREENS = 2                      # needed so the directed matching rule is visible
_GEN_RETRIES = 300                   # retry budget for generate_input()


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


def _nearest_green_lane(src_lane: int, green_lanes: tuple[int, ...]) -> int:
    """Old behavior baseline, including leftmost tie-break to mimic min(abs(center)))."""
    src_center = _LANE_CENTERS[src_lane]
    return min(green_lanes, key=lambda lane: (abs(_LANE_CENTERS[lane] - src_center), _LANE_CENTERS[lane]))


def _right_green_lane(src_lane: int, green_lanes: tuple[int, ...]) -> int:
    """New directed behavior: nearest green strictly to the right, else wrap to the leftmost green."""
    right = [lane for lane in green_lanes if _LANE_CENTERS[lane] > _LANE_CENTERS[src_lane]]
    if right:
        return min(right, key=lambda lane: _LANE_CENTERS[lane] - _LANE_CENTERS[src_lane])
    return min(green_lanes, key=lambda lane: _LANE_CENTERS[lane])


def _feasible_roles(counts: list[int]) -> list[tuple[tuple[int, ...], tuple[int, ...]]]:
    """
    Return feasible (reds, greens) assignments that force the D13 rule to matter.

    The accepted assignments must satisfy:
    - exactly two green lanes,
    - one or two red lanes,
    - at least one red lane routes differently under right-match vs nearest-match,
    - each receiving green lane stays within the stack capacity.
    """
    feasible = []
    for greens in combinations(range(_NUM_LANES), _NUM_GREENS):
        green_set = set(greens)
        others = [i for i in range(_NUM_LANES) if i not in green_set]
        for red_count in (1, 2):
            for reds in combinations(others, red_count):
                targets = {lane: counts[lane] for lane in greens}
                for src in reds:
                    targets[_right_green_lane(src, greens)] += counts[src]
                if any(total > _MAX_LANE_TOTAL for total in targets.values()):
                    continue
                if not any(_right_green_lane(src, greens) != _nearest_green_lane(src, greens) for src in reds):
                    continue
                feasible.append((reds, greens))
    return feasible


def _build_output(
    lane_colors: list[int], input_counts: list[int], reds: tuple[int, ...], greens: tuple[int, ...], bg: int
) -> tuple[np.ndarray, dict[int, int]]:
    """Build the expected output grid and report how many sprites each green lane gains."""
    red_set = set(reds)
    gains_by_green = {lane: 0 for lane in greens}
    for src in reds:
        gains_by_green[_right_green_lane(src, greens)] += input_counts[src]

    output_counts = input_counts[:]
    for lane, gain in gains_by_green.items():
        output_counts[lane] += gain

    out = np.full((_H, _W), bg, dtype=int)
    for lane_idx in range(_NUM_LANES):
        color = _GREY if lane_idx in red_set else lane_colors[lane_idx]
        _paint_lane(out, lane_idx, color, output_counts[lane_idx], bg)
    return out, gains_by_green


def _gen_attempt() -> np.ndarray | None:
    rng = np.random.default_rng()

    # Choose palette
    bg = int(rng.choice(_BG_CHOICES))
    sprite_palette = [c for c in range(10) if c not in (bg, _GREY)]
    lane_colors = rng.choice(sprite_palette, size=_NUM_LANES, replace=False).tolist()
    input_counts = rng.integers(_MIN_SPRITES, _MAX_SPRITES + 1, size=_NUM_LANES).tolist()

    feasible = _feasible_roles(input_counts)
    if not feasible:
        return None

    reds, greens = feasible[int(rng.integers(len(feasible)))]
    out, gains_by_green = _build_output(lane_colors, input_counts, reds, greens, bg)

    # Derive INPUT from output.
    inp = out.copy()
    for lane_idx in reds:
        _paint_lane(inp, lane_idx, lane_colors[lane_idx], input_counts[lane_idx], bg)

    # Erase the gained sprites from each receiving green lane and add markers.
    for lane_idx, gain in gains_by_green.items():
        c0 = _LANE_STARTS[lane_idx]
        total = input_counts[lane_idx] + gain
        for r0 in _lane_tops(total)[:gain]:
            inp[r0:r0 + _SPRITE_H, c0:c0 + _SPRITE_W] = bg
        inp[_MARKER_ROW, c0:c0 + _SPRITE_W] = 3

    for lane_idx in reds:
        c0 = _LANE_STARTS[lane_idx]
        inp[_MARKER_ROW, c0:c0 + _SPRITE_W] = 2

    # Oracle check
    if not np.array_equal(main(inp), out):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(_GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {_GEN_RETRIES} attempts")
