# Variant of 9aaea919
# Dimension: D9 — 车道数量与布局 (meta)
# Change: 固定4条车道，列中心分别在5、11、17、23，间距6列... → 不等间距车道

import numpy as np

from arc_utils import bbox_coords, components_mask, get_bg, paste
from itertools import combinations, product


# Design analysis for the D9 variant
# ==================================
# source_task:
# - None. This variant only changes the lane layout dimension of the original task.
#
# a) Concrete behavior trace
# - The lane object itself is still a 5-column slot that carries repeated 3x5 plus sprites and a
#   1x5 marker bar on the bottom row. What changes is only the horizontal placement of those lane
#   slots: instead of fixed starts [3, 9, 15, 21], each sample picks 4 starts whose pairwise gaps
#   are not all equal.
# - Example:
#   Old layout starts [3, 9, 15, 21] means centers [5, 11, 17, 23]. A red lane at start 9 and a
#   green lane at start 21 are easy to read even if a solver silently assumes a period of 6.
#   New layout might be [2, 8, 15, 23], with center gaps 6, 7, 8. The transfer rule is unchanged,
#   but the solver must read the exact columns from each marker/lane rather than stepping left/right
#   by a constant delta.
# - Flow impact:
#   Input construction: lane starts are now sampled per example.
#   Solver: lane extraction should be based on the marker's own occupied columns, not a hard-coded
#   global layout.
#   Output construction: identical transfer logic, but sprites appear in the sampled lane columns.
#
# b) Complexity calibration
# - The variant is visible because the four lane slots no longer form a uniform rhythm across the
#   grid; adjacent gaps vary among a small set of sizes.
# - It is not overcomplicated: lane count stays 4, lane width stays 5, vertical logic stays the
#   same, and only one spatial degree of freedom changes.
# - Layout choices are constrained so there is always at least one background column between lanes.
#   This prevents adjacent same-color markers from merging into one longer red bar.
#
# c) Implicit-assumption review of the original solver
# - Marker detection itself was already layout-agnostic: it finds thin 4-connected bars by color.
# - The lane-component extraction used bbox column ranges from each marker. That still works for
#   irregular spacing, but here it is tightened to use the marker's actual column set, which makes
#   the dependency explicit and avoids any hidden reliance on contiguous global lane stepping.
# - Destination matching by x-center remains valid because the task still maps each red lane to the
#   nearest green lane in screen space; with exactly one green lane this is unambiguous.
# - Vertical period inference from existing destination sprites remains valid because D9 changes no
#   vertical geometry.
#
# d) Placement legality
# - Every lane is 5 columns wide and all starts satisfy a minimum start gap of 6, so lanes never
#   overlap and markers remain separated by at least one background cell.
# - The existing max-lane-total bound still guarantees that the green lane can absorb transferred
#   sprites without leaving the grid.
# - Output-first generation is preserved: build the final state, derive the input by undoing the
#   transfer, then verify with main().
#
# e) Visual observability
# - A human can now see that the lane columns are unevenly spaced from the examples themselves.
#   The rule is still inferable because the markers continue to directly indicate source/destination
#   lanes, and the only new requirement is "track each marked lane where it actually is".
#
# f) Visual cleanliness
# - The grid stays sparse and regular vertically. Keeping lane width, sprite shape, and marker style
#   unchanged prevents the variant from becoming visually noisy.
#
# g) Color-change inference
# - Colors keep the original semantics: red bars mark sources, green bar marks the destination,
#   source sprites become grey, and inserted sprites adopt the destination lane color. No new color
#   mapping is introduced by the D9 change.
#
# 9) New solver assumptions and generator guarantees
# - main() now assumes each lane can be identified from a 1x5 marker or from a visible 5-column
#   sprite stack; generate_input() guarantees this by preserving the original object vocabulary.
# - main() assumes distinct lanes are separated enough that a marker bar is a standalone component;
#   generate_input() guarantees start differences >= 6.
# - main() assumes the destination lane already contains at least one sprite so its color and period
#   can be inferred; generate_input() keeps the original minimum of one sprite per lane.


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


def _single_color_components_in_cols(
    g: np.ndarray, lane_cols: list[int], bg: int
) -> list[list[tuple[int, int]]]:
    h, w = g.shape
    colmask = np.zeros((h, w), dtype=bool)
    colmask[:, lane_cols] = True
    m = (g != bg) & colmask
    comps = [[(int(r), int(c)) for r, c in pts] for pts in components_mask(m, connectivity=4)]
    out = []
    for pts in comps:
        cols_in = {int(g[r, c]) for r, c in pts}
        if len(cols_in) == 1:
            out.append(pts)
    return out


def _lane_sprites_from_marker(
    g: np.ndarray, marker_pts: list[tuple[int, int]], bg: int
) -> list[list[tuple[int, int]]]:
    lane_cols = sorted({c for _r, c in marker_pts})
    marker_set = set(marker_pts)
    comps = _single_color_components_in_cols(g, lane_cols, bg)
    comps = [[p for p in pts if p not in marker_set] for pts in comps]
    comps = [pts for pts in comps if pts]
    if not comps:
        return []
    sprite_sz = max(len(pts) for pts in comps)
    return [pts for pts in comps if len(pts) == sprite_sz]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Move sprites from red-tagged lanes into green-tagged lanes even when lane gaps are irregular.

    Input structure:
    - The background is the most frequent color. Red (2) and green (3) 1x5 bars mark lane columns,
      and the marked lanes can sit at uneven horizontal offsets.
    - Each lane still contains repeated identical monochrome sprites stacked on a shared vertical
      lattice, but the x-position of each lane must be read independently from its own marker.

    Rule:
    - Every sprite in a red-marked lane turns grey (5), and the same number of new sprites is added
      to the nearest green-marked lane using that lane's own color.
    - New sprites fill empty vertical slots from the bottom upward, aligned to the destination lane's
      columns. All marker bars are removed afterward.

    Key insight:
    - The marker bars reveal the exact 5-column footprint of each relevant lane, so the transfer rule
      is unchanged; only the assumption of uniform lane spacing must be discarded.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = get_bg(g)

    red_markers = _find_marker_bars(g, 2)
    green_markers = _find_marker_bars(g, 3)
    assert red_markers and green_markers, "expected both red and green marker bars to be present"

    green_state = {}
    for gm in green_markers:
        dst_sprites = _lane_sprites_from_marker(g, gm, bg)
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
            r0, c0, r1, c1 = bbox_coords(np.array(dst_sprites_sorted[0][2]))
            period = (r1 - r0 + 1) + 1

        lane_cols = sorted({c for _r, c in gm})
        center = float(sum(lane_cols) / len(lane_cols))
        green_state[id(gm)] = {
            "center": center,
            "dst_color": dst_color,
            "period": period,
            "occupied": set(tops),
            "bottom0": max(tops),
        }

    assert green_state, "expected at least one green lane with valid sprites"

    out = g.copy()

    for rm in red_markers:
        src_sprites = _lane_sprites_from_marker(g, rm, bg)
        if not src_sprites:
            continue

        for pts in src_sprites:
            for r, c in pts:
                out[r, c] = 5

        src_cols = sorted({c for _r, c in rm})
        src_center = float(sum(src_cols) / len(src_cols))
        best = min(green_state.values(), key=lambda st: abs(st["center"] - src_center))

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

    for rm in red_markers:
        for r, c in rm:
            out[r, c] = bg
    for gm in green_markers:
        for r, c in gm:
            out[r, c] = bg

    return out


# --------------- generator ---------------

# --- CFG: canvas & layout ---
_H, _W = 30, 30
_NUM_LANES = 4
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
_MAX_LANE_TOTAL = 7
_GEN_RETRIES = 300

# --- CFG: irregular horizontal layouts ---
_START_GAP_CHOICES = (6, 7, 8)


def _build_lane_layouts() -> list[list[int]]:
    layouts = []
    for s0 in range(0, 5):
        for d1, d2, d3 in product(_START_GAP_CHOICES, repeat=3):
            if len({d1, d2, d3}) == 1:
                continue
            starts = [s0, s0 + d1, s0 + d1 + d2, s0 + d1 + d2 + d3]
            if starts[-1] + _SPRITE_W <= _W:
                layouts.append(starts)
    uniq = sorted({tuple(starts) for starts in layouts})
    return [list(starts) for starts in uniq]


_LANE_LAYOUTS = _build_lane_layouts()
assert _LANE_LAYOUTS, "expected at least one irregular lane layout"


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


def _paint_lane(grid: np.ndarray, lane_start: int, color: int, count: int, bg: int) -> None:
    """Stamp `count` sprites of `color` into the given lane start column."""
    patch = _sprite_patch(color, bg)
    for r0 in _lane_tops(count):
        paste(grid, patch, r0, lane_start, transparent=bg)


def _feasible_roles(counts: list[int]) -> dict[int, list[tuple[tuple[int, ...], int]]]:
    """Return feasible (reds, green) assignments grouped by number of red lanes."""
    grouped: dict[int, list[tuple[tuple[int, ...], int]]] = {1: [], 2: []}
    for green in range(_NUM_LANES):
        others = [i for i in range(_NUM_LANES) if i != green]
        for red_count in (1, 2):
            for reds in combinations(others, red_count):
                if counts[green] + sum(counts[i] for i in reds) <= _MAX_LANE_TOTAL:
                    grouped[red_count].append((reds, green))
    return grouped


def _gen_attempt() -> np.ndarray | None:
    rng = np.random.default_rng()

    bg = int(rng.choice(_BG_CHOICES))
    sprite_palette = [c for c in range(10) if c not in (bg, _GREY)]
    lane_colors = rng.choice(sprite_palette, size=_NUM_LANES, replace=False).tolist()
    lane_starts = _LANE_LAYOUTS[int(rng.integers(len(_LANE_LAYOUTS)))]
    input_counts = rng.integers(_MIN_SPRITES, _MAX_SPRITES + 1, size=_NUM_LANES).tolist()

    feasible = _feasible_roles(input_counts)
    feasible_red_counts = [rc for rc in (1, 2) if feasible[rc]]
    if not feasible_red_counts:
        return None

    red_count = int(rng.choice(feasible_red_counts))
    reds, green = feasible[red_count][int(rng.integers(len(feasible[red_count])))]
    red_set = set(reds)
    gain = sum(input_counts[i] for i in reds)

    output_counts = input_counts[:]
    output_counts[green] += gain
    out = np.full((_H, _W), bg, dtype=int)
    for lane_idx in range(_NUM_LANES):
        color = _GREY if lane_idx in red_set else lane_colors[lane_idx]
        _paint_lane(out, lane_starts[lane_idx], color, output_counts[lane_idx], bg)

    inp = out.copy()
    for lane_idx in reds:
        _paint_lane(inp, lane_starts[lane_idx], lane_colors[lane_idx], input_counts[lane_idx], bg)

    green_c0 = lane_starts[green]
    for r0 in _lane_tops(output_counts[green])[:gain]:
        inp[r0 : r0 + _SPRITE_H, green_c0 : green_c0 + _SPRITE_W] = bg
    inp[_MARKER_ROW, green_c0 : green_c0 + _SPRITE_W] = 3

    for lane_idx in reds:
        c0 = lane_starts[lane_idx]
        inp[_MARKER_ROW, c0 : c0 + _SPRITE_W] = 2

    if not np.array_equal(main(inp), out):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(_GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {_GEN_RETRIES} attempts")
