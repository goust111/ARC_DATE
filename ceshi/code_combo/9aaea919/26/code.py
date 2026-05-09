# Combo variant of 9aaea919 (index 26)
# combo_spec: D15/0+D18/1
# Name: 允许同色车道+隐形车道

import numpy as np

from arc_utils import bbox_coords, components_mask, get_bg, paste
from itertools import combinations


# Design analysis for the D15/0 + D18/1 combo
# --------------------------------------------
# spec:
#   grid: unchanged 30x30 canvas with 4 fixed 5-column lanes and marker bars on row 29.
#   sprite: unchanged 3x5 monochrome plus/diamond, stacked on a period-4 lattice.
#   action: unchanged transfer rule -- red-marked lanes donate all of their sprites to the
#     green-marked lane, turn grey in place, and all marker bars disappear in the output.
#   D15 change:
#     - Exactly one visible color is duplicated across two non-hidden lanes.
#     - Every red source lane must still differ from the green destination lane.
#   D18 change:
#     - Exactly one unmarked lane uses the background color as its sprite color, so that
#       lane becomes visually absent in both input and output.
#
# a) Concrete behavior and trace:
#   Example layout:
#     lane 0 = hidden unmarked lane with conceptual color bg and count 2,
#     lane 1 = red-marked visible color 6 and count 2,
#     lane 2 = green-marked visible color 8 and count 1,
#     lane 3 = visible unmarked color 8 and count 3.
#   In the input, lane 0 looks empty because its sprites are drawn with the background
#   color, while lanes 2 and 3 visibly share color 8. After the transform, lane 1 turns
#   grey, lane 2 gains two new sprites of color 8 at higher lattice slots, lane 3 stays
#   unchanged, and lane 0 remains invisible. Compared with the base task, the solver can
#   no longer rely on either "all four lanes are visible" or "all visible lane colors are
#   unique"; only marker geometry and fixed lane columns remain dependable.
#
# b) Complexity calibration:
#   The combo is forced, not optional: every generated sample contains exactly one hidden
#   unmarked lane and exactly one duplicated visible color pair. That makes both changes
#   obvious across a few examples without introducing multiple hidden lanes or several
#   independent duplicate pairs that would make the board harder to parse.
#
# c) Implicit-assumption audit:
#   - `_find_marker_bars` depends only on marker geometry and remains valid.
#   - Any logic that infers lanes from global color uniqueness would break under D15, so
#     lane extraction must stay tied to the columns under a given marker.
#   - Any logic that expects all four lanes to be detectable via `g != bg` would break
#     under D18. The solver avoids that by only reading marked lanes; the hidden lane is
#     guaranteed to be unmarked, so invisibility never blocks source or destination lookup.
#   - Destination recolor must come from sprites already present in the green-marked lane,
#     not from a palette invariant.
#   - The generator must explicitly prevent the hidden lane from also being red or green,
#     because a hidden marked lane would violate the solver's visible-sprite expectation.
#
# d) Placement legality:
#   Geometry, spacing, and sprite lattice stay unchanged. The hidden lane still has a
#   conceptual count in 1..3, but painting with `transparent=bg` leaves those sprites
#   invisible and non-overlapping. Output-first construction is retained, and the original
#   `_MAX_LANE_TOTAL` bound still guarantees enough vertical space in the destination lane.
#
# e) Visual observability:
#   Humans can spot the duplicate-color phenomenon because two visible lanes always share a
#   color. The hidden-lane phenomenon is inferable because the task still uses four fixed
#   lane slots at regular spacing, yet one unmarked slot is entirely blank in both input
#   and output. Marker bars on the other lanes preserve the lane scaffold.
#
# f) Visual neatness:
#   Only one lane disappears and only one color pair duplicates. Visible sprite colors are
#   also kept distinct from the red/green marker-bar colors, so the scene remains a clean
#   set of vertical stacks in fixed columns without semantic color collisions.
#
# g) Change inferability:
#   Color behavior remains structurally simple: red-marked sprites become grey, transferred
#   sprites adopt the green lane's existing color, one other visible lane may share that
#   color or a red lane color, and the hidden lane is exactly the lane whose sprite color
#   equals the background. No numeric color ordering or non-visual tie-breakers are used.
#
# main()/generator contract review:
#   - `main()` assumes each marked lane has at least one visible monochrome sprite
#     component; `generate_input()` guarantees the hidden lane is always unmarked.
#   - `main()` assumes the green lane already contains visible sprites to supply its color
#     and lattice period; the generator keeps destination color non-background and places
#     1-3 visible destination sprites before transfer.
#   - `main()` tests candidate placement cells against background only inside the chosen
#     green lane columns; invisible sprites in another lane cannot interfere.


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
        cols_in = {int(g[r, c]) for r, c in pts}
        if len(cols_in) == 1:
            out.append(pts)
    return out


def _sprites_in_marker_lane(
    g: np.ndarray, marker_pts: list[tuple[int, int]], bg: int
) -> tuple[list[list[tuple[int, int]]], float]:
    """Extract visible sprite components inside the columns owned by one marker bar."""
    _, c0, _, c1 = bbox_coords(np.array(marker_pts))
    lane_center = (c0 + c1) / 2.0
    marker_set = set(marker_pts)
    comps = _single_color_components_in_cols(g, range(c0, c1 + 1), bg)
    comps = [[p for p in pts if p not in marker_set] for pts in comps]
    comps = [pts for pts in comps if pts]
    if not comps:
        return [], lane_center

    sprite_sizes = [len(pts) for pts in comps if len(pts) != len(marker_pts)]
    if not sprite_sizes:
        return [], lane_center
    sprite_sz = max(sprite_sizes)
    sprites = [pts for pts in comps if len(pts) == sprite_sz]
    sprites.sort(key=lambda pts: (min(r for r, _ in pts), min(c for _, c in pts)))
    return sprites, lane_center


def _lane_period(sprites: list[list[tuple[int, int]]]) -> int:
    tops = sorted({min(r for r, _ in pts) for pts in sprites})
    if len(tops) >= 2:
        diffs = [b - a for a, b in zip(tops, tops[1:]) if b - a > 0]
        if diffs:
            return int(np.gcd.reduce(diffs))
    r0, _c0, r1, _c1 = bbox_coords(np.array(sprites[0]))
    return (r1 - r0 + 1) + 1


def main(grid: np.ndarray) -> np.ndarray:
    """
    Move sprites from red-marked lanes into the green-marked lane, even when another lane is invisible and two visible lanes share a color.

    Input structure:
    - The background is still the most common color. Thin red (2) and green (3) 1x5 bars on the last row mark the source and destination lanes.
    - The four lane slots stay at fixed columns, but one unmarked lane may have its sprite color equal to the background, making that lane look empty.
    - Among the remaining visible lanes, two lanes may share the same monochrome plus/diamond color, so palette uniqueness is not reliable.

    Rule:
    - Every visible sprite in each red-marked lane turns grey (5) in place.
    - The same number of sprites is inserted into the nearest green-marked lane, filling empty lattice slots from the bottom upward and using the destination lane's own visible color.
    - All marker bars are erased to background, while unmarked lanes stay unchanged whether they were visible or hidden.

    Key insight:
    - The stable signal is lane geometry, not color inventory: read source and destination lanes from their marker columns, count visible source sprites, and ignore the fact that another lane may vanish into the background or mimic a different lane's color.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = get_bg(g)

    red_markers = _find_marker_bars(g, 2)
    green_markers = _find_marker_bars(g, 3)
    assert red_markers and green_markers, "expected both red and green marker bars to be present"

    green_state = {}
    for gm in green_markers:
        dst_sprites, dst_center = _sprites_in_marker_lane(g, gm, bg)
        if not dst_sprites:
            continue

        dst_color = int(g[dst_sprites[0][0][0], dst_sprites[0][0][1]])
        tops = {min(r for r, _ in pts) for pts in dst_sprites}
        green_state[id(gm)] = {
            "center": dst_center,
            "dst_color": dst_color,
            "period": _lane_period(dst_sprites),
            "occupied": set(tops),
            "bottom0": max(tops),
        }

    assert green_state, "expected at least one green lane with visible sprites"

    out = g.copy()

    for rm in red_markers:
        src_sprites, src_center = _sprites_in_marker_lane(g, rm, bg)
        if not src_sprites:
            continue

        for pts in src_sprites:
            for r, c in pts:
                out[r, c] = 5

        best = min(green_state.values(), key=lambda st: abs(st["center"] - src_center))

        src_ref_pts = src_sprites[0]
        src_ref_r0 = min(r for r, _ in src_ref_pts)
        dc_shift = int(round(best["center"] - src_center))
        src_offsets = [(r - src_ref_r0, c + dc_shift) for r, c in src_ref_pts]
        sprite_h = max(dr for dr, _c in src_offsets) + 1

        placed = 0
        n_new = len(src_sprites)
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


# --------------- generator (O3) ---------------

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
_MARKER_COLORS = (2, 3)

# --- CFG: counts & limits ---
_MIN_SPRITES = 1
_MAX_SPRITES = 3
_MAX_LANE_TOTAL = 7
_GEN_RETRIES = 300


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


def _sample_lane_colors(
    rng: np.random.Generator, bg: int, reds: tuple[int, ...], green: int
) -> tuple[int, list[int]]:
    """
    Make exactly one unmarked hidden lane and exactly one visible duplicated-color pair.
    """
    unmarked = [i for i in range(_NUM_LANES) if i not in reds and i != green]
    hidden = unmarked[int(rng.integers(len(unmarked)))]
    visible = [i for i in range(_NUM_LANES) if i != hidden]

    valid_dup_pairs = []
    for pair in combinations(visible, 2):
        if green in pair and any(r in pair for r in reds):
            continue
        valid_dup_pairs.append(pair)
    assert valid_dup_pairs, "expected at least one valid visible duplicate pair"

    dup_pair = valid_dup_pairs[int(rng.integers(len(valid_dup_pairs)))]
    # Keep sprite colors distinct from the red/green marker bars so the combo stays
    # visually legible even when one lane is hidden and another color is duplicated.
    palette = [c for c in range(10) if c not in (bg, _GREY, *_MARKER_COLORS)]
    dup_color, singleton_color = [int(x) for x in rng.choice(palette, size=2, replace=False)]

    lane_colors = [-1] * _NUM_LANES
    lane_colors[hidden] = bg
    lane_colors[dup_pair[0]] = dup_color
    lane_colors[dup_pair[1]] = dup_color
    remaining_visible = [i for i in visible if i not in dup_pair]
    assert len(remaining_visible) == 1
    lane_colors[remaining_visible[0]] = singleton_color

    assert lane_colors[hidden] == bg
    assert all(lane_colors[r] != lane_colors[green] for r in reds)
    visible_non_bg = [lane_colors[i] for i in visible]
    assert len(set(visible_non_bg)) == 2
    dup_count = sum(1 for pair in combinations(visible, 2) if lane_colors[pair[0]] == lane_colors[pair[1]])
    assert dup_count == 1
    return hidden, lane_colors


def _gen_attempt() -> np.ndarray | None:
    rng = np.random.default_rng()

    bg = int(rng.choice(_BG_CHOICES))
    input_counts = rng.integers(_MIN_SPRITES, _MAX_SPRITES + 1, size=_NUM_LANES).tolist()

    feasible = _feasible_roles(input_counts)
    feasible_red_counts = [rc for rc in (1, 2) if feasible[rc]]
    if not feasible_red_counts:
        return None

    red_count = int(rng.choice(feasible_red_counts))
    reds, green = feasible[red_count][int(rng.integers(len(feasible[red_count])))]
    red_set = set(reds)
    hidden, lane_colors = _sample_lane_colors(rng, bg, reds, green)
    gain = sum(input_counts[i] for i in reds)

    output_counts = input_counts[:]
    output_counts[green] += gain
    out = np.full((_H, _W), bg, dtype=int)
    for lane_idx in range(_NUM_LANES):
        color = _GREY if lane_idx in red_set else lane_colors[lane_idx]
        _paint_lane(out, lane_idx, color, output_counts[lane_idx], bg)

    inp = out.copy()
    for lane_idx in reds:
        _paint_lane(inp, lane_idx, lane_colors[lane_idx], input_counts[lane_idx], bg)

    green_c0 = _LANE_STARTS[green]
    for r0 in _lane_tops(output_counts[green])[:gain]:
        inp[r0 : r0 + _SPRITE_H, green_c0 : green_c0 + _SPRITE_W] = bg
    inp[_MARKER_ROW, green_c0 : green_c0 + _SPRITE_W] = 3

    for lane_idx in reds:
        c0 = _LANE_STARTS[lane_idx]
        inp[_MARKER_ROW, c0 : c0 + _SPRITE_W] = 2

    unmarked = {i for i in range(_NUM_LANES) if i not in red_set and i != green}
    visible = [i for i in range(_NUM_LANES) if i != hidden]
    if hidden not in unmarked:
        return None
    if lane_colors[hidden] != bg:
        return None
    if any(lane_colors[i] == bg for i in visible):
        return None
    if any(lane_colors[i] in _MARKER_COLORS for i in visible):
        return None
    if any(lane_colors[i] == lane_colors[j] for i, j in combinations(range(_NUM_LANES), 2) if hidden in (i, j)):
        return None
    if sum(1 for i, j in combinations(visible, 2) if lane_colors[i] == lane_colors[j]) != 1:
        return None
    if any(lane_colors[i] == lane_colors[green] for i in reds):
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
