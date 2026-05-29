# Combo variant of 9aaea919 (index 9)
# combo_spec: D4/0+D6/2
# Name: 保持源色+复制而非移动

import numpy as np

from arc_utils import bbox_coords, components_mask, get_bg, paste
from itertools import combinations


# Design analysis for combo D4/0 + D6/2
# a) Concrete behavior and flow tracking
#    The shared scaffold from the seed task stays fixed: a 30x30 canvas, four
#    5-column lanes, a 3x5 plus/diamond sprite on a period-4 vertical lattice,
#    red 1x5 bars for sources, and one green 1x5 bar for the destination.
#    The seed behavior was "move by count": red-lane sprites visually remained
#    but turned grey, while equally many sprites were created in the green lane
#    using the green lane's color. This combo changes both touched stages:
#    source lanes no longer grey out, and copied sprites no longer recolor.
#    Example trace: suppose lane A has two blue sprites and is red-marked, lane
#    B has one yellow sprite and is green-marked. Seed output: lane A becomes
#    grey, lane B becomes three yellow sprites. D4-only output: lane A becomes
#    grey, lane B becomes one yellow plus two blue. D6-only output: lane A stays
#    blue, lane B becomes three yellow. Combo output: lane A stays blue and lane
#    B becomes one yellow plus two blue. The operation is therefore a pure clone
#    from source lane into destination lane, with no recolor on either side.
# b) Complexity calibration
#    This is not an over-simplified "do nothing" variant because the target lane
#    still grows by a count tied to the red markers, and the copied sprites land
#    in specific empty lattice slots bottom-up. The visible effect is strong:
#    source lanes remain intact, while the destination lane becomes a mixed-color
#    stack containing its old color plus one or two source-lane colors. Colors
#    are still controlled tightly because each lane is monochrome and the task
#    uses at most two source lanes, so the destination shows at most three colors.
# c) Hidden assumptions audit of the seed main()
#    The seed line painting source sprites grey relied on "transfer consumes the
#    source visually"; that assumption is false here, so that overwrite must be
#    removed. The seed line painting new sprites with `dst_color` relied on
#    "transfer adopts destination color"; that is also false here, so copied
#    sprites must use the current source lane color. The rest of the pipeline
#    still holds: marker detection, lane extraction, destination-period
#    inference, nearest-green matching, and bottom-up empty-slot search are all
#    compatible with a copy rule. The only new order-sensitive assumption is that
#    multiple red lanes are processed deterministically; both solver and
#    generator use the marker ordering returned by `_find_marker_bars`, which is
#    left-to-right for these same-row bars.
# d) Placement legality
#    No geometry changes are introduced. The generator still caps each input lane
#    at 1..3 sprites and keeps the destination total at most 7, so copied sprites
#    always fit within rows 1..27 on the fixed lattice. Lanes do not overlap in
#    columns, and the marker row sits below the sprite area, so connected-component
#    parsing stays stable. The generator simulates the same occupied-slot update
#    as the solver when adding copied sprites, preventing overlap or off-grid
#    placements.
# e) Visual observability
#    A human can infer the fusion from a few samples because both changed cues are
#    directly visible: red-marked lanes do not fade, and the destination receives
#    extra sprites in the exact colors that remain in the marked source lanes.
#    Since lane colors are chosen distinct, "kept source color" cannot be confused
#    with coincidental agreement with the destination lane color.
# f) Visual cleanliness
#    The output remains tidy because every sprite stays monochrome and occupies a
#    separate lattice slot. Mixed colors occur only as separate whole sprites in
#    one vertical lane, not as overlapping pixels inside one object. Marker bars
#    still vanish, so the final scene contains only lane stacks.
# g) Change inferability
#    The color behavior has strong structural support: red bars identify the
#    donors, the donors keep their colors in place, and those same colors appear
#    in new destination slots. The count rule is equally visible because the
#    number of added destination sprites matches the total count across red lanes.
# h) Non-degeneracy target for generation
#    A generated sample should differ from the seed task on the same input in two
#    measurable ways: at least one copied destination sprite uses a source color
#    that is not the destination color, and at least one red-lane sprite remains
#    unchanged instead of becoming grey. The generator therefore always preserves
#    distinct per-lane colors and always includes at least one red-marked lane.
# i) New solver assumptions to keep aligned with generation
#    - Each marked lane still contains at least one valid monochrome sprite
#      component within its 5-column band.
#    - The green lane still contains at least one pre-existing sprite so its
#      center, color mix baseline, and vertical period can be inferred.
#    - When two source lanes feed one destination, the generator must add their
#      copies in the same lane order as the solver so the stacked color groups
#      are deterministic and reproducible.


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


def _lane_sprites_for_marker(
    g: np.ndarray, marker_pts: list[tuple[int, int]], bg: int
) -> dict[str, object]:
    """Extract sorted monochrome sprite components that belong to the lane under one marker bar."""
    _r0, c0, _r1, c1 = bbox_coords(np.array(marker_pts))
    lane_cols = range(c0, c1 + 1)
    marker_set = set(marker_pts)
    comps = _single_color_components_in_cols(g, lane_cols, bg)
    comps = [[p for p in pts if p not in marker_set] for pts in comps]
    comps = [pts for pts in comps if pts]

    sprite_entries = []
    if comps:
        sprite_sizes = [len(p) for p in comps if len(p) != len(marker_pts)]
        if sprite_sizes:
            sprite_sz = max(sprite_sizes)
            for pts in comps:
                if len(pts) != sprite_sz:
                    continue
                rmin = min(r for r, _ in pts)
                cmin = min(c for _, c in pts)
                color = int(g[pts[0][0], pts[0][1]])
                sprite_entries.append((rmin, cmin, color, pts))
            sprite_entries.sort()

    return {
        "c0": c0,
        "c1": c1,
        "center": (c0 + c1) / 2.0,
        "sprites": sprite_entries,
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Copy sprites from red-marked lanes into green-marked lanes while preserving source colors and sources.

    Input structure:
    - The background is the most frequent color, and thin red (2) / green (3) bars sit under 5-column lanes.
    - Each lane contains repeated identical monochrome plus sprites placed on a regular vertical lattice.

    Rule:
    - Every red-marked lane contributes copies of all its sprites to the nearest green-marked lane by horizontal center.
    - The copied sprites keep the source lane color instead of adopting the destination lane color, and they fill empty lane slots from the bottom upward.
    - The original source sprites remain unchanged in place, and all marker bars are erased.

    Key insight:
    - The bars describe a cloning relationship rather than a recolor-and-move relationship: the green lane grows by source counts, and the new destination colors are inherited directly from the donor lanes.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = get_bg(g)

    red_markers = _find_marker_bars(g, 2)
    green_markers = _find_marker_bars(g, 3)
    assert red_markers and green_markers, "expected both red and green marker bars to be present"

    green_state = {}
    for gm in green_markers:
        lane = _lane_sprites_for_marker(g, gm, bg)
        dst_sprites = lane["sprites"]
        if not dst_sprites:
            continue

        tops = sorted({rmin for rmin, _cmin, _color, _pts in dst_sprites})
        if len(tops) >= 2:
            diffs = [b - a for a, b in zip(tops, tops[1:]) if b - a > 0]
            period = int(np.gcd.reduce(diffs)) if diffs else 4
        else:
            r0, c0, r1, c1 = bbox_coords(np.array(dst_sprites[0][3]))
            period = (r1 - r0 + 1) + 1

        green_state[id(gm)] = {
            "center": lane["center"],
            "period": period,
            "occupied": set(tops),
            "bottom0": max(tops),
        }

    assert green_state, "expected at least one green lane with valid sprites"

    out = g.copy()

    for rm in red_markers:
        lane = _lane_sprites_for_marker(g, rm, bg)
        src_sprites = lane["sprites"]
        if not src_sprites:
            continue

        rc = lane["center"]
        best = min(green_state.values(), key=lambda st: abs(st["center"] - rc))

        src_ref_r0, _src_ref_c0, src_color, src_ref_pts = src_sprites[0]
        dc_shift = int(round(best["center"] - rc))
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
                out[new_top + dr, c] = src_color
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
_MARKER_COLORS = {2, 3}

# --- CFG: counts & limits ---
_MIN_SPRITES = 1
_MAX_SPRITES = 3
_MAX_LANE_TOTAL = 7
_GEN_RETRIES = 300
_GEN_CALL_INDEX = 0


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


def _gen_attempt(preferred_red_count: int | None = None) -> np.ndarray | None:
    rng = np.random.default_rng()

    bg = int(rng.choice(_BG_CHOICES))
    # Keep lane-object colors distinct from marker colors so the clone rule stays visually explicit.
    sprite_palette = [c for c in range(10) if c not in {bg, _GREY, *_MARKER_COLORS}]
    lane_colors = rng.choice(sprite_palette, size=_NUM_LANES, replace=False).tolist()
    input_counts = rng.integers(_MIN_SPRITES, _MAX_SPRITES + 1, size=_NUM_LANES).tolist()

    feasible = _feasible_roles(input_counts)
    feasible_red_counts = [rc for rc in (1, 2) if feasible[rc]]
    if not feasible_red_counts:
        return None

    if preferred_red_count in feasible_red_counts:
        red_count = preferred_red_count
    else:
        red_count = int(rng.choice(feasible_red_counts))
    reds, green = feasible[red_count][int(rng.integers(len(feasible[red_count])))]
    reds = tuple(sorted(reds))
    red_set = set(reds)

    out = np.full((_H, _W), bg, dtype=int)
    for lane_idx in range(_NUM_LANES):
        _paint_lane(out, lane_idx, lane_colors[lane_idx], input_counts[lane_idx], bg)

    green_c0 = _LANE_STARTS[green]
    occupied = set(_lane_tops(input_counts[green]))
    transferred = []
    for lane_idx in reds:
        patch = _sprite_patch(lane_colors[lane_idx], bg)
        placed = 0
        k = 0
        while placed < input_counts[lane_idx]:
            new_top = _BOTTOM_TOP - _PERIOD * k
            k += 1
            if new_top < 0 or new_top + _SPRITE_H > _H:
                if new_top < 0:
                    return None
                continue
            if new_top in occupied:
                continue
            paste(out, patch, new_top, green_c0, transparent=bg)
            occupied.add(new_top)
            transferred.append((lane_idx, new_top))
            placed += 1

    inp = out.copy()
    for _lane_idx, r0 in transferred:
        inp[r0 : r0 + _SPRITE_H, green_c0 : green_c0 + _SPRITE_W] = bg
    inp[_MARKER_ROW, green_c0 : green_c0 + _SPRITE_W] = 3

    for lane_idx in reds:
        c0 = _LANE_STARTS[lane_idx]
        inp[_MARKER_ROW, c0 : c0 + _SPRITE_W] = 2

    green_lane_colors = {
        int(out[r, c])
        for r in range(_H)
        for c in range(green_c0, green_c0 + _SPRITE_W)
        if int(out[r, c]) != bg
    }
    expected_green_colors = {lane_colors[green], *(lane_colors[l] for l in reds)}
    if green_lane_colors != expected_green_colors:
        return None

    for lane_idx in red_set:
        c0 = _LANE_STARTS[lane_idx]
        src_slice_in = inp[:, c0 : c0 + _SPRITE_W]
        src_slice_out = out[:, c0 : c0 + _SPRITE_W]
        if not np.array_equal(src_slice_in[:_MARKER_ROW], src_slice_out[:_MARKER_ROW]):
            return None

    if not np.array_equal(main(inp), out):
        return None
    return inp


def generate_input() -> np.ndarray:
    global _GEN_CALL_INDEX
    preferred_red_count = 1 if _GEN_CALL_INDEX % 2 == 0 else 2
    for _ in range(_GEN_RETRIES):
        grid = _gen_attempt(preferred_red_count=preferred_red_count)
        if grid is not None:
            _GEN_CALL_INDEX += 1
            return grid
    raise RuntimeError(f"failed to generate a valid input after {_GEN_RETRIES} attempts")
