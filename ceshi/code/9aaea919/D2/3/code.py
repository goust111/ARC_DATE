# Variant of 9aaea919
# Dimension: D2 — 源对象类型 (src)
# Change: 十字/菱形精灵(3x5外接矩形，11像素)，每车道单色，均匀垂直排列... → 多色精灵

import numpy as np

from arc_utils import bbox_coords, components_mask, get_bg, paste
from itertools import combinations


# Design analysis for the D2 variant
# a) Concretization and behavior trace:
#    The sprite keeps the original 3x5 plus/diamond silhouette:
#      .AAA.
#      BBBBB
#      .AAA.
#    but each lane now has an ordered two-color scheme (outer band A, center band B).
#    Input still shows 4 fixed vertical lanes plus 1x5 marker bars on the last row.
#    Example: if a red lane contains (8, 4) sprites and the green lane contains (6, 1)
#    sprites, the source sprites turn fully grey in the output while the transferred
#    copies appear in the green lane as .666./11111/.666., stacked into the empty slots.
#    The old solver identified sprites via "single-color component inside lane columns".
#    That breaks here because one sprite now splits into same-color subparts. The variant
#    therefore changes the detection stage: first find whole non-background components per
#    lane, then read their two role colors from the fixed 3x5 shape.
# b) Complexity calibration:
#    The color logic is limited to exactly two structural roles repeated across all sprites:
#    outer 3+3 pixels and center 5 pixels. This is visually obvious in 2-3 examples and
#    does not introduce hidden many-to-many mappings.
# c) Hidden-assumption audit:
#    - Old helper `_single_color_components_in_cols(...)` assumed every sprite is monochrome.
#      That is false now, so it is removed.
#    - Old marker finding tolerated any thin component of color 2/3. In the variant a sprite
#      band may itself be a thin 1x5 strip, so markers are now restricted to the lowest 1x5
#      bars of the marker color.
#    - Old destination recoloring wrote one scalar `dst_color`. That would destroy the second
#      color here, so the solver now carries the destination's full two-color patch.
# d) Placement legality:
#    The generator keeps the original 30x30 canvas, four fixed 5-column lanes, sprite height
#    3, period 4, and bottom top-row 25. This preserves all non-overlap guarantees: sprites
#    only occupy lane-local 3x5 slots, markers stay on row 29, and the green lane total is
#    capped so stacked sprites remain in bounds.
# e) Visual observability:
#    The rule is inferable from examples because each lane repeatedly shows the same ordered
#    two-color sprite, and every transferred sprite adopts the already-visible pair of the
#    destination lane. No sorting or mapping depends on raw color ids.
# f) Visual cleanliness:
#    No extra overlays are added. Lanes remain separated by background columns, marker bars
#    stay isolated below the sprites, and source greying still uses a single neutral color.
# g) Color-change inferability:
#    The recoloring rule is structural rather than symbolic: "outer band becomes destination
#    outer color, center band becomes destination center color". The destination lane itself
#    provides the clue, so humans do not need to infer an arbitrary palette lookup.
# New solver assumptions after the change:
#    - Every valid sprite is one 4-connected 11-cell component with bbox 3x5.
#    - Within a sprite, the six outer-band cells share one color and the five center-row cells
#      share another color, and those colors differ.
#    - Each green lane already contains at least one valid sprite so its ordered two-color
#      scheme can be copied; the generator guarantees that.


_OUTER_CELLS = [(0, 1), (0, 2), (0, 3), (2, 1), (2, 2), (2, 3)]
_CENTER_CELLS = [(1, 0), (1, 1), (1, 2), (1, 3), (1, 4)]
_SPRITE_MASK = np.array(
    [
        [0, 1, 1, 1, 0],
        [1, 1, 1, 1, 1],
        [0, 1, 1, 1, 0],
    ],
    dtype=bool,
)


def _find_marker_bars(g: np.ndarray, marker_color: int) -> list[list[tuple[int, int]]]:
    """
    Marker bars are the lowest horizontal 1x5 components of the given marker color.
    """
    cands = []
    for pts_arr in components_mask(g == marker_color, connectivity=4):
        r0, c0, r1, c1 = bbox_coords(pts_arr)
        hh = r1 - r0 + 1
        ww = c1 - c0 + 1
        if hh == 1 and ww == 5 and len(pts_arr) == 5:
            pts = [(int(r), int(c)) for r, c in pts_arr]
            cands.append((r1, c0, pts))
    if not cands:
        return []
    lowest = max(r1 for r1, _c0, _pts in cands)
    bars = [pts for r1, _c0, pts in sorted(cands) if r1 == lowest]
    return bars


def _sprite_components_in_cols(
    g: np.ndarray, cols: range, bg: int, banned: set[tuple[int, int]]
) -> list[list[tuple[int, int]]]:
    h, w = g.shape
    colmask = np.zeros((h, w), dtype=bool)
    colmask[:, cols] = True
    mask = (g != bg) & colmask
    comps = []
    for pts_arr in components_mask(mask, connectivity=4):
        pts = [(int(r), int(c)) for r, c in pts_arr if (int(r), int(c)) not in banned]
        if not pts:
            continue
        coords = np.array(pts, dtype=int)
        r0, c0, r1, c1 = bbox_coords(coords)
        if (r1 - r0 + 1, c1 - c0 + 1, len(pts)) == (3, 5, 11):
            comps.append(pts)
    return comps


def _ordered_patch_from_pts(g: np.ndarray, pts: list[tuple[int, int]], bg: int) -> tuple[int, int, np.ndarray]:
    coords = np.array(pts, dtype=int)
    r0, c0, r1, c1 = bbox_coords(coords)
    patch = np.full((r1 - r0 + 1, c1 - c0 + 1), bg, dtype=int)
    for r, c in pts:
        patch[r - r0, c - c0] = int(g[r, c])
    assert patch.shape == (3, 5), "expected 3x5 sprite patch"
    assert np.array_equal(patch != bg, _SPRITE_MASK), "expected plus/diamond mask"

    outer_colors = {int(patch[r, c]) for r, c in _OUTER_CELLS}
    center_colors = {int(patch[r, c]) for r, c in _CENTER_CELLS}
    assert len(outer_colors) == 1 and len(center_colors) == 1, "expected two uniform role bands"
    outer = next(iter(outer_colors))
    center = next(iter(center_colors))
    assert outer != center, "expected a true two-color sprite"
    return outer, center, patch


def main(grid: np.ndarray) -> np.ndarray:
    """
    Move sprites from red-marked lanes into the green-marked lane, preserving the two-band shape logic.

    Input structure:
    - The background is the most common color. Red (2) and green (3) 1x5 bars on the lowest row mark lanes.
    - Each lane contains repeated 3x5 plus/diamond sprites on a regular vertical lattice, but every sprite now uses two colors: one for the outer band and one for the full center row.

    Rule:
    - All sprites in a red-marked lane are greyed out in place.
    - The same number of sprites is recreated in the nearest green-marked lane, stacked into empty slots from the bottom upward, using that destination lane's ordered two-color scheme.
    - All marker bars are erased back to the background.

    Key insight: Count transfer is unchanged from the original task; the new challenge is that recoloring is no longer a single fill color, but a structural copy of the destination lane's two-band sprite palette.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = get_bg(g)

    red_markers = _find_marker_bars(g, 2)
    green_markers = _find_marker_bars(g, 3)
    assert red_markers and green_markers, "expected both red and green marker bars to be present"

    green_state = {}
    for gm in green_markers:
        _, dst_c0, _, dst_c1 = bbox_coords(np.array(gm))
        dst_cols = range(dst_c0, dst_c1 + 1)
        dst_set = set(gm)
        dst_sprites = _sprite_components_in_cols(g, dst_cols, bg, dst_set)
        if not dst_sprites:
            continue

        dst_sprites_sorted = []
        for pts in dst_sprites:
            rmin = min(r for r, _ in pts)
            cmin = min(c for _, c in pts)
            dst_sprites_sorted.append((rmin, cmin, pts))
        dst_sprites_sorted.sort()
        _, _, dst_patch = _ordered_patch_from_pts(g, dst_sprites_sorted[0][2], bg)

        tops = sorted({rmin for rmin, _cmin, _pts in dst_sprites_sorted})
        if len(tops) >= 2:
            diffs = [b - a for a, b in zip(tops, tops[1:]) if b - a > 0]
            period = int(np.gcd.reduce(diffs)) if diffs else 4
        else:
            period = dst_patch.shape[0] + 1

        gc = (dst_c0 + dst_c1) / 2.0
        green_state[id(gm)] = {
            "center": gc,
            "dst_patch": dst_patch,
            "period": period,
            "occupied": set(tops),
            "bottom0": max(tops),
        }

    assert green_state, "expected at least one green lane with valid sprites"

    out = g.copy()

    for rm in red_markers:
        _, src_c0, _, src_c1 = bbox_coords(np.array(rm))
        src_cols = range(src_c0, src_c1 + 1)
        src_set = set(rm)
        src_sprites = _sprite_components_in_cols(g, src_cols, bg, src_set)
        if not src_sprites:
            continue

        for pts in src_sprites:
            _ordered_patch_from_pts(g, pts, bg)
            for r, c in pts:
                out[r, c] = 5

        rc = (src_c0 + src_c1) / 2.0
        best = min(green_state.values(), key=lambda st: abs(st["center"] - rc))
        dst_patch = best["dst_patch"]
        shape_offsets = [(int(r), int(c)) for r, c in np.argwhere(dst_patch != bg)]
        sprite_h = dst_patch.shape[0]
        sprite_w = dst_patch.shape[1]
        dst_left = int(round(best["center"] - (sprite_w - 1) / 2.0))

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
            for dr, dc in shape_offsets:
                rr = new_top + dr
                cc = dst_left + dc
                if not (0 <= rr < h and 0 <= cc < w):
                    ok = False
                    break
                if int(out[rr, cc]) != bg:
                    ok = False
                    break
            if not ok:
                continue

            paste(out, dst_patch, new_top, dst_left, transparent=bg)
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

# --- CFG: counts & limits ---
_MIN_SPRITES = 1
_MAX_SPRITES = 3
_MAX_LANE_TOTAL = 7
_GEN_RETRIES = 300


def _sprite_patch(outer_color: int, center_color: int, bg: int) -> np.ndarray:
    """Return a 3x5 two-color plus/diamond sprite patch with bg-transparent corners."""
    patch = np.full((_SPRITE_H, _SPRITE_W), bg, dtype=int)
    patch[0, 1:4] = outer_color
    patch[1, :] = center_color
    patch[2, 1:4] = outer_color
    return patch


def _grey_patch(bg: int) -> np.ndarray:
    """Greyed-out source sprite used in the output."""
    return _sprite_patch(_GREY, _GREY, bg)


def _lane_tops(count: int) -> list[int]:
    """Return top-row positions for `count` sprites, ordered top-to-bottom."""
    return [_BOTTOM_TOP - _PERIOD * k for k in range(count - 1, -1, -1)]


def _paint_lane(grid: np.ndarray, lane_idx: int, patch: np.ndarray, count: int, bg: int) -> None:
    """Stamp `count` identical sprites into the given lane."""
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


def _gen_attempt() -> np.ndarray | None:
    rng = np.random.default_rng()

    bg = int(rng.choice(_BG_CHOICES))
    palette = [c for c in range(10) if c not in (bg, _GREY)]
    rng.shuffle(palette)
    lane_schemes = [(int(palette[i]), int(palette[i + _NUM_LANES])) for i in range(_NUM_LANES)]
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
        patch = _grey_patch(bg) if lane_idx in red_set else _sprite_patch(*lane_schemes[lane_idx], bg)
        _paint_lane(out, lane_idx, patch, output_counts[lane_idx], bg)

    inp = out.copy()
    for lane_idx in reds:
        _paint_lane(inp, lane_idx, _sprite_patch(*lane_schemes[lane_idx], bg), input_counts[lane_idx], bg)

    green_c0 = _LANE_STARTS[green]
    for r0 in _lane_tops(output_counts[green])[:gain]:
        inp[r0 : r0 + _SPRITE_H, green_c0 : green_c0 + _SPRITE_W] = bg
    inp[_MARKER_ROW, green_c0 : green_c0 + _SPRITE_W] = 3

    for lane_idx in reds:
        c0 = _LANE_STARTS[lane_idx]
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
