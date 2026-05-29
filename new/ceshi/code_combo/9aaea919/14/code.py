# Combo variant of 9aaea919 (index 14)
# combo_spec: D6/0+D17/0
# Name: 源擦除+旁观车道变灰

import numpy as np

from arc_utils import bbox_coords, components_mask, get_bg, paste
from itertools import combinations


# Design analysis for combo D6/0 + D17/0
#
# a) Concretization and behavior trace:
#    The base task still lives on a 30x30 canvas with four fixed 5-column lanes,
#    bottom-row marker bars, and repeated 3x5 plus/diamond sprites on a period-4
#    vertical lattice. The transfer mechanism is unchanged: red-marked lanes donate
#    their sprite counts to the nearest green-marked lane, which receives added
#    destination-colored copies stacked upward into empty lattice slots.
#    The fusion changes the final state of every non-target lane:
#    - red source lanes no longer leave grey residue; they are fully erased to bg.
#    - unmarked spectator lanes no longer stay untouched; they become grey.
#    Example with lane roles [red, unmarked, green, unmarked] and counts [2, 1, 2, 3]:
#    original output -> [grey x2, original x1, green x4, original x3]
#    D6-only output  -> [blank,    original x1, green x4, original x3]
#    D17-only output -> [grey x2, grey x1,     green x4, grey x3]
#    combo output    -> [blank,    grey x1,     green x4, grey x3]
#    The important difference is that "all non-target lanes lose their color" is
#    now true globally, but red lanes go one step further and disappear entirely.
#
# b) Complexity calibration:
#    The puzzle stays legible because the structural scaffolding is unchanged:
#    fixed lane positions, fixed sprite shape, fixed marker semantics, and the same
#    bottom-up transfer pattern. The visual change is still strong in every sample:
#    at least one unmarked lane must bleach to grey, and at least one red lane must
#    become blank. This is enough to distinguish the fusion from both the original
#    task and either single variant without adding new object classes or hidden rules.
#
# c) Implicit-assumption review:
#    The original solver assumed only red lanes needed post-processing, and that
#    their residue color was always grey. The fusion breaks both assumptions.
#    A robust implementation must therefore:
#    - detect all sprite lanes from the input, not only the marked ones;
#    - apply a global "non-green lanes lose color" rule first;
#    - override red lanes specifically to bg rather than leaving them grey;
#    - keep all transfer counting based on the input grid, not the partially mutated
#      output grid, so the extra greying/erasing never corrupts sprite extraction.
#    The placement loop still assumes destination occupancy is checked against bg only;
#    this remains valid because only the green lane receives new sprites and lanes are
#    column-disjoint under the spec.
#
# d) Placement legality:
#    The generator preserves the original feasibility guard
#    counts[green] + sum(counts[reds]) <= 7, which keeps all stacked target sprites
#    inside the 30-row canvas on the same period-4 lattice. Red lanes are erased, so
#    they cannot overlap anything in the output. Spectator lanes keep their original
#    geometry and count; they only recolor to grey, which introduces no collision mode.
#
# e) Visual observability:
#    A human can infer the combo from a few examples because two independent visual
#    cues repeat across samples: red-marked lanes vanish entirely, while unmarked lanes
#    remain present but turn grey. The target lane is the only lane that keeps a vivid
#    color and gains extra sprites, so the rule "only the target survives in color"
#    is visible without using numeric color ids or hidden ordering.
#
# f) Visual cleanliness:
#    No new overlays are introduced. Outputs remain tidy because each lane still holds
#    one uniform post-state: blank for red lanes, grey for spectator lanes, colored for
#    the green lane. Marker bars are removed at the end, so there is no ambiguity about
#    whether a bottom-row bar is part of a sprite.
#
# g) Change inferability:
#    The only learned color mapping is structural: green-marked destination keeps its
#    own lane color, spectators become grey, and sources become background. This uses
#    just three visibly distinct outcomes tied to lane roles, which is inferable from
#    examples. The solver's new assumptions are: each lane has at least one sprite,
#    the sprite components dominate marker size within a lane, and there is at least
#    one green lane with at least one sprite. The generator enforces all of these.


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


def _lane_sprites_in_cols(
    g: np.ndarray,
    cols: range,
    bg: int,
    ignored: set[tuple[int, int]] | None = None,
) -> list[list[tuple[int, int]]]:
    comps = _single_color_components_in_cols(g, cols, bg)
    if ignored:
        comps = [[p for p in pts if p not in ignored] for pts in comps]
        comps = [pts for pts in comps if pts]
    if not comps:
        return []

    sprite_sizes = [len(pts) for pts in comps]
    sprite_sz = max(sprite_sizes)
    return [pts for pts in comps if len(pts) == sprite_sz]


def _sort_sprites(sprites: list[list[tuple[int, int]]]) -> list[tuple[int, int, list[tuple[int, int]]]]:
    out = []
    for pts in sprites:
        rmin = min(r for r, _ in pts)
        cmin = min(c for _, c in pts)
        out.append((rmin, cmin, pts))
    out.sort()
    return out


def _find_lane_ranges(
    g: np.ndarray,
    bg: int,
    marker_cells: set[tuple[int, int]],
) -> list[tuple[int, int]]:
    mask = g != bg
    for r, c in marker_cells:
        mask[r, c] = False
    ranges = set()
    for pts_arr in components_mask(mask, connectivity=4):
        r0, c0, r1, c1 = bbox_coords(pts_arr)
        hh = r1 - r0 + 1
        ww = c1 - c0 + 1
        if hh > 1 and ww > 1:
            ranges.add((int(c0), int(c1)))
    return sorted(ranges)


def _lane_from_marker(
    g: np.ndarray,
    marker_pts: list[tuple[int, int]],
    bg: int,
) -> tuple[int, int, list[list[tuple[int, int]]]]:
    _, c0, _, c1 = bbox_coords(np.array(marker_pts))
    sprites = _lane_sprites_in_cols(g, range(c0, c1 + 1), bg, ignored=set(marker_pts))
    return c0, c1, sprites


def main(grid: np.ndarray) -> np.ndarray:
    """
    Move red-lane sprite counts into the green lane, while blanking the sources and greying the spectators.

    Input structure:
    - The dominant color is background, and 1x5 red (2) / green (3) bars on the last row mark source and destination lanes.
    - Each of the four fixed-width lanes contains 1-3 identical monochrome plus sprites placed on the same period-4 vertical lattice.

    Rule:
    - Red-marked lanes donate their sprite counts to the nearest green-marked lane; the inserted copies keep the source shape but adopt the destination lane's color and stack upward into empty slots.
    - Any lane that is not the green destination loses its original color in the output: unmarked lanes become grey (5), while red source lanes are erased all the way to background.
    - All marker bars are removed after the transfer.

    Key insight:
    - The transfer count still comes only from the red markers, but the post-state is role-dependent across all lanes: the destination is the only colored survivor, spectators remain as grey evidence, and sources disappear completely.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = get_bg(g)

    red_markers = _find_marker_bars(g, 2)
    green_markers = _find_marker_bars(g, 3)
    assert red_markers and green_markers, "expected both red and green marker bars to be present"

    marker_cells = {pt for bar in red_markers + green_markers for pt in bar}
    lane_ranges = _find_lane_ranges(g, bg, marker_cells)
    assert lane_ranges, "expected at least one sprite lane"

    green_state = {}
    target_ranges = set()
    for gm in green_markers:
        dst_c0, dst_c1, dst_sprites = _lane_from_marker(g, gm, bg)
        target_ranges.add((dst_c0, dst_c1))
        if not dst_sprites:
            continue

        dst_sprites_sorted = _sort_sprites(dst_sprites)
        dst_color = int(g[dst_sprites_sorted[0][2][0][0], dst_sprites_sorted[0][2][0][1]])

        tops = sorted({rmin for rmin, _cmin, _pts in dst_sprites_sorted})
        if len(tops) >= 2:
            diffs = [b - a for a, b in zip(tops, tops[1:]) if b - a > 0]
            period = int(np.gcd.reduce(diffs)) if diffs else 4
        else:
            r0, c0, r1, c1 = bbox_coords(np.array(dst_sprites_sorted[0][2]))
            period = (r1 - r0 + 1) + 1

        gc = (dst_c0 + dst_c1) / 2.0
        green_state[id(gm)] = {
            "center": gc,
            "dst_color": dst_color,
            "period": period,
            "occupied": set(tops),
            "bottom0": max(tops),
        }

    assert green_state, "expected at least one green lane with valid sprites"

    out = g.copy()

    # First apply the global D17 part: every non-target lane loses its original color.
    for c0, c1 in lane_ranges:
        if (c0, c1) in target_ranges:
            continue
        lane_sprites = _lane_sprites_in_cols(g, range(c0, c1 + 1), bg)
        for pts in lane_sprites:
            for r, c in pts:
                out[r, c] = 5

    # Then apply the stronger D6 override for source lanes and perform the transfer.
    for rm in red_markers:
        src_c0, src_c1, src_sprites = _lane_from_marker(g, rm, bg)
        if not src_sprites:
            continue

        for pts in src_sprites:
            for r, c in pts:
                out[r, c] = bg

        rc = (src_c0 + src_c1) / 2.0
        best = min(green_state.values(), key=lambda st: abs(st["center"] - rc))

        src_sprites_sorted = _sort_sprites(src_sprites)
        src_ref_r0, _src_ref_c0, src_ref_pts = src_sprites_sorted[0]

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


def _gen_attempt() -> np.ndarray | None:
    rng = np.random.default_rng()

    bg = int(rng.choice(_BG_CHOICES))
    # Reserve marker colors 2/3 and grey 5 for lane-role semantics so examples stay legible.
    sprite_palette = [c for c in range(10) if c not in (bg, 2, 3, _GREY)]
    lane_colors = rng.choice(sprite_palette, size=_NUM_LANES, replace=False).tolist()
    input_counts = rng.integers(_MIN_SPRITES, _MAX_SPRITES + 1, size=_NUM_LANES).tolist()

    feasible = _feasible_roles(input_counts)
    feasible_red_counts = [rc for rc in (1, 2) if feasible[rc]]
    if not feasible_red_counts:
        return None

    red_count = int(rng.choice(feasible_red_counts))
    reds, green = feasible[red_count][int(rng.integers(len(feasible[red_count])))]
    red_set = set(reds)
    gain = sum(input_counts[i] for i in reds)

    # Output-first fusion state:
    # - green lane keeps its color and gains transferred sprites,
    # - red lanes become empty,
    # - unmarked spectator lanes become grey.
    output_counts = input_counts[:]
    output_counts[green] += gain
    out = np.full((_H, _W), bg, dtype=int)
    for lane_idx in range(_NUM_LANES):
        if lane_idx == green:
            _paint_lane(out, lane_idx, lane_colors[lane_idx], output_counts[lane_idx], bg)
        elif lane_idx not in red_set:
            _paint_lane(out, lane_idx, _GREY, output_counts[lane_idx], bg)

    inp = out.copy()

    # Restore all non-green lanes to their input colors.
    for lane_idx in range(_NUM_LANES):
        if lane_idx == green:
            continue
        _paint_lane(inp, lane_idx, lane_colors[lane_idx], input_counts[lane_idx], bg)

    # Remove transferred sprites from the target lane in the input and add bars.
    green_c0 = _LANE_STARTS[green]
    for r0 in _lane_tops(output_counts[green])[:gain]:
        inp[r0:r0 + _SPRITE_H, green_c0:green_c0 + _SPRITE_W] = bg
    inp[_MARKER_ROW, green_c0:green_c0 + _SPRITE_W] = 3

    for lane_idx in reds:
        c0 = _LANE_STARTS[lane_idx]
        inp[_MARKER_ROW, c0:c0 + _SPRITE_W] = 2

    if not np.array_equal(main(inp), out):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(_GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {_GEN_RETRIES} attempts")
