# Variant of 9aaea919
# Dimension: D6 — 源车道残留处理 (trajectory)
# Change: 源车道精灵原位变灰(5)，形状保留但颜色变为灰色... → 源车道精灵保持原色不变

import numpy as np

from arc_utils import bbox_coords, components_mask, get_bg, paste
from itertools import combinations


# ---------------------------------------------------------------------------
# Design analysis for the D6 keep_source variant
# ---------------------------------------------------------------------------
# a) Concretization and behavior trace
# - The grid layout is unchanged from the seed task: 30x30 canvas, four fixed
#   5-column lanes, 3x5 plus/diamond sprites repeated every 4 rows, and 1x5
#   marker bars on the last row.
# - Red bars still mark source lanes and a green bar still marks the single
#   destination lane. The destination lane still receives as many new sprites as
#   the total number of sprites across all red-marked lanes, stacked into empty
#   slots from the bottom upward and recolored to the destination lane color.
# - The only semantic change is what remains in each source lane after the
#   transfer. Original behavior: source sprites stayed in place but were recolored
#   grey(5). Variant behavior: source sprites remain fully unchanged in place,
#   so the visual story is "copy into the green lane" instead of "move away".
# - Concrete trace:
#   input: lane A has 2 red-marked blue sprites, lane B is green-marked with 1
#   yellow sprite. output in the seed would show lane A still containing 2
#   sprites but recolored grey, while lane B would contain 3 yellow sprites.
#   output in this variant shows lane A still containing 2 blue sprites, while
#   lane B still grows to 3 yellow sprites. The marker bars disappear in both.
#
# b) Complexity calibration
# - The task remains visually clean because the same bar markers still identify
#   source and destination roles, and the only changed cue is whether source
#   sprites fade to grey or stay colored.
# - The lane count, sprite count, and destination count limits are kept exactly
#   as before so the puzzle does not gain extra branching or ambiguous matching.
# - The change is still visible in 2-3 examples because every red-marked lane
#   preserves an obvious colored stack in the output rather than turning grey.
#
# c) Implicit-assumption audit of the original main()
# - The original `out = g.copy()` still works: preserving the source now simply
#   means we should stop overwriting source pixels with grey.
# - `_single_color_components_in_cols(...)` still finds the same lane sprites,
#   because source detection depends on lane columns and monochrome connected
#   components, not on the later grey recolor.
# - The original "grey out all source sprites" loop encoded the old D6
#   assumption. In this variant that loop becomes incorrect and must be removed.
# - Destination placement checks against `out[rr, c] == bg` remain valid because
#   new sprites only occupy empty slots in the green lane; keeping source pixels
#   unchanged does not create extra occupancy inside the destination columns.
# - Marker removal remains valid because markers are still auxiliary indicators
#   rather than output objects.
#
# d) Placement legality
# - Generation still enforces that the destination lane total count does not
#   exceed `_MAX_LANE_TOTAL`, which keeps all stacked sprites in-bounds.
# - Lanes are fixed and disjoint, markers are on row 29 while the lowest sprite
#   ends on row 27, so markers never connect to sprites and component extraction
#   remains stable.
# - The output-first generator builds the final copied state first, then derives
#   the input by removing only the gained destination sprites and by adding the
#   markers. This guarantees `main(input) == output` when the solver is correct.
#
# e) Visual observability
# - The rule is inferable from a few examples because the same source lane appears
#   in both input and output with identical colored sprites, while the destination
#   lane grows. No color-id ordering or hidden metadata is used.
#
# f) Visual cleanliness
# - Nothing new overlaps: source lanes stay in their original columns, added
#   sprites appear only in empty destination slots, and marker bars vanish.
#
# g) Color-change inferability
# - The destination recolor rule is unchanged and already structurally signaled
#   by the green-marked destination lane. The source side now has no recolor at
#   all, which is even easier to infer.
#
# h) Behavioral validation target
# - A generated sample is non-degenerate for this variant if at least one
#   red-marked lane exists and the output keeps those source-lane sprite pixels
#   equal to their input colors while the destination lane gains extra sprites.
#   On the same input, the seed solver would differ by painting those source
#   pixels grey.
#
# i) New solver assumptions to keep generator aligned with
# - Each marked lane must still contain at least one valid monochrome sprite
#   component in its 5-column span.
# - The green lane must still contain at least one existing sprite so its color
#   and period can be inferred from the input.
# - Multiple red lanes may map to the same green lane; the generator must keep
#   the total copied count within the lane capacity so placement never runs out
#   of legal upward slots.


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


def main(grid: np.ndarray) -> np.ndarray:
    """
    Copy sprites from red-marked lanes into the green-marked lane while preserving the sources.

    Input structure:
    - The background is the most frequent color. Thin red (2) and green (3) bars on the last row mark lane roles.
    - Each marked or unmarked lane contains repeated identical one-color sprites aligned to a regular vertical lattice.

    Rule:
    - Every red-marked lane contributes copies of all its sprites to the nearest green-marked lane by horizontal center.
    - Each copied sprite uses the source shape but is recolored to the destination lane color and placed into empty slots from the bottom upward.
    - Source-lane sprites stay exactly as they were, and all marker bars are erased.

    Key insight: The red marker means duplication rather than removal: counts are added into the green lane, while the source stacks remain intact.
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
            r0, c0, r1, c1 = bbox_coords(np.array(dst_sprites_sorted[0][2]))
            period = (r1 - r0 + 1) + 1

        gc = (dst_c0 + dst_c1) / 2.0
        green_state[id(gm)] = {
            "marker": gm,
            "center": gc,
            "dst_color": dst_color,
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
        src_comps = _single_color_components_in_cols(g, src_cols, bg)
        src_comps = [[p for p in pts if p not in src_set] for pts in src_comps]
        src_comps = [pts for pts in src_comps if pts]
        if not src_comps:
            continue

        src_sz = max(len(p) for p in src_comps if len(p) != len(rm))
        src_sprites = [p for p in src_comps if len(p) == src_sz]
        if not src_sprites:
            continue

        rc = (src_c0 + src_c1) / 2.0
        best = min(green_state.values(), key=lambda st: abs(st["center"] - rc))

        src_sprites_sorted = []
        for pts in src_sprites:
            rmin = min(r for r, _ in pts)
            cmin = min(c for _, c in pts)
            src_sprites_sorted.append((rmin, cmin, pts))
        src_sprites_sorted.sort()
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
    sprite_palette = [c for c in range(10) if c not in (bg, _GREY)]
    lane_colors = rng.choice(sprite_palette, size=_NUM_LANES, replace=False).tolist()
    input_counts = rng.integers(_MIN_SPRITES, _MAX_SPRITES + 1, size=_NUM_LANES).tolist()

    feasible = _feasible_roles(input_counts)
    feasible_red_counts = [rc for rc in (1, 2) if feasible[rc]]
    if not feasible_red_counts:
        return None

    red_count = int(rng.choice(feasible_red_counts))
    reds, green = feasible[red_count][int(rng.integers(len(feasible[red_count])))]
    gain = sum(input_counts[i] for i in reds)

    output_counts = input_counts[:]
    output_counts[green] += gain
    out = np.full((_H, _W), bg, dtype=int)
    for lane_idx in range(_NUM_LANES):
        _paint_lane(out, lane_idx, lane_colors[lane_idx], output_counts[lane_idx], bg)

    inp = out.copy()

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
