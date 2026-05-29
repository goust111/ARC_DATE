# Combo variant of 9aaea919 (index 41)
# combo_spec: D4/0+D6/2+D11/0
# Name: 保色部分克隆

from pathlib import Path
import sys
from itertools import combinations

import numpy as np

_SOLVER_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_SOLVER_DIR) not in sys.path:
    sys.path.append(str(_SOLVER_DIR))

from arc_utils import bbox_coords, components_mask, get_bg, paste


# Design analysis for combo D4/0 + D6/2 + D11/0
# ------------------------------------------------
# a) Concrete behavior and flow trace
#    The canvas remains the 30x30 four-lane layout from the seed task: each lane is 5 columns wide, sprites are
#    identical 3x5 plus/diamond shapes, and marker bars still sit on the last row. The fusion changes three linked
#    stages at once:
#    - original: every sprite in a red-marked lane is "consumed", source pixels turn grey, and equally many new
#      sprites appear in the green lane recolored to the green lane color.
#    - combo: a red-marked lane is a copier, not a remover; the source stack stays unchanged, only floor(k/2)
#      copies are added if the lane has k sprites, and each added sprite keeps the source lane color.
#    Example: if a red lane has 3 maroon sprites and the green lane has 2 blue sprites, the seed output would show
#    3 grey maroon-lane sprites and 5 blue green-lane sprites. The combo output instead keeps the 3 maroon sprites
#    intact and adds only 1 new maroon sprite to the green lane, so the destination ends with 2 blue + 1 maroon.
#    With two red lanes, copied sprites still appear in deterministic source-lane groups because we process source
#    lanes left-to-right and fill destination slots bottom-up in that order.
# b) Complexity calibration
#    The change is visible in two independent ways: source lanes no longer fade, and the destination no longer
#    absorbs everything or recolor to a single tone. To keep this legible rather than noisy, the generator preserves
#    the original lane geometry, keeps each lane monochrome, and caps the destination total at 5 sprites. That gives
#    a small number of repeated states a human can compare across examples: original destination color, 1-2 source
#    colors, and a copied count that is strictly smaller than the total red count.
# c) Hidden-assumption audit of the original main()
#    - `out = g.copy()` still works, but the old source-greying loop is now invalid and must disappear entirely.
#    - The old paint step `out[new_top + dr, c] = best["dst_color"]` assumed transferred sprites inherit the
#      destination color. The combo breaks that assumption; paint color must come from the current source lane.
#    - `n_new = len(src_sprites)` assumed complete transfer. The combo changes that to `len(src_sprites) // 2`.
#    - Because source stacks stay unchanged, the exact identity of the copied subset inside a lane is not visually
#      observable; only the count matters. The solver therefore uses the lane template plus copy count, while the
#      generator guarantees that all sprites in a lane are identical so this abstraction is safe.
#    - When two red lanes are both odd, `sum(k_i // 2)` would undershoot `floor(sum(k_i) / 2)`. The generator must
#      avoid that family so the solver's per-lane half-copy rule matches the intended overall-half description.
# d) Placement legality
#    New sprites are still placed only on the destination lane's period-4 lattice, searching empty slots from the
#    bottom upward. The generator checks `counts[green] + gain <= 5`, so inserted sprites stay in bounds and never
#    overlap existing destination sprites. Source lanes are untouched, so they introduce no extra collision risk.
# e) Visual observability
#    A human can see the fusion because red-marked lanes lose neither count nor color, yet the green lane still
#    grows. The destination growth is also smaller than "copy everything", and the new destination sprites reuse the
#    exact source colors. We force at least one contributing red lane with count >= 2, so every generated example
#    shows a non-zero partial copy.
# f) Visual cleanliness
#    No new markers or mixed-color single components are introduced. Every sprite remains a separate monochrome
#    component on the existing lane grid, so even when the destination shows two or three colors, the output stays
#    easy to parse.
# g) Change inferability
#    The color rule is structurally supported: the only colors added to the destination are colors already present in
#    red-marked source lanes, and the original destination color remains present because the green lane starts with at
#    least one sprite. The count rule is supported by repeated small integers (1->0, 2->1, 3->1 per lane), which is
#    learnable from a few examples once the solver sees that red lanes act as sources but do not disappear.
# h) Degeneration target for validation
#    A generated sample is valid only if it differs from the seed behavior on the same input. Here that means:
#    source lanes remain identical to input, the destination gains at least one sprite, and gained count is strictly
#    less than the total number of red-lane sprites. We measure those properties after implementation.
# i) New solver assumptions that the generator must satisfy
#    - Each marked lane must contain at least one valid monochrome sprite component in its 5-column span.
#    - The green lane must already contain at least one sprite so its lattice and baseline occupancy are inferable.
#    - Red lanes are processed left-to-right so grouped source colors stack deterministically in the destination.
#    - If two red lanes are used, they cannot both have odd sprite counts; otherwise the solver's per-lane half-copy
#      count would not equal the intended floor(total_red / 2).


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
    bars = [t[4] for t in cands]
    bars.sort(key=lambda pts: bbox_coords(np.array(pts))[1])
    return bars


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
    """Extract sorted monochrome sprite components that belong to one marked lane."""
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


def _period_from_sorted_sprites(sprites: list[tuple[int, int, int, list[tuple[int, int]]]]) -> int:
    tops = sorted({rmin for rmin, _cmin, _color, _pts in sprites})
    if len(tops) >= 2:
        diffs = [b - a for a, b in zip(tops, tops[1:]) if b - a > 0]
        return int(np.gcd.reduce(diffs)) if diffs else 4
    r0, _c0, r1, _c1 = bbox_coords(np.array(sprites[0][3]))
    return (r1 - r0 + 1) + 1


def main(grid: np.ndarray) -> np.ndarray:
    """
    Partially clone sprites from red-marked lanes into the green-marked lane while preserving source colors.

    Input structure:
    - The background is the most common color. Thin red (2) and green (3) bars mark fixed 5-column lanes.
    - Each lane contains repeated copies of the same monochrome plus/diamond sprite on a regular vertical lattice.

    Rule:
    - A red-marked lane with k sprites keeps its own stack unchanged, but contributes only floor(k/2) copied sprites to the nearest green lane.
    - Each added sprite keeps the source lane color rather than being recolored to the destination lane color.
    - Copied sprites fill empty destination slots from the bottom upward, and all marker bars are removed.

    Key insight:
    - Red markers now signal partial duplication, not removal: only half the stack is cloned, the source colors are preserved, and the destination grows by grouped source-colored copies.
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

        green_state[id(gm)] = {
            "center": lane["center"],
            "period": _period_from_sorted_sprites(dst_sprites),
            "occupied": {rmin for rmin, _cmin, _color, _pts in dst_sprites},
            "bottom0": max(rmin for rmin, _cmin, _color, _pts in dst_sprites),
        }

    assert green_state, "expected at least one green lane with valid sprites"

    out = g.copy()

    for rm in red_markers:
        lane = _lane_sprites_for_marker(g, rm, bg)
        src_sprites = lane["sprites"]
        if not src_sprites:
            continue

        n_new = len(src_sprites) // 2
        if n_new == 0:
            continue

        rc = lane["center"]
        best = min(green_state.values(), key=lambda st: abs(st["center"] - rc))

        src_ref_r0, _src_ref_c0, src_color, src_ref_pts = src_sprites[0]
        dc_shift = int(round(best["center"] - rc))
        src_offsets = [(r - src_ref_r0, c + dc_shift) for r, c in src_ref_pts]
        sprite_h = max(dr for dr, _c in src_offsets) + 1

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

# --- CFG: counts & limits ---
_MIN_SPRITES = 1
_MAX_SPRITES = 3
_MAX_LANE_TOTAL = 5
_GEN_RETRIES = 300
_SCENARIO_ORDER = ((1, 0), (1, 1), (2, 0), (2, 1))
_SCENARIO_CURSOR = 0


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
    """Stamp `count` sprites of one color into the given lane."""
    patch = _sprite_patch(color, bg)
    c0 = _LANE_STARTS[lane_idx]
    for r0 in _lane_tops(count):
        paste(grid, patch, r0, c0, transparent=bg)


def _transfer_counts(counts: list[int], reds: tuple[int, ...]) -> dict[int, int] | None:
    copy_counts = {lane_idx: counts[lane_idx] // 2 for lane_idx in reds}
    if len(reds) == 2 and all(counts[lane_idx] % 2 == 1 for lane_idx in reds):
        return None
    if sum(copy_counts.values()) == 0:
        return None
    return copy_counts


def _feasible_roles(
    counts: list[int],
) -> dict[tuple[int, int], list[tuple[tuple[int, ...], int, dict[int, int], int]]]:
    """
    Return feasible assignments grouped by observable branch type.

    The key is `(red_lane_count, total_red_parity)` so repeated calls can rotate through
    both one-vs-two source lanes and even-vs-odd total source counts.
    """
    grouped: dict[tuple[int, int], list[tuple[tuple[int, ...], int, dict[int, int], int]]] = {
        key: [] for key in _SCENARIO_ORDER
    }
    for green in range(_NUM_LANES):
        others = [i for i in range(_NUM_LANES) if i != green]
        for red_count in (1, 2):
            for reds in combinations(others, red_count):
                copy_counts = _transfer_counts(counts, reds)
                if copy_counts is None:
                    continue
                gain = sum(copy_counts.values())
                if counts[green] + gain <= _MAX_LANE_TOTAL:
                    total_red = sum(counts[lane_idx] for lane_idx in reds)
                    grouped[(red_count, total_red % 2)].append((tuple(sorted(reds)), green, copy_counts, gain))
    return grouped


def _gen_attempt() -> np.ndarray | None:
    global _SCENARIO_CURSOR

    rng = np.random.default_rng()

    bg = int(rng.choice(_BG_CHOICES))
    sprite_palette = [c for c in range(10) if c not in (bg, _GREY)]
    lane_colors = rng.choice(sprite_palette, size=_NUM_LANES, replace=False).tolist()
    input_counts = rng.integers(_MIN_SPRITES, _MAX_SPRITES + 1, size=_NUM_LANES).tolist()

    feasible = _feasible_roles(input_counts)
    available_keys = [key for key in _SCENARIO_ORDER if feasible[key]]
    if not available_keys:
        return None

    chosen_key = None
    for offset in range(len(_SCENARIO_ORDER)):
        key = _SCENARIO_ORDER[(_SCENARIO_CURSOR + offset) % len(_SCENARIO_ORDER)]
        if feasible[key]:
            chosen_key = key
            _SCENARIO_CURSOR = (_SCENARIO_CURSOR + offset + 1) % len(_SCENARIO_ORDER)
            break
    assert chosen_key is not None

    candidates = feasible[chosen_key]
    reds, green, copy_counts, gain = candidates[int(rng.integers(len(candidates)))]

    out = np.full((_H, _W), bg, dtype=int)
    for lane_idx in range(_NUM_LANES):
        base_count = input_counts[lane_idx] + gain if lane_idx == green else input_counts[lane_idx]
        if lane_idx == green:
            _paint_lane(out, lane_idx, lane_colors[lane_idx], input_counts[lane_idx], bg)
        else:
            _paint_lane(out, lane_idx, lane_colors[lane_idx], base_count, bg)

    green_c0 = _LANE_STARTS[green]
    occupied = set(_lane_tops(input_counts[green]))
    transferred_tops: list[tuple[int, int]] = []
    for lane_idx in reds:
        patch = _sprite_patch(lane_colors[lane_idx], bg)
        placed = 0
        k = 0
        while placed < copy_counts[lane_idx]:
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
            transferred_tops.append((lane_idx, new_top))
            placed += 1

    inp = out.copy()
    for lane_idx, r0 in transferred_tops:
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
