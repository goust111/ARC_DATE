# Variant of 9aaea919
# Dimension: D11 — 转移数量规则 (meta)
# Change: 转移数量N = 红色车道中所有精灵的总数(1对1完全转移)... → 转移数量减半

# Design analysis for the D11 variant
# ----------------------------------
# a) Concrete behavior and trace
#    - The canvas, four fixed 5-column lanes, 1x5 marker bars, and 3x5 plus sprites stay unchanged.
#    - The visible change is in red-marked source lanes: only the lower half of each red stack is treated
#      as "consumed". Those consumed sprites turn grey in place; the upper remainder stays in its original
#      lane color. The destination green lane still receives recolored copies stacked upward from the bottom.
#    - Example with one red lane holding 3 sprites and one green lane holding 2 sprites:
#      input  : red lane has 3 colored sprites, green lane has 2 colored sprites, markers are present.
#      output : red lane keeps its top 2 sprites colored, its bottom 1 sprite becomes grey, and the green
#               lane gains exactly 1 extra sprite at the highest newly filled slot.
#      Original task would have greyed all 3 source sprites and added 3 new destination sprites instead.
#      The D11 difference is therefore immediately visible in both the source lane and the destination lane.
#
# b) Complexity calibration
#    - The key novelty should be "half transfer", not a second hidden ordering puzzle. To keep the rule
#      legible, the moved subset is always a contiguous bottom tail inside each red lane.
#    - Two red markers are still allowed, but the generator avoids the case where both red lanes have odd
#      counts. That avoids needing an extra global tie-break rule for the leftover odd sprite while still
#      making the total number of transferred sprites equal floor(total_red / 2).
#
# c) Implicit assumptions in the original solver that must change
#    - `n_new = len(src_sprites)` assumed complete transfer. That is no longer valid; it must become the
#      size of the movable tail only.
#    - Greying every source sprite relied on "all removed". In this variant, only selected sprites should
#      be greyed, otherwise the output collapses back toward the original task.
#    - The destination placement logic can stay intact because shape, lane geometry, and bottom-up stacking
#      are unchanged; only the count of newly inserted sprites changes.
#
# d) Placement legality
#    - Source lanes keep their original stack positions; only recoloring changes there, so no new overlap
#      risk is introduced.
#    - Destination additions still occupy empty lattice slots above the existing green stack. The generator
#      only accepts role assignments whose post-transfer green count fits in the 30x30 canvas.
#    - Because each red lane contributes `count // 2`, the maximum green total becomes 5 instead of 7.
#
# e) Visual observability
#    - Every generated puzzle has at least one transferred sprite, so the variant is not degenerate.
#    - Each active red lane that contributes transfer shows both colors at once in the output: original
#      color on top, grey on the moved bottom tail. This makes "partial transfer" visible without relying
#      on hidden metadata or color-id ordering.
#
# f) Visual cleanliness
#    - No new objects or extra markers are introduced. The same lane-separated layout keeps the scene easy
#      to parse, and the partial greying happens inside an already well-isolated vertical stack.
#
# g) Color-change inferability
#    - Recoloring still follows the original structural cue: new sprites use the destination lane color.
#      Grey still means "consumed in source". The variant changes quantity, not the color mapping system.
#
# h) Variant-specific generator audit target
#    - For generated samples we later check that `actual_gain < total_red_sprites` always holds; otherwise
#      a sample would behave like the original full-transfer task and would count as a degeneration.
#
# i) New solver assumptions that the generator must satisfy
#    - If there are two red lanes, at least one must have an even count so that
#      `sum(count_i // 2) == floor(sum(count_i) / 2)`.
#    - At least one red lane must have count >= 2 so that the puzzle actually exhibits a non-zero transfer.

from pathlib import Path
import sys
from itertools import combinations

import numpy as np

_SOLVER_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_SOLVER_DIR) not in sys.path:
    sys.path.append(str(_SOLVER_DIR))

from arc_utils import bbox_coords, components_mask, get_bg, paste


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


def _sorted_lane_sprites(g: np.ndarray, marker_pts: list[tuple[int, int]], bg: int) -> list[list[tuple[int, int]]]:
    _, c0, _, c1 = bbox_coords(np.array(marker_pts))
    lane_cols = range(c0, c1 + 1)
    marker_set = set(marker_pts)
    comps = _single_color_components_in_cols(g, lane_cols, bg)
    comps = [[p for p in pts if p not in marker_set] for pts in comps]
    comps = [pts for pts in comps if pts]
    if not comps:
        return []

    sprite_sz = max(len(p) for p in comps)
    sprites = [p for p in comps if len(p) == sprite_sz]

    decorated = []
    for pts in sprites:
        rmin = min(r for r, _ in pts)
        cmin = min(c for _, c in pts)
        decorated.append((rmin, cmin, pts))
    decorated.sort()
    return [pts for _rmin, _cmin, pts in decorated]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Move only part of each red-marked source stack into the green-marked destination stack.

    Input structure:
    - The background is the most frequent color. Thin red (2) and green (3) bars mark lane roles.
    - Each marked lane contains identical monochrome plus sprites arranged on a fixed vertical lattice.

    Rule:
    - In a red lane with k sprites, only the bottom floor(k/2) sprites are treated as transferred.
    - Those transferred source sprites turn grey (5) in place, and the same number of copies are added
      to the nearest green lane, recolored to that destination lane's sprite color and stacked bottom-up.
    - Untransferred source sprites keep their original color, and all marker bars are removed.

    Key insight:
    - A red marker no longer means "move the whole stack". It means "consume only the lower half", so the
      destination gains half as many sprites as the marked source stacks contribute overall.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = get_bg(g)

    red_markers = _find_marker_bars(g, 2)
    green_markers = _find_marker_bars(g, 3)
    assert red_markers and green_markers, "expected both red and green marker bars to be present"

    green_state = {}
    for gm in green_markers:
        dst_sprites = _sorted_lane_sprites(g, gm, bg)
        if not dst_sprites:
            continue

        _, dst_c0, _, dst_c1 = bbox_coords(np.array(gm))
        dst_color = int(g[dst_sprites[0][0][0], dst_sprites[0][0][1]])

        tops = sorted({min(r for r, _ in pts) for pts in dst_sprites})
        if len(tops) >= 2:
            diffs = [b - a for a, b in zip(tops, tops[1:]) if b - a > 0]
            period = int(np.gcd.reduce(diffs)) if diffs else 4
        else:
            r0, _c0, r1, _c1 = bbox_coords(np.array(dst_sprites[0]))
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

    for rm in red_markers:
        src_sprites = _sorted_lane_sprites(g, rm, bg)
        if not src_sprites:
            continue

        n_new = len(src_sprites) // 2
        moved_sprites = src_sprites[-n_new:] if n_new else []

        for pts in moved_sprites:
            for r, c in pts:
                out[r, c] = 5

        _, src_c0, _, src_c1 = bbox_coords(np.array(rm))
        rc = (src_c0 + src_c1) / 2.0
        best = min(green_state.values(), key=lambda st: abs(st["center"] - rc))

        src_ref_pts = src_sprites[0]
        src_ref_r0 = min(r for r, _ in src_ref_pts)
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
_MAX_LANE_TOTAL = 5
_GEN_RETRIES = 300


def _sprite_patch(color: int, bg: int) -> np.ndarray:
    """Return a 3x5 plus sprite patch with bg-transparent corners."""
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


def _paint_lane_with_grey_tail(
    grid: np.ndarray, lane_idx: int, color: int, count: int, grey_tail: int, bg: int
) -> None:
    """Paint a lane where only the bottom `grey_tail` sprites are greyed out."""
    tops = _lane_tops(count)
    c0 = _LANE_STARTS[lane_idx]
    base_patch = _sprite_patch(color, bg)
    grey_patch = _sprite_patch(_GREY, bg)
    grey_tops = set(tops[-grey_tail:]) if grey_tail else set()
    for r0 in tops:
        patch = grey_patch if r0 in grey_tops else base_patch
        paste(grid, patch, r0, c0, transparent=bg)


def _transfer_gain(counts: list[int], reds: tuple[int, ...]) -> int | None:
    red_counts = [counts[i] for i in reds]
    if len(red_counts) == 2 and all(c % 2 == 1 for c in red_counts):
        return None
    gain = sum(c // 2 for c in red_counts)
    if gain == 0:
        return None
    return gain


def _feasible_roles(counts: list[int]) -> dict[int, list[tuple[tuple[int, ...], int, int]]]:
    """Return feasible (reds, green, gain) assignments grouped by number of red lanes."""
    grouped: dict[int, list[tuple[tuple[int, ...], int, int]]] = {1: [], 2: []}
    for green in range(_NUM_LANES):
        others = [i for i in range(_NUM_LANES) if i != green]
        for red_count in (1, 2):
            for reds in combinations(others, red_count):
                gain = _transfer_gain(counts, reds)
                if gain is None:
                    continue
                if counts[green] + gain <= _MAX_LANE_TOTAL:
                    grouped[red_count].append((reds, green, gain))
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
    reds, green, gain = feasible[red_count][int(rng.integers(len(feasible[red_count])))]
    red_set = set(reds)

    output_counts = input_counts[:]
    output_counts[green] += gain

    out = np.full((_H, _W), bg, dtype=int)
    for lane_idx in range(_NUM_LANES):
        if lane_idx in red_set:
            grey_tail = input_counts[lane_idx] // 2
            _paint_lane_with_grey_tail(out, lane_idx, lane_colors[lane_idx], input_counts[lane_idx], grey_tail, bg)
        else:
            _paint_lane(out, lane_idx, lane_colors[lane_idx], output_counts[lane_idx], bg)

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

    if not np.array_equal(main(inp), out):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(_GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {_GEN_RETRIES} attempts")
