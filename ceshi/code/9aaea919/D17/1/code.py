# Variant of 9aaea919
# Dimension: D17 — 未标记车道行为 (transform)
# Change: 未标记车道(无红/绿标记)在输出中完全不变... → 未标记车道精灵复制到目标

# Design analysis for D17
# a) Concrete behavior:
#    The original task has four fixed 5-column sprite lanes. One green bar marks the
#    receiving lane, one or two red bars mark moving lanes, and the remaining lanes are
#    unmarked. In this variant, those unmarked lanes are no longer passive: every sprite
#    in an unmarked lane stays in place and also creates a copy in the green lane. Red
#    lanes still behave as before: their sprites become grey and also contribute to the
#    green lane. To keep the new rule visible, the destination lane is split into two
#    priority zones on the same vertical lattice: red contributions occupy the lowest
#    free slots, while unmarked-lane copies occupy the highest free slots.
#    Example trace: if the green lane initially has 2 sprites, red lanes contribute 2,
#    and unmarked lanes contribute 1, then the old task would produce a contiguous stack
#    of 4 gained sprites above the green base. The new task instead places the 2 red
#    transfers directly above the green base, places the 1 copied sprite at the highest
#    available slot, and leaves a visible gap between the two batches.
# b) Complexity calibration:
#    The role system stays small and repeated across every sample: red = move, green =
#    receive, unmarked = copy. There is still exactly one destination lane, and all
#    sprites keep the same plus shape and 4-row lattice, so the task is not overloaded.
#    The separate high/low placement policy makes the copied behavior visually explicit
#    without introducing extra colors or extra object types.
# c) Old assumptions that break:
#    - The original solver only inspected red-marked lanes as donors. That is no longer
#      enough; it must enumerate every sprite lane and classify each lane by marker role.
#    - The original generator allowed the green lane to fill up to all 7 slots. In this
#      variant, if the destination becomes completely full then "high priority vs low
#      priority" can collapse into an indistinguishable solid stack. The generator must
#      therefore leave at least one empty lattice slot in the destination.
#    - The original placement loop used one monotone bottom-up policy for every donor.
#      That would erase the intended distinction between moved and copied contributions.
# d) Placement legality:
#    The lane lattice remains unchanged: 7 total slot positions at rows 1,5,9,13,17,21,25.
#    The generator enforces total input sprites across all four lanes <= 6, so the green
#    lane can receive every non-green sprite and still leave at least one empty slot. This
#    guarantees no overlap, no out-of-bounds placement, and a visible gap between the low
#    red batch and the high copy batch.
# e) Visual observability:
#    Humans can see that red lanes lose their original color and turn grey, while unmarked
#    lanes keep their sprites unchanged. The green lane gains extra sprites from both kinds
#    of donor, but the copy batch is separated upward from the moved batch. Across a few
#    examples, that is enough to infer three roles instead of the old two-role system.
# f) Visual neatness:
#    Nothing new is drawn outside the existing lane structure. The only added complexity is
#    sparse occupancy inside the green lane, which remains clean because sprites stay aligned
#    to the same 5-column lane and same vertical period.
# g) Color reasoning:
#    No color-id arithmetic or hidden ordering is used. New sprites always take the green
#    lane's own sprite color, exactly as in the original task; the only new distinction is
#    whether the source lane is red-marked or unmarked.
# h) New solver assumptions and generator support:
#    - Every sprite component is a 3x5 monochrome plus with 11 cells.
#    - Every marker bar is a 1x5 mono bar on the lane columns.
#    - There is exactly one green-marked lane and at least one red-marked lane.
#    - Each lane has at least one sprite, so every unmarked lane genuinely exercises the
#      copy rule.
#    The generator constructs inputs to satisfy all of these assumptions and validates them
#    with an oracle call to main().

from pathlib import Path
import sys
from itertools import combinations

import numpy as np

ROOT = Path(__file__).resolve().parents[5]
if str(ROOT / "solvers") not in sys.path:
    sys.path.append(str(ROOT / "solvers"))

from arc_utils import bbox_coords, components_mask, get_bg, paste


def _collect_lane_states(
    g: np.ndarray, bg: int
) -> tuple[dict[int, dict[str, object]], set[int], set[int]]:
    lane_sprites: dict[int, list[list[tuple[int, int]]]] = {}
    red_cols: set[int] = set()
    green_cols: set[int] = set()

    for pts_arr in components_mask(g != bg, connectivity=4):
        pts = [(int(r), int(c)) for r, c in pts_arr]
        colors = {int(g[r, c]) for r, c in pts}
        if len(colors) != 1:
            continue
        color = next(iter(colors))
        r0, c0, r1, c1 = bbox_coords(pts_arr)
        hh = r1 - r0 + 1
        ww = c1 - c0 + 1

        if hh == 1 and ww == 5 and len(pts) == 5 and color in (2, 3):
            if color == 2:
                red_cols.add(c0)
            else:
                green_cols.add(c0)
            continue

        if hh == 3 and ww == 5 and len(pts) == 11:
            lane_sprites.setdefault(c0, []).append(pts)

    lane_states: dict[int, dict[str, object]] = {}
    for c0, sprites in lane_sprites.items():
        sprites_sorted = []
        for pts in sprites:
            rmin = min(r for r, _ in pts)
            cmin = min(c for _, c in pts)
            sprites_sorted.append((rmin, cmin, pts))
        sprites_sorted.sort()

        tops = [rmin for rmin, _cmin, _pts in sprites_sorted]
        if len(tops) >= 2:
            diffs = [b - a for a, b in zip(tops, tops[1:]) if b - a > 0]
            period = int(np.gcd.reduce(diffs)) if diffs else 4
        else:
            r0, _c0, r1, _c1 = bbox_coords(np.array(sprites_sorted[0][2]))
            period = (r1 - r0 + 1) + 1

        sample_r, sample_c = sprites_sorted[0][2][0]
        lane_states[c0] = {
            "c0": c0,
            "center": c0 + 2.0,
            "sprite_color": int(g[sample_r, sample_c]),
            "sprites": [pts for _rmin, _cmin, pts in sprites_sorted],
            "tops": tops,
            "period": period,
            "bottom0": max(tops),
        }

    return lane_states, red_cols, green_cols


def _shifted_offsets(src_lane: dict[str, object], dst_lane: dict[str, object]) -> list[tuple[int, int]]:
    ref_pts = src_lane["sprites"][0]
    ref_r0 = min(r for r, _ in ref_pts)
    dc_shift = int(round(float(dst_lane["center"]) - float(src_lane["center"])))
    return [(r - ref_r0, c + dc_shift) for r, c in ref_pts]


def _candidate_tops(dst_lane: dict[str, object]) -> list[int]:
    tops = []
    top = int(dst_lane["bottom0"])
    period = int(dst_lane["period"])
    while top >= 0:
        tops.append(top)
        top -= period
    return tops


def _place_batch(
    out: np.ndarray,
    src_lane: dict[str, object],
    dst_lane: dict[str, object],
    order: list[int],
    n_new: int,
    occupied: set[int],
    bg: int,
) -> None:
    if n_new <= 0:
        return

    h, w = out.shape
    offsets = _shifted_offsets(src_lane, dst_lane)
    sprite_h = max(dr for dr, _c in offsets) + 1
    dst_color = int(dst_lane["sprite_color"])

    placed = 0
    for new_top in order:
        if placed == n_new:
            break
        if new_top in occupied:
            continue
        if new_top < 0 or new_top + sprite_h > h:
            continue

        ok = True
        for dr, c in offsets:
            rr = new_top + dr
            if not (0 <= rr < h and 0 <= c < w):
                ok = False
                break
            if int(out[rr, c]) != bg:
                ok = False
                break
        if not ok:
            continue

        for dr, c in offsets:
            out[new_top + dr, c] = dst_color
        occupied.add(new_top)
        placed += 1

    assert placed == n_new, "could not place every requested sprite in the destination lane"


def main(grid: np.ndarray) -> np.ndarray:
    """
    Route marked and unmarked lane sprites into the green destination lane.

    Input structure:
    - The background is the dominant color. A single green 1x5 bar marks the receiving lane,
      one or two red 1x5 bars mark moving lanes, and the remaining sprite lanes are unmarked.
    - Each lane contains repeated monochrome plus sprites aligned to the same vertical lattice.

    Rule:
    - Sprites in red-marked lanes are greyed in place and also contribute the same number of
      destination-colored sprites into the green lane.
    - Sprites in unmarked lanes stay where they are and additionally create copies in the
      green lane.
    - Red contributions take the lowest free lattice slots, unmarked copies take the highest
      free lattice slots, and all marker bars are erased.

    Key insight:
    - The destination lane is fed by two donor classes at once: red lanes behave like moves,
      unmarked lanes behave like copies, and the spatial split between low slots and high
      slots is what makes the extra donor role observable.
    """
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)

    lane_states, red_cols, green_cols = _collect_lane_states(g, bg)
    assert lane_states, "expected sprite lanes to be present"
    assert red_cols and green_cols, "expected both red and green marker bars to be present"
    assert len(green_cols) == 1, "expected exactly one green-marked lane"

    green_c0 = next(iter(green_cols))
    assert green_c0 in lane_states, "expected the green marker to align with a sprite lane"
    dst_lane = lane_states[green_c0]

    red_lanes = [lane_states[c0] for c0 in sorted(red_cols) if c0 in lane_states]
    unmarked_lanes = [lane_states[c0] for c0 in sorted(lane_states) if c0 not in red_cols and c0 not in green_cols]
    assert red_lanes, "expected at least one red-marked source lane"
    assert unmarked_lanes, "expected at least one unmarked donor lane in this variant"

    out = g.copy()

    for lane in red_lanes:
        for pts in lane["sprites"]:
            for r, c in pts:
                out[r, c] = 5

    occupied = set(int(t) for t in dst_lane["tops"])
    low_to_high = _candidate_tops(dst_lane)
    high_to_low = list(reversed(low_to_high))

    for lane in red_lanes:
        _place_batch(out, lane, dst_lane, low_to_high, len(lane["sprites"]), occupied, bg)

    for lane in unmarked_lanes:
        _place_batch(out, lane, dst_lane, high_to_low, len(lane["sprites"]), occupied, bg)

    for c0 in red_cols:
        for c in range(c0, c0 + 5):
            out[29, c] = bg
    for c0 in green_cols:
        for c in range(c0, c0 + 5):
            out[29, c] = bg

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
_MAX_STACK = 7

# --- CFG: color palette ---
_BG_CHOICES = [0, 1, 4, 6, 7, 8, 9]
_GREY = 5

# --- CFG: counts & limits ---
_MIN_SPRITES = 1
_MAX_SPRITES = 3
_MAX_TOTAL_INPUT = 6
_GEN_RETRIES = 300


def _sprite_patch(color: int, bg: int) -> np.ndarray:
    patch = np.full((_SPRITE_H, _SPRITE_W), bg, dtype=int)
    patch[0, 1:4] = color
    patch[1, :] = color
    patch[2, 1:4] = color
    return patch


def _lane_tops(count: int) -> list[int]:
    return [_BOTTOM_TOP - _PERIOD * k for k in range(count - 1, -1, -1)]


def _all_slot_tops() -> list[int]:
    return _lane_tops(_MAX_STACK)


def _paint_lane_at_tops(grid: np.ndarray, lane_idx: int, color: int, tops: list[int], bg: int) -> None:
    patch = _sprite_patch(color, bg)
    c0 = _LANE_STARTS[lane_idx]
    for r0 in tops:
        paste(grid, patch, r0, c0, transparent=bg)


def _paint_lane(grid: np.ndarray, lane_idx: int, color: int, count: int, bg: int) -> None:
    _paint_lane_at_tops(grid, lane_idx, color, _lane_tops(count), bg)


def _feasible_roles(counts: list[int]) -> dict[int, list[tuple[tuple[int, ...], int]]]:
    grouped: dict[int, list[tuple[tuple[int, ...], int]]] = {1: [], 2: []}
    if sum(counts) > _MAX_TOTAL_INPUT:
        return grouped

    for green in range(_NUM_LANES):
        others = [i for i in range(_NUM_LANES) if i != green]
        for red_count in (1, 2):
            for reds in combinations(others, red_count):
                grouped[red_count].append((reds, green))
    return grouped


def _destination_tops(green_count: int, red_gain: int, copy_gain: int) -> tuple[list[int], list[int], list[int]]:
    slots = _all_slot_tops()
    green_tops = _lane_tops(green_count)
    occupied = set(green_tops)

    red_tops = []
    for top in reversed(slots):
        if top in occupied:
            continue
        red_tops.append(top)
        occupied.add(top)
        if len(red_tops) == red_gain:
            break

    copy_tops = []
    for top in slots:
        if top in occupied:
            continue
        copy_tops.append(top)
        occupied.add(top)
        if len(copy_tops) == copy_gain:
            break

    assert len(red_tops) == red_gain
    assert len(copy_tops) == copy_gain
    assert occupied != set(slots), "variant needs at least one visible empty slot in the destination lane"
    return green_tops, red_tops, copy_tops


def _gen_attempt() -> np.ndarray | None:
    rng = np.random.default_rng()

    bg = int(rng.choice(_BG_CHOICES))
    sprite_palette = [c for c in range(10) if c not in (bg, _GREY)]
    lane_colors = rng.choice(sprite_palette, size=_NUM_LANES, replace=False).tolist()

    counts_ok = False
    input_counts: list[int] = []
    for _ in range(40):
        cand = rng.integers(_MIN_SPRITES, _MAX_SPRITES + 1, size=_NUM_LANES).tolist()
        if sum(cand) <= _MAX_TOTAL_INPUT:
            input_counts = cand
            counts_ok = True
            break
    if not counts_ok:
        return None

    feasible = _feasible_roles(input_counts)
    feasible_red_counts = [rc for rc in (1, 2) if feasible[rc]]
    if not feasible_red_counts:
        return None

    red_count = int(rng.choice(feasible_red_counts))
    reds, green = feasible[red_count][int(rng.integers(len(feasible[red_count])))]
    red_set = set(reds)
    unmarked = [i for i in range(_NUM_LANES) if i not in red_set and i != green]
    red_gain = sum(input_counts[i] for i in reds)
    copy_gain = sum(input_counts[i] for i in unmarked)

    green_tops, red_tops, copy_tops = _destination_tops(input_counts[green], red_gain, copy_gain)

    out = np.full((_H, _W), bg, dtype=int)
    for lane_idx in range(_NUM_LANES):
        if lane_idx == green:
            continue
        color = _GREY if lane_idx in red_set else lane_colors[lane_idx]
        _paint_lane(out, lane_idx, color, input_counts[lane_idx], bg)
    _paint_lane_at_tops(out, green, lane_colors[green], green_tops + red_tops + copy_tops, bg)

    inp = out.copy()
    for lane_idx in reds:
        _paint_lane(inp, lane_idx, lane_colors[lane_idx], input_counts[lane_idx], bg)

    green_c0 = _LANE_STARTS[green]
    for r0 in red_tops + copy_tops:
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
