# Variant of 9aaea919
# Dimension: D3 — 目标位置与堆叠规则 (dst)
# Change: stacking — 新精灵在目标车道底部已有精灵之上，按周期4(3精灵行+1间隙行)向上堆叠空槽位... → 交错插入

import numpy as np

from arc_utils import bbox_coords, components_mask, get_bg, paste
from itertools import combinations


# Design analysis for this D3 variant
# a) Concrete behavior:
#    - The canvas, lanes, sprite shape, markers, and recoloring stay the same as the
#      source task.
#    - Only the destination lane changes: its visible green-lane sprites can already sit
#      on the usual period-4 lattice with holes between them, e.g. tops at 13 and 25 with
#      row 17/21 slots empty.
#    - Old behavior on that input would scan from the bottom and place at 21 first.
#      New behavior treats the lane as a set of empty slots between existing sprites:
#      repeatedly find the widest occupied-to-occupied gap and insert into its middle slot.
#      For a 13...25 gap, that means 17 is chosen before 21, which makes the difference
#      visible in the output.
# b) Complexity calibration:
#    - The generator keeps only one destination lane special; all other lanes remain
#      bottom-stacked. This isolates the D3 change without adding unrelated ambiguity.
#    - Destination inputs are restricted to 2-3 existing sprites with the bottom slot
#      occupied and at least one interior hole. That keeps the rule readable while still
#      producing several distinct gap patterns.
# c) Hidden assumptions from the original solver that no longer hold:
#    - The original `while` loop assumed "next valid slot" can be found by scanning
#      upward from the bottom anchor. That fails here because the correct next slot may be
#      inside a higher gap instead of the lowest empty slot.
#    - `occupied` can no longer be treated as a monotone stack prefix; we must explicitly
#      enumerate the lane lattice and compute an insertion order from gaps.
# d) Placement legality:
#    - All slots still lie on the original 7-position lattice: 25,21,17,13,9,5,1.
#    - The generator only chooses destination occupancies whose final size is <= 7, so
#      every inserted sprite stays in bounds and non-overlapping.
# e) Visual observability:
#    - Generated samples are rejected unless the first positions chosen by the new policy
#      differ from simple bottom-up stacking for the same input lane.
#    - At least one inserted sprite must fill an interior hole, so the "fill the middle"
#      behavior is actually visible to a human.
# f) Visual cleanliness:
#    - Only one lane contains holes; markers remain separate on the last row and sprites
#      still occupy the original 5-column lane width, so the scene stays uncluttered.
# g) Color inference:
#    - Colors keep their original role: red markers identify sources, the green marker
#      identifies the destination, source sprites turn grey, and inserted sprites use the
#      destination lane color. No color-id-based ordering is used anywhere.
# Additional solver assumptions introduced here and enforced by the generator:
#    - The destination lane always includes the bottom slot, so `max(existing_tops)` is a
#      reliable lattice anchor for reconstructing candidate rows.
#    - Source and unmarked lanes remain simple contiguous bottom stacks, preserving the
#      original shape/color evidence while the destination alone exhibits the new D3 logic.


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


def _sorted_sprites(components: list[list[tuple[int, int]]]) -> list[tuple[int, int, list[tuple[int, int]]]]:
    items = []
    for pts in components:
        rmin = min(r for r, _ in pts)
        cmin = min(c for _, c in pts)
        items.append((rmin, cmin, pts))
    items.sort()
    return items


def _lane_slots(bottom0: int, period: int, sprite_h: int, h: int) -> list[int]:
    slots = []
    top = int(bottom0)
    while 0 <= top and top + sprite_h <= h:
        slots.append(top)
        top -= period
    return slots


def _interleaved_fill_order(occupied_tops: set[int], lane_slots: list[int]) -> list[int]:
    """
    Repeatedly split the widest occupied-to-occupied gap and take its middle empty slot.
    If a gap has two central slots, pick the upper one. After all interior holes are gone,
    continue upward from the current topmost sprite.
    """
    occupied = set(int(t) for t in occupied_tops)
    all_slots = sorted(int(t) for t in lane_slots)
    order: list[int] = []

    while True:
        occ = sorted(occupied)
        best: tuple[int, int, int] | None = None
        for upper, lower in zip(occ, occ[1:]):
            gap = [t for t in all_slots if upper < t < lower and t not in occupied]
            if not gap:
                continue
            chosen = gap[(len(gap) - 1) // 2]
            cand = (len(gap), lower, chosen)
            if best is None or cand > best:
                best = cand
        if best is None:
            break
        chosen = best[2]
        order.append(chosen)
        occupied.add(chosen)

    above = [t for t in all_slots if t < min(occupied) and t not in occupied]
    order.extend(sorted(above, reverse=True))

    below = [t for t in all_slots if t > max(occupied) and t not in occupied]
    order.extend(sorted(below))
    return order


def _stacking_fill_order(occupied_tops: set[int], lane_slots: list[int]) -> list[int]:
    """Original baseline: scan empty lattice slots from the bottom upward."""
    occupied = set(int(t) for t in occupied_tops)
    return [t for t in lane_slots if t not in occupied]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Transfer sprites from red-marked lanes to green-marked lanes, recoloring and interleaving.

    Input structure:
    - The background is the most common color; thin red (2) and green (3) bars mark source
      and destination lanes.
    - Each lane contains repeated monochrome plus sprites on the same period-4 vertical
      lattice, but a destination lane may skip some middle lattice slots.

    Rule:
    - Sprites in red-marked lanes turn grey (5), and the same number of copies move to the
      nearest green-marked lane recolored with that lane's own sprite color.
    - In the destination lane, new sprites fill interior holes first by repeatedly splitting
      the widest gap between existing sprites; only after those holes are exhausted do extra
      sprites continue upward on the lattice.
    - All marker bars are removed.

    Key insight:
    - The transfer still conserves sprite count, but destination placement is governed by a
      gap-filling order on the lane lattice rather than a simple bottom-up stack.
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

        dst_sprites_sorted = _sorted_sprites(dst_sprites)
        dst_color = int(g[dst_sprites_sorted[0][2][0][0], dst_sprites_sorted[0][2][0][1]])

        tops = sorted({rmin for rmin, _cmin, _pts in dst_sprites_sorted})
        if len(tops) >= 2:
            diffs = [b - a for a, b in zip(tops, tops[1:]) if b - a > 0]
            period = int(np.gcd.reduce(diffs)) if diffs else 4
        else:
            r0, _c0, r1, _c1 = bbox_coords(np.array(dst_sprites_sorted[0][2]))
            period = (r1 - r0 + 1) + 1
        sprite_h = max(r for r, _ in dst_sprites_sorted[0][2]) - min(r for r, _ in dst_sprites_sorted[0][2]) + 1

        gc = (dst_c0 + dst_c1) / 2.0
        green_state[id(gm)] = {
            "center": gc,
            "dst_color": dst_color,
            "period": period,
            "sprite_h": sprite_h,
            "occupied": set(tops),
            "lane_slots": _lane_slots(max(tops), period, sprite_h, h),
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

        for pts in src_sprites:
            for r, c in pts:
                out[r, c] = 5

        rc = (src_c0 + src_c1) / 2.0
        best = min(green_state.values(), key=lambda st: abs(st["center"] - rc))

        src_sprites_sorted = _sorted_sprites(src_sprites)
        src_ref_r0, _src_ref_c0, src_ref_pts = src_sprites_sorted[0]

        dc_shift = int(round(best["center"] - rc))
        src_offsets = [(r - src_ref_r0, c + dc_shift) for r, c in src_ref_pts]

        n_new = len(src_sprites)
        fill_order = _interleaved_fill_order(best["occupied"], best["lane_slots"])
        placed = 0
        for new_top in fill_order:
            if placed >= n_new:
                break
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
_SLOT_TOPS = [_BOTTOM_TOP - _PERIOD * k for k in range(7)]

# --- CFG: color palette ---
_BG_CHOICES = [0, 1, 4, 6, 7, 8, 9]
_GREY = 5

# --- CFG: counts & limits ---
_MIN_SPRITES = 1
_MAX_SPRITES = 3
_MAX_LANE_TOTAL = 7
_GEN_RETRIES = 400


def _sprite_patch(color: int, bg: int) -> np.ndarray:
    """Return a 3x5 plus/diamond sprite patch with bg-transparent corners."""
    patch = np.full((_SPRITE_H, _SPRITE_W), bg, dtype=int)
    patch[0, 1:4] = color
    patch[1, :] = color
    patch[2, 1:4] = color
    return patch


def _lane_tops(count: int) -> list[int]:
    """Return contiguous bottom-stack top rows for `count` sprites, ordered top-to-bottom."""
    return [_BOTTOM_TOP - _PERIOD * k for k in range(count - 1, -1, -1)]


def _paint_lane_tops(grid: np.ndarray, lane_idx: int, color: int, tops: list[int], bg: int) -> None:
    patch = _sprite_patch(color, bg)
    c0 = _LANE_STARTS[lane_idx]
    for r0 in tops:
        paste(grid, patch, r0, c0, transparent=bg)


def _paint_lane(grid: np.ndarray, lane_idx: int, color: int, count: int, bg: int) -> None:
    _paint_lane_tops(grid, lane_idx, color, _lane_tops(count), bg)


def _candidate_green_tops() -> list[tuple[int, ...]]:
    cands = []
    for size in (2, 3):
        for combo in combinations(sorted(_SLOT_TOPS), size):
            occ = set(combo)
            if _BOTTOM_TOP not in occ:
                continue
            if len(combo) >= 2 and max(combo) - min(combo) == _PERIOD * (len(combo) - 1):
                continue
            cands.append(combo)
    return cands


_GREEN_TOP_CANDIDATES = _candidate_green_tops()


def _role_choices() -> list[tuple[tuple[int, ...], int]]:
    pairs = []
    for green in range(_NUM_LANES):
        others = [i for i in range(_NUM_LANES) if i != green]
        for red_count in (1, 2):
            for reds in combinations(others, red_count):
                pairs.append((reds, green))
    return pairs


_ROLE_CHOICES = _role_choices()


def _green_layout_options(gain: int) -> list[tuple[tuple[int, ...], list[int], list[int]]]:
    options = []
    for occ in _GREEN_TOP_CANDIDATES:
        occ_set = set(occ)
        diffs = [b - a for a, b in zip(occ, occ[1:])]
        if len(diffs) >= 1 and int(np.gcd.reduce(diffs)) != _PERIOD:
            continue
        fill_order = _interleaved_fill_order(occ_set, _SLOT_TOPS)
        stacking_order = _stacking_fill_order(occ_set, _SLOT_TOPS)
        if len(occ) + gain > _MAX_LANE_TOTAL or len(fill_order) < gain:
            continue
        inserted = fill_order[:gain]
        stacking_inserted = stacking_order[:gain]
        if inserted == stacking_inserted:
            continue
        if sorted(list(occ) + inserted) == sorted(list(occ) + stacking_inserted):
            continue
        interior = [t for t in inserted if min(occ) < t < max(occ)]
        if not interior:
            continue
        output_tops = sorted(list(occ) + inserted)
        options.append((occ, inserted, output_tops))
    return options


def _gen_attempt() -> np.ndarray | None:
    rng = np.random.default_rng()

    bg = int(rng.choice(_BG_CHOICES))
    sprite_palette = [c for c in range(10) if c not in (bg, _GREY)]
    lane_colors = rng.choice(sprite_palette, size=_NUM_LANES, replace=False).tolist()

    reds, green = _ROLE_CHOICES[int(rng.integers(len(_ROLE_CHOICES)))]
    red_set = set(reds)

    lane_counts = [int(rng.integers(_MIN_SPRITES, _MAX_SPRITES + 1)) for _ in range(_NUM_LANES)]
    gain = sum(lane_counts[i] for i in reds)

    green_options = _green_layout_options(gain)
    if not green_options:
        return None
    green_input_tops, inserted_tops, green_output_tops = green_options[int(rng.integers(len(green_options)))]
    lane_counts[green] = len(green_input_tops)

    out = np.full((_H, _W), bg, dtype=int)
    for lane_idx in range(_NUM_LANES):
        if lane_idx == green:
            _paint_lane_tops(out, lane_idx, lane_colors[lane_idx], green_output_tops, bg)
        elif lane_idx in red_set:
            _paint_lane(out, lane_idx, _GREY, lane_counts[lane_idx], bg)
        else:
            _paint_lane(out, lane_idx, lane_colors[lane_idx], lane_counts[lane_idx], bg)

    inp = out.copy()
    for lane_idx in reds:
        _paint_lane(inp, lane_idx, lane_colors[lane_idx], lane_counts[lane_idx], bg)

    green_c0 = _LANE_STARTS[green]
    for r0 in inserted_tops:
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
