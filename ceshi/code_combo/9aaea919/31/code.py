# Combo variant of 9aaea919 (index 31)
# combo_spec: D3/1+D11/1+D16/1
# Name: 交错插入+空位决定量+更多精灵

import numpy as np

from arc_utils import bbox_coords, components_mask, get_bg, paste
from itertools import combinations


# Design analysis for the fused combo variant
# a) Concrete behavior and trace:
#    - Keep the original 30x30 canvas, four fixed 5-column lanes, 3x5 plus/diamond
#      sprites, red source bars, and one green destination bar.
#    - The destination lane is no longer a contiguous bottom stack in the input. Example:
#      if its occupied tops are {13, 25}, the old solver would place new tops at 21 then 17
#      by bottom-up scanning. The fused rule instead uses the same lane lattice but fills the
#      interior gap first: 17, then 21, then 9, 5, 1 above the current top if more transfer
#      remains.
#    - The number of inserted sprites is no longer guaranteed to equal source supply. If the
#      red lanes contain 4 sprites total but the destination has only 2 empty lattice slots,
#      all 4 source sprites still turn grey while only 2 new destination sprites are added.
#    - Counts per lane in the input are expanded to 1-5. In practice the generator biases the
#      destination toward 3-5 existing sprites with holes, which creates denser and more varied
#      empty-slot patterns than the original 1-3 regime.
# b) Complexity calibration:
#    - The object vocabulary stays unchanged; only the occupancy pattern and transfer arithmetic
#      change. This keeps the task recognizable as 9aaea919 instead of becoming a new puzzle.
#    - To make D3 visible, some generated samples use a non-binding transfer where the number of
#      inserted sprites cuts off partway through the interleaved order, so the output differs from
#      simple bottom-up stacking on the same input.
#    - To make D11 visible, other samples deliberately over-supply the destination so capacity
#      binds. Those samples still start from a holey destination lane, so one solver must handle
#      both behaviors under the same lane model.
# c) Hidden assumptions from the original solver that fail here:
#    - `n_new = len(src_sprites)` is invalid because transfer quantity is now capped by free slots.
#    - The original while-loop assumed the destination occupancy is a stack prefix growing only
#      upward from the bottom. That fails when the correct next slot is inside a middle gap.
#    - Processing each red lane independently can accidentally let iteration order decide how much
#      capacity remains. The fused solver must aggregate total source supply first, then apply one
#      explicit `min(supply, free_slots)` against the destination lane's slot order.
# d) Placement legality:
#    - Every painted sprite still lands on the original period-4 lane lattice and within the same
#      5-column lane window. The generator only uses slot sets drawn from the legal 7-row lattice,
#      and it builds outputs first, so no overlaps or out-of-bounds placements are introduced.
# e) Visual observability:
#    - One branch of samples visibly differs from bottom-up stacking because only the first part of
#      the interleaved order is realized. Another branch visibly differs from the original task
#      because more greyed source sprites exist than transferred destination sprites.
#    - These two visible effects cannot both be maximally exposed in a single sample: if capacity
#      binds all the way to "fill every remaining slot", the final occupied set matches any order.
#      The generator therefore mixes the two branches across samples while keeping the same holey
#      destination structure and 1-5 count regime throughout.
# f) Visual cleanliness:
#    - Only the destination lane carries holes; red and unmarked lanes remain contiguous bottom
#      stacks. This avoids clutter while still making the destination's gap logic salient.
# g) Color inference and new solver assumptions:
#    - Colors keep purely structural roles: red marks source lanes, green marks the destination,
#      grey marks consumed sources, and inserted sprites take the destination lane color.
#    - The solver now assumes every marked lane contains at least one monochrome plus sprite, and
#      that all sprites in a lane share one regular lattice period. The generator enforces both.


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
    mask = (g != bg) & colmask
    comps = [[(int(r), int(c)) for r, c in pts] for pts in components_mask(mask, connectivity=4)]
    out = []
    for pts in comps:
        colors = {int(g[r, c]) for r, c in pts}
        if len(colors) == 1:
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


def _lane_slots(anchor_top: int, period: int, sprite_h: int, h: int) -> list[int]:
    """Enumerate the full legal lattice through one observed sprite top."""
    slots = {int(anchor_top)}

    top = int(anchor_top) - int(period)
    while top >= 0:
        if top + sprite_h <= h:
            slots.add(int(top))
        top -= int(period)

    top = int(anchor_top) + int(period)
    while top + sprite_h <= h:
        if top >= 0:
            slots.add(int(top))
        top += int(period)

    return sorted(slots)


def _interleaved_fill_order(occupied_tops: set[int], lane_slots: list[int]) -> list[int]:
    """
    Fill interior gaps before growing beyond the current occupied span.
    For an even-sized gap, choose the upper central slot first.
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
    """Baseline order from the source task: scan empty slots from bottom upward."""
    occupied = set(int(t) for t in occupied_tops)
    return [t for t in sorted(lane_slots, reverse=True) if t not in occupied]


def _extract_lane_state(g: np.ndarray, marker: list[tuple[int, int]], bg: int) -> dict | None:
    """Read one marked lane: sprite components, color, center, and legal lattice slots."""
    h, _w = g.shape
    _, c0, _, c1 = bbox_coords(np.array(marker))
    lane_cols = range(c0, c1 + 1)
    marker_set = set(marker)

    comps = _single_color_components_in_cols(g, lane_cols, bg)
    comps = [[p for p in pts if p not in marker_set] for pts in comps]
    comps = [pts for pts in comps if pts]
    if not comps:
        return None

    sprite_sizes = [len(pts) for pts in comps if len(pts) != len(marker)]
    if not sprite_sizes:
        return None
    sprite_sz = max(sprite_sizes)
    sprites = [pts for pts in comps if len(pts) == sprite_sz]
    if not sprites:
        return None

    sprite_rows = []
    for pts in sprites:
        r0, c0_pts, r1, _c1_pts = bbox_coords(np.array(pts))
        sprite_rows.append((r0, c0_pts, r1 - r0 + 1, pts))
    sprite_rows.sort()

    color = int(g[sprite_rows[0][3][0][0], sprite_rows[0][3][0][1]])
    tops = sorted({r0 for r0, _c0_pts, _hh, _pts in sprite_rows})
    sprite_h = int(sprite_rows[0][2])
    if len(tops) >= 2:
        diffs = [b - a for a, b in zip(tops, tops[1:]) if b - a > 0]
        period = int(np.gcd.reduce(diffs)) if diffs else sprite_h + 1
    else:
        period = sprite_h + 1

    slot_tops = _lane_slots(tops[0], period, sprite_h, h)
    return {
        "marker": marker,
        "center": (c0 + c1) / 2.0,
        "color": color,
        "sprites": [pts for _r0, _c0_pts, _hh, pts in sprite_rows],
        "tops": tops,
        "occupied": set(tops),
        "period": period,
        "sprite_h": sprite_h,
        "slot_tops": slot_tops,
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Transfer marked source-lane sprites into the marked destination lane with capped, interleaved placement.

    Input structure:
    - The dominant background surrounds four fixed vertical lanes, with thin red (2) bars marking source lanes
      and a thin green (3) bar marking one destination lane.
    - Every lane contains repeated monochrome plus sprites on one period-4 lattice; the destination lane may
      already occupy a non-contiguous subset of those slots, leaving visible holes between existing sprites.

    Rule:
    - Every sprite in each red-marked lane turns grey (5). The total number that can reappear in the nearest
      green-marked lane is limited to `min(total source sprites, empty destination slots)`.
    - New destination sprites use the destination lane color and are placed by filling interior holes first:
      repeatedly choose the middle slot of the widest occupied-to-occupied gap, then continue above the current
      topmost sprite only after all interior holes are exhausted.
    - All marker bars are removed.

    Key insight:
    - The solver must reason about a destination lane as a sparse lattice with finite capacity. The answer depends
      jointly on two things: how many source sprites are available and which empty slots on the destination lattice
      are reached first by the gap-filling order.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = get_bg(g)

    red_markers = _find_marker_bars(g, 2)
    green_markers = _find_marker_bars(g, 3)
    assert red_markers and green_markers, "expected both red and green marker bars to be present"

    green_state: dict[int, dict] = {}
    for gm in green_markers:
        state = _extract_lane_state(g, gm, bg)
        if state is not None:
            green_state[id(gm)] = state
    assert green_state, "expected at least one green lane with valid sprites"

    out = g.copy()
    grouped_supply: dict[int, dict] = {}

    for rm in red_markers:
        src_state = _extract_lane_state(g, rm, bg)
        if src_state is None:
            continue

        for pts in src_state["sprites"]:
            for r, c in pts:
                out[r, c] = 5

        best_key = min(green_state, key=lambda key: abs(green_state[key]["center"] - src_state["center"]))
        group = grouped_supply.setdefault(
            best_key,
            {
                "supply": 0,
                "template_pts": None,
                "template_r0": None,
                "template_center": None,
            },
        )
        group["supply"] += len(src_state["sprites"])
        if group["template_pts"] is None:
            top_sprite = src_state["sprites"][0]
            group["template_pts"] = top_sprite
            group["template_r0"] = min(r for r, _c in top_sprite)
            group["template_center"] = src_state["center"]

    for best_key, group in grouped_supply.items():
        best = green_state[best_key]
        fill_order = _interleaved_fill_order(best["occupied"], best["slot_tops"])
        n_new = min(group["supply"], len(fill_order))
        if n_new <= 0:
            continue

        dc_shift = int(round(best["center"] - group["template_center"]))
        src_offsets = [(r - group["template_r0"], c + dc_shift) for r, c in group["template_pts"]]

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
                out[new_top + dr, c] = best["color"]
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

# --- CFG: vertical lattice ---
_BOTTOM_TOP = 25
_PERIOD = 4
_SLOT_TOPS = [_BOTTOM_TOP - _PERIOD * k for k in range(7)]
_MAX_STACK_SLOTS = len(_SLOT_TOPS)

# --- CFG: colors ---
_BG_CHOICES = [0, 1, 4, 6, 7, 8, 9]
_GREY = 5

# --- CFG: counts & generation ---
_MIN_SPRITES = 1
_MAX_SPRITES = 5
_GEN_RETRIES = 500
_CASE_CYCLE = (
    (True, 1),
    (False, 1),
    (True, 2),
    (False, 2),
)
_GEN_CALL_COUNT = 0


def _sprite_patch(color: int, bg: int) -> np.ndarray:
    patch = np.full((_SPRITE_H, _SPRITE_W), bg, dtype=int)
    patch[0, 1:4] = color
    patch[1, :] = color
    patch[2, 1:4] = color
    return patch


def _lane_tops(count: int) -> list[int]:
    return [_BOTTOM_TOP - _PERIOD * k for k in range(count - 1, -1, -1)]


def _paint_lane_tops(grid: np.ndarray, lane_idx: int, color: int, tops: list[int], bg: int) -> None:
    patch = _sprite_patch(color, bg)
    c0 = _LANE_STARTS[lane_idx]
    for r0 in tops:
        paste(grid, patch, r0, c0, transparent=bg)


def _paint_lane(grid: np.ndarray, lane_idx: int, color: int, count: int, bg: int) -> None:
    _paint_lane_tops(grid, lane_idx, color, _lane_tops(count), bg)


def _candidate_green_layouts() -> list[dict]:
    cands = []
    for size in range(2, 6):
        for combo in combinations(sorted(_SLOT_TOPS), size):
            occ = set(combo)
            if _BOTTOM_TOP not in occ:
                continue
            if len(combo) >= 2 and max(combo) - min(combo) == _PERIOD * (len(combo) - 1):
                continue
            fill_order = _interleaved_fill_order(occ, _SLOT_TOPS)
            stacking_order = _stacking_fill_order(occ, _SLOT_TOPS)
            free_slots = len(fill_order)
            if free_slots <= 0:
                continue
            interior_holes = [t for t in fill_order if min(combo) < t < max(combo)]
            if not interior_holes:
                continue
            visible_ns = [
                n
                for n in range(1, free_slots)
                if set(fill_order[:n]) != set(stacking_order[:n])
            ]
            cands.append(
                {
                    "occ": combo,
                    "size": size,
                    "fill_order": fill_order,
                    "stacking_order": stacking_order,
                    "free_slots": free_slots,
                    "visible_ns": visible_ns,
                    "interior_holes": interior_holes,
                }
            )
    return cands


_GREEN_LAYOUTS = _candidate_green_layouts()
_INTERLEAVE_LAYOUTS = [cfg for cfg in _GREEN_LAYOUTS if cfg["visible_ns"]]
_CAPACITY_LAYOUTS = [cfg for cfg in _GREEN_LAYOUTS if cfg["free_slots"] <= 5]


def _role_choices() -> list[tuple[tuple[int, ...], int]]:
    pairs = []
    for green in range(_NUM_LANES):
        others = [i for i in range(_NUM_LANES) if i != green]
        for red_count in (1, 2):
            for reds in combinations(others, red_count):
                pairs.append((reds, green))
    return pairs


_ROLE_CHOICES = _role_choices()
_ROLE_CHOICES_BY_COUNT = {
    red_count: [pair for pair in _ROLE_CHOICES if len(pair[0]) == red_count]
    for red_count in (1, 2)
}


def _choose_red_counts_for_sum(
    rng: np.random.Generator, red_count: int, total: int
) -> list[int] | None:
    if total < red_count or total > red_count * _MAX_SPRITES:
        return None
    if red_count == 1:
        return [total] if _MIN_SPRITES <= total <= _MAX_SPRITES else None

    low = max(_MIN_SPRITES, total - _MAX_SPRITES)
    high = min(_MAX_SPRITES, total - _MIN_SPRITES)
    if low > high:
        return None
    first = int(rng.integers(low, high + 1))
    second = total - first
    if not (_MIN_SPRITES <= second <= _MAX_SPRITES):
        return None
    counts = [first, second]
    rng.shuffle(counts)
    return counts


def _sample_case(
    rng: np.random.Generator, reds: tuple[int, ...], prefer_interleave: bool
) -> tuple[dict, list[int], int] | None:
    red_count = len(reds)

    if prefer_interleave:
        layouts = [cfg for cfg in _INTERLEAVE_LAYOUTS if any(n <= red_count * _MAX_SPRITES for n in cfg["visible_ns"])]
        if not layouts:
            return None
        layout = layouts[int(rng.integers(len(layouts)))]
        feasible_ns = sorted(n for n in layout["visible_ns"] if red_count <= n <= red_count * _MAX_SPRITES)
        if not feasible_ns:
            return None
        transfer_choices = feasible_ns[len(feasible_ns) // 2 :]
        transfer = int(rng.choice(transfer_choices))
        red_counts = _choose_red_counts_for_sum(rng, red_count, transfer)
        if red_counts is None:
            return None
        supply = transfer
        return layout, red_counts, supply

    layouts = [cfg for cfg in _CAPACITY_LAYOUTS if cfg["free_slots"] < red_count * _MAX_SPRITES]
    if not layouts:
        return None
    layout = layouts[int(rng.integers(len(layouts)))]
    capacity = layout["free_slots"]
    min_supply = capacity + 1
    max_supply = red_count * _MAX_SPRITES
    if min_supply > max_supply:
        return None
    supply_low = min_supply + (max_supply - min_supply) // 2
    supply = int(rng.integers(supply_low, max_supply + 1))
    red_counts = _choose_red_counts_for_sum(rng, red_count, supply)
    if red_counts is None:
        return None
    return layout, red_counts, supply


def _pick_roles_for_case(
    rng: np.random.Generator, prefer_interleave: bool, desired_red_count: int
) -> tuple[tuple[int, ...], int, tuple[dict, list[int], int]] | None:
    for red_count in (desired_red_count, 1 if desired_red_count == 2 else 2):
        role_choices = list(_ROLE_CHOICES_BY_COUNT[red_count])
        rng.shuffle(role_choices)
        for reds, green in role_choices:
            case = _sample_case(rng, reds, prefer_interleave)
            if case is not None:
                return reds, green, case
    return None


def _gen_attempt(prefer_interleave: bool, desired_red_count: int) -> np.ndarray | None:
    rng = np.random.default_rng()

    bg = int(rng.choice(_BG_CHOICES))
    sprite_palette = [c for c in range(10) if c not in (bg, _GREY)]
    lane_colors = rng.choice(sprite_palette, size=_NUM_LANES, replace=False).tolist()

    picked = _pick_roles_for_case(rng, prefer_interleave, desired_red_count)
    if picked is None:
        return None
    reds, green, case = picked
    layout, red_counts, supply = case

    lane_counts = [int(rng.integers(_MIN_SPRITES, _MAX_SPRITES + 1)) for _ in range(_NUM_LANES)]
    for lane_idx, count in zip(reds, red_counts):
        lane_counts[lane_idx] = int(count)
    lane_counts[green] = int(layout["size"])

    transferred = min(supply, layout["free_slots"])
    inserted_tops = layout["fill_order"][:transferred]
    green_output_tops = sorted(list(layout["occ"]) + inserted_tops)

    out = np.full((_H, _W), bg, dtype=int)
    red_set = set(reds)
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
        inp[r0 : r0 + _SPRITE_H, green_c0 : green_c0 + _SPRITE_W] = bg
    inp[_MARKER_ROW, green_c0 : green_c0 + _SPRITE_W] = 3

    for lane_idx in reds:
        c0 = _LANE_STARTS[lane_idx]
        inp[_MARKER_ROW, c0 : c0 + _SPRITE_W] = 2

    if not np.array_equal(main(inp), out):
        return None
    return inp


def generate_input() -> np.ndarray:
    global _GEN_CALL_COUNT
    prefer_interleave, desired_red_count = _CASE_CYCLE[_GEN_CALL_COUNT % len(_CASE_CYCLE)]
    _GEN_CALL_COUNT += 1
    for _ in range(_GEN_RETRIES):
        grid = _gen_attempt(prefer_interleave, desired_red_count)
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {_GEN_RETRIES} attempts")
