# Combo variant of 9aaea919 (index 7)
# combo_spec: D3/1+D11/1
# Name: 交错插入+空位决定转移量

import numpy as np

from arc_utils import bbox_coords, components_mask, get_bg, paste
from itertools import combinations


# Design analysis for the D3/1 + D11/1 combo
# a) Concrete behavior:
#    - The scene is still the original 30x30 four-lane layout with 3x5 plus sprites,
#      red source bars, and one green destination bar.
#    - The destination lane may already have holes on the fixed 7-slot lattice
#      [25, 21, 17, 13, 9, 5, 1]. Example: if its visible tops are {13, 25}, the
#      original solver would stack new sprites at 21 then 17, while the fused rule first
#      fills the middle of the widest occupied-to-occupied gap, so it chooses 17 before 21.
#    - Transfer count is no longer "all red sprites no matter what"; it is
#      `min(total source supply, destination free slots)`. If the destination has only one
#      open slot left, every source sprite still greys out, but only one recolored copy is
#      added to the green lane.
# b) Complexity calibration:
#    - A single green lane remains the only receiver and all sprites keep the same shape,
#      so the task stays close to 9aaea919.
#    - The generator deliberately mixes two sample families:
#      1) D3-visible cases where supply is smaller than capacity, so the chosen inserted
#         subset differs from bottom-up stacking and that difference remains visible.
#      2) D11-visible cases where supply exceeds capacity, so the grey-source count and the
#         number of newly added destination sprites visibly disagree.
#    - Forcing both effects into every single sample would overconstrain the lattice:
#      whenever capacity binds, every remaining slot gets filled, which erases the visual
#      evidence of interleaving. Mixing the two families avoids that degeneration.
# c) Hidden-assumption audit:
#    - The original per-red placement loop assumed transfer amount equals per-lane source
#      count. That no longer holds, so the solver must first aggregate supply per target
#      lane and only then apply one global `min(supply, free_slots)`.
#    - The original bottom-up scan assumed destination occupancy is a contiguous stack
#      prefix. That fails once the green lane can start with interior holes, so we must
#      reconstruct the lane lattice and compute an explicit gap-filling order.
#    - The solver still relies on a stable lattice anchor. The generator therefore keeps
#      the bottom slot occupied in every green input, making `max(existing_tops)` a valid
#      way to recover the lane's bottom row.
# d) Placement legality:
#    - Every sprite stays within the original 5-column lane width and the original 7 legal
#      vertical slots. Generator options are filtered so inserted sprites never overlap and
#      output lane size never exceeds 7.
#    - Source and unmarked lanes remain simple contiguous bottom stacks, preventing
#      ambiguous component parsing.
# e) Visual observability:
#    - D3-visible samples are rejected unless the fused output occupancy differs from the
#      occupancy produced by ordinary bottom-up stacking on the same input.
#    - D11-visible samples are rejected unless `supply > transferred`, so the mismatch
#      between greyed source sprites and added destination sprites is visible to a human.
# f) Visual cleanliness:
#    - Only the destination lane is allowed to contain holes; the remaining lanes preserve
#      the original clean stack layout. Markers stay on the bottom row and never overlap
#      sprites.
# g) Color inference:
#    - Colors keep structural roles only: red marks sources, green marks the destination,
#      grey marks consumed sprites, and inserted sprites inherit the destination lane's own
#      color. No ordering depends on numeric color ids.
# Additional solver assumptions introduced here and enforced by the generator:
#    - Each marked lane contains at least one monochrome plus sprite component.
#    - The green input lane always includes the bottom slot and at least one empty slot.
#    - D3-visible and D11-visible branches both appear across generated samples, so the
#      combo does not collapse back to the original bottom-stack regime.


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
    """All legal slot top rows for this lane, ordered bottom-to-top."""
    tops = []
    top = int(bottom0)
    while 0 <= top and top + sprite_h <= h:
        tops.append(top)
        top -= period
    return tops


def _interleaved_fill_order(occupied_tops: set[int], lane_slots: list[int]) -> list[int]:
    """
    Fill interior holes before extending the stack upward.

    Repeatedly pick the widest occupied-to-occupied gap and insert at its middle empty slot.
    When a gap has two central slots, choose the upper one. After all interior holes are
    filled, continue upward from the current topmost occupied slot.
    """
    occupied = {int(t) for t in occupied_tops}
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
    """Baseline order from the original task: empty slots scanned bottom-up."""
    occupied = {int(t) for t in occupied_tops}
    return [t for t in lane_slots if t not in occupied]


def _extract_lane_state(g: np.ndarray, marker: list[tuple[int, int]], bg: int) -> dict | None:
    """Parse one marked lane into sorted sprite components and its vertical lattice."""
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

    sprites_sorted = _sorted_sprites(sprites)
    tops = sorted({r0 for r0, _c0, _pts in sprites_sorted})
    color = int(g[sprites_sorted[0][2][0][0], sprites_sorted[0][2][0][1]])

    r0, _c0, r1, _c1 = bbox_coords(np.array(sprites_sorted[0][2]))
    sprite_h = int(r1 - r0 + 1)
    if len(tops) >= 2:
        diffs = [b - a for a, b in zip(tops, tops[1:]) if b - a > 0]
        period = int(np.gcd.reduce(diffs)) if diffs else sprite_h + 1
    else:
        period = sprite_h + 1

    return {
        "center": (c0 + c1) / 2.0,
        "color": color,
        "sprites": [pts for _r0, _c0, pts in sprites_sorted],
        "occupied": set(tops),
        "slot_tops": _lane_slots(max(tops), period, sprite_h, h),
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Transfer marked source sprites into the marked destination lane with hole-filling and capacity limits.

    Input structure:
    - The background is the dominant color, red 1x5 bars mark source lanes, and a green 1x5
      bar marks the destination lane.
    - Each lane contains repeated monochrome plus sprites on the same period-4 lattice, and
      the destination lane may already skip some interior lattice slots.

    Rule:
    - Every sprite in a red-marked lane turns grey (5). Their total supply is grouped by the
      nearest green-marked lane, but that lane accepts only as many copies as it has empty
      lattice slots.
    - Accepted copies are recolored to the destination lane's own sprite color and placed by
      interleaving into the destination lane's holes: split the widest occupied gap first,
      then continue upward only after interior holes are exhausted.
    - All marker bars are removed.

    Key insight:
    - Two quantities matter at once: source supply and destination capacity. The destination
      does not simply stack everything upward; it chooses a gap-filling subset of the free
      slots, and the number of chosen slots is capped by remaining room.
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


# --------------- generator (O3) ---------------

# --- CFG: canvas & layout ---
_H, _W = 30, 30
_NUM_LANES = 4
_LANE_STARTS = [3, 9, 15, 21]
_MARKER_ROW = 29
_SPRITE_H, _SPRITE_W = 3, 5

# --- CFG: vertical lattice ---
_BOTTOM_TOP = 25
_PERIOD = 4
_SLOT_TOPS = [_BOTTOM_TOP - _PERIOD * k for k in range(7)]  # bottom-to-top
_MAX_STACK_SLOTS = len(_SLOT_TOPS)

# --- CFG: color palette ---
_BG_CHOICES = [0, 1, 4, 6, 7, 8, 9]
_GREY = 5

# --- CFG: generation mix ---
_OTHER_MIN_SPRITES = 1
_OTHER_MAX_SPRITES = 3
_GEN_RETRIES = 500
_GEN_CASE_INDEX = 0


def _sprite_patch(color: int, bg: int) -> np.ndarray:
    """Return a 3x5 plus sprite patch with transparent corners."""
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


def _interior_holes(occ: tuple[int, ...]) -> list[int]:
    occ_set = set(occ)
    lo = min(occ)
    hi = max(occ)
    return [t for t in sorted(_SLOT_TOPS) if lo < t < hi and t not in occ_set]


def _supply_partitions() -> dict[int, list[tuple[int, ...]]]:
    parts: dict[int, list[tuple[int, ...]]] = {total: [] for total in range(1, 7)}
    for total in range(1, 7):
        if 1 <= total <= 3:
            parts[total].append((total,))
        for a in range(1, 4):
            for b in range(1, 4):
                if a + b == total:
                    parts[total].append((a, b))
    return parts


_SUPPLY_PARTS = _supply_partitions()


def _build_d3_options() -> list[dict]:
    opts = []
    asc_slots = sorted(_SLOT_TOPS)
    for size in range(2, _MAX_STACK_SLOTS):
        for combo in combinations(asc_slots, size):
            occ = tuple(sorted(combo))
            occ_set = set(occ)
            if _BOTTOM_TOP not in occ_set:
                continue
            if not _interior_holes(occ):
                continue

            fill_order = _interleaved_fill_order(occ_set, _SLOT_TOPS)
            stacking_order = _stacking_fill_order(occ_set, _SLOT_TOPS)
            free_slots = len(fill_order)
            for transferred in range(1, min(6, free_slots - 1) + 1):
                inserted = fill_order[:transferred]
                baseline = stacking_order[:transferred]
                if inserted == baseline:
                    continue
                if sorted(list(occ) + inserted) == sorted(list(occ) + baseline):
                    continue
                if not any(min(occ) < t < max(occ) for t in inserted):
                    continue
                if not _SUPPLY_PARTS[transferred]:
                    continue
                opts.append(
                    {
                        "occ": occ,
                        "fill_order": fill_order,
                        "transferred": transferred,
                        "inserted": inserted,
                        "output_tops": sorted(list(occ) + inserted),
                        "free_slots": free_slots,
                    }
                )
    return opts


def _build_d11_options() -> list[dict]:
    opts = []
    asc_slots = sorted(_SLOT_TOPS)
    for size in range(4, _MAX_STACK_SLOTS):
        for combo in combinations(asc_slots, size):
            occ = tuple(sorted(combo))
            occ_set = set(occ)
            if _BOTTOM_TOP not in occ_set:
                continue
            if not _interior_holes(occ):
                continue

            fill_order = _interleaved_fill_order(occ_set, _SLOT_TOPS)
            free_slots = len(fill_order)
            if not (1 <= free_slots <= 3):
                continue
            if free_slots + 1 > 6:
                continue
            opts.append(
                {
                    "occ": occ,
                    "fill_order": fill_order,
                    "transferred": free_slots,
                    "inserted": list(fill_order),
                    "output_tops": sorted(_SLOT_TOPS),
                    "free_slots": free_slots,
                }
            )
    return opts


_D3_OPTIONS = _build_d3_options()
_D11_OPTIONS = _build_d11_options()


def _pick_roles(rng: np.random.Generator, red_count: int, green: int) -> tuple[int, ...]:
    others = [i for i in range(_NUM_LANES) if i != green]
    return tuple(sorted(rng.choice(others, size=red_count, replace=False).tolist()))


def _gen_attempt(case_kind: str) -> np.ndarray | None:
    rng = np.random.default_rng()

    bg = int(rng.choice(_BG_CHOICES))
    sprite_palette = [c for c in range(10) if c not in (bg, _GREY, 2, 3)]
    lane_colors = rng.choice(sprite_palette, size=_NUM_LANES, replace=False).tolist()

    if case_kind == "D3":
        if not _D3_OPTIONS:
            return None
        option = _D3_OPTIONS[int(rng.integers(len(_D3_OPTIONS)))]
        supply = option["transferred"]
    else:
        if not _D11_OPTIONS:
            return None
        option = _D11_OPTIONS[int(rng.integers(len(_D11_OPTIONS)))]
        supply = int(rng.integers(option["free_slots"] + 1, 7))

    if not _SUPPLY_PARTS[supply]:
        return None

    green = int(rng.integers(_NUM_LANES))
    supply_split = _SUPPLY_PARTS[supply][int(rng.integers(len(_SUPPLY_PARTS[supply])))]
    reds = _pick_roles(rng, len(supply_split), green)
    red_set = set(reds)

    lane_counts = [int(rng.integers(_OTHER_MIN_SPRITES, _OTHER_MAX_SPRITES + 1)) for _ in range(_NUM_LANES)]
    lane_counts[green] = len(option["occ"])
    for lane_idx, count in zip(reds, supply_split):
        lane_counts[lane_idx] = int(count)

    out = np.full((_H, _W), bg, dtype=int)
    for lane_idx in range(_NUM_LANES):
        if lane_idx == green:
            _paint_lane_tops(out, lane_idx, lane_colors[lane_idx], option["output_tops"], bg)
        elif lane_idx in red_set:
            _paint_lane(out, lane_idx, _GREY, lane_counts[lane_idx], bg)
        else:
            _paint_lane(out, lane_idx, lane_colors[lane_idx], lane_counts[lane_idx], bg)

    inp = out.copy()
    for lane_idx in reds:
        _paint_lane(inp, lane_idx, lane_colors[lane_idx], lane_counts[lane_idx], bg)

    green_c0 = _LANE_STARTS[green]
    for r0 in option["inserted"]:
        inp[r0 : r0 + _SPRITE_H, green_c0 : green_c0 + _SPRITE_W] = bg
    inp[_MARKER_ROW, green_c0 : green_c0 + _SPRITE_W] = 3

    for lane_idx in reds:
        c0 = _LANE_STARTS[lane_idx]
        inp[_MARKER_ROW, c0 : c0 + _SPRITE_W] = 2

    if not np.array_equal(main(inp), out):
        return None
    return inp


def generate_input() -> np.ndarray:
    global _GEN_CASE_INDEX

    case_order = []
    if _D3_OPTIONS and _D11_OPTIONS:
        preferred = "D3" if (_GEN_CASE_INDEX % 2 == 0) else "D11"
        alternate = "D11" if preferred == "D3" else "D3"
        case_order = [preferred, alternate]
    elif _D3_OPTIONS:
        case_order = ["D3"]
    elif _D11_OPTIONS:
        case_order = ["D11"]

    for case_kind in case_order:
        for _ in range(_GEN_RETRIES):
            grid = _gen_attempt(case_kind)
            if grid is not None:
                if len(case_order) == 2:
                    _GEN_CASE_INDEX += 1
                return grid
    raise RuntimeError(f"failed to generate a valid input after {_GEN_RETRIES} attempts")
