# Variant of 9aaea919
# Dimension: D11 — 转移数量规则 (meta)
# Change: 转移数量N = 红色车道中所有精灵的总数(1对1完全转移)... → 转移数量由目标车道空位决定

import numpy as np

from arc_utils import bbox_coords, components_mask, get_bg, paste


# Design analysis for the D11 variant
# - Concrete behavior:
#   The canvas is still a 30x30 field with four fixed 5-column lanes, 3x5 plus sprites,
#   red 1x5 bars for sources, and one green 1x5 bar for the destination. The new visible
#   event is that every red-lane sprite still turns grey, but the green lane only receives
#   as many new sprites as it has empty lattice slots left. Example: if the green lane
#   already holds 6 sprites, there is only 1 open slot at the very top; if the red lanes
#   contain 4 sprites total, the output shows 4 greyed source sprites but only 1 added
#   green sprite.
# - Complexity calibration:
#   The object vocabulary stays unchanged so the task does not drift away from 9aaea919.
#   To make the new rule observable, the generator often starts the green lane at 5 or 6
#   sprites so that the "capacity binds" branch appears frequently, while a smaller share
#   of samples still uses the non-binding branch where all supply fits.
# - Hidden-assumption audit:
#   The original `n_new = len(src_sprites)` line depended on conservation of sprite count.
#   That assumption is invalid here. The original per-red loop also let destination capacity
#   be consumed implicitly by iteration order; in this variant the solver first aggregates
#   total supply per destination and then applies one explicit `min(supply, free_slots)`.
#   The original generator filtered out any case where green+red would exceed stack height;
#   that filter must be removed because over-supply is now the point of the variant.
# - Placement legality:
#   All sprites remain on the same period-4 vertical lattice, markers stay on row 29, and
#   output green counts never exceed the physical maximum stack of 7 sprites. The generator
#   keeps at least one sprite in the green lane so color and period remain detectable.
# - Visual observability:
#   Humans can infer the rule because many generated examples show a mismatch between the
#   number of greyed source sprites and the number of newly added destination sprites, and
#   that mismatch tracks the destination lane's visible remaining capacity.
# - Visual cleanliness / color inference:
#   Colors still play purely structural roles: red and green bars identify lanes, grey marks
#   consumed sources, and transferred sprites keep the destination lane's own color. No
#   ordering depends on color ids.
# - New solver assumptions checked against the generator:
#   The solver assumes lane sprites are aligned to one regular stack lattice and that each
#   marked lane contains at least one monochrome plus sprite. The generator enforces both.


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


def _lane_slots(bottom0: int, period: int, sprite_h: int, h: int) -> list[int]:
    """All legal top rows for this lane's vertical lattice, ordered top-to-bottom."""
    tops = []
    k = 0
    while True:
        top = bottom0 - k * period
        if top < 0:
            break
        if top + sprite_h <= h:
            tops.append(int(top))
        k += 1
    return sorted(tops)


def _extract_lane_state(g: np.ndarray, marker: list[tuple[int, int]], bg: int) -> dict | None:
    """Read one marked lane: monochrome sprite components, color, center, and stack lattice."""
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

    dst_color = int(g[sprite_rows[0][3][0][0], sprite_rows[0][3][0][1]])
    tops = sorted({r0 for r0, _c0_pts, _hh, _pts in sprite_rows})
    sprite_h = int(sprite_rows[0][2])
    if len(tops) >= 2:
        diffs = [b - a for a, b in zip(tops, tops[1:]) if b - a > 0]
        period = int(np.gcd.reduce(diffs)) if diffs else sprite_h + 1
    else:
        period = sprite_h + 1

    bottom0 = max(tops)
    slot_tops = _lane_slots(bottom0, period, sprite_h, h)
    return {
        "marker": marker,
        "center": (c0 + c1) / 2.0,
        "color": dst_color,
        "sprites": [pts for _r0, _c0_pts, _hh, pts in sprite_rows],
        "tops": tops,
        "occupied": set(tops),
        "period": period,
        "sprite_h": sprite_h,
        "bottom0": bottom0,
        "slot_tops": slot_tops,
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Move marked source-lane sprites into the marked destination lane until that lane runs out of room.

    Input structure:
    - The background is the dominant color, and thin red (2) / green (3) bars on the last row mark source and destination lanes.
    - Each lane contains repeated monochrome plus sprites placed on one regular vertical stack lattice.

    Rule:
    - Every sprite in a red-marked lane turns grey (5), and the destination lane tries to absorb those sprites bottom-up in its own color.
    - The number of new sprites is capped by the destination lane's empty lattice slots: only `min(total source sprites, free destination slots)` are placed.
    - All marker bars disappear after the transfer.

    Key insight:
    - The output is controlled by a bottleneck, not by conservation: all source sprites are consumed, but the green lane can only grow as far as its remaining stack capacity allows.
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
        free_slots = [top for top in sorted(best["slot_tops"], reverse=True) if top not in best["occupied"]]
        n_new = min(group["supply"], len(free_slots))
        if n_new <= 0:
            continue

        dc_shift = int(round(best["center"] - group["template_center"]))
        src_offsets = [(r - group["template_r0"], c + dc_shift) for r, c in group["template_pts"]]

        placed = 0
        for new_top in free_slots:
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
            if placed == n_new:
                break

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
_MAX_STACK_SLOTS = 1 + _BOTTOM_TOP // _PERIOD

# --- CFG: color palette ---
_BG_CHOICES = [0, 1, 4, 6, 7, 8, 9]
_GREY = 5

# --- CFG: counts & limits ---
_OTHER_MIN_SPRITES = 1
_OTHER_MAX_SPRITES = 3
_TIGHT_GREEN_COUNTS = [5, 6]
_LOOSE_GREEN_COUNTS = [1, 2, 3, 4]
_GEN_RETRIES = 300
_TIGHT_CASE_PROB = 0.8


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


def _pick_roles(rng: np.random.Generator) -> tuple[tuple[int, ...], int]:
    """Pick 1-2 red lanes and one distinct green lane."""
    green = int(rng.integers(_NUM_LANES))
    others = [i for i in range(_NUM_LANES) if i != green]
    red_count = 1 if rng.random() < 0.45 else 2
    reds = tuple(sorted(rng.choice(others, size=red_count, replace=False).tolist()))
    return reds, green


def _sample_counts(
    rng: np.random.Generator,
    reds: tuple[int, ...],
    green: int,
    tight_case: bool,
) -> list[int] | None:
    """Choose lane counts so the intended capacity branch is explicit."""
    counts = rng.integers(_OTHER_MIN_SPRITES, _OTHER_MAX_SPRITES + 1, size=_NUM_LANES).tolist()
    counts[green] = int(rng.choice(_TIGHT_GREEN_COUNTS if tight_case else _LOOSE_GREEN_COUNTS))

    if tight_case:
        for lane_idx in reds:
            counts[lane_idx] = int(rng.integers(2, _OTHER_MAX_SPRITES + 1))
        capacity = _MAX_STACK_SLOTS - counts[green]
        supply = sum(counts[i] for i in reds)
        if capacity <= 0 or supply <= capacity:
            return None
    else:
        capacity = _MAX_STACK_SLOTS - counts[green]
        supply = sum(counts[i] for i in reds)
        if supply > capacity:
            return None

    return counts


def _gen_attempt() -> np.ndarray | None:
    rng = np.random.default_rng()

    bg = int(rng.choice(_BG_CHOICES))
    sprite_palette = [c for c in range(10) if c not in (bg, _GREY)]
    lane_colors = rng.choice(sprite_palette, size=_NUM_LANES, replace=False).tolist()

    reds, green = _pick_roles(rng)
    counts = _sample_counts(rng, reds, green, tight_case=rng.random() < _TIGHT_CASE_PROB)
    if counts is None:
        return None

    red_set = set(reds)
    supply = sum(counts[i] for i in reds)
    capacity = _MAX_STACK_SLOTS - counts[green]
    transferred = min(supply, capacity)

    output_counts = counts[:]
    output_counts[green] += transferred
    out = np.full((_H, _W), bg, dtype=int)
    for lane_idx in range(_NUM_LANES):
        color = _GREY if lane_idx in red_set else lane_colors[lane_idx]
        _paint_lane(out, lane_idx, color, output_counts[lane_idx], bg)

    inp = out.copy()
    for lane_idx in reds:
        _paint_lane(inp, lane_idx, lane_colors[lane_idx], counts[lane_idx], bg)

    green_c0 = _LANE_STARTS[green]
    for r0 in _lane_tops(output_counts[green])[:transferred]:
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
