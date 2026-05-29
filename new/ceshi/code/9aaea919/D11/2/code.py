# Variant of 9aaea919
# Dimension: D11 — 转移数量规则 (meta)
# Change: 转移数量N = 红色车道中所有精灵的总数(1对1完全转移)... → 源车道精灵数量差值转移

import numpy as np

from arc_utils import bbox_coords, components_mask, get_bg, paste
from itertools import combinations


# Design analysis for the D11 variant
# a) Concrete behavior and trace:
#    - The canvas, four 5-column lanes, 1x5 marker bars, and 3x5 plus sprites stay unchanged.
#    - The new visible pattern is only triggered when there are two red markers. Those two red lanes are
#      no longer two independent sources; instead they form a comparison pair.
#    - Example: red counts = 3 and 1, green count = 2.
#      Original task: both red lanes grey out completely and green gains 4 sprites.
#      Variant: only the lane with 3 sprites donates, and only its excess 2 sprites (the top two, since
#      stacks are bottom-aligned) turn grey; the 1-sprite red lane stays colored; green gains 2 sprites.
#    - The changed step is therefore "how many source sprites are selected for transfer", not lane layout.
# b) Complexity calibration:
#    - The red-pair comparison uses only two red lanes and a single green lane, so the rule is still
#      inferable from a few examples.
#    - Counts are kept in the original 1-3 range, which yields visible differences of 1 or 2 sprites.
#      Generator forbids equal red counts, otherwise the variant would visually collapse to "remove markers".
# c) Old-solver assumption audit:
#    - The original per-red-lane loop assumes every red lane contributes its full sprite count. That would
#      incorrectly sum both red lanes here.
#    - The original greying step recolors every sprite in each red lane. That no longer matches the variant:
#      only the extra sprites in the larger red lane should grey out.
#    - The original generator feasibility check uses sum(red counts). It must switch to abs(count_a-count_b).
#    - The original input reconstruction repaints every red lane from grey and erases that full summed gain
#      from the green lane. In the variant only donor excess sprites are grey, and the green gain is diff only.
# d) Placement legality:
#    - All sprites remain on the original lane lattice, so there are no new geometric degrees of freedom.
#    - The donor's excess sprites are chosen from the top of the larger stack, preserving bottom alignment.
#    - Green placement still uses the destination lane's detected period and fills empty slots upward, so
#      no overlaps or out-of-bounds writes are introduced.
# e) Visual observability:
#    - The two red markers now mean "compare these two lanes" rather than "sum both lanes".
#    - Because one red lane remains untouched and only the larger lane loses its top excess, the subtraction
#      pattern is directly visible without relying on color ids or hidden ordering.
# f) Visual cleanliness:
#    - The variant does not add any new markers or overlays; it only changes how many top sprites in one
#      source lane become grey. The lane structure stays uncluttered.
# g) Color inference:
#    - Recoloring remains structurally anchored: grey means transferred-away source sprites, and new green
#      sprites still take the destination lane's sprite color. No free-form color mapping is introduced.
# h) New main() assumptions and how generate_input() guarantees them:
#    - When two red markers exist, there is exactly one larger lane. The generator enforces unequal counts.
#    - The larger lane's excess equals its topmost sprites because all lanes are bottom-stacked on the same
#      period-4 lattice; the generator keeps that lattice unchanged.
#    - Exactly one green lane is generated, and its final count plus transferred diff fits in the grid.


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


def _lane_info_from_marker(
    g: np.ndarray, marker: list[tuple[int, int]], bg: int
) -> dict[str, object] | None:
    _, c0, _, c1 = bbox_coords(np.array(marker))
    cols = range(c0, c1 + 1)
    marker_set = set(marker)
    comps = _single_color_components_in_cols(g, cols, bg)
    sprite_comps = []
    for pts in comps:
        pts_wo_marker = [p for p in pts if p not in marker_set]
        if pts_wo_marker:
            sprite_comps.append(pts_wo_marker)
    if not sprite_comps:
        return None

    sprite_sizes = [len(pts) for pts in sprite_comps if len(pts) != len(marker)]
    if not sprite_sizes:
        return None
    sprite_sz = max(sprite_sizes)
    sprites = [pts for pts in sprite_comps if len(pts) == sprite_sz]
    if not sprites:
        return None

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

    pr, pc = sprites_sorted[0][2][0]
    return {
        "marker": marker,
        "center": (c0 + c1) / 2.0,
        "color": int(g[pr, pc]),
        "period": period,
        "bottom0": max(tops),
        "occupied": set(tops),
        "sprites_sorted": sprites_sorted,
        "count": len(sprites_sorted),
    }


def _grey_sorted_sprites(out: np.ndarray, sprites_sorted: list[tuple[int, int, list[tuple[int, int]]]]) -> None:
    for _rmin, _cmin, pts in sprites_sorted:
        for r, c in pts:
            out[r, c] = 5


def _place_transferred_sprites(
    out: np.ndarray,
    src_info: dict[str, object],
    dst_state: dict[str, object],
    n_new: int,
    bg: int,
) -> None:
    if n_new <= 0:
        return

    h, w = out.shape
    src_ref_r0, _src_ref_c0, src_ref_pts = src_info["sprites_sorted"][0]
    dc_shift = int(round(float(dst_state["center"]) - float(src_info["center"])))
    src_offsets = [(r - src_ref_r0, c + dc_shift) for r, c in src_ref_pts]
    sprite_h = max(dr for dr, _c in src_offsets) + 1

    placed = 0
    k = 0
    while placed < n_new:
        new_top = int(dst_state["bottom0"]) - k * int(dst_state["period"])
        k += 1
        if new_top < 0 or new_top + sprite_h > h:
            if new_top < 0:
                break
            continue
        if new_top in dst_state["occupied"]:
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
            out[new_top + dr, c] = int(dst_state["dst_color"])
        dst_state["occupied"].add(new_top)
        placed += 1


def main(grid: np.ndarray) -> np.ndarray:
    """
    Compare red-marked lanes and transfer only the excess sprites into the green lane.

    Input structure:
    - The background is the most common color. Thin red (2) bars and one thin green (3) bar sit under lanes.
    - Each lane contains repeated identical monochrome plus sprites on the same vertical lattice as before.

    Rule:
    - With one red lane, it behaves like the base task: all of that lane's sprites grey out and the same count
      is added into the green lane.
    - With two red lanes, compare their sprite counts. Only the lane with more sprites donates, and only its
      excess over the smaller red lane is affected: those top excess sprites turn grey and that difference count
      is stacked into the green lane using the destination lane's color.
    - All marker bars are erased afterward.

    Key insight: Two red markers now define a subtraction. The preserved bottom-aligned overlap between the
    red lanes stays in place, while only the unmatched excess from the larger stack is copied to the green lane.
    """
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)

    red_markers = _find_marker_bars(g, 2)
    green_markers = _find_marker_bars(g, 3)
    assert red_markers and green_markers, "expected both red and green marker bars to be present"

    green_state = {}
    for gm in green_markers:
        info = _lane_info_from_marker(g, gm, bg)
        if info is None:
            continue
        green_state[id(gm)] = {
            "center": info["center"],
            "dst_color": info["color"],
            "period": info["period"],
            "occupied": set(info["occupied"]),
            "bottom0": info["bottom0"],
        }
    assert green_state, "expected at least one green lane with valid sprites"

    red_infos = []
    for rm in red_markers:
        info = _lane_info_from_marker(g, rm, bg)
        if info is not None:
            red_infos.append(info)
    assert 1 <= len(red_infos) <= 2, "expected one or two valid red lanes"

    out = g.copy()

    if len(red_infos) == 1:
        donor = red_infos[0]
        donor_sprites = donor["sprites_sorted"]
        _grey_sorted_sprites(out, donor_sprites)
        best = min(green_state.values(), key=lambda st: abs(float(st["center"]) - float(donor["center"])))
        _place_transferred_sprites(out, donor, best, len(donor_sprites), bg)
    else:
        left, right = sorted(red_infos, key=lambda info: float(info["center"]))
        if int(left["count"]) > int(right["count"]):
            donor = left
            baseline = int(right["count"])
        elif int(right["count"]) > int(left["count"]):
            donor = right
            baseline = int(left["count"])
        else:
            donor = None
            baseline = int(left["count"])

        if donor is not None:
            diff = int(donor["count"]) - baseline
            donor_excess = donor["sprites_sorted"][:diff]
            _grey_sorted_sprites(out, donor_excess)
            best = min(green_state.values(), key=lambda st: abs(float(st["center"]) - float(donor["center"])))
            _place_transferred_sprites(out, donor, best, diff, bg)

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
    """Stamp `count` sprites of `color` into the given lane."""
    patch = _sprite_patch(color, bg)
    c0 = _LANE_STARTS[lane_idx]
    for r0 in _lane_tops(count):
        paste(grid, patch, r0, c0, transparent=bg)


def _paint_tops(grid: np.ndarray, lane_idx: int, color: int, tops: list[int], bg: int) -> None:
    """Stamp sprites of `color` into a lane at explicitly chosen top rows."""
    patch = _sprite_patch(color, bg)
    c0 = _LANE_STARTS[lane_idx]
    for r0 in tops:
        paste(grid, patch, r0, c0, transparent=bg)


def _feasible_roles(counts: list[int]) -> list[tuple[tuple[int, int], int, int, int]]:
    """Return feasible (reds, green, donor, diff) assignments for the subtraction variant."""
    feasible = []
    for green in range(_NUM_LANES):
        others = [i for i in range(_NUM_LANES) if i != green]
        for reds in combinations(others, 2):
            a, b = counts[reds[0]], counts[reds[1]]
            diff = abs(a - b)
            if diff == 0:
                continue
            if counts[green] + diff > _MAX_LANE_TOTAL:
                continue
            donor = reds[0] if a > b else reds[1]
            feasible.append((reds, green, donor, diff))
    return feasible


def _gen_attempt() -> np.ndarray | None:
    rng = np.random.default_rng()

    bg = int(rng.choice(_BG_CHOICES))
    sprite_palette = [c for c in range(10) if c not in (bg, _GREY)]
    lane_colors = rng.choice(sprite_palette, size=_NUM_LANES, replace=False).tolist()
    input_counts = rng.integers(_MIN_SPRITES, _MAX_SPRITES + 1, size=_NUM_LANES).tolist()

    feasible = _feasible_roles(input_counts)
    if not feasible:
        return None

    reds, green, donor, diff = feasible[int(rng.integers(len(feasible)))]

    output_counts = input_counts[:]
    output_counts[green] += diff

    out = np.full((_H, _W), bg, dtype=int)
    for lane_idx in range(_NUM_LANES):
        _paint_lane(out, lane_idx, lane_colors[lane_idx], output_counts[lane_idx], bg)

    donor_tops = _lane_tops(input_counts[donor])[:diff]
    _paint_tops(out, donor, _GREY, donor_tops, bg)

    inp = out.copy()
    _paint_tops(inp, donor, lane_colors[donor], donor_tops, bg)

    green_c0 = _LANE_STARTS[green]
    for r0 in _lane_tops(output_counts[green])[:diff]:
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
