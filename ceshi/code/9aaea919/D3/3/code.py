# Variant of 9aaea919
# Dimension: D3 — 目标位置与堆叠规则 (dst)
# Change: stacking — 新精灵在目标车道底部已有精灵之上，按周期4(3精灵行+1间隙行)向上堆叠空槽位... → 均匀分散放置

# Design analysis for the D3 variant
# ==================================
# variant_spec:
#   unchanged_structure:
#     grid: 30x30
#     lanes: 4 fixed 5-column lanes at columns 3-7, 9-13, 15-19, 21-25
#     sprite: monochrome 3x5 plus/diamond, repeated on a period-4 vertical lattice
#     markers: red(2) marks source lane(s), green(3) marks exactly one destination lane
#     source_effect: marked red-lane sprites become grey(5)
#     destination_effect: receives the same number of new sprites, recolored to the destination lane color
#     marker_cleanup: all marker bars are erased to background
#   changed_structure:
#     dst_rule:
#       old: fill destination empty lattice slots bottom-up, so new sprites form one tight block above existing sprites
#       new: list every empty lattice slot in the destination lane from top to bottom, then distribute the new sprites
#            evenly across that list instead of packing them at the bottom
#       precise_selection:
#         let E = number of empty slots, N = number of transferred sprites
#         order empty slots by top row ascending (top to bottom)
#         choose slot indices floor((2k+1) * E / (2N)) for k = 0..N-1
#         this is a centered equal-width bin rule; it spreads placements across the whole empty region
#       example:
#         existing destination tops: [21, 25]
#         all lattice tops: [1, 5, 9, 13, 17, 21, 25]
#         empty tops: [1, 5, 9, 13, 17]
#         transfer N=2
#         original stacking would use [13, 17]
#         this variant uses indices floor([1.25, 3.75]) = [1, 3], so tops [5, 13]
#         difference: the new sprites no longer hug the old bottom block; they are visibly spaced apart
# complexity_calibration:
#   why_not_too_simple:
#     the rule is not "pick random gaps"; it is a single deterministic arithmetic spacing rule
#     the destination still uses one repeated lane/lattice structure, so examples stay learnable
#   why_not_ambiguous:
#     generator only emits cases where the number of empty slots is strictly larger than the transfer count,
#     and where the distributed choice differs from the original bottom-packed choice
#     this makes the D3 change visible in the output rather than silently collapsing back to the seed behavior
# implicit_assumption_audit:
#   solver_lines_changed_in_spirit:
#     original while-loop over k assumed new slots are discovered in bottom-up priority order
#     original best["occupied"].add(new_top) inside the red-lane loop made placement depend on source processing order
#     original generator erased topmost gained sprites from a contiguous output stack, which only matches the old packing rule
#   variant_fix:
#     first count all transfers per destination lane
#     then enumerate that lane's full empty-slot list once
#     then choose the evenly spaced subset in one batch
#   preserved_assumptions:
#     destination lane already contains at least one sprite, so period and sprite shape can be inferred
#     lane sprites all share one monochrome shape, and the input green lane uses the same aligned silhouette as the output
# placement_legality:
#   all legal lane tops are taken from the same period-4 lattice anchored by the detected bottom sprite
#   generator only chooses roles where transferred sprites fit into currently empty lattice slots
#   solver still checks every chosen placement against bounds and background before stamping
# visual_observability:
#   no hidden ordering uses color ids or other invisible metadata
#   the new behavior is readable from the vertical spacing pattern alone: the destination lane gains separated inserts
#   generator rejects cases where distributed placement equals the old stacked placement
# visual_tidiness:
#   the lane system, sprite shape, and marker bars stay unchanged, so the scene remains sparse and legible
#   only the vertical arrangement within the destination lane changes

import numpy as np

from arc_utils import bbox_coords, components_mask, get_bg, paste
from itertools import combinations


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


def _sorted_sprites(comps: list[list[tuple[int, int]]]) -> list[tuple[int, int, list[tuple[int, int]]]]:
    sprites = []
    for pts in comps:
        rmin = min(r for r, _ in pts)
        cmin = min(c for _, c in pts)
        sprites.append((rmin, cmin, pts))
    sprites.sort()
    return sprites


def _valid_lane_tops(bottom0: int, period: int, sprite_h: int, h: int) -> list[int]:
    tops_desc = []
    cur = int(bottom0)
    while cur >= 0:
        if cur + sprite_h <= h:
            tops_desc.append(cur)
        cur -= period
    return sorted(tops_desc)


def _select_even_slot_indices(num_slots: int, num_pick: int) -> list[int]:
    assert 0 <= num_pick <= num_slots
    if num_pick == 0:
        return []
    # Center of each equal-width bin, rounded down to an integer slot index.
    return [((2 * k + 1) * num_slots) // (2 * num_pick) for k in range(num_pick)]


def _select_even_slots(empty_tops: list[int], num_pick: int) -> list[int]:
    idxs = _select_even_slot_indices(len(empty_tops), num_pick)
    return [empty_tops[i] for i in idxs]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Transfer sprites from red-marked lanes to green-marked lanes, recoloring and redistributing them.

    Input structure:
    - The background is the most common color. Thin red (2) and green (3) bars mark source and destination lanes.
    - Each lane contains repeated monochrome plus sprites aligned to the same vertical period-4 lattice.

    Rule:
    - Sprites in every red-marked lane turn grey (5), and the same number of replacement sprites appear in the nearest green lane.
    - The destination does not pack the replacements at the bottom; instead, it spreads them evenly across that lane's empty lattice slots.
    - All marker bars disappear in the output.

    Key insight:
    - The destination lane already reveals the sprite shape, the lattice period, and which slots are occupied, so the only new work is to count the transfers and choose an evenly spaced subset of the remaining slots.
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

        ref_r0, _ref_c0, ref_pts = dst_sprites_sorted[0]
        sprite_h = max(r for r, _c in ref_pts) - ref_r0 + 1
        template_offsets = [(r - ref_r0, c) for r, c in ref_pts]

        gc = (dst_c0 + dst_c1) / 2.0
        green_state[id(gm)] = {
            "marker": gm,
            "center": gc,
            "dst_color": dst_color,
            "period": period,
            "occupied": set(tops),
            "bottom0": max(tops),
            "sprite_h": sprite_h,
            "template_offsets": template_offsets,
            "transfer_count": 0,
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
        best["transfer_count"] += len(src_sprites)

    for st in green_state.values():
        n_new = int(st["transfer_count"])
        if n_new == 0:
            continue

        valid_tops = _valid_lane_tops(st["bottom0"], st["period"], st["sprite_h"], h)
        empty_tops = [top for top in valid_tops if top not in st["occupied"]]
        assert n_new <= len(empty_tops), "transfer does not fit in the destination lane"
        chosen_tops = _select_even_slots(empty_tops, n_new)

        for new_top in chosen_tops:
            ok = True
            for dr, c in st["template_offsets"]:
                rr = new_top + dr
                if not (0 <= rr < h and 0 <= c < w):
                    ok = False
                    break
                if int(out[rr, c]) != bg:
                    ok = False
                    break
            assert ok, "expected a clean evenly selected destination slot"
            for dr, c in st["template_offsets"]:
                out[new_top + dr, c] = st["dst_color"]

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
_LANE_STARTS = [3, 9, 15, 21]       # col-0 of each lane's 5-wide sprite slot
_MARKER_ROW = 29                     # row for red/green 1x5 marker bars
_SPRITE_H, _SPRITE_W = 3, 5         # bounding box of the plus/diamond sprite

# --- CFG: vertical lattice ---
_BOTTOM_TOP = 25                     # top-row of the bottommost sprite
_PERIOD = 4                          # vertical period (3 sprite rows + 1 gap)
_ALL_TOPS = list(range(1, _BOTTOM_TOP + 1, _PERIOD))
_MAX_SLOT_COUNT = len(_ALL_TOPS)

# --- CFG: color palette ---
_BG_CHOICES = [0, 1, 4, 6, 7, 8, 9] # valid background colors (excludes 2, 3, 5)
_GREY = 5                            # color used for greyed-out source sprites

# --- CFG: counts & limits ---
_MIN_SPRITES = 1                     # min sprites per lane in input
_MAX_SPRITES = 3                     # max sprites per lane in input
_GEN_RETRIES = 300                   # retry budget for generate_input()


def _sprite_patch(color: int, bg: int) -> np.ndarray:
    """Return a 3x5 plus/diamond sprite patch with bg-transparent corners."""
    patch = np.full((_SPRITE_H, _SPRITE_W), bg, dtype=int)
    patch[0, 1:4] = color
    patch[1, :] = color
    patch[2, 1:4] = color
    return patch


def _lane_tops(count: int) -> list[int]:
    """Return top-row positions for `count` bottom-stacked sprites, ordered top-to-bottom."""
    return [_BOTTOM_TOP - _PERIOD * k for k in range(count - 1, -1, -1)]


def _empty_lane_tops(existing_count: int) -> list[int]:
    occupied = set(_lane_tops(existing_count))
    return [top for top in _ALL_TOPS if top not in occupied]


def _paint_lane_tops(grid: np.ndarray, lane_idx: int, color: int, tops: list[int], bg: int) -> None:
    patch = _sprite_patch(color, bg)
    c0 = _LANE_STARTS[lane_idx]
    for r0 in tops:
        paste(grid, patch, r0, c0, transparent=bg)


def _paint_lane(grid: np.ndarray, lane_idx: int, color: int, count: int, bg: int) -> None:
    """Stamp `count` contiguous bottom-stacked sprites of `color` into the given lane."""
    _paint_lane_tops(grid, lane_idx, color, _lane_tops(count), bg)


def _stacked_new_tops(existing_count: int, gain: int) -> list[int]:
    empty_tops = _empty_lane_tops(existing_count)
    return empty_tops[-gain:]


def _distributed_new_tops(existing_count: int, gain: int) -> list[int]:
    empty_tops = _empty_lane_tops(existing_count)
    return _select_even_slots(empty_tops, gain)


def _feasible_roles(counts: list[int]) -> dict[int, list[tuple[tuple[int, ...], int]]]:
    """Return visible D3-variant role assignments grouped by number of red lanes."""
    grouped: dict[int, list[tuple[tuple[int, ...], int]]] = {1: [], 2: []}
    for green in range(_NUM_LANES):
        empty_slots = _MAX_SLOT_COUNT - counts[green]
        others = [i for i in range(_NUM_LANES) if i != green]
        for red_count in (1, 2):
            for reds in combinations(others, red_count):
                gain = sum(counts[i] for i in reds)
                if gain > empty_slots:
                    continue
                if gain < 2:
                    continue
                if empty_slots - gain < 2:
                    continue
                if _distributed_new_tops(counts[green], gain) == _stacked_new_tops(counts[green], gain):
                    continue
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
    red_set = set(reds)
    gain = sum(input_counts[i] for i in reds)

    green_existing_tops = _lane_tops(input_counts[green])
    green_new_tops = _distributed_new_tops(input_counts[green], gain)
    green_output_tops = sorted(green_existing_tops + green_new_tops)

    out = np.full((_H, _W), bg, dtype=int)
    for lane_idx in range(_NUM_LANES):
        if lane_idx == green:
            _paint_lane_tops(out, lane_idx, lane_colors[lane_idx], green_output_tops, bg)
        else:
            color = _GREY if lane_idx in red_set else lane_colors[lane_idx]
            _paint_lane(out, lane_idx, color, input_counts[lane_idx], bg)

    inp = out.copy()
    for lane_idx in reds:
        _paint_lane(inp, lane_idx, lane_colors[lane_idx], input_counts[lane_idx], bg)

    green_c0 = _LANE_STARTS[green]
    for r0 in green_new_tops:
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
