# Combo variant of 9aaea919 (index 6)
# combo_spec: D3/0+D5/0
# Name: 顶部向下堆叠+标记在顶行

import sys
from itertools import combinations
from pathlib import Path

import numpy as np

_SOLVER_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_SOLVER_DIR) not in sys.path:
    sys.path.append(str(_SOLVER_DIR))

from arc_utils import bbox_coords, components_mask, get_bg, paste


# Design analysis for combo D3/0 + D5/0
# a) Concrete behavior and trace
#    - The canvas remains 30x30 with four fixed 5-column lanes and the same 3x5 plus sprite on a period-4 lattice.
#    - The two changed pieces are coupled by a vertical symmetry:
#      marker bars move from row 29 to row 0, and the destination fill order flips from bottom-first to top-first.
#    - Concrete trace: suppose the green lane already shows sprites at top rows {21, 25} and a red lane contributes 2 sprites.
#      Original task: markers sit on row 29 and the destination adds rows 17 then 13.
#      This combo: markers sit on row 0 and the destination adds rows 1 then 5, leaving the old bottom block in place.
#    - The visible consequence is strong: the role cues are read from the top edge, and the newly inserted sprites appear
#      in the highest empty slots rather than adjacent to the bottom block.
# b) Complexity calibration
#    - Only two binary inversions are introduced: marker row and slot priority. Lane count, shape, colors, matching by
#      nearest lane center, and greying rule stay unchanged, so the puzzle is still one coherent transfer task.
#    - To keep the changed stacking direction observable, generation forbids fully filling the destination lane; otherwise
#      top-first and bottom-first would end with the same occupied set.
#    - Input lane counts remain 1-3, which guarantees the top slot is empty before transformation and keeps the top-row
#      markers visually isolated from lane sprites.
# c) Hidden-assumption review of the original main()
#    - Marker detection as thin 4-connected bars still works, but the old samples relied on a bottom gap. With markers on
#      row 0, the generator must avoid any input sprite at row 1 or the marker could touch a same-color sprite vertically.
#    - Monochrome component extraction inside each lane still holds because sprite geometry and lane columns do not change.
#    - Period inference from destination sprites still holds because existing green-lane sprites stay on the original
#      residue class; only the priority among empty slots changes.
#    - The broken assumption is the old `bottom0 - k * period` scan, which hard-codes bottom-up growth. The combo solver
#      must reconstruct the full legal slot ladder and scan it from smallest top row downward.
# d) Placement legality
#    - Legal top rows are the period-4 lattice rows where a 3-high sprite fits inside the 30-row canvas: 1,5,9,13,17,21,25.
#    - Output-first generation paints only those rows, erases only whole added sprites, and keeps total destination occupancy
#      below 7, so no overlap or out-of-bounds placement occurs.
#    - Markers on row 0 are one full row above the highest legal sprite top row 1, and because the input never uses row 1,
#      marker bars stay isolated connected components.
# e) Visual observability
#    - The changed rule is readable from a few examples: the clues migrate from the bottom edge to the top edge, and the
#      created destination sprites appear high in the lane instead of hugging the existing bottom cluster.
#    - No hidden numeric ordering is used; slot priority depends only on visible vertical position.
# f) Visual cleanliness
#    - All objects still occupy clean lane-aligned strips with fixed spacing. The top markers form a single tidy legend row
#      and the added sprites occupy empty lattice cells rather than overlapping other structures.
# g) Color inferability
#    - Color semantics remain structured and easy to infer: red-tagged lanes grey out, copied sprites take the destination
#      lane color, and all markers disappear.
# h) New solver assumptions introduced here
#    - `main()` now assumes the destination lane's legal slots can be reconstructed from the observed lattice and that the
#      highest empty slots are background in the input; `generate_input()` guarantees this by keeping input sprites bottom-
#      stacked and erasing exactly the top-priority added sprites from the destination lane.
#    - `main()` also assumes top-row markers remain disconnected from lane sprites; `generate_input()` guarantees this by
#      capping input counts at 3, so row 1 is always empty before transformation.


_SRC_MARKER = 2
_DST_MARKER = 3
_GREY = 5


def _find_marker_bars(g: np.ndarray, marker_color: int) -> list[list[tuple[int, int]]]:
    """Marker bars are thin 4-connected components of the given marker color."""
    cands = []
    for pts_arr in components_mask(g == marker_color, connectivity=4):
        r0, c0, r1, c1 = bbox_coords(pts_arr)
        hh = r1 - r0 + 1
        ww = c1 - c0 + 1
        if hh == 1 or ww == 1:
            pts = [(int(r), int(c)) for r, c in pts_arr]
            cands.append((0 if len(pts) == 5 else 1, -r0, hh * ww, len(pts), pts))
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


def _sorted_sprites(sprites: list[list[tuple[int, int]]]) -> list[tuple[int, int, list[tuple[int, int]]]]:
    items = []
    for pts in sprites:
        rmin = min(r for r, _ in pts)
        cmin = min(c for _, c in pts)
        items.append((rmin, cmin, pts))
    items.sort()
    return items


def _extract_lane_sprites(
    g: np.ndarray, marker_pts: list[tuple[int, int]], bg: int
) -> tuple[float, list[list[tuple[int, int]]]]:
    marker_arr = np.array(marker_pts)
    _, lane_c0, _, lane_c1 = bbox_coords(marker_arr)
    lane_cols = range(lane_c0, lane_c1 + 1)
    marker_set = set(marker_pts)
    comps = _single_color_components_in_cols(g, lane_cols, bg)
    comps = [[p for p in pts if p not in marker_set] for pts in comps]
    comps = [pts for pts in comps if pts]
    center = (lane_c0 + lane_c1) / 2.0
    if not comps:
        return center, []

    sprite_sizes = [len(pts) for pts in comps if len(pts) != len(marker_pts)]
    if not sprite_sizes:
        return center, []

    sprite_sz = max(sprite_sizes)
    sprites = [pts for pts in comps if len(pts) == sprite_sz]
    return center, sprites


def _lane_slot_rows(occupied_tops: list[int], period: int, sprite_h: int, h: int) -> list[int]:
    """Reconstruct all legal lattice top rows for a lane, ordered from top to bottom."""
    top0 = min(occupied_tops)
    while top0 - period >= 0:
        top0 -= period
    rows = []
    r = top0
    while r + sprite_h <= h:
        rows.append(r)
        r += period
    return rows


def main(grid: np.ndarray) -> np.ndarray:
    """
    Transfer sprites from red-tagged lanes into green-tagged lanes using a top-first destination fill, with all role markers moved to the top edge.

    Input structure:
    - The background is the most common color. Thin red (2) and green (3) 1x5 bars on row 0 identify source and destination lanes.
    - Each fixed lane contains repeated monochrome plus sprites on a regular vertical lattice, and the destination lane's visible sprites still occupy ordinary lane slots rather than touching the top-row marker.

    Rule:
    - Every sprite in a red-marked lane turns grey (5), while the same number of copies is inserted into the nearest green-marked lane using that lane's own color.
    - The destination lane fills its highest empty legal slots first, proceeding downward through the lattice instead of growing upward from the bottom cluster.
    - After all transfers, both red and green marker bars are erased back to the background.

    Key insight:
    - The top-row bars still reveal which lanes donate and receive, but the placement logic must be rebuilt from the destination lane's lattice: enumerate all legal slot rows and choose empty ones from top to bottom.
    """
    g = np.asarray(grid, dtype=int)
    h, _w = g.shape
    bg = get_bg(g)

    red_markers = _find_marker_bars(g, _SRC_MARKER)
    green_markers = _find_marker_bars(g, _DST_MARKER)
    assert red_markers and green_markers, "expected both red and green marker bars to be present"

    green_state = {}
    for gm in green_markers:
        gc, dst_sprites = _extract_lane_sprites(g, gm, bg)
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
        green_state[id(gm)] = {
            "center": gc,
            "dst_color": dst_color,
            "occupied": set(tops),
            "slots": _lane_slot_rows(tops, period, sprite_h, h),
        }

    assert green_state, "expected at least one green lane with valid sprites"

    out = g.copy()

    for rm in red_markers:
        rc, src_sprites = _extract_lane_sprites(g, rm, bg)
        if not src_sprites:
            continue

        for pts in src_sprites:
            for r, c in pts:
                out[r, c] = _GREY

        best = min(green_state.values(), key=lambda st: abs(st["center"] - rc))

        src_sprites_sorted = _sorted_sprites(src_sprites)
        src_ref_r0, _src_ref_c0, src_ref_pts = src_sprites_sorted[0]
        dc_shift = int(round(best["center"] - rc))
        src_offsets = [(r - src_ref_r0, c + dc_shift) for r, c in src_ref_pts]

        n_new = len(src_sprites)
        placed = 0
        for new_top in best["slots"]:
            if placed >= n_new:
                break
            if new_top in best["occupied"]:
                continue

            ok = True
            for dr, c in src_offsets:
                rr = new_top + dr
                if not (0 <= rr < h and 0 <= c < g.shape[1]):
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

    for markers in (red_markers, green_markers):
        for marker in markers:
            for r, c in marker:
                out[r, c] = bg

    return out


# --------------- generator ---------------

# --- CFG: canvas & layout ---
_H, _W = 30, 30
_NUM_LANES = 4
_LANE_STARTS = [3, 9, 15, 21]
_MARKER_ROW = 0
_SPRITE_H, _SPRITE_W = 3, 5

# --- CFG: vertical lattice ---
_BOTTOM_TOP = 25
_PERIOD = 4
_TOP_SLOT0 = _BOTTOM_TOP % _PERIOD
_ALL_TOPS = list(range(_TOP_SLOT0, _BOTTOM_TOP + 1, _PERIOD))

# --- CFG: color palette ---
_BG_CHOICES = [0, 1, 4, 6, 7, 8, 9]

# --- CFG: counts & limits ---
_MIN_SPRITES = 1
_MAX_SPRITES = 3
_MAX_LANE_TOTAL = len(_ALL_TOPS)
_GEN_RETRIES = 300


def _sprite_patch(color: int, bg: int) -> np.ndarray:
    """Return a 3x5 plus sprite patch with bg-transparent corners."""
    patch = np.full((_SPRITE_H, _SPRITE_W), bg, dtype=int)
    patch[0, 1:4] = color
    patch[1, :] = color
    patch[2, 1:4] = color
    return patch


def _lane_tops(count: int) -> list[int]:
    """Return bottom-stacked top rows for `count` sprites, ordered top-to-bottom."""
    return [_BOTTOM_TOP - _PERIOD * k for k in range(count - 1, -1, -1)]


def _topdown_added_tops(existing_count: int, gain: int) -> list[int]:
    """Return the variant's destination rows: highest empty slots first."""
    occupied = set(_lane_tops(existing_count))
    empties = [r for r in _ALL_TOPS if r not in occupied]
    return empties[:gain]


def _paint_lane_at_tops(grid: np.ndarray, lane_idx: int, color: int, tops: list[int], bg: int) -> None:
    patch = _sprite_patch(color, bg)
    c0 = _LANE_STARTS[lane_idx]
    for r0 in tops:
        paste(grid, patch, r0, c0, transparent=bg)


def _paint_lane(grid: np.ndarray, lane_idx: int, color: int, count: int, bg: int) -> None:
    """Stamp `count` bottom-stacked sprites of `color` into the given lane."""
    _paint_lane_at_tops(grid, lane_idx, color, _lane_tops(count), bg)


def _feasible_roles(counts: list[int]) -> dict[int, list[tuple[tuple[int, ...], int]]]:
    """Return feasible (reds, green) assignments grouped by number of red lanes."""
    grouped: dict[int, list[tuple[tuple[int, ...], int]]] = {1: [], 2: []}
    for green in range(_NUM_LANES):
        others = [i for i in range(_NUM_LANES) if i != green]
        for red_count in (1, 2):
            for reds in combinations(others, red_count):
                total = counts[green] + sum(counts[i] for i in reds)
                if total < _MAX_LANE_TOTAL:
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

    added_tops = _topdown_added_tops(input_counts[green], gain)
    if len(added_tops) != gain:
        return None

    out = np.full((_H, _W), bg, dtype=int)
    for lane_idx in range(_NUM_LANES):
        color = _GREY if lane_idx in red_set else lane_colors[lane_idx]
        if lane_idx == green:
            tops = sorted(_lane_tops(input_counts[green]) + added_tops)
            _paint_lane_at_tops(out, lane_idx, color, tops, bg)
        else:
            _paint_lane(out, lane_idx, color, input_counts[lane_idx], bg)

    inp = out.copy()
    for lane_idx in reds:
        _paint_lane(inp, lane_idx, lane_colors[lane_idx], input_counts[lane_idx], bg)

    green_c0 = _LANE_STARTS[green]
    for r0 in added_tops:
        inp[r0 : r0 + _SPRITE_H, green_c0 : green_c0 + _SPRITE_W] = bg
    inp[_MARKER_ROW, green_c0 : green_c0 + _SPRITE_W] = _DST_MARKER

    for lane_idx in reds:
        c0 = _LANE_STARTS[lane_idx]
        inp[_MARKER_ROW, c0 : c0 + _SPRITE_W] = _SRC_MARKER

    if not np.array_equal(main(inp), out):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(_GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {_GEN_RETRIES} attempts")
