# Variant of 9aaea919
# Dimension: D12 — 源-目标基数关系 (meta)
# Change: many:1 — 1-2个红色源车道转移到1个绿色目标车道... → many:many — 多源多目标配对

import numpy as np

from itertools import permutations

from arc_utils import bbox_coords, components_mask, get_bg, paste


# Design analysis for the D12 many:many variant
# a) Concrete behavior:
#    - Keep the original 30x30 canvas, four fixed 5-column lanes, 3x5 plus sprites, and last-row marker bars.
#    - Change only the source/target cardinality: exactly two red marker bars and exactly two green marker bars.
#    - Each red lane still turns grey in place, but now each red lane transfers to exactly one green lane.
#    - The pairing is a global one-to-one assignment by horizontal distance, not a per-lane greedy lookup.
#    - Small example: with red lanes at x=5 and x=11, green lanes at x=17 and x=23, a local many:1 rule sends
#      both reds to x=17. The new rule must split them 1:1, so one source feeds x=17 and the other feeds x=23.
# b) Complexity calibration:
#    - The variant must visibly differ from the original, so the generator forces "contention" layouts where both
#      red lanes would prefer the same green lane under the old greedy rule.
#    - The two red lanes are also forced to have different sprite counts, so the chosen pairing changes the output.
# c) Audit of original assumptions:
#    - `best = min(green_state.values(), key=...)` assumed every red can reuse the same destination; this is invalid.
#    - Updating one green lane's occupied slots while processing reds sequentially was harmless before because reuse
#      was allowed. Now the full matching must be solved before any placement to avoid order-dependent behavior.
#    - Sorting by lane center is still valid because lane geometry is unchanged and the rule is based on visible x-position.
# d) Placement legality:
#    - Each green lane receives sprites from only one red lane, so the original bottom-up stacking remains legal.
#    - With input counts limited to 1..3, every green lane ends with at most 6 sprites, which fits in the 30x30 grid.
# e) Visual observability:
#    - The pairing depends only on visible lane positions and visible stack sizes; no color-id ordering is used.
#    - Because the source counts differ, examples expose which green lane received which source stack.
# f) Visual cleanliness:
#    - No new object types are introduced. Markers stay on the last row, sprites stay inside their lanes, and all
#      additions happen on the existing vertical lattice, so the picture remains easy to parse.
# g) Color inference:
#    - Colors behave exactly as in the original task: sources become grey, transferred sprites take the target lane color.
# h) New solver assumptions checked against the generator:
#    - There are exactly two red and two green markers.
#    - Every marked lane contains at least one monochrome sprite stack with the original plus shape.
#    - The marker bars do not touch sprites, and lane columns remain separated by background gaps.


def _find_marker_bars(g: np.ndarray, marker_color: int) -> list[list[tuple[int, int]]]:
    """Marker bars are thin 4-connected components of the requested color."""
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


def _sorted_sprites(sprites: list[list[tuple[int, int]]]) -> list[tuple[int, int, list[tuple[int, int]]]]:
    items = []
    for pts in sprites:
        rmin = min(r for r, _ in pts)
        cmin = min(c for _, c in pts)
        items.append((rmin, cmin, pts))
    items.sort()
    return items


def _extract_lane_state(g: np.ndarray, marker: list[tuple[int, int]], bg: int) -> dict | None:
    """Collect geometry, color, and sprite-template data for one marked lane."""
    _, c0, _, c1 = bbox_coords(np.array(marker))
    cols = range(c0, c1 + 1)
    marker_set = set(marker)

    comps = _single_color_components_in_cols(g, cols, bg)
    comps = [[p for p in pts if p not in marker_set] for pts in comps]
    comps = [pts for pts in comps if pts]
    if not comps:
        return None

    non_marker_sizes = [len(p) for p in comps if len(p) != len(marker)]
    if not non_marker_sizes:
        return None
    sprite_size = max(non_marker_sizes)
    sprites = [pts for pts in comps if len(pts) == sprite_size]
    if not sprites:
        return None

    sprites_sorted = _sorted_sprites(sprites)
    tops = sorted({rmin for rmin, _cmin, _pts in sprites_sorted})
    if len(tops) >= 2:
        diffs = [b - a for a, b in zip(tops, tops[1:]) if b - a > 0]
        period = int(np.gcd.reduce(diffs)) if diffs else 4
    else:
        r0, _cc0, r1, _cc1 = bbox_coords(np.array(sprites_sorted[0][2]))
        period = (r1 - r0 + 1) + 1

    ref_top, _ref_left, ref_pts = sprites_sorted[0]
    ref_color = int(g[ref_pts[0][0], ref_pts[0][1]])
    sprite_h = max(r for r, _ in ref_pts) - ref_top + 1
    return {
        "marker": marker,
        "center": (c0 + c1) / 2.0,
        "color": ref_color,
        "sprites": [pts for _r0, _c0, pts in sprites_sorted],
        "tops": tops,
        "period": period,
        "occupied": set(tops),
        "bottom0": max(tops),
        "ref_top": ref_top,
        "ref_pts": ref_pts,
        "sprite_h": sprite_h,
        "cols": cols,
    }


def _choose_pairing(red_states: list[dict], green_states: list[dict]) -> list[tuple[dict, dict]]:
    """
    Minimum-total-distance one-to-one pairing.

    Ties are broken by the number of crossing assignments, then lexicographically, which
    produces the left-to-right non-crossing pairing on the fixed lane line.
    """
    reds = sorted(red_states, key=lambda st: st["center"])
    greens = sorted(green_states, key=lambda st: st["center"])
    assert len(reds) == len(greens) and reds, "expected equal non-empty red/green marker sets"

    best_key = None
    best_perm = None
    for perm in permutations(range(len(greens))):
        cost = sum(abs(reds[i]["center"] - greens[j]["center"]) for i, j in enumerate(perm))
        crossings = sum(int(perm[i] > perm[j]) for i in range(len(perm)) for j in range(i + 1, len(perm)))
        key = (cost, crossings, tuple(perm))
        if best_key is None or key < best_key:
            best_key = key
            best_perm = perm

    assert best_perm is not None
    return [(reds[i], greens[best_perm[i]]) for i in range(len(reds))]


def _place_transfer(out: np.ndarray, src: dict, dst: dict, bg: int) -> None:
    """Recreate all sprites from one source lane into its assigned destination lane."""
    dc_shift = int(round(dst["center"] - src["center"]))
    src_offsets = [(r - src["ref_top"], c + dc_shift) for r, c in src["ref_pts"]]

    placed = 0
    needed = len(src["sprites"])
    step = 0
    while placed < needed:
        new_top = dst["bottom0"] - step * dst["period"]
        step += 1
        if new_top < 0:
            break
        if new_top + src["sprite_h"] > out.shape[0]:
            continue
        if new_top in dst["occupied"]:
            continue

        ok = True
        for dr, cc in src_offsets:
            rr = new_top + dr
            if not (0 <= rr < out.shape[0] and 0 <= cc < out.shape[1]):
                ok = False
                break
            if int(out[rr, cc]) != bg:
                ok = False
                break
        if not ok:
            continue

        for dr, cc in src_offsets:
            out[new_top + dr, cc] = dst["color"]
        dst["occupied"].add(new_top)
        placed += 1

    assert placed == needed, "destination lane lacked enough empty lattice slots"


def main(grid: np.ndarray) -> np.ndarray:
    """
    Match each red source lane to one green destination lane, then transfer the stacks.

    Input structure:
    - The background is the dominant color; the last row contains exactly two red 1x5 bars and two green 1x5 bars.
    - Each marked lane contains repeated monochrome plus sprites on the same vertical lattice used in the original task.

    Rule:
    - Every red lane turns grey in place, and its whole sprite stack is recreated inside exactly one green lane.
    - Red and green lanes are paired by a global nearest assignment in x-position; equal-cost ties use left-to-right non-crossing order.
    - Recreated sprites take the destination lane's color, occupy empty slots from the bottom upward, and all marker bars are erased.

    Key insight:
    - The main challenge is to solve the red-to-green pairing across all marked lanes before placing anything, because local greedy matching can send both sources to the same destination and break the variant.
    """
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)

    red_markers = _find_marker_bars(g, 2)
    green_markers = _find_marker_bars(g, 3)
    assert len(red_markers) == 2 and len(green_markers) == 2, "expected exactly two red and two green markers"

    red_states = [_extract_lane_state(g, rm, bg) for rm in red_markers]
    green_states = [_extract_lane_state(g, gm, bg) for gm in green_markers]
    assert all(st is not None for st in red_states + green_states), "every marked lane must contain a valid sprite stack"
    red_states = [st for st in red_states if st is not None]
    green_states = [st for st in green_states if st is not None]

    pairs = _choose_pairing(red_states, green_states)
    out = g.copy()

    for src, _dst in pairs:
        for pts in src["sprites"]:
            for r, c in pts:
                out[r, c] = 5

    for src, dst in pairs:
        _place_transfer(out, src, dst, bg)

    for marker in red_markers + green_markers:
        for r, c in marker:
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
_MAX_LANE_TOTAL = 6
_GEN_RETRIES = 300

# Layouts chosen to force a genuine pairing problem:
# both red lanes would pick the same green lane under the original per-lane greedy rule.
_ROLE_LAYOUTS = [
    ((0, 1), (2, 3)),
    ((2, 3), (0, 1)),
]


def _sprite_patch(color: int, bg: int) -> np.ndarray:
    """Return the 3x5 plus/diamond sprite with transparent corners."""
    patch = np.full((_SPRITE_H, _SPRITE_W), bg, dtype=int)
    patch[0, 1:4] = color
    patch[1, :] = color
    patch[2, 1:4] = color
    return patch


def _lane_tops(count: int) -> list[int]:
    """Top rows of a bottom-aligned stack, ordered from top to bottom."""
    return [_BOTTOM_TOP - _PERIOD * k for k in range(count - 1, -1, -1)]


def _paint_lane(grid: np.ndarray, lane_idx: int, color: int, count: int, bg: int) -> None:
    patch = _sprite_patch(color, bg)
    c0 = _LANE_STARTS[lane_idx]
    for r0 in _lane_tops(count):
        paste(grid, patch, r0, c0, transparent=bg)


def _lane_center(lane_idx: int) -> float:
    return _LANE_STARTS[lane_idx] + (_SPRITE_W - 1) / 2.0


def _pair_roles(reds: tuple[int, int], greens: tuple[int, int]) -> dict[int, int]:
    red_states = [{"center": _lane_center(idx), "lane": idx} for idx in reds]
    green_states = [{"center": _lane_center(idx), "lane": idx} for idx in greens]
    pairs = _choose_pairing(red_states, green_states)
    return {src["lane"]: dst["lane"] for src, dst in pairs}


def _gen_attempt() -> np.ndarray | None:
    rng = np.random.default_rng()

    bg = int(rng.choice(_BG_CHOICES))
    sprite_palette = [c for c in range(10) if c not in (bg, _GREY)]
    lane_colors = rng.choice(sprite_palette, size=_NUM_LANES, replace=False).tolist()

    reds, greens = _ROLE_LAYOUTS[int(rng.integers(len(_ROLE_LAYOUTS)))]
    pair_map = _pair_roles(reds, greens)

    input_counts = rng.integers(_MIN_SPRITES, _MAX_SPRITES + 1, size=_NUM_LANES).tolist()
    if input_counts[reds[0]] == input_counts[reds[1]]:
        return None

    output_counts = input_counts[:]
    for red_lane, green_lane in pair_map.items():
        output_counts[green_lane] += input_counts[red_lane]
    if any(output_counts[g] > _MAX_LANE_TOTAL for g in greens):
        return None

    out = np.full((_H, _W), bg, dtype=int)
    for lane_idx in range(_NUM_LANES):
        color = _GREY if lane_idx in reds else lane_colors[lane_idx]
        _paint_lane(out, lane_idx, color, output_counts[lane_idx], bg)

    inp = out.copy()
    for lane_idx in reds:
        _paint_lane(inp, lane_idx, lane_colors[lane_idx], input_counts[lane_idx], bg)

    for red_lane, green_lane in pair_map.items():
        gain = input_counts[red_lane]
        green_c0 = _LANE_STARTS[green_lane]
        for r0 in _lane_tops(output_counts[green_lane])[:gain]:
            inp[r0 : r0 + _SPRITE_H, green_c0 : green_c0 + _SPRITE_W] = bg

    for lane_idx in reds:
        c0 = _LANE_STARTS[lane_idx]
        inp[_MARKER_ROW, c0 : c0 + _SPRITE_W] = 2
    for lane_idx in greens:
        c0 = _LANE_STARTS[lane_idx]
        inp[_MARKER_ROW, c0 : c0 + _SPRITE_W] = 3

    if not np.array_equal(main(inp), out):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(_GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {_GEN_RETRIES} attempts")
