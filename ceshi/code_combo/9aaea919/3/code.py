# Combo variant of 9aaea919 (index 3)
# combo_spec: D2/2+D4/0
# Name: 多形状精灵+保持源颜色

from __future__ import annotations

"""
Design analysis for the combo variant
=====================================

source_task:
- Base solver: original 9aaea919.
- Reference variants consulted: D2/2 (different per-lane shapes) and D4/0 (preserve source color).

a) Concrete behavior and trace
------------------------------
- The canvas is still a 30x30 board with four fixed 5-column lanes, bottom markers on row 29, and a
  period-4 vertical lattice whose sprite top rows come from {1, 5, 9, 13, 17, 21, 25}.
- The fusion change is twofold at once:
  1. each lane owns a different 3x5 sprite silhouette chosen from {plus, T, L, square};
  2. transferred sprites keep their source lane color instead of adopting the destination lane color.
- Example trace:
  - input: lane A is red-marked and contains 2 orange L sprites, lane C is green-marked and contains
    2 blue square sprites.
  - original task output: lane A turns grey, lane C gains 2 more blue plus-shaped copies.
  - D2-only output: lane A turns grey, lane C gains 2 blue L sprites.
  - this combo output: lane A turns grey, lane C gains 2 orange L sprites.
- The observable difference is therefore coupled, not additive: the destination lane now mixes both
  foreign shapes and foreign colors in one stack, so the solver must carry source geometry and source
  color through the same transfer.

b) Complexity calibration
-------------------------
- Four shapes are enough to produce clear silhouette diversity while keeping one fixed 3x5 footprint and
  one fixed period-4 lattice. That avoids introducing extra scale or rotation rules.
- Lane colors remain distinct and are drawn from a palette that excludes marker colors 2 and 3, so the
  "keep source color" behavior stays easy to see and does not visually merge with the marker bars.
- The generator allows one-source and two-source cases, which yields destination lanes with 2 or 3 visible
  color/shape groups. That is rich enough to expose the fusion rule without becoming combinatorially noisy.

c) Hidden assumptions in the original main()
--------------------------------------------
- Original assumption: useful components in a lane can be selected by maximum pixel count.
  This breaks because plus=11 pixels, square=9, T/L=7.
- Original assumption: transferred pixels should be painted with the destination lane color.
  This breaks because D4/0 requires source-color preservation.
- Original assumption: shifting the source template by destination-center offset is safe.
  This still works if the shift is expressed as slot-local column offsets inside the 5-column lane, not as
  absolute coordinates tied to the plus shape.
- Original assumption: tracking occupied destination slots only by sprite top row is enough.
  This still holds because every allowed shape is 3 rows tall and all lanes share the same period 4.
- Replacement strategy:
  - extract each lane's repeated sprite type by normalized shape signature, not by area;
  - keep source slot-local offsets when copying a template into the destination lane;
  - paint inserted sprites with the current source lane color.

d) Placement legality
---------------------
- Every sprite lives inside one fixed 5-column lane slot, so horizontal overlap across lanes cannot occur.
- Every sprite has height 3 and uses the shared period 4, so distinct top rows imply non-overlap vertically.
- The generator enforces `dst_count + sum(src_counts) <= 7`, which keeps the highest inserted top row at 1.
- Markers occupy row 29, while the lowest sprite row is 27, leaving a guaranteed blank separator row 28.

e) Visual observability
-----------------------
- The rule relies only on visible structure: marker position, lane membership, sprite silhouette, and sprite color.
- No ordering depends on invisible facts such as numeric color id rank; source lanes are processed left-to-right
  by marker position, which a human can also inspect.
- Because the destination lane always starts with at least one native sprite and receives at least one transferred
  sprite from a different-shape lane, the output always exhibits both shape heterogeneity and color heterogeneity.

f) Visual cleanliness
---------------------
- Mixed outputs are confined to a single destination lane; the other lanes remain clean vertical stacks.
- Each sprite remains monochrome and separated by one blank row, so foreign colors and silhouettes are readable.
- Reserving colors 2 and 3 for markers prevents the bottom annotations from being mistaken for lane objects.

g) Change inferability
----------------------
- The source lanes are exactly the ones that lose their original colors and become grey.
- Those lost colors reappear in the destination lane, attached to shapes that also match the source lanes.
- Since the destination lane's own original color remains present at the bottom, a viewer can distinguish
  "native destination sprites" from "inserted source batches" across only a few examples.

h) Generator strategy adjustment
--------------------------------
- The generator builds input and expected output independently from the sampled lane shapes, colors, counts,
  and role assignment. It does not rely on `main()` to synthesize the answer.
- After construction it still checks `np.array_equal(main(inp), out)` as an oracle.
- It also verifies a fusion-specific metric on pixels: the output destination lane must contain exactly
  `1 + number_of_red_lanes` distinct colors and the same number of distinct shape signatures.

i) New assumptions introduced by the combo solver
-------------------------------------------------
- Assumption: each lane contains repeated copies of one monochrome shape signature.
  Generator guarantee: one sampled shape key is assigned per lane and repeated for all counts in that lane.
- Assumption: all allowed shapes fit inside the common 3x5 slot with height 3.
  Generator guarantee: the shape library is fixed to four such templates.
- Assumption: the destination lane already has at least one native sprite so its occupied lattice is observable.
  Generator guarantee: counts stay in 1..3 for every lane, including the green lane.
- Assumption: red markers are processed left-to-right when multiple sources feed one destination.
  Generator guarantee: role tuples are sorted and expected outputs are constructed in the same order.
"""

import sys
from pathlib import Path

import numpy as np

_UTILS_DIR = None
for _parent in Path(__file__).resolve().parents:
    for _cand_name in ("solvers", "arc2_solver_select"):
        _cand = _parent / _cand_name
        if (_cand / "arc_utils.py").exists():
            _UTILS_DIR = _cand
            break
    if _UTILS_DIR is not None:
        break
if _UTILS_DIR is None:
    raise ImportError("could not locate arc_utils.py from combo variant path")
if str(_UTILS_DIR) not in sys.path:
    sys.path.insert(0, str(_UTILS_DIR))

from arc_utils import bbox_coords, components_mask, get_bg, paste


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
        colors = {int(g[r, c]) for r, c in pts}
        if len(colors) == 1:
            out.append(pts)
    return out


def _component_signature(pts: list[tuple[int, int]]) -> tuple[int, int, tuple[tuple[int, int], ...]]:
    arr = np.array(pts, dtype=int)
    r0, c0, r1, c1 = bbox_coords(arr)
    norm = tuple(sorted((int(r - r0), int(c - c0)) for r, c in pts))
    return (r1 - r0 + 1, c1 - c0 + 1, norm)


def _sorted_sprites(groups: list[list[tuple[int, int]]]) -> list[list[tuple[int, int]]]:
    return sorted(groups, key=lambda pts: (min(r for r, _ in pts), min(c for _, c in pts)))


def _extract_lane_sprites(
    g: np.ndarray, cols: range, marker_pts: list[tuple[int, int]], bg: int
) -> list[list[tuple[int, int]]]:
    """
    Extract the repeated sprite components from a marked lane.

    The combo variant cannot select by max component area because different lanes use different shapes.
    We instead group components by normalized signature and keep the repeated group for that lane.
    """
    marker_set = set(marker_pts)
    grouped: dict[tuple[int, int, tuple[tuple[int, int], ...]], list[list[tuple[int, int]]]] = {}
    for pts in _single_color_components_in_cols(g, cols, bg):
        kept = [p for p in pts if p not in marker_set]
        if not kept:
            continue
        sig = _component_signature(kept)
        grouped.setdefault(sig, []).append(kept)
    if not grouped:
        return []
    best_group = max(grouped.values(), key=lambda group: (len(group), len(group[0])))
    return _sorted_sprites(best_group)


def _infer_period(sprites: list[list[tuple[int, int]]]) -> int:
    tops = sorted(min(r for r, _ in pts) for pts in sprites)
    if len(tops) >= 2:
        diffs = [b - a for a, b in zip(tops, tops[1:]) if b - a > 0]
        if diffs:
            return int(np.gcd.reduce(diffs))
    r0, _c0, r1, _c1 = bbox_coords(np.array(sprites[0], dtype=int))
    return (r1 - r0 + 1) + 1


def main(grid: np.ndarray) -> np.ndarray:
    """
    Transfer marked source-lane sprites into the destination lane while preserving both shape and source color.

    Input structure:
    - The background is the most common color. Thin red (2) and green (3) bars on the last row mark source and destination lanes.
    - The four lanes share one 3-row-high, period-4 vertical lattice, but each lane repeats its own monochrome 3x5 sprite shape.

    Rule:
    - Every sprite in a red-marked lane turns grey (5), and the same number of sprites is recreated in the nearest green-marked lane.
    - Recreated sprites keep the exact source-lane silhouette and keep the source lane's original color; they fill empty lattice slots from the bottom upward.
    - When multiple red lanes feed the same destination, their batches stack in left-to-right source-lane order. All marker bars are then removed.

    Key insight:
    - The destination lane no longer supplies either defining attribute of the inserted objects: shape comes from the source lane, and color also comes from the source lane. Only the destination lane contributes the empty slots that receive the transferred batch.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = get_bg(g)

    red_markers = sorted(_find_marker_bars(g, 2), key=lambda pts: bbox_coords(np.array(pts, dtype=int))[1])
    green_markers = sorted(_find_marker_bars(g, 3), key=lambda pts: bbox_coords(np.array(pts, dtype=int))[1])
    assert red_markers and green_markers, "expected both red and green marker bars to be present"

    green_state = {}
    for gm in green_markers:
        _r0, dst_c0, _r1, dst_c1 = bbox_coords(np.array(gm, dtype=int))
        dst_cols = range(dst_c0, dst_c1 + 1)
        dst_sprites = _extract_lane_sprites(g, dst_cols, gm, bg)
        if not dst_sprites:
            continue

        tops = sorted(min(r for r, _ in pts) for pts in dst_sprites)
        green_state[(dst_c0, dst_c1)] = {
            "center": (dst_c0 + dst_c1) / 2.0,
            "c0": dst_c0,
            "period": _infer_period(dst_sprites),
            "occupied": set(tops),
            "bottom0": max(tops),
        }

    assert green_state, "expected at least one green lane with valid sprites"

    out = g.copy()

    for rm in red_markers:
        _r0, src_c0, _r1, src_c1 = bbox_coords(np.array(rm, dtype=int))
        src_cols = range(src_c0, src_c1 + 1)
        src_sprites = _extract_lane_sprites(g, src_cols, rm, bg)
        if not src_sprites:
            continue

        for pts in src_sprites:
            for r, c in pts:
                out[r, c] = 5

        src_center = (src_c0 + src_c1) / 2.0
        best = min(green_state.values(), key=lambda st: abs(st["center"] - src_center))

        src_ref_pts = src_sprites[0]
        src_ref_r0 = min(r for r, _ in src_ref_pts)
        src_color = int(g[src_ref_pts[0][0], src_ref_pts[0][1]])
        src_offsets = [(r - src_ref_r0, c - src_c0) for r, c in src_ref_pts]
        sprite_h = max(dr for dr, _dc in src_offsets) + 1

        placed = 0
        probe = 0
        while placed < len(src_sprites):
            new_top = best["bottom0"] - probe * best["period"]
            probe += 1
            if new_top < 0 or new_top + sprite_h > h:
                if new_top < 0:
                    break
                continue
            if new_top in best["occupied"]:
                continue

            ok = True
            for dr, dc in src_offsets:
                rr = new_top + dr
                cc = best["c0"] + dc
                if not (0 <= rr < h and 0 <= cc < w):
                    ok = False
                    break
                if int(out[rr, cc]) != bg:
                    ok = False
                    break
            if not ok:
                continue

            for dr, dc in src_offsets:
                out[new_top + dr, best["c0"] + dc] = src_color
            best["occupied"].add(new_top)
            placed += 1

    for rm in red_markers:
        for r, c in rm:
            out[r, c] = bg
    for gm in green_markers:
        for r, c in gm:
            out[r, c] = bg

    return out


# --------------- generator (combo variant) ---------------

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
_SPRITE_COLOR_CHOICES = [0, 1, 4, 6, 7, 8, 9]
_GREY = 5

# --- CFG: counts & limits ---
_MIN_SPRITES = 1
_MAX_SPRITES = 3
_MAX_LANE_TOTAL = 7
_GEN_RETRIES = 300

# --- CFG: shape library ---
_SHAPE_MASKS: dict[str, tuple[str, str, str]] = {
    "plus": (".XXX.", "XXXXX", ".XXX."),
    "tee": ("XXXXX", "..X..", "..X.."),
    "ell": ("X....", "X....", "XXXXX"),
    "square": (".XXX.", ".XXX.", ".XXX."),
}
_SHAPE_KEYS = tuple(_SHAPE_MASKS)


def _shape_signature(shape_key: str) -> tuple[int, int, tuple[tuple[int, int], ...]]:
    pts = []
    for r, row in enumerate(_SHAPE_MASKS[shape_key]):
        for c, ch in enumerate(row):
            if ch == "X":
                pts.append((r, c))
    return _component_signature(pts)


def _sprite_patch(shape_key: str, color: int, bg: int) -> np.ndarray:
    patch = np.full((_SPRITE_H, _SPRITE_W), bg, dtype=int)
    for r, row in enumerate(_SHAPE_MASKS[shape_key]):
        for c, ch in enumerate(row):
            if ch == "X":
                patch[r, c] = color
    return patch


def _lane_tops(count: int) -> list[int]:
    return [_BOTTOM_TOP - _PERIOD * k for k in range(count - 1, -1, -1)]


def _paint_lane(grid: np.ndarray, lane_idx: int, shape_key: str, color: int, count: int, bg: int) -> None:
    patch = _sprite_patch(shape_key, color, bg)
    c0 = _LANE_STARTS[lane_idx]
    for r0 in _lane_tops(count):
        paste(grid, patch, r0, c0, transparent=bg)


def _sample_balanced_roles_and_counts(
    rng: np.random.Generator,
) -> tuple[list[int], tuple[int, ...], int] | None:
    """
    Sample branch structure first so one-source and two-source cases both appear regularly.

    The previous generator chose counts first and then filtered for feasible role assignments, which made
    two-source cases noticeably rarer because they more often violate the lane-capacity limit.
    """
    red_count = int(rng.choice((1, 2)))
    green = int(rng.integers(_NUM_LANES))
    others = [i for i in range(_NUM_LANES) if i != green]
    reds = tuple(sorted(int(x) for x in rng.choice(others, size=red_count, replace=False).tolist()))

    for _ in range(64):
        counts = rng.integers(_MIN_SPRITES, _MAX_SPRITES + 1, size=_NUM_LANES).tolist()
        if counts[green] + sum(counts[i] for i in reds) <= _MAX_LANE_TOTAL:
            return counts, reds, green
    return None


def _lane_component_entries(
    g: np.ndarray, lane_idx: int, bg: int
) -> list[tuple[int, int, int, tuple[int, int, tuple[tuple[int, int], ...]], list[tuple[int, int]]]]:
    c0 = _LANE_STARTS[lane_idx]
    cols = range(c0, c0 + _SPRITE_W)
    entries = []
    for pts in _single_color_components_in_cols(g, cols, bg):
        rmin = min(r for r, _ in pts)
        cmin = min(c for _, c in pts)
        color = int(g[pts[0][0], pts[0][1]])
        sig = _component_signature(pts)
        entries.append((rmin, cmin, color, sig, pts))
    entries.sort()
    return entries


def _build_expected_output(
    lane_shapes: list[str], lane_colors: list[int], counts: list[int], reds: tuple[int, ...], green: int, bg: int
) -> np.ndarray | None:
    out = np.full((_H, _W), bg, dtype=int)
    red_set = set(reds)
    for lane_idx in range(_NUM_LANES):
        color = _GREY if lane_idx in red_set else lane_colors[lane_idx]
        _paint_lane(out, lane_idx, lane_shapes[lane_idx], color, counts[lane_idx], bg)

    occupied = set(_lane_tops(counts[green]))
    green_c0 = _LANE_STARTS[green]
    for lane_idx in reds:
        patch = _sprite_patch(lane_shapes[lane_idx], lane_colors[lane_idx], bg)
        placed = 0
        probe = 0
        while placed < counts[lane_idx]:
            new_top = _BOTTOM_TOP - _PERIOD * probe
            probe += 1
            if new_top < 0 or new_top + _SPRITE_H > _H:
                return None
            if new_top in occupied:
                continue
            paste(out, patch, new_top, green_c0, transparent=bg)
            occupied.add(new_top)
            placed += 1
    return out


def _build_input(
    lane_shapes: list[str], lane_colors: list[int], counts: list[int], reds: tuple[int, ...], green: int, bg: int
) -> np.ndarray:
    inp = np.full((_H, _W), bg, dtype=int)
    for lane_idx in range(_NUM_LANES):
        _paint_lane(inp, lane_idx, lane_shapes[lane_idx], lane_colors[lane_idx], counts[lane_idx], bg)

    green_c0 = _LANE_STARTS[green]
    inp[_MARKER_ROW, green_c0 : green_c0 + _SPRITE_W] = 3
    for lane_idx in reds:
        c0 = _LANE_STARTS[lane_idx]
        inp[_MARKER_ROW, c0 : c0 + _SPRITE_W] = 2
    return inp


def _fusion_visible(
    out: np.ndarray, lane_shapes: list[str], lane_colors: list[int], counts: list[int], reds: tuple[int, ...], green: int, bg: int
) -> bool:
    green_entries = _lane_component_entries(out, green, bg)
    expected_total = counts[green] + sum(counts[i] for i in reds)
    if len(green_entries) != expected_total:
        return False

    colors = {color for _r0, _c0, color, _sig, _pts in green_entries}
    sigs = {sig for _r0, _c0, _color, sig, _pts in green_entries}
    expected_colors = {lane_colors[green]} | {lane_colors[i] for i in reds}
    expected_sigs = {_shape_signature(lane_shapes[green])} | {_shape_signature(lane_shapes[i]) for i in reds}
    if colors != expected_colors or sigs != expected_sigs:
        return False

    dst_color = lane_colors[green]
    dst_sig = _shape_signature(lane_shapes[green])
    return any(color != dst_color and sig != dst_sig for _r0, _c0, color, sig, _pts in green_entries)


def _gen_attempt() -> np.ndarray | None:
    rng = np.random.default_rng()

    bg = int(rng.choice(_BG_CHOICES))
    sprite_palette = [c for c in _SPRITE_COLOR_CHOICES if c != bg]
    lane_colors = rng.choice(sprite_palette, size=_NUM_LANES, replace=False).tolist()
    lane_shapes = rng.permutation(_SHAPE_KEYS).tolist()
    sampled = _sample_balanced_roles_and_counts(rng)
    if sampled is None:
        return None
    counts, reds, green = sampled

    out = _build_expected_output(lane_shapes, lane_colors, counts, reds, green, bg)
    if out is None:
        return None
    inp = _build_input(lane_shapes, lane_colors, counts, reds, green, bg)

    if not _fusion_visible(out, lane_shapes, lane_colors, counts, reds, green, bg):
        return None

    try:
        solved = main(inp)
    except Exception:
        return None
    if not np.array_equal(solved, out):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(_GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {_GEN_RETRIES} attempts")
