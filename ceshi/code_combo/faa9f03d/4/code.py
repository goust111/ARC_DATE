# Combo variant of faa9f03d (index 4)
# combo_spec: D1/3+D14/0
# Name: 间隙用特殊色填充 + 输入交叉处显示背景色
#
# Design analysis
# ---------------
# spec_like:
#   fused_rule:
#     - Keep the original 12x12 rectilinear one-pixel line network with red/yellow
#       singleton markers and binary orthogonal crossings.
#     - Apply D14/0 first at the input level: every true crossing cell is hidden as
#       background instead of showing either crossing color.
#     - Apply D1/3 at the output level: every ordinary repaired gap cell is shown in
#       one fixed special color rather than in the inferred line color. Yellow-driven
#       reroutes are not recolored to the special color; they stay in the line color.
#   concrete_pixel_behavior:
#     - If a red marker removes a corner cell of an orange polyline, the output no
#       longer restores that cell as orange. It restores the logical connection but
#       paints the repaired cell gray.
#     - If two lines logically cross at a hidden input background cell, both colors
#       still claim that cell during reconstruction; the output shows the winning line
#       color there, not gray.
#     - Therefore background holes split into two visible classes after solving:
#       single-claim holes become gray repairs, while two-claim holes become colored
#       crossings.
#   traced_example:
#     original:
#       - Row fragment blue .. gap .. blue becomes fully blue again.
#       - A crossing center may already show the losing color in the input.
#     fused_variant:
#       - The same row fragment becomes blue-gray-blue in the output.
#       - A true crossing center is blank in the input, then becomes whichever line
#         wins the fragmentation contest in the output.
#     difference:
#       - The solver must recover both "which blank cells are crossings?" and "which
#         gray repairs belong to which hidden line color?" from the same sparse line
#         geometry.
#   complexity_calibration:
#     - The special repair color is fixed to gray (ARC color 5) and line colors are
#       sampled from a palette that excludes gray. This makes the new transform
#       human-visible without relying on hidden numeric color ordering.
#     - Every generated sample includes at least two red gaps, so the gray-repair
#       behavior repeats within a sample instead of appearing as a one-off accident.
#     - Hidden crossings still require a unique fragmentation winner; the generator
#       forces extra dominant-color red gaps so those wins are structurally justified
#       rather than falling back to color-id ties.
#   implicit_assumptions_review:
#     - Marker assignment by straight line-of-sight still holds because hiding a
#       crossing only removes one cell, not an entire visible fragment.
#     - Generic gap filling can still detect hidden crossings because they remain
#       encoded as two perpendicular one-cell-facing gaps.
#     - The old endpoint assertions on the rendered output are no longer valid:
#       visible line colors now intentionally stop at gray repair cells. Structural
#       validation must therefore run on the reconstructed logical per-color masks,
#       not on the final rendered image.
#     - The old `g[pt] == bg` overlap fallback based on marker heuristics is unsafe
#       for hidden crossings; the solver must directly use visible-fragment counts at
#       those cells.
#   placement_legality:
#     - Crossings stay blocked for both red and yellow marker placement.
#     - Yellow rays are forbidden from passing through any hidden crossing so one
#       background cell never mixes "yellow wins" and "hidden crossing" semantics.
#     - Red markers are selected on visible turns/endpoints with local spacing, and
#       gray is reserved so no input cell can already use the repair color.
#   observability_and_tidiness:
#     - Humans can see the fused rule from a few examples because each sample repeats
#       both motifs: blank crossing centers in the input and gray repaired segments in
#       the output.
#     - Restricting line colors away from gray keeps the rendered outputs readable:
#       gray always means "ordinary repair trace", never a native line.

from __future__ import annotations

import importlib.util
import random
import sys
from pathlib import Path

import numpy as np

from arc_utils import components_mask, get_bg


BASE_PATH = Path("/home/lixinyu/snap/new/ceshi/code/faa9f03d/D14/0/code.py")
SPEC = importlib.util.spec_from_file_location("faa9f03d_d14_0_combo_base", BASE_PATH)
_base = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = _base
assert SPEC.loader is not None
SPEC.loader.exec_module(_base)


DIRS4 = _base.DIRS4
H = _base.H
W = _base.W
BG = _base.BG

FILL_COLOR = 5
LINE_PALETTE = [1, 3, 6, 7, 8, 9]
MARKER_MODE_WEIGHTS = [5, 4]  # red-only / both
RED_COUNT_RANGE = (2, 4)
HIDDEN_CROSSING_DOMINANT_REDS = 2
SPECIAL_FILL_RANGE = (2, 12)
GEN_RETRIES = _base.GEN_RETRIES

_base.LINE_PALETTE = LINE_PALETTE

_pts = _base._pts
_axis_support = _base._axis_support
_perp_support = _base._perp_support
_component_count = _base._component_count
_neighbors = _base._neighbors
_orientation = _base._orientation
_gen_component_count = _base._gen_component_count
_crossing_cells = _base._crossing_cells
_yellow_candidates = _base._yellow_candidates
_choose_red_markers = _base._choose_red_markers
_choose_red_markers_for_color = _base._choose_red_markers_for_color
_build_logical = _base._build_logical
_marker_isolation_ok = _base._marker_isolation_ok


def _solve(grid: np.ndarray, *, return_debug: bool = False):
    g = np.asarray(grid, dtype=int)
    bg = int(get_bg(g))
    H0, W0 = g.shape
    assert (H0, W0) == (H, W), "expected 12x12 grid"

    colors = set(map(int, g.ravel())) - {bg}
    assert colors, "expected foreground line and marker colors"
    assert FILL_COLOR not in colors, "input must not already contain the special repair color"

    marker_colors: set[int] = set()
    for color in colors:
        comps = components_mask(g == color, connectivity=4)
        assert comps, "foreground color must have at least one component"
        if max(len(comp) for comp in comps) == 1:
            marker_colors.add(int(color))

    line_colors = colors - marker_colors
    assert line_colors, "expected at least one non-marker line color"
    assert marker_colors, "expected singleton marker colors"
    assert FILL_COLOR not in line_colors, "special repair color cannot also be a line color"

    original = {color: _pts(g == color) for color in line_colors}
    assignment: dict[tuple[int, int], int] = {}
    assign_dir: dict[tuple[int, int], tuple[int, int]] = {}

    marker_mask = np.isin(g, list(marker_colors))
    for r0, c0 in _pts(marker_mask):
        candidates: list[tuple[int, int, int, int, int]] = []
        for dr, dc in DIRS4:
            r, c = r0 + dr, c0 + dc
            dist = 1
            while 0 <= r < H and 0 <= c < W:
                value = int(g[r, c])
                if value in line_colors:
                    axis = 0 if dr == 0 else 1
                    supported = _axis_support(original[value], r, c, axis)
                    candidates.append((dist, 0 if supported else 1, value, dr, dc))
                    break
                if value == bg or value in marker_colors:
                    r += dr
                    c += dc
                    dist += 1
                    continue
                break

        assert candidates, f"marker at {(r0, c0)} cannot see a line color"
        candidates.sort()
        _dist, _unsupported, color, dr, dc = candidates[0]
        assignment[(r0, c0)] = int(color)
        assign_dir[(r0, c0)] = (int(dr), int(dc))

    masks = {color: set(pts) for color, pts in original.items()}
    marker_pts = {color: set() for color in line_colors}
    ordinary_fill = {color: set() for color in line_colors}
    for pt, color in assignment.items():
        masks[color].add(pt)
        marker_pts[color].add(pt)
        if int(g[pt]) == 2:
            ordinary_fill[color].add(pt)

    comp_count = {color: _component_count(g.shape, original[color]) for color in line_colors}

    def has_axis_support(color: int, r: int, c: int, axis: int) -> bool:
        return (r, c) in marker_pts[color] or _axis_support(masks[color], r, c, axis)

    def should_fill(color: int, a: tuple[int, int], b: tuple[int, int], axis: int) -> bool:
        r1, c1 = a
        r2, c2 = b
        gap = c2 - c1 - 1 if axis == 0 else r2 - r1 - 1
        if gap <= 0:
            return False

        a_axis = has_axis_support(color, r1, c1, axis)
        b_axis = has_axis_support(color, r2, c2, axis)
        if gap == 1:
            if a_axis or b_axis:
                return True
            a_perp = _perp_support(masks[color], r1, c1, axis)
            b_perp = _perp_support(masks[color], r2, c2, axis)
            return not (a_perp and b_perp)

        if (r1, c1) in marker_pts[color] and (r2, c2) in marker_pts[color]:
            return False
        return a_axis and b_axis

    for color in sorted(line_colors):
        additions: set[tuple[int, int]] = set()
        for r in range(H):
            cols = sorted(c for rr, c in masks[color] if rr == r)
            for c1, c2 in zip(cols, cols[1:]):
                if should_fill(color, (r, c1), (r, c2), axis=0):
                    additions.update((r, c) for c in range(c1 + 1, c2))

        for c in range(W):
            rows = sorted(r for r, cc in masks[color] if cc == c)
            for r1, r2 in zip(rows, rows[1:]):
                if should_fill(color, (r1, c), (r2, c), axis=1):
                    additions.update((r, c) for r in range(r1 + 1, r2))

        masks[color].update(additions)
        ordinary_fill[color].update(additions)

    control_ray = {color: set() for color in line_colors}
    if 4 in marker_colors:
        for control in _pts(g == 4):
            color = assignment[control]
            dr, dc = assign_dir[control]

            r, c = control[0] + dr, control[1] + dc
            pivot: tuple[int, int] | None = None
            while 0 <= r < H and 0 <= c < W:
                if (r, c) in masks[color]:
                    pivot = (r, c)
                    break
                r += dr
                c += dc
            assert pivot is not None, f"yellow control at {control} has no assigned endpoint"

            old_start = (pivot[0] + dr, pivot[1] + dc)
            if old_start in masks[color]:
                stack = [old_start]
                old_arm: set[tuple[int, int]] = set()
                while stack:
                    q = stack.pop()
                    if q == pivot or q in old_arm or q not in masks[color]:
                        continue
                    old_arm.add(q)
                    qr, qc = q
                    for ndr, ndc in DIRS4:
                        stack.append((qr + ndr, qc + ndc))

                assert old_arm, f"yellow control at {control} found no old arm to erase"
                for q in old_arm:
                    masks[color].discard(q)
                    marker_pts[color].discard(q)
                    ordinary_fill[color].discard(q)
                    control_ray[color].discard(q)

            pr, pc = -dr, -dc
            r, c = pivot
            while 0 <= r < H and 0 <= c < W:
                masks[color].add((r, c))
                control_ray[color].add((r, c))
                r += pr
                c += pc

    claims: dict[tuple[int, int], list[int]] = {}
    for color, pts in masks.items():
        for pt in pts:
            claims.setdefault(pt, []).append(int(color))

    def marker_brackets(color: int, pt: tuple[int, int]) -> bool:
        r, c = pt
        same_row = [mc for mr, mc in marker_pts[color] if mr == r]
        if any(mc < c for mc in same_row) and any(mc > c for mc in same_row):
            return True
        same_col = [mr for mr, mc in marker_pts[color] if mc == c]
        return any(mr < r for mr in same_col) and any(mr > r for mr in same_col)

    out = np.full_like(g, bg)
    for pt, colors_here in claims.items():
        forced = {color for color in colors_here if pt in control_ray[color]}
        if forced:
            assert len(forced) == 1, f"ambiguous yellow-control rays at {pt}: {forced}"
            out[pt] = next(iter(forced))
            continue

        if len(colors_here) == 1:
            color = colors_here[0]
            out[pt] = FILL_COLOR if pt in ordinary_fill[color] else color
            continue

        if int(g[pt]) == bg:
            out[pt] = min(colors_here, key=lambda color: (-comp_count[color], color))
            continue

        original_crossing = [color for color in colors_here if int(g[pt]) == color]
        if original_crossing:
            out[pt] = min(colors_here, key=lambda color: (-comp_count[color], color))
        else:
            bracketed = [color for color in colors_here if marker_brackets(color, pt)]
            if bracketed:
                out[pt] = min(bracketed, key=lambda color: (-comp_count[color], color))
            else:
                out[pt] = min(colors_here)

    # Validate the reconstructed logical color masks, not the rendered output:
    # visible line colors now intentionally stop at gray repair traces.
    for color in line_colors:
        mask = np.zeros((H, W), dtype=bool)
        for r, c in masks[color]:
            mask[r, c] = True
        for comp in components_mask(mask, connectivity=4):
            pts_set = set(map(tuple, comp))
            for r, c in pts_set:
                nbrs = [(nr, nc) for nr, nc in [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)] if (nr, nc) in pts_set]
                assert len(nbrs) <= 2, f"logical color {color} has a junction at {(r, c)}"
                if len(nbrs) != 1:
                    continue
                assert r in (0, H - 1) or c in (0, W - 1), f"logical dangling endpoint {(r, c)} for color {color}"
                if (r in (0, H - 1)) and (c in (0, W - 1)):
                    continue
                nr, nc = nbrs[0]
                dr, dc = nr - r, nc - c
                perp = (
                    (r == 0 and dr == 1)
                    or (r == H - 1 and dr == -1)
                    or (c == 0 and dc == 1)
                    or (c == W - 1 and dc == -1)
                )
                assert perp, f"logical boundary endpoint {(r, c)} color {color} not perpendicular"

    crossing_colors = set()
    for pt, colors_here in claims.items():
        if len(colors_here) <= 1:
            continue
        assert len(colors_here) == 2, f"expected binary crossing at {pt}, got {colors_here}"
        orientations = [_orientation(masks[color], pt) for color in colors_here]
        assert None not in orientations and sorted(orientations) == ["h", "v"], f"invalid crossing geometry at {pt}"
        crossing_colors.update(colors_here)
    for color in line_colors:
        assert color in crossing_colors, f"color {color} has no logical crossing with any other color"

    if return_debug:
        return out, {
            "claims": claims,
            "line_colors": set(line_colors),
            "ordinary_fill": ordinary_fill,
            "control_ray": control_ray,
            "comp_count": comp_count,
            "bg": bg,
        }
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reconstruct a one-cell-thick rectilinear line network by recoloring singleton markers, restoring background-hidden crossings, and rendering ordinary repaired gaps in a dedicated special color.

    Input structure:
    - A 12x12 grid with a dominant background.
    - Non-marker colors form one-pixel-wide horizontal/vertical stroke fragments.
    - Marker colors are exactly the foreground colors whose 4-connected components are all singleton pixels.
    - Red singleton markers stand in for removed line cells, yellow singleton markers mark endpoint reroutes, and every true crossing cell is hidden as background rather than shown in either line color.

    Rule:
    - Assign each marker to the nearest visible line color in row/column line-of-sight, preferring an endpoint supported along that same axis.
    - For each line color, bridge the axis-aligned gap between consecutive facing endpoints in a row or column; these bridges recover both ordinary red-marker gaps and blank hidden-crossing centers.
    - A yellow control still erases the old arm on the far side of its pivot and paints the reassigned ray from that pivot through the marker to the boundary in the line's own color.
    - When a repaired cell belongs to only one reconstructed line, display it in the fixed special repair color; when two reconstructed lines overlap at a hidden crossing, yellow rays win first, otherwise the more fragmented visible pre-repair line color claims the crossing cell.

    Key insight: Background holes now split into two semantic classes. A hole claimed by one line is just a repaired gap and must be shown as gray, while a hole claimed by two perpendicular lines is a hidden crossing and must be shown in the winning line color. The same local reconstruction step reveals both.
    """
    return _solve(grid, return_debug=False)


def _combo_metrics(grid: np.ndarray) -> dict[str, int]:
    out, debug = _solve(grid, return_debug=True)
    hidden_crossings = sum(1 for pt, colors in debug["claims"].items() if len(colors) > 1 and int(grid[pt]) == debug["bg"])
    return {
        "special_fill_cells": int(np.count_nonzero(out == FILL_COLOR)),
        "hidden_crossings": int(hidden_crossings),
    }


def _derive_input(
    logical: dict[int, set[tuple[int, int]]],
    paths: dict[int, list[set[tuple[int, int]]]],
    dominant: int,
) -> tuple[np.ndarray, np.ndarray] | None:
    crossings = set(_crossing_cells(logical))
    mode = random.choices(["red", "both"], weights=MARKER_MODE_WEIGHTS)[0]
    yellow_count = random.randint(*_base.YELLOW_COUNT_RANGE) if mode == "both" else 0
    red_count = max(random.randint(*RED_COUNT_RANGE), HIDDEN_CROSSING_DOMINANT_REDS)

    yellow_infos = []
    blocked = set(crossings)
    if yellow_count:
        for cand in _yellow_candidates(logical, paths, crossings):
            color, pivot, out_dir, target_ray, old_arm = cand
            marker = (pivot[0] + out_dir[0], pivot[1] + out_dir[1])
            if target_ray & crossings:
                continue
            if marker in blocked or any(nb in blocked for nb in _neighbors(marker)):
                continue
            if any((old & old_arm) or (ray & old_arm) for _, _, _, ray, old in yellow_infos):
                continue
            yellow_infos.append(cand)
            blocked.update(target_ray | old_arm | {marker})
            if len(yellow_infos) == yellow_count:
                break
        if len(yellow_infos) != yellow_count:
            return None

    forced_reds = _choose_red_markers_for_color(
        logical, paths, blocked, dominant, HIDDEN_CROSSING_DOMINANT_REDS
    )
    if len(forced_reds) != HIDDEN_CROSSING_DOMINANT_REDS:
        return None
    blocked_with_forced = blocked | {pt for _, pt in forced_reds}
    extra_needed = red_count - HIDDEN_CROSSING_DOMINANT_REDS
    extra_reds = _choose_red_markers(
        logical, paths, blocked_with_forced, extra_needed
    ) if extra_needed else []
    if extra_needed and not extra_reds:
        return None
    red_markers = forced_reds + extra_reds

    inp = np.full((H, W), BG, dtype=int)
    comp_count_masks: dict[int, set[tuple[int, int]]] = {c: set() for c in logical}

    removed_for_yellow: dict[int, set[tuple[int, int]]] = {c: set() for c in logical}
    old_arms: dict[int, set[tuple[int, int]]] = {c: set() for c in logical}
    yellow_rays: dict[int, set[tuple[int, int]]] = {c: set() for c in logical}
    yellow_markers: list[tuple[int, tuple[int, int]]] = []
    for color, pivot, out_dir, target_ray, old_arm in yellow_infos:
        marker = (pivot[0] + out_dir[0], pivot[1] + out_dir[1])
        removed_for_yellow[color].update(target_ray - {pivot})
        old_arms[color].update(old_arm)
        yellow_rays[color].update(target_ray)
        yellow_markers.append((color, marker))

    red_by_pt = {pt: color for color, pt in red_markers}
    yellow_by_pt = {pt: color for color, pt in yellow_markers}

    for color, pts in logical.items():
        for pt in pts:
            if pt in removed_for_yellow[color]:
                continue
            if pt in crossings:
                continue
            if pt in red_by_pt or pt in yellow_by_pt:
                continue
            inp[pt] = color
            comp_count_masks[color].add(pt)
    for color, pts in old_arms.items():
        for pt in pts:
            if inp[pt] != BG:
                return None
            inp[pt] = color
            comp_count_masks[color].add(pt)

    for pt in red_by_pt:
        inp[pt] = 2
    for pt in yellow_by_pt:
        inp[pt] = 4

    if not _marker_isolation_ok(inp):
        return None
    for pt, color in yellow_by_pt.items():
        if not any(inp[nb] == color for nb in _neighbors(pt)):
            return None

    comp_count = {color: _gen_component_count(pts) for color, pts in comp_count_masks.items()}
    for pt in crossings:
        colors_here = [color for color in logical if pt in logical[color]]
        forced = [color for color in colors_here if pt in yellow_rays.get(color, set())]
        if forced:
            return None
        best = max(comp_count[color] for color in colors_here)
        if sum(comp_count[color] == best for color in colors_here) != 1:
            return None

    try:
        target, debug = _solve(inp, return_debug=True)
    except Exception:
        return None

    metrics = _combo_metrics(inp)
    if metrics["hidden_crossings"] <= 0:
        return None
    if not SPECIAL_FILL_RANGE[0] <= metrics["special_fill_cells"] <= SPECIAL_FILL_RANGE[1]:
        return None
    if int(np.count_nonzero(inp == FILL_COLOR)) != 0:
        return None
    if int(np.count_nonzero(target == FILL_COLOR)) != metrics["special_fill_cells"]:
        return None
    if not _base.FILL_RANGE[0] <= int(np.count_nonzero(target)) <= _base.FILL_RANGE[1]:
        return None

    visible_colors = set(map(int, np.unique(target))) - {BG, FILL_COLOR}
    if visible_colors != set(logical):
        return None
    if set(debug["line_colors"]) != set(logical):
        return None
    if not all(inp[pt] == BG for pt, colors in debug["claims"].items() if len(colors) > 1):
        return None

    return inp, target


def _gen_attempt() -> np.ndarray | None:
    built = _build_logical()
    if built is None:
        return None
    logical, paths, dominant = built
    derived = _derive_input(logical, paths, dominant)
    return derived[0] if derived is not None else None


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("failed to generate a valid faa9f03d combo input")
