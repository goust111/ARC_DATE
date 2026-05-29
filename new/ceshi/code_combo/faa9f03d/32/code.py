# Combo variant of faa9f03d (index 32)
# combo_spec: D3/1+D6/2+D10/1
# Name: 双像素方向标记 + 标记夹持填充 + 距离优先逐步修复
#
# Design analysis for the fused variant
# ====================================
# a) Concrete behavior and trace
# - Base task recap:
#   - line colors form thin rectilinear polylines;
#   - red marks missing line cells;
#   - yellow marks reroute one arm through the mark to the boundary.
# - Fused change:
#   - every marker is now a 2-cell domino (horizontal or vertical);
#   - only red marker pairs can trigger gap filling, and only when the two
#     dominoes clamp the missing corridor from both sides on the same axis;
#   - red repairs happen one pair at a time from shorter gap to longer gap,
#     recomputing the visible line state after each repair;
#   - yellow dominoes keep the original "erase old arm / extend new ray"
#     semantics, but they also use the domino axis for assignment.
# - Concrete trace on one row:
#   - input may show `... line | RR .. RR | ...` where each `RR` is a horizontal
#     red domino and the blank cells between them are background.
#   - old solver family would immediately recolor every marker cell and fill
#     every eligible row/column gap in bulk.
#   - new solver treats each red domino as pending until a matching partner is
#     selected. It picks the currently shortest clamped corridor, recolors that
#     pair plus the interior cells, updates the visible line, then looks for the
#     next pair. A farther pair can therefore become valid only after a nearer
#     repair has restored supporting cells nearby.
#
# b) Complexity calibration
# - The only new symbolic freedom is domino orientation; it has just two values
#   and is repeated across both red and yellow markers, so the visual cue is
#   learnable from a few examples.
# - Red gaps are limited to 1-4 blank cells between the two dominoes. This keeps
#   the bracket relation readable and prevents giant ambiguous corridors.
# - Samples usually include 2-3 red repairs and 1 yellow reroute. That is enough
#   to make the sequential behavior visible without overcrowding the 12x12 grid.
#
# c) Implicit assumptions audited
# - Old code assumed marker colors were detectable by singleton components. That
#   breaks immediately: here marker colors are explicitly `2` and `4`, and each
#   component must be a size-2 domino.
# - Old fill logic assumed marker pixels were already part of the repaired line.
#   That would erase the D10 effect. In this variant unresolved red dominoes are
#   *not* line cells yet; they become line cells only when their pair is chosen.
# - Old fill logic scanned all consecutive visible cells in one bulk pass. That
#   also breaks D10. Here the solver repeatedly rebuilds red-pair candidates from
#   the current state and consumes exactly one nearest pair each iteration.
# - Old yellow handling anchored on a single control pixel. That fails with a
#   domino because only the line-facing domino cell is the correct anchor.
#
# d) Placement legality guaranteed by the generator
# - Every red domino pair is carved from one straight run of one logical output
#   component, with real same-color cells immediately outside the pair in the
#   target and no crossings inside the removed corridor.
# - Every yellow domino occupies the first two cells of an intended output ray;
#   the opposite side of the pivot has empty space for the temporary old arm
#   shown in the input.
# - Marker dominoes do not overlap each other, do not overlap crossings, and do
#   not merge into larger marker components because the generator reserves a halo.
#
# e) Visual observability
# - The fused rule is visible in the grids themselves:
#   - markers are bars, not dots;
#   - the bar orientation matches the direction of the restored segment;
#   - repaired corridors are bracketed by two red bars;
#   - a yellow bar points along the new arm direction.
# - The generator rejects cases where the ordered solver matches a bulk
#   no-order baseline, so the distance-priority rule is not merely present in
#   code; it changes the produced output on accepted samples.
#
# f) Visual cleanliness
# - Red dominoes are sampled only from straight runs, never from crossings.
# - Yellow dominoes occupy a clean two-cell ray prefix, which keeps the old-arm
#   versus new-arm contrast legible.
# - Crossing recoloring is preserved from the seed task so the output still looks
#   like an ARC line-network repair task rather than a free-form drawing.
#
# g) Color inference
# - No hidden numeric convention is introduced beyond the explicit marker colors
#   `2` and `4`. The recovered line color is always inferred from visible
#   same-row/same-column line fragments.
#
# h) New solver assumptions and generator support
# - Solver assumption: unresolved red dominoes are pairable only when both
#   dominoes have a unique nearest visible line color on their own axis.
#   Generator support: red gaps are cut from clean straight runs so this
#   assignment is unambiguous.
# - Solver assumption: if a red pair is chosen, the cells between the two
#   dominoes are background at that moment. Generator support: the carved red
#   corridors start fully blank and never overlap yellow old arms or crossings.
# - Solver assumption: yellow dominoes are processed after the red sequential
#   repairs and then have a unique pivot on the line-facing side. Generator
#   support: yellow candidates are sampled exactly from ray prefixes of the
#   target polyline, so that pivot is well-defined.

from __future__ import annotations

import random

import numpy as np

from arc_utils import components_mask, get_bg


DIRS4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
MARKER_IDS = {2, 4}
H = W = 12
BG = 0
LINE_PALETTE = [1, 3, 5, 6, 7, 8, 9]
POLYLINE_PIECE_WEIGHTS = [5, 4, 2, 1]
MIN_SPAN_LEN = 4
FULL_SPAN_PROB = 0.65
ARM_MIN_OFFSET = 2
CROSSING_COUNT_WEIGHTS = [2, 3, 3, 2]
SERPENTINE_MINOR_LO = 6
SERPENTINE_MINOR_HI = 5
CROSS_MINOR_MARGIN = 2
MAX_COLORS = 4
MAX_CROSSINGS = 12
RED_CHAIN_COUNT = 1
YELLOW_COUNT_RANGE = (0, 0)
FILL_RANGE = (33, 58)
COMPONENTS_PER_COLOR = (1, 4)
TOTAL_COMPONENTS = (4, 6)
SPACED_VALUES_RETRIES = 60
EXTRA_POLYLINE_RETRIES = 80
GEN_RETRIES = 500


def _pts(mask: np.ndarray) -> set[tuple[int, int]]:
    return {tuple(map(int, rc)) for rc in np.argwhere(mask)}


def _neighbors(pt: tuple[int, int]) -> list[tuple[int, int]]:
    r, c = pt
    return [(r + dr, c + dc) for dr, dc in DIRS4 if 0 <= r + dr < H and 0 <= c + dc < W]


def _component_count(shape: tuple[int, int], pts: set[tuple[int, int]]) -> int:
    mask = np.zeros(shape, dtype=bool)
    for r, c in pts:
        mask[r, c] = True
    return len(components_mask(mask, connectivity=4))


def _dirs_in(pts: set[tuple[int, int]], pt: tuple[int, int]) -> list[tuple[int, int]]:
    r, c = pt
    return [(dr, dc) for dr, dc in DIRS4 if (r + dr, c + dc) in pts]


def _orientation(pts: set[tuple[int, int]], pt: tuple[int, int]) -> str | None:
    dirs = _dirs_in(pts, pt)
    if len(dirs) != 2 or (dirs[0][0] + dirs[1][0], dirs[0][1] + dirs[1][1]) != (0, 0):
        return None
    return "v" if dirs[0][0] else "h"


def _is_turn_or_endpoint(pts: set[tuple[int, int]], pt: tuple[int, int]) -> bool:
    dirs = _dirs_in(pts, pt)
    if len(dirs) == 1:
        return True
    return len(dirs) == 2 and (dirs[0][0] + dirs[1][0], dirs[0][1] + dirs[1][1]) != (0, 0)


def _axis_support(mask: set[tuple[int, int]], r: int, c: int, axis: int) -> bool:
    if axis == 0:
        return (r, c - 1) in mask or (r, c + 1) in mask
    return (r - 1, c) in mask or (r + 1, c) in mask


def _domino_axis(cells: set[tuple[int, int]]) -> int | None:
    if len(cells) != 2:
        return None
    (r1, c1), (r2, c2) = sorted(cells)
    if abs(r1 - r2) + abs(c1 - c2) != 1:
        return None
    return 0 if r1 == r2 else 1


def _split_marker_component(cells: set[tuple[int, int]]) -> list[set[tuple[int, int]]] | None:
    axis = _domino_axis(cells)
    if axis is not None:
        return [set(cells)]
    if len(cells) != 4:
        return None
    ordered = sorted(cells)
    if len({r for r, _ in ordered}) == 1:
        row = ordered[0][0]
        cols = sorted(c for _r, c in ordered)
        if cols == list(range(cols[0], cols[0] + 4)):
            return [
                {(row, cols[0]), (row, cols[1])},
                {(row, cols[2]), (row, cols[3])},
            ]
    if len({c for _, c in ordered}) == 1:
        col = ordered[0][1]
        rows = sorted(r for r, _c in ordered)
        if rows == list(range(rows[0], rows[0] + 4)):
            return [
                {(rows[0], col), (rows[1], col)},
                {(rows[2], col), (rows[3], col)},
            ]
    return None


def _axis_dirs(axis: int) -> tuple[tuple[int, int], tuple[int, int]]:
    return ((0, -1), (0, 1)) if axis == 0 else ((-1, 0), (1, 0))


def _ordered_ray_from(start: tuple[int, int], direction: tuple[int, int]) -> list[tuple[int, int]]:
    out = []
    r, c = start
    dr, dc = direction
    while 0 <= r < H and 0 <= c < W:
        out.append((r, c))
        r += dr
        c += dc
    return out


def _ray_from(start: tuple[int, int], direction: tuple[int, int]) -> set[tuple[int, int]]:
    return set(_ordered_ray_from(start, direction))


def _marker_brackets(marker_pts: dict[int, set[tuple[int, int]]], color: int, pt: tuple[int, int]) -> bool:
    r, c = pt
    same_row = [mc for mr, mc in marker_pts[color] if mr == r]
    if any(mc < c for mc in same_row) and any(mc > c for mc in same_row):
        return True
    same_col = [mr for mr, mc in marker_pts[color] if mc == c]
    return any(mr < r for mr in same_col) and any(mr > r for mr in same_col)


def _parse_markers(g: np.ndarray) -> tuple[list[dict[str, object]], set[int]]:
    colors = set(map(int, g.ravel())) - {int(get_bg(g))}
    marker_colors = colors & MARKER_IDS
    marker_components: list[dict[str, object]] = []
    for marker_color in sorted(marker_colors):
        comps = components_mask(g == marker_color, connectivity=4)
        for comp in comps:
            cells = {tuple(map(int, rc)) for rc in comp}
            parts = _split_marker_component(cells)
            assert parts is not None, f"marker color {marker_color} must form domino components"
            for part in parts:
                axis = _domino_axis(part)
                assert axis is not None
                marker_components.append(
                    {
                        "marker_color": int(marker_color),
                        "cells": tuple(sorted(part)),
                        "cells_set": set(part),
                        "axis": int(axis),
                    }
                )
    return marker_components, marker_colors


def _assign_marker(info: dict[str, object], g: np.ndarray, masks: dict[int, set[tuple[int, int]]], line_colors: set[int]) -> tuple[int, tuple[int, int], tuple[int, int], int]:
    axis = int(info["axis"])
    bg = int(get_bg(g))
    candidates: list[tuple[int, int, int, tuple[int, int], tuple[int, int], int]] = []
    for anchor in info["cells"]:
        for dr, dc in _axis_dirs(axis):
            r, c = anchor[0] + dr, anchor[1] + dc
            dist = 1
            while 0 <= r < g.shape[0] and 0 <= c < g.shape[1]:
                hit_color = None
                for color in line_colors:
                    if (r, c) in masks[color]:
                        hit_color = int(color)
                        break
                if hit_color is not None:
                    supported = _axis_support(masks[hit_color], r, c, axis)
                    candidates.append((dist, 0 if supported else 1, int(hit_color), anchor, (dr, dc), axis))
                    break
                value = int(g[r, c])
                if value == bg or value in MARKER_IDS:
                    r += dr
                    c += dc
                    dist += 1
                    continue
                break
    assert candidates, f"marker {info['cells']} cannot see a line color on its axis"
    candidates.sort(key=lambda item: item[:2])
    best_key = candidates[0][:2]
    best = [cand for cand in candidates if cand[:2] == best_key]
    visible_colors = {cand[2] for cand in best}
    assert len(visible_colors) == 1, f"ambiguous marker assignment for {info['cells']}"
    _dist, _unsupported, color, anchor, direction, _axis = best[0]
    return int(color), anchor, direction, axis


def _eligible_red_pair(
    left: dict[str, object],
    right: dict[str, object],
    color: int,
    masks: dict[int, set[tuple[int, int]]],
) -> tuple[int, list[tuple[int, int]]] | None:
    axis = int(left["axis"])
    if int(right["axis"]) != axis:
        return None
    cells1 = sorted(left["cells"])
    cells2 = sorted(right["cells"])
    if axis == 0:
        if cells1[0][0] != cells2[0][0]:
            return None
        row = cells1[0][0]
        left_cols = sorted(c for _r, c in cells1)
        right_cols = sorted(c for _r, c in cells2)
        if left_cols[-1] >= right_cols[0]:
            return None
        interior = [(row, c) for c in range(left_cols[-1] + 1, right_cols[0])]
        if not interior:
            return None
        fill_cells = [pt for pt in interior if pt not in masks[color]]
        if not fill_cells:
            return None
    else:
        if cells1[0][1] != cells2[0][1]:
            return None
        col = cells1[0][1]
        left_rows = sorted(r for r, _c in cells1)
        right_rows = sorted(r for r, _c in cells2)
        if left_rows[-1] >= right_rows[0]:
            return None
        interior = [(r, col) for r in range(left_rows[-1] + 1, right_rows[0])]
        if not interior:
            return None
        fill_cells = [pt for pt in interior if pt not in masks[color]]
        if not fill_cells:
            return None
    return len(fill_cells), fill_cells


def _has_pending_between(left: dict[str, object], right: dict[str, object], assigned: list[dict[str, object]]) -> bool:
    axis = int(left["axis"])
    if axis == 0:
        row = left["cells"][0][0]
        l_hi = max(c for _r, c in left["cells"])
        r_lo = min(c for _r, c in right["cells"])
        for other in assigned:
            if other is left or other is right or int(other["axis"]) != axis or other["cells"][0][0] != row:
                continue
            o_lo = min(c for _r, c in other["cells"])
            o_hi = max(c for _r, c in other["cells"])
            if l_hi < o_lo and o_hi < r_lo:
                return True
        return False
    col = left["cells"][0][1]
    l_hi = max(r for r, _c in left["cells"])
    r_lo = min(r for r, _c in right["cells"])
    for other in assigned:
        if other is left or other is right or int(other["axis"]) != axis or other["cells"][0][1] != col:
            continue
        o_lo = min(r for r, _c in other["cells"])
        o_hi = max(r for r, _c in other["cells"])
        if l_hi < o_lo and o_hi < r_lo:
            return True
    return False


def _collect_red_candidates(
    pending_red: list[dict[str, object]],
    g: np.ndarray,
    masks: dict[int, set[tuple[int, int]]],
    line_colors: set[int],
) -> list[tuple[int, int, int, tuple[int, int], dict[str, object], dict[str, object], list[tuple[int, int]], int]]:
    assigned: list[dict[str, object]] = []
    for info in pending_red:
        color, anchor, direction, axis = _assign_marker(info, g, masks, line_colors)
        enriched = dict(info)
        enriched["line_color"] = int(color)
        enriched["anchor"] = anchor
        enriched["dir_to_line"] = direction
        enriched["axis"] = int(axis)
        assigned.append(enriched)

    candidates = []
    for i in range(len(assigned)):
        for j in range(i + 1, len(assigned)):
            a = assigned[i]
            b = assigned[j]
            if int(a["line_color"]) != int(b["line_color"]):
                continue
            axis = int(a["axis"])
            pair = (a, b) if tuple(a["cells"])[0] <= tuple(b["cells"])[0] else (b, a)
            if _has_pending_between(pair[0], pair[1], assigned):
                continue
            eligible = _eligible_red_pair(pair[0], pair[1], int(a["line_color"]), masks)
            if eligible is None:
                continue
            gap_len, interior = eligible
            top_left = min(pair[0]["cells"][0], pair[1]["cells"][0])
            center_r = sum(pt[0] for pt in pair[0]["cells"] + pair[1]["cells"]) / 4.0
            center_c = sum(pt[1] for pt in pair[0]["cells"] + pair[1]["cells"]) / 4.0
            center_bias = abs(center_r - (H - 1) / 2) + abs(center_c - (W - 1) / 2)
            candidates.append((gap_len, center_bias, top_left[0], top_left[1], top_left, pair[0], pair[1], interior, axis))
    candidates.sort(key=lambda x: (x[0], x[1], x[2], x[3]))
    return candidates


def _resolve_claims(
    g: np.ndarray,
    bg: int,
    line_colors: set[int],
    masks: dict[int, set[tuple[int, int]]],
    marker_pts: dict[int, set[tuple[int, int]]],
    comp_count: dict[int, int],
    control_ray: dict[int, set[tuple[int, int]]],
) -> np.ndarray:
    claims: dict[tuple[int, int], list[int]] = {}
    for color, pts in masks.items():
        for pt in pts:
            claims.setdefault(pt, []).append(int(color))

    out = np.full_like(g, bg)
    for pt, colors_here in claims.items():
        forced = {color for color in colors_here if pt in control_ray[color]}
        if forced:
            assert len(forced) == 1, f"ambiguous yellow-control rays at {pt}: {forced}"
            out[pt] = next(iter(forced))
            continue

        if len(colors_here) == 1:
            out[pt] = colors_here[0]
            continue

        original_crossing = [color for color in colors_here if int(g[pt]) == color]
        if original_crossing:
            out[pt] = min(colors_here, key=lambda color: (-comp_count[color], color))
            continue

        bracketed = [color for color in colors_here if _marker_brackets(marker_pts, color, pt)]
        if bracketed:
            out[pt] = min(bracketed, key=lambda color: (-comp_count[color], color))
        else:
            out[pt] = min(colors_here)
    return out


def _validate_output(out: np.ndarray, bg: int, line_colors: set[int]) -> None:
    for color in line_colors:
        for comp in components_mask(out == color, connectivity=4):
            pts_set = set(map(tuple, comp))
            for r, c in pts_set:
                nbrs = [(nr, nc) for nr, nc in [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)] if (nr, nc) in pts_set]
                if len(nbrs) != 1:
                    continue
                on_boundary = r == 0 or r == H - 1 or c == 0 or c == W - 1
                if on_boundary:
                    nr, nc = nbrs[0]
                    dr, dc = nr - r, nc - c
                    if not ((r in (0, H - 1)) and (c in (0, W - 1))):
                        perp = (
                            (r == 0 and dr == 1)
                            or (r == H - 1 and dr == -1)
                            or (c == 0 and dc == 1)
                            or (c == W - 1 and dc == -1)
                        )
                        assert perp, f"boundary endpoint ({r},{c}) color {color} not perpendicular"
                    continue


def _solve(grid: np.ndarray, *, ordered_red: bool) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    bg = int(get_bg(g))
    colors = set(map(int, g.ravel())) - {bg}
    assert colors, "expected foreground line and marker colors"
    marker_components, marker_colors = _parse_markers(g)
    line_colors = colors - MARKER_IDS
    assert line_colors, "expected at least one non-marker line color"
    assert marker_colors, "expected at least one marker color"

    original = {color: _pts(g == color) for color in line_colors}
    masks = {color: set(pts) for color, pts in original.items()}
    marker_pts = {color: set() for color in line_colors}
    assigned_markers: list[dict[str, object]] = []
    control_ray = {color: set() for color in line_colors}

    red_components = [info for info in marker_components if int(info["marker_color"]) == 2]
    yellow_components = [info for info in marker_components if int(info["marker_color"]) == 4]

    if ordered_red:
        pending = [dict(info) for info in red_components]
        while pending:
            candidates = _collect_red_candidates(pending, g, masks, line_colors)
            assert candidates, f"unrepairable red markers: {[p['cells'] for p in pending]}"
            _gap, _center_bias, _r, _c, _top_left, left, right, interior, _axis = candidates[0]
            color = int(left["line_color"])
            repaired_cells = set(left["cells"]) | set(right["cells"]) | set(interior)
            masks[color].update(repaired_cells)
            marker_pts[color].update(left["cells"])
            marker_pts[color].update(right["cells"])
            assigned_markers.extend([left, right])
            used = {tuple(left["cells"]), tuple(right["cells"])}
            pending = [p for p in pending if tuple(p["cells"]) not in used]
    else:
        assigned_red = []
        for info in red_components:
            color, anchor, direction, axis = _assign_marker(info, g, masks, line_colors)
            enriched = dict(info)
            enriched["line_color"] = int(color)
            enriched["anchor"] = anchor
            enriched["dir_to_line"] = direction
            enriched["axis"] = int(axis)
            assigned_red.append(enriched)
        candidates = _collect_red_candidates(assigned_red, g, masks, line_colors)
        used: set[tuple[tuple[int, int], ...]] = set()
        for _gap, _center_bias, _r, _c, _top_left, left, right, interior, _axis in sorted(candidates, key=lambda x: (x[2], x[3], x[0], x[1])):
            if tuple(left["cells"]) in used or tuple(right["cells"]) in used:
                continue
            color = int(left["line_color"])
            masks[color].update(set(left["cells"]) | set(right["cells"]) | set(interior))
            marker_pts[color].update(left["cells"])
            marker_pts[color].update(right["cells"])
            assigned_markers.extend([left, right])
            used.add(tuple(left["cells"]))
            used.add(tuple(right["cells"]))

    yellow_infos: list[dict[str, object]] = []
    for info in yellow_components:
        color, anchor, direction, axis = _assign_marker(info, g, masks, line_colors)
        enriched = dict(info)
        enriched["line_color"] = int(color)
        enriched["anchor"] = anchor
        enriched["dir_to_line"] = direction
        enriched["axis"] = int(axis)
        yellow_infos.append(enriched)
        marker_pts[color].update(info["cells"])
        assigned_markers.append(enriched)

    comp_basis = {color: set(original[color]) for color in line_colors}
    for info in assigned_markers:
        comp_basis[int(info["line_color"])].update(info["cells"])
    comp_count = {color: _component_count(g.shape, comp_basis[color]) for color in line_colors}

    for info in yellow_infos:
        color = int(info["line_color"])
        dr, dc = info["dir_to_line"]
        anchor = info["anchor"]
        cells_set = set(info["cells"])

        r, c = anchor[0] + dr, anchor[1] + dc
        pivot: tuple[int, int] | None = None
        while 0 <= r < H and 0 <= c < W:
            if (r, c) in masks[color] and (r, c) not in cells_set:
                pivot = (r, c)
                break
            r += dr
            c += dc
        assert pivot is not None, f"yellow marker {info['cells']} has no assigned pivot"

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
            assert old_arm, f"yellow marker {info['cells']} found no old arm to erase"
            for q in old_arm:
                masks[color].discard(q)
                control_ray[color].discard(q)

        pr, pc = -dr, -dc
        r, c = pivot
        while 0 <= r < H and 0 <= c < W:
            masks[color].add((r, c))
            control_ray[color].add((r, c))
            r += pr
            c += pc

    out = _resolve_claims(g, bg, line_colors, masks, marker_pts, comp_count, control_ray)
    _validate_output(out, bg, line_colors)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Repair a one-cell-thick colored line network whose instructions are carried by oriented 2-cell markers, using pair-clamped fills processed from near to far.

    Input structure:
    - A 12x12 grid with a dominant background and 2-4 non-marker line colors.
    - Line colors form one-cell-thick horizontal/vertical polyline fragments that may cross other colors.
    - Marker colors are red (2) and/or yellow (4), and each marker component is exactly one horizontal or vertical domino.
    - Red dominoes sit in pairs around a blank straight corridor; yellow dominoes sit on the first two cells of an intended new ray from a pivot.

    Rule:
    - Read each domino as one object and infer its candidate line color only along the domino's own axis.
    - Repeatedly find the currently shortest red-domino pair that clamps a blank corridor on that same axis, recolor both dominoes plus the cells between them, and then recompute the remaining candidates on the updated line state.
    - After the red repairs, each yellow domino erases the old arm on the far side of its pivot and extends the assigned color from the pivot through the domino to the grid edge.
    - At crossings, yellow-control rays override everything else; otherwise the displayed crossing color is chosen by the same fragmentation-first rule as the seed task.

    Key insight: The decisive objects are no longer isolated pixels but oriented dominoes, and unresolved red dominoes are not yet part of the line. You must pair the nearest compatible clamps first, because each repair changes which farther clamps are supported and therefore what the remaining network means.
    """
    return _solve(grid, ordered_red=True)


def _bulk_main(grid: np.ndarray) -> np.ndarray:
    return _solve(grid, ordered_red=False)


def _segment(a: tuple[int, int], b: tuple[int, int]) -> list[tuple[int, int]]:
    r0, c0 = a
    r1, c1 = b
    if r0 == r1:
        step = 1 if c1 >= c0 else -1
        return [(r0, c) for c in range(c0, c1 + step, step)]
    step = 1 if r1 >= r0 else -1
    return [(r, c0) for r in range(r0, r1 + step, step)]


def _path(points: list[tuple[int, int]]) -> list[tuple[int, int]]:
    cells: list[tuple[int, int]] = []
    for a, b in zip(points, points[1:]):
        seg = _segment(a, b)
        cells.extend(seg[1:] if cells else seg)
    return cells


def _on_boundary(r: int, c: int) -> bool:
    return r == 0 or r == H - 1 or c == 0 or c == W - 1


def _valid_path(cells: list[tuple[int, int]]) -> bool:
    pts = set(cells)
    if len(pts) != len(cells) or len(pts) < 2:
        return False
    for r, c in pts:
        if not (0 <= r < H and 0 <= c < W):
            return False
        if len(_dirs_in(pts, (r, c))) > 2:
            return False
    if _component_count((H, W), pts) != 1:
        return False
    for r, c in pts:
        if not _on_boundary(r, c):
            continue
        for dr, dc in _dirs_in(pts, (r, c)):
            nr, nc = r + dr, c + dc
            if (r in (0, H - 1) and nr == r) or (c in (0, W - 1) and nc == c):
                return False
        nbrs = _dirs_in(pts, (r, c))
        if len(nbrs) != 1:
            continue
        if (r in (0, H - 1)) and (c in (0, W - 1)):
            continue
        dr, dc = nbrs[0]
        if not ((r == 0 and dr == 1) or (r == H - 1 and dr == -1) or (c == 0 and dc == 1) or (c == W - 1 and dc == -1)):
            return False
    for r, c in pts:
        if len(_dirs_in(pts, (r, c))) == 1 and not _on_boundary(r, c):
            return False
    return True


def _rc(ax: int, major: int, minor: int) -> tuple[int, int]:
    return (major, minor) if ax == 0 else (minor, major)


def _dim(ax: int) -> int:
    return H if ax == 0 else W


def _random_span(limit: int) -> tuple[int, int]:
    if random.random() < 0.5:
        return 0, random.randint(MIN_SPAN_LEN, limit - 1)
    return random.randint(0, limit - MIN_SPAN_LEN - 1), limit - 1


def _build_polyline_waypoints(ax: int) -> list[tuple[int, int]]:
    pieces = random.choices(range(1, len(POLYLINE_PIECE_WEIGHTS) + 1), weights=POLYLINE_PIECE_WEIGHTS)[0]
    a0, a1 = _random_span(_dim(ax))
    pieces = min(pieces, max(1, a1 - a0))
    minors = random.sample(range(1, _dim(1 - ax) - 1), pieces)
    bends = sorted(random.sample(range(a0 + 1, a1), pieces - 1)) if pieces > 1 else []
    pts = [_rc(ax, a0, minors[0])]
    for i, b in enumerate(bends):
        pts.append(_rc(ax, b, minors[i]))
        pts.append(_rc(ax, b, minors[i + 1]))
    pts.append(_rc(ax, a1, minors[-1]))
    if not _on_boundary(*pts[0]):
        r0, c0 = pts[0]
        if ax == 0:
            pts.insert(0, (r0, random.choice([0, W - 1])))
        else:
            pts.insert(0, (random.choice([0, H - 1]), c0))
    if not _on_boundary(*pts[-1]):
        rn, cn = pts[-1]
        if ax == 0:
            pts.append((rn, random.choice([0, W - 1])))
        else:
            pts.append((random.choice([0, H - 1]), cn))
    return pts


def _build_arm(ax: int, main_coord: int, cross_lo: int, cross_hi: int, is_start: bool) -> list[tuple[int, int]]:
    dim_minor = _dim(1 - ax)
    arm_choices = [v for v in range(1, dim_minor - 1) if abs(v - main_coord) >= ARM_MIN_OFFSET]
    if not arm_choices:
        return []
    minor = random.choice(arm_choices)
    if is_start:
        if cross_lo <= 1:
            return []
        arm_major = random.randint(1, cross_lo - 1)
        return [_rc(ax, 0, minor), _rc(ax, arm_major, minor), _rc(ax, arm_major, main_coord)]
    if cross_hi >= _dim(ax) - 2:
        return []
    arm_major = random.randint(cross_hi + 1, _dim(ax) - 2)
    return [_rc(ax, arm_major, main_coord), _rc(ax, arm_major, minor), _rc(ax, _dim(ax) - 1, minor)]


def _path_through(cell: tuple[int, int], orient: str) -> list[tuple[int, int]] | None:
    r, c = cell
    ax = 0 if orient == "v" else 1
    major = r if ax == 0 else c
    minor = c if ax == 0 else r
    if major == 0 or major == _dim(ax) - 1:
        return None
    lo = 0 if random.random() < FULL_SPAN_PROB else random.randint(0, major - 1)
    hi = _dim(ax) - 1 if random.random() < FULL_SPAN_PROB else random.randint(major + 1, _dim(ax) - 1)
    pts = [_rc(ax, lo, minor), _rc(ax, hi, minor)]
    flavor = random.choice(("straight", "start_arm", "end_arm", "both_arms"))
    if flavor in ("start_arm", "both_arms") and lo > 0:
        arm = _build_arm(ax, minor, lo, hi, is_start=True)
        if arm:
            pts = arm + pts[1:]
    if flavor in ("end_arm", "both_arms") and hi < _dim(ax) - 1:
        arm = _build_arm(ax, minor, lo, hi, is_start=False)
        if arm:
            pts = pts[:-1] + arm
    if not _on_boundary(*pts[0]):
        pts[0] = _rc(ax, 0, minor)
    if not _on_boundary(*pts[-1]):
        pts[-1] = _rc(ax, _dim(ax) - 1, minor)
    cells = _path(pts)
    return cells if _valid_path(cells) and _orientation(set(cells), cell) == orient else None


def _spaced_values(limit: int, count: int) -> list[int] | None:
    choices = list(range(1, limit - 1))
    for _ in range(SPACED_VALUES_RETRIES):
        vals = sorted(random.sample(choices, count))
        if all(b - a >= 2 for a, b in zip(vals, vals[1:])):
            return vals
    return None


def _serpentine_path(count: int) -> tuple[list[tuple[int, int]], list[tuple[int, int]]] | None:
    ax = random.randint(0, 1)
    lanes = _spaced_values(_dim(ax), count)
    if lanes is None:
        return None
    if random.random() < 0.5:
        m0, m1 = 1, random.randint(SERPENTINE_MINOR_LO, _dim(1 - ax) - 2)
    else:
        m0, m1 = random.randint(1, SERPENTINE_MINOR_HI), _dim(1 - ax) - 2
    cross_minor = random.randint(m0 + CROSS_MINOR_MARGIN, m1 - CROSS_MINOR_MARGIN)

    start_at_m0 = random.random() < 0.5
    pts = [_rc(ax, lanes[0], m0 if start_at_m0 else m1)]
    fwd = start_at_m0
    for i, lane in enumerate(lanes):
        end_m = m1 if fwd else m0
        if pts[-1] != _rc(ax, lane, end_m):
            pts.append(_rc(ax, lane, end_m))
        if i + 1 < len(lanes):
            pts.append(_rc(ax, lanes[i + 1], end_m))
            fwd = not fwd

    if not _on_boundary(*pts[0]):
        r0, c0 = pts[0]
        if ax == 0:
            pts.insert(0, (r0, 0 if c0 == m0 else W - 1))
        else:
            pts.insert(0, (0 if r0 == m0 else H - 1, c0))
    if not _on_boundary(*pts[-1]):
        rn, cn = pts[-1]
        if ax == 0:
            pts.append((rn, 0 if cn == m0 else W - 1))
        else:
            pts.append((0 if rn == m0 else H - 1, cn))

    cells = _path(pts)
    crossings = [_rc(ax, lane, cross_minor) for lane in lanes]
    return (cells, crossings) if _valid_path(cells) else None


def _path_crossing_all(crossings: list[tuple[int, int]]) -> list[tuple[int, int]] | None:
    rows = {r for r, _ in crossings}
    cols = {c for _, c in crossings}
    if len(cols) == 1:
        ax, main_coord, cross_vals = 0, next(iter(cols)), rows
    elif len(rows) == 1:
        ax, main_coord, cross_vals = 1, next(iter(rows)), cols
    else:
        return None

    lo, hi = min(cross_vals), max(cross_vals)
    if lo <= 1 or hi >= _dim(ax) - 2:
        return None

    pts = [_rc(ax, 0, main_coord), _rc(ax, _dim(ax) - 1, main_coord)]
    flavor = random.choice(("straight", "start_arm", "end_arm", "both_arms"))
    if flavor in ("start_arm", "both_arms"):
        arm = _build_arm(ax, main_coord, lo, hi, is_start=True)
        if arm:
            pts = arm + pts[1:]
    if flavor in ("end_arm", "both_arms"):
        arm = _build_arm(ax, main_coord, lo, hi, is_start=False)
        if arm:
            pts = pts[:-1] + arm
    cells = _path(pts)
    return cells if _valid_path(cells) else None


def _can_add(logical: dict[int, set[tuple[int, int]]], color: int, cells: list[tuple[int, int]]) -> bool:
    if not _valid_path(cells):
        return False
    pts = set(cells)
    for pt in pts:
        if pt in logical[color]:
            return False
        if any(nb in logical[color] for nb in _neighbors(pt)):
            return False
        if sum(pt in pset for pset in logical.values()) >= 2:
            return False
    return True


def _crossing_cells(logical: dict[int, set[tuple[int, int]]]) -> list[tuple[int, int]]:
    cells = set().union(*logical.values())
    return [pt for pt in cells if sum(pt in logical[color] for color in logical) > 1]


def _valid_crossings(logical: dict[int, set[tuple[int, int]]], dominant: int) -> bool:
    crossings = _crossing_cells(logical)
    if not 1 <= len(crossings) <= MAX_CROSSINGS:
        return False
    for pt in crossings:
        colors = [color for color in logical if pt in logical[color]]
        if len(colors) != 2 or dominant not in colors:
            return False
        orientations = [_orientation(logical[color], pt) for color in colors]
        if None in orientations or sorted(orientations) != ["h", "v"]:
            return False
    return True


def _find_crossing_point(logical: dict[int, set[tuple[int, int]]]) -> tuple[tuple[int, int], str] | None:
    candidates = []
    for pts in logical.values():
        for r, c in pts:
            orient = _orientation(pts, (r, c))
            if orient is None:
                continue
            perp = "v" if orient == "h" else "h"
            major = r if perp == "v" else c
            if major <= 1 or major >= (H if perp == "v" else W) - 2:
                continue
            candidates.append(((r, c), perp))
    random.shuffle(candidates)
    return candidates[0] if candidates else None


def _build_logical() -> tuple[dict[int, set[tuple[int, int]]], dict[int, list[set[tuple[int, int]]]], int] | None:
    colors = random.sample(LINE_PALETTE, 4)
    dominant = colors[0]
    logical: dict[int, set[tuple[int, int]]] = {c: set() for c in colors}
    paths: dict[int, list[set[tuple[int, int]]]] = {c: [] for c in colors}

    cross_col = random.randint(3, W - 4)
    cross_rows = sorted(random.sample(range(3, H - 2), 3))

    dominant_cells = {(r, cross_col) for r in range(H)}
    dominant_cells.update({(1, c) for c in range(W)})
    logical[dominant].update(dominant_cells)
    paths[dominant].append(set(dominant_cells))

    for color, row in zip(colors[1:], cross_rows):
        row_cells = {(row, c) for c in range(W)}
        logical[color].update(row_cells)
        paths[color].append(set(row_cells))

    if not _valid_crossings(logical, dominant):
        return None
    return logical, paths, dominant


def _render_target(logical: dict[int, set[tuple[int, int]]], comp_count: dict[int, int], yellow_rays: dict[int, set[tuple[int, int]]]) -> np.ndarray:
    out = np.full((H, W), BG, dtype=int)
    for pt in set().union(*logical.values()):
        colors_here = [color for color in logical if pt in logical[color]]
        forced = [color for color in colors_here if pt in yellow_rays.get(color, set())]
        if forced:
            out[pt] = forced[0]
        elif len(colors_here) == 1:
            out[pt] = colors_here[0]
        else:
            out[pt] = min(colors_here, key=lambda color: (-comp_count[color], color))
    return out


def _marker_components_ok(grid: np.ndarray) -> bool:
    for color in MARKER_IDS:
        for comp in components_mask(grid == color, connectivity=4):
            cells = {tuple(map(int, rc)) for rc in comp}
            if _split_marker_component(cells) is None:
                return False
    return True


def _yellow_candidates(
    logical: dict[int, set[tuple[int, int]]],
    paths: dict[int, list[set[tuple[int, int]]]],
    crossings: set[tuple[int, int]],
) -> list[tuple[int, tuple[int, int], tuple[int, int], set[tuple[int, int]], set[tuple[int, int]], tuple[tuple[int, int], tuple[int, int]]]]:
    candidates = []
    occupied = set().union(*logical.values())
    for color, comps in paths.items():
        for comp in comps:
            for pivot in comp:
                if pivot in crossings or not _is_turn_or_endpoint(comp, pivot):
                    continue
                for out_dir in DIRS4:
                    first = (pivot[0] + out_dir[0], pivot[1] + out_dir[1])
                    second = (pivot[0] + 2 * out_dir[0], pivot[1] + 2 * out_dir[1])
                    if first not in comp or second not in comp:
                        continue
                    target_ray = _ray_from(pivot, out_dir)
                    if not target_ray.issubset(comp):
                        continue
                    old_dir = (-out_dir[0], -out_dir[1])
                    old_cells: set[tuple[int, int]] = set()
                    r, c = pivot[0] + old_dir[0], pivot[1] + old_dir[1]
                    while 0 <= r < H and 0 <= c < W and (r, c) not in occupied:
                        old_cells.add((r, c))
                        r += old_dir[0]
                        c += old_dir[1]
                    if not old_cells:
                        continue
                    old_len = random.randint(1, len(old_cells))
                    ordered_old = _ordered_ray_from((pivot[0] + old_dir[0], pivot[1] + old_dir[1]), old_dir)
                    old_arm = set(ordered_old[:old_len])
                    if old_arm and old_arm.issubset(old_cells):
                        candidates.append((color, pivot, out_dir, target_ray, old_arm, (first, second)))
    random.shuffle(candidates)
    return candidates


def _run_groups(values: list[int]) -> list[list[int]]:
    if not values:
        return []
    groups = [[values[0]]]
    for v in values[1:]:
        if v == groups[-1][-1] + 1:
            groups[-1].append(v)
        else:
            groups.append([v])
    return groups


def _red_chain_candidates(
    logical: dict[int, set[tuple[int, int]]],
    blocked: set[tuple[int, int]],
) -> list[tuple[int, list[tuple[tuple[int, int], tuple[int, int]]], list[tuple[int, int]], int]]:
    crossings = set(_crossing_cells(logical))
    candidates = []
    patterns = ((0, 1, 1), (1, 1, 0))
    for color, pts in logical.items():
        rows: dict[int, list[int]] = {}
        cols: dict[int, list[int]] = {}
        for r, c in pts:
            rows.setdefault(r, []).append(c)
            cols.setdefault(c, []).append(r)
        for r, cols_here in rows.items():
            for group in _run_groups(sorted(cols_here)):
                if len(group) < 10:
                    continue
                for g1, g2, g3 in patterns:
                    span = 8 + g1 + g2 + g3
                    if span > len(group):
                        continue
                    for s in range(0, len(group) - span + 1):
                        if s == 0 or s + span == len(group):
                            continue
                        idx = s
                        d1 = ((r, group[idx]), (r, group[idx + 1]))
                        idx += 2
                        gap1 = [(r, group[idx + k]) for k in range(g1)]
                        idx += g1
                        d2 = ((r, group[idx]), (r, group[idx + 1]))
                        idx += 2
                        gap2 = [(r, group[idx + k]) for k in range(g2)]
                        idx += g2
                        d3 = ((r, group[idx]), (r, group[idx + 1]))
                        idx += 2
                        gap3 = [(r, group[idx + k]) for k in range(g3)]
                        idx += g3
                        d4 = ((r, group[idx]), (r, group[idx + 1]))
                        cells = set(d1) | set(d2) | set(d3) | set(d4) | set(gap1) | set(gap2) | set(gap3)
                        if cells & blocked or cells & crossings:
                            continue
                        candidates.append((color, [d1, d2, d3, d4], gap1 + gap2 + gap3, 0))
        for c, rows_here in cols.items():
            for group in _run_groups(sorted(rows_here)):
                if len(group) < 10:
                    continue
                for g1, g2, g3 in patterns:
                    span = 8 + g1 + g2 + g3
                    if span > len(group):
                        continue
                    for s in range(0, len(group) - span + 1):
                        if s == 0 or s + span == len(group):
                            continue
                        idx = s
                        d1 = ((group[idx], c), (group[idx + 1], c))
                        idx += 2
                        gap1 = [(group[idx + k], c) for k in range(g1)]
                        idx += g1
                        d2 = ((group[idx], c), (group[idx + 1], c))
                        idx += 2
                        gap2 = [(group[idx + k], c) for k in range(g2)]
                        idx += g2
                        d3 = ((group[idx], c), (group[idx + 1], c))
                        idx += 2
                        gap3 = [(group[idx + k], c) for k in range(g3)]
                        idx += g3
                        d4 = ((group[idx], c), (group[idx + 1], c))
                        cells = set(d1) | set(d2) | set(d3) | set(d4) | set(gap1) | set(gap2) | set(gap3)
                        if cells & blocked or cells & crossings:
                            continue
                        candidates.append((color, [d1, d2, d3, d4], gap1 + gap2 + gap3, 1))
    random.shuffle(candidates)
    return candidates


def _derive_input(logical: dict[int, set[tuple[int, int]]], paths: dict[int, list[set[tuple[int, int]]]], dominant: int) -> tuple[np.ndarray, np.ndarray] | None:
    crossings = set(_crossing_cells(logical))
    yellow_count = random.randint(*YELLOW_COUNT_RANGE)
    yellow_infos = []
    blocked = set(crossings)
    if yellow_count:
        for cand in _yellow_candidates(logical, paths, crossings):
            color, pivot, out_dir, target_ray, old_arm, marker_pair = cand
            marker_cells = set(marker_pair)
            halo = set(marker_cells)
            for cell in marker_cells:
                halo.update(_neighbors(cell))
            if any(pt in blocked for pt in marker_cells) or any(pt in blocked for pt in halo - marker_cells):
                continue
            if any((old & old_arm) or (ray & old_arm) or (set(pair) & old_arm) for _, _, _, ray, old, pair in yellow_infos):
                continue
            yellow_infos.append(cand)
            blocked.update(target_ray | old_arm | marker_cells)
            if len(yellow_infos) == yellow_count:
                break
        if len(yellow_infos) != yellow_count:
            return None

    red_infos = []
    for cand in _red_chain_candidates(logical, blocked):
        color, dominos, interior, axis = cand
        if color != dominant:
            continue
        cells = set(interior)
        for domino in dominos:
            cells.update(domino)
        halo = set(cells)
        for cell in cells:
            halo.update(_neighbors(cell))
        if any(pt in blocked for pt in cells) or any(pt in blocked for pt in halo - cells):
            continue
        red_infos.append(cand)
        blocked.update(cells)
        if len(red_infos) == RED_CHAIN_COUNT:
            break
    if len(red_infos) != RED_CHAIN_COUNT:
        return None

    inp = np.full((H, W), BG, dtype=int)
    comp_count_masks: dict[int, set[tuple[int, int]]] = {c: set() for c in logical}
    crossing_display: dict[tuple[int, int], int] = {}
    for pt in crossings:
        colors = [color for color in logical if pt in logical[color]]
        if dominant not in colors:
            return None
        crossing_display[pt] = colors[0] if colors[1] == dominant else colors[1]

    removed_for_yellow: dict[int, set[tuple[int, int]]] = {c: set() for c in logical}
    old_arms: dict[int, set[tuple[int, int]]] = {c: set() for c in logical}
    yellow_rays: dict[int, set[tuple[int, int]]] = {c: set() for c in logical}
    yellow_markers: list[tuple[int, tuple[tuple[int, int], tuple[int, int]]]] = []
    for color, pivot, _out_dir, target_ray, old_arm, marker_pair in yellow_infos:
        removed_for_yellow[color].update(target_ray - {pivot})
        old_arms[color].update(old_arm)
        yellow_rays[color].update(target_ray)
        yellow_markers.append((color, marker_pair))

    red_marker_cells = {
        pt: color
        for color, dominos, _interior, _axis in red_infos
        for domino in dominos
        for pt in domino
    }
    red_gap_cells = {pt for _color, _dominos, interior, _axis in red_infos for pt in interior}
    yellow_by_pt = {pt: color for color, pair in yellow_markers for pt in pair}

    for color, pts in logical.items():
        for pt in pts:
            if pt in removed_for_yellow[color]:
                continue
            if pt in red_gap_cells or pt in red_marker_cells:
                continue
            if pt in crossings and crossing_display[pt] != color:
                continue
            if pt in yellow_by_pt:
                continue
            inp[pt] = color
            comp_count_masks[color].add(pt)

    for color, pts in old_arms.items():
        for pt in pts:
            if inp[pt] != BG:
                return None
            inp[pt] = color
            comp_count_masks[color].add(pt)

    for pt, color in red_marker_cells.items():
        if inp[pt] != BG:
            return None
        inp[pt] = 2
        comp_count_masks[color].add(pt)

    for pt, color in yellow_by_pt.items():
        if inp[pt] != BG:
            return None
        inp[pt] = 4
        comp_count_masks[color].add(pt)

    if not _marker_components_ok(inp):
        return None

    try:
        solved = main(inp)
        bulk = _bulk_main(inp)
    except Exception:
        return None
    if np.array_equal(solved, bulk):
        return None

    colors = sorted(logical)
    counts = {c: len(components_mask(solved == c, connectivity=4)) for c in colors}
    if not all(COMPONENTS_PER_COLOR[0] <= v <= COMPONENTS_PER_COLOR[1] for v in counts.values()):
        return None
    if not TOTAL_COMPONENTS[0] <= sum(counts.values()) <= TOTAL_COMPONENTS[1]:
        return None
    if not FILL_RANGE[0] <= int(np.count_nonzero(solved)) <= FILL_RANGE[1]:
        return None
    return inp, solved


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
