# Combo variant of faa9f03d (index 23)
# combo_spec: D1/0+D5/0+D10/0
# Name: 仅填充间隙 + 完整优先消歧 + 迭代收敛

from __future__ import annotations

import random

import numpy as np

from arc_utils import components_mask, get_bg

# Design analysis for this combo variant
# a) Concrete behavior:
#    The original task mixed two marker behaviors: red markers repaired gaps, yellow markers flipped an
#    endpoint arm into a border ray. Here that split is removed: every singleton marker, whether red or
#    yellow, is only a same-color repair anchor. The solver still assigns each marker to the nearest line
#    color in row/column sight, but after that it performs only gap filling. The new D10 layer means one
#    fill pass is not enough: a marker can first reconnect to a nearby supported stub, and only then become
#    supported enough to bridge a longer gap on its far side. Example on a row:
#    visible `AA . M .. AAAA` becomes `AAA M .. AAAA` after pass 1, then `AAA M AAAAA` after pass 2.
#    D5 is also flipped: when two colors claim the same crossing, the currently more complete color
#    (fewer connected components) wins, not the more fragmented one.
# b) Complexity calibration:
#    If the generator allowed only trivial one-cell repairs, iterative convergence would be invisible and
#    the combo would collapse to a one-pass task. The constructed samples therefore force at least one
#    two-stage cascade and at least one crossing where complete-first and fragmented-first disagree.
#    The family is kept narrow on purpose: a vertical backbone crosses two horizontal lines, which is
#    enough repetition for humans to see both the iterative fill and the reversed crossing preference.
# c) Hidden assumptions audit:
#    The original `main()` assumed yellow controls produce forced rays (`control_ray`) and that component
#    counts can be measured once before filling. Both assumptions break here. The new solver removes all
#    ray-specific state, recomputes component counts from the current visible grid each round, and resolves
#    overlaps every round before deciding whether another fill step is possible. Marker assignment remains
#    fixed from the input, because the markers themselves are the only stable evidence of missing cells.
# d) Placement legality:
#    The generator uses full logical polylines first, then carves visible fragments without overlap
#    conflicts other than intended crossings. Marker cells are isolated singletons, stay in-bounds, and
#    have unambiguous line-of-sight back to their intended line color. Crossings occur only at orthogonal
#    full-span lines, so every claimed overlap is geometrically legal.
# e) Visual observability:
#    The combo is visible from pixels alone. Both marker colors appear as the same kind of repair anchor,
#    there is no hidden dependence on numeric color ids beyond ordinary tie-breaking, and the samples are
#    rejected unless iterative repair adds cells that a one-pass complete-first solver would miss and
#    unless complete-first changes at least one crossing relative to fragmented-first.
# f) Visual cleanliness:
#    The scene is intentionally sparse: three one-cell-thick lines, two crossings, and two isolated
#    markers. There are no edge-running segments, no same-color junctions, and no stacked control/ray
#    semantics, so the important interactions stay legible.
# g) Inferability:
#    The color change is structurally cued by completeness. The vertical line is visibly less broken than
#    the horizontal ones in the input, and it consistently takes the crossing after repair. Because the
#    same relation repeats at two rows and both marker colors use the same repair behavior, the rule can
#    be inferred from a handful of examples instead of requiring hidden palette memorization.


DIRS4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
H = W = 12
BG = 0
LINE_PALETTE = [1, 3, 5, 6, 7, 8, 9]
MARKER_COLORS = (2, 4)
GEN_RETRIES = 200


def _pts(mask: np.ndarray) -> set[tuple[int, int]]:
    return {tuple(map(int, rc)) for rc in np.argwhere(mask)}


def _axis_support(mask: set[tuple[int, int]], r: int, c: int, axis: int) -> bool:
    if axis == 0:
        return (r, c - 1) in mask or (r, c + 1) in mask
    return (r - 1, c) in mask or (r + 1, c) in mask


def _perp_support(mask: set[tuple[int, int]], r: int, c: int, axis: int) -> bool:
    if axis == 0:
        return (r - 1, c) in mask or (r + 1, c) in mask
    return (r, c - 1) in mask or (r, c + 1) in mask


def _component_count(shape: tuple[int, int], pts: set[tuple[int, int]]) -> int:
    if not pts:
        return 0
    mask = np.zeros(shape, dtype=bool)
    for r, c in pts:
        mask[r, c] = True
    return len(components_mask(mask, connectivity=4))


def _choose_complete(colors_here: list[int] | set[int], comp_count: dict[int, int]) -> int:
    return min(colors_here, key=lambda color: (comp_count[color], color))


def _choose_fragmented(colors_here: list[int] | set[int], comp_count: dict[int, int]) -> int:
    return min(colors_here, key=lambda color: (-comp_count[color], color))


def _analyze_input(
    grid: np.ndarray,
) -> tuple[
    np.ndarray,
    int,
    set[int],
    set[int],
    dict[int, set[tuple[int, int]]],
    dict[tuple[int, int], int],
    dict[int, set[tuple[int, int]]],
]:
    g = np.asarray(grid, dtype=int)
    bg = int(get_bg(g))
    colors = set(map(int, g.ravel())) - {bg}
    assert colors, "expected foreground line and marker colors"

    marker_colors: set[int] = set()
    for color in colors:
        comps = components_mask(g == color, connectivity=4)
        assert comps, "foreground color must have at least one component"
        if max(len(comp) for comp in comps) == 1:
            marker_colors.add(int(color))

    line_colors = colors - marker_colors
    assert line_colors, "expected at least one non-marker line color"
    assert marker_colors, "expected singleton marker colors"

    original = {color: _pts(g == color) for color in line_colors}
    assignment: dict[tuple[int, int], int] = {}

    marker_mask = np.isin(g, list(marker_colors))
    for r0, c0 in _pts(marker_mask):
        candidates: list[tuple[int, int, int]] = []
        for dr, dc in DIRS4:
            r, c = r0 + dr, c0 + dc
            dist = 1
            while 0 <= r < g.shape[0] and 0 <= c < g.shape[1]:
                value = int(g[r, c])
                if value in line_colors:
                    axis = 0 if dr == 0 else 1
                    supported = _axis_support(original[value], r, c, axis)
                    candidates.append((dist, 0 if supported else 1, value))
                    break
                if value == bg or value in marker_colors:
                    r += dr
                    c += dc
                    dist += 1
                    continue
                break

        assert candidates, f"marker at {(r0, c0)} cannot see a line color"
        candidates.sort()
        assignment[(r0, c0)] = int(candidates[0][2])

    marker_pts = {color: set() for color in line_colors}
    for pt, color in assignment.items():
        marker_pts[color].add(pt)

    masks = {color: set(pts) | marker_pts[color] for color, pts in original.items()}
    return g, bg, line_colors, marker_colors, original, assignment, masks


def _has_axis_support(
    masks: dict[int, set[tuple[int, int]]],
    marker_pts: dict[int, set[tuple[int, int]]],
    color: int,
    r: int,
    c: int,
    axis: int,
) -> bool:
    del marker_pts
    return _axis_support(masks[color], r, c, axis)


def _should_fill(
    masks: dict[int, set[tuple[int, int]]],
    marker_pts: dict[int, set[tuple[int, int]]],
    color: int,
    a: tuple[int, int],
    b: tuple[int, int],
    axis: int,
) -> bool:
    r1, c1 = a
    r2, c2 = b
    gap = c2 - c1 - 1 if axis == 0 else r2 - r1 - 1
    if gap <= 0:
        return False

    a_axis = _has_axis_support(masks, marker_pts, color, r1, c1, axis)
    b_axis = _has_axis_support(masks, marker_pts, color, r2, c2, axis)
    if gap == 1:
        if a_axis or b_axis:
            return True
        a_perp = _perp_support(masks[color], r1, c1, axis)
        b_perp = _perp_support(masks[color], r2, c2, axis)
        return not (a_perp and b_perp)

    if (r1, c1) in marker_pts[color] and (r2, c2) in marker_pts[color]:
        return False
    return a_axis and b_axis


def _fill_round(
    masks: dict[int, set[tuple[int, int]]],
    marker_pts: dict[int, set[tuple[int, int]]],
    shape: tuple[int, int],
) -> dict[int, set[tuple[int, int]]]:
    H0, W0 = shape
    expanded = {color: set(pts) for color, pts in masks.items()}
    for color in sorted(masks):
        additions: set[tuple[int, int]] = set()
        for r in range(H0):
            cols = sorted(c for rr, c in masks[color] if rr == r)
            for c1, c2 in zip(cols, cols[1:]):
                if _should_fill(masks, marker_pts, color, (r, c1), (r, c2), axis=0):
                    additions.update((r, c) for c in range(c1 + 1, c2))

        for c in range(W0):
            rows = sorted(r for r, cc in masks[color] if cc == c)
            for r1, r2 in zip(rows, rows[1:]):
                if _should_fill(masks, marker_pts, color, (r1, c), (r2, c), axis=1):
                    additions.update((r, c) for r in range(r1 + 1, r2))

        expanded[color].update(additions)
    return expanded


def _resolve_claims(
    claims: dict[tuple[int, int], list[int]],
    shape: tuple[int, int],
    bg: int,
    comp_count: dict[int, int],
    prefer_complete: bool,
) -> np.ndarray:
    chooser = _choose_complete if prefer_complete else _choose_fragmented
    out = np.full(shape, bg, dtype=int)
    for pt, colors_here in claims.items():
        out[pt] = colors_here[0] if len(colors_here) == 1 else chooser(colors_here, comp_count)
    return out


def _repair(grid: np.ndarray, *, prefer_complete: bool, iterative: bool) -> np.ndarray:
    g, bg, line_colors, _marker_colors, _original, assignment, current = _analyze_input(grid)
    marker_pts = {color: set() for color in line_colors}
    for pt, color in assignment.items():
        marker_pts[color].add(pt)

    limit = g.shape[0] * g.shape[1]
    for _ in range(limit):
        comp_count = {color: _component_count(g.shape, current[color]) for color in line_colors}
        candidate = _fill_round(current, marker_pts, g.shape)
        claims: dict[tuple[int, int], list[int]] = {}
        for color, pts in candidate.items():
            for pt in pts:
                claims.setdefault(pt, []).append(int(color))

        out = _resolve_claims(claims, g.shape, bg, comp_count, prefer_complete)
        nxt = {color: _pts(out == color) for color in line_colors}
        if not iterative or nxt == current:
            return out
        current = nxt

    raise AssertionError("repair did not converge")


def _visible_neighbors(pts: set[tuple[int, int]], pt: tuple[int, int]) -> list[tuple[int, int]]:
    r, c = pt
    return [(nr, nc) for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)) if (nr, nc) in pts]


def _assert_output(out: np.ndarray, line_colors: set[int], bg: int) -> None:
    H0, W0 = out.shape
    for color in line_colors:
        for comp in components_mask(out == color, connectivity=4):
            pts_set = set(map(tuple, comp))
            for r, c in pts_set:
                nbrs = _visible_neighbors(pts_set, (r, c))
                if len(nbrs) != 1:
                    continue
                on_boundary = r == 0 or r == H0 - 1 or c == 0 or c == W0 - 1
                if on_boundary:
                    nr, nc = nbrs[0]
                    dr, dc = nr - r, nc - c
                    if not ((r in (0, H0 - 1)) and (c in (0, W0 - 1))):
                        perp = (
                            (r == 0 and dr == 1)
                            or (r == H0 - 1 and dr == -1)
                            or (c == 0 and dc == 1)
                            or (c == W0 - 1 and dc == -1)
                        )
                        assert perp, f"boundary endpoint ({r},{c}) color {color} not perpendicular"
                    continue
                adj_other = any(
                    0 <= r + dr < H0
                    and 0 <= c + dc < W0
                    and int(out[r + dr, c + dc]) not in (bg, color)
                    for dr, dc in DIRS4
                )
                assert adj_other, f"dangling endpoint ({r},{c}) color {color}: not on boundary or crossing"

    color_cells = {color: _pts(out == color) for color in line_colors}
    crossing_colors = set()
    for r in range(H0):
        for c in range(W0):
            a = int(out[r, c])
            if a not in line_colors:
                continue
            for b in line_colors - {a}:
                if (((r, c - 1) in color_cells[b] and (r, c + 1) in color_cells[b]) or
                        ((r - 1, c) in color_cells[b] and (r + 1, c) in color_cells[b])):
                    crossing_colors.add(a)
                    crossing_colors.add(b)
    for color in line_colors:
        assert color in crossing_colors, f"color {color} has no crossing with any other color"


def main(grid: np.ndarray) -> np.ndarray:
    """
    Rebuild a one-cell-thick rectilinear line network by recoloring singleton markers, repeatedly filling supported gaps, and resolving crossings in favor of the more complete color.

    Input structure:
    - A 12x12 grid has a dominant background plus several non-marker line colors.
    - The visible lines are sparse horizontal/vertical fragments of larger one-cell-thick rectilinear paths.
    - Marker colors are exactly the foreground colors whose 4-connected components are all single pixels.
    - Red and yellow markers no longer mean different actions: both are just isolated repair anchors that belong to some nearby line color.

    Rule:
    - Assign each marker to the nearest line color seen along its row or column, preferring a candidate that is already supported along that same axis.
    - Treat the assigned marker as part of that line color, then fill the cells between facing same-color points in a shared row or column whenever the local endpoint geometry supports a bridge.
    - After one fill pass, recompute the visible colored network and repeat the same repair process; newly repaired cells can create fresh supported endpoints that unlock longer bridges on later rounds.
    - Whenever multiple colors claim the same cell, show the currently more complete color there, measured by having fewer visible 4-connected components in that round; ties fall back to the smaller color id. Stop when another round changes nothing.

    Key insight: The singleton markers only tell you which fragments belong together; the real challenge is that one repair can enable the next, so the network must be completed to a fixed point, and crossings consistently follow the line that already looks less broken.
    """
    out = _repair(grid, prefer_complete=True, iterative=True)
    g = np.asarray(grid, dtype=int)
    bg = int(get_bg(g))
    line_colors = set(map(int, out.ravel())) - {bg}
    _assert_output(out, line_colors, bg)
    return out


def _paint_row(grid: np.ndarray, row: int, cols: range | list[int], color: int) -> None:
    for c in cols:
        grid[row, int(c)] = int(color)


def _paint_col(grid: np.ndarray, col: int, rows: range | list[int], color: int) -> None:
    for r in rows:
        grid[int(r), col] = int(color)


def _case_metrics(inp: np.ndarray) -> tuple[int, int]:
    combo = _repair(inp, prefer_complete=True, iterative=True)
    one_pass = _repair(inp, prefer_complete=True, iterative=False)
    fragmented = _repair(inp, prefer_complete=False, iterative=False)
    cascade_gain = int(np.count_nonzero(combo != one_pass))
    crossing_gain = int(np.count_nonzero(combo != fragmented))
    return cascade_gain, crossing_gain


def _build_case() -> np.ndarray | None:
    colors = sorted(random.sample(LINE_PALETTE, 3))
    backbone = colors[0]
    horiz_a = colors[1]
    horiz_b = colors[2]

    col = random.choice((7, 8))
    row_a = random.randint(2, 4)
    row_b = random.randint(7, 9)

    inp = np.full((H, W), BG, dtype=int)

    # Backbone vertical line: keep one crossing hidden by `horiz_a`, keep the other visible.
    _paint_col(inp, col, list(range(0, row_a)) + list(range(row_a + 1, H)), backbone)

    # `horiz_a`: more fragmented than the backbone, but fully repairable in one round.
    left_end_a = col - 6
    marker_a = col - 4
    rejoin_a = col - 2
    _paint_row(inp, row_a, range(0, left_end_a + 1), horiz_a)
    _paint_row(inp, row_a, range(rejoin_a, W), horiz_a)
    inp[row_a, marker_a] = random.choice(MARKER_COLORS)

    # `horiz_b`: true two-stage cascade; the last pre-crossing cell appears only after iteration.
    left_end_b = col - 4
    marker_b = col - 2
    right_start_b = col + 1
    _paint_row(inp, row_b, range(0, left_end_b + 1), horiz_b)
    _paint_row(inp, row_b, range(right_start_b, W), horiz_b)
    inp[row_b, col] = backbone
    inp[row_b, marker_b] = random.choice(MARKER_COLORS)

    for color in MARKER_COLORS:
        pts = _pts(inp == color)
        for r, c in pts:
            assert all(abs(r - rr) + abs(c - cc) > 1 for rr, cc in pts - {(r, c)})

    try:
        combo = main(inp)
    except Exception:
        return None

    cascade_gain, crossing_gain = _case_metrics(inp)
    if cascade_gain <= 0 or crossing_gain <= 0:
        return None

    non_bg = int(np.count_nonzero(combo))
    if not 30 <= non_bg <= 40:
        return None

    comps_total = sum(len(components_mask(combo == color, connectivity=4)) for color in colors)
    if not 4 <= comps_total <= 6:
        return None

    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        case = _build_case()
        if case is not None:
            return case
    raise RuntimeError("failed to generate a valid combo faa9f03d input")
