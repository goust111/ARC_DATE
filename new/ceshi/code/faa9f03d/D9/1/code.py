# Variant of faa9f03d
# Dimension: D9 — 折线的几何约束 (meta)
# Change: 直角折线（rectilinear polylines），每分量0-6个转角，每色1-4个连通分量，端点垂直于边界... → T形/十字形交叉节点

from __future__ import annotations

import random
import numpy as np

from arc_utils import components_mask, get_bg


# Design analysis
# yaml:
#   variant_focus: D9_same_color_junctions
#   concrete_shape:
#     output_grid: 12x12 with one-cell-thick orthogonal colored networks plus logical color crossings.
#     original_behavior: every same-color component behaves like a path; every pixel degree <= 2.
#     new_behavior:
#       - at least one color now contains a visible same-color T node (degree 3), and sometimes a + node (degree 4)
#       - the branch is created by attaching a straight orthogonal ray from an existing straight trunk cell to a boundary
#       - red markers may sit on branch hubs, so one missing pixel can reconnect 3 or 4 arms at once
#       - yellow markers may rotate one arm of a tree while preserving the perpendicular siblings at the pivot
#   behavior_trace:
#     case:
#       output: a horizontal trunk crosses a vertical stem of the same color, producing a T
#       input_original_logic: removing the hub pixel with a red marker still leaves a row gap and a column gap, but old solver
#         ranks marker ownership mainly by nearest supported endpoint and resolves some ties by color id
#       input_variant_logic:
#         - marker ownership prefers the color that can complete the richer local orthogonal pattern around the marker
#         - yellow deletion removes only the straight arm behind the pivot, not an arbitrary whole connected remainder
#       difference:
#         - branch hubs can be reconstructed even when the decisive evidence is "three visible arms of one color" rather than a simple path endpoint
#         - arm rotation is branch-safe: perpendicular siblings at a T node survive
#   complexity_calibration:
#     - keep the networks rectilinear and one-cell thick; do not introduce arbitrary mazes
#     - limit branch count to 1-2 junction anchors on one dominant color so the new geometry is obvious but not noisy
#     - keep 2-4 line colors and 1-12 crossings, matching the original visual regime
#     - avoid hidden tie-breaks by color id; use visible cues only and reject ambiguous generated cases
#   implicit_assumptions_audited:
#     marker_assignment:
#       original_assumption: one best candidate is identified by nearest supported path endpoint
#       variant_issue: a red marker on a T hub can expose multiple same-distance colors; local arm-count evidence matters
#       fix: score candidates by visible direction count, axis support, and aligned span before accepting them
#     yellow_erase:
#       original_assumption: deleting the far side of the pivot can flood through the whole component with no collateral damage
#       variant_issue: in a tree, that can erase sibling branches if the deletion is not constrained
#       fix: delete only the straight collinear arm until the first perpendicular support cell
#     output_validation:
#       original_assumption: only leaves matter; degree>2 never occurs
#       variant_issue: degree 3/4 is now valid but must still be orthogonal, thin, and boundary-rooted
#       fix: allow degree <= 4 and require every junction to have both a horizontal and a vertical support
#   placement_legality:
#     - branch rays are only attached through empty cells and must terminate on the boundary
#     - branch cells may not touch other colors by side adjacency unless they are intended overlaps at a crossing cell
#     - yellow target rays are kept crossing-free; red markers are isolated singletons
#   visual_observability:
#     - every generated sample contains at least one visible same-color junction in the solved output
#     - most samples also place at least one red marker on such a junction, so the new tree geometry is part of the repair itself
#     - color-id ordering is never used to decide ownership or crossing winners
#   cleanliness:
#     - only one dominant color gets extra branches, preventing the grid from becoming a thicket
#     - branch length is at least 2 cells, so the new geometry is visually obvious
#   generator_guarantees_for_solver:
#     - marker sight-lines are unambiguous under the visible scoring used by main()
#     - every yellow old arm is straight, so main()'s constrained deletion matches construction
#     - pure filled-vs-filled crossing ties are rejected during generation


DIRS4 = ((-1, 0), (1, 0), (0, -1), (0, 1))


def _pts(mask: np.ndarray) -> set[tuple[int, int]]:
    return {tuple(map(int, rc)) for rc in np.argwhere(mask)}


def _neighbors(pt: tuple[int, int]) -> list[tuple[int, int]]:
    r, c = pt
    return [(r + dr, c + dc) for dr, dc in DIRS4]


def _dirs_in(pts: set[tuple[int, int]], pt: tuple[int, int]) -> list[tuple[int, int]]:
    r, c = pt
    return [(dr, dc) for dr, dc in DIRS4 if (r + dr, c + dc) in pts]


def _axis_support(mask: set[tuple[int, int]], r: int, c: int, axis: int) -> bool:
    if axis == 0:
        return (r, c - 1) in mask or (r, c + 1) in mask
    return (r - 1, c) in mask or (r + 1, c) in mask


def _perp_support(mask: set[tuple[int, int]], r: int, c: int, axis: int) -> bool:
    if axis == 0:
        return (r - 1, c) in mask or (r + 1, c) in mask
    return (r, c - 1) in mask or (r, c + 1) in mask


def _component_count(shape: tuple[int, int], pts: set[tuple[int, int]]) -> int:
    mask = np.zeros(shape, dtype=bool)
    for r, c in pts:
        mask[r, c] = True
    return len(components_mask(mask, connectivity=4))


def _degree(pts: set[tuple[int, int]], pt: tuple[int, int]) -> int:
    return len(_dirs_in(pts, pt))


def _orientation(pts: set[tuple[int, int]], pt: tuple[int, int]) -> str | None:
    dirs = _dirs_in(pts, pt)
    if len(dirs) != 2 or (dirs[0][0] + dirs[1][0], dirs[0][1] + dirs[1][1]) != (0, 0):
        return None
    return "v" if dirs[0][0] else "h"


def _is_corner(pts: set[tuple[int, int]], pt: tuple[int, int]) -> bool:
    dirs = _dirs_in(pts, pt)
    return len(dirs) == 2 and (dirs[0][0] + dirs[1][0], dirs[0][1] + dirs[1][1]) != (0, 0)


def _is_marker_site(pts: set[tuple[int, int]], pt: tuple[int, int]) -> bool:
    deg = _degree(pts, pt)
    return deg == 1 or deg >= 3 or _is_corner(pts, pt)


def _on_boundary(r: int, c: int, H: int, W: int) -> bool:
    return r == 0 or r == H - 1 or c == 0 or c == W - 1


def _span_len(pts: set[tuple[int, int]], pt: tuple[int, int], axis: int) -> int:
    r, c = pt
    total = 1
    if axis == 0:
        cc = c - 1
        while (r, cc) in pts:
            total += 1
            cc -= 1
        cc = c + 1
        while (r, cc) in pts:
            total += 1
            cc += 1
        return total
    rr = r - 1
    while (rr, c) in pts:
        total += 1
        rr -= 1
    rr = r + 1
    while (rr, c) in pts:
        total += 1
        rr += 1
    return total


def _visible_directions(
    grid: np.ndarray,
    marker_colors: set[int],
    line_color: int,
    pt: tuple[int, int],
) -> list[tuple[int, int, int]]:
    H, W = grid.shape
    out: list[tuple[int, int, int]] = []
    r0, c0 = pt
    for dr, dc in DIRS4:
        r, c = r0 + dr, c0 + dc
        dist = 1
        while 0 <= r < H and 0 <= c < W:
            value = int(grid[r, c])
            if value == line_color:
                out.append((dr, dc, dist))
                break
            if value == int(get_bg(grid)) or value in marker_colors:
                r += dr
                c += dc
                dist += 1
                continue
            break
    return out


def _marker_completion_score(
    grid: np.ndarray,
    marker_colors: set[int],
    color: int,
    pt: tuple[int, int],
    original_pts: set[tuple[int, int]],
) -> tuple[int, int, int]:
    dirs = _visible_directions(grid, marker_colors, color, pt)
    axis_count = sum(1 for dr, dc, _ in dirs if dr == 0) + sum(1 for dr, dc, _ in dirs if dc == 0)
    same_axis_pairs = 0
    has_lr = any((dr, dc) == (0, -1) for dr, dc, _ in dirs) and any((dr, dc) == (0, 1) for dr, dc, _ in dirs)
    has_ud = any((dr, dc) == (-1, 0) for dr, dc, _ in dirs) and any((dr, dc) == (1, 0) for dr, dc, _ in dirs)
    if has_lr:
        same_axis_pairs += 1
    if has_ud:
        same_axis_pairs += 1
    best_span = max(
        (
            _span_len(original_pts, (r, c), 0 if dr == 0 else 1)
            for dr, dc, _dist in dirs
            for r, c in [(
                pt[0] + dr * _dist,
                pt[1] + dc * _dist,
            )]
        ),
        default=0,
    )
    return same_axis_pairs, len(dirs), best_span


def _pick_unique_min(candidates: list[tuple[tuple[int, ...], int, tuple[int, int]]], label: str) -> tuple[int, tuple[int, int]]:
    assert candidates, label
    best_key = min(key for key, _color, _dir in candidates)
    best = [(color, direction) for key, color, direction in candidates if key == best_key]
    colors = {color for color, _direction in best}
    assert len(colors) == 1, f"ambiguous {label}: {best}"
    return best[0]


def _pick_unique_max(scored: list[tuple[int, tuple[int, ...]]], label: str) -> int:
    assert scored, label
    best_key = max(score for _color, score in scored)
    best = [color for color, score in scored if score == best_key]
    assert len(best) == 1, f"ambiguous {label}: {best}"
    return best[0]


def _straight_old_arm(
    pts: set[tuple[int, int]],
    pivot: tuple[int, int],
    direction: tuple[int, int],
) -> set[tuple[int, int]]:
    dr, dc = direction
    axis = 0 if dr == 0 else 1
    old_arm: set[tuple[int, int]] = set()
    r, c = pivot[0] + dr, pivot[1] + dc
    while (r, c) in pts:
        if _perp_support(pts, r, c, axis):
            break
        old_arm.add((r, c))
        r += dr
        c += dc
    return old_arm


def _marker_brackets(marker_pts: dict[int, set[tuple[int, int]]], color: int, pt: tuple[int, int]) -> bool:
    r, c = pt
    same_row = [mc for mr, mc in marker_pts[color] if mr == r]
    if any(mc < c for mc in same_row) and any(mc > c for mc in same_row):
        return True
    same_col = [mr for mr, mc in marker_pts[color] if mc == c]
    return any(mr < r for mr in same_col) and any(mr > r for mr in same_col)


def _claim_score(
    masks: dict[int, set[tuple[int, int]]],
    marker_pts: dict[int, set[tuple[int, int]]],
    comp_count: dict[int, int],
    color: int,
    pt: tuple[int, int],
) -> tuple[int, int, int, int]:
    return (
        comp_count[color],
        1 if _marker_brackets(marker_pts, color, pt) else 0,
        _span_len(masks[color], pt, 0),
        _span_len(masks[color], pt, 1),
    )


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
        forced = [color for color in colors_here if pt in control_ray.get(color, set())]
        if forced:
            out[pt] = _pick_unique_max(
                [(color, _claim_score(masks, marker_pts, comp_count, color, pt)) for color in forced],
                f"yellow-control claim at {pt}",
            )
            continue

        if len(colors_here) == 1:
            out[pt] = colors_here[0]
            continue

        if any(int(g[pt]) == color for color in colors_here):
            pool = colors_here
        else:
            bracketed = [color for color in colors_here if _marker_brackets(marker_pts, color, pt)]
            pool = bracketed if bracketed else colors_here

        out[pt] = _pick_unique_max(
            [(color, _claim_score(masks, marker_pts, comp_count, color, pt)) for color in pool],
            f"crossing claim at {pt}",
        )
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Repair a one-cell-thick colored line tree by recoloring singleton markers, filling orthogonal gaps, and rotating yellow-marked arms without destroying same-color branches.

    Input structure:
    - A rectangular grid with a dominant background.
    - Non-marker colors form thin horizontal/vertical stroke fragments that may now include same-color T or + junctions.
    - Marker colors are exactly those whose 4-connected components are isolated single pixels.
    - Red singletons indicate missing line cells; yellow singletons indicate a branch arm that should be redirected through the marker side.

    Rule:
    - Assign each marker to the visible line color that best completes a local orthogonal pattern around it, using line-of-sight, arm count, axis support, and span as visible evidence.
    - For each line color, fill the row/column gaps between facing supports exactly as in the original repair process; this can now restore branch hubs as well as simple paths.
    - A yellow control deletes only the straight old arm behind its pivot until the first preserved junction, then paints the opposite ray from the pivot through the marker to the grid edge.
    - At overlap cells, yellow rays win; otherwise choose the most visibly supported repaired claimant, prioritizing fragmentation and marker bracketing rather than color id.

    Key insight: The singleton markers are still local repair cues, but once same-color junctions are allowed the decisive evidence is no longer "which path endpoint is closest" alone; the correct color is the one whose surrounding visible arms assemble the intended tree hub or redirected branch.
    """
    g = np.asarray(grid, dtype=int)
    bg = int(get_bg(g))
    H, W = g.shape

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
    assign_dir: dict[tuple[int, int], tuple[int, int]] = {}

    marker_mask = np.isin(g, list(marker_colors))
    for r0, c0 in _pts(marker_mask):
        candidates: list[tuple[tuple[int, ...], int, tuple[int, int]]] = []
        for dr, dc in DIRS4:
            r, c = r0 + dr, c0 + dc
            dist = 1
            while 0 <= r < H and 0 <= c < W:
                value = int(g[r, c])
                if value in line_colors:
                    axis = 0 if dr == 0 else 1
                    supported = _axis_support(original[value], r, c, axis)
                    pairs, vis_dirs, best_span = _marker_completion_score(
                        g, marker_colors, value, (r0, c0), original[value]
                    )
                    key = (
                        dist,
                        -pairs,
                        -vis_dirs,
                        0 if supported else 1,
                        -best_span,
                    )
                    candidates.append((key, int(value), (int(dr), int(dc))))
                    break
                if value == bg or value in marker_colors:
                    r += dr
                    c += dc
                    dist += 1
                    continue
                break

        color, direction = _pick_unique_min(candidates, f"marker at {(r0, c0)} cannot see a unique line color")
        assignment[(r0, c0)] = int(color)
        assign_dir[(r0, c0)] = direction

    masks = {color: set(pts) for color, pts in original.items()}
    marker_pts = {color: set() for color in line_colors}
    for pt, color in assignment.items():
        masks[color].add(pt)
        marker_pts[color].add(pt)

    comp_count = {color: _component_count(g.shape, masks[color]) for color in line_colors}

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

    for color in line_colors:
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
            assert pivot is not None, f"yellow control at {control} has no assigned pivot"

            old_arm = _straight_old_arm(masks[color], pivot, (dr, dc))
            assert old_arm, f"yellow control at {control} found no straight old arm to erase"
            for q in old_arm:
                masks[color].discard(q)
                marker_pts[color].discard(q)
                control_ray[color].discard(q)

            pr, pc = -dr, -dc
            r, c = pivot
            while 0 <= r < H and 0 <= c < W:
                masks[color].add((r, c))
                control_ray[color].add((r, c))
                r += pr
                c += pc

    out = _resolve_claims(g, bg, line_colors, masks, marker_pts, comp_count, control_ray)

    for color in line_colors:
        for comp in components_mask(out == color, connectivity=4):
            pts_set = set(map(tuple, comp))
            for r, c in pts_set:
                dirs = _dirs_in(pts_set, (r, c))
                deg = len(dirs)
                assert 1 <= deg <= 4, f"bad degree {deg} at {(r, c)} color {color}"
                if deg >= 3:
                    assert any(dc != 0 for _dr, dc in dirs) and any(dr != 0 for dr, _dc in dirs), (
                        f"non-orthogonal junction at {(r, c)} color {color}"
                    )
                if deg != 1:
                    continue
                on_boundary = _on_boundary(r, c, H, W)
                if on_boundary:
                    nr, nc = r + dirs[0][0], c + dirs[0][1]
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
                adj_other = any(
                    0 <= r + dr < H
                    and 0 <= c + dc < W
                    and int(out[r + dr, c + dc]) not in (bg, color)
                    for dr, dc in DIRS4
                )
                assert adj_other, f"dangling endpoint ({r},{c}) color {color}: not on boundary or crossing"

        pts_set = _pts(out == color)
        for r in range(H - 1):
            for c in range(W - 1):
                block = {(r, c), (r + 1, c), (r, c + 1), (r + 1, c + 1)}
                assert not block.issubset(pts_set), f"2x2 thickness at {(r, c)} color {color}"

    color_cells = {c: set() for c in line_colors}
    for r in range(H):
        for c in range(W):
            v = int(out[r, c])
            if v in line_colors:
                color_cells[v].add((r, c))
    crossing_colors = set()
    for r in range(H):
        for c in range(W):
            a = int(out[r, c])
            if a not in line_colors:
                continue
            for b in line_colors - {a}:
                if (((r, c - 1) in color_cells[b] and (r, c + 1) in color_cells[b])
                        or ((r - 1, c) in color_cells[b] and (r + 1, c) in color_cells[b])):
                    crossing_colors.add(a)
                    crossing_colors.add(b)
    for color in line_colors:
        assert color in crossing_colors, f"color {color} has no crossing with any other color"

    return out


# --------------- generator ---------------

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

BRANCH_COUNT_RANGE = (1, 2)
BRANCH_MIN_LEN = 2
BRANCH_ATTACH_RETRIES = 80
BRANCH_DOUBLE_PROB = 0.25

MARKER_MODE_WEIGHTS = [0, 1]  # red / both; keep yellow branch-rotation active for D9 observability
YELLOW_COUNT_RANGE = (0, 1)
RED_COUNT_RANGE = (1, 4)
MIN_CELLS_FOR_RED = 2

COMPONENTS_PER_COLOR = (1, 4)
TOTAL_COMPONENTS = (3, 6)
FILL_RANGE = (34, 66)

SPACED_VALUES_RETRIES = 60
EXTRA_POLYLINE_RETRIES = 80
GEN_RETRIES = 1500


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


def _gen_component_count(pts: set[tuple[int, int]]) -> int:
    if not pts:
        return 0
    mask = np.zeros((H, W), dtype=bool)
    for r, c in pts:
        mask[r, c] = True
    return len(components_mask(mask, connectivity=4))


def _valid_network(pts: set[tuple[int, int]]) -> bool:
    if not pts:
        return False
    for r, c in pts:
        if not (0 <= r < H and 0 <= c < W):
            return False
    if _gen_component_count(pts) != 1:
        return False
    for r in range(H - 1):
        for c in range(W - 1):
            block = {(r, c), (r + 1, c), (r, c + 1), (r + 1, c + 1)}
            if block.issubset(pts):
                return False
    for pt in pts:
        dirs = _dirs_in(pts, pt)
        deg = len(dirs)
        if deg < 1 or deg > 4:
            return False
        if deg >= 3:
            if not (any(dc != 0 for _dr, dc in dirs) and any(dr != 0 for dr, _dc in dirs)):
                return False
        if _on_boundary(pt[0], pt[1], H, W):
            for dr, dc in dirs:
                nr, nc = pt[0] + dr, pt[1] + dc
                if (pt[0] in (0, H - 1) and nr == pt[0]) or (pt[1] in (0, W - 1) and nc == pt[1]):
                    return False
        if deg == 1 and not _on_boundary(pt[0], pt[1], H, W):
            return False
    return True


def _valid_path(cells: list[tuple[int, int]]) -> bool:
    pts = set(cells)
    if len(pts) != len(cells) or len(pts) < 2:
        return False
    return _valid_network(pts) and all(_degree(pts, pt) <= 2 for pt in pts)


def _junction_cells(pts: set[tuple[int, int]]) -> set[tuple[int, int]]:
    return {pt for pt in pts if _degree(pts, pt) >= 3}


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
    if not _on_boundary(*pts[0], H, W):
        r0, c0 = pts[0]
        pts.insert(0, (r0, random.choice([0, W - 1])) if ax == 0 else (random.choice([0, H - 1]), c0))
    if not _on_boundary(*pts[-1], H, W):
        rn, cn = pts[-1]
        pts.append((rn, random.choice([0, W - 1])) if ax == 0 else (random.choice([0, H - 1]), cn))
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
    if not _on_boundary(*pts[0], H, W):
        pts[0] = _rc(ax, 0, minor)
    if not _on_boundary(*pts[-1], H, W):
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
    if not _on_boundary(*pts[0], H, W):
        r0, c0 = pts[0]
        pts.insert(0, (r0, 0 if c0 == m0 else W - 1) if ax == 0 else (0 if r0 == m0 else H - 1, c0))
    if not _on_boundary(*pts[-1], H, W):
        rn, cn = pts[-1]
        pts.append((rn, 0 if cn == m0 else W - 1) if ax == 0 else (0 if rn == m0 else H - 1, cn))
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
            if 1 < major < (H if perp == "v" else W) - 2:
                candidates.append(((r, c), perp))
    random.shuffle(candidates)
    return candidates[0] if candidates else None


def _branch_ray(
    occupied: set[tuple[int, int]],
    junction: tuple[int, int],
    direction: tuple[int, int],
) -> list[tuple[int, int]] | None:
    cells: list[tuple[int, int]] = []
    dr, dc = direction
    r, c = junction[0] + dr, junction[1] + dc
    while 0 <= r < H and 0 <= c < W:
        pt = (r, c)
        if pt in occupied:
            return None
        for nb in _neighbors(pt):
            if nb == junction or nb in cells:
                continue
            if nb in occupied:
                return None
        cells.append(pt)
        if _on_boundary(r, c, H, W):
            break
        r += dr
        c += dc
    if len(cells) < BRANCH_MIN_LEN or not _on_boundary(*cells[-1], H, W):
        return None
    return cells


def _branch_options(
    comp: set[tuple[int, int]],
    occupied: set[tuple[int, int]],
    forbidden: set[tuple[int, int]],
) -> list[set[tuple[int, int]]]:
    options: list[set[tuple[int, int]]] = []
    for pt in random.sample(list(comp), len(comp)):
        if pt in forbidden:
            continue
        orient = _orientation(comp, pt)
        if orient is None:
            continue
        dirs = [(-1, 0), (1, 0)] if orient == "h" else [(0, -1), (0, 1)]
        valid = []
        for direction in dirs:
            ray = _branch_ray(occupied, pt, direction)
            if ray is not None:
                branch = set(ray)
                if _valid_network(comp | branch):
                    valid.append(branch)
                    options.append(branch)
        if len(valid) == 2 and random.random() < BRANCH_DOUBLE_PROB:
            both = valid[0] | valid[1]
            if _valid_network(comp | both):
                options.append(both)
    return options


def _attach_branches_to_dominant(
    logical: dict[int, set[tuple[int, int]]],
    paths: dict[int, list[set[tuple[int, int]]]],
    dominant: int,
) -> bool:
    comp = paths[dominant][0]
    occupied = set().union(*logical.values())
    crossings = set(_crossing_cells(logical))
    target = random.randint(*BRANCH_COUNT_RANGE)
    attached = 0
    for _ in range(BRANCH_ATTACH_RETRIES):
        options = _branch_options(comp, occupied, crossings)
        if not options:
            break
        branch = random.choice(options)
        comp = comp | branch
        logical[dominant].update(branch)
        paths[dominant][0] = comp
        occupied.update(branch)
        attached += 1
        if attached >= target:
            break
    return attached >= 1 and bool(_junction_cells(comp))


def _empty_ray_to_boundary(
    occupied: set[tuple[int, int]],
    start: tuple[int, int],
    direction: tuple[int, int],
    allow_adjacent: set[tuple[int, int]] | None = None,
) -> list[tuple[int, int]] | None:
    cells: list[tuple[int, int]] = []
    allow_adjacent = allow_adjacent or set()
    dr, dc = direction
    r, c = start
    while 0 <= r < H and 0 <= c < W:
        pt = (r, c)
        if pt in occupied:
            return None
        for nb in _neighbors(pt):
            if nb in cells:
                continue
            if nb in occupied and nb not in allow_adjacent:
                return None
        cells.append(pt)
        if _on_boundary(r, c, H, W):
            return cells
        r += dr
        c += dc
    return None


def _build_special_yellow(
    logical: dict[int, set[tuple[int, int]]],
    paths: dict[int, list[set[tuple[int, int]]]],
    dominant: int,
) -> tuple[int, tuple[int, int], tuple[int, int], set[tuple[int, int]], set[tuple[int, int]]] | None:
    crossings = set(_crossing_cells(logical))
    comp = paths[dominant][0]
    occupied = set().union(*logical.values())
    for pivot in random.sample(list(comp), len(comp)):
        if pivot in crossings:
            continue
        for out_dir in random.sample(list(DIRS4), len(DIRS4)):
            marker = (pivot[0] - out_dir[0], pivot[1] - out_dir[1])
            if not (0 <= marker[0] < H and 0 <= marker[1] < W):
                continue
            s1 = (pivot[0] + out_dir[0], pivot[1] + out_dir[1])
            s2 = (pivot[0] + 2 * out_dir[0], pivot[1] + 2 * out_dir[1])
            if not (0 <= s2[0] < H and 0 <= s2[1] < W):
                continue
            if s1 not in comp or s2 not in comp:
                continue
            axis = 0 if out_dir[0] == 0 else 1
            if not _perp_support(comp, s2[0], s2[1], axis):
                continue
            ray_cells = _empty_ray_to_boundary(occupied, marker, (-out_dir[0], -out_dir[1]), {pivot})
            if ray_cells is None or any(pt in crossings for pt in ray_cells):
                continue
            target_ray = {pivot, *ray_cells}
            new_dom = set(logical[dominant])
            new_dom.discard(s1)
            new_dom.update(ray_cells)
            mask = np.zeros((H, W), dtype=bool)
            for r, c in new_dom:
                mask[r, c] = True
            new_comps = [set(map(tuple, comp_arr)) for comp_arr in components_mask(mask, connectivity=4)]
            if not new_comps or not all(_valid_network(comp_set) for comp_set in new_comps):
                continue
            logical[dominant] = new_dom
            paths[dominant] = new_comps
            return dominant, pivot, out_dir, target_ray, {s1}
    return None


def _ray_from(pivot: tuple[int, int], direction: tuple[int, int]) -> set[tuple[int, int]]:
    out = set()
    r, c = pivot
    dr, dc = direction
    while 0 <= r < H and 0 <= c < W:
        out.add((r, c))
        r += dr
        c += dc
    return out


def _yellow_candidates(
    logical: dict[int, set[tuple[int, int]]],
    paths: dict[int, list[set[tuple[int, int]]]],
    crossings: set[tuple[int, int]],
) -> list[tuple[int, tuple[int, int], tuple[int, int], set[tuple[int, int]], set[tuple[int, int]]]]:
    candidates = []
    occupied = set().union(*logical.values())
    for color, comps in paths.items():
        for comp in comps:
            for pivot in comp:
                if pivot in crossings:
                    continue
                for out_dir in DIRS4:
                    marker = (pivot[0] - out_dir[0], pivot[1] - out_dir[1])
                    if not (0 <= marker[0] < H and 0 <= marker[1] < W):
                        continue
                    if marker in occupied:
                        continue
                    target_ray = _ray_from(pivot, (-out_dir[0], -out_dir[1]))
                    if not target_ray.issubset(comp) or len(target_ray) < 2:
                        continue
                    if any(pt in crossings for pt in target_ray):
                        continue
                    if _degree(comp, pivot) == 1:
                        continue
                    old_cells: list[tuple[int, int]] = []
                    r, c = pivot[0] + out_dir[0], pivot[1] + out_dir[1]
                    while 0 <= r < H and 0 <= c < W and (r, c) not in occupied:
                        old_cells.append((r, c))
                        r += out_dir[0]
                        c += out_dir[1]
                    if not old_cells:
                        continue
                    arm_len = random.randint(1, len(old_cells))
                    candidates.append((color, pivot, out_dir, target_ray, set(old_cells[:arm_len])))
    random.shuffle(candidates)
    return candidates


def _choose_red_markers(
    logical: dict[int, set[tuple[int, int]]],
    paths: dict[int, list[set[tuple[int, int]]]],
    blocked: set[tuple[int, int]],
    count: int,
    require_junction: bool,
) -> list[tuple[int, tuple[int, int]]]:
    junction_candidates = [
        (color, pt)
        for color, comps in paths.items()
        for comp in comps
        for pt in comp
        if pt not in blocked and _degree(comp, pt) >= 3
    ]
    random.shuffle(junction_candidates)
    chosen: list[tuple[int, tuple[int, int]]] = []
    used = set(blocked)
    per_color_left = {color: len(logical[color]) for color in logical}

    if require_junction:
        for color, pt in junction_candidates:
            if per_color_left[color] <= MIN_CELLS_FOR_RED:
                continue
            if any(nb in used for nb in [pt] + _neighbors(pt)):
                continue
            chosen.append((color, pt))
            used.add(pt)
            per_color_left[color] -= 1
            break
        if not chosen:
            return []

    candidates = [
        (color, pt)
        for color, comps in paths.items()
        for comp in comps
        for pt in comp
        if pt not in blocked and _is_marker_site(comp, pt)
    ]
    random.shuffle(candidates)
    for color, pt in candidates:
        if len(chosen) == count:
            break
        if per_color_left[color] <= MIN_CELLS_FOR_RED:
            continue
        if any(nb in used for nb in [pt] + _neighbors(pt)):
            continue
        if any(existing_pt == pt for _existing_color, existing_pt in chosen):
            continue
        chosen.append((color, pt))
        used.add(pt)
        per_color_left[color] -= 1
    return chosen if len(chosen) == count else []


def _build_logical() -> tuple[dict[int, set[tuple[int, int]]], dict[int, list[set[tuple[int, int]]]], int] | None:
    desired_crossings = random.choices(range(1, len(CROSSING_COUNT_WEIGHTS) + 1), weights=CROSSING_COUNT_WEIGHTS)[0]
    n_colors = random.randint(2, min(MAX_COLORS, TOTAL_COMPONENTS[1]))
    colors = random.sample(LINE_PALETTE, n_colors)
    dominant = random.choice(colors)
    logical: dict[int, set[tuple[int, int]]] = {c: set() for c in colors}
    paths: dict[int, list[set[tuple[int, int]]]] = {c: [] for c in colors}

    built = _serpentine_path(desired_crossings)
    if built is None:
        return None
    dom_cells, forced_crossings = built
    if not _can_add(logical, dominant, dom_cells):
        return None
    logical[dominant].update(dom_cells)
    paths[dominant].append(set(dom_cells))

    crossing_color = random.choice([c for c in colors if c != dominant])
    cross_cells = _path_crossing_all(forced_crossings)
    if cross_cells is None or not _can_add(logical, crossing_color, cross_cells):
        return None
    logical[crossing_color].update(cross_cells)
    paths[crossing_color].append(set(cross_cells))

    for color in [c for c in colors if c not in (dominant, crossing_color)]:
        cells = None
        for _ in range(EXTRA_POLYLINE_RETRIES):
            cp = _find_crossing_point(logical)
            if cp is None:
                break
            trial = _path_through(cp[0], cp[1])
            if trial is not None and _can_add(logical, color, trial):
                cells = trial
                break
        if cells is None:
            return None
        logical[color].update(cells)
        paths[color].append(set(cells))

    if not _valid_crossings(logical, dominant):
        return None
    if not _attach_branches_to_dominant(logical, paths, dominant):
        return None
    if not any(_junction_cells(comp) for comp in paths[dominant]):
        return None
    return logical, paths, dominant


def _render_target(
    logical: dict[int, set[tuple[int, int]]],
    comp_count: dict[int, int],
    yellow_rays: dict[int, set[tuple[int, int]]],
    marker_pts: dict[int, set[tuple[int, int]]],
    inp: np.ndarray,
) -> np.ndarray:
    return _resolve_claims(inp, BG, set(logical), logical, marker_pts, comp_count, yellow_rays)


def _marker_isolation_ok(grid: np.ndarray) -> bool:
    for color in (2, 4):
        pts = list(map(tuple, np.argwhere(grid == color)))
        for pt in pts:
            if any(grid[nb] == color for nb in _neighbors(pt) if 0 <= nb[0] < H and 0 <= nb[1] < W):
                return False
    return True


def _grid_junction_count(grid: np.ndarray, colors: set[int]) -> int:
    total = 0
    for color in colors:
        pts = _pts(grid == color)
        total += sum(1 for pt in pts if _degree(pts, pt) >= 3)
    return total


def _derive_input(
    logical: dict[int, set[tuple[int, int]]],
    paths: dict[int, list[set[tuple[int, int]]]],
    dominant: int,
) -> tuple[np.ndarray, np.ndarray] | None:
    crossings = set(_crossing_cells(logical))
    mode = random.choices(["red", "both"], weights=MARKER_MODE_WEIGHTS)[0]
    yellow_count = 1 if mode == "both" else 0
    red_count = random.randint(*RED_COUNT_RANGE)

    yellow_infos = []
    blocked = set(crossings)
    if yellow_count:
        special = _build_special_yellow(logical, paths, dominant)
        if special is None:
            return None
        yellow_infos = [special]
        color, pivot, out_dir, target_ray, old_arm = special
        marker = (pivot[0] - out_dir[0], pivot[1] - out_dir[1])
        crossings = set(_crossing_cells(logical))
        blocked = set(crossings)
        blocked.update(target_ray | old_arm | {marker})
        if len(yellow_infos) != yellow_count:
            return None

    red_markers = _choose_red_markers(logical, paths, blocked, red_count, require_junction=True)
    if not red_markers:
        return None

    crossing_display: dict[tuple[int, int], int] = {}
    for pt in crossings:
        colors_here = [color for color in logical if pt in logical[color]]
        if dominant not in colors_here or len(colors_here) != 2:
            return None
        crossing_display[pt] = colors_here[0] if colors_here[1] == dominant else colors_here[1]

    inp = np.full((H, W), BG, dtype=int)
    comp_count_masks: dict[int, set[tuple[int, int]]] = {c: set() for c in logical}
    removed_for_yellow: dict[int, set[tuple[int, int]]] = {c: set() for c in logical}
    old_arms: dict[int, set[tuple[int, int]]] = {c: set() for c in logical}
    yellow_rays: dict[int, set[tuple[int, int]]] = {c: set() for c in logical}
    marker_pts: dict[int, set[tuple[int, int]]] = {c: set() for c in logical}

    yellow_markers: list[tuple[int, tuple[int, int]]] = []
    for color, pivot, out_dir, target_ray, old_arm in yellow_infos:
        marker = (pivot[0] - out_dir[0], pivot[1] - out_dir[1])
        removed_for_yellow[color].update(target_ray - {pivot})
        old_arms[color].update(old_arm)
        yellow_rays[color].update(target_ray)
        yellow_markers.append((color, marker))
        marker_pts[color].add(marker)

    red_by_pt = {pt: color for color, pt in red_markers}
    yellow_by_pt = {pt: color for color, pt in yellow_markers}
    for pt, color in red_by_pt.items():
        marker_pts[color].add(pt)

    for color, pts in logical.items():
        for pt in pts:
            if pt in removed_for_yellow[color]:
                continue
            if pt in crossings and crossing_display[pt] != color:
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

    for pt, color in red_by_pt.items():
        inp[pt] = 2
        comp_count_masks[color].add(pt)
    for pt, color in yellow_by_pt.items():
        inp[pt] = 4
        comp_count_masks[color].add(pt)

    if not _marker_isolation_ok(inp):
        return None
    for pt, color in yellow_by_pt.items():
        if not any(0 <= nb[0] < H and 0 <= nb[1] < W and inp[nb] == color for nb in _neighbors(pt)):
            return None

    comp_count = {color: _gen_component_count(pts) for color, pts in comp_count_masks.items()}
    target = _render_target(logical, comp_count, yellow_rays, marker_pts, inp)

    for pt in crossings:
        if target[pt] == inp[pt]:
            return None

    colors = set(logical)
    counts = {c: len(components_mask(target == c, connectivity=4)) for c in colors}
    if not all(COMPONENTS_PER_COLOR[0] <= v <= COMPONENTS_PER_COLOR[1] for v in counts.values()):
        return None
    if not TOTAL_COMPONENTS[0] <= sum(counts.values()) <= TOTAL_COMPONENTS[1]:
        return None
    fill = int(np.count_nonzero(target))
    if not FILL_RANGE[0] <= fill <= FILL_RANGE[1]:
        return None
    if _grid_junction_count(target, colors) < 1:
        return None

    try:
        solved = main(inp)
    except Exception:
        return None
    return (inp, target) if np.array_equal(solved, target) else None


def _gen_attempt() -> tuple[np.ndarray, np.ndarray] | None:
    built = _build_logical()
    if built is None:
        return None
    logical, paths, dominant = built
    return _derive_input(logical, paths, dominant)


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        result = _gen_attempt()
        if result is not None:
            return result[0]
    raise RuntimeError("failed to generate a valid faa9f03d D9 input")
