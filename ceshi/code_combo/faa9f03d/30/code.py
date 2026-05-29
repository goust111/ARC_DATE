# Combo variant of faa9f03d (index 30)
# combo_spec: D2/0+D9/1+D12/1
# Name: 对角线段 + T形分叉 + 黄色控制90度转弯

from __future__ import annotations

import random

import numpy as np

from arc_utils import components_mask, get_bg


# Design analysis
# yaml:
#   variant_focus: diagonal_segments + same_color_T_junctions + yellow_90deg_turn
#   concrete_shape:
#     output_grid: 12x12 with 1-cell-wide colored networks drawn on the 8-neighbor lattice.
#     always_present:
#       - one dominant color forms an orthogonal T-shaped tree whose hub is repaired by a red marker
#       - one second color crosses that tree at a non-hub cell, so crossing recoloring still matters
#       - one third color forms a diagonal V-corner; its missing arm is restored by a yellow control that turns
#         90 degrees in diagonal space, and another red marker can lie on the surviving diagonal arm
#     allowed_geometry:
#       - straight spans may be horizontal, vertical, or 45-degree diagonal
#       - same-color degree 3 junctions are allowed, but only in visually simple T layouts
#       - yellow pivots are true endpoints in the input and become corners in the output
#   behavior_trace:
#     case:
#       target:
#         - color A is a full horizontal row plus a vertical stem, making a visible T hub
#         - color B is a full vertical line crossing A away from the hub
#         - color C reaches a pivot along NW-SE, then turns into the perpendicular NE-SW diagonal
#       input_original_logic:
#         - original faa9f03d only sees row/column rays, forbids same-color degree 3, and sends yellow straight
#           through its marker cell
#       input_combo_logic:
#         - the red hub on A must be assigned using "three visible arms" evidence, then both the row and column
#           gaps are refilled
#         - the red diagonal marker on C must be repaired along a 45-degree line
#         - the yellow marker next to C does not become output paint; it identifies a diagonal endpoint that must
#           turn onto the nearer perpendicular diagonal border ray
#       difference:
#         - line-of-sight, support tests, and gap filling all have to work on four line families instead of two
#         - same-color trees are now valid, so deletion/fill logic can no longer assume degree <= 2
#         - yellow control is visibly distinct from the base task because the output never passes through the
#           yellow pixel and the new arm is perpendicular to the surviving diagonal arm
#   complexity_calibration:
#     - keep only one explicit same-color T hub and one explicit yellow-controlled diagonal corner per sample
#     - keep crossings simple: exactly one orthogonal crossing pair is enough to preserve the base-task flavor
#     - allow diagonal geometry, but restrict it to straight 45-degree rays rather than arbitrary 8-neighbor mazes
#   implicit_assumptions_audited:
#     marker_detection:
#       original_assumption: 4-connected foreground components identify line colors
#       combo_issue: diagonal lines would fragment under 4-connectivity
#       fix: treat logical line structure with 8-connectivity, while still keeping markers as isolated singletons
#     marker_assignment:
#       original_assumption: nearest cardinal hit plus same-axis support is enough
#       combo_issue: diagonal repair sites and T hubs need richer local evidence
#       fix: score candidates by distance, visible opposite pairs, visible arm count, line support, and span
#     gap_fill:
#       original_assumption: only rows and columns can contain intended gaps
#       combo_issue: diagonal missing cells must be restored too
#       fix: run the fill pass on horizontal, vertical, main-diagonal, and anti-diagonal families
#     yellow_transform:
#       original_assumption: yellow sits on the future painted ray
#       combo_issue: D12/1 changes yellow into a turn cue only
#       fix: exclude yellow cells from the paint mask and choose the nearer of the two perpendicular directions
#     validation:
#       original_assumption: boundary leaves are orthogonal and components are paths
#       combo_issue: diagonal endpoints and degree-3 hubs are both legal now
#       fix: only enforce thinness, degree bounds, and "interior leaves must be explained by a crossing"
#   placement_legality:
#     - the diagonal V is kept far from the T row and the crossing column, so marker assignment remains unique
#     - the yellow marker is in-bounds, isolated, and adjacent only to its intended pivot color in line-of-sight
#     - red markers are separated from one another and from yellow, so all marker colors stay singleton-valued
#   visual_observability:
#     - every generated sample visibly contains: a diagonal segment, a same-color T hub, and a yellow-controlled
#       corner whose output ray avoids the yellow cell
#     - the crossing winner is determined by visible fragmentation: the T color is split by its hub marker
#     - no rule relies on hidden palette ordering except as a last-resort tie-break that generation avoids
#   cleanliness:
#     - only three line colors are used
#     - the diagonal object is kept in a dedicated upper band while the orthogonal crossing lives lower down
#     - the new behaviors are separated enough to be readable but still coexist in one puzzle
#   generator_guarantees_for_solver:
#     - every marker has a unique best visible owner under the score used below
#     - every yellow pivot is an endpoint with exactly one surviving same-color neighbor in the keep direction
#     - the intended yellow turn direction is strictly nearer to a boundary than its perpendicular alternative
#     - all diagonal gaps lie on a single 45-degree line family and all same-color junctions are thin T nodes


H = W = 12
BG = 0
LINE_PALETTE = [1, 3, 5, 6, 7, 8, 9]

DIRS4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
DIRS8 = (
    (-1, 0), (1, 0), (0, -1), (0, 1),
    (-1, -1), (-1, 1), (1, -1), (1, 1),
)
CANON_DIRS = ((0, 1), (1, 0), (1, 1), (1, -1))

GEN_RETRIES = 400


def _pts(mask: np.ndarray) -> set[tuple[int, int]]:
    return {tuple(map(int, rc)) for rc in np.argwhere(mask)}


def _in_bounds(r: int, c: int) -> bool:
    return 0 <= r < H and 0 <= c < W


def _on_boundary(r: int, c: int) -> bool:
    return r == 0 or r == H - 1 or c == 0 or c == W - 1


def _neighbors8(pt: tuple[int, int]) -> list[tuple[int, int]]:
    r, c = pt
    return [(r + dr, c + dc) for dr, dc in DIRS8 if _in_bounds(r + dr, c + dc)]


def _dirs_in(pts: set[tuple[int, int]], pt: tuple[int, int]) -> list[tuple[int, int]]:
    r, c = pt
    return [(dr, dc) for dr, dc in DIRS8 if (r + dr, c + dc) in pts]


def _component_count(shape: tuple[int, int], pts: set[tuple[int, int]]) -> int:
    mask = np.zeros(shape, dtype=bool)
    for r, c in pts:
        mask[r, c] = True
    return len(components_mask(mask, connectivity=4))


def _component_count8(shape: tuple[int, int], pts: set[tuple[int, int]]) -> int:
    mask = np.zeros(shape, dtype=bool)
    for r, c in pts:
        mask[r, c] = True
    return len(components_mask(mask, connectivity=8))


def _degree(pts: set[tuple[int, int]], pt: tuple[int, int]) -> int:
    return len(_dirs_in(pts, pt))


def _canonical(direction: tuple[int, int]) -> tuple[int, int]:
    dr, dc = direction
    if (dr, dc) in CANON_DIRS:
        return (dr, dc)
    return (-dr, -dc)


def _line_key(pt: tuple[int, int], direction: tuple[int, int]) -> tuple[str, int]:
    r, c = pt
    d = _canonical(direction)
    if d == (0, 1):
        return ("row", r)
    if d == (1, 0):
        return ("col", c)
    if d == (1, 1):
        return ("diag", r - c)
    assert d == (1, -1)
    return ("anti", r + c)


def _line_t(pt: tuple[int, int], direction: tuple[int, int]) -> int:
    d = _canonical(direction)
    r, c = pt
    return c if d == (0, 1) else r


def _support_along(pts: set[tuple[int, int]], pt: tuple[int, int], direction: tuple[int, int]) -> bool:
    dr, dc = _canonical(direction)
    r, c = pt
    return (r + dr, c + dc) in pts or (r - dr, c - dc) in pts


def _offaxis_support(pts: set[tuple[int, int]], pt: tuple[int, int], direction: tuple[int, int]) -> bool:
    d = _canonical(direction)
    nd = (-d[0], -d[1])
    return any(vec not in (d, nd) for vec in _dirs_in(pts, pt))


def _span_len(pts: set[tuple[int, int]], pt: tuple[int, int], direction: tuple[int, int]) -> int:
    dr, dc = _canonical(direction)
    total = 1
    r, c = pt
    rr, cc = r + dr, c + dc
    while (rr, cc) in pts:
        total += 1
        rr += dr
        cc += dc
    rr, cc = r - dr, c - dc
    while (rr, cc) in pts:
        total += 1
        rr -= dr
        cc -= dc
    return total


def _opposite_pairs(directions: list[tuple[int, int, int]]) -> int:
    seen = {(dr, dc) for dr, dc, _dist in directions}
    return sum(1 for dr, dc in CANON_DIRS if (dr, dc) in seen and (-dr, -dc) in seen)


def _visible_directions(
    grid: np.ndarray,
    bg: int,
    marker_colors: set[int],
    line_color: int,
    pt: tuple[int, int],
) -> list[tuple[int, int, int]]:
    out: list[tuple[int, int, int]] = []
    r0, c0 = pt
    for dr, dc in DIRS8:
        r, c = r0 + dr, c0 + dc
        dist = 1
        while _in_bounds(r, c):
            value = int(grid[r, c])
            if value == line_color:
                out.append((dr, dc, dist))
                break
            if value == bg or value in marker_colors:
                r += dr
                c += dc
                dist += 1
                continue
            break
    return out


def _marker_completion_score(
    grid: np.ndarray,
    bg: int,
    marker_colors: set[int],
    color: int,
    pt: tuple[int, int],
    original_pts: set[tuple[int, int]],
) -> tuple[int, int, int]:
    dirs = _visible_directions(grid, bg, marker_colors, color, pt, )
    best_span = 0
    for dr, dc, dist in dirs:
        hit = (pt[0] + dr * dist, pt[1] + dc * dist)
        best_span = max(best_span, _span_len(original_pts, hit, (dr, dc)))
    return _opposite_pairs(dirs), len(dirs), best_span


def _pick_unique_min(
    candidates: list[tuple[tuple[int, ...], int, tuple[int, int]]],
    label: str,
) -> tuple[int, tuple[int, int]]:
    assert candidates, label
    best_key = min(key for key, _color, _direction in candidates)
    best = [(color, direction) for key, color, direction in candidates if key == best_key]
    colors = {color for color, _direction in best}
    assert len(colors) == 1, f"ambiguous {label}: {best}"
    return best[0]


def _pick_best_color(scored: list[tuple[int, tuple[int, ...]]], label: str) -> int:
    assert scored, label
    best_key = max(score for _color, score in scored)
    best = [color for color, score in scored if score == best_key]
    assert best, label
    return min(best)


def _marker_brackets(marker_pts: dict[int, set[tuple[int, int]]], color: int, pt: tuple[int, int]) -> bool:
    for direction in CANON_DIRS:
        key = _line_key(pt, direction)
        t = _line_t(pt, direction)
        ts = [_line_t(mpt, direction) for mpt in marker_pts[color] if _line_key(mpt, direction) == key]
        if any(v < t for v in ts) and any(v > t for v in ts):
            return True
    return False


def _claim_score(
    masks: dict[int, set[tuple[int, int]]],
    marker_pts: dict[int, set[tuple[int, int]]],
    comp_count: dict[int, int],
    color: int,
    pt: tuple[int, int],
) -> tuple[int, int, int, int]:
    spans = sorted((_span_len(masks[color], pt, d) for d in CANON_DIRS), reverse=True)
    return (
        comp_count[color],
        1 if _marker_brackets(marker_pts, color, pt) else 0,
        spans[0],
        spans[1],
    )


def _turn_choices(direction: tuple[int, int]) -> tuple[tuple[int, int], tuple[int, int]]:
    choices = [vec for vec in DIRS8 if vec[0] * direction[0] + vec[1] * direction[1] == 0]
    assert len(choices) == 2, f"bad turn choices for {direction}: {choices}"
    return choices[0], choices[1]


def _steps_to_boundary(shape: tuple[int, int], pivot: tuple[int, int], direction: tuple[int, int]) -> int:
    h, w = shape
    r, c = pivot
    dr, dc = direction
    steps = 0
    while 0 <= r + dr < h and 0 <= c + dc < w:
        r += dr
        c += dc
        steps += 1
    return steps


def _ray_cells(start: tuple[int, int], direction: tuple[int, int]) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    r, c = start
    dr, dc = direction
    while _in_bounds(r, c):
        out.append((r, c))
        r += dr
        c += dc
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Repair sparse one-cell-wide colored networks by recoloring singleton markers, bridging gaps on orthogonal and 45-degree axes, and turning yellow-controlled endpoints by 90 degrees.

    Input structure:
    - The grid has one dominant background and several non-marker colors forming thin horizontal, vertical, and diagonal stroke fragments.
    - Same-color networks may now include T-shaped degree-3 junctions, so a component is not restricted to a simple path.
    - Marker colors are exactly the foreground colors whose connected components are isolated single pixels.
    - Red singletons stand for missing line cells, while yellow singletons sit just beyond a visible endpoint and act as turn controls rather than painted output cells.

    Rule:
    - Assign each marker to the unique visible line color that best completes a local straight-line pattern, scoring by distance, visible opposite pairs, visible arm count, support, and span across all eight directions.
    - Recolor red markers as line cells, then fill gaps between compatible same-color points on rows, columns, and both diagonal families; this can restore diagonal spans and same-color T hubs.
    - A yellow control keeps the endpoint arm that reaches its pivot, ignores the yellow cell itself, and adds a new ray from the pivot in whichever perpendicular direction reaches a boundary sooner.
    - At overlaps, yellow-generated rays win; otherwise the crossing color with the more fragmented pre-repair structure claims the cell, with marker bracketing and local span used as visible tie-break evidence.

    Key insight: The puzzle is still about using singleton markers to reconstruct hidden continuations, but the continuations are no longer limited to rectilinear paths. The solver must recognize straightness on four line families, allow same-color branching at hubs, and interpret yellow as a 90-degree turn cue rather than a paint-through extension.
    """
    g = np.asarray(grid, dtype=int)
    bg = int(get_bg(g))
    h, w = g.shape

    colors = set(map(int, g.ravel())) - {bg}
    assert colors, "expected foreground line and marker colors"

    marker_colors = {color for color in colors if color in (2, 4)}
    line_colors = colors - marker_colors
    assert line_colors, "expected at least one non-marker line color"
    assert marker_colors, "expected singleton marker colors"

    original = {color: _pts(g == color) for color in line_colors}
    assignment: dict[tuple[int, int], int] = {}
    assign_dir: dict[tuple[int, int], tuple[int, int]] = {}

    marker_mask = np.isin(g, list(marker_colors))
    for r0, c0 in _pts(marker_mask):
        candidates: list[tuple[tuple[int, ...], int, tuple[int, int]]] = []
        for dr, dc in DIRS8:
            r, c = r0 + dr, c0 + dc
            dist = 1
            while 0 <= r < h and 0 <= c < w:
                value = int(g[r, c])
                if value in line_colors:
                    supported = _support_along(original[value], (r, c), (dr, dc))
                    pairs, vis_dirs, best_span = _marker_completion_score(
                        g, bg, marker_colors, value, (r0, c0), original[value]
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

    yellow_controls = _pts(g == 4) if 4 in marker_colors else set()
    masks = {color: set(pts) for color, pts in original.items()}
    marker_pts = {color: set() for color in line_colors}
    yellow_marker_pts = {color: set() for color in line_colors}
    for pt, color in assignment.items():
        if pt in yellow_controls:
            yellow_marker_pts[color].add(pt)
        else:
            masks[color].add(pt)
            marker_pts[color].add(pt)

    comp_count = {
        color: _component_count(g.shape, masks[color] | yellow_marker_pts[color])
        for color in line_colors
    }

    def has_line_support(color: int, pt: tuple[int, int], direction: tuple[int, int]) -> bool:
        return pt in marker_pts[color] or _support_along(masks[color], pt, direction)

    def should_fill(color: int, a: tuple[int, int], b: tuple[int, int], direction: tuple[int, int]) -> bool:
        gap = _line_t(b, direction) - _line_t(a, direction) - 1
        if gap <= 0:
            return False
        a_axis = has_line_support(color, a, direction)
        b_axis = has_line_support(color, b, direction)
        if gap == 1:
            if a_axis or b_axis:
                return True
            return not (_offaxis_support(masks[color], a, direction) and _offaxis_support(masks[color], b, direction))
        if a in marker_pts[color] and b in marker_pts[color]:
            return False
        return a_axis and b_axis

    changed = True
    while changed:
        changed = False
        for color in sorted(line_colors):
            additions: set[tuple[int, int]] = set()
            for direction in CANON_DIRS:
                groups: dict[tuple[str, int], list[tuple[int, int]]] = {}
                for pt in masks[color]:
                    groups.setdefault(_line_key(pt, direction), []).append(pt)
                for pts_on_line in groups.values():
                    pts_on_line.sort(key=lambda pt: _line_t(pt, direction))
                    for a, b in zip(pts_on_line, pts_on_line[1:]):
                        ta = _line_t(a, direction)
                        tb = _line_t(b, direction)
                        if tb <= ta + 1:
                            continue
                        if not should_fill(color, a, b, direction):
                            continue
                        step = _canonical(direction)
                        rr, cc = a[0] + step[0], a[1] + step[1]
                        while (rr, cc) != b:
                            additions.add((rr, cc))
                            rr += step[0]
                            cc += step[1]
            if additions - masks[color]:
                masks[color].update(additions)
                changed = True

    control_ray = {color: set() for color in line_colors}
    if 4 in marker_colors:
        for control in yellow_controls:
            color = assignment[control]
            keep_dir = assign_dir[control]

            r, c = control[0] + keep_dir[0], control[1] + keep_dir[1]
            pivot: tuple[int, int] | None = None
            while 0 <= r < h and 0 <= c < w:
                if (r, c) in masks[color]:
                    pivot = (r, c)
                    break
                r += keep_dir[0]
                c += keep_dir[1]
            assert pivot is not None, f"yellow control at {control} has no assigned pivot"

            neighbors = _dirs_in(masks[color], pivot)
            assert len(neighbors) == 1 and neighbors[0] == keep_dir, (
                f"yellow control at {control} expected endpoint aligned with {keep_dir}, got {neighbors}"
            )

            choices = sorted(
                (_steps_to_boundary(g.shape, pivot, turn_dir), turn_dir)
                for turn_dir in _turn_choices(keep_dir)
            )
            assert choices[0][0] > 0, f"yellow control at {control} cannot extend to boundary"
            assert choices[0][0] != choices[1][0], f"yellow control at {control} has tied turn distances"
            turn_dir = choices[0][1]
            for pt in _ray_cells(pivot, turn_dir):
                masks[color].add(pt)
                control_ray[color].add(pt)

    claims: dict[tuple[int, int], list[int]] = {}
    for color, pts in masks.items():
        for pt in pts:
            claims.setdefault(pt, []).append(int(color))

    out = np.full_like(g, bg)
    for pt, colors_here in claims.items():
        forced = [color for color in colors_here if pt in control_ray[color]]
        if forced:
            out[pt] = _pick_best_color(
                [(color, _claim_score(masks, marker_pts, comp_count, color, pt)) for color in forced],
                f"yellow claim at {pt}",
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
        out[pt] = _pick_best_color(
            [(color, _claim_score(masks, marker_pts, comp_count, color, pt)) for color in pool],
            f"crossing claim at {pt}",
        )

    for color in line_colors:
        pts_set = _pts(out == color)
        for r in range(h - 1):
            for c in range(w - 1):
                block = {(r, c), (r + 1, c), (r, c + 1), (r + 1, c + 1)}
                assert not block.issubset(pts_set), f"2x2 thickness at {(r, c)} color {color}"

    return out


def _segment(a: tuple[int, int], b: tuple[int, int]) -> list[tuple[int, int]]:
    r0, c0 = a
    r1, c1 = b
    dr = r1 - r0
    dc = c1 - c0
    if dr == 0 and dc == 0:
        return [a]
    if dr != 0 and dc != 0:
        assert abs(dr) == abs(dc), f"non-45-degree diagonal segment: {a}->{b}"
    else:
        assert dr == 0 or dc == 0, f"non-straight segment: {a}->{b}"
    sr = 0 if dr == 0 else (1 if dr > 0 else -1)
    sc = 0 if dc == 0 else (1 if dc > 0 else -1)
    steps = max(abs(dr), abs(dc))
    return [(r0 + i * sr, c0 + i * sc) for i in range(steps + 1)]


def _path(points: list[tuple[int, int]]) -> list[tuple[int, int]]:
    cells: list[tuple[int, int]] = []
    for a, b in zip(points, points[1:]):
        seg = _segment(a, b)
        cells.extend(seg[1:] if cells else seg)
    return cells


def _red_sites_on_diag(ray: list[tuple[int, int]], pivot: tuple[int, int]) -> list[tuple[int, int]]:
    return [pt for pt in ray if pt != pivot and _on_boundary(*pt) is False]


def _valid_thin_network(pts: set[tuple[int, int]]) -> bool:
    if not pts or _component_count8((H, W), pts) != 1:
        return False
    for r in range(H - 1):
        for c in range(W - 1):
            if {(r, c), (r + 1, c), (r, c + 1), (r + 1, c + 1)}.issubset(pts):
                return False
    return True


def _target_grid(
    logical: dict[int, set[tuple[int, int]]],
    dominant: int,
    crossing_pt: tuple[int, int],
) -> np.ndarray:
    out = np.full((H, W), BG, dtype=int)
    for color, pts in logical.items():
        for pt in pts:
            if pt == crossing_pt:
                continue
            out[pt] = color
    out[crossing_pt] = dominant
    return out


def _marker_isolation_ok(grid: np.ndarray) -> bool:
    for color in (2, 4):
        pts = list(map(tuple, np.argwhere(grid == color)))
        for pt in pts:
            for nb in _neighbors8(pt):
                if nb != pt and grid[nb] == color:
                    return False
    return True


def _choose_diagonal_spec(occupied: set[tuple[int, int]], blocked_cols: set[int]) -> tuple[
    tuple[int, int], tuple[int, int], tuple[int, int], list[tuple[int, int]], list[tuple[int, int]], tuple[int, int]
] | None:
    keep = (-1, -1)
    turn = (-1, 1)
    for _ in range(120):
        pr = random.randint(3, 4)
        pc = random.randint(3, 7)
        pivot = (pr, pc)
        marker = (pr + 1, pc + 1)
        if not _in_bounds(*marker):
            continue
        keep_ray = _ray_cells(pivot, keep)
        turn_ray = _ray_cells(pivot, turn)
        if len(keep_ray) < 4 or len(turn_ray) < 4:
            continue
        opp_turn = (1, -1)
        if _steps_to_boundary((H, W), pivot, turn) >= _steps_to_boundary((H, W), pivot, opp_turn):
            continue
        c_pts = set(keep_ray) | set(turn_ray)
        if any(pt in occupied for pt in c_pts):
            continue
        if any(pt[1] in blocked_cols for pt in c_pts):
            continue
        expanded = set(c_pts)
        for pt in c_pts:
            expanded.update(_neighbors8(pt))
        if expanded & occupied:
            continue
        if marker in c_pts or marker in occupied:
            continue
        return keep, turn, pivot, keep_ray, turn_ray, marker
    return None


def _build_case() -> tuple[np.ndarray, np.ndarray] | None:
    a, b, c = random.sample(LINE_PALETTE, 3)
    dominant = a

    row_a = random.randint(6, 8)
    hub_col = random.randint(2, 4)
    cross_col_choices = [v for v in range(7, 10) if v != hub_col]
    cross_col = random.choice(cross_col_choices)

    stem_from_top = random.random() < 0.5
    a_pts = {(row_a, cc) for cc in range(W)}
    if stem_from_top:
        a_pts |= {(rr, hub_col) for rr in range(0, row_a + 1)}
    else:
        a_pts |= {(rr, hub_col) for rr in range(row_a, H)}

    b_pts = {(rr, cross_col) for rr in range(H)}
    crossing_pt = (row_a, cross_col)

    occupied = set(a_pts) | set(b_pts)
    diag_spec = _choose_diagonal_spec(occupied, {cross_col - 1, cross_col, cross_col + 1})
    if diag_spec is None:
        return None
    keep_dir, turn_dir, pivot, keep_ray, turn_ray, yellow_marker = diag_spec
    c_pts = set(keep_ray) | set(turn_ray)

    logical = {a: set(a_pts), b: set(b_pts), c: set(c_pts)}
    if not all(_valid_thin_network(logical[color]) for color in logical):
        return None

    target = _target_grid(logical, dominant, crossing_pt)

    red_a = (row_a, hub_col)
    red_c_choices = [pt for pt in keep_ray[2:-1] if pt != pivot]
    if not red_c_choices:
        return None
    red_c = random.choice(red_c_choices)

    marker_pts = [red_a, red_c, yellow_marker]
    for i, p in enumerate(marker_pts):
        for q in marker_pts[i + 1:]:
            if max(abs(p[0] - q[0]), abs(p[1] - q[1])) <= 1:
                return None

    inp = np.full((H, W), BG, dtype=int)
    removed_for_yellow = set(turn_ray[1:])

    for pt in logical[a]:
        if pt == red_a:
            continue
        if pt == crossing_pt:
            continue
        inp[pt] = a
    for pt in logical[b]:
        inp[pt] = b
    for pt in logical[c]:
        if pt in removed_for_yellow or pt == red_c:
            continue
        inp[pt] = c

    inp[crossing_pt] = b
    inp[red_a] = 2
    inp[red_c] = 2
    inp[yellow_marker] = 4

    if not _marker_isolation_ok(inp):
        return None
    if any(inp[nb] == 4 for nb in _neighbors8(yellow_marker)):
        return None
    if not any(inp[nb] == c for nb in _neighbors8(yellow_marker)):
        return None

    try:
        solved = main(inp)
    except Exception:
        return None
    if not np.array_equal(solved, target):
        return None
    return inp, target


def _gen_attempt() -> np.ndarray | None:
    built = _build_case()
    return built[0] if built is not None else None


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("failed to generate a valid faa9f03d combo input")
