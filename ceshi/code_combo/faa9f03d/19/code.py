# Combo variant of faa9f03d (index 19)
# combo_spec: D1/2+D12/2
# Name: 反射旧臂 + 黄色控制折线反射
#
# Design analysis for this combo
# --------------------------------
# concrete_behavior_and_trace:
#   - Base task: red markers still stand for deleted line cells and ordinary same-color
#     row/column gaps are filled back in. The yellow step is the part that changes.
#   - This combo restricts yellow cases to a visible endpoint pivot P whose surviving side
#     is not a single straight arm but a whole 2-segment boundary-reaching branch.
#   - Example in canonical orientation:
#       visible branch in input: from P go up 3, then left to the border.
#       missing branch in output: from P go right 3, then up to the border.
#       input shows only the visible branch plus a yellow singleton at the first missing
#       cell to the right of P.
#   - Original task would delete the old side and shoot a straight ray through the yellow
#     marker to the boundary. D1/2 alone would copy only the first straight length of the
#     old arm, stopping after 3 cells. This combo instead rotates the entire visible branch
#     around P by 90 degrees onto the yellow side, so the copied branch also turns.
# complexity_calibration:
#   - Yellow is always present, and its reconstructed branch always has at least one turn
#     beyond the first reflected segment. That prevents collapse to either the base task
#     or the D1/2 straight-length-only variant.
#   - The yellow-controlled component is a simple 2-branch polyline, so the solver has one
#     stable interpretation per sample rather than many competing reflection choices.
# implicit_assumptions_audit:
#   - Marker assignment by nearest axial line-of-sight still works, because the yellow
#     marker is adjacent to its pivot and farther cells on the marker side are removed.
#   - Yellow markers can no longer be inserted into the fill mask as ordinary line cells;
#     doing so would erase the distinction between "marker as control" and "marker as data".
#   - The yellow repair now assumes the assigned pivot is an input endpoint with exactly one
#     same-color neighbor and that neighbor is perpendicular to the missing target side.
#     The generator guarantees this and blocks red markers from the whole yellow footprint.
#   - Rotating the whole visible branch would create self-intersections if nearby geometry
#     were too dense. The generator therefore reserves a clean footprint around the special
#     component and rejects placements whose rotated copy would collide or leave the grid.
# placement_legality:
#   - The yellow component is built output-first as a valid one-cell-thick boundary-reaching
#     polyline. Input derivation removes exactly one whole branch from its pivot onward and
#     replaces the first removed cell with the yellow marker.
#   - Base crossing colors are generated first; the yellow component is then added only when
#     it stays disjoint from existing geometry, so no hidden overlap cases are introduced.
# visual_observability_and_cleanliness:
#   - Across examples, humans can compare the visible branch and the restored yellow-side
#     branch and notice a repeated 90-degree whole-branch rotation, not a straight edge ray.
#   - No hidden color-id rule decides the new shape. Geometry alone determines the repair.
# combo_metric_contract:
#   - Every generated yellow case has a reflected target branch with at least one turn.
#   - Therefore `main(input)` must differ from both the base solver and the D1/2 solver on
#     the same input by more than just recoloring the yellow pixel itself.

from __future__ import annotations

import random

import numpy as np

from arc_utils import components_mask, get_bg


DIRS4 = ((-1, 0), (1, 0), (0, -1), (0, 1))


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
    mask = np.zeros(shape, dtype=bool)
    for r, c in pts:
        mask[r, c] = True
    return len(components_mask(mask, connectivity=4))


def _neighbors(pt: tuple[int, int]) -> list[tuple[int, int]]:
    r, c = pt
    return [(r + dr, c + dc) for dr, dc in DIRS4 if 0 <= r + dr < H and 0 <= c + dc < W]


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


def _gen_component_count(pts: set[tuple[int, int]]) -> int:
    if not pts:
        return 0
    mask = np.zeros((H, W), dtype=bool)
    for r, c in pts:
        mask[r, c] = True
    return len(components_mask(mask, connectivity=4))


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
    if _gen_component_count(pts) != 1:
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
        if not (
            (r == 0 and dr == 1)
            or (r == H - 1 and dr == -1)
            or (c == 0 and dc == 1)
            or (c == W - 1 and dc == -1)
        ):
            return False
    for r, c in pts:
        if len(_dirs_in(pts, (r, c))) == 1 and not _on_boundary(r, c):
            return False
    return True


def _rot_cw(vec: tuple[int, int]) -> tuple[int, int]:
    r, c = vec
    return (c, -r)


def _rot_ccw(vec: tuple[int, int]) -> tuple[int, int]:
    r, c = vec
    return (-c, r)


def _rotation_to_map(
    source_dir: tuple[int, int],
    target_dir: tuple[int, int],
):
    for rot in (_rot_cw, _rot_ccw):
        if rot(source_dir) == target_dir:
            return rot
    raise AssertionError(f"cannot rotate {source_dir} onto {target_dir}")


def _trace_branch(
    pts: set[tuple[int, int]],
    pivot: tuple[int, int],
    first_dir: tuple[int, int],
) -> list[tuple[int, int]]:
    cur = (pivot[0] + first_dir[0], pivot[1] + first_dir[1])
    prev = pivot
    out: list[tuple[int, int]] = []
    while cur in pts:
        out.append(cur)
        nxt = [nb for nb in _neighbors(cur) if nb in pts and nb != prev]
        if not nxt:
            break
        assert len(nxt) == 1, f"branch at {cur} is not a simple path"
        prev, cur = cur, nxt[0]
    return out


def _rotate_branch(
    branch: list[tuple[int, int]],
    pivot: tuple[int, int],
    source_dir: tuple[int, int],
    target_dir: tuple[int, int],
) -> list[tuple[int, int]]:
    rot = _rotation_to_map(source_dir, target_dir)
    pr, pc = pivot
    out: list[tuple[int, int]] = []
    for r, c in branch:
        dr, dc = r - pr, c - pc
        rr, cc = rot((dr, dc))
        out.append((pr + rr, pc + cc))
    return out


def _transform_rot(pt: tuple[int, int], rot_k: int) -> tuple[int, int]:
    r, c = pt
    if rot_k == 0:
        return (r, c)
    if rot_k == 1:
        return (c, W - 1 - r)
    if rot_k == 2:
        return (H - 1 - r, W - 1 - c)
    if rot_k == 3:
        return (H - 1 - c, r)
    raise AssertionError(rot_k)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Repair a one-cell-thick colored line network by recoloring singleton markers, filling ordinary orthogonal gaps, and rebuilding yellow-controlled branches by whole-branch quarter-turn reflection.

    Input structure:
    - A rectangular grid has one dominant background and several non-marker colors that form thin axis-aligned stroke fragments.
    - Foreground colors whose 4-connected components are all singleton pixels are marker colors.
    - Red singleton markers stand for missing line cells; yellow singleton markers sit at the first missing cell of an absent branch whose pivot endpoint is still visible.

    Rule:
    - Assign each marker to the nearest visible line color in row/column line-of-sight, preferring hits already supported along that same axis.
    - Recolor non-yellow markers as line cells, then fill ordinary row/column gaps between compatible facing endpoints of the same color.
    - For a yellow control, locate its pivot on the assigned color, trace the one surviving branch attached to that pivot, and rotate that entire ordered branch by 90 degrees so its first step goes through the yellow marker.
    - At overlaps, yellow-generated cells win; otherwise original-vs-filled crossings still prefer the more fragmented pre-repair color, with lower color id as the remaining tie-break.

    Key insight: Yellow no longer means "erase an old arm and shoot a straight ray". It marks a missing branch whose full visible counterpart on the other side of the pivot must be quarter-turned onto the yellow side, including later bends.
    """
    g = np.asarray(grid, dtype=int)
    bg = int(get_bg(g))
    H_local, W_local = g.shape
    assert (H_local, W_local) == (H, W), "generator and solver both assume 12x12"

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
        color: _component_count(g.shape, masks[color] | marker_pts[color] | yellow_marker_pts[color])
        for color in line_colors
    }

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

    control_reflect = {color: set() for color in line_colors}
    if 4 in marker_colors:
        for control in yellow_controls:
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

            nbr_dirs = [
                ndr_dc
                for ndr_dc in DIRS4
                if (pivot[0] + ndr_dc[0], pivot[1] + ndr_dc[1]) in masks[color]
            ]
            assert len(nbr_dirs) == 1, f"yellow pivot {pivot} should be an endpoint in the input"
            source_dir = nbr_dirs[0]
            target_dir = (-dr, -dc)
            assert source_dir[0] * target_dir[0] + source_dir[1] * target_dir[1] == 0, (
                f"yellow pivot {pivot} has non-perpendicular source {source_dir} and target {target_dir}"
            )

            source_branch = _trace_branch(masks[color], pivot, source_dir)
            assert source_branch, f"yellow pivot {pivot} has no traceable source branch"
            rotated_branch = _rotate_branch(source_branch, pivot, source_dir, target_dir)
            assert rotated_branch[0] == control, (
                f"yellow control {control} is not the first rotated branch cell from pivot {pivot}"
            )
            for cell in rotated_branch:
                rr, cc = cell
                assert 0 <= rr < H and 0 <= cc < W, f"rotated branch leaves grid at {cell}"
                masks[color].add(cell)
                control_reflect[color].add(cell)
            control_reflect[color].add(pivot)

    claims: dict[tuple[int, int], list[int]] = {}
    for color, pts in masks.items():
        for pt in pts:
            claims.setdefault(pt, []).append(int(color))

    out = np.full_like(g, bg)

    def marker_brackets(color: int, pt: tuple[int, int]) -> bool:
        r, c = pt
        same_row = [mc for mr, mc in marker_pts[color] if mr == r]
        if any(mc < c for mc in same_row) and any(mc > c for mc in same_row):
            return True
        same_col = [mr for mr, mc in marker_pts[color] if mc == c]
        return any(mr < r for mr in same_col) and any(mr > r for mr in same_col)

    for pt, colors_here in claims.items():
        forced = {color for color in colors_here if pt in control_reflect[color]}
        if forced:
            assert len(forced) == 1, f"ambiguous yellow reflections at {pt}: {forced}"
            out[pt] = next(iter(forced))
            continue
        if len(colors_here) == 1:
            out[pt] = colors_here[0]
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

    for color in line_colors:
        for comp in components_mask(out == color, connectivity=4):
            pts_set = set(map(tuple, comp))
            for r, c in pts_set:
                nbrs = [
                    (nr, nc)
                    for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))
                    if (nr, nc) in pts_set
                ]
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
                adj_other = any(
                    0 <= r + dr < H
                    and 0 <= c + dc < W
                    and int(out[r + dr, c + dc]) not in (bg, color)
                    for dr, dc in DIRS4
                )
                assert adj_other, f"dangling endpoint ({r},{c}) color {color}: not on boundary or crossing"

    return out


# --------------- generator (combo) ---------------

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
MAX_CROSSINGS = 12

MARKER_MODE_WEIGHTS = [0, 3, 6]  # red-only / yellow-only / both
YELLOW_COUNT_RANGE = (1, 1)
RED_COUNT_RANGE = (1, 4)
MIN_CELLS_FOR_RED = 2

COMPONENTS_PER_COLOR = (1, 4)
TOTAL_COMPONENTS = (3, 6)
FILL_RANGE = (30, 58)

SPACED_VALUES_RETRIES = 60
SPECIAL_RETRIES = 80
GEN_RETRIES = 900


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
    for i, bend in enumerate(bends):
        pts.append(_rc(ax, bend, minors[i]))
        pts.append(_rc(ax, bend, minors[i + 1]))
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


def _choose_red_markers(
    logical: dict[int, set[tuple[int, int]]],
    paths: dict[int, list[set[tuple[int, int]]]],
    blocked: set[tuple[int, int]],
    count: int,
) -> list[tuple[int, tuple[int, int]]]:
    candidates = [
        (color, pt)
        for color, comps in paths.items()
        for comp in comps
        for pt in comp
        if pt not in blocked and _is_turn_or_endpoint(comp, pt)
    ]
    random.shuffle(candidates)
    chosen: list[tuple[int, tuple[int, int]]] = []
    used: set[tuple[int, int]] = set(blocked)
    per_color_left = {color: len(logical[color]) for color in logical}
    for color, pt in candidates:
        if len(chosen) == count:
            break
        if per_color_left[color] <= MIN_CELLS_FOR_RED:
            continue
        if any(nb in used for nb in [pt] + _neighbors(pt)):
            continue
        chosen.append((color, pt))
        used.add(pt)
        per_color_left[color] -= 1
    return chosen if len(chosen) == count else []


def _render_target(
    logical: dict[int, set[tuple[int, int]]],
    comp_count: dict[int, int],
    yellow_spans: dict[int, set[tuple[int, int]]],
) -> np.ndarray:
    out = np.full((H, W), BG, dtype=int)
    for pt in set().union(*logical.values()):
        colors_here = [color for color in logical if pt in logical[color]]
        forced = [color for color in colors_here if pt in yellow_spans.get(color, set())]
        if forced:
            out[pt] = forced[0]
        elif len(colors_here) == 1:
            out[pt] = colors_here[0]
        else:
            out[pt] = min(colors_here, key=lambda color: (-comp_count[color], color))
    return out


def _marker_isolation_ok(grid: np.ndarray) -> bool:
    for color in (2, 4):
        for pt in map(tuple, np.argwhere(grid == color)):
            if any(grid[nb] == color for nb in _neighbors(pt)):
                return False
    return True


def _build_special_component() -> dict[str, object] | None:
    for _ in range(SPECIAL_RETRIES):
        k = random.randint(3, 5)
        max_a = min(k - 1, W - 2 - k)
        if max_a < 2:
            continue
        a = random.randint(2, max_a)
        base_points = [(k - a, 0), (k - a, k), (k, k), (k, k + a), (0, k + a)]
        cells = _path(base_points)
        if not _valid_path(cells):
            continue

        pivot = (k, k)
        source_branch = [(k - d, k) for d in range(1, a + 1)] + [(k - a, c) for c in range(k - 1, -1, -1)]
        target_branch = [(k, k + d) for d in range(1, a + 1)] + [(r, k + a) for r in range(k - 1, -1, -1)]
        rot_k = random.randint(0, 3)

        cells_t = [_transform_rot(pt, rot_k) for pt in cells]
        pivot_t = _transform_rot(pivot, rot_k)
        source_t = [_transform_rot(pt, rot_k) for pt in source_branch]
        target_t = [_transform_rot(pt, rot_k) for pt in target_branch]
        marker_t = target_t[0]
        footprint = set(cells_t) | set(_neighbors(marker_t)) | {pivot_t}
        if not _valid_path(cells_t):
            continue
        if len(set(cells_t)) != len(cells_t):
            continue
        return {
            "cells": cells_t,
            "pivot": pivot_t,
            "source_branch": source_t,
            "target_branch": target_t,
            "marker": marker_t,
            "footprint": footprint,
        }
    return None


def _build_logical() -> tuple[
    dict[int, set[tuple[int, int]]],
    dict[int, list[set[tuple[int, int]]]],
    int,
    dict[str, object],
] | None:
    desired_crossings = random.choices(range(1, len(CROSSING_COUNT_WEIGHTS) + 1), weights=CROSSING_COUNT_WEIGHTS)[0]
    colors = random.sample(LINE_PALETTE, 3)
    dominant, crossing_color, special_color = colors
    random.shuffle(colors)
    dominant = colors[0]
    crossing_color = colors[1]
    special_color = colors[2]

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

    cross_cells = _path_crossing_all(forced_crossings)
    if cross_cells is None or not _can_add(logical, crossing_color, cross_cells):
        return None
    logical[crossing_color].update(cross_cells)
    paths[crossing_color].append(set(cross_cells))

    special = _build_special_component()
    if special is None:
        return None
    special_cells = list(special["cells"])
    if not _can_add(logical, special_color, special_cells):
        return None
    logical[special_color].update(special_cells)
    paths[special_color].append(set(special_cells))

    if not _valid_crossings(logical, dominant):
        return None

    special["color"] = special_color
    return logical, paths, dominant, special


def _derive_input(
    logical: dict[int, set[tuple[int, int]]],
    paths: dict[int, list[set[tuple[int, int]]]],
    dominant: int,
    special: dict[str, object],
) -> tuple[np.ndarray, np.ndarray] | None:
    crossings = set(_crossing_cells(logical))
    mode = random.choices(["red", "yellow", "both"], weights=MARKER_MODE_WEIGHTS)[0]
    yellow_count = random.randint(*YELLOW_COUNT_RANGE) if mode in ("yellow", "both") else 0
    red_count = random.randint(*RED_COUNT_RANGE) if mode in ("red", "both") else 0
    assert yellow_count == 1, "combo generator always expects exactly one yellow control"

    special_color = int(special["color"])
    pivot = tuple(special["pivot"])
    target_branch = list(special["target_branch"])
    source_branch = list(special["source_branch"])
    marker = tuple(special["marker"])
    yellow_footprint = set(source_branch) | set(target_branch) | {pivot} | set(_neighbors(marker))

    blocked = set(crossings) | yellow_footprint
    red_markers = _choose_red_markers(logical, paths, blocked, red_count) if red_count else []
    if red_count and not red_markers:
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
    yellow_spans: dict[int, set[tuple[int, int]]] = {c: set() for c in logical}
    removed_for_yellow[special_color].update(target_branch)
    yellow_spans[special_color].update({pivot} | set(target_branch))

    red_by_pt = {pt: color for color, pt in red_markers}
    yellow_by_pt = {marker: special_color}

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

    for pt, color in red_by_pt.items():
        inp[pt] = 2
        comp_count_masks[color].add(pt)
    for pt, color in yellow_by_pt.items():
        inp[pt] = 4
        comp_count_masks[color].add(pt)

    if not _marker_isolation_ok(inp):
        return None
    if not any(inp[nb] == special_color for nb in _neighbors(marker)):
        return None

    comp_count = {color: _gen_component_count(pts) for color, pts in comp_count_masks.items()}
    target = _render_target(logical, comp_count, yellow_spans)
    for pt in crossings:
        if target[pt] == inp[pt]:
            return None

    colors = sorted(logical)
    counts = {c: len(components_mask(target == c, connectivity=4)) for c in colors}
    if not all(COMPONENTS_PER_COLOR[0] <= v <= COMPONENTS_PER_COLOR[1] for v in counts.values()):
        return None
    if not TOTAL_COMPONENTS[0] <= sum(counts.values()) <= TOTAL_COMPONENTS[1]:
        return None
    if not FILL_RANGE[0] <= int(np.count_nonzero(target)) <= FILL_RANGE[1]:
        return None

    try:
        solved = main(inp)
    except Exception:
        return None
    return (inp, target) if np.array_equal(solved, target) else None


def _gen_attempt() -> np.ndarray | None:
    built = _build_logical()
    if built is None:
        return None
    logical, paths, dominant, special = built
    derived = _derive_input(logical, paths, dominant, special)
    return derived[0] if derived is not None else None


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("failed to generate a valid faa9f03d combo input")
