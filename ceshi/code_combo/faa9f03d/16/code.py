# Combo variant of faa9f03d (index 16)
# combo_spec: D9/1+D11/2
# Name: T形分叉节点 + 交叉处显示混合标记色

from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[5]
SOLVER_DIR = ROOT / "solvers"
if str(SOLVER_DIR) not in sys.path:
    sys.path.insert(0, str(SOLVER_DIR))

from arc_utils import components_mask, get_bg


# Design analysis
# yaml:
#   variant_focus: combo_D9_1_plus_D11_2
#   concrete_shape:
#     output_grid: 12x12 with thin orthogonal colored networks, explicit grey crossing cells, and at least one same-color T junction.
#     original_behavior: different-color crossings were shown as one winner color; same-color components were simple paths.
#     fused_behavior:
#       - one dominant color is upgraded from a path to a tree by attaching 1-2 orthogonal branches to a straight trunk cell
#       - every logical different-color crossing is displayed as grey 5 in both input and output, hiding both underlying line colors
#       - red markers may remove a branch hub itself, so one repair can simultaneously reconnect row and column arms
#       - optional yellow markers still redirect one straight arm, but the deleted arm is constrained so sibling branches survive
#   behavior_trace:
#     case:
#       output: green has a visible T junction, while a blue vertical and the green trunk meet at a grey crossing.
#       input_original_logic: a crossing cell would show either blue or green, and same-color repairs would only need path logic.
#       input_fused_logic:
#         - the grey crossing is not owned visibly by either color; its horizontal and vertical owners must be inferred after repair
#         - the red marker on the T hub is assigned by visible arm pattern rather than nearest endpoint alone
#         - if a yellow marker exists, only the straight collinear arm behind the pivot is erased; the perpendicular branch at the hub remains
#       difference:
#         - the solver must distinguish same-color degree-3 junctions from grey different-color crossings
#         - the final grid keeps the crossing grey, so ownership is logical rather than visible
#   complexity_calibration:
#     - reserve grey exclusively for crossings and remove it from the normal line palette
#     - force at least one visible same-color junction and at least one grey crossing in every generated sample
#     - keep only one dominant branched color so the tree topology is legible instead of maze-like
#     - keep red/yellow markers out of the crossing neighborhood so humans can read the two hidden crossing directions from opposite neighbors
#   implicit_assumptions_audited:
#     marker_assignment:
#       original_assumption: singleton-only colors are all markers
#       fused_issue: grey crossing cells are often singleton pixels too, but they are not markers
#       fix: use fixed marker semantics red=2, yellow=4, grey crossing=5, and let marker sight-lines pass through grey
#     crossing_recovery:
#       original_assumption: a visible overlap cell already shows one participating line color
#       fused_issue: grey hides both colors, and same-color T nodes elsewhere mean degree>=3 no longer implies a crossing
#       fix: reconstruct grey ownership only from opposite horizontal/vertical supports after gap filling, while allowing degree 3/4 away from crossings
#     yellow_erase:
#       original_assumption: deleting the far side of the pivot can flood through the whole component
#       fused_issue: that would wrongly destroy a sibling branch at a T hub and could also remove support needed near a grey crossing
#       fix: delete only the straight collinear arm until the first perpendicular support cell
#     output_validation:
#       original_assumption: visible line colors alone determine crossings and all degrees are <= 2
#       fused_issue: grey cells split visible components and same-color degree 3 is legitimate
#       fix: validate thinness and endpoints on logical per-color masks that reinsert grey crossing ownership
#   placement_legality:
#     - branch rays are only attached through empty cells and must terminate on the boundary
#     - branch cells may not touch other colors by side adjacency unless they are intended logical overlaps at a grey crossing
#     - red and yellow markers are isolated singletons and are kept outside the crossing 4-neighborhood
#     - yellow target rays and deleted old arms are kept outside the crossing buffer, preserving immediate supports for grey inference
#   visual_observability:
#     - every sample visibly contains both motifs: a same-color T junction and one or more grey crossing cells
#     - at least one red marker is placed on a junction when possible, making the branch topology part of the repair evidence
#     - grey stays grey in the output, so the crossing rule is directly observable from examples
#   cleanliness:
#     - only one special overlay type is added, the grey crossing, and it never stacks with red/yellow markers
#     - branch length is at least 2 cells, so the T motif is visually obvious
#     - crossing neighborhoods are kept clear, so T hubs and grey crossings do not visually merge into an unreadable cluster
#   generator_guarantees_for_solver:
#     - marker sight-lines are unambiguous under the visible completion score used by main()
#     - every grey crossing has exactly one horizontal claimant and one vertical claimant after repair
#     - every yellow old arm is straight and crossing-free, so constrained deletion matches construction
#     - generated outputs always contain at least one same-color junction and at least one grey crossing, preventing regression to either single-variant alone


DIRS4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
BG = 0
RED_MARKER = 2
YELLOW_MARKER = 4
CROSS_COLOR = 5
MARKER_COLORS = {RED_MARKER, YELLOW_MARKER}


def _pts(mask: np.ndarray) -> set[tuple[int, int]]:
    return {tuple(map(int, rc)) for rc in np.argwhere(mask)}


def _neighbors(pt: tuple[int, int]) -> list[tuple[int, int]]:
    r, c = pt
    return [(r + dr, c + dc) for dr, dc in DIRS4]


def _expanded(cells: set[tuple[int, int]]) -> set[tuple[int, int]]:
    out = set(cells)
    for pt in list(cells):
        out.update(nb for nb in _neighbors(pt) if 0 <= nb[0] < H and 0 <= nb[1] < W)
    return out


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
            if value == BG or value in marker_colors or value == CROSS_COLOR:
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


def _cross_axis_claim(mask: set[tuple[int, int]], pt: tuple[int, int], axis: int) -> bool:
    r, c = pt
    if axis == 0:
        return (r, c - 1) in mask and (r, c + 1) in mask
    return (r - 1, c) in mask and (r + 1, c) in mask


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
    masks: dict[int, set[tuple[int, int]]],
    marker_pts: dict[int, set[tuple[int, int]]],
    comp_count: dict[int, int],
    control_ray: dict[int, set[tuple[int, int]]],
    crossing_claims: dict[tuple[int, int], dict[int, int]],
) -> np.ndarray:
    claims: dict[tuple[int, int], list[int]] = {}
    for color, pts in masks.items():
        for pt in pts:
            claims.setdefault(pt, []).append(int(color))
    for pt, axes in crossing_claims.items():
        claims[pt] = sorted(axes)

    out = np.full_like(g, bg)
    for pt, colors_here in claims.items():
        if pt in crossing_claims:
            out[pt] = CROSS_COLOR
            continue

        forced = [color for color in colors_here if pt in control_ray.get(color, set())]
        if forced:
            out[pt] = _pick_unique_max(
                [(color, _claim_score(masks, marker_pts, comp_count, color, pt)) for color in forced],
                f"yellow-control claim at {pt}",
            )
            continue

        assert len(colors_here) == 1, f"unexpected non-grey overlap at {pt}: {colors_here}"
        out[pt] = colors_here[0]
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Repair a thin orthogonal colored line network by recovering branch hubs, preserving grey mixed crossings, and rotating yellow-marked arms without destroying same-color branches.

    Input structure:
    - A rectangular grid with a dominant background.
    - Non-grey line colors form one-cell-thick horizontal/vertical fragments; some solved components contain same-color T or + junctions.
    - Red (2) and yellow (4) pixels are isolated singleton markers; grey (5) pixels are reserved mixed-crossing cells rather than ordinary line color.
    - Each grey cell stands for exactly two hidden orthogonal line colors, while each red/yellow marker belongs to one nearby line color.

    Rule:
    - Assign each red/yellow marker to the visible line color that best completes the local orthogonal pattern around it, scanning through background, other markers, and grey crossing cells.
    - Recolor assigned markers logically and fill row/column gaps between supported facing fragments; this restores ordinary spans, branch hubs, and missing corners.
    - A yellow control deletes only the straight old arm behind its pivot until the first preserved junction, then paints the opposite ray from the pivot through the marker side to the boundary.
    - After those repairs, infer each grey cell's horizontal and vertical owners from opposite repaired neighbors and keep the visible crossing grey in the final output.

    Key insight: The hard part is separating two kinds of degree changes. A same-color T node is recovered because one color visibly assembles multiple arms around a marker, while a grey crossing is not recolored at all; its two hidden owners are inferred only from the repaired straight continuations on opposite sides.
    """
    g = np.asarray(grid, dtype=int)
    bg = int(get_bg(g))
    H, W = g.shape

    colors = set(map(int, g.ravel())) - {bg}
    assert colors, "expected foreground line, crossing, and marker colors"

    marker_colors = colors & MARKER_COLORS
    assert marker_colors, "expected at least one red or yellow marker"
    for color in marker_colors:
        comps = components_mask(g == color, connectivity=4)
        assert comps and max(len(comp) for comp in comps) == 1, f"marker color {color} must be singleton-only"

    cross_pts = _pts(g == CROSS_COLOR)
    line_colors = colors - marker_colors - {CROSS_COLOR}
    assert line_colors, "expected at least one non-marker line color"

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
                if value == bg or value in marker_colors or value == CROSS_COLOR:
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
    if YELLOW_MARKER in marker_colors:
        for control in _pts(g == YELLOW_MARKER):
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

    crossing_claims: dict[tuple[int, int], dict[int, int]] = {}
    for pt in cross_pts:
        axes: dict[int, int] = {}
        for color in line_colors:
            has_h = _cross_axis_claim(masks[color], pt, axis=0)
            has_v = _cross_axis_claim(masks[color], pt, axis=1)
            assert not (has_h and has_v), f"color {color} forms a junction at grey crossing {pt}"
            if has_h:
                axes[color] = 0
            elif has_v:
                axes[color] = 1
        assert len(axes) == 2, f"grey crossing {pt} needs exactly two inferred colors"
        assert sorted(axes.values()) == [0, 1], f"grey crossing {pt} must be orthogonal"
        crossing_claims[pt] = axes

    out = _resolve_claims(g, bg, masks, marker_pts, comp_count, control_ray, crossing_claims)

    logical_masks = {color: set(pts) for color, pts in masks.items()}
    for pt, axes in crossing_claims.items():
        for color in axes:
            logical_masks[color].add(pt)

    for color in line_colors:
        mask = np.zeros(g.shape, dtype=bool)
        for r, c in logical_masks[color]:
            mask[r, c] = True
        for comp in components_mask(mask, connectivity=4):
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
                    and (
                        (r + dr, c + dc) in cross_pts
                        or int(out[r + dr, c + dc]) not in (bg, color)
                    )
                    for dr, dc in DIRS4
                )
                assert adj_other, f"dangling endpoint ({r},{c}) color {color}: not on boundary or crossing"

        pts_set = logical_masks[color]
        for r in range(H - 1):
            for c in range(W - 1):
                block = {(r, c), (r + 1, c), (r, c + 1), (r + 1, c + 1)}
                assert not block.issubset(pts_set), f"2x2 thickness at {(r, c)} color {color}"

    crossing_colors = set()
    for axes in crossing_claims.values():
        crossing_colors.update(axes)
    for color in line_colors:
        assert color in crossing_colors, f"color {color} has no crossing with any other color"

    return out


# --------------- generator ---------------

H = W = 12
LINE_PALETTE = [1, 3, 6, 7, 8, 9]

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

MARKER_MODE_WEIGHTS = [3, 1]  # red / both; grey crossings are always present, yellow is optional
YELLOW_COUNT_RANGE = (0, 1)
RED_COUNT_RANGE = (1, 4)
MIN_CELLS_FOR_RED = 2

VISIBLE_COMPONENTS_PER_COLOR = (1, 5)
TOTAL_VISIBLE_COMPONENTS = (3, 9)
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
    crossing_buffer = _expanded(crossings)
    comp = paths[dominant][0]
    occupied = set().union(*logical.values())
    for pivot in random.sample(list(comp), len(comp)):
        if pivot in crossing_buffer:
            continue
        for out_dir in random.sample(list(DIRS4), len(DIRS4)):
            marker = (pivot[0] - out_dir[0], pivot[1] - out_dir[1])
            if not (0 <= marker[0] < H and 0 <= marker[1] < W):
                continue
            if marker in crossing_buffer:
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
            if ray_cells is None or any(pt in crossing_buffer for pt in ray_cells):
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
    n_colors = random.randint(2, min(MAX_COLORS, TOTAL_VISIBLE_COMPONENTS[1]))
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
    yellow_rays: dict[int, set[tuple[int, int]]],
    crossings: set[tuple[int, int]],
) -> np.ndarray:
    out = np.full((H, W), BG, dtype=int)
    for pt in set().union(*logical.values()):
        colors_here = [color for color in logical if pt in logical[color]]
        if pt in crossings:
            out[pt] = CROSS_COLOR
            continue
        forced = [color for color in colors_here if pt in yellow_rays.get(color, set())]
        if forced:
            out[pt] = forced[0]
        else:
            assert len(colors_here) == 1, f"unexpected visible overlap at {pt}: {colors_here}"
            out[pt] = colors_here[0]
    return out


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
    crossing_buffer = _expanded(crossings)
    mode = random.choices(["red", "both"], weights=MARKER_MODE_WEIGHTS)[0]
    yellow_count = 1 if mode == "both" else 0
    red_count = random.randint(*RED_COUNT_RANGE)

    yellow_infos = []
    blocked = set(crossing_buffer)
    if yellow_count:
        special = _build_special_yellow(logical, paths, dominant)
        if special is None:
            return None
        yellow_infos = [special]
        color, pivot, out_dir, target_ray, old_arm = special
        marker = (pivot[0] - out_dir[0], pivot[1] - out_dir[1])
        crossings = set(_crossing_cells(logical))
        crossing_buffer = _expanded(crossings)
        blocked = set(crossing_buffer)
        blocked.update(target_ray | old_arm | {marker, pivot})
        if len(yellow_infos) != yellow_count:
            return None

    red_markers = _choose_red_markers(logical, paths, blocked, red_count, require_junction=True)
    if not red_markers:
        return None

    inp = np.full((H, W), BG, dtype=int)
    removed_for_yellow: dict[int, set[tuple[int, int]]] = {c: set() for c in logical}
    old_arms: dict[int, set[tuple[int, int]]] = {c: set() for c in logical}
    yellow_rays: dict[int, set[tuple[int, int]]] = {c: set() for c in logical}

    yellow_markers: list[tuple[int, tuple[int, int]]] = []
    for color, pivot, out_dir, target_ray, old_arm in yellow_infos:
        marker = (pivot[0] - out_dir[0], pivot[1] - out_dir[1])
        removed_for_yellow[color].update(target_ray - {pivot})
        old_arms[color].update(old_arm)
        yellow_rays[color].update(target_ray)
        yellow_markers.append((color, marker))

    red_by_pt = {pt: color for color, pt in red_markers}
    yellow_by_pt = {pt: color for color, pt in yellow_markers}

    for color, pts in logical.items():
        for pt in pts:
            if pt in crossings:
                continue
            if pt in removed_for_yellow[color]:
                continue
            if pt in red_by_pt or pt in yellow_by_pt:
                continue
            inp[pt] = color
    for color, pts in old_arms.items():
        for pt in pts:
            if inp[pt] != BG or pt in crossings:
                return None
            inp[pt] = color

    for pt in crossings:
        if inp[pt] != BG:
            return None
        inp[pt] = CROSS_COLOR

    for pt, color in red_by_pt.items():
        if inp[pt] not in (BG, color):
            return None
        inp[pt] = RED_MARKER
    for pt, color in yellow_by_pt.items():
        if inp[pt] not in (BG, color):
            return None
        inp[pt] = YELLOW_MARKER

    if not _marker_isolation_ok(inp):
        return None
    for pt, color in yellow_by_pt.items():
        if not any(0 <= nb[0] < H and 0 <= nb[1] < W and inp[nb] == color for nb in _neighbors(pt)):
            return None

    target = _render_target(logical, yellow_rays, crossings)
    if not np.array_equal(target == CROSS_COLOR, inp == CROSS_COLOR):
        return None

    colors = set(logical)
    visible_counts = {c: len(components_mask(target == c, connectivity=4)) for c in colors}
    if not all(VISIBLE_COMPONENTS_PER_COLOR[0] <= v <= VISIBLE_COMPONENTS_PER_COLOR[1] for v in visible_counts.values()):
        return None
    if not TOTAL_VISIBLE_COMPONENTS[0] <= sum(visible_counts.values()) <= TOTAL_VISIBLE_COMPONENTS[1]:
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
    raise RuntimeError("failed to generate a valid faa9f03d combo input")
