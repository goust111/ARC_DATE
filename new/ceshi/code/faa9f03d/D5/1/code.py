# Variant of faa9f03d
# Dimension: D5 — 交叉处的颜色消歧规则 (transform)
# Change: 黄色控制射线优先；否则原始交叉处选碎片度更高的颜色（pre-repair连通分量更多者胜出），纯填充交叉选较小颜色ID... → 交叉处选线段更长的颜色

# Design analysis for the D5 variant
# ----------------------------------
# a) Concrete behavior and trace
#    The objects stay the same: 12x12 grids, 1-cell-thick rectilinear colored polylines,
#    red singletons that mark missing cells, and yellow singletons that redirect an endpoint.
#    The only semantic change is what happens when two repaired colors both claim one crossing cell.
#    Example: suppose a vertical blue run of length 9 and a horizontal green run of length 5 both
#    pass through the same cell after all repairs. The original task could choose blue because blue
#    had more pre-repair fragments. In this variant, blue wins because the straight segment through
#    the crossing is locally longer (9 > 5), even if green had more fragments globally.
#    This changes the conflict-resolution stage only; marker assignment, gap filling, and yellow-arm
#    flipping remain intact.
# b) Complexity calibration
#    The variant should be visible from a few examples, so crossings must not collapse into equal
#    local spans too often. The generator therefore rejects ambiguous equal-length crossings and also
#    requires at least one crossing where the new local-length winner differs from the old
#    fragmentation winner. The task still uses only 2-4 colors and 1-12 crossings, so the rule is
#    observable without adding unrelated object types.
# c) Hidden-assumption review against the original main()
#    - Marker detection and line-of-sight assignment do not depend on fragmentation, so they remain.
#    - Gap filling also does not depend on fragmentation, so it remains.
#    - Yellow control processing must still happen before conflict resolution, because a control ray
#      can overwrite a crossing regardless of local span lengths.
#    - The original conflict code relied on `comp_count` and on a color-id fallback for pure fills.
#      Those assumptions are invalid here: a globally fragmented color can lose if its local straight
#      run is shorter, and color ids are not a visible cue. The new code instead measures the
#      straight run through the candidate crossing cell inside each repaired color mask.
# d) Placement legality
#    The generator keeps the original output-first construction: it builds valid polylines first,
#    then removes cells for red markers, flips arms for yellow markers, and finally verifies that the
#    solver reconstructs the target exactly. Additional acceptance checks ensure each crossing has a
#    unique local-length winner, so the solver never has to guess.
# e) Visual observability
#    The new rule is geometric and local. A human can compare the visible straight spans that meet at
#    a crossing: the crossing adopts the color of the longer through-segment. Because the generator
#    enforces at least one disagreement with the old fragmentation rule, the dataset actually exhibits
#    the new behavior rather than accidentally looking like the original task.
# f) Visual cleanliness
#    No new overlay type is added. The same single-pixel markers and sparse orthogonal lines are used,
#    so crossings remain visually legible. Rejecting equal-length crossings also removes visually
#    confusing near-symmetries.
# g) Color-change inference
#    Colors themselves are not ordered. The only non-yellow crossing decision uses visible geometry:
#    compare the local straight segment lengths through the contested cell. The generator keeps the
#    palette size small and repeats the same kind of cue across crossings.
# h) New solver assumptions and generator guarantees
#    - Assumption: every non-yellow ambiguous cell has a unique best local straight-span profile.
#      Guarantee: target rendering rejects ties and generation retries.
#    - Assumption: the local span should be measured on the repaired masks after red filling and
#      yellow redirection. Guarantee: the generator renders the target from that same repaired logic.
#    - Assumption: every displayed crossing still consists of two orthogonal colors. Guarantee:
#      `_valid_crossings` preserves the original orthogonal-crossing construction.

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


def _straight_run_len(pts: set[tuple[int, int]], pt: tuple[int, int], axis: int) -> int:
    r, c = pt
    length = 1
    if axis == 0:
        cc = c - 1
        while (r, cc) in pts:
            length += 1
            cc -= 1
        cc = c + 1
        while (r, cc) in pts:
            length += 1
            cc += 1
        return length

    rr = r - 1
    while (rr, c) in pts:
        length += 1
        rr -= 1
    rr = r + 1
    while (rr, c) in pts:
        length += 1
        rr += 1
    return length


def _local_span_profile(pts: set[tuple[int, int]], pt: tuple[int, int]) -> tuple[int, int]:
    r, c = pt
    lengths: list[int] = []
    if (r, c - 1) in pts or (r, c + 1) in pts:
        lengths.append(_straight_run_len(pts, pt, axis=0))
    if (r - 1, c) in pts or (r + 1, c) in pts:
        lengths.append(_straight_run_len(pts, pt, axis=1))
    if not lengths:
        return (1, 0)
    lengths.sort(reverse=True)
    if len(lengths) == 1:
        lengths.append(0)
    return (lengths[0], lengths[1])


def _longer_segment_winner(
    masks: dict[int, set[tuple[int, int]]],
    colors_here: list[int],
    pt: tuple[int, int],
    visible_color: int | None = None,
) -> int:
    profiles = {color: _local_span_profile(masks[color], pt) for color in colors_here}
    best_profile = max(profiles.values())
    best = [color for color, profile in profiles.items() if profile == best_profile]
    if len(best) == 1:
        return best[0]
    if visible_color in best:
        return int(visible_color)
    raise AssertionError(f"ambiguous equal-length crossing at {pt}: {profiles}")


def main(grid: np.ndarray) -> np.ndarray:
    """
    Repair a one-cell-thick colored line network by restoring singleton markers, filling straight gaps, and redirecting yellow-marked endpoints.

    Input structure:
    - A rectangular grid with a dominant background.
    - Non-marker colors appear as one-cell-thick horizontal/vertical line fragments.
    - Marker colors are exactly the colors whose 4-connected components are all singleton pixels.
    - Red singletons mark removed line cells; yellow singletons sit next to endpoints whose arms must be flipped.

    Rule:
    - Assign each marker to the nearest visible line color seen in a cardinal direction, preferring a sightline that hits a cell already supported along that same axis.
    - For each line color, bridge facing cells in the same row or column when their local endpoint geometry indicates they were split by a marker gap.
    - A yellow control removes the old arm beyond its pivot endpoint and extends the line through the control pixel all the way to the opposite boundary.
    - When multiple colors claim one cell, any yellow-control ray still has priority; otherwise the crossing takes the color whose repaired straight segment through that cell is longer.

    Key insight: The singleton markers tell you which sparse fragments belong to the same rectilinear path, but crossing ownership is now decided locally by geometry: compare the straight spans that actually pass through the contested cell after repair.
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
    for pt, color in assignment.items():
        masks[color].add(pt)
        marker_pts[color].add(pt)

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

        visible_color = int(g[pt]) if int(g[pt]) in colors_here else None
        out[pt] = _longer_segment_winner(masks, colors_here, pt, visible_color)

    # Assert: every visible endpoint is on the boundary (or at a crossing)
    for color in line_colors:
        for comp in components_mask(out == color, connectivity=4):
            pts_set = set(map(tuple, comp))
            for r, c in pts_set:
                nbrs = [(nr, nc) for nr, nc in [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]
                        if (nr, nc) in pts_set]
                if len(nbrs) != 1:
                    continue
                on_boundary = (r == 0 or r == H - 1 or c == 0 or c == W - 1)
                if on_boundary:
                    nr, nc = nbrs[0]
                    dr, dc = nr - r, nc - c
                    if not ((r in (0, H - 1)) and (c in (0, W - 1))):
                        perp = ((r == 0 and dr == 1) or (r == H - 1 and dr == -1)
                                or (c == 0 and dc == 1) or (c == W - 1 and dc == -1))
                        assert perp, f"boundary endpoint ({r},{c}) color {color} not perpendicular"
                    continue
                adj_other = any(
                    0 <= r + dr < H and 0 <= c + dc < W
                    and int(out[r + dr, c + dc]) not in (bg, color)
                    for dr, dc in DIRS4
                )
                assert adj_other, f"dangling endpoint ({r},{c}) color {color}: not on boundary or crossing"

    # Assert: every line color crosses at least one other color
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


# --------------- generator (o5x) ---------------

# --- CFG: grid & palette ---
H = W = 12
BG = 0
DIRS4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
LINE_PALETTE = [1, 3, 5, 6, 7, 8, 9]

# --- CFG: polyline shape ---
POLYLINE_PIECE_WEIGHTS = [5, 4, 2, 1]   # weights for 1-4 segments per polyline
MIN_SPAN_LEN = 4                         # minimum length of a polyline's primary span
FULL_SPAN_PROB = 0.65                    # probability path_through extends to grid edge
ARM_MIN_OFFSET = 2                       # minimum distance between arm column and main line

# --- CFG: serpentine & crossings ---
CROSSING_COUNT_WEIGHTS = [2, 3, 3, 2]   # weights for 1-4 crossing rows in serpentine
SERPENTINE_MINOR_LO = 6                  # max value for minor-axis lower bound
SERPENTINE_MINOR_HI = 5                  # max value for minor-axis upper offset from 0
CROSS_MINOR_MARGIN = 2                   # margin from serpentine edges to crossing column
MAX_COLORS = 4                           # maximum number of line colors
MAX_CROSSINGS = 12                       # upper bound on crossing cell count

# --- CFG: markers ---
MARKER_MODE_WEIGHTS = [6, 1, 2]         # weights for red-only / yellow-only / both
YELLOW_COUNT_RANGE = (1, 2)
RED_COUNT_RANGE = (2, 6)
MIN_CELLS_FOR_RED = 2                    # color must have > this many cells to place a red marker

# --- CFG: output validation ---
COMPONENTS_PER_COLOR = (1, 4)
TOTAL_COMPONENTS = (4, 6)
FILL_RANGE = (33, 55)                    # non-bg cell count in output

# --- CFG: retry budgets ---
SPACED_VALUES_RETRIES = 60
EXTRA_POLYLINE_RETRIES = 80
GEN_RETRIES = 600


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
        if not ((r == 0 and dr == 1) or (r == H - 1 and dr == -1)
                or (c == 0 and dc == 1) or (c == W - 1 and dc == -1)):
            return False
    for r, c in pts:
        if len(_dirs_in(pts, (r, c))) == 1 and not _on_boundary(r, c):
            return False
    return True


# ---------------------------------------------------------------------------
# Axis-generic helpers: ax=0 means the polyline's "main" axis is rows
# (vertical primary), ax=1 means columns (horizontal primary).
# p[ax] is the major coordinate, p[1-ax] is the minor coordinate.
# ---------------------------------------------------------------------------

def _rc(ax: int, major: int, minor: int) -> tuple[int, int]:
    return (major, minor) if ax == 0 else (minor, major)


def _dim(ax: int) -> int:
    return H if ax == 0 else W


def _random_span(limit: int) -> tuple[int, int]:
    if random.random() < 0.5:
        return 0, random.randint(MIN_SPAN_LEN, limit - 1)
    return random.randint(0, limit - MIN_SPAN_LEN - 1), limit - 1


def _build_polyline_waypoints(ax: int) -> list[tuple[int, int]]:
    """Build waypoint list for a random polyline with axis ax (0=vert-primary, 1=horiz-primary)."""
    pieces = random.choices(
        range(1, len(POLYLINE_PIECE_WEIGHTS) + 1),
        weights=POLYLINE_PIECE_WEIGHTS,
    )[0]
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


def _random_polyline() -> list[tuple[int, int]]:
    return _path(_build_polyline_waypoints(random.randint(0, 1)))


def _build_arm(ax: int, main_coord: int, cross_lo: int, cross_hi: int,
               is_start: bool) -> list[tuple[int, int]]:
    """Build L-shaped arm waypoints for _path_crossing_all / _path_through."""
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
    return [_rc(ax, arm_major, main_coord), _rc(ax, arm_major, minor),
            _rc(ax, _dim(ax) - 1, minor)]


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
            ext_c = 0 if c0 == m0 else W - 1
            pts.insert(0, (r0, ext_c))
        else:
            ext_r = 0 if r0 == m0 else H - 1
            pts.insert(0, (ext_r, c0))
    if not _on_boundary(*pts[-1]):
        rn, cn = pts[-1]
        if ax == 0:
            ext_c = 0 if cn == m0 else W - 1
            pts.append((rn, ext_c))
        else:
            ext_r = 0 if rn == m0 else H - 1
            pts.append((ext_r, cn))

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
                if pivot in crossings or not _is_turn_or_endpoint(comp, pivot):
                    continue
                for out_dir in DIRS4:
                    mr, mc = pivot[0] + out_dir[0], pivot[1] + out_dir[1]
                    if not (0 <= mr < H and 0 <= mc < W) or (mr, mc) not in comp:
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
                    old_arm = set(list(_ray_from(
                        (pivot[0] + old_dir[0], pivot[1] + old_dir[1]), old_dir,
                    ))[:old_len])
                    if old_arm and old_arm.issubset(old_cells):
                        candidates.append((color, pivot, out_dir, target_ray, old_arm))
    random.shuffle(candidates)
    return candidates


def _choose_red_markers(
    logical: dict[int, set[tuple[int, int]]],
    paths: dict[int, list[set[tuple[int, int]]]],
    blocked: set[tuple[int, int]],
    count: int,
) -> list[tuple[int, tuple[int, int]]]:
    candidates = [(color, pt) for color, comps in paths.items() for comp in comps
                  for pt in comp if pt not in blocked and _is_turn_or_endpoint(comp, pt)]
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


def _find_crossing_point(
    logical: dict[int, set[tuple[int, int]]],
) -> tuple[tuple[int, int], str] | None:
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
    desired_crossings = random.choices(
        range(1, len(CROSSING_COUNT_WEIGHTS) + 1),
        weights=CROSSING_COUNT_WEIGHTS,
    )[0]
    n_colors = random.randint(2, min(MAX_COLORS, TOTAL_COMPONENTS[1] - desired_crossings))
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
    return logical, paths, dominant


def _render_target(
    logical: dict[int, set[tuple[int, int]]],
    yellow_rays: dict[int, set[tuple[int, int]]],
) -> np.ndarray | None:
    out = np.full((H, W), BG, dtype=int)
    for pt in set().union(*logical.values()):
        colors_here = [color for color in logical if pt in logical[color]]
        forced = [color for color in colors_here if pt in yellow_rays.get(color, set())]
        if forced:
            if len(forced) != 1:
                return None
            out[pt] = forced[0]
        elif len(colors_here) == 1:
            out[pt] = colors_here[0]
        else:
            try:
                out[pt] = _longer_segment_winner(logical, colors_here, pt)
            except AssertionError:
                return None
    return out


def _marker_isolation_ok(grid: np.ndarray) -> bool:
    for color in (2, 4):
        for pt in map(tuple, np.argwhere(grid == color)):
            if any(grid[nb] == color for nb in _neighbors(pt)):
                return False
    return True


def _derive_input(
    logical: dict[int, set[tuple[int, int]]],
    paths: dict[int, list[set[tuple[int, int]]]],
    dominant: int,
) -> tuple[np.ndarray, np.ndarray] | None:
    del dominant
    crossings = set(_crossing_cells(logical))
    mode = random.choices(["red", "yellow", "both"], weights=MARKER_MODE_WEIGHTS)[0]
    yellow_count = random.randint(*YELLOW_COUNT_RANGE) if mode in ("yellow", "both") else 0
    red_count = random.randint(*RED_COUNT_RANGE) if mode in ("red", "both") else 0

    yellow_infos = []
    blocked = set(crossings)
    if yellow_count:
        for cand in _yellow_candidates(logical, paths, crossings):
            color, pivot, out_dir, target_ray, old_arm = cand
            marker = (pivot[0] + out_dir[0], pivot[1] + out_dir[1])
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

    red_markers = _choose_red_markers(logical, paths, blocked, red_count) if red_count else []
    if red_count and not red_markers:
        return None

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

    crossing_info: dict[tuple[int, int], tuple[int, int, bool]] = {}
    safe_crossings: list[tuple[int, int]] = []
    for pt in crossings:
        colors = [color for color in logical if pt in logical[color]]
        try:
            winner = _longer_segment_winner(logical, colors, pt)
        except AssertionError:
            return None
        losers = [color for color in colors if color != winner]
        if len(losers) != 1:
            return None
        forced = any(pt in yellow_rays[color] for color in colors)
        crossing_info[pt] = (winner, losers[0], forced)
        if not forced:
            safe_crossings.append(pt)
    if not safe_crossings:
        return None

    target = _render_target(logical, yellow_rays)
    if target is None:
        return None

    red_by_pt = {pt: color for color, pt in red_markers}
    yellow_by_pt = {pt: color for color, pt in yellow_markers}

    candidate_pts = random.sample(safe_crossings, len(safe_crossings))
    for divergence_pt in candidate_pts:
        inp = np.full((H, W), BG, dtype=int)
        comp_count_masks: dict[int, set[tuple[int, int]]] = {c: set() for c in logical}
        crossing_display = {pt: loser for pt, (_winner, loser, _forced) in crossing_info.items()}
        crossing_display[divergence_pt] = crossing_info[divergence_pt][0]

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
        failed = False
        for color, pts in old_arms.items():
            for pt in pts:
                if inp[pt] != BG:
                    failed = True
                    break
                inp[pt] = color
                comp_count_masks[color].add(pt)
            if failed:
                break
        if failed:
            continue

        for pt, color in red_by_pt.items():
            inp[pt] = 2
            comp_count_masks[color].add(pt)
        for pt, color in yellow_by_pt.items():
            inp[pt] = 4
            comp_count_masks[color].add(pt)

        if not _marker_isolation_ok(inp):
            continue
        if any(not any(inp[nb] == color for nb in _neighbors(pt)) for pt, color in yellow_by_pt.items()):
            continue

        comp_count = {color: _gen_component_count(pts) for color, pts in comp_count_masks.items()}
        if any(target[pt] == inp[pt] for pt in crossings if pt != divergence_pt):
            continue
        colors_here = [color for color in logical if divergence_pt in logical[color]]
        old_winner = min(colors_here, key=lambda color: (-comp_count[color], color))
        if target[divergence_pt] != crossing_info[divergence_pt][0]:
            continue
        if target[divergence_pt] == old_winner:
            continue

        colors = sorted(logical)
        counts = {c: len(components_mask(target == c, connectivity=4)) for c in colors}
        if not all(COMPONENTS_PER_COLOR[0] <= v <= COMPONENTS_PER_COLOR[1] for v in counts.values()):
            continue
        if not TOTAL_COMPONENTS[0] <= sum(counts.values()) <= TOTAL_COMPONENTS[1]:
            continue
        if not FILL_RANGE[0] <= int(np.count_nonzero(target)) <= FILL_RANGE[1]:
            continue

        try:
            solved = main(inp)
        except Exception:
            continue
        if np.array_equal(solved, target):
            return inp, target
    return None


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
    raise RuntimeError("failed to generate a valid faa9f03d input")
