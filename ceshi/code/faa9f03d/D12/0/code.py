# Variant of faa9f03d
# Dimension: D12 — 黄色控制的延伸方向 (transform)
# Change: 黄色标记方向由标记相对于端点的位置决定（从端点穿过标记延伸到边缘），旧臂在端点的对侧被擦除... → 延伸到最近障碍物而非网格边缘

from __future__ import annotations

import random

import numpy as np

from arc_utils import components_mask, get_bg


# Design analysis for D12:
# - Concrete behavior:
#   Input still contains sparse rectilinear line fragments plus singleton markers.
#   Red markers still stand for missing line cells. Yellow markers still sit next to a line
#   endpoint and indicate which side should become the new arm. The new difference is that the
#   repaired yellow arm is not allowed to run to the border anymore: after passing through the
#   yellow marker, it grows only until the first already-occupied non-background cell further
#   along that ray, and that blocking cell is included as the terminal collision cell.
# - Pipeline impact:
#   The old solver treated every yellow control as "erase opposite arm, then fill from pivot to
#   boundary". In this variant the critical changed step is the yellow-extension phase. Example:
#   if a pivot at (5,3) points right and another color occupies (5,8), the old logic would paint
#   (5,3..11); the new logic paints only (5,3..8). Cells (5,9..11) must be absent from both the
#   intended output and the generated input, otherwise the samples would silently collapse back to
#   the original edge-reaching rule.
# - Complexity calibration:
#   The generator always includes at least one yellow marker, and every chosen yellow arm is forced
#   to terminate at an interior crossing before the boundary. That gives repeated visible evidence
#   that the ray is obstacle-limited rather than border-limited, while keeping only 1-2 yellow
#   cases per grid so the scene stays readable.
# - Hidden assumptions audited in main():
#   Marker assignment and red-gap filling remain unchanged because they do not rely on the yellow
#   arm reaching the edge. The yellow block is the sensitive part: the old `while in bounds` paint
#   loop assumed "no obstacle matters before the border". That assumption is false now, so the new
#   code explicitly walks past the marker and stops at the first occupied cell. The endpoint check
#   also needed tightening: a yellow-stopped arm may now end at an interior crossing cell, so the
#   validation must treat multi-claim cells as legitimate terminals even when no visible neighbor
#   of another color is adjacent.
# - Placement legality in the generator:
#   Yellow candidates are selected only when (1) the target side is a straight arm, (2) the nearest
#   obstacle is a crossing on that arm before the boundary, (3) there is empty space on the opposite
#   side to host the fake old arm, and (4) multiple yellow edits do not overlap. The generator
#   trims the tail beyond the obstacle from the final target, removes the obstacle-side segment from
#   the input, and inserts the fake old arm on the far side so the solver has exactly one valid
#   repair to perform.
# - Visual observability / cleanliness:
#   The blocking event is always geometric and visible. No invisible color-id ordering is used to
#   decide the new stop position. At obstacle crossings reached by yellow rays, the input shows the
#   blocker color while the output shows the yellow-controlled color, making the collision-based
#   stop directly inspectable from a few examples.
# - New solver assumptions checked against the generator:
#   The solver now assumes every yellow marker has some occupied blocker beyond the marker on its
#   chosen ray. The generator guarantees this by only selecting yellow candidates whose straight arm
#   hits an interior crossing and by trimming only the tail beyond that crossing.


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


def _rc(ax: int, major: int, minor: int) -> tuple[int, int]:
    return (major, minor) if ax == 0 else (minor, major)


def _dim(ax: int) -> int:
    return H if ax == 0 else W


def _random_span(limit: int) -> tuple[int, int]:
    if random.random() < 0.5:
        return 0, random.randint(MIN_SPAN_LEN, limit - 1)
    return random.randint(0, limit - MIN_SPAN_LEN - 1), limit - 1


def _build_polyline_waypoints(ax: int) -> list[tuple[int, int]]:
    pieces = random.choices(
        range(1, len(POLYLINE_PIECE_WEIGHTS) + 1),
        weights=POLYLINE_PIECE_WEIGHTS,
    )[0]
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


def _build_arm(
    ax: int,
    main_coord: int,
    cross_lo: int,
    cross_hi: int,
    is_start: bool,
) -> list[tuple[int, int]]:
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
    return [
        _rc(ax, arm_major, main_coord),
        _rc(ax, arm_major, minor),
        _rc(ax, _dim(ax) - 1, minor),
    ]


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


def _logical_components(logical: dict[int, set[tuple[int, int]]]) -> dict[int, list[set[tuple[int, int]]]]:
    paths: dict[int, list[set[tuple[int, int]]]] = {}
    for color, pts in logical.items():
        mask = np.zeros((H, W), dtype=bool)
        for r, c in pts:
            mask[r, c] = True
        paths[color] = [set(map(tuple, comp)) for comp in components_mask(mask, connectivity=4)]
    return paths


def _ray_list(pivot: tuple[int, int], direction: tuple[int, int]) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    r, c = pivot
    dr, dc = direction
    while 0 <= r < H and 0 <= c < W:
        out.append((r, c))
        r += dr
        c += dc
    return out


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


def main(grid: np.ndarray) -> np.ndarray:
    """
    Repair a sparse one-cell-thick line network by recoloring singleton markers, filling orthogonal gaps, and redirecting yellow-marked arms until they hit an obstacle.

    Input structure:
    - The grid has a dominant background plus several one-cell-thick rectilinear line colors.
    - Any color whose 4-connected components are all singletons is a marker color rather than a line color.
    - Red singletons mark missing line cells; yellow singletons sit next to a line endpoint and indicate the new growth side.

    Rule:
    - Assign each marker to the nearest visible line color in straight line-of-sight, preferring candidates already supported along that axis.
    - For each recovered line color, fill the row/column gap between consecutive facing endpoints using the same local support logic as the base task.
    - A yellow control removes the old arm on the opposite side of its pivot, then extends the pivot through the yellow marker only until the first non-background cell beyond the marker; that blocking cell is included.
    - When multiple colors claim one cell, yellow-control traces override other claims; otherwise use the same fragmentation-based crossing disambiguation as in the base task.

    Key insight: The singleton markers still tell you which fragments belong together, but yellow no longer means "grow to the border" — it means "flip the arm and stop at the first visible blocker on that ray".
    """
    g = np.asarray(grid, dtype=int)
    bg = int(get_bg(g))
    h, w = g.shape

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
            while 0 <= r < h and 0 <= c < w:
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

    for color in sorted(line_colors):
        additions: set[tuple[int, int]] = set()
        for r in range(h):
            cols = sorted(c for rr, c in masks[color] if rr == r)
            for c1, c2 in zip(cols, cols[1:]):
                if should_fill(color, (r, c1), (r, c2), axis=0):
                    additions.update((r, c) for c in range(c1 + 1, c2))

        for c in range(w):
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
            while 0 <= r < h and 0 <= c < w:
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
            occupied = set().union(*masks.values())
            control_ray[color].add(pivot)
            masks[color].add(pivot)

            r, c = control
            control_ray[color].add((r, c))
            masks[color].add((r, c))

            hit_obstacle = False
            r += pr
            c += pc
            while 0 <= r < h and 0 <= c < w:
                pt = (r, c)
                masks[color].add(pt)
                control_ray[color].add(pt)
                if pt in occupied:
                    hit_obstacle = True
                    break
                r += pr
                c += pc
            assert hit_obstacle, f"yellow control at {control} reached no blocking cell"

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
        else:
            bracketed = [color for color in colors_here if marker_brackets(color, pt)]
            if bracketed:
                out[pt] = min(bracketed, key=lambda color: (-comp_count[color], color))
            else:
                out[pt] = min(colors_here)

    crossing_pts = {pt for pt, colors_here in claims.items() if len(colors_here) > 1}

    for color in line_colors:
        for comp in components_mask(out == color, connectivity=4):
            pts_set = set(map(tuple, comp))
            for r, c in pts_set:
                nbrs = [
                    (nr, nc)
                    for nr, nc in [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]
                    if (nr, nc) in pts_set
                ]
                if len(nbrs) != 1:
                    continue
                if (r, c) in crossing_pts:
                    continue
                on_boundary = r == 0 or r == h - 1 or c == 0 or c == w - 1
                if on_boundary:
                    nr, nc = nbrs[0]
                    dr, dc = nr - r, nc - c
                    if not ((r in (0, h - 1)) and (c in (0, w - 1))):
                        perp = (
                            (r == 0 and dr == 1)
                            or (r == h - 1 and dr == -1)
                            or (c == 0 and dc == 1)
                            or (c == w - 1 and dc == -1)
                        )
                        assert perp, f"boundary endpoint ({r},{c}) color {color} not perpendicular"
                    continue
                adj_other = any(
                    0 <= r + dr < h
                    and 0 <= c + dc < w
                    and int(out[r + dr, c + dc]) not in (bg, color)
                    for dr, dc in DIRS4
                )
                assert adj_other, f"dangling endpoint ({r},{c}) color {color}: not on boundary or crossing"

    color_cells = {c: set() for c in line_colors}
    for r in range(h):
        for c in range(w):
            v = int(out[r, c])
            if v in line_colors:
                color_cells[v].add((r, c))
    crossing_colors = set()
    for r in range(h):
        for c in range(w):
            a = int(out[r, c])
            if a not in line_colors:
                continue
            for b in line_colors - {a}:
                if (
                    ((r, c - 1) in color_cells[b] and (r, c + 1) in color_cells[b])
                    or ((r - 1, c) in color_cells[b] and (r + 1, c) in color_cells[b])
                ):
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

MARKER_MODE_WEIGHTS = [2, 3]  # yellow-only / both; yellow must always appear in D12
YELLOW_COUNT_RANGE = (1, 2)
RED_COUNT_RANGE = (2, 5)
MIN_CELLS_FOR_RED = 2
MIN_BLOCK_DIST = 2

COMPONENTS_PER_COLOR = (1, 4)
TOTAL_COMPONENTS = (4, 6)
FILL_RANGE = (28, 52)

SPACED_VALUES_RETRIES = 60
EXTRA_POLYLINE_RETRIES = 80
GEN_RETRIES = 400


def _yellow_candidates(
    logical: dict[int, set[tuple[int, int]]],
    paths: dict[int, list[set[tuple[int, int]]]],
    crossings: set[tuple[int, int]],
) -> list[dict]:
    candidates: list[dict] = []
    occupied = set().union(*logical.values())
    for color, comps in paths.items():
        for comp in comps:
            for pivot in comp:
                if pivot in crossings or not _is_turn_or_endpoint(comp, pivot):
                    continue
                for out_dir in DIRS4:
                    marker = (pivot[0] + out_dir[0], pivot[1] + out_dir[1])
                    if marker not in comp:
                        continue
                    ray = _ray_list(pivot, out_dir)
                    if not set(ray).issubset(comp):
                        continue
                    obstacle_idx = None
                    for idx, pt in enumerate(ray[1:], start=1):
                        if pt in crossings:
                            obstacle_idx = idx
                            break
                    if obstacle_idx is None or obstacle_idx < MIN_BLOCK_DIST:
                        continue
                    target_seg = set(ray[: obstacle_idx + 1])
                    removed_tail = set(ray[obstacle_idx + 1 :])
                    if not removed_tail:
                        continue
                    old_dir = (-out_dir[0], -out_dir[1])
                    old_cells: list[tuple[int, int]] = []
                    r, c = pivot[0] + old_dir[0], pivot[1] + old_dir[1]
                    while 0 <= r < H and 0 <= c < W and (r, c) not in occupied:
                        old_cells.append((r, c))
                        r += old_dir[0]
                        c += old_dir[1]
                    if not old_cells:
                        continue
                    old_len = random.randint(1, len(old_cells))
                    old_arm = set(old_cells[:old_len])
                    candidates.append(
                        {
                            "color": color,
                            "pivot": pivot,
                            "out_dir": out_dir,
                            "marker": marker,
                            "target_seg": target_seg,
                            "removed_tail": removed_tail,
                            "old_arm": old_arm,
                            "obstacle": ray[obstacle_idx],
                            "zone": target_seg | removed_tail | old_arm | {pivot, marker},
                        }
                    )
    random.shuffle(candidates)
    return candidates


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
    comp_count: dict[int, int],
    yellow_rays: dict[int, set[tuple[int, int]]],
) -> np.ndarray:
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


def _marker_isolation_ok(grid: np.ndarray) -> bool:
    for color in (2, 4):
        for pt in map(tuple, np.argwhere(grid == color)):
            if any(grid[nb] == color for nb in _neighbors(pt)):
                return False
    return True


def _derive_input(
    logical_base: dict[int, set[tuple[int, int]]],
    paths_base: dict[int, list[set[tuple[int, int]]]],
    dominant: int,
) -> tuple[np.ndarray, np.ndarray] | None:
    base_crossings = set(_crossing_cells(logical_base))
    mode = random.choices(["yellow", "both"], weights=MARKER_MODE_WEIGHTS)[0]
    yellow_count = random.randint(*YELLOW_COUNT_RANGE)
    red_count = random.randint(*RED_COUNT_RANGE) if mode == "both" else 0

    yellow_infos: list[dict] = []
    reserved_marker_zone = set(base_crossings)
    used_zone: set[tuple[int, int]] = set()
    for cand in _yellow_candidates(logical_base, paths_base, base_crossings):
        marker = cand["marker"]
        if marker in reserved_marker_zone or any(nb in reserved_marker_zone for nb in _neighbors(marker)):
            continue
        if cand["zone"] & used_zone:
            continue
        yellow_infos.append(cand)
        used_zone.update(cand["zone"])
        reserved_marker_zone.update(cand["target_seg"] | cand["old_arm"] | {marker})
        if len(yellow_infos) == yellow_count:
            break
    if len(yellow_infos) != yellow_count:
        return None

    logical_target = {color: set(pts) for color, pts in logical_base.items()}
    yellow_rays: dict[int, set[tuple[int, int]]] = {c: set() for c in logical_target}
    removed_for_yellow: dict[int, set[tuple[int, int]]] = {c: set() for c in logical_target}
    old_arms: dict[int, set[tuple[int, int]]] = {c: set() for c in logical_target}
    yellow_markers: list[tuple[int, tuple[int, int]]] = []
    for cand in yellow_infos:
        color = cand["color"]
        logical_target[color].difference_update(cand["removed_tail"])
        removed_for_yellow[color].update(cand["target_seg"] - {cand["pivot"]})
        old_arms[color].update(cand["old_arm"])
        yellow_rays[color].update(cand["target_seg"])
        yellow_markers.append((color, cand["marker"]))

    crossings = set(_crossing_cells(logical_target))
    for cand in yellow_infos:
        obstacle = cand["obstacle"]
        if obstacle not in crossings or _on_boundary(*obstacle):
            return None
        if obstacle not in yellow_rays[cand["color"]]:
            return None

    paths_target = _logical_components(logical_target)
    blocked_for_red = set(crossings)
    for cand in yellow_infos:
        blocked_for_red.update(cand["target_seg"] | cand["old_arm"] | {cand["marker"]})
    red_markers = _choose_red_markers(logical_target, paths_target, blocked_for_red, red_count) if red_count else []
    if red_count and not red_markers:
        return None

    inp = np.full((H, W), BG, dtype=int)
    comp_count_masks: dict[int, set[tuple[int, int]]] = {c: set() for c in logical_target}

    crossing_display: dict[tuple[int, int], int] = {}
    for pt in crossings:
        colors = [color for color in logical_target if pt in logical_target[color]]
        if len(colors) != 2:
            return None
        forced = [color for color in colors if pt in yellow_rays[color]]
        if forced:
            if len(forced) != 1:
                return None
            crossing_display[pt] = next(color for color in colors if color != forced[0])
        else:
            if dominant not in colors:
                return None
            crossing_display[pt] = colors[0] if colors[1] == dominant else colors[1]

    red_by_pt = {pt: color for color, pt in red_markers}
    yellow_by_pt = {pt: color for color, pt in yellow_markers}

    for color, pts in logical_target.items():
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
        if not any(inp[nb] == color for nb in _neighbors(pt)):
            return None

    comp_count = {color: _gen_component_count(pts) for color, pts in comp_count_masks.items()}
    target = _render_target(logical_target, comp_count, yellow_rays)
    for pt in crossings:
        if target[pt] == inp[pt]:
            return None

    counts = {c: len(components_mask(target == c, connectivity=4)) for c in sorted(logical_target)}
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
    logical, paths, dominant = built
    derived = _derive_input(logical, paths, dominant)
    return derived[0] if derived is not None else None


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("failed to generate a valid faa9f03d D12 input")
