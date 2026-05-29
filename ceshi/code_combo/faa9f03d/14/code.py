# Combo variant of faa9f03d (index 14)
# combo_spec: D2/0+D7/1
# Name: 允许对角线段 + 输出放大2x2
#
# Design analysis for the fused variant:
# - Concrete behavior trace:
#   In the base task, a red marker can stand in for a missing orthogonal cell and a
#   yellow marker can flip an orthogonal endpoint to the opposite side. Here the
#   same repair logic is lifted from 2 straight-axis families (row / column) to
#   4 families (row / column / main diagonal / anti-diagonal). A diagonal red
#   marker therefore bridges two diagonal fragments, and a yellow marker can
#   redirect a 45-degree arm as well. After the 12x12 repair is finished, every
#   solved pixel is expanded to a 2x2 block, producing a 24x24 output.
# - Complexity calibration:
#   The generator always includes one genuine diagonal repair site, but keeps the
#   number of colors low (3) and the network thin (degree <= 2 per color) so the
#   examples stay legible. This prevents the variant from degenerating into the
#   original all-orthogonal task while avoiding excessive crossing clutter.
# - Implicit assumptions that changed:
#   Original code assumed 4-neighbor connectivity, row/column line-of-sight, and
#   orthogonal gap filling. Every such use is replaced here by 8-neighbor
#   connectivity plus a "straight family" abstraction over the 4 legal line
#   directions. Yellow-arm erasure is restricted to a straight ray, and the
#   generator only creates yellow cases that satisfy this stricter assumption.
# - Placement legality:
#   All generated polylines are built from horizontal / vertical / 45-degree
#   segments, remain in-bounds, avoid same-color self-overlap, and only overlap
#   different colors at true crossings. Yellow old-arms are added only into empty
#   space; marker cells are kept isolated from each other.
# - Visual observability:
#   Each instance contains at least one diagonal gap marked in red, so the repair
#   is visibly diagonal before scaling. The 2x enlargement is applied only after
#   repair, which makes the solved diagonal more obvious while still requiring the
#   solver to infer the original 12x12 structure from the examples.
# - Cleanliness:
#   Crossings are capped and forced to involve exactly two colors with different
#   local orientations. The diagonal color is a single clean stroke, so the final
#   24x24 output reads as enlarged repaired lines rather than noisy texture.
# - New solver assumptions and generator guarantees:
#   `main()` assumes markers can see their owner color along one of the 8 rays,
#   gap endpoints belong to the same straight family, and yellow controls only act
#   on straight tails. `generate_input()` constructs exactly those cases and
#   rejects any sample whose rendered target does not round-trip through `main()`.

from __future__ import annotations

import random

import numpy as np

from arc_utils import components_mask, get_bg


H = W = 12
BG = 0

DIRS4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
DIRS8 = (
    (-1, -1), (-1, 0), (-1, 1),
    (0, -1),            (0, 1),
    (1, -1),  (1, 0),   (1, 1),
)
LINE_DIRS = ((0, 1), (1, 0), (1, 1), (1, -1))
LINE_PALETTE = [1, 3, 5, 6, 7, 8, 9]

POLYLINE_PIECE_WEIGHTS = [5, 4, 2, 1]
MIN_SPAN_LEN = 4
FULL_SPAN_PROB = 0.65
ARM_MIN_OFFSET = 2
CROSSING_COUNT_WEIGHTS = [2, 3, 3]
SERPENTINE_MINOR_LO = 6
SERPENTINE_MINOR_HI = 5
CROSS_MINOR_MARGIN = 2
SPACED_VALUES_RETRIES = 60
GEN_RETRIES = 400
MAX_CROSSINGS = 10


def _pts(mask: np.ndarray) -> set[tuple[int, int]]:
    return {tuple(map(int, rc)) for rc in np.argwhere(mask)}


def _opposite(direction: tuple[int, int]) -> tuple[int, int]:
    return (-int(direction[0]), -int(direction[1]))


def _axis_of(direction: tuple[int, int]) -> tuple[int, int]:
    dr, dc = map(int, direction)
    assert dr != 0 or dc != 0
    if dr == 0:
        return (0, 1)
    if dc == 0:
        return (1, 0)
    if dr == dc:
        return (1, 1)
    assert dr == -dc
    return (1, -1)


def _orientation_name(axis: tuple[int, int] | None) -> str | None:
    if axis is None:
        return None
    return {
        (0, 1): "h",
        (1, 0): "v",
        (1, 1): "d1",
        (1, -1): "d2",
    }[axis]


def _neighbors8(pt: tuple[int, int]) -> list[tuple[int, int]]:
    r, c = pt
    return [(r + dr, c + dc) for dr, dc in DIRS8 if 0 <= r + dr < H and 0 <= c + dc < W]


def _dirs_in(pts: set[tuple[int, int]], pt: tuple[int, int]) -> list[tuple[int, int]]:
    r, c = pt
    return [(dr, dc) for dr, dc in DIRS8 if (r + dr, c + dc) in pts]


def _point_axis(pts: set[tuple[int, int]], pt: tuple[int, int]) -> tuple[int, int] | None:
    dirs = _dirs_in(pts, pt)
    if len(dirs) == 1:
        return _axis_of(dirs[0])
    if len(dirs) == 2 and dirs[1] == _opposite(dirs[0]):
        return _axis_of(dirs[0])
    return None


def _is_turn_or_endpoint(pts: set[tuple[int, int]], pt: tuple[int, int]) -> bool:
    dirs = _dirs_in(pts, pt)
    return len(dirs) == 1 or (len(dirs) == 2 and dirs[1] != _opposite(dirs[0]))


def _component_count(shape: tuple[int, int], pts: set[tuple[int, int]]) -> int:
    mask = np.zeros(shape, dtype=bool)
    for r, c in pts:
        mask[r, c] = True
    return len(components_mask(mask, connectivity=8))


def _segment(a: tuple[int, int], b: tuple[int, int]) -> list[tuple[int, int]]:
    r0, c0 = a
    r1, c1 = b
    dr = r1 - r0
    dc = c1 - c0
    if r0 == r1:
        step = (0, 1 if dc > 0 else -1)
        n = abs(dc)
    elif c0 == c1:
        step = (1 if dr > 0 else -1, 0)
        n = abs(dr)
    elif abs(dr) == abs(dc):
        step = (1 if dr > 0 else -1, 1 if dc > 0 else -1)
        n = abs(dr)
    else:
        raise ValueError(f"unsupported segment: {a} -> {b}")
    return [(r0 + i * step[0], c0 + i * step[1]) for i in range(n + 1)]


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
        deg = len(_dirs_in(pts, (r, c)))
        if deg > 2:
            return False
        if deg == 1 and not _on_boundary(r, c):
            return False
    return _component_count((H, W), pts) == 1


def _line_key(axis: tuple[int, int], pt: tuple[int, int]) -> int:
    r, c = pt
    if axis == (0, 1):
        return r
    if axis == (1, 0):
        return c
    if axis == (1, 1):
        return r - c
    return r + c


def _project(axis: tuple[int, int], pt: tuple[int, int]) -> int:
    r, c = pt
    if axis == (0, 1):
        return c
    return r


def _axis_support(mask: set[tuple[int, int]], pt: tuple[int, int], axis: tuple[int, int]) -> bool:
    r, c = pt
    dr, dc = axis
    return (r + dr, c + dc) in mask or (r - dr, c - dc) in mask


def _other_support(mask: set[tuple[int, int]], pt: tuple[int, int], axis: tuple[int, int]) -> bool:
    r, c = pt
    dr, dc = axis
    along = {(dr, dc), (-dr, -dc)}
    for ndr, ndc in DIRS8:
        if (ndr, ndc) in along:
            continue
        if (r + ndr, c + ndc) in mask:
            return True
    return False


def _cells_between(a: tuple[int, int], b: tuple[int, int]) -> list[tuple[int, int]]:
    seg = _segment(a, b)
    return seg[1:-1]


def _scale2x(grid: np.ndarray) -> np.ndarray:
    return np.repeat(np.repeat(grid, 2, axis=0), 2, axis=1)


def _render_small(
    logical: dict[int, set[tuple[int, int]]],
    comp_count: dict[int, int],
    forced_rays: dict[int, set[tuple[int, int]]],
) -> np.ndarray:
    out = np.full((H, W), BG, dtype=int)
    all_pts = set().union(*logical.values())
    for pt in all_pts:
        colors_here = [color for color in logical if pt in logical[color]]
        forced = [color for color in colors_here if pt in forced_rays.get(color, set())]
        if forced:
            out[pt] = forced[0]
        elif len(colors_here) == 1:
            out[pt] = colors_here[0]
        else:
            out[pt] = min(colors_here, key=lambda color: (-comp_count[color], color))
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Repair a one-cell-thick colored line network that may use horizontal, vertical, or 45-degree segments, then enlarge the repaired image by 2x2 pixels.

    Input structure:
    - A 12x12 grid with a dominant background.
    - Non-marker colors form one-cell-thick polyline fragments built only from horizontal, vertical, or 45-degree diagonal steps.
    - Marker colors are singleton pixels: red marks missing line cells, yellow marks a missing arm direction next to an endpoint/turn.

    Rule:
    - Assign each marker to the nearest visible line color along one of the 8 straight rays, preferring hits already supported along that same line family.
    - After recoloring markers, fill gaps between facing same-color cells when they lie on a shared row, column, or 45-degree diagonal and their local support indicates a broken segment.
    - For each yellow marker, remove the old straight arm on the far side of the pivot and extend the line through the marked direction until the grid edge.
    - Where colors overlap, yellow-controlled rays take priority; all other overlaps choose the color with more pre-repair fragments, breaking ties by lower color id.

    Key insight: The markers still identify which sparse fragments belong together, but the straight-line reasoning now has four admissible orientations instead of two, and the final answer is the repaired 12x12 network enlarged into 2x2 blocks.
    """
    g = np.asarray(grid, dtype=int)
    bg = int(get_bg(g))

    colors = set(map(int, g.ravel())) - {bg}
    assert colors, "expected foreground"

    marker_colors: set[int] = set()
    for color in colors:
        comps = components_mask(g == color, connectivity=8)
        if comps and max(len(comp) for comp in comps) == 1:
            marker_colors.add(int(color))

    line_colors = colors - marker_colors
    assert line_colors and marker_colors, "expected line colors and singleton markers"

    original = {color: _pts(g == color) for color in line_colors}
    assignment: dict[tuple[int, int], int] = {}
    assign_dir: dict[tuple[int, int], tuple[int, int]] = {}

    marker_mask = np.isin(g, list(marker_colors))
    for r0, c0 in _pts(marker_mask):
        candidates: list[tuple[int, int, int, int, int]] = []
        for dr, dc in DIRS8:
            r, c = r0 + dr, c0 + dc
            dist = 1
            while 0 <= r < H and 0 <= c < W:
                value = int(g[r, c])
                if value in line_colors:
                    axis = _axis_of((dr, dc))
                    supported = _axis_support(original[value], (r, c), axis)
                    candidates.append((dist, 0 if supported else 1, value, dr, dc))
                    break
                if value == bg or value in marker_colors:
                    r += dr
                    c += dc
                    dist += 1
                    continue
                break
        assert candidates, f"marker at {(r0, c0)} cannot see a line"
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

    def has_axis_support(color: int, pt: tuple[int, int], axis: tuple[int, int]) -> bool:
        return pt in marker_pts[color] or _axis_support(masks[color], pt, axis)

    def should_fill(color: int, a: tuple[int, int], b: tuple[int, int], axis: tuple[int, int]) -> bool:
        gap = abs(_project(axis, b) - _project(axis, a)) - 1
        if gap <= 0:
            return False
        a_axis = has_axis_support(color, a, axis)
        b_axis = has_axis_support(color, b, axis)
        if gap == 1:
            if a_axis or b_axis:
                return True
            return not (_other_support(masks[color], a, axis) and _other_support(masks[color], b, axis))
        if a in marker_pts[color] and b in marker_pts[color]:
            return False
        return a_axis and b_axis

    for color in sorted(line_colors):
        additions: set[tuple[int, int]] = set()
        for axis in LINE_DIRS:
            groups: dict[int, list[tuple[int, int]]] = {}
            for pt in masks[color]:
                groups.setdefault(_line_key(axis, pt), []).append(pt)
            for pts_on_line in groups.values():
                pts_on_line.sort(key=lambda pt: _project(axis, pt))
                for a, b in zip(pts_on_line, pts_on_line[1:]):
                    if should_fill(color, a, b, axis):
                        additions.update(_cells_between(a, b))
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
            assert pivot is not None, f"yellow control at {control} has no pivot"

            old_arm: set[tuple[int, int]] = set()
            r, c = pivot[0] + dr, pivot[1] + dc
            while (r, c) in masks[color]:
                old_arm.add((r, c))
                r += dr
                c += dc
            assert old_arm, f"yellow control at {control} has no old arm"
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

    out_small = _render_small(masks, comp_count, control_ray)
    return _scale2x(out_small)


def _rc(ax: int, major: int, minor: int) -> tuple[int, int]:
    return (major, minor) if ax == 0 else (minor, major)


def _dim(ax: int) -> int:
    return H if ax == 0 else W


def _random_span(limit: int) -> tuple[int, int]:
    if random.random() < 0.5:
        return 0, random.randint(MIN_SPAN_LEN, limit - 1)
    return random.randint(0, limit - MIN_SPAN_LEN - 1), limit - 1


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


def _diag_line_through(cell: tuple[int, int], slope: int) -> list[tuple[int, int]]:
    r, c = cell
    if slope == 1:
        delta = min(r, c)
        start = (r - delta, c - delta)
        steps = min(H - 1 - start[0], W - 1 - start[1])
        end = (start[0] + steps, start[1] + steps)
    else:
        delta = min(r, W - 1 - c)
        start = (r - delta, c + delta)
        steps = min(H - 1 - start[0], start[1])
        end = (start[0] + steps, start[1] - steps)
    return _segment(start, end)


def _crossing_cells(logical: dict[int, set[tuple[int, int]]]) -> list[tuple[int, int]]:
    cells = set().union(*logical.values())
    return [pt for pt in cells if sum(pt in logical[color] for color in logical) > 1]


def _valid_crossings(logical: dict[int, set[tuple[int, int]]]) -> bool:
    crossings = _crossing_cells(logical)
    if not 1 <= len(crossings) <= MAX_CROSSINGS:
        return False
    touched = {color: False for color in logical}
    for pt in crossings:
        colors_here = [color for color in logical if pt in logical[color]]
        if len(colors_here) != 2:
            return False
        orientations = [_orientation_name(_point_axis(logical[color], pt)) for color in colors_here]
        if None in orientations or len(set(orientations)) != 2:
            return False
        for color in colors_here:
            touched[color] = True
    return all(touched.values())


def _build_logical() -> tuple[
    dict[int, set[tuple[int, int]]],
    dict[int, list[set[tuple[int, int]]]],
    int,
] | None:
    colors = random.sample(LINE_PALETTE, 3)
    logical: dict[int, set[tuple[int, int]]] = {c: set() for c in colors}
    paths: dict[int, list[set[tuple[int, int]]]] = {c: [] for c in colors}

    row = random.randint(2, H - 3)
    col = random.randint(2, W - 3)
    dominant = colors[0]
    cross_color = colors[1]
    diag_color = colors[2]

    horizontal = {(row, c) for c in range(W)}
    vertical = {(r, col) for r in range(H)}
    logical[dominant].update(horizontal)
    logical[cross_color].update(vertical)
    paths[dominant].append(set(horizontal))
    paths[cross_color].append(set(vertical))

    diagonals: list[list[tuple[int, int]]] = []
    for c0 in range(W):
        diagonals.append(_diag_line_through((0, c0), 1))
        diagonals.append(_diag_line_through((0, c0), -1))
    for r0 in range(1, H):
        diagonals.append(_diag_line_through((r0, 0), 1))
        diagonals.append(_diag_line_through((r0, W - 1), -1))
    random.shuffle(diagonals)

    diag_cells: list[tuple[int, int]] | None = None
    for cells in diagonals:
        pts = set(cells)
        if len(pts) < 6 or (row, col) in pts:
            continue
        overlaps = [pt for pt in pts if pt in horizontal or pt in vertical]
        if not overlaps:
            continue
        if any(len(_dirs_in(pts, pt)) != 2 for pt in overlaps):
            continue
        trial_logical = {color: set(pts0) for color, pts0 in logical.items()}
        trial_logical[diag_color] = pts
        if _valid_crossings(trial_logical):
            diag_cells = cells
            break
    if diag_cells is None:
        return None

    logical[diag_color].update(diag_cells)
    paths[diag_color].append(set(diag_cells))
    return logical, paths, diag_color


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
                for out_dir in _dirs_in(comp, pivot):
                    ray = [pivot]
                    r, c = pivot
                    while True:
                        r += out_dir[0]
                        c += out_dir[1]
                        if (r, c) in comp:
                            ray.append((r, c))
                            continue
                        break
                    if len(ray) < 2 or not _on_boundary(*ray[-1]):
                        continue
                    if any(pt in crossings for pt in ray[1:]):
                        continue
                    old_dir = _opposite(out_dir)
                    old_cells = []
                    r, c = pivot[0] + old_dir[0], pivot[1] + old_dir[1]
                    while 0 <= r < H and 0 <= c < W and (r, c) not in occupied:
                        old_cells.append((r, c))
                        r += old_dir[0]
                        c += old_dir[1]
                    if not old_cells:
                        continue
                    old_len = random.randint(1, len(old_cells))
                    old_arm = set(old_cells[:old_len])
                    candidates.append((color, pivot, out_dir, set(ray), old_arm))
    random.shuffle(candidates)
    return candidates


def _choose_diag_red(
    diag_color: int,
    paths: dict[int, list[set[tuple[int, int]]]],
    crossings: set[tuple[int, int]],
    blocked: set[tuple[int, int]],
) -> tuple[int, tuple[int, int]] | None:
    comp = paths[diag_color][0]
    candidates = [
        pt
        for pt in comp
        if pt not in crossings
        and pt not in blocked
        and 0 < pt[0] < H - 1
        and 0 < pt[1] < W - 1
        and _orientation_name(_point_axis(comp, pt)) in {"d1", "d2"}
    ]
    random.shuffle(candidates)
    for pt in candidates:
        if any(nb in blocked for nb in [pt] + _neighbors8(pt)):
            continue
        return diag_color, pt
    return None


def _marker_isolation_ok(grid: np.ndarray) -> bool:
    markers = _pts(np.isin(grid, [2, 4]))
    for pt in markers:
        if any(nb in markers for nb in _neighbors8(pt)):
            return False
    for color in (2, 4):
        comps = components_mask(grid == color, connectivity=8)
        if any(len(comp) != 1 for comp in comps):
            return False
    return True


def _derive_input(
    logical: dict[int, set[tuple[int, int]]],
    paths: dict[int, list[set[tuple[int, int]]]],
    diag_color: int,
) -> tuple[np.ndarray, np.ndarray] | None:
    crossings = set(_crossing_cells(logical))

    blocked = set(crossings)
    red_info = _choose_diag_red(diag_color, paths, crossings, blocked)
    if red_info is None:
        return None
    red_color, red_pt = red_info
    blocked.update({red_pt, *_neighbors8(red_pt)})

    inp = np.full((H, W), BG, dtype=int)
    comp_count_masks: dict[int, set[tuple[int, int]]] = {c: set() for c in logical}
    yellow_rays: dict[int, set[tuple[int, int]]] = {c: set() for c in logical}
    removed_for_yellow: dict[int, set[tuple[int, int]]] = {c: set() for c in logical}
    old_arms: dict[int, set[tuple[int, int]]] = {c: set() for c in logical}
    yellow_by_pt: dict[tuple[int, int], int] = {}

    red_by_pt = {red_pt: red_color}
    crossing_display: dict[tuple[int, int], int] = {}
    for pt in crossings:
        colors_here = [color for color in logical if pt in logical[color]]
        safe = []
        for shown in colors_here:
            hidden = [color for color in colors_here if color != shown]
            if all(len(_dirs_in(logical[color], pt)) != 1 for color in hidden):
                safe.append(shown)
        crossing_display[pt] = random.choice(safe or colors_here)

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

    comp_count = {color: _component_count((H, W), pts) for color, pts in comp_count_masks.items()}
    target_small = _render_small(logical, comp_count, yellow_rays)
    target = _scale2x(target_small)

    if all(target_small[pt] == inp[pt] for pt in crossings):
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
    logical, paths, diag_color = built
    derived = _derive_input(logical, paths, diag_color)
    return derived[0] if derived is not None else None


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("failed to generate a valid fused faa9f03d input")
