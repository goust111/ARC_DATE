# Variant of faa9f03d
# Dimension: D5 — 交叉处的颜色消歧规则 (transform)
# Change: 黄色控制射线优先；否则原始交叉处选碎片度更高的颜色（pre-repair连通分量更多者胜出），纯填充交叉选较小颜色ID... → 交叉处选碎片度更低的颜色（更完整者胜出）

from __future__ import annotations

import numpy as np

from arc_utils import components_mask, get_bg
import random

# Design analysis for this D5 variant
# - Concrete behavior: the input still contains sparse rectilinear line fragments plus red/yellow singleton
#   markers. The only semantic change is at cells where two repaired colors claim the same crossing.
#   Yellow-ray crossings still force the ray color. Non-yellow crossings now show the color whose
#   pre-repair mask is more complete, measured by having fewer 4-connected components after markers are
#   assigned but before gap filling. Example: if a visible crossing belongs to colors A and B and the
#   pre-repair counts are A:3, B:1, the original task output A there while this variant outputs B.
# - Complexity calibration: the raw solver change is tiny, but the generator cannot stay unchanged.
#   With the old construction, the color displayed in the input often stays the less fragmented one, so
#   flipping the rule would produce the same visible crossing and the variant would be unobservable.
#   To expose D5, generation now insists on at least one "inversion crossing" where the input shows the
#   old fragmented-rule winner and the output flips to the more complete rival.
# - Hidden assumptions audit: `main()` computes component counts from marker-colored pre-repair masks and
#   uses those counts only in the crossing-resolution branches (`original_crossing` and bracketed pure-fill
#   crossings). All geometric assumptions for marker assignment, gap filling, yellow arm erasure, boundary
#   endpoints, and crossing participation remain valid, so only the comp-count ordering is reversed there.
# - Placement legality: the generator still constructs full logical polylines first, then removes crossing
#   ownership, red cells, and yellow arms while preserving isolated marker pixels, non-overlapping old arms,
#   boundary-perpendicular endpoints, and solver verifiability through `main(input) == target`.
# - Visual observability: the variant avoids hidden numeric tricks. The cue remains purely visual: the
#   output crossing follows the line that looks less broken overall, not the one shattered into more
#   pieces. Samples are rejected unless at least one crossing would have been colored differently by the
#   original fragmented-first rule.
# - Cleanliness: no extra object types are introduced. The grid still contains only background, line
#   colors, and the same red/yellow singleton markers, so the scene stays readable despite overlaps.


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


def _choose_complete(colors_here: list[int] | set[int], comp_count: dict[int, int]) -> int:
    return min(colors_here, key=lambda color: (comp_count[color], color))


def _choose_fragmented(colors_here: list[int] | set[int], comp_count: dict[int, int]) -> int:
    return min(colors_here, key=lambda color: (-comp_count[color], color))


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reconstruct a one-cell-thick rectilinear line network by recoloring singleton markers, filling axis-aligned gaps, and redirecting yellow-controlled arms to the border.

    Input structure:
    - A rectangular grid contains a dominant background plus several non-marker line colors.
    - Each line color appears as sparse one-cell-thick horizontal/vertical fragments from a larger rectilinear polyline.
    - Marker colors are precisely the colors whose every 4-connected component is a singleton pixel.
    - Red singletons stand in for removed line cells, while yellow singletons sit beside an endpoint and indicate an arm flip.

    Rule:
    - Assign each marker to the nearest seen line color along its row or column, preferring a candidate already supported along that axis.
    - For every color, fill the cells between facing same-color endpoints in a shared row or column when the local endpoint geometry supports a repair.
    - A yellow control removes the old arm beyond its pivot endpoint and extends the same color from the pivot through the marker to the grid edge.
    - When repaired colors overlap, yellow-ray claims still override everything else; otherwise any comp-count-based crossing now shows the less fragmented pre-repair color (fewer connected components), while purely fill-only crossings with no marker bracketing still fall back to the smaller color id.

    Key insight: The markers still tell you which fragments should be reconnected and which arm a yellow control should reverse, but the crossing color must now be taken from the line that already looks more intact before repair, not from the more shattered one.
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
            out[pt] = _choose_complete(colors_here, comp_count)
        else:
            bracketed = [color for color in colors_here if marker_brackets(color, pt)]
            if bracketed:
                out[pt] = _choose_complete(bracketed, comp_count)
            else:
                out[pt] = min(colors_here)

    # Assert: every visible endpoint is on the boundary (or at a crossing)
    for color in line_colors:
        for comp in components_mask(out == color, connectivity=4):
            pts_set = set(map(tuple, comp))
            for r, c in pts_set:
                nbrs = [(nr, nc) for nr, nc in [(r-1,c),(r+1,c),(r,c-1),(r,c+1)]
                        if (nr, nc) in pts_set]
                if len(nbrs) != 1:
                    continue
                on_boundary = (r == 0 or r == H - 1 or c == 0 or c == W - 1)
                if on_boundary:
                    # Perpendicularity check
                    nr, nc = nbrs[0]
                    dr, dc = nr - r, nc - c
                    if not ((r in (0, H-1)) and (c in (0, W-1))):
                        perp = ((r == 0 and dr == 1) or (r == H-1 and dr == -1)
                                or (c == 0 and dc == 1) or (c == W-1 and dc == -1))
                        assert perp, f"boundary endpoint ({r},{c}) color {color} not perpendicular"
                    continue
                adj_other = any(
                    0 <= r+dr < H and 0 <= c+dc < W
                    and int(out[r+dr, c+dc]) not in (bg, color)
                    for dr, dc in DIRS4)
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
                if (((r, c-1) in color_cells[b] and (r, c+1) in color_cells[b])
                        or ((r-1, c) in color_cells[b] and (r+1, c) in color_cells[b])):
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
POLYLINE_PIECE_WEIGHTS = [5, 4, 2, 1]  # weights for 1-4 segments per polyline
MIN_SPAN_LEN = 4                        # minimum length of a polyline's primary span
FULL_SPAN_PROB = 0.65                    # probability path_through extends to grid edge
ARM_MIN_OFFSET = 2                       # minimum distance between arm column and main line

# --- CFG: serpentine & crossings ---
CROSSING_COUNT_WEIGHTS = [2, 3, 3, 2]  # weights for 1-4 crossing rows in serpentine
SERPENTINE_MINOR_LO = 6                 # max value for minor-axis lower bound
SERPENTINE_MINOR_HI = 5                 # max value for minor-axis upper offset from 0
CROSS_MINOR_MARGIN = 2                  # margin from serpentine edges to crossing column
MAX_COLORS = 4                          # maximum number of line colors
MAX_CROSSINGS = 12                      # upper bound on crossing cell count

# --- CFG: markers ---
MARKER_MODE_WEIGHTS = [5, 0, 4]  # weights for red-only / yellow-only / both; red is required to expose D5
YELLOW_COUNT_RANGE = (1, 2)
RED_COUNT_RANGE = (2, 6)
MIN_CELLS_FOR_RED = 2             # color must have > this many cells to place a red marker
MIN_RED_ON_DOMINANT = 1           # keep the dominant crossing color visibly more fragmented in some samples

# --- CFG: output validation ---
COMPONENTS_PER_COLOR = (1, 4)
TOTAL_COMPONENTS = (4, 6)
FILL_RANGE = (33, 55)            # non-bg cell count in output

# --- CFG: retry budgets ---
SPACED_VALUES_RETRIES = 60
EXTRA_POLYLINE_RETRIES = 80
GEN_RETRIES = 450


def _neighbors(pt: tuple[int, int]) -> list[tuple[int, int]]:
    r, c = pt
    return [(r + dr, c + dc) for dr, dc in DIRS4 if 0 <= r + dr < H and 0 <= c + dc < W]


def _segment(a: tuple[int, int], b: tuple[int, int]) -> list[tuple[int, int]]:
    r0, c0 = a; r1, c1 = b
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
            if (r in (0, H-1) and nr == r) or (c in (0, W-1) and nc == c):
                return False
        nbrs = _dirs_in(pts, (r, c))
        if len(nbrs) != 1:
            continue
        if (r in (0, H-1)) and (c in (0, W-1)):
            continue
        dr, dc = nbrs[0]
        if not ((r == 0 and dr == 1) or (r == H-1 and dr == -1)
                or (c == 0 and dc == 1) or (c == W-1 and dc == -1)):
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
    """Build waypoint list for a random polyline with axis ax (0=vert-primary, 1=horiz-primary).
    Primary span is along dim(ax), bends alternate into dim(1-ax)."""
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
    # Extend non-boundary endpoints to nearest edge (perpendicular to primary axis)
    if not _on_boundary(*pts[0]):
        r0, c0 = pts[0]
        if ax == 0:  # vert-primary → extend horizontally
            pts.insert(0, (r0, random.choice([0, W - 1])))
        else:  # horiz-primary → extend vertically
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
    """Build L-shaped arm waypoints for _path_crossing_all / _path_through.
    ax=0: arm extends vertically (row-axis), ax=1: horizontally.
    Returns waypoints to prepend (is_start) or append (!is_start)."""
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
    # If endpoints still not on boundary after arm attempts, extend to full span
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
    ax = random.randint(0, 1)  # 0 = vertical serpentine, 1 = horizontal
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

    # Extend endpoints outward to the minor-axis boundary
    if not _on_boundary(*pts[0]):
        r0, c0 = pts[0]
        if ax == 0:  # vert serpentine → extend col outward
            ext_c = 0 if c0 == m0 else W - 1
            pts.insert(0, (r0, ext_c))
        else:  # horiz serpentine → extend row outward
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
        r += dr; c += dc
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
                        r += old_dir[0]; c += old_dir[1]
                    if not old_cells:
                        continue
                    old_len = random.randint(1, len(old_cells))
                    old_arm = set(list(_ray_from(
                        (pivot[0] + old_dir[0], pivot[1] + old_dir[1]), old_dir))[:old_len])
                    if old_arm and old_arm.issubset(old_cells):
                        candidates.append((color, pivot, out_dir, target_ray, old_arm))
    random.shuffle(candidates)
    return candidates


def _choose_red_markers(
    logical: dict[int, set[tuple[int, int]]],
    paths: dict[int, list[set[tuple[int, int]]]],
    blocked: set[tuple[int, int]],
    count: int,
    preferred_color: int | None = None,
    min_preferred: int = 0,
) -> list[tuple[int, tuple[int, int]]]:
    candidates = [(color, pt) for color, comps in paths.items() for comp in comps
                  for pt in comp if pt not in blocked and _is_turn_or_endpoint(comp, pt)]
    random.shuffle(candidates)
    chosen: list[tuple[int, tuple[int, int]]] = []
    used: set[tuple[int, int]] = set(blocked)
    per_color_left = {color: len(logical[color]) for color in logical}
    preferred_hits = 0
    ordered = [
        [item for item in candidates if item[0] == preferred_color],
        [item for item in candidates if item[0] != preferred_color],
    ] if preferred_color is not None else [candidates]
    for bucket in ordered:
        for color, pt in bucket:
            if len(chosen) == count:
                break
            if per_color_left[color] <= MIN_CELLS_FOR_RED:
                continue
            if any(nb in used for nb in [pt] + _neighbors(pt)):
                continue
            chosen.append((color, pt))
            used.add(pt)
            per_color_left[color] -= 1
            if color == preferred_color:
                preferred_hits += 1
        if len(chosen) == count:
            break
    if preferred_hits < min_preferred:
        return []
    return chosen if len(chosen) == count else []


def _find_crossing_point(
    logical: dict[int, set[tuple[int, int]]],
) -> tuple[tuple[int, int], str] | None:
    candidates = []
    for color, pts in logical.items():
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
    desired_crossings = random.choices(range(1, len(CROSSING_COUNT_WEIGHTS) + 1), weights=CROSSING_COUNT_WEIGHTS)[0]
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
    *,
    prefer_complete: bool = True,
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
            chooser = _choose_complete if prefer_complete else _choose_fragmented
            out[pt] = chooser(colors_here, comp_count)
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

    red_markers = _choose_red_markers(
        logical,
        paths,
        blocked,
        red_count,
        preferred_color=dominant if red_count else None,
        min_preferred=MIN_RED_ON_DOMINANT if red_count else 0,
    ) if red_count else []
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

    red_by_pt = {pt: color for color, pt in red_markers}
    yellow_by_pt = {pt: color for color, pt in yellow_markers}

    crossing_pairs: dict[tuple[int, int], list[int]] = {}
    for pt in crossings:
        colors = [color for color in logical if pt in logical[color]]
        if dominant not in colors:
            return None
        crossing_pairs[pt] = colors

    def build_candidate(
        crossing_display: dict[tuple[int, int], int],
    ) -> tuple[np.ndarray, dict[int, set[tuple[int, int]]]] | None:
        inp = np.full((H, W), BG, dtype=int)
        comp_count_masks: dict[int, set[tuple[int, int]]] = {c: set() for c in logical}
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
            if not any(inp[nb] == color for nb in _neighbors(pt)):
                return None
        return inp, comp_count_masks

    candidate_displays: list[dict[tuple[int, int], int]] = []
    if crossings:
        all_dominant = {pt: dominant for pt in crossings}
        all_other = {
            pt: crossing_pairs[pt][0] if crossing_pairs[pt][1] == dominant else crossing_pairs[pt][1]
            for pt in crossings
        }
        candidate_displays.extend([all_dominant, all_other])
        for _ in range(32):
            candidate_displays.append({
                pt: random.choice(crossing_pairs[pt])
                for pt in crossings
            })

    best: tuple[int, int, int, np.ndarray, np.ndarray] | None = None
    for crossing_display in candidate_displays:
        built = build_candidate(crossing_display)
        if built is None:
            continue
        inp, comp_count_masks = built
        comp_count = {color: _gen_component_count(pts) for color, pts in comp_count_masks.items()}
        target = _render_target(logical, comp_count, yellow_rays, prefer_complete=True)
        old_target = _render_target(logical, comp_count, yellow_rays, prefer_complete=False)
        inversion_count = 0
        changed_crossings = 0
        eligible_crossings = 0
        for pt in crossings:
            forced = {color for color in crossing_pairs[pt] if pt in yellow_rays.get(color, set())}
            if forced:
                continue
            eligible_crossings += 1
            old_winner = _choose_fragmented(crossing_pairs[pt], comp_count)
            new_winner = _choose_complete(crossing_pairs[pt], comp_count)
            if target[pt] != inp[pt]:
                changed_crossings += 1
            if old_winner != new_winner and inp[pt] == old_winner and int(target[pt]) == new_winner:
                inversion_count += 1
        if eligible_crossings == 0 or inversion_count == 0 or changed_crossings == 0:
            continue
        old_new_diff = int(np.count_nonzero(old_target != target))
        score = (inversion_count, changed_crossings, old_new_diff)
        if best is None or score > best[:3]:
            best = (*score, inp, target)
    if best is None:
        return None
    _inversion_count, _changed_crossings, _old_new_diff, inp, target = best

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
    logical, paths, dominant = built
    derived = _derive_input(logical, paths, dominant)
    return derived[0] if derived is not None else None


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("failed to generate a valid faa9f03d input")
