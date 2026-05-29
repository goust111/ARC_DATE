# Combo variant of faa9f03d (index 5)
# combo_spec: D2/1+D6/0
# Name: 内部悬挂端点 + 仅填充固定长度以内间隙
#
# Design analysis
# - concretization:
#   - The fused scene still contains colored 1-pixel-wide orthogonal polylines plus singleton
#     red/yellow markers, but now at least one same-color facing pair is a real internal stop:
#     two red endpoints face each other across a 4-cell blank corridor and remain disconnected.
#   - The same sample also contains a different red-red bridge whose blank corridor is only 2-3
#     cells long; after recoloring the markers, that short corridor is repaired.
#   - So the visual problem is no longer "all facing gaps should be closed". The solver must infer
#     a fixed numeric fill cap and simultaneously accept that some interior dangling endpoints are
#     legitimate final structure.
# - behavior trace:
#   - Old base logic: after marker assignment, support geometry decides whether a gap closes, and
#     any surviving endpoint must end on the boundary or sit at a crossing.
#   - Fused logic: after the same marker assignment, gap filling ignores support and depends only
#     on `gap <= 3`; later validation allows true interior endpoints to remain.
#   - Example: two red markers at `(5,3)` and `(5,8)` with blanks at columns `4..7` are recolored
#     to the same line color, but the 4-cell corridor stays blank. Another pair at `(9,2)` and
#     `(9,6)` with a 3-cell corridor is filled completely.
# - complexity calibration:
#   - The threshold is fixed to 3 and every generated sample forces both sides of the contrast:
#     one repaired short bridge and one unrepaired long hanging gap. This makes the new rule
#     observable from a few examples instead of hiding behind rare edge cases.
#   - The long-gap witness uses one existing crossing color plus two extra internal stub components
#     of that same color, so the puzzle still lives in the original multi-color crossing context.
# - implicit assumptions audited:
#   - Marker assignment still depends on nearest row/column line-of-sight with axis-support tie
#     break. The generator preserves this by making mandatory red endpoints look into their own
#     component at distance 1.
#   - Yellow controls still need a pivot whose marked-side ray already belongs to one component;
#     hanging stubs are never used as yellow pivots, so the old yellow rewrite remains valid.
#   - The original final assertion "all endpoints must hit a boundary or crossing" is invalid
#     under D2/1 and is replaced with a weaker structural check: degree never exceeds 2, while
#     boundary endpoints, when they exist, still meet the edge perpendicularly.
# - placement legality:
#   - The hanging pair is searched inside a fully clear interior corridor; its two components, the
#     long blank gap, and a one-step neighborhood are reserved before other markers are placed.
#   - Short-bridge interiors, yellow rays, and hanging corridors are mutually blocked so the
#     generator cannot accidentally stack multiple explanations on the same cells.
# - visual observability:
#   - A human can compare the mandatory 3-cell repaired bridge and 4-cell unrepaired hanging gap
#     and infer the threshold directly from length, not from hidden support tests or color ids.
#   - The long unrepaired case ends in the middle of the board, so the relaxation from boundary
#     endpoints to internal endpoints is visible in the output itself.
# - cleanliness / inference:
#   - Only two line colors are used, which keeps crossings, rerouted yellow rays, short bridges,
#     and interior stops readable in the same 12x12 canvas.

from __future__ import annotations

import random

import numpy as np

from arc_utils import components_mask, get_bg


DIRS4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
MAX_FILL_GAP = 3


def _pts(mask: np.ndarray) -> set[tuple[int, int]]:
    return {tuple(map(int, rc)) for rc in np.argwhere(mask)}


def _axis_support(mask: set[tuple[int, int]], r: int, c: int, axis: int) -> bool:
    if axis == 0:
        return (r, c - 1) in mask or (r, c + 1) in mask
    return (r - 1, c) in mask or (r + 1, c) in mask


def _component_count(shape: tuple[int, int], pts: set[tuple[int, int]]) -> int:
    mask = np.zeros(shape, dtype=bool)
    for r, c in pts:
        mask[r, c] = True
    return len(components_mask(mask, connectivity=4))


def main(grid: np.ndarray) -> np.ndarray:
    """
    Repair a one-cell-thick colored line network by recoloring singleton markers, filling only short axis gaps, and redirecting yellow-marked arms to the edge while allowing true internal endpoints to survive.

    Input structure:
    - A rectangular grid with a dominant background.
    - Non-marker colors form one-cell-thick horizontal/vertical stroke fragments with crossings between colors.
    - Marker colors are exactly the colors whose 4-connected components are isolated single pixels.
    - Red markers replace missing turn/endpoint cells; some red pairs bound a short corridor that should close, while others mark genuine internal hanging endpoints separated by a longer corridor.
    - Yellow markers sit just beyond a line endpoint and indicate that the visible arm should be rerouted through the marker to the boundary.

    Rule:
    - Assign every marker to the nearest visible line color in row/column line-of-sight, preferring the sightline whose seen cell already has support along that same axis.
    - For each line color, whenever two consecutive same-color cells face each other in one row or column, fill the intervening cells iff the gap length is at most 3.
    - A yellow control erases the old straight arm on the far side of its endpoint and paints the same color from that endpoint through the control pixel out to the grid edge.
    - At crossings, yellow-control rays win; otherwise an original-vs-filled crossing uses the more fragmented pre-repair line color, while a crossing formed only by filled cells uses the lower color id.

    Key insight: The singleton markers still tell you which fragments belong together, but after recoloring them the decisive distinction is numeric: short same-axis breaks are repairs, whereas longer same-color separations can be intentional interior endpoints and must remain open.
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

    def should_fill(a: tuple[int, int], b: tuple[int, int], axis: int) -> bool:
        r1, c1 = a
        r2, c2 = b
        gap = c2 - c1 - 1 if axis == 0 else r2 - r1 - 1
        return 0 < gap <= MAX_FILL_GAP

    for color in sorted(line_colors):
        additions: set[tuple[int, int]] = set()
        for r in range(H):
            cols = sorted(c for rr, c in masks[color] if rr == r)
            for c1, c2 in zip(cols, cols[1:]):
                if should_fill((r, c1), (r, c2), axis=0):
                    additions.update((r, c) for c in range(c1 + 1, c2))

        for c in range(W):
            rows = sorted(r for r, cc in masks[color] if cc == c)
            for r1, r2 in zip(rows, rows[1:]):
                if should_fill((r1, c), (r2, c), axis=1):
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
            out[pt] = min(colors_here, key=lambda color: (-comp_count[color], color))
        else:
            bracketed = [color for color in colors_here if marker_brackets(color, pt)]
            if bracketed:
                out[pt] = min(bracketed, key=lambda color: (-comp_count[color], color))
            else:
                out[pt] = min(colors_here)

    # Assert: each visible color stays degree-2-or-less; boundary endpoints, when present, still
    # meet the edge perpendicularly, but interior endpoints are now allowed.
    for color in line_colors:
        for comp in components_mask(out == color, connectivity=4):
            pts_set = set(map(tuple, comp))
            for r, c in pts_set:
                nbrs = [(nr, nc) for nr, nc in [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]
                        if (nr, nc) in pts_set]
                assert len(nbrs) <= 2, f"junction at {(r, c)} color {color}"
                if len(nbrs) != 1:
                    continue
                if r == 0 or r == H - 1 or c == 0 or c == W - 1:
                    nr, nc = nbrs[0]
                    dr, dc = nr - r, nc - c
                    if not ((r in (0, H - 1)) and (c in (0, W - 1))):
                        perp = ((r == 0 and dr == 1) or (r == H - 1 and dr == -1)
                                or (c == 0 and dc == 1) or (c == W - 1 and dc == -1))
                        assert perp, f"boundary endpoint ({r},{c}) color {color} not perpendicular"

    # Assert: every line color crosses at least one other color.
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


# --------------- generator (combo D2/1 + D6/0) ---------------

# --- CFG: grid & palette ---
H = W = 12
BG = 0
LINE_PALETTE = [1, 3, 5, 6, 7, 8, 9]

# --- CFG: polyline shape ---
POLYLINE_PIECE_WEIGHTS = [5, 4, 2, 1]
MIN_SPAN_LEN = 4
FULL_SPAN_PROB = 0.65
ARM_MIN_OFFSET = 2

# --- CFG: crossings / colors ---
CROSSING_COUNT_WEIGHTS = [4, 3]
SERPENTINE_MINOR_LO = 6
SERPENTINE_MINOR_HI = 5
CROSS_MINOR_MARGIN = 2
N_COLORS = 2
MAX_CROSSINGS = 12

# --- CFG: markers / combo witnesses ---
MARKER_MODE_WEIGHTS = [1]  # red-only
YELLOW_COUNT_RANGE = (1, 2)
RED_COUNT_RANGE = (4, 6)
MIN_CELLS_FOR_RED = 2
BRIDGE_GAP_RANGE = (2, MAX_FILL_GAP)
BRIDGE_COUNT = 1
HANG_GAP = MAX_FILL_GAP + 1
HANG_SEG_CHOICES = ((2, 3), (3, 2), (3, 3))
HANG_PAIR_COUNT = 1

# --- CFG: output validation ---
COMPONENTS_PER_COLOR = (1, 4)
TOTAL_COMPONENTS = (4, 6)
FILL_RANGE = (33, 55)

# --- CFG: retry budgets ---
SPACED_VALUES_RETRIES = 60
EXTRA_POLYLINE_RETRIES = 80
HANGING_RETRIES = 80
GEN_RETRIES = 400


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


def _valid_open_path(cells: list[tuple[int, int]]) -> bool:
    pts = set(cells)
    if len(pts) != len(cells) or len(pts) < 2:
        return False
    if _gen_component_count(pts) != 1:
        return False
    for r, c in pts:
        if not (0 <= r < H and 0 <= c < W):
            return False
        if len(_dirs_in(pts, (r, c))) > 2:
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


def _build_arm(ax: int, main_coord: int, cross_lo: int, cross_hi: int,
               is_start: bool) -> list[tuple[int, int]]:
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
                        (pivot[0] + old_dir[0], pivot[1] + old_dir[1]), old_dir
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


def _bridge_candidates(
    paths: dict[int, list[set[tuple[int, int]]]],
    crossings: set[tuple[int, int]],
) -> list[tuple[int, tuple[int, int], tuple[int, int], set[tuple[int, int]]]]:
    candidates: list[tuple[int, tuple[int, int], tuple[int, int], set[tuple[int, int]]]] = []
    for color, comps in paths.items():
        for comp in comps:
            for start in comp:
                if start in crossings or not _is_turn_or_endpoint(comp, start):
                    continue
                for dr, dc, orient in ((0, 1, "h"), (1, 0, "v")):
                    prev = (start[0] - dr, start[1] - dc)
                    nxt = (start[0] + dr, start[1] + dc)
                    if prev in comp or nxt not in comp:
                        continue
                    cells = [start]
                    cur = nxt
                    while cur in comp:
                        cells.append(cur)
                        if cur in crossings:
                            break
                        if cur != start and _is_turn_or_endpoint(comp, cur):
                            gap = len(cells) - 2
                            if BRIDGE_GAP_RANGE[0] <= gap <= BRIDGE_GAP_RANGE[1]:
                                candidates.append((color, start, cur, set(cells[1:-1])))
                            break
                        if _orientation(comp, cur) != orient:
                            break
                        cur = (cur[0] + dr, cur[1] + dc)
    random.shuffle(candidates)
    return candidates


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


def _corridor_clear(occupied: set[tuple[int, int]], cells: set[tuple[int, int]]) -> bool:
    for pt in cells:
        if pt in occupied:
            return False
        if any(nb in occupied for nb in _neighbors(pt)):
            return False
    return True


def _add_hanging_pairs(
    logical: dict[int, set[tuple[int, int]]],
    paths: dict[int, list[set[tuple[int, int]]]],
) -> list[tuple[int, tuple[int, int], tuple[int, int], set[tuple[int, int]]]]:
    infos: list[tuple[int, tuple[int, int], tuple[int, int], set[tuple[int, int]]]] = []
    colors = list(logical)
    for _ in range(HANG_PAIR_COUNT):
        occupied = set().union(*logical.values())
        placed = False
        for _retry in range(HANGING_RETRIES):
            color = random.choice(colors)
            horizontal = random.random() < 0.5
            seg1, seg2 = random.choice(HANG_SEG_CHOICES)
            if horizontal:
                r = random.randint(1, H - 2)
                start = random.randint(1, 10 - (seg1 + HANG_GAP + seg2) + 1)
                left = {(r, c) for c in range(start, start + seg1)}
                right = {(r, c) for c in range(start + seg1 + HANG_GAP, start + seg1 + HANG_GAP + seg2)}
                gap_cells = {(r, c) for c in range(start + seg1, start + seg1 + HANG_GAP)}
                a = (r, start + seg1 - 1)
                b = (r, start + seg1 + HANG_GAP)
            else:
                c = random.randint(1, W - 2)
                start = random.randint(1, 10 - (seg1 + HANG_GAP + seg2) + 1)
                left = {(r, c) for r in range(start, start + seg1)}
                right = {(r, c) for r in range(start + seg1 + HANG_GAP, start + seg1 + HANG_GAP + seg2)}
                gap_cells = {(r, c) for r in range(start + seg1, start + seg1 + HANG_GAP)}
                a = (start + seg1 - 1, c)
                b = (start + seg1 + HANG_GAP, c)
            comp1 = set(sorted(left))
            comp2 = set(sorted(right))
            if not _valid_open_path(sorted(comp1)) or not _valid_open_path(sorted(comp2)):
                continue
            reserve = left | right | gap_cells
            if not _corridor_clear(occupied, reserve):
                continue
            logical[color].update(left | right)
            paths[color].append(set(left))
            paths[color].append(set(right))
            infos.append((color, a, b, gap_cells))
            placed = True
            break
        if not placed:
            return []
    return infos


TRANSFORMS = ("id", "rot90", "rot180", "rot270", "flip_h", "flip_v", "diag", "anti")


def _transform_pt(pt: tuple[int, int], name: str) -> tuple[int, int]:
    r, c = pt
    if name == "id":
        return r, c
    if name == "rot90":
        return c, W - 1 - r
    if name == "rot180":
        return H - 1 - r, W - 1 - c
    if name == "rot270":
        return H - 1 - c, r
    if name == "flip_h":
        return r, W - 1 - c
    if name == "flip_v":
        return H - 1 - r, c
    if name == "diag":
        return c, r
    if name == "anti":
        return H - 1 - c, W - 1 - r
    raise ValueError(name)


def _transform_pts(pts: set[tuple[int, int]], name: str) -> set[tuple[int, int]]:
    return {_transform_pt(pt, name) for pt in pts}


def _build_logical() -> tuple[
    dict[int, set[tuple[int, int]]],
    dict[int, list[set[tuple[int, int]]]],
    int,
    list[tuple[int, tuple[int, int], tuple[int, int], set[tuple[int, int]]]],
    list[tuple[int, tuple[int, int], tuple[int, int], set[tuple[int, int]]]],
] | None:
    colors = random.sample(LINE_PALETTE, N_COLORS)
    bridge_color, dominant = colors[0], colors[1]
    xf = random.choice(TRANSFORMS)

    bridge_main = set(_path([(0, 1), (4, 1), (4, 4), (7, 4), (7, 7), (11, 7)]))
    dominant_main = set(_path([(2, 0), (2, 3), (11, 3)]))
    hang_hi = {(r, 10) for r in range(1, 4)}
    hang_lo = {(r, 10) for r in range(8, 11)}
    hanging_infos = [(dominant, (3, 10), (8, 10), {(4, 10), (5, 10), (6, 10), (7, 10)})]
    bridge_infos = [(bridge_color, (7, 4), (7, 7), {(7, 5), (7, 6)})]

    bridge_main = _transform_pts(bridge_main, xf)
    dominant_main = _transform_pts(dominant_main, xf)
    hang_hi = _transform_pts(hang_hi, xf)
    hang_lo = _transform_pts(hang_lo, xf)
    hanging_infos = [
        (color, _transform_pt(a, xf), _transform_pt(b, xf), _transform_pts(gap_cells, xf))
        for color, a, b, gap_cells in hanging_infos
    ]
    bridge_infos = [
        (color, _transform_pt(a, xf), _transform_pt(b, xf), _transform_pts(interior, xf))
        for color, a, b, interior in bridge_infos
    ]

    if not (_valid_path(sorted(bridge_main)) and _valid_path(sorted(dominant_main))):
        return None
    if not (_valid_open_path(sorted(hang_hi)) and _valid_open_path(sorted(hang_lo))):
        return None

    logical = {
        bridge_color: set(bridge_main),
        dominant: set(dominant_main | hang_hi | hang_lo),
    }
    paths = {
        bridge_color: [set(bridge_main)],
        dominant: [set(dominant_main), set(hang_hi), set(hang_lo)],
    }
    if not _valid_crossings(logical, dominant):
        return None
    return logical, paths, dominant, hanging_infos, bridge_infos


def _marker_isolation_ok(grid: np.ndarray) -> bool:
    for color in (2, 4):
        for pt in map(tuple, np.argwhere(grid == color)):
            if any(grid[nb] == color for nb in _neighbors(pt)):
                return False
    return True


def _display_component_counts(target: np.ndarray, colors: list[int]) -> dict[int, int]:
    return {c: len(components_mask(target == c, connectivity=4)) for c in colors}


def _count_internal_endpoints(target: np.ndarray, colors: list[int]) -> int:
    total = 0
    for color in colors:
        for comp in components_mask(target == color, connectivity=4):
            pts = set(map(tuple, comp))
            for r, c in pts:
                if _on_boundary(r, c):
                    continue
                if len(_dirs_in(pts, (r, c))) == 1:
                    total += 1
    return total


def _derive_input(
    logical: dict[int, set[tuple[int, int]]],
    paths: dict[int, list[set[tuple[int, int]]]],
    dominant: int,
    hanging_infos: list[tuple[int, tuple[int, int], tuple[int, int], set[tuple[int, int]]]],
    bridge_infos: list[tuple[int, tuple[int, int], tuple[int, int], set[tuple[int, int]]]],
) -> tuple[np.ndarray, np.ndarray] | None:
    crossings = set(_crossing_cells(logical))
    mode = "red"
    yellow_count = 0
    red_count = random.randint(*RED_COUNT_RANGE)

    blocked = set(crossings)
    hanging_red_markers: list[tuple[int, tuple[int, int]]] = []
    for color, a, b, gap_cells in hanging_infos:
        hanging_red_markers.extend([(color, a), (color, b)])
        blocked.update(gap_cells | {a, b} | set(_neighbors(a)) | set(_neighbors(b)))

    chosen_bridge_infos: list[tuple[int, tuple[int, int], tuple[int, int], set[tuple[int, int]]]] = []
    for cand in bridge_infos:
        color, a, b, interior = cand
        if a in blocked or b in blocked or any(pt in blocked for pt in interior):
            return None
        chosen_bridge_infos.append(cand)
        blocked.update(interior | {a, b} | set(_neighbors(a)) | set(_neighbors(b)))
    if len(chosen_bridge_infos) != BRIDGE_COUNT:
        return None

    yellow_infos = []
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

    bridge_red_markers = [(color, a) for color, a, _b, _interior in chosen_bridge_infos]
    bridge_red_markers += [(color, b) for color, _a, b, _interior in chosen_bridge_infos]
    forced_red_markers = hanging_red_markers + bridge_red_markers
    extra_red_count = red_count - len(forced_red_markers)
    if extra_red_count < 0:
        return None
    extra_red_markers = _choose_red_markers(logical, paths, blocked, extra_red_count) if extra_red_count else []
    if extra_red_count and not extra_red_markers:
        return None
    red_markers = forced_red_markers + extra_red_markers

    inp = np.full((H, W), BG, dtype=int)
    comp_count_masks: dict[int, set[tuple[int, int]]] = {c: set() for c in logical}
    crossing_display: dict[tuple[int, int], int] = {}
    for pt in crossings:
        colors = [color for color in logical if pt in logical[color]]
        if dominant not in colors:
            return None
        crossing_display[pt] = colors[0] if colors[1] == dominant else colors[1]

    removed_for_bridge: dict[int, set[tuple[int, int]]] = {c: set() for c in logical}
    removed_for_yellow: dict[int, set[tuple[int, int]]] = {c: set() for c in logical}
    old_arms: dict[int, set[tuple[int, int]]] = {c: set() for c in logical}
    yellow_markers: list[tuple[int, tuple[int, int]]] = []

    for color, _a, _b, interior in chosen_bridge_infos:
        removed_for_bridge[color].update(interior)
    for color, pivot, out_dir, target_ray, old_arm in yellow_infos:
        marker = (pivot[0] + out_dir[0], pivot[1] + out_dir[1])
        removed_for_yellow[color].update(target_ray - {pivot})
        old_arms[color].update(old_arm)
        yellow_markers.append((color, marker))

    red_by_pt = {pt: color for color, pt in red_markers}
    yellow_by_pt = {pt: color for color, pt in yellow_markers}

    for color, pts in logical.items():
        for pt in pts:
            if pt in removed_for_bridge[color]:
                continue
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

    try:
        target = main(inp)
    except Exception:
        return None

    colors = sorted(logical)
    bridge_filled = 0
    for color, _a, _b, interior in chosen_bridge_infos:
        if interior and all(int(target[pt]) == color for pt in interior):
            bridge_filled += 1
    if bridge_filled != len(chosen_bridge_infos):
        return None

    hanging_kept = 0
    for color, a, b, gap_cells in hanging_infos:
        if int(target[a]) == color and int(target[b]) == color and all(int(target[pt]) == BG for pt in gap_cells):
            hanging_kept += 1
    if hanging_kept != len(hanging_infos):
        return None

    if _count_internal_endpoints(target, colors) < 2 * len(hanging_infos):
        return None

    if not any(int(target[pt]) != int(inp[pt]) for pt in set(red_by_pt) | set(yellow_by_pt) | crossings):
        return None

    counts = _display_component_counts(target, colors)
    if not all(COMPONENTS_PER_COLOR[0] <= v <= COMPONENTS_PER_COLOR[1] for v in counts.values()):
        return None
    if not TOTAL_COMPONENTS[0] <= sum(counts.values()) <= TOTAL_COMPONENTS[1]:
        return None
    if not FILL_RANGE[0] <= int(np.count_nonzero(target)) <= FILL_RANGE[1]:
        return None

    return inp, target


def _gen_attempt() -> np.ndarray | None:
    built = _build_logical()
    if built is None:
        return None
    logical, paths, dominant, hanging_infos, bridge_infos = built
    derived = _derive_input(logical, paths, dominant, hanging_infos, bridge_infos)
    return derived[0] if derived is not None else None


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("failed to generate a valid input for faa9f03d combo D2/1+D6/0")
