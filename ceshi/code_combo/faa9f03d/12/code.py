# Combo variant of faa9f03d (index 12)
# combo_spec: D6/2+D15/0
# Name: 仅标记夹持时填充 + 支撑检查扩大为2格
#
# Design analysis for the fused variant
# - concretization:
#   - The original task can fill a straight row/column gap even when the two visible
#     endpoints are ordinary line cells, as long as they look structurally supported.
#     It also explicitly rejects the case where both ends of a long gap are markers.
#   - The fused rule inverts that. A gap is fillable only when both ends are markers,
#     and the surrounding line evidence is now checked with a 2-cell axis window.
#   - Concrete trace on one row:
#       output:  A A X R . . R A A
#       input:   A A X 2 . . 2 A A
#     Here `X` is a crossing cell logically belonging to color A but visibly shown in
#     another color. Old logic refuses the 2..2 gap because both ends are markers.
#     A plain "markers-only" variant with 1-step support also refuses it because the
#     left marker sees no adjacent visible A. The fused rule succeeds because the left
#     marker may validate against the A two cells away across the hidden crossing.
# - complexity calibration:
#   - The generator always builds at least one such marker-clamped corridor, and at
#     least one of its two ends must rely on the 2-cell support check. This makes the
#     D15/0 contribution observable instead of being a dormant tie-break.
#   - Yellow rerouting remains optional but secondary; it preserves the original task's
#     multi-object context without obscuring the main clamped-gap phenomenon.
# - implicit assumptions audited:
#   - Marker assignment still uses nearest row/column line-of-sight. The support-aware
#     tie-break now also uses the 2-cell span, because hidden crossings can remove the
#     immediately adjacent visible support.
#   - Markers no longer count as their own support. For clamped filling, support must
#     come from actual non-marker line structure around the marker.
#   - The original gap=1 special-case based on perpendicular support is invalid here:
#     once markers are mandatory, a one-cell gap is treated the same as any longer gap.
#   - Crossing resolution still depends on pre-fill fragmentation. The generator keeps
#     its `comp_count_masks` aligned with the solver by omitting blanked clamp interiors
#     and including marker cells under their assigned color.
# - placement legality:
#   - Clamp endpoints are chosen from one straight logical run, never on a crossing
#     cell, and their interior corridor contains no crossings or yellow artifacts.
#   - At least one outer support cell next to a clamp end is a hidden crossing, and the
#     next cell beyond it remains visibly the clamp color; this is what activates D15/0.
#   - Yellow markers and extra red markers are blocked away from the clamp corridor so
#     that markers remain isolated, non-overlapping, and visually readable.
# - observability and cleanliness:
#   - In every generated example there is an obvious red-red corridor whose interior is
#     restored in the output. The clue that matters is not merely "red pair => fill",
#     but that one end may look under-supported until you inspect two cells away.
#   - The scene stays tidy because the new evidence comes from existing crossings on the
#     same polyline rather than from extra decorative objects or invisible color-order
#     conventions.

from __future__ import annotations

import random

import numpy as np

from arc_utils import components_mask, get_bg


DIRS4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
SUPPORT_SPAN = 2


def _pts(mask: np.ndarray) -> set[tuple[int, int]]:
    return {tuple(map(int, rc)) for rc in np.argwhere(mask)}


def _axis_support(
    mask: set[tuple[int, int]],
    r: int,
    c: int,
    axis: int,
    max_dist: int = 1,
) -> bool:
    if axis == 0:
        return any((r, c - d) in mask or (r, c + d) in mask for d in range(1, max_dist + 1))
    return any((r - d, c) in mask or (r + d, c) in mask for d in range(1, max_dist + 1))


def _perp_support(mask: set[tuple[int, int]], r: int, c: int, axis: int) -> bool:
    if axis == 0:
        return (r - 1, c) in mask or (r + 1, c) in mask
    return (r, c - 1) in mask or (r, c + 1) in mask


def _component_count(shape: tuple[int, int], pts: set[tuple[int, int]]) -> int:
    mask = np.zeros(shape, dtype=bool)
    for r, c in pts:
        mask[r, c] = True
    return len(components_mask(mask, connectivity=4))


def _solve(grid: np.ndarray, support_span: int, fill_mode: str) -> np.ndarray:
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
                    supported = _axis_support(original[value], r, c, axis, max_dist=support_span)
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
    line_only = {color: masks[color] - marker_pts[color] for color in line_colors}

    def has_axis_support(color: int, r: int, c: int, axis: int) -> bool:
        if fill_mode == "original":
            return (r, c) in marker_pts[color] or _axis_support(masks[color], r, c, axis, max_dist=1)
        return _axis_support(line_only[color], r, c, axis, max_dist=support_span)

    def should_fill(color: int, a: tuple[int, int], b: tuple[int, int], axis: int) -> bool:
        r1, c1 = a
        r2, c2 = b
        gap = c2 - c1 - 1 if axis == 0 else r2 - r1 - 1
        if gap <= 0:
            return False

        a_axis = has_axis_support(color, r1, c1, axis)
        b_axis = has_axis_support(color, r2, c2, axis)
        if fill_mode == "original":
            if gap == 1:
                if a_axis or b_axis:
                    return True
                a_perp = _perp_support(masks[color], r1, c1, axis)
                b_perp = _perp_support(masks[color], r2, c2, axis)
                return not (a_perp and b_perp)

            if (r1, c1) in marker_pts[color] and (r2, c2) in marker_pts[color]:
                return False
            return a_axis and b_axis

        if (r1, c1) not in marker_pts[color] or (r2, c2) not in marker_pts[color]:
            return False
        return True

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
                    for nr, nc in [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]
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
                if (
                    ((r, c - 1) in color_cells[b] and (r, c + 1) in color_cells[b])
                    or ((r - 1, c) in color_cells[b] and (r + 1, c) in color_cells[b])
                ):
                    crossing_colors.add(a)
                    crossing_colors.add(b)
    for color in line_colors:
        assert color in crossing_colors, f"color {color} has no crossing with any other color"

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Repair a one-cell-thick colored line network by recoloring singleton markers, filling only marker-clamped gaps, and redirecting yellow-marked arms to the edge.

    Input structure:
    - A rectangular grid with a dominant background.
    - Line colors form one-cell-thick horizontal/vertical stroke fragments with crossings between colors.
    - Marker colors are exactly those whose 4-connected components are isolated single pixels.
    - Red markers sit at both ends of a missing straight corridor; yellow markers sit just beyond a line endpoint and indicate a rerouted arm.

    Rule:
    - Assign every marker to the nearest visible line color in row/column line-of-sight, preferring sightlines whose contacted line cell still has same-axis support within 2 cells.
    - For each line color, fill a row/column gap only when both facing endpoints are markers of that color, and each marker is backed by same-axis non-marker line structure within 2 cells.
    - A yellow control erases the old straight arm on the far side of its endpoint and paints the same color from that endpoint through the control pixel to the grid edge.
    - At crossings, yellow-control rays win; otherwise an original-vs-filled crossing uses the more fragmented pre-repair line color, while pure filled crossings use the lower color id.

    Key insight: The markers are no longer generic missing cells. They act as clamps that authorize a fill only when they appear in a pair, and the nearby line evidence must sometimes be read through a one-cell disturbance by looking two cells away along the same axis.
    """
    return _solve(grid, support_span=SUPPORT_SPAN, fill_mode="clamped")


# --------------- generator (combo D6/2 + D15/0) ---------------

# --- CFG: grid & palette ---
H = W = 12
BG = 0
LINE_PALETTE = [1, 3, 5, 6, 7, 8, 9]

# --- CFG: polyline shape ---
POLYLINE_PIECE_WEIGHTS = [5, 4, 2, 1]
MIN_SPAN_LEN = 4
FULL_SPAN_PROB = 0.65
ARM_MIN_OFFSET = 2

# --- CFG: serpentine & crossings ---
CROSSING_COUNT_WEIGHTS = [2, 3, 3, 2]
SERPENTINE_MINOR_LO = 6
SERPENTINE_MINOR_HI = 5
CROSS_MINOR_MARGIN = 2
MAX_COLORS = 4
MAX_CROSSINGS = 12

# --- CFG: markers ---
YELLOW_USE_PROB = 0.0
EXTRA_RED_RANGE = (0, 0)
CLAMP_GAP_RANGE = (1, 5)
MIN_CELLS_FOR_RED = 2

# --- CFG: output validation ---
COMPONENTS_PER_COLOR = (1, 6)
TOTAL_COMPONENTS = (2, 8)
FILL_RANGE = (24, 64)

# --- CFG: retry budgets ---
SPACED_VALUES_RETRIES = 60
EXTRA_POLYLINE_RETRIES = 80
GEN_RETRIES = 500


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
                    old_arm = set(list(_ray_from((pivot[0] + old_dir[0], pivot[1] + old_dir[1]), old_dir))[:old_len])
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


def _visible_cross_color(color: int, pt: tuple[int, int], crossings: set[tuple[int, int]], crossing_display: dict[tuple[int, int], int]) -> bool:
    return pt not in crossings or crossing_display[pt] == color


def _straight_runs(pts: set[tuple[int, int]], axis: int) -> list[list[tuple[int, int]]]:
    groups: dict[int, list[int]] = {}
    for r, c in pts:
        key = r if axis == 0 else c
        val = c if axis == 0 else r
        groups.setdefault(key, []).append(val)

    runs: list[list[tuple[int, int]]] = []
    for key, vals in groups.items():
        vals = sorted(vals)
        start = 0
        for i in range(1, len(vals) + 1):
            if i == len(vals) or vals[i] != vals[i - 1] + 1:
                block = vals[start:i]
                if len(block) >= 2:
                    if axis == 0:
                        runs.append([(key, c) for c in block])
                    else:
                        runs.append([(r, key) for r in block])
                start = i
    return runs


def _best_visible_color(
    pt: tuple[int, int],
    visible_cells: dict[tuple[int, int], int],
    visible_by_color: dict[int, set[tuple[int, int]]],
    support_span: int,
) -> int | None:
    candidates: list[tuple[int, int, int]] = []
    for dr, dc in DIRS4:
        r, c = pt[0] + dr, pt[1] + dc
        dist = 1
        while 0 <= r < H and 0 <= c < W:
            color = visible_cells.get((r, c))
            if color is not None:
                axis = 0 if dr == 0 else 1
                supported = _axis_support(visible_by_color[color], r, c, axis, max_dist=support_span)
                candidates.append((dist, 0 if supported else 1, color))
                break
            r += dr
            c += dc
            dist += 1
    if not candidates:
        return None
    candidates.sort()
    return candidates[0][2]


def _choose_clamp_gap(
    logical: dict[int, set[tuple[int, int]]],
    crossings: set[tuple[int, int]],
    crossing_display: dict[tuple[int, int], int],
) -> tuple[int, list[tuple[int, int]], set[tuple[int, int]], int] | None:
    candidates: list[tuple[int, list[tuple[int, int]], set[tuple[int, int]], int]] = []
    min_gap, max_gap = CLAMP_GAP_RANGE
    visible_by_color_base = {
        color: {pt for pt in pts if _visible_cross_color(color, pt, crossings, crossing_display)}
        for color, pts in logical.items()
    }
    for color, pts in logical.items():
        for axis in (0, 1):
            for run in _straight_runs(pts, axis):
                n = len(run)
                for gap in range(min_gap, min(max_gap, n - 3) + 1):
                    for i in range(1, n - gap - 1):
                        j = i + gap + 1
                        if j >= n - 1:
                            continue
                        a = run[i]
                        b = run[j]
                        if a in crossings or b in crossings:
                            continue
                        interior = set(run[i + 1:j])
                        visible_by_color = {k: set(v) for k, v in visible_by_color_base.items()}
                        visible_by_color[color].discard(a)
                        visible_by_color[color].discard(b)
                        for pt in interior:
                            if pt not in crossings or crossing_display[pt] == color:
                                visible_by_color[color].discard(pt)
                        visible_cells = {
                            pt: clr
                            for clr, clr_pts in visible_by_color.items()
                            for pt in clr_pts
                        }
                        best2_a = _best_visible_color(a, visible_cells, visible_by_color, support_span=SUPPORT_SPAN)
                        best2_b = _best_visible_color(b, visible_cells, visible_by_color, support_span=SUPPORT_SPAN)
                        if best2_a != color or best2_b != color:
                            continue
                        best1_a = _best_visible_color(a, visible_cells, visible_by_color, support_span=1)
                        best1_b = _best_visible_color(b, visible_cells, visible_by_color, support_span=1)
                        delta_count = int(best1_a != color) + int(best1_b != color)
                        if delta_count == 0:
                            continue
                        candidates.append((color, [a, b], interior, delta_count))
    if not candidates:
        return None
    random.shuffle(candidates)
    candidates.sort(key=lambda item: (-item[3], -len(item[2])))
    return candidates[0]


def _blocked_with_neighbors(cells: set[tuple[int, int]]) -> set[tuple[int, int]]:
    out = set(cells)
    for pt in list(cells):
        out.update(_neighbors(pt))
    return out


def _derive_input(
    logical: dict[int, set[tuple[int, int]]],
    paths: dict[int, list[set[tuple[int, int]]]],
    dominant: int,
) -> tuple[np.ndarray, np.ndarray] | None:
    crossings = set(_crossing_cells(logical))
    crossing_display: dict[tuple[int, int], int] = {}
    for pt in crossings:
        colors = [color for color in logical if pt in logical[color]]
        if dominant not in colors:
            return None
        crossing_display[pt] = min(colors)

    clamp = _choose_clamp_gap(logical, crossings, crossing_display)
    if clamp is None:
        return None
    clamp_color, clamp_markers_pts, clamp_blank, assignment_delta_count = clamp
    clamp_markers = [(clamp_color, pt) for pt in clamp_markers_pts]
    clamp_block = _blocked_with_neighbors(set(clamp_markers_pts) | clamp_blank)

    yellow_count = 1 if random.random() < YELLOW_USE_PROB else 0
    yellow_infos = []
    blocked = set(crossings) | clamp_block
    if yellow_count:
        for cand in _yellow_candidates(logical, paths, crossings):
            color, pivot, out_dir, target_ray, old_arm = cand
            marker = (pivot[0] + out_dir[0], pivot[1] + out_dir[1])
            if marker in blocked or any(nb in blocked for nb in _neighbors(marker)):
                continue
            if target_ray & clamp_blank or old_arm & clamp_blank:
                continue
            if any((old & old_arm) or (ray & old_arm) for _, _, _, ray, old in yellow_infos):
                continue
            yellow_infos.append(cand)
            blocked.update(target_ray | old_arm | {marker})
            break

    extra_red_count = random.randint(*EXTRA_RED_RANGE)
    extra_red = _choose_red_markers(logical, paths, blocked, extra_red_count) if extra_red_count else []
    if extra_red_count and not extra_red:
        return None
    red_markers = clamp_markers + extra_red

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

    removed_for_red: dict[int, set[tuple[int, int]]] = {c: set() for c in logical}
    removed_for_red[clamp_color].update(clamp_blank)

    red_by_pt = {pt: color for color, pt in red_markers}
    yellow_by_pt = {pt: color for color, pt in yellow_markers}

    comp_count_full: dict[int, set[tuple[int, int]]] = {c: set() for c in logical}
    for color, pts in logical.items():
        for pt in pts:
            if pt in removed_for_yellow[color] or pt in removed_for_red[color]:
                continue
            comp_count_full[color].add(pt)
    for color, pts in old_arms.items():
        comp_count_full[color].update(pts)
    for pt, color in red_by_pt.items():
        comp_count_full[color].add(pt)
    for pt, color in yellow_by_pt.items():
        comp_count_full[color].add(pt)

    comp_count = {color: _gen_component_count(pts) for color, pts in comp_count_full.items()}
    target = _render_target(logical, comp_count, yellow_rays)
    crossing_display = {pt: int(target[pt]) for pt in crossings}

    inp = np.full((H, W), BG, dtype=int)
    comp_count_masks: dict[int, set[tuple[int, int]]] = {c: set() for c in logical}

    for color, pts in logical.items():
        for pt in pts:
            if pt in removed_for_yellow[color] or pt in removed_for_red[color]:
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
    if not np.array_equal(solved, target):
        return None

    try:
        original_like = _solve(inp, support_span=1, fill_mode="original")
    except Exception:
        original_like = None
    if original_like is not None and np.array_equal(original_like, target):
        return None

    filled_cells = np.argwhere((solved != inp) & (inp == BG))
    if len(filled_cells) < len(clamp_blank):
        return None
    if assignment_delta_count < 1:
        return None

    return inp, target


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
    raise RuntimeError("failed to generate a valid faa9f03d combo D6/2+D15/0 input")
