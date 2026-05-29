# Variant of faa9f03d
# Dimension: D11 — 交叉结构约束 (meta)
# Change: 每个交叉格仅两色重叠，所有线色都必须参与至少一个交叉，交叉处两线必须正交... → 交叉处显示两色混合标记

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


# Design analysis for the D11 variant
# -----------------------------------
# a) Concrete behavior
#    - Original crossing display: a logical overlap cell was shown as one of the two line
#      colors, so the solver could treat it as an ordinary visible line pixel.
#    - Variant crossing display: every logical overlap cell is shown as a dedicated mixed
#      marker color 5 (grey) in both input and output. The underlying two line colors are
#      not shown directly at the crossing.
#    - Small trace:
#      original input row/col around a crossing might show:
#        3
#        3
#      1 1 1
#        3
#      where the center is displayed as color 1 or 3.
#      Variant input/output instead shows:
#        3
#        3
#      1 5 1
#        3
#      The solver must keep extending the horizontal and vertical lines through the grey
#      cell, then paint the final visible crossing back to grey.
# b) Complexity calibration
#    - Grey is reserved and never used as a normal line color. That makes the mixed-crossing
#      cue visually obvious in every sample.
#    - Red/yellow markers are kept away from crossing cells and their 4-neighbors so the
#      "infer the two hidden colors from opposite neighbors" rule remains stable and easy to
#      read for humans.
# c) Old assumptions that break
#    - Marker detection by "all singleton components" would wrongly classify each grey
#      crossing as a marker. The variant therefore uses fixed semantics:
#      red=2, yellow=4, mixed-crossing=5.
#    - Original crossing resolution chose one visible line color. That is no longer valid.
#      The new solver reconstructs hidden ownership at grey cells from opposite neighbors and
#      always renders the final crossing as grey.
# d) Placement legality
#    - Crossings are still exactly two-color and orthogonal.
#    - Yellow rewrites are rejected if they would touch a crossing cell or its 4-neighbors.
#      This prevents erasing one of the immediate support pixels needed for crossing recovery.
#    - Red markers are also kept away from crossing neighborhoods for the same reason.
# e) Visual observability
#    - Every generated instance contains at least one grey crossing. On the repaired output,
#      those same positions remain grey, making the D11 change directly visible from examples.
# f) Visual cleanliness
#    - Only one special overlay type is added: grey crossings. There is no stacking of grey
#      with red/yellow markers, and yellow control arms never pass through a grey crossing.
# g) Color inference
#    - No invisible ordering by color id is used for crossings. A grey cell belongs to the
#      horizontal line whose two opposite horizontal neighbors match, and to the vertical line
#      whose two opposite vertical neighbors match.
# h) New solver assumptions and generator guarantees
#    - Assumption: each grey cell has exactly one horizontal claimant and one vertical claimant.
#      Guarantee: `_valid_crossings` enforces two colors and orthogonal orientations only.
#    - Assumption: immediate opposite neighbors around a grey cell survive in the input-derived
#      reconstruction. Guarantee: marker/yellow placement excludes crossing neighborhoods.


DIRS4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
BG = 0
RED_MARKER = 2
YELLOW_MARKER = 4
CROSS_COLOR = 5
MARKER_COLORS = {RED_MARKER, YELLOW_MARKER}


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


def _cross_axis_claim(mask: set[tuple[int, int]], pt: tuple[int, int], axis: int) -> bool:
    r, c = pt
    if axis == 0:
        return (r, c - 1) in mask and (r, c + 1) in mask
    return (r - 1, c) in mask and (r + 1, c) in mask


def main(grid: np.ndarray) -> np.ndarray:
    """
    Repair a one-cell-thick colored line network with red/yellow singleton markers while treating grey cells as mixed-color crossings that hide both participating lines.

    Input structure:
    - A rectangular grid with a dominant background.
    - Non-grey line colors form one-cell-thick horizontal/vertical stroke fragments.
    - Red (2) and yellow (4) pixels are isolated singleton markers; grey (5) pixels are reserved mixed-crossing markers, not line cells.
    - Each grey crossing stands for exactly two hidden orthogonal line colors whose identities must be inferred from the surrounding fragments.

    Rule:
    - Assign each red/yellow marker to the nearest visible line color in row/column line-of-sight, allowing the scan to pass through background, other markers, and grey crossing cells.
    - Recolor assigned markers to their line color logically, then fill row/column gaps between supported facing endpoints for each color.
    - A yellow control removes the old arm on the far side of its endpoint and extends the line through the marker side to the boundary.
    - After those repairs, infer every grey crossing's two hidden owners from opposite horizontal and vertical neighbors, and render the final crossing cell as grey.

    Key insight: Grey cells are not choices between two visible colors; they are explicit occlusion markers. The surrounding repaired fragments reveal which color continues straight across horizontally and which color continues straight across vertically, so the solver can rebuild connectivity first and only then repaint the crossing as grey.
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
                if value == bg or value in marker_colors or value == CROSS_COLOR:
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

    crossing_claims: dict[tuple[int, int], dict[int, int]] = {}
    for pt in cross_pts:
        axes: dict[int, int] = {}
        for color in line_colors:
            has_h = _cross_axis_claim(masks[color], pt, axis=0)
            has_v = _cross_axis_claim(masks[color], pt, axis=1)
            assert not (has_h and has_v), f"color {color} forms a junction at crossing {pt}"
            if has_h:
                axes[color] = 0
            elif has_v:
                axes[color] = 1
        assert len(axes) == 2, f"grey crossing {pt} needs exactly two inferred colors"
        assert sorted(axes.values()) == [0, 1], f"grey crossing {pt} must be orthogonal"
        crossing_claims[pt] = axes

    claims: dict[tuple[int, int], list[int]] = {}
    for color, pts in masks.items():
        for pt in pts:
            claims.setdefault(pt, []).append(int(color))
    for pt, axes in crossing_claims.items():
        claims[pt] = sorted(axes)

    out = np.full_like(g, bg)
    for pt, colors_here in claims.items():
        forced = {color for color in colors_here if pt in control_ray[color]}
        if pt in crossing_claims:
            assert not forced, f"yellow-control ray cannot own grey crossing {pt}"
            out[pt] = CROSS_COLOR
            continue
        if forced:
            assert len(forced) == 1, f"ambiguous yellow-control rays at {pt}: {forced}"
            out[pt] = next(iter(forced))
            continue
        assert len(colors_here) == 1, f"unexpected non-grey overlap at {pt}: {colors_here}"
        out[pt] = colors_here[0]

    logical_masks = {color: set(pts) for color, pts in masks.items()}
    for pt, axes in crossing_claims.items():
        for color in axes:
            logical_masks[color].add(pt)

    # Assert: every logical endpoint is on the boundary and approaches it perpendicularly.
    for color in line_colors:
        mask = np.zeros(g.shape, dtype=bool)
        for r, c in logical_masks[color]:
            mask[r, c] = True
        for comp in components_mask(mask, connectivity=4):
            pts_set = set(map(tuple, comp))
            for r, c in pts_set:
                nbrs = [(nr, nc) for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))
                        if (nr, nc) in pts_set]
                assert len(nbrs) <= 2, f"color {color} has a junction at {(r, c)}"
                if len(nbrs) != 1:
                    continue
                on_boundary = (r == 0 or r == H - 1 or c == 0 or c == W - 1)
                assert on_boundary, f"logical endpoint {(r, c)} color {color} is interior"
                nr, nc = nbrs[0]
                dr, dc = nr - r, nc - c
                if (r in (0, H - 1)) and (c in (0, W - 1)):
                    continue
                perp = ((r == 0 and dr == 1) or (r == H - 1 and dr == -1)
                        or (c == 0 and dc == 1) or (c == W - 1 and dc == -1))
                assert perp, f"boundary endpoint ({r},{c}) color {color} not perpendicular"

    crossing_colors = set()
    for axes in crossing_claims.values():
        crossing_colors.update(axes)
    for color in line_colors:
        assert color in crossing_colors, f"color {color} has no crossing with any other color"

    return out


# --------------- generator (o5x, adapted for mixed grey crossings) ---------------

# --- CFG: grid & palette ---
H = W = 12
LINE_PALETTE = [1, 3, 6, 7, 8, 9]

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
MARKER_MODE_WEIGHTS = [4, 2, 3]
YELLOW_COUNT_RANGE = (1, 2)
RED_COUNT_RANGE = (2, 6)
MIN_CELLS_FOR_RED = 2

# --- CFG: output validation ---
VISIBLE_COMPONENTS_PER_COLOR = (1, 5)
TOTAL_VISIBLE_COMPONENTS = (3, 9)
FILL_RANGE = (33, 55)

# --- CFG: retry budgets ---
SPACED_VALUES_RETRIES = 60
EXTRA_POLYLINE_RETRIES = 80
GEN_RETRIES = 300


def _neighbors(pt: tuple[int, int]) -> list[tuple[int, int]]:
    r, c = pt
    return [(r + dr, c + dc) for dr, dc in DIRS4 if 0 <= r + dr < H and 0 <= c + dc < W]


def _expanded(cells: set[tuple[int, int]]) -> set[tuple[int, int]]:
    out = set(cells)
    for pt in list(cells):
        out.update(_neighbors(pt))
    return out


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
    crossing_buffer = _expanded(crossings)
    for color, comps in paths.items():
        for comp in comps:
            for pivot in comp:
                if pivot in crossing_buffer or not _is_turn_or_endpoint(comp, pivot):
                    continue
                for out_dir in DIRS4:
                    mr, mc = pivot[0] + out_dir[0], pivot[1] + out_dir[1]
                    if not (0 <= mr < H and 0 <= mc < W) or (mr, mc) not in comp:
                        continue
                    target_ray = _ray_from(pivot, out_dir)
                    if not target_ray.issubset(comp) or target_ray & crossing_buffer:
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
                    if old_arm and old_arm.issubset(old_cells) and not (old_arm & crossing_buffer):
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
    n_colors = random.randint(2, min(MAX_COLORS, TOTAL_VISIBLE_COMPONENTS[1] - desired_crossings))
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
    for color in (RED_MARKER, YELLOW_MARKER):
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
    crossing_buffer = _expanded(crossings)
    mode = random.choices(["red", "yellow", "both"], weights=MARKER_MODE_WEIGHTS)[0]
    yellow_count = random.randint(*YELLOW_COUNT_RANGE) if mode in ("yellow", "both") else 0
    red_count = random.randint(*RED_COUNT_RANGE) if mode in ("red", "both") else 0

    yellow_infos = []
    blocked = set(crossing_buffer)
    if yellow_count:
        for cand in _yellow_candidates(logical, paths, crossings):
            color, pivot, out_dir, target_ray, old_arm = cand
            marker = (pivot[0] + out_dir[0], pivot[1] + out_dir[1])
            affected = target_ray | old_arm | {marker, pivot}
            if affected & crossing_buffer:
                continue
            if marker in blocked or any(nb in blocked for nb in _neighbors(marker)):
                continue
            if any((old & old_arm) or (ray & old_arm) for _, _, _, ray, old in yellow_infos):
                continue
            yellow_infos.append(cand)
            blocked.update(affected)
            if len(yellow_infos) == yellow_count:
                break
        if len(yellow_infos) != yellow_count:
            return None

    red_markers = _choose_red_markers(logical, paths, blocked, red_count) if red_count else []
    if red_count and not red_markers:
        return None

    inp = np.full((H, W), BG, dtype=int)
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

    for color, pts in logical.items():
        for pt in pts:
            if pt in crossings or pt in removed_for_yellow[color]:
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
        if not any(inp[nb] == color for nb in _neighbors(pt)):
            return None

    target = _render_target(logical, yellow_rays, crossings)
    if not np.array_equal(target == CROSS_COLOR, inp == CROSS_COLOR):
        return None

    visible_counts = {c: len(components_mask(target == c, connectivity=4)) for c in logical}
    if not all(VISIBLE_COMPONENTS_PER_COLOR[0] <= v <= VISIBLE_COMPONENTS_PER_COLOR[1] for v in visible_counts.values()):
        return None
    if not TOTAL_VISIBLE_COMPONENTS[0] <= sum(visible_counts.values()) <= TOTAL_VISIBLE_COMPONENTS[1]:
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
