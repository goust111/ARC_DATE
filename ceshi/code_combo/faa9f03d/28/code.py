# Combo variant of faa9f03d (index 28)
# combo_spec: D8/1+D12/0+D13/0
# Name: 高密度多色 + 延伸到障碍物 + 可变尺寸

from __future__ import annotations

import random

import numpy as np

from arc_utils import components_mask, get_bg


# Design analysis for combo 28:
# - Concrete behavior:
#   Start from the base task's "repair sparse rectilinear networks from singleton markers" idea,
#   then change three pressure points together. Example on a 14x14 grid: suppose a pink line has a
#   yellow marker one cell to the left of a turn-pivot, and three cells farther left there is a
#   teal crossing cell. In the base task the repaired arm would continue past teal all the way to
#   the border. In this combo it must stop exactly at that first occupied blocker cell, and the
#   output keeps only the pivot..blocker segment. At the same time the scene contains 5-6 line
#   colors, so such blockers are common and visually meaningful rather than rare corner cases.
# - Complexity calibration:
#   The generator always uses 5 or 6 non-marker colors and always includes at least one yellow
#   control. The dense layout is not "free-form chaos": one dominant serpentine line is crossed by
#   every other color, which creates many visible obstacles while keeping the crossing rule legible.
#   Yellow candidates are filtered so the blocker is interior and close, making the obstacle-stop
#   effect repeated and easy to notice across examples.
# - Hidden assumptions audited in main():
#   The original solver assumed a fixed 12x12 board only inside the generator helpers, so main() is
#   kept shape-driven and never relies on global H/W. The more important broken assumption is the
#   yellow phase: "walk to the edge" is no longer valid, so the new loop stops at the first occupied
#   cell after the marker. Endpoint validation also has to accept obstacle-limited terminals at
#   multi-claim crossings. Marker assignment still assumes each visible line color has at least one
#   non-singleton component; the generator explicitly validates that dense recoloring does not turn a
#   true line color into an all-singleton marker impostor.
# - Placement legality:
#   The generator is still output-first. It builds legal 1-pixel rectilinear paths, forbids
#   triple-overlaps, and constrains every non-dominant color to intersect only the dominant line.
#   Yellow edits reserve a non-overlapping zone containing the new ray, the removed tail, the fake
#   old arm, and the marker cell, so separate controls cannot collide or accidentally create new
#   blockers. All helpers are size-aware, so no coordinates assume 12x12.
# - Visual observability:
#   Humans can see the combo because three signals co-occur: the board size changes across samples,
#   there are clearly 5-6 different line colors, and yellow-controlled arms terminate at nearby
#   interior collisions instead of at the boundary. Input crossings show the blocker color while the
#   output switches those cells back to the winning repaired color, exposing the obstacle event.
# - Visual cleanliness:
#   Dense does not mean unreadable. Every crossing is still only two colors, all lines stay
#   one-cell-thick, markers remain isolated singletons, and all non-dominant lines cross the same
#   dominant scaffold rather than each other. That keeps local neighborhoods interpretable.
# - Inferability of changes:
#   No hidden use of color ids decides where a yellow ray stops; the stop is geometric. Color ids
#   only participate in the unchanged tie-break after fragmentation, and the dominant scaffold is
#   intentionally fragmented in the input so the crossing winner is usually supported by visible
#   structure rather than by a bare numeric tie-break.


DIRS4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
LINE_PALETTE = [1, 3, 5, 6, 7, 8, 9]
BG = 0
MIN_SIZE = 8
MAX_SIZE = 16

H = 12
W = 12
LAST_META: dict[str, object] = {}

SPACED_VALUES_RETRIES = 120
EXTRA_POLYLINE_RETRIES = 220
GEN_RETRIES = 700


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


def main(grid: np.ndarray) -> np.ndarray:
    """
    Repair a dense multi-color rectilinear network on variable square boards by recoloring singleton markers, filling supported gaps, and rerouting yellow-controlled arms until they hit the nearest obstacle.

    Input structure:
    - The board is square but not fixed-size; examples can vary from 8x8 up to 16x16.
    - Five or six non-marker colors form one-cell-thick horizontal/vertical line fragments with many crossings.
    - Any foreground color whose 4-connected components are all singleton pixels is a marker color.
    - Red singleton markers stand for missing line cells; yellow singleton markers sit beside a line endpoint/turn and indicate the side where the repaired arm should grow.

    Rule:
    - Assign each marker to the nearest visible line color in straight row/column sight, preferring candidates that already have support along that axis.
    - After recoloring the markers conceptually, fill the row/column gaps between consecutive facing endpoints using the same support test as the original task.
    - For each yellow control, erase the old arm on the far side of the pivot, then extend the pivot through the marker only until the first non-background cell on that ray; include that blocking cell as the new terminal collision cell.
    - If several colors claim one output cell, yellow-control rays override everything else; otherwise crossings are resolved by the same fragmentation rule as the base task.

    Key insight: In the fused variant the network is dense enough that other colored paths become visible blockers, so yellow does not mean "grow to the edge" anymore. You still repair by grouping sparse fragments, but now the stopping point is determined by the nearest obstacle and the whole procedure must be independent of any fixed 12x12 frame.
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
            masks[color].add(pivot)
            control_ray[color].add(pivot)
            masks[color].add(control)
            control_ray[color].add(control)

            r, c = control[0] + pr, control[1] + pc
            hit_obstacle = False
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

    return out


def _set_size(size: int) -> None:
    global H, W
    H = size
    W = size


def _choose_size() -> int:
    sizes = list(range(9, MAX_SIZE + 1))
    weights = [1, 2, 2, 3, 4, 4, 3, 2]
    return random.choices(sizes, weights=weights)[0]


def _min_span_len(limit: int) -> int:
    return max(3, min(5, limit // 2))


def _arm_min_offset() -> int:
    return 1 if min(H, W) <= 9 else 2


def _cross_minor_margin() -> int:
    return 1 if min(H, W) <= 10 else 2


def _full_span_prob() -> float:
    return 0.55 if min(H, W) <= 10 else 0.68


def _yellow_block_max_dist() -> int:
    return max(3, min(5, min(H, W) // 3 + 1))


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


def _logical_components(logical: dict[int, set[tuple[int, int]]]) -> dict[int, list[set[tuple[int, int]]]]:
    paths: dict[int, list[set[tuple[int, int]]]] = {}
    for color, pts in logical.items():
        mask = np.zeros((H, W), dtype=bool)
        for r, c in pts:
            mask[r, c] = True
        paths[color] = [set(map(tuple, comp)) for comp in components_mask(mask, connectivity=4)]
    return paths


def _rc(ax: int, major: int, minor: int) -> tuple[int, int]:
    return (major, minor) if ax == 0 else (minor, major)


def _dim(ax: int) -> int:
    return H if ax == 0 else W


def _random_span(limit: int) -> tuple[int, int]:
    min_span = _min_span_len(limit)
    if limit <= min_span + 1:
        return 0, limit - 1
    if random.random() < 0.5:
        return 0, random.randint(min_span, limit - 1)
    return random.randint(0, limit - min_span - 1), limit - 1


def _pick_path_flavor() -> str:
    return random.choices(
        ["straight", "start_arm", "end_arm", "both_arms"],
        weights=[1, 3, 3, 2],
    )[0]


def _build_arm(
    ax: int,
    main_coord: int,
    cross_lo: int,
    cross_hi: int,
    is_start: bool,
) -> list[tuple[int, int]]:
    dim_minor = _dim(1 - ax)
    arm_choices = [
        v for v in range(1, dim_minor - 1) if abs(v - main_coord) >= _arm_min_offset()
    ]
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
    lo = 0 if random.random() < _full_span_prob() else random.randint(0, major - 1)
    hi = _dim(ax) - 1 if random.random() < _full_span_prob() else random.randint(major + 1, _dim(ax) - 1)
    pts = [_rc(ax, lo, minor), _rc(ax, hi, minor)]
    flavor = _pick_path_flavor()
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


def _straight_path_through(cell: tuple[int, int], orient: str) -> list[tuple[int, int]] | None:
    r, c = cell
    ax = 0 if orient == "v" else 1
    major = r if ax == 0 else c
    minor = c if ax == 0 else r
    if major == 0 or major == _dim(ax) - 1:
        return None
    cells = _path([_rc(ax, 0, minor), _rc(ax, _dim(ax) - 1, minor)])
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
    minor_dim = _dim(1 - ax)
    margin = _cross_minor_margin()
    interior = list(range(1, minor_dim - 1))
    pair_candidates = [
        (a, b)
        for a in interior
        for b in interior
        if b - a >= 2 * margin + 1 and (a == 1 or b == minor_dim - 2)
    ]
    if not pair_candidates:
        pair_candidates = [
            (a, b) for a in interior for b in interior if b - a >= 2 * margin + 1
        ]
    if not pair_candidates:
        return None
    m0, m1 = random.choice(pair_candidates)
    cross_minor = random.randint(m0 + margin, m1 - margin)

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

    pts = [_rc(ax, 0, main_coord), _rc(ax, _dim(ax) - 1, main_coord)]
    flavor = _pick_path_flavor()
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
    if len(crossings) < len(logical) - 1:
        return False
    crossing_colors = set()
    for pt in crossings:
        colors = [color for color in logical if pt in logical[color]]
        if len(colors) != 2 or dominant not in colors:
            return False
        orientations = [_orientation(logical[color], pt) for color in colors]
        if None in orientations or sorted(orientations) != ["h", "v"]:
            return False
        crossing_colors.update(colors)
    return all(color in crossing_colors for color in logical)


def _has_turn(cells: list[tuple[int, int]]) -> bool:
    pts = set(cells)
    return any(_is_turn_or_endpoint(pts, pt) and len(_dirs_in(pts, pt)) == 2 for pt in pts)


def _straight_line_cells(orient: str, fixed: int) -> list[tuple[int, int]]:
    ax = 0 if orient == "v" else 1
    return _path([_rc(ax, 0, fixed), _rc(ax, _dim(ax) - 1, fixed)])


def _yellow_ready_bent(
    dominant_orient: str,
    dom_minor: int,
    anchor_value: int,
) -> list[tuple[int, int]] | None:
    max_dist = _yellow_block_max_dist()
    dist_choices: list[int] = []
    if dom_minor - max_dist >= 1:
        dist_choices.extend([-d for d in range(2, max_dist + 1) if dom_minor - d >= 1])
    if dom_minor + max_dist <= (_dim(1 if dominant_orient == "v" else 0) - 2):
        dist_choices.extend([d for d in range(2, max_dist + 1) if dom_minor + d <= (_dim(1 if dominant_orient == "v" else 0) - 2)])
    random.shuffle(dist_choices)
    for signed_dist in dist_choices:
        if dominant_orient == "v":
            turn_col = dom_minor + signed_dist
            start_row = 0 if anchor_value <= H // 2 else H - 1
            end_col = W - 1 if turn_col < dom_minor else 0
            pts = [(start_row, turn_col), (anchor_value, turn_col), (anchor_value, end_col)]
            anchor = (anchor_value, dom_minor)
        else:
            turn_row = dom_minor + signed_dist
            start_col = 0 if anchor_value <= W // 2 else W - 1
            end_row = H - 1 if turn_row < dom_minor else 0
            pts = [(turn_row, start_col), (turn_row, anchor_value), (end_row, anchor_value)]
            anchor = (dom_minor, anchor_value)
        cells = _path(pts)
        if not _valid_path(cells):
            continue
        if set(cells) & {anchor} != {anchor}:
            continue
        return cells
    return None


def _ray_list(pivot: tuple[int, int], direction: tuple[int, int]) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    r, c = pivot
    dr, dc = direction
    while 0 <= r < H and 0 <= c < W:
        out.append((r, c))
        r += dr
        c += dc
    return out


def _yellow_candidates(
    logical: dict[int, set[tuple[int, int]]],
    paths: dict[int, list[set[tuple[int, int]]]],
    crossings: set[tuple[int, int]],
) -> list[dict]:
    candidates: list[dict] = []
    occupied = set().union(*logical.values())
    max_dist = _yellow_block_max_dist()
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
                    if obstacle_idx is None or obstacle_idx < 2 or obstacle_idx > max_dist:
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
                    old_len = random.randint(1, min(len(old_cells), max_dist))
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
                            "block_dist": obstacle_idx,
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
        if per_color_left[color] <= 2:
            continue
        if any(nb in used for nb in [pt] + _neighbors(pt)):
            continue
        chosen.append((color, pt))
        used.add(pt)
        per_color_left[color] -= 1
    return chosen if len(chosen) == count else []


def _find_extra_straight(
    logical: dict[int, set[tuple[int, int]]],
    dominant: int,
    crossings: set[tuple[int, int]],
) -> list[tuple[int, int]] | None:
    candidates = []
    dominant_pts = logical[dominant]
    for r, c in dominant_pts:
        if (r, c) in crossings:
            continue
        orient = _orientation(dominant_pts, (r, c))
        if orient is None:
            continue
        perp = "v" if orient == "h" else "h"
        major = r if perp == "v" else c
        if major <= 1 or major >= (H if perp == "v" else W) - 2:
            continue
        if any(abs(r - cr) + abs(c - cc) <= 1 for cr, cc in crossings):
            continue
        trial = _straight_path_through((r, c), perp)
        if trial is None:
            continue
        trial_set = set(trial)
        if trial_set & dominant_pts != {(r, c)}:
            continue
        if any(trial_set & logical[other] for other in logical if other != dominant):
            continue
        candidates.append(trial)
    random.shuffle(candidates)
    return candidates[0] if candidates else None


def _build_logical() -> tuple[dict[int, set[tuple[int, int]]], dict[int, list[set[tuple[int, int]]]], int] | None:
    n_colors = 5 if H <= 10 else random.choice([5, 6])
    colors = random.sample(LINE_PALETTE, n_colors)
    dominant = random.choice(colors)
    logical: dict[int, set[tuple[int, int]]] = {c: set() for c in colors}
    paths: dict[int, list[set[tuple[int, int]]]] = {c: [] for c in colors}

    dominant_orient = random.choice(["v", "h"])
    perp_orient = "h" if dominant_orient == "v" else "v"
    dom_minor_choices = list(range(2, (_dim(1 if dominant_orient == "v" else 0)) - 2))
    if not dom_minor_choices:
        dom_minor_choices = list(range(1, (_dim(1 if dominant_orient == "v" else 0)) - 1))
    if not dom_minor_choices:
        return None
    dom_minor = random.choice(dom_minor_choices)
    dom_cells = _straight_line_cells(dominant_orient, dom_minor)
    if not _can_add(logical, dominant, dom_cells):
        return None
    logical[dominant].update(dom_cells)
    paths[dominant].append(set(dom_cells))

    other_colors = [c for c in colors if c != dominant]
    bent_color = random.choice(other_colors)
    anchor_axis_len = H if perp_orient == "h" else W
    anchor_values = _spaced_values(anchor_axis_len, len(other_colors))
    if anchor_values is None:
        return None
    anchor_values.sort(key=lambda v: min(v, anchor_axis_len - 1 - v))

    bent_cells = None
    bent_anchor_value = None
    for anchor_value in anchor_values:
        anchor = (anchor_value, dom_minor) if dominant_orient == "v" else (dom_minor, anchor_value)
        trial = _yellow_ready_bent(dominant_orient, dom_minor, anchor_value)
        if trial is None:
            continue
        trial_set = set(trial)
        if trial_set & logical[dominant] != {anchor}:
            continue
        if any(trial_set & logical[other] for other in logical if other not in (dominant, bent_color)):
            continue
        if not _can_add(logical, bent_color, trial):
            continue
        bent_cells = trial
        bent_anchor_value = anchor_value
        break
    if bent_cells is None or bent_anchor_value is None:
        return None
    logical[bent_color].update(bent_cells)
    paths[bent_color].append(set(bent_cells))

    used_anchor_values = {bent_anchor_value}
    for color in other_colors:
        if color == bent_color:
            continue
        cells = None
        for anchor_value in anchor_values:
            if anchor_value in used_anchor_values:
                continue
            anchor = (anchor_value, dom_minor) if dominant_orient == "v" else (dom_minor, anchor_value)
            trial = _straight_line_cells(perp_orient, anchor_value)
            trial_set = set(trial)
            if trial_set & logical[dominant] != {anchor}:
                continue
            if any(trial_set & logical[other] for other in logical if other not in (dominant, color)):
                continue
            if not _can_add(logical, color, trial):
                continue
            cells = trial
            used_anchor_values.add(anchor_value)
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


def _line_colors_not_markers(grid: np.ndarray, line_colors: set[int]) -> bool:
    for color in line_colors:
        comps = components_mask(grid == color, connectivity=4)
        if not comps:
            return False
        if max(len(comp) for comp in comps) == 1:
            return False
    return True


def _fill_count_bounds(n_colors: int) -> tuple[int, int]:
    area = H * W
    lo = max(n_colors * 5, int(area * 0.26))
    hi = int(area * 0.60)
    return lo, hi


def _derive_input(
    logical_base: dict[int, set[tuple[int, int]]],
    paths_base: dict[int, list[set[tuple[int, int]]]],
    dominant: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, object]] | None:
    base_crossings = set(_crossing_cells(logical_base))
    yellow_count = 1
    red_count = 0

    yellow_infos: list[dict] = []
    reserved_marker_zone = set(base_crossings)
    used_zone: set[tuple[int, int]] = set()
    for cand in _yellow_candidates(logical_base, paths_base, base_crossings):
        marker = cand["marker"]
        if marker in reserved_marker_zone:
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
        if len(colors) != 2 or dominant not in colors:
            return None
        forced = [color for color in colors if pt in yellow_rays[color]]
        if forced:
            if len(forced) != 1:
                return None
            crossing_display[pt] = next(color for color in colors if color != forced[0])
        else:
            crossing_display[pt] = next(color for color in colors if color != dominant)

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
    if not _line_colors_not_markers(inp, set(logical_target)):
        return None

    comp_count = {color: _gen_component_count(pts) for color, pts in comp_count_masks.items()}
    target = _render_target(logical_target, comp_count, yellow_rays)
    for pt in crossings:
        colors = [color for color in logical_target if pt in logical_target[color]]
        forced = [color for color in colors if pt in yellow_rays[color]]
        if forced:
            if target[pt] != forced[0]:
                return None
        elif target[pt] != dominant:
            return None
        if target[pt] == inp[pt]:
            return None

    counts = {c: len(components_mask(target == c, connectivity=4)) for c in sorted(logical_target)}
    if not all(1 <= v <= 4 for v in counts.values()):
        return None
    fill_lo, fill_hi = _fill_count_bounds(len(logical_target))
    fill_count = int(np.count_nonzero(target))
    if not fill_lo <= fill_count <= fill_hi:
        return None

    try:
        solved = main(inp)
    except Exception:
        return None
    if not np.array_equal(solved, target):
        return None

    meta = {
        "size": H,
        "n_colors": len(logical_target),
        "crossings": len(crossings),
        "yellow_count": len(yellow_infos),
        "yellow_block_dists": [cand["block_dist"] for cand in yellow_infos],
        "yellow_tail_lengths": [len(cand["removed_tail"]) for cand in yellow_infos],
        "fill_count": fill_count,
    }
    return inp, target, meta


def _gen_attempt() -> tuple[np.ndarray, np.ndarray, dict[str, object]] | None:
    _set_size(_choose_size())
    built = _build_logical()
    if built is None:
        return None
    logical, paths, dominant = built
    return _derive_input(logical, paths, dominant)


def generate_example() -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    for _ in range(GEN_RETRIES):
        result = _gen_attempt()
        if result is not None:
            inp, target, meta = result
            LAST_META.clear()
            LAST_META.update(meta)
            return inp, target, meta
    raise RuntimeError("failed to generate a valid combo-28 faa9f03d input")


def generate_input() -> np.ndarray:
    return generate_example()[0]
