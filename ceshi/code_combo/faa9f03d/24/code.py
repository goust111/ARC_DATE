# Combo variant of faa9f03d (index 24)
# combo_spec: D3/0+D4/1+D11/0
# Name: 三种标记颜色 + 颜色编码目标 + 三色交叉

from __future__ import annotations

import random

import numpy as np

from arc_utils import components_mask, get_bg


# ---------------------------------------------------------------------------
# Design analysis for the combo variant
#
# a) Concrete fused behavior and trace
#    The fused rule uses exactly three marker colors and exactly three logical
#    line colors, with a fixed legend:
#      - red marker 2  -> line color 1, and its role is "gap/repair"
#      - yellow marker 4 -> line color 3, and its role is "direction control"
#      - pink marker 6 -> line color 7, and its role is "crossing priority"
#    This is the D4/1 part: marker color no longer asks "which nearby line is
#    this for?" because the marker color itself answers that question. The
#    D3/0 part is that the three marker colors still do three different jobs.
#    The D11/0 part is that the pink marker is placed on a true three-color
#    overlap cell, so it forces the visible winner of a 3-way crossing.
#
#    Small trace:
#      - logical output has color-1, color-3, color-7 paths overlapping at one
#        central cell T; nearby there are also two ordinary 2-color overlap
#        cells.
#      - input replaces one color-1 corner with red(2), removes the upward arm
#        of a color-3 turn and places yellow(4) on that missing arm, adds a
#        short old arm on the opposite side, and replaces T itself with pink(6).
#      - solver decodes 2->1, 4->3, 6->7 directly; red/pink are repainted as
#        line cells, yellow still needs geometry only for its direction.
#      - gaps are filled, the yellow-controlled color-3 arm is restored to the
#        edge, and at T the pink marker forces color 7 to be visible even though
#        fragmentation would otherwise favor a different color.
#
# b) Complexity calibration
#    The design is intentionally narrow instead of fully free-form:
#      - exactly three line colors repeat across all samples, so the legend is
#        inferable from a few examples rather than hidden in many palettes
#      - every generated sample contains all three marker families, at least one
#        3-color crossing, and at least one ordinary 2-color crossing nearby
#      - pink is always on the 3-color crossing itself, not somewhere remote, so
#        the D11 change is visually obvious instead of requiring long tracing
#
# c) Original assumptions that break
#      - source solver assumed singleton markers must be assigned to colors by
#        nearest visible line-of-sight; false here, because marker hue is the
#        assignment
#      - source solver bundled all recolored markers together; here yellow must
#        stay separate because its marker color identifies a different target
#        line than red/pink
#      - source crossing logic assumed at most two colors claim a cell; here the
#        claim set may contain three colors and the same winner rule must scale
#        to that larger set
#
# d) Placement legality guaranteed by the generator
#      - the three logical polylines are hand-parameterized and validated as
#        1-cell-thick rectilinear paths with boundary endpoints
#      - the two marker cells away from crossings (red and yellow) are kept
#        isolated from same-colored markers and from one another
#      - the yellow old arm is only placed into empty space opposite the chosen
#        pivot, so it never collides with existing logical paths
#
# e/f/g) Human observability, cleanliness, inferability
#      - the scene is clean: one compact crossing region in the middle, one red
#        repair corner off to the side, and one yellow reroute near the end of
#        the color-3 path
#      - the color legend is repeated in every sample because marker colors are
#        globally fixed, and each marker disappears into the same line color in
#        the output
#      - the pink marker sits exactly on the unusual 3-color crossing cell, so
#        the "third marker = crossing winner color" rule is locally readable
# ---------------------------------------------------------------------------


DIRS4 = ((-1, 0), (1, 0), (0, -1), (0, 1))

RED = 2
YELLOW = 4
PINK = 6

RED_LINE = 1
YELLOW_LINE = 3
PINK_LINE = 7

KNOWN_MARKERS = {RED, YELLOW, PINK}
MARKER_TO_LINE = {RED: RED_LINE, YELLOW: YELLOW_LINE, PINK: PINK_LINE}
LINE_COLORS = {RED_LINE, YELLOW_LINE, PINK_LINE}


def _pts(mask: np.ndarray) -> set[tuple[int, int]]:
    return {tuple(map(int, rc)) for rc in np.argwhere(mask)}


def _neighbors(pt: tuple[int, int], H: int, W: int) -> list[tuple[int, int]]:
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


def _on_boundary(r: int, c: int, H: int, W: int) -> bool:
    return r == 0 or r == H - 1 or c == 0 or c == W - 1


def _valid_path(cells: list[tuple[int, int]], H: int, W: int) -> bool:
    pts = set(cells)
    if len(pts) != len(cells) or len(pts) < 2:
        return False
    mask = np.zeros((H, W), dtype=bool)
    for r, c in pts:
        if not (0 <= r < H and 0 <= c < W):
            return False
        mask[r, c] = True
    if len(components_mask(mask, connectivity=4)) != 1:
        return False
    for r, c in pts:
        nbrs = _dirs_in(pts, (r, c))
        if len(nbrs) > 2:
            return False
        if not _on_boundary(r, c, H, W):
            continue
        for dr, dc in nbrs:
            nr, nc = r + dr, c + dc
            if (r in (0, H - 1) and nr == r) or (c in (0, W - 1) and nc == c):
                return False
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
    return True


def _ray_from(pivot: tuple[int, int], direction: tuple[int, int], H: int, W: int) -> set[tuple[int, int]]:
    out = set()
    r, c = pivot
    dr, dc = direction
    while 0 <= r < H and 0 <= c < W:
        out.add((r, c))
        r += dr
        c += dc
    return out


def _transform_pt(pt: tuple[int, int], mode: int, H: int, W: int) -> tuple[int, int]:
    r, c = pt
    if mode == 0:
        return (r, c)
    if mode == 1:
        return (r, W - 1 - c)
    if mode == 2:
        return (H - 1 - r, c)
    if mode == 3:
        return (H - 1 - r, W - 1 - c)
    raise ValueError(mode)


def _transform_dir(direction: tuple[int, int], mode: int) -> tuple[int, int]:
    dr, dc = direction
    if mode == 0:
        return (dr, dc)
    if mode == 1:
        return (dr, -dc)
    if mode == 2:
        return (-dr, dc)
    if mode == 3:
        return (-dr, -dc)
    raise ValueError(mode)


def _crossing_map(logical: dict[int, set[tuple[int, int]]]) -> dict[tuple[int, int], list[int]]:
    cells = set().union(*logical.values())
    return {pt: [color for color in logical if pt in logical[color]] for pt in cells if sum(pt in logical[color] for color in logical) > 1}


def _find_same_color_ray(
    grid: np.ndarray,
    marker: tuple[int, int],
    color: int,
    original: dict[int, set[tuple[int, int]]],
) -> tuple[int, int]:
    H, W = grid.shape
    candidates: list[tuple[int, int, int, int]] = []
    for dr, dc in DIRS4:
        r, c = marker[0] + dr, marker[1] + dc
        dist = 1
        while 0 <= r < H and 0 <= c < W:
            value = int(grid[r, c])
            if value == color:
                axis = 0 if dr == 0 else 1
                supported = _axis_support(original[color], r, c, axis)
                candidates.append((dist, 0 if supported else 1, dr, dc))
                break
            if value == 0 or value in KNOWN_MARKERS:
                r += dr
                c += dc
                dist += 1
                continue
            break
    assert candidates, f"marker {marker} cannot see its mapped color {color}"
    candidates.sort()
    _dist, _unsupported, dr, dc = candidates[0]
    return (int(dr), int(dc))


def main(grid: np.ndarray) -> np.ndarray:
    """
    Repair a sparse rectilinear line network whose three singleton marker colors simultaneously encode role and target line color, including a forced 3-color crossing winner.

    Input structure:
    - The grid has a dominant background plus three logical line colors and up to three singleton marker colors.
    - Marker hue is the legend: red(2) belongs to line color 1, yellow(4) belongs to line color 3, and pink(6) belongs to line color 7.
    - Red marks a missing repair cell, yellow sits on the missing arm of a rerouted line, and pink is placed directly on a crossing cell whose visible winner must become color 7.

    Rule:
    - Decode every marker by color, not by nearest-line assignment: repaint red as color 1 and pink as color 7, while yellow still uses same-color line-of-sight only to decide which arm direction to restore.
    - Fill row/column gaps between facing same-color cells after the repair markers are restored; for yellow, erase the old opposite arm beyond the pivot and extend the mapped color through the marker to the boundary.
    - Resolve overlaps in this order: yellow-control rays first, then pink-forced crossing cells, then the fragmentation-based winner rule over all claiming colors (which may now be two or three).

    Key insight: The three marker colors form a compact legend system: each marker family has its own geometric role, but its hue also says which line color it belongs to, and the pink marker makes that legend visible at a three-color crossing because it directly chooses the shown winner there.
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

    assert marker_colors <= KNOWN_MARKERS, f"unexpected marker colors: {marker_colors}"
    line_colors = colors - marker_colors
    assert line_colors <= LINE_COLORS, f"unexpected line colors: {line_colors}"
    assert marker_colors, "expected singleton markers"

    original = {color: _pts(g == color) for color in LINE_COLORS}
    assignment = {pt: MARKER_TO_LINE[int(g[pt])] for pt in _pts(np.isin(g, list(marker_colors)))}
    assign_dir: dict[tuple[int, int], tuple[int, int]] = {}

    if YELLOW in marker_colors:
        for control in _pts(g == YELLOW):
            assign_dir[control] = _find_same_color_ray(g, control, MARKER_TO_LINE[YELLOW], original)

    masks = {color: set(original[color]) for color in LINE_COLORS}
    repair_marker_pts = {color: set() for color in LINE_COLORS}
    pink_forced: dict[tuple[int, int], int] = {}
    for pt, color in assignment.items():
        marker_color = int(g[pt])
        if marker_color == YELLOW:
            continue
        masks[color].add(pt)
        repair_marker_pts[color].add(pt)
        if marker_color == PINK:
            pink_forced[pt] = color

    comp_count = {color: _component_count(g.shape, masks[color]) for color in LINE_COLORS}

    def has_axis_support(color: int, r: int, c: int, axis: int) -> bool:
        return (r, c) in repair_marker_pts[color] or _axis_support(masks[color], r, c, axis)

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
        if (r1, c1) in repair_marker_pts[color] and (r2, c2) in repair_marker_pts[color]:
            return False
        return a_axis and b_axis

    for color in sorted(LINE_COLORS):
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

    control_ray = {color: set() for color in LINE_COLORS}
    if YELLOW in marker_colors:
        for control in _pts(g == YELLOW):
            color = MARKER_TO_LINE[YELLOW]
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
                for q in old_arm:
                    masks[color].discard(q)
                    repair_marker_pts[color].discard(q)
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

    def marker_brackets(color: int, pt: tuple[int, int]) -> bool:
        r, c = pt
        same_row = [mc for mr, mc in repair_marker_pts[color] if mr == r]
        if any(mc < c for mc in same_row) and any(mc > c for mc in same_row):
            return True
        same_col = [mr for mr, mc in repair_marker_pts[color] if mc == c]
        return any(mr < r for mr in same_col) and any(mr > r for mr in same_col)

    out = np.full_like(g, bg)
    for pt, colors_here in claims.items():
        forced = {color for color in colors_here if pt in control_ray[color]}
        if forced:
            assert len(forced) == 1, f"ambiguous yellow rays at {pt}: {forced}"
            out[pt] = next(iter(forced))
            continue
        if pt in pink_forced:
            assert pink_forced[pt] in colors_here, f"pink forced color missing at {pt}"
            out[pt] = pink_forced[pt]
            continue
        if len(colors_here) == 1:
            out[pt] = colors_here[0]
            continue
        if int(g[pt]) in colors_here:
            out[pt] = min(colors_here, key=lambda color: (-comp_count[color], color))
            continue
        bracketed = [color for color in colors_here if marker_brackets(color, pt)]
        if bracketed:
            out[pt] = min(bracketed, key=lambda color: (-comp_count[color], color))
        else:
            out[pt] = min(colors_here)

    for color in LINE_COLORS:
        for comp in components_mask(out == color, connectivity=4):
            pts_set = set(map(tuple, comp))
            for r, c in pts_set:
                nbrs = [(nr, nc) for nr, nc in [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)] if (nr, nc) in pts_set]
                if len(nbrs) != 1:
                    continue
                on_boundary = _on_boundary(r, c, H, W)
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
                    0 <= r + dr < H and 0 <= c + dc < W and int(out[r + dr, c + dc]) not in (bg, color)
                    for dr, dc in DIRS4
                )
                assert adj_other, f"dangling endpoint ({r},{c}) color {color}"

    return out


# --------------- generator (combo variant) ---------------

H = W = 12
BG = 0
GEN_RETRIES = 500
FILL_RANGE = (33, 55)


def _build_scene() -> tuple[
    dict[int, set[tuple[int, int]]],
    tuple[int, int],
    tuple[int, int],
    tuple[int, int],
    tuple[int, int],
]:
    r = random.randint(4, 6)
    c = random.randint(5, 7)
    y = random.randint(r + 3, 9)

    color1 = set(
        _path(
            [
                (r + 2, 0),
                (r + 2, 2),
                (r + 1, 2),
                (r + 1, c),
                (r, c),
                (r, c + 1),
                (0, c + 1),
            ]
        )
    )
    color3 = set(_path([(0, c), (y, c), (y, W - 1)]))
    color7 = set(_path([(r, 0), (r, W - 1)]))

    assert _valid_path(sorted(color1), H, W)
    assert _valid_path(sorted(color3), H, W)
    assert _valid_path(sorted(color7), H, W)

    logical = {RED_LINE: color1, YELLOW_LINE: color3, PINK_LINE: color7}
    crossing_map = _crossing_map(logical)
    triple = (r, c)
    assert set(crossing_map) == {triple, (r, c + 1), (r + 1, c)}
    assert len(crossing_map[triple]) == 3
    assert len(crossing_map[(r, c + 1)]) == 2
    assert len(crossing_map[(r + 1, c)]) == 2

    red_marker = (r + 1, 2)
    yellow_pivot = (y, c)
    yellow_dir = (0, 1)
    pink_marker = triple
    return logical, red_marker, pink_marker, yellow_pivot, yellow_dir


def _transform_scene(
    logical: dict[int, set[tuple[int, int]]],
    red_marker: tuple[int, int],
    pink_marker: tuple[int, int],
    yellow_pivot: tuple[int, int],
    yellow_dir: tuple[int, int],
    mode: int,
) -> tuple[dict[int, set[tuple[int, int]]], tuple[int, int], tuple[int, int], tuple[int, int], tuple[int, int]]:
    tlogical = {color: {_transform_pt(pt, mode, H, W) for pt in pts} for color, pts in logical.items()}
    return (
        tlogical,
        _transform_pt(red_marker, mode, H, W),
        _transform_pt(pink_marker, mode, H, W),
        _transform_pt(yellow_pivot, mode, H, W),
        _transform_dir(yellow_dir, mode),
    )


def _render_target(
    logical: dict[int, set[tuple[int, int]]],
    comp_count: dict[int, int],
    yellow_rays: dict[int, set[tuple[int, int]]],
    pink_forced: dict[tuple[int, int], int],
) -> np.ndarray:
    out = np.full((H, W), BG, dtype=int)
    for pt in set().union(*logical.values()):
        colors_here = [color for color in logical if pt in logical[color]]
        forced = [color for color in colors_here if pt in yellow_rays.get(color, set())]
        if forced:
            out[pt] = forced[0]
        elif pt in pink_forced:
            out[pt] = pink_forced[pt]
        elif len(colors_here) == 1:
            out[pt] = colors_here[0]
        else:
            out[pt] = min(colors_here, key=lambda color: (-comp_count[color], color))
    return out


def _marker_isolation_ok(grid: np.ndarray) -> bool:
    for color in KNOWN_MARKERS:
        for pt in map(tuple, np.argwhere(grid == color)):
            if any(grid[nb] == color for nb in _neighbors(pt, H, W)):
                return False
    return True


def _build_input_for_crossing_display(
    logical: dict[int, set[tuple[int, int]]],
    red_marker: tuple[int, int],
    pink_marker: tuple[int, int],
    yellow_pivot: tuple[int, int],
    yellow_dir: tuple[int, int],
    crossing_display: dict[tuple[int, int], int],
) -> tuple[np.ndarray, np.ndarray, dict[str, int | bool]] | None:
    marker = (yellow_pivot[0] + yellow_dir[0], yellow_pivot[1] + yellow_dir[1])
    if not (0 <= marker[0] < H and 0 <= marker[1] < W):
        return None
    if marker not in logical[YELLOW_LINE]:
        return None

    target_ray = _ray_from(yellow_pivot, yellow_dir, H, W) & logical[YELLOW_LINE]
    removed_for_yellow = target_ray - {yellow_pivot}
    assert pink_marker not in removed_for_yellow

    old_arm_len = random.randint(2, min(3, 11))
    old_arm = set()
    r, c = yellow_pivot[0] - yellow_dir[0], yellow_pivot[1] - yellow_dir[1]
    while 0 <= r < H and 0 <= c < W and (r, c) not in set().union(*logical.values()):
        old_arm.add((r, c))
        if len(old_arm) == old_arm_len:
            break
        r -= yellow_dir[0]
        c -= yellow_dir[1]
    if len(old_arm) < 2:
        return None

    crossings = _crossing_map(logical)
    inp = np.full((H, W), BG, dtype=int)
    comp_count_masks = {color: set() for color in LINE_COLORS}
    yellow_rays = {color: set() for color in LINE_COLORS}
    yellow_rays[YELLOW_LINE] = set(target_ray)
    pink_forced = {pink_marker: PINK_LINE}

    for color, pts in logical.items():
        for pt in pts:
            if color == YELLOW_LINE and pt in removed_for_yellow:
                continue
            if pt == red_marker or pt == pink_marker or pt == marker:
                continue
            if pt in crossings and crossing_display.get(pt) != color:
                continue
            inp[pt] = color
            comp_count_masks[color].add(pt)

    for pt in old_arm:
        if inp[pt] != BG:
            return None
        inp[pt] = YELLOW_LINE
        comp_count_masks[YELLOW_LINE].add(pt)

    inp[red_marker] = RED
    comp_count_masks[RED_LINE].add(red_marker)
    inp[pink_marker] = PINK
    comp_count_masks[PINK_LINE].add(pink_marker)
    inp[marker] = YELLOW

    if not _marker_isolation_ok(inp):
        return None
    if not any(inp[nb] == YELLOW_LINE for nb in _neighbors(marker, H, W)):
        return None

    comp_count = {color: _component_count((H, W), comp_count_masks[color]) for color in LINE_COLORS}
    target = _render_target(logical, comp_count, yellow_rays, pink_forced)
    target_no_pink = _render_target(logical, comp_count, yellow_rays, {})

    if target[pink_marker] != PINK_LINE:
        return None
    if target_no_pink[pink_marker] == target[pink_marker]:
        return None
    if inp[pink_marker] == target[pink_marker]:
        return None

    counts = {color: len(components_mask(target == color, connectivity=4)) for color in LINE_COLORS}
    if not all(1 <= counts[color] <= 4 for color in LINE_COLORS):
        return None
    if not 4 <= sum(counts.values()) <= 6:
        return None
    if not FILL_RANGE[0] <= int(np.count_nonzero(target)) <= FILL_RANGE[1]:
        return None

    try:
        solved = main(inp)
    except Exception:
        return None
    if not np.array_equal(solved, target):
        return None

    metrics = {
        "triple_crossings": sum(len(v) == 3 for v in crossings.values()),
        "pink_changes_winner": bool(target_no_pink[pink_marker] != target[pink_marker]),
        "yellow_removed_cells": len(removed_for_yellow),
    }
    return inp, target, metrics


def _derive_input(
    logical: dict[int, set[tuple[int, int]]],
    red_marker: tuple[int, int],
    pink_marker: tuple[int, int],
    yellow_pivot: tuple[int, int],
    yellow_dir: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray, dict[str, int | bool]] | None:
    crossings = _crossing_map(logical)
    pairwise = [pt for pt, colors in crossings.items() if len(colors) == 2]
    options = [crossings[pt] for pt in pairwise]
    random.shuffle(pairwise)

    trials: list[dict[tuple[int, int], int]] = []
    if len(pairwise) == 2:
        a, b = pairwise
        ca = crossings[a]
        cb = crossings[b]
        trials = [
            {a: ca[0], b: cb[0], pink_marker: PINK_LINE},
            {a: ca[0], b: cb[1], pink_marker: PINK_LINE},
            {a: ca[1], b: cb[0], pink_marker: PINK_LINE},
            {a: ca[1], b: cb[1], pink_marker: PINK_LINE},
        ]
    else:
        trials = [{pt: colors[0] for pt, colors in crossings.items()}]

    random.shuffle(trials)
    for display in trials:
        built = _build_input_for_crossing_display(logical, red_marker, pink_marker, yellow_pivot, yellow_dir, display)
        if built is not None:
            return built
    return None


def _gen_attempt(return_meta: bool = False):
    logical, red_marker, pink_marker, yellow_pivot, yellow_dir = _build_scene()
    mode = random.randint(0, 3)
    logical, red_marker, pink_marker, yellow_pivot, yellow_dir = _transform_scene(
        logical, red_marker, pink_marker, yellow_pivot, yellow_dir, mode
    )
    derived = _derive_input(logical, red_marker, pink_marker, yellow_pivot, yellow_dir)
    if derived is None:
        return None
    inp, target, metrics = derived
    if return_meta:
        return inp, target, metrics
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("failed to generate a valid fused faa9f03d input")
