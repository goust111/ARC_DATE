# Combo variant of faa9f03d (index 8)
# combo_spec: D3/1+D4/1
# Name: 双像素方向标记 + 标记颜色编码目标线色
#
# Design analysis for the combo variant
# =====================================
# a) Concrete behavior and trace
# - The original task had two singleton roles: red gap markers and yellow
#   direction markers. This combo removes that role-by-color split.
# - Every marker is now a 2-cell domino. Its orientation is meaningful:
#   horizontal domino -> only horizontal completion/visibility is valid;
#   vertical domino -> only vertical completion/visibility is valid.
# - Marker color no longer says "gap" or "control". Instead it encodes which
#   line color the marker belongs to through a fixed legend:
#     2 -> 1, 4 -> 3, 6 -> 5, 8 -> 7
# - Role is inferred from geometry:
#   - if the mapped line color is visible on both sides of the domino along its
#     own axis, the domino is a gap marker and both cells must be recolored.
#   - if the mapped line color is visible on exactly one side, the domino is a
#     direction-control marker; the visible side points toward the pivot, while
#     the opposite side is the new ray to be extended.
# - Small trace:
#   output has color-5 cells at `(6,4),(6,5),(6,6),(6,7),(6,8)`.
#   input replaces `(6,5),(6,6)` by marker color 6. The solver first decodes
#   `6 -> 5`, then reads the domino as horizontal, then notices color 5 on both
#   left and right. Therefore it fills only horizontally and restores those two
#   cells as color 5. A vertical interpretation would be wrong.
#   For a control case, output has a color-3 ray from pivot `(4,3)` to the
#   right edge. Input shows a horizontal marker-color-4 domino at `(4,4),(4,5)`
#   and an old color-3 arm extending left of the pivot. The solver decodes
#   `4 -> 3`, sees color 3 only on the left side of the domino, erases that old
#   arm, and extends right through the domino to the boundary.
#
# b) Complexity calibration
# - The legend uses four marker colors and four line colors. That is enough to
#   create multi-color evidence without exploding the palette.
# - Domino orientation repeats across both gap and control cases, so the axis
#   signal is visible several times in a few examples instead of appearing as a
#   one-off exception.
# - Marker role is not another free label; it is derived from whether the
#   mapped line is seen on one side or two. This keeps the combo richer than
#   the seed task without turning it into an arbitrary case distinction.
#
# c) Implicit assumptions that had to change
# - Old `main()` inferred marker colors by singleton components. Invalid now:
#   marker colors are explicit legend colors and every component must be a
#   domino.
# - Old marker assignment used line-of-sight to decide the target line color.
#   Invalid now: color is decoded from the marker itself, while line-of-sight
#   is only used to classify the domino as gap vs control and to locate pivots.
# - Old yellow handling relied on a dedicated yellow color. Invalid now: the new
#   solver must discover control markers from one-sided visibility instead.
# - Old gap logic treated marker pixels as generic support. Invalid now:
#   dominoes only support the axis encoded by their orientation.
#
# d) Placement legality guaranteed by the generator
# - Every marker is exactly one isolated domino of a legend color.
# - Gap dominoes replace two consecutive cells on a straight non-crossing run of
#   their target line, leaving target-color visibility on both sides.
# - Control dominoes occupy the first two cells of the final ray, leaving target
#   visibility only on the pivot side and empty space on the extension side.
# - Halo checks keep marker dominoes from touching and prevent same-color
#   dominoes from merging into larger marker components.
#
# e) Visual observability
# - Humans can see both fused dimensions directly: shape/orientation is visible
#   because markers are dominoes, and color coding is visible because each
#   legend color repeatedly turns into the same output line color.
# - The mapping is fixed globally rather than resampled per grid, so multiple
#   examples can support the same legend instead of forcing the viewer to solve
#   a fresh arbitrary cipher every time.
#
# f) Visual cleanliness
# - Gap markers are restricted to straight runs instead of corners, so the
#   domino shape is readable and does not collide with turn semantics.
# - Control dominoes sit on the new ray rather than beside it, which makes the
#   intended extension direction legible without adding extra marker types.
#
# g) Change inferability
# - The legend uses a one-to-one low-cardinality mapping with fixed colors:
#   marker colors are the even colors and output lines are the preceding odd
#   colors. This is simple enough for ARC-style induction and still visibly
#   distinct from the seed rule where marker color only meant role.
#
# h) New solver assumptions and generator support
# - `main()` assumes a marker color always belongs to the fixed legend and that
#   its mapped line color appears elsewhere in the grid. `_derive_input()`
#   enforces that by only choosing line colors that have marker partners.
# - `main()` assumes a control domino sees its mapped line color on exactly one
#   side along its own axis. `_yellow_candidates()` constructs only such cases.
# - `main()` assumes a gap domino sees its mapped line color on both sides of a
#   straight run. `_choose_red_markers()` only samples those runs.

from __future__ import annotations

import random

import numpy as np

from arc_utils import components_mask, get_bg


DIRS4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
MARKER_TO_LINE = {2: 1, 4: 3, 6: 5, 8: 7}
LINE_TO_MARKER = {line: marker for marker, line in MARKER_TO_LINE.items()}
MARKER_COLORS = set(MARKER_TO_LINE)
LINE_PALETTE = sorted(LINE_TO_MARKER)


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


def _domino_axis(cells: set[tuple[int, int]]) -> int | None:
    if len(cells) != 2:
        return None
    (r1, c1), (r2, c2) = sorted(cells)
    if abs(r1 - r2) + abs(c1 - c2) != 1:
        return None
    return 0 if r1 == r2 else 1


def _axis_dirs(axis: int) -> tuple[tuple[int, int], tuple[int, int]]:
    return ((0, -1), (0, 1)) if axis == 0 else ((-1, 0), (1, 0))


def _anchor_for_direction(cells: tuple[tuple[int, int], tuple[int, int]], direction: tuple[int, int]) -> tuple[int, int]:
    dr, dc = direction
    return min(cells, key=lambda pt: dr * pt[0] + dc * pt[1])


def _scan_for_color(
    g: np.ndarray,
    start: tuple[int, int],
    direction: tuple[int, int],
    target_color: int,
    marker_colors: set[int],
    bg: int,
) -> tuple[int, tuple[int, int]] | None:
    H, W = g.shape
    dr, dc = direction
    r, c = start[0] + dr, start[1] + dc
    dist = 1
    while 0 <= r < H and 0 <= c < W:
        value = int(g[r, c])
        if value == target_color:
            return dist, (r, c)
        if value == bg or value in marker_colors:
            r += dr
            c += dc
            dist += 1
            continue
        return None
    return None


def _marker_info_from_component(
    g: np.ndarray,
    cells: set[tuple[int, int]],
    marker_color: int,
    bg: int,
    original: dict[int, set[tuple[int, int]]],
) -> dict[str, object]:
    axis = _domino_axis(cells)
    assert axis is not None, f"marker color {marker_color} must form domino components"
    target_color = int(MARKER_TO_LINE[marker_color])
    cells_tuple = tuple(sorted(cells))

    visible: list[tuple[tuple[int, int], tuple[int, int], int, tuple[int, int], int]] = []
    for direction in _axis_dirs(axis):
        anchor = _anchor_for_direction(cells_tuple, direction)
        hit = _scan_for_color(g, anchor, direction, target_color, MARKER_COLORS, bg)
        if hit is None:
            continue
        dist, endpoint = hit
        supported = 0 if _axis_support(original[target_color], endpoint[0], endpoint[1], axis) else 1
        visible.append((direction, anchor, dist, endpoint, supported))

    assert visible, f"marker {cells_tuple} cannot see its mapped line color {target_color}"
    if len(visible) == 2:
        return {
            "marker_color": int(marker_color),
            "line_color": target_color,
            "cells": cells_tuple,
            "cells_set": set(cells_tuple),
            "axis": int(axis),
            "role": "gap",
        }

    direction, anchor, _dist, _endpoint, _supported = visible[0]
    return {
        "marker_color": int(marker_color),
        "line_color": target_color,
        "cells": cells_tuple,
        "cells_set": set(cells_tuple),
        "axis": int(axis),
        "role": "control",
        "dir_to_line": direction,
        "anchor": anchor,
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Repair a thin orthogonal line network whose instructions are encoded by color-coded domino markers: the domino's axis tells which direction is valid, and the marker color tells which line color it belongs to.

    Input structure:
    - A rectangular grid with a dominant background.
    - Odd colors in the fixed line palette form one-cell-thick orthogonal stroke fragments.
    - Even colors in the fixed marker palette form isolated 2x1 or 1x2 domino markers.
    - A marker domino does not advertise its role directly: some dominoes replace a two-cell gap on a straight run, while others occupy the first two cells of a redirected endpoint ray.

    Rule:
    - Decode each marker color with the fixed legend `2->1, 4->3, 6->5, 8->7`; then inspect only along the domino's own axis.
    - If the mapped line color is visible on both sides of that axis, the domino is a gap marker and both domino cells are recolored to that line color, supporting completion only on that axis.
    - If the mapped line color is visible on exactly one side, the domino is a control marker: find the pivot on the visible side, erase the old arm on the opposite side, and extend the line from the pivot through the domino to the grid edge.
    - At crossings, control rays win; otherwise an original-vs-filled crossing uses the more fragmented pre-repair line color, while pure filled crossings use the lower color id.

    Key insight: The solver must fuse two independent clues. Marker color no longer tells you "what action to take"; it only identifies the destination line color. The domino's geometry and one-sided vs two-sided visibility determine whether the object repairs a gap or redirects a ray, and its orientation forbids completion on the perpendicular axis.
    """
    g = np.asarray(grid, dtype=int)
    bg = int(get_bg(g))
    H, W = g.shape

    colors = set(map(int, g.ravel())) - {bg}
    assert colors, "expected foreground line and marker colors"

    marker_colors = colors & MARKER_COLORS
    line_colors = colors - MARKER_COLORS
    assert line_colors, "expected at least one non-marker line color"
    assert marker_colors, "expected at least one marker color"
    assert line_colors <= set(LINE_PALETTE), f"unexpected line colors: {line_colors}"

    original = {color: _pts(g == color) for color in line_colors}
    marker_components: list[dict[str, object]] = []
    marker_by_pt: dict[tuple[int, int], dict[str, object]] = {}
    gap_marker_pts = {color: set() for color in line_colors}

    for marker_color in sorted(marker_colors):
        comps = components_mask(g == marker_color, connectivity=4)
        assert comps, "marker color must have at least one component"
        for comp in comps:
            cells = {tuple(map(int, rc)) for rc in comp}
            info = _marker_info_from_component(g, cells, int(marker_color), bg, original)
            marker_components.append(info)
            mapped = int(info["line_color"])
            assert mapped in line_colors, f"mapped line color {mapped} not present"
            for pt in info["cells"]:
                marker_by_pt[pt] = info
                if info["role"] == "gap":
                    gap_marker_pts[mapped].add(pt)

    masks = {color: set(pts) for color, pts in original.items()}
    marker_pts = {color: set() for color in line_colors}
    for info in marker_components:
        color = int(info["line_color"])
        for pt in info["cells"]:
            masks[color].add(pt)
            marker_pts[color].add(pt)

    comp_count = {color: _component_count(g.shape, masks[color]) for color in line_colors}

    def _marker_supports_axis(color: int, pt: tuple[int, int], axis: int) -> bool:
        info = marker_by_pt.get(pt)
        return info is not None and int(info["line_color"]) == color and int(info["axis"]) == axis

    def has_axis_support(color: int, r: int, c: int, axis: int) -> bool:
        return _marker_supports_axis(color, (r, c), axis) or _axis_support(masks[color], r, c, axis)

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
    for info in marker_components:
        if info["role"] != "control":
            continue
        color = int(info["line_color"])
        dr, dc = info["dir_to_line"]
        anchor = info["anchor"]
        cells_set = set(info["cells"])

        r, c = anchor[0] + dr, anchor[1] + dc
        pivot: tuple[int, int] | None = None
        while 0 <= r < H and 0 <= c < W:
            if (r, c) in masks[color] and (r, c) not in cells_set:
                pivot = (r, c)
                break
            r += dr
            c += dc
        assert pivot is not None, f"control marker {info['cells']} has no assigned pivot"

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

            assert old_arm, f"control marker {info['cells']} found no old arm to erase"
            for q in old_arm:
                masks[color].discard(q)
                marker_pts[color].discard(q)
                gap_marker_pts[color].discard(q)
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
        same_row = [mc for mr, mc in gap_marker_pts[color] if mr == r]
        if any(mc < c for mc in same_row) and any(mc > c for mc in same_row):
            return True
        same_col = [mr for mr, mc in gap_marker_pts[color] if mc == c]
        return any(mr < r for mr in same_col) and any(mr > r for mr in same_col)

    for pt, colors_here in claims.items():
        forced = {color for color in colors_here if pt in control_ray[color]}
        if forced:
            assert len(forced) == 1, f"ambiguous control rays at {pt}: {forced}"
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
                nbrs = [(nr, nc) for nr, nc in [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)] if (nr, nc) in pts_set]
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
                if (((r, c - 1) in color_cells[b] and (r, c + 1) in color_cells[b]) or ((r - 1, c) in color_cells[b] and (r + 1, c) in color_cells[b])):
                    crossing_colors.add(a)
                    crossing_colors.add(b)
    for color in line_colors:
        assert color in crossing_colors, f"color {color} has no crossing with any other color"

    return out


# --------------- generator (combo) ---------------

# --- CFG: grid & palette ---
H = W = 12
BG = 0

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
MARKER_MODE_WEIGHTS = [3, 2, 4]
YELLOW_COUNT_RANGE = (1, 2)
RED_COUNT_RANGE = (2, 4)
MIN_CELLS_FOR_RED = 4

# --- CFG: output validation ---
COMPONENTS_PER_COLOR = (1, 4)
TOTAL_COMPONENTS = (4, 6)
FILL_RANGE = (33, 55)

# --- CFG: retry budgets ---
SPACED_VALUES_RETRIES = 60
EXTRA_POLYLINE_RETRIES = 80
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
        if not ((r == 0 and dr == 1) or (r == H - 1 and dr == -1) or (c == 0 and dc == 1) or (c == W - 1 and dc == -1)):
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


def _ordered_ray_from(start: tuple[int, int], direction: tuple[int, int]) -> list[tuple[int, int]]:
    out = []
    r, c = start
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
) -> list[tuple[int, tuple[int, int], tuple[int, int], set[tuple[int, int]], set[tuple[int, int]], tuple[tuple[int, int], tuple[int, int]]]]:
    candidates = []
    occupied = set().union(*logical.values())
    for color, comps in paths.items():
        for comp in comps:
            for pivot in comp:
                if pivot in crossings or not _is_turn_or_endpoint(comp, pivot):
                    continue
                for out_dir in DIRS4:
                    first = (pivot[0] + out_dir[0], pivot[1] + out_dir[1])
                    second = (pivot[0] + 2 * out_dir[0], pivot[1] + 2 * out_dir[1])
                    if first not in comp or second not in comp:
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
                    ordered_old = _ordered_ray_from((pivot[0] + old_dir[0], pivot[1] + old_dir[1]), old_dir)
                    old_arm = set(ordered_old[:old_len])
                    if old_arm and old_arm.issubset(old_cells):
                        candidates.append((color, pivot, out_dir, target_ray, old_arm, (first, second)))
    random.shuffle(candidates)
    return candidates


def _straight_pair_candidates(
    logical: dict[int, set[tuple[int, int]]],
    paths: dict[int, list[set[tuple[int, int]]]],
    blocked: set[tuple[int, int]],
) -> list[tuple[int, tuple[tuple[int, int], tuple[int, int]], int]]:
    crossings = set(_crossing_cells(logical))
    candidates: list[tuple[int, tuple[tuple[int, int], tuple[int, int]], int]] = []
    seen: set[tuple[int, tuple[tuple[int, int], tuple[int, int]]]] = set()
    for color, comps in paths.items():
        for comp in comps:
            for pt in comp:
                for dr, dc, axis, orient in ((0, 1, 0, "h"), (1, 0, 1, "v")):
                    q = (pt[0] + dr, pt[1] + dc)
                    prev = (pt[0] - dr, pt[1] - dc)
                    nxt = (q[0] + dr, q[1] + dc)
                    if q not in comp or prev not in comp or nxt not in comp:
                        continue
                    if pt in blocked or q in blocked or pt in crossings or q in crossings:
                        continue
                    if _orientation(comp, pt) != orient or _orientation(comp, q) != orient:
                        continue
                    pair = tuple(sorted((pt, q)))
                    key = (color, pair)
                    if key in seen:
                        continue
                    seen.add(key)
                    candidates.append((color, pair, axis))
    random.shuffle(candidates)
    return candidates


def _choose_red_markers(
    logical: dict[int, set[tuple[int, int]]],
    paths: dict[int, list[set[tuple[int, int]]]],
    blocked: set[tuple[int, int]],
    count: int,
) -> list[tuple[int, tuple[tuple[int, int], tuple[int, int]], int]]:
    candidates = _straight_pair_candidates(logical, paths, blocked)
    chosen: list[tuple[int, tuple[tuple[int, int], tuple[int, int]], int]] = []
    used = set(blocked)
    per_color_left = {color: len(logical[color]) for color in logical}
    for color, pair, axis in candidates:
        if len(chosen) == count:
            break
        if per_color_left[color] <= MIN_CELLS_FOR_RED:
            continue
        cells = set(pair)
        halo = set(cells)
        for cell in cells:
            halo.update(_neighbors(cell))
        if any(pt in used for pt in halo - cells) or any(pt in used for pt in cells):
            continue
        chosen.append((color, pair, axis))
        used.update(halo)
        per_color_left[color] -= 2
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


def _render_target(logical: dict[int, set[tuple[int, int]]], comp_count: dict[int, int], yellow_rays: dict[int, set[tuple[int, int]]]) -> np.ndarray:
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


def _marker_components_ok(grid: np.ndarray) -> bool:
    for color in MARKER_COLORS:
        for comp in components_mask(grid == color, connectivity=4):
            cells = {tuple(map(int, rc)) for rc in comp}
            if _domino_axis(cells) is None:
                return False
    return True


def _marker_roles_ok(grid: np.ndarray) -> bool:
    bg = int(get_bg(grid))
    line_colors = set(map(int, grid.ravel())) - {bg} - MARKER_COLORS
    if not line_colors:
        return False
    original = {color: _pts(grid == color) for color in line_colors}
    for marker_color in MARKER_COLORS & set(map(int, grid.ravel())):
        mapped = MARKER_TO_LINE[marker_color]
        if mapped not in line_colors:
            return False
        for comp in components_mask(grid == marker_color, connectivity=4):
            cells = {tuple(map(int, rc)) for rc in comp}
            try:
                info = _marker_info_from_component(grid, cells, marker_color, bg, original)
            except Exception:
                return False
            if info["role"] not in {"gap", "control"}:
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
            color, pivot, out_dir, target_ray, old_arm, marker_pair = cand
            marker_cells = set(marker_pair)
            halo = set(marker_cells)
            for cell in marker_cells:
                halo.update(_neighbors(cell))
            if any(pt in blocked for pt in marker_cells) or any(pt in blocked for pt in halo - marker_cells):
                continue
            if any((old & old_arm) or (ray & old_arm) or (set(pair) & old_arm) for _, _, _, ray, old, pair in yellow_infos):
                continue
            yellow_infos.append(cand)
            blocked.update(target_ray | old_arm | marker_cells)
            if len(yellow_infos) == yellow_count:
                break
        if len(yellow_infos) != yellow_count:
            return None

    red_markers = _choose_red_markers(logical, paths, blocked, red_count) if red_count else []
    if red_count and not red_markers:
        return None

    inp = np.full((H, W), BG, dtype=int)
    comp_count_masks: dict[int, set[tuple[int, int]]] = {c: set() for c in logical}
    crossing_display: dict[tuple[int, int], int] = {}
    for pt in crossings:
        colors = [color for color in logical if pt in logical[color]]
        if dominant not in colors:
            return None
        crossing_display[pt] = colors[0] if colors[1] == dominant else colors[1]

    removed_for_yellow: dict[int, set[tuple[int, int]]] = {c: set() for c in logical}
    old_arms: dict[int, set[tuple[int, int]]] = {c: set() for c in logical}
    yellow_rays: dict[int, set[tuple[int, int]]] = {c: set() for c in logical}
    yellow_markers: list[tuple[int, tuple[tuple[int, int], tuple[int, int]]]] = []
    for color, pivot, out_dir, target_ray, old_arm, marker_pair in yellow_infos:
        removed_for_yellow[color].update(target_ray - {pivot})
        old_arms[color].update(old_arm)
        yellow_rays[color].update(target_ray)
        yellow_markers.append((color, marker_pair))

    red_by_pt = {pt: LINE_TO_MARKER[color] for color, pair, _axis in red_markers for pt in pair}
    yellow_by_pt = {pt: LINE_TO_MARKER[color] for color, pair in yellow_markers for pt in pair}

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

    for pt, marker_color in red_by_pt.items():
        if inp[pt] != BG:
            return None
        inp[pt] = marker_color
        comp_count_masks[MARKER_TO_LINE[marker_color]].add(pt)
    for pt, marker_color in yellow_by_pt.items():
        if inp[pt] != BG:
            return None
        inp[pt] = marker_color
        comp_count_masks[MARKER_TO_LINE[marker_color]].add(pt)

    if not _marker_components_ok(inp):
        return None
    if not _marker_roles_ok(inp):
        return None

    comp_count = {color: _gen_component_count(pts) for color, pts in comp_count_masks.items()}
    target = _render_target(logical, comp_count, yellow_rays)
    for pt in crossings:
        if target[pt] == inp[pt]:
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
    raise RuntimeError("failed to generate a valid combo faa9f03d input")
