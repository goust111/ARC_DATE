# Combo variant of faa9f03d (index 6)
# combo_spec: D2/2+D8/1
# Name: 2像素宽线条 + 5-6种高密度线色

from __future__ import annotations

import random

import numpy as np

from arc_utils import components_mask, get_bg


# Design analysis for the fused variant
# a) Concrete behavior:
#    The original task works on a 12x12 single-pixel skeleton. This variant first
#    promotes every logical line cell into a 2x2 slot on a 24x24 grid. A line slot
#    is a solid 2x2 block, while a marker still occupies exactly one pixel inside an
#    otherwise empty 2x2 slot. Example: a logical horizontal run [A][gap-red][A]
#    becomes two 2x2 line blocks with one empty 2x2 slot containing a single red
#    pixel; solving colors the full missing slot, not just the marker pixel.
# b) Complexity calibration:
#    Using 5-6 colors on a 10x10 logical grid would be unreadable if arbitrary
#    polylines overlapped freely. To keep the rule inferable, generation uses one
#    dominant serpentine color plus 4-5 vertical crossing colors. This creates many
#    crossings, but every overlap still has a clean "horizontal vs vertical" reading.
# c) Hidden assumptions audit:
#    The original solver assumes every visible line cell is itself the atomic graph
#    node. That breaks for width-2 ribbons. The new solver therefore normalizes the
#    fine grid into 2x2 logical slots first; all original support / gap / control-ray
#    reasoning is then performed on that slot graph.
# d) Placement legality:
#    The generator guarantees that line colors only occupy full 2x2 slots, markers
#    only occupy singleton pixels inside empty slots, crossings are only pairwise,
#    and vertical colors never share the serpentine's turn columns, preventing
#    same-orientation overlaps or triple crossings.
# e) Visual observability:
#    Humans can still infer the original repair rule because markers remain vivid
#    isolated pixels, while the widened ribbons make the "missing block", "wrong
#    crossing color", and "yellow redirect" effects visible at a glance.
# f) Visual cleanliness:
#    Restricting crossings to one dominant serpentine against straight vertical
#    colors keeps the dense 5-6-color network legible instead of becoming a tangled
#    multicolor braid.
# g) Inferability of changes:
#    No hidden color remapping is introduced. The only color-specific roles are the
#    same structural ones as the original task: singleton red/yellow markers versus
#    ordinary line colors. The fusion changes geometry and density, not semantics.


DIRS4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
BG = 0
SCALE = 2
LH = LW = 12
H = W = LH * SCALE
LINE_PALETTE = [1, 3, 5, 6, 7, 8, 9]
MARKER_PIXEL = {2: (0, 0), 4: (1, 1)}
SERP_TURN_COLS = (1, 10)
GEN_RETRIES = 400
RED_COUNT_RANGE = (3, 4)


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


def _compress_to_logical(grid: np.ndarray) -> tuple[np.ndarray, int]:
    g = np.asarray(grid, dtype=int)
    bg = int(get_bg(g))
    h, w = g.shape
    assert h % SCALE == 0 and w % SCALE == 0, "grid must split cleanly into 2x2 slots"

    logical = np.full((h // SCALE, w // SCALE), bg, dtype=int)
    for br in range(logical.shape[0]):
        for bc in range(logical.shape[1]):
            block = g[br * SCALE : (br + 1) * SCALE, bc * SCALE : (bc + 1) * SCALE]
            vals = [int(v) for v in block.ravel() if int(v) != bg]
            if not vals:
                continue
            assert len(set(vals)) == 1, f"mixed-color slot at {(br, bc)}"
            if len(vals) == SCALE * SCALE:
                logical[br, bc] = vals[0]
                continue
            assert len(vals) == 1, f"partial line block at {(br, bc)}"
            logical[br, bc] = vals[0]
    return logical, bg


def _expand_output(logical: np.ndarray, bg: int) -> np.ndarray:
    out = np.full((logical.shape[0] * SCALE, logical.shape[1] * SCALE), bg, dtype=int)
    for r in range(logical.shape[0]):
        for c in range(logical.shape[1]):
            color = int(logical[r, c])
            if color == bg:
                continue
            out[r * SCALE : (r + 1) * SCALE, c * SCALE : (c + 1) * SCALE] = color
    return out


def _expand_input(logical: np.ndarray) -> np.ndarray:
    out = np.full((logical.shape[0] * SCALE, logical.shape[1] * SCALE), BG, dtype=int)
    for r in range(logical.shape[0]):
        for c in range(logical.shape[1]):
            color = int(logical[r, c])
            if color == BG:
                continue
            if color in MARKER_PIXEL:
                dr, dc = MARKER_PIXEL[color]
                out[r * SCALE + dr, c * SCALE + dc] = color
            else:
                out[r * SCALE : (r + 1) * SCALE, c * SCALE : (c + 1) * SCALE] = color
    return out


def _solve_logical(grid: np.ndarray) -> np.ndarray:
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
            r, c = pivot
            while 0 <= r < h and 0 <= c < w:
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
                nbrs = [(nr, nc) for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)) if (nr, nc) in pts_set]
                if len(nbrs) != 1:
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
            value = int(out[r, c])
            if value in line_colors:
                color_cells[value].add((r, c))

    crossing_colors = set()
    for r in range(h):
        for c in range(w):
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


def main(grid: np.ndarray) -> np.ndarray:
    """
    Repair a widened rectilinear color network by reading each 2x2 slot as one logical line cell, then applying the original marker-and-crossing logic on that slot graph.

    Input structure:
    - The visible grid is 24x24 and background-dominated.
    - Every ordinary line cell is rendered as a solid 2x2 block, so horizontal/vertical polylines look 2 pixels wide.
    - Red and yellow controls remain singleton pixels, each sitting alone inside an otherwise empty 2x2 slot.
    - There are 5-6 non-marker line colors, producing a denser crossing pattern than the base task.

    Rule:
    - First collapse the grid into 2x2 logical slots: full-color blocks become line cells, singleton pixels become marker cells.
    - On that logical grid, assign each marker to the nearest visible line color in row/column line-of-sight, fill the missing orthogonal gaps, and use yellow controls to delete the obsolete arm and extend the chosen color toward the marked side until the boundary.
    - When multiple logical lines claim the same slot, yellow-control rays override everything else; otherwise the displayed color is the one whose pre-repair logical network is more fragmented, with lower color id as the fallback tie-break.

    Key insight:
    The geometric change is only in the rasterization: the real combinatorics still live on a coarser slot lattice. Once each 2x2 ribbon segment is treated as one logical cell, the same repair logic as the original task explains the widened, high-density examples.
    """
    logical, bg = _compress_to_logical(grid)
    solved = _solve_logical(logical)
    return _expand_output(solved, bg)


def _neighbors(pt: tuple[int, int]) -> list[tuple[int, int]]:
    r, c = pt
    return [(r + dr, c + dc) for dr, dc in DIRS4 if 0 <= r + dr < LH and 0 <= c + dc < LW]


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
    mask = np.zeros((LH, LW), dtype=bool)
    for r, c in pts:
        mask[r, c] = True
    return len(components_mask(mask, connectivity=4))


def _on_boundary(r: int, c: int) -> bool:
    return r == 0 or r == LH - 1 or c == 0 or c == LW - 1


def _valid_path(cells: list[tuple[int, int]]) -> bool:
    pts = set(cells)
    if len(pts) != len(cells) or len(pts) < 2:
        return False
    for r, c in pts:
        if not (0 <= r < LH and 0 <= c < LW):
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
            if (r in (0, LH - 1) and nr == r) or (c in (0, LW - 1) and nc == c):
                return False
        nbrs = _dirs_in(pts, (r, c))
        if len(nbrs) != 1:
            continue
        if (r in (0, LH - 1)) and (c in (0, LW - 1)):
            continue
        dr, dc = nbrs[0]
        if not ((r == 0 and dr == 1) or (r == LH - 1 and dr == -1) or (c == 0 and dc == 1) or (c == LW - 1 and dc == -1)):
            return False
    for r, c in pts:
        if len(_dirs_in(pts, (r, c))) == 1 and not _on_boundary(r, c):
            return False
    return True


def _pick_spaced_rows(count: int) -> list[int] | None:
    choices = list(range(1, LH - 1))
    for _ in range(80):
        rows = sorted(random.sample(choices, count))
        if all(b - a >= 2 for a, b in zip(rows, rows[1:])):
            return rows
    return None


def _build_dominant_serpentine(rows: list[int], start_left: bool) -> list[tuple[int, int]]:
    c0, c1 = SERP_TURN_COLS
    start_boundary = 0 if start_left else LW - 1
    current = c1 if start_left else c0
    pts = [(rows[0], start_boundary), (rows[0], current)]
    for row in rows[1:]:
        pts.append((row, current))
        current = c0 if current == c1 else c1
        pts.append((row, current))
    end_boundary = 0 if current == c0 else LW - 1
    pts.append((rows[-1], end_boundary))
    cells = _path(pts)
    assert _valid_path(cells)
    return cells


def _build_vertical_path(col: int) -> list[tuple[int, int]]:
    cells = _path([(0, col), (LH - 1, col)])
    assert _valid_path(cells)
    return cells


def _crossing_cells(logical: dict[int, set[tuple[int, int]]]) -> list[tuple[int, int]]:
    cells = set().union(*logical.values())
    return [pt for pt in cells if sum(pt in logical[color] for color in logical) > 1]


def _valid_crossings(logical: dict[int, set[tuple[int, int]]], dominant: int) -> bool:
    crossings = _crossing_cells(logical)
    if len(crossings) < len(logical) - 1:
        return False
    for pt in crossings:
        colors = [color for color in logical if pt in logical[color]]
        if len(colors) != 2 or dominant not in colors:
            return False
        orientations = [_orientation(logical[color], pt) for color in colors]
        if None in orientations or sorted(orientations) != ["h", "v"]:
            return False
    return True


def _marker_isolation_ok(grid: np.ndarray) -> bool:
    for color in (2, 4):
        for pt in map(tuple, np.argwhere(grid == color)):
            if any(grid[nb] == color for nb in _neighbors(pt)):
                return False
    return True


def _ray_from(pivot: tuple[int, int], direction: tuple[int, int]) -> set[tuple[int, int]]:
    out = set()
    r, c = pivot
    dr, dc = direction
    while 0 <= r < LH and 0 <= c < LW:
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
                    if not (0 <= mr < LH and 0 <= mc < LW) or (mr, mc) not in comp:
                        continue
                    target_ray = _ray_from(pivot, out_dir)
                    if not target_ray.issubset(comp):
                        continue
                    old_dir = (-out_dir[0], -out_dir[1])
                    old_cells: set[tuple[int, int]] = set()
                    r, c = pivot[0] + old_dir[0], pivot[1] + old_dir[1]
                    while 0 <= r < LH and 0 <= c < LW and (r, c) not in occupied:
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
    dominant: int,
    count: int,
) -> list[tuple[int, tuple[int, int]]]:
    dom_candidates = []
    all_candidates = []
    for color, comps in paths.items():
        for comp in comps:
            for pt in comp:
                if pt in blocked or not _is_turn_or_endpoint(comp, pt):
                    continue
                item = (color, pt)
                all_candidates.append(item)
                if color == dominant:
                    dom_candidates.append(item)
    random.shuffle(dom_candidates)
    random.shuffle(all_candidates)

    chosen: list[tuple[int, tuple[int, int]]] = []
    used: set[tuple[int, int]] = set(blocked)
    per_color_left = {color: len(logical[color]) for color in logical}

    for pool in (dom_candidates, all_candidates):
        for color, pt in pool:
            if len(chosen) == count:
                break
            if per_color_left[color] <= 2:
                continue
            if any(nb in used for nb in [pt] + _neighbors(pt)):
                continue
            if (color, pt) in chosen:
                continue
            chosen.append((color, pt))
            used.add(pt)
            per_color_left[color] -= 1
        if len(chosen) == count:
            break
    return chosen if len(chosen) == count else []


def _render_logical_target(
    logical: dict[int, set[tuple[int, int]]],
    comp_count: dict[int, int],
    yellow_rays: dict[int, set[tuple[int, int]]],
) -> np.ndarray:
    out = np.full((LH, LW), BG, dtype=int)
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


def _build_logical() -> tuple[dict[int, set[tuple[int, int]]], dict[int, list[set[tuple[int, int]]]], int] | None:
    n_colors = random.randint(5, 6)
    colors = random.sample(LINE_PALETTE, n_colors)
    dominant = random.choice(colors)
    others = [color for color in colors if color != dominant]

    row_count = random.randint(3, 4)
    rows = _pick_spaced_rows(row_count)
    if rows is None:
        return None

    columns = sorted(random.sample(list(range(SERP_TURN_COLS[0] + 2, SERP_TURN_COLS[1] - 1)), len(others)))

    logical: dict[int, set[tuple[int, int]]] = {color: set() for color in colors}
    paths: dict[int, list[set[tuple[int, int]]]] = {color: [] for color in colors}

    dom_cells = _build_dominant_serpentine(rows, start_left=bool(random.randint(0, 1)))
    logical[dominant].update(dom_cells)
    paths[dominant].append(set(dom_cells))

    for color, col in zip(others, columns):
        cells = _build_vertical_path(col)
        logical[color].update(cells)
        paths[color].append(set(cells))

    if not _valid_crossings(logical, dominant):
        return None
    return logical, paths, dominant


def _derive_logical_input(
    logical: dict[int, set[tuple[int, int]]],
    paths: dict[int, list[set[tuple[int, int]]]],
    dominant: int,
) -> tuple[np.ndarray, np.ndarray] | None:
    crossings = set(_crossing_cells(logical))

    yellow_infos = []
    blocked = set(crossings)
    for cand in _yellow_candidates(logical, paths, crossings):
        color, pivot, out_dir, target_ray, old_arm = cand
        marker = (pivot[0] + out_dir[0], pivot[1] + out_dir[1])
        if marker in blocked:
            continue
        yellow_infos.append(cand)
        blocked.update(target_ray | old_arm | {marker})
        break
    if len(yellow_infos) != 1:
        return None

    red_count = random.randint(*RED_COUNT_RANGE)
    red_markers = _choose_red_markers(logical, paths, blocked, dominant, red_count)
    if len(red_markers) != red_count:
        return None

    inp = np.full((LH, LW), BG, dtype=int)
    comp_count_masks: dict[int, set[tuple[int, int]]] = {c: set() for c in logical}

    crossing_display: dict[tuple[int, int], int] = {}
    for pt in crossings:
        colors = [color for color in logical if pt in logical[color]]
        if len(colors) != 2 or dominant not in colors:
            return None
        crossing_display[pt] = next(color for color in colors if color != dominant)

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
        solved = _solve_logical(inp)
    except Exception:
        return None

    line_colors = sorted(set(int(v) for v in solved.ravel()) - {BG})
    if not 5 <= len(line_colors) <= 6:
        return None

    changed_total = int(np.count_nonzero(solved != inp))
    changed_crossings = sum(int(solved[pt] != inp[pt]) for pt in crossings)
    changed_non_marker = sum(
        int(inp[r, c] not in (BG, 2, 4) and solved[r, c] != inp[r, c]) for r in range(LH) for c in range(LW)
    )

    if changed_total < len(red_markers) + 6:
        return None
    if changed_crossings < max(4, len(crossings) // 3):
        return None
    if changed_non_marker < 3:
        return None

    return inp, solved


def _gen_attempt() -> np.ndarray | None:
    built = _build_logical()
    if built is None:
        return None
    logical, paths, dominant = built
    derived = _derive_logical_input(logical, paths, dominant)
    if derived is None:
        return None
    logical_inp, logical_target = derived
    fine_inp = _expand_input(logical_inp)
    fine_target = _expand_output(logical_target, BG)
    try:
        solved = main(fine_inp)
    except Exception:
        return None
    return fine_inp if np.array_equal(solved, fine_target) else None


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("failed to generate a valid fused faa9f03d input")
