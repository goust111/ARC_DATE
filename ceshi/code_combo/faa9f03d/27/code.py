# Combo variant of faa9f03d (index 27)
# combo_spec: D3/2+D4/2+D15/1
# Name: 统一标记颜色 + 全局最优匹配 + 移除支撑检查
#
# Design analysis for this fused variant
# - Concrete behavior:
#   All markers use one visible singleton color. Their job is inferred from
#   local geometry after assignment: a marker sitting on a missing bend cell is a
#   gap marker, while a marker sitting just outside an exposed tip is a control
#   marker. Assignment itself no longer happens marker-by-marker with local
#   support checks; instead all markers are paired to visible line endpoints by a
#   single global minimum-distance matching.
# - Small traced example:
#   Suppose marker M1 is a bend-gap on the blue line and marker M2 is a control
#   beside the green line. In the input, both markers are color 2. M1 may sit
#   closer to one blue endpoint than to any green endpoint, but M2 may be
#   adjacent to both a blue tip and a green tip. A local greedy rule can attach
#   both markers to blue-side evidence and fail. The fused rule instead chooses a
#   one-to-one global endpoint matching, then uses each chosen line color to read
#   the marker's role from position.
# - Complexity calibration:
#   The generator always includes both roles while keeping the marker set small:
#   exactly one gap marker and one control marker for two line colors. Matching
#   is therefore a visible global pairing problem, but still small enough to
#   inspect by eye from a few examples.
# - Hidden assumptions audited:
#   Original code assumed marker color encoded role, assignment direction came
#   from the first visible line, and gap filling relied on axis/perpendicular
#   support. None of those remain safe. The new solver separates role detection
#   from endpoint matching, computes assignments globally, and lets fill legality
#   depend only on aligned same-color cells and bounded gap length.
# - Placement legality:
#   The generator still works output-first. Control markers are placed next to
#   true visible endpoints; gap markers are placed only on true turns; all marker
#   cells stay isolated; removed rays, old arms, crossings, and marker
#   neighborhoods are blocked from later placements. This guarantees the solver's
#   role tests and control-ray edits stay well-formed.
# - Visual observability and neatness:
#   Reusing one marker color makes role cues purely spatial, while the global
#   matching ambiguity appears as repeated "same-looking markers near several
#   candidate fragments" patterns. Samples with arbitrary equal-cost matchings are
#   rejected, so the visible structure, not hidden color-id tie breaks, explains
#   the result.
# - Inferability of the fused changes:
#   The solver still has to infer role from position alone and match the two
#   markers jointly to the two line colors. Gap filling is limited to bounded
#   one-cell spans so the simplified no-support rule stays visually tidy.

from __future__ import annotations

import itertools
import random

import numpy as np

from arc_utils import components_mask, get_bg


DIRS4 = ((-1, 0), (1, 0), (0, -1), (0, 1))


def _pts(mask: np.ndarray) -> set[tuple[int, int]]:
    return {tuple(map(int, rc)) for rc in np.argwhere(mask)}


def _neighbors(pt: tuple[int, int]) -> list[tuple[int, int]]:
    r, c = pt
    return [(r + dr, c + dc) for dr, dc in DIRS4 if 0 <= r + dr < H and 0 <= c + dc < W]


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


def _is_turn(pts: set[tuple[int, int]], pt: tuple[int, int]) -> bool:
    dirs = _dirs_in(pts, pt)
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


def _visible_endpoints(pts: set[tuple[int, int]]) -> list[tuple[int, int]]:
    return sorted(pt for pt in pts if len(_dirs_in(pts, pt)) == 1)


def _marker_role(
    original: dict[int, set[tuple[int, int]]],
    pt: tuple[int, int],
    color: int,
) -> tuple[str, tuple[int, int] | None] | None:
    r0, c0 = pt
    nbr_dirs = [(dr, dc) for dr, dc in DIRS4 if (r0 + dr, c0 + dc) in original[color]]
    if len(nbr_dirs) == 1:
        dr, dc = nbr_dirs[0]
        pivot = (r0 + dr, c0 + dc)
        if len(_dirs_in(original[color], pivot)) == 1:
            return ("control", (dr, dc))
        return None
    if len(nbr_dirs) == 2 and (nbr_dirs[0][0] + nbr_dirs[1][0], nbr_dirs[0][1] + nbr_dirs[1][1]) != (0, 0):
        pivots = [(r0 + dr, c0 + dc) for dr, dc in nbr_dirs]
        if all(len(_dirs_in(original[color], pivot)) == 1 for pivot in pivots):
            return ("gap", None)
    return None


def _assign_markers(
    original: dict[int, set[tuple[int, int]]],
    markers: list[tuple[int, int]],
    mode: str,
) -> tuple[
    dict[tuple[int, int], int],
    dict[tuple[int, int], tuple[int, int]],
    dict[tuple[int, int], tuple[str, tuple[int, int] | None]],
    int,
    int | None,
]:
    endpoints = {color: _visible_endpoints(pts) for color, pts in original.items()}
    assert all(endpoints.values()), "expected visible line endpoints"
    colors = sorted(original)
    assert len(markers) == len(colors), "global color matching expects one marker per line color"

    roles: dict[tuple[int, int], dict[int, tuple[str, tuple[int, int] | None]]] = {}
    candidates: dict[tuple[int, int], list[int]] = {}
    nearest_endpoint: dict[tuple[tuple[int, int], int], tuple[int, int]] = {}
    for marker in markers:
        roles[marker] = {}
        cand: list[int] = []
        for color in colors:
            role = _marker_role(original, marker, color)
            if role is None:
                continue
            roles[marker][color] = role
            cand.append(color)
            nearest_endpoint[(marker, color)] = min(
                endpoints[color],
                key=lambda pt: (abs(marker[0] - pt[0]) + abs(marker[1] - pt[1]), pt[0], pt[1]),
            )
        assert cand, f"marker at {marker} has no feasible line color"
        candidates[marker] = cand

    if mode == "local":
        assignment: dict[tuple[int, int], int] = {}
        matched: dict[tuple[int, int], tuple[int, int]] = {}
        role_choice: dict[tuple[int, int], tuple[str, tuple[int, int] | None]] = {}
        total = 0
        for marker in sorted(markers):
            color = min(
                candidates[marker],
                key=lambda c: (
                    abs(marker[0] - nearest_endpoint[(marker, c)][0]) + abs(marker[1] - nearest_endpoint[(marker, c)][1]),
                    nearest_endpoint[(marker, c)][0],
                    nearest_endpoint[(marker, c)][1],
                    c,
                ),
            )
            endpoint = nearest_endpoint[(marker, color)]
            assignment[marker] = color
            matched[marker] = endpoint
            role_choice[marker] = roles[marker][color]
            total += abs(marker[0] - endpoint[0]) + abs(marker[1] - endpoint[1])
        return assignment, matched, role_choice, total, None

    order = sorted(markers, key=lambda pt: (len(candidates[pt]), pt[0], pt[1]))
    solutions: list[tuple[int, tuple[int, ...]]] = []

    def rec(i: int, used: set[int], chosen: list[int], total: int) -> None:
        if i == len(order):
            solutions.append((total, tuple(chosen)))
            return
        marker = order[i]
        for idx in sorted(
            candidates[marker],
            key=lambda c: (
                abs(marker[0] - nearest_endpoint[(marker, c)][0]) + abs(marker[1] - nearest_endpoint[(marker, c)][1]),
                nearest_endpoint[(marker, c)][0],
                nearest_endpoint[(marker, c)][1],
                c,
            ),
        ):
            if idx in used:
                continue
            endpoint = nearest_endpoint[(marker, idx)]
            dist = abs(marker[0] - endpoint[0]) + abs(marker[1] - endpoint[1])
            used.add(idx)
            chosen.append(idx)
            rec(i + 1, used, chosen, total + dist)
            chosen.pop()
            used.remove(idx)

    rec(0, set(), [], 0)
    assert solutions, "no injective global endpoint matching for markers"
    solutions.sort(
        key=lambda item: (
            item[0],
            tuple((nearest_endpoint[(marker, color)][0], nearest_endpoint[(marker, color)][1], color) for marker, color in zip(order, item[1])),
        )
    )
    best_cost, best_idxs = solutions[0]
    second_cost = solutions[1][0] if len(solutions) > 1 else None

    assignment = {}
    matched = {}
    role_choice = {}
    for marker, idx in zip(order, best_idxs):
        assignment[marker] = idx
        endpoint = nearest_endpoint[(marker, idx)]
        matched[marker] = endpoint
        role_choice[marker] = roles[marker][idx]
    return assignment, matched, role_choice, best_cost, second_cost


def _solve(
    grid: np.ndarray,
    *,
    match_mode: str,
    support_mode: bool,
    return_debug: bool = False,
) -> np.ndarray | tuple[np.ndarray, dict[str, object]]:
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
    marker_mask = np.isin(g, list(marker_colors))
    markers = sorted(_pts(marker_mask))
    assert markers, "expected at least one marker"

    assignment, matched, role_choice, best_cost, second_cost = _assign_markers(original, markers, match_mode)

    control_markers: set[tuple[int, int]] = set()
    control_dirs: dict[tuple[int, int], tuple[int, int]] = {}
    gap_marker_pts = {color: set() for color in line_colors}
    for marker, color in assignment.items():
        role, direction = role_choice[marker]
        if role == "control":
            assert direction is not None
            control_markers.add(marker)
            control_dirs[marker] = direction
        else:
            gap_marker_pts[color].add(marker)

    masks = {color: set(pts) for color, pts in original.items()}
    marker_pts = {color: set() for color in line_colors}
    for marker, color in assignment.items():
        masks[color].add(marker)
        marker_pts[color].add(marker)

    comp_count = {color: _component_count(g.shape, masks[color]) for color in line_colors}

    def has_axis_support(color: int, r: int, c: int, axis: int) -> bool:
        return (r, c) in marker_pts[color] or _axis_support(masks[color], r, c, axis)

    def should_fill(color: int, a: tuple[int, int], b: tuple[int, int], axis: int) -> bool:
        r1, c1 = a
        r2, c2 = b
        gap = c2 - c1 - 1 if axis == 0 else r2 - r1 - 1
        if gap <= 0 or gap > MAX_FILL_GAP:
            return False
        if not support_mode:
            return True

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
    for control in sorted(control_markers):
        color = assignment[control]
        dr, dc = control_dirs[control]

        r, c = control[0] + dr, control[1] + dc
        pivot: tuple[int, int] | None = None
        while 0 <= r < h and 0 <= c < w:
            if (r, c) in masks[color]:
                pivot = (r, c)
                break
            r += dr
            c += dc
        assert pivot is not None, f"control marker at {control} has no assigned endpoint"

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

            assert old_arm, f"control marker at {control} found no old arm to erase"
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
                    0 <= r + dr < h and 0 <= c + dc < w and int(out[r + dr, c + dc]) not in (bg, color)
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

    if not return_debug:
        return out
    return out, {
        "markers": markers,
        "assignment": assignment,
        "matched_endpoints": matched,
        "roles": role_choice,
        "best_cost": best_cost,
        "second_cost": second_cost,
    }


def _safe_solve(grid: np.ndarray, *, match_mode: str, support_mode: bool) -> np.ndarray | None:
    try:
        return _solve(grid, match_mode=match_mode, support_mode=support_mode)
    except Exception:
        return None


def variant_metrics(grid: np.ndarray) -> dict[str, object]:
    out, debug = _solve(grid, match_mode="global", support_mode=False, return_debug=True)
    local = _safe_solve(grid, match_mode="local", support_mode=False)
    support = _safe_solve(grid, match_mode="global", support_mode=True)
    _local_out, local_debug = _solve(grid, match_mode="local", support_mode=False, return_debug=True)
    changed = sum(
        debug["matched_endpoints"][marker] != local_debug["matched_endpoints"][marker]  # type: ignore[index]
        for marker in debug["markers"]  # type: ignore[index]
    )
    return {
        "output": out,
        "marker_count": len(debug["markers"]),  # type: ignore[arg-type]
        "best_cost": debug["best_cost"],
        "second_cost": debug["second_cost"],
        "global_vs_local_endpoint_changes": changed,
        "local_matches_global": local is not None and np.array_equal(local, out),
        "support_matches_global": support is not None and np.array_equal(support, out),
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Repair a one-cell-thick colored line network whose singleton markers all share one visible color, using a global endpoint assignment and support-free gap filling.

    Input structure:
    - A rectangular grid with a dominant background.
    - Non-marker colors form one-cell-thick horizontal/vertical polyline fragments that may cross.
    - Marker colors are exactly those colors whose 4-connected components are all singleton pixels.
    - All markers use the same visible color; a marker on a removed bend cell is a gap marker, while a marker just outside an exposed tip is a redirection control.

    Rule:
    - For each line color, measure a marker's distance to that color by the nearest visible endpoint, then assign all markers jointly to distinct line colors by global minimum Manhattan-distance matching.
    - For the assigned line color, read the marker's function from local position: a bend-cell marker restores a gap, while an endpoint-adjacent marker erases the opposite arm and extends the line through the marker to the boundary.
    - Fill aligned same-color gaps whenever two consecutive cells in a row or column face each other with a bounded gap; no axis-support or perpendicular-support check is applied.
    - At overlaps, control-ray cells win; otherwise crossings are resolved by pre-repair fragmentation, with lower color id breaking ties only after the geometric claims are established.

    Key insight: Marker color no longer tells you either the job or the owner. Position tells you the job, and ownership comes from a single global spatial matching against visible endpoints. Once that matching is right, the network repairs itself by simple bounded gap filling with no support-gating.
    """
    return _solve(grid, match_mode="global", support_mode=False)


# --------------- generator ---------------

H = W = 12
BG = 0
LINE_PALETTE = [1, 3, 5, 6, 7, 8, 9]
MARKER_COLOR = 2

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

CONTROL_COUNT = 1
GAP_COUNT = 1
MIN_CELLS_FOR_GAP = 2
MAX_FILL_GAP = 1

COMPONENTS_PER_COLOR = (1, 4)
TOTAL_COMPONENTS = (4, 6)
FILL_RANGE = (33, 55)

SPACED_VALUES_RETRIES = 60
EXTRA_POLYLINE_RETRIES = 80
GEN_RETRIES = 1200


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


def _control_candidates(
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


def _gap_candidates(
    logical: dict[int, set[tuple[int, int]]],
    paths: dict[int, list[set[tuple[int, int]]]],
    blocked: set[tuple[int, int]],
) -> list[tuple[int, tuple[int, int]]]:
    crossings = set(_crossing_cells(logical))
    candidates = []
    for color, comps in paths.items():
        for comp in comps:
            for pt in comp:
                if pt in blocked or not _is_turn(comp, pt) or any(nb in blocked for nb in _neighbors(pt)):
                    continue
                own_nbrs = [(pt[0] + dr, pt[1] + dc) for dr, dc in _dirs_in(comp, pt)]
                if any(nb in crossings for nb in own_nbrs):
                    continue
                candidates.append((color, pt))
    random.shuffle(candidates)
    return candidates


def _find_crossing_point(logical: dict[int, set[tuple[int, int]]]) -> tuple[tuple[int, int], str] | None:
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
    n_colors = 2
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
    control_rays: dict[int, set[tuple[int, int]]],
) -> np.ndarray:
    out = np.full((H, W), BG, dtype=int)
    for pt in set().union(*logical.values()):
        colors_here = [color for color in logical if pt in logical[color]]
        forced = [color for color in colors_here if pt in control_rays.get(color, set())]
        if forced:
            out[pt] = forced[0]
        elif len(colors_here) == 1:
            out[pt] = colors_here[0]
        else:
            out[pt] = min(colors_here, key=lambda color: (-comp_count[color], color))
    return out


def _marker_isolation_ok(grid: np.ndarray) -> bool:
    for pt in map(tuple, np.argwhere(grid == MARKER_COLOR)):
        if any(grid[nb] == MARKER_COLOR for nb in _neighbors(pt)):
            return False
    return True


def _materialize_case(
    logical: dict[int, set[tuple[int, int]]],
    dominant: int,
    crossings: set[tuple[int, int]],
    control_infos: list[tuple[int, tuple[int, int], tuple[int, int], set[tuple[int, int]], set[tuple[int, int]]]],
    gap_markers: list[tuple[int, tuple[int, int]]],
) -> tuple[np.ndarray, np.ndarray] | None:
    inp = np.full((H, W), BG, dtype=int)
    comp_count_masks: dict[int, set[tuple[int, int]]] = {c: set() for c in logical}
    crossing_display: dict[tuple[int, int], int] = {}
    for pt in crossings:
        colors = [color for color in logical if pt in logical[color]]
        if dominant not in colors:
            return None
        crossing_display[pt] = colors[0] if colors[1] == dominant else colors[1]

    removed_for_control: dict[int, set[tuple[int, int]]] = {c: set() for c in logical}
    old_arms: dict[int, set[tuple[int, int]]] = {c: set() for c in logical}
    control_rays: dict[int, set[tuple[int, int]]] = {c: set() for c in logical}
    control_markers: list[tuple[int, tuple[int, int]]] = []
    for color, pivot, out_dir, target_ray, old_arm in control_infos:
        marker = (pivot[0] + out_dir[0], pivot[1] + out_dir[1])
        removed_for_control[color].update(target_ray - {pivot})
        old_arms[color].update(old_arm)
        control_rays[color].update(target_ray)
        control_markers.append((color, marker))

    gap_by_pt = {pt: color for color, pt in gap_markers}
    control_by_pt = {pt: color for color, pt in control_markers}

    for color, pts in logical.items():
        for pt in pts:
            if pt in removed_for_control[color]:
                continue
            if pt in crossings and crossing_display[pt] != color:
                continue
            if pt in gap_by_pt or pt in control_by_pt:
                continue
            inp[pt] = color
            comp_count_masks[color].add(pt)
    for color, pts in old_arms.items():
        for pt in pts:
            if inp[pt] != BG:
                return None
            inp[pt] = color
            comp_count_masks[color].add(pt)

    for pt, color in gap_by_pt.items():
        inp[pt] = MARKER_COLOR
        comp_count_masks[color].add(pt)
    for pt, color in control_by_pt.items():
        inp[pt] = MARKER_COLOR
        comp_count_masks[color].add(pt)

    if not _marker_isolation_ok(inp):
        return None
    for pt, color in control_by_pt.items():
        if not any(inp[nb] == color for nb in _neighbors(pt)):
            return None

    comp_count = {color: _gen_component_count(pts) for color, pts in comp_count_masks.items()}
    target = _render_target(logical, comp_count, control_rays)
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
        solved = _solve(inp, match_mode="global", support_mode=False)
    except Exception:
        return None
    if not np.array_equal(solved, target):
        return None
    return inp, target


def _derive_input(
    logical: dict[int, set[tuple[int, int]]],
    paths: dict[int, list[set[tuple[int, int]]]],
    dominant: int,
) -> tuple[np.ndarray, np.ndarray] | None:
    crossings = set(_crossing_cells(logical))
    controls = _control_candidates(logical, paths, crossings)
    if not controls:
        return None

    base_blocked = set(crossings)
    gap_candidates_all = _gap_candidates(logical, paths, base_blocked)
    if len(gap_candidates_all) < GAP_COUNT:
        return None

    for control in controls[:24]:
        color, pivot, out_dir, target_ray, old_arm = control
        marker = (pivot[0] + out_dir[0], pivot[1] + out_dir[1])
        blocked = set(base_blocked)
        if marker in blocked or any(nb in blocked for nb in _neighbors(marker)):
            continue
        blocked.update(target_ray | old_arm | {marker})

        valid_gaps: list[tuple[int, tuple[int, int]]] = []
        per_color_left = {c: len(logical[c]) for c in logical}
        for gap in gap_candidates_all:
            gap_color, pt = gap
            if gap_color == color:
                continue
            if pt in blocked:
                continue
            if any(nb in blocked for nb in _neighbors(pt)):
                continue
            if per_color_left[gap_color] <= MIN_CELLS_FOR_GAP:
                continue
            valid_gaps.append(gap)

        random.shuffle(valid_gaps)
        tested = 0
        for combo in itertools.combinations(valid_gaps, GAP_COUNT):
            used = set(blocked)
            okay = True
            color_counts: dict[int, int] = {}
            for gap_color, pt in combo:
                color_counts[gap_color] = color_counts.get(gap_color, 0) + 1
                if len(logical[gap_color]) - color_counts[gap_color] <= MIN_CELLS_FOR_GAP:
                    okay = False
                    break
                if any(nb in used for nb in [pt] + _neighbors(pt)):
                    okay = False
                    break
                used.add(pt)
            if not okay:
                continue
            tested += 1
            built = _materialize_case(logical, dominant, crossings, [control], list(combo))
            if built is not None:
                return built
            if tested >= 80:
                break
    return None


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
    raise RuntimeError("failed to generate a valid fused faa9f03d input")
