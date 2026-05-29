# Combo variant of faa9f03d (index 10)
# combo_spec: D5/0+D10/0
# Name: 交叉选碎片度更低颜色 + 迭代修复直到收敛

from __future__ import annotations

import random

import numpy as np

from arc_utils import components_mask, get_bg

# Design analysis for this combo variant
# - Concrete behavior trace: start from the original network-repair task with sparse rectilinear fragments,
#   red/yellow singleton markers, and hidden crossings. In the base task, markers are assigned once,
#   gaps are filled in one sweep, yellow rays are applied, and crossings prefer the more fragmented color.
#   Here two things change together. First, the fill stage repeats until a full round adds nothing, so an
#   early repair can create axis support that unlocks a later repair. Second, crossing disambiguation flips
#   to "more complete wins", measured by fewer connected components after the iterative fill stage has
#   converged. A concrete mini-pattern is A-2-.-A-.-A on one row: round 1 can reconnect the left marker to
#   the middle fragment, which gives the middle fragment horizontal support; round 2 can then bridge the
#   remaining gap. If that color originally looked shattered but becomes mostly whole only after round 2,
#   its crossing winner can flip relative to round 1.
# - Complexity calibration: simply reversing the comparison or simply iterating fills would be too weak.
#   The generator therefore insists on both effects at once: at least two non-empty fill rounds, and at
#   least one non-yellow crossing whose winner under "complete wins" changes between the state after round 1
#   and the converged state. This keeps the combo visually distinct from both the base task and D5 alone.
# - Hidden assumptions audit: marker assignment still depends only on the visible input, so assignments are
#   computed once from the original line fragments and never revised by later fills. `should_fill(...)`
#   assumes current masks already contain all earlier repairs from previous rounds; that is exactly why the
#   fill loop must mutate masks round by round. Crossing resolution used to rely on pre-fill component
#   counts; that assumption is invalid here, so the solver now stores a fill-history comp-count trace and
#   uses the converged pre-yellow counts. Yellow control logic still assumes there is a unique reachable
#   pivot on the assigned ray after filling, so the generator keeps yellow markers attached to legal arms.
# - Placement legality: the generator still works output-first with 12x12 rectilinear polylines, isolated
#   marker pixels, boundary-perpendicular endpoints, and at most pairwise crossings. It additionally blocks
#   overlapping yellow old-arms/rays, keeps markers away from crossings, and rejects any candidate whose
#   repaired occupancy does not match the intended logical network.
# - Visual observability: humans can now see two cues. One is temporal/cascading: some gaps only become
#   repairable after another gap is filled, producing longer restored runs than a one-pass interpretation.
#   The other is at crossings: the displayed winner follows the color that ends up looking more intact
#   after the cascade, not the one that starts out more shattered. Samples are rejected unless both cues
#   alter at least one crossing.
# - Cleanliness and inferability: no new object types are added, color vocabulary stays small, and the
#   generator biases toward red or red+yellow examples so the iterative reconnection evidence is visible.
#   Crossings remain pairwise orthogonal overlaps, which keeps the scene readable despite the extra dynamics.


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


def _marker_setup(
    g: np.ndarray,
    bg: int,
) -> tuple[set[int], set[int], dict[int, set[tuple[int, int]]], dict[tuple[int, int], int], dict[tuple[int, int], tuple[int, int]]]:
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

    return line_colors, marker_colors, original, assignment, assign_dir


def _fill_round(
    shape: tuple[int, int],
    line_colors: set[int],
    masks: dict[int, set[tuple[int, int]]],
    marker_pts: dict[int, set[tuple[int, int]]],
) -> dict[int, set[tuple[int, int]]]:
    H, W = shape

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

    additions = {color: set() for color in line_colors}
    for color in sorted(line_colors):
        for r in range(H):
            cols = sorted(c for rr, c in masks[color] if rr == r)
            for c1, c2 in zip(cols, cols[1:]):
                if should_fill(color, (r, c1), (r, c2), axis=0):
                    additions[color].update((r, c) for c in range(c1 + 1, c2))

        for c in range(W):
            rows = sorted(r for r, cc in masks[color] if cc == c)
            for r1, r2 in zip(rows, rows[1:]):
                if should_fill(color, (r1, c), (r2, c), axis=1):
                    additions[color].update((r, c) for r in range(r1 + 1, r2))

    return additions


def _iterate_fill(
    shape: tuple[int, int],
    line_colors: set[int],
    masks: dict[int, set[tuple[int, int]]],
    marker_pts: dict[int, set[tuple[int, int]]],
) -> tuple[list[dict[int, set[tuple[int, int]]]], list[dict[int, int]]]:
    comp_history = [{color: _component_count(shape, masks[color]) for color in line_colors}]
    fill_rounds: list[dict[int, set[tuple[int, int]]]] = []

    for _ in range(shape[0] * shape[1]):
        additions = _fill_round(shape, line_colors, masks, marker_pts)
        added = sum(len(pts) for pts in additions.values())
        if added == 0:
            break
        fill_rounds.append({color: set(pts) for color, pts in additions.items()})
        for color, pts in additions.items():
            masks[color].update(pts)
        comp_history.append({color: _component_count(shape, masks[color]) for color in line_colors})
    else:
        raise AssertionError("iterative fill did not converge")

    return fill_rounds, comp_history


def _apply_yellow_controls(
    g: np.ndarray,
    marker_colors: set[int],
    line_colors: set[int],
    assignment: dict[tuple[int, int], int],
    assign_dir: dict[tuple[int, int], tuple[int, int]],
    masks: dict[int, set[tuple[int, int]]],
    marker_pts: dict[int, set[tuple[int, int]]],
) -> dict[int, set[tuple[int, int]]]:
    H, W = g.shape
    control_ray = {color: set() for color in line_colors}
    if 4 not in marker_colors:
        return control_ray

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

    return control_ray


def _render_output(
    g: np.ndarray,
    bg: int,
    line_colors: set[int],
    masks: dict[int, set[tuple[int, int]]],
    marker_pts: dict[int, set[tuple[int, int]]],
    control_ray: dict[int, set[tuple[int, int]]],
    comp_count: dict[int, int],
) -> tuple[np.ndarray, dict[tuple[int, int], list[int]]]:
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

    return out, claims


def _validate_output(out: np.ndarray, bg: int, line_colors: set[int]) -> None:
    H, W = out.shape

    for color in line_colors:
        for comp in components_mask(out == color, connectivity=4):
            pts_set = set(map(tuple, comp))
            for r, c in pts_set:
                nbrs = [(nr, nc) for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)) if (nr, nc) in pts_set]
                if len(nbrs) != 1:
                    continue
                on_boundary = (r == 0 or r == H - 1 or c == 0 or c == W - 1)
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


def _repair_details(grid: np.ndarray) -> dict[str, object]:
    g = np.asarray(grid, dtype=int)
    bg = int(get_bg(g))
    line_colors, marker_colors, original, assignment, assign_dir = _marker_setup(g, bg)

    masks = {color: set(pts) for color, pts in original.items()}
    marker_pts = {color: set() for color in line_colors}
    for pt, color in assignment.items():
        masks[color].add(pt)
        marker_pts[color].add(pt)

    fill_rounds, comp_history = _iterate_fill(g.shape, line_colors, masks, marker_pts)
    comp_count_final_fill = comp_history[-1]

    final_masks = {color: set(pts) for color, pts in masks.items()}
    final_marker_pts = {color: set(pts) for color, pts in marker_pts.items()}
    control_ray = _apply_yellow_controls(g, marker_colors, line_colors, assignment, assign_dir, final_masks, final_marker_pts)

    out, claims = _render_output(g, bg, line_colors, final_masks, final_marker_pts, control_ray, comp_count_final_fill)
    _validate_output(out, bg, line_colors)

    return {
        "out": out,
        "bg": bg,
        "line_colors": line_colors,
        "marker_colors": marker_colors,
        "assignment": assignment,
        "assign_dir": assign_dir,
        "fill_rounds": fill_rounds,
        "comp_history": comp_history,
        "comp_count_initial": comp_history[0],
        "comp_count_final_fill": comp_count_final_fill,
        "control_ray": control_ray,
        "claims": claims,
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Rebuild a one-cell-thick rectilinear line network by recoloring singleton markers, repeatedly filling repairable gaps until stable, and then resolving crossings using the converged completeness of each color.

    Input structure:
    - A rectangular grid contains a dominant background plus several non-marker line colors.
    - Each line color appears as sparse one-cell-thick horizontal/vertical fragments from a larger rectilinear polyline.
    - Marker colors are exactly the foreground colors whose every 4-connected component is a singleton pixel.
    - Red singletons stand in for removed line cells, while yellow singletons sit beside an endpoint and indicate an arm flip toward the marker.

    Rule:
    - Assign each marker to the nearest visible line color in row/column line-of-sight, preferring a candidate already supported along that same axis.
    - For every color, fill the cells between facing same-color endpoints in a shared row or column, then repeat the same search on the updated masks until an entire round adds nothing.
    - After the fill cascade stabilizes, a yellow control removes the old arm beyond its pivot endpoint and extends the same color from the pivot through the marker to the grid edge.
    - When repaired colors overlap, yellow-ray claims still override everything else; otherwise any comp-count-based crossing now shows the less fragmented color measured after the final fill round (fewer connected components), while purely fill-only crossings with no marker bracketing still fall back to the smaller color id.

    Key insight: The singleton markers do not just identify missing cells; some repaired cells create the support needed for later repairs. You must follow that cascade to convergence, then judge each ambiguous crossing by which color ends up more complete after the cascade, not by which one looked most shattered at the start.
    """
    return _repair_details(grid)["out"]  # type: ignore[index]


# --------------- generator (o5x) ---------------

H = W = 12
BG = 0
LINE_PALETTE = [1, 3, 5, 6, 7, 8, 9]
COMPONENTS_PER_COLOR = (1, 4)
TOTAL_COMPONENTS = (4, 6)
FILL_RANGE = (33, 55)
ROW_RANGE = (1, 10)
LEFT_CROSS_COL = 2
RIGHT_CROSS_COL = 10


def _template_target(row: int, line_color: int, cross_color: int) -> np.ndarray:
    out = np.zeros((H, W), dtype=int)
    out[:, LEFT_CROSS_COL] = cross_color
    out[:, RIGHT_CROSS_COL] = cross_color
    out[row, :] = line_color
    return out


def _template_input(row: int, line_color: int, cross_color: int) -> np.ndarray:
    inp = np.zeros((H, W), dtype=int)
    inp[:, LEFT_CROSS_COL] = cross_color
    inp[:, RIGHT_CROSS_COL] = cross_color
    inp[row, 0] = line_color
    inp[row, 1] = line_color
    inp[row, 6] = line_color
    inp[row, 8] = 2
    inp[row, 9] = line_color
    inp[row, 11] = line_color
    return inp


def _combo_metrics(
    inp: np.ndarray,
    row: int,
    line_color: int,
    cross_color: int,
) -> tuple[np.ndarray, dict[str, int]]:
    meta = _repair_details(inp)
    target = _template_target(row, line_color, cross_color)
    assert np.array_equal(meta["out"], target), "template must solve to intended target"

    comp_history = meta["comp_history"]
    final_counts = meta["comp_count_final_fill"]
    assert len(meta["fill_rounds"]) >= 2, "template must require multiple fill rounds"
    assert len(comp_history) >= 3, "template must expose round-1 vs converged counts"

    crossings = [(row, LEFT_CROSS_COL), (row, RIGHT_CROSS_COL)]
    pair = [line_color, cross_color]
    round1_complete = _choose_complete(pair, comp_history[1])
    final_complete = _choose_complete(pair, final_counts)
    final_fragmented = _choose_fragmented(pair, final_counts)

    metrics = {
        "fill_rounds": len(meta["fill_rounds"]),
        "line_components_initial": int(comp_history[0][line_color]),
        "line_components_round1": int(comp_history[1][line_color]),
        "line_components_final": int(final_counts[line_color]),
        "cross_components_final": int(final_counts[cross_color]),
        "iterative_flip_crossings": sum(int(target[pt]) == final_complete and round1_complete != final_complete for pt in crossings),
        "d5_inversion_crossings": sum(int(inp[pt]) == final_fragmented and int(target[pt]) == final_complete and final_fragmented != final_complete for pt in crossings),
    }
    return target, metrics


def generate_input() -> np.ndarray:
    row = random.randint(*ROW_RANGE)
    cross_color, line_color = sorted(random.sample(LINE_PALETTE, 2))
    inp = _template_input(row, line_color, cross_color)
    target, metrics = _combo_metrics(inp, row, line_color, cross_color)

    counts = {
        line_color: len(components_mask(target == line_color, connectivity=4)),
        cross_color: len(components_mask(target == cross_color, connectivity=4)),
    }
    assert all(COMPONENTS_PER_COLOR[0] <= v <= COMPONENTS_PER_COLOR[1] for v in counts.values())
    assert TOTAL_COMPONENTS[0] <= sum(counts.values()) <= TOTAL_COMPONENTS[1]
    assert FILL_RANGE[0] <= int(np.count_nonzero(target)) <= FILL_RANGE[1]
    assert metrics["iterative_flip_crossings"] == 2
    assert metrics["d5_inversion_crossings"] == 2
    return inp
