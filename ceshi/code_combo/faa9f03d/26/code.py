# Combo variant of faa9f03d (index 26)
# combo_spec: D1/1+D5/1+D10/2
# Name: 射线截断 + 长度消歧 + 先黄后红

from __future__ import annotations

import random

import numpy as np

from arc_utils import components_mask, get_bg


# ---------------------------------------------------------------------------
# Design analysis for the combo variant
#
# a) Concrete behavior and trace
#    Start from the original task's sparse rectilinear line network with red and
#    yellow singleton markers. The fused rule changes three linked stages:
#    yellow is now a truncation marker, crossing color is chosen by local straight
#    span length, and yellow acts before red-gap filling.
#    Example trace on one color:
#      original task: turn -> pivot, yellow near pivot, old arm behind pivot gets
#      erased, a new ray grows through yellow to the border, then red gaps fill.
#      combo task:    turn -> pivot -> yellow -> visible tail to border in input;
#      step 1 removes the tail beyond yellow, step 2 fills any red gap that only
#      became a facing-endpoint gap after that truncation, step 3 resolves
#      crossings by comparing the repaired local through-segment lengths.
#    The visible difference is chained: the yellow move deletes pixels, that
#    deletion can expose a later red fill, and the final crossing winner is read
#    from the post-truncation geometry rather than pre-repair fragmentation.
#
# b) Complexity calibration
#    To keep the fusion legible, every generated sample contains both yellow and
#    red markers, at least one yellow truncation tail of length >= 2, and at
#    least one crossing whose local-length winner differs from the old fragment
#    winner. The generator also rejects samples where swapping the step order
#    back to red-first reproduces the same output, because that would make D10/2
#    invisible.
#
# c) Hidden-assumption review
#    The original solver assumed yellow produces boundary endpoints and therefore
#    validated every repaired endpoint as boundary-or-crossing. That is invalid
#    here: a yellow tip can be an interior endpoint, so endpoint checks must
#    explicitly whitelist those marker cells.
#    The original fill loop also assumed all fills happen before yellow action.
#    In this combo that order is wrong, because pre-truncation tails can hide the
#    consecutive pair that should later be filled. The fill routine therefore
#    runs on masks after yellow truncation.
#    The original crossing code depended on `comp_count` and a color-id fallback.
#    Those assumptions are replaced by local geometry measured on the repaired
#    masks; ties are rejected during generation so the solver never guesses.
#
# d) Placement legality
#    Yellow candidates are restricted to turns whose marked side is a clean
#    straight tail without crossings. Red markers are chosen on the already
#    truncated target so they do not land on removed tails or on yellow tips.
#    Crossings are preserved when tails are removed, and marker isolation is
#    checked after rendering the input grid.
#
# e) Visual observability
#    A human can see the combo in a few examples because the same cues repeat:
#    yellow sits on an overlong arm and clips it, a nearby red hole becomes the
#    next fill target only after that clip, and contested crossings choose the
#    color with the longer visible straight run through the crossing cell.
#
# f) Visual cleanliness
#    The grid still contains only thin orthogonal lines plus singleton markers.
#    The generator blocks local overlap between yellow neighborhoods, red markers,
#    and crossings so the truncation site, the later fill, and the crossing cue
#    remain spatially readable instead of piling up into one dense patch.
#
# g) Change inferability
#    No invisible ordering by color id is used for the combo-specific crossing
#    decision. The only new cue is geometric length at the crossing. The palette
#    remains small and the generator insists on samples where the combo output
#    differs from both the old fragmentation-based tie-break and the old red-first
#    processing order.
#
# h) New solver assumptions and generator guarantees
#    - Assumption: each yellow marker has a unique retained pivot and a non-empty
#      straight tail beyond the marker. Guarantee: yellow candidates are built
#      from actual turn-plus-tail structures and rechecked in input rendering.
#    - Assumption: local-length crossings have a unique winner. Guarantee: target
#      rendering rejects equal-length ties.
#    - Assumption: order matters on at least one sample mechanism. Guarantee:
#      generation rejects any candidate where a red-first truncation-last solver
#      returns the same output as the combo solver.
# ---------------------------------------------------------------------------


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


def _neighbors(pt: tuple[int, int], h: int, w: int) -> list[tuple[int, int]]:
    r, c = pt
    return [(r + dr, c + dc) for dr, dc in DIRS4 if 0 <= r + dr < h and 0 <= c + dc < w]


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


def _on_boundary(r: int, c: int, h: int, w: int) -> bool:
    return r == 0 or r == h - 1 or c == 0 or c == w - 1


def _components_from_pts(shape: tuple[int, int], pts: set[tuple[int, int]]) -> list[set[tuple[int, int]]]:
    if not pts:
        return []
    mask = np.zeros(shape, dtype=bool)
    for r, c in pts:
        mask[r, c] = True
    return [set(map(tuple, comp)) for comp in components_mask(mask, connectivity=4)]


def _ray_from(pivot: tuple[int, int], direction: tuple[int, int], h: int, w: int) -> set[tuple[int, int]]:
    out = set()
    r, c = pivot
    dr, dc = direction
    while 0 <= r < h and 0 <= c < w:
        out.add((r, c))
        r += dr
        c += dc
    return out


def _straight_run_len(pts: set[tuple[int, int]], pt: tuple[int, int], axis: int) -> int:
    r, c = pt
    length = 1
    if axis == 0:
        cc = c - 1
        while (r, cc) in pts:
            length += 1
            cc -= 1
        cc = c + 1
        while (r, cc) in pts:
            length += 1
            cc += 1
        return length

    rr = r - 1
    while (rr, c) in pts:
        length += 1
        rr -= 1
    rr = r + 1
    while (rr, c) in pts:
        length += 1
        rr += 1
    return length


def _local_span_profile(pts: set[tuple[int, int]], pt: tuple[int, int]) -> tuple[int, int]:
    r, c = pt
    lengths: list[int] = []
    if (r, c - 1) in pts or (r, c + 1) in pts:
        lengths.append(_straight_run_len(pts, pt, axis=0))
    if (r - 1, c) in pts or (r + 1, c) in pts:
        lengths.append(_straight_run_len(pts, pt, axis=1))
    if not lengths:
        return (1, 0)
    lengths.sort(reverse=True)
    if len(lengths) == 1:
        lengths.append(0)
    return (lengths[0], lengths[1])


def _longer_segment_winner(
    masks: dict[int, set[tuple[int, int]]],
    colors_here: list[int],
    pt: tuple[int, int],
    visible_color: int | None = None,
) -> int:
    profiles = {color: _local_span_profile(masks[color], pt) for color in colors_here}
    best_profile = max(profiles.values())
    best = [color for color, profile in profiles.items() if profile == best_profile]
    if len(best) == 1:
        return best[0]
    if visible_color in best:
        return int(visible_color)
    raise AssertionError(f"ambiguous equal-length crossing at {pt}: {profiles}")


def _assign_markers(
    g: np.ndarray,
    line_colors: set[int],
    marker_colors: set[int],
    original: dict[int, set[tuple[int, int]]],
) -> tuple[dict[tuple[int, int], int], dict[tuple[int, int], tuple[int, int]]]:
    h, w = g.shape
    assignment: dict[tuple[int, int], int] = {}
    assign_dir: dict[tuple[int, int], tuple[int, int]] = {}
    marker_mask = np.isin(g, list(marker_colors))
    for r0, c0 in _pts(marker_mask):
        marker_value = int(g[r0, c0])
        candidates: list[tuple[int, ...]] = []
        for dr, dc in DIRS4:
            r, c = r0 + dr, c0 + dc
            dist = 1
            while 0 <= r < h and 0 <= c < w:
                value = int(g[r, c])
                if value in line_colors:
                    axis = 0 if dr == 0 else 1
                    supported = _axis_support(original[value], r, c, axis)
                    if marker_value == 4:
                        corner_like = _perp_support(original[value], r, c, axis)
                        candidates.append(
                            (dist, 0 if corner_like else 1, 0 if supported else 1, value, dr, dc)
                        )
                    else:
                        candidates.append((dist, 0 if supported else 1, value, dr, dc))
                    break
                if value == 0 or value in marker_colors:
                    r += dr
                    c += dc
                    dist += 1
                    continue
                break

        assert candidates, f"marker at {(r0, c0)} cannot see a line color"
        candidates.sort()
        if marker_value == 4:
            _dist, _corner, _unsupported, color, dr, dc = candidates[0]
        else:
            _dist, _unsupported, color, dr, dc = candidates[0]
        assignment[(r0, c0)] = int(color)
        assign_dir[(r0, c0)] = (int(dr), int(dc))
    return assignment, assign_dir


def _fill_gaps(
    masks: dict[int, set[tuple[int, int]]],
    marker_pts: dict[int, set[tuple[int, int]]],
    shape: tuple[int, int],
) -> None:
    h, w = shape

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

    for color in sorted(masks):
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


def _apply_yellow_truncation(
    g: np.ndarray,
    masks: dict[int, set[tuple[int, int]]],
    marker_pts: dict[int, set[tuple[int, int]]],
    assignment: dict[tuple[int, int], int],
    assign_dir: dict[tuple[int, int], tuple[int, int]],
) -> dict[int, set[tuple[int, int]]]:
    h, w = g.shape
    line_colors = set(masks)
    control_tips = {color: set() for color in line_colors}
    if 4 not in set(map(int, g.ravel())):
        return control_tips

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
        assert pivot is not None, f"yellow control at {control} has no retained pivot"

        pr, pc = -dr, -dc
        tail: set[tuple[int, int]] = set()
        r, c = control[0] + pr, control[1] + pc
        while 0 <= r < h and 0 <= c < w and (r, c) in masks[color]:
            tail.add((r, c))
            r += pr
            c += pc

        assert tail, f"yellow control at {control} found no tail to erase"
        for q in tail:
            masks[color].discard(q)
            marker_pts[color].discard(q)
        control_tips[color].add(control)

    return control_tips


def _render_from_masks(
    g: np.ndarray,
    bg: int,
    masks: dict[int, set[tuple[int, int]]],
    control_tips: dict[int, set[tuple[int, int]]],
    conflict_mode: str,
    comp_count: dict[int, int],
) -> np.ndarray:
    claims: dict[tuple[int, int], list[int]] = {}
    for color, pts in masks.items():
        for pt in pts:
            claims.setdefault(pt, []).append(int(color))

    out = np.full_like(g, bg)
    for pt, colors_here in claims.items():
        forced = {color for color in colors_here if pt in control_tips[color]}
        if forced:
            assert len(forced) == 1, f"ambiguous yellow tips at {pt}: {forced}"
            out[pt] = next(iter(forced))
            continue

        if len(colors_here) == 1:
            out[pt] = colors_here[0]
            continue

        if conflict_mode == "length":
            visible_color = int(g[pt]) if int(g[pt]) in colors_here else None
            out[pt] = _longer_segment_winner(masks, colors_here, pt, visible_color)
        else:
            out[pt] = min(colors_here, key=lambda color: (-comp_count[color], color))
    return out


def _validate_output(
    out: np.ndarray,
    line_colors: set[int],
    bg: int,
    control_tips: dict[int, set[tuple[int, int]]],
) -> None:
    h, w = out.shape
    control_tips_flat = set().union(*control_tips.values()) if control_tips else set()
    for color in line_colors:
        for comp in components_mask(out == color, connectivity=4):
            pts_set = set(map(tuple, comp))
            for r, c in pts_set:
                nbrs = [
                    (nr, nc)
                    for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))
                    if (nr, nc) in pts_set
                ]
                if len(nbrs) != 1:
                    continue
                on_boundary = _on_boundary(r, c, h, w)
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
                if (r, c) in control_tips_flat:
                    continue
                adj_other = any(
                    0 <= r + dr < h
                    and 0 <= c + dc < w
                    and int(out[r + dr, c + dc]) not in (bg, color)
                    for dr, dc in DIRS4
                )
                assert adj_other, f"dangling endpoint ({r},{c}) color {color}: not boundary, crossing, or yellow tip"

    color_cells = {c: set() for c in line_colors}
    for r in range(h):
        for c in range(w):
            v = int(out[r, c])
            if v in line_colors:
                color_cells[v].add((r, c))
    crossing_colors = set()
    for r in range(h):
        for c in range(w):
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


def _solve_core(grid: np.ndarray, *, yellow_first: bool, conflict_mode: str) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    bg = int(get_bg(g))

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
    assignment, assign_dir = _assign_markers(g, line_colors, marker_colors, original)

    masks = {color: set(pts) for color, pts in original.items()}
    marker_pts = {color: set() for color in line_colors}
    for pt, color in assignment.items():
        masks[color].add(pt)
        marker_pts[color].add(pt)

    comp_count = {color: _component_count(g.shape, masks[color]) for color in line_colors}
    if yellow_first:
        control_tips = _apply_yellow_truncation(g, masks, marker_pts, assignment, assign_dir)
        _fill_gaps(masks, marker_pts, g.shape)
    else:
        _fill_gaps(masks, marker_pts, g.shape)
        control_tips = _apply_yellow_truncation(g, masks, marker_pts, assignment, assign_dir)

    out = _render_from_masks(g, bg, masks, control_tips, conflict_mode, comp_count)
    _validate_output(out, line_colors, bg, control_tips)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reconstruct a thin rectilinear line network by clipping yellow-marked tails first, then repairing red gaps, and finally settling crossings by local span length.

    Input structure:
    - A rectangular grid with one dominant background color.
    - Non-marker colors form one-cell-thick horizontal and vertical stroke fragments that may logically cross.
    - Marker colors are exactly the colors whose 4-connected components are all singleton pixels.
    - Red singleton markers stand for removed line cells, while yellow singleton markers lie on overlong straight tails attached to turns or endpoints.

    Rule:
    - Assign each marker to the nearest visible line color along a row or column; yellow markers prefer the sightline that looks like the kept corner side rather than the expendable tail.
    - Execute every yellow control first: recolor the yellow cell to its line color conceptually, keep the side leading back into the network, and erase the same-colored straight tail beyond the marker.
    - After those truncations, fill the row or column gaps between facing same-color endpoints, including red-marker gaps that only become exposed after a yellow tail is removed.
    - When multiple repaired colors claim one cell, a yellow tip wins if present; otherwise choose the color whose repaired straight segment through that cell is locally longer.

    Key insight:
    The puzzle is not just "repair markers independently". Yellow changes the geometry before red repair happens. Removing a marked tail can create the exact facing endpoints that reveal where a red gap should be filled, and the final crossing color must then be read from the post-truncation local segment lengths.
    """
    return _solve_core(grid, yellow_first=True, conflict_mode="length")


# --------------- generator ---------------

H = W = 12
BG = 0
LINE_PALETTE = [1, 3, 5, 6, 7, 8, 9]

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

YELLOW_COUNT_RANGE = (1, 1)
RED_COUNT_RANGE = (1, 1)
MIN_CELLS_FOR_RED = 2
MIN_TRUNCATION_LEN = 2

COMPONENTS_PER_COLOR = (1, 4)
TOTAL_COMPONENTS = (4, 6)
FILL_RANGE = (28, 52)

SPACED_VALUES_RETRIES = 60
EXTRA_POLYLINE_RETRIES = 80
GEN_RETRIES = 1600
RED_SELECTION_RETRIES = 48


def _gen_component_count(pts: set[tuple[int, int]]) -> int:
    return _component_count((H, W), pts)


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
        if not _on_boundary(r, c, H, W):
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
        if len(_dirs_in(pts, (r, c))) == 1 and not _on_boundary(r, c, H, W):
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
    if not _on_boundary(*pts[0], H, W):
        r0, c0 = pts[0]
        if ax == 0:
            pts.insert(0, (r0, random.choice([0, W - 1])))
        else:
            pts.insert(0, (random.choice([0, H - 1]), c0))
    if not _on_boundary(*pts[-1], H, W):
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
    if not _on_boundary(*pts[0], H, W):
        pts[0] = _rc(ax, 0, minor)
    if not _on_boundary(*pts[-1], H, W):
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

    if not _on_boundary(*pts[0], H, W):
        r0, c0 = pts[0]
        if ax == 0:
            pts.insert(0, (r0, 0 if c0 == m0 else W - 1))
        else:
            pts.insert(0, (0 if r0 == m0 else H - 1, c0))
    if not _on_boundary(*pts[-1], H, W):
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
        if any(nb in logical[color] for nb in _neighbors(pt, H, W)):
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


def _yellow_truncation_candidates(
    logical: dict[int, set[tuple[int, int]]],
    paths: dict[int, list[set[tuple[int, int]]]],
    crossings: set[tuple[int, int]],
) -> list[tuple[int, tuple[int, int], tuple[int, int], tuple[int, int], set[tuple[int, int]]]]:
    candidates = []
    for color, comps in paths.items():
        for comp in comps:
            for pivot in comp:
                if pivot in crossings or not _is_turn(comp, pivot):
                    continue
                for out_dir in DIRS4:
                    marker = (pivot[0] + out_dir[0], pivot[1] + out_dir[1])
                    if marker not in comp:
                        continue
                    tail: set[tuple[int, int]] = set()
                    r, c = marker[0] + out_dir[0], marker[1] + out_dir[1]
                    while (r, c) in comp:
                        tail.add((r, c))
                        r += out_dir[0]
                        c += out_dir[1]
                    if len(tail) < MIN_TRUNCATION_LEN:
                        continue
                    if marker in crossings or any(cell in crossings for cell in tail):
                        continue
                    candidates.append((color, pivot, out_dir, marker, tail))
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
        if any(nb in used for nb in [pt] + _neighbors(pt, H, W)):
            continue
        chosen.append((color, pt))
        used.add(pt)
        per_color_left[color] -= 1
    return chosen if len(chosen) == count else []


def _red_marker_candidates(
    logical: dict[int, set[tuple[int, int]]],
    paths: dict[int, list[set[tuple[int, int]]]],
    blocked: set[tuple[int, int]],
) -> list[tuple[int, tuple[int, int]]]:
    candidates = [
        (color, pt)
        for color, comps in paths.items()
        for comp in comps
        for pt in comp
        if pt not in blocked and _is_turn_or_endpoint(comp, pt)
    ]
    random.shuffle(candidates)
    out: list[tuple[int, tuple[int, int]]] = []
    per_color_left = {color: len(logical[color]) for color in logical}
    for color, pt in candidates:
        if per_color_left[color] <= MIN_CELLS_FOR_RED:
            continue
        if any(nb in blocked for nb in [pt] + _neighbors(pt, H, W)):
            continue
        out.append((color, pt))
    return out


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
    yellow_tips: dict[int, set[tuple[int, int]]],
) -> np.ndarray | None:
    out = np.full((H, W), BG, dtype=int)
    all_pts = set().union(*logical.values()) if logical else set()
    for pt in all_pts:
        colors_here = [color for color in logical if pt in logical[color]]
        forced = [color for color in colors_here if pt in yellow_tips.get(color, set())]
        if forced:
            if len(forced) != 1:
                return None
            out[pt] = forced[0]
        elif len(colors_here) == 1:
            out[pt] = colors_here[0]
        else:
            try:
                out[pt] = _longer_segment_winner(logical, colors_here, pt)
            except AssertionError:
                return None
    return out


def _marker_isolation_ok(grid: np.ndarray) -> bool:
    for color in (2, 4):
        for pt in map(tuple, np.argwhere(grid == color)):
            if any(grid[nb] == color for nb in _neighbors(pt, H, W)):
                return False
    return True


def _valid_target_structure(
    logical: dict[int, set[tuple[int, int]]],
    yellow_markers: set[tuple[int, int]],
) -> bool:
    for color, pts in logical.items():
        for comp in _components_from_pts((H, W), pts):
            for r, c in comp:
                nbrs = _dirs_in(comp, (r, c))
                if len(nbrs) > 2:
                    return False
                if len(nbrs) != 1:
                    continue
                if _on_boundary(r, c, H, W):
                    continue
                if (r, c) in yellow_markers:
                    continue
                adj_other = any(
                    nb in other
                    for other_color, other in logical.items()
                    if other_color != color
                    for nb in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))
                )
                if not adj_other:
                    return False
    return True


def _derive_input(
    logical: dict[int, set[tuple[int, int]]],
    paths: dict[int, list[set[tuple[int, int]]]],
    dominant: int,
) -> tuple[np.ndarray, np.ndarray] | None:
    del dominant
    base_crossings = set(_crossing_cells(logical))
    if YELLOW_COUNT_RANGE != (1, 1):
        return None

    yellow_options: list[tuple[list[tuple[int, tuple[int, int], tuple[int, int], tuple[int, int], set[tuple[int, int]]]], set[tuple[int, int]]]] = []
    blocked0 = set(base_crossings)
    for cand in _yellow_truncation_candidates(logical, paths, base_crossings):
        color, pivot, out_dir, marker, tail = cand
        del color, out_dir
        local_cells = {pivot, marker} | tail
        if local_cells & blocked0:
            continue
        if any(nb in blocked0 for nb in _neighbors(marker, H, W) + _neighbors(pivot, H, W)):
            continue
        yellow_options.append((
            [cand],
            blocked0 | local_cells | set(_neighbors(marker, H, W)) | set(_neighbors(pivot, H, W)),
        ))
    if not yellow_options:
        return None

    random.shuffle(yellow_options)
    for yellow_infos, blocked in yellow_options:
        target_logical = {c: set(pts) for c, pts in logical.items()}
        removed_tails: dict[int, set[tuple[int, int]]] = {c: set() for c in logical}
        yellow_by_pt: dict[tuple[int, int], int] = {}
        yellow_tips: dict[int, set[tuple[int, int]]] = {c: set() for c in logical}
        for color, pivot, out_dir, marker, tail in yellow_infos:
            del pivot, out_dir
            target_logical[color].difference_update(tail)
            removed_tails[color].update(tail)
            yellow_by_pt[marker] = color
            yellow_tips[color].add(marker)

        target_crossings = set(_crossing_cells(target_logical))
        if target_crossings != base_crossings:
            continue
        if not _valid_target_structure(target_logical, set(yellow_by_pt)):
            continue

        target_paths = {c: _components_from_pts((H, W), target_logical[c]) for c in target_logical}

        input_logical = {c: set(pts) for c, pts in target_logical.items()}
        for color, pts in removed_tails.items():
            input_logical[color].update(pts)

        base_target = _render_target(target_logical, yellow_tips)
        if base_target is None:
            continue

        crossing_display_base: dict[tuple[int, int], int] = {}
        crossing_winner: dict[tuple[int, int], int] = {}
        valid_crossing_display = True
        for pt in target_crossings:
            colors_here = [color for color in target_logical if pt in target_logical[color]]
            if len(colors_here) != 2:
                valid_crossing_display = False
                break
            winner = int(base_target[pt])
            losers = [color for color in colors_here if color != winner]
            if len(losers) != 1:
                valid_crossing_display = False
                break
            crossing_display_base[pt] = losers[0]
            crossing_winner[pt] = winner
        if not valid_crossing_display:
            continue

        marker_info_by_pt = {info[3]: info for info in yellow_infos}
        truncated_cells = sum(len(tail) for tail in removed_tails.values())
        if truncated_cells < MIN_TRUNCATION_LEN:
            continue

        red_blocked = blocked | set(yellow_by_pt)
        red_candidates = _red_marker_candidates(target_logical, target_paths, red_blocked)
        if not red_candidates:
            continue
        crossing_variants = [None] + random.sample(list(target_crossings), len(target_crossings))
        tried = 0
        for divergence_pt in crossing_variants:
            crossing_display = dict(crossing_display_base)
            if divergence_pt is not None:
                crossing_display[divergence_pt] = crossing_winner[divergence_pt]
            for red_marker in red_candidates:
                tried += 1
                if tried > RED_SELECTION_RETRIES:
                    break
                red_markers = [red_marker]

                inp = np.full((H, W), BG, dtype=int)
                comp_count_masks: dict[int, set[tuple[int, int]]] = {c: set() for c in logical}
                red_by_pt = {pt: color for color, pt in red_markers}

                for color, pts in input_logical.items():
                    for pt in pts:
                        if pt in target_crossings and crossing_display[pt] != color:
                            continue
                        if pt in red_by_pt or pt in yellow_by_pt:
                            continue
                        inp[pt] = color
                        comp_count_masks[color].add(pt)

                for pt, color in red_by_pt.items():
                    inp[pt] = 2
                    comp_count_masks[color].add(pt)
                for pt, color in yellow_by_pt.items():
                    inp[pt] = 4
                    comp_count_masks[color].add(pt)

                if not _marker_isolation_ok(inp):
                    continue
                bad_marker = False
                for pt, color in yellow_by_pt.items():
                    if not any(inp[nb] == color for nb in _neighbors(pt, H, W)):
                        bad_marker = True
                        break
                    _, pivot, out_dir, _marker, _tail = marker_info_by_pt[pt]
                    if (pivot[0] + out_dir[0], pivot[1] + out_dir[1]) != pt:
                        bad_marker = True
                        break
                    beyond = (pt[0] + out_dir[0], pt[1] + out_dir[1])
                    if beyond not in input_logical[color]:
                        bad_marker = True
                        break
                if bad_marker:
                    continue

                try:
                    target = main(inp)
                except Exception:
                    continue

                try:
                    wrong_order = _solve_core(inp, yellow_first=False, conflict_mode="length")
                except Exception:
                    wrong_order = None
                if wrong_order is not None and np.array_equal(wrong_order, target):
                    continue

                try:
                    frag_order = _solve_core(inp, yellow_first=True, conflict_mode="fragment")
                except Exception:
                    frag_order = None
                if frag_order is not None and np.array_equal(frag_order, target):
                    continue
                if wrong_order is None and frag_order is None:
                    continue

                if not np.any(inp == 2) or not np.any(inp == 4):
                    continue

                comp_count = {color: _gen_component_count(pts) for color, pts in comp_count_masks.items()}
                length_diff_crossings = 0
                for pt in target_crossings:
                    colors_here = [color for color in target_logical if pt in target_logical[color]]
                    frag_winner = min(colors_here, key=lambda color: (-comp_count[color], color))
                    if frag_winner != int(target[pt]):
                        length_diff_crossings += 1
                if length_diff_crossings == 0:
                    continue

                order_diff_cells = None if wrong_order is None else int(np.count_nonzero(wrong_order != target))
                if order_diff_cells == 0:
                    continue

                counts = {c: len(components_mask(target == c, connectivity=4)) for c in sorted(logical)}
                if not all(COMPONENTS_PER_COLOR[0] <= v <= COMPONENTS_PER_COLOR[1] for v in counts.values()):
                    continue
                if not TOTAL_COMPONENTS[0] <= sum(counts.values()) <= TOTAL_COMPONENTS[1]:
                    continue
                if not FILL_RANGE[0] <= int(np.count_nonzero(target)) <= FILL_RANGE[1]:
                    continue

                return inp, target
            if tried > RED_SELECTION_RETRIES:
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
    raise RuntimeError("failed to generate a valid faa9f03d combo input")
