# Combo variant of faa9f03d (index 25)
# combo_spec: D2/1+D6/1+D9/0
# Name: 内部悬挂端点 + 全部填充 + 纯直线无转角

from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np

_ARC_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_ARC_UTILS_DIR) not in sys.path:
    sys.path.append(str(_ARC_UTILS_DIR))

from arc_utils import components_mask, get_bg


# Design analysis for the combo variant D2/1 + D6/1 + D9/0
# a) Concrete behavior and trace:
#    - The fusion removes all turns: every logical object is now a single horizontal or vertical
#      segment, so the task becomes a set of one-dimensional repairs rather than polyline routing.
#    - D2/1 removes the old "must hit the border" clue. A repaired segment may stop strictly
#      inside the grid, so a dangling interior endpoint is no longer evidence of an error.
#    - D6/1 changes the fill trigger from "supported facing endpoints only" to "every pair of
#      same-color pixels that share a row or column". Example row:
#        output: [0, a, a, a, a, a, a, a, a, a, a, 0]
#        input : [0, a, a, 0, 2, 0, 0, 4, 0, a, a, 0]
#      First recolor the singleton markers 2/4 to line color `a`. Then the variant fills all
#      gaps between consecutive same-color cells in that row, producing the full 10-cell segment.
#      The original solver would not fill the central gap between the two markers because both
#      endpoints of that sub-gap are markers and the gap length is > 1.
#    - To keep the variant visually learnable, the generator also places one repeated color on two
#      separate short segments in different rows/columns but with disjoint spans, so the solver
#      sees "same color can appear multiple times" without accidentally creating a rectangle.
#
# b) Complexity calibration:
#    - The task is simpler than the base problem, but not degenerate: there are still multiple
#      colors, both marker hues appear, and at least one sample color occurs on multiple distinct
#      straight segments.
#    - The aggressive D6/1 rule is forced to matter because every generated instance contains
#      long gaps whose only anchors are two singleton markers separated by background.
#    - Freedom is intentionally limited: no crossings, no turns, no edge-running strokes. That
#      keeps the visual hypothesis space small enough for ARC while still making the new behavior
#      obvious from 2-4 examples.
#
# c) Implicit-assumption audit of the original solver:
#    - Original assumption: visible endpoints should end on the boundary or at a crossing.
#      Invalid here; internal endpoints are expected, so all boundary/perpendicularity asserts
#      must be removed.
#    - Original assumption: yellow markers carry a direction-specific "erase old arm and extend to
#      edge" meaning. That meaning depends on turns/boundary arms; under D9/0 + D2/1 it becomes
#      largely invisible and would reintroduce hidden state. This combo therefore treats both
#      singleton marker hues as the same kind of missing line cell: assign a line color, then fill.
#    - Original assumption: whether a gap should be filled depends on axis support and on whether
#      both ends are markers. D6/1 explicitly invalidates all of those checks; once two same-color
#      pixels are collinear, their in-between cells belong to the repaired segment.
#    - New solver assumption: markers have an unambiguous nearest visible line color in row/column
#      line of sight. The generator guarantees this by using only parallel line families and by
#      keeping marker columns/rows away from foreign colored stubs.
#
# d) Placement legality:
#    - Generation is output-first. The final straight segments are placed first, then specific
#      cells are hidden or turned into singleton markers.
#    - Same-color repeated segments are given disjoint spans on the fill axis, preventing the
#      D6/1 rule from joining them through a shared column/row.
#    - Parallel rows/columns are spaced by at least two cells so markers of the same hue remain
#      isolated 1x1 components and no accidental 4-neighbor contacts appear.
#
# e) Visual observability:
#    - D2/1 is visible because the two long repaired segments always end strictly inside the grid.
#    - D6/1 is visible because the longest repaired region is the marker-to-marker interior, which
#      would stay blank under the old support-gated rule.
#    - D9/0 is visible because every color forms only straight pieces; there are no corners to
#      explain away, so a human can focus on the "same color + same axis => fill everything"
#      relationship.
#
# f) Visual cleanliness:
#    - No two colors overlap or cross. The picture is a small set of separated stripes, which
#      keeps marker ownership legible.
#    - The repeated-color short segments use disjoint spans, so the output does not become a noisy
#      block or ladder.
#
# g) New solver assumptions and generator guarantees:
#    - Solver assumption: each foreground singleton marker can see at least one true line color in
#      a cardinal direction. Generator guarantee: every marker is embedded inside its target line's
#      span and no foreign line color is closer on another axis.
#    - Solver assumption: line colors are not themselves all-singleton. Generator guarantee: every
#      line color has at least one connected component of length >= 2 in the input.
#    - Solver assumption: one fill pass using recolored markers as anchors is enough. Generator
#      guarantee: each color family is restricted to one orientation and is arranged so that no new
#      cross-axis fill opportunities need to be discovered recursively.


H = W = 12
BG = 0
MARKER_COLORS = (2, 4)
LINE_PALETTE = [1, 3, 5, 6, 7, 8, 9]
DIRS4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
GEN_RETRIES = 80


def _pts(mask: np.ndarray) -> set[tuple[int, int]]:
    return {tuple(map(int, rc)) for rc in np.argwhere(mask)}


def _component_count(shape: tuple[int, int], pts: set[tuple[int, int]]) -> int:
    mask = np.zeros(shape, dtype=bool)
    for r, c in pts:
        mask[r, c] = True
    return len(components_mask(mask, connectivity=4))


def _line_cells(orient: str, fixed: int, start: int, end: int) -> list[tuple[int, int]]:
    if orient == "h":
        return [(fixed, c) for c in range(start, end + 1)]
    return [(r, fixed) for r in range(start, end + 1)]


def _marker_colors_in_grid(g: np.ndarray, bg: int) -> set[int]:
    colors = set(map(int, g.ravel())) - {int(bg)}
    marker_colors: set[int] = set()
    for color in colors:
        comps = components_mask(g == int(color), connectivity=4)
        if comps and max(len(comp) for comp in comps) == 1:
            marker_colors.add(int(color))
    return marker_colors


def _assign_markers(
    g: np.ndarray,
    line_colors: set[int],
    marker_colors: set[int],
) -> tuple[dict[tuple[int, int], int], dict[tuple[int, int], tuple[int, int]]]:
    H0, W0 = g.shape
    original = {color: _pts(g == color) for color in line_colors}
    assignment: dict[tuple[int, int], int] = {}
    assign_dir: dict[tuple[int, int], tuple[int, int]] = {}
    marker_mask = np.isin(g, list(marker_colors))

    for r0, c0 in sorted(_pts(marker_mask)):
        candidates: list[tuple[int, int, int, int, int]] = []
        for dr, dc in DIRS4:
            r, c = r0 + dr, c0 + dc
            dist = 1
            while 0 <= r < H0 and 0 <= c < W0:
                value = int(g[r, c])
                if value in line_colors:
                    candidates.append((dist, value, dr, dc, len(original[value])))
                    break
                if value == int(get_bg(g)) or value in marker_colors:
                    r += dr
                    c += dc
                    dist += 1
                    continue
                break
        assert candidates, f"marker at {(r0, c0)} cannot see a line color"
        candidates.sort(key=lambda item: (item[0], item[1], item[4]))
        _dist, color, dr, dc, _count = candidates[0]
        assignment[(r0, c0)] = int(color)
        assign_dir[(r0, c0)] = (int(dr), int(dc))
    return assignment, assign_dir


def main(grid: np.ndarray) -> np.ndarray:
    """
    Repair a one-cell-thick straight-line scene by recoloring singleton markers and fully
    filling every same-color gap on the marker's axis family.

    Input structure:
    - The grid has a dominant background plus several non-marker line colors.
    - Every line object is a pure horizontal or vertical straight segment; there are no turns.
    - Marker colors are exactly those foreground colors whose connected components are all
      singleton pixels, typically red and/or yellow.

    Rule:
    - For each singleton marker, look outward in the four cardinal directions and assign the
      marker to the nearest visible line color in line-of-sight.
    - After recoloring the markers, for each line color fill every gap between consecutive
      same-color pixels that lie on the same row or on the same column.
    - If two colors ever claim the same cell, prefer the color already visible there; otherwise
      prefer the more fragmented pre-fill color, breaking ties by smaller color id.

    Key insight:
    - Because all geometry degenerates to straight segments, the old endpoint-support and
      boundary-direction logic is noise. The real signal is simply that singleton markers belong
      to nearby line colors, and once those colors are known, every collinear same-color break
      should be closed completely even when both anchors are markers.
    """
    g = np.asarray(grid, dtype=int)
    bg = int(get_bg(g))
    colors = set(map(int, g.ravel())) - {bg}
    assert colors, "expected non-background pixels"

    marker_colors = _marker_colors_in_grid(g, bg)
    line_colors = colors - marker_colors
    assert line_colors, "expected at least one non-marker line color"
    assert marker_colors, "expected singleton marker colors"

    assignment, _assign_dir = _assign_markers(g, line_colors, marker_colors)

    masks = {color: _pts(g == color) for color in line_colors}
    marker_pts = {color: set() for color in line_colors}
    for pt, color in assignment.items():
        masks[color].add(pt)
        marker_pts[color].add(pt)

    comp_count = {color: _component_count(g.shape, masks[color]) for color in line_colors}

    for color in sorted(line_colors):
        additions: set[tuple[int, int]] = set()
        for r in range(g.shape[0]):
            cols = sorted(c for rr, c in masks[color] if rr == r)
            for c1, c2 in zip(cols, cols[1:]):
                additions.update((r, c) for c in range(c1 + 1, c2))
        for c in range(g.shape[1]):
            rows = sorted(r for r, cc in masks[color] if cc == c)
            for r1, r2 in zip(rows, rows[1:]):
                additions.update((r, c) for r in range(r1 + 1, r2))
        masks[color].update(additions)

    claims: dict[tuple[int, int], list[int]] = {}
    for color, pts in masks.items():
        for pt in pts:
            claims.setdefault(pt, []).append(int(color))

    out = np.full_like(g, bg)
    for pt, colors_here in claims.items():
        if len(colors_here) == 1:
            out[pt] = colors_here[0]
            continue
        visible = [color for color in colors_here if int(g[pt]) == color]
        if visible:
            out[pt] = min(visible, key=lambda color: (-comp_count[color], color))
        else:
            out[pt] = min(colors_here, key=lambda color: (-comp_count[color], color))

    return out


# --------------- generator (combo D2/1 + D6/1 + D9/0) ---------------


def _paint_cells(grid: np.ndarray, cells: list[tuple[int, int]], color: int) -> None:
    for r, c in cells:
        grid[r, c] = int(color)


def _make_specs(orient: str) -> tuple[list[dict], np.ndarray]:
    colors = random.sample(LINE_PALETTE, 3)
    repeated = colors[0]
    long_a = colors[1]
    long_b = colors[2]

    specs: list[dict]
    if orient == "h":
        specs = [
            {"color": long_a, "orient": "h", "fixed": 1, "start": 1, "end": 10, "kind": "double"},
            {"color": long_b, "orient": "h", "fixed": 4, "start": 1, "end": 10, "kind": "double"},
            {"color": repeated, "orient": "h", "fixed": 7, "start": 0, "end": 3, "kind": "short_left"},
            {"color": repeated, "orient": "h", "fixed": 10, "start": 8, "end": 11, "kind": "short_right"},
        ]
    else:
        specs = [
            {"color": long_a, "orient": "v", "fixed": 1, "start": 1, "end": 10, "kind": "double"},
            {"color": long_b, "orient": "v", "fixed": 4, "start": 1, "end": 10, "kind": "double"},
            {"color": repeated, "orient": "v", "fixed": 7, "start": 0, "end": 3, "kind": "short_left"},
            {"color": repeated, "orient": "v", "fixed": 10, "start": 8, "end": 11, "kind": "short_right"},
        ]

    target = np.full((H, W), BG, dtype=int)
    for spec in specs:
        _paint_cells(target, _line_cells(spec["orient"], spec["fixed"], spec["start"], spec["end"]), spec["color"])
    return specs, target


def _apply_pattern(
    inp: np.ndarray,
    spec: dict,
    marker_order: tuple[int, int],
) -> None:
    color = int(spec["color"])
    cells = _line_cells(spec["orient"], spec["fixed"], spec["start"], spec["end"])
    if spec["kind"] == "double":
        rel = {
            0: color,
            1: color,
            3: marker_order[0],
            6: marker_order[1],
            8: color,
            9: color,
        }
    elif spec["kind"] == "short_left":
        rel = {
            0: color,
            1: color,
            2: marker_order[0],
            3: color,
        }
    else:
        rel = {
            0: color,
            1: marker_order[1],
            2: color,
            3: color,
        }
    for idx, (r, c) in enumerate(cells):
        if idx in rel:
            inp[r, c] = int(rel[idx])


def _marker_isolation_ok(grid: np.ndarray) -> bool:
    for color in MARKER_COLORS:
        for r, c in map(tuple, np.argwhere(grid == color)):
            for dr, dc in DIRS4:
                rr, cc = r + dr, c + dc
                if 0 <= rr < H and 0 <= cc < W and int(grid[rr, cc]) == int(color):
                    return False
    return True


def _has_internal_endpoint(target: np.ndarray, color: int) -> bool:
    for comp in components_mask(target == int(color), connectivity=4):
        pts = {tuple(map(int, rc)) for rc in comp}
        for r, c in pts:
            nbrs = sum(((r + dr, c + dc) in pts) for dr, dc in DIRS4)
            if nbrs == 1 and 0 < r < H - 1 and 0 < c < W - 1:
                return True
    return False


def _derive_input(specs: list[dict], target: np.ndarray) -> np.ndarray | None:
    inp = np.full((H, W), BG, dtype=int)
    orders = [(2, 4), (4, 2)]
    random.shuffle(orders)

    for i, spec in enumerate(specs):
        order = orders[i % len(orders)]
        _apply_pattern(inp, spec, order)

    if not _marker_isolation_ok(inp):
        return None

    marker_colors = _marker_colors_in_grid(inp, BG)
    if set(MARKER_COLORS) - marker_colors:
        return None

    if not any(_has_internal_endpoint(target, color) for color in sorted(set(spec["color"] for spec in specs))):
        return None

    solved = main(inp)
    return inp if np.array_equal(solved, target) else None


def _gen_attempt() -> np.ndarray | None:
    orient = random.choice(["h", "v"])
    specs, target = _make_specs(orient)
    return _derive_input(specs, target)


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("failed to generate a valid faa9f03d combo D2/1+D6/1+D9/0 input")
