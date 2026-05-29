# Combo variant of a395ee82 (index 13)
# combo_spec: D1/4+D14/0
# Name: 连通性匹配象限旋转

from __future__ import annotations

import math
import random

import numpy as np

from arc_utils import bbox_h, bbox_w, find_components, get_bg

# Design analysis for the combo variant
# -------------------------------------
# a) Concrete behavior / trace:
# - The base task had one solid template in color T and a detached singleton lattice
#   whose unique T-colored point marked the origin. Every decoded offset reused the
#   same template orientation.
# - D1/4 removes that color cue: the source marker becomes a monochrome O-colored
#   thin graph, and the origin must be recovered from connectivity. In this combo the
#   graph is interpreted as one unique highest-degree center plus several leaf nodes.
# - D14/0 simultaneously changes the stamping rule: after a leaf offset is decoded,
#   its template copy is rotated by quadrant around the recovered center
#   (upper-right 0, upper-left 90, lower-left 180, lower-right 270).
# - Concrete example: if the graph center is at fine-grid position (10, 10), then a
#   leaf at (8, 12) decodes to coarse delta (-1, +1) and keeps the template upright;
#   a leaf at (8, 8) decodes to (-1, -1) and receives the 90-degree turn. The
#   original solver would have looked for a T-colored singleton and would have reused
#   one fixed orientation for both leaves, so both the origin-detection step and the
#   transform step change together.
#
# b) Complexity calibration:
# - The graph always shows all four diagonal quadrants via four base leaves, so every
#   sample contains all four rotation cases. An optional fifth leaf repeats one
#   quadrant to provide redundancy without adding many degrees of freedom.
# - The template bbox is fixed to 4x4 and required to have four distinct quarter-turns,
#   making the quadrant rule visually obvious.
# - Only a few graph topologies are generated: a 4-arm center plus at most one extra
#   branch. This keeps the connectivity inference human-solvable.
#
# c) Implicit-assumption audit against the original main():
# - Original `template = max(comps, key=area)` fails because the connector graph can
#   have more pixels than the template. This combo instead identifies the template as
#   the smaller-bbox monochrome component; generator guarantees template bbox = 4x4
#   while graph bbox is at least 5x5.
# - Original origin recovery by `color == templ_col` is invalid because the graph has
#   no T-colored marker. We replace it with graph-degree analysis.
# - The D1 single-variant trick "even BFS distance = node" conflicts with D14 because
#   quadrant rotation forbids axis-aligned stamp positions. This combo therefore uses
#   `origin + graph leaves` as the actual stamp nodes and treats degree-2 path pixels
#   as connectors only.
# - Original stamping reused one `templ_rel`; D14/0 breaks that, so each decoded delta
#   now selects a rotated copy before translation.
#
# d) Placement legality:
# - Output copies live on a 4x4 anchor lattice, so every quarter-turn stays inside its
#   own box. Generator chooses the base template anchor from decoded delta bounds, so
#   all copies remain inside 22x22.
# - The input graph and the visible template bbox are separated by at least one bg
#   cell. Graph routing uses only orthogonal 1-pixel-wide paths, and all graph pixels
#   stay inside the board.
# - Extra leaves are chosen from a safe subset whose routed paths may branch but do
#   not erase the four base quadrant leaves.
#
# e) Visual observability:
# - Humans can see that the source side is no longer isolated singletons: it is a thin
#   tree with one obvious branch center and several leaf endpoints.
# - The output shows four different rotated copies arranged by diagonal quadrant, with
#   repeated evidence when the optional fifth leaf is present.
#
# f) Visual cleanliness:
# - The input still has exactly two foreground objects: one compact template and one
#   detached thin graph. No extra color legend or hidden ordering is introduced.
# - The generator avoids output overlap, so rotated copies remain readable rather than
#   being hidden behind O-colored collision regions.
#
# g) Rule inferability / anti-degeneration:
# - Colors remain only bg / T / O. O consistently means "graph color" in the input and
#   "origin-copy / overlap color" in the output.
# - Every sample includes all four diagonal quadrants, so the rotation map is directly
#   evidenced instead of being a one-off coincidence.
# - The graph contains zero T-colored source markers, which blocks regression to the
#   original color-match rule.
#
# h) Quantitative fusion metric:
# - Metric 1: the graph must contain a unique highest-degree pixel (degree 4) and
#   4-5 leaves; this confirms connectivity matching is active.
# - Metric 2: among the decoded non-origin deltas, all four quadrant turn values
#   {0,1,2,3} must appear at least once; this confirms quadrant-conditioned rotation
#   is active and not collapsing to one orientation.
#
# i) New solver assumptions and generator guarantees:
# - main() assumes there are exactly two nontrivial monochrome components: a 4x4
#   template and one thin O-colored graph. generate_input() constructs exactly that.
# - main() assumes the graph origin is the unique maximum-degree pixel and every stamp
#   target besides the origin is a leaf. The routed graph family guarantees both.
# - main() assumes all leaves lie in strict diagonal quadrants and their coordinate
#   differences share a common gcd equal to the fine-grid spacing. The generator uses
#   leaf offsets built from a fixed coarse lattice with GUNIT = 2.


GRID_SIZE = 22
BOX_SIZE = 4
TEMPLATE_AREA_RANGE = (7, 10)
GUNIT = 2

GENERATE_RETRIES = 500
GRAPH_PLACE_RETRIES = 100
SPANNING_RETRIES = 80
TEMPLATE_RETRIES = 160

BASE_LEAVES = [(-1, -1), (-1, 1), (1, -1), (1, 1)]
SAFE_EXTRA_LEAVES = [(-2, 1), (-1, -2), (2, -1), (1, 2)]


def _bbox_area(comp: dict) -> int:
    r0, c0, r1, c1 = map(int, comp["bbox"])
    return int(bbox_h((r0, c0, r1, c1)) * bbox_w((r0, c0, r1, c1)))


def _rotate_cells(cells: list[tuple[int, int]], size: int, turns: int) -> list[tuple[int, int]]:
    """Rotate bbox-relative cells by 90-degree CCW steps inside a size x size box."""
    turns %= 4
    out = [(int(r), int(c)) for r, c in cells]
    for _ in range(turns):
        out = [(int(c), int(size - 1 - r)) for r, c in out]
    return out


def _quadrant_turns(dr: int, dc: int) -> int:
    """
    Map a center-relative coarse delta to the quadrant rotation.

    Upper-right -> 0, upper-left -> 1, lower-left -> 2, lower-right -> 3.
    The center keeps the canonical orientation.
    """
    dr = int(dr)
    dc = int(dc)
    if dr == 0 and dc == 0:
        return 0
    assert dr != 0 and dc != 0
    if dr < 0 and dc > 0:
        return 0
    if dr < 0 and dc < 0:
        return 1
    if dr > 0 and dc < 0:
        return 2
    return 3


def _graph_degrees(coords: set[tuple[int, int]]) -> dict[tuple[int, int], int]:
    deg: dict[tuple[int, int], int] = {}
    for r, c in coords:
        cnt = 0
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            if (int(r + dr), int(c + dc)) in coords:
                cnt += 1
        deg[(int(r), int(c))] = int(cnt)
    return deg


def _extract_origin_and_leaves(graph_comp: dict) -> tuple[tuple[int, int], list[tuple[int, int]]]:
    graph_coords = {(int(r), int(c)) for r, c in graph_comp["coords"]}
    assert graph_coords

    deg = _graph_degrees(graph_coords)
    max_deg = max(int(v) for v in deg.values())
    origin_candidates = [pt for pt, d in deg.items() if int(d) == int(max_deg)]
    assert max_deg == 4
    assert len(origin_candidates) == 1
    origin = origin_candidates[0]

    leaves = [pt for pt, d in deg.items() if int(d) == 1]
    assert 4 <= len(leaves) <= 5

    orr, orc = map(int, origin)
    for rr, cc in leaves:
        assert int(rr - orr) != 0 and int(cc - orc) != 0
    return origin, sorted((int(r), int(c)) for r, c in leaves)


def _render_output(
    H: int,
    W: int,
    bg: int,
    templ_col: int,
    overlap_col: int,
    base_r: int,
    base_c: int,
    size: int,
    templ_cells: list[tuple[int, int]],
    deltas: list[tuple[int, int]],
) -> np.ndarray:
    """Render the output implied by the decoded coarse deltas."""
    total = np.zeros((H, W), dtype=int)
    center = np.zeros((H, W), dtype=int)
    for dr, dc in deltas:
        rot_cells = _rotate_cells(templ_cells, size, _quadrant_turns(dr, dc))
        is_origin = int(dr) == 0 and int(dc) == 0
        off_r = int(base_r + dr * size)
        off_c = int(base_c + dc * size)
        for rr, cc in rot_cells:
            R = int(off_r + rr)
            C = int(off_c + cc)
            if 0 <= R < H and 0 <= C < W:
                total[R, C] += 1
                if is_origin:
                    center[R, C] += 1

    out = np.full((H, W), int(bg), dtype=int)
    out[total > 0] = int(templ_col)
    out[(total == 1) & (center > 0)] = int(overlap_col)
    out[total >= 2] = int(overlap_col)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Stamp quadrant-rotated copies of a monochrome template using a connectivity-coded graph.

    Input structure:
    - bg is the dominant background color.
    - One compact monochrome 4x4 component is the template, shown in its canonical orientation with color T.
    - A detached monochrome O-colored thin graph marks copy locations: one center node branches to several leaf nodes through 1-pixel orthogonal paths.

    Rule:
    - Identify the graph center as the unique highest-degree pixel, and treat only the graph leaves as non-origin stamp targets.
    - Convert each leaf offset from the center into coarse lattice steps by dividing by the shared spacing unit.
    - Stamp one template copy per decoded offset, rotating by quadrant around the center:
      upper-right 0 degrees, upper-left 90 degrees, lower-left 180 degrees, lower-right 270 degrees.
    - The center's own copy keeps the canonical orientation but is recolored to O; any overlapped pixels are also recolored to O.

    Key insight: The special source location is no longer the uniquely colored marker from the original task. You must read the topology of the thin graph to find the center, then use each leaf's diagonal quadrant to choose the correct quarter-turn of the template.
    """
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))
    H, W = g.shape

    comps = find_components(g, bg=bg, connectivity=4, monochromatic=True, return_format="dict")
    big = [comp for comp in comps if int(comp["area"]) > 1]
    assert len(big) == 2

    template = min(big, key=_bbox_area)
    graph_comp = big[0] if big[1] is template else big[1]

    templ_coords = template["coords"]
    templ_col = int(template["color"])
    overlap_col = int(graph_comp["color"])
    r0, c0, r1, c1 = map(int, template["bbox"])
    h = int(bbox_h((r0, c0, r1, c1)))
    w = int(bbox_w((r0, c0, r1, c1)))
    assert h == w == BOX_SIZE
    templ_rel = [(int(r) - int(r0), int(c) - int(c0)) for r, c in templ_coords]

    origin, leaves = _extract_origin_and_leaves(graph_comp)
    orr, orc = map(int, origin)

    leaf_offsets = [(int(r - orr), int(c - orc)) for r, c in leaves]
    gcd_vals = [abs(int(x)) for dr, dc in leaf_offsets for x in (dr, dc) if int(x) != 0]
    assert gcd_vals
    gunit = max(1, int(math.gcd(*gcd_vals)))

    deltas = [(0, 0)] + [(int(dr // gunit), int(dc // gunit)) for dr, dc in leaf_offsets]
    for dr, dc in deltas[1:]:
        assert dr != 0 and dc != 0

    total = np.zeros((H, W), dtype=int)
    center = np.zeros((H, W), dtype=int)

    for dr, dc in deltas:
        rot_cells = _rotate_cells(templ_rel, BOX_SIZE, _quadrant_turns(dr, dc))
        off_r = int(r0 + dr * BOX_SIZE)
        off_c = int(c0 + dc * BOX_SIZE)
        is_origin = int(dr) == 0 and int(dc) == 0
        for rr, cc in rot_cells:
            R = int(off_r + rr)
            C = int(off_c + cc)
            if 0 <= R < H and 0 <= C < W:
                total[R, C] += 1
                if is_origin:
                    center[R, C] += 1

    out = np.full((H, W), int(bg), dtype=int)
    out[total > 0] = int(templ_col)
    out[(total == 1) & (center > 0)] = int(overlap_col)
    out[total >= 2] = int(overlap_col)
    return out


def _random_polyomino(n: int, rows: int, cols: int) -> list[tuple[int, int]] | None:
    """Grow a random 4-connected polyomino of n cells inside a rows x cols box."""
    start = (random.randint(0, rows - 1), random.randint(0, cols - 1))
    cells = {start}
    frontier = []
    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        nr, nc = start[0] + dr, start[1] + dc
        if 0 <= nr < rows and 0 <= nc < cols:
            frontier.append((nr, nc))

    while len(cells) < n and frontier:
        cell = random.choice(frontier)
        frontier.remove(cell)
        if cell in cells:
            continue
        cells.add(cell)
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = cell[0] + dr, cell[1] + dc
            if 0 <= nr < rows and 0 <= nc < cols and (nr, nc) not in cells:
                frontier.append((nr, nc))
    return list(cells) if len(cells) == n else None


def _spanning_polyomino(n: int, rows: int, cols: int) -> list[tuple[int, int]] | None:
    """Random 4-connected polyomino that touches all four sides of its bbox."""
    for _ in range(SPANNING_RETRIES):
        cells = _random_polyomino(n, rows, cols)
        if cells is None:
            continue
        rs = [int(r) for r, _c in cells]
        cs = [int(c) for _r, c in cells]
        if min(rs) == 0 and max(rs) == rows - 1 and min(cs) == 0 and max(cs) == cols - 1:
            return cells
    return None


def _is_rotation_distinct(cells: list[tuple[int, int]], size: int) -> bool:
    """Require four visually distinct quarter-turns inside the square bbox."""
    seen = set()
    for turns in range(4):
        seen.add(frozenset(_rotate_cells(cells, size, turns)))
    return len(seen) == 4


def _gen_template(size: int) -> list[tuple[int, int]] | None:
    """Square, spanning, 4-connected template with visible rotation asymmetry."""
    for _ in range(TEMPLATE_RETRIES):
        area = random.randint(*TEMPLATE_AREA_RANGE)
        cells = _spanning_polyomino(area, size, size)
        if cells is None:
            continue
        if _is_rotation_distinct(cells, size):
            return cells
    return None


def _gen_leaf_deltas() -> list[tuple[int, int]]:
    """All four diagonal quadrants are present; an optional fifth leaf repeats one quadrant."""
    leaves = list(BASE_LEAVES)
    if random.choice([False, True]):
        leaves.append(random.choice(SAFE_EXTRA_LEAVES))
    return leaves


def _segment_points(r0: int, c0: int, r1: int, c1: int) -> set[tuple[int, int]]:
    """Inclusive integer points on one horizontal or vertical segment."""
    assert r0 == r1 or c0 == c1
    pts: set[tuple[int, int]] = set()
    if r0 == r1:
        step = 1 if c1 >= c0 else -1
        for c in range(int(c0), int(c1) + step, step):
            pts.add((int(r0), int(c)))
    else:
        step = 1 if r1 >= r0 else -1
        for r in range(int(r0), int(r1) + step, step):
            pts.add((int(r), int(c0)))
    return pts


def _route_leaf_pixels(dr: int, dc: int) -> set[tuple[int, int]]:
    """Route one leaf from the origin using a quadrant-specific elbow path."""
    fr = int(dr * GUNIT)
    fc = int(dc * GUNIT)
    if dr < 0 and dc > 0:
        bend = (int(fr), 0)
    elif dr < 0 and dc < 0:
        bend = (0, int(fc))
    elif dr > 0 and dc < 0:
        bend = (int(fr), 0)
    else:
        bend = (0, int(fc))
    pts = _segment_points(0, 0, bend[0], bend[1])
    pts |= _segment_points(bend[0], bend[1], fr, fc)
    return pts


def _graph_relative_pixels(leaves: list[tuple[int, int]]) -> set[tuple[int, int]]:
    pts = {(0, 0)}
    for dr, dc in leaves:
        pts |= _route_leaf_pixels(int(dr), int(dc))
    return pts


def _delta_bounds(deltas: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    min_dr = min(int(dr) for dr, _dc in deltas)
    max_dr = max(int(dr) for dr, _dc in deltas)
    min_dc = min(int(dc) for _dr, dc in deltas)
    max_dc = max(int(dc) for _dr, dc in deltas)
    return int(min_dr), int(max_dr), int(min_dc), int(max_dc)


def _place_graph(
    graph_rel: set[tuple[int, int]],
    base_r: int,
    base_c: int,
    templ_cells: list[tuple[int, int]],
    bg: int,
    templ_col: int,
    overlap_col: int,
    expected: np.ndarray,
) -> np.ndarray | None:
    """Place the routed graph away from the template bbox and verify the solver output."""
    min_gr = min(int(r) for r, _c in graph_rel)
    max_gr = max(int(r) for r, _c in graph_rel)
    min_gc = min(int(c) for _r, c in graph_rel)
    max_gc = max(int(c) for _r, c in graph_rel)

    orr_lo = int(-min_gr)
    orr_hi = int(GRID_SIZE - 1 - max_gr)
    orc_lo = int(-min_gc)
    orc_hi = int(GRID_SIZE - 1 - max_gc)
    if orr_lo > orr_hi or orc_lo > orc_hi:
        return None

    tr0, tr1 = int(base_r), int(base_r + BOX_SIZE - 1)
    tc0, tc1 = int(base_c), int(base_c + BOX_SIZE - 1)

    for _ in range(GRAPH_PLACE_RETRIES):
        orr = random.randint(orr_lo, orr_hi)
        orc = random.randint(orc_lo, orc_hi)

        sr0 = int(orr + min_gr)
        sr1 = int(orr + max_gr)
        sc0 = int(orc + min_gc)
        sc1 = int(orc + max_gc)
        if not (sr1 + 1 < tr0 or tr1 + 1 < sr0 or sc1 + 1 < tc0 or tc1 + 1 < sc0):
            continue

        grid = np.full((GRID_SIZE, GRID_SIZE), int(bg), dtype=int)
        for tr, tc in templ_cells:
            grid[int(base_r + tr), int(base_c + tc)] = int(templ_col)
        for rr, cc in graph_rel:
            grid[int(orr + rr), int(orc + cc)] = int(overlap_col)

        try:
            out = main(grid)
        except Exception:
            continue
        if np.array_equal(out, expected):
            return grid
    return None


def _gen_attempt() -> np.ndarray | None:
    bg, templ_col, overlap_col = random.sample(range(10), 3)

    templ_cells = _gen_template(BOX_SIZE)
    if templ_cells is None:
        return None

    leaves = _gen_leaf_deltas()
    deltas = [(0, 0)] + [(int(dr), int(dc)) for dr, dc in leaves]
    min_dr, max_dr, min_dc, max_dc = _delta_bounds(deltas)

    base_r_lo = int(-min_dr * BOX_SIZE)
    base_r_hi = int(GRID_SIZE - max_dr * BOX_SIZE - BOX_SIZE)
    base_c_lo = int(-min_dc * BOX_SIZE)
    base_c_hi = int(GRID_SIZE - max_dc * BOX_SIZE - BOX_SIZE)
    if base_r_lo > base_r_hi or base_c_lo > base_c_hi:
        return None

    base_r = random.randint(base_r_lo, base_r_hi)
    base_c = random.randint(base_c_lo, base_c_hi)

    expected = _render_output(
        GRID_SIZE,
        GRID_SIZE,
        bg,
        templ_col,
        overlap_col,
        base_r,
        base_c,
        BOX_SIZE,
        templ_cells,
        deltas,
    )

    graph_rel = _graph_relative_pixels(leaves)
    return _place_graph(
        graph_rel,
        base_r,
        base_c,
        templ_cells,
        bg,
        templ_col,
        overlap_col,
        expected,
    )


def generate_input() -> np.ndarray:
    for _ in range(GENERATE_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"Failed to generate input after {GENERATE_RETRIES} attempts")
