# Combo variant of a395ee82 (index 7)
# combo_spec: D1/4+D3/1+D14/0
# Name: 连通性孔洞象限旋转

from __future__ import annotations

import math
import random
import sys
from collections import deque

import numpy as np

if "/home/lixinyu/snap/new/solvers" not in sys.path:
    sys.path.append("/home/lixinyu/snap/new/solvers")

from arc_utils import bbox_h, bbox_w, components_mask, find_components, get_bg


# Design analysis
# ---------------
# a) Concrete behavior and trace
#    - Preserve the original two-region input layout: one region is the template and one region is
#      the source marker structure. The difference is that the source is now a thin connected graph,
#      not isolated singletons, and the template is now a square holey mask so quarter-turn rotation
#      is geometrically meaningful.
#    - Concrete fused example:
#      1. Suppose the template bbox is 4x4 and contains one enclosed hole plus an edge notch, so its
#         four rotations are visually different.
#      2. Suppose the graph has one central junction and four leaves whose normalized offsets are
#         (-1,+1), (-1,-2), (+2,-1), (+1,+1). The central junction is not color-marked.
#      3. The solver first identifies the holey square component as the template, then identifies the
#         graph origin as the unique highest-degree pixel in the thin O-colored graph, then reads the
#         four leaf offsets relative to that origin.
#      4. At output time, the template is stamped at each leaf/origin offset, but the copy in the
#         upper-right quadrant stays canonical, the upper-left copy rotates 90 degrees CCW, the
#         lower-left copy rotates 180 degrees, and the lower-right copy rotates 270 degrees. The
#         origin copy uses the canonical orientation but is recolored O.
#    - Relative to the original task, all three moving parts change together:
#      color no longer reveals the origin, the template can no longer be treated as a solid bbox, and
#      the copied mask is no longer orientation-invariant.
# b) Complexity calibration
#    - The graph always exposes all four diagonal quadrants, so the quadrant-to-rotation mapping is
#      inferable from only a few examples instead of relying on rare cases.
#    - The connected graph is restricted to one 4-way center plus four leaves. This keeps the origin
#      uniquely identifiable by connectivity while avoiding multiple competing high-degree nodes.
#    - The template library is restricted to 4x4 connected masks with 1-2 enclosed hole pixels and
#      four distinct quarter-turns. That keeps the hypothesis space small enough for ARC while making
#      rotation and hole preservation both visually obvious.
# c) Implicit assumption audit against the original solver
#    - `template = max(comps, key=area)` is no longer valid: the graph can easily contain more pixels
#      than the template. The fused solver instead identifies the template by the presence of enclosed
#      holes inside a square bbox.
#    - `origin = singleton whose color == templ_col` is invalid because the source markers form one
#      monochrome graph and the origin is not color-marked. The solver now finds the unique maximum-
#      degree graph pixel.
#    - The D1 single-variant used even BFS parity to separate nodes from connectors, but that relies
#      on all edges being one midpoint long. Once we need diagonal quadrants for D14, the graph arms
#      become longer L-shaped paths; parity would create fake nodes on the arms. The fused solver
#      therefore uses graph endpoints (degree 1) plus the unique hub instead.
#    - The original constant-orientation stamp loop is also invalid. Each recovered offset must first
#      be mapped to a quadrant and then rotated before stamping.
# d) Placement legality
#    - Every output copy lives in a 4x4 anchor box, and all recovered coarse offsets are diagonal with
#      coordinate magnitudes in {1, 2}. This guarantees that different boxes are separated by at least
#      one full box width/height in some axis, so filled pixels never overlap.
#    - The template component and the input graph component are placed with at least one background
#      cell between their bboxes, so the solver always sees exactly two major foreground components.
#    - All graph pixels and all stamped template pixels stay within the 22x22 canvas by construction.
# e) Visual observability
#    - The input already looks different from the original task: there is no special T-colored dot;
#      instead there is a thin orthogonal graph with one obvious branching center.
#    - The output visibly preserves template holes under rotation. Because the template is forced to
#      have four distinct rotations, a human can see that different quadrants produce different hole
#      layouts instead of merely translated copies.
# f) Visual cleanliness
#    - The graph is a single thin tree and the template is a single dense block, so the two object
#      types remain easy to distinguish.
#    - Output copies are non-overlapping, avoiding clutter that would obscure the hole structure.
# g) Color inference and non-degeneracy
#    - Colors remain only `{bg, T, O}`. O still plays a dual role: it colors the input graph and the
#      output origin copy. The rule is still structural rather than color-id based.
#    - Generator-side metrics can verify all three fused behaviors:
#      hole count in the template is always 1 or 2, the graph has zero T-colored origin cue and a
#      unique degree-4 center, and the four non-origin quadrants realize four distinct rotated masks.
# h) New solver assumptions that the generator must satisfy
#    - Exactly one major component has a square bbox with 1-2 enclosed holes; that is the template.
#    - The graph is a single-pixel-thick tree with exactly one max-degree hub and four degree-1 leaf
#      endpoints, and every leaf lies in a strict diagonal quadrant relative to the hub.
#    - The normalized graph offsets use a single gcd unit of 2 pixels, so coarse deltas are integral.


GRID_SIZE = 22
BOX_SIZE = 4
TEMPLATE_AREA_RANGE = (9, 12)
TEMPLATE_HOLE_PIXELS = (1, 2)
GUNIT = 2
GENERATE_RETRIES = 200

QUADRANT_ORDER = [(-1, 1), (-1, -1), (1, -1), (1, 1)]
QUADRANT_CHOICES = {
    (-1, 1): [(-1, 1), (-2, 1), (-1, 2)],
    (-1, -1): [(-1, -1), (-2, -1), (-1, -2)],
    (1, -1): [(1, -1), (2, -1), (1, -2)],
    (1, 1): [(1, 1), (2, 1), (1, 2)],
}


def _enclosed_bg_mask(filled_mask: np.ndarray) -> np.ndarray:
    """Return the background cells inside a bbox that are not 4-connected to the bbox edge."""
    h, w = filled_mask.shape
    bg_mask = ~filled_mask.astype(bool)
    holes = np.zeros((h, w), dtype=bool)
    for coords in components_mask(bg_mask, connectivity=4):
        if coords.size == 0:
            continue
        touches_edge = any(
            int(r) == 0 or int(r) == h - 1 or int(c) == 0 or int(c) == w - 1
            for r, c in coords
        )
        if not touches_edge:
            holes[coords[:, 0], coords[:, 1]] = True
    return holes


def _cells_to_mask(cells: list[tuple[int, int]], size: int) -> np.ndarray:
    mask = np.zeros((size, size), dtype=bool)
    for r, c in cells:
        mask[int(r), int(c)] = True
    return mask


def _rotate_cells(cells: list[tuple[int, int]], size: int, turns: int) -> list[tuple[int, int]]:
    """Rotate bbox-relative cells by 90-degree CCW steps inside a square bbox."""
    turns %= 4
    out = [(int(r), int(c)) for r, c in cells]
    for _ in range(turns):
        out = [(int(c), int(size - 1 - r)) for r, c in out]
    return out


def _quadrant_turns(dr: int, dc: int) -> int:
    """Map diagonal coarse offsets to the quadrant-conditioned rotation."""
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


def _component_mask(comp: dict) -> tuple[np.ndarray, int, int]:
    r0, c0, r1, c1 = map(int, comp["bbox"])
    mask = np.zeros((int(bbox_h((r0, c0, r1, c1))), int(bbox_w((r0, c0, r1, c1)))), dtype=bool)
    for r, c in comp["coords"]:
        mask[int(r) - int(r0), int(c) - int(c0)] = True
    return mask, int(r0), int(c0)


def _identify_template_and_graph(comps: list[dict]) -> tuple[dict, dict]:
    big = [comp for comp in comps if int(comp["area"]) > 1]
    assert len(big) == 2
    template_candidates: list[dict] = []
    for comp in big:
        mask, _r0, _c0 = _component_mask(comp)
        if mask.shape != (BOX_SIZE, BOX_SIZE):
            continue
        hole_pixels = int(_enclosed_bg_mask(mask).sum())
        if hole_pixels in TEMPLATE_HOLE_PIXELS:
            template_candidates.append(comp)
    assert len(template_candidates) == 1
    template = template_candidates[0]
    graph_comp = big[1] if big[0] is template else big[0]
    return template, graph_comp


def _extract_origin_and_leaves(graph_comp: dict) -> tuple[tuple[int, int], list[tuple[int, int]]]:
    coords = {(int(r), int(c)) for r, c in graph_comp["coords"]}
    assert coords

    deg = _graph_degrees(coords)
    max_deg = max(deg.values())
    origin_candidates = [pt for pt, d in deg.items() if int(d) == int(max_deg)]
    assert len(origin_candidates) == 1
    assert int(max_deg) >= 3
    origin = origin_candidates[0]

    leaves = [pt for pt, d in deg.items() if int(d) == 1]
    assert len(leaves) == 4
    quadrants = {(int(np.sign(r - origin[0])), int(np.sign(c - origin[1]))) for r, c in leaves}
    assert quadrants == set(QUADRANT_ORDER)
    return origin, sorted(leaves)


def _render_output(
    H: int,
    W: int,
    bg: int,
    templ_col: int,
    overlap_col: int,
    base_r: int,
    base_c: int,
    templ_cells: list[tuple[int, int]],
    deltas: list[tuple[int, int]],
) -> np.ndarray:
    total = np.zeros((H, W), dtype=int)
    center = np.zeros((H, W), dtype=int)
    for dr, dc in deltas:
        rot_cells = _rotate_cells(templ_cells, BOX_SIZE, _quadrant_turns(dr, dc))
        is_origin = int(dr) == 0 and int(dc) == 0
        off_r = int(base_r + dr * BOX_SIZE)
        off_c = int(base_c + dc * BOX_SIZE)
        for rr, cc in rot_cells:
            total[int(off_r + rr), int(off_c + cc)] += 1
            if is_origin:
                center[int(off_r + rr), int(off_c + cc)] += 1

    out = np.full((H, W), int(bg), dtype=int)
    out[total > 0] = int(templ_col)
    out[(total == 1) & (center > 0)] = int(overlap_col)
    out[total >= 2] = int(overlap_col)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Stamp rotated copies of a holey square template using a connected graph to recover the origin.

    Input structure:
    - `bg` is the dominant background color.
    - One dense monochrome component in color `T` is the canonical template. Its bbox is a 4x4 square,
      and the filled mask contains 1-2 enclosed background holes.
    - Elsewhere, one monochrome component in color `O` forms a thin orthogonal graph. The graph has a
      unique high-degree hub and four leaf endpoints; no graph pixel is specially recolored to mark the origin.

    Rule:
    - Identify the template by its holey square structure, and identify the graph origin as the unique
      maximum-degree graph pixel.
    - Use the four graph leaves plus the origin itself as lattice nodes, normalize their offsets by the
      common pixel spacing, and convert those coarse offsets into translated template-box anchors.
    - Stamp the template's filled cells only, preserving holes. The upper-right quadrant uses the input
      orientation, the upper-left rotates 90 degrees CCW, the lower-left rotates 180 degrees, and the
      lower-right rotates 270 degrees. The origin copy keeps the canonical orientation.
    - Recolor the origin copy to `O`, and recolor any multiply covered filled pixel to `O` as well.

    Key insight: The special source location is no longer encoded by color, and the copied object is no
    longer rotation-invariant. The solver must first recover a hub-and-leaf graph from connectivity, then
    stamp the exact holey mask with a quadrant-conditioned quarter-turn.
    """
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))
    H, W = g.shape

    comps = find_components(g, bg=bg, connectivity=4, monochromatic=True, return_format="dict")
    template, graph_comp = _identify_template_and_graph(comps)

    templ_mask, r0, c0 = _component_mask(template)
    templ_col = int(template["color"])
    overlap_col = int(graph_comp["color"])
    assert templ_mask.shape == (BOX_SIZE, BOX_SIZE)
    hole_pixels = int(_enclosed_bg_mask(templ_mask).sum())
    assert hole_pixels in TEMPLATE_HOLE_PIXELS
    templ_rel = [tuple(map(int, rc)) for rc in np.argwhere(templ_mask)]

    origin, leaves = _extract_origin_and_leaves(graph_comp)
    orr, orc = map(int, origin)
    nodes = [origin] + leaves

    raw_deltas = [(int(r - orr), int(c - orc)) for r, c in nodes]
    gcd_vals = [abs(int(x)) for dr, dc in raw_deltas for x in (dr, dc) if int(x) != 0]
    assert gcd_vals
    gunit = max(1, int(math.gcd(*gcd_vals)))
    deltas = [(int(dr // gunit), int(dc // gunit)) for dr, dc in raw_deltas]
    assert (0, 0) in deltas
    assert {(int(np.sign(dr)), int(np.sign(dc))) for dr, dc in deltas if (dr, dc) != (0, 0)} == set(QUADRANT_ORDER)

    total = np.zeros((H, W), dtype=int)
    center = np.zeros((H, W), dtype=int)

    for dr, dc in deltas:
        turns = _quadrant_turns(dr, dc)
        rot_cells = _rotate_cells(templ_rel, BOX_SIZE, turns)
        is_origin = int(dr) == 0 and int(dc) == 0
        off_r = int(r0 + dr * BOX_SIZE)
        off_c = int(c0 + dc * BOX_SIZE)
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


def _template_candidates() -> dict[int, list[list[tuple[int, int]]]]:
    """Enumerate legal 4x4 holey templates, grouped by enclosed-hole count."""
    grouped: dict[int, list[list[tuple[int, int]]]] = {1: [], 2: []}
    total = BOX_SIZE * BOX_SIZE
    for bits in range(1, 1 << total):
        cells = [(idx // BOX_SIZE, idx % BOX_SIZE) for idx in range(total) if (bits >> idx) & 1]
        area = len(cells)
        if area < TEMPLATE_AREA_RANGE[0] or area > TEMPLATE_AREA_RANGE[1]:
            continue

        mask = _cells_to_mask(cells, BOX_SIZE)
        comps = components_mask(mask, connectivity=4)
        if len(comps) != 1:
            continue
        if not mask[0, :].any() or not mask[BOX_SIZE - 1, :].any():
            continue
        if not mask[:, 0].any() or not mask[:, BOX_SIZE - 1].any():
            continue

        hole_pixels = int(_enclosed_bg_mask(mask).sum())
        if hole_pixels not in grouped:
            continue

        rots = {frozenset(_rotate_cells(cells, BOX_SIZE, turns)) for turns in range(4)}
        if len(rots) != 4:
            continue
        grouped[hole_pixels].append(cells)
    return grouped


TEMPLATE_LIBRARY = _template_candidates()


def _gen_template() -> tuple[list[tuple[int, int]], int]:
    hole_order = list(TEMPLATE_HOLE_PIXELS)
    random.shuffle(hole_order)
    for hole_pixels in hole_order:
        candidates = TEMPLATE_LIBRARY[hole_pixels]
        if candidates:
            return random.choice(candidates), int(hole_pixels)
    raise RuntimeError("No legal template candidates found")


def _gen_leaf_deltas() -> list[tuple[int, int]]:
    return [random.choice(QUADRANT_CHOICES[q]) for q in QUADRANT_ORDER]


def _graph_relative_pixels(leaf_deltas: list[tuple[int, int]]) -> set[tuple[int, int]]:
    pts = {(0, 0)}
    for dr, dc in leaf_deltas:
        vr = abs(int(dr)) * GUNIT
        hc = abs(int(dc)) * GUNIT
        if dr < 0 and dc > 0:
            for step in range(1, vr + 1):
                pts.add((-step, 0))
            for step in range(1, hc + 1):
                pts.add((-vr, step))
        elif dr < 0 and dc < 0:
            for step in range(1, hc + 1):
                pts.add((0, -step))
            for step in range(1, vr + 1):
                pts.add((-step, -hc))
        elif dr > 0 and dc < 0:
            for step in range(1, vr + 1):
                pts.add((step, 0))
            for step in range(1, hc + 1):
                pts.add((vr, -step))
        else:
            assert dr > 0 and dc > 0
            for step in range(1, hc + 1):
                pts.add((0, step))
            for step in range(1, vr + 1):
                pts.add((step, hc))
    return {(int(r), int(c)) for r, c in pts}


def _delta_bounds(deltas: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    min_dr = min(dr for dr, _dc in deltas)
    max_dr = max(dr for dr, _dc in deltas)
    min_dc = min(dc for _dr, dc in deltas)
    max_dc = max(dc for _dr, dc in deltas)
    return int(min_dr), int(max_dr), int(min_dc), int(max_dc)


def _place_graph(
    graph_rel: set[tuple[int, int]],
    templ_cells: list[tuple[int, int]],
    base_r: int,
    base_c: int,
    bg: int,
    templ_col: int,
    overlap_col: int,
    expected: np.ndarray,
) -> np.ndarray | None:
    rel_rs = [r for r, _c in graph_rel]
    rel_cs = [c for _r, c in graph_rel]
    min_r, max_r = min(rel_rs), max(rel_rs)
    min_c, max_c = min(rel_cs), max(rel_cs)

    orr_lo = int(-min_r)
    orr_hi = int(GRID_SIZE - 1 - max_r)
    orc_lo = int(-min_c)
    orc_hi = int(GRID_SIZE - 1 - max_c)
    if orr_lo > orr_hi or orc_lo > orc_hi:
        return None

    tr0, tr1 = int(base_r), int(base_r + BOX_SIZE - 1)
    tc0, tc1 = int(base_c), int(base_c + BOX_SIZE - 1)

    candidates = [(int(orr), int(orc)) for orr in range(orr_lo, orr_hi + 1) for orc in range(orc_lo, orc_hi + 1)]
    random.shuffle(candidates)
    for orr, orc in candidates:
        graph_pts = {(int(orr + rr), int(orc + cc)) for rr, cc in graph_rel}

        sr0 = int(orr + min_r)
        sr1 = int(orr + max_r)
        sc0 = int(orc + min_c)
        sc1 = int(orc + max_c)
        if not (sr1 + 1 < tr0 or tr1 + 1 < sr0 or sc1 + 1 < tc0 or tc1 + 1 < sc0):
            continue

        grid = np.full((GRID_SIZE, GRID_SIZE), bg, dtype=int)
        for tr, tc in templ_cells:
            grid[int(base_r + tr), int(base_c + tc)] = int(templ_col)
        for rr, cc in graph_pts:
            grid[int(rr), int(cc)] = int(overlap_col)

        try:
            out = main(grid)
        except Exception:
            continue
        if np.array_equal(out, expected):
            return grid
    return None


def _gen_attempt() -> np.ndarray | None:
    bg, templ_col, overlap_col = random.sample(range(10), 3)

    templ_cells, _hole_pixels = _gen_template()
    leaf_deltas = _gen_leaf_deltas()
    deltas = [(0, 0)] + leaf_deltas
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
        templ_cells,
        deltas,
    )

    graph_rel = _graph_relative_pixels(leaf_deltas)
    return _place_graph(graph_rel, templ_cells, base_r, base_c, bg, templ_col, overlap_col, expected)


def generate_input() -> np.ndarray:
    for _ in range(GENERATE_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"Failed to generate input after {GENERATE_RETRIES} attempts")
