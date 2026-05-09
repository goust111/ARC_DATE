# Combo variant of 35ab12c3 (index 31)
# combo_spec: D6/2+D14/1+D19/0
# Name: 重叠擦除+边-阴影交替绘制+锚点在任意顶点

import itertools
import math

import numpy as np

from arc_utils import bbox_coords, colors_present, get_bg, paste, random_free_location
from arc_utils import get_bg as _bg_color

# -----------------------------------------------------------------------------
# Design analysis for the fused combo
#
# a) Concretization / behavior trace
# - Base task: detect each same-color vertex set, order it into a polyline/polygon
#   cycle, draw all main edges, then draw one shadow family copied from the chosen
#   start vertex's adjacent helper pixel. Shape pixels dominate because shadows are
#   written only onto background.
# - Fused combo:
#   1. every vertex can contribute its own shadow anchor (D14),
#   2. rendering is edge-by-edge: draw one main edge, then immediately draw all
#      shadow copies of that edge (D19),
#   3. a same-shape main pixel colliding with a same-shape shadow pixel erases to
#      background at that moment (D6).
# - Small trace: use a 1x3 thin rectangle with two anchors, one near the top-left
#   corner and one near the lower-right corner. Under the base task, only the start
#   corner's shadow exists, all main edges are laid down first, and overlaps keep the
#   main color. Under the fused task, both anchors generate shadows; after drawing the
#   top edge, its shadow may already occupy cells that the later right/bottom edge
#   wants. When those later main cells arrive they erase the earlier shadow to
#   background, creating holes. The second anchor adds another shadow family that the
#   base task would never see.
#
# b) Complexity calibration
# - If the generator allowed only one anchor or overlap-free offsets, the combo would
#   visually collapse back toward the original task. Therefore each generated grid
#   contains one "primary" shadowed shape with exactly two anchors on different
#   vertices, and that shape is accepted only if three deltas are all positive:
#     * fused output != start-anchor-only output  (D14 is active)
#     * fused output != two-phase XOR output      (D19 is active)
#     * fused output != interleaved shape-wins    (D6 is active)
# - Other groups remain ordinary shapes or a sparse distractor, so examples still look
#   like the source task instead of a single over-constrained gadget.
#
# c) Implicit-assumption audit of main()
# - Original `carried` read only from `cycle[0]`; D14 breaks that assumption, so shadow
#   specs must be collected from all vertices and filtered by unique anchor adjacency.
# - Original phase split ("all main edges, then all shadows") encoded a hidden priority
#   rule. D19 removes that assumption, so rendering must be parameterized by schedule.
# - Original shadow compositor relied on "write only onto background". D6 breaks that:
#   main-vs-shadow collisions must explicitly erase. A pure set-based XOR would make
#   D19 inert, so the erase logic is implemented as a sequential state machine that
#   acts at draw time.
# - The solver now assumes that a valid shadow anchor pixel is attributable to exactly
#   one vertex of its host shape. The generator guarantees this.
#
# d) Placement legality
# - Every local patch is built input-first (vertices + anchor pixels), then verified by
#   running `main(patch_in)` to get the patch output oracle.
# - Global placement reserves the full patch bounding box, not only visible output
#   cells. This prevents another object from being pasted into XOR holes or anchor-only
#   margins, which would otherwise break `_clear_in_original` or shadow attribution.
# - Inter-group overlaps are disallowed; only intra-shape main/shadow interactions are
#   used to express the combo.
#
# e) Visual observability
# - The primary shape always exposes two anchor colors in input and multiple shadow
#   directions in output, so humans can infer "anchors may sit on arbitrary vertices".
# - The output always contains visible background holes created by same-shape
#   main/shadow cancellation, not just extra colored lines.
# - At least one sample-local cell changes when switching from two-phase XOR rendering
#   to interleaved XOR rendering, so the edge/shadow alternation is not latent.
#
# f) Visual cleanliness
# - The generator uses a small number of objects, one-cell padding between occupied
#   patch boxes, and at most one multi-shadow primary shape. This keeps the woven
#   effect local and legible instead of letting different objects interfere globally.
# - Shadow-shadow overlap is not required for the combo and is avoided by the primary
#   shape search when it would obscure the pattern.
#
# g) Color inference
# - Each shadow color still appears as exactly one input anchor pixel and one shadow
#   family in output. No hidden color-id ordering is used; all decisions depend on
#   visible geometry and adjacency.
#
# Post-change solver assumptions and generator guarantees
# ------------------------------------------------------
# 1. Each accepted shadow anchor is 8-adjacent to exactly one shape vertex.
# 2. The primary patch has positive deltas for D6, D14, and D19 against dedicated
#    ablation renderers, so the saved tasks are not degenerate.
# 3. Distinct object patches are globally separated by bbox placement, so the solver's
#    same-shape erase logic does not accidentally erase another object.
# -----------------------------------------------------------------------------

DIR8 = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]


def _aligned_8(r0: int, c0: int, r1: int, c1: int) -> bool:
    dr = r1 - r0
    dc = c1 - c0
    if dr == 0 and dc != 0:
        return True
    if dc == 0 and dr != 0:
        return True
    if dr != 0 and abs(dr) == abs(dc):
        return True
    return False


def _step_dir(r0: int, c0: int, r1: int, c1: int) -> tuple[int, int]:
    dr = r1 - r0
    dc = c1 - c0
    sr = 0 if dr == 0 else (1 if dr > 0 else -1)
    sc = 0 if dc == 0 else (1 if dc > 0 else -1)
    return sr, sc


def _segment_cells(r0: int, c0: int, r1: int, c1: int):
    sr, sc = _step_dir(r0, c0, r1, c1)
    steps = max(abs(r1 - r0), abs(c1 - c0))
    for k in range(steps + 1):
        yield r0 + sr * k, c0 + sc * k


def _clear_in_original(g: np.ndarray, r0: int, c0: int, r1: int, c1: int, bg: int, allow: set[int]) -> bool:
    first = True
    for rr, cc in _segment_cells(r0, c0, r1, c1):
        if first:
            first = False
            continue
        if rr == r1 and cc == c1:
            break
        v = int(g[rr, cc])
        if v != bg and v not in allow:
            return False
    return True


def _angle_cycle(points_rc: list[tuple[int, int]]) -> list[tuple[int, int]]:
    cr = sum(r for r, _ in points_rc) / len(points_rc)
    cc = sum(c for _, c in points_rc) / len(points_rc)

    def key(p: tuple[int, int]):
        r, c = p
        ang = math.atan2(r - cr, c - cc)
        rad = (r - cr) ** 2 + (c - cc) ** 2
        return (ang, rad, r, c)

    return sorted(points_rc, key=key)


def _rotate_cycle(cycle: list[tuple[int, int]], start_idx: int) -> list[tuple[int, int]]:
    return cycle[start_idx:] + cycle[:start_idx]


def _shape_edges(vertices: list[tuple[int, int]]) -> list[tuple[tuple[int, int], tuple[int, int]]]:
    if len(vertices) == 2:
        return [(vertices[0], vertices[1])]
    return [(vertices[i], vertices[(i + 1) % len(vertices)]) for i in range(len(vertices))]


def _shape_cells(vertices: list[tuple[int, int]]) -> set[tuple[int, int]]:
    cells: set[tuple[int, int]] = set()
    for a, b in _shape_edges(vertices):
        cells.update(_segment_cells(a[0], a[1], b[0], b[1]))
    return cells


def _collect_shadow_specs(
    g: np.ndarray,
    cycle: list[tuple[int, int]],
    col: int,
    bg: int,
    *,
    all_vertices: bool,
) -> list[dict[str, object]]:
    h, w = g.shape
    scan_vertices = cycle if all_vertices else cycle[:1]
    hits: dict[tuple[int, int], list[dict[str, object]]] = {}
    for vr, vc in scan_vertices:
        for drn, dcn in DIR8:
            rr = vr + drn
            cc = vc + dcn
            if 0 <= rr < h and 0 <= cc < w:
                v = int(g[rr, cc])
                if v != bg and v != col:
                    hits.setdefault((rr, cc), []).append(
                        {
                            "anchor_vertex": (vr, vc),
                            "anchor_cell": (rr, cc),
                            "offset": (drn, dcn),
                            "color": v,
                        }
                    )

    specs: list[dict[str, object]] = []
    for anchor_rc in sorted(hits):
        choices = hits[anchor_rc]
        if len(choices) == 1:
            specs.append(choices[0])
    return specs


def _shape_plan(
    g: np.ndarray,
    col: int,
    bg: int,
    *,
    all_vertices: bool,
) -> tuple[list[tuple[tuple[int, int], tuple[int, int]]], list[dict[str, object]]] | None:
    h, w = g.shape
    pts_rc = [tuple(map(int, p)) for p in np.argwhere(g == col)]
    if len(pts_rc) < 2:
        return None

    cycle = _angle_cycle(pts_rc)
    if len(cycle) < 2:
        return None

    seed_idxs = []
    for i, (r, c) in enumerate(cycle):
        for drn, dcn in DIR8:
            rr = r + drn
            cc = c + dcn
            if 0 <= rr < h and 0 <= cc < w:
                v = int(g[rr, cc])
                if v != bg and v != col:
                    seed_idxs.append(i)
                    break

    if seed_idxs:
        start = min(seed_idxs, key=lambda i: (cycle[i][0], cycle[i][1]))
    else:
        start = min(range(len(cycle)), key=lambda i: (cycle[i][0], cycle[i][1]))

    cycle = _rotate_cycle(cycle, start)
    shadow_specs = _collect_shadow_specs(g, cycle, col, bg, all_vertices=all_vertices)

    edges = _shape_edges(cycle)
    drawable_edges: list[tuple[tuple[int, int], tuple[int, int]]] = []
    for (r0, c0), (r1, c1) in edges:
        if not _aligned_8(r0, c0, r1, c1):
            continue
        if not _clear_in_original(g, r0, c0, r1, c1, bg, allow={col}):
            continue
        drawable_edges.append(((r0, c0), (r1, c1)))
    return drawable_edges, shadow_specs


def _collect_shape_cells_from_edges(
    edges: list[tuple[tuple[int, int], tuple[int, int]]],
) -> set[tuple[int, int]]:
    cells: set[tuple[int, int]] = set()
    for (r0, c0), (r1, c1) in edges:
        cells.update(_segment_cells(r0, c0, r1, c1))
    return cells


def _collect_shadow_cells_for_spec(
    g: np.ndarray,
    edges: list[tuple[tuple[int, int], tuple[int, int]]],
    spec: dict[str, object],
    bg: int,
) -> set[tuple[int, int]]:
    h, w = g.shape
    drn, dcn = spec["offset"]
    cells: set[tuple[int, int]] = set()
    for (r0, c0), (r1, c1) in edges:
        drs, dcs = _step_dir(r0, c0, r1, c1)
        if drs * dcn - dcs * drn == 0:
            continue
        rr0 = r0 + drn
        cc0 = c0 + dcn
        rr1 = r1 + drn
        cc1 = c1 + dcn
        if not (0 <= rr0 < h and 0 <= cc0 < w and 0 <= rr1 < h and 0 <= cc1 < w):
            continue
        if not _aligned_8(rr0, cc0, rr1, cc1):
            continue
        if not _clear_in_original(g, rr0, cc0, rr1, cc1, bg, allow=set()):
            continue
        cells.update(_segment_cells(rr0, cc0, rr1, cc1))
    return cells


def _apply_cells(
    out: np.ndarray,
    owner: np.ndarray,
    kind: np.ndarray,
    cells,
    *,
    color: int,
    shape_id: int,
    draw_kind: int,
    bg: int,
    erase_overlap: bool,
) -> None:
    for rr, cc in cells:
        cur = int(out[rr, cc])
        cur_owner = int(owner[rr, cc])
        cur_kind = int(kind[rr, cc])
        if cur == bg:
            out[rr, cc] = color
            owner[rr, cc] = shape_id
            kind[rr, cc] = draw_kind
        elif erase_overlap and cur_owner == shape_id and cur_kind != 0 and cur_kind != draw_kind:
            out[rr, cc] = bg
            owner[rr, cc] = 0
            kind[rr, cc] = 0


def _render_grid(
    grid: np.ndarray,
    *,
    all_vertices: bool,
    interleaved: bool,
    erase_overlap: bool,
) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    bg = _bg_color(g)
    out = g.copy()
    owner = np.zeros_like(out, dtype=int)
    kind = np.zeros_like(out, dtype=int)  # 0 none, 1 main, 2 shadow
    colors = sorted(set(int(v) for v in np.unique(g)) - {bg})

    for shape_id, col in enumerate(colors, start=1):
        plan = _shape_plan(g, col, bg, all_vertices=all_vertices)
        if plan is None:
            continue
        drawable_edges, shadow_specs = plan
        if not drawable_edges:
            continue

        affected = _collect_shape_cells_from_edges(drawable_edges)
        for spec in shadow_specs:
            affected.add(spec["anchor_cell"])
            affected.update(_collect_shadow_cells_for_spec(g, drawable_edges, spec, bg))

        for rr, cc in affected:
            out[rr, cc] = bg
            owner[rr, cc] = 0
            kind[rr, cc] = 0

        if interleaved:
            for (r0, c0), (r1, c1) in drawable_edges:
                _apply_cells(
                    out,
                    owner,
                    kind,
                    _segment_cells(r0, c0, r1, c1),
                    color=col,
                    shape_id=shape_id,
                    draw_kind=1,
                    bg=bg,
                    erase_overlap=erase_overlap,
                )
                for spec in shadow_specs:
                    drn, dcn = spec["offset"]
                    drs, dcs = _step_dir(r0, c0, r1, c1)
                    if drs * dcn - dcs * drn == 0:
                        continue
                    rr0 = r0 + drn
                    cc0 = c0 + dcn
                    rr1 = r1 + drn
                    cc1 = c1 + dcn
                    if not (0 <= rr0 < g.shape[0] and 0 <= cc0 < g.shape[1] and 0 <= rr1 < g.shape[0] and 0 <= cc1 < g.shape[1]):
                        continue
                    if not _aligned_8(rr0, cc0, rr1, cc1):
                        continue
                    if not _clear_in_original(g, rr0, cc0, rr1, cc1, bg, allow=set()):
                        continue
                    _apply_cells(
                        out,
                        owner,
                        kind,
                        _segment_cells(rr0, cc0, rr1, cc1),
                        color=int(spec["color"]),
                        shape_id=shape_id,
                        draw_kind=2,
                        bg=bg,
                        erase_overlap=erase_overlap,
                    )
        else:
            for (r0, c0), (r1, c1) in drawable_edges:
                _apply_cells(
                    out,
                    owner,
                    kind,
                    _segment_cells(r0, c0, r1, c1),
                    color=col,
                    shape_id=shape_id,
                    draw_kind=1,
                    bg=bg,
                    erase_overlap=erase_overlap,
                )
            for (r0, c0), (r1, c1) in drawable_edges:
                for spec in shadow_specs:
                    drn, dcn = spec["offset"]
                    drs, dcs = _step_dir(r0, c0, r1, c1)
                    if drs * dcn - dcs * drn == 0:
                        continue
                    rr0 = r0 + drn
                    cc0 = c0 + dcn
                    rr1 = r1 + drn
                    cc1 = c1 + dcn
                    if not (0 <= rr0 < g.shape[0] and 0 <= cc0 < g.shape[1] and 0 <= rr1 < g.shape[0] and 0 <= cc1 < g.shape[1]):
                        continue
                    if not _aligned_8(rr0, cc0, rr1, cc1):
                        continue
                    if not _clear_in_original(g, rr0, cc0, rr1, cc1, bg, allow=set()):
                        continue
                    _apply_cells(
                        out,
                        owner,
                        kind,
                        _segment_cells(rr0, cc0, rr1, cc1),
                        color=int(spec["color"]),
                        shape_id=shape_id,
                        draw_kind=2,
                        bg=bg,
                        erase_overlap=erase_overlap,
                    )
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Overview:
        Reconstruct each shape's outline and weave its shadows in edge-by-edge order.

    Input structure:
        Several colors form shape vertices (>=2 pixels of one color). Other singleton
        colors may appear 8-adjacent to shape vertices as shadow anchors. A distractor
        color may also exist as sparse non-alignable pixels.

    Rule:
        Order each shape color into its cycle, scan every vertex for uniquely
        attributable shadow anchors, then for each drawable edge draw the main edge and
        immediately draw every non-collinear parallel shadow edge. If a same-shape main
        pixel meets a same-shape shadow pixel, erase that cell to background.

    Key insight:
        The combo is not "draw all shapes, then XOR with all shadows". The erase effect
        happens during interleaved drawing, so both anchor coverage and edge order
        matter.
    """
    return _render_grid(np.asarray(grid, dtype=int), all_vertices=True, interleaved=True, erase_overlap=True)


def _main_two_phase_xor(grid: np.ndarray) -> np.ndarray:
    return _render_grid(np.asarray(grid, dtype=int), all_vertices=True, interleaved=False, erase_overlap=True)


def _main_interleaved_shape_wins(grid: np.ndarray) -> np.ndarray:
    return _render_grid(np.asarray(grid, dtype=int), all_vertices=True, interleaved=True, erase_overlap=False)


def _main_start_anchor_only(grid: np.ndarray) -> np.ndarray:
    return _render_grid(np.asarray(grid, dtype=int), all_vertices=False, interleaved=True, erase_overlap=True)


# --------------- generator ---------------

BG = 0
COLORS = [1, 2, 3, 4, 6, 7, 8]
CANVAS_H_RANGE = (15, 23)
CANVAS_W_JITTER = (-3, 4)
CANVAS_W_CLIP = (15, 22)

SHAPE_COUNT_RANGE = (2, 5)
PRIMARY_SHADOW_COUNT = 2
DISTRACTOR_PROB = 0.45
PLACEMENT_PADDING = 1
PLACEMENT_PAD_CONNECTIVITY = 8

LINE_VERTICES_RANGE = (2, 8)
LINE_TOTAL_LENGTH = 13
LINE_MIN_GAP = 2
TRIANGLE_SIDE_RANGE = (2, 8)
RECT_H_RANGE = (2, 8)
RECT_W_RANGE = (2, 8)
DIAMOND_ARM_RANGE = (2, 6)
PENTAGON_ROOF_RANGE = (2, 5)
PENTAGON_WALL_RANGE = (2, 6)
HEXAGON_SLANT_RANGE = (2, 5)
HEXAGON_WALL_RANGE = (2, 5)
THIN_RECT_W_RANGE = (3, 8)

SHAPE_WEIGHTS = np.array([0.30, 0.18, 0.18, 0.14, 0.11, 0.09], dtype=float)

DISTRACTOR_COUNT_RANGE = (1, 4)
DISTRACTOR_BBOX_RANGE = (2, 5)
DISTRACTOR_RETRIES = 50

PRIMARY_PATCH_ATTEMPTS = 80
PLAIN_PATCH_ATTEMPTS = 50
SHAPE_PLACEMENT_RETRIES = 60
GENERATE_RETRIES = 500
MIN_D19_DELTA = 2
MIN_D14_DELTA = 2
MIN_D6_DELTA = 1
# Keep the weave effect visually explicit instead of allowing one-cell deltas.
MIN_ERASE_CELLS = 3


def _aligned(dr: int, dc: int) -> bool:
    return (dr == 0 and dc != 0) or (dc == 0 and dr != 0) or (dr != 0 and abs(dr) == abs(dc))


def _transform_coords(coords: list[tuple[int, int]]) -> list[tuple[int, int]]:
    arr = np.array(coords, dtype=int)
    if np.random.rand() < 0.5:
        arr = arr[:, ::-1]
    if np.random.rand() < 0.5:
        arr[:, 0] *= -1
    if np.random.rand() < 0.5:
        arr[:, 1] *= -1
    arr -= arr.min(axis=0)
    return [tuple(map(int, p)) for p in arr]


def _build_line() -> list[tuple[int, int]]:
    n_vertices = int(np.random.randint(*LINE_VERTICES_RANGE))
    remaining = LINE_TOTAL_LENGTH
    pos = [0]
    for i in range(n_vertices - 1):
        steps_left = n_vertices - 2 - i
        gap_hi = remaining - LINE_MIN_GAP * steps_left
        gap = int(np.random.randint(LINE_MIN_GAP, gap_hi + 1))
        pos.append(pos[-1] + gap)
        remaining -= gap
    return _transform_coords([(0, x) for x in pos])


def _build_triangle() -> list[tuple[int, int]]:
    side = int(np.random.randint(*TRIANGLE_SIDE_RANGE))
    return _transform_coords([(0, 0), (0, side), (side, side)])


def _build_rectangle() -> list[tuple[int, int]]:
    h = int(np.random.randint(*RECT_H_RANGE))
    w = int(np.random.randint(*RECT_W_RANGE))
    return _transform_coords([(0, 0), (0, w), (h, w), (h, 0)])


def _build_diamond() -> list[tuple[int, int]]:
    arm = int(np.random.randint(*DIAMOND_ARM_RANGE))
    return _transform_coords([(0, arm), (arm, 2 * arm), (2 * arm, arm), (arm, 0)])


def _build_pentagon() -> list[tuple[int, int]]:
    roof = int(np.random.randint(*PENTAGON_ROOF_RANGE))
    wall = int(np.random.randint(*PENTAGON_WALL_RANGE))
    h = roof + wall
    return _transform_coords([(0, roof), (roof, 0), (h, 0), (h, 2 * roof), (roof, 2 * roof)])


def _build_hexagon() -> list[tuple[int, int]]:
    slant = int(np.random.randint(*HEXAGON_SLANT_RANGE))
    wall = int(np.random.randint(*HEXAGON_WALL_RANGE))
    mid = slant + wall
    bot = 2 * slant + wall
    return _transform_coords([(0, slant), (slant, 0), (mid, 0), (bot, slant), (mid, 2 * slant), (slant, 2 * slant)])


def _build_thin_rectangle() -> list[tuple[int, int]]:
    w = int(np.random.randint(*THIN_RECT_W_RANGE))
    return _transform_coords([(0, 0), (0, w), (1, w), (1, 0)])


_SHAPE_BUILDERS = [_build_line, _build_triangle, _build_rectangle, _build_diamond, _build_pentagon, _build_hexagon]


def _make_plain_vertices() -> list[tuple[int, int]]:
    weights = SHAPE_WEIGHTS / SHAPE_WEIGHTS.sum()
    idx = int(np.random.choice(len(_SHAPE_BUILDERS), p=weights))
    return _SHAPE_BUILDERS[idx]()


def _make_primary_vertices() -> list[tuple[int, int]]:
    return _build_thin_rectangle()


def _count_vertex_adjacencies(vertices: list[tuple[int, int]], pos: tuple[int, int]) -> int:
    return sum(1 for vr, vc in vertices if max(abs(pos[0] - vr), abs(pos[1] - vc)) == 1)


def _shadow_candidates(vertices: list[tuple[int, int]]) -> list[tuple[int, tuple[int, int]]]:
    cycle = _angle_cycle(vertices)
    outline = _shape_cells(cycle)
    candidates: list[tuple[int, tuple[int, int]]] = []
    for vidx, anchor in enumerate(cycle):
        for dr, dc in DIR8:
            pos = (anchor[0] + dr, anchor[1] + dc)
            if pos in outline:
                continue
            if _count_vertex_adjacencies(cycle, pos) != 1:
                continue
            candidates.append((vidx, (dr, dc)))
    return candidates


def _build_input_patch(
    vertices: list[tuple[int, int]],
    shape_color: int,
    shadows: list[tuple[int, tuple[int, int], int]],
) -> np.ndarray | None:
    cycle = _angle_cycle(vertices)
    coords = list(cycle)
    for vidx, (dr, dc), _shadow_color in shadows:
        vr, vc = cycle[vidx]
        coords.append((vr + dr, vc + dc))
    r0, c0, r1, c1 = bbox_coords(np.array(coords, dtype=int))
    h, w = int(r1 - r0 + 1), int(c1 - c0 + 1)
    inp = np.full((h, w), BG, dtype=int)

    local_vertices = [(r - r0, c - c0) for r, c in cycle]
    for r, c in local_vertices:
        inp[r, c] = shape_color

    for vidx, (dr, dc), shadow_color in shadows:
        vr, vc = local_vertices[vidx]
        ar, ac = vr + dr, vc + dc
        if not (0 <= ar < h and 0 <= ac < w):
            return None
        if inp[ar, ac] != BG:
            return None
        inp[ar, ac] = int(shadow_color)
        if _count_vertex_adjacencies(local_vertices, (ar, ac)) != 1:
            return None
    return inp


def _combo_metrics(inp: np.ndarray) -> dict[str, int]:
    fused = main(inp)
    two_phase = _main_two_phase_xor(inp)
    single_anchor = _main_start_anchor_only(inp)
    shape_wins = _main_interleaved_shape_wins(inp)
    return {
        "d19": int(np.count_nonzero(fused != two_phase)),
        "d14": int(np.count_nonzero(fused != single_anchor)),
        "d6": int(np.count_nonzero(fused != shape_wins)),
        "erase_cells": int(np.count_nonzero((fused == BG) & (shape_wins != BG))),
    }


def _make_primary_patch(shape_color: int, shadow_colors: list[int]) -> tuple[np.ndarray, np.ndarray, dict[str, int]] | None:
    if len(shadow_colors) != PRIMARY_SHADOW_COUNT:
        return None

    for _ in range(PRIMARY_PATCH_ATTEMPTS):
        vertices = _make_primary_vertices()
        cycle = _angle_cycle(vertices)
        candidates = _shadow_candidates(cycle)
        if len(candidates) < PRIMARY_SHADOW_COUNT:
            continue

        order = list(itertools.combinations(candidates, PRIMARY_SHADOW_COUNT))
        np.random.shuffle(order)
        for combo in order:
            used_vertices = {combo_item[0] for combo_item in combo}
            if len(used_vertices) != PRIMARY_SHADOW_COUNT:
                continue

            shadows = [(combo[i][0], combo[i][1], int(shadow_colors[i])) for i in range(PRIMARY_SHADOW_COUNT)]
            inp = _build_input_patch(cycle, shape_color, shadows)
            if inp is None:
                continue

            out = main(inp)
            if np.array_equal(inp, out):
                continue
            if len(colors_present(inp, exclude=[BG])) != 1 + PRIMARY_SHADOW_COUNT:
                continue

            metrics = _combo_metrics(inp)
            if metrics["d19"] < MIN_D19_DELTA:
                continue
            if metrics["d14"] < MIN_D14_DELTA:
                continue
            if metrics["d6"] < MIN_D6_DELTA:
                continue
            if metrics["erase_cells"] < MIN_ERASE_CELLS:
                continue
            return inp, out, metrics
    return None


def _make_plain_patch(shape_color: int) -> tuple[np.ndarray, np.ndarray] | None:
    for _ in range(PLAIN_PATCH_ATTEMPTS):
        vertices = _make_plain_vertices()
        inp = _build_input_patch(vertices, shape_color, [])
        if inp is None:
            continue
        out = main(inp)
        if np.array_equal(inp, out):
            continue
        return inp, out
    return None


def _sample_distractor_coords(h: int, w: int, count: int) -> list[tuple[int, int]] | None:
    coords: list[tuple[int, int]] = []
    seen: set[tuple[int, int]] = set()
    while len(coords) < count:
        cand = (int(np.random.randint(h)), int(np.random.randint(w)))
        if cand in seen:
            continue
        coords.append(cand)
        seen.add(cand)
    for i in range(len(coords)):
        for j in range(i + 1, len(coords)):
            if _aligned(coords[j][0] - coords[i][0], coords[j][1] - coords[i][1]):
                return None
    return coords


def _make_distractor_patch(color: int) -> tuple[np.ndarray, np.ndarray]:
    count = int(np.random.randint(*DISTRACTOR_COUNT_RANGE))
    if count == 1:
        patch = np.full((1, 1), color, dtype=int)
        return patch.copy(), patch.copy()
    for _ in range(DISTRACTOR_RETRIES):
        h = int(np.random.randint(*DISTRACTOR_BBOX_RANGE))
        w = int(np.random.randint(*DISTRACTOR_BBOX_RANGE))
        coords = _sample_distractor_coords(h, w, count)
        if coords is None:
            continue
        patch = np.full((h, w), BG, dtype=int)
        for r, c in coords:
            patch[r, c] = color
        return patch.copy(), patch.copy()
    patch = np.full((1, 1), color, dtype=int)
    return patch.copy(), patch.copy()


def _sample_canvas_size() -> tuple[int, int]:
    h = int(np.random.randint(*CANVAS_H_RANGE))
    w = int(np.clip(h + np.random.randint(*CANVAS_W_JITTER), *CANVAS_W_CLIP))
    return h, w


def _try_place(
    occ: np.ndarray,
    inp: np.ndarray,
    out: np.ndarray,
    patch_in: np.ndarray,
    patch_out: np.ndarray,
) -> bool:
    footprint = np.full(patch_out.shape, 9, dtype=int)
    try:
        loc = random_free_location(
            occ,
            footprint,
            transparent=BG,
            padding=PLACEMENT_PADDING,
            padding_connectivity=PLACEMENT_PAD_CONNECTIVITY,
        )
    except ValueError:
        return False
    paste(occ, footprint, loc[0], loc[1], transparent=BG)
    paste(inp, patch_in, loc[0], loc[1], transparent=BG)
    paste(out, patch_out, loc[0], loc[1], transparent=BG)
    return True


def _gen_attempt() -> np.ndarray | None:
    h, w = _sample_canvas_size()
    occ = np.full((h, w), BG, dtype=int)
    inp = np.full((h, w), BG, dtype=int)
    out = np.full((h, w), BG, dtype=int)

    has_distractor = bool(np.random.rand() < DISTRACTOR_PROB)
    shape_count = int(np.random.randint(*SHAPE_COUNT_RANGE))
    total_colors = shape_count + PRIMARY_SHADOW_COUNT + int(has_distractor)
    if total_colors > len(COLORS):
        return None

    palette = list(np.random.permutation(COLORS)[:total_colors])
    shape_colors = [int(v) for v in palette[:shape_count]]
    shadow_colors = [int(v) for v in palette[shape_count:shape_count + PRIMARY_SHADOW_COUNT]]
    distractor_color = int(palette[-1]) if has_distractor else None

    primary = _make_primary_patch(shape_colors[0], shadow_colors)
    if primary is None:
        return None
    patch_in, patch_out, primary_metrics = primary
    if not _try_place(occ, inp, out, patch_in, patch_out):
        return None

    for shape_color in shape_colors[1:]:
        placed = False
        for _ in range(SHAPE_PLACEMENT_RETRIES):
            patch = _make_plain_patch(shape_color)
            if patch is None:
                continue
            if _try_place(occ, inp, out, patch[0], patch[1]):
                placed = True
                break
        if not placed:
            return None

    if has_distractor and distractor_color is not None:
        if not _try_place(occ, inp, out, *_make_distractor_patch(distractor_color)):
            return None

    final_out = main(inp)
    if not np.array_equal(final_out, out):
        return None
    if get_bg(inp) != BG or get_bg(out) != BG:
        return None
    if len(colors_present(inp, exclude=[BG])) != total_colors:
        return None

    final_metrics = _combo_metrics(inp)
    if final_metrics["d19"] < MIN_D19_DELTA:
        return None
    if final_metrics["d14"] < MIN_D14_DELTA:
        return None
    if final_metrics["d6"] < MIN_D6_DELTA:
        return None
    if final_metrics["erase_cells"] < MIN_ERASE_CELLS:
        return None
    if any(final_metrics[k] < primary_metrics[k] for k in ("d19", "d14", "d6")):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GENERATE_RETRIES):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError(f"generate_input failed after {GENERATE_RETRIES} attempts")
