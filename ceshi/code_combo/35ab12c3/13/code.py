# Combo variant of 35ab12c3 (index 13)
# combo_spec: D1/0+D11/2
# Name: 互补色配对+仅使用2种颜色

from __future__ import annotations

import math

import numpy as np

from arc_utils import bbox_coords, colors_present, get_bg as _bg_color, paste, random_free_location

# -----------------------------------------------------------------------------
# Design analysis for the fused variant
#
# a) Concrete behavior and trace
#    - Original task: each non-bg color was one independent shape-group, so the
#      solver could iterate by color and connect same-colored vertices.
#    - Fused task: the whole grid now uses exactly two non-bg colors total, and
#      every real shape mixes both of them. A shape is solved by merging both
#      colors inside one local spatial island, then connecting them as a single
#      cycle/polygon. The adjacent extra pixel is still copied as a parallel
#      shadow, but it must also use one of those same two colors.
#    - Small trace:
#      Original island: blue vertices only -> connect blue-to-blue.
#      Fused island: red/blue/red/blue vertices around one rectangle plus a
#      blue shadow anchor next to the lexicographically smallest red vertex.
#      New solver must first isolate the island spatially, then decide which one
#      adjacent pixel is the anchor, then connect all remaining red+blue points
#      together. Running the old color-by-color solver would split red and blue
#      into unrelated fragments and would miss the intended rectangle.
#
# b) Complexity calibration
#    - Only two colors are available, so colors can no longer separate shapes.
#      To avoid ambiguity, every shape is generated as its own compact island and
#      different islands are separated by a Chebyshev gap larger than the solver
#      grouping radius.
#    - The complementary rule is still visible because every island contains both
#      colors on alternating vertices; humans can see that one mixed-color object
#      becomes one connected outline.
#    - We keep the palette hypothesis space small: the two present colors are the
#      only complementary pair in that grid.
#
# c) Hidden-assumption audit of the original solver
#    - `for col in colors:` assumed one color == one shape. Broken here; solver
#      must group by space first, not by color.
#    - `_clear_in_original(... allow={col})` assumed only one active vertex color
#      could lie on a legal segment. Broken here; legal interiors may contain
#      either of the two paired colors, so both must be allowed.
#    - "different-colored neighbor means shadow anchor" is no longer sufficient,
#      because partner-color vertices are legitimate shape vertices. We replace
#      that with a stronger structural cue: within each island there is exactly
#      one 8-neighbor pair, and exactly one interpretation of that pair yields a
#      valid alternating-color cycle.
#    - The original generator allowed distractors. With only two colors total,
#      distractors would be visually confusable with real vertices, so they are
#      removed in this fusion.
#
# d) Placement legality
#    - Generator places solved patches using their full output footprint plus
#      `PLACEMENT_PADDING = GROUP_LINK_RADIUS`, so islands remain disconnected in
#      the sparse-component graph used by the solver.
#    - All shapes are compact even-vertex polygons with edge lengths <= 4, so one
#      island is easy to recover while still leaving room for 2-3 shapes.
#
# e) Visual observability
#    - Across examples, a human sees only two colors, yet each local red/blue
#      constellation becomes one unified outline instead of two same-color ones.
#      The nearby extra pixel creates a parallel copy in the opposite color.
#    - No hidden use of numeric color ids is needed: the rule is visible from the
#      mixed-color local geometry itself.
#
# f) Visual cleanliness
#    - We keep only compact rectangles / diamonds / hexagons, one shadow anchor
#      per island, and no distractors. This avoids same-color clutter and keeps
#      each mixed-color object readable.
#
# g) Inferability of the color change
#    - The output outline color is the anchor-adjacent start vertex color; the
#      shadow uses the opposite color. Because input vertices alternate around
#      the cycle, the "cross-color pairing" rule is directly supported inside
#      every island.
#
# 9) New solver assumptions and generator guarantees
#    - Assumption 1: every active island contains exactly one shadow anchor.
#      Guarantee: generator adds exactly one anchor pixel per shape.
#    - Assumption 2: every shape has an even number of vertices whose colors
#      alternate around the angle-sorted cycle.
#      Guarantee: generator uses only 4- or 6-vertex convex shapes and colors
#      them alternately after rotating the cycle to the chosen start vertex.
#    - Assumption 3: the anchor uses the opposite color of the start vertex and
#      is 8-adjacent to exactly that one vertex.
#      Guarantee: shadow offset is sampled from positions adjacent to exactly one
#      vertex and we color the anchor with the opposite color.
#    - Assumption 4: different shapes are separated in the sparse graph built
#      with `GROUP_LINK_RADIUS`.
#      Guarantee: solved patches are pasted with at least that much padding.
# -----------------------------------------------------------------------------

DIR8 = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]

BG = 0
COLORS = [1, 2, 3, 4, 6, 7, 8]

GROUP_LINK_RADIUS = 4
MAX_EDGE_STEP = 4

# ---- CFG: canvas / placement ----
CANVAS_H_RANGE = (17, 23)
CANVAS_W_JITTER = (-2, 3)
CANVAS_W_CLIP = (17, 22)
SHAPE_COUNT_RANGE = (2, 4)
PLACEMENT_PADDING = GROUP_LINK_RADIUS
PLACEMENT_PAD_CONNECTIVITY = 8
GENERATE_RETRIES = 500
SHAPE_PLACEMENT_RETRIES = 100
MAX_PATCH_SIDE = 10

# ---- CFG: geometry ----
RECT_H_RANGE = (2, 5)
RECT_W_RANGE = (2, 5)
DIAMOND_ARM_RANGE = (2, 4)
HEXAGON_SLANT_RANGE = (2, 4)
HEXAGON_WALL_RANGE = (2, 5)
SHAPE_WEIGHTS = np.array([0.40, 0.30, 0.30], dtype=float)


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


def _step(a: tuple[int, int], b: tuple[int, int]) -> tuple[int, int]:
    return _step_dir(a[0], a[1], b[0], b[1])


def _segment_cells(r0: int, c0: int, r1: int, c1: int):
    sr, sc = _step_dir(r0, c0, r1, c1)
    steps = max(abs(r1 - r0), abs(c1 - c0))
    for k in range(steps + 1):
        yield r0 + sr * k, c0 + sc * k


def _gen_segment_cells(a: tuple[int, int], b: tuple[int, int]) -> list[tuple[int, int]]:
    return list(_segment_cells(a[0], a[1], b[0], b[1]))


def _clear_in_original(
    g: np.ndarray,
    r0: int,
    c0: int,
    r1: int,
    c1: int,
    bg: int,
    allow: set[int],
) -> bool:
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


def _draw_line_bg_only(out: np.ndarray, r0: int, c0: int, r1: int, c1: int, col: int, bg: int) -> None:
    for rr, cc in _segment_cells(r0, c0, r1, c1):
        if out[rr, cc] == bg:
            out[rr, cc] = col


def _angle_cycle(points_rc: list[tuple[int, int]]) -> list[tuple[int, int]]:
    cr = sum(r for r, _ in points_rc) / len(points_rc)
    cc = sum(c for _, c in points_rc) / len(points_rc)

    def key(p: tuple[int, int]):
        r, c = p
        ang = math.atan2(r - cr, c - cc)
        rad = (r - cr) ** 2 + (c - cc) ** 2
        return (ang, rad, r, c)

    return sorted(points_rc, key=key)


def _rotate_cycle(cycle: list[tuple[int, int]], start: tuple[int, int]) -> list[tuple[int, int]]:
    idx = cycle.index(start)
    return cycle[idx:] + cycle[:idx]


def _shape_edges(vertices: list[tuple[int, int]]) -> list[tuple[tuple[int, int], tuple[int, int]]]:
    return [(vertices[i], vertices[(i + 1) % len(vertices)]) for i in range(len(vertices))]


def _shape_cells(vertices: list[tuple[int, int]]) -> set[tuple[int, int]]:
    cells: set[tuple[int, int]] = set()
    for a, b in _shape_edges(vertices):
        cells.update(_gen_segment_cells(a, b))
    return cells


def _edge_dirs(vertices: list[tuple[int, int]]) -> list[tuple[int, int]]:
    return [_step(a, b) for a, b in _shape_edges(vertices)]


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


def _build_rectangle() -> list[tuple[int, int]]:
    h = int(np.random.randint(*RECT_H_RANGE))
    w = int(np.random.randint(*RECT_W_RANGE))
    return _transform_coords([(0, 0), (0, w), (h, w), (h, 0)])


def _build_diamond() -> list[tuple[int, int]]:
    arm = int(np.random.randint(*DIAMOND_ARM_RANGE))
    return _transform_coords([(0, arm), (arm, 2 * arm), (2 * arm, arm), (arm, 0)])


def _build_hexagon() -> list[tuple[int, int]]:
    slant = int(np.random.randint(*HEXAGON_SLANT_RANGE))
    wall = int(np.random.randint(*HEXAGON_WALL_RANGE))
    mid = slant + wall
    bot = 2 * slant + wall
    return _transform_coords([(0, slant), (slant, 0), (mid, 0), (bot, slant), (mid, 2 * slant), (slant, 2 * slant)])


_SHAPE_BUILDERS = [_build_rectangle, _build_diamond, _build_hexagon]


def _make_vertices() -> list[tuple[int, int]]:
    weights = SHAPE_WEIGHTS / SHAPE_WEIGHTS.sum()
    idx = int(np.random.choice(len(_SHAPE_BUILDERS), p=weights))
    return _angle_cycle(_SHAPE_BUILDERS[idx]())


def _sparse_components(points_rc: list[tuple[int, int]], radius: int) -> list[list[tuple[int, int]]]:
    if not points_rc:
        return []
    comps: list[list[tuple[int, int]]] = []
    used = [False] * len(points_rc)
    for start in range(len(points_rc)):
        if used[start]:
            continue
        used[start] = True
        stack = [start]
        comp: list[tuple[int, int]] = []
        while stack:
            i = stack.pop()
            pr, pc = points_rc[i]
            comp.append(points_rc[i])
            for j, (qr, qc) in enumerate(points_rc):
                if used[j]:
                    continue
                if max(abs(pr - qr), abs(pc - qc)) <= radius:
                    used[j] = True
                    stack.append(j)
        comps.append(sorted(comp))
    return comps


def _adjacent_pairs(points_rc: list[tuple[int, int]]) -> list[tuple[tuple[int, int], tuple[int, int]]]:
    pairs: list[tuple[tuple[int, int], tuple[int, int]]] = []
    for i, a in enumerate(points_rc):
        for b in points_rc[i + 1 :]:
            if max(abs(a[0] - b[0]), abs(a[1] - b[1])) == 1:
                pairs.append((a, b))
    return pairs


def _colors_alternate(cycle: list[tuple[int, int]], g: np.ndarray) -> bool:
    return all(int(g[cycle[i]]) != int(g[cycle[(i + 1) % len(cycle)]]) for i in range(len(cycle)))


def _interpret_island(
    g: np.ndarray,
    island: list[tuple[int, int]],
    bg: int,
) -> dict[str, object] | None:
    island_colors = {int(g[r, c]) for r, c in island}
    if len(island) < 5 or len(island_colors) != 2:
        return None

    adj_pairs = _adjacent_pairs(island)
    if len(adj_pairs) != 1:
        return None

    candidates: list[dict[str, object]] = []
    for a, b in adj_pairs:
        for anchor, start in ((a, b), (b, a)):
            vertex_pts = [pt for pt in island if pt != anchor]
            if len(vertex_pts) < 4 or len(vertex_pts) % 2 != 0:
                continue
            if sum(1 for r, c in vertex_pts if max(abs(anchor[0] - r), abs(anchor[1] - c)) == 1) != 1:
                continue
            if start not in vertex_pts:
                continue

            cycle = _rotate_cycle(_angle_cycle(vertex_pts), start)
            if not _colors_alternate(cycle, g):
                continue
            if int(g[anchor]) == int(g[start]):
                continue
            if int(g[anchor]) != int(g[cycle[1]]) or int(g[anchor]) != int(g[cycle[-1]]):
                continue

            drn = anchor[0] - start[0]
            dcn = anchor[1] - start[1]
            if (drn, dcn) == (0, 0):
                continue

            edges = _shape_edges(cycle)
            drawable: list[tuple[tuple[int, int], tuple[int, int]]] = []
            ok = True
            for (r0, c0), (r1, c1) in edges:
                if not _aligned_8(r0, c0, r1, c1):
                    ok = False
                    break
                if not _clear_in_original(g, r0, c0, r1, c1, bg, allow=island_colors):
                    ok = False
                    break
                drs, dcs = _step_dir(r0, c0, r1, c1)
                if drs * dcn - dcs * drn == 0:
                    ok = False
                    break
                drawable.append(((r0, c0), (r1, c1)))
            if not ok:
                continue

            candidates.append(
                {
                    "anchor": anchor,
                    "start": start,
                    "cycle": cycle,
                    "edges": drawable,
                    "offset": (drn, dcn),
                    "anchor_color": int(g[anchor]),
                    "main_color": int(g[start]),
                }
            )

    if len(candidates) != 1:
        return None
    return candidates[0]


def _choose_shadow(
    vertices: list[tuple[int, int]],
    shape_cells: set[tuple[int, int]],
    start: tuple[int, int],
) -> tuple[tuple[int, int], tuple[int, int]] | None:
    directions = _edge_dirs(vertices)
    candidates: list[tuple[int, int]] = []
    for dr, dc in DIR8:
        if any(ds_r * dc - ds_c * dr == 0 for ds_r, ds_c in directions):
            continue
        pos = (start[0] + dr, start[1] + dc)
        if pos in shape_cells:
            continue
        if sum(1 for vr, vc in vertices if max(abs(pos[0] - vr), abs(pos[1] - vc)) == 1) != 1:
            continue
        candidates.append((dr, dc))
    if not candidates:
        return None
    return start, candidates[int(np.random.randint(len(candidates)))]


def _patch_bbox(
    shape_cells: set[tuple[int, int]],
    vertices: list[tuple[int, int]],
    shadow: tuple[tuple[int, int], tuple[int, int]],
) -> tuple[int, int, int, int]:
    (ar, ac), (dr, dc) = shadow
    coords = list(shape_cells) + vertices
    coords.append((ar + dr, ac + dc))
    coords.extend((r + dr, c + dc) for r, c in shape_cells)
    r0, c0, r1, c1 = bbox_coords(np.array(coords, dtype=int))
    return int(r0), int(c0), int(r1), int(c1)


def _single_sparse_island(inp: np.ndarray) -> bool:
    pts = [tuple(map(int, p)) for p in np.argwhere(inp != BG)]
    return len(_sparse_components(pts, GROUP_LINK_RADIUS)) == 1


def _sample_palette() -> tuple[int, int]:
    a, b = np.random.choice(COLORS, size=2, replace=False).tolist()
    return int(a), int(b)


def _try_place(inp: np.ndarray, out: np.ndarray, patch_in: np.ndarray, patch_out: np.ndarray) -> bool:
    try:
        loc = random_free_location(
            out,
            patch_out,
            transparent=BG,
            padding=PLACEMENT_PADDING,
            padding_connectivity=PLACEMENT_PAD_CONNECTIVITY,
        )
    except ValueError:
        return False
    if loc is None:
        return False
    paste(out, patch_out, loc[0], loc[1], transparent=BG)
    paste(inp, patch_in, loc[0], loc[1], transparent=BG)
    return True


def _sample_canvas_size() -> tuple[int, int]:
    h = int(np.random.randint(*CANVAS_H_RANGE))
    w = int(np.clip(h + np.random.randint(*CANVAS_W_JITTER), *CANVAS_W_CLIP))
    return h, w


def _make_shape_patch(color_a: int, color_b: int) -> tuple[np.ndarray, np.ndarray] | None:
    cycle = _make_vertices()
    if len(cycle) < 4 or len(cycle) % 2 != 0:
        return None

    for a, b in _shape_edges(cycle):
        if max(abs(a[0] - b[0]), abs(a[1] - b[1])) > MAX_EDGE_STEP:
            return None

    start = min(cycle)
    shape_cells = _shape_cells(cycle)
    shadow = _choose_shadow(cycle, shape_cells, start)
    if shadow is None:
        return None

    r0, c0, r1, c1 = _patch_bbox(shape_cells, cycle, shadow)
    h, w = r1 - r0 + 1, c1 - c0 + 1
    if max(h, w) > MAX_PATCH_SIDE:
        return None

    inp = np.full((h, w), BG, dtype=int)
    rotated = _rotate_cycle(cycle, start)
    start_color, partner_color = (color_a, color_b) if np.random.rand() < 0.5 else (color_b, color_a)
    shifted_vertices = [(r - r0, c - c0) for r, c in rotated]
    for i, (r, c) in enumerate(shifted_vertices):
        inp[r, c] = start_color if i % 2 == 0 else partner_color

    (ar, ac), offset = shadow
    shadow_pos = (ar - r0 + offset[0], ac - c0 + offset[1])
    if not (0 <= shadow_pos[0] < h and 0 <= shadow_pos[1] < w):
        return None
    if inp[shadow_pos] != BG:
        return None
    inp[shadow_pos] = partner_color

    if len(_adjacent_pairs([tuple(map(int, p)) for p in np.argwhere(inp != BG)])) != 1:
        return None
    if not _single_sparse_island(inp):
        return None

    interpreted = _interpret_island(inp, [tuple(map(int, p)) for p in np.argwhere(inp != BG)], BG)
    if interpreted is None:
        return None
    if tuple(interpreted["anchor"]) != shadow_pos:
        return None
    if tuple(interpreted["start"]) != shifted_vertices[0]:
        return None

    out = main(inp)
    if np.array_equal(inp, out):
        return None
    if set(colors_present(out, exclude=[BG])) != {start_color, partner_color}:
        return None
    if np.count_nonzero(out == partner_color) <= np.count_nonzero(inp == partner_color):
        return None
    return inp, out


def _gen_attempt() -> np.ndarray | None:
    h, w = _sample_canvas_size()
    inp = np.full((h, w), BG, dtype=int)
    out = np.full((h, w), BG, dtype=int)

    shape_count = int(np.random.randint(*SHAPE_COUNT_RANGE))
    color_a, color_b = _sample_palette()

    for _ in range(shape_count):
        placed = False
        for _ in range(SHAPE_PLACEMENT_RETRIES):
            patch = _make_shape_patch(color_a, color_b)
            if patch is None:
                continue
            if _try_place(inp, out, *patch):
                placed = True
                break
        if not placed:
            return None

    pts = [tuple(map(int, p)) for p in np.argwhere(inp != BG)]
    if set(colors_present(inp, exclude=[BG])) != {color_a, color_b}:
        return None
    if len(_sparse_components(pts, GROUP_LINK_RADIUS)) != shape_count:
        return None
    if _bg_color(inp) != BG or _bg_color(out) != BG:
        return None
    if not np.array_equal(main(inp), out):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GENERATE_RETRIES):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError(f"generate_input failed after {GENERATE_RETRIES} attempts")


def main(grid: np.ndarray) -> np.ndarray:
    """
    Connect each compact mixed-color island into one outline and copy its
    adjacent marker as a parallel shadow, using only the two colors already
    present in the grid.

    Input structure:
    The input contains 2-3 separated islands. Each island has one even-vertex
    convex shape whose vertices alternate between the only two non-background
    colors in the whole grid, plus exactly one extra pixel 8-adjacent to the
    lexicographically smallest vertex.

    Rule:
    First split the input into sparse spatial islands instead of grouping by
    color. Inside each island, find the unique adjacent-pixel pair; exactly one
    interpretation of that pair leaves an alternating-color cycle of vertices.
    Connect that cycle with H/V/45-degree edges using the start-vertex color,
    then copy every edge along the anchor offset using the anchor color, writing
    only onto background cells.

    Key insight:
    The two colors are complementary members of the same shape, not separate
    groups. The only reliable way to solve the task is to merge both colors
    locally, recover which adjacent pixel is the shadow anchor, and then apply
    the original connect-and-shift geometry inside each island.
    """
    g = np.asarray(grid, dtype=int)
    bg = int(_bg_color(g))
    out = g.copy()

    points = [tuple(map(int, p)) for p in np.argwhere(g != bg)]
    if not points:
        return out

    islands = _sparse_components(points, GROUP_LINK_RADIUS)
    for island in islands:
        solved = _interpret_island(g, island, bg)
        if solved is None:
            continue

        drawable_edges = list(solved["edges"])
        drn, dcn = tuple(solved["offset"])
        main_color = int(solved["main_color"])
        anchor_color = int(solved["anchor_color"])

        for (r0, c0), (r1, c1) in drawable_edges:
            _draw_line_bg_only(out, r0, c0, r1, c1, main_color, bg)

        for (r0, c0), (r1, c1) in drawable_edges:
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
            _draw_line_bg_only(out, rr0, cc0, rr1, cc1, anchor_color, bg)

    return out
