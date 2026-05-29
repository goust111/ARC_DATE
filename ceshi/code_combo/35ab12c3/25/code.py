# Combo variant of 35ab12c3 (index 25)
# combo_spec: D7/1+D11/0
# Name: 颜色复用空间分离+色块连通分量源
#
# Design analysis for the combo variant
# -------------------------------------
# a) Concrete behavior and trace
# - Original task: each shape color is a set of isolated same-color pixels. The
#   solver globally merges all pixels of one color, orders them by angle, draws
#   aligned edges, then copies a different-color neighbor of the chosen start
#   vertex along parallel offset lines.
# - This combo changes two stages at once:
#   1. D7: a "vertex" is now a small connected color block rather than a single
#      pixel. The solver must segment same-color components and collapse each
#      eligible component to its centroid.
#   2. D11: the same main color may appear in several different shapes, so color
#      alone no longer determines grouping. The solver must split one color into
#      local groups after centroid extraction.
# - Trace example:
#   input contains color 6 with three `+` blocks centered at (4,4),(4,12),(12,12),
#   plus another far-away pair of `+` blocks of the same color centered at
#   (14,3),(18,3). The old solver would treat all 25 color-6 pixels as one big
#   vertex set and would mix the two shapes together. The combo solver first finds
#   five-cell components, reduces them to centers, builds aligned-clear edges
#   between centers, splits them into {triangle centers} and {vertical pair}, then
#   expands each group independently. If a shadow seed sits at (3,3) next to the
#   centroid (4,4), only the triangle gets that offset shadow.
#
# b) Complexity calibration
# - Vertex components use only two symmetric five-cell templates: `+` and `X`.
#   They are visibly multi-pixel, always 8-connected, and always have integer
#   centroids, avoiding centroid-rounding ambiguity.
# - Color reuse is forced but limited: 2-4 shapes are generated, and at least one
#   color is reused by multiple shapes. Shadow and distractor colors stay distinct
#   from the set of unique shape colors to keep the puzzle readable.
# - Same-color reused shapes are placed far apart and are additionally filtered so
#   their centroid sets do not create cross-group aligned-clear links, preventing
#   accidental merging.
#
# c) Implicit assumption review
# - `np.argwhere(g == col)` as "the vertices of col" no longer holds. One five-cell
#   block would explode into five fake vertices, so the solver must use
#   connected-component extraction first.
# - "One color = one shape" also breaks. After centroid extraction, the solver
#   must build a graph on the centroids and take connected components of that
#   graph to recover local same-color shapes.
# - Path-clear tests cannot blindly allow every same-color cell, because another
#   same-color block from a different local shape could sit on the ray. The combo
#   solver therefore allows only the endpoint blocks when testing whether two
#   centroids belong to the same local group or whether an edge is drawable.
# - Start-vertex shadow detection used to inspect a single colored pixel. It now
#   inspects the centroid neighborhood of the chosen vertex block. The generator
#   enforces that the shadow seed is placed in an empty 8-neighbor of that
#   centroid and is adjacent to exactly one centroid.
#
# d) Placement legality
# - Shapes are built on an abstract step-4 lattice and then rendered as five-cell
#   blocks, which keeps different vertex blocks of one shape disconnected under
#   8-connectivity.
# - Candidate canvas placements are filtered against the final output patch, so
#   main lines and shadow lines do not overlap unrelated objects.
# - For reused colors, placement also checks centroid-level non-alignment against
#   already placed same-color groups, matching the solver's grouping assumptions.
#
# e) Visual observability
# - The D7 change is directly visible because vertices are clearly 3x3-ish blobs
#   instead of dots.
# - The D11 change is directly visible because the same main color appears in
#   multiple separated local groups, and each local group expands independently in
#   the output.
# - Across 2-4 training examples, a human can infer that lines go through blob
#   centers and that same-color blobs must first be partitioned spatially.
#
# f) Visual cleanliness
# - The layout keeps at least one cell of background around each output object and
#   avoids sharing shadow colors with shape colors.
# - Only one shadow seed is attached to a given shape, so the output remains a
#   sparse collection of shapes with a single offset copy when present.
#
# g) Non-degeneracy target and generator guarantees
# - Every real shape vertex is a five-cell component, so the combo inputs are not
#   solvable by the original per-pixel interpretation.
# - At least one color is reused across multiple shape groups, so the combo inputs
#   are not equivalent to the unique-color original task.
# - The generator validates each sample with the combo solver and also exposes
#   statistics helpers so tests can confirm that both D7 and D11 behaviors are
#   present in generated data.

import math

import numpy as np

from arc_utils import bbox_coords, colors_present, draw_line, find_components, get_bg, neighbors8, paste

DIR8 = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
BG = 0
COLORS = [1, 2, 3, 4, 6, 7, 8]
VERTEX_MIN_AREA = 4


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


def _clear_on_grid(
    g: np.ndarray,
    r0: int,
    c0: int,
    r1: int,
    c1: int,
    bg: int,
    allow_coords: set[tuple[int, int]],
) -> bool:
    first = True
    for rr, cc in _segment_cells(r0, c0, r1, c1):
        if first:
            first = False
            continue
        if rr == r1 and cc == c1:
            break
        if int(g[rr, cc]) != bg and (rr, cc) not in allow_coords:
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


def _component_centroid(coords: np.ndarray) -> tuple[int, int] | None:
    rr = float(coords[:, 0].mean())
    cc = float(coords[:, 1].mean())
    r = int(round(rr))
    c = int(round(cc))
    if abs(rr - r) > 1e-9 or abs(cc - c) > 1e-9:
        return None
    return r, c


def _vertex_components(g: np.ndarray, col: int, bg: int) -> list[dict[str, object]]:
    comps = find_components(
        g,
        bg=bg,
        connectivity=8,
        monochromatic=True,
        colors=[col],
        min_area=VERTEX_MIN_AREA,
        return_format="dict",
    )
    out: list[dict[str, object]] = []
    for comp in comps:
        coords = np.asarray(comp["coords"], dtype=int)
        center = _component_centroid(coords)
        if center is None:
            continue
        out.append(
            {
                "center": center,
                "coords": {tuple(map(int, p)) for p in coords},
                "area": int(coords.shape[0]),
            }
        )
    out.sort(key=lambda comp: tuple(comp["center"]))
    return out


def _shape_groups_for_color(g: np.ndarray, col: int, bg: int) -> list[list[dict[str, object]]]:
    comps = _vertex_components(g, col, bg)
    n = len(comps)
    if n == 0:
        return []
    if n == 1:
        return [[comps[0]]]

    adj = [set() for _ in range(n)]
    for i in range(n):
        r0, c0 = comps[i]["center"]
        for j in range(i + 1, n):
            r1, c1 = comps[j]["center"]
            if not _aligned_8(r0, c0, r1, c1):
                continue
            allow = set(comps[i]["coords"]) | set(comps[j]["coords"])
            if not _clear_on_grid(g, r0, c0, r1, c1, bg, allow):
                continue
            adj[i].add(j)
            adj[j].add(i)

    groups: list[list[dict[str, object]]] = []
    seen = [False] * n
    for i in range(n):
        if seen[i]:
            continue
        stack = [i]
        seen[i] = True
        comp_idx: list[int] = []
        while stack:
            cur = stack.pop()
            comp_idx.append(cur)
            for nxt in adj[cur]:
                if not seen[nxt]:
                    seen[nxt] = True
                    stack.append(nxt)
        grp = [comps[k] for k in comp_idx]
        grp.sort(key=lambda comp: tuple(comp["center"]))
        groups.append(grp)

    groups.sort(key=lambda grp: tuple(grp[0]["center"]))
    return groups


def fusion_stats(grid: np.ndarray) -> dict[str, int]:
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    per_color_group_counts: dict[int, int] = {}
    vertex_component_count = 0
    for col in sorted(set(map(int, np.unique(g))) - {bg}):
        groups = _shape_groups_for_color(g, int(col), bg)
        shape_groups = [grp for grp in groups if len(grp) >= 2]
        if not shape_groups:
            continue
        per_color_group_counts[int(col)] = len(shape_groups)
        vertex_component_count += sum(len(grp) for grp in shape_groups)
    shape_group_count = sum(per_color_group_counts.values())
    reused_color_count = sum(1 for count in per_color_group_counts.values() if count >= 2)
    reused_group_surplus = sum(count - 1 for count in per_color_group_counts.values() if count >= 2)
    return {
        "shape_group_count": int(shape_group_count),
        "vertex_component_count": int(vertex_component_count),
        "reused_color_count": int(reused_color_count),
        "reused_group_surplus": int(reused_group_surplus),
    }


def _global_groups_for_color(g: np.ndarray, col: int, bg: int) -> list[list[dict[str, object]]]:
    comps = _vertex_components(g, col, bg)
    return [comps] if comps else []


def _solve_with_group_fn(grid: np.ndarray, group_fn) -> np.ndarray:
    """
    Overview:
    Expand each same-color shape by connecting its vertex-block centroids and copy
    any different-color neighbor of the chosen start centroid along parallel lines.

    Input structure:
    Background is 0. Real shape vertices are small 8-connected same-color blocks
    with integer centroids. A main color may be reused by multiple spatially
    separated shapes. Optional shadow seeds are single different-color pixels in
    the 8-neighborhood of one centroid. Optional distractors stay unchanged.

    Rule:
    For each non-background color, find its large connected components, convert
    them to centroids, split the centroids into local same-color groups using
    aligned-and-clear centroid connectivity, then process each local group
    independently. Order a group's centroids by angle, pick the start centroid
    using the presence of a different-color neighbor, draw each aligned clear edge
    between consecutive centroids, and draw parallel offset copies for the carried
    shadow seed color(s).

    Key insight:
    Color is no longer enough. The solver must first compress each color block to
    one semantic vertex and only then recover local same-color shapes by spatial
    connectivity between those centroids.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = get_bg(g)

    out = g.copy()

    colors = sorted(set(map(int, np.unique(g))) - {bg})
    for col in colors:
        for group in group_fn(g, int(col), bg):
            if len(group) < 2:
                continue

            centers = [tuple(comp["center"]) for comp in group]
            cycle = _angle_cycle(centers)
            center_to_coords = {tuple(comp["center"]): set(comp["coords"]) for comp in group}

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

            cycle = cycle[start:] + cycle[:start]

            carried: list[tuple[int, int, int]] = []
            sr0, sc0 = cycle[0]
            for drn, dcn in DIR8:
                rr = sr0 + drn
                cc = sc0 + dcn
                if 0 <= rr < h and 0 <= cc < w:
                    v = int(g[rr, cc])
                    if v != bg and v != col:
                        carried.append((drn, dcn, v))

            if len(cycle) == 2:
                edges = [(cycle[0], cycle[1])]
            else:
                edges = [(cycle[i], cycle[(i + 1) % len(cycle)]) for i in range(len(cycle))]

            drawable_edges: list[tuple[tuple[int, int], tuple[int, int]]] = []
            for (r0, c0), (r1, c1) in edges:
                if not _aligned_8(r0, c0, r1, c1):
                    continue
                allow = center_to_coords[(r0, c0)] | center_to_coords[(r1, c1)]
                if not _clear_on_grid(g, r0, c0, r1, c1, bg, allow):
                    continue
                drawable_edges.append(((r0, c0), (r1, c1)))

            for (r0, c0), (r1, c1) in drawable_edges:
                _draw_line_bg_only(out, r0, c0, r1, c1, int(col), bg)

            for (r0, c0), (r1, c1) in drawable_edges:
                drs, dcs = _step_dir(r0, c0, r1, c1)
                for drn, dcn, ncol in carried:
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
                    if not _clear_on_grid(g, rr0, cc0, rr1, cc1, bg, set()):
                        continue
                    _draw_line_bg_only(out, rr0, cc0, rr1, cc1, ncol, bg)

    return out


def _main_no_split(grid: np.ndarray) -> np.ndarray:
    return _solve_with_group_fn(grid, _global_groups_for_color)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Overview:
    Expand each same-color shape by connecting its vertex-block centroids and copy
    any different-color neighbor of the chosen start centroid along parallel lines.

    Input structure:
    Background is 0. Real shape vertices are small 8-connected same-color blocks
    with integer centroids. A main color may be reused by multiple spatially
    separated shapes. Optional shadow seeds are single different-color pixels in
    the 8-neighborhood of one centroid. Optional distractors stay unchanged.

    Rule:
    For each non-background color, find its large connected components, convert
    them to centroids, split the centroids into local same-color groups using
    aligned-and-clear centroid connectivity, then process each local group
    independently. Order a group's centroids by angle, pick the start centroid
    using the presence of a different-color neighbor, draw each aligned clear edge
    between consecutive centroids, and draw parallel offset copies for the carried
    shadow seed color(s).

    Key insight:
    Color is no longer enough. The solver must first compress each color block to
    one semantic vertex and only then recover local same-color shapes by spatial
    connectivity between those centroids.
    """
    return _solve_with_group_fn(grid, _shape_groups_for_color)


# --------------- generator ---------------

CANVAS_H_RANGE = (15, 23)
CANVAS_W_JITTER = (-3, 4)
CANVAS_W_CLIP = (15, 22)

SHAPE_COUNT_RANGE = (3, 5)
MAX_SHADOWS_CAP = 3
DISTRACTOR_PROB = 0.5
PLACEMENT_PADDING = 1

CENTER_STEP = 4
LINE_VERTICES_RANGE = (2, 6)
LINE_TOTAL_UNITS_RANGE = (4, 6)
LINE_MIN_GAP_UNITS = 1
TRIANGLE_SIDE_UNITS = (1, 3)
RECT_H_UNITS = (1, 3)
RECT_W_UNITS = (1, 3)
DIAMOND_ARM_UNITS = (1, 3)
PENTAGON_ROOF_UNITS = (1, 3)
PENTAGON_WALL_UNITS = (1, 3)
HEXAGON_SLANT_UNITS = (1, 2)
HEXAGON_WALL_UNITS = (1, 3)

SHAPE_WEIGHTS = np.array([0.28, 0.18, 0.18, 0.14, 0.12, 0.10], dtype=float)

DISTRACTOR_COUNT_RANGE = (1, 4)
DISTRACTOR_BBOX_RANGE = (2, 5)
DISTRACTOR_RETRIES = 50

SHAPE_PLACEMENT_RETRIES = 60
GENERATE_RETRIES = 500

BLOCK_TEMPLATES: dict[str, tuple[tuple[int, int], ...]] = {
    "plus": ((0, 0), (-1, 0), (1, 0), (0, -1), (0, 1)),
    "x": ((0, 0), (-1, -1), (-1, 1), (1, -1), (1, 1)),
}


def _aligned(dr: int, dc: int) -> bool:
    return (dr == 0 and dc != 0) or (dc == 0 and dr != 0) or (dr != 0 and abs(dr) == abs(dc))


def _step(a: tuple[int, int], b: tuple[int, int]) -> tuple[int, int]:
    dr = b[0] - a[0]
    dc = b[1] - a[1]
    return (0 if dr == 0 else (1 if dr > 0 else -1), 0 if dc == 0 else (1 if dc > 0 else -1))


def _gen_segment_cells(a: tuple[int, int], b: tuple[int, int]) -> list[tuple[int, int]]:
    drs, dcs = _step(a, b)
    steps = max(abs(b[0] - a[0]), abs(b[1] - a[1]))
    return [(a[0] + drs * k, a[1] + dcs * k) for k in range(steps + 1)]


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


def _scale_centers(coords: list[tuple[int, int]]) -> list[tuple[int, int]]:
    return [(int(r) * CENTER_STEP, int(c) * CENTER_STEP) for r, c in coords]


def _build_line() -> list[tuple[int, int]]:
    n_vertices = int(np.random.randint(*LINE_VERTICES_RANGE))
    total_units = int(np.random.randint(*LINE_TOTAL_UNITS_RANGE))
    if total_units < n_vertices - 1:
        total_units = n_vertices - 1
    remaining = total_units
    pos = [0]
    for i in range(n_vertices - 1):
        steps_left = n_vertices - 2 - i
        gap_hi = remaining - LINE_MIN_GAP_UNITS * steps_left
        gap = int(np.random.randint(LINE_MIN_GAP_UNITS, gap_hi + 1))
        pos.append(pos[-1] + gap)
        remaining -= gap
    return _scale_centers(_transform_coords([(0, x) for x in pos]))


def _build_triangle() -> list[tuple[int, int]]:
    side = int(np.random.randint(*TRIANGLE_SIDE_UNITS))
    return _scale_centers(_transform_coords([(0, 0), (0, side), (side, side)]))


def _build_rectangle() -> list[tuple[int, int]]:
    h = int(np.random.randint(*RECT_H_UNITS))
    w = int(np.random.randint(*RECT_W_UNITS))
    return _scale_centers(_transform_coords([(0, 0), (0, w), (h, w), (h, 0)]))


def _build_diamond() -> list[tuple[int, int]]:
    arm = int(np.random.randint(*DIAMOND_ARM_UNITS))
    return _scale_centers(_transform_coords([(0, arm), (arm, 2 * arm), (2 * arm, arm), (arm, 0)]))


def _build_pentagon() -> list[tuple[int, int]]:
    roof = int(np.random.randint(*PENTAGON_ROOF_UNITS))
    wall = int(np.random.randint(*PENTAGON_WALL_UNITS))
    h = roof + wall
    return _scale_centers(_transform_coords([(0, roof), (roof, 0), (h, 0), (h, 2 * roof), (roof, 2 * roof)]))


def _build_hexagon() -> list[tuple[int, int]]:
    slant = int(np.random.randint(*HEXAGON_SLANT_UNITS))
    wall = int(np.random.randint(*HEXAGON_WALL_UNITS))
    mid = slant + wall
    bot = 2 * slant + wall
    return _scale_centers(_transform_coords([(0, slant), (slant, 0), (mid, 0), (bot, slant), (mid, 2 * slant), (slant, 2 * slant)]))


_SHAPE_BUILDERS = [_build_line, _build_triangle, _build_rectangle, _build_diamond, _build_pentagon, _build_hexagon]


def _make_vertices() -> list[tuple[int, int]]:
    weights = SHAPE_WEIGHTS / SHAPE_WEIGHTS.sum()
    idx = int(np.random.choice(len(_SHAPE_BUILDERS), p=weights))
    return _SHAPE_BUILDERS[idx]()


def _shape_edges(vertices: list[tuple[int, int]]) -> list[tuple[tuple[int, int], tuple[int, int]]]:
    if len(vertices) == 2:
        return [(vertices[0], vertices[1])]
    return [(vertices[i], vertices[(i + 1) % len(vertices)]) for i in range(len(vertices))]


def _edge_dirs(vertices: list[tuple[int, int]]) -> list[tuple[int, int]]:
    return [_step(a, b) for a, b in _shape_edges(vertices)]


def _apply_template(patch: np.ndarray, center: tuple[int, int], color: int, template: tuple[tuple[int, int], ...]) -> None:
    for dr, dc in template:
        patch[center[0] + dr, center[1] + dc] = int(color)


def _all_block_cells(vertices: list[tuple[int, int]], template: tuple[tuple[int, int], ...]) -> set[tuple[int, int]]:
    return {(r + dr, c + dc) for r, c in vertices for dr, dc in template}


def _choose_shadow(
    vertices: list[tuple[int, int]],
    template: tuple[tuple[int, int], ...],
) -> tuple[tuple[int, int], tuple[int, int]] | None:
    anchor = min(vertices)
    directions = _edge_dirs(vertices)
    occupied = _all_block_cells(vertices, template)
    template_offsets = set(template)
    candidates: list[tuple[int, int]] = []
    for dr, dc in DIR8:
        if (dr, dc) in template_offsets:
            continue
        if any(ds_r * dc - ds_c * dr == 0 for ds_r, ds_c in directions):
            continue
        pos = (anchor[0] + dr, anchor[1] + dc)
        if pos in occupied:
            continue
        if sum(1 for vr, vc in vertices if max(abs(pos[0] - vr), abs(pos[1] - vc)) == 1) != 1:
            continue
        candidates.append((dr, dc))
    if not candidates:
        return None
    return anchor, candidates[int(np.random.randint(len(candidates)))]


def _patch_bbox_from_input_cells(cells: set[tuple[int, int]]) -> tuple[int, int, int, int]:
    r0, c0, r1, c1 = bbox_coords(np.array(sorted(cells), dtype=int))
    return int(r0), int(c0), int(r1), int(c1)


def _make_shape_patch(shape_color: int, shadow_color: int | None) -> tuple[np.ndarray, np.ndarray, list[tuple[int, int]]] | None:
    vertices = _make_vertices()
    template_names = list(BLOCK_TEMPLATES)
    np.random.shuffle(template_names)

    for template_name in template_names:
        template = BLOCK_TEMPLATES[template_name]
        shadow = None
        if shadow_color is not None:
            shadow = _choose_shadow(vertices, template)
            if shadow is None:
                continue

        input_cells = _all_block_cells(vertices, template)
        if shadow is not None:
            (ar, ac), offset = shadow
            input_cells.add((ar + offset[0], ac + offset[1]))

        r0, c0, r1, c1 = _patch_bbox_from_input_cells(input_cells)
        h, w = r1 - r0 + 1, c1 - c0 + 1
        inp = np.full((h, w), BG, dtype=int)

        shifted_vertices = [(r - r0, c - c0) for r, c in vertices]
        for center in shifted_vertices:
            _apply_template(inp, center, shape_color, template)

        if shadow is not None:
            (ar, ac), offset = shadow
            sr = ar - r0 + offset[0]
            sc = ac - c0 + offset[1]
            if inp[sr, sc] != BG:
                continue
            inp[sr, sc] = int(shadow_color)

            anchor = (ar - r0, ac - c0)
            nb_hits = sum(
                1
                for rr, cc in neighbors8(anchor[0], anchor[1])
                if 0 <= rr < h and 0 <= cc < w and inp[rr, cc] == shadow_color
            )
            if nb_hits != 1:
                continue

        out = main(inp)
        if np.array_equal(inp, out):
            continue

        groups = _shape_groups_for_color(inp, int(shape_color), BG)
        shape_groups = [grp for grp in groups if len(grp) >= 2]
        if len(shape_groups) != 1:
            continue
        centers = sorted(tuple(comp["center"]) for comp in shape_groups[0])
        if centers != sorted(shifted_vertices):
            continue
        return inp, out, shifted_vertices

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
        patch = np.full((1, 1), int(color), dtype=int)
        return patch.copy(), patch.copy()
    for _ in range(DISTRACTOR_RETRIES):
        h = int(np.random.randint(*DISTRACTOR_BBOX_RANGE))
        w = int(np.random.randint(*DISTRACTOR_BBOX_RANGE))
        coords = _sample_distractor_coords(h, w, count)
        if coords is None:
            continue
        patch = np.full((h, w), BG, dtype=int)
        for r, c in coords:
            patch[r, c] = int(color)
        return patch.copy(), patch.copy()
    patch = np.full((1, 1), int(color), dtype=int)
    return patch.copy(), patch.copy()


def _has_reused_shape_color(shape_colors: list[int]) -> bool:
    return len(set(shape_colors)) < len(shape_colors)


def _sample_palette(shape_count: int, has_distractor: bool) -> tuple[list[int], list[int], int | None] | None:
    min_unique_shapes = max(1, (shape_count + 1) // 2)
    max_unique_shapes = shape_count - 1
    unique_shape_count = int(np.random.randint(min_unique_shapes, max_unique_shapes + 1))

    shape_palette = list(map(int, np.random.permutation(COLORS)[:unique_shape_count]))
    shape_colors = shape_palette.copy()
    extra_count = shape_count - unique_shape_count
    if extra_count > 0:
        extras = np.random.choice(shape_palette, size=extra_count, replace=True).tolist()
        shape_colors.extend(int(v) for v in extras)
    np.random.shuffle(shape_colors)

    shadow_pool = [col for col in COLORS if col not in shape_palette]
    max_shadows = min(MAX_SHADOWS_CAP, shape_count, len(shadow_pool) - int(has_distractor))
    if max_shadows < 0:
        return None

    shadow_count = int(np.random.randint(max_shadows + 1))
    shadow_pool = list(map(int, np.random.permutation(shadow_pool)))
    shadow_colors = shadow_pool[:shadow_count]
    distractor_color = shadow_pool[shadow_count] if has_distractor else None
    return list(map(int, shape_colors)), list(map(int, shadow_colors)), None if distractor_color is None else int(distractor_color)


def _assign_shadows(shape_count: int, shadow_colors: list[int]) -> dict[int, int]:
    if not shadow_colors:
        return {}
    targets = np.random.choice(shape_count, size=len(shadow_colors), replace=False).tolist()
    return {int(idx): int(col) for idx, col in zip(sorted(targets), shadow_colors)}


def _dilate_8(mask: np.ndarray, radius: int) -> np.ndarray:
    radius = int(radius)
    if radius <= 0:
        return mask.copy()
    out = np.zeros_like(mask, dtype=bool)
    h, w = mask.shape
    for dr in range(-radius, radius + 1):
        for dc in range(-radius, radius + 1):
            r_src = slice(max(0, -dr), h - max(0, dr))
            c_src = slice(max(0, -dc), w - max(0, dc))
            r_dst = slice(max(0, dr), h - max(0, -dr))
            c_dst = slice(max(0, dc), w - max(0, -dc))
            out[r_dst, c_dst] |= mask[r_src, c_src]
    return out


def _candidate_locations(canvas_out: np.ndarray, patch_out: np.ndarray) -> list[tuple[int, int]]:
    h, w = patch_out.shape
    H, W = canvas_out.shape
    if h > H or w > W:
        return []

    occupied = canvas_out != BG
    blocked = _dilate_8(occupied, PLACEMENT_PADDING)
    rr, cc = np.nonzero(patch_out != BG)
    if len(rr) == 0:
        return []

    candidates: list[tuple[int, int]] = []
    for r0 in range(H - h + 1):
        for c0 in range(W - w + 1):
            if blocked[r0 + rr, c0 + cc].any():
                continue
            candidates.append((int(r0), int(c0)))
    return candidates


def _cross_group_alignment_exists(
    new_centers: list[tuple[int, int]],
    prior_centers: list[tuple[int, int]],
) -> bool:
    for r0, c0 in new_centers:
        for r1, c1 in prior_centers:
            if _aligned_8(r0, c0, r1, c1):
                return True
    return False


def _try_place_shape(
    inp: np.ndarray,
    out: np.ndarray,
    patch_in: np.ndarray,
    patch_out: np.ndarray,
    patch_centers: list[tuple[int, int]],
    shape_color: int,
    placed_shape_groups: list[dict[str, object]],
) -> bool:
    candidates = _candidate_locations(out, patch_out)
    valid: list[tuple[int, int]] = []
    for r0, c0 in candidates:
        global_centers = [(r0 + r, c0 + c) for r, c in patch_centers]
        bad = False
        for info in placed_shape_groups:
            if int(info["shape_color"]) != shape_color:
                continue
            if _cross_group_alignment_exists(global_centers, info["centers"]):
                bad = True
                break
        if not bad:
            valid.append((r0, c0))

    if not valid:
        return False

    loc = valid[int(np.random.randint(len(valid)))]
    paste(out, patch_out, loc[0], loc[1], transparent=BG)
    paste(inp, patch_in, loc[0], loc[1], transparent=BG)
    placed_shape_groups.append(
        {
            "shape_color": int(shape_color),
            "centers": [(loc[0] + r, loc[1] + c) for r, c in patch_centers],
        }
    )
    return True


def _try_place_patch(inp: np.ndarray, out: np.ndarray, patch_in: np.ndarray, patch_out: np.ndarray) -> bool:
    candidates = _candidate_locations(out, patch_out)
    if not candidates:
        return False
    loc = candidates[int(np.random.randint(len(candidates)))]
    paste(out, patch_out, loc[0], loc[1], transparent=BG)
    paste(inp, patch_in, loc[0], loc[1], transparent=BG)
    return True


def _sample_canvas_size() -> tuple[int, int]:
    H = int(np.random.randint(*CANVAS_H_RANGE))
    W = int(np.clip(H + np.random.randint(*CANVAS_W_JITTER), *CANVAS_W_CLIP))
    return H, W


def _gen_attempt() -> np.ndarray | None:
    H, W = _sample_canvas_size()
    out = np.full((H, W), BG, dtype=int)
    inp = np.full((H, W), BG, dtype=int)

    has_distractor = bool(np.random.rand() < DISTRACTOR_PROB)
    shape_count = int(np.random.randint(*SHAPE_COUNT_RANGE))

    palette = _sample_palette(shape_count, has_distractor)
    if palette is None:
        return None
    shape_colors, shadow_colors, distractor_color = palette
    if not _has_reused_shape_color(shape_colors):
        return None

    shadow_map = _assign_shadows(shape_count, shadow_colors)
    placed_shape_groups: list[dict[str, object]] = []

    for idx, shape_color in enumerate(shape_colors):
        placed = False
        for _ in range(SHAPE_PLACEMENT_RETRIES):
            patch = _make_shape_patch(int(shape_color), shadow_map.get(idx))
            if patch is None:
                continue
            patch_in, patch_out, patch_centers = patch
            if _try_place_shape(inp, out, patch_in, patch_out, patch_centers, int(shape_color), placed_shape_groups):
                placed = True
                break
        if not placed:
            return None

    if has_distractor and distractor_color is not None:
        if not _try_place_patch(inp, out, *_make_distractor_patch(distractor_color)):
            return None

    present_colors = set(shape_colors) | set(shadow_colors)
    if distractor_color is not None:
        present_colors.add(int(distractor_color))

    stats = fusion_stats(inp)
    if stats["shape_group_count"] != shape_count:
        return None
    if stats["vertex_component_count"] < 2 * shape_count:
        return None
    if stats["reused_color_count"] < 1 or stats["reused_group_surplus"] < 1:
        return None
    if get_bg(inp) != BG or get_bg(out) != BG:
        return None
    if set(colors_present(inp, exclude=[BG])) != present_colors:
        return None
    if np.array_equal(_main_no_split(inp), out):
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
