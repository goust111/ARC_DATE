# Variant of 35ab12c3
# Dimension: D7 — 源对象——形状顶点来源 (src)
# Change: 同色像素组作为多边形/折线顶点，每组2-7个同色像素，形状类型包括折线、三角形、矩形、菱形、五边形、六边形... → 色块连通分量作为源

# Design analysis for the D7 variant
# ----------------------------------
# Reference-task note:
# - The prompt asks to read the solver of source task `b5ca7ac4` first.
# - In this workspace there is no local solver file for `b5ca7ac4`; only
#   `/home/lixinyu/snap/new/ARC-data/ARC-AGI-2/data/evaluation/b5ca7ac4.json`
#   is available. The implementation therefore only borrows the visible
#   "treat a same-color connected piece as one semantic unit" idea.
#
# a) Concrete behavior and trace
# - Original task: each shape color contributes 2-7 isolated pixels. The solver
#   treats every pixel of that color as a vertex, orders them by polar angle,
#   connects aligned consecutive vertices, then uses a different-color pixel next
#   to the chosen start vertex as the shadow seed.
# - Variant: each shape vertex is now a small same-color connected component
#   instead of a lone pixel. The solver must first segment same-color components,
#   then reduce each valid component to one lattice point: its centroid.
# - Example trace:
#   input had `+` blocks centered at (4,4), (4,12), (12,12), plus a shadow seed
#   at (3,3). The old solver would see fifteen same-color pixels and connect many
#   wrong pairs. The new solver first finds three components, computes centers
#   (4,4), (4,12), (12,12), then draws the intended triangle edges and the offset
#   shadow lines from the seed at offset (-1,-1).
#
# b) Complexity calibration
# - A literal 2x2 square has a half-integer centroid, which would force an extra
#   rounding rule not present in the original task and would make line placement
#   ambiguous on the ARC lattice.
# - To keep D7 visible but still logically crisp, generated vertex components are
#   restricted to two 3x3 symmetric templates:
#     1. `+` block: center + orthogonal arms
#     2. `X` block: center + diagonal arms
# - Both are unmistakably multi-pixel connected components, both have an integer
#   centroid at the center cell, and together they expose both diagonal and
#   orthogonal empty neighbor slots for shadow seeding.
#
# c) Old implicit assumptions that break
# - `np.argwhere(g == col)` in the original `main()` assumes every colored cell is
#   itself a vertex. This is false now; it would explode one vertex-block into
#   five fake vertices.
# - Start-vertex detection originally scans the 8-neighborhood of a single vertex
#   pixel. In the variant it must scan around the component centroid instead.
# - The original generator can place vertices only 2 cells apart. With 3x3
#   vertex-blocks that would merge components, so spacing has to be enlarged.
#
# d) Placement legality
# - Generator places centroid positions on a step-4 lattice. That guarantees two
#   3x3 templates of the same color remain disconnected under 8-connectivity.
# - Each local patch is validated by running `main(patch_in)`, and the canvas
#   placement still uses `patch_out` for collision checks, so final outputs do not
#   overlap or leave the grid.
#
# e) Visual observability
# - The D7 change is immediately visible: shape "vertices" are now compact 3x3
#   blobs instead of isolated dots.
# - The output still preserves those blobs, so a human can infer that lines are
#   drawn through blob centers rather than through every blob pixel.
#
# f) Clean visualization
# - Only one shadow seed is attached per shape, as in the original generator.
# - Vertex components are small and symmetric; they remain readable after line
#   drawing because the line color matches the block color and the shadow is drawn
#   only in the offset color.
#
# g) Color reasoning
# - No new color mapping is introduced. Colors keep their original roles:
#   shape color, optional shadow color, optional distractor color.
#
# h) Non-degeneracy target
# - Inputs are deliberately generated so every real shape color has >=2 connected
#   components of area 5, and old per-pixel logic would therefore disagree with
#   the new centroid-based logic on most samples.
#
# i) New solver assumptions and how the generator enforces them
# - Assumption: a real vertex component has area >= 4 and an integer centroid.
#   Generator enforces this with 3x3 `+` / `X` templates of area 5.
# - Assumption: shadow seeds sit in an empty 8-neighbor of the anchor centroid.
#   Generator only uses template-complement offsets and checks uniqueness.
# - Assumption: centroids of consecutive vertices are H/V/45-degree aligned.
#   Generator still builds all abstract shapes from that same alignment family.

import math

import numpy as np

from arc_utils import (
    bbox_coords,
    colors_present,
    draw_line,
    find_components,
    get_bg,
    neighbors8,
    paste,
    random_free_location,
)

DIR8 = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
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


def _color_vertex_centers(g: np.ndarray, col: int, bg: int) -> list[tuple[int, int]]:
    comps = find_components(
        g,
        bg=bg,
        connectivity=8,
        monochromatic=True,
        colors=[col],
        min_area=VERTEX_MIN_AREA,
        return_format="dict",
    )
    centers: list[tuple[int, int]] = []
    for comp in comps:
        center = _component_centroid(np.asarray(comp["coords"], dtype=int))
        if center is not None:
            centers.append(center)
    return sorted(centers)


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = get_bg(g)

    out = g.copy()

    colors = sorted(set(int(v) for v in np.unique(g)) - {bg})
    for col in colors:
        centers = _color_vertex_centers(g, col, bg)
        if len(centers) < 2:
            continue

        cycle = _angle_cycle(centers)
        if len(cycle) < 2:
            continue

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
            if not _clear_in_original(g, r0, c0, r1, c1, bg, allow={col}):
                continue
            drawable_edges.append(((r0, c0), (r1, c1)))

        for (r0, c0), (r1, c1) in drawable_edges:
            _draw_line_bg_only(out, r0, c0, r1, c1, col, bg)

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
                if not _clear_in_original(g, rr0, cc0, rr1, cc1, bg, allow=set()):
                    continue
                _draw_line_bg_only(out, rr0, cc0, rr1, cc1, ncol, bg)

    return out


# --------------- generator ---------------

BG = 0
COLORS = [1, 2, 3, 4, 6, 7, 8]
CANVAS_H_RANGE = (15, 23)
CANVAS_W_JITTER = (-3, 4)
CANVAS_W_CLIP = (15, 22)

SHAPE_COUNT_RANGE = (2, 5)
MAX_SHADOWS_CAP = 3
DISTRACTOR_PROB = 0.5
PLACEMENT_PADDING = 1
PLACEMENT_PAD_CONNECTIVITY = 8

CENTER_STEP = 4
LINE_VERTICES_RANGE = (2, 6)          # 2-5 centers
LINE_TOTAL_UNITS_RANGE = (4, 6)       # total span in step-units: 4-5
LINE_MIN_GAP_UNITS = 1
TRIANGLE_SIDE_UNITS = (1, 3)          # 1-2
RECT_H_UNITS = (1, 3)
RECT_W_UNITS = (1, 3)
DIAMOND_ARM_UNITS = (1, 3)
PENTAGON_ROOF_UNITS = (1, 3)
PENTAGON_WALL_UNITS = (1, 3)
HEXAGON_SLANT_UNITS = (1, 2)          # fixed 1
HEXAGON_WALL_UNITS = (1, 3)

SHAPE_WEIGHTS = np.array([0.28, 0.18, 0.18, 0.14, 0.12, 0.10], dtype=float)

DISTRACTOR_COUNT_RANGE = (1, 4)
DISTRACTOR_BBOX_RANGE = (2, 5)
DISTRACTOR_RETRIES = 50

SHAPE_PLACEMENT_RETRIES = 50
GENERATE_RETRIES = 400

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
        patch[center[0] + dr, center[1] + dc] = color


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


def _make_shape_patch(shape_color: int, shadow_color: int | None) -> tuple[np.ndarray, np.ndarray] | None:
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

        centers = _color_vertex_centers(inp, int(shape_color), BG)
        if len(centers) != len(vertices):
            continue
        if sorted(centers) != sorted(shifted_vertices):
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


def _sample_palette(shape_count: int, has_distractor: bool) -> tuple[list[int], list[int], int | None] | None:
    max_shadows = min(MAX_SHADOWS_CAP, shape_count, len(COLORS) - shape_count - int(has_distractor))
    if max_shadows < 0:
        return None
    shadow_count = int(np.random.randint(max_shadows + 1))
    total = shape_count + shadow_count + int(has_distractor)
    palette = list(np.random.permutation(COLORS)[:total])
    shape_colors = palette[:shape_count]
    shadow_colors = palette[shape_count:shape_count + shadow_count]
    distractor_color = int(palette[-1]) if has_distractor else None
    return shape_colors, shadow_colors, distractor_color


def _assign_shadows(shape_count: int, shadow_colors: list[int]) -> dict[int, int]:
    if not shadow_colors:
        return {}
    targets = np.random.choice(shape_count, size=len(shadow_colors), replace=False).tolist()
    return {int(idx): int(col) for idx, col in zip(sorted(targets), shadow_colors)}


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
    shadow_map = _assign_shadows(shape_count, shadow_colors)

    total_colors = shape_count + len(shadow_colors) + int(has_distractor)

    for idx, shape_color in enumerate(shape_colors):
        placed = False
        for _ in range(SHAPE_PLACEMENT_RETRIES):
            patch = _make_shape_patch(int(shape_color), shadow_map.get(idx))
            if patch is None:
                continue
            if _try_place(inp, out, *patch):
                placed = True
                break
        if not placed:
            return None

    if has_distractor and distractor_color is not None:
        if not _try_place(inp, out, *_make_distractor_patch(distractor_color)):
            return None

    if get_bg(inp) != BG or get_bg(out) != BG:
        return None
    if len(colors_present(inp, exclude=[BG])) != total_colors:
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
