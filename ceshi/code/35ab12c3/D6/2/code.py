# Variant of 35ab12c3
# Dimension: D6 — 阴影优先级规则 (transform)
# Change: 形状颜色优先（shape color wins）——当阴影像素与形状像素重叠时，形状颜色覆盖阴影颜色
# ... → 重叠区域擦除为背景

import math
import numpy as np

from arc_utils import get_bg as _bg_color
from arc_utils import bbox_coords, colors_present, get_bg, neighbors8, paste, random_free_location

# -----------------------------------------------------------------------------
# Design analysis for the D6 variant
#
# a) Concretization / behavior trace
# - Original task: each shape color is expanded into a polyline/polygon outline, then a
#   carried neighbor color is copied along a parallel offset path. The old compositor is
#   "write main shape first, then write shadow only onto background", so any overlap is
#   silently resolved in favor of the shape color.
# - Variant: keep the same detection logic (same vertices, same start-anchor search, same
#   shifted shadow geometry), but replace the compositor. For each shape, collect the set
#   of cells covered by the main outline and the set of cells covered by each shadow copy.
#   Cells in shape-only area stay the shape color, cells in shadow-only area stay the
#   shadow color, and cells in the intersection become background. This creates visible
#   cut-out holes.
# - Small trace:
#   input vertices: a 1-cell-tall rectangle with a shadow anchor on an outward diagonal.
#   old output: the shifted bottom edge overlaps the top edge; the top edge stays filled.
#   new output: the same overlapped run is erased to background, so the top edge acquires
#   a gap. Nothing else about vertex ordering or shadow offset inference changes.
#
# b) Complexity calibration
# - If the generator kept avoiding overlaps, the new rule would often be invisible and the
#   task would degenerate back to the original. Therefore each generated puzzle contains at
#   least one shadowed shape, and shadowed shapes are biased toward "thin rectangles" whose
#   shifted copies overlap the original outline in multiple cells.
# - Freedom is still controlled: the shape families, palette, canvas size, distractors, and
#   placement policy remain close to the original solver/generator.
#
# c) Implicit-assumption audit of main()
# - Original _draw_line_bg_only depended on the old invariant "shape wins" because writing
#   only to background preserves existing shape pixels under shadow overlaps. That invariant
#   is no longer valid.
# - The seed selection, neighbor carrying, edge alignment checks, and original-grid path
#   clearance checks do not depend on the old priority rule, so they remain unchanged.
# - The new compositor must explicitly clear every cell touched by either the shape or the
#   shadow before repainting the surviving exclusive regions; otherwise original input
#   vertices / shadow anchors would leak through at erased cells.
#
# d) Placement legality
# - Local patches are still bounded by the union of shape cells and shifted shadow cells.
# - Global placement still uses random_free_location with padding, so different groups do
#   not overlap each other. Only intra-shape shape-vs-shadow overlap is allowed, and that
#   overlap is intentional because it is the feature under test.
# - The solver still assumes that a carried shadow color is adjacent to exactly one vertex;
#   the generator enforces that for each shadow anchor.
#
# e) Visual observability
# - Every generated grid has at least one shadow.
# - Every shadowed shape is accepted only if the inferred shadow overlaps the main shape in
#   at least two cells, so the erase/XOR-like behavior is visible in 2-4 examples.
# - No hidden ordering by color id is introduced; all decisions are geometric.
#
# f) Visual cleanliness
# - Inter-group padding is preserved, so holes appear inside a single shape/shadow pair
#   instead of creating cluttered multi-object collisions.
# - Distractors remain sparse and unchanged.
#
# g) Color inference
# - Colors keep their original role: one color for the shape, one distinct color for the
#   shadow, optional one for a distractor. The new rule is structural (erase on overlap),
#   not a color remapping, so humans can infer it from geometry alone.
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


def _collect_drawable_edges(
    g: np.ndarray,
    cycle: list[tuple[int, int]],
    col: int,
    bg: int,
) -> list[tuple[tuple[int, int], tuple[int, int]]]:
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
    return drawable_edges


def _collect_shape_cells(
    edges: list[tuple[tuple[int, int], tuple[int, int]]],
) -> set[tuple[int, int]]:
    cells: set[tuple[int, int]] = set()
    for (r0, c0), (r1, c1) in edges:
        cells.update(_segment_cells(r0, c0, r1, c1))
    return cells


def _collect_shadow_cells(
    g: np.ndarray,
    edges: list[tuple[tuple[int, int], tuple[int, int]]],
    offset: tuple[int, int],
    bg: int,
) -> set[tuple[int, int]]:
    drn, dcn = offset
    h, w = g.shape
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


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = _bg_color(g)

    out = g.copy()

    colors = sorted(set(int(v) for v in np.unique(g)) - {bg})
    for col in colors:
        pts_rc = [tuple(map(int, p)) for p in np.argwhere(g == col)]
        if len(pts_rc) < 2:
            continue

        cycle = _angle_cycle(pts_rc)
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

        drawable_edges = _collect_drawable_edges(g, cycle, col, bg)
        if not drawable_edges:
            continue

        shape_cells = _collect_shape_cells(drawable_edges)
        shadow_layers: list[tuple[int, set[tuple[int, int]]]] = []
        overlap_cells: set[tuple[int, int]] = set()

        for drn, dcn, ncol in carried:
            shadow_cells = _collect_shadow_cells(g, drawable_edges, (drn, dcn), bg)
            if not shadow_cells:
                continue
            shadow_layers.append((ncol, shadow_cells))
            overlap_cells.update(shape_cells & shadow_cells)

        affected = set(shape_cells)
        for _, shadow_cells in shadow_layers:
            affected.update(shadow_cells)
        for rr, cc in affected:
            out[rr, cc] = bg

        for rr, cc in (shape_cells - overlap_cells):
            out[rr, cc] = col

        for ncol, shadow_cells in shadow_layers:
            for rr, cc in (shadow_cells - shape_cells):
                out[rr, cc] = ncol

    return out


# --------------- generator (D6 variant) ---------------

# ---- CFG: canvas and palette ----
BG = 0
COLORS = [1, 2, 3, 4, 6, 7, 8]
CANVAS_H_RANGE = (15, 23)
CANVAS_W_JITTER = (-3, 4)
CANVAS_W_CLIP = (15, 22)

# ---- CFG: structural counts ----
SHAPE_COUNT_RANGE = (2, 5)
MAX_SHADOWS_CAP = 3
MIN_SHADOWS = 1
DISTRACTOR_PROB = 0.5
PLACEMENT_PADDING = 1
PLACEMENT_PAD_CONNECTIVITY = 8

# ---- CFG: shape geometry ----
LINE_VERTICES_RANGE = (2, 8)
LINE_TOTAL_LENGTH = 13
LINE_MIN_GAP = 2
TRIANGLE_SIDE_RANGE = (2, 8)
RECT_H_RANGE = (2, 9)
RECT_W_RANGE = (2, 9)
DIAMOND_ARM_RANGE = (2, 6)
PENTAGON_ROOF_RANGE = (2, 5)
PENTAGON_WALL_RANGE = (2, 7)
HEXAGON_SLANT_RANGE = (2, 5)
HEXAGON_WALL_RANGE = (2, 6)
OVERLAP_RECT_W_RANGE = (4, 9)
OVERLAP_MIN_CELLS = 2
OVERLAP_SHAPE_BIAS = 0.75

# ---- CFG: shape mixture ----
SHAPE_WEIGHTS = np.array([0.28, 0.18, 0.18, 0.14, 0.12, 0.10], dtype=float)

# ---- CFG: distractor geometry ----
DISTRACTOR_COUNT_RANGE = (1, 4)
DISTRACTOR_BBOX_RANGE = (2, 5)
DISTRACTOR_RETRIES = 50

# ---- CFG: retry budgets ----
SHAPE_PLACEMENT_RETRIES = 40
SHADOW_SELECTION_RETRIES = 60
GENERATE_RETRIES = 300


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
    hh = int(np.random.randint(*RECT_H_RANGE))
    ww = int(np.random.randint(*RECT_W_RANGE))
    return _transform_coords([(0, 0), (0, ww), (hh, ww), (hh, 0)])


def _build_diamond() -> list[tuple[int, int]]:
    arm = int(np.random.randint(*DIAMOND_ARM_RANGE))
    return _transform_coords([(0, arm), (arm, 2 * arm), (2 * arm, arm), (arm, 0)])


def _build_pentagon() -> list[tuple[int, int]]:
    roof = int(np.random.randint(*PENTAGON_ROOF_RANGE))
    wall = int(np.random.randint(*PENTAGON_WALL_RANGE))
    hh = roof + wall
    return _transform_coords([(0, roof), (roof, 0), (hh, 0), (hh, 2 * roof), (roof, 2 * roof)])


def _build_hexagon() -> list[tuple[int, int]]:
    slant = int(np.random.randint(*HEXAGON_SLANT_RANGE))
    wall = int(np.random.randint(*HEXAGON_WALL_RANGE))
    mid = slant + wall
    bot = 2 * slant + wall
    return _transform_coords([(0, slant), (slant, 0), (mid, 0), (bot, slant), (mid, 2 * slant), (slant, 2 * slant)])


def _build_overlap_rectangle() -> list[tuple[int, int]]:
    ww = int(np.random.randint(*OVERLAP_RECT_W_RANGE))
    coords = [(0, 0), (0, ww), (1, ww), (1, 0)]
    arr = np.array(coords, dtype=int)
    if np.random.rand() < 0.5:
        arr[:, 0] *= -1
    if np.random.rand() < 0.5:
        arr[:, 1] *= -1
    arr -= arr.min(axis=0)
    return [tuple(map(int, p)) for p in arr]


_SHAPE_BUILDERS = [_build_line, _build_triangle, _build_rectangle, _build_diamond, _build_pentagon, _build_hexagon]


def _make_vertices(require_overlap: bool = False) -> list[tuple[int, int]]:
    if require_overlap and np.random.rand() < OVERLAP_SHAPE_BIAS:
        return _build_overlap_rectangle()
    weights = SHAPE_WEIGHTS / SHAPE_WEIGHTS.sum()
    idx = int(np.random.choice(len(_SHAPE_BUILDERS), p=weights))
    return _SHAPE_BUILDERS[idx]()


def _shape_edges(vertices: list[tuple[int, int]]) -> list[tuple[tuple[int, int], tuple[int, int]]]:
    if len(vertices) == 2:
        return [(vertices[0], vertices[1])]
    return [(vertices[i], vertices[(i + 1) % len(vertices)]) for i in range(len(vertices))]


def _shape_cells(vertices: list[tuple[int, int]]) -> set[tuple[int, int]]:
    cells: set[tuple[int, int]] = set()
    for a, b in _shape_edges(vertices):
        cells.update(_gen_segment_cells(a, b))
    return cells


def _shadow_cells_for_offset(
    vertices: list[tuple[int, int]],
    offset: tuple[int, int],
) -> set[tuple[int, int]]:
    drn, dcn = offset
    cells: set[tuple[int, int]] = set()
    for a, b in _shape_edges(vertices):
        drs, dcs = _step(a, b)
        if drs * dcn - dcs * drn == 0:
            continue
        a2 = (a[0] + drn, a[1] + dcn)
        b2 = (b[0] + drn, b[1] + dcn)
        if not _aligned_8(a2[0], a2[1], b2[0], b2[1]):
            continue
        cells.update(_gen_segment_cells(a2, b2))
    return cells


def _choose_shadow(
    vertices: list[tuple[int, int]],
    shape_cells: set[tuple[int, int]],
    require_overlap: bool,
) -> tuple[tuple[int, int], tuple[int, int], int] | None:
    candidates: list[tuple[int, int, tuple[int, int], tuple[int, int]]] = []
    for anchor in vertices:
        for dr, dc in DIR8:
            pos = (anchor[0] + dr, anchor[1] + dc)
            if pos in shape_cells:
                continue
            if sum(1 for vr, vc in vertices if max(abs(pos[0] - vr), abs(pos[1] - vc)) == 1) != 1:
                continue
            shadow_cells = _shadow_cells_for_offset(vertices, (dr, dc))
            if not shadow_cells:
                continue
            overlap = len(shape_cells & shadow_cells)
            if require_overlap and overlap < OVERLAP_MIN_CELLS:
                continue
            candidates.append((overlap, abs(dr) + abs(dc), anchor, (dr, dc)))
    if not candidates:
        return None
    overlaps = np.array([cand[0] for cand in candidates], dtype=float)
    rank = overlaps - overlaps.min() + 1.0
    probs = rank / rank.sum()
    idx = int(np.random.choice(len(candidates), p=probs))
    overlap, _span, anchor, offset = candidates[idx]
    return anchor, offset, int(overlap)


def _patch_bbox(
    shape_cells: set[tuple[int, int]],
    vertices: list[tuple[int, int]],
    shadow: tuple[tuple[int, int], tuple[int, int], int] | None,
) -> tuple[int, int, int, int]:
    coords = list(shape_cells) + vertices
    if shadow is not None:
        (ar, ac), (dr, dc), _ = shadow
        coords.append((ar + dr, ac + dc))
        coords.extend((r + dr, c + dc) for r, c in shape_cells)
    r0, c0, r1, c1 = bbox_coords(np.array(coords, dtype=int))
    return int(r0), int(c0), int(r1), int(c1)


def _make_shape_patch(
    shape_color: int,
    shadow_color: int | None,
) -> tuple[np.ndarray, np.ndarray, int] | None:
    require_overlap = shadow_color is not None
    for _ in range(SHADOW_SELECTION_RETRIES if require_overlap else 1):
        vertices = _make_vertices(require_overlap=require_overlap)
        cells = _shape_cells(vertices)

        shadow = None
        overlap_count = 0
        if shadow_color is not None:
            shadow = _choose_shadow(vertices, cells, require_overlap=True)
            if shadow is None:
                continue
            overlap_count = shadow[2]

        r0, c0, r1, c1 = _patch_bbox(cells, vertices, shadow)
        hh, ww = r1 - r0 + 1, c1 - c0 + 1
        inp = np.full((hh, ww), BG, dtype=int)

        shifted_vertices = [(r - r0, c - c0) for r, c in vertices]
        for r, c in shifted_vertices:
            inp[r, c] = shape_color

        if shadow is not None:
            (ar, ac), offset, _ = shadow
            anchor = (ar - r0, ac - c0)
            sa = (anchor[0] + offset[0], anchor[1] + offset[1])
            if not (0 <= sa[0] < hh and 0 <= sa[1] < ww) or inp[sa] != BG:
                continue
            inp[sa] = int(shadow_color)
            nb_hits = sum(
                1
                for rr, cc in neighbors8(sa[0], sa[1])
                if 0 <= rr < hh and 0 <= cc < ww and inp[rr, cc] == shape_color
            )
            if nb_hits != 1:
                continue

        out = main(inp)
        if shadow_color is not None and overlap_count < OVERLAP_MIN_CELLS:
            continue
        if np.array_equal(inp, out):
            continue
        return inp, out, overlap_count
    return None


def _sample_distractor_coords(hh: int, ww: int, count: int) -> list[tuple[int, int]] | None:
    coords: list[tuple[int, int]] = []
    seen: set[tuple[int, int]] = set()
    while len(coords) < count:
        cand = (int(np.random.randint(hh)), int(np.random.randint(ww)))
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
        hh = int(np.random.randint(*DISTRACTOR_BBOX_RANGE))
        ww = int(np.random.randint(*DISTRACTOR_BBOX_RANGE))
        coords = _sample_distractor_coords(hh, ww, count)
        if coords is None:
            continue
        patch = np.full((hh, ww), BG, dtype=int)
        for r, c in coords:
            patch[r, c] = color
        return patch.copy(), patch.copy()
    patch = np.full((1, 1), color, dtype=int)
    return patch.copy(), patch.copy()


def _sample_palette(shape_count: int, has_distractor: bool) -> tuple[list[int], list[int], int | None] | None:
    max_shadows = min(MAX_SHADOWS_CAP, shape_count, len(COLORS) - shape_count - int(has_distractor))
    if max_shadows < MIN_SHADOWS:
        return None
    shadow_count = int(np.random.randint(MIN_SHADOWS, max_shadows + 1))
    total = shape_count + shadow_count + int(has_distractor)
    palette = list(np.random.permutation(COLORS)[:total])
    shape_colors = palette[:shape_count]
    shadow_colors = palette[shape_count:shape_count + shadow_count]
    distractor_color = int(palette[-1]) if has_distractor else None
    return shape_colors, shadow_colors, distractor_color


def _assign_shadows(shape_count: int, shadow_colors: list[int]) -> dict[int, int]:
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
    hh = int(np.random.randint(*CANVAS_H_RANGE))
    ww = int(np.clip(hh + np.random.randint(*CANVAS_W_JITTER), *CANVAS_W_CLIP))
    return hh, ww


def _gen_attempt() -> np.ndarray | None:
    hh, ww = _sample_canvas_size()
    out = np.full((hh, ww), BG, dtype=int)
    inp = np.full((hh, ww), BG, dtype=int)

    has_distractor = bool(np.random.rand() < DISTRACTOR_PROB)
    shape_count = int(np.random.randint(*SHAPE_COUNT_RANGE))

    palette = _sample_palette(shape_count, has_distractor)
    if palette is None:
        return None
    shape_colors, shadow_colors, distractor_color = palette
    shadow_map = _assign_shadows(shape_count, shadow_colors)

    total_colors = shape_count + len(shadow_colors) + int(has_distractor)
    overlap_total = 0

    for idx, shape_color in enumerate(shape_colors):
        placed = False
        for _ in range(SHAPE_PLACEMENT_RETRIES):
            patch = _make_shape_patch(int(shape_color), shadow_map.get(idx))
            if patch is None:
                continue
            patch_in, patch_out, overlap_count = patch
            if _try_place(inp, out, patch_in, patch_out):
                placed = True
                overlap_total += overlap_count
                break
        if not placed:
            return None

    if overlap_total < OVERLAP_MIN_CELLS:
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
