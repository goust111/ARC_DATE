# Combo variant of 35ab12c3 (index 5)
# combo_spec: D1/2+D8/2
# Name: 距离约束配对+距离编码停止
#
# Design analysis
# ===============
# a) Concretization / behavior trace
# - Original task: a same-color set such as A=(1,1), B=(1,8), C=(7,8) is treated as
#   one cycle/polyline, so the solver draws A-B and B-C. If a shadow anchor of color
#   7 touches A, the whole multi-edge trace is copied in parallel.
# - D1/2 changes the match stage: only one same-color pair is selected, namely the
#   unique shortest aligned and unobstructed pair. In the example above, if another
#   vertex D creates several candidate pairs, only the nearest pair survives.
# - D8/2 changes the stop rule: after a pair is chosen, the second endpoint is only a
#   direction cue. The actual painted length becomes n, where n is the total number
#   of input pixels of that color.
# - Fused behavior: suppose one color has 4 vertices and its closest drawable pair is
#   A=(1,1), B=(1,7). The solver starts from the chosen endpoint A (or the endpoint
#   carrying the shadow anchor), points toward B, but paints only 4 steps:
#   (1,1)->(1,5). B remains visible as a distant isolated dot. If a shadow anchor
#   touches A, only that truncated ray is copied in parallel.
#
# b) Complexity calibration
# - If a group had only two vertices, "closest pair" would be trivial and D1/2 would
#   become invisible. The generator therefore uses only 3-vertex elbows and 4-vertex
#   zigzags, guaranteeing multiple candidate pairs.
# - If the closest-pair distance were <= the color count, the D8/2 truncation would
#   disappear or collide with the endpoint. The generator enforces:
#   shortest_pair_length > vertex_count.
# - To keep the encoded distances human-readable, the count only takes values 3 or 4.
#   This gives repeated evidence across examples without exploding the hypothesis
#   space.
#
# c) Implicit-assumption review against the original main()
# - The original `_angle_cycle(...)` assumed every same-color point belongs to one
#   global polygon/polyline. That assumption is invalid here; the solver must search
#   over all same-color pairs instead.
# - The original `_clear_in_original(...)` skipped the final endpoint because the
#   endpoint was explicitly occupied. In this combo the painted stop cell is usually
#   background, so ray validation must check every cell after the start.
# - The original shadow phase copied every drawable edge. Here it must copy only the
#   chosen truncated ray; otherwise the output would regress toward either the
#   original task or the D1-only variant.
# - New solver assumptions introduced by the fusion:
#   the chosen pair is unique, the ray of length `count(color)` stays in-bounds and
#   background-only, and a shadow anchor (if present) touches exactly one chosen
#   endpoint. The generator explicitly guarantees each of these.
#
# d) Placement legality
# - Each local patch contains: all same-color vertices, the truncated main ray,
#   optionally one shadow-anchor pixel, and optionally one parallel truncated shadow
#   ray. The patch bbox is computed from the union of those cells.
# - Global placement still uses `random_free_location(..., padding=1)`, so different
#   objects remain separated by background and cannot create accidental foreign-color
#   neighbors or mid-ray collisions.
#
# e) Visual observability
# - A human can count that a color appears 3 or 4 times in the input, then observe
#   that the output ray of that color always has exactly that many steps.
# - The same human also sees that among several same-color vertices, only one pair is
#   used to determine direction, because the other same-color vertices remain as dots.
# - No color-id arithmetic or hidden ordering is used; only visible distance, visible
#   adjacency, and visible pixel count matter.
#
# f) Visual cleanliness
# - The generator avoids dense polygons and only creates one ray per shape color, with
#   at most one parallel shadow copy. This keeps each motif compact and readable.
# - Shadow offsets are forced to be non-collinear with the chosen ray and the shifted
#   ray must stay disjoint from all vertices and from the main ray.
#
# g) Color-change inferability
# - Colors keep the original semantics: one color names the vertex group, a different
#   optional color names the shadow anchor/trace, and distractor colors remain
#   unchanged. There is no hidden remapping rule to infer.
#
# h) Generator-side guarantees for the fused solver assumptions
# - Every shape color has a unique shortest aligned candidate pair.
# - That shortest pair is one intended edge of the latent elbow/zigzag.
# - The shortest-pair distance is strictly larger than the same-color pixel count, so
#   the distance-coded stop is visible.
# - The truncated main ray and truncated shadow ray are collision-free in the input.

import numpy as np

from arc_utils import colors_present, get_bg, get_bg as _bg_color, paste, random_free_location

DIR8 = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
STEP_DIRS = DIR8[:]


def _aligned_8(r0: int, c0: int, r1: int, c1: int) -> bool:
    dr = r1 - r0
    dc = c1 - c0
    return (dr == 0 and dc != 0) or (dc == 0 and dr != 0) or (dr != 0 and abs(dr) == abs(dc))


def _step_dir(r0: int, c0: int, r1: int, c1: int) -> tuple[int, int]:
    dr = r1 - r0
    dc = c1 - c0
    return (0 if dr == 0 else (1 if dr > 0 else -1), 0 if dc == 0 else (1 if dc > 0 else -1))


def _ray_cells(r0: int, c0: int, dr: int, dc: int, steps: int):
    for k in range(steps + 1):
        yield r0 + dr * k, c0 + dc * k


def _clear_segment_in_original(
    g: np.ndarray,
    r0: int,
    c0: int,
    r1: int,
    c1: int,
    bg: int,
    allow: set[int],
) -> bool:
    drs, dcs = _step_dir(r0, c0, r1, c1)
    steps = max(abs(r1 - r0), abs(c1 - c0))
    for k in range(1, steps):
        rr = r0 + drs * k
        cc = c0 + dcs * k
        v = int(g[rr, cc])
        if v != bg and v not in allow:
            return False
    return True


def _clear_ray_in_original(g: np.ndarray, r0: int, c0: int, dr: int, dc: int, steps: int, bg: int) -> bool:
    for k in range(1, steps + 1):
        rr = r0 + dr * k
        cc = c0 + dc * k
        if int(g[rr, cc]) != bg:
            return False
    return True


def _draw_ray_bg_only(out: np.ndarray, r0: int, c0: int, dr: int, dc: int, steps: int, col: int, bg: int) -> None:
    for rr, cc in _ray_cells(r0, c0, dr, dc, steps):
        if int(out[rr, cc]) == bg:
            out[rr, cc] = col


def _sorted_pair(a: tuple[int, int], b: tuple[int, int]) -> tuple[tuple[int, int], tuple[int, int]]:
    return (a, b) if a <= b else (b, a)


def _seg_len(a: tuple[int, int], b: tuple[int, int]) -> int:
    return max(abs(a[0] - b[0]), abs(a[1] - b[1]))


def _has_foreign_neighbor(g: np.ndarray, pt: tuple[int, int], bg: int, col: int) -> bool:
    h, w = g.shape
    r, c = pt
    for dr, dc in DIR8:
        rr = r + dr
        cc = c + dc
        if 0 <= rr < h and 0 <= cc < w:
            v = int(g[rr, cc])
            if v != bg and v != col:
                return True
    return False


def _closest_drawable_pair(
    g: np.ndarray,
    pts_rc: list[tuple[int, int]],
    bg: int,
    col: int,
) -> tuple[tuple[int, int], tuple[int, int]] | None:
    candidates: list[tuple[int, int, int, int, int, tuple[int, int], tuple[int, int]]] = []
    for i in range(len(pts_rc)):
        for j in range(i + 1, len(pts_rc)):
            a = pts_rc[i]
            b = pts_rc[j]
            if not _aligned_8(a[0], a[1], b[0], b[1]):
                continue
            if not _clear_segment_in_original(g, a[0], a[1], b[0], b[1], bg, allow={col}):
                continue
            aa, bb = _sorted_pair(a, b)
            candidates.append((_seg_len(aa, bb), aa[0], aa[1], bb[0], bb[1], aa, bb))
    if not candidates:
        return None
    candidates.sort(key=lambda item: item[:5])
    return candidates[0][5], candidates[0][6]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Overview:
        Pick exactly one same-color pair per color: the unique closest drawable pair.

    Input structure:
        Non-background colors are either shape colors with 3 or 4 same-color vertices,
        optional one-pixel shadow anchors in another color, or distractor colors that
        do not form aligned candidate pairs.

    Rule:
        For each shape color, choose the closest aligned unobstructed pair. Use the
        chosen pair only to determine direction. Starting from the selected endpoint
        (the one with the foreign-color neighbor if present, otherwise the smaller
        endpoint), draw a ray whose length equals the number of input pixels of that
        shape color. Then copy that truncated ray in parallel for each carried
        neighbor color adjacent to the chosen start pixel.

    Key insight:
        Two visible numeric clues interact: geometry decides which same-color pair is
        relevant, while the count of same-color pixels decides where the ray stops.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = _bg_color(g)
    out = g.copy()

    colors = sorted(set(int(v) for v in np.unique(g)) - {bg})
    for col in colors:
        pts_rc = [tuple(map(int, p)) for p in np.argwhere(g == col)]
        if len(pts_rc) < 2:
            continue

        pair = _closest_drawable_pair(g, pts_rc, bg, col)
        if pair is None:
            continue

        a, b = pair
        seed_pts = [pt for pt in (a, b) if _has_foreign_neighbor(g, pt, bg, col)]
        start = min(seed_pts) if seed_pts else min(a, b)
        end = b if start == a else a
        ray_steps = len(pts_rc)
        drs, dcs = _step_dir(start[0], start[1], end[0], end[1])
        rr1 = start[0] + drs * ray_steps
        cc1 = start[1] + dcs * ray_steps
        if not (0 <= rr1 < h and 0 <= cc1 < w):
            continue
        if not _clear_ray_in_original(g, start[0], start[1], drs, dcs, ray_steps, bg):
            continue

        carried: list[tuple[int, int, int]] = []
        for drn, dcn in DIR8:
            rr = start[0] + drn
            cc = start[1] + dcn
            if 0 <= rr < h and 0 <= cc < w:
                v = int(g[rr, cc])
                if v != bg and v != col:
                    carried.append((drn, dcn, v))

        _draw_ray_bg_only(out, start[0], start[1], drs, dcs, ray_steps, col, bg)

        for drn, dcn, ncol in carried:
            if drs * dcn - dcs * drn == 0:
                continue
            rr0 = start[0] + drn
            cc0 = start[1] + dcn
            rr1 = rr0 + drs * ray_steps
            cc1 = cc0 + dcs * ray_steps
            if not (0 <= rr0 < h and 0 <= cc0 < w and 0 <= rr1 < h and 0 <= cc1 < w):
                continue
            if not _clear_ray_in_original(g, rr0, cc0, drs, dcs, ray_steps, bg):
                continue
            _draw_ray_bg_only(out, rr0, cc0, drs, dcs, ray_steps, ncol, bg)

    return out


# --------------- generator ---------------

BG = 0
COLORS = [1, 2, 3, 4, 6, 7, 8]
CANVAS_H_RANGE = (15, 23)
CANVAS_W_JITTER = (-3, 4)
CANVAS_W_CLIP = (15, 22)

SHAPE_COUNT_RANGE = (2, 5)
MAX_SHADOWS_CAP = 3
DISTRACTOR_PROB = 0.45
PLACEMENT_PADDING = 1
PLACEMENT_PAD_CONNECTIVITY = 8

SEG_LEN_RANGE = (5, 10)
EXTRA_SEG_LEN_BONUS = 2
ZIGZAG_PROB = 0.45
SHAPE_BUILD_RETRIES = 120

DISTRACTOR_COUNT_RANGE = (1, 4)
DISTRACTOR_BBOX_RANGE = (2, 5)
DISTRACTOR_RETRIES = 50

SHAPE_PLACEMENT_RETRIES = 50
GENERATE_RETRIES = 400


def _aligned(dr: int, dc: int) -> bool:
    return (dr == 0 and dc != 0) or (dc == 0 and dr != 0) or (dr != 0 and abs(dr) == abs(dc))


def _step(a: tuple[int, int], b: tuple[int, int]) -> tuple[int, int]:
    return _step_dir(a[0], a[1], b[0], b[1])


def _normalize_coords(coords: list[tuple[int, int]]) -> list[tuple[int, int]]:
    arr = np.array(coords, dtype=int)
    arr -= arr.min(axis=0)
    return [tuple(map(int, p)) for p in arr]


def _cross(d1: tuple[int, int], d2: tuple[int, int]) -> int:
    return d1[0] * d2[1] - d1[1] * d2[0]


def _move(pt: tuple[int, int], direction: tuple[int, int], length: int) -> tuple[int, int]:
    return pt[0] + direction[0] * length, pt[1] + direction[1] * length


def _polyline_edges(vertices: list[tuple[int, int]]) -> list[tuple[tuple[int, int], tuple[int, int]]]:
    return [(vertices[i], vertices[i + 1]) for i in range(len(vertices) - 1)]


def _sample_distinct_lengths(k: int) -> list[int]:
    vals = np.random.choice(np.arange(SEG_LEN_RANGE[0], SEG_LEN_RANGE[1]), size=k, replace=False)
    vals = sorted(int(v) for v in vals)
    return vals


def _build_elbow3() -> list[tuple[int, int]]:
    d1 = STEP_DIRS[int(np.random.randint(len(STEP_DIRS)))]
    turn_dirs = [d for d in STEP_DIRS if _cross(d1, d) != 0]
    d2 = turn_dirs[int(np.random.randint(len(turn_dirs)))]
    l1, l2 = _sample_distinct_lengths(2)
    if np.random.rand() < 0.5:
        l1, l2 = l2, l1
    p0 = (0, 0)
    p1 = _move(p0, d1, l1)
    p2 = _move(p1, d2, l2 + EXTRA_SEG_LEN_BONUS)
    return _normalize_coords([p0, p1, p2])


def _build_zigzag4() -> list[tuple[int, int]] | None:
    d1 = STEP_DIRS[int(np.random.randint(len(STEP_DIRS)))]
    d2_choices = [d for d in STEP_DIRS if _cross(d1, d) != 0]
    d2 = d2_choices[int(np.random.randint(len(d2_choices)))]
    d3_choices = [d for d in STEP_DIRS if _cross(d2, d) != 0 and d != (-d2[0], -d2[1])]
    d3 = d3_choices[int(np.random.randint(len(d3_choices)))]

    l1, l2, l3 = _sample_distinct_lengths(3)
    lens = [l1, l2 + EXTRA_SEG_LEN_BONUS, l3 + EXTRA_SEG_LEN_BONUS + 1]
    if np.random.rand() < 0.5:
        lens[0], lens[1] = lens[1], lens[0]

    p0 = (0, 0)
    p1 = _move(p0, d1, lens[0])
    p2 = _move(p1, d2, lens[1])
    p3 = _move(p2, d3, lens[2])
    coords = [p0, p1, p2, p3]
    if len(set(coords)) != len(coords):
        return None

    edges = _polyline_edges(coords)
    edge_cells = [set(_ray_cells(a[0], a[1], *_step(a, b), _seg_len(a, b))) for a, b in edges]
    for i in range(len(edge_cells)):
        for j in range(i + 1, len(edge_cells)):
            overlap = edge_cells[i] & edge_cells[j]
            allowed = set()
            if j == i + 1:
                allowed.add(edges[i][1])
            if overlap - allowed:
                return None
    return _normalize_coords(coords)


def _make_vertices() -> list[tuple[int, int]] | None:
    if np.random.rand() < ZIGZAG_PROB:
        return _build_zigzag4()
    return _build_elbow3()


def _candidate_pairs(points: list[tuple[int, int]]) -> list[tuple[int, tuple[int, int], tuple[int, int]]]:
    pairs: list[tuple[int, tuple[int, int], tuple[int, int]]] = []
    for i in range(len(points)):
        for j in range(i + 1, len(points)):
            a = points[i]
            b = points[j]
            if not _aligned_8(a[0], a[1], b[0], b[1]):
                continue
            aa, bb = _sorted_pair(a, b)
            pairs.append((_seg_len(aa, bb), aa, bb))
    pairs.sort(key=lambda item: (item[0], item[1], item[2]))
    return pairs


def _choose_shadow(
    vertices: list[tuple[int, int]],
    pair: tuple[tuple[int, int], tuple[int, int]],
    ray_steps: int,
) -> tuple[tuple[int, int], tuple[int, int]] | None:
    a, b = pair
    vertex_set = set(vertices)
    for anchor, other in ((a, b), (b, a)):
        drs, dcs = _step(anchor, other)
        main_cells = set(_ray_cells(anchor[0], anchor[1], drs, dcs, ray_steps))
        candidates: list[tuple[int, int]] = []
        for dr, dc in DIR8:
            if drs * dc - dcs * dr == 0:
                continue
            pos = (anchor[0] + dr, anchor[1] + dc)
            if pos in vertex_set or pos in main_cells:
                continue
            if sum(1 for vr, vc in vertices if max(abs(pos[0] - vr), abs(pos[1] - vc)) == 1) != 1:
                continue
            shifted = {(r + dr, c + dc) for r, c in main_cells}
            if shifted & main_cells:
                continue
            if shifted & vertex_set:
                continue
            candidates.append((dr, dc))
        if candidates:
            return anchor, candidates[int(np.random.randint(len(candidates)))]
    return None


def _patch_bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    arr = np.array(coords, dtype=int)
    r0, c0 = arr.min(axis=0)
    r1, c1 = arr.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _make_shape_patch(shape_color: int, shadow_color: int | None) -> tuple[np.ndarray, np.ndarray] | None:
    for _ in range(SHAPE_BUILD_RETRIES):
        vertices = _make_vertices()
        if vertices is None or len(vertices) < 3:
            continue

        candidates = _candidate_pairs(vertices)
        if len(candidates) < 2:
            continue
        if candidates[0][0] == candidates[1][0]:
            continue

        pair = (candidates[0][1], candidates[0][2])
        intended_edges = {_sorted_pair(a, b) for a, b in _polyline_edges(vertices)}
        if _sorted_pair(*pair) not in intended_edges:
            continue

        ray_steps = len(vertices)
        if candidates[0][0] <= ray_steps:
            continue

        start = pair[0]
        end = pair[1]
        shadow = None
        if shadow_color is not None:
            shadow = _choose_shadow(vertices, pair, ray_steps)
            if shadow is None:
                continue
            start = shadow[0]
            end = pair[1] if start == pair[0] else pair[0]

        drs, dcs = _step(start, end)
        main_cells = list(_ray_cells(start[0], start[1], drs, dcs, ray_steps))
        if any(cell in set(vertices) and cell != start for cell in main_cells[1:]):
            continue

        coords = list(vertices) + main_cells
        if shadow is not None:
            anchor, offset = shadow
            coords.append((anchor[0] + offset[0], anchor[1] + offset[1]))
            coords.extend((r + offset[0], c + offset[1]) for r, c in main_cells)
        r0, c0, r1, c1 = _patch_bbox(coords)
        h, w = r1 - r0 + 1, c1 - c0 + 1

        inp = np.full((h, w), BG, dtype=int)
        for r, c in vertices:
            inp[r - r0, c - c0] = shape_color

        if shadow is not None:
            anchor, offset = shadow
            sa = (anchor[0] + offset[0] - r0, anchor[1] + offset[1] - c0)
            if not (0 <= sa[0] < h and 0 <= sa[1] < w):
                continue
            if int(inp[sa]) != BG:
                continue
            inp[sa] = int(shadow_color)

        out = inp.copy()
        for r, c in main_cells:
            rr = r - r0
            cc = c - c0
            if int(out[rr, cc]) == BG:
                out[rr, cc] = shape_color

        if shadow is not None:
            dr, dc = shadow[1]
            for r, c in main_cells:
                rr = r + dr - r0
                cc = c + dc - c0
                if int(out[rr, cc]) == BG:
                    out[rr, cc] = int(shadow_color)

        if not np.array_equal(main(inp), out):
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
    shadow_colors = palette[shape_count : shape_count + shadow_count]
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
    h = int(np.random.randint(*CANVAS_H_RANGE))
    w = int(np.clip(h + np.random.randint(*CANVAS_W_JITTER), *CANVAS_W_CLIP))
    return h, w


def _gen_attempt() -> np.ndarray | None:
    h, w = _sample_canvas_size()
    out = np.full((h, w), BG, dtype=int)
    inp = np.full((h, w), BG, dtype=int)

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
