# Variant of 35ab12c3
# Dimension: D4 — 轨迹涂色模式 (trajectory)
# Change: continuous_trail——沿路径的每个背景格子都涂上形状颜色，形成连续实线... → 仅涂顶点不涂边

# Design analysis for this variant
# --------------------------------
# a) Concrete behavior and trace
#    - Original task: each same-color vertex set is ordered into a cycle/polyline,
#      every valid H/V/45deg edge is filled cell-by-cell, and a single adjacent
#      foreign-color "shadow anchor" is copied along parallel edges.
#    - Variant: keep the same shape detection and the same shadow-offset inference,
#      but invert the rendering stage.
#      Input still contains only shape vertices plus one shadow-anchor pixel.
#      Output no longer fills any edge interiors. Instead:
#        1. every detected vertex is highlighted by a small plus marker
#           {(0,0),(-1,0),(1,0),(0,-1),(0,1)};
#        2. every carried shadow offset is applied to every vertex center, and each
#           translated center receives the same plus marker in the shadow color.
#      Example trace:
#        input shape color 4 at (5,5),(5,9), shadow color 7 at (4,5).
#        Original output would paint 4 across columns 5..9 and 7 across 4..8.
#        Variant output paints a plus around (5,5) and (5,9), and a plus around
#        (4,5) and (4,9); cells between the vertices stay background.
#    - The changed stage is strictly the rendering stage of both solver and
#      output-first generator. Vertex ordering, anchor choice, and shadow-offset
#      discovery stay aligned with the original task.
#
# b) Complexity calibration
#    - A single center pixel per vertex would be too weak: the output would often
#      equal the input except for copied shadow centers, making the rule harder to
#      see. The plus marker makes the transformation visually obvious.
#    - A 3x3 square marker would create too much overlap/clutter, so the plus is a
#      controlled five-cell marker.
#    - To keep markers distinct, the generator raises geometric minima (segment and
#      side lengths >= 3 where needed) and rejects any patch whose plus markers
#      overlap within the same layer or across shape/shadow colors.
#
# c) Hidden assumptions audit against original main()
#    - _aligned_8/_clear_in_original: these still matter, because we still need to
#      distinguish true shapes from distractors and preserve the original notion of
#      which vertex sequence is drawable.
#    - _draw_line_bg_only: this was the old semantic core and is no longer valid.
#      Replaced by _paint_marker_bg_only, which only stamps local plus markers.
#    - Phase 1/2 edge loops assumed edge-cell filling. In the variant, edges are
#      still computed only as a recognition filter; rendering uses the vertex set.
#    - out = g.copy() remains safe because input already contains the vertex centers
#      and the shadow anchor center. Marker arms are only painted on background, so
#      unrelated colors are never overwritten.
#    - Colors with <2 pixels are still skipped as primary shapes, which is required
#      so single shadow-anchor pixels and 1-pixel distractors are not misread as
#      shapes.
#
# d) Placement legality
#    - Generator remains output-first in spirit: build a local patch whose output is
#      the union of vertex markers (and optional shadow markers), then derive the
#      input by keeping only vertex centers and the single shadow anchor center.
#    - Shadow offsets are still chosen from an 8-neighbor of the anchor vertex,
#      must not land on the original shape skeleton, and must be 8-adjacent to
#      exactly one vertex.
#    - Additional legality checks reject marker overlap, cross-color collisions,
#      border clipping inside patches, and any patch that disagrees with main().
#
# e) Visual observability
#    - Across 2-4 examples, the rule is visible: inputs show sparse vertices, while
#      outputs show only local plus signs at those vertices and at the copied shadow
#      positions. The absence of connecting trails is the dominant, repeated cue.
#    - No invisible attribute such as color id ordering is used for the new rule.
#
# f) Visual cleanliness
#    - The plus marker is local and sparse.
#    - The generator rejects overlapping markers and keeps inter-object spacing via
#      random_free_location padding, so different groups stay visually separated.
#
# g) Color-change inferability
#    - Colors are not remapped. Each shape keeps its own color, and an optional
#      shadow color still behaves as a translated companion of one anchor pixel.
#      The only semantic change is where those colors are painted.
#
# New solver-side assumptions introduced by this variant
# -----------------------------------------------------
# 1. Every intended shape has the full expected set of drawable edges after the same
#    angle-cycle ordering as the original task. The generator guarantees this.
# 2. Every intended shadow offset is non-collinear with all edges of its shape and
#    yields non-overlapping translated vertex markers. The generator guarantees this.
# 3. Marker arms have room to remain legible. The generator enforces this by larger
#    geometric minima and explicit overlap rejection.

import math

import numpy as np

from arc_utils import bbox_coords, colors_present, get_bg, neighbors8, paste, random_free_location
from arc_utils import get_bg as _bg_color

DIR8 = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
MARKER_OFFSETS = [(0, 0), (-1, 0), (1, 0), (0, -1), (0, 1)]


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


def _marker_cells(r: int, c: int):
    for dr, dc in MARKER_OFFSETS:
        yield r + dr, c + dc


def _paint_marker_bg_only(out: np.ndarray, r: int, c: int, col: int, bg: int) -> None:
    h, w = out.shape
    for rr, cc in _marker_cells(r, c):
        if 0 <= rr < h and 0 <= cc < w and out[rr, cc] == bg:
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

        expected_edges = 1 if len(cycle) == 2 else len(cycle)
        if len(drawable_edges) != expected_edges:
            continue

        for r, c in cycle:
            _paint_marker_bg_only(out, r, c, col, bg)

        edge_dirs = [_step_dir(r0, c0, r1, c1) for (r0, c0), (r1, c1) in drawable_edges]
        for drn, dcn, ncol in carried:
            if any(drs * dcn - dcs * drn == 0 for drs, dcs in edge_dirs):
                continue
            for vr, vc in cycle:
                rr = vr + drn
                cc = vc + dcn
                if 0 <= rr < h and 0 <= cc < w:
                    _paint_marker_bg_only(out, rr, cc, ncol, bg)

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

LINE_VERTICES_RANGE = (2, 6)
LINE_TOTAL_LENGTH = 15
LINE_MIN_GAP = 3
TRIANGLE_SIDE_RANGE = (3, 8)
RECT_H_RANGE = (3, 9)
RECT_W_RANGE = (3, 9)
DIAMOND_ARM_RANGE = (3, 6)
PENTAGON_ROOF_RANGE = (3, 5)
PENTAGON_WALL_RANGE = (3, 7)
HEXAGON_SLANT_RANGE = (3, 5)
HEXAGON_WALL_RANGE = (3, 6)

SHAPE_WEIGHTS = np.array([0.28, 0.18, 0.18, 0.14, 0.12, 0.10], dtype=float)

DISTRACTOR_COUNT_RANGE = (1, 4)
DISTRACTOR_BBOX_RANGE = (2, 5)
DISTRACTOR_RETRIES = 50

SHAPE_PLACEMENT_RETRIES = 40
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


_SHAPE_BUILDERS = [_build_line, _build_triangle, _build_rectangle, _build_diamond, _build_pentagon, _build_hexagon]


def _make_vertices() -> list[tuple[int, int]]:
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


def _edge_dirs(vertices: list[tuple[int, int]]) -> list[tuple[int, int]]:
    return [_step(a, b) for a, b in _shape_edges(vertices)]


def _shadow_candidates(
    vertices: list[tuple[int, int]],
    shape_cells: set[tuple[int, int]],
) -> list[tuple[tuple[int, int], tuple[int, int]]]:
    anchor = min(vertices)
    directions = _edge_dirs(vertices)
    candidates: list[tuple[tuple[int, int], tuple[int, int]]] = []
    for dr, dc in DIR8:
        if any(ds_r * dc - ds_c * dr == 0 for ds_r, ds_c in directions):
            continue
        pos = (anchor[0] + dr, anchor[1] + dc)
        if pos in shape_cells:
            continue
        if sum(1 for vr, vc in vertices if max(abs(pos[0] - vr), abs(pos[1] - vc)) == 1) != 1:
            continue
        candidates.append((anchor, (dr, dc)))
    if candidates:
        order = np.random.permutation(len(candidates))
        candidates = [candidates[int(i)] for i in order]
    return candidates


def _marker_cell_set(center: tuple[int, int]) -> set[tuple[int, int]]:
    r, c = center
    return {(r + dr, c + dc) for dr, dc in MARKER_OFFSETS}


def _markers_pairwise_disjoint(centers: list[tuple[int, int]]) -> bool:
    seen: set[tuple[int, int]] = set()
    for center in centers:
        cells = _marker_cell_set(center)
        if seen & cells:
            return False
        seen |= cells
    return True


def _paint_marker_strict(out: np.ndarray, center: tuple[int, int], color: int) -> bool:
    h, w = out.shape
    for rr, cc in _marker_cells(center[0], center[1]):
        if not (0 <= rr < h and 0 <= cc < w):
            return False
        if out[rr, cc] not in (BG, color):
            return False
    for rr, cc in _marker_cells(center[0], center[1]):
        out[rr, cc] = color
    return True


def _patch_bbox(
    vertices: list[tuple[int, int]],
    shadow_centers: list[tuple[int, int]],
) -> tuple[int, int, int, int]:
    coords = set(vertices) | set(shadow_centers)
    for center in list(coords):
        coords.update(_marker_cell_set(center))
    r0, c0, r1, c1 = bbox_coords(np.array(sorted(coords), dtype=int))
    return int(r0), int(c0), int(r1), int(c1)


def _make_shape_patch(shape_color: int, shadow_color: int | None) -> tuple[np.ndarray, np.ndarray] | None:
    vertices = _make_vertices()
    cells = _shape_cells(vertices)

    shadow_options: list[tuple[tuple[int, int], tuple[int, int]]] = [None]  # type: ignore[list-item]
    if shadow_color is not None:
        shadow_options = _shadow_candidates(vertices, cells)
        if not shadow_options:
            return None

    for shadow in shadow_options:
        shadow_centers: list[tuple[int, int]] = []
        if shadow is not None:
            (ar, ac), offset = shadow
            shadow_centers = [(r + offset[0], c + offset[1]) for r, c in vertices]
        if not _markers_pairwise_disjoint(vertices):
            return None
        if shadow_centers and not _markers_pairwise_disjoint(shadow_centers):
            continue

        r0, c0, r1, c1 = _patch_bbox(vertices, shadow_centers)
        h, w = r1 - r0 + 1, c1 - c0 + 1
        out = np.full((h, w), BG, dtype=int)
        inp = np.full((h, w), BG, dtype=int)

        shifted_vertices = [(r - r0, c - c0) for r, c in vertices]
        shifted_shadow_centers = [(r - r0, c - c0) for r, c in shadow_centers]

        valid = True
        for r, c in shifted_vertices:
            inp[r, c] = shape_color
            if not _paint_marker_strict(out, (r, c), shape_color):
                valid = False
                break
        if not valid:
            continue

        if shadow is not None:
            anchor = (shadow[0][0] - r0, shadow[0][1] - c0)
            offset = shadow[1]
            shadow_anchor = (anchor[0] + offset[0], anchor[1] + offset[1])
            if not (0 <= shadow_anchor[0] < h and 0 <= shadow_anchor[1] < w):
                continue
            if inp[shadow_anchor] != BG:
                continue
            inp[shadow_anchor] = int(shadow_color)
            for center in shifted_shadow_centers:
                if not _paint_marker_strict(out, center, int(shadow_color)):
                    valid = False
                    break
            if not valid:
                continue
            nb_hits = sum(
                1
                for rr, cc in neighbors8(anchor[0], anchor[1])
                if 0 <= rr < h and 0 <= cc < w and inp[rr, cc] == shadow_color
            )
            if nb_hits != 1:
                continue

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
