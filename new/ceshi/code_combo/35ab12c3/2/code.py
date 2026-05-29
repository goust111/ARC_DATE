# Combo variant of 35ab12c3 (index 2)
# combo_spec: D3/0+D5/0
# Name: 最近邻贪心连接+无阴影
#
# Design analysis for the fused variant
# a) Concrete behavior and trace:
#    - Each multi-pixel non-background color is still a candidate shape vertex set.
#    - Original task:
#      1. Same-color vertices were angle-sorted around a centroid.
#      2. 3+ vertices formed a closed polygon/cycle.
#      3. A foreign 8-neighbor near the chosen start vertex was propagated as a
#         parallel shadow along every edge.
#    - Fused variant:
#      1. Start from the smallest visible vertex `(min r, min c)`.
#      2. Repeatedly connect to the nearest unvisited same-color vertex
#         (distance squared, ties by `(r,c)` only as a stability fallback).
#      3. Stop after all vertices are visited; this is an OPEN polyline.
#      4. Do not propagate any foreign-color neighbor at all; lone adjacent
#         decoy pixels stay unchanged.
#    - Small example:
#      vertices = {(0,0), (0,4), (4,4), (4,8)}, plus a blue pixel at (-1,0)+anchor.
#      Original output: a closed 4-edge cycle plus a blue shifted copy.
#      Fused output:    (0,0)->(0,4)->(4,4)->(4,8) only; the blue pixel remains
#      a single pixel. This combines both changes in one observable pattern.
# b) Complexity calibration:
#    - Using the old rectangle/diamond generator would often make greedy order look
#      similar to angle order, weakening D3. The fused generator therefore produces
#      3-5 vertex monotone turning paths whose greedy open-chain cells differ from
#      the old angle-cycle cells.
#    - D5/0 could become invisible if no foreign adjacent pixels exist, so the
#      generator guarantees at least one single-pixel decoy color per grid.
#    - The task is still not trivial: the solver must infer grouping by color,
#      greedy nearest-neighbor traversal, and that foreign adjacent pixels are inert.
# c) Hidden-assumption audit:
#    - Original `_angle_cycle()` and "close the loop when len>2" are invalid here;
#      replaced by greedy traversal and consecutive edges only.
#    - Original start selection preferred vertices touching a foreign color so the
#      shadow offset could be read. That premise disappears because there is no
#      shadow mechanism, so the start is simply the smallest vertex.
#    - Original `carried` collection and shifted parallel drawing phase are removed.
#    - `_clear_in_original()` still assumes intermediate cells are background or the
#      same color. The generator enforces that each intended greedy segment does not
#      pass through a third vertex, avoiding ambiguous hidden path structure.
# d) Placement legality:
#    - Each local patch is built output-first: draw the intended greedy polyline in
#      `out`, keep only vertices in `inp`, then optionally add one inert decoy pixel
#      that is outside the path and 8-adjacent to exactly one vertex.
#    - Global placement still uses `random_free_location(..., padding=1)` so patches
#      do not overlap and remain separated by background.
# e) Visual observability:
#    - Humans can see that same-color points become one open broken line, not a
#      closed polygon, and nearby foreign singletons do not echo into a shadow.
#    - No hidden dependence on color ids is used for ordering. The generator rejects
#      distance ties so examples do not rely on imperceptible tie-breaking.
# f) Visual cleanliness:
#    - Open monotone paths avoid self-overcrowding, and decoys are isolated single
#      pixels. One-cell padding between patches keeps different groups separated.
# g) Color inference / non-degeneracy:
#    - No color remapping is introduced. Colors retain their identity.
#    - The generator rejects shapes whose greedy-output cells equal the old
#      angle-cycle cells, so the D3 change is genuinely exercised.
#    - It also guarantees at least one decoy color, so the D5 removal is visible.

import math

import numpy as np

from arc_utils import bbox_coords, colors_present, draw_line, get_bg as _bg_color
from arc_utils import get_bg, paste, random_free_location

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


def _aligned(dr: int, dc: int) -> bool:
    return (dr == 0 and dc != 0) or (dc == 0 and dr != 0) or (dr != 0 and abs(dr) == abs(dc))


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


def _dist2(a: tuple[int, int], b: tuple[int, int]) -> int:
    dr = a[0] - b[0]
    dc = a[1] - b[1]
    return dr * dr + dc * dc


def _greedy_path(points_rc: list[tuple[int, int]], start: tuple[int, int]) -> list[tuple[int, int]]:
    path = [start]
    unvisited = {p for p in points_rc}
    unvisited.remove(start)
    current = start
    while unvisited:
        nxt = min(unvisited, key=lambda p: (_dist2(current, p), p[0], p[1]))
        path.append(nxt)
        unvisited.remove(nxt)
        current = nxt
    return path


def _greedy_path_trace(
    points_rc: list[tuple[int, int]],
    start: tuple[int, int],
) -> tuple[list[tuple[int, int]], bool]:
    path = [start]
    unique_nearest = True
    unvisited = {p for p in points_rc}
    unvisited.remove(start)
    current = start
    while unvisited:
        ranked = sorted(unvisited, key=lambda p: (_dist2(current, p), p[0], p[1]))
        if len(ranked) > 1 and _dist2(current, ranked[0]) == _dist2(current, ranked[1]):
            unique_nearest = False
        current = ranked[0]
        path.append(current)
        unvisited.remove(current)
    return path, unique_nearest


def _path_edges(vertices: list[tuple[int, int]]) -> list[tuple[tuple[int, int], tuple[int, int]]]:
    return [(vertices[i], vertices[i + 1]) for i in range(len(vertices) - 1)]


def _cells_from_edges(edges: list[tuple[tuple[int, int], tuple[int, int]]]) -> set[tuple[int, int]]:
    cells: set[tuple[int, int]] = set()
    for a, b in edges:
        cells.update(_segment_cells(a[0], a[1], b[0], b[1]))
    return cells


def _angle_edges(vertices: list[tuple[int, int]]) -> list[tuple[tuple[int, int], tuple[int, int]]]:
    cycle = _angle_cycle(vertices)
    if len(cycle) == 2:
        return [(cycle[0], cycle[1])]
    return [(cycle[i], cycle[(i + 1) % len(cycle)]) for i in range(len(cycle))]


# Overview:
#   Connect each same-color vertex set by an open nearest-neighbor greedy path.
# Input structure:
#   Background 0; a shape color appears on 3-5 sparse vertices (or 2 for a line),
#   and some extra colors may appear as inert single-pixel decoys/distractors.
# Rule:
#   For each color with at least two pixels, start from its smallest vertex and
#   repeatedly connect to the nearest unvisited same-color vertex using only
#   drawable H/V/45deg segments through background or same-color cells. Do not
#   propagate any adjacent foreign-color pixel.
# Key insight:
#   The output differs from the original task in two visible ways at once:
#   shapes become open greedy chains rather than closed angle cycles, and foreign
#   adjacent markers stay singletons instead of generating a parallel shadow copy.
def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    bg = _bg_color(g)
    out = g.copy()

    colors = sorted(set(int(v) for v in np.unique(g)) - {bg})
    for col in colors:
        pts_rc = [tuple(map(int, p)) for p in np.argwhere(g == col)]
        if len(pts_rc) < 2:
            continue

        start = min(pts_rc)
        path = _greedy_path(pts_rc, start)
        edges = _path_edges(path)

        for (r0, c0), (r1, c1) in edges:
            if not _aligned_8(r0, c0, r1, c1):
                continue
            if not _clear_in_original(g, r0, c0, r1, c1, bg, allow={col}):
                continue
            _draw_line_bg_only(out, r0, c0, r1, c1, col, bg)

    return out


# --------------- generator ---------------

BG = 0
COLORS = [1, 2, 3, 4, 6, 7, 8]
CANVAS_H_RANGE = (15, 23)
CANVAS_W_JITTER = (-3, 4)
CANVAS_W_CLIP = (15, 22)

SHAPE_COUNT_RANGE = (2, 5)
MAX_DECOYS_CAP = 3
DISTRACTOR_PROB = 0.5
PLACEMENT_PADDING = 1
PLACEMENT_PAD_CONNECTIVITY = 8

PATH_VERTICES_RANGE = (3, 6)
PATH_TOTAL_LENGTH_MAX = 10
PATH_MIN_GAP = 2
PATH_BUILD_RETRIES = 80
PATH_DIRECTIONS = [(0, 1), (1, 0), (1, 1)]

DISTRACTOR_COUNT_RANGE = (2, 4)
DISTRACTOR_BBOX_RANGE = (2, 5)
DISTRACTOR_RETRIES = 50

SHAPE_PLACEMENT_RETRIES = 40
GENERATE_RETRIES = 300


def _step(a: tuple[int, int], b: tuple[int, int]) -> tuple[int, int]:
    dr = b[0] - a[0]
    dc = b[1] - a[1]
    return (0 if dr == 0 else (1 if dr > 0 else -1), 0 if dc == 0 else (1 if dc > 0 else -1))


def _gen_segment_cells(a: tuple[int, int], b: tuple[int, int]) -> list[tuple[int, int]]:
    drs, dcs = _step(a, b)
    steps = max(abs(b[0] - a[0]), abs(b[1] - a[1]))
    return [(a[0] + drs * k, a[1] + dcs * k) for k in range(steps + 1)]


def _shape_cells(vertices: list[tuple[int, int]]) -> set[tuple[int, int]]:
    cells: set[tuple[int, int]] = set()
    for a, b in _path_edges(vertices):
        cells.update(_gen_segment_cells(a, b))
    return cells


def _angle_no_shadow_main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    bg = _bg_color(g)
    out = g.copy()

    colors = sorted(set(int(v) for v in np.unique(g)) - {bg})
    for col in colors:
        pts_rc = [tuple(map(int, p)) for p in np.argwhere(g == col)]
        if len(pts_rc) < 2:
            continue

        cycle = _angle_cycle(pts_rc)
        edges = [(cycle[0], cycle[1])] if len(cycle) == 2 else [
            (cycle[i], cycle[(i + 1) % len(cycle)]) for i in range(len(cycle))
        ]

        for (r0, c0), (r1, c1) in edges:
            if not _aligned_8(r0, c0, r1, c1):
                continue
            if not _clear_in_original(g, r0, c0, r1, c1, bg, allow={col}):
                continue
            _draw_line_bg_only(out, r0, c0, r1, c1, col, bg)

    return out


def _segment_hits_other_vertex(
    a: tuple[int, int],
    b: tuple[int, int],
    vertices: list[tuple[int, int]],
) -> bool:
    verts = set(vertices)
    for rr, cc in _gen_segment_cells(a, b)[1:-1]:
        if (rr, cc) in verts:
            return True
    return False


def _sample_gaps(n_segments: int) -> list[int] | None:
    min_total = PATH_MIN_GAP * n_segments
    if min_total > PATH_TOTAL_LENGTH_MAX:
        return None
    total = int(np.random.randint(min_total, PATH_TOTAL_LENGTH_MAX + 1))
    remaining = total
    gaps: list[int] = []
    for i in range(n_segments):
        steps_left = n_segments - 1 - i
        hi = remaining - PATH_MIN_GAP * steps_left
        gap = int(np.random.randint(PATH_MIN_GAP, hi + 1))
        gaps.append(gap)
        remaining -= gap
    return gaps


def _build_monotone_vertices() -> list[tuple[int, int]] | None:
    n_vertices = int(np.random.randint(*PATH_VERTICES_RANGE))
    n_segments = n_vertices - 1
    gaps = _sample_gaps(n_segments)
    if gaps is None:
        return None

    directions: list[tuple[int, int]] = []
    for i in range(n_segments):
        candidates = list(PATH_DIRECTIONS)
        if directions:
            prev = directions[-1]
            if len(candidates) > 1 and np.random.rand() < 0.75:
                candidates = [d for d in candidates if d != prev]
            if not candidates:
                candidates = list(PATH_DIRECTIONS)
            if i == n_segments - 1 and len(set(directions)) == 1:
                forced = [d for d in candidates if d != prev]
                if forced:
                    candidates = forced
        direction = candidates[int(np.random.randint(len(candidates)))]
        directions.append(direction)

    if len(set(directions)) < 2:
        return None

    vertices = [(0, 0)]
    for (dr, dc), gap in zip(directions, gaps):
        last = vertices[-1]
        vertices.append((last[0] + dr * gap, last[1] + dc * gap))

    if np.random.rand() < 0.5:
        vertices = [(c, r) for r, c in vertices]
    return vertices


def _is_variant_distinct(vertices: list[tuple[int, int]]) -> bool:
    greedy_cells = _cells_from_edges(_path_edges(vertices))
    angle_cells = _cells_from_edges(_angle_edges(vertices))
    return greedy_cells != angle_cells


def _make_vertices() -> list[tuple[int, int]] | None:
    for _ in range(PATH_BUILD_RETRIES):
        vertices = _build_monotone_vertices()
        if vertices is None:
            continue
        if vertices[0] != min(vertices):
            continue
        greedy, unique_nearest = _greedy_path_trace(vertices, vertices[0])
        if greedy != vertices or not unique_nearest:
            continue
        if any(not _aligned_8(a[0], a[1], b[0], b[1]) for a, b in _path_edges(vertices)):
            continue
        if any(_segment_hits_other_vertex(a, b, vertices) for a, b in _path_edges(vertices)):
            continue
        if not _is_variant_distinct(vertices):
            continue
        return vertices
    return None


def _choose_decoy_anchor(
    vertices: list[tuple[int, int]],
    shape_cells: set[tuple[int, int]],
) -> tuple[tuple[int, int], tuple[int, int]] | None:
    anchor = vertices[0]
    directions = [_step(a, b) for a, b in _path_edges(vertices)]
    candidates: list[tuple[int, int]] = []
    for dr, dc in DIR8:
        if any(ds_r * dc - ds_c * dr == 0 for ds_r, ds_c in directions):
            continue
        pos = (anchor[0] + dr, anchor[1] + dc)
        if pos in shape_cells:
            continue
        hits = sum(1 for vr, vc in vertices if max(abs(pos[0] - vr), abs(pos[1] - vc)) == 1)
        if hits != 1:
            continue
        candidates.append((dr, dc))
    if not candidates:
        return None
    return anchor, candidates[int(np.random.randint(len(candidates)))]


def _patch_bbox(
    shape_cells: set[tuple[int, int]],
    vertices: list[tuple[int, int]],
    decoy: tuple[tuple[int, int], tuple[int, int]] | None,
) -> tuple[int, int, int, int]:
    coords = list(shape_cells) + vertices
    if decoy is not None:
        (ar, ac), (dr, dc) = decoy
        coords.append((ar + dr, ac + dc))
    r0, c0, r1, c1 = bbox_coords(np.array(coords, dtype=int))
    return int(r0), int(c0), int(r1), int(c1)


def _make_shape_patch(shape_color: int, decoy_color: int | None) -> tuple[np.ndarray, np.ndarray] | None:
    vertices = _make_vertices()
    if vertices is None:
        return None
    cells = _shape_cells(vertices)

    decoy = None
    if decoy_color is not None:
        decoy = _choose_decoy_anchor(vertices, cells)
        if decoy is None:
            return None

    r0, c0, r1, c1 = _patch_bbox(cells, vertices, decoy)
    h, w = r1 - r0 + 1, c1 - c0 + 1
    out = np.full((h, w), BG, dtype=int)
    inp = np.full((h, w), BG, dtype=int)

    shifted_vertices = [(r - r0, c - c0) for r, c in vertices]
    for a, b in _path_edges(shifted_vertices):
        draw_line(out, a[0], a[1], end_r=b[0], end_c=b[1], color=shape_color)
    for r, c in shifted_vertices:
        inp[r, c] = shape_color

    if decoy is not None:
        (ar, ac), offset = decoy
        marker = (ar - r0 + offset[0], ac - c0 + offset[1])
        if not (0 <= marker[0] < h and 0 <= marker[1] < w):
            return None
        if inp[marker] != BG or out[marker] != BG:
            return None
        inp[marker] = int(decoy_color)
        out[marker] = int(decoy_color)

    if np.array_equal(_angle_no_shadow_main(inp), out):
        return None
    if not np.array_equal(main(inp), out):
        return None
    return inp, out


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


def _make_distractor_patch(color: int) -> tuple[np.ndarray, np.ndarray] | None:
    count = int(np.random.randint(*DISTRACTOR_COUNT_RANGE))
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
    return None


def _sample_palette(shape_count: int, has_distractor: bool) -> tuple[list[int], list[int], int | None] | None:
    max_decoys = min(MAX_DECOYS_CAP, shape_count, len(COLORS) - shape_count - int(has_distractor))
    if max_decoys < 1:
        return None
    decoy_count = int(np.random.randint(1, max_decoys + 1))
    total = shape_count + decoy_count + int(has_distractor)
    palette = list(np.random.permutation(COLORS)[:total])
    shape_colors = palette[:shape_count]
    decoy_colors = palette[shape_count : shape_count + decoy_count]
    distractor_color = int(palette[-1]) if has_distractor else None
    return shape_colors, decoy_colors, distractor_color


def _assign_decoys(shape_count: int, decoy_colors: list[int]) -> dict[int, int]:
    targets = np.random.choice(shape_count, size=len(decoy_colors), replace=False).tolist()
    return {int(idx): int(col) for idx, col in zip(sorted(targets), decoy_colors)}


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


def _gen_attempt(has_distractor: bool | None = None) -> np.ndarray | None:
    h, w = _sample_canvas_size()
    out = np.full((h, w), BG, dtype=int)
    inp = np.full((h, w), BG, dtype=int)

    if has_distractor is None:
        has_distractor = bool(np.random.rand() < DISTRACTOR_PROB)
    shape_count = int(np.random.randint(*SHAPE_COUNT_RANGE))

    palette = _sample_palette(shape_count, has_distractor)
    if palette is None:
        return None
    shape_colors, decoy_colors, distractor_color = palette
    decoy_map = _assign_decoys(shape_count, decoy_colors)

    total_colors = shape_count + len(decoy_colors) + int(has_distractor)

    for idx, shape_color in enumerate(shape_colors):
        placed = False
        for _ in range(SHAPE_PLACEMENT_RETRIES):
            patch = _make_shape_patch(int(shape_color), decoy_map.get(idx))
            if patch is None:
                continue
            if _try_place(inp, out, *patch):
                placed = True
                break
        if not placed:
            return None

    if has_distractor and distractor_color is not None:
        distractor_patch = _make_distractor_patch(distractor_color)
        if distractor_patch is None:
            return None
        if not _try_place(inp, out, *distractor_patch):
            return None

    if get_bg(inp) != BG or get_bg(out) != BG:
        return None
    if len(colors_present(inp, exclude=[BG])) != total_colors:
        return None
    if not np.array_equal(main(inp), out):
        return None
    return inp


def generate_input() -> np.ndarray:
    # Choose the distractor branch first, then retry within that branch.
    # This avoids successful samples being skewed toward whichever branch is
    # easier to place on a given canvas.
    target_has_distractor = bool(np.random.rand() < DISTRACTOR_PROB)
    for _ in range(GENERATE_RETRIES):
        result = _gen_attempt(target_has_distractor)
        if result is not None:
            return result
    raise RuntimeError(f"generate_input failed after {GENERATE_RETRIES} attempts")
