# Variant of 35ab12c3
# Dimension: D21 — 边界处理 (meta)
# Change: 阴影超出网格边界时直接截断（不画超出部分），使用bounds check确保所有坐标在[0,H)x[0,W)内... → 边界反射

from __future__ import annotations

import math

import numpy as np

from arc_utils import colors_present, get_bg, neighbors8


"""
Design analysis for the D21 reflection variant
==============================================

a) Concrete behavior and where it changes
-----------------------------------------
Original task:
- A shape color contributes a cycle/polyline of vertices.
- Main edges are drawn directly between same-color vertices when H/V/45deg aligned.
- If the chosen start vertex has a differently colored 8-neighbor, that neighbor
  defines a shadow offset. Each drawable main edge is copied in parallel by the
  same offset, but the old code only kept that copied edge when both shifted
  endpoints stayed inside the grid.

Variant:
- Main edges are unchanged.
- Only shadow drawing changes. For every cell on a drawable main edge, first apply
  the shadow offset, then mirror any out-of-bounds row/col back into the grid.
  Reflection is per-axis. If a translated cell would go above the top boundary it
  bounces downward; if it would go beyond the right boundary it bounces leftward;
  if both axes leave at once it makes a corner bounce.
- The reflected shadow can therefore form a V/kink near a border instead of being
  clipped away.

Small trace:
- Suppose an edge occupies (0,3),(1,3),(2,3),(3,3) and the shadow offset is (0,1)
  in a width-5 grid.
- Old logic shifts to col 4 for every cell until the source edge hits col 4; any
  shifted endpoint beyond col 4 causes the whole copied edge to be skipped.
- New logic translates the per-cell path to cols 4,4,4,4 for this example if still
  in-bounds, but for an edge that reaches the border and needs col 5, the mirrored
  coordinate becomes col 3 instead of disappearing. The output visibly differs from
  truncation because extra shadow cells appear inside the grid after the bounce.

b) Complexity calibration
-------------------------
- The variant changes exactly one mechanism: shadow boundary handling.
- The generator forces at least one shadow in every sample, and at least one of
  those shadows must differ from the old clipped output. Without that, D21 would be
  invisible.
- The shape family, counts, colors, vertex ordering, and anchor semantics remain
  close to the original task so the puzzle does not become a different task.

c) Old assumptions that break under reflection
----------------------------------------------
Relevant old shadow code assumptions:
- `rr0 = r0 + drn`, `rr1 = r1 + drn`, ... followed by one bounds check assumed the
  shadow copy is still a single straight in-bounds segment. That is false now.
- `_aligned_8(rr0, cc0, rr1, cc1)` on shifted endpoints assumed the copied edge
  stays H/V/diag after translation. Under reflection the path can kink, so endpoint
  alignment is no longer meaningful.
- `_draw_line_bg_only(...)` assumed one straight segment was enough to render the
  shadow. Under reflection we must draw a per-cell reflected path instead.

What remains valid:
- Main-edge detection, angle ordering, seed selection, and the "skip collinear
  shadow offsets" rule all still hold.
- Background-only drawing still resolves overlaps cleanly: main shape wins over
  shadow, and previously painted shadow cells are preserved.

d) Placement legality in the generator
--------------------------------------
- Because reflection depends on the global canvas border, local patch generation is
  no longer sufficient. The generator places each shape directly in the full canvas.
- Candidate placement is accepted only if:
  1. the input pixels are in-bounds and non-overlapping,
  2. the solver output for that single candidate shape fits with a >=1-cell 8-neigh
     gap from already placed output regions,
  3. for the designated reflective shadow, the reflected output differs from the old
     clipped solver on the same input.
- This guarantees the final composed input is still solver-friendly and visually
  separated.

e) Visual observability
-----------------------
- Every sample has at least one shadow, and at least one shadow must activate the
  new border reflection rule.
- The reflection is observable through extra shadow cells near a border; this is a
  visible spatial effect, not a hidden color-id convention.

f) Visual cleanliness
---------------------
- Shapes are spaced using their full solver output footprints, not just input
  vertices, so reflected shadows do not tangle with unrelated groups.
- Only one shadow layer per shape is kept, matching the original generator style and
  avoiding excessive overlaps near borders.

g) Color inference
------------------
- Colors remain simple: one unique color per shape, one unique color per shadow, and
  an optional distractor color. No color remapping rule is introduced.

New solver assumptions introduced by this variant
-------------------------------------------------
- The generator must still ensure that only one vertex of a shadowed shape is
  8-adjacent to the shadow anchor color; otherwise the chosen start vertex could be
  ambiguous.
- The generator must keep distinct groups separated in the final output, because the
  solver still assumes unrelated non-bg pixels should not block or merge with a
  shape's reflected shadow path.
"""


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


def _draw_line_bg_only(out: np.ndarray, r0: int, c0: int, r1: int, c1: int, col: int, bg: int) -> None:
    for rr, cc in _segment_cells(r0, c0, r1, c1):
        if out[rr, cc] == bg:
            out[rr, cc] = col


def _draw_path_bg_only(out: np.ndarray, path: list[tuple[int, int]], col: int, bg: int) -> None:
    for rr, cc in path:
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


def _reflect_index(idx: int, size: int) -> int:
    if size <= 1:
        return 0
    period = 2 * (size - 1)
    x = idx % period
    if x > size - 1:
        x = period - x
    return int(x)


def _shadow_path_reflected(
    r0: int,
    c0: int,
    r1: int,
    c1: int,
    drn: int,
    dcn: int,
    h: int,
    w: int,
) -> list[tuple[int, int]]:
    path: list[tuple[int, int]] = []
    for rr, cc in _segment_cells(r0, c0, r1, c1):
        path.append((_reflect_index(rr + drn, h), _reflect_index(cc + dcn, w)))
    return path


def _shadow_path_clipped(
    r0: int,
    c0: int,
    r1: int,
    c1: int,
    drn: int,
    dcn: int,
    h: int,
    w: int,
) -> list[tuple[int, int]]:
    path: list[tuple[int, int]] = []
    for rr, cc in _segment_cells(r0, c0, r1, c1):
        rr2 = rr + drn
        cc2 = cc + dcn
        if 0 <= rr2 < h and 0 <= cc2 < w:
            path.append((int(rr2), int(cc2)))
    return path


def _clear_path_in_original(g: np.ndarray, path: list[tuple[int, int]], bg: int, allow: set[int]) -> bool:
    if len(path) <= 2:
        return True
    for i in range(1, len(path) - 1):
        rr, cc = path[i]
        v = int(g[rr, cc])
        if v != bg and v not in allow:
            return False
    return True


def _main_common(grid: np.ndarray, *, reflect_shadow: bool) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = int(get_bg(g))
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

        for (r0, c0), (r1, c1) in drawable_edges:
            _draw_line_bg_only(out, r0, c0, r1, c1, col, bg)

        for (r0, c0), (r1, c1) in drawable_edges:
            drs, dcs = _step_dir(r0, c0, r1, c1)
            for drn, dcn, ncol in carried:
                if drs * dcn - dcs * drn == 0:
                    continue
                if reflect_shadow:
                    path = _shadow_path_reflected(r0, c0, r1, c1, drn, dcn, h, w)
                    if not _clear_path_in_original(g, path, bg, allow=set()):
                        continue
                    _draw_path_bg_only(out, path, ncol, bg)
                else:
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


def main(grid: np.ndarray) -> np.ndarray:
    return _main_common(grid, reflect_shadow=True)


def _main_clipped(grid: np.ndarray) -> np.ndarray:
    return _main_common(grid, reflect_shadow=False)


# --------------- generator ---------------

BG = 0
COLORS = [1, 2, 3, 4, 6, 7, 8]

CANVAS_H_RANGE = (15, 23)
CANVAS_W_JITTER = (-3, 4)
CANVAS_W_CLIP = (15, 22)

SHAPE_COUNT_RANGE = (2, 5)
MAX_SHADOWS_CAP = 3
DISTRACTOR_PROB = 0.45

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

SHAPE_WEIGHTS = np.array([0.28, 0.18, 0.18, 0.14, 0.12, 0.10], dtype=float)

DISTRACTOR_COUNT_RANGE = (1, 4)
DISTRACTOR_BBOX_RANGE = (2, 5)
DISTRACTOR_RETRIES = 50

SHAPE_GEOMETRY_RETRIES = 50
SHAPE_PLACEMENT_RETRIES = 120
DISTRACTOR_PLACEMENT_RETRIES = 80
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


def _choose_shadow_offsets(vertices: list[tuple[int, int]], shape_cells: set[tuple[int, int]]) -> list[tuple[int, int]]:
    anchor = min(vertices)
    directions = _edge_dirs(vertices)
    candidates: list[tuple[int, int]] = []
    for dr, dc in DIR8:
        if any(ds_r * dc - ds_c * dr == 0 for ds_r, ds_c in directions):
            continue
        pos = (anchor[0] + dr, anchor[1] + dc)
        if pos in shape_cells:
            continue
        if sum(1 for vr, vc in vertices if max(abs(pos[0] - vr), abs(pos[1] - vc)) == 1) != 1:
            continue
        candidates.append((int(dr), int(dc)))
    return candidates


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


def _make_distractor_cells() -> set[tuple[int, int]]:
    count = int(np.random.randint(*DISTRACTOR_COUNT_RANGE))
    if count == 1:
        return {(0, 0)}
    for _ in range(DISTRACTOR_RETRIES):
        h = int(np.random.randint(*DISTRACTOR_BBOX_RANGE))
        w = int(np.random.randint(*DISTRACTOR_BBOX_RANGE))
        coords = _sample_distractor_coords(h, w, count)
        if coords is not None:
            return set(coords)
    return {(0, 0)}


def _sample_palette(shape_count: int, has_distractor: bool) -> tuple[list[int], list[int], int | None] | None:
    max_shadows = min(MAX_SHADOWS_CAP, shape_count, len(COLORS) - shape_count - int(has_distractor))
    if max_shadows < 1:
        return None
    shadow_count = int(np.random.randint(1, max_shadows + 1))
    total = shape_count + shadow_count + int(has_distractor)
    palette = list(np.random.permutation(COLORS)[:total])
    shape_colors = [int(v) for v in palette[:shape_count]]
    shadow_colors = [int(v) for v in palette[shape_count:shape_count + shadow_count]]
    distractor_color = int(palette[-1]) if has_distractor else None
    return shape_colors, shadow_colors, distractor_color


def _assign_shadows(shape_count: int, shadow_colors: list[int]) -> dict[int, int]:
    targets = np.random.choice(shape_count, size=len(shadow_colors), replace=False).tolist()
    return {int(idx): int(col) for idx, col in zip(sorted(targets), shadow_colors)}


def _shape_reflection_score(vertices: list[tuple[int, int]], offset: tuple[int, int]) -> int:
    anchor = min(vertices)
    max_r = max(r for r, _ in _shape_cells(vertices))
    max_c = max(c for _, c in _shape_cells(vertices))
    score = 0
    if offset[0] == 1:
        score += 2
    if offset[1] == 1:
        score += 2
    if offset[1] == -1 and anchor[1] > 0:
        score += 1
    if offset[0] == -1 and anchor[0] > 0:
        score += 1
    score += int(max_r > anchor[0]) + int(max_c > anchor[1])
    return score


def _candidate_locations(
    H: int,
    W: int,
    vertices: list[tuple[int, int]],
    offset: tuple[int, int] | None,
    require_reflection: bool,
) -> list[tuple[int, int]]:
    max_r = max(r for r, _ in vertices)
    max_c = max(c for _, c in vertices)
    anchor = min(vertices)
    base_r0 = 0
    base_r1 = H - 1 - max_r
    base_c0 = 0
    base_c1 = W - 1 - max_c
    if base_r1 < base_r0 or base_c1 < base_c0:
        return []

    candidates: list[tuple[int, int]] = []

    def add(loc_r: int, loc_c: int) -> None:
        if base_r0 <= loc_r <= base_r1 and base_c0 <= loc_c <= base_c1:
            if offset is None:
                candidates.append((int(loc_r), int(loc_c)))
                return
            shadow_r = loc_r + anchor[0] + offset[0]
            shadow_c = loc_c + anchor[1] + offset[1]
            if 0 <= shadow_r < H and 0 <= shadow_c < W:
                candidates.append((int(loc_r), int(loc_c)))

    if require_reflection and offset is not None:
        shape_cells = _shape_cells(vertices)
        max_sr = max(r for r, _ in shape_cells)
        max_sc = max(c for _, c in shape_cells)
        fixed_r = None
        fixed_c = None
        if offset[0] == 1:
            fixed_r = H - 1 - max_sr
            for _ in range(20):
                add(fixed_r, int(np.random.randint(base_c0, base_c1 + 1)))
        if offset[1] == 1:
            fixed_c = W - 1 - max_sc
            for _ in range(20):
                add(int(np.random.randint(base_r0, base_r1 + 1)), fixed_c)
        if offset[1] == -1 and anchor[1] > 0:
            fixed_c = 0
            for _ in range(20):
                add(int(np.random.randint(base_r0, base_r1 + 1)), 0)
        if offset[0] == -1 and anchor[0] > 0:
            fixed_r = 0
            for _ in range(20):
                add(0, int(np.random.randint(base_c0, base_c1 + 1)))
        if fixed_r is not None and fixed_c is not None:
            for _ in range(10):
                add(fixed_r, fixed_c)

    for _ in range(50):
        add(int(np.random.randint(base_r0, base_r1 + 1)), int(np.random.randint(base_c0, base_c1 + 1)))

    if not candidates:
        return []

    seen = set()
    uniq: list[tuple[int, int]] = []
    for loc in candidates:
        if loc not in seen:
            seen.add(loc)
            uniq.append(loc)
    np.random.shuffle(uniq)
    return uniq


def _render_shape_candidate(
    H: int,
    W: int,
    vertices: list[tuple[int, int]],
    shape_color: int,
    shadow_color: int | None,
    offset: tuple[int, int] | None,
    loc: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray] | None:
    inp = np.full((H, W), BG, dtype=int)
    shifted_vertices = [(r + loc[0], c + loc[1]) for r, c in vertices]
    for r, c in shifted_vertices:
        if not (0 <= r < H and 0 <= c < W):
            return None
        inp[r, c] = int(shape_color)

    if shadow_color is not None and offset is not None:
        anchor = min(shifted_vertices)
        sr = anchor[0] + offset[0]
        sc = anchor[1] + offset[1]
        if not (0 <= sr < H and 0 <= sc < W):
            return None
        if inp[sr, sc] != BG:
            return None
        inp[sr, sc] = int(shadow_color)
        nb_hits = sum(
            1
            for rr, cc in neighbors8(sr, sc)
            if 0 <= rr < H and 0 <= cc < W and inp[rr, cc] == shape_color
        )
        if nb_hits != 1:
            return None

    out = main(inp)
    return inp, out


def _cells_of(grid: np.ndarray) -> set[tuple[int, int]]:
    return {tuple(map(int, p)) for p in np.argwhere(grid != BG)}


def _touches_with_padding(cells: set[tuple[int, int]], occupied: set[tuple[int, int]], pad: int = 1) -> bool:
    if not cells or not occupied:
        return False
    for r, c in cells:
        for dr in range(-pad, pad + 1):
            for dc in range(-pad, pad + 1):
                if (r + dr, c + dc) in occupied:
                    return True
    return False


def _merge_patches(dst: np.ndarray, patch: np.ndarray) -> None:
    mask = patch != BG
    dst[mask] = patch[mask]


def _difference_from_clipped(inp: np.ndarray) -> int:
    return int(np.count_nonzero(main(inp) != _main_clipped(inp)))


def _reflection_events(inp: np.ndarray) -> int:
    g = np.asarray(inp, dtype=int)
    h, w = g.shape
    bg = int(get_bg(g))
    total = 0

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
            continue
        cycle = cycle[start:] + cycle[:start]
        sr0, sc0 = cycle[0]

        carried: list[tuple[int, int, int]] = []
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

        for (r0, c0), (r1, c1) in edges:
            if not _aligned_8(r0, c0, r1, c1):
                continue
            if not _clear_in_original(g, r0, c0, r1, c1, bg, allow={col}):
                continue
            drs, dcs = _step_dir(r0, c0, r1, c1)
            for drn, dcn, _ncol in carried:
                if drs * dcn - dcs * drn == 0:
                    continue
                for rr, cc in _segment_cells(r0, c0, r1, c1):
                    rr2 = rr + drn
                    cc2 = cc + dcn
                    if not (0 <= rr2 < h and 0 <= cc2 < w):
                        total += 1
    return total


def _place_shape(
    full_inp: np.ndarray,
    occupied_out: set[tuple[int, int]],
    H: int,
    W: int,
    shape_color: int,
    shadow_color: int | None,
    require_reflection: bool,
) -> tuple[np.ndarray, set[tuple[int, int]]] | None:
    for _ in range(SHAPE_GEOMETRY_RETRIES):
        vertices = _make_vertices()
        if shadow_color is None:
            offsets = [None]
        else:
            shape_cells = _shape_cells(vertices)
            offsets_raw = _choose_shadow_offsets(vertices, shape_cells)
            if not offsets_raw:
                continue
            offsets = sorted(offsets_raw, key=lambda o: _shape_reflection_score(vertices, o), reverse=True)

        for offset in offsets:
            locs = _candidate_locations(H, W, vertices, offset, require_reflection=require_reflection)
            for loc in locs[:SHAPE_PLACEMENT_RETRIES]:
                rendered = _render_shape_candidate(H, W, vertices, shape_color, shadow_color, offset, loc)
                if rendered is None:
                    continue
                patch_in, patch_out = rendered
                candidate_cells = _cells_of(patch_out)
                if not candidate_cells:
                    continue
                if _touches_with_padding(candidate_cells, occupied_out, pad=1):
                    continue
                if shadow_color is not None and require_reflection and _difference_from_clipped(patch_in) == 0:
                    continue
                if np.any((full_inp != BG) & (patch_in != BG)):
                    continue
                return patch_in, candidate_cells
    return None


def _place_distractor(
    full_inp: np.ndarray,
    occupied_out: set[tuple[int, int]],
    H: int,
    W: int,
    color: int,
) -> np.ndarray | None:
    cells_local = _make_distractor_cells()
    max_r = max(r for r, _ in cells_local)
    max_c = max(c for _, c in cells_local)
    if max_r >= H or max_c >= W:
        return None
    for _ in range(DISTRACTOR_PLACEMENT_RETRIES):
        loc_r = int(np.random.randint(0, H - max_r))
        loc_c = int(np.random.randint(0, W - max_c))
        patch = np.full((H, W), BG, dtype=int)
        for r, c in cells_local:
            patch[loc_r + r, loc_c + c] = int(color)
        cells = _cells_of(patch)
        if _touches_with_padding(cells, occupied_out, pad=1):
            continue
        if np.any((full_inp != BG) & (patch != BG)):
            continue
        return patch
    return None


def _sample_canvas_size() -> tuple[int, int]:
    H = int(np.random.randint(*CANVAS_H_RANGE))
    W = int(np.clip(H + np.random.randint(*CANVAS_W_JITTER), *CANVAS_W_CLIP))
    return H, W


def _gen_attempt() -> np.ndarray | None:
    H, W = _sample_canvas_size()
    inp = np.full((H, W), BG, dtype=int)
    has_distractor = bool(np.random.rand() < DISTRACTOR_PROB)
    shape_count = int(np.random.randint(*SHAPE_COUNT_RANGE))

    palette = _sample_palette(shape_count, has_distractor)
    if palette is None:
        return None
    shape_colors, shadow_colors, distractor_color = palette
    shadow_map = _assign_shadows(shape_count, shadow_colors)
    reflective_idx = int(np.random.choice(list(shadow_map.keys())))

    occupied_out: set[tuple[int, int]] = set()

    for idx, shape_color in enumerate(shape_colors):
        shadow_color = shadow_map.get(idx)
        placed = _place_shape(
            inp,
            occupied_out,
            H,
            W,
            int(shape_color),
            None if shadow_color is None else int(shadow_color),
            require_reflection=(idx == reflective_idx),
        )
        if placed is None:
            return None
        patch_in, candidate_cells = placed
        _merge_patches(inp, patch_in)
        occupied_out |= candidate_cells

    if has_distractor and distractor_color is not None:
        patch = _place_distractor(inp, occupied_out, H, W, int(distractor_color))
        if patch is None:
            return None
        _merge_patches(inp, patch)
        occupied_out |= _cells_of(patch)

    out = main(inp)
    total_colors = shape_count + len(shadow_colors) + int(has_distractor)
    if get_bg(inp) != BG or get_bg(out) != BG:
        return None
    if len(colors_present(inp, exclude=[BG])) != total_colors:
        return None
    if _difference_from_clipped(inp) == 0:
        return None
    if _reflection_events(inp) == 0:
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GENERATE_RETRIES):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError(f"generate_input failed after {GENERATE_RETRIES} attempts")
