# Variant of 35ab12c3
# Dimension: D8 — 终止条件/端点规则 (dst)
# Change: 无独立终点——路径在同色像素对之间直接连接，终点就是配对的另一个像素... → 到达网格边缘停止

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

SOLVER_UTILS = Path("/home/lixinyu/snap/new/solvers")
if str(SOLVER_UTILS) not in sys.path:
    sys.path.append(str(SOLVER_UTILS))

from arc_utils import bbox_coords, colors_present, get_bg


"""
Design analysis for the D8 variant
==================================

a) Concretization and behavior trace
------------------------------------
Original 35ab12c3 first orders each same-color vertex set into a cycle/polyline and
then draws the segment from each vertex to its paired next vertex. The D8 change is
only about where those strokes stop: instead of stopping at the paired vertex, each
stroke keeps the same H/V/45-degree direction and continues until the grid edge.

Concrete example:
- Input has color 4 at (5,5) and (5,8).
- Original: draw only cells (5,5)..(5,8).
- Variant: draw from (5,5) rightward to column W-1, and because a 2-point shape has
  two endpoints, also draw from (5,8) leftward to column 0. The segment becomes a
  full row ray pair reaching both borders.
- If a shadow pixel of color 7 sits at offset (-1,-1) from the chosen start vertex,
  the shadow also becomes a parallel ray pair with the same offset, still skipping
  collinear offsets and still respecting background-only painting.

So the changed stage is: "segment endpoint selection". Vertex grouping, angular
ordering, start-vertex detection via a neighboring shadow pixel, and shadow offset
inheritance stay intact.

b) Complexity calibration
-------------------------
If the generator kept the original dense multi-shape placement unchanged, border-bound
rays would collide too often and many examples would look like accidental clutter.
The variant therefore keeps the same object grammar in the solver, but the generator
uses a stable visible subfamily: exactly 2 shapes on slightly larger canvases, simple
line/triangle/rectangle geometry, and no sampled distractor or shadow groups. This
reduces accidental congestion while making the ray-to-edge effect obvious in 2-4
examples.

c) Implicit-assumption audit for the original main()
----------------------------------------------------
- `_clear_in_original(..., r0,c0,r1,c1, ...)` assumed there is a meaningful endpoint
  to skip. That is no longer true once the ray goes beyond the partner vertex, so the
  new solver must validate the entire post-source ray corridor to the border.
- `_draw_line_bg_only(..., r0,c0,r1,c1, ...)` only knew a finite segment. It becomes
  `_draw_ray_bg_only(..., r0,c0,dr,dc, ...)`.
- `edges = [(vi, v{i+1}) ...]` is still the correct source of directions, but for
  `len(cycle)==2` the original single segment gave only one direction. The variant
  needs two opposite rays so that both endpoints visibly emit outward.
- Shadow logic relied on the same finite endpoint pair. In the variant the copied
  path must use the same direction but run to the border as well.

d) Placement legality
---------------------
Because rays now span to the canvas boundary, local patch placement is no longer
valid: the stopping position depends on the final global coordinates. The generator
therefore samples full-canvas placements directly, renders each candidate shape's
global ray footprint, and rejects placements that overlap or even 8-touch existing
objects. It also requires every emitted ray to extend at least one cell past its
paired vertex, otherwise that ray would collapse back to original behavior.

e) Visual observability
-----------------------
The rule is visible because the input still contains only sparse vertices, while the
output shows rays that continue well beyond those vertices until a border. No color-id
ordering is used for semantics. The only ordering cue remains geometric: angular
order around the centroid plus the unique shadow-adjacent anchor when a shadow exists.

f) Visual cleanliness
---------------------
The generator forbids overlaps and 8-neighbor contact between distinct groups in the
final output. This matters more than in the original task because each shape now owns
long border-reaching corridors. Simpler shape mixtures and a lower shadow density keep
the output legible.

g) Color inferability
---------------------
Color behavior is unchanged from the original task: each shape keeps its own color,
and an optional adjacent pixel indicates one parallel shadow color. No hidden mapping
from one arbitrary color id to another is introduced.

New solver assumptions introduced by D8
---------------------------------------
1. Each intended shape must have clear ray corridors from every emitting vertex to the
   relevant border, except for same-color vertices on the main ray.
2. If a shadow exists, its single anchor pixel must be the only different-color
   neighbor of exactly one vertex; otherwise the solver could choose the wrong start.
3. Every emitted ray should have positive tail length beyond its paired vertex; the
   generator enforces this so the D8 behavior is actually present rather than merely
   compatible.
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


def _ray_cells(r0: int, c0: int, dr: int, dc: int, h: int, w: int):
    rr, cc = r0, c0
    while 0 <= rr < h and 0 <= cc < w:
        yield rr, cc
        nr = rr + dr
        nc = cc + dc
        if not (0 <= nr < h and 0 <= nc < w):
            break
        rr, cc = nr, nc


def _steps_to_border(r0: int, c0: int, dr: int, dc: int, h: int, w: int) -> int:
    steps = 0
    rr = r0
    cc = c0
    while 0 <= rr + dr < h and 0 <= cc + dc < w:
        rr += dr
        cc += dc
        steps += 1
    return steps


def _clear_ray_in_original(
    g: np.ndarray, r0: int, c0: int, dr: int, dc: int, bg: int, allow: set[int]
) -> bool:
    h, w = g.shape
    first = True
    for rr, cc in _ray_cells(r0, c0, dr, dc, h, w):
        if first:
            first = False
            continue
        v = int(g[rr, cc])
        if v != bg and v not in allow:
            return False
    return True


def _draw_ray_bg_only(out: np.ndarray, r0: int, c0: int, dr: int, dc: int, col: int, bg: int) -> None:
    h, w = out.shape
    for rr, cc in _ray_cells(r0, c0, dr, dc, h, w):
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


def _seed_indices(g: np.ndarray, cycle: list[tuple[int, int]], col: int, bg: int) -> list[int]:
    h, w = g.shape
    seed_idxs: list[int] = []
    for i, (r, c) in enumerate(cycle):
        for drn, dcn in DIR8:
            rr = r + drn
            cc = c + dcn
            if 0 <= rr < h and 0 <= cc < w:
                v = int(g[rr, cc])
                if v != bg and v != col:
                    seed_idxs.append(i)
                    break
    return seed_idxs


def _ordered_cycle_and_carried(
    g: np.ndarray, col: int, bg: int
) -> tuple[list[tuple[int, int]], list[tuple[int, int, int]]]:
    h, w = g.shape
    pts_rc = [tuple(map(int, p)) for p in np.argwhere(g == col)]
    cycle = _angle_cycle(pts_rc)

    seed_idxs = _seed_indices(g, cycle, col, bg)
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
    return cycle, carried


def _ray_specs(cycle: list[tuple[int, int]]) -> list[tuple[tuple[int, int], tuple[int, int], tuple[int, int]]]:
    if len(cycle) == 2:
        a, b = cycle
        return [
            (a, _step_dir(a[0], a[1], b[0], b[1]), b),
            (b, _step_dir(b[0], b[1], a[0], a[1]), a),
        ]
    rays = []
    for i in range(len(cycle)):
        a = cycle[i]
        b = cycle[(i + 1) % len(cycle)]
        rays.append((a, _step_dir(a[0], a[1], b[0], b[1]), b))
    return rays


def _trace_group(
    g: np.ndarray, col: int, bg: int
) -> tuple[list[tuple[tuple[int, int], tuple[int, int], tuple[int, int]]], list[tuple[int, int, int]]]:
    cycle, carried = _ordered_cycle_and_carried(g, col, bg)
    drawable_rays: list[tuple[tuple[int, int], tuple[int, int], tuple[int, int]]] = []
    for src, step, paired in _ray_specs(cycle):
        r0, c0 = src
        r1, c1 = paired
        drs, dcs = step
        if not _aligned_8(r0, c0, r1, c1):
            continue
        if not _clear_ray_in_original(g, r0, c0, drs, dcs, bg, allow={col}):
            continue
        drawable_rays.append((src, step, paired))
    return drawable_rays, carried


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = get_bg(g)
    out = g.copy()

    colors = sorted(set(int(v) for v in np.unique(g)) - {bg})
    for col in colors:
        pts_rc = [tuple(map(int, p)) for p in np.argwhere(g == col)]
        if len(pts_rc) < 2:
            continue

        drawable_rays, carried = _trace_group(g, col, bg)

        for (r0, c0), (drs, dcs), _paired in drawable_rays:
            _draw_ray_bg_only(out, r0, c0, drs, dcs, col, bg)

        for (r0, c0), (drs, dcs), _paired in drawable_rays:
            for drn, dcn, ncol in carried:
                if drs * dcn - dcs * drn == 0:
                    continue
                rr0 = r0 + drn
                cc0 = c0 + dcn
                if not (0 <= rr0 < h and 0 <= cc0 < w):
                    continue
                if not _clear_ray_in_original(g, rr0, cc0, drs, dcs, bg, allow=set()):
                    continue
                _draw_ray_bg_only(out, rr0, cc0, drs, dcs, ncol, bg)

    return out


# --------------- generator (global ray placement for D8 variant) ---------------

BG = 0
COLORS = [1, 2, 3, 4, 6, 7, 8]
CANVAS_H_RANGE = (20, 23)
CANVAS_W_JITTER = (-1, 2)
CANVAS_W_CLIP = (20, 22)

SHAPE_COUNT_RANGE = (2, 3)
MAX_SHADOWS_CAP = 0
DISTRACTOR_PROB = 0.0
PLACEMENT_RETRIES = 90
GEOMETRY_RETRIES = 30
GENERATE_RETRIES = 300

LINE_VERTICES_RANGE = (2, 3)
LINE_TOTAL_LENGTH = 8
LINE_MIN_GAP = 2
TRIANGLE_SIDE_RANGE = (2, 6)
RECT_H_RANGE = (2, 6)
RECT_W_RANGE = (2, 6)

SHAPE_WEIGHTS = np.array([0.65, 0.20, 0.15], dtype=float)

DISTRACTOR_COUNT_RANGE = (1, 4)
DISTRACTOR_BBOX_RANGE = (2, 5)
DISTRACTOR_RETRIES = 80


def _aligned(dr: int, dc: int) -> bool:
    return (dr == 0 and dc != 0) or (dc == 0 and dr != 0) or (dr != 0 and abs(dr) == abs(dc))


def _step(a: tuple[int, int], b: tuple[int, int]) -> tuple[int, int]:
    return _step_dir(a[0], a[1], b[0], b[1])


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


_SHAPE_BUILDERS = [_build_line, _build_triangle, _build_rectangle]


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


def _ray_dirs(vertices: list[tuple[int, int]]) -> list[tuple[int, int]]:
    cycle = _angle_cycle(vertices)
    return [step for _src, step, _paired in _ray_specs(cycle)]


def _choose_shadow(
    vertices: list[tuple[int, int]],
    shape_cells: set[tuple[int, int]],
) -> tuple[tuple[int, int], tuple[int, int]] | None:
    directions = _ray_dirs(vertices)
    candidates: list[tuple[tuple[int, int], tuple[int, int]]] = []
    for anchor in vertices:
        for dr, dc in DIR8:
            if any(ds_r * dc - ds_c * dr == 0 for ds_r, ds_c in directions):
                continue
            pos = (anchor[0] + dr, anchor[1] + dc)
            if pos in shape_cells:
                continue
            if sum(1 for vr, vc in vertices if max(abs(pos[0] - vr), abs(pos[1] - vc)) == 1) != 1:
                continue
            candidates.append((anchor, (dr, dc)))
    if not candidates:
        return None
    return candidates[int(np.random.randint(len(candidates)))]


def _shape_bbox(
    vertices: list[tuple[int, int]],
    shadow: tuple[tuple[int, int], tuple[int, int]] | None,
) -> tuple[int, int, int, int]:
    coords = list(vertices)
    if shadow is not None:
        (ar, ac), (dr, dc) = shadow
        coords.append((ar + dr, ac + dc))
    r0, c0, r1, c1 = bbox_coords(np.array(coords, dtype=int))
    return int(r0), int(c0), int(r1), int(c1)


def _translate_shape(
    vertices: list[tuple[int, int]],
    shadow: tuple[tuple[int, int], tuple[int, int]] | None,
    dr: int,
    dc: int,
) -> tuple[list[tuple[int, int]], tuple[tuple[int, int], tuple[int, int]] | None]:
    new_vertices = [(r + dr, c + dc) for r, c in vertices]
    new_shadow = None
    if shadow is not None:
        (ar, ac), offset = shadow
        new_shadow = ((ar + dr, ac + dc), offset)
    return new_vertices, new_shadow


def _render_single_group(
    h: int,
    w: int,
    vertices: list[tuple[int, int]],
    shape_color: int,
    shadow: tuple[tuple[int, int], tuple[int, int]] | None,
    shadow_color: int | None,
) -> tuple[np.ndarray, np.ndarray, dict[str, object]] | None:
    g = np.full((h, w), BG, dtype=int)
    for r, c in vertices:
        if not (0 <= r < h and 0 <= c < w):
            return None
        if g[r, c] != BG:
            return None
        g[r, c] = int(shape_color)

    if shadow is not None and shadow_color is not None:
        (ar, ac), (dr, dc) = shadow
        sr = ar + dr
        sc = ac + dc
        if not (0 <= sr < h and 0 <= sc < w) or g[sr, sc] != BG:
            return None
        g[sr, sc] = int(shadow_color)

    drawable_rays, carried = _trace_group(g, int(shape_color), BG)
    if shadow is None and carried:
        return None
    if shadow is not None:
        if len(carried) != 1 or carried[0][2] != int(shadow_color):
            return None

    tail_lengths: list[int] = []
    for (r0, c0), (drs, dcs), paired in drawable_rays:
        tail = _steps_to_border(r0, c0, drs, dcs, h, w) - max(abs(paired[0] - r0), abs(paired[1] - c0))
        tail_lengths.append(int(tail))
    if not tail_lengths or min(tail_lengths) <= 0:
        return None

    out = g.copy()
    for (r0, c0), (drs, dcs), _paired in drawable_rays:
        _draw_ray_bg_only(out, r0, c0, drs, dcs, int(shape_color), BG)
    for (r0, c0), (drs, dcs), _paired in drawable_rays:
        for drn, dcn, ncol in carried:
            if drs * dcn - dcs * drn == 0:
                continue
            rr0 = r0 + drn
            cc0 = c0 + dcn
            if not (0 <= rr0 < h and 0 <= cc0 < w):
                return None
            if not _clear_ray_in_original(g, rr0, cc0, drs, dcs, BG, allow=set()):
                return None
            _draw_ray_bg_only(out, rr0, cc0, drs, dcs, int(ncol), BG)

    meta = {
        "tail_lengths": tail_lengths,
        "ray_count": len(drawable_rays),
        "has_shadow": shadow is not None,
    }
    return g, out, meta


def _candidate_blocked(scene: np.ndarray) -> set[tuple[int, int]]:
    h, w = scene.shape
    blocked: set[tuple[int, int]] = set()
    for r, c in np.argwhere(scene != BG):
        rr = int(r)
        cc = int(c)
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                nr = rr + dr
                nc = cc + dc
                if 0 <= nr < h and 0 <= nc < w:
                    blocked.add((nr, nc))
    return blocked


def _can_place(scene_out: np.ndarray, cand_out: np.ndarray) -> bool:
    blocked = _candidate_blocked(scene_out)
    for r, c in np.argwhere(cand_out != BG):
        if (int(r), int(c)) in blocked:
            return False
    return True


def _commit(scene_in: np.ndarray, scene_out: np.ndarray, cand_in: np.ndarray, cand_out: np.ndarray) -> None:
    in_mask = cand_in != BG
    out_mask = cand_out != BG
    scene_in[in_mask] = cand_in[in_mask]
    scene_out[out_mask] = cand_out[out_mask]


def _sample_canvas_size() -> tuple[int, int]:
    h = int(np.random.randint(*CANVAS_H_RANGE))
    w = int(np.clip(h + np.random.randint(*CANVAS_W_JITTER), *CANVAS_W_CLIP))
    return h, w


def _sample_palette(shape_count: int, has_distractor: bool) -> tuple[list[int], list[int], int | None] | None:
    max_shadows = min(MAX_SHADOWS_CAP, shape_count, len(COLORS) - shape_count - int(has_distractor))
    if max_shadows < 0:
        return None
    shadow_count = int(np.random.randint(max_shadows + 1))
    total = shape_count + shadow_count + int(has_distractor)
    palette = list(np.random.permutation(COLORS)[:total])
    shape_colors = [int(v) for v in palette[:shape_count]]
    shadow_colors = [int(v) for v in palette[shape_count:shape_count + shadow_count]]
    distractor_color = int(palette[-1]) if has_distractor else None
    return shape_colors, shadow_colors, distractor_color


def _assign_shadows(shape_count: int, shadow_colors: list[int]) -> dict[int, int]:
    if not shadow_colors:
        return {}
    targets = np.random.choice(shape_count, size=len(shadow_colors), replace=False).tolist()
    return {int(idx): int(col) for idx, col in zip(sorted(targets), shadow_colors)}


def _make_shape_candidate(h: int, w: int, shape_color: int, shadow_color: int | None):
    for _ in range(GEOMETRY_RETRIES):
        vertices = _make_vertices()
        cells = _shape_cells(vertices)
        shadow = None
        if shadow_color is not None:
            shadow = _choose_shadow(vertices, cells)
            if shadow is None:
                continue

        r0, c0, r1, c1 = _shape_bbox(vertices, shadow)
        patch_h = r1 - r0 + 1
        patch_w = c1 - c0 + 1
        if patch_h > h or patch_w > w:
            continue

        base_vertices = [(r - r0, c - c0) for r, c in vertices]
        base_shadow = None
        if shadow is not None:
            (ar, ac), offset = shadow
            base_shadow = ((ar - r0, ac - c0), offset)

        for _ in range(PLACEMENT_RETRIES):
            off_r = int(np.random.randint(0, h - patch_h + 1))
            off_c = int(np.random.randint(0, w - patch_w + 1))
            shifted_vertices, shifted_shadow = _translate_shape(base_vertices, base_shadow, off_r, off_c)
            rendered = _render_single_group(h, w, shifted_vertices, int(shape_color), shifted_shadow, shadow_color)
            if rendered is None:
                continue
            cand_in, cand_out, meta = rendered
            return cand_in, cand_out, meta
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


def _make_distractor_candidate(h: int, w: int, color: int) -> tuple[np.ndarray, np.ndarray] | None:
    count = int(np.random.randint(*DISTRACTOR_COUNT_RANGE))
    if count == 1:
        cand = np.full((h, w), BG, dtype=int)
        r = int(np.random.randint(h))
        c = int(np.random.randint(w))
        cand[r, c] = int(color)
        return cand.copy(), cand.copy()
    for _ in range(DISTRACTOR_RETRIES):
        box_h = int(np.random.randint(*DISTRACTOR_BBOX_RANGE))
        box_w = int(np.random.randint(*DISTRACTOR_BBOX_RANGE))
        coords = _sample_distractor_coords(box_h, box_w, count)
        if coords is None:
            continue
        off_r = int(np.random.randint(0, h - box_h + 1))
        off_c = int(np.random.randint(0, w - box_w + 1))
        cand = np.full((h, w), BG, dtype=int)
        for r, c in coords:
            cand[off_r + r, off_c + c] = int(color)
        return cand.copy(), cand.copy()
    return None


def _gen_attempt(return_meta: bool = False):
    h, w = _sample_canvas_size()
    scene_in = np.full((h, w), BG, dtype=int)
    scene_out = np.full((h, w), BG, dtype=int)

    has_distractor = bool(np.random.rand() < DISTRACTOR_PROB)
    shape_count = int(np.random.randint(*SHAPE_COUNT_RANGE))

    palette = _sample_palette(shape_count, has_distractor)
    if palette is None:
        return None
    shape_colors, shadow_colors, distractor_color = palette
    shadow_map = _assign_shadows(shape_count, shadow_colors)

    metrics = {
        "shape_count": shape_count,
        "shadow_count": len(shadow_colors),
        "ray_count": 0,
        "tail_lengths": [],
    }

    for idx, shape_color in enumerate(shape_colors):
        placed = False
        for _ in range(PLACEMENT_RETRIES):
            candidate = _make_shape_candidate(h, w, int(shape_color), shadow_map.get(idx))
            if candidate is None:
                continue
            cand_in, cand_out, meta = candidate
            if not _can_place(scene_out, cand_out):
                continue
            _commit(scene_in, scene_out, cand_in, cand_out)
            metrics["ray_count"] += int(meta["ray_count"])
            metrics["tail_lengths"].extend(int(v) for v in meta["tail_lengths"])
            placed = True
            break
        if not placed:
            return None

    if has_distractor and distractor_color is not None:
        placed = False
        for _ in range(PLACEMENT_RETRIES):
            candidate = _make_distractor_candidate(h, w, int(distractor_color))
            if candidate is None:
                continue
            cand_in, cand_out = candidate
            if not _can_place(scene_out, cand_out):
                continue
            _commit(scene_in, scene_out, cand_in, cand_out)
            placed = True
            break
        if not placed:
            return None

    total_colors = shape_count + len(shadow_colors) + int(has_distractor)
    if get_bg(scene_in) != BG or get_bg(scene_out) != BG:
        return None
    if len(colors_present(scene_in, exclude=[BG])) != total_colors:
        return None
    solved = main(scene_in)
    if not np.array_equal(solved, scene_out):
        return None

    if return_meta:
        return scene_in, metrics
    return scene_in


def generate_input() -> np.ndarray:
    for _ in range(GENERATE_RETRIES):
        result = _gen_attempt(return_meta=False)
        if result is not None:
            return result
    raise RuntimeError(f"generate_input failed after {GENERATE_RETRIES} attempts")
