# Variant of 35ab12c3
# Dimension: D15 — 段间清晰路径约束 (path)
# Change: clear_in_original——两个顶点之间的连线路径上不能有非背景非同色像素阻挡（在原始输入中检查），否则该边不画 → 路径穿过时覆盖所有像素

import math
import sys
from pathlib import Path

import numpy as np

_SOLVER_ROOT = Path("/home/lixinyu/snap/new/solvers")
if str(_SOLVER_ROOT) not in sys.path:
    sys.path.append(str(_SOLVER_ROOT))

from arc_utils import bbox_coords, colors_present, get_bg, neighbors8, paste, random_free_location


# Design analysis for the D15 variant
# a) Concrete behavior:
#    The old task only connected two same-color vertices when the segment path was
#    clean in the original input, and even then it only filled background cells.
#    The new variant removes both protections: if two vertices are H/V/45-degree
#    aligned, that segment is always drawn, and every traversed cell is overwritten.
#    To make that visible, the generator injects one forced interaction scene:
#      - shape A is a 2-vertex vertical line;
#      - shape B is a horizontal line whose middle vertex lies on A's future path;
#      - A also has one shadow anchor so its parallel shadow line crosses B too.
#    Small trace:
#      old logic: A main line is blocked by B's middle vertex, so only A's shadow
#      appears; later B draws but cannot overwrite the shadow because it only fills
#      background.
#      new logic: A main line draws through B's vertex, A's shadow draws too, then
#      later B draws and overwrites both crossing cells. This exposes both parts of
#      D15: "ignore blockers" and "later paint crushes earlier paint".
# b) Complexity calibration:
#    The grid still contains 2-4 shape colors, optional extra shadows, and optional
#    distractors. Only one local interaction scene is forced; the rest remain close
#    to the original task so the puzzle does not collapse into a trivial two-line
#    template. The forced scene has exactly one blocker vertex and one shadow-cross,
#    which is enough to be visible without turning the whole canvas into clutter.
# c) Hidden assumptions audit:
#    In the original solver, `sorted(colors)` was harmless because shapes never
#    overlapped in the output. Under D15, that would create invisible color-id-based
#    precedence. The solver therefore switches the group processing order to a
#    visible spatial rule: sort color groups by their top-left input pixel. This is
#    the only non-D15 adaptation, and it is necessary so overwrite precedence is
#    inferable from examples instead of depending on hidden palette ids.
#    The original `_clear_in_original(...)` checks and `_draw_line_bg_only(...)`
#    writes both depend on the old "blocked / only-fill-bg" semantics; both must be
#    bypassed or replaced everywhere a main edge or carried shadow edge is drawn.
# d) Placement legality:
#    Full future output patches are still used for placement, so different groups do
#    not accidentally interact outside the intentionally crafted overlap scene. The
#    forced scene contains internal overlap by construction; extra groups are placed
#    with a one-cell buffer against its final output footprint.
# e) Visual observability:
#    The overwrite rule is readable from 2-3 examples because every sample contains
#    the same kind of local evidence: one line passes through a foreign vertex, and
#    one later line repaints a shadow crossing. No ordering depends on numeric color
#    ids; the precedence comes from visible position only.
# f) Visual cleanliness:
#    Overlap is concentrated inside one compact patch. The rest of the canvas keeps
#    the original sparse look, so the important crushing event is easy to isolate.
# g) Color inference:
#    Colors are still just object labels. No color remapping or hidden palette
#    ordering is used. Extra colors only indicate extra shapes, shadows, or a
#    distractor, exactly as in the base task.

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
    g: np.ndarray, r0: int, c0: int, r1: int, c1: int, bg: int, allow: set[int]
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
        if int(out[rr, cc]) == bg:
            out[rr, cc] = int(col)


def _draw_line_overwrite(out: np.ndarray, r0: int, c0: int, r1: int, c1: int, col: int) -> None:
    for rr, cc in _segment_cells(r0, c0, r1, c1):
        out[rr, cc] = int(col)


def _angle_cycle(points_rc: list[tuple[int, int]]) -> list[tuple[int, int]]:
    cr = sum(r for r, _ in points_rc) / len(points_rc)
    cc = sum(c for _, c in points_rc) / len(points_rc)

    def key(p: tuple[int, int]):
        r, c = p
        ang = math.atan2(r - cr, c - cc)
        rad = (r - cr) ** 2 + (c - cc) ** 2
        return (ang, rad, r, c)

    return sorted(points_rc, key=key)


def _group_order(g: np.ndarray, bg: int) -> list[int]:
    cols = []
    for col in set(int(v) for v in np.unique(g)) - {bg}:
        pts = [tuple(map(int, p)) for p in np.argwhere(g == col)]
        cols.append((min(pts), int(col)))
    cols.sort(key=lambda item: (item[0][0], item[0][1]))
    return [col for _, col in cols]


def _solve(grid: np.ndarray, *, respect_blockers: bool, overwrite_paths: bool) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = get_bg(g)
    out = g.copy()
    colors = _group_order(g, bg)

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
            if respect_blockers and not _clear_in_original(g, r0, c0, r1, c1, bg, allow={col}):
                continue
            drawable_edges.append(((r0, c0), (r1, c1)))

        for (r0, c0), (r1, c1) in drawable_edges:
            if overwrite_paths:
                _draw_line_overwrite(out, r0, c0, r1, c1, col)
            else:
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
                if respect_blockers and not _clear_in_original(g, rr0, cc0, rr1, cc1, bg, allow=set()):
                    continue
                if overwrite_paths:
                    _draw_line_overwrite(out, rr0, cc0, rr1, cc1, ncol)
                else:
                    _draw_line_bg_only(out, rr0, cc0, rr1, cc1, ncol, bg)

    return out


def main(grid: np.ndarray) -> np.ndarray:
    return _solve(grid, respect_blockers=False, overwrite_paths=True)


def _legacy_main(grid: np.ndarray) -> np.ndarray:
    return _solve(grid, respect_blockers=True, overwrite_paths=False)


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

SHAPE_PLACEMENT_RETRIES = 40
GENERATE_RETRIES = 300


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


def _transform_scene(parts: list[list[tuple[int, int]]]) -> list[list[tuple[int, int]]]:
    sizes = [len(part) for part in parts]
    arr = np.array([coord for part in parts for coord in part], dtype=int)
    if np.random.rand() < 0.5:
        arr[:, 0] *= -1
    if np.random.rand() < 0.5:
        arr[:, 1] *= -1
    arr -= arr.min(axis=0)
    out: list[list[tuple[int, int]]] = []
    idx = 0
    for size in sizes:
        out.append([tuple(map(int, p)) for p in arr[idx : idx + size]])
        idx += size
    return out


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


def _choose_shadow(
    vertices: list[tuple[int, int]], shape_cells: set[tuple[int, int]]
) -> tuple[tuple[int, int], tuple[int, int]] | None:
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
        candidates.append((dr, dc))
    if not candidates:
        return None
    return anchor, candidates[int(np.random.randint(len(candidates)))]


def _patch_bbox(
    shape_cells: set[tuple[int, int]],
    vertices: list[tuple[int, int]],
    shadow: tuple[tuple[int, int], tuple[int, int]] | None,
) -> tuple[int, int, int, int]:
    coords = list(shape_cells) + vertices
    if shadow is not None:
        (ar, ac), (dr, dc) = shadow
        coords.append((ar + dr, ac + dc))
        coords.extend((r + dr, c + dc) for r, c in shape_cells)
    r0, c0, r1, c1 = bbox_coords(np.array(coords, dtype=int))
    return int(r0), int(c0), int(r1), int(c1)


def _make_shape_patch(shape_color: int, shadow_color: int | None) -> tuple[np.ndarray, np.ndarray] | None:
    vertices = _make_vertices()
    cells = _shape_cells(vertices)

    shadow = None
    if shadow_color is not None:
        shadow = _choose_shadow(vertices, cells)
        if shadow is None:
            return None

    r0, c0, r1, c1 = _patch_bbox(cells, vertices, shadow)
    h, w = r1 - r0 + 1, c1 - c0 + 1
    inp = np.full((h, w), BG, dtype=int)

    shifted_vertices = [(r - r0, c - c0) for r, c in vertices]
    for r, c in shifted_vertices:
        inp[r, c] = int(shape_color)

    if shadow is not None:
        (ar, ac), offset = shadow
        anchor = (ar - r0, ac - c0)
        shadow_pos = (anchor[0] + offset[0], anchor[1] + offset[1])
        if not (0 <= shadow_pos[0] < h and 0 <= shadow_pos[1] < w):
            return None
        if int(inp[shadow_pos]) != BG:
            return None
        inp[shadow_pos] = int(shadow_color)

    out = main(inp)
    if np.array_equal(out, inp):
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
            patch[r, c] = int(color)
        return patch.copy(), patch.copy()
    patch = np.full((1, 1), color, dtype=int)
    return patch.copy(), patch.copy()


def _make_interaction_patch(shape_a: int, shape_b: int, shadow_col: int) -> tuple[np.ndarray, np.ndarray] | None:
    half_r = int(np.random.randint(3, 6))
    half_c = int(np.random.randint(3, 6))
    mid_r, mid_c = half_r, half_c
    side = -1 if np.random.rand() < 0.5 else 1

    a_vertices = [(0, mid_c), (2 * half_r, mid_c)]
    b_vertices = [(mid_r, 0), (mid_r, mid_c), (mid_r, 2 * half_c)]
    shadow_pixel = [(0, mid_c + side)]
    a_vertices, b_vertices, shadow_pixel = _transform_scene([a_vertices, b_vertices, shadow_pixel])

    coords = a_vertices + b_vertices + shadow_pixel
    r0, c0, r1, c1 = bbox_coords(np.array(coords, dtype=int))
    h, w = r1 - r0 + 1, c1 - c0 + 1
    inp = np.full((h, w), BG, dtype=int)

    for r, c in a_vertices:
        inp[r - r0, c - c0] = int(shape_a)
    for r, c in b_vertices:
        if int(inp[r - r0, c - c0]) != BG:
            return None
        inp[r - r0, c - c0] = int(shape_b)
    sr, sc = shadow_pixel[0]
    if int(inp[sr - r0, sc - c0]) != BG:
        return None
    inp[sr - r0, sc - c0] = int(shadow_col)

    out = main(inp)
    legacy = _legacy_main(inp)
    diff = int(np.count_nonzero(out != legacy))
    if diff < 3:
        return None
    return inp, out


def _assign_shadows(shape_count: int, shadow_colors: list[int]) -> dict[int, int]:
    if not shadow_colors:
        return {}
    targets = np.random.choice(shape_count, size=len(shadow_colors), replace=False).tolist()
    return {int(idx): int(col) for idx, col in zip(sorted(targets), shadow_colors)}


def _try_place(inp: np.ndarray, occ: np.ndarray, patch_in: np.ndarray, patch_out: np.ndarray) -> bool:
    try:
        loc = random_free_location(
            occ,
            patch_out,
            transparent=BG,
            padding=PLACEMENT_PADDING,
            padding_connectivity=PLACEMENT_PAD_CONNECTIVITY,
        )
    except ValueError:
        return False
    paste(occ, patch_out, loc[0], loc[1], transparent=BG)
    paste(inp, patch_in, loc[0], loc[1], transparent=BG)
    return True


def _sample_canvas_size() -> tuple[int, int]:
    h = int(np.random.randint(*CANVAS_H_RANGE))
    w = int(np.clip(h + np.random.randint(*CANVAS_W_JITTER), *CANVAS_W_CLIP))
    return h, w


def _sample_palette(
    shape_count: int, has_distractor: bool
) -> tuple[list[int], int, list[int], list[int], int | None] | None:
    extra_shapes = shape_count - 2
    special_shadow_count = 1
    max_extra_shadows = min(
        MAX_SHADOWS_CAP - special_shadow_count,
        extra_shapes,
        len(COLORS) - shape_count - special_shadow_count - int(has_distractor),
    )
    if max_extra_shadows < 0:
        return None
    extra_shadow_count = int(np.random.randint(max_extra_shadows + 1))
    total = shape_count + special_shadow_count + extra_shadow_count + int(has_distractor)
    if total > len(COLORS):
        return None
    palette = list(np.random.permutation(COLORS)[:total])
    special_shapes = [int(palette[0]), int(palette[1])]
    special_shadow = int(palette[2])
    extra_shape_colors = [int(v) for v in palette[3 : 3 + extra_shapes]]
    extra_shadow_colors = [int(v) for v in palette[3 + extra_shapes : 3 + extra_shapes + extra_shadow_count]]
    distractor_color = int(palette[-1]) if has_distractor else None
    return special_shapes, special_shadow, extra_shape_colors, extra_shadow_colors, distractor_color


def _gen_attempt() -> np.ndarray | None:
    h, w = _sample_canvas_size()
    inp = np.full((h, w), BG, dtype=int)
    occ = np.full((h, w), BG, dtype=int)

    shape_count = int(np.random.randint(*SHAPE_COUNT_RANGE))
    has_distractor = bool(np.random.rand() < DISTRACTOR_PROB)

    palette = _sample_palette(shape_count, has_distractor)
    if palette is None:
        return None
    special_shapes, special_shadow, extra_shape_colors, extra_shadow_colors, distractor_color = palette

    interaction = _make_interaction_patch(special_shapes[0], special_shapes[1], special_shadow)
    if interaction is None:
        return None
    if not _try_place(inp, occ, *interaction):
        return None

    extra_shadow_map = _assign_shadows(len(extra_shape_colors), extra_shadow_colors)
    for idx, shape_color in enumerate(extra_shape_colors):
        placed = False
        for _ in range(SHAPE_PLACEMENT_RETRIES):
            patch = _make_shape_patch(int(shape_color), extra_shadow_map.get(idx))
            if patch is None:
                continue
            if _try_place(inp, occ, *patch):
                placed = True
                break
        if not placed:
            return None

    if has_distractor and distractor_color is not None:
        if not _try_place(inp, occ, *_make_distractor_patch(distractor_color)):
            return None

    out = main(inp)
    legacy = _legacy_main(inp)

    expected_colors = 3 + len(extra_shape_colors) + len(extra_shadow_colors) + int(has_distractor)
    if get_bg(inp) != BG or get_bg(out) != BG:
        return None
    if len(colors_present(inp, exclude=[BG])) != expected_colors:
        return None
    if not np.array_equal(out, occ):
        return None
    if np.array_equal(out, legacy):
        return None
    if np.count_nonzero(out != legacy) < 3:
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GENERATE_RETRIES):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError(f"generate_input failed after {GENERATE_RETRIES} attempts")
