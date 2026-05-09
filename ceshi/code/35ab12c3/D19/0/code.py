# Variant of 35ab12c3
# Dimension: D19 — 线段绘制优先级 (multi_step)
# Change: 两阶段绘制：Phase1先画所有主形状边（只涂背景格子），Phase2再画所有阴影平行线（只涂背景格子）。同一形状内主线... → 边-阴影交替绘制

import math

import numpy as np

from arc_utils import bbox_coords, colors_present, get_bg, paste, random_free_location
from arc_utils import get_bg as _bg_color

"""
Design analysis for D19 (source_task: none)

a) Concrete behavior and trace
   Input still shows sparse same-color vertices plus one 8-neighbor shadow-anchor
   pixel in another color. Output is still built by connecting each color group
   with H/V/45-degree segments and drawing shadow-parallel copies. The D19 change
   is only the schedule inside one shape:
   old: draw all main edges, then all shadow edges;
   new: for each drawable edge in cycle order, draw that main edge and immediately
        draw its shadow edge(s).
   Because the primitive remains "write only onto background", the observable
   consequence is that an earlier shadow can occupy cells that a later main edge
   would have used, so the later main edge is blocked there. Example:
   rectangle vertices at (0,0),(0,4),(3,4),(3,0), anchor=(0,0), shadow offset=(1,1).
   Edge order is top -> right -> bottom -> left. After drawing the top edge, its
   shadow occupies row 1, cols 1..5. The later right edge wants to paint col 4,
   rows 0..3; cell (1,4) is no longer background, so the interleaved variant keeps
   the shadow color there. In the original two-phase solver, the right edge already
   exists before any shadow is attempted, so (1,4) stays the main color.

b) Complexity calibration
   If shadows are optional and conflict-free, D19 becomes invisible and the variant
   degenerates to the original task. To keep the rule inferable, the generator
   forces at least one shadowed closed shape per grid whose interleaved rendering
   differs from the old two-phase rendering by at least one cell. Other shapes may
   still be ordinary unshadowed lines/polygons so the overall task remains close to
   the base task instead of turning into a single highly constrained gadget.

c) Hidden assumptions in the original main()
   - _draw_line_bg_only was harmless for main edges only because no shadow was drawn
     until every main edge had already been placed.
   - The split between "Phase 1 all mains" and "Phase 2 all shadows" encoded the
     old priority rule even though the write primitive itself did not.
   - _clear_in_original checks only the input grid, not the evolving output, so the
     only place where D19 can matter is the order in which out is mutated.
   Therefore the solver change must be minimal and local: preserve edge discovery,
   anchor selection, and shadow validity checks, but interleave the existing draw
   calls so out-state matters differently.

d) Placement legality
   The generator still builds one local patch per shape and pastes patches onto a
   larger canvas with a >=1-cell buffer using random_free_location(..., padding=1).
   Each patch bbox covers vertices, the shifted vertices, and the shadow-anchor cell,
   so no rendered segment can leave the patch. Because patches are separated before
   paste, inter-shape overlaps do not occur; only intra-shape edge-vs-shadow clashes
   are used to express D19.

e) Visual observability
   Humans can see the change because at least one shadowed polygon in every sample
   contains a few "missing" main-edge cells replaced by the shadow color exactly
   where an earlier parallel shadow passed first. This is a local structural cue,
   not a hidden color-id trick.

f) Visual cleanliness
   The generator keeps the original roughly square canvas, small number of groups,
   unique colors per group, and one-cell spacing. The conflict cells are confined to
   one shape at a time, so outputs stay legible instead of becoming globally noisy.

g) Color inferability
   No new color mapping is introduced. Shape colors and shadow colors remain unique,
   and the shadow color is still supplied by the single adjacent anchor pixel in the
   input, so the solver cue remains structural and local.

h) Generator-side enforcement
   New helper _main_two_phase preserves the old rendering for comparison only. The
   generator accepts at least one shadowed shape only if main(inp) and
   _main_two_phase(inp) differ on that local patch. This guarantees that the saved
   dataset actually exercises D19 instead of accidentally producing old-behavior
   instances that happen to pass the solver oracle.

i) New assumptions introduced by the variant solver
   The interleaved solver now assumes that any same-shape shadow/main occlusion it
   encounters is intended. The generator guarantees this by:
   - attaching shadow anchors to exactly one vertex;
   - using only H/V/45-degree edges;
   - keeping separate shapes spatially disjoint;
   - forcing at least one shadowed closed shape with positive old-vs-new delta.
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


def _angle_cycle(points_rc: list[tuple[int, int]]) -> list[tuple[int, int]]:
    cr = sum(r for r, _ in points_rc) / len(points_rc)
    cc = sum(c for _, c in points_rc) / len(points_rc)

    def key(p: tuple[int, int]):
        r, c = p
        ang = math.atan2(r - cr, c - cc)
        rad = (r - cr) ** 2 + (c - cc) ** 2
        return (ang, rad, r, c)

    return sorted(points_rc, key=key)


def _rotate_cycle(cycle: list[tuple[int, int]], start_idx: int) -> list[tuple[int, int]]:
    return cycle[start_idx:] + cycle[:start_idx]


def _shape_plan(g: np.ndarray, col: int, bg: int) -> tuple[list[tuple[tuple[int, int], tuple[int, int]]], list[tuple[int, int, int]]] | None:
    h, w = g.shape
    pts_rc = [tuple(map(int, p)) for p in np.argwhere(g == col)]
    if len(pts_rc) < 2:
        return None

    cycle = _angle_cycle(pts_rc)
    if len(cycle) < 2:
        return None

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

    cycle = _rotate_cycle(cycle, start)

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
    return drawable_edges, carried


def _draw_parallel_shadows(
    out: np.ndarray,
    g: np.ndarray,
    r0: int,
    c0: int,
    r1: int,
    c1: int,
    carried: list[tuple[int, int, int]],
    bg: int,
) -> None:
    h, w = g.shape
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


def _render_grid(g: np.ndarray, interleaved: bool) -> np.ndarray:
    bg = _bg_color(g)
    out = g.copy()
    colors = sorted(set(int(v) for v in np.unique(g)) - {bg})
    for col in colors:
        plan = _shape_plan(g, col, bg)
        if plan is None:
            continue
        drawable_edges, carried = plan
        if interleaved:
            for (r0, c0), (r1, c1) in drawable_edges:
                _draw_line_bg_only(out, r0, c0, r1, c1, col, bg)
                _draw_parallel_shadows(out, g, r0, c0, r1, c1, carried, bg)
        else:
            for (r0, c0), (r1, c1) in drawable_edges:
                _draw_line_bg_only(out, r0, c0, r1, c1, col, bg)
            for (r0, c0), (r1, c1) in drawable_edges:
                _draw_parallel_shadows(out, g, r0, c0, r1, c1, carried, bg)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    return _render_grid(g, interleaved=True)


def _main_two_phase(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    return _render_grid(g, interleaved=False)


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
SHADOW_SHAPE_WEIGHTS = np.array([0.0, 0.20, 0.28, 0.18, 0.18, 0.16], dtype=float)

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


def _make_vertices(*, prefer_shadow_shape: bool = False) -> list[tuple[int, int]]:
    if prefer_shadow_shape:
        weights = SHADOW_SHAPE_WEIGHTS / SHADOW_SHAPE_WEIGHTS.sum()
    else:
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


def _cycle_shape_cells(vertices: list[tuple[int, int]]) -> set[tuple[int, int]]:
    ordered = _angle_cycle(vertices)
    return _shape_cells(ordered)


def _edge_dirs(vertices: list[tuple[int, int]]) -> list[tuple[int, int]]:
    return [_step(a, b) for a, b in _shape_edges(vertices)]


def _patch_bbox(vertices: list[tuple[int, int]], shadow: tuple[tuple[int, int], tuple[int, int]] | None) -> tuple[int, int, int, int]:
    coords = list(vertices)
    if shadow is not None:
        (ar, ac), (dr, dc) = shadow
        coords.append((ar + dr, ac + dc))
        coords.extend((r + dr, c + dc) for r, c in vertices)
    r0, c0, r1, c1 = bbox_coords(np.array(coords, dtype=int))
    return int(r0), int(c0), int(r1), int(c1)


def _build_input_patch(
    vertices: list[tuple[int, int]],
    shape_color: int,
    shadow_color: int | None = None,
    shadow: tuple[tuple[int, int], tuple[int, int]] | None = None,
) -> np.ndarray | None:
    r0, c0, r1, c1 = _patch_bbox(vertices, shadow)
    h, w = r1 - r0 + 1, c1 - c0 + 1
    inp = np.full((h, w), BG, dtype=int)
    shifted_vertices = [(r - r0, c - c0) for r, c in vertices]
    for r, c in shifted_vertices:
        inp[r, c] = shape_color
    if shadow is not None:
        if shadow_color is None:
            return None
        (ar, ac), offset = shadow
        anchor = (ar - r0, ac - c0)
        sa = (anchor[0] + offset[0], anchor[1] + offset[1])
        if not (0 <= sa[0] < h and 0 <= sa[1] < w):
            return None
        if inp[sa] != BG:
            return None
        if sum(1 for vr, vc in shifted_vertices if max(abs(sa[0] - vr), abs(sa[1] - vc)) == 1) != 1:
            return None
        inp[sa] = int(shadow_color)
    return inp


def _evaluate_shadow_variant(
    vertices: list[tuple[int, int]],
    shape_color: int,
    shadow_color: int,
    shadow: tuple[tuple[int, int], tuple[int, int]],
) -> tuple[np.ndarray, np.ndarray, int] | None:
    inp = _build_input_patch(vertices, shape_color, shadow_color, shadow)
    if inp is None:
        return None
    out_variant = main(inp)
    out_old = _main_two_phase(inp)
    diff = int(np.count_nonzero(out_variant != out_old))
    return inp, out_variant, diff


def _choose_shadow_setup(
    vertices: list[tuple[int, int]],
    shape_color: int,
    shadow_color: int,
    *,
    require_conflict: bool,
) -> tuple[np.ndarray, np.ndarray, int] | None:
    anchor = min(vertices)
    shape_cells = _cycle_shape_cells(vertices)
    directions = _edge_dirs(_angle_cycle(vertices))

    candidates: list[tuple[int, np.ndarray, np.ndarray]] = []
    positive: list[tuple[int, np.ndarray, np.ndarray]] = []
    for dr, dc in DIR8:
        if any(ds_r * dc - ds_c * dr == 0 for ds_r, ds_c in directions):
            continue
        pos = (anchor[0] + dr, anchor[1] + dc)
        if pos in shape_cells:
            continue
        if sum(1 for vr, vc in vertices if max(abs(pos[0] - vr), abs(pos[1] - vc)) == 1) != 1:
            continue
        evaluated = _evaluate_shadow_variant(vertices, shape_color, shadow_color, (anchor, (dr, dc)))
        if evaluated is None:
            continue
        inp, out_variant, diff = evaluated
        rec = (diff, inp, out_variant)
        candidates.append(rec)
        if diff > 0:
            positive.append(rec)

    if require_conflict and not positive:
        return None
    if require_conflict:
        best = max(diff for diff, _, _ in positive)
        pool = [rec for rec in positive if rec[0] == best]
    elif positive and np.random.rand() < 0.75:
        pool = positive
    else:
        pool = candidates
    if not pool:
        return None
    diff, inp, out_variant = pool[int(np.random.randint(len(pool)))]
    return inp, out_variant, diff


def _make_shape_patch(
    shape_color: int,
    shadow_color: int | None,
    *,
    require_conflict: bool = False,
) -> tuple[np.ndarray, np.ndarray, int] | None:
    vertices = _make_vertices(prefer_shadow_shape=shadow_color is not None)

    if shadow_color is None:
        inp = _build_input_patch(vertices, shape_color)
        if inp is None:
            return None
        out = main(inp)
        if not np.array_equal(main(inp), out):
            return None
        return inp, out, 0

    chosen = _choose_shadow_setup(vertices, shape_color, int(shadow_color), require_conflict=require_conflict)
    if chosen is None:
        return None
    inp, out, diff = chosen
    if require_conflict and diff <= 0:
        return None
    if not np.array_equal(main(inp), out):
        return None
    return inp, out, diff


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
    if max_shadows <= 0:
        return None
    shadow_count = int(np.random.randint(1, max_shadows + 1))
    total = shape_count + shadow_count + int(has_distractor)
    palette = list(np.random.permutation(COLORS)[:total])
    shape_colors = palette[:shape_count]
    shadow_colors = palette[shape_count:shape_count + shadow_count]
    distractor_color = int(palette[-1]) if has_distractor else None
    return shape_colors, shadow_colors, distractor_color


def _assign_shadows(shape_count: int, shadow_colors: list[int]) -> tuple[dict[int, int], int]:
    targets = np.random.choice(shape_count, size=len(shadow_colors), replace=False).tolist()
    shadow_map = {int(idx): int(col) for idx, col in zip(sorted(targets), shadow_colors)}
    conflict_idx = int(np.random.choice(list(shadow_map)))
    return shadow_map, conflict_idx


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
    shadow_map, conflict_shape_idx = _assign_shadows(shape_count, shadow_colors)

    total_colors = shape_count + len(shadow_colors) + int(has_distractor)
    total_diff = 0

    for idx, shape_color in enumerate(shape_colors):
        placed = False
        for _ in range(SHAPE_PLACEMENT_RETRIES):
            patch = _make_shape_patch(
                int(shape_color),
                shadow_map.get(idx),
                require_conflict=(idx == conflict_shape_idx),
            )
            if patch is None:
                continue
            patch_in, patch_out, diff = patch
            if _try_place(inp, out, patch_in, patch_out):
                total_diff += diff
                placed = True
                break
        if not placed:
            return None

    if total_diff <= 0:
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
    if np.array_equal(_main_two_phase(inp), out):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GENERATE_RETRIES):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError(f"generate_input failed after {GENERATE_RETRIES} attempts")
