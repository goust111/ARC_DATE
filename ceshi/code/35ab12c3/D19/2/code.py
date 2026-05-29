# Variant of 35ab12c3
# Dimension: D19 — 线段绘制优先级 (multi_step)
# Change: 两阶段绘制：Phase1先画所有主形状边（只涂背景格子），Phase2再画所有阴影平行线（只涂背景格子）。同一形状内主线...
#         → 全覆盖绘制（非仅背景）: 取消“只涂背景格子”的限制，后画的线可以覆盖先画的线。
#
# Design analysis
# ---------------
# a) Concrete behavior and trace
#    - Original task: input gives a shape color as sparse vertices plus one shadow-anchor
#      pixel in another color. Solver reconstructs the polygon/polyline edges and then
#      draws a parallel shadow copy. Because both phases only paint background cells,
#      any shadow cell that lands on top of the main shape is suppressed; the main
#      shape always survives.
#    - Variant: keep the same inference mechanism (find vertices, find the one vertex
#      that has an adjacent foreign-colored anchor, reuse that offset for all edges),
#      but change the write semantics. Every drawn segment now overwrites existing
#      colors. With the same two-phase order, shadow lines are still drawn after the
#      main shape, so a shadow may repaint part of the main shape.
#    - Small trace:
#        vertices = [(0,0), (0,4), (4,4)], shadow anchor at (1,-1) relative to (0,0)
#        main edges: top horizontal, right vertical, down-right diagonal
#        shifted shadow edges: one shifted edge crosses the original diagonal.
#        Original output keeps the crossing cell in shape color.
#        Variant output repaints that crossing cell in shadow color because shadow is
#        drawn later and overwrite is allowed.
#
# b) Complexity calibration
#    - If shadowed shapes are sampled arbitrarily, many grids would still look identical
#      to the original bg-only rule, making D19 invisible.
#    - To avoid that degeneration, every generated grid contains exactly one shadowed
#      shape from a controlled triangle family whose shifted shadow is guaranteed to
#      overlap the main shape on at least one cell. Other shapes remain ordinary.
#    - Keeping only one shadowed shape per grid avoids hidden ordering between multiple
#      shadow colors. The only meaningful priority is visibly “main first, shadow later”.
#
# c) Old implicit assumptions that break under D19
#    - `_draw_line_bg_only(...)`: relied on the old invariant that writes should never
#      replace an existing color. This is the direct D19 change point.
#    - `_paint_shadow(...)` in the generator: also relied on bg-only painting, so the
#      generated expected output would silently keep shape color on overlaps. That must
#      be changed to overwrite as well.
#    - Iteration order across distinct colors is still based on sorted color ids in the
#      original solver. Under D19 this would become a hidden rule if different shapes
#      overlapped. The generator therefore preserves the old placement invariant that
#      different groups do not overlap each other at all; only self-overlap between a
#      shape and its own shadow is allowed.
#
# d) Placement legality
#    - Each shape/distractor is still built as a local patch and placed with at least
#      one-cell padding, so distinct groups remain disjoint and in-bounds.
#    - The special shadowed patch allows overlap only inside the patch, between the
#      shape and its own shifted copy. This keeps the solver unambiguous and avoids
#      cross-shape color-order artifacts.
#
# e) Visual observability
#    - Every generated puzzle must differ from the original bg-only solver on the same
#      input. The generator rejects any sample where that is not true.
#    - The visible cue is consistent across examples: the shadow color can “bite into”
#      the main shape exactly where the parallel copy crosses it.
#
# f) Visual cleanliness
#    - Only one overlap-driving shadowed shape is forced per grid. Other groups are
#      separated by padding, so the output does not devolve into dense multi-color noise.
#
# g) Color reasoning
#    - No color remapping is introduced. Colors still serve only as group identity.
#    - The variant does not use color ids for any visible ordering decision. Cross-group
#      overlaps are forbidden, so the sorted-color loop in the solver remains harmless.
#
# h) New assumptions after the solver change
#    - A shadowed shape still needs exactly one anchor pixel that is 8-adjacent to
#      exactly one vertex, so the chosen start vertex is unambiguous.
#    - The shadow offset must remain non-collinear with every edge direction, matching
#      the original inference rule.
#    - The generator guarantees those conditions and also guarantees at least one
#      shape-vs-shadow overlap, so the new overwrite semantics are observable.

import math

import numpy as np

from arc_utils import get_bg as _bg_color
from arc_utils import bbox_coords, colors_present, draw_line, get_bg, neighbors8, paste, random_free_location

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


def _draw_line_with_policy(
    out: np.ndarray,
    r0: int,
    c0: int,
    r1: int,
    c1: int,
    col: int,
    bg: int,
    *,
    overwrite: bool,
) -> None:
    for rr, cc in _segment_cells(r0, c0, r1, c1):
        if overwrite or out[rr, cc] == bg:
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


def _solve(grid: np.ndarray, *, overwrite: bool) -> np.ndarray:
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

        for (r0, c0), (r1, c1) in drawable_edges:
            _draw_line_with_policy(out, r0, c0, r1, c1, col, bg, overwrite=overwrite)

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
                _draw_line_with_policy(out, rr0, cc0, rr1, cc1, ncol, bg, overwrite=overwrite)

    return out


def main(grid: np.ndarray) -> np.ndarray:
    return _solve(grid, overwrite=True)


def _main_bg_only_reference(grid: np.ndarray) -> np.ndarray:
    return _solve(grid, overwrite=False)


# --------------- generator ---------------

BG = 0
COLORS = [1, 2, 3, 4, 6, 7, 8]
CANVAS_H_RANGE = (15, 23)
CANVAS_W_JITTER = (-3, 4)
CANVAS_W_CLIP = (15, 22)

SHAPE_COUNT_RANGE = (2, 5)
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


def _choose_shadow(
    vertices: list[tuple[int, int]],
    shape_cells: set[tuple[int, int]],
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


def _build_overlap_shadow_case() -> tuple[list[tuple[int, int]], tuple[tuple[int, int], tuple[int, int]]] | None:
    side = int(np.random.randint(*TRIANGLE_SIDE_RANGE))
    pts = _transform_coords([(0, 0), (0, side), (side, side), (1, -1)])
    vertices = pts[:3]
    shadow_pos = pts[3]
    shape_cells = _shape_cells(vertices)
    if shadow_pos in shape_cells:
        return None
    anchor_hits = [v for v in vertices if max(abs(shadow_pos[0] - v[0]), abs(shadow_pos[1] - v[1])) == 1]
    if len(anchor_hits) != 1:
        return None
    anchor = anchor_hits[0]
    offset = (shadow_pos[0] - anchor[0], shadow_pos[1] - anchor[1])
    if offset not in DIR8:
        return None
    if any(ds_r * offset[1] - ds_c * offset[0] == 0 for ds_r, ds_c in _edge_dirs(vertices)):
        return None
    shifted_cells = {(r + offset[0], c + offset[1]) for r, c in shape_cells}
    if not (shape_cells & shifted_cells):
        return None
    return vertices, (anchor, offset)


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


def _shift_mask(src: np.ndarray, offset: tuple[int, int]) -> np.ndarray:
    dr, dc = offset
    h, w = src.shape
    shifted = np.zeros_like(src)
    r_src = slice(max(0, -dr), h - max(0, dr))
    c_src = slice(max(0, -dc), w - max(0, dc))
    r_dst = slice(max(0, dr), h - max(0, -dr))
    c_dst = slice(max(0, dc), w - max(0, -dc))
    shifted[r_dst, c_dst] = src[r_src, c_src]
    return shifted


def _paint_shadow(out: np.ndarray, shape_color: int, shadow_color: int, offset: tuple[int, int], *, overwrite: bool) -> int:
    src = out == shape_color
    shifted = _shift_mask(src, offset)
    overlap_cells = int((shifted & src).sum())
    if overwrite:
        out[shifted] = shadow_color
    else:
        out[shifted & (out == BG)] = shadow_color
    return overlap_cells


def _make_shape_patch(
    shape_color: int,
    shadow_color: int | None,
    *,
    require_overlap_priority: bool,
) -> tuple[np.ndarray, np.ndarray] | None:
    if shadow_color is not None and require_overlap_priority:
        case = _build_overlap_shadow_case()
        if case is None:
            return None
        vertices, shadow = case
    else:
        vertices = _make_vertices()
        shadow = None
        if shadow_color is not None:
            shape_cells = _shape_cells(vertices)
            shadow = _choose_shadow(vertices, shape_cells)
            if shadow is None:
                return None

    cells = _shape_cells(vertices)
    r0, c0, r1, c1 = _patch_bbox(cells, vertices, shadow)
    h, w = r1 - r0 + 1, c1 - c0 + 1
    out = np.full((h, w), BG, dtype=int)
    inp = np.full((h, w), BG, dtype=int)

    shifted_vertices = [(r - r0, c - c0) for r, c in vertices]
    for a, b in _shape_edges(shifted_vertices):
        draw_line(out, a[0], a[1], end_r=b[0], end_c=b[1], color=shape_color)
    for r, c in shifted_vertices:
        inp[r, c] = shape_color

    overlap_cells = 0
    if shadow is not None:
        (ar, ac), offset = shadow
        anchor = (ar - r0, ac - c0)
        overlap_cells = _paint_shadow(out, shape_color, int(shadow_color), offset, overwrite=True)
        sa = (anchor[0] + offset[0], anchor[1] + offset[1])
        if not (0 <= sa[0] < h and 0 <= sa[1] < w) or inp[sa] != BG:
            return None
        inp[sa] = int(shadow_color)
        nb_hits = sum(
            1
            for rr, cc in neighbors8(sa[0], sa[1])
            if 0 <= rr < h and 0 <= cc < w and inp[rr, cc] == shape_color
        )
        if nb_hits != 1:
            return None

    if require_overlap_priority:
        if shadow is None or overlap_cells <= 0:
            return None
        if np.array_equal(_main_bg_only_reference(inp), out):
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


def _sample_palette(shape_count: int, has_distractor: bool) -> tuple[list[int], int, int | None] | None:
    total = shape_count + 1 + int(has_distractor)
    if total > len(COLORS):
        return None
    palette = list(np.random.permutation(COLORS)[:total])
    shape_colors = [int(v) for v in palette[:shape_count]]
    shadow_color = int(palette[shape_count])
    distractor_color = int(palette[-1]) if has_distractor else None
    return shape_colors, shadow_color, distractor_color


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
    shape_colors, shadow_color, distractor_color = palette
    priority_idx = int(np.random.randint(shape_count))

    total_colors = shape_count + 1 + int(has_distractor)

    for idx, shape_color in enumerate(shape_colors):
        placed = False
        for _ in range(SHAPE_PLACEMENT_RETRIES):
            patch = _make_shape_patch(
                int(shape_color),
                int(shadow_color) if idx == priority_idx else None,
                require_overlap_priority=(idx == priority_idx),
            )
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
    if np.array_equal(_main_bg_only_reference(inp), out):
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
