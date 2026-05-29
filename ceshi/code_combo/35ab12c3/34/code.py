# Combo variant of 35ab12c3 (index 34)
# combo_spec: D20/0+D22/2
# Name: 顶点重着色为阴影颜色+阴影颜色由映射决定
#
# Design analysis for the fused variant
# ------------------------------------
# a) 具体化与行为追踪
#    原题里，输入中某个顶点旁边的异色像素同时承担两件事：
#    1) 它提示“这个顶点是阴影起点”；
#    2) 它自己的颜色就是输出阴影线的颜色。
#    融合后，这两件事被拆开并进一步叠加：
#    - 输入里的阴影提示像素统一改成提示色 4；
#    - 输出里的真正阴影色不再来自输入锚点，而是由形状主色通过固定映射决定：
#      1->6, 2->7, 3->8；
#    - 同时，凡是带阴影的形状，其所有顶点在输出里也改成同一个映射阴影色。
#    小例子：输入若有两个 1 色顶点，左上邻接一个 4 色提示点。
#    原题输出会保留两个 1 色顶点、画 1 色主线、再画提示点原色的阴影。
#    本融合变体输出则是：主线中段仍为 1，平行阴影改成 6，两个端点也改成 6。
#    差异因此集中在两处：阴影色的来源改为“按主色查表”，顶点最终颜色也跟着阴影色走。
# b) 复杂度校准
#    若仍允许任意提示色、任意形状色和可选无阴影对象，人类很难在 2-4 个样例中区分
#    “提示色只是提示”与“输出色由映射决定”。因此这里把形状色限制为 {1,2,3}，
#    提示色固定为 4，阴影色固定为 {6,7,8}，并且每个生成样例中的每个形状都带阴影。
#    这样每个样例都会重复出现“原色线段 + 映射阴影色 + 顶点同阴影色”的证据。
# c) 隐式假设审查（main 相关）
#    - `out = g.copy()` 在原题中会把输入里的锚点色直接保留到输出；这里不成立，
#      因为 4 只是提示色，不应直接出现在输出中，所以要先把 4 清成背景。
#    - “寻找 seed 顶点”原来接受任意异色邻居；现在只能把 4 识别为合法提示，
#      否则会把别的颜色误当作阴影线索。
#    - “carried” 原来携带 (偏移, 阴影色)；现在只需要携带偏移，阴影色由主色查映射表。
#    - 顶点改色必须发生在主线和阴影都画完之后；否则若先覆写顶点颜色，
#      阴影复制时就可能丢失基于主色的源形状。
# d) 放置合法性
#    生成器仍是 output-first 的局部 patch 放置。每个 patch 的 bbox 同时覆盖：
#    形状主线、完整平移后的阴影、以及那个 4 色提示像素。全局放置继续使用
#    `random_free_location(..., padding=1, connectivity=8)`，确保对象之间不重叠并保留缓冲带。
# e) 视觉可观察性
#    解题者能直接看到输入里所有提示点都同色 4，但输出阴影却随主色变化，
#    且顶点颜色总是和阴影一致、与主线不同。这个“颜色关系反转”在肉眼上非常显著。
# f) 可视化整洁度
#    统一提示色减少了输入中的杂色；输出只保留主线与其单层阴影，不引入多层嵌套或额外标记。
#    由于不同对象之间保留至少 1 格缓冲，网格不会出现难以分辨的黏连。
# g) 变化的可推断性
#    颜色变化不是依赖不可见排序，而是固定可见配对：
#    1->6, 2->7, 3->8。提示点始终是 4，因此人类能从多组重复配对中归纳出：
#    “4 只是信号，真正输出色由形状本色决定，而且顶点会并入阴影色。”
# h) main 的新隐式假设
#    main 现在假设：只有 {1,2,3} 是形状顶点色，只有 4 是阴影提示色；
#    每个形状至多有一个起始提示点，且它只 8 邻接于该形状的起始顶点。
#    generate_input 通过固定色域、每形状恰好一个提示点、以及输出级 padding>=1
#    保证这些假设始终成立。

import math

import numpy as np

from arc_utils import bbox_coords, colors_present, draw_line, get_bg, neighbors8, paste, random_free_location
from arc_utils import get_bg as _bg_color

DIR8 = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]

BG = 0
CUE_COLOR = 4
SHAPE_COLORS = [1, 2, 3]
SHADOW_COLOR_MAP = {1: 6, 2: 7, 3: 8}

CANVAS_H_RANGE = (15, 23)
CANVAS_W_JITTER = (-3, 4)
CANVAS_W_CLIP = (15, 22)

SHAPE_COUNT_RANGE = (2, 4)  # 2-3 shapes, all shadowed
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

SHAPE_PLACEMENT_RETRIES = 40
GENERATE_RETRIES = 300


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


def _patch_bbox(
    shape_cells: set[tuple[int, int]],
    vertices: list[tuple[int, int]],
    shadow: tuple[tuple[int, int], tuple[int, int]],
) -> tuple[int, int, int, int]:
    coords = list(shape_cells) + vertices
    (ar, ac), (dr, dc) = shadow
    coords.append((ar + dr, ac + dc))
    coords.extend((r + dr, c + dc) for r, c in shape_cells)
    r0, c0, r1, c1 = bbox_coords(np.array(coords, dtype=int))
    return int(r0), int(c0), int(r1), int(c1)


def _paint_shadow(out: np.ndarray, shape_color: int, shadow_color: int, offset: tuple[int, int]) -> None:
    dr, dc = offset
    h, w = out.shape
    src = out == shape_color
    shifted = np.zeros_like(src)
    r_src = slice(max(0, -dr), h - max(0, dr))
    c_src = slice(max(0, -dc), w - max(0, dc))
    r_dst = slice(max(0, dr), h - max(0, -dr))
    c_dst = slice(max(0, dc), w - max(0, -dc))
    shifted[r_dst, c_dst] = src[r_src, c_src]
    mask = shifted & (out == BG)
    out[mask] = shadow_color


# Overview:
#   Reconstruct each sparse shape by joining same-color vertices, infer a single 4-colored
#   cue pixel as the shadow offset, then render the shadow with a deterministic color map.
# Input structure:
#   Colors 1/2/3 are sparse vertices of shapes. Color 4 is a cue pixel placed in the
#   8-neighborhood of one start vertex. No distractors are generated in this combo.
# Rule:
#   Draw the main H/V/45deg edges in the original shape color, draw the parallel shadow
#   copy in the mapped color 1->6, 2->7, 3->8, and finally recolor all vertices of that
#   shape to the same mapped shadow color.
# Key insight:
#   The cue pixel's color no longer tells you the shadow color; it only marks the offset.
#   The real shadow/vertex color is determined by the shape's own color.
def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = _bg_color(g)

    out = g.copy()
    out[out == CUE_COLOR] = bg

    for col in SHAPE_COLORS:
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
                if 0 <= rr < h and 0 <= cc < w and int(g[rr, cc]) == CUE_COLOR:
                    seed_idxs.append(i)
                    break

        if seed_idxs:
            start = min(seed_idxs, key=lambda i: (cycle[i][0], cycle[i][1]))
        else:
            start = min(range(len(cycle)), key=lambda i: (cycle[i][0], cycle[i][1]))

        cycle = cycle[start:] + cycle[:start]

        carried_offsets: list[tuple[int, int]] = []
        sr0, sc0 = cycle[0]
        for drn, dcn in DIR8:
            rr = sr0 + drn
            cc = sc0 + dcn
            if 0 <= rr < h and 0 <= cc < w and int(g[rr, cc]) == CUE_COLOR:
                carried_offsets.append((drn, dcn))

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

        shadow_color = SHADOW_COLOR_MAP[col]
        for (r0, c0), (r1, c1) in drawable_edges:
            drs, dcs = _step_dir(r0, c0, r1, c1)
            for drn, dcn in carried_offsets:
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
                _draw_line_bg_only(out, rr0, cc0, rr1, cc1, shadow_color, bg)

        if carried_offsets:
            for r, c in cycle:
                out[r, c] = shadow_color

    return out


def _make_shape_patch(shape_color: int) -> tuple[np.ndarray, np.ndarray] | None:
    vertices = _make_vertices()
    cells = _shape_cells(vertices)
    shadow = _choose_shadow(vertices, cells)
    if shadow is None:
        return None

    r0, c0, r1, c1 = _patch_bbox(cells, vertices, shadow)
    h, w = r1 - r0 + 1, c1 - c0 + 1
    out = np.full((h, w), BG, dtype=int)
    inp = np.full((h, w), BG, dtype=int)

    shifted_vertices = [(r - r0, c - c0) for r, c in vertices]
    for a, b in _shape_edges(shifted_vertices):
        draw_line(out, a[0], a[1], end_r=b[0], end_c=b[1], color=shape_color)
    for r, c in shifted_vertices:
        inp[r, c] = shape_color

    (ar, ac), offset = shadow
    anchor = (ar - r0, ac - c0)
    shadow_color = SHADOW_COLOR_MAP[shape_color]
    _paint_shadow(out, shape_color, shadow_color, offset)
    for r, c in shifted_vertices:
        out[r, c] = shadow_color

    sa = (anchor[0] + offset[0], anchor[1] + offset[1])
    if not (0 <= sa[0] < h and 0 <= sa[1] < w) or inp[sa] != BG:
        return None
    inp[sa] = CUE_COLOR

    cue_hits = sum(
        1
        for rr, cc in neighbors8(anchor[0], anchor[1])
        if 0 <= rr < h and 0 <= cc < w and inp[rr, cc] == CUE_COLOR
    )
    if cue_hits != 1:
        return None

    if not np.array_equal(main(inp), out):
        return None
    return inp, out


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


def _sample_shape_colors(shape_count: int) -> list[int]:
    return list(np.random.permutation(SHAPE_COLORS)[:shape_count])


def _gen_attempt() -> np.ndarray | None:
    h, w = _sample_canvas_size()
    out = np.full((h, w), BG, dtype=int)
    inp = np.full((h, w), BG, dtype=int)

    shape_count = int(np.random.randint(*SHAPE_COUNT_RANGE))
    shape_colors = _sample_shape_colors(shape_count)

    for shape_color in shape_colors:
        placed = False
        for _ in range(SHAPE_PLACEMENT_RETRIES):
            patch = _make_shape_patch(int(shape_color))
            if patch is None:
                continue
            if _try_place(inp, out, *patch):
                placed = True
                break
        if not placed:
            return None

    if get_bg(inp) != BG or get_bg(out) != BG:
        return None
    input_colors = set(colors_present(inp, exclude=[BG]))
    expected_input_colors = set(shape_colors) | {CUE_COLOR}
    if input_colors != expected_input_colors:
        return None
    expected_output_colors = set(shape_colors) | {SHADOW_COLOR_MAP[col] for col in shape_colors}
    output_colors = set(colors_present(out, exclude=[BG]))
    if output_colors != expected_output_colors:
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
