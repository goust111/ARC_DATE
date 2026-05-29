# Variant of 35ab12c3
# Dimension: D2 — 路径方向类型 (path)
# Change: straight_ray + diagonal_path——顶点间通过水平、垂直或严格45度对角线连接，要求dr==0或... → 仅对角线路径（禁止正交）

import math

import numpy as np

from arc_utils import get_bg as _bg_color
from arc_utils import bbox_coords, colors_present, draw_line, get_bg, neighbors8, paste, random_free_location

# Design analysis for the D2 variant
# ----------------------------------
# a) Concrete behavior + trace
#    - 原题里，同色顶点只要 H / V / 45° 对齐就会被连接；本变体里只有严格 45° 对角线可连接。
#    - 因此输入中的“可行动对象”只保留两类：
#      1) 纯对角线链：多个同色点共线在一条斜线上，输出补齐整条斜线。
#      2) 纯对角菱形/风筝：按角度环序得到的每一条边都必须是 45° 对角线。
#    - 为了让“禁止正交”肉眼可见，generator 固定加入一个正交 distractor：
#      例如同色点位于同一行的 (2,2),(2,6)。旧规则会补成横线；新规则下这组点完全不动。
#    - 具体追踪：
#      old: _aligned_8((2,2),(2,6)) == True -> 边可画 -> 输出出现横线。
#      new: _aligned_diag_only((2,2),(2,6)) == False -> drawable_edges 为空 -> 输出保持两个孤立点。
#
# b) Complexity calibration
#    - 如果只把 _aligned 改成“只对角”，但 generator 继续产出矩形/水平线，大量样本会退化为“什么都不发生”。
#    - 如果仍允许任意折线，solver 的 angle-cycle 排序会把某些 3 点/5 点形状重新配对，导致非预期边。
#    - 所以生成器只保留两个低分支但高辨识度的族：对角链 + 4 顶点菱形/风筝。
#    - 每个样例都加入 1 个正交 distractor，保证“旧 solver 与新 solver 输出不同”的核心证据稳定出现。
#
# c) Hidden assumptions audit on main()
#    - _segment_cells 只有在端点真的沿允许方向对齐时才安全；本变体继续依赖这一点，所以所有调用前都先过 aligned 判定。
#    - _clear_in_original 原本允许中间穿过同色点；这在对角链里仍然成立，且正好让多顶点共线链塌缩成一条完整对角线。
#    - angle_cycle 不显式保存原拓扑，只按几何角度排序；因此 generator 必须只产出“角度环序本身就是目标边序”的顶点集。
#    - carried neighbor 从起始顶点复制影子；generator 必须保证 shadow anchor 只邻接一个顶点，避免 solver 选错 seed。
#
# d) Placement legality
#    - patch 仍采用 output-first：先画完整对角形状和影子，再把输入退化成顶点 + 单个 shadow-anchor 像素。
#    - 所有 patch 在全局 canvas 上用 random_free_location 放置，并保留 >=1 背景缓冲，避免跨对象遮挡。
#    - shadow offset 仍限制为 8 邻域，但不能与任何边方向共线，否则整条平移影子会落回主形状。
#
# e) Visual observability
#    - 对人类解题者，最关键的对比是：
#      “斜着摆的同色点会被连起来；横着/竖着摆的同色点不会。”
#    - 所以样例里同时出现可行动的对角对象和不可行动的正交对象，2-3 组样例即可归纳新规则。
#
# f) Visual cleanliness
#    - 只保留 2 种主形状，减少自交和复杂覆盖。
#    - 阴影只复制一层，且对象间强制留白；不同颜色区域不会挤在一起。
#
# g) Color inferability
#    - 颜色不参与排序、不参与映射；每种颜色只表示一个对象或一个影子层。
#    - solver 按颜色分别处理只是为了独立连线，不编码额外不可见规则。
#
# New solver assumptions introduced by the variant
# -----------------------------------------------
# 1) 真正会变化的 shape 颜色组必须至少存在一条对角可画边。
# 2) 正交同色点组如果出现，会被当成 inert distractor；generator 固定这样构造并显式验证 old/new solver 分歧。
# 3) 任意带 shadow 的对象都只保留一个不同色邻居像素，这样 main() 的 seed 选择与 generator 的 anchor 一致。

DIR8 = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]


def _aligned_diag_only(r0: int, c0: int, r1: int, c1: int) -> bool:
    dr = r1 - r0
    dc = c1 - c0
    return dr != 0 and abs(dr) == abs(dc)


def _aligned_legacy(r0: int, c0: int, r1: int, c1: int) -> bool:
    dr = r1 - r0
    dc = c1 - c0
    if dr == 0 and dc != 0:
        return True
    if dc == 0 and dr != 0:
        return True
    if dr != 0 and abs(dr) == abs(dc):
        return True
    return False


def _aligned_delta_diag_only(dr: int, dc: int) -> bool:
    return dr != 0 and abs(dr) == abs(dc)


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


def _solve(grid: np.ndarray, aligned_fn) -> np.ndarray:
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
            if not aligned_fn(r0, c0, r1, c1):
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
                rr0 = r0 + drn
                cc0 = c0 + dcn
                rr1 = r1 + drn
                cc1 = c1 + dcn
                if not (0 <= rr0 < h and 0 <= cc0 < w and 0 <= rr1 < h and 0 <= cc1 < w):
                    continue
                if not aligned_fn(rr0, cc0, rr1, cc1):
                    continue
                if not _clear_in_original(g, rr0, cc0, rr1, cc1, bg, allow=set()):
                    continue
                _draw_line_bg_only(out, rr0, cc0, rr1, cc1, ncol, bg)

    return out


def main(grid: np.ndarray) -> np.ndarray:
    return _solve(grid, _aligned_diag_only)


def _legacy_main(grid: np.ndarray) -> np.ndarray:
    return _solve(grid, _aligned_legacy)


# --------------- generator (D2 diagonal-only variant) ---------------

BG = 0
COLORS = [1, 2, 3, 4, 6, 7, 8]
CANVAS_H_RANGE = (15, 23)
CANVAS_W_JITTER = (-3, 4)
CANVAS_W_CLIP = (15, 22)

SHAPE_COUNT_RANGE = (2, 5)
MAX_SHADOWS_CAP = 3
DISTRACTOR_PROB = 1.0
PLACEMENT_PADDING = 1
PLACEMENT_PAD_CONNECTIVITY = 8

DIAG_CHAIN_VERTICES_RANGE = (2, 6)
DIAG_CHAIN_TOTAL_LENGTH = 12
DIAG_CHAIN_MIN_GAP = 2
KITE_ARM_RANGE = (2, 6)
SHAPE_WEIGHTS = np.array([0.55, 0.45], dtype=float)

ORTH_DISTRACTOR_COUNT_RANGE = (2, 4)
ORTH_DISTRACTOR_TOTAL_SPAN = 10
ORTH_DISTRACTOR_MIN_GAP = 2

SHAPE_PLACEMENT_RETRIES = 40
GENERATE_RETRIES = 400


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


def _sample_positions(n_vertices: int, total_span: int, min_gap: int) -> list[int]:
    remaining = total_span
    pos = [0]
    for i in range(n_vertices - 1):
        steps_left = n_vertices - 2 - i
        gap_hi = remaining - min_gap * steps_left
        gap = int(np.random.randint(min_gap, gap_hi + 1))
        pos.append(pos[-1] + gap)
        remaining -= gap
    return pos


def _build_diag_chain() -> list[tuple[int, int]]:
    n_vertices = int(np.random.randint(*DIAG_CHAIN_VERTICES_RANGE))
    pos = _sample_positions(n_vertices, DIAG_CHAIN_TOTAL_LENGTH, DIAG_CHAIN_MIN_GAP)
    return _transform_coords([(x, x) for x in pos])


def _build_kite() -> list[tuple[int, int]]:
    a = int(np.random.randint(*KITE_ARM_RANGE))
    b = int(np.random.randint(*KITE_ARM_RANGE))
    return _transform_coords([(0, b), (a, a + b), (a + b, a), (b, 0)])


_SHAPE_BUILDERS = [_build_diag_chain, _build_kite]


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


def _ordered_diag_vertices(coords: list[tuple[int, int]]) -> list[tuple[int, int]] | None:
    cycle = _angle_cycle(coords)
    for a, b in _shape_edges(cycle):
        if not _aligned_delta_diag_only(b[0] - a[0], b[1] - a[1]):
            return None
    return cycle


def _make_vertices() -> list[tuple[int, int]] | None:
    weights = SHAPE_WEIGHTS / SHAPE_WEIGHTS.sum()
    for _ in range(20):
        idx = int(np.random.choice(len(_SHAPE_BUILDERS), p=weights))
        vertices = _ordered_diag_vertices(_SHAPE_BUILDERS[idx]())
        if vertices is not None:
            return vertices
    return None


def _choose_shadow(
    vertices: list[tuple[int, int]],
    shape_cells: set[tuple[int, int]],
) -> tuple[tuple[int, int], tuple[int, int]] | None:
    directions = _edge_dirs(vertices)
    for idx in np.random.permutation(len(vertices)):
        anchor = vertices[int(idx)]
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
        if candidates:
            return anchor, candidates[int(np.random.randint(len(candidates)))]
    return None


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


def _make_shape_patch(shape_color: int, shadow_color: int | None) -> tuple[np.ndarray, np.ndarray] | None:
    vertices = _make_vertices()
    if vertices is None:
        return None
    cells = _shape_cells(vertices)

    shadow = None
    if shadow_color is not None:
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

    if shadow is not None:
        (ar, ac), offset = shadow
        anchor = (ar - r0, ac - c0)
        _paint_shadow(out, shape_color, int(shadow_color), offset)
        sa = (anchor[0] + offset[0], anchor[1] + offset[1])
        if not (0 <= sa[0] < h and 0 <= sa[1] < w) or inp[sa] != BG:
            return None
        inp[sa] = int(shadow_color)
        nb_hits = sum(
            1
            for rr, cc in neighbors8(anchor[0], anchor[1])
            if 0 <= rr < h and 0 <= cc < w and inp[rr, cc] == shadow_color
        )
        if nb_hits != 1:
            return None

    if not np.array_equal(main(inp), out):
        return None
    return inp, out


def _make_orth_distractor_patch(color: int) -> tuple[np.ndarray, np.ndarray]:
    count = int(np.random.randint(*ORTH_DISTRACTOR_COUNT_RANGE))
    pos = _sample_positions(count, ORTH_DISTRACTOR_TOTAL_SPAN, ORTH_DISTRACTOR_MIN_GAP)
    horizontal = bool(np.random.rand() < 0.5)
    if horizontal:
        patch = np.full((1, pos[-1] + 1), BG, dtype=int)
        for c in pos:
            patch[0, c] = color
    else:
        patch = np.full((pos[-1] + 1, 1), BG, dtype=int)
        for r in pos:
            patch[r, 0] = color
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
        if not _try_place(inp, out, *_make_orth_distractor_patch(distractor_color)):
            return None

    if get_bg(inp) != BG or get_bg(out) != BG:
        return None
    if len(colors_present(inp, exclude=[BG])) != total_colors:
        return None
    if not np.array_equal(main(inp), out):
        return None
    if has_distractor and np.array_equal(_legacy_main(inp), out):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GENERATE_RETRIES):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError(f"generate_input failed after {GENERATE_RETRIES} attempts")
