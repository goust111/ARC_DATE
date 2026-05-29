# Variant of 35ab12c3
# Dimension: D3 — 顶点排序与连接拓扑 (path)
# Change: 极角排序（angle_cycle）——所有同色顶点按围绕质心的极角排序，2个顶点连直线段，3+个顶点按顺序连成闭合多边形... → 星形连接（所有顶点连到质心）
#
# Design analysis for this variant
# --------------------------------
# source_task: None
#
# a) 具体化与行为追踪
# - 每个 shape 颜色在输入里仍然只出现少量“顶点像素”，但这些顶点不再被解释为一个环的顶点序列。
# - 变体中，solver 先求同色顶点集合的质心；若质心正好落在整数网格且每个顶点都能用 H/V/45° 直线连到该点，
#   则该点就是中心 hub。否则退化到“离浮点质心最近的已有顶点”作为 hub。
# - 输出不再把顶点按极角首尾相连；而是把每个顶点都直接连到 hub，形成放射状 star。
# - 影子机制保持：选择一个带异色 8 邻居的 seed 顶点，把该邻居颜色沿每一条主 spoke 做同偏移平移复制。
# - 小例子：
#   输入同色顶点为 {(0,2),(2,0),(2,4),(4,2)}，质心是 (2,2)。
#   原题输出：按极角得到一个菱形外轮廓。
#   本变体输出：四个顶点都连到 (2,2)，得到十字/菱形骨架的星形。
#   差异在“边”的定义：原题边来自顶点-顶点，变体边来自顶点-hub。
#
# b) 复杂度校准
# - 若继续大量生成 2 顶点对象，D3 改动几乎不可见，因为 2 点时原题和变体都只是一条线。
# - 因此 generator 收紧到 4 或 6 顶点的辐射对象，方向来自 8 邻域中的 2 或 3 对对向射线。
# - 这样每个样例里大多数 shape 都会稳定表现为 star，而不是偶然退化成原题行为。
# - 方向只从少量离散档位中取值，利于人类从 2-4 个样例中归纳。
#
# c) 原 main() 的隐式假设审查
# - `_angle_cycle(...)` 假设“3+ 顶点要围绕质心排成一圈”；本变体不成立，必须移除。
# - `cycle = cycle[start:] + cycle[:start]` 假设“选定 seed 后，后续边集仍由相邻顶点组成”；本变体不成立。
# - `edges = [(cycle[i], cycle[(i + 1) % len(cycle)]) ...]` 直接编码了闭环拓扑；必须改为 `vertex -> hub`。
# - shadow 的 `carried` 逻辑只依赖“有一个 seed 顶点及其邻居偏移”，这在变体中仍成立，可以保留。
# - `_clear_in_original(... allow={col})` 在旧题中允许边穿过同色顶点；新题中同样无害，保留该宽松性。
# - 旧题没有检查“中心端点是否被别色占据”，因为边终点本来总是同色顶点；新题 hub 常是背景格，
#   所以 solver 需要额外要求：若 hub 不是已有顶点，则该格必须是背景，避免画线终点被异色堵住。
#
# d) 放置合法性
# - generator 直接构造局部 star patch：先决定 hub=原点，再放置对向 spokes 的端点。
# - 由于每条 spoke 都是独立射线，局部 occupied cells 可精确计算，bbox 也可精确包含 shape + shadow。
# - 整体画布仍用 `random_free_location(... padding=1)` 放置 patch，保证不同对象间留出背景缓冲。
# - shadow offset 仍限制为非共线方向，避免平移后的影子与主 spokes 大面积重合。
#
# e) 视觉可观察性
# - 规则只依赖几何结构：顶点围绕中心、输出向中心汇聚，不依赖颜色 id 排序。
# - 生成的 4/6 端点对象，在输出里会出现明显中心汇聚点；这与原题的外轮廓封边很容易肉眼区分。
# - 同一网格中重复出现多个 star，可为解题者提供“所有 shape 都向中心连”的重复证据。
#
# f) 可视化整洁度
# - 只保留单层 shadow，不引入多级递归影子，避免中心区域叠加过多。
# - spokes 仅使用 H/V/45° 八向直线，输出结构规整；对象间有 padding，减少视觉打架。
# - 每个 shape 只使用一个中心 hub，不做多中心或交叉重连，避免杂乱。
#
# g) 颜色变化可推断性
# - 颜色规则未改变：每个组颜色唯一，shadow 用另一个唯一颜色，仅由 seed 邻居触发。
# - 没有隐藏的颜色映射，也不按颜色值做排序或分支。
#
# 9) 新 main() 的隐式假设与 generator 保证
# - 假设 1：generator 主要产出整数质心的星形端点集；由对向等长 spokes 保证。
# - 假设 2：hub 到每个顶点必须 H/V/45° 可达；由方向只取 8 邻域单位向量保证。
# - 假设 3：若存在 shadow，其 anchor 像素必须仅邻接一个顶点；由 `_choose_shadow` 构造保证。
# - 假设 4：hub 若不是原输入顶点，则该格为空；局部 patch 构造保证。
# - 假设 5：不同 shape 的输出填充区域互不相撞；全局放置使用输出 patch 做碰撞检测保证。

import numpy as np

from arc_utils import bbox_coords, colors_present, draw_line, get_bg, neighbors8, paste, random_free_location
from arc_utils import get_bg as _bg_color

DIR8 = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
OPPOSITE_PAIRS = [
    ((-1, 0), (1, 0)),
    ((0, -1), (0, 1)),
    ((-1, -1), (1, 1)),
    ((-1, 1), (1, -1)),
]


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


def _centroid_hub(points_rc: list[tuple[int, int]]) -> tuple[int, int]:
    n = len(points_rc)
    sum_r = sum(r for r, _ in points_rc)
    sum_c = sum(c for _, c in points_rc)
    if sum_r % n == 0 and sum_c % n == 0:
        hub = (sum_r // n, sum_c // n)
        if all(_aligned_8(r, c, hub[0], hub[1]) for r, c in points_rc if (r, c) != hub):
            return hub

    cr = sum_r / n
    cc = sum_c / n
    return min(points_rc, key=lambda p: ((p[0] - cr) ** 2 + (p[1] - cc) ** 2, p[0], p[1]))


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = _bg_color(g)

    out = g.copy()

    colors = sorted(set(int(v) for v in np.unique(g)) - {bg})
    for col in colors:
        pts_rc = sorted(tuple(map(int, p)) for p in np.argwhere(g == col))
        if len(pts_rc) < 2:
            continue

        seed_vertices = []
        for r, c in pts_rc:
            for drn, dcn in DIR8:
                rr = r + drn
                cc = c + dcn
                if 0 <= rr < h and 0 <= cc < w:
                    v = int(g[rr, cc])
                    if v != bg and v != col:
                        seed_vertices.append((r, c))
                        break
        seed = min(seed_vertices) if seed_vertices else min(pts_rc)

        carried: list[tuple[int, int, int]] = []
        for drn, dcn in DIR8:
            rr = seed[0] + drn
            cc = seed[1] + dcn
            if 0 <= rr < h and 0 <= cc < w:
                v = int(g[rr, cc])
                if v != bg and v != col:
                    carried.append((drn, dcn, v))

        hub = _centroid_hub(pts_rc)
        pts_set = set(pts_rc)
        if hub not in pts_set:
            hr, hc = hub
            if not (0 <= hr < h and 0 <= hc < w):
                continue
            if int(g[hr, hc]) != bg:
                continue

        drawable_edges: list[tuple[tuple[int, int], tuple[int, int]]] = []
        for r0, c0 in pts_rc:
            if (r0, c0) == hub:
                continue
            r1, c1 = hub
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


# --------------- generator (star-centroid variant) ---------------

# ---- CFG: canvas and palette ----
BG = 0
COLORS = [1, 2, 3, 4, 6, 7, 8]
CANVAS_H_RANGE = (15, 23)
CANVAS_W_JITTER = (-3, 4)
CANVAS_W_CLIP = (15, 22)

# ---- CFG: structural counts ----
SHAPE_COUNT_RANGE = (2, 5)
MAX_SHADOWS_CAP = 3
DISTRACTOR_PROB = 0.5
PLACEMENT_PADDING = 1
PLACEMENT_PAD_CONNECTIVITY = 8

# ---- CFG: star geometry ----
STAR_PAIR_COUNT_CHOICES = np.array([2, 3], dtype=int)   # 4 or 6 vertices
STAR_PAIR_COUNT_WEIGHTS = np.array([0.65, 0.35], dtype=float)
STAR_ARM_LEN_RANGE = (2, 7)                             # 2-6 inclusive

# ---- CFG: distractor geometry ----
DISTRACTOR_COUNT_RANGE = (1, 4)
DISTRACTOR_BBOX_RANGE = (2, 5)
DISTRACTOR_RETRIES = 50

# ---- CFG: retry budgets ----
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


def _make_vertices() -> tuple[list[tuple[int, int]], tuple[int, int]]:
    pair_weights = STAR_PAIR_COUNT_WEIGHTS / STAR_PAIR_COUNT_WEIGHTS.sum()
    pair_count = int(np.random.choice(STAR_PAIR_COUNT_CHOICES, p=pair_weights))
    pair_ids = np.random.choice(len(OPPOSITE_PAIRS), size=pair_count, replace=False).tolist()
    vertices: list[tuple[int, int]] = []
    for idx in pair_ids:
        d0, d1 = OPPOSITE_PAIRS[int(idx)]
        arm_len = int(np.random.randint(*STAR_ARM_LEN_RANGE))
        vertices.append((d0[0] * arm_len, d0[1] * arm_len))
        vertices.append((d1[0] * arm_len, d1[1] * arm_len))
    vertices = sorted(set(vertices))
    return vertices, (0, 0)


def _shape_cells(vertices: list[tuple[int, int]], hub: tuple[int, int]) -> set[tuple[int, int]]:
    cells: set[tuple[int, int]] = set()
    for vertex in vertices:
        cells.update(_gen_segment_cells(vertex, hub))
    return cells


def _edge_dirs(vertices: list[tuple[int, int]], hub: tuple[int, int]) -> list[tuple[int, int]]:
    return [_step(vertex, hub) for vertex in vertices if vertex != hub]


def _choose_shadow(
    vertices: list[tuple[int, int]],
    hub: tuple[int, int],
    shape_cells: set[tuple[int, int]],
) -> tuple[tuple[int, int], tuple[int, int]] | None:
    anchor = min(vertices)
    directions = _edge_dirs(vertices, hub)
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
    hub: tuple[int, int],
    shadow: tuple[tuple[int, int], tuple[int, int]] | None,
) -> tuple[int, int, int, int]:
    coords = list(shape_cells) + vertices + [hub]
    if shadow is not None:
        (_ar, _ac), (dr, dc) = shadow
        coords.extend((r + dr, c + dc) for r, c in shape_cells)
    r0, c0, r1, c1 = bbox_coords(np.array(coords, dtype=int))
    return int(r0), int(c0), int(r1), int(c1)


def _paint_shadow(out: np.ndarray, shape_color: int, shadow_color: int, offset: tuple[int, int]) -> None:
    dr, dc = offset
    h, w = out.shape
    src = (out == shape_color)
    shifted = np.zeros_like(src)
    r_src = slice(max(0, -dr), h - max(0, dr))
    c_src = slice(max(0, -dc), w - max(0, dc))
    r_dst = slice(max(0, dr), h - max(0, -dr))
    c_dst = slice(max(0, dc), w - max(0, -dc))
    shifted[r_dst, c_dst] = src[r_src, c_src]
    mask = shifted & (out == BG)
    out[mask] = shadow_color


def _make_shape_patch(shape_color: int, shadow_color: int | None) -> tuple[np.ndarray, np.ndarray] | None:
    vertices, hub = _make_vertices()
    cells = _shape_cells(vertices, hub)

    shadow = None
    if shadow_color is not None:
        shadow = _choose_shadow(vertices, hub, cells)
        if shadow is None:
            return None

    r0, c0, r1, c1 = _patch_bbox(cells, vertices, hub, shadow)
    h, w = r1 - r0 + 1, c1 - c0 + 1
    out = np.full((h, w), BG, dtype=int)
    inp = np.full((h, w), BG, dtype=int)

    shifted_vertices = [(r - r0, c - c0) for r, c in vertices]
    shifted_hub = (hub[0] - r0, hub[1] - c0)
    for vertex in shifted_vertices:
        draw_line(out, vertex[0], vertex[1], end_r=shifted_hub[0], end_c=shifted_hub[1], color=shape_color)
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
        if sum(1 for vr, vc in shifted_vertices if max(abs(sa[0] - vr), abs(sa[1] - vc)) == 1) != 1:
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
