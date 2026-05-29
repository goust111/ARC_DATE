# Variant of 35ab12c3
# Dimension: D8 — 终止条件/端点规则 (dst)
# Change: 无独立终点——路径在同色像素对之间直接连接，终点就是配对的另一个像素
# ... → 连接到最近的不同色像素
#
# Design analysis
# a) Concrete behavior:
#    原题把一个颜色组视为若干顶点，并在同色顶点之间连 H/V/45 度边；若起点旁有异色邻居，
#    则把该异色像素沿同样的边做平行复制。这个变体保留“颜色组 + seed 邻居 + 两阶段绘制”
#    的骨架，但把“边的终点”改掉：被激活的同色顶点不再互连，而是各自连向最近的、且与它
#    H/V/45 度对齐的异色像素。为了让规则稳定可见，生成器构造的是“同色顶点簇围绕一个异色
#    单像素目标”的局部对象；输出因此变成多条同色射线汇聚到异色点，并继续在 seed->target
#    的偏移下复制一层异色平行射线。
#    小例子：原题若输入有 3 个绿色顶点 + 一个粉色 seed 邻居，输出会闭合/折线连接绿色顶点，
#    再复制一层粉色平行边。现在同样的输入里，3 个绿色顶点都直接连向最近粉色点；粉色不再
#    是“只决定偏移的注脚”，而是主线的终点。
# b) Complexity calibration:
#    若完全放开“每个像素都找任意异色最近点”，不同对象会互相抢终点，且视觉上过乱。这里
#    收敛成：只有“至少 2 个像素且存在异色 8 邻居 seed”的颜色组会扩展；每组只有一个近邻
#    异色目标，其他颜色做单像素/非激活 distractor。这样 D8 的跨颜色终点在每个样例里都会
#    重复出现，但自由度仍受控。
# c) Hidden assumptions audited in main():
#    - 原代码 `len(pts_rc) < 2: continue`：依赖“要扩展的是多顶点同色组”。变体保留这一点，
#      单像素异色目标本身不扩展，否则会产生额外反向射线并使颜色覆盖顺序变得脆弱。
#    - 原代码按 `cycle` 建相邻同色边：这个前提已失效，所以只保留 `cycle` 用于稳定 seed 选取。
#    - 原代码 `seed_idxs` 只影响起点与 shadow 偏移；现在它还承担“这个颜色组是否被激活”的判定。
#      没有异色邻居的多像素组直接视为 distractor，以避免它们误连到远处其他对象。
#    - 原代码 `_clear_in_original(..., allow={col})` 默认路径内部只会遇到同色。变体里终点是异色，
#      因而仍然只允许内部穿过本色，终点异色仅出现在最后一个格子。
#    - 原代码第二阶段平行复制依赖 seed 的邻居偏移；变体仍保持这一点，所以生成器必须保证恰好
#      一个 seed 顶点与目标异色像素 8 邻接，且至少还有一条非共线主射线，使复制层可见。
# d) Placement legality:
#    每个局部对象都在自己的 patch 内先验证 `main(inp)`，再用 `random_free_location` 贴到全局。
#    patch 的 bbox 覆盖了主射线和可能出现的平行射线，因此放置时不会截断。全局仍保留 >=1 的
#    背景缓冲，避免不同对象争抢“最近异色终点”。
# e) Visual observability:
#    人眼可见的规律是“同色点簇朝一个异色单点汇聚”，而不是“同色点互连成壳”。异色单点总是
#    紧邻一个 seed 顶点，平行复制仍保留原题的 shadow 味道，因此从 2-3 个例子可以稳定归纳出
#    “终点换成最近异色点”这一变化。
# f) Cleanliness:
#    生成器只选互异方向的射线，避免大量共享主干；另外要求至少一条非共线射线能复制出异色平行
#    线，从而输出既有中心汇聚，也不过度重叠。
# g) Color inferability:
#    没有基于颜色 id 的排序规则；颜色只通过局部几何角色体现：多像素色是“发射源”，单像素异色
#    是“最近终点/复制色”。同一网格里会重复出现多个这样的双色对象，足以支撑归纳。
# 9) New solver assumptions guaranteed by generator:
#    - 每个可扩展颜色组恰有一个 seed 顶点与异色目标 8 邻接。
#    - 该异色目标对组内所有顶点都是最近且可对齐的异色像素。
#    - 不同对象之间距离足够大，避免跨对象最近点竞争。
#    - 至少一条主射线与 seed 偏移不共线，使第二阶段复制真正发生。

import math

import numpy as np

from arc_utils import bbox_coords, colors_present, draw_line, get_bg, get_bg as _bg_color, paste, random_free_location

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


def _nearest_different_aligned(
    r0: int,
    c0: int,
    src_col: int,
    non_bg_points: list[tuple[int, int, int]],
) -> tuple[int, int] | None:
    best_key: tuple[int, int, int] | None = None
    best_target: tuple[int, int] | None = None
    for r1, c1, col1 in non_bg_points:
        if col1 == src_col:
            continue
        if not _aligned_8(r0, c0, r1, c1):
            continue
        dist = max(abs(r1 - r0), abs(c1 - c0))
        if dist == 0:
            continue
        key = (dist, r1, c1)
        if best_key is None or key < best_key:
            best_key = key
            best_target = (r1, c1)
    return best_target


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = _bg_color(g)
    out = g.copy()
    non_bg_points = [tuple(map(int, (r, c, g[r, c]))) for r, c in np.argwhere(g != bg)]

    colors = sorted(set(int(v) for v in np.unique(g)) - {bg})
    for col in colors:
        pts_rc = [tuple(map(int, p)) for p in np.argwhere(g == col)]
        if len(pts_rc) < 2:
            continue

        cycle = _angle_cycle(pts_rc)

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

        if not seed_idxs:
            continue

        start = min(seed_idxs, key=lambda i: (cycle[i][0], cycle[i][1]))
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

        rays: list[tuple[tuple[int, int], tuple[int, int]]] = []
        for r0, c0 in cycle:
            target = _nearest_different_aligned(r0, c0, col, non_bg_points)
            if target is None:
                continue
            r1, c1 = target
            if not _clear_in_original(g, r0, c0, r1, c1, bg, allow={col}):
                continue
            rays.append(((r0, c0), (r1, c1)))

        for (r0, c0), (r1, c1) in rays:
            _draw_line_bg_only(out, r0, c0, r1, c1, col, bg)

        for (r0, c0), (r1, c1) in rays:
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


# --------------- generator ---------------

BG = 0
COLORS = [1, 2, 3, 4, 6, 7, 8]
CANVAS_H_RANGE = (15, 23)
CANVAS_W_JITTER = (-3, 4)
CANVAS_W_CLIP = (15, 22)

SHAPE_COUNT_RANGE = (2, 4)        # 2-3 shapes; each needs a paired target color.
DISTRACTOR_PROB = 0.45
PLACEMENT_PADDING = 1
PLACEMENT_PAD_CONNECTIVITY = 8

VERTEX_COUNT_RANGE = (2, 6)       # 2-5 vertices around one hub
RAY_DISTANCE_RANGE = (2, 7)       # extra vertices sit 2-6 cells from the hub
SHAPE_PATCH_RETRIES = 60
GENERATE_RETRIES = 300

DISTRACTOR_COUNT_RANGE = (1, 4)
DISTRACTOR_BBOX_RANGE = (2, 5)
DISTRACTOR_RETRIES = 50


def _aligned(dr: int, dc: int) -> bool:
    return (dr == 0 and dc != 0) or (dc == 0 and dr != 0) or (dr != 0 and abs(dr) == abs(dc))


def _gen_segment_cells(a: tuple[int, int], b: tuple[int, int]) -> list[tuple[int, int]]:
    drs, dcs = _step_dir(a[0], a[1], b[0], b[1])
    steps = max(abs(b[0] - a[0]), abs(b[1] - a[1]))
    return [(a[0] + drs * k, a[1] + dcs * k) for k in range(steps + 1)]


def _sample_canvas_size() -> tuple[int, int]:
    h = int(np.random.randint(*CANVAS_H_RANGE))
    w = int(np.clip(h + np.random.randint(*CANVAS_W_JITTER), *CANVAS_W_CLIP))
    return h, w


def _build_star_geometry() -> tuple[list[tuple[int, int]], tuple[int, int], tuple[int, int]] | None:
    vertex_count = int(np.random.randint(*VERTEX_COUNT_RANGE))
    offset = DIR8[int(np.random.randint(len(DIR8)))]
    hub = (0, 0)
    seed = (-offset[0], -offset[1])
    seed_dir = (seed[0], seed[1])

    extra_dirs = [d for d in DIR8 if d != seed_dir]
    np.random.shuffle(extra_dirs)

    non_collinear_needed = False
    vertices = [seed]
    for direction in extra_dirs:
        if len(vertices) >= vertex_count:
            break
        dist = int(np.random.randint(*RAY_DISTANCE_RANGE))
        candidate = (direction[0] * dist, direction[1] * dist)
        if candidate == seed:
            continue
        vertices.append(candidate)
        if seed_dir[0] * direction[1] - seed_dir[1] * direction[0] != 0:
            non_collinear_needed = True

    if len(vertices) != vertex_count or not non_collinear_needed:
        return None
    return vertices, hub, offset


def _shifted_bbox(vertices: list[tuple[int, int]], hub: tuple[int, int], offset: tuple[int, int]) -> tuple[int, int, int, int]:
    coords: set[tuple[int, int]] = set(vertices)
    coords.add(hub)
    for vertex in vertices:
        coords.update(_gen_segment_cells(vertex, hub))
        drs, dcs = _step_dir(vertex[0], vertex[1], hub[0], hub[1])
        if drs * offset[1] - dcs * offset[0] == 0:
            continue
        start = (vertex[0] + offset[0], vertex[1] + offset[1])
        end = (hub[0] + offset[0], hub[1] + offset[1])
        coords.update(_gen_segment_cells(start, end))
    arr = np.array(sorted(coords), dtype=int)
    return bbox_coords(arr)


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
    total = 2 * shape_count + int(has_distractor)
    if total > len(COLORS):
        return None
    palette = list(np.random.permutation(COLORS)[:total])
    shape_colors = [int(v) for v in palette[:shape_count]]
    target_colors = [int(v) for v in palette[shape_count:2 * shape_count]]
    distractor_color = int(palette[-1]) if has_distractor else None
    return shape_colors, target_colors, distractor_color


def _all_vertices_choose_hub(inp: np.ndarray, vertices: list[tuple[int, int]], hub: tuple[int, int], shape_color: int) -> bool:
    non_bg_points = [tuple(map(int, (r, c, inp[r, c]))) for r, c in np.argwhere(inp != BG)]
    for r, c in vertices:
        target = _nearest_different_aligned(r, c, shape_color, non_bg_points)
        if target != hub:
            return False
    return True


def _make_shape_patch(shape_color: int, target_color: int) -> tuple[np.ndarray, np.ndarray] | None:
    for _ in range(SHAPE_PATCH_RETRIES):
        built = _build_star_geometry()
        if built is None:
            continue
        vertices, hub, offset = built
        r0, c0, r1, c1 = _shifted_bbox(vertices, hub, offset)
        h = r1 - r0 + 1
        w = c1 - c0 + 1
        inp = np.full((h, w), BG, dtype=int)

        shifted_vertices = [(r - r0, c - c0) for r, c in vertices]
        shifted_hub = (hub[0] - r0, hub[1] - c0)

        for r, c in shifted_vertices:
            if inp[r, c] != BG:
                break
            inp[r, c] = shape_color
        else:
            if inp[shifted_hub] != BG:
                continue
            inp[shifted_hub] = target_color

            seed_adjacent = sum(
                1 for r, c in shifted_vertices
                if max(abs(r - shifted_hub[0]), abs(c - shifted_hub[1])) == 1
            )
            if seed_adjacent != 1:
                continue
            if not _all_vertices_choose_hub(inp, shifted_vertices, shifted_hub, shape_color):
                continue

            out = main(inp)
            if np.array_equal(inp, out):
                continue
            if int(np.count_nonzero(out == target_color)) <= 1:
                continue
            if len(colors_present(inp, exclude=[BG])) != 2:
                continue
            return inp, out
    return None


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


def _gen_attempt() -> np.ndarray | None:
    h, w = _sample_canvas_size()
    inp = np.full((h, w), BG, dtype=int)
    out = np.full((h, w), BG, dtype=int)

    has_distractor = bool(np.random.rand() < DISTRACTOR_PROB)
    shape_count = int(np.random.randint(*SHAPE_COUNT_RANGE))
    palette = _sample_palette(shape_count, has_distractor)
    if palette is None:
        return None
    shape_colors, target_colors, distractor_color = palette

    for shape_color, target_color in zip(shape_colors, target_colors):
        patch = _make_shape_patch(shape_color, target_color)
        if patch is None:
            return None
        if not _try_place(inp, out, *patch):
            return None

    if has_distractor and distractor_color is not None:
        if not _try_place(inp, out, *_make_distractor_patch(distractor_color)):
            return None

    if get_bg(inp) != BG or get_bg(out) != BG:
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
