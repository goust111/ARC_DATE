# Variant of 35ab12c3
# Dimension: D19 — 线段绘制优先级 (multi_step)
# Change: 两阶段绘制：Phase1先画所有主形状边（只涂背景格子），Phase2再画所有阴影平行线（只涂背景格子）。
#         变为：主线和阴影线混合后按线段长度从长到短统一绘制，仍然只占用背景格，因此长线会在交叉点优先“抢占”像素。

import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path("/home/lixinyu/snap/new/solvers")
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from arc_utils import bbox_coords, colors_present, get_bg, neighbors8, paste, random_free_location
from arc_utils import get_bg as _bg_color

# Design analysis for the D19 variant.
#
# a) 具体化与行为追踪
#    原题在每个形状内先画所有主边，再画所有阴影平行边，而且每一步都只填背景格。
#    所以一旦主边和阴影边相交，主边永远赢，因为它更早占住交叉格。
#    这里改成“所有候选边统一排序后依次画”，固定采用“长边先画、短边后画”。
#    小例子：矩形顶点 (0,0),(0,7),(3,7),(3,0)，阴影锚点在 (1,1)。
#    顶边及其阴影长 8，竖边及其阴影长 4。原题在交叉点 (1,7) 保留主竖边颜色；
#    本变体中长阴影横边先画，后来的短主竖边不能覆盖，于是 (1,7) 变成阴影色。
#    变化只发生在“候选边已经确定之后的渲染顺序”，输入识别、可画性判定、阴影偏移都保持原题思路。
#
# b) 复杂度校准
#    如果继续允许“无阴影”或“所有边等长”的样本，D19 可能完全不可见。
#    因此生成器强制至少放 1 个带阴影的非正方形矩形，并从 8 邻域中挑选真正能制造冲突的偏移，
#    让长边和短边在主线/阴影线之间发生交叉，
#    同时保留 1-3 个其他形状/干扰点，避免任务被过度简化成单对象。
#
# c) 隐式假设审查
#    - out = g.copy(): 仍然保留输入中的顶点和阴影锚点。D19 没要求覆盖这些输入像素，因此保持。
#    - _clear_in_original(...): 仍只参考原输入判断线段是否合法；排序变化不应改变“哪些线能画”。
#    - 原 main 里“先主后阴影”是硬编码优先级。这里把主/阴影都收集成 jobs，再按长度统一排序。
#    - 旧版 _make_shape_patch 直接先画完整主形状、再整体平移阴影，这依赖旧优先级；
#      变体里改成先构造输入 patch，再调用新的 main 得到输出，避免生成器偷偷沿用旧规则。
#
# d) 放置合法性
#    patch_out 直接取新 solver 的输出，放置时仍使用 random_free_location + 1 格 padding，
#    从而保证不同组在最终输出中不相交、不贴边，也不会出现跨组优先级竞争。
#
# e) 视觉可观察性
#    排序只看线段长度，不使用颜色 id。为了让规则从少量样例可见，至少有一个矩形满足：
#    1. 带阴影；2. 长短边明显不同；3. 新旧规则输出不同（生成器显式检测并拒绝退化样本）。
#
# f) 可视化整洁度
#    只要求一个“优先级矩形”承担 D19 现象，其余对象延续原题的稀疏布局，避免全图大量覆盖而难以读图。
#
# g) 颜色变化可推断性
#    颜色不参与排序，也没有额外颜色映射。阴影色仍由输入中的邻接单像素直接给出，人类可以从结构关系推断。
#
# 9) 新 solver 的输入假设与生成器保证
#    - 假设：某个带阴影形状的阴影锚点 8 邻接且只邻接到一个顶点。生成器沿用并检查此条件。
#    - 假设：要看到 D19，至少要有一组“长度不同且发生交叉”的主/阴影边。生成器固定首个形状为非正方形矩形，
#      并要求 priority_delta_cells >= 2。

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


def _segment_len(r0: int, c0: int, r1: int, c1: int) -> int:
    return max(abs(r1 - r0), abs(c1 - c0)) + 1


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


def _collect_line_jobs(
    g: np.ndarray, col: int, bg: int
) -> tuple[list[tuple[tuple[int, int], tuple[int, int], int, int]], list[tuple[tuple[int, int], tuple[int, int], int, int]]]:
    h, w = g.shape
    pts_rc = [tuple(map(int, p)) for p in np.argwhere(g == col)]
    if len(pts_rc) < 2:
        return [], []

    cycle = _angle_cycle(pts_rc)
    if len(cycle) < 2:
        return [], []

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

    main_jobs = [
        ((r0, c0), (r1, c1), col, _segment_len(r0, c0, r1, c1))
        for (r0, c0), (r1, c1) in drawable_edges
    ]

    shadow_jobs: list[tuple[tuple[int, int], tuple[int, int], int, int]] = []
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
            shadow_jobs.append((((rr0, cc0)), ((rr1, cc1)), ncol, _segment_len(rr0, cc0, rr1, cc1)))
    return main_jobs, shadow_jobs


def _priority_job_key(job: tuple[tuple[int, int], tuple[int, int], int, int]) -> tuple[int, int, int, int, int]:
    (r0, c0), (r1, c1), _col, seg_len = job
    a, b = sorted(((r0, c0), (r1, c1)))
    return (-seg_len, a[0], a[1], b[0], b[1])


def _render_grid(g: np.ndarray, *, priority_mode: bool) -> np.ndarray:
    g = np.asarray(g, dtype=int)
    bg = _bg_color(g)
    out = g.copy()

    colors = sorted(set(int(v) for v in np.unique(g)) - {bg})
    for col in colors:
        main_jobs, shadow_jobs = _collect_line_jobs(g, col, bg)
        if priority_mode:
            jobs = sorted(main_jobs + shadow_jobs, key=_priority_job_key)
        else:
            jobs = main_jobs + shadow_jobs
        for (r0, c0), (r1, c1), draw_col, _seg_len in jobs:
            _draw_line_bg_only(out, r0, c0, r1, c1, draw_col, bg)

    return out


def main(grid: np.ndarray) -> np.ndarray:
    return _render_grid(np.asarray(grid, dtype=int), priority_mode=True)


def _legacy_main(grid: np.ndarray) -> np.ndarray:
    return _render_grid(np.asarray(grid, dtype=int), priority_mode=False)


def _priority_delta_cells(grid: np.ndarray) -> int:
    return int(np.count_nonzero(main(grid) != _legacy_main(grid)))


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
PRIORITY_RECT_LONG_RANGE = (6, 10)
PRIORITY_RECT_SHORT_RANGE = (2, 5)
MIN_PRIORITY_DELTA_CELLS = 2

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


def _build_priority_rectangle() -> list[tuple[int, int]]:
    long_side = int(np.random.randint(*PRIORITY_RECT_LONG_RANGE))
    short_side = int(np.random.randint(*PRIORITY_RECT_SHORT_RANGE))
    if np.random.rand() < 0.5:
        h, w = short_side, long_side
    else:
        h, w = long_side, short_side
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


def _make_vertices(*, priority_shape: bool) -> list[tuple[int, int]]:
    if priority_shape:
        return _build_priority_rectangle()
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


def _choose_priority_shadow(
    vertices: list[tuple[int, int]],
    shape_cells: set[tuple[int, int]],
) -> list[tuple[tuple[int, int], tuple[int, int]]]:
    anchor = min(vertices)
    candidates: list[tuple[tuple[int, int], tuple[int, int]]] = []
    for dr, dc in DIR8:
        pos = (anchor[0] + dr, anchor[1] + dc)
        if pos in shape_cells:
            continue
        if sum(1 for vr, vc in vertices if max(abs(pos[0] - vr), abs(pos[1] - vc)) == 1) != 1:
            continue
        candidates.append((anchor, (dr, dc)))
    np.random.shuffle(candidates)
    return candidates


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


def _make_shape_patch(shape_color: int, shadow_color: int | None, *, priority_shape: bool) -> tuple[np.ndarray, np.ndarray] | None:
    vertices = _make_vertices(priority_shape=priority_shape)
    cells = _shape_cells(vertices)

    if shadow_color is None:
        shadow_candidates = [None]
    elif priority_shape:
        shadow_candidates = _choose_priority_shadow(vertices, cells)
        if not shadow_candidates:
            return None
    else:
        shadow = _choose_shadow(vertices, cells)
        if shadow is None:
            return None
        shadow_candidates = [shadow]

    for shadow in shadow_candidates:
        r0, c0, r1, c1 = _patch_bbox(cells, vertices, shadow)
        h, w = r1 - r0 + 1, c1 - c0 + 1
        inp = np.full((h, w), BG, dtype=int)

        shifted_vertices = [(r - r0, c - c0) for r, c in vertices]
        for r, c in shifted_vertices:
            inp[r, c] = shape_color

        if shadow is not None:
            (ar, ac), offset = shadow
            anchor = (ar - r0, ac - c0)
            sa = (anchor[0] + offset[0], anchor[1] + offset[1])
            if not (0 <= sa[0] < h and 0 <= sa[1] < w):
                continue
            if inp[sa] != BG:
                continue
            inp[sa] = int(shadow_color)
            nb_hits = sum(
                1
                for rr, cc in neighbors8(anchor[0], anchor[1])
                if 0 <= rr < h and 0 <= cc < w and inp[rr, cc] == shadow_color
            )
            if nb_hits != 1:
                continue

        out = main(inp)
        if priority_shape and _priority_delta_cells(inp) < MIN_PRIORITY_DELTA_CELLS:
            continue
        return inp, out
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
    if max_shadows < 1:
        return None
    shadow_count = int(np.random.randint(1, max_shadows + 1))
    total = shape_count + shadow_count + int(has_distractor)
    palette = list(np.random.permutation(COLORS)[:total])
    shape_colors = palette[:shape_count]
    shadow_colors = palette[shape_count : shape_count + shadow_count]
    distractor_color = int(palette[-1]) if has_distractor else None
    return shape_colors, shadow_colors, distractor_color


def _assign_shadows(shape_count: int, shadow_colors: list[int], priority_idx: int) -> dict[int, int]:
    if not shadow_colors:
        return {}
    mapping = {int(priority_idx): int(shadow_colors[0])}
    if len(shadow_colors) == 1:
        return mapping
    pool = [idx for idx in range(shape_count) if idx != priority_idx]
    targets = np.random.choice(pool, size=len(shadow_colors) - 1, replace=False).tolist()
    for idx, col in zip(sorted(targets), shadow_colors[1:]):
        mapping[int(idx)] = int(col)
    return mapping


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
    priority_idx = 0

    palette = _sample_palette(shape_count, has_distractor)
    if palette is None:
        return None
    shape_colors, shadow_colors, distractor_color = palette
    shadow_map = _assign_shadows(shape_count, shadow_colors, priority_idx)

    total_colors = shape_count + len(shadow_colors) + int(has_distractor)

    for idx, shape_color in enumerate(shape_colors):
        placed = False
        for _ in range(SHAPE_PLACEMENT_RETRIES):
            patch = _make_shape_patch(
                int(shape_color),
                shadow_map.get(idx),
                priority_shape=(idx == priority_idx),
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
    if _priority_delta_cells(inp) < MIN_PRIORITY_DELTA_CELLS:
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
