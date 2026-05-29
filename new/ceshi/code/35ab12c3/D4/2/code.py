# Variant of 35ab12c3
# Dimension: D4 — 轨迹涂色模式 (trajectory)
# Change: continuous_trail——沿路径的每个背景格子都涂上形状颜色，形成连续实线... → 渐变长度轨迹

import math
import numpy as np

from arc_utils import get_bg as _bg_color
from arc_utils import bbox_coords, colors_present, get_bg, paste, random_free_location

# -----------------------------------------------------------------------------
# Design analysis for the D4 variant
#
# 1. Concrete behavior
#    原题对每条合法边都会把起点和终点之间的所有背景格子全部填满。这个变体只改
#    “轨迹怎么涂”这一点：边的终点仍然存在，因为输入里本来就给了顶点；变化的是
#    端点之间的背景轨迹只画“从起点出发的一段前缀”，而且同一形状里按边序号递减。
#
#    固定比例序列:
#      第1条边: 4/4  (完整轨迹)
#      第2条边: 3/4  (只画前 3/4 的内部轨迹)
#      第3条边: 2/4
#      第4条边及以后: 1/4
#
#    这里“长度”只统计两顶点之间的内部背景格。终点顶点本身仍会出现在输出中：
#    对主形状来说终点像素已经在输入里；对 shadow 来说 solver 会显式补出平移后的
#    顶点。这样输出会呈现“前缀实线 + 尾端孤立顶点”的渐淡效果，而不是简单缩短
#    整条线段。
#
#    具体小例子（水平边，长度 6 步）:
#      输入:   A . . . . . A
#      原题:   A A A A A A A
#      变体第2边:
#             内部背景格有 5 个，只保留 floor(5 * 3/4) = 3 个
#             输出: A A A A . . A
#
# 2. Complexity calibration / observability
#    若保留原始短边（长度 2-3），“3/4”和“1/2”经常会量化成相同像素数，肉眼看不出
#    变化。因此生成器把形状限制为 3-4 条边的三角形/矩形/菱形，并要求每条边长度
#    至少为 4。这样一个样例里会反复出现“满 / 较长 / 更短 / 最短”的轨迹层级。
#
#    同时，每个形状都强制带一个 shadow-anchor。原因是渐变规则依赖“边顺序”，而边
#    顺序又依赖起点；有 anchor 的顶点就是清晰可见的起点。这样解题者不需要猜测
#    从哪个顶点开始数边。
#
# 3. Implicit-assumption audit against original main()
#    - _aligned_8 / _clear_in_original:
#      仍然成立。边是否可连的几何约束没有改变。
#    - cycle/start 选择:
#      原题只用它决定画线顺序；现在它还决定“哪条边最完整、哪条最短”，因此生成器
#      必须稳定地暴露唯一的起点。为此我们保证每个 shape 恰好有一个 shadow 像素且
#      它只贴邻一个顶点。
#    - Phase 1 全量 draw:
#      原题无条件把整条边画满；在变体里这会出错，因为会抹掉应保留的空洞。这里改为
#      只画按边序号计算出的前缀内部格，再保留终点。
#    - Phase 2 shadow copy:
#      原题复制的是完整平行线；变体中 shadow 必须复制同一条“部分轨迹”，否则主线
#      渐淡而 shadow 不渐淡，会破坏单一规则。
#
# 4. Placement legality
#    生成器继续使用 output-first placement：先构造局部输入 patch，再直接调用本
#    solver 得到局部输出 patch。random_free_location 依据 patch_out 放置，因此不会
#    与别的对象输出重叠。patch bbox 仍按“完整形状 + 完整 shadow 平移包围盒”分配，
#    所以即使某条边只画前缀，也一定不会越界。
#
# 5. Visual cleanliness / color inference
#    这个变体不引入新的颜色映射，颜色仍只标识不同 shape / shadow / distractor。
#    规则全部靠可见结构推断：同色顶点、唯一 shadow-anchor、以及同一形状内按顺序
#    变短的轨迹。生成器降低对象数到 2-3 个、并保持至少 1 格缓冲，避免多个渐变轨迹
#    互相缠绕。
#
# 6. New assumptions introduced by this variant
#    solver 现在假设“起点是可识别的”以及“边足够长，3/4 与 1/2 可见”。生成器通过：
#    - 每个 shape 都带唯一 shadow-anchor；
#    - 只采样边长 >= 4 的三角形/矩形/菱形；
#    - 每个 shape 至少有 3 条边；
#    来保证这些前提。
# -----------------------------------------------------------------------------

DIR8 = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
TRAIL_NUMERATORS = (4, 3, 2, 1)
TRAIL_DENOMINATOR = 4


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


def _edge_ratio_numerator(edge_idx: int) -> int:
    return TRAIL_NUMERATORS[min(edge_idx, len(TRAIL_NUMERATORS) - 1)]


def _partial_internal_count(total_steps: int, edge_idx: int) -> int:
    internal = max(0, total_steps - 1)
    num = _edge_ratio_numerator(edge_idx)
    return int(math.floor(internal * num / TRAIL_DENOMINATOR))


def _draw_partial_edge_bg_only(
    out: np.ndarray,
    r0: int,
    c0: int,
    r1: int,
    c1: int,
    col: int,
    bg: int,
    edge_idx: int,
) -> None:
    cells = list(_segment_cells(r0, c0, r1, c1))
    total_steps = len(cells) - 1
    keep_internal = _partial_internal_count(total_steps, edge_idx)
    for rr, cc in cells[1 : 1 + keep_internal]:
        if out[rr, cc] == bg:
            out[rr, cc] = col
    if total_steps >= 1:
        rr, cc = cells[-1]
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


def _ordered_drawable_edges(
    g: np.ndarray,
    col: int,
    bg: int,
) -> tuple[list[tuple[tuple[int, int], tuple[int, int]]], list[tuple[int, int, int]]]:
    h, w = g.shape
    pts_rc = [tuple(map(int, p)) for p in np.argwhere(g == col)]
    if len(pts_rc) < 2:
        return [], []

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
    return drawable_edges, carried


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = _bg_color(g)
    out = g.copy()

    colors = sorted(set(int(v) for v in np.unique(g)) - {bg})
    for col in colors:
        drawable_edges, carried = _ordered_drawable_edges(g, col, bg)
        if not drawable_edges:
            continue

        for edge_idx, ((r0, c0), (r1, c1)) in enumerate(drawable_edges):
            _draw_partial_edge_bg_only(out, r0, c0, r1, c1, col, bg, edge_idx)

        for edge_idx, ((r0, c0), (r1, c1)) in enumerate(drawable_edges):
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
                _draw_partial_edge_bg_only(out, rr0, cc0, rr1, cc1, ncol, bg, edge_idx)

    return out


def _main_full_reference(grid: np.ndarray) -> np.ndarray:
    """Original continuous-trail behavior, kept only for generator validation."""
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = _bg_color(g)
    out = g.copy()

    colors = sorted(set(int(v) for v in np.unique(g)) - {bg})
    for col in colors:
        drawable_edges, carried = _ordered_drawable_edges(g, col, bg)
        if not drawable_edges:
            continue

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


# --------------- generator ---------------

BG = 0
COLORS = [1, 2, 3, 4, 6, 7, 8]
CANVAS_H_RANGE = (15, 23)
CANVAS_W_JITTER = (-3, 4)
CANVAS_W_CLIP = (15, 22)

SHAPE_COUNT_RANGE = (2, 4)  # 2-3 shapes
DISTRACTOR_PROB = 0.35
PLACEMENT_PADDING = 1
PLACEMENT_PAD_CONNECTIVITY = 8

TRIANGLE_SIDE_RANGE = (4, 8)
RECT_H_RANGE = (4, 8)
RECT_W_RANGE = (4, 8)
DIAMOND_ARM_RANGE = (4, 7)

SHAPE_WEIGHTS = np.array([0.34, 0.33, 0.33], dtype=float)

DISTRACTOR_COUNT_RANGE = (1, 4)
DISTRACTOR_BBOX_RANGE = (2, 5)
DISTRACTOR_RETRIES = 50

SHAPE_PLACEMENT_RETRIES = 50
GENERATE_RETRIES = 400


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


_SHAPE_BUILDERS = [_build_triangle, _build_rectangle, _build_diamond]


def _make_vertices() -> list[tuple[int, int]]:
    weights = SHAPE_WEIGHTS / SHAPE_WEIGHTS.sum()
    idx = int(np.random.choice(len(_SHAPE_BUILDERS), p=weights))
    return _SHAPE_BUILDERS[idx]()


def _shape_edges(vertices: list[tuple[int, int]]) -> list[tuple[tuple[int, int], tuple[int, int]]]:
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
    shadow: tuple[tuple[int, int], tuple[int, int]] | None,
) -> tuple[int, int, int, int]:
    coords = list(shape_cells) + vertices
    if shadow is not None:
        (_, _), (dr, dc) = shadow
        coords.extend((r + dr, c + dc) for r, c in shape_cells)
        coords.extend((r + dr, c + dc) for r, c in vertices)
    r0, c0, r1, c1 = bbox_coords(np.array(coords, dtype=int))
    return int(r0), int(c0), int(r1), int(c1)


def _edge_lengths(vertices: list[tuple[int, int]]) -> list[int]:
    return [max(abs(b[0] - a[0]), abs(b[1] - a[1])) for a, b in _shape_edges(vertices)]


def _has_visible_gradient(vertices: list[tuple[int, int]]) -> bool:
    lengths = _edge_lengths(vertices)
    if len(lengths) < 3:
        return False
    previous = None
    for edge_idx, steps in enumerate(lengths):
        kept = _partial_internal_count(steps, edge_idx)
        if previous is not None and kept > previous:
            return False
        previous = kept
    return any(_partial_internal_count(steps, idx) < max(0, steps - 1) for idx, steps in enumerate(lengths[1:], start=1))


def _make_shape_patch(shape_color: int, shadow_color: int) -> tuple[np.ndarray, np.ndarray] | None:
    vertices = _make_vertices()
    if not _has_visible_gradient(vertices):
        return None

    cells = _shape_cells(vertices)
    shadow = _choose_shadow(vertices, cells)
    if shadow is None:
        return None

    r0, c0, r1, c1 = _patch_bbox(cells, vertices, shadow)
    h, w = r1 - r0 + 1, c1 - c0 + 1
    inp = np.full((h, w), BG, dtype=int)

    shifted_vertices = [(r - r0, c - c0) for r, c in vertices]
    for r, c in shifted_vertices:
        inp[r, c] = shape_color

    (ar, ac), offset = shadow
    anchor = (ar - r0, ac - c0)
    sa = (anchor[0] + offset[0], anchor[1] + offset[1])
    if not (0 <= sa[0] < h and 0 <= sa[1] < w) or inp[sa] != BG:
        return None
    inp[sa] = shadow_color

    out = main(inp)
    if np.array_equal(out, _main_full_reference(inp)):
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


def _sample_palette(shape_count: int, has_distractor: bool) -> tuple[list[int], list[int], int | None]:
    total = shape_count * 2 + int(has_distractor)
    palette = list(np.random.permutation(COLORS)[:total])
    shape_colors = palette[:shape_count]
    shadow_colors = palette[shape_count : 2 * shape_count]
    distractor_color = int(palette[-1]) if has_distractor else None
    return shape_colors, shadow_colors, distractor_color


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
    shape_colors, shadow_colors, distractor_color = _sample_palette(shape_count, has_distractor)
    total_colors = shape_count * 2 + int(has_distractor)

    for shape_color, shadow_color in zip(shape_colors, shadow_colors):
        placed = False
        for _ in range(SHAPE_PLACEMENT_RETRIES):
            patch = _make_shape_patch(int(shape_color), int(shadow_color))
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
    if np.array_equal(main(inp), _main_full_reference(inp)):
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
