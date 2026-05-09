# Variant of 35ab12c3
# Dimension: D10 — 形状种类与拓扑 (meta)
# Change: 六种形状类型（折线/三角形/矩形/菱形/五边形/六边形），通过加权随机选择，包含开放折线和闭合多边形两种拓扑... → 嵌套形状（形状内含形状）
#
# Design analysis
# - Concrete behavior:
#   原题把每种颜色看成“顶点集”，输出只连边；若某个顶点旁有异色像素，还会沿平移方向复制一份平行阴影。
#   这个变体保留“同色顶点 -> 连成 H/V/45° 闭合多边形”的主轴，但把输出改成区域型结果：
#   1) 若一种颜色只形成一个闭合多边形，则输出把该多边形整体填满。
#   2) 若一种颜色的顶点能拆成“外层多边形 + 内层更小的同色多边形”，则输出填充外层减去内层真内部，
#      保留两条边界，因此得到同色环带。
#   例子：输入若有 8 个红点，分别是外矩形四角和内矩形四角，原 solver 会把 8 点按单一环排序，
#   很容易把内外点串错；变体 solver 则先枚举可闭合子集，找出两个互不相交且一内一外的合法多边形，
#   再输出“外填充 - 内部留空”。
# - Complexity calibration:
#   若允许任意非凸或任意顶点数的嵌套，同色点集的分组会很歧义；因此生成器限制为凸、8 方向边、
#   顶点数 3-6 的多边形。嵌套层只允许两层，且每个样例至少出现一个嵌套组，使 D10 变化肉眼可见。
#   同时仍保留 2-4 个对象和可选干扰色，避免任务退化成单一大框填色。
# - Implicit-assumption audit against the original main():
#   1) 原 `main()` 的 `_angle_cycle` 默认“一种颜色只有一个环或一条折线”，这在嵌套同色组里失效；
#      现在只能把 `_angle_cycle` 当作“验证某个候选子集是否可闭合”的工具，而不能直接对整组使用。
#   2) 原 `seed_idxs` / `carried` 依赖“异色邻居表示阴影偏移”，变体不再有这个语义，相关逻辑必须删除；
#      否则内层同色多边形附近若恰好有别的对象，会被误读成阴影。
#   3) 原 `_draw_line_bg_only` 只描边、且默认背景优先即可；变体需要区域填充，因此必须显式构造边界 mask
#      并填洞，不能再靠逐边画线完成输出。
#   4) 原 `colors = sorted(...)` 仅用于遍历，和颜色 id 无关；变体继续保留这种无害遍历，不做基于颜色值的判断。
# - Placement legality:
#   生成器仍采用 output-first。本地 patch 先构造期望输出（实心多边形或环带），再反推输入只保留顶点。
#   每个 patch 都要求 `main(patch_input) == patch_output`；整图再用 `random_free_location(..., padding=1)` 放置，
#   保证不同对象之间至少隔 1 格背景，避免填充区域互相干扰。
# - Visual observability / neatness:
#   所有形状都用单色填充或单色环带，结构差异完全来自空间拓扑而不是颜色 id。嵌套对象内部一定留空，
#   在 2-4 个样例里很容易看到“同色四角/六角点 -> 变成一个有洞的框”。
# - Color inference:
#   不做颜色映射；每个对象颜色在输入输出保持不变。可选干扰色也保持不变，因此解题者只需归纳“哪些同色点
#   表示一个可闭合区域 / 两层嵌套区域”。
# - New solver assumptions and generator guarantees:
#   1) 一种颜色要么是干扰点，要么恰好对应 1 个闭合多边形，或 2 个严格嵌套的闭合多边形。
#      generator 按此构造，并禁止一色出现 3 层以上或多个不相交多边形。
#   2) 多边形是凸的，边只走 H/V/45°；generator 只从这些模板采样。
#   3) 内层多边形完全落在外层真内部，不接触外边界；generator 通过显式 inset 保证。
#   4) 对于同色嵌套组，所有顶点都必须由这 1 或 2 个多边形解释完；solver 仅接受“候选分解覆盖全部同色点”。

import itertools
import math

import numpy as np
from scipy import ndimage

from arc_utils import bbox_coords, colors_present, draw_line, get_bg, paste, random_free_location

DIR8 = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]


def _aligned_8(r0: int, c0: int, r1: int, c1: int) -> bool:
    dr = r1 - r0
    dc = c1 - c0
    return (dr == 0 and dc != 0) or (dc == 0 and dr != 0) or (dr != 0 and abs(dr) == abs(dc))


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


def _angle_cycle(points_rc: list[tuple[int, int]]) -> list[tuple[int, int]]:
    cr = sum(r for r, _ in points_rc) / len(points_rc)
    cc = sum(c for _, c in points_rc) / len(points_rc)

    def key(p: tuple[int, int]):
        r, c = p
        ang = math.atan2(r - cr, c - cc)
        rad = (r - cr) ** 2 + (c - cc) ** 2
        return (ang, rad, r, c)

    return sorted(points_rc, key=key)


def _polygon_edges(vertices: list[tuple[int, int]]) -> list[tuple[tuple[int, int], tuple[int, int]]]:
    return [(vertices[i], vertices[(i + 1) % len(vertices)]) for i in range(len(vertices))]


def _polygon_masks(vertices: list[tuple[int, int]], shape: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    r0, c0, r1, c1 = bbox_coords(np.array(vertices, dtype=int))
    h = r1 - r0 + 1
    w = c1 - c0 + 1

    outline_local = np.zeros((h + 2, w + 2), dtype=bool)
    shifted = [(r - r0 + 1, c - c0 + 1) for r, c in vertices]
    for a, b in _polygon_edges(shifted):
        draw_line(outline_local, a[0], a[1], end_r=b[0], end_c=b[1], color=True)

    filled_local = ndimage.binary_fill_holes(outline_local)

    outline = np.zeros(shape, dtype=bool)
    filled = np.zeros(shape, dtype=bool)
    outline[r0 : r1 + 1, c0 : c1 + 1] = outline_local[1:-1, 1:-1]
    filled[r0 : r1 + 1, c0 : c1 + 1] = filled_local[1:-1, 1:-1]
    return outline, filled


def _analyze_polygon(g: np.ndarray, col: int, pts_rc: list[tuple[int, int]], bg: int):
    if len(pts_rc) < 3 or len(pts_rc) > 6:
        return None

    cycle = _angle_cycle(pts_rc)
    for (r0, c0), (r1, c1) in _polygon_edges(cycle):
        if not _aligned_8(r0, c0, r1, c1):
            return None
        if not _clear_in_original(g, r0, c0, r1, c1, bg, allow={col}):
            return None

    outline, filled = _polygon_masks(cycle, g.shape)
    if not outline.any() or not filled.any():
        return None
    vals = g[filled]
    if np.any((vals != bg) & (vals != col)):
        return None
    if int(filled.sum()) <= len(cycle):
        return None

    return {
        "cycle": cycle,
        "outline": outline,
        "filled": filled,
        "interior": filled & ~outline,
        "area": int(filled.sum()),
    }


def _nested_mask(outer: dict, inner: dict) -> np.ndarray | None:
    if np.any(inner["filled"] & ~outer["interior"]):
        return None
    if np.any(outer["outline"] & inner["filled"]):
        return None
    if not all(bool(outer["interior"][r, c]) for r, c in inner["cycle"]):
        return None
    return outer["filled"] & ~inner["interior"]


def _solve_color_mask(g: np.ndarray, col: int, bg: int) -> np.ndarray | None:
    pts_rc = [tuple(map(int, p)) for p in np.argwhere(g == col)]
    n = len(pts_rc)
    if n < 3:
        return None

    candidates: dict[frozenset[int], dict] = {}
    for size in range(3, min(6, n) + 1):
        for idxs in itertools.combinations(range(n), size):
            info = _analyze_polygon(g, col, [pts_rc[i] for i in idxs], bg)
            if info is not None:
                candidates[frozenset(idxs)] = info

    if not candidates:
        return None

    all_idxs = frozenset(range(n))
    best_ring = None
    best_score = None
    keys = list(candidates.keys())
    for i, a in enumerate(keys):
        for b in keys[i + 1 :]:
            if a & b:
                continue
            if a | b != all_idxs:
                continue
            pa = candidates[a]
            pb = candidates[b]
            outer, inner = (pa, pb) if pa["area"] >= pb["area"] else (pb, pa)
            ring = _nested_mask(outer, inner)
            if ring is None:
                continue
            score = (int(ring.sum()), outer["area"], -inner["area"])
            if best_score is None or score > best_score:
                best_score = score
                best_ring = ring

    if best_ring is not None:
        return best_ring

    full = candidates.get(all_idxs)
    if full is not None:
        return full["filled"]
    return None


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    out = g.copy()

    for col in sorted(set(int(v) for v in np.unique(g)) - {bg}):
        mask = _solve_color_mask(g, col, bg)
        if mask is not None:
            out[mask] = int(col)

    return out


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

TRIANGLE_SIDE_RANGE = (3, 7)
RECT_H_RANGE = (3, 8)
RECT_W_RANGE = (3, 8)
DIAMOND_ARM_RANGE = (3, 6)
PENTAGON_ROOF_RANGE = (2, 4)
PENTAGON_WALL_RANGE = (3, 6)
HEXAGON_SLANT_RANGE = (2, 4)
HEXAGON_WALL_RANGE = (2, 5)

NESTED_RECT_H_RANGE = (6, 11)
NESTED_RECT_W_RANGE = (6, 11)
NESTED_DIAMOND_ARM_RANGE = (4, 7)

SIMPLE_SHAPE_WEIGHTS = np.array([0.18, 0.28, 0.24, 0.14, 0.16], dtype=float)
NESTED_SHAPE_WEIGHTS = np.array([0.56, 0.44], dtype=float)

DISTRACTOR_COUNT_RANGE = (1, 4)
DISTRACTOR_BBOX_RANGE = (2, 5)
DISTRACTOR_RETRIES = 50

SHAPE_PLACEMENT_RETRIES = 40
GENERATE_RETRIES = 300


def _aligned(dr: int, dc: int) -> bool:
    return (dr == 0 and dc != 0) or (dc == 0 and dr != 0) or (dr != 0 and abs(dr) == abs(dc))


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


def _transform_nested_coords(
    outer: list[tuple[int, int]],
    inner: list[tuple[int, int]],
) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    coords = _transform_coords(outer + inner)
    return coords[: len(outer)], coords[len(outer) :]


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


def _build_nested_rectangle() -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    h = int(np.random.randint(*NESTED_RECT_H_RANGE))
    w = int(np.random.randint(*NESTED_RECT_W_RANGE))
    max_inset = min(h, w) // 2 - 1
    inset = int(np.random.randint(2, max_inset + 1))
    outer = [(0, 0), (0, w), (h, w), (h, 0)]
    inner = [(inset, inset), (inset, w - inset), (h - inset, w - inset), (h - inset, inset)]
    return _transform_nested_coords(outer, inner)


def _build_nested_diamond() -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    arm = int(np.random.randint(*NESTED_DIAMOND_ARM_RANGE))
    inner_arm = int(np.random.randint(2, arm - 1))
    outer = [(0, arm), (arm, 2 * arm), (2 * arm, arm), (arm, 0)]
    inner = [(arm - inner_arm, arm), (arm, arm + inner_arm), (arm + inner_arm, arm), (arm, arm - inner_arm)]
    return _transform_nested_coords(outer, inner)


_SIMPLE_BUILDERS = [_build_triangle, _build_rectangle, _build_diamond, _build_pentagon, _build_hexagon]
_NESTED_BUILDERS = [_build_nested_rectangle, _build_nested_diamond]


def _shift_to_origin(coords: list[tuple[int, int]]) -> list[tuple[int, int]]:
    arr = np.array(coords, dtype=int)
    arr -= arr.min(axis=0)
    return [tuple(map(int, p)) for p in arr]


def _make_simple_vertices() -> list[tuple[int, int]]:
    weights = SIMPLE_SHAPE_WEIGHTS / SIMPLE_SHAPE_WEIGHTS.sum()
    idx = int(np.random.choice(len(_SIMPLE_BUILDERS), p=weights))
    return _SIMPLE_BUILDERS[idx]()


def _make_nested_vertices() -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    weights = NESTED_SHAPE_WEIGHTS / NESTED_SHAPE_WEIGHTS.sum()
    idx = int(np.random.choice(len(_NESTED_BUILDERS), p=weights))
    return _NESTED_BUILDERS[idx]()


def _shape_mask_from_vertices(vertices: list[tuple[int, int]], inner_vertices: list[tuple[int, int]] | None = None) -> np.ndarray:
    all_vertices = vertices if inner_vertices is None else vertices + inner_vertices
    r0, c0, r1, c1 = bbox_coords(np.array(all_vertices, dtype=int))
    h = r1 - r0 + 1
    w = c1 - c0 + 1

    outer_shifted = [(r - r0, c - c0) for r, c in vertices]
    _outline, outer_fill = _polygon_masks(outer_shifted, (h, w))
    if inner_vertices is None:
        return outer_fill

    inner_shifted = [(r - r0, c - c0) for r, c in inner_vertices]
    inner_outline, inner_fill = _polygon_masks(inner_shifted, (h, w))
    return outer_fill & ~(inner_fill & ~inner_outline)


def _make_shape_patch(shape_color: int, nested: bool) -> tuple[np.ndarray, np.ndarray] | None:
    if nested:
        outer, inner = _make_nested_vertices()
        all_coords = _shift_to_origin(outer + inner)
        outer = all_coords[: len(outer)]
        inner = all_coords[len(outer) :]
        vertices = outer + inner
        r0, c0, r1, c1 = bbox_coords(np.array(vertices, dtype=int))
        h = r1 - r0 + 1
        w = c1 - c0 + 1
        inp = np.full((h, w), BG, dtype=int)
        out = np.full((h, w), BG, dtype=int)
        mask = _shape_mask_from_vertices(outer, inner)
        out[mask] = int(shape_color)
        for r, c in outer + inner:
            inp[r, c] = int(shape_color)
    else:
        outer = _make_simple_vertices()
        outer = _shift_to_origin(outer)
        r0, c0, r1, c1 = bbox_coords(np.array(outer, dtype=int))
        h = r1 - r0 + 1
        w = c1 - c0 + 1
        inp = np.full((h, w), BG, dtype=int)
        out = np.full((h, w), BG, dtype=int)
        mask = _shape_mask_from_vertices(outer, None)
        out[mask] = int(shape_color)
        for r, c in outer:
            inp[r, c] = int(shape_color)

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


def _sample_canvas_size() -> tuple[int, int]:
    h = int(np.random.randint(*CANVAS_H_RANGE))
    w = int(np.clip(h + np.random.randint(*CANVAS_W_JITTER), *CANVAS_W_CLIP))
    return h, w


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

    shape_count = int(np.random.randint(*SHAPE_COUNT_RANGE))
    has_distractor = bool(np.random.rand() < DISTRACTOR_PROB)
    total_colors = shape_count + int(has_distractor)
    palette = list(np.random.permutation(COLORS)[:total_colors])
    shape_colors = palette[:shape_count]
    distractor_color = int(palette[-1]) if has_distractor else None

    nested_count = int(np.random.randint(1, shape_count + 1))
    nested_targets = set(np.random.choice(shape_count, size=nested_count, replace=False).tolist())

    for idx, shape_color in enumerate(shape_colors):
        placed = False
        for _ in range(SHAPE_PLACEMENT_RETRIES):
            patch = _make_shape_patch(int(shape_color), nested=(idx in nested_targets))
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
