# Variant of 35ab12c3
# Dimension: D20 — 形状-阴影颜色关系 (meta)
# Change: 每个阴影使用独立的唯一颜色，阴影颜色与形状颜色无系统性映射关系... → 阴影颜色编码偏移方向
#
# Design analysis
# a) Concrete behavior / trace:
#    原题里，某个形状顶点旁边的异色像素既是“提示像素”，它所在的几何位置也直接就是阴影偏移。
#    例如顶点 v 左上有一个 7，那么 solver 复制出的阴影也沿 (-1,-1) 平移；7 只是被原样涂过去。
#    这个变体里，阴影方向由颜色固定编码：
#      red(2)->up, blue(1)->right, orange(7)->down, teal(8)->left。
#    输入里仍保留一个与锚点顶点 8 邻接的提示像素，但它的摆放位置不再等于真实阴影偏移。
#    solver 必须读颜色再查方向表，然后沿该方向复制整条形状。输出中提示像素本身会被擦掉；
#    只保留真正方向上的阴影线。这样同色提示可以跨多个形状重复出现，而不会再被当作“同色顶点组”。
#    小例子：若形状顶点左下有一个 red(2) 提示像素，原逻辑会把阴影向左下复制；新逻辑则忽略左下这个位置，
#    只根据 red=up，把阴影整体向上复制。两者在同一输入上会得到不同输出。
# b) Complexity calibration:
#    方向只保留 4 个正交方向，而不是 8 个方向。这样颜色类目少、跨样例重复足够，人在 2-4 个样例里能归纳；
#    同时提示像素位置与真实方向脱钩，避免退化回原题。
# c) Implicit assumptions audit for main():
#    - 原 `colors = all non-bg colors` 假设每个非背景色要么是形状色要么是一次性阴影色；现在不成立。
#      因此 main() 必须跳过固定的方向提示色集合，否则同色提示会被误连成假形状。
#    - 原 `carried.append((drn, dcn, v))` 假设邻居几何偏移就是阴影偏移；现在不成立。
#      新逻辑改为只记录提示颜色，再由 `COLOR_TO_OFFSET[v]` 恢复真实平移方向。
#    - 原 `out = g.copy()` 默认输入提示像素应保留到输出；现在不成立。
#      新逻辑先把所有方向提示色从输出底稿里清掉，再只画真正的阴影。
#    - 原 `_clear_in_original(... allow=set())` 默认阴影路径上不会有其他提示像素挡路。
#      生成器因此必须保证提示像素既不落在形状路径上，也不落在真实阴影路径上。
# d) Placement legality:
#    生成器仍按 patch 放置，并使用 `random_free_location(... padding=1)` 保证不同对象输出之间至少隔 1 格背景。
#    单个 patch 的 bbox 同时覆盖形状、真实阴影和提示像素，避免越界。
# e) Visual observability:
#    每个形状都带一个阴影提示；提示色固定来自 4 个方向色。并且提示像素的位置被强制与真实方向不同，
#    所以观察者不能再靠“提示像素在哪边”解题，必须从颜色和输出中的复制方向建立映射。
# f) Visual cleanliness:
#    每个形状只给 1 个提示像素，且提示像素在输出中消失，避免多个标记与阴影叠在一起。
#    同时关闭 distractor，并把形状数限制到 2-3，保证网格不杂乱。
# g) Color inferability:
#    固定 4 个方向色：1/2/7/8；形状色固定来自另一组 3/4/6，二者不混用。
#    方向色类别少且跨样例重复；提示像素颜色是唯一决定方向的线索，结构性足够强。

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

_ARC_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_ARC_UTILS_DIR) not in sys.path:
    sys.path.append(str(_ARC_UTILS_DIR))

from arc_utils import get_bg as _bg_color
from arc_utils import bbox_coords, colors_present, draw_line, get_bg, paste, random_free_location

DIR8 = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
CARDINAL_DIRS = [(-1, 0), (0, 1), (1, 0), (0, -1)]

# Fixed direction-color encoding for the variant.
OFFSET_TO_COLOR = {
    (-1, 0): 2,  # red -> up
    (0, 1): 1,   # blue -> right
    (1, 0): 7,   # orange -> down
    (0, -1): 8,  # teal -> left
}
COLOR_TO_OFFSET = {col: offset for offset, col in OFFSET_TO_COLOR.items()}
SHADOW_COLORS = sorted(COLOR_TO_OFFSET)


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


def _is_shadow_marker_color(col: int) -> bool:
    return int(col) in COLOR_TO_OFFSET


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = _bg_color(g)

    out = g.copy()
    for shadow_color in SHADOW_COLORS:
        out[out == shadow_color] = bg

    colors = sorted(set(int(v) for v in np.unique(g)) - {bg} - set(SHADOW_COLORS))
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
                if 0 <= rr < h and 0 <= cc < w and _is_shadow_marker_color(int(g[rr, cc])):
                    seed_idxs.append(i)
                    break

        if seed_idxs:
            start = min(seed_idxs, key=lambda i: (cycle[i][0], cycle[i][1]))
        else:
            start = min(range(len(cycle)), key=lambda i: (cycle[i][0], cycle[i][1]))

        cycle = cycle[start:] + cycle[:start]

        carried: list[int] = []
        sr0, sc0 = cycle[0]
        for drn, dcn in DIR8:
            rr = sr0 + drn
            cc = sc0 + dcn
            if 0 <= rr < h and 0 <= cc < w:
                v = int(g[rr, cc])
                if _is_shadow_marker_color(v):
                    carried.append(v)

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

        for (r0, c0), (r1, c1) in drawable_edges:
            drs, dcs = _step_dir(r0, c0, r1, c1)
            for ncol in carried:
                drn, dcn = COLOR_TO_OFFSET[ncol]
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
SHAPE_COLORS = [3, 4, 6]
CANVAS_H_RANGE = (15, 23)
CANVAS_W_JITTER = (-3, 4)
CANVAS_W_CLIP = (15, 22)

SHAPE_COUNT_RANGE = (2, 4)         # 2-3 shapes
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
LOCAL_GEOMETRY_RETRIES = 60
GENERATE_RETRIES = 300
MIN_SHADOW_CELLS = 2


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


def _patch_bbox(
    shape_cells: set[tuple[int, int]],
    vertices: list[tuple[int, int]],
    cue: tuple[int, int],
    actual_offset: tuple[int, int],
) -> tuple[int, int, int, int]:
    coords = list(shape_cells) + vertices + [cue]
    coords.extend((r + actual_offset[0], c + actual_offset[1]) for r, c in shape_cells)
    r0, c0, r1, c1 = bbox_coords(np.array(coords, dtype=int))
    return int(r0), int(c0), int(r1), int(c1)


def _shadow_direction_visible(out: np.ndarray, shape_color: int, shadow_color: int, actual_offset: tuple[int, int]) -> bool:
    shape_coords = np.argwhere(out == shape_color)
    shadow_coords = np.argwhere(out == shadow_color)
    if len(shape_coords) == 0 or len(shadow_coords) < MIN_SHADOW_CELLS:
        return False
    sr0, sc0, sr1, sc1 = bbox_coords(shape_coords)
    tr0, tc0, tr1, tc1 = bbox_coords(shadow_coords)
    dr, dc = actual_offset
    if dr == -1:
        return tr0 < sr0
    if dr == 1:
        return tr1 > sr1
    if dc == -1:
        return tc0 < sc0
    return tc1 > sc1


def _make_shape_patch(shape_color: int, shadow_color: int) -> tuple[np.ndarray, np.ndarray] | None:
    actual_offset = COLOR_TO_OFFSET[int(shadow_color)]
    for _ in range(LOCAL_GEOMETRY_RETRIES):
        vertices = _make_vertices()
        shape_cells = _shape_cells(vertices)
        anchor = min(vertices)
        shifted_shape = {(r + actual_offset[0], c + actual_offset[1]) for r, c in shape_cells}

        marker_dirs = list(DIR8)
        np.random.shuffle(marker_dirs)
        for marker_dir in marker_dirs:
            if marker_dir == actual_offset:
                continue
            cue = (anchor[0] + marker_dir[0], anchor[1] + marker_dir[1])
            if cue in shape_cells or cue in shifted_shape:
                continue
            if sum(1 for vr, vc in vertices if max(abs(cue[0] - vr), abs(cue[1] - vc)) == 1) != 1:
                continue

            r0, c0, r1, c1 = _patch_bbox(shape_cells, vertices, cue, actual_offset)
            h, w = r1 - r0 + 1, c1 - c0 + 1
            inp = np.full((h, w), BG, dtype=int)

            shifted_vertices = [(r - r0, c - c0) for r, c in vertices]
            cue_rel = (cue[0] - r0, cue[1] - c0)
            for r, c in shifted_vertices:
                inp[r, c] = int(shape_color)
            if inp[cue_rel] != BG:
                continue
            inp[cue_rel] = int(shadow_color)

            out = main(inp)
            if int(out[cue_rel]) != BG:
                continue
            if not _shadow_direction_visible(out, int(shape_color), int(shadow_color), actual_offset):
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


def _sample_canvas_size() -> tuple[int, int]:
    H = int(np.random.randint(*CANVAS_H_RANGE))
    W = int(np.clip(H + np.random.randint(*CANVAS_W_JITTER), *CANVAS_W_CLIP))
    return H, W


def _sample_shadow_colors(shape_count: int) -> list[int]:
    while True:
        shadow_colors = [int(np.random.choice(SHADOW_COLORS)) for _ in range(shape_count)]
        if len(set(shadow_colors)) >= min(2, shape_count):
            return shadow_colors


def _gen_attempt() -> np.ndarray | None:
    H, W = _sample_canvas_size()
    out = np.full((H, W), BG, dtype=int)
    inp = np.full((H, W), BG, dtype=int)

    shape_count = int(np.random.randint(*SHAPE_COUNT_RANGE))
    shape_colors = list(np.random.permutation(SHAPE_COLORS)[:shape_count])
    shadow_colors = _sample_shadow_colors(shape_count)

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

    total_colors = shape_count + len(set(shadow_colors))
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
