# Variant of 35ab12c3
# Dimension: D15 — 段间清晰路径约束 (path)
# Change: clear_in_original——两个顶点之间的连线路径上不能有非背景非同色像素阻挡（在原始输入中检查），否则该边不画 → 允许路径穿过其他颜色像素

import math
import numpy as np

from arc_utils import get_bg as _bg_color
from arc_utils import bbox_coords, colors_present, draw_line, get_bg, neighbors8, paste, random_free_location


# Design analysis for the D15 variant
# ==================================
# 1. Concrete behavior / trace
#    原题在 main() 里先枚举每个颜色的顶点，再对每条候选边调用
#    `_clear_in_original(...)`。只要两端点之间的原始输入路径上出现了别的颜色，
#    这整条边就被丢弃，完全不画。
#
#    变体要求改成：边仍然允许“穿过”别的颜色组，但不能覆盖穿过点。这里原 solver
#    已经有 `_draw_line_bg_only(...)`，它只会在背景格着色。因此最小且正确的改法是：
#    去掉 `_clear_in_original(...)` 过滤，保留背景优先绘制。这样线段仍沿完整几何路径
#    尝试落笔，但碰到原输入里已有的异色像素时会留下缺口。
#
#    小例子：
#      输入中 red 顶点在 (2,2) 与 (2,8)，blue 单像素挡在 (2,5)。
#      原题：red 这条横线因路径不清空而整段不画。
#      本变体：输出仍保留 blue 在 (2,5)，并把 red 画到 (2,3)(2,4)(2,6)(2,7)，
#      于是出现一个被 blue 打断的断续横线。
#
# 2. Complexity calibration
#    如果只改 solver，不改 generator，那么大多数样本仍会保持对象彼此间隔，视觉上退化回
#    原题。因此 generator 必须稳定地产生“段内阻挡像素”。我保留原有形状/阴影机制，
#    只把原来的可选 distractor 改成强制存在的 blocker group：同色 1-3 个像素，
#    直接放在别的形状边的内部。这样：
#    - 变体差异肉眼可见：边会被少量异色像素切断。
#    - 复杂度仍受控：blocker 仍是普通 distractor 颜色组，不引入新的几何类型。
#
# 3. Hidden-assumption audit for main()
#    - `_aligned_8(...)`：仍成立，边的几何只允许 H/V/45°。
#    - `_angle_cycle(...)`：仍成立，顶点排序规则不依赖路径清晰性。
#    - seed/start vertex 选择：仍成立，但 generator 必须避免把 blocker 放到任一顶点
#      的 8 邻域里，否则 blocker 会被误读为 shadow anchor。
#    - 原来的 `_clear_in_original(...)` 调用：这是唯一依赖“边必须全清空”的代码。
#      变体中该前提不成立，必须删除这两个过滤点：
#        a. 主边 drawable_edges 构建处
#        b. carried shadow 的平行边构建处
#    - `_draw_line_bg_only(...)`：原题里只是“避免覆盖”；在本变体里它变成实现断续缺口
#      的关键，因此必须保留，不能改成普通 draw_line。
#
# 4. Placement legality
#    generator 继续先放互不重叠的形状 patch，保持原题的大部分空间整洁度。之后再把
#    blocker 像素压到某些边的内部单元格上，并同步写入 input/output。
#    为避免非法歧义，blocker 位置必须满足：
#    - 在某条边的内部，且离两端点至少 2 格（不与顶点 8 邻接）
#    - 其 8 邻域中没有其他输入前景像素（避免看起来像 shadow anchor）
#    - 同组 blocker 像素两两不 H/V/diag 对齐（保证 solver 不会把它们当作新 shape）
#
# 5. Visual observability / cleanliness
#    每个样本都强制至少有一个 blocker，因此 2-4 个样例足以让人看到“线会穿过去但不覆盖”
#    的规律。blocker 数量仍限制为 1-3 个，小而稀疏，避免整张图过乱。
#
# 6. Color inference
#    颜色不承载隐藏顺序信息；blocker 只是一个普通颜色组，输入输出都保留该颜色本身。
#    变化来自几何位置（它恰好躺在别人的连线段上），而不是 color id。
#
# 7. Post-change solver assumptions guaranteed by generate_input()
#    - main() 仍会把“每个 >=2 像素颜色组”都当成候选 shape；因此 blocker 组必须两两
#      非 H/V/diag 对齐，generator 已显式保证这一点。
#    - main() 会把起始顶点周围的异色邻居当成 carried shadow；因此 blocker 不能进入任何
#      顶点的 8 邻域，generator 只在远离端点的段内部下放 blocker。
#    - main() 的新规则只有在“某条本应可画的边内部真的存在异色像素”时才和原题不同；
#      generator 强制每个样本至少布置一个这类 blocker，并用 oracle 校验 `main(inp)==out`。


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


def main(grid: np.ndarray) -> np.ndarray:
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
                _draw_line_bg_only(out, rr0, cc0, rr1, cc1, ncol, bg)

    return out


# --------------- generator ---------------

BG = 0
COLORS = [1, 2, 3, 4, 6, 7, 8]
CANVAS_H_RANGE = (15, 23)
CANVAS_W_JITTER = (-3, 4)
CANVAS_W_CLIP = (15, 22)

SHAPE_COUNT_RANGE = (2, 5)
MAX_SHADOWS_CAP = 3
BLOCKER_GROUP_REQUIRED = True
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

BLOCKER_COUNT_RANGE = (1, 4)
BLOCKER_PICK_RETRIES = 80

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
    src = (out == shape_color)
    shifted = np.zeros_like(src)
    r_src = slice(max(0, -dr), h - max(0, dr))
    c_src = slice(max(0, -dc), w - max(0, dc))
    r_dst = slice(max(0, dr), h - max(0, -dr))
    c_dst = slice(max(0, dc), w - max(0, -dc))
    shifted[r_dst, c_dst] = src[r_src, c_src]
    mask = shifted & (out == BG)
    out[mask] = shadow_color


def _make_shape_patch(shape_color: int, shadow_color: int | None):
    vertices = _make_vertices()
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
    shifted_edges = _shape_edges(shifted_vertices)

    for a, b in shifted_edges:
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
    return inp, out, {"vertices": shifted_vertices, "edges": shifted_edges}


def _sample_palette(shape_count: int, has_blocker: bool) -> tuple[list[int], list[int], int | None] | None:
    max_shadows = min(MAX_SHADOWS_CAP, shape_count, len(COLORS) - shape_count - int(has_blocker))
    if max_shadows < 0:
        return None
    shadow_count = int(np.random.randint(max_shadows + 1))
    total = shape_count + shadow_count + int(has_blocker)
    palette = list(np.random.permutation(COLORS)[:total])
    shape_colors = palette[:shape_count]
    shadow_colors = palette[shape_count:shape_count + shadow_count]
    blocker_color = int(palette[-1]) if has_blocker else None
    return shape_colors, shadow_colors, blocker_color


def _assign_shadows(shape_count: int, shadow_colors: list[int]) -> dict[int, int]:
    if not shadow_colors:
        return {}
    targets = np.random.choice(shape_count, size=len(shadow_colors), replace=False).tolist()
    return {int(idx): int(col) for idx, col in zip(sorted(targets), shadow_colors)}


def _try_place(inp: np.ndarray, out: np.ndarray, patch_in: np.ndarray, patch_out: np.ndarray) -> tuple[int, int] | None:
    try:
        loc = random_free_location(
            out,
            patch_out,
            transparent=BG,
            padding=PLACEMENT_PADDING,
            padding_connectivity=PLACEMENT_PAD_CONNECTIVITY,
        )
    except ValueError:
        return None
    paste(out, patch_out, loc[0], loc[1], transparent=BG)
    paste(inp, patch_in, loc[0], loc[1], transparent=BG)
    return int(loc[0]), int(loc[1])


def _sample_canvas_size() -> tuple[int, int]:
    H = int(np.random.randint(*CANVAS_H_RANGE))
    W = int(np.clip(H + np.random.randint(*CANVAS_W_JITTER), *CANVAS_W_CLIP))
    return H, W


def _segment_far_interior(a: tuple[int, int], b: tuple[int, int]) -> list[tuple[int, int]]:
    cells = _gen_segment_cells(a, b)
    if len(cells) < 5:
        return []
    return cells[2:-2]


def _collect_blocker_candidates(
    shape_records: list[dict[str, object]],
    inp: np.ndarray,
) -> list[tuple[int, int]]:
    all_vertices = [tuple(v) for rec in shape_records for v in rec["vertices"]]  # type: ignore[index]
    h, w = inp.shape
    seen: set[tuple[int, int]] = set()
    candidates: list[tuple[int, int]] = []

    for rec in shape_records:
        for a, b in rec["edges"]:  # type: ignore[index]
            for r, c in _segment_far_interior(a, b):
                if not (0 <= r < h and 0 <= c < w):
                    continue
                if int(inp[r, c]) != BG:
                    continue
                if any(max(abs(r - vr), abs(c - vc)) <= 1 for vr, vc in all_vertices):
                    continue
                crowded = False
                for rr in range(max(0, r - 1), min(h, r + 2)):
                    for cc in range(max(0, c - 1), min(w, c + 2)):
                        if (rr, cc) == (r, c):
                            continue
                        if int(inp[rr, cc]) != BG:
                            crowded = True
                            break
                    if crowded:
                        break
                if crowded:
                    continue
                cell = (int(r), int(c))
                if cell not in seen:
                    seen.add(cell)
                    candidates.append(cell)
    return candidates


def _choose_blocker_coords(candidates: list[tuple[int, int]]) -> list[tuple[int, int]] | None:
    if not candidates:
        return None
    max_target = min(BLOCKER_COUNT_RANGE[1] - 1, len(candidates))
    min_target = min(BLOCKER_COUNT_RANGE[0], max_target)
    for target in range(max_target, min_target - 1, -1):
        for _ in range(BLOCKER_PICK_RETRIES):
            order = np.random.permutation(len(candidates))
            picked: list[tuple[int, int]] = []
            for idx in order:
                cand = candidates[int(idx)]
                if any(_aligned(cand[0] - rr, cand[1] - cc) for rr, cc in picked):
                    continue
                picked.append(cand)
                if len(picked) == target:
                    return picked
    return [candidates[int(np.random.randint(len(candidates)))]]


def _paint_blockers(inp: np.ndarray, out: np.ndarray, blocker_color: int, coords: list[tuple[int, int]]) -> None:
    for r, c in coords:
        inp[r, c] = blocker_color
        out[r, c] = blocker_color


def _gen_attempt() -> np.ndarray | None:
    H, W = _sample_canvas_size()
    out = np.full((H, W), BG, dtype=int)
    inp = np.full((H, W), BG, dtype=int)

    shape_count = int(np.random.randint(*SHAPE_COUNT_RANGE))
    has_blocker = BLOCKER_GROUP_REQUIRED

    palette = _sample_palette(shape_count, has_blocker)
    if palette is None:
        return None
    shape_colors, shadow_colors, blocker_color = palette
    shadow_map = _assign_shadows(shape_count, shadow_colors)
    total_colors = shape_count + len(shadow_colors) + int(has_blocker)

    shape_records: list[dict[str, object]] = []

    for idx, shape_color in enumerate(shape_colors):
        placed = False
        for _ in range(SHAPE_PLACEMENT_RETRIES):
            patch = _make_shape_patch(int(shape_color), shadow_map.get(idx))
            if patch is None:
                continue
            patch_in, patch_out, meta = patch
            loc = _try_place(inp, out, patch_in, patch_out)
            if loc is None:
                continue
            global_vertices = [(r + loc[0], c + loc[1]) for r, c in meta["vertices"]]
            shape_records.append(
                {
                    "color": int(shape_color),
                    "vertices": global_vertices,
                    "edges": _shape_edges(global_vertices),
                }
            )
            placed = True
            break
        if not placed:
            return None

    if has_blocker:
        if blocker_color is None:
            return None
        blocker_candidates = _collect_blocker_candidates(shape_records, inp)
        blocker_coords = _choose_blocker_coords(blocker_candidates)
        if blocker_coords is None or len(blocker_coords) == 0:
            return None
        _paint_blockers(inp, out, int(blocker_color), blocker_coords)

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
