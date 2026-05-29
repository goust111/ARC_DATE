# Variant of 35ab12c3
# Dimension: D8 — 终止条件/端点规则 (dst)
# Change: 无独立终点——路径在同色像素对之间直接连接，终点就是配对的另一个像素
#   -> 距离编码停止：方向仍由“下一个同色顶点”给出，但射线只走固定步数；
#      该步数等于该颜色在输入中的像素总数。
#
# Design analysis
# ===============
# variant_spec:
#   core_change:
#     original: |
#       对每个形状色，solver 先按角度排序得到顶点循环，然后把每条边直接画到下一个同色像素。
#       终点是显式存在的第二个顶点，因此“边长”来自两个顶点之间的几何距离。
#     variant: |
#       仍然使用“当前顶点 -> 下一个同色顶点”来确定方向，但不再把后者当终点。
#       设该颜色在输入中的像素数为 n，则每条射线只从起点沿该方向前进 n 步。
#       输出保留所有原始顶点，所以当真实目标顶点更远时，会看到“射线停在半途 + 远处仍有同色点”。
#   concrete_trace:
#     input_shape: |
#       例：同色顶点 4 个，构成一个大矩形；左上顶点旁边有 1 个阴影锚点。
#       原题：左上 -> 右上 会画完整上边；右上 -> 右下 画完整右边；形成完整轮廓。
#       变体：同色计数 n=4，因此每条边只画 4 步。若矩形边长为 7，则每条边会提前 3 格停止，
#       右上等顶点仍以孤立输入点形式保留。阴影只沿这些“截断射线”平移复制。
#     difference: |
#       旧逻辑依赖“另一个同色点就是终点”；新逻辑只把它当方向提示，真正终点由可见计数 n 编码。
#   complexity_calibration:
#     choices:
#       - 把可编码距离限制为 2-4（即仅生成 2/3/4 顶点形状），避免距离档位太多。
#       - 同时把真实边长约束为严格大于 n，使“停在半途”在绝大多数样本中肉眼可见。
#       - 保留原题的阴影机制、颜色唯一性、可选干扰物，避免把任务改成完全不同的问题。
#   implicit_assumption_audit:
#     affected_lines:
#       - _clear_in_original(...) 原来默认最后一个格子是显式端点，可跳过；现在停止点是背景格，不能跳过。
#       - drawable_edges 原来存 (start,end)；现在 end 需重算为 start + n * dir。
#       - 阴影阶段原来平移完整边；现在必须平移截断后的射线，否则会错误延伸到旧终点。
#       - len(cycle)==2 时原题只有一条边；新题仍保持这一点，否则会把 2 点对象变成双向射线。
#     new_assumptions:
#       - generator 必须保证每条“方向边”的真实目标距离 > n，否则会退化成原题或穿过另一个顶点。
#       - generator 必须保证截断射线和其平移阴影在输入中途不撞上非背景像素，否则 solver 会合法地跳过该射线。
#   placement_legality:
#     generator_strategy: |
#       仍采用局部 patch + 全局随机放置。每个 patch 的 bbox 覆盖：
#       输入顶点、主射线、可选阴影锚点、以及阴影射线。全局放置仍使用 random_free_location，
#       并保留 1 格缓冲，避免不同对象在输出里互相阻塞。
#   visual_observability:
#     rationale: |
#       人类可直接数出某颜色输入像素个数 n，并观察该颜色所有输出射线都只延伸 n 步；
#       同时远处的同色顶点仍保留，从而能排除“连接到另一点”为规则。
#       颜色 id 不参与排序或映射，只有几何关系与可见计数参与推理。
#   cleanliness:
#     controls:
#       - 只生成边长足够长的 2/3/4 顶点形状，避免大量重叠或过短射线。
#       - 阴影偏移仍限制为非共线 8 邻域偏移，保持平行复制清晰可见。
#       - 干扰物仍要求任意两点不水平/垂直/45 度对齐，防止被误识别成形状。

import math

import numpy as np

from arc_utils import get_bg as _bg_color
from arc_utils import bbox_coords, colors_present, get_bg, neighbors8, paste, random_free_location

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


def _ray_cells(r0: int, c0: int, dr: int, dc: int, steps: int):
    for k in range(steps + 1):
        yield r0 + dr * k, c0 + dc * k


def _clear_ray_in_original(g: np.ndarray, r0: int, c0: int, dr: int, dc: int, steps: int, bg: int) -> bool:
    for k in range(1, steps + 1):
        rr = r0 + dr * k
        cc = c0 + dc * k
        if int(g[rr, cc]) != bg:
            return False
    return True


def _draw_ray_bg_only(out: np.ndarray, r0: int, c0: int, dr: int, dc: int, steps: int, col: int, bg: int) -> None:
    for rr, cc in _ray_cells(r0, c0, dr, dc, steps):
        if int(out[rr, cc]) == bg:
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


def _ordered_cycle(points_rc: list[tuple[int, int]], start_pt: tuple[int, int]) -> list[tuple[int, int]]:
    cycle = _angle_cycle(points_rc)
    start = cycle.index(start_pt)
    return cycle[start:] + cycle[:start]


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

        ray_steps = len(pts_rc)
        drawable_rays: list[tuple[int, int, int, int]] = []
        for (r0, c0), (r1, c1) in edges:
            if not _aligned_8(r0, c0, r1, c1):
                continue
            drs, dcs = _step_dir(r0, c0, r1, c1)
            rr1 = r0 + drs * ray_steps
            cc1 = c0 + dcs * ray_steps
            if not (0 <= rr1 < h and 0 <= cc1 < w):
                continue
            if not _clear_ray_in_original(g, r0, c0, drs, dcs, ray_steps, bg):
                continue
            drawable_rays.append((r0, c0, drs, dcs))

        for r0, c0, drs, dcs in drawable_rays:
            _draw_ray_bg_only(out, r0, c0, drs, dcs, ray_steps, col, bg)

        for r0, c0, drs, dcs in drawable_rays:
            for drn, dcn, ncol in carried:
                if drs * dcn - dcs * drn == 0:
                    continue
                rr0 = r0 + drn
                cc0 = c0 + dcn
                rr1 = rr0 + drs * ray_steps
                cc1 = cc0 + dcs * ray_steps
                if not (0 <= rr0 < h and 0 <= cc0 < w and 0 <= rr1 < h and 0 <= cc1 < w):
                    continue
                if not _clear_ray_in_original(g, rr0, cc0, drs, dcs, ray_steps, bg):
                    continue
                _draw_ray_bg_only(out, rr0, cc0, drs, dcs, ray_steps, ncol, bg)

    return out


# --------------- generator ---------------

BG = 0
COLORS = [1, 2, 3, 4, 6, 7, 8]
CANVAS_H_RANGE = (15, 23)
CANVAS_W_JITTER = (-3, 4)
CANVAS_W_CLIP = (15, 22)

SHAPE_COUNT_RANGE = (2, 5)
MAX_SHADOWS_CAP = 3
DISTRACTOR_PROB = 0.45
PLACEMENT_PADDING = 1
PLACEMENT_PAD_CONNECTIVITY = 8

SEG_LEN_RANGE = (4, 10)
TRI_SIDE_RANGE = (5, 10)
RECT_H_RANGE = (5, 10)
RECT_W_RANGE = (5, 10)
DIAMOND_ARM_RANGE = (5, 8)
SHAPE_WEIGHTS = np.array([0.18, 0.30, 0.28, 0.24], dtype=float)

DISTRACTOR_COUNT_RANGE = (1, 4)
DISTRACTOR_BBOX_RANGE = (2, 5)
DISTRACTOR_RETRIES = 50

SHAPE_PATCH_RETRIES = 60
SHAPE_PLACEMENT_RETRIES = 50
GENERATE_RETRIES = 400


def _aligned(dr: int, dc: int) -> bool:
    return (dr == 0 and dc != 0) or (dc == 0 and dr != 0) or (dr != 0 and abs(dr) == abs(dc))


def _step(a: tuple[int, int], b: tuple[int, int]) -> tuple[int, int]:
    return _step_dir(a[0], a[1], b[0], b[1])


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


def _build_segment() -> list[tuple[int, int]]:
    length = int(np.random.randint(*SEG_LEN_RANGE))
    return _transform_coords([(0, 0), (0, length)])


def _build_triangle() -> list[tuple[int, int]]:
    side = int(np.random.randint(*TRI_SIDE_RANGE))
    return _transform_coords([(0, 0), (0, side), (side, side)])


def _build_rectangle() -> list[tuple[int, int]]:
    h = int(np.random.randint(*RECT_H_RANGE))
    w = int(np.random.randint(*RECT_W_RANGE))
    return _transform_coords([(0, 0), (0, w), (h, w), (h, 0)])


def _build_diamond() -> list[tuple[int, int]]:
    arm = int(np.random.randint(*DIAMOND_ARM_RANGE))
    return _transform_coords([(0, arm), (arm, 2 * arm), (2 * arm, arm), (arm, 0)])


_SHAPE_BUILDERS = [_build_segment, _build_triangle, _build_rectangle, _build_diamond]


def _make_vertices() -> list[tuple[int, int]]:
    weights = SHAPE_WEIGHTS / SHAPE_WEIGHTS.sum()
    idx = int(np.random.choice(len(_SHAPE_BUILDERS), p=weights))
    return _SHAPE_BUILDERS[idx]()


def _shape_edges(vertices: list[tuple[int, int]]) -> list[tuple[tuple[int, int], tuple[int, int]]]:
    start = min(vertices)
    cycle = _ordered_cycle(vertices, start)
    if len(cycle) == 2:
        return [(cycle[0], cycle[1])]
    return [(cycle[i], cycle[(i + 1) % len(cycle)]) for i in range(len(cycle))]


def _ray_plan(vertices: list[tuple[int, int]]) -> list[dict[str, object]] | None:
    rays: list[dict[str, object]] = []
    ray_steps = len(vertices)
    vertex_set = set(vertices)
    used_cells: set[tuple[int, int]] = set()
    for a, b in _shape_edges(vertices):
        if not _aligned_8(a[0], a[1], b[0], b[1]):
            return None
        dist = max(abs(b[0] - a[0]), abs(b[1] - a[1]))
        if dist <= ray_steps:
            return None
        drs, dcs = _step(a, b)
        cells = list(_ray_cells(a[0], a[1], drs, dcs, ray_steps))
        if any(cell in vertex_set and cell != a for cell in cells[1:]):
            return None
        if any(cell in used_cells for cell in cells[1:]):
            return None
        used_cells.update(cells)
        rays.append({"start": a, "dir": (drs, dcs), "cells": cells})
    return rays


def _choose_shadow(
    vertices: list[tuple[int, int]],
    main_cells: set[tuple[int, int]],
    ray_dirs: list[tuple[int, int]],
) -> tuple[tuple[int, int], tuple[int, int]] | None:
    anchor = min(vertices)
    candidates: list[tuple[int, int]] = []
    for dr, dc in DIR8:
        if any(ds_r * dc - ds_c * dr == 0 for ds_r, ds_c in ray_dirs):
            continue
        pos = (anchor[0] + dr, anchor[1] + dc)
        if pos in main_cells or pos in set(vertices):
            continue
        if sum(1 for vr, vc in vertices if max(abs(pos[0] - vr), abs(pos[1] - vc)) == 1) != 1:
            continue
        candidates.append((dr, dc))
    if not candidates:
        return None
    return anchor, candidates[int(np.random.randint(len(candidates)))]


def _patch_bbox(
    vertices: list[tuple[int, int]],
    main_cells: set[tuple[int, int]],
    shadow: tuple[tuple[int, int], tuple[int, int]] | None,
) -> tuple[int, int, int, int]:
    coords = list(vertices) + list(main_cells)
    if shadow is not None:
        (ar, ac), (dr, dc) = shadow
        coords.append((ar + dr, ac + dc))
        coords.extend((r + dr, c + dc) for r, c in main_cells)
    r0, c0, r1, c1 = bbox_coords(np.array(coords, dtype=int))
    return int(r0), int(c0), int(r1), int(c1)


def _paint_shifted_cells_bg_only(out: np.ndarray, cells: set[tuple[int, int]], color: int, offset: tuple[int, int]) -> None:
    dr, dc = offset
    for r, c in cells:
        rr = r + dr
        cc = c + dc
        if int(out[rr, cc]) == BG:
            out[rr, cc] = color


def _make_shape_patch(shape_color: int, shadow_color: int | None) -> tuple[np.ndarray, np.ndarray] | None:
    for _ in range(SHAPE_PATCH_RETRIES):
        vertices = _make_vertices()
        rays = _ray_plan(vertices)
        if rays is None:
            continue

        main_cells = {tuple(cell) for ray in rays for cell in ray["cells"]}  # type: ignore[index]
        ray_dirs = [ray["dir"] for ray in rays]  # type: ignore[index]

        shadow = None
        if shadow_color is not None:
            shadow = _choose_shadow(vertices, main_cells, ray_dirs)
            if shadow is None:
                continue

        r0, c0, r1, c1 = _patch_bbox(vertices, main_cells, shadow)
        h, w = r1 - r0 + 1, c1 - c0 + 1
        inp = np.full((h, w), BG, dtype=int)

        shifted_vertices = [(r - r0, c - c0) for r, c in vertices]
        for r, c in shifted_vertices:
            inp[r, c] = shape_color

        if shadow is not None:
            (ar, ac), offset = shadow
            sa = (ar + offset[0] - r0, ac + offset[1] - c0)
            if not (0 <= sa[0] < h and 0 <= sa[1] < w):
                continue
            if int(inp[sa]) != BG:
                continue
            inp[sa] = int(shadow_color)

        out = inp.copy()
        shifted_main_cells = {(r - r0, c - c0) for r, c in main_cells}
        for r, c in shifted_main_cells:
            if int(out[r, c]) == BG:
                out[r, c] = shape_color

        if shadow is not None:
            _paint_shifted_cells_bg_only(out, shifted_main_cells, int(shadow_color), shadow[1])
            anchor = (shadow[0][0] - r0, shadow[0][1] - c0)
            nb_hits = sum(
                1
                for rr, cc in neighbors8(anchor[0], anchor[1])
                if 0 <= rr < h and 0 <= cc < w and int(inp[rr, cc]) == int(shadow_color)
            )
            if nb_hits != 1:
                continue

        if not np.array_equal(main(inp), out):
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
    if max_shadows < 0:
        return None
    shadow_count = int(np.random.randint(max_shadows + 1))
    total = shape_count + shadow_count + int(has_distractor)
    palette = list(np.random.permutation(COLORS)[:total])
    shape_colors = palette[:shape_count]
    shadow_colors = palette[shape_count : shape_count + shadow_count]
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
