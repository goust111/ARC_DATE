# Variant of 35ab12c3
# Dimension: D7 — 源对象——形状顶点来源 (src)
# Change: 同色像素组作为多边形/折线顶点，每组2-7个同色像素，形状类型包括折线、三角形、矩形、菱形、五边形、六边形... → 边缘像素作为源
#
# Design analysis
# ===============
# variant_spec:
#   core_change:
#     - 原题把某颜色的所有像素都当作候选顶点；本变体只把该颜色位于外边框上的像素
#       (首行/末行/首列/末列) 当作真实顶点。
#     - 同色的内部像素允许存在，但它们是“诱饵像素”，不会参与连线；它们在输出中保留原样。
#   concrete_visual:
#     - 每个 shape 的真实顶点固定在边框，主要分布在四个角附近。
#     - 输出仍然连接这些顶点形成折线/多边形，并把起始顶点的异色邻居沿平行路径复制。
#     - 与原题不同的是，输入里会反复出现“同色内部点不触发连线”的例子，迫使求解者发现
#       “边框像素才是源顶点”这一规则。
#   behavior_trace:
#     - 原逻辑：对某颜色收集全部同色像素，按极角排序后依次尝试连边。
#     - 新逻辑：先过滤到边框同色像素，再做同样的极角排序和连边；内部同色点只作为保留杂点。
#     - 小例子：若颜色 2 在左上角有边框点 {(0,4),(4,0)}，并在内部多一个 (4,2)，
#       原逻辑会把 3 个点一起排序并可能生成额外边；新逻辑只连接前两个边框点，
#       (4,2) 在输出中仍是孤立像素。
#   complexity_calibration:
#     - 若只把原 generator 的顶点移到边框，而 solver 仍按“所有同色像素”处理，
#       则新旧逻辑在这些输入上会退化为同一输出，不构成真正变体。
#     - 因此 generator 必须系统性加入内部同色诱饵，并显式拒绝“旧 solver 也得到同样输出”的样本。
#     - 形状族控制在 3 类：角落对角线、角落直角三角形、角落四边形；自由度不大，但足够
#       形成 2-4 个边框源对象并重复支撑“只认边框顶点”的规律。
#   implicit_assumption_audit_main:
#     - `_solve(..., border_only=True)` 依赖：每个 shape 至少有 2 个边框同色像素。
#       generator 通过每个 shape 的模板构造保证这一点。
#     - 起始顶点选择仍依赖“存在异色 8 邻居的顶点”；generator 若给 shape 分配 shadow，
#       就在该 shape 词典序最小的边框顶点 inward 方向放 1 个 shadow-anchor，并检查它只邻接
#       1 个真实顶点。
#     - `_clear_in_original` 仍假设边之间的中间路径上没有别色阻挡；generator 用 group-level
#       间隔检查保证不同对象之间至少隔 1 格背景，且诱饵像素不会落在别的组路径上。
#     - 旧代码里“收集某色所有像素”这个前提在变体中不成立；这里显式改成 `_is_border(...)`
#       过滤，否则内部诱饵会被误当顶点。
#   placement_legality:
#     - 每个 shape 限制在一个角落扇区内，且每个角最多一个 shape。
#     - 接受候选前，检查该 group 的输出细胞与已有 group 的输出细胞在 8 邻域内都不接触。
#     - shadow 只沿朝向画布内部的固定偏移复制，因此不会跑出边界，也不会把 shadow 颜色带回边框。
#   observability_and_cleanliness:
#     - 样例里同一颜色会同时出现“边框点”和“内部点”，但只有边框点被连接；这是肉眼可见的。
#     - 角落局部 shape 避免多组在中心大面积交叉，输出更整洁。
#   color_inference:
#     - 颜色不参与排序或映射；每个对象只靠位置/边框身份决定行为。
#     - shadow 颜色仍只由与起始顶点相邻的异色 anchor 决定，和原题一致。

from __future__ import annotations

import math
import numpy as np

from arc_utils import bbox_coords, colors_present, get_bg, neighbors8

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


def _is_border(r: int, c: int, h: int, w: int) -> bool:
    return r == 0 or r == h - 1 or c == 0 or c == w - 1


def _solve(grid: np.ndarray, *, border_only: bool) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = get_bg(g)
    out = g.copy()

    colors = sorted(set(int(v) for v in np.unique(g)) - {bg})
    for col in colors:
        pts_rc = []
        for r, c in np.argwhere(g == col):
            rr = int(r)
            cc = int(c)
            if border_only and not _is_border(rr, cc, h, w):
                continue
            pts_rc.append((rr, cc))
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


def main(grid: np.ndarray) -> np.ndarray:
    return _solve(grid, border_only=True)


# --------------- generator ---------------

BG = 0
COLORS = [1, 2, 3, 4, 6, 7, 8]
CANVAS_H_RANGE = (15, 23)
CANVAS_W_JITTER = (-3, 4)
CANVAS_W_CLIP = (15, 22)

SHAPE_COUNT_RANGE = (2, 5)
MAX_SHADOWS_CAP = 3
DISTRACTOR_PROB = 0.45

SHAPE_RETRIES = 60
DISTRACTOR_RETRIES = 80
GENERATE_RETRIES = 250

SHAPE_WEIGHTS = np.array([0.30, 0.35, 0.35], dtype=float)  # line / tri / quad
CORNERS = ["tl", "tr", "bl", "br"]


def _shape_edges(vertices: list[tuple[int, int]]) -> list[tuple[tuple[int, int], tuple[int, int]]]:
    if len(vertices) == 2:
        return [(vertices[0], vertices[1])]
    return [(vertices[i], vertices[(i + 1) % len(vertices)]) for i in range(len(vertices))]


def _shape_cells(vertices: list[tuple[int, int]]) -> set[tuple[int, int]]:
    cells: set[tuple[int, int]] = set()
    for a, b in _shape_edges(vertices):
        cells.update(_segment_cells(a[0], a[1], b[0], b[1]))
    return cells


def _transform_corner(coords: list[tuple[int, int]], h: int, w: int, corner: str) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    for r, c in coords:
        if corner == "tl":
            out.append((r, c))
        elif corner == "tr":
            out.append((r, w - 1 - c))
        elif corner == "bl":
            out.append((h - 1 - r, c))
        else:
            out.append((h - 1 - r, w - 1 - c))
    return out


def _inward_offset(corner: str) -> tuple[int, int]:
    if corner == "tl":
        return 1, 1
    if corner == "tr":
        return 1, -1
    if corner == "bl":
        return -1, 1
    return -1, -1


def _sample_canvas_size() -> tuple[int, int]:
    h = int(np.random.randint(*CANVAS_H_RANGE))
    w = int(np.clip(h + np.random.randint(*CANVAS_W_JITTER), *CANVAS_W_CLIP))
    return h, w


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


def _build_vertices(h: int, w: int, corner: str, family: int) -> list[tuple[int, int]] | None:
    span_hi = min(h, w) // 3
    if span_hi < 5:
        return None

    if family == 0:
        a = int(np.random.randint(4, span_hi + 1))
        base = [(0, a), (a, 0)]
    elif family == 1:
        a = int(np.random.randint(4, span_hi + 1))
        base = [(0, 0), (0, a), (a, 0)]
    else:
        b = int(np.random.randint(5, span_hi + 1))
        a = int(np.random.randint(2, b - 1))
        base = [(0, a), (a, 0), (b, 0), (0, b)]
    return _transform_corner(base, h, w, corner)


def _shadow_spec(
    h: int,
    w: int,
    corner: str,
    vertices: list[tuple[int, int]],
    shape_cells: set[tuple[int, int]],
    shadow_color: int | None,
) -> dict | None:
    if shadow_color is None:
        return {"color": None, "cells": set(), "anchor": None, "offset": None}

    offset = _inward_offset(corner)
    anchor = min(vertices)
    shadow_anchor = (anchor[0] + offset[0], anchor[1] + offset[1])
    if not (0 <= shadow_anchor[0] < h and 0 <= shadow_anchor[1] < w):
        return None
    if shadow_anchor in shape_cells:
        return None
    if sum(1 for vr, vc in vertices if max(abs(shadow_anchor[0] - vr), abs(shadow_anchor[1] - vc)) == 1) != 1:
        return None

    shadow_cells = set()
    for r, c in shape_cells:
        rr = r + offset[0]
        cc = c + offset[1]
        if not (0 <= rr < h and 0 <= cc < w):
            return None
        if (rr, cc) not in shape_cells:
            shadow_cells.add((rr, cc))

    return {"color": int(shadow_color), "cells": shadow_cells, "anchor": shadow_anchor, "offset": offset}


def _paint_cells(grid: np.ndarray, cells: set[tuple[int, int]], color: int) -> None:
    for r, c in cells:
        grid[r, c] = color


def _local_expected_output(
    h: int,
    w: int,
    vertices: list[tuple[int, int]],
    shape_color: int,
    decoy: tuple[int, int],
    shadow: dict,
) -> np.ndarray:
    out = np.full((h, w), BG, dtype=int)
    _paint_cells(out, _shape_cells(vertices), shape_color)
    out[decoy] = shape_color
    if shadow["color"] is not None:
        _paint_cells(out, shadow["cells"], int(shadow["color"]))
    return out


def _local_input(
    h: int,
    w: int,
    vertices: list[tuple[int, int]],
    shape_color: int,
    decoy: tuple[int, int],
    shadow: dict,
) -> np.ndarray:
    inp = np.full((h, w), BG, dtype=int)
    for r, c in vertices:
        inp[r, c] = shape_color
    inp[decoy] = shape_color
    if shadow["color"] is not None and shadow["anchor"] is not None:
        inp[shadow["anchor"]] = int(shadow["color"])
    return inp


def _candidate_decoy_cells(
    h: int,
    w: int,
    vertices: list[tuple[int, int]],
    shape_cells: set[tuple[int, int]],
    shadow_cells: set[tuple[int, int]],
) -> list[tuple[int, int]]:
    coords = np.array(list(shape_cells | shadow_cells | set(vertices)), dtype=int)
    r0, c0, r1, c1 = bbox_coords(coords)
    r0 = max(1, int(r0) - 1)
    c0 = max(1, int(c0) - 1)
    r1 = min(h - 2, int(r1) + 1)
    c1 = min(w - 2, int(c1) + 1)
    center_r = sum(r for r, _ in vertices) / len(vertices)
    center_c = sum(c for _, c in vertices) / len(vertices)
    candidates = []
    for r in range(r0, r1 + 1):
        for c in range(c0, c1 + 1):
            cell = (r, c)
            if cell in shape_cells or cell in shadow_cells:
                continue
            candidates.append(cell)
    candidates.sort(key=lambda p: ((p[0] - center_r) ** 2 + (p[1] - center_c) ** 2, p[0], p[1]))
    return candidates


def _find_decoy(
    h: int,
    w: int,
    vertices: list[tuple[int, int]],
    shape_color: int,
    shadow: dict,
) -> tuple[int, int] | None:
    shape_cells = _shape_cells(vertices)
    shadow_cells = set(shadow["cells"])
    for decoy in _candidate_decoy_cells(h, w, vertices, shape_cells, shadow_cells):
        inp = _local_input(h, w, vertices, shape_color, decoy, shadow)
        expected = _local_expected_output(h, w, vertices, shape_color, decoy, shadow)
        if not np.array_equal(main(inp), expected):
            continue
        if np.array_equal(_solve(inp, border_only=False), expected):
            continue
        return decoy
    return None


def _expand_cells(cells: set[tuple[int, int]], h: int, w: int) -> set[tuple[int, int]]:
    expanded: set[tuple[int, int]] = set()
    for r, c in cells:
        for rr in range(max(0, r - 1), min(h - 1, r + 1) + 1):
            for cc in range(max(0, c - 1), min(w - 1, c + 1) + 1):
                expanded.add((rr, cc))
    return expanded


def _group_bundle(
    h: int,
    w: int,
    corner: str,
    shape_color: int,
    shadow_color: int | None,
    family: int,
) -> dict | None:
    vertices = _build_vertices(h, w, corner, family)
    if vertices is None:
        return None
    shape_cells = _shape_cells(vertices)
    shadow = _shadow_spec(h, w, corner, vertices, shape_cells, shadow_color)
    if shadow is None:
        return None
    decoy = _find_decoy(h, w, vertices, shape_color, shadow)
    if decoy is None:
        return None

    input_cells = set(vertices) | {decoy}
    output_cells = set(shape_cells) | {decoy}
    if shadow["color"] is not None:
        input_cells.add(shadow["anchor"])
        output_cells |= set(shadow["cells"])

    return {
        "vertices": vertices,
        "shape_cells": shape_cells,
        "shape_color": int(shape_color),
        "decoy": decoy,
        "shadow": shadow,
        "input_cells": input_cells,
        "output_cells": output_cells,
        "forbidden": _expand_cells(output_cells, h, w),
    }


def _place_group(inp: np.ndarray, out: np.ndarray, spec: dict) -> None:
    for r, c in spec["vertices"]:
        inp[r, c] = spec["shape_color"]
    inp[spec["decoy"]] = spec["shape_color"]
    _paint_cells(out, spec["shape_cells"], spec["shape_color"])
    out[spec["decoy"]] = spec["shape_color"]

    shadow = spec["shadow"]
    if shadow["color"] is not None:
        inp[shadow["anchor"]] = int(shadow["color"])
        _paint_cells(out, shadow["cells"], int(shadow["color"]))


def _sample_distractor_coords(h: int, w: int, count: int) -> list[tuple[int, int]] | None:
    coords: list[tuple[int, int]] = []
    seen: set[tuple[int, int]] = set()
    while len(coords) < count:
        cand = (int(np.random.randint(1, h - 1)), int(np.random.randint(1, w - 1)))
        if cand in seen:
            continue
        coords.append(cand)
        seen.add(cand)
    for i in range(len(coords)):
        for j in range(i + 1, len(coords)):
            if _aligned_8(coords[i][0], coords[i][1], coords[j][0], coords[j][1]):
                return None
    return coords


def _try_add_distractor(
    inp: np.ndarray,
    out: np.ndarray,
    color: int,
    forbidden: set[tuple[int, int]],
) -> bool:
    h, w = inp.shape
    for _ in range(DISTRACTOR_RETRIES):
        count = int(np.random.randint(1, 4))
        coords = _sample_distractor_coords(h, w, count)
        if coords is None:
            continue
        cells = set(coords)
        if cells & forbidden:
            continue
        _paint_cells(inp, cells, color)
        _paint_cells(out, cells, color)
        forbidden |= _expand_cells(cells, h, w)
        return True
    return False


def _gen_attempt() -> np.ndarray | None:
    h, w = _sample_canvas_size()
    inp = np.full((h, w), BG, dtype=int)
    out = np.full((h, w), BG, dtype=int)

    shape_count = int(np.random.randint(*SHAPE_COUNT_RANGE))
    has_distractor = bool(np.random.rand() < DISTRACTOR_PROB)
    palette = _sample_palette(shape_count, has_distractor)
    if palette is None:
        return None
    shape_colors, shadow_colors, distractor_color = palette
    shadow_map = _assign_shadows(shape_count, shadow_colors)

    chosen_corners = np.random.choice(len(CORNERS), size=shape_count, replace=False).tolist()
    corners = [CORNERS[i] for i in chosen_corners]
    forbidden: set[tuple[int, int]] = set()

    for idx, (shape_color, corner) in enumerate(zip(shape_colors, corners)):
        placed = False
        for _ in range(SHAPE_RETRIES):
            family = int(np.random.choice(3, p=SHAPE_WEIGHTS / SHAPE_WEIGHTS.sum()))
            spec = _group_bundle(h, w, corner, int(shape_color), shadow_map.get(idx), family)
            if spec is None:
                continue
            if spec["output_cells"] & forbidden:
                continue
            _place_group(inp, out, spec)
            forbidden |= spec["forbidden"]
            placed = True
            break
        if not placed:
            return None

    if has_distractor and distractor_color is not None:
        if not _try_add_distractor(inp, out, int(distractor_color), forbidden):
            return None

    total_colors = shape_count + len(shadow_colors) + int(has_distractor)
    if get_bg(inp) != BG or get_bg(out) != BG:
        return None
    if len(colors_present(inp, exclude=[BG])) != total_colors:
        return None
    if not np.array_equal(main(inp), out):
        return None
    if np.array_equal(_solve(inp, border_only=False), out):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GENERATE_RETRIES):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError(f"generate_input failed after {GENERATE_RETRIES} attempts")
