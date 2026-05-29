# Variant of 7ed72f31
# Dimension: D2 — 反射轴的定义方式 (src)
# Change: 红色(2)像素构成反射轴：多个红像素同行→水平轴，同列→竖直轴，单个红像素→点反射中心... → 对称线检测作为反射轴

from __future__ import annotations

import random
from typing import Iterable

import numpy as np

from arc_utils import get_bg, label_mask


# Design analysis for the D2 variant
# ==================================
# spec_variant:
#   core_change:
#     old: 每个局部红色标记自己定义一条轴或一个点中心，solver 先找“红+payload”混合连通块，再在该局部框内镜像
#     new: 整个网格存在唯一一条“全局分隔线”——一整行或一整列红色(2)；所有彩色精灵都要统一跨这条线镜像
#   concrete_input:
#     - 背景色仍然占主导，且不为 2
#     - 恰好一整行或一整列全部为 2，视觉上像横向/纵向隔板
#     - 3-6 个单色 4 连通精灵全部放在隔板同一侧；另一侧输入中保持为空背景
#     - 精灵与隔板之间至少留 1 格空隙，彼此之间也留 1 格空隙，避免视觉拥挤
#   concrete_output:
#     - 保留原输入所有像素
#     - 每个非背景非红色像素都按全局轴做镜像，在另一侧补出同色副本
#   small_trace:
#     - 例：第 8 行全红，某个蓝色像素在 (5, 4)
#     - 原任务逻辑：红线是独立连通块，没有和 payload 连在一起，`payload` 为空，因此原 solver 对它什么也不做
#     - 本变体逻辑：先检测到“整行全红”是全局轴，再把 (5, 4) 映到 (11, 4)
#     - 差异：原题依赖局部红标记几何；变体依赖全局分隔线检测
#   complexity_calibration:
#     - 只保留“整行/整列”两种轴类型，不再混入点反射；这样规则足够集中，2-4 个样例即可看出“所有精灵共用同一条轴”
#     - 通过随机横轴/竖轴、多个不同颜色和不规则形状，保证视觉上明显不同于原题的“每组各有自己的红标记”
#   implicit_assumptions_in_old_main:
#     - `label_mask(g != bg, connectivity=4)` 仍然无害，只是现在红色隔板会成为一个独立大连通块
#     - `if MARK not in vals: continue` 与后续 `reds` 提取默认“红标记和 payload 在同一组件内”；变体里这不成立，因为隔板与所有精灵都分离
#     - `axis = ("row"/"col"/"point")` 由局部红像素几何决定；变体里必须改为扫描整行/整列
#     - `eligible(...)` 依赖组件 bbox 近邻范围，只会抓局部 payload；变体里需要对所有 payload 组件统一处理
#     - “isolated red pixel” 分支完全依赖单个红点；变体里不存在，应删除
#   placement_legality:
#     - generator 只放置一条全局红线，禁止第二条整行/整列红线，避免 solver 轴检测歧义
#     - 精灵全部位于轴的一侧，而且这侧的厚度不大于另一侧，以保证镜像后一定留在网格内
#     - 输入时目标侧保持背景，因此镜像输出不会覆盖已有对象
#   visual_observability:
#     - 一整条红色分隔线在所有样例里都非常醒目，人类可直接把它视为“镜子”
#     - 同一侧出现多个颜色/形状不同的精灵，输出中它们一起跨线复制，能明确支持“全局统一规则”
#   cleanliness_and_color_reasoning:
#     - 每个精灵单色、颜色数控制在 3-6，避免颜色规则成为额外谜题
#     - 不加入无关噪声精灵，避免解题者怀疑某些对象不参与
#   new_main_assumptions_to_audit:
#     - 恰好存在一条完整红线（行或列）作为轴
#     - 除轴以外，红色不再承担别的语义；payload 颜色均非背景、非红色
#     - 所有输入精灵都位于可完整反射的一侧，且其镜像落点在输入中是背景
#     - generator 必须保证以上条件，否则新 solver 会遇到歧义或覆盖问题


MARK = 2


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = map(int, shape)
    return 0 <= int(r) < h and 0 <= int(c) < w


def _detect_global_axis(grid: np.ndarray) -> tuple[str, int]:
    g = np.array(grid, dtype=int)
    rows = [int(r) for r in range(g.shape[0]) if np.all(g[int(r), :] == int(MARK))]
    cols = [int(c) for c in range(g.shape[1]) if np.all(g[:, int(c)] == int(MARK))]
    if len(rows) + len(cols) != 1:
        raise ValueError("expected exactly one full red separator line")
    if rows:
        return "row", int(rows[0])
    return "col", int(cols[0])


def _reflect(axis: tuple[str, int], r: int, c: int) -> tuple[int, int]:
    kind, pos = axis
    if kind == "row":
        return int(2 * int(pos) - int(r)), int(c)
    return int(r), int(2 * int(pos) - int(c))


def main(grid: np.ndarray) -> np.ndarray:
    """
    Global-axis reflection variant of 7ed72f31.

    The unique full red row/column is the reflection axis.
    Every non-red, non-background 4-connected component is mirrored across it.
    Originals and the red separator remain in place.
    """
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))
    axis = _detect_global_axis(g)
    out = g.copy()

    payload_mask = (g != int(bg)) & (g != int(MARK))
    labs, n = label_mask(payload_mask, connectivity=4)
    for k in range(1, int(n) + 1):
        for r, c in np.argwhere(labs == int(k)):
            rr, cc = _reflect(axis, int(r), int(c))
            if not _in_bounds(g.shape, rr, cc):
                continue
            if int(g[int(rr), int(cc)]) != int(bg):
                continue
            out[int(rr), int(cc)] = int(g[int(r), int(c)])

    return out


def _normalize(cells: Iterable[tuple[int, int]]) -> list[tuple[int, int]]:
    pts = [(int(r), int(c)) for r, c in cells]
    min_r = min(r for r, _ in pts)
    min_c = min(c for _, c in pts)
    return sorted((int(r - min_r), int(c - min_c)) for r, c in pts)


def _random_shape(rng: random.Random, size: int) -> list[tuple[int, int]]:
    cells = {(0, 0)}
    while len(cells) < int(size):
        frontier = set()
        for r, c in cells:
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                rr, cc = int(r + dr), int(c + dc)
                if (rr, cc) in cells:
                    continue
                if abs(rr) > 3 or abs(cc) > 3:
                    continue
                frontier.add((rr, cc))
        if not frontier:
            break
        cells.add(rng.choice(sorted(frontier)))
    return _normalize(cells)


def _bbox(cells: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rs = [int(r) for r, _ in cells]
    cs = [int(c) for _, c in cells]
    return int(min(rs)), int(min(cs)), int(max(rs)), int(max(cs))


def _shift(cells: list[tuple[int, int]], dr: int, dc: int) -> list[tuple[int, int]]:
    return [(int(r + dr), int(c + dc)) for r, c in cells]


def _expanded_bbox_ok(grid: np.ndarray, cells: list[tuple[int, int]], bg: int) -> bool:
    h, w = grid.shape
    r0, c0, r1, c1 = _bbox(cells)
    r0 = max(0, int(r0 - 1))
    c0 = max(0, int(c0 - 1))
    r1 = min(int(h - 1), int(r1 + 1))
    c1 = min(int(w - 1), int(c1 + 1))
    return bool(np.all(grid[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1] == int(bg)))


def _make_case(rng: random.Random) -> tuple[np.ndarray, np.ndarray, dict]:
    h = rng.randint(18, 28)
    w = rng.randint(18, 28)
    bg = rng.choice([c for c in range(10) if c != int(MARK)])
    grid = np.full((h, w), int(bg), dtype=int)

    orient = rng.choice(["row", "col"])
    if orient == "row":
        source_side = rng.choice(["top", "bottom"])
        if source_side == "top":
            axis_pos = rng.randint(6, (h - 1) // 2)
        else:
            axis_pos = rng.randint(h // 2, h - 7)
        grid[int(axis_pos), :] = int(MARK)
        row_lo = 1 if source_side == "top" else int(axis_pos + 2)
        row_hi = int(axis_pos - 2) if source_side == "top" else h - 2
        col_lo, col_hi = 1, w - 2
        axis = ("row", int(axis_pos))
    else:
        source_side = rng.choice(["left", "right"])
        if source_side == "left":
            axis_pos = rng.randint(6, (w - 1) // 2)
        else:
            axis_pos = rng.randint(w // 2, w - 7)
        grid[:, int(axis_pos)] = int(MARK)
        row_lo, row_hi = 1, h - 2
        col_lo = 1 if source_side == "left" else int(axis_pos + 2)
        col_hi = int(axis_pos - 2) if source_side == "left" else w - 2
        axis = ("col", int(axis_pos))

    n_sprites = rng.randint(3, 6)
    colors = [c for c in range(10) if c not in (int(bg), int(MARK))]
    rng.shuffle(colors)
    colors = colors[: int(n_sprites)]

    placed = 0
    for color in colors:
        success = False
        for _ in range(120):
            size = rng.randint(3, 8)
            shape = _random_shape(rng, size)
            sh_h = max(r for r, _ in shape) + 1
            sh_w = max(c for _, c in shape) + 1

            if row_hi - row_lo + 1 < sh_h or col_hi - col_lo + 1 < sh_w:
                continue

            top = rng.randint(int(row_lo), int(row_hi - sh_h + 1))
            left = rng.randint(int(col_lo), int(col_hi - sh_w + 1))
            cells = _shift(shape, int(top), int(left))

            if not _expanded_bbox_ok(grid, cells, int(bg)):
                continue

            mirrored = [_reflect(axis, int(r), int(c)) for r, c in cells]
            if any(not _in_bounds(grid.shape, rr, cc) for rr, cc in mirrored):
                continue
            if any(int(grid[int(rr), int(cc)]) != int(bg) for rr, cc in mirrored):
                continue

            for r, c in cells:
                grid[int(r), int(c)] = int(color)
            success = True
            placed += 1
            break

        if not success:
            break

    if placed < 3:
        raise ValueError("failed to place enough sprites")

    output = grid.copy()
    for r, c in np.argwhere((grid != int(bg)) & (grid != int(MARK))):
        rr, cc = _reflect(axis, int(r), int(c))
        output[int(rr), int(cc)] = int(grid[int(r), int(c)])

    meta = {
        "shape": (int(h), int(w)),
        "axis_kind": axis[0],
        "axis_pos": int(axis[1]),
        "source_side": source_side,
        "sprite_count": int(placed),
        "bg": int(bg),
    }
    return grid, output, meta


def generate_input() -> np.ndarray:
    for _ in range(200):
        rng = random.Random()
        try:
            inp, out, _meta = _make_case(rng)
        except ValueError:
            continue
        got = main(inp)
        if np.array_equal(got, out):
            return inp
    raise RuntimeError("failed to generate a valid D2 variant instance")
