# Variant of 7ed72f31
# Dimension: D2 — 反射轴的定义方式 (src)
# Change: 红色(2)像素构成反射轴：多个红像素同行→水平轴，同列→竖直轴，单个红像素→点反射中心... → 控制像素在网格边缘指定反射轴

from __future__ import annotations

import random
import numpy as np

from arc_utils import get_bg, label_mask


"""
design_analysis:
  variant_summary:
    original: 红色像素本身构成局部反射轴/中心，彩色精灵通常与红色标记直接相连
    new: 红色像素只出现在网格边缘，单个红色控制像素定义一条贯穿整行/整列的隐式射线；彩色精灵不再贴着红色标记，而是根据"离哪条射线最近"来决定反射轴

  a_concrete_behavior_and_trace:
    input_structure:
      - 红色控制像素全部位于边缘，且不放在角上
      - 顶边/底边红点 => 该列是一条竖直反射轴
      - 左边/右边红点 => 该行是一条水平反射轴
      - 彩色精灵是若干单色 4 连通块，位于网格内部，与控制像素分离
    output_structure:
      - 原精灵保留
      - 每个精灵按最近控制射线做镜像，镜像副本画到背景上
      - 红色控制像素本身保持不变，不把整条轴显式画出来
    worked_example:
      - 设顶边有红点 (0, 6)，故列 6 是竖直轴；左边有红点 (9, 0)，故行 9 是水平轴
      - 一个蓝色 L 形精灵 bbox 中心更靠近列 6，而不是行 9，则它整体左右镜像到列 6 的另一侧
      - 原题逻辑会先把红色与附近负载并成同一连通块，再从红像素形状判断 row/col/point
      - 本变体里红点和精灵分离，原题那套"局部红块几何"完全失效；必须先从边缘红点恢复出全局射线，再做最近轴匹配

  b_complexity_calibration:
    - 每个样例同时放 1-2 条竖直轴和 1-2 条水平轴，确保"最近轴匹配"不是伪装成单轴任务
    - 每条轴至少服务一个精灵，避免某类控制像素成为摆设
    - 精灵使用 3-8 像素的不对称单色形状，反射方向能肉眼看出
    - 距离判定只采用少量离散类别：最近竖轴或最近横轴；不再引入颜色映射或多阶段规则

  c_hidden_assumptions_in_original_main:
    - `label_mask(g != bg, connectivity=4)` 默认红色和负载会进入同一非背景连通块
      - 变体中不成立：红点在边缘且与精灵分离，所以不能再从混合连通块里找轴
    - `reds` 的几何形状决定 axis(row/col/point)
      - 变体中不成立：红点永远是单像素，但单像素不再表示点反射中心，而是边缘射线的发射器
    - `eligible(...)` 通过 marker-containing bbox 捕获要反射的 payload
      - 变体中不成立：精灵不再靠 bbox 绑定到局部红标记，必须做全局最近轴匹配
    - `marker_pts` 分支把孤立红点解释为点反射中心
      - 变体中不成立：孤立边缘红点附近通常没有 payload；若沿用旧逻辑，多数样例会退化成"什么都不做"

  d_placement_legality:
    - 控制像素只放在非角落边缘，避免角点同时满足两种方向而歧义
    - 每个精灵原像与镜像都必须完整落在网格内
    - 输出中任意两组像素不重叠，且保留至少 1 格缓冲，避免视觉粘连
    - 精灵到最近轴的距离必须严格小于到其他轴的距离，并留出至少 1 格裕量，保证 solver 与 generator 一致

  e_visual_observability:
    - 人类能从 2-4 个样例看出：红点不再嵌在精灵里，而是固定贴边；复制方向与边缘红点的列/行对齐
    - 同一张图里既有横轴也有竖轴，可以观察到不同精灵被不同边缘红点控制
    - 不使用对称精灵，避免"镜像了但看不出来"

  f_visual_cleanliness:
    - 禁止控制点进入角落
    - 精灵与镜像副本之间、不同组之间都留空隙
    - 不画整条红线，画面上只有少量边缘红点和内部彩色精灵，层次清楚

  g_color_inference:
    - 颜色不参与规则，只用于区分不同精灵
    - 每个精灵使用唯一的非背景非红色，避免同色多块被误认为一个对象

  new_main_assumptions_to_audit:
    - 假设所有红点都在边缘；generator 保证这一点
    - 假设不会出现两个轴与同一精灵等距；generator 用距离裕量保证
    - 假设镜像目标落在背景上；generator 在 output-first 放置时检查
    - 假设不同颜色组件独立处理即可；generator 给每个精灵唯一颜色
"""


MARK = 2


def _axis_key(axis: dict[str, int | tuple[int, int]]) -> tuple[str, int]:
    return str(axis["kind"]), int(axis["value"])


def _extract_axes(grid: np.ndarray) -> list[dict[str, int | tuple[int, int]]]:
    H, W = grid.shape
    axes: list[dict[str, int | tuple[int, int]]] = []
    seen: set[tuple[str, int]] = set()
    for r, c in np.argwhere(grid == int(MARK)):
        r = int(r)
        c = int(c)
        if r not in (0, H - 1) and c not in (0, W - 1):
            continue
        if r in (0, H - 1) and c not in (0, W - 1):
            axis = {"kind": "col", "value": int(c), "control": (r, c)}
        elif c in (0, W - 1) and r not in (0, H - 1):
            axis = {"kind": "row", "value": int(r), "control": (r, c)}
        elif r in (0, H - 1):
            axis = {"kind": "col", "value": int(c), "control": (r, c)}
        else:
            axis = {"kind": "row", "value": int(r), "control": (r, c)}
        key = _axis_key(axis)
        if key not in seen:
            seen.add(key)
            axes.append(axis)
    return axes


def _center_distance(coords: list[tuple[int, int]], axis: dict[str, int | tuple[int, int]]) -> float:
    rs = [int(r) for r, _ in coords]
    cs = [int(c) for _, c in coords]
    center_r = (min(rs) + max(rs)) / 2.0
    center_c = (min(cs) + max(cs)) / 2.0
    if str(axis["kind"]) == "col":
        return abs(center_c - int(axis["value"]))
    return abs(center_r - int(axis["value"]))


def _nearest_axis(coords: list[tuple[int, int]], axes: list[dict[str, int | tuple[int, int]]]) -> tuple[dict[str, int | tuple[int, int]], float, float]:
    scored = sorted((_center_distance(coords, axis), idx, axis) for idx, axis in enumerate(axes))
    best_dist, _idx, best_axis = scored[0]
    second_dist = float("inf") if len(scored) == 1 else float(scored[1][0])
    return best_axis, float(best_dist), float(second_dist)


def _reflect(axis: dict[str, int | tuple[int, int]], r: int, c: int) -> tuple[int, int]:
    if str(axis["kind"]) == "col":
        return int(r), int(2 * int(axis["value"]) - int(c))
    return int(2 * int(axis["value"]) - int(r)), int(c)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Mirror monochrome sprites using implicit reflection axes emitted by border red controls.

    Input structure:
    - bg = most common color. Red (2) appears only on the border as control pixels.
    - A top/bottom border red pixel defines a vertical reflection axis through its column.
    - A left/right border red pixel defines a horizontal reflection axis through its row.
    - Colored sprites are detached from the controls.

    Rule:
    - Split sprites by color and 4-connectivity.
    - Assign each sprite to the nearest control ray, measured by bbox-center distance to the axis.
    - Mirror the whole sprite across that axis. Originals and red controls remain.
    """
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))
    H, W = g.shape
    out = g.copy()

    axes = _extract_axes(g)
    if not axes:
        return out

    colors = sorted({int(v) for v in g.flatten()} - {int(bg), int(MARK)})
    for color in colors:
        lab, n = label_mask(g == int(color), connectivity=4)
        for k in range(1, int(n) + 1):
            coords = [(int(r), int(c)) for r, c in np.argwhere(lab == int(k))]
            if not coords:
                continue
            axis, _best, _second = _nearest_axis(coords, axes)
            for r, c in coords:
                rr, cc = _reflect(axis, int(r), int(c))
                if 0 <= int(rr) < H and 0 <= int(cc) < W and int(out[int(rr), int(cc)]) in (int(bg), int(color)):
                    out[int(rr), int(cc)] = int(color)

    return out


def _normalize(cells: set[tuple[int, int]] | list[tuple[int, int]] | tuple[tuple[int, int], ...]) -> tuple[tuple[int, int], ...]:
    pts = [(int(r), int(c)) for r, c in cells]
    rs = [r for r, _ in pts]
    cs = [c for _, c in pts]
    r0 = min(rs)
    c0 = min(cs)
    return tuple(sorted((int(r - r0), int(c - c0)) for r, c in pts))


def _shape_dims(shape: tuple[tuple[int, int], ...]) -> tuple[int, int]:
    return int(max(r for r, _ in shape) + 1), int(max(c for _, c in shape) + 1)


def _mirror_lr_shape(shape: tuple[tuple[int, int], ...]) -> tuple[tuple[int, int], ...]:
    max_c = max(c for _, c in shape)
    return _normalize({(int(r), int(max_c - c)) for r, c in shape})


def _mirror_ud_shape(shape: tuple[tuple[int, int], ...]) -> tuple[tuple[int, int], ...]:
    max_r = max(r for r, _ in shape)
    return _normalize({(int(max_r - r), int(c)) for r, c in shape})


def _grow_shape(target_px: int, max_h: int, max_w: int) -> tuple[tuple[int, int], ...]:
    cells = {(0, 0)}
    for _ in range(int(target_px) - 1):
        frontier: set[tuple[int, int]] = set()
        for r, c in cells:
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                nr = int(r + dr)
                nc = int(c + dc)
                if (nr, nc) not in cells:
                    frontier.add((nr, nc))
        valid = []
        for nr, nc in frontier:
            cand = cells | {(nr, nc)}
            rs = [r for r, _ in cand]
            cs = [c for _, c in cand]
            if max(rs) - min(rs) + 1 <= int(max_h) and max(cs) - min(cs) + 1 <= int(max_w):
                valid.append((nr, nc))
        if not valid:
            break
        cells.add(random.choice(valid))
    return _normalize(cells)


def _random_shape() -> tuple[tuple[int, int], ...]:
    for _ in range(200):
        target_px = random.randint(3, 8)
        shape = _grow_shape(target_px, max_h=random.randint(2, 4), max_w=random.randint(3, 5))
        h, w = _shape_dims(shape)
        if h < 2 or w < 2:
            continue
        if shape == _mirror_lr_shape(shape):
            continue
        if shape == _mirror_ud_shape(shape):
            continue
        return shape
    raise RuntimeError("failed to create asymmetric shape")


def _pick_spaced_positions(lo: int, hi: int, count: int, gap: int) -> list[int] | None:
    vals = list(range(int(lo), int(hi) + 1))
    random.shuffle(vals)
    chosen: list[int] = []
    for v in vals:
        if all(abs(int(v) - int(prev)) >= int(gap) for prev in chosen):
            chosen.append(int(v))
            if len(chosen) == int(count):
                return sorted(chosen)
    return None


def _coords_from_shape(shape: tuple[tuple[int, int], ...], r0: int, c0: int) -> list[tuple[int, int]]:
    return [(int(r0 + r), int(c0 + c)) for r, c in shape]


def _clearance_ok(cells: set[tuple[int, int]], occupied: set[tuple[int, int]], H: int, W: int) -> bool:
    for r, c in cells:
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                rr = int(r + dr)
                cc = int(c + dc)
                if 0 <= rr < int(H) and 0 <= cc < int(W) and (rr, cc) in occupied:
                    return False
    return True


def _place_component(
    shape: tuple[tuple[int, int], ...],
    axis: dict[str, int | tuple[int, int]],
    axes: list[dict[str, int | tuple[int, int]]],
    H: int,
    W: int,
    occupied_out: set[tuple[int, int]],
) -> tuple[list[tuple[int, int]], list[tuple[int, int]]] | None:
    h, w = _shape_dims(shape)
    max_r = max(r for r, _ in shape)
    max_c = max(c for _, c in shape)
    side_order = ["left", "right"] if str(axis["kind"]) == "col" else ["up", "down"]
    random.shuffle(side_order)

    for side in side_order:
        for _ in range(220):
            if str(axis["kind"]) == "col":
                axis_c = int(axis["value"])
                r0_lo = 1
                r0_hi = int(H - h - 2)
                if r0_lo > r0_hi:
                    continue
                r0 = random.randint(r0_lo, r0_hi)
                if side == "left":
                    c0_lo = max(1, 2 * axis_c - (W - 1))
                    c0_hi = axis_c - 1 - max_c
                else:
                    c0_lo = axis_c + 1
                    c0_hi = min(W - w - 2, 2 * axis_c - max_c)
            else:
                axis_r = int(axis["value"])
                c0_lo = 1
                c0_hi = int(W - w - 2)
                if c0_lo > c0_hi:
                    continue
                c0 = random.randint(c0_lo, c0_hi)
                if side == "up":
                    r0_lo = max(1, 2 * axis_r - (H - 1))
                    r0_hi = axis_r - 1 - max_r
                else:
                    r0_lo = axis_r + 1
                    r0_hi = min(H - h - 2, 2 * axis_r - max_r)

            if str(axis["kind"]) == "col":
                if c0_lo > c0_hi:
                    continue
                c0 = random.randint(c0_lo, c0_hi)
            else:
                if r0_lo > r0_hi:
                    continue
                r0 = random.randint(r0_lo, r0_hi)

            orig = _coords_from_shape(shape, int(r0), int(c0))
            best_axis, best_dist, second_dist = _nearest_axis(orig, axes)
            if _axis_key(best_axis) != _axis_key(axis):
                continue
            if best_dist < 1.0 or second_dist - best_dist < 1.0:
                continue

            mirrored = [_reflect(axis, r, c) for r, c in orig]
            cell_set = set(orig) | set(mirrored)
            if len(cell_set) != len(orig) + len(mirrored):
                continue
            if any(not (0 <= int(r) < H and 0 <= int(c) < W) for r, c in cell_set):
                continue
            if not _clearance_ok(cell_set, occupied_out, H, W):
                continue
            return orig, mirrored
    return None


def _build_instance() -> tuple[np.ndarray, np.ndarray]:
    H = random.randint(20, 26)
    W = random.randint(20, 26)
    bg = random.choice([c for c in range(10) if c != int(MARK)])

    n_vert = random.randint(1, 2)
    n_horz = random.randint(1, 2)
    vert_cols = _pick_spaced_positions(4, W - 5, n_vert, gap=5)
    horz_rows = _pick_spaced_positions(4, H - 5, n_horz, gap=5)
    if vert_cols is None or horz_rows is None:
        raise RuntimeError("failed to choose axis positions")

    axes: list[dict[str, int | tuple[int, int]]] = []
    controls: list[tuple[int, int]] = []
    for c in vert_cols:
        ctrl = (0, int(c)) if random.random() < 0.5 else (H - 1, int(c))
        controls.append(ctrl)
        axes.append({"kind": "col", "value": int(c), "control": ctrl})
    for r in horz_rows:
        ctrl = (int(r), 0) if random.random() < 0.5 else (int(r), W - 1)
        controls.append(ctrl)
        axes.append({"kind": "row", "value": int(r), "control": ctrl})

    n_sprites = random.randint(max(3, len(axes)), min(6, len(axes) + 2))
    assignments = list(axes)
    while len(assignments) < int(n_sprites):
        assignments.append(random.choice(axes))
    random.shuffle(assignments)

    palette = [c for c in range(10) if c not in (int(bg), int(MARK))]
    if len(palette) < int(n_sprites):
        raise RuntimeError("not enough colors")
    colors = random.sample(palette, int(n_sprites))

    inp = np.full((H, W), int(bg), dtype=int)
    out = np.full((H, W), int(bg), dtype=int)
    for r, c in controls:
        inp[int(r), int(c)] = int(MARK)
        out[int(r), int(c)] = int(MARK)

    occupied_out = set((int(r), int(c)) for r, c in controls)

    for color, axis in zip(colors, assignments):
        placed = None
        for _ in range(120):
            shape = _random_shape()
            placed = _place_component(shape, axis, axes, H, W, occupied_out)
            if placed is not None:
                break
        if placed is None:
            raise RuntimeError("failed to place component")

        orig, mirrored = placed
        for r, c in orig:
            inp[int(r), int(c)] = int(color)
            out[int(r), int(c)] = int(color)
        for r, c in mirrored:
            out[int(r), int(c)] = int(color)
        occupied_out |= set(orig) | set(mirrored)

    return inp, out


def generate_input() -> np.ndarray:
    for _ in range(300):
        try:
            inp, out = _build_instance()
        except RuntimeError:
            continue
        if np.array_equal(main(inp), out):
            return inp
    raise RuntimeError("failed to generate a valid D2 variant instance")
