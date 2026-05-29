# Combo variant of 7ed72f31 (index 8)
# combo_spec: D2/3+D7/1
# Name: 边缘射线轴+扩大检测半径

from __future__ import annotations

import random

import numpy as np

from arc_utils import get_bg, label_mask


"""
design_analysis:
  variant_summary:
    original: 红色像素既可能是嵌在 sprite 里的轴，也可能是孤立点反射中心；孤立点只看切比雪夫距离 1 的邻域
    combo: 边缘红点不再是局部几何的一部分，而是发射穿过整行/整列的隐式反射射线；同时保留“孤立红点做点反射”的支路，但其可捕获 sprite 的半径从 1 扩大到 2

  a_concrete_behavior_and_trace:
    worked_example:
      - 顶边红点 (0, 6) 定义列 6 的竖直反射轴；左边红点 (9, 0) 定义行 9 的水平反射轴
      - 一个蓝色 L 形块位于列 6 左侧，且其 bbox 中心离列 6 比离行 9 更近，所以它绕列 6 左右镜像
      - 另有一个内部孤立红点 (13, 13)，其上方两格有一个绿色短横条 {(11, 12), (11, 13), (11, 14)}
      - 原题孤立红点分支只看 1 格邻域，看不到这条绿色短横条，所以不会触发点反射
      - 融合版先用半径 2 捕获该绿色组件，再绕 (13, 13) 做点反射，于是在下方两格生成对应副本
    difference_from_original:
      - 原 solver 假设“红点和 payload 在同一个非背景 4 连通块里”，这里边缘控制点与内部 sprite 完全分离，该假设失效
      - 原 solver 对孤立红点只扫描 8 邻域，这里必须扩到切比雪夫距离 <= 2，否则 generator 中的点反射组会全部漏掉

  b_complexity_calibration:
    - 每个样例同时放 1-2 条竖轴和 1-2 条横轴，避免退化成单轴镜像
    - 每个样例至少放 1 个“只在半径 2 可见”的点反射组，确保 D7/1 不是摆设
    - 轴控 sprite 使用 3-8 像素的不对称单色 4 连通形状，镜像方向肉眼可见
    - 点反射 sprite 保持小而清晰，全部位于红中心的第 2 圈上，视觉上能直接看出“不是贴着红点”

  c_hidden_assumptions_in_original_main:
    - `label_mask(g != bg, connectivity=4)` 后从混合连通块里提取红轴
      - 融合后不成立：边缘控制点和内部 sprite 分离，必须先全局提取边缘射线
    - `len(reds) == 1` 表示点反射中心
      - 融合后不成立：边缘红点也是单像素，但它表示整行/整列射线，不是点反射中心
    - `eligible(...)` 依赖 marker bbox 捕获 payload
      - 融合后不成立：轴控 sprite 与红点脱钩，必须按“离哪条射线最近”分配
    - 孤立红点只扫描 8 邻域
      - 融合后不成立：generator 故意把点反射 sprite 放在切比雪夫距离 2 处
    - 多项改动叠加后的新增冲突:
      - 点反射 sprite 也可能是普通彩色组件，所以 solver 必须先让内部孤立红点“认领”半径 2 内的组件，再把剩余组件交给边缘射线；否则会被重复处理

  d_placement_legality:
    - 边缘控制点不放角落，避免同一像素同时像横轴和竖轴
    - 轴控 sprite 的原像与镜像都必须完全在网格内，且离最近轴严格最近
    - 点反射中心周围切比雪夫半径 2 内，不允许混入其他对象，避免误认领
    - 输出中的所有非背景像素不重叠，并留 1 格缓冲，防止视觉粘连

  e_visual_observability:
    - 解题者能同时看到“边缘红点控制内部镜像”和“内部孤立红点隔 2 格抓取组件”的两种行为
    - 点反射组全部隔着一圈背景，半径扩大是可见的，不依赖颜色编号等不可视因素
    - 轴控组和点控组都保留原像，只新增镜像副本，输入输出对比清楚

  f_visual_cleanliness:
    - 控制点数量有限，颜色唯一，避免同色多块互相干扰
    - 点反射组使用短横/短竖小组件，边缘轴组使用更大的不对称块，视觉层次稳定
    - 不显式画整条射线，只保留边缘红点，画面不会被红线塞满

  g_color_inference:
    - 颜色不承载规则，只区分对象
    - 每个对象组使用唯一的非背景非红色，保证“哪个副本对应哪个原像”始终可见

  new_main_assumptions_to_audit:
    - 假设所有边缘红点都位于边界且不在角上；generator 保证
    - 假设点反射中心是内部孤立红点，且其半径 2 内只会命中 intended 组件；generator 用半径 2 净空保证
    - 假设普通轴控组件不会与点反射组件共享像素；generator 给每组唯一颜色并做输出占位检查
    - 假设某个轴控组件到最近射线的距离有严格优势；generator 检查最近轴 margin
"""


MARK = 2


def _axis_key(axis: dict[str, int | tuple[int, int]]) -> tuple[str, int]:
    return str(axis["kind"]), int(axis["value"])


def _in_bounds(H: int, W: int, r: int, c: int) -> bool:
    return 0 <= int(r) < int(H) and 0 <= int(c) < int(W)


def _reflect_axis(axis: dict[str, int | tuple[int, int]], r: int, c: int) -> tuple[int, int]:
    if str(axis["kind"]) == "col":
        return int(r), int(2 * int(axis["value"]) - int(c))
    return int(2 * int(axis["value"]) - int(r)), int(c)


def _reflect_point(center: tuple[int, int], r: int, c: int) -> tuple[int, int]:
    pr, pc = map(int, center)
    return int(2 * pr - int(r)), int(2 * pc - int(c))


def _extract_axes(grid: np.ndarray) -> list[dict[str, int | tuple[int, int]]]:
    H, W = grid.shape
    axes: list[dict[str, int | tuple[int, int]]] = []
    seen: set[tuple[str, int]] = set()
    for r, c in np.argwhere(grid == int(MARK)):
        r = int(r)
        c = int(c)
        if r not in (0, H - 1) and c not in (0, W - 1):
            continue
        if r in (0, H - 1):
            axis = {"kind": "col", "value": int(c), "control": (int(r), int(c))}
        else:
            axis = {"kind": "row", "value": int(r), "control": (int(r), int(c))}
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


def _nearest_axis(
    coords: list[tuple[int, int]], axes: list[dict[str, int | tuple[int, int]]]
) -> tuple[dict[str, int | tuple[int, int]], float, float]:
    scored = sorted((_center_distance(coords, axis), idx, axis) for idx, axis in enumerate(axes))
    best_dist, _idx, best_axis = scored[0]
    second_dist = float("inf") if len(scored) == 1 else float(scored[1][0])
    return best_axis, float(best_dist), float(second_dist)


def _extract_isolated_point_centers(grid: np.ndarray, bg: int) -> list[tuple[int, int]]:
    H, W = grid.shape
    centers: list[tuple[int, int]] = []
    lab, n = label_mask(grid != int(bg), connectivity=4)
    for k in range(1, int(n) + 1):
        coords = [(int(r), int(c)) for r, c in np.argwhere(lab == int(k))]
        if len(coords) != 1:
            continue
        r, c = coords[0]
        if int(grid[int(r), int(c)]) != int(MARK):
            continue
        if r in (0, H - 1) or c in (0, W - 1):
            continue
        centers.append((int(r), int(c)))
    return centers


def main(grid: np.ndarray) -> np.ndarray:
    """
    Solve the combo by prioritizing radius-2 point-reflection groups, then mirroring
    the remaining colored components across the nearest border-emitted axis.
    """
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))
    H, W = g.shape
    out = g.copy()

    colors = sorted({int(v) for v in g.flatten()} - {int(bg), int(MARK)})
    axes = _extract_axes(g)
    centers = _extract_isolated_point_centers(g, bg)

    color_lab8 = {int(color): label_mask(g == int(color), connectivity=8)[0] for color in colors}
    claimed = np.zeros((H, W), dtype=bool)

    # Stage 1: interior isolated red points claim nearby components using radius 2.
    for pr, pc in centers:
        touched: set[tuple[int, int]] = set()
        for rr in range(max(0, int(pr - 2)), min(H - 1, int(pr + 2)) + 1):
            for cc in range(max(0, int(pc - 2)), min(W - 1, int(pc + 2)) + 1):
                if max(abs(int(rr - pr)), abs(int(cc - pc))) > 2:
                    continue
                if int(rr) == int(pr) and int(cc) == int(pc):
                    continue
                v = int(g[int(rr), int(cc)])
                if v == int(bg) or v == int(MARK):
                    continue
                lbl = int(color_lab8[int(v)][int(rr), int(cc)])
                if lbl != 0:
                    touched.add((int(v), int(lbl)))

        for color, lbl in sorted(touched):
            coords = [(int(r), int(c)) for r, c in np.argwhere(color_lab8[int(color)] == int(lbl))]
            if not coords or any(bool(claimed[int(r), int(c)]) for r, c in coords):
                continue
            for r, c in coords:
                claimed[int(r), int(c)] = True
                rr, cc = _reflect_point((int(pr), int(pc)), int(r), int(c))
                if _in_bounds(H, W, rr, cc) and int(out[int(rr), int(cc)]) in (int(bg), int(color)):
                    out[int(rr), int(cc)] = int(color)

    # Stage 2: all remaining colored components follow the nearest border ray.
    if not axes:
        return out

    for color in colors:
        mask = (g == int(color)) & (~claimed)
        lab4, n4 = label_mask(mask, connectivity=4)
        for k in range(1, int(n4) + 1):
            coords = [(int(r), int(c)) for r, c in np.argwhere(lab4 == int(k))]
            if not coords:
                continue
            axis, _best, _second = _nearest_axis(coords, axes)
            for r, c in coords:
                rr, cc = _reflect_axis(axis, int(r), int(c))
                if _in_bounds(H, W, rr, cc) and int(out[int(rr), int(cc)]) in (int(bg), int(color)):
                    out[int(rr), int(cc)] = int(color)

    return out


def _main_radius1_ablation(grid: np.ndarray) -> np.ndarray:
    """
    Control solver used only by the generator's anti-degeneration check:
    identical to `main` except isolated red points only see radius 1.
    """
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))
    H, W = g.shape
    out = g.copy()

    colors = sorted({int(v) for v in g.flatten()} - {int(bg), int(MARK)})
    axes = _extract_axes(g)
    centers = _extract_isolated_point_centers(g, bg)

    color_lab8 = {int(color): label_mask(g == int(color), connectivity=8)[0] for color in colors}
    claimed = np.zeros((H, W), dtype=bool)

    for pr, pc in centers:
        touched: set[tuple[int, int]] = set()
        for rr in range(max(0, int(pr - 1)), min(H - 1, int(pr + 1)) + 1):
            for cc in range(max(0, int(pc - 1)), min(W - 1, int(pc + 1)) + 1):
                if int(rr) == int(pr) and int(cc) == int(pc):
                    continue
                v = int(g[int(rr), int(cc)])
                if v == int(bg) or v == int(MARK):
                    continue
                lbl = int(color_lab8[int(v)][int(rr), int(cc)])
                if lbl != 0:
                    touched.add((int(v), int(lbl)))

        for color, lbl in sorted(touched):
            coords = [(int(r), int(c)) for r, c in np.argwhere(color_lab8[int(color)] == int(lbl))]
            if not coords or any(bool(claimed[int(r), int(c)]) for r, c in coords):
                continue
            for r, c in coords:
                claimed[int(r), int(c)] = True
                rr, cc = _reflect_point((int(pr), int(pc)), int(r), int(c))
                if _in_bounds(H, W, rr, cc) and int(out[int(rr), int(cc)]) in (int(bg), int(color)):
                    out[int(rr), int(cc)] = int(color)

    if not axes:
        return out

    for color in colors:
        mask = (g == int(color)) & (~claimed)
        lab4, n4 = label_mask(mask, connectivity=4)
        for k in range(1, int(n4) + 1):
            coords = [(int(r), int(c)) for r, c in np.argwhere(lab4 == int(k))]
            if not coords:
                continue
            axis, _best, _second = _nearest_axis(coords, axes)
            for r, c in coords:
                rr, cc = _reflect_axis(axis, int(r), int(c))
                if _in_bounds(H, W, rr, cc) and int(out[int(rr), int(cc)]) in (int(bg), int(color)):
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
            cand = cells | {(int(nr), int(nc))}
            rs = [r for r, _ in cand]
            cs = [c for _, c in cand]
            if max(rs) - min(rs) + 1 <= int(max_h) and max(cs) - min(cs) + 1 <= int(max_w):
                valid.append((int(nr), int(nc)))
        if not valid:
            break
        cells.add(random.choice(valid))
    return _normalize(cells)


def _random_axis_shape() -> tuple[tuple[int, int], ...]:
    for _ in range(200):
        shape = _grow_shape(
            target_px=random.randint(3, 8),
            max_h=random.randint(2, 4),
            max_w=random.randint(3, 5),
        )
        h, w = _shape_dims(shape)
        if h < 2 or w < 2:
            continue
        if shape == _mirror_lr_shape(shape):
            continue
        if shape == _mirror_ud_shape(shape):
            continue
        return shape
    raise RuntimeError("failed to create asymmetric axis-controlled shape")


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


def _clearance_ok(cells: set[tuple[int, int]], occupied: set[tuple[int, int]], H: int, W: int, radius: int = 1) -> bool:
    for r, c in cells:
        for rr in range(int(r - radius), int(r + radius) + 1):
            for cc in range(int(c - radius), int(c + radius) + 1):
                if 0 <= int(rr) < int(H) and 0 <= int(cc) < int(W) and (int(rr), int(cc)) in occupied:
                    return False
    return True


def _place_axis_component(
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
        for _ in range(240):
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
                if c0_lo > c0_hi:
                    continue
                c0 = random.randint(c0_lo, c0_hi)
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
                if r0_lo > r0_hi:
                    continue
                r0 = random.randint(r0_lo, r0_hi)

            orig = _coords_from_shape(shape, int(r0), int(c0))
            best_axis, best_dist, second_dist = _nearest_axis(orig, axes)
            if _axis_key(best_axis) != _axis_key(axis):
                continue
            if best_dist < 1.0 or second_dist - best_dist < 1.0:
                continue

            mirrored = [_reflect_axis(axis, r, c) for r, c in orig]
            cell_set = set(orig) | set(mirrored)
            if len(cell_set) != len(orig) + len(mirrored):
                continue
            if any(not _in_bounds(H, W, r, c) for r, c in cell_set):
                continue
            if not _clearance_ok(cell_set, occupied_out, H, W, radius=1):
                continue
            return orig, mirrored
    return None


def _random_point_ring_shape() -> tuple[tuple[int, int], ...]:
    side = random.choice(["top", "bottom", "left", "right"])
    length = random.randint(2, 4)
    start = random.randint(-2, 3 - int(length))
    if side == "top":
        return tuple((int(-2), int(start + i)) for i in range(int(length)))
    if side == "bottom":
        return tuple((int(2), int(start + i)) for i in range(int(length)))
    if side == "left":
        return tuple((int(start + i), int(-2)) for i in range(int(length)))
    return tuple((int(start + i), int(2)) for i in range(int(length)))


def _place_point_component(
    H: int,
    W: int,
    occupied_out: set[tuple[int, int]],
    occupied_input: set[tuple[int, int]],
) -> tuple[tuple[int, int], list[tuple[int, int]], list[tuple[int, int]]] | None:
    for _ in range(320):
        pr = random.randint(3, H - 4)
        pc = random.randint(3, W - 4)
        center = (int(pr), int(pc))
        offsets = _random_point_ring_shape()
        orig = [(int(pr + dr), int(pc + dc)) for dr, dc in offsets]
        mirrored = [(int(pr - dr), int(pc - dc)) for dr, dc in offsets]
        cell_set = set(orig) | set(mirrored) | {center}
        if len(cell_set) != len(orig) + len(mirrored) + 1:
            continue
        if any(not _in_bounds(H, W, r, c) for r, c in cell_set):
            continue
        if not _clearance_ok(cell_set, occupied_out, H, W, radius=1):
            continue
        if not _clearance_ok({center}, occupied_input, H, W, radius=2):
            continue
        return center, orig, mirrored
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
        raise RuntimeError("failed to choose border-ray axes")

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

    n_point_groups = random.randint(1, 2)
    max_axis_groups = min(6, 8 - int(n_point_groups))
    n_axis_groups = random.randint(len(axes), max_axis_groups)

    assignments = list(axes)
    while len(assignments) < int(n_axis_groups):
        assignments.append(random.choice(axes))
    random.shuffle(assignments)

    palette = [c for c in range(10) if c not in (int(bg), int(MARK))]
    random.shuffle(palette)
    if len(palette) < int(n_axis_groups + n_point_groups):
        raise RuntimeError("not enough colors")
    axis_colors = palette[: int(n_axis_groups)]
    point_colors = palette[int(n_axis_groups) : int(n_axis_groups + n_point_groups)]

    inp = np.full((H, W), int(bg), dtype=int)
    out = np.full((H, W), int(bg), dtype=int)
    for r, c in controls:
        inp[int(r), int(c)] = int(MARK)
        out[int(r), int(c)] = int(MARK)

    occupied_input = set((int(r), int(c)) for r, c in controls)
    occupied_out = set((int(r), int(c)) for r, c in controls)

    for color, axis in zip(axis_colors, assignments):
        placed = None
        for _ in range(140):
            shape = _random_axis_shape()
            placed = _place_axis_component(shape, axis, axes, H, W, occupied_out)
            if placed is not None:
                break
        if placed is None:
            raise RuntimeError("failed to place axis-controlled component")

        orig, mirrored = placed
        for r, c in orig:
            inp[int(r), int(c)] = int(color)
            out[int(r), int(c)] = int(color)
        for r, c in mirrored:
            out[int(r), int(c)] = int(color)
        occupied_input |= set(orig)
        occupied_out |= set(orig) | set(mirrored)

    for color in point_colors:
        placed = _place_point_component(H, W, occupied_out, occupied_input)
        if placed is None:
            raise RuntimeError("failed to place radius-2 point component")
        center, orig, mirrored = placed

        inp[int(center[0]), int(center[1])] = int(MARK)
        out[int(center[0]), int(center[1])] = int(MARK)
        for r, c in orig:
            inp[int(r), int(c)] = int(color)
            out[int(r), int(c)] = int(color)
        for r, c in mirrored:
            out[int(r), int(c)] = int(color)

        occupied_input |= {center} | set(orig)
        occupied_out |= {center} | set(orig) | set(mirrored)

    return inp, out


def generate_input() -> np.ndarray:
    for _ in range(400):
        try:
            inp, out = _build_instance()
        except RuntimeError:
            continue
        combo_out = main(inp)
        if not np.array_equal(combo_out, out):
            continue
        if np.array_equal(_main_radius1_ablation(inp), combo_out):
            continue
        if np.array_equal(combo_out, inp):
            continue
        if np.array_equal(combo_out, out):
            return inp
    raise RuntimeError("failed to generate a valid combo instance")
