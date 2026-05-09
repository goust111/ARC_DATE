# Variant of 7ed72f31
# Dimension: D1 — 反射变换类型 (transform)
# Change: 根据红色标记几何形状选择镜像方式：红线→轴反射，单红像素→点反射... → 旋转替代反射

from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np

_ARC_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_ARC_UTILS_DIR) not in sys.path:
    sys.path.append(str(_ARC_UTILS_DIR))

from arc_utils import get_bg, label_mask


"""
design_analysis:
  variant_core:
    original: 红色水平/竖直线作为反射轴，把单色 payload 镜像到轴另一侧；单红点做 180 度点反射。
    new: 红色线不再作为反射轴，而是只提供一个旋转中心。水平线和竖线都让 payload 绕红线中点顺时针旋转 90 度；
      单红点仍做 180 度旋转。这样“线的几何形状决定变换类别”的外观保留，但真正要推断的是旋转而不是镜像。
  concrete_visualization:
    horizontal_line:
      input: 红线长 3 或 5，payload 全在红线上方或下方，并与红线 4 连通。
      output: 原 payload 保留，新增拷贝出现在红线右侧或左侧，而不是上下镜像到对面。
    vertical_line:
      input: 红线长 3 或 5，payload 全在左侧或右侧，并与红线 4 连通。
      output: 新增拷贝出现在上方或下方，而不是左右镜像到对面。
    isolated_red_point:
      input: 单红点与一个单色 8 连通小精灵对角相邻，但没有 4 连通接触。
      output: 该小精灵绕红点 180 度旋转到对侧。
  behavior_trace:
    example:
      input:
        - 水平红线中心在 (10,10)，长度 3，payload 像素有 (9,10),(8,10),(8,11)
      original_solver:
        - 按行反射到 (11,10),(12,10),(12,11)，结果仍在红线下方
      variant_solver:
        - 按 90 度顺时针旋转到 (10,11),(10,12),(11,12)，结果改为出现在红线右侧
      difference:
        - 复制侧从“轴对侧”变成“正交侧”，因此同一输入下原任务与变体输出明显不同
  complexity_calibration:
    - 保留三类 marker：水平线、竖线、孤立红点；避免把问题简化成单一情形。
    - 线类统一为 90 度顺时针，点类统一为 180 度；离散档位少，训练样例中容易重复支撑。
    - payload 强制为不对称连通形，避免 90 度旋转与镜像在视觉上近似或不可见。
  implicit_assumption_audit:
    - 原 solver 只需行/列坐标作为反射轴；新 solver 需要真实旋转中心，因此红线必须有可落在整数格点上的中点。
    - 对于 90 度格点旋转，中心的行列坐标必须同为整数或同为半整数。水平/竖直红线若长度为偶数，则中点是一整一半，旋转后会落在半格。
    - 因此 generator 必须把红线长度限制为奇数；solver 也把偶数长度红线视为不合法并跳过，而不是错误旋转。
    - 原 solver 的 bbox/邻域筛选仍然可用，因为变体只改几何变换，不改“哪些像素参与”。
  placement_legality:
    - 放置时预留的是最终 output 占用区域，而不是仅 input 占用区域，避免不同 group 的旋转副本相撞。
    - 每个 group 之间留 1 格 padding，减轻视觉拥挤并避免 solver 的局部 bbox 意外捕获别组像素。
    - 孤立红点 group 把 payload 限制在单一对角象限，使其与红点仅 8 邻接、非 4 邻接。
  observability:
    - 每个样本至少包含 1 个水平线 group、1 个竖线 group、1 个孤立点 group。
    - 线类 group 数量通常多于点类，保证“90 度旋转替代镜像”是主导现象。
    - payload 使用不同颜色，减少跨组混淆。
  cleanliness:
    - 不生成原题中可选的“近 bbox 干扰点”，因为那会在旋转版本里放大歧义且不属于 D1 核心变化。
    - 通过 reserve-output + padding 的放置策略，让每组在视觉上成为清晰分离的小局部结构。
  new_solver_assumptions:
    - 红线是连续的单行/单列奇数长度线段。
    - 混合组件中的非红 payload 颜色单一；generator 以唯一颜色保证这一点。
    - 孤立红点 8 邻域里只接触一个颜色组件；generator 通过单象限单色构造保证。
"""


MARK = 2


def _rot90_cw_center2(center2: tuple[int, int], r: int, c: int) -> tuple[int, int] | None:
    cr2, cc2 = map(int, center2)
    rr2 = cr2 + 2 * int(c) - cc2
    cc2_new = cc2 - 2 * int(r) + cr2
    if rr2 % 2 != 0 or cc2_new % 2 != 0:
        return None
    return rr2 // 2, cc2_new // 2


def _rot180_point(p: tuple[int, int], r: int, c: int) -> tuple[int, int]:
    pr, pc = map(int, p)
    return 2 * pr - int(r), 2 * pc - int(c)


def main(grid: np.ndarray) -> np.ndarray:
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))
    H, W = g.shape

    lab, n = label_mask(g != bg, connectivity=4)
    comps = []
    for k in range(1, int(n) + 1):
        coords = [(int(r), int(c)) for r, c in np.argwhere(lab == int(k))]
        if not coords:
            continue
        vals = [int(g[r, c]) for r, c in coords]
        rs = [r for r, _ in coords]
        cs = [c for _, c in coords]
        comps.append(
            {
                "coords": coords,
                "vals": vals,
                "bbox": (min(rs), min(cs), max(rs), max(cs)),
            }
        )

    out = g.copy()

    def inb(r: int, c: int) -> bool:
        return 0 <= int(r) < H and 0 <= int(c) < W

    for comp in comps:
        coords = comp["coords"]
        vals = comp["vals"]
        if MARK not in vals:
            continue

        reds = [(r, c) for (r, c), v in zip(coords, vals) if int(v) == MARK]
        rs = {r for r, _ in reds}
        cs = {c for _, c in reds}

        axis: tuple[str, tuple[int, int] | int] | None = None
        if len(reds) == 1:
            axis = ("point", reds[0])
        elif len(rs) == 1:
            row = next(iter(rs))
            red_cols = sorted(c for _, c in reds)
            if red_cols == list(range(red_cols[0], red_cols[-1] + 1)) and len(red_cols) % 2 == 1:
                axis = ("row", (2 * int(row), int(red_cols[0] + red_cols[-1])))
        elif len(cs) == 1:
            col = next(iter(cs))
            red_rows = sorted(r for r, _ in reds)
            if red_rows == list(range(red_rows[0], red_rows[-1] + 1)) and len(red_rows) % 2 == 1:
                axis = ("col", (int(red_rows[0] + red_rows[-1]), 2 * int(col)))
        if axis is None:
            continue

        payload = {int(v) for v in vals if int(v) != MARK}
        if not payload:
            continue

        r0, c0, r1, c1 = map(int, comp["bbox"])
        if axis[0] == "row":

            def eligible(r: int, c: int, v: int) -> bool:
                return int(v) in payload and c0 <= int(c) <= c1

        elif axis[0] == "col":

            def eligible(r: int, c: int, v: int) -> bool:
                return int(v) in payload and r0 <= int(r) <= r1

        else:
            rr0 = max(0, r0 - 1)
            rr1 = min(H - 1, r1 + 1)
            cc0 = max(0, c0 - 1)
            cc1 = min(W - 1, c1 + 1)

            def eligible(r: int, c: int, v: int) -> bool:
                return int(v) in payload and rr0 <= int(r) <= rr1 and cc0 <= int(c) <= cc1

        for r in range(H):
            for c in range(W):
                v = int(g[r, c])
                if v == bg or v == MARK or not eligible(r, c, v):
                    continue

                if axis[0] == "point":
                    dst = _rot180_point(axis[1], r, c)
                else:
                    dst = _rot90_cw_center2(axis[1], r, c)
                    if dst is None:
                        continue
                rr, cc = dst
                if inb(rr, cc):
                    out[rr, cc] = v

    marker_pts = [comp["coords"][0] for comp in comps if len(comp["coords"]) == 1 and int(comp["vals"][0]) == MARK]
    for pr, pc in marker_pts:
        neigh_colors: set[int] = set()
        neigh_pixels: list[tuple[int, int, int]] = []
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == 0 and dc == 0:
                    continue
                rr, cc = pr + dr, pc + dc
                if not inb(rr, cc):
                    continue
                v = int(g[rr, cc])
                if v == bg or v == MARK:
                    continue
                neigh_colors.add(v)
                neigh_pixels.append((rr, cc, v))
        for v in neigh_colors:
            labv, _ = label_mask(g == int(v), connectivity=8)
            labs = {
                int(labv[rr, cc])
                for rr, cc, vv in neigh_pixels
                if int(vv) == int(v) and int(labv[rr, cc]) != 0
            }
            for lbl in labs:
                for r, c in np.argwhere(labv == int(lbl)):
                    rr, cc = _rot180_point((pr, pc), int(r), int(c))
                    if inb(rr, cc):
                        out[rr, cc] = int(v)

    return out


def _neighbors4(p: tuple[int, int]) -> list[tuple[int, int]]:
    r, c = p
    return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


def _neighbors8(p: tuple[int, int]) -> list[tuple[int, int]]:
    r, c = p
    return [
        (r + dr, c + dc)
        for dr in (-1, 0, 1)
        for dc in (-1, 0, 1)
        if not (dr == 0 and dc == 0)
    ]


def _shape_array(cells: set[tuple[int, int]]) -> np.ndarray:
    rs = [r for r, _ in cells]
    cs = [c for _, c in cells]
    r0, c0 = min(rs), min(cs)
    h = max(rs) - r0 + 1
    w = max(cs) - c0 + 1
    arr = np.zeros((h, w), dtype=int)
    for r, c in cells:
        arr[r - r0, c - c0] = 1
    return arr


def _is_interesting_line_shape(cells: set[tuple[int, int]]) -> bool:
    if len(cells) < 4:
        return False
    arr = _shape_array(cells)
    if arr.shape[0] < 2 or arr.shape[1] < 2:
        return False
    variants = [
        np.flipud(arr),
        np.fliplr(arr),
        np.rot90(arr, 1),
        np.rot90(arr, 2),
        np.rot90(arr, 3),
    ]
    for var in variants:
        if arr.shape == var.shape and np.array_equal(arr, var):
            return False
    return True


def _is_interesting_isolated_shape(cells: set[tuple[int, int]]) -> bool:
    if len(cells) < 3:
        return False
    arr = _shape_array(cells)
    return arr.shape[0] >= 2 and arr.shape[1] >= 2


def _grow_connected(
    allowed: set[tuple[int, int]],
    seed: tuple[int, int],
    target_size: int,
    connectivity: int,
) -> set[tuple[int, int]] | None:
    cells = {seed}
    while len(cells) < int(target_size):
        frontier = set()
        for cell in cells:
            nbrs = _neighbors4(cell) if int(connectivity) == 4 else _neighbors8(cell)
            for nxt in nbrs:
                if nxt in allowed and nxt not in cells:
                    frontier.add(nxt)
        if not frontier:
            return None
        cells.add(random.choice(tuple(frontier)))
    return cells


def _build_line_group(kind: str, color: int) -> dict | None:
    for _ in range(200):
        line_len = random.choice([3, 5])
        half = line_len // 2
        size = random.randint(4, 7)
        if kind == "row":
            side = random.choice(["above", "below"])
            if side == "above":
                allowed = {(r, c) for r in range(-4, 0) for c in range(-half - 2, half + 3)}
                seed = (-1, random.randint(-half, half))
            else:
                allowed = {(r, c) for r in range(1, 5) for c in range(-half - 2, half + 3)}
                seed = (1, random.randint(-half, half))
            reds = {(0, c) for c in range(-half, half + 1)}
        else:
            side = random.choice(["left", "right"])
            if side == "left":
                allowed = {(r, c) for r in range(-half - 2, half + 3) for c in range(-4, 0)}
                seed = (random.randint(-half, half), -1)
            else:
                allowed = {(r, c) for r in range(-half - 2, half + 3) for c in range(1, 5)}
                seed = (random.randint(-half, half), 1)
            reds = {(r, 0) for r in range(-half, half + 1)}

        payload = _grow_connected(allowed, seed, size, connectivity=4)
        if payload is None or not _is_interesting_line_shape(payload):
            continue

        rotated = {(c, -r) for r, c in payload}
        if payload & reds or rotated & reds or rotated & payload:
            continue

        all_cells = [(r, c, MARK) for r, c in reds] + [(r, c, color) for r, c in payload]
        out_extra = [(r, c, color) for r, c in rotated]
        return {"kind": kind, "input_cells": all_cells, "output_extra": out_extra}
    return None


def _build_isolated_group(color: int) -> dict | None:
    quadrants = {
        "tl": ({(r, c) for r in range(-4, 0) for c in range(-4, 0)}, (-1, -1)),
        "tr": ({(r, c) for r in range(-4, 0) for c in range(1, 5)}, (-1, 1)),
        "bl": ({(r, c) for r in range(1, 5) for c in range(-4, 0)}, (1, -1)),
        "br": ({(r, c) for r in range(1, 5) for c in range(1, 5)}, (1, 1)),
    }
    for _ in range(200):
        allowed, seed = random.choice(list(quadrants.values()))
        size = random.randint(3, 6)
        payload = _grow_connected(allowed, seed, size, connectivity=8)
        if payload is None or not _is_interesting_isolated_shape(payload):
            continue
        rotated = {(-r, -c) for r, c in payload}
        if (0, 0) in payload or (0, 0) in rotated or payload & rotated:
            continue
        all_cells = [(0, 0, MARK)] + [(r, c, color) for r, c in payload]
        out_extra = [(r, c, color) for r, c in rotated]
        return {"kind": "isolated", "input_cells": all_cells, "output_extra": out_extra}
    return None


def _candidate_bbox(cells: list[tuple[int, int, int]]) -> tuple[int, int, int, int]:
    rs = [r for r, _, _ in cells]
    cs = [c for _, c, _ in cells]
    return min(rs), min(cs), max(rs), max(cs)


def _translate_cells(cells: list[tuple[int, int, int]], dr: int, dc: int) -> list[tuple[int, int, int]]:
    return [(r + int(dr), c + int(dc), v) for r, c, v in cells]


def _place_candidate(
    H: int,
    W: int,
    blocked: np.ndarray,
    cand: dict,
) -> tuple[list[tuple[int, int, int]], list[tuple[int, int, int]]] | None:
    all_output = list(cand["input_cells"]) + list(cand["output_extra"])
    r0, c0, r1, c1 = _candidate_bbox(all_output)
    row_lo = -r0
    row_hi = H - 1 - r1
    col_lo = -c0
    col_hi = W - 1 - c1
    if row_lo > row_hi or col_lo > col_hi:
        return None

    anchors = [(r, c) for r in range(row_lo, row_hi + 1) for c in range(col_lo, col_hi + 1)]
    random.shuffle(anchors)
    for ar, ac in anchors:
        placed_output = _translate_cells(all_output, ar, ac)
        if any(blocked[r, c] for r, c, _ in placed_output):
            continue
        return _translate_cells(cand["input_cells"], ar, ac), _translate_cells(cand["output_extra"], ar, ac)
    return None


def generate_input() -> np.ndarray:
    colors_all = list(range(10))
    colors_all.remove(MARK)

    for _ in range(200):
        H = random.randint(20, 26)
        W = random.randint(20, 26)
        bg = random.choice(colors_all)
        payload_colors = [c for c in colors_all if c != bg]
        random.shuffle(payload_colors)

        kinds = ["row", "col", "isolated"]
        extra_count = random.randint(1, 3)
        extra_pool = ["row", "col", "isolated", "row", "col"]
        for _j in range(extra_count):
            kinds.append(random.choice(extra_pool))
        random.shuffle(kinds)
        if len(payload_colors) < len(kinds):
            continue

        blocked = np.zeros((H, W), dtype=bool)
        inp = np.full((H, W), bg, dtype=int)
        expected = np.full((H, W), bg, dtype=int)
        ok = True

        for idx, kind in enumerate(kinds):
            color = payload_colors[idx]
            cand = _build_line_group(kind, color) if kind in {"row", "col"} else _build_isolated_group(color)
            if cand is None:
                ok = False
                break
            placed = _place_candidate(H, W, blocked, cand)
            if placed is None:
                ok = False
                break
            input_cells, output_extra = placed
            all_output = input_cells + output_extra
            rr0, cc0, rr1, cc1 = _candidate_bbox(all_output)
            pad_r0 = max(0, rr0 - 1)
            pad_c0 = max(0, cc0 - 1)
            pad_r1 = min(H - 1, rr1 + 1)
            pad_c1 = min(W - 1, cc1 + 1)
            blocked[pad_r0 : pad_r1 + 1, pad_c0 : pad_c1 + 1] = True

            for r, c, v in input_cells:
                inp[r, c] = int(v)
                expected[r, c] = int(v)
            for r, c, v in output_extra:
                expected[r, c] = int(v)

        if not ok:
            continue

        solved = main(inp)
        if not np.array_equal(solved, expected):
            continue
        return inp

    raise RuntimeError("failed to generate a valid rotation-variant sample")
