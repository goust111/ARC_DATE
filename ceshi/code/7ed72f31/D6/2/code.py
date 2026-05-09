# Variant of 7ed72f31
# Dimension: D6 — 网格形状与输出裁剪 (grid_shape)
# Change: 输出与输入相同尺寸(18-28 x 18-28)，在原网格上直接叠加镜像结果... → 输出仅保留镜像部分

from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np

sys.path.append(str(Path("/home/lixinyu/snap/new/solvers")))
from arc_utils import get_bg, label_mask

# Design analysis for the D6 variant
# ----------------------------------
# a) 具体化与行为追踪
# - 输入仍然是原题风格的大画布：背景色占主导；红色(2)标记给出反射轴/中心；彩色 payload
#   只存在于轴的一侧或中心点的一侧。
# - 输出不再保留输入里的原 sprite 与红标记，而是只保留“新生成”的镜像像素，并把结果裁剪到这些像素
#   的最小外接框。视觉上等于把原题 output 里的“新增那一半”抠出来，形成负像。
# - 例子：若输入中有一条水平红线，绿色 payload 在其下方，原题 output 会显示“上方新镜像 + 原输入整张图”；
#   本变体 output 只显示“上方新镜像”，并去掉整张图周围无关背景。
#
# b) 复杂度校准
# - 仍保留三种原始机制：水平轴、垂直轴、孤立红点点反射。
# - 为了让删除/保留反转足够明显，generator 总是放置至少 1 个水平轴、1 个垂直轴、1 个孤立红点组，
#   且最终 output 会被裁成严格小于输入的尺寸。
# - 不引入新的颜色映射或多阶段逻辑，避免偏离 D6 核心。
#
# c) 原 solver 中与本变体相关的隐式假设审查
# - `out = g.copy()` 依赖“原图要保留”；本变体不成立，必须改为纯背景画布。
# - 两段反射逻辑都从 `g` 读取 payload，再把结果写入 `out`；这一点仍成立，因为本变体只改变输出保留策略，
#   不改变“如何找到镜像像素”。
# - 原 solver 对嵌入轴的行/列组使用 bbox/同色扫描，而不是只遍历组件自身像素。原题这允许同色干扰物被吸入；
#   本变体的 generator 会避免在相关 bbox 内放置同色闲散像素，以保证求解稳定。
# - 孤立红点分支按“与红点 8 邻接的同色 8 连通分量”工作；generator 需保证每个孤立红点只接触 1 个颜色分量，
#   且该红点没有 4 邻接非背景像素。
#
# d) 放置合法性
# - 所有 group 都在一个留有外边距的 active rectangle 内生成；输入 payload、红标记、以及镜像结果都必须落在
#   该区域内，且彼此不重叠。
# - 统一维护 `occupied`，禁止不同 group 的输入或镜像像素互相覆盖。
# - 行/列镜像组只在轴的一侧生成 payload；孤立红点组只在某个对角象限生成 payload，这样点反射后必然落到对面。
#
# e) 视觉可观察性
# - 输入仍能直接看出“红线/红点 + 一侧彩色 sprite”的原始反射语义。
# - 输出只剩另一侧的镜像，并且整体尺寸缩小；2-3 个样例就能看出“并不是把整张图做变换，而是把新生成的那部分裁出来”。
#
# f) 可视化整洁度
# - group 之间不重叠，颜色尽量 distinct，不生成 bbox 内同色噪声。
# - active rectangle 外保留背景边框，使裁剪效果和负像效果都更清楚。
#
# g) 颜色变化可推断性
# - 颜色不变：镜像只是位置变化，不做颜色映射。颜色只用于区分不同 group。
#
# h) 变体后的 generator 校验
# - generator 先构造 full-size 的“仅镜像像素”画布，再裁剪成期望 output。
# - 之后断言 `main(input) == expected_output`，确保 solver 与 generator 一致。
#
# i) 新 solver 的额外假设
# - 至少存在 1 个镜像像素；generator 通过至少 3 个有效 group 保证这一点。
# - 反射结果的总 bbox 严格小于输入网格；generator 通过四周外边距保证这一点。

MARK = 2
_SYS_RNG = random.SystemRandom()


def _crop_non_bg(grid: np.ndarray, bg: int) -> np.ndarray:
    pts = np.argwhere(grid != int(bg))
    if pts.size == 0:
        return np.full((1, 1), int(bg), dtype=int)
    r0, c0 = pts.min(axis=0)
    r1, c1 = pts.max(axis=0)
    return grid[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1]


def main(grid: np.ndarray) -> np.ndarray:
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))
    H, W = g.shape

    lab, n = label_mask(g != int(bg), connectivity=4)
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
                "bbox": (int(min(rs)), int(min(cs)), int(max(rs)), int(max(cs))),
            }
        )

    out = np.full_like(g, int(bg))

    def inb(r: int, c: int) -> bool:
        return 0 <= int(r) < H and 0 <= int(c) < W

    def refl_point(p: tuple[int, int], r: int, c: int) -> tuple[int, int]:
        pr, pc = map(int, p)
        return int(2 * pr - r), int(2 * pc - c)

    def refl_row(r0: int, r: int, c: int) -> tuple[int, int]:
        return int(2 * int(r0) - int(r)), int(c)

    def refl_col(c0: int, r: int, c: int) -> tuple[int, int]:
        return int(r), int(2 * int(c0) - int(c))

    # 1) Components containing red define row/col/point reflection rules.
    for comp in comps:
        coords = comp["coords"]
        vals = comp["vals"]
        if int(MARK) not in set(map(int, vals)):
            continue

        reds = [(int(r), int(c)) for (r, c), v in zip(coords, vals) if int(v) == int(MARK)]
        rs = {int(r) for r, _ in reds}
        cs = {int(c) for _, c in reds}
        if len(reds) == 1:
            axis = ("point", (int(reds[0][0]), int(reds[0][1])))
        elif len(rs) == 1:
            axis = ("row", int(next(iter(rs))))
        elif len(cs) == 1:
            axis = ("col", int(next(iter(cs))))
        else:
            continue

        r0, c0, r1, c1 = map(int, comp["bbox"])
        payload = {int(v) for v in vals if int(v) != int(MARK)}
        if not payload:
            continue

        if axis[0] == "row":

            def eligible(r: int, c: int, v: int) -> bool:
                return int(v) in payload and int(c0) <= int(c) <= int(c1)

        elif axis[0] == "col":

            def eligible(r: int, c: int, v: int) -> bool:
                return int(v) in payload and int(r0) <= int(r) <= int(r1)

        else:
            rr0 = max(0, int(r0 - 1))
            rr1 = min(int(H - 1), int(r1 + 1))
            cc0 = max(0, int(c0 - 1))
            cc1 = min(int(W - 1), int(c1 + 1))

            def eligible(r: int, c: int, v: int) -> bool:
                return int(v) in payload and int(rr0) <= int(r) <= int(rr1) and int(cc0) <= int(c) <= int(cc1)

        for r in range(int(H)):
            for c in range(int(W)):
                v = int(g[int(r), int(c)])
                if v == int(bg) or v == int(MARK):
                    continue
                if not eligible(int(r), int(c), int(v)):
                    continue

                if axis[0] == "row":
                    rr, cc = refl_row(int(axis[1]), int(r), int(c))
                elif axis[0] == "col":
                    rr, cc = refl_col(int(axis[1]), int(r), int(c))
                else:
                    rr, cc = refl_point(axis[1], int(r), int(c))

                if inb(int(rr), int(cc)):
                    out[int(rr), int(cc)] = int(v)

    # 2) Truly isolated red points reflect the adjacent 8-connected color component.
    marker_pts = []
    for comp in comps:
        if len(comp["coords"]) == 1 and int(comp["vals"][0]) == int(MARK):
            marker_pts.append(comp["coords"][0])

    for pr, pc in marker_pts:
        pr = int(pr)
        pc = int(pc)
        neigh_colors: set[int] = set()
        neigh_pixels: list[tuple[int, int, int]] = []
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == 0 and dc == 0:
                    continue
                rr, cc = int(pr + dr), int(pc + dc)
                if not inb(rr, cc):
                    continue
                v = int(g[rr, cc])
                if v == int(bg) or v == int(MARK):
                    continue
                neigh_colors.add(int(v))
                neigh_pixels.append((int(rr), int(cc), int(v)))
        if not neigh_colors:
            continue

        for v in neigh_colors:
            labv, _ = label_mask(g == int(v), connectivity=8)
            labs = {
                int(labv[int(rr), int(cc)])
                for rr, cc, vv in neigh_pixels
                if int(vv) == int(v) and int(labv[int(rr), int(cc)]) != 0
            }
            for lbl in labs:
                for r, c in np.argwhere(labv == int(lbl)):
                    rr, cc = refl_point((int(pr), int(pc)), int(r), int(c))
                    if inb(rr, cc):
                        out[int(rr), int(cc)] = int(v)

    return _crop_non_bg(out, int(bg))


def _neighbors(cell: tuple[int, int], connectivity: int) -> list[tuple[int, int]]:
    r, c = cell
    if int(connectivity) == 4:
        return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]
    return [
        (r + dr, c + dc)
        for dr in (-1, 0, 1)
        for dc in (-1, 0, 1)
        if not (dr == 0 and dc == 0)
    ]


def _grow_shape(
    rng: random.Random,
    allowed: set[tuple[int, int]],
    size: int,
    connectivity: int,
    start: tuple[int, int],
) -> list[tuple[int, int]] | None:
    if start not in allowed:
        return None
    shape = {start}
    while len(shape) < int(size):
        frontier = set()
        for cell in shape:
            for nb in _neighbors(cell, int(connectivity)):
                if nb in allowed and nb not in shape:
                    frontier.add(nb)
        if not frontier:
            return None
        shape.add(rng.choice(sorted(frontier)))
    return sorted(shape)


def _make_row_group(rng: random.Random, color: int) -> dict:
    line_len = rng.randint(3, 6)
    depth = rng.randint(2, 4)
    side = rng.choice([-1, 1])
    allowed = {(side * dr, dc) for dr in range(1, depth + 1) for dc in range(line_len)}
    size = rng.randint(3, min(8, len(allowed)))
    start = (side, rng.randrange(line_len))
    payload = _grow_shape(rng, allowed, size, connectivity=4, start=start)
    if payload is None:
        raise ValueError("row payload generation failed")
    red = [(0, dc) for dc in range(line_len)]
    mirrored = [(-r, c) for r, c in payload]
    return {
        "kind": "row",
        "input_cells": [(r, c, int(color)) for r, c in payload] + [(r, c, MARK) for r, c in red],
        "output_cells": [(r, c, int(color)) for r, c in mirrored],
    }


def _make_col_group(rng: random.Random, color: int) -> dict:
    line_len = rng.randint(3, 6)
    depth = rng.randint(2, 4)
    side = rng.choice([-1, 1])
    allowed = {(dr, side * dc) for dr in range(line_len) for dc in range(1, depth + 1)}
    size = rng.randint(3, min(8, len(allowed)))
    start = (rng.randrange(line_len), side)
    payload = _grow_shape(rng, allowed, size, connectivity=4, start=start)
    if payload is None:
        raise ValueError("col payload generation failed")
    red = [(dr, 0) for dr in range(line_len)]
    mirrored = [(r, -c) for r, c in payload]
    return {
        "kind": "col",
        "input_cells": [(r, c, int(color)) for r, c in payload] + [(r, c, MARK) for r, c in red],
        "output_cells": [(r, c, int(color)) for r, c in mirrored],
    }


def _make_isolated_group(rng: random.Random, color: int) -> dict:
    sr, sc = rng.choice([(-1, -1), (-1, 1), (1, -1), (1, 1)])
    h = rng.randint(2, 4)
    w = rng.randint(2, 4)
    allowed = {(sr * dr, sc * dc) for dr in range(1, h + 1) for dc in range(1, w + 1)}
    size = rng.randint(1, min(6, len(allowed)))
    start = (sr, sc)
    payload = _grow_shape(rng, allowed, size, connectivity=8, start=start)
    if payload is None:
        raise ValueError("isolated payload generation failed")
    mirrored = [(-r, -c) for r, c in payload]
    return {
        "kind": "isolated",
        "input_cells": [(r, c, int(color)) for r, c in payload] + [(0, 0, MARK)],
        "output_cells": [(r, c, int(color)) for r, c in mirrored],
    }


def _translate_cells(cells: list[tuple[int, int, int]], dr: int, dc: int) -> list[tuple[int, int, int]]:
    return [(int(r + dr), int(c + dc), int(v)) for r, c, v in cells]


def _coords_bbox(cells: list[tuple[int, int, int]]) -> tuple[int, int, int, int]:
    rs = [r for r, _, _ in cells]
    cs = [c for _, c, _ in cells]
    return int(min(rs)), int(min(cs)), int(max(rs)), int(max(cs))


def _has_buffer_conflict(coords: set[tuple[int, int]], occupied: set[tuple[int, int]]) -> bool:
    for r, c in coords:
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if (int(r + dr), int(c + dc)) in occupied:
                    return True
    return False


def _place_group(
    rng: random.Random,
    group: dict,
    occupied: set[tuple[int, int]],
    active_rect: tuple[int, int, int, int],
) -> tuple[list[tuple[int, int, int]], list[tuple[int, int, int]]] | None:
    all_cells = list(group["input_cells"]) + list(group["output_cells"])
    r0, c0, r1, c1 = _coords_bbox(all_cells)
    ar0, ac0, ar1, ac1 = active_rect
    row_lo = int(ar0 - r0)
    row_hi = int(ar1 - r1)
    col_lo = int(ac0 - c0)
    col_hi = int(ac1 - c1)
    if row_lo > row_hi or col_lo > col_hi:
        return None

    for _ in range(150):
        dr = rng.randint(row_lo, row_hi)
        dc = rng.randint(col_lo, col_hi)
        inp = _translate_cells(group["input_cells"], dr, dc)
        out = _translate_cells(group["output_cells"], dr, dc)
        coords = {(r, c) for r, c, _ in inp + out}
        if coords & occupied or _has_buffer_conflict(coords, occupied):
            continue
        return inp, out
    return None


def _build_example(rng: random.Random) -> tuple[np.ndarray, np.ndarray, dict]:
    H = rng.randint(18, 28)
    W = rng.randint(18, 28)
    bg = rng.choice([0, 1, 3, 4, 5, 6, 7, 8, 9])
    palette = [c for c in range(10) if c not in (bg, MARK)]
    rng.shuffle(palette)

    margin_top = rng.randint(2, 4)
    margin_bottom = rng.randint(2, 4)
    margin_left = rng.randint(2, 4)
    margin_right = rng.randint(2, 4)
    active_rect = (
        int(margin_top),
        int(margin_left),
        int(H - 1 - margin_bottom),
        int(W - 1 - margin_right),
    )
    ar0, ac0, ar1, ac1 = active_rect
    if ar1 - ar0 + 1 < 8 or ac1 - ac0 + 1 < 8:
        raise ValueError("active area too small")

    inp = np.full((H, W), int(bg), dtype=int)
    full_out = np.full((H, W), int(bg), dtype=int)
    occupied: set[tuple[int, int]] = set()

    kinds = ["row", "col", "isolated"]
    extra_count = rng.randint(0, 2)
    for _ in range(extra_count):
        kinds.append(rng.choice(["row", "col", "isolated"]))
    rng.shuffle(kinds)

    builders = {"row": _make_row_group, "col": _make_col_group, "isolated": _make_isolated_group}
    placed_kinds: list[str] = []
    for idx, kind in enumerate(kinds):
        color = int(palette[idx])
        placed = None
        for _ in range(80):
            group = builders[kind](rng, color)
            placed = _place_group(rng, group, occupied, active_rect)
            if placed is not None:
                break
        if placed is None:
            raise ValueError(f"failed to place {kind} group")
        inp_cells, out_cells = placed
        for r, c, v in inp_cells:
            inp[int(r), int(c)] = int(v)
            occupied.add((int(r), int(c)))
        for r, c, v in out_cells:
            full_out[int(r), int(c)] = int(v)
            occupied.add((int(r), int(c)))
        placed_kinds.append(kind)

    expected = _crop_non_bg(full_out, int(bg))
    got = main(inp)
    if not np.array_equal(got, expected):
        raise AssertionError("solver/generator mismatch")
    if expected.shape[0] >= H or expected.shape[1] >= W:
        raise AssertionError("output was not cropped smaller than input")
    if np.any(expected == MARK):
        raise AssertionError("output leaked red markers")
    return inp, expected, {"kinds": placed_kinds, "input_shape": (H, W), "output_shape": expected.shape}


def generate_input() -> np.ndarray:
    for _ in range(400):
        rng = random.Random(_SYS_RNG.randrange(1 << 63))
        try:
            inp, _expected, _meta = _build_example(rng)
            return inp
        except Exception:
            continue
    raise RuntimeError("failed to generate a valid D6 variant instance")
