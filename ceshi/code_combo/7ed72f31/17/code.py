# Combo variant of 7ed72f31 (index 17)
# combo_spec: D6/1+D8/2
# Name: 迭代扩展反射

from __future__ import annotations

import random
import sys
from collections import deque
from pathlib import Path

import numpy as np

sys.path.append(str(Path("/home/lixinyu/snap/new/solvers")))
from arc_utils import get_bg


# Design analysis for the fused variant
# -------------------------------------
# a) 具体化与行为追踪
# - 原题：每个红标记组只读静态输入中的原始 payload，一次性把镜像结果叠加到同尺寸输出；越界结果直接丢弃。
# - 融合后：把红标记视为可反复使用的反射器。每一轮都用“上一轮结束时的整张当前画布”做输入，所有本轮新像素
#   到下一轮才允许继续参与反射；同时画布不再固定在输入尺寸内，而是放在整数坐标平面上，最后再把原输入矩形
#   向外扩展到刚好包住所有新增像素。
# - 具体小例子：
#   1. 输入左上角附近有一个红十字：水平轴 r=1、垂直轴 c=1；一个绿色 L 形 seed 只在右下象限。
#   2. 原 solver 若只做单轮：会得到“上方一份”和“左侧一份”，但左上角那份需要经过两次反射，原逻辑不会出现。
#   3. 融合 solver：
#      - round 1: seed 经水平轴得到上方副本，经垂直轴得到左侧副本。
#      - round 2: 上方副本再经垂直轴得到左上角副本；左侧副本再经水平轴得到同一批像素。
#      - round 3: 不再有新像素，收敛。
#      - 若左上角副本坐标为负，则输出在上/左自动补背景边框，把它纳入更大的网格。
# - 差异点不只是“可越界”，而是“越界的二轮镜像也会继续参与后续轮次”，两项改动必须一起出现才有完整效果。
#
# b) 复杂度校准
# - 若只有单一反射轴或单一中心，迭代通常在 1 轮后就稳定，视觉上容易退化回原题。
# - 因此 generator 固定生成一个红十字系统：一条水平线 + 一条垂直线 + 一个只在单象限出现的非对称 seed。
#   这样最少需要 2 轮才能补全四象限，且把十字放在输入边缘附近，就会稳定触发 D6 的扩展行为。
# - 不再加入自由散点或同色噪声，否则“全轴反射 + 迭代传播”会把无关像素也复制出去，视觉上会变脏。
#
# c) 原 main() 隐式假设审查
# - 原代码 `out = g.copy()` 假设输出尺寸固定；融合后失效，必须改为无限稀疏坐标。
# - 原代码靠“含红混合 4 连通分量的 bbox”决定 payload 可见范围；一旦二轮像素离开原 bbox，这个前提就失效。
#   因此新 solver 改为直接提取红色直线 run，并把它们解释为整条反射轴。
# - 原代码从静态 `g` 扫描 eligible 像素；融合后必须改为每轮读当前 payload snapshot，否则新像素无法继续传播。
# - 原代码只需 `inb()` 检查；融合后越界不再是错误，而是输出边界的一部分。
#
# d) 放置合法性
# - generator 只放置一个十字系统，避免多轴之间相互干扰造成无限平移链。
# - 红十字始终完整落在输入中；seed 只放在一个象限且与红线相邻，输入合法、可读。
# - 由于只使用一对相交的正交反射轴，闭包是有限的四象限补全，不会无限生长。
#
# e) 视觉可观察性
# - 输入里只看到一个象限的彩色 sprite；输出里出现四象限完成图案，且画布向靠边的方向增大。
# - 人类从 2-3 个样例就能看出：
#   1. 红水平线/垂直线还是反射轴；
#   2. 新像素不是一轮完成，而是继续被另一条轴使用；
#   3. 输出尺寸比输入大，是为了容纳越界镜像。
#
# f) 可视化整洁度
# - 不生成第二个系统、不生成无关同色碎片，避免“全轴规则”把杂物扩散到整图。
# - seed 强制非对称，保证四象限补全在视觉上清楚可辨，不会被误读为简单复制。
#
# g) 新 solver 的额外假设及其对应生成保证
# - 假设红标记可被解析成水平/垂直 straight run。generator 始终使用规范十字。
# - 假设迭代闭包有限。generator 只使用一组相交正交轴，不使用平行双轴或双中心。
# - 假设不同颜色不会在同一目标格冲突。generator 只放一个单色 seed。


MARK = 2
_RNG = random.Random()


def _red_runs(g: np.ndarray) -> list[dict]:
    h, w = g.shape
    rows: list[dict] = []
    cols: list[dict] = []
    row_cells: set[tuple[int, int]] = set()
    col_cells: set[tuple[int, int]] = set()

    for r in range(h):
        c = 0
        while c < w:
            if int(g[r, c]) != int(MARK):
                c += 1
                continue
            c0 = c
            while c < w and int(g[r, c]) == int(MARK):
                c += 1
            c1 = c - 1
            if c1 - c0 + 1 >= 2:
                rows.append({"kind": "row", "row": int(r), "c0": int(c0), "c1": int(c1)})
                for cc in range(c0, c1 + 1):
                    row_cells.add((int(r), int(cc)))

    for c in range(w):
        r = 0
        while r < h:
            if int(g[r, c]) != int(MARK):
                r += 1
                continue
            r0 = r
            while r < h and int(g[r, c]) == int(MARK):
                r += 1
            r1 = r - 1
            if r1 - r0 + 1 >= 2:
                cols.append({"kind": "col", "col": int(c), "r0": int(r0), "r1": int(r1)})
                for rr in range(r0, r1 + 1):
                    col_cells.add((int(rr), int(c)))

    point_markers: list[dict] = []
    for r in range(h):
        for c in range(w):
            if int(g[r, c]) != int(MARK):
                continue
            pt = (int(r), int(c))
            if pt in row_cells or pt in col_cells:
                continue
            point_markers.append({"kind": "point", "center": pt})

    markers = rows + cols + point_markers
    markers.sort(
        key=lambda m: (
            0 if m["kind"] == "row" else 1 if m["kind"] == "col" else 2,
            int(m.get("row", m.get("center", (0, 0))[0])),
            int(m.get("col", m.get("center", (0, 0))[1])),
        )
    )
    return markers


def _neighbors8(pt: tuple[int, int]) -> list[tuple[int, int]]:
    r, c = pt
    return [
        (int(r + dr), int(c + dc))
        for dr in (-1, 0, 1)
        for dc in (-1, 0, 1)
        if not (dr == 0 and dc == 0)
    ]


def _component8_from(
    payload: dict[tuple[int, int], int],
    start: tuple[int, int],
) -> set[tuple[int, int]]:
    color = int(payload[start])
    q = deque([start])
    seen = {start}
    while q:
        r, c = q.popleft()
        for rr, cc in _neighbors8((int(r), int(c))):
            nb = (int(rr), int(cc))
            if nb in seen:
                continue
            if int(payload.get(nb, -1)) != int(color):
                continue
            seen.add(nb)
            q.append(nb)
    return seen


def _point_reflections(
    payload: dict[tuple[int, int], int],
    center: tuple[int, int],
) -> dict[tuple[int, int], int]:
    pr, pc = map(int, center)
    touched = [pt for pt in _neighbors8((int(pr), int(pc))) if pt in payload]
    if not touched:
        return {}

    out: dict[tuple[int, int], int] = {}
    seen_components: set[frozenset[tuple[int, int]]] = set()
    for start in touched:
        comp = _component8_from(payload, start)
        key = frozenset(comp)
        if key in seen_components:
            continue
        seen_components.add(key)
        color = int(payload[start])
        for r, c in comp:
            rr = int(2 * pr - r)
            cc = int(2 * pc - c)
            out[(int(rr), int(cc))] = int(color)
    return out


def _simulate(
    grid: np.ndarray,
    *,
    max_rounds: int | None = None,
) -> tuple[dict[tuple[int, int], int], int, list[int], int]:
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))
    h, w = g.shape
    markers = _red_runs(g)

    canvas: dict[tuple[int, int], int] = {}
    payload: dict[tuple[int, int], int] = {}
    for r in range(h):
        for c in range(w):
            v = int(g[r, c])
            if v == int(bg):
                continue
            pt = (int(r), int(c))
            canvas[pt] = int(v)
            if v != int(MARK):
                payload[pt] = int(v)

    if not markers or not payload:
        return canvas, int(bg), [0], 0

    added_per_round: list[int] = []
    rounds = 0
    limit = int(max_rounds) if max_rounds is not None else 32

    while rounds < limit:
        snapshot = dict(payload)
        proposals: dict[tuple[int, int], int] = {}

        for marker in markers:
            kind = marker["kind"]
            if kind == "row":
                row = int(marker["row"])
                for (r, c), color in snapshot.items():
                    if int(r) == int(row):
                        continue
                    tgt = (int(2 * row - r), int(c))
                    if tgt not in proposals:
                        proposals[tgt] = int(color)
            elif kind == "col":
                col = int(marker["col"])
                for (r, c), color in snapshot.items():
                    if int(c) == int(col):
                        continue
                    tgt = (int(r), int(2 * col - c))
                    if tgt not in proposals:
                        proposals[tgt] = int(color)
            else:
                for tgt, color in _point_reflections(snapshot, marker["center"]).items():
                    if tgt not in proposals:
                        proposals[tgt] = int(color)

        new_pixels: dict[tuple[int, int], int] = {}
        for pt, color in proposals.items():
            cur = canvas.get(pt)
            if cur is None:
                new_pixels[pt] = int(color)
            elif int(cur) == int(bg):
                new_pixels[pt] = int(color)

        added_per_round.append(len(new_pixels))
        if not new_pixels:
            break

        for pt, color in new_pixels.items():
            canvas[pt] = int(color)
            payload[pt] = int(color)
        rounds += 1

        if max_rounds is not None and rounds >= int(max_rounds):
            break

    return canvas, int(bg), added_per_round, rounds


def _render(
    canvas: dict[tuple[int, int], int],
    bg: int,
    input_shape: tuple[int, int],
) -> np.ndarray:
    h, w = map(int, input_shape)
    if not canvas:
        return np.full((h, w), int(bg), dtype=int)

    rs = [r for r, _c in canvas]
    cs = [c for _r, c in canvas]
    min_r = min(0, min(rs))
    min_c = min(0, min(cs))
    max_r = max(h - 1, max(rs))
    max_c = max(w - 1, max(cs))

    out = np.full((int(max_r - min_r + 1), int(max_c - min_c + 1)), int(bg), dtype=int)
    for (r, c), v in canvas.items():
        out[int(r - min_r), int(c - min_c)] = int(v)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    canvas, bg, _added, _rounds = _simulate(np.array(grid, dtype=int))
    return _render(canvas, int(bg), tuple(np.array(grid, dtype=int).shape))


def _norm_shape(cells: set[tuple[int, int]]) -> set[tuple[int, int]]:
    min_r = min(r for r, _c in cells)
    min_c = min(c for _r, c in cells)
    return {(int(r - min_r), int(c - min_c)) for r, c in cells}


def _random_seed_shape(rng: random.Random) -> set[tuple[int, int]]:
    for _ in range(400):
        hh = rng.randint(3, 4)
        ww = rng.randint(3, 4)
        target = rng.randint(4, min(7, hh * ww - 1))
        start = (rng.randint(0, hh - 1), rng.randint(0, ww - 1))
        cells = {start}
        while len(cells) < target:
            r, c = rng.choice(tuple(cells))
            nbrs = []
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                rr, cc = int(r + dr), int(c + dc)
                if 0 <= rr < hh and 0 <= cc < ww:
                    nbrs.append((rr, cc))
            rng.shuffle(nbrs)
            for nb in nbrs:
                if nb not in cells:
                    cells.add(nb)
                    break
            else:
                break

        cells = _norm_shape(cells)
        max_r = max(r for r, _c in cells)
        max_c = max(c for _r, c in cells)
        if max_r < 2 or max_c < 2:
            continue

        row_m = _norm_shape({(max_r - r, c) for r, c in cells})
        col_m = _norm_shape({(r, max_c - c) for r, c in cells})
        pt_m = _norm_shape({(max_r - r, max_c - c) for r, c in cells})
        if cells == row_m or cells == col_m or cells == pt_m:
            continue

        return {(int(r + 1), int(c + 1)) for r, c in cells}
    raise RuntimeError("failed to generate asymmetric seed")


def _oriented_system(
    seed: set[tuple[int, int]],
    corner: str,
    color: int,
) -> tuple[set[tuple[int, int, int]], set[tuple[int, int, int]]]:
    sgn_r = 1 if corner in {"tl", "tr"} else -1
    sgn_c = 1 if corner in {"tl", "bl"} else -1
    max_seed_r = max(r for r, _c in seed)
    max_seed_c = max(c for _r, c in seed)
    arm_up = 1
    arm_left = 1
    arm_down = max(2, max_seed_r + 1)
    arm_right = max(2, max_seed_c + 1)

    red: set[tuple[int, int, int]] = set()
    for c in range(-arm_left, arm_right + 1):
        red.add((0, int(c), int(MARK)))
    for r in range(-arm_up, arm_down + 1):
        red.add((int(r), 0, int(MARK)))

    payload = {(int(r), int(c), int(color)) for r, c in seed}

    def tr(cell: tuple[int, int, int]) -> tuple[int, int, int]:
        r, c, v = cell
        return (int(sgn_r * r), int(sgn_c * c), int(v))

    return {tr(cell) for cell in red}, {tr(cell) for cell in payload}


def _place_system(
    red_local: set[tuple[int, int, int]],
    payload_local: set[tuple[int, int, int]],
    h: int,
    w: int,
    corner: str,
    bg: int,
) -> np.ndarray:
    if corner == "tl":
        center_r, center_c = 1, 1
    elif corner == "tr":
        center_r, center_c = 1, w - 2
    elif corner == "bl":
        center_r, center_c = h - 2, 1
    else:
        center_r, center_c = h - 2, w - 2

    g = np.full((h, w), int(bg), dtype=int)
    for r, c, v in red_local | payload_local:
        rr = int(r + center_r)
        cc = int(c + center_c)
        if not (0 <= rr < h and 0 <= cc < w):
            raise RuntimeError("system placement overflow")
        g[rr, cc] = int(v)
    return g


def _fusion_metrics(inp: np.ndarray) -> dict[str, int]:
    canvas1, bg1, add1, rounds1 = _simulate(inp, max_rounds=1)
    out1 = _render(canvas1, int(bg1), inp.shape)
    outf = main(inp)
    return {
        "expanded": int(outf.shape[0] > inp.shape[0] or outf.shape[1] > inp.shape[1]),
        "iterative_extra_cells": int(np.count_nonzero(outf != int(get_bg(inp))) - np.count_nonzero(out1 != int(get_bg(inp)))),
        "round1_new": int(add1[0] if add1 else 0),
        "rounds1": int(rounds1),
    }


def _gen_attempt() -> np.ndarray | None:
    rng = _RNG
    bg = int(rng.choice([c for c in range(10) if c != int(MARK)]))
    color = int(rng.choice([c for c in range(1, 10) if c not in {int(MARK), int(bg)}]))
    corner = rng.choice(["tl", "tr", "bl", "br"])
    h = rng.randint(18, 24)
    w = rng.randint(18, 24)

    try:
        seed = _random_seed_shape(rng)
        red_local, payload_local = _oriented_system(seed, corner, int(color))
        inp = _place_system(red_local, payload_local, int(h), int(w), corner, int(bg))
    except RuntimeError:
        return None

    out = main(inp)
    metrics = _fusion_metrics(inp)
    if metrics["expanded"] != 1:
        return None
    if metrics["iterative_extra_cells"] <= 0:
        return None
    if out.shape == inp.shape:
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(400):
        inp = _gen_attempt()
        if inp is not None:
            return inp
    raise RuntimeError("failed to generate valid iterative-expand reflection input")
