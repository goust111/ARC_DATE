from __future__ import annotations

# Variant of 7ed72f31
# Dimension: D13 — 对称性修复替代镜像生成 (transform)
# Change: 在背景上生成全新的镜像副本（添加像素）... → 对称性补全：仅添加缺失像素使对称完整
#
# Design analysis
# a) Concrete behavior and trace
#    - 原题输入通常只有轴一侧的单色 payload，输出把整块 payload 镜像到另一侧。
#    - 本变体输入改为“已部分对称”：红线/红点仍然给出对称轴或对称中心，但 payload
#      可以同时出现在两侧，只是某些对称位置缺了一格。
#    - 输出不再无条件生成一整块新副本，而是仅把“已有像素的对称对应位置中仍为背景”的格子补上。
#    - 小例子：红横线在第 5 行，输入在第 4 行有蓝点 (4,7)，第 6 行已有蓝点 (6,7)，
#      但第 3 行有蓝点 (3,8) 而第 7 行 (7,8) 为空。原逻辑会把所有观察到的蓝点都写到对侧；
#      变体逻辑仍然会从 (3,8) 推出 (7,8)，但遇到 (4,7) ↔ (6,7) 时不会“重画”或覆盖，
#      只补背景缺口。因此输入输出差异只发生在缺失格，而不是整片新副本。
# b) Complexity calibration
#    - 如果所有输入都退化成“纯单侧 payload”，那就看起来仍像原题。本 generator 强制每个
#      组至少保留一个已成对像素，同时至少保留一个单侧缺口，使 D13 的变化可见。
#    - 颜色仍保持每组单色、组间颜色唯一，避免对称补全和组匹配发生歧义。
# c) Implicit-assumption audit of original main()
#    - 原 solver 的关键隐含假设是：把反射结果直接写进 out 不会破坏正确性，因为目标位置原本是背景。
#      这一点在原题成立，在本变体中不成立，因为输入里可能已经有正确的对称像素。
#      因此所有 `out[rr, cc] = v` 都必须改成“仅当目标仍为背景时写入”。
#    - 原 solver 对 row/col 组件不是只看组件内部，而是按 payload 颜色 + 组件 bbox 的投影范围
#      扫描整张图。这在原题用于容纳靠近轴的散点；本变体仍可用，但要求 generator 保证：
#      同色像素只属于一个组，且组之间留出间距，避免误吸收。
#    - isolated red 分支会把所有与红点 8 邻接的同色组件都做点反射；本变体允许输入中心两侧都
#      已有一些同色碎片，只要颜色唯一，它们被同时闭包成中心对称图案即可。
# d) Placement legality
#    - generator 先构造完整对称输出组，再删掉部分成对像素得到输入；放置时按输出 bbox 检查边界，
#      并给每组留 2 格间距，保证不重叠、不越界。
#    - row/col 组使用唯一颜色，且完整输出 bbox 就是将来补全所需的空间，所以 solver 反射出的像素
#      不会撞到其他组。
# e) Visual observability
#    - 每个 row/col/point 组都至少有一个“已对称”的位置和一个“待补全”的位置，解题者能从 2-4 个
#      样例中看出：不是整块复制，而是哪里缺就补哪里。
#    - 红标记的语义保持不变，降低学习成本；变化只体现在 payload 已经部分存在于对侧。
# f) Visual cleanliness
#    - 不生成自由散点干扰物，不复用颜色，不让组互相贴边，保证每个红标记附近只有一组清晰图形。
# g) Color inference
#    - 不引入颜色映射，颜色只标识“哪组像素属于同一个对称补全对象”，避免额外认知负担。
# h) New assumptions introduced by the revised main()
#    - 新 main() 假设正确输出只需“补背景”，不会要求覆盖错误颜色。generator 通过唯一颜色、
#      非重叠放置和 output-first 校验保证这一点。
#    - row/col 分支不再沿整个列/行投影捕捉同色像素，而是限制在红标记附近的 2D bbox 内。
#      这是为了让 D13 样本中出现“同色旁观像素”时仍只补全标记精灵本身，从而避免退化回原题。
#    - isolated 分支允许同色多碎片，只要它们都围绕同一个红点；generator 保证不会有第二个同色组。
# i) Generator-side guarantees
#    - 每个样例都至少包含一个真正的 D13 组：输入里已经有至少一对对称像素，同时仍缺至少一个对称点。
#    - `generate_input()` 内部执行 oracle 校验：`main(input) == intended_output`，失败则重采样。

import random
from typing import Iterable

import numpy as np

from arc_utils import get_bg, label_mask

MARK = 2
ALL_COLORS = list(range(10))


def main(grid: np.ndarray) -> np.ndarray:
    """
    Symmetry completion variant of 7ed72f31.

    Red geometry still defines the symmetry:
    - red row -> mirror across that row
    - red column -> mirror across that column
    - single red pixel -> point reflection around that point

    Difference from the original task:
    - reflect every observed payload pixel as before
    - but only fill missing background cells; existing non-background pixels stay unchanged
    """
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))
    H, W = g.shape

    lab, n = label_mask(g != bg, connectivity=4)
    comps: list[dict[str, object]] = []
    for k in range(1, int(n) + 1):
        coords = [(int(r), int(c)) for r, c in np.argwhere(lab == int(k))]
        if not coords:
            continue
        vals = [int(g[r, c]) for r, c in coords]
        rs = [r for r, _c in coords]
        cs = [c for _r, c in coords]
        comps.append(
            {
                "coords": coords,
                "vals": vals,
                "bbox": (int(min(rs)), int(min(cs)), int(max(rs)), int(max(cs))),
            }
        )

    out = g.copy()

    def inb(r: int, c: int) -> bool:
        return 0 <= int(r) < H and 0 <= int(c) < W

    def paint_if_missing(r: int, c: int, v: int) -> None:
        if inb(int(r), int(c)) and int(out[int(r), int(c)]) == bg:
            out[int(r), int(c)] = int(v)

    def refl_point(p: tuple[int, int], r: int, c: int) -> tuple[int, int]:
        pr, pc = map(int, p)
        return int(2 * pr - r), int(2 * pc - c)

    def refl_row(r0: int, r: int, c: int) -> tuple[int, int]:
        return int(2 * int(r0) - int(r)), int(c)

    def refl_col(c0: int, r: int, c: int) -> tuple[int, int]:
        return int(r), int(2 * int(c0) - int(c))

    # Marker-containing components: reflect observed payload pixels, but only into missing bg cells.
    for comp in comps:
        coords = comp["coords"]
        vals = comp["vals"]
        if MARK not in set(map(int, vals)):
            continue

        reds = [(int(r), int(c)) for (r, c), v in zip(coords, vals) if int(v) == MARK]
        if not reds:
            continue

        rs = {int(r) for r, _c in reds}
        cs = {int(c) for _r, c in reds}
        if len(reds) == 1:
            axis = ("point", (int(reds[0][0]), int(reds[0][1])))
        elif len(rs) == 1:
            axis = ("row", int(next(iter(rs))))
        elif len(cs) == 1:
            axis = ("col", int(next(iter(cs))))
        else:
            continue

        r0, c0, r1, c1 = map(int, comp["bbox"])
        payload = {int(v) for v in vals if int(v) != MARK}
        if not payload:
            continue

        rr0 = max(0, int(r0 - 1))
        rr1 = min(int(H - 1), int(r1 + 1))
        cc0 = max(0, int(c0 - 1))
        cc1 = min(int(W - 1), int(c1 + 1))

        if axis[0] == "row":

            def eligible(r: int, c: int, v: int) -> bool:
                return int(v) in payload and int(rr0) <= int(r) <= int(rr1) and int(cc0) <= int(c) <= int(cc1)

        elif axis[0] == "col":

            def eligible(r: int, c: int, v: int) -> bool:
                return int(v) in payload and int(rr0) <= int(r) <= int(rr1) and int(cc0) <= int(c) <= int(cc1)

        else:
            def eligible(r: int, c: int, v: int) -> bool:
                return int(v) in payload and int(rr0) <= int(r) <= int(rr1) and int(cc0) <= int(c) <= int(cc1)

        for r in range(int(H)):
            for c in range(int(W)):
                v = int(g[int(r), int(c)])
                if v == bg or v == MARK or not eligible(int(r), int(c), int(v)):
                    continue

                if axis[0] == "row":
                    rr, cc = refl_row(int(axis[1]), int(r), int(c))
                elif axis[0] == "col":
                    rr, cc = refl_col(int(axis[1]), int(r), int(c))
                else:
                    rr, cc = refl_point(axis[1], int(r), int(c))

                paint_if_missing(int(rr), int(cc), int(v))

    # Isolated red points: close each adjacent same-color component under point reflection.
    marker_pts = []
    for comp in comps:
        if len(comp["coords"]) == 1 and int(comp["vals"][0]) == MARK:
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
                if v == bg or v == MARK:
                    continue
                neigh_colors.add(int(v))
                neigh_pixels.append((int(rr), int(cc), int(v)))
        if not neigh_colors:
            continue

        for v in neigh_colors:
            labv, _nv = label_mask(g == int(v), connectivity=8)
            labs = {
                int(labv[int(rr), int(cc)])
                for rr, cc, vv in neigh_pixels
                if int(vv) == int(v) and int(labv[int(rr), int(cc)]) != 0
            }
            for lbl in labs:
                for r, c in np.argwhere(labv == int(lbl)):
                    rr, cc = refl_point((int(pr), int(pc)), int(r), int(c))
                    paint_if_missing(int(rr), int(cc), int(v))

    return out


def _bbox(coords: Iterable[tuple[int, int]]) -> tuple[int, int, int, int]:
    coords = list(coords)
    rs = [r for r, _c in coords]
    cs = [c for _r, c in coords]
    return min(rs), min(cs), max(rs), max(cs)


def _rand_choice(rng: random.Random, seq: list[tuple[int, int]]) -> tuple[int, int]:
    return seq[rng.randrange(len(seq))]


def _grow_connected_4(
    rng: random.Random,
    seed: tuple[int, int],
    target_size: int,
    r_lo: int,
    r_hi: int,
    c_lo: int,
    c_hi: int,
) -> set[tuple[int, int]]:
    cells = {seed}
    for _ in range(200):
        if len(cells) >= int(target_size):
            break
        base = _rand_choice(rng, list(cells))
        cand = []
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nr, nc = int(base[0] + dr), int(base[1] + dc)
            if nr < r_lo or nr > r_hi or nc < c_lo or nc > c_hi:
                continue
            if (nr, nc) not in cells:
                cand.append((nr, nc))
        if cand:
            cells.add(_rand_choice(rng, cand))
    if len(cells) < int(target_size):
        raise ValueError("could not grow 4-connected shape")
    return cells


def _grow_connected_8(
    rng: random.Random,
    seed: tuple[int, int],
    target_size: int,
    sign_r: int,
    sign_c: int,
    max_r: int,
    max_c: int,
) -> set[tuple[int, int]]:
    cells = {seed}
    for _ in range(300):
        if len(cells) >= int(target_size):
            break
        base = _rand_choice(rng, list(cells))
        cand = []
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == 0 and dc == 0:
                    continue
                nr = int(base[0] + dr)
                nc = int(base[1] + dc)
                if sign_r * nr < 1 or sign_r * nr > max_r:
                    continue
                if sign_c * nc < 1 or sign_c * nc > max_c:
                    continue
                if abs(nr) + abs(nc) == 1:
                    continue
                if (nr, nc) not in cells:
                    cand.append((nr, nc))
        if cand:
            cells.add(_rand_choice(rng, cand))
    if len(cells) < int(target_size):
        raise ValueError("could not grow 8-connected shape")
    return cells


def _input_states(rng: random.Random, n_pairs: int) -> list[str]:
    if n_pairs < 2:
        raise ValueError("need at least two reflection pairs")
    states = ["both"]
    rest = [rng.choice(["both", "a", "b"]) for _ in range(n_pairs - 1)]
    states.extend(rest)
    if all(s == "both" for s in states):
        states[-1] = rng.choice(["a", "b"])
    if not any(s == "both" for s in states):
        states[0] = "both"
    return states


def _build_row_group(rng: random.Random, color: int) -> dict[str, object]:
    line_len = rng.randint(2, 6)
    side_height = rng.randint(2, 4)
    left_extra = rng.randint(0, 2)
    right_extra = rng.randint(0, 2)
    c_lo = -left_extra
    c_hi = line_len - 1 + right_extra
    seed = (1, rng.randint(0, line_len - 1))
    size = rng.randint(4, min(8, (side_height * (c_hi - c_lo + 1))))
    half = _grow_connected_4(rng, seed, size, 1, side_height, c_lo, c_hi)

    states = _input_states(rng, len(half))
    half_sorted = sorted(half)
    input_payload: set[tuple[int, int]] = set()
    output_payload: set[tuple[int, int]] = set()
    singletons = 0
    for (r, c), state in zip(half_sorted, states):
        a = (-int(r), int(c))
        b = (int(r), int(c))
        output_payload.add(a)
        output_payload.add(b)
        if state == "both":
            input_payload.add(a)
            input_payload.add(b)
        elif state == "a":
            input_payload.add(a)
            singletons += 1
        else:
            input_payload.add(b)
            singletons += 1
    if singletons == 0:
        raise ValueError("row group degenerated to full symmetry")

    reds = {(0, c) for c in range(line_len)}
    out_cells = [(r, c, color) for r, c in sorted(output_payload)] + [(r, c, MARK) for r, c in sorted(reds)]
    inp_cells = [(r, c, color) for r, c in sorted(input_payload)] + [(r, c, MARK) for r, c in sorted(reds)]
    bbox = _bbox((r, c) for r, c, _v in out_cells)
    return {
        "type": "row",
        "color": int(color),
        "out_cells": out_cells,
        "inp_cells": inp_cells,
        "bbox": bbox,
        "pair_count": len(half_sorted),
        "missing_pairs": singletons,
        "kept_pairs": sum(1 for s in states if s == "both"),
    }


def _build_col_group(rng: random.Random, color: int) -> dict[str, object]:
    line_len = rng.randint(2, 6)
    side_width = rng.randint(2, 4)
    top_extra = rng.randint(0, 2)
    bottom_extra = rng.randint(0, 2)
    r_lo = -top_extra
    r_hi = line_len - 1 + bottom_extra
    seed = (rng.randint(0, line_len - 1), 1)
    size = rng.randint(4, min(8, ((r_hi - r_lo + 1) * side_width)))
    half = _grow_connected_4(rng, seed, size, r_lo, r_hi, 1, side_width)

    states = _input_states(rng, len(half))
    half_sorted = sorted(half)
    input_payload: set[tuple[int, int]] = set()
    output_payload: set[tuple[int, int]] = set()
    singletons = 0
    for (r, c), state in zip(half_sorted, states):
        a = (int(r), -int(c))
        b = (int(r), int(c))
        output_payload.add(a)
        output_payload.add(b)
        if state == "both":
            input_payload.add(a)
            input_payload.add(b)
        elif state == "a":
            input_payload.add(a)
            singletons += 1
        else:
            input_payload.add(b)
            singletons += 1
    if singletons == 0:
        raise ValueError("col group degenerated to full symmetry")

    reds = {(r, 0) for r in range(line_len)}
    out_cells = [(r, c, color) for r, c in sorted(output_payload)] + [(r, c, MARK) for r, c in sorted(reds)]
    inp_cells = [(r, c, color) for r, c in sorted(input_payload)] + [(r, c, MARK) for r, c in sorted(reds)]
    bbox = _bbox((r, c) for r, c, _v in out_cells)
    return {
        "type": "col",
        "color": int(color),
        "out_cells": out_cells,
        "inp_cells": inp_cells,
        "bbox": bbox,
        "pair_count": len(half_sorted),
        "missing_pairs": singletons,
        "kept_pairs": sum(1 for s in states if s == "both"),
    }


def _build_point_group(rng: random.Random, color: int) -> dict[str, object]:
    sign_r = rng.choice([-1, 1])
    sign_c = rng.choice([-1, 1])
    max_r = rng.randint(2, 4)
    max_c = rng.randint(2, 4)
    seed = (sign_r, sign_c)
    size = rng.randint(2, min(5, max_r * max_c))
    half = _grow_connected_8(rng, seed, size, sign_r, sign_c, max_r, max_c)

    states = _input_states(rng, len(half))
    half_sorted = sorted(half)
    input_payload: set[tuple[int, int]] = set()
    output_payload: set[tuple[int, int]] = set()
    singletons = 0
    for (r, c), state in zip(half_sorted, states):
        a = (int(r), int(c))
        b = (-int(r), -int(c))
        output_payload.add(a)
        output_payload.add(b)
        if state == "both":
            input_payload.add(a)
            input_payload.add(b)
        elif state == "a":
            input_payload.add(a)
            singletons += 1
        else:
            input_payload.add(b)
            singletons += 1
    if singletons == 0:
        raise ValueError("point group degenerated to full symmetry")

    out_cells = [(r, c, color) for r, c in sorted(output_payload)] + [(0, 0, MARK)]
    inp_cells = [(r, c, color) for r, c in sorted(input_payload)] + [(0, 0, MARK)]
    bbox = _bbox((r, c) for r, c, _v in out_cells)
    return {
        "type": "point",
        "color": int(color),
        "out_cells": out_cells,
        "inp_cells": inp_cells,
        "bbox": bbox,
        "pair_count": len(half_sorted),
        "missing_pairs": singletons,
        "kept_pairs": sum(1 for s in states if s == "both"),
    }


def _translate_cells(cells: list[tuple[int, int, int]], top: int, left: int, r0: int, c0: int) -> list[tuple[int, int, int]]:
    return [(int(top + r - r0), int(left + c - c0), int(v)) for r, c, v in cells]


def _place_group(
    rng: random.Random,
    inp: np.ndarray,
    out: np.ndarray,
    reserved: np.ndarray,
    group: dict[str, object],
    bg: int,
) -> bool:
    H, W = inp.shape
    r0, c0, r1, c1 = map(int, group["bbox"])
    gh = int(r1 - r0 + 1)
    gw = int(c1 - c0 + 1)
    moat = 2
    if gh + 2 * moat > H or gw + 2 * moat > W:
        return False

    for _ in range(200):
        top = rng.randint(moat, H - gh - moat)
        left = rng.randint(moat, W - gw - moat)
        rr0 = top - moat
        cc0 = left - moat
        rr1 = top + gh + moat - 1
        cc1 = left + gw + moat - 1
        if reserved[rr0 : rr1 + 1, cc0 : cc1 + 1].any():
            continue

        out_cells = _translate_cells(group["out_cells"], top, left, r0, c0)
        inp_cells = _translate_cells(group["inp_cells"], top, left, r0, c0)
        if any(out[r, c] != bg for r, c, _v in out_cells):
            continue
        if any(inp[r, c] != bg for r, c, _v in inp_cells):
            continue

        for r, c, v in out_cells:
            out[r, c] = int(v)
        for r, c, v in inp_cells:
            inp[r, c] = int(v)
        reserved[rr0 : rr1 + 1, cc0 : cc1 + 1] = True
        group["placed_bbox"] = (int(top), int(left), int(top + gh - 1), int(left + gw - 1))
        if group["type"] == "row":
            group["axis_global"] = int(top - r0)
        elif group["type"] == "col":
            group["axis_global"] = int(left - c0)
        return True
    return False


def _place_linear_decoy(
    rng: random.Random,
    inp: np.ndarray,
    out: np.ndarray,
    reserved: np.ndarray,
    group: dict[str, object],
    bg: int,
) -> bool:
    if group["type"] not in {"row", "col"}:
        return False
    H, W = inp.shape
    color = int(group["color"])
    g_r0, g_c0, g_r1, g_c1 = map(int, group["placed_bbox"])
    axis = int(group["axis_global"])
    options: list[list[tuple[int, int]]] = []

    if group["type"] == "row":
        for side in (-1, 1):
            base_r = g_r0 - 3 if side < 0 else g_r1 + 3
            for extra in range(3):
                r = base_r - extra if side < 0 else base_r + extra
                rr = int(2 * axis - r)
                if not (0 <= r < H and 0 <= rr < H):
                    continue
                width = rng.randint(1, min(2, g_c1 - g_c0 + 1))
                start_lo = g_c0
                start_hi = g_c1 - width + 1
                if start_lo > start_hi:
                    continue
                for c_start in range(start_lo, start_hi + 1):
                    cells = [(r, c_start + k) for k in range(width)]
                    refl = [(rr, c_start + k) for k in range(width)]
                    if all(out[a, b] == bg and inp[a, b] == bg for a, b in cells) and all(out[a, b] == bg for a, b in refl):
                        options.append(cells)
    else:
        for side in (-1, 1):
            base_c = g_c0 - 3 if side < 0 else g_c1 + 3
            for extra in range(3):
                c = base_c - extra if side < 0 else base_c + extra
                cc = int(2 * axis - c)
                if not (0 <= c < W and 0 <= cc < W):
                    continue
                height = rng.randint(1, min(2, g_r1 - g_r0 + 1))
                start_lo = g_r0
                start_hi = g_r1 - height + 1
                if start_lo > start_hi:
                    continue
                for r_start in range(start_lo, start_hi + 1):
                    cells = [(r_start + k, c) for k in range(height)]
                    refl = [(r_start + k, cc) for k in range(height)]
                    if all(out[a, b] == bg and inp[a, b] == bg for a, b in cells) and all(out[a, b] == bg for a, b in refl):
                        options.append(cells)

    rng.shuffle(options)
    for cells in options:
        rr0, cc0, rr1, cc1 = _bbox(cells)
        if reserved[max(0, rr0 - 1) : min(H, rr1 + 2), max(0, cc0 - 1) : min(W, cc1 + 2)].any():
            continue
        for r, c in cells:
            inp[r, c] = color
            out[r, c] = color
        reserved[max(0, rr0 - 1) : min(H, rr1 + 2), max(0, cc0 - 1) : min(W, cc1 + 2)] = True
        return True
    return False


def _build_group(rng: random.Random, kind: str, color: int) -> dict[str, object]:
    if kind == "row":
        return _build_row_group(rng, color)
    if kind == "col":
        return _build_col_group(rng, color)
    if kind == "point":
        return _build_point_group(rng, color)
    raise ValueError(f"unknown kind: {kind}")


def _generate_case(rng: random.Random) -> tuple[np.ndarray, np.ndarray, dict[str, int]]:
    H = rng.randint(20, 28)
    W = rng.randint(20, 28)
    bg = rng.choice([c for c in ALL_COLORS if c != MARK])

    available_colors = [c for c in ALL_COLORS if c not in (bg, MARK)]
    rng.shuffle(available_colors)

    n_groups = rng.randint(3, 5)
    kinds: list[str] = []
    linear_count = rng.randint(2, n_groups)
    for _ in range(linear_count):
        kinds.append(rng.choice(["row", "col"]))
    while len(kinds) < n_groups:
        kinds.append("point")
    rng.shuffle(kinds)

    inp = np.full((H, W), bg, dtype=int)
    out = np.full((H, W), bg, dtype=int)
    reserved = np.zeros((H, W), dtype=bool)

    groups = []
    for i in range(n_groups):
        groups.append(_build_group(rng, kinds[i], available_colors[i]))

    rng.shuffle(groups)
    for group in groups:
        if not _place_group(rng, inp, out, reserved, group, bg):
            raise ValueError("placement failed")
        if group["type"] in {"row", "col"} and rng.random() < 0.75:
            _place_linear_decoy(rng, inp, out, reserved, group, bg)

    delta = int(np.count_nonzero((inp == bg) & (out != bg)))
    if delta <= 0:
        raise ValueError("no missing pixels to repair")

    meta = {
        "groups": int(n_groups),
        "linear_groups": int(sum(1 for g in groups if g["type"] in {"row", "col"})),
        "point_groups": int(sum(1 for g in groups if g["type"] == "point")),
        "kept_pairs": int(sum(int(g["kept_pairs"]) for g in groups)),
        "missing_pairs": int(sum(int(g["missing_pairs"]) for g in groups)),
        "delta_pixels": int(delta),
    }
    return inp, out, meta


def _generate_valid_case(rng: random.Random) -> tuple[np.ndarray, np.ndarray, dict[str, int]]:
    last_err: Exception | None = None
    for _ in range(200):
        try:
            inp, intended_out, meta = _generate_case(rng)
            actual_out = main(inp)
            if not np.array_equal(actual_out, intended_out):
                raise AssertionError("oracle mismatch")
            return inp, intended_out, meta
        except Exception as exc:  # pragma: no cover - exercised by stochastic retries
            last_err = exc
    raise RuntimeError(f"failed to generate valid D13 case: {last_err}")


def generate_input() -> np.ndarray:
    """
    Output-first generator for the D13 variant.

    The intended output is a fully symmetry-completed grid.
    The returned input already contains some symmetric counterparts, plus some missing ones.
    """
    seed = random.randrange(1 << 30)
    rng = random.Random(seed)
    try:
        inp, _intended_out, _meta = _generate_valid_case(rng)
        return inp
    except Exception as exc:
        raise RuntimeError(f"failed to generate valid D13 sample (seed={seed}): {exc}") from exc
