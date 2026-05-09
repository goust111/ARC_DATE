# Combo variant of 7ed72f31 (index 4)
# combo_spec: D1/3+D3/2
# Name: 颜色决定变换并交换

from __future__ import annotations

import random
from typing import Iterable

import numpy as np

from arc_utils import get_bg, label_mask


"""
design_analysis:
  variant_core:
    summary:
      - 保留原题的红色标记体系：水平红线、垂直红线、孤立红点。
      - 融合后 payload 颜色同时决定两件事：
        1. 做反射还是做 90 度旋转。
        2. 变换副本使用哪一种交换后的颜色。
      - 红标记几何不再决定“反射 vs 旋转”这个大类，只提供几何参考系：
        - 水平红线：反射时按该行镜像；旋转时绕红线中心旋转。
        - 垂直红线：反射时按该列镜像；旋转时绕红线中心旋转。
        - 孤立红点：反射时做点反射；旋转时绕红点顺时针转 90 度。

  a_concretization_and_trace:
    chosen_color_system:
      - 反射色对: 3 <-> 8
      - 旋转色对: 4 <-> 6
      - 因此 3/8 永远触发 reflect，4/6 永远触发 rotate，同时副本颜色在各自对内交换。
    traced_example:
      - 输入里有一条长度 5 的水平红线，绿色 3 的不对称小精灵位于红线之上。
      - 原题:
        - 仅由“水平红线”决定行为，故输出是在红线另一侧做同色镜像，副本仍为 3。
      - 本融合变体:
        - 先读 payload 颜色 3，推出它属于 reflect 类。
        - 再使用红线几何执行行反射。
        - 最后副本不保留 3，而是交换成 8。
      - 若把同样几何改成颜色 4:
        - 原题仍会做上下镜像。
        - 本变体则改为绕红线中心做四分之一转，并把副本颜色换成 6。
      - 差异:
        - 同一标记几何在不同颜色下会产生不同空间变换。
        - 即使两组都生成副本，副本颜色也不同，因此解题者需要同时归纳两张表：
          - 颜色 -> 变换类别
          - 颜色 -> 交换目标色

  b_complexity_calibration:
    avoided_over_simplification:
      - 不采用“某个几何总是旋转”的设计，否则 D1 改动会退化。
      - 不采用每种颜色都映射到全新目标色的高自由度设计，否则 2-3 个样例很难归纳。
      - 使用两对对称交换色：
        - 3 <-> 8 共享 reflect
        - 4 <-> 6 共享 rotate
      - 这样总共只有四种 payload 色，视觉上紧凑，并且颜色交换与变换类别之间存在结构化联系。
    support_repetition:
      - 每个样例强制包含:
        - 一个 reflect 行组
        - 一个 reflect 列组
        - 一个 rotate 行组
        - 一个 rotate 列组
      - 同时再加入 reflect / rotate 各一个孤立点组，给颜色规则与几何规则额外重复支撑。

  c_implicit_assumption_audit_of_original_main:
    broken_original_assumptions:
      - 原代码把红标记几何直接当成变换类型。
        - 融合后必须先读 payload 颜色，再决定 reflect / rotate。
      - 原代码对混合组件做 bbox+同色全图扫描。
        - 现在重复颜色是必须的，用全图扫描会把别组同色像素误卷入变换。
        - 因此新 solver 只处理 marker-containing component 自己的 payload 像素。
      - 原代码默认副本颜色等于源颜色。
        - 融合后必须在几何变换后再应用颜色交换。
      - 原代码对孤立红点统一做点反射。
        - 新规则里孤立点组也要由 payload 颜色决定 reflect / rotate。
    new_assumptions:
      - 混合 marker component 的 payload 仍是单色。
      - 旋转用到的红线长度必须为奇数，以便存在离散中心像素。
      - 行/列线组的 payload 完全位于红线单侧。
      - 孤立红点只 8 邻接一个单色组件。

  d_placement_legality:
    generator_policy:
      - 使用 output-first：先在局部坐标里构造完整输出组，再删去副本得到输入组。
      - 所有组按输出占位放置，并在放置后给输出像素打 8 邻域 moat，防止:
        - 不同组在输入或输出中连通
        - 同色不同组被 solver 混淆
        - 孤立红点额外接触到无关像素
      - 仅使用已经验证不重叠的局部模板；若放不下整图就重试整案。

  e_visual_observability:
    why_human_can_infer:
      - 行组里同时出现 reflect 色与 rotate 色，肉眼可直接对比“同样是红线，颜色不同导致空间结果不同”。
      - 列组再重复一次同样对比，说明几何只是参考系，不是决定变换类别的根因。
      - 交换色采用成对互换，且两个方向都在一个样例中出现：
        - 例如某些组看到 3 -> 8，另一些组看到 8 -> 3。
      - payload 模板刻意不对称，避免“旋转后看起来像镜像”或“镜像后与原形状太像”的退化。

  f_visual_cleanliness:
    cleanliness_choices:
      - 去掉原题里可能被 bbox 误卷入的 stray sprites。
      - 所有组都很小，彼此之间保留背景缓冲。
      - 红线统一长度 5，视觉上中心明确，旋转参考点容易看出。

  g_change_inferability:
    color_discipline:
      - 只使用四种 payload 色: 3, 4, 6, 8。
      - 背景只从 0/1/7/9 中选，避免与 payload/target 冲突。
      - 颜色变化不是孤立发生，而是伴随稳定几何副本一起出现，因此人类能从位置+颜色双线索推断。

  h_generator_side_non_degeneracy_check:
    metric:
      - 只验证 main(input)==output 不够，因为样例可能退化到“全是反射”或者“只换色不旋转也刚好一样”。
      - 生成器额外比较两个消融求解器:
        - reflect_swap_only: 颜色仍换色，但所有组都按原题几何反射。
        - transform_no_swap: 颜色决定 reflect/rotate，但副本不换色。
      - 要求当前样例输出同时与这两个消融输出不同。
      - 这能量化保证:
        - 旋转分支真的被触发。
        - 换色分支真的改变了结果。

  i_main_assumption_review_after_modification:
    solver_loops_and_guarantees:
      - 混合组件分支假设每组 payload 单色且 marker 几何无歧义。
        - 生成器只造单色 row/col 线组。
      - 行/列旋转分支假设 payload 在红线单侧。
        - 模板构造时只在单侧取坐标。
      - 孤立点分支按颜色+8 连通标签抓相邻组件。
        - 生成器用 moat 保证孤立点只接触自己的 payload。
      - paint 只写回输入中的背景格。
        - 生成器预先保证合法副本区域为空白，因此不会出现“正确答案需要覆盖原物体”的情况。
"""


MARK = 2
REFLECT_COLORS = {3, 8}
ROTATE_COLORS = {4, 6}
COLOR_SWAP = {3: 8, 8: 3, 4: 6, 6: 4}
BG_CHOICES = (0, 1, 7, 9)
LINE_HALF = 2

ROW_TEMPLATES = (
    ((-1, -2), (-1, -1), (-2, -1), (-3, -1)),
    ((-1, 1), (-2, 1), (-2, 2), (-3, 2)),
    ((-1, -2), (-2, -2), (-3, -2), (-3, -1)),
    ((-1, 2), (-2, 2), (-3, 2), (-3, 1)),
    ((-1, -1), (-2, -1), (-2, 0), (-3, 0), (-3, 1)),
    ((-1, 1), (-2, 1), (-2, 0), (-3, 0), (-3, -1)),
)

ISOLATED_TEMPLATES = (
    ((1, 1), (1, 2), (2, 2)),
    ((1, 1), (2, 1), (2, 2), (3, 2)),
    ((1, 1), (2, 2), (3, 2), (3, 3)),
    ((1, 1), (2, 1), (3, 2)),
)


def _transform_kind(color: int) -> str:
    return "reflect" if int(color) in REFLECT_COLORS else "rotate"


def _swap_color(color: int) -> int:
    return int(COLOR_SWAP.get(int(color), int(color)))


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = map(int, shape)
    return 0 <= int(r) < h and 0 <= int(c) < w


def _reflect_point(pivot: tuple[int, int], r: int, c: int) -> tuple[int, int]:
    pr, pc = map(int, pivot)
    return int(2 * pr - r), int(2 * pc - c)


def _reflect_row(axis_r: int, r: int, c: int) -> tuple[int, int]:
    return int(2 * int(axis_r) - int(r)), int(c)


def _reflect_col(axis_c: int, r: int, c: int) -> tuple[int, int]:
    return int(r), int(2 * int(axis_c) - int(c))


def _rotate_cw(pivot: tuple[int, int], r: int, c: int) -> tuple[int, int]:
    pr, pc = map(int, pivot)
    dr = int(r - pr)
    dc = int(c - pc)
    return int(pr + dc), int(pc - dr)


def _rotate_ccw(pivot: tuple[int, int], r: int, c: int) -> tuple[int, int]:
    pr, pc = map(int, pivot)
    dr = int(r - pr)
    dc = int(c - pc)
    return int(pr - dc), int(pc + dr)


def _components(grid: np.ndarray, bg: int) -> list[dict[str, object]]:
    lab, n = label_mask(grid != int(bg), connectivity=4)
    comps: list[dict[str, object]] = []
    for k in range(1, int(n) + 1):
        coords = [(int(r), int(c)) for r, c in np.argwhere(lab == int(k))]
        if not coords:
            continue
        vals = [int(grid[r, c]) for r, c in coords]
        comps.append({"coords": coords, "vals": vals})
    return comps


def _solve(
    grid: np.ndarray,
    *,
    use_color_transform: bool,
    use_color_swap: bool,
) -> np.ndarray:
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))
    h, w = map(int, g.shape)
    out = g.copy()
    comps = _components(g, int(bg))

    def paint(src_pixels: list[tuple[int, int, int]], mapper) -> None:
        for r, c, v in src_pixels:
            rr, cc = mapper(int(r), int(c))
            if _in_bounds((h, w), int(rr), int(cc)) and int(g[rr, cc]) == int(bg):
                out[int(rr), int(cc)] = _swap_color(int(v)) if use_color_swap else int(v)

    for comp in comps:
        coords = comp["coords"]
        vals = comp["vals"]
        reds = [(int(r), int(c)) for (r, c), v in zip(coords, vals) if int(v) == int(MARK)]
        payload = [(int(r), int(c), int(v)) for (r, c), v in zip(coords, vals) if int(v) != int(MARK)]
        if not reds or not payload:
            continue

        payload_colors = {int(v) for _r, _c, v in payload}
        if len(payload_colors) != 1:
            continue
        color = int(next(iter(payload_colors)))
        mode = _transform_kind(int(color)) if use_color_transform else "reflect"

        red_rows = {int(r) for r, _c in reds}
        red_cols = {int(c) for _r, c in reds}

        if len(reds) == 1:
            pivot = reds[0]
            if mode == "reflect":
                paint(payload, lambda r, c, p=pivot: _reflect_point(p, r, c))
            else:
                paint(payload, lambda r, c, p=pivot: _rotate_cw(p, r, c))
            continue

        if len(red_rows) == 1:
            axis_r = int(next(iter(red_rows)))
            if mode == "reflect":
                paint(payload, lambda r, c, rr=axis_r: _reflect_row(rr, r, c))
                continue

            center_c = int(sorted(int(c) for _r, c in reds)[len(reds) // 2])
            pivot = (int(axis_r), int(center_c))
            if max(int(r) for r, _c, _v in payload) < int(axis_r):
                paint(payload, lambda r, c, p=pivot: _rotate_cw(p, r, c))
            elif min(int(r) for r, _c, _v in payload) > int(axis_r):
                paint(payload, lambda r, c, p=pivot: _rotate_ccw(p, r, c))
            continue

        if len(red_cols) == 1:
            axis_c = int(next(iter(red_cols)))
            if mode == "reflect":
                paint(payload, lambda r, c, cc=axis_c: _reflect_col(cc, r, c))
                continue

            center_r = int(sorted(int(r) for r, _c in reds)[len(reds) // 2])
            pivot = (int(center_r), int(axis_c))
            if max(int(c) for _r, c, _v in payload) < int(axis_c):
                paint(payload, lambda r, c, p=pivot: _rotate_ccw(p, r, c))
            elif min(int(c) for _r, c, _v in payload) > int(axis_c):
                paint(payload, lambda r, c, p=pivot: _rotate_cw(p, r, c))

    for comp in comps:
        if len(comp["coords"]) != 1 or int(comp["vals"][0]) != int(MARK):
            continue
        pr, pc = map(int, comp["coords"][0])
        touched: list[tuple[int, int, int]] = []
        neigh_colors: set[int] = set()
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == 0 and dc == 0:
                    continue
                rr = int(pr + dr)
                cc = int(pc + dc)
                if not _in_bounds((h, w), rr, cc):
                    continue
                v = int(g[rr, cc])
                if v == int(bg) or v == int(MARK):
                    continue
                touched.append((int(rr), int(cc), int(v)))
                neigh_colors.add(int(v))

        for color in sorted(neigh_colors):
            labs, _ = label_mask(g == int(color), connectivity=8)
            labels = {
                int(labs[int(rr), int(cc)])
                for rr, cc, vv in touched
                if int(vv) == int(color) and int(labs[int(rr), int(cc)]) != 0
            }
            mode = _transform_kind(int(color)) if use_color_transform else "reflect"
            for lbl in labels:
                pixels = [(int(r), int(c), int(color)) for r, c in np.argwhere(labs == int(lbl))]
                if mode == "reflect":
                    paint(pixels, lambda r, c, p=(int(pr), int(pc)): _reflect_point(p, r, c))
                else:
                    paint(pixels, lambda r, c, p=(int(pr), int(pc)): _rotate_cw(p, r, c))

    return out


def main(grid: np.ndarray) -> np.ndarray:
    return _solve(np.array(grid, dtype=int), use_color_transform=True, use_color_swap=True)


def _row_coords(coords: tuple[tuple[int, int], ...], below: bool) -> list[tuple[int, int]]:
    if not below:
        return [(int(r), int(c)) for r, c in coords]
    return [(-int(r), int(c)) for r, c in coords]


def _col_coords(coords: tuple[tuple[int, int], ...], right: bool) -> list[tuple[int, int]]:
    left = [(int(c), int(r)) for r, c in coords]
    if not right:
        return left
    return [(int(r), -int(c)) for r, c in left]


def _isolated_coords(coords: tuple[tuple[int, int], ...], quadrant: str) -> list[tuple[int, int]]:
    out = []
    for r, c in coords:
        rr, cc = int(r), int(c)
        if quadrant == "se":
            out.append((rr, cc))
        elif quadrant == "sw":
            out.append((rr, -cc))
        elif quadrant == "ne":
            out.append((-rr, cc))
        else:
            out.append((-rr, -cc))
    return out


def _normalize_group(
    out_cells: dict[tuple[int, int], int],
    in_cells: dict[tuple[int, int], int],
    meta: dict[str, object],
) -> dict[str, object]:
    pts = list(out_cells)
    min_r = min(int(r) for r, _c in pts)
    min_c = min(int(c) for _r, c in pts)
    out_norm = {(int(r - min_r), int(c - min_c)): int(v) for (r, c), v in out_cells.items()}
    in_norm = {(int(r - min_r), int(c - min_c)): int(v) for (r, c), v in in_cells.items()}
    h = 1 + max(int(r) for r, _c in out_norm)
    w = 1 + max(int(c) for _r, c in out_norm)
    meta = dict(meta)
    meta["height"] = int(h)
    meta["width"] = int(w)
    return {"out": out_norm, "in": in_norm, "h": int(h), "w": int(w), "meta": meta}


def _build_row_group(src: int, rng: random.Random) -> dict[str, object]:
    mode = _transform_kind(int(src))
    dst = _swap_color(int(src))
    payload = _row_coords(rng.choice(ROW_TEMPLATES), below=bool(rng.randrange(2)))
    out_cells: dict[tuple[int, int], int] = {}
    in_cells: dict[tuple[int, int], int] = {}
    for dc in range(-LINE_HALF, LINE_HALF + 1):
        out_cells[(0, int(dc))] = int(MARK)
        in_cells[(0, int(dc))] = int(MARK)

    for r, c in payload:
        out_cells[(int(r), int(c))] = int(src)
        in_cells[(int(r), int(c))] = int(src)

    if mode == "reflect":
        copy = [(-int(r), int(c)) for r, c in payload]
    elif max(int(r) for r, _c in payload) < 0:
        copy = [(int(c), -int(r)) for r, c in payload]
    else:
        copy = [(-int(c), int(r)) for r, c in payload]

    if len(set(copy)) != len(copy):
        raise RuntimeError("duplicate row-copy cells")
    if set(copy) & set(payload):
        raise RuntimeError("row copy overlaps payload")
    if set(copy) & {(0, int(dc)) for dc in range(-LINE_HALF, LINE_HALF + 1)}:
        raise RuntimeError("row copy overlaps marker")

    for r, c in copy:
        out_cells[(int(r), int(c))] = int(dst)

    return _normalize_group(
        out_cells,
        in_cells,
        {
            "type": "row",
            "source": int(src),
            "target": int(dst),
            "mode": mode,
            "copy_pixels": int(len(copy)),
        },
    )


def _build_col_group(src: int, rng: random.Random) -> dict[str, object]:
    mode = _transform_kind(int(src))
    dst = _swap_color(int(src))
    payload = _col_coords(rng.choice(ROW_TEMPLATES), right=bool(rng.randrange(2)))
    out_cells: dict[tuple[int, int], int] = {}
    in_cells: dict[tuple[int, int], int] = {}
    for dr in range(-LINE_HALF, LINE_HALF + 1):
        out_cells[(int(dr), 0)] = int(MARK)
        in_cells[(int(dr), 0)] = int(MARK)

    for r, c in payload:
        out_cells[(int(r), int(c))] = int(src)
        in_cells[(int(r), int(c))] = int(src)

    if mode == "reflect":
        copy = [(int(r), -int(c)) for r, c in payload]
    elif max(int(c) for _r, c in payload) < 0:
        copy = [(-int(c), int(r)) for r, c in payload]
    else:
        copy = [(int(c), -int(r)) for r, c in payload]

    if len(set(copy)) != len(copy):
        raise RuntimeError("duplicate col-copy cells")
    if set(copy) & set(payload):
        raise RuntimeError("col copy overlaps payload")
    if set(copy) & {(int(dr), 0) for dr in range(-LINE_HALF, LINE_HALF + 1)}:
        raise RuntimeError("col copy overlaps marker")

    for r, c in copy:
        out_cells[(int(r), int(c))] = int(dst)

    return _normalize_group(
        out_cells,
        in_cells,
        {
            "type": "col",
            "source": int(src),
            "target": int(dst),
            "mode": mode,
            "copy_pixels": int(len(copy)),
        },
    )


def _build_iso_group(src: int, rng: random.Random) -> dict[str, object]:
    mode = _transform_kind(int(src))
    dst = _swap_color(int(src))
    quad = rng.choice(("se", "sw", "ne", "nw"))
    payload = _isolated_coords(rng.choice(ISOLATED_TEMPLATES), quad)
    out_cells: dict[tuple[int, int], int] = {(0, 0): int(MARK)}
    in_cells: dict[tuple[int, int], int] = {(0, 0): int(MARK)}

    for r, c in payload:
        out_cells[(int(r), int(c))] = int(src)
        in_cells[(int(r), int(c))] = int(src)

    if mode == "reflect":
        copy = [(-int(r), -int(c)) for r, c in payload]
    else:
        copy = [(int(c), -int(r)) for r, c in payload]

    if len(set(copy)) != len(copy):
        raise RuntimeError("duplicate iso-copy cells")
    if set(copy) & set(payload):
        raise RuntimeError("iso copy overlaps payload")
    if (0, 0) in set(copy):
        raise RuntimeError("iso copy overlaps marker")

    for r, c in copy:
        out_cells[(int(r), int(c))] = int(dst)

    return _normalize_group(
        out_cells,
        in_cells,
        {
            "type": "iso",
            "source": int(src),
            "target": int(dst),
            "mode": mode,
            "copy_pixels": int(len(copy)),
        },
    )


def _build_group(kind: str, src: int, rng: random.Random) -> dict[str, object]:
    if kind == "row":
        return _build_row_group(int(src), rng)
    if kind == "col":
        return _build_col_group(int(src), rng)
    if kind == "iso":
        return _build_iso_group(int(src), rng)
    raise ValueError(f"unknown group kind: {kind}")


def _can_place(
    group: dict[str, object],
    top: int,
    left: int,
    h: int,
    w: int,
    forbidden: set[tuple[int, int]],
) -> bool:
    for r, c in group["out"]:
        rr = int(top + r)
        cc = int(left + c)
        if not (0 <= rr < int(h) and 0 <= cc < int(w)):
            return False
        if (int(rr), int(cc)) in forbidden:
            return False
    return True


def _stamp_moat(forbidden: set[tuple[int, int]], cells: Iterable[tuple[int, int]]) -> None:
    for r, c in cells:
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                forbidden.add((int(r + dr), int(c + dc)))


def generate_case(rng: random.Random | None = None) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    rng = rng or random.Random()

    for _attempt in range(500):
        h = rng.randint(20, 29)
        w = rng.randint(20, 29)
        bg = rng.choice(BG_CHOICES)

        reflect_pair = rng.choice(((3, 8), (8, 3)))
        rotate_pair = rng.choice(((4, 6), (6, 4)))
        specs = [
            ("row", int(reflect_pair[0])),
            ("col", int(reflect_pair[1])),
            ("row", int(rotate_pair[0])),
            ("col", int(rotate_pair[1])),
            ("iso", int(reflect_pair[0])),
            ("iso", int(rotate_pair[0])),
        ]
        rng.shuffle(specs)

        try:
            groups = [_build_group(kind, int(src), rng) for kind, src in specs]
        except RuntimeError:
            continue
        groups.sort(key=lambda g: len(g["out"]), reverse=True)

        inp = np.full((h, w), int(bg), dtype=int)
        out = np.full((h, w), int(bg), dtype=int)
        forbidden: set[tuple[int, int]] = set()
        placements: list[dict[str, object]] = []
        ok = True

        for group in groups:
            placed = False
            gh = int(group["h"])
            gw = int(group["w"])
            for _ in range(700):
                top = rng.randint(0, int(h - gh))
                left = rng.randint(0, int(w - gw))
                if not _can_place(group, int(top), int(left), int(h), int(w), forbidden):
                    continue

                abs_out = []
                for (r, c), v in group["out"].items():
                    rr = int(top + r)
                    cc = int(left + c)
                    out[rr, cc] = int(v)
                    abs_out.append((int(rr), int(cc)))
                for (r, c), v in group["in"].items():
                    inp[int(top + r), int(left + c)] = int(v)

                _stamp_moat(forbidden, abs_out)
                meta = dict(group["meta"])
                meta["top"] = int(top)
                meta["left"] = int(left)
                placements.append(meta)
                placed = True
                break

            if not placed:
                ok = False
                break

        if not ok:
            continue

        solved = main(inp)
        if not np.array_equal(solved, out):
            continue

        reflect_swap_only = _solve(inp, use_color_transform=False, use_color_swap=True)
        transform_no_swap = _solve(inp, use_color_transform=True, use_color_swap=False)
        diff_vs_reflect_swap = int(np.count_nonzero(out != reflect_swap_only))
        diff_vs_transform_no_swap = int(np.count_nonzero(out != transform_no_swap))
        if diff_vs_reflect_swap == 0 or diff_vs_transform_no_swap == 0:
            continue

        mode_counts = {
            "reflect": sum(1 for p in placements if str(p["mode"]) == "reflect"),
            "rotate": sum(1 for p in placements if str(p["mode"]) == "rotate"),
        }
        if min(mode_counts.values()) == 0:
            continue

        meta = {
            "bg": int(bg),
            "grid_shape": (int(h), int(w)),
            "placements": placements,
            "mode_counts": mode_counts,
            "fusion_metric": {
                "diff_vs_reflect_swap_only": int(diff_vs_reflect_swap),
                "diff_vs_transform_no_swap": int(diff_vs_transform_no_swap),
            },
            "sources": [int(p["source"]) for p in placements],
            "targets": [int(p["target"]) for p in placements],
        }
        return inp, out, meta

    raise RuntimeError("failed to generate a valid combo case")


def generate_input() -> np.ndarray:
    inp, _out, _meta = generate_case()
    return inp
