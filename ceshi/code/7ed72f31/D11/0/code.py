# Variant of 7ed72f31
# Dimension: D11 — 源-目标基数关系 (meta)
# Change: 1:1 — 每个标记组独立对应一个payload精灵，独立产生一个镜像副本
#      → 多对一：多个payload共享一个反射轴
#
# Design analysis
# ===============
# variant_spec:
#   core_change:
#     original: 每条红色反射线只服务一个payload精灵
#     new: 一条红色反射线同时服务两个payload精灵
#     concrete_form:
#       row_axis:
#         - 一条水平红线
#         - 红线左段上方放一个单色sprite
#         - 红线右段下方放另一个不同颜色的单色sprite
#         - 两者都跨同一条红线做镜像
#       col_axis:
#         - 一条竖直红线
#         - 红线上段左侧放一个单色sprite
#         - 红线下段右侧放另一个不同颜色的单色sprite
#         - 两者都跨同一条红线做镜像
#   behavior_trace:
#     original_small_example:
#       - 输入: 水平红线只在上方接一个绿色sprite
#       - solver: 识别该组件中的红线为水平轴, 把绿色像素映射到红线下方
#     variant_small_example:
#       - 输入: 同一条水平红线, 左上有绿色sprite, 右下有黄色sprite
#       - solver: 仍先识别同一条水平轴
#       - 之后不再假设“这条轴只有一个payload颜色”, 而是把该组件中的所有非红像素逐点跨轴镜像
#       - 结果: 绿色出现在左下, 黄色出现在右上
#     delta:
#       - 旧题的“一个marker组 -> 一个payload”变成“一个marker组 -> 两个payload”
#       - 视觉上能直接看到同一条红线两侧都有原始sprite, 且输出中二者都被复制到对侧
#   complexity_calibration:
#     - 只引入两payload共享一条线轴, 不额外引入颜色重编码或多步变换
#     - 每个共享轴只服务两个payload, 复杂度固定在少量档位, 人眼可在2-4个样例里归纳
#     - 两个payload分居轴线的不同段, 避免它们的镜像落点互相覆盖
#   implicit_assumption_audit:
#     - 原solver按颜色集合 + bbox在整张图里扫像素, 旧题里大多“恰好无害”
#     - 变体里同一marker组件会有多个颜色, 若继续依赖颜色+bbox, 同色杂点更容易被误吸入
#     - 新solver改为: 只反射当前marker组件里的真实非红像素
#     - 原solver对 isolated red pixel 的点反射逻辑不依赖 D11, 保持不变
#   placement_legality:
#     - 生成器按 output-first 构造完整输出组, 再删去镜像副本得到输入
#     - 每个组在放置前检查: bbox不越界, 组与组之间保留至少1格边距, 输出像素不重叠
#     - 行轴组让上/下两个payload使用不同列段; 列轴组让左/右两个payload使用不同 row 段
#   observability:
#     - 每个样例至少含2个共享轴组
#     - 行轴与列轴都出现, 让“共享轴”不是单一样式偶然现象
#     - 两侧payload颜色不同, 共享关系更醒目
#   visual_cleanliness:
#     - 不再生成原题里那类容易被 bbox 误吸的自由散点, 避免噪声掩盖共享轴
#     - 组间用bbox+margin分开, 保持图面整洁
#   color_inference:
#     - 颜色不发生映射, payload镜像后保持原色
#     - 颜色只用于区分同一轴服务的两个sprite, 不增加额外推理负担
#   main_generate_contract:
#     - main() 现在假设共享线轴组件中的payload镜像目标区域为空
#     - generate_input() 通过将两个payload放在轴线的不同段来保证这一点

from __future__ import annotations

import numpy as np

from arc_utils import get_bg, label_mask


MARK = 2


def main(grid: np.ndarray) -> np.ndarray:
    """
    Mirror colored sprites using red (2) markers as reflection axes.

    D11 variant:
    - A single horizontal/vertical red line may serve two payload sprites at once.
    - The solver reflects every non-red pixel inside that marker-bearing component,
      so both payloads sharing the same axis are mirrored.
    - Isolated red pixels keep the original point-reflection behavior.
    """
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))
    H, W = g.shape

    def inb(r: int, c: int) -> bool:
        return 0 <= int(r) < H and 0 <= int(c) < W

    def refl_point(p: tuple[int, int], r: int, c: int) -> tuple[int, int]:
        pr, pc = map(int, p)
        return int(2 * pr - r), int(2 * pc - c)

    def refl_row(r0: int, r: int, c: int) -> tuple[int, int]:
        return int(2 * int(r0) - int(r)), int(c)

    def refl_col(c0: int, r: int, c: int) -> tuple[int, int]:
        return int(r), int(2 * int(c0) - int(c))

    out = g.copy()

    # 4-connected components of all non-background pixels.
    lab, n = label_mask(g != int(bg), connectivity=4)

    for k in range(1, int(n) + 1):
        coords = [(int(r), int(c)) for r, c in np.argwhere(lab == int(k))]
        if not coords:
            continue

        reds = [(int(r), int(c)) for r, c in coords if int(g[r, c]) == int(MARK)]
        if not reds:
            continue

        rs = {int(r) for r, _c in reds}
        cs = {int(c) for _r, c in reds}
        axis: tuple[str, int | tuple[int, int]] | None = None
        if len(reds) == 1:
            axis = ("point", (int(reds[0][0]), int(reds[0][1])))
        elif len(rs) == 1:
            cvals = sorted(int(c) for _r, c in reds)
            if cvals == list(range(cvals[0], cvals[-1] + 1)):
                axis = ("row", int(next(iter(rs))))
        elif len(cs) == 1:
            rvals = sorted(int(r) for r, _c in reds)
            if rvals == list(range(rvals[0], rvals[-1] + 1)):
                axis = ("col", int(next(iter(cs))))
        if axis is None:
            continue

        # D11-sensitive change:
        # reflect only the actual pixels inside this component, rather than
        # scanning the whole grid by color+bbox. This safely supports multiple
        # payload colors sharing the same marker axis.
        for r, c in coords:
            v = int(g[r, c])
            if v == int(bg) or v == int(MARK):
                continue

            if axis[0] == "row":
                rr, cc = refl_row(int(axis[1]), int(r), int(c))
            elif axis[0] == "col":
                rr, cc = refl_col(int(axis[1]), int(r), int(c))
            else:
                rr, cc = refl_point(axis[1], int(r), int(c))  # type: ignore[arg-type]

            if inb(int(rr), int(cc)):
                out[int(rr), int(cc)] = int(v)

    # Original isolated-point rule remains unchanged.
    marker_pts = []
    for k in range(1, int(n) + 1):
        coords = [(int(r), int(c)) for r, c in np.argwhere(lab == int(k))]
        if len(coords) == 1:
            r, c = coords[0]
            if int(g[r, c]) == int(MARK):
                marker_pts.append((int(r), int(c)))

    for pr, pc in marker_pts:
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

    return out


def _rng(rng: np.random.Generator | None = None) -> np.random.Generator:
    return rng if rng is not None else np.random.default_rng()


def _choice_int(rng: np.random.Generator, lo: int, hi: int) -> int:
    return int(rng.integers(int(lo), int(hi) + 1))


def _neighbors4(r: int, c: int, h: int, w: int) -> list[tuple[int, int]]:
    out = []
    for rr, cc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
        if 0 <= rr < h and 0 <= cc < w:
            out.append((int(rr), int(cc)))
    return out


def _random_connected_shape(
    rng: np.random.Generator,
    h: int,
    w: int,
    n_cells: int,
    anchors: list[tuple[int, int]],
) -> list[tuple[int, int]]:
    n_cells = min(int(n_cells), int(h * w))
    cells = {anchors[_choice_int(rng, 0, len(anchors) - 1)]}
    frontier = set(_neighbors4(*next(iter(cells)), h, w)) - cells

    while len(cells) < int(n_cells):
        if not frontier:
            raise ValueError("shape frontier exhausted")
        idx = _choice_int(rng, 0, len(frontier) - 1)
        pick = list(frontier)[idx]
        frontier.remove(pick)
        cells.add(pick)
        for nb in _neighbors4(*pick, h, w):
            if nb not in cells:
                frontier.add(nb)

    return sorted((int(r), int(c)) for r, c in cells)


def _reflect_row(axis_r: int, cells: list[tuple[int, int, int]]) -> list[tuple[int, int, int]]:
    return [(int(2 * axis_r - r), int(c), int(v)) for r, c, v in cells]


def _reflect_col(axis_c: int, cells: list[tuple[int, int, int]]) -> list[tuple[int, int, int]]:
    return [(int(r), int(2 * axis_c - c), int(v)) for r, c, v in cells]


def _pick_two_colors(rng: np.random.Generator, palette: list[int]) -> tuple[int, int]:
    cols = rng.choice(np.array(palette, dtype=int), size=2, replace=False)
    return int(cols[0]), int(cols[1])


def _build_shared_row_group(
    rng: np.random.Generator,
    palette: list[int],
) -> dict:
    side_h = _choice_int(rng, 2, 4)
    top_w = _choice_int(rng, 2, 4)
    bot_w = _choice_int(rng, 2, 4)
    gap = _choice_int(rng, 1, 3)
    left_pad = _choice_int(rng, 0, 1)
    right_pad = _choice_int(rng, 0, 1)
    axis_len = left_pad + top_w + gap + bot_w + right_pad
    axis_r = side_h
    color_top, color_bot = _pick_two_colors(rng, palette)

    n_top = _choice_int(rng, 3, min(7, side_h * top_w))
    n_bot = _choice_int(rng, 3, min(7, side_h * bot_w))
    top_rel = _random_connected_shape(
        rng,
        side_h,
        top_w,
        n_top,
        anchors=[(side_h - 1, c) for c in range(top_w)],
    )
    bot_rel = _random_connected_shape(
        rng,
        side_h,
        bot_w,
        n_bot,
        anchors=[(0, c) for c in range(bot_w)],
    )

    top_cells = [(int(r), int(left_pad + c), int(color_top)) for r, c in top_rel]
    bot_offset = left_pad + top_w + gap
    bot_cells = [(int(axis_r + 1 + r), int(bot_offset + c), int(color_bot)) for r, c in bot_rel]
    red_cells = [(int(axis_r), int(c), int(MARK)) for c in range(axis_len)]

    inp_cells = top_cells + bot_cells + red_cells
    out_cells = inp_cells + _reflect_row(axis_r, top_cells) + _reflect_row(axis_r, bot_cells)

    return {
        "kind": "row",
        "bbox": (2 * side_h + 1, axis_len),
        "input_cells": inp_cells,
        "output_cells": out_cells,
        "meta": {"shared_axis": True, "payload_count": 2},
    }


def _build_shared_col_group(
    rng: np.random.Generator,
    palette: list[int],
) -> dict:
    side_w = _choice_int(rng, 2, 4)
    left_h = _choice_int(rng, 2, 4)
    right_h = _choice_int(rng, 2, 4)
    gap = _choice_int(rng, 1, 3)
    top_pad = _choice_int(rng, 0, 1)
    bot_pad = _choice_int(rng, 0, 1)
    axis_len = top_pad + left_h + gap + right_h + bot_pad
    axis_c = side_w
    color_left, color_right = _pick_two_colors(rng, palette)

    n_left = _choice_int(rng, 3, min(7, left_h * side_w))
    n_right = _choice_int(rng, 3, min(7, right_h * side_w))
    left_rel = _random_connected_shape(
        rng,
        left_h,
        side_w,
        n_left,
        anchors=[(r, side_w - 1) for r in range(left_h)],
    )
    right_rel = _random_connected_shape(
        rng,
        right_h,
        side_w,
        n_right,
        anchors=[(r, 0) for r in range(right_h)],
    )

    left_cells = [(int(top_pad + r), int(c), int(color_left)) for r, c in left_rel]
    right_offset = top_pad + left_h + gap
    right_cells = [(int(right_offset + r), int(axis_c + 1 + c), int(color_right)) for r, c in right_rel]
    red_cells = [(int(r), int(axis_c), int(MARK)) for r in range(axis_len)]

    inp_cells = left_cells + right_cells + red_cells
    out_cells = inp_cells + _reflect_col(axis_c, left_cells) + _reflect_col(axis_c, right_cells)

    return {
        "kind": "col",
        "bbox": (axis_len, 2 * side_w + 1),
        "input_cells": inp_cells,
        "output_cells": out_cells,
        "meta": {"shared_axis": True, "payload_count": 2},
    }


def _stamp(
    grid: np.ndarray,
    cells: list[tuple[int, int, int]],
    top: int,
    left: int,
) -> None:
    for r, c, v in cells:
        grid[int(top + r), int(left + c)] = int(v)


def _sample_layout(
    rng: np.random.Generator,
    H: int,
    W: int,
    groups: list[dict],
) -> list[tuple[int, int]]:
    reserved = np.zeros((H, W), dtype=bool)
    placements: list[tuple[int, int]] = []

    groups_sorted = sorted(enumerate(groups), key=lambda kv: kv[1]["bbox"][0] * kv[1]["bbox"][1], reverse=True)
    placed: dict[int, tuple[int, int]] = {}

    for idx, group in groups_sorted:
        gh, gw = map(int, group["bbox"])
        ok = False
        for _ in range(400):
            top = _choice_int(rng, 0, H - gh)
            left = _choice_int(rng, 0, W - gw)
            r0 = max(0, top - 1)
            r1 = min(H, top + gh + 1)
            c0 = max(0, left - 1)
            c1 = min(W, left + gw + 1)
            if reserved[r0:r1, c0:c1].any():
                continue
            reserved[r0:r1, c0:c1] = True
            placed[idx] = (int(top), int(left))
            ok = True
            break
        if not ok:
            raise ValueError("failed to place group")

    for i in range(len(groups)):
        placements.append(placed[i])
    return placements


def _build_sample(
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray, dict]:
    rng = _rng(rng)
    H = _choice_int(rng, 22, 28)
    W = _choice_int(rng, 22, 28)
    bg = int(rng.choice([c for c in range(10) if c != MARK]))
    avail = [c for c in range(10) if c not in (bg, MARK)]
    pal_size = _choice_int(rng, 4, min(6, len(avail)))
    palette = [int(x) for x in rng.choice(np.array(avail, dtype=int), size=pal_size, replace=False)]

    n_groups = _choice_int(rng, 3, 4)
    kinds = ["row", "col"]
    group_types = ["row", "col"]
    for _ in range(max(0, n_groups - 2)):
        group_types.append(str(rng.choice(kinds)))
    rng.shuffle(group_types)

    groups = []
    for kind in group_types:
        group = _build_shared_row_group(rng, palette) if kind == "row" else _build_shared_col_group(rng, palette)
        groups.append(group)

    placements = _sample_layout(rng, H, W, groups)
    inp = np.full((H, W), int(bg), dtype=int)
    out = np.full((H, W), int(bg), dtype=int)

    for group, (top, left) in zip(groups, placements):
        _stamp(inp, group["input_cells"], top, left)
        _stamp(out, group["output_cells"], top, left)

    meta = {
        "bg": int(bg),
        "group_types": group_types,
        "shared_axis_groups": len(groups),
        "payloads_per_shared_axis": [2 for _ in groups],
    }
    return inp, out, meta


def generate_input() -> np.ndarray:
    """
    Output-first generator for the D11 shared-axis variant.

    Guarantees:
    - every line marker group has exactly two payload sprites sharing the axis
    - at least one row-axis and one col-axis shared group appear
    - main(input) reproduces the constructed output
    """
    for _ in range(200):
        try:
            inp, out, _meta = _build_sample()
        except ValueError:
            continue
        if np.array_equal(main(inp), out):
            return inp
    raise RuntimeError("failed to generate a valid sample")
