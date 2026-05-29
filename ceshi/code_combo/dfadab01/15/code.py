# Combo variant of dfadab01 (index 15)
# combo_spec: D6/1+D7/0
# Name: 环绕放置+随机散布

from __future__ import annotations

import random

import numpy as np


# Design analysis
# ---------------
# spec:
#   fused_rule_summary:
#     - 保留原题的固定 4x4 字形字典、marker 颜色集合、以及“可选图例 + 右下角标签”的语义。
#     - 同时引入两项改动：真实 marker 改为不规则随机散布；marker 展开出的 4x4 字形不再裁剪，而是在网格上按环面坐标写入。
#   concrete_trace:
#     - 小例子：10x10 网格里，真实 marker 可以散布在 (1,2)、(8,7)、(4,0)、(6,5) 这类不等间距位置，而不是原题那种规则行列槽位。
#     - 若 marker 2 在 (8,7)，原题 clipped 只会留下行 8..9、列 7..9 内的黄色边框残片。
#     - 本融合版会把该 4x4 边框按模运算写入：(10,*) 回到 0 行，(*,10) 回到 0 列，所以同一字形会在底边和顶边、右边和左边分裂出现。
#     - 因为 marker 不再等间距，解题者不能先推断“网格槽位”，而必须逐个找单像素 marker，再对每个 marker 应用同一个环面字形规则。
#   complexity_calibration:
#     - 若随机散布不加约束，环面上的 4x4 字形很容易互相覆盖，输出会变脏且 solver 写入顺序会影响结果。
#     - 因此生成器用“4x4 环面包围盒互斥”做硬约束：任意两个真实 marker 的 4x4 占位集合都不相交。
#     - 为了让变体肉眼可见，同一输入里强制至少有一个 wrapped marker，以及一个同色的 interior marker，形成“同字形内部正常显示 / 边缘分裂显示”的对照。
#     - marker 数量保持中等：10x10 用 4 个，20x20 用 5-9 个，既能体现随机散布，也给图例留下空间。
#   implicit_assumptions_review:
#     - 原 _has_source_glyph 假设图例样例是完整的界内 4x4 图案；该假设仍成立，因为生成器把图例限制在完整的 5x5 内部窗口中，不允许图例环绕。
#     - 原 solver 虽然遍历所有 marker 颜色，但并不真正依赖“规则槽位”；因此 D7/0 不需要改 solver 的识别流程，只需要让 generator 不再产出等距布局。
#     - 原 _paint_clipped 依赖“越界像素直接丢弃”；D6/1 使这个前提失效，必须改成对行列取模的 _paint_toroidal。
#     - 主循环逐个写出 glyph 仍然安全，但现在它额外依赖“不同 marker 的环面 4x4 占位不重叠”；generate_input() 会构造性保证这一点。
#   placement_legality:
#     - 真实 marker 通过拒绝采样放置；每加入一个 marker，都会占用其对应 4x4 环面包围盒的 16 个单元，后续 marker 不得再触碰这些单元。
#     - 图例放在一个完全空白的 5x5 输入窗口里：前 4x4 是样例字形，右下角一格是标签 marker，保证不会和真实 marker 或其他图例像素重叠。
#   observability_and_cleanliness:
#     - 输出只包含固定字形和黑底，不使用颜色 id 的数值大小做任何隐藏规则。
#     - 至少一个字形会跨边界分裂，且同图保留一个同色未环绕实例；再配合不规则散点输入，解题者能从少量样例中看出“随机定位 + 环绕放置”的组合变化。
#     - 4x4 包围盒互斥和适中的 marker 数量保证输出区域不互相盖住，避免视觉噪声。
#   inference_check:
#     - 颜色变化仍然只有原题固定的四种 marker->glyph 对应，没有额外条件分支。
#     - 随机散布不会破坏可推断性，因为规则对每个 marker 独立同构；样例中重复出现的同色 interior/wrapped 对照足以支撑归纳。


GLYPHS: dict[int, tuple[int, np.ndarray]] = {
    2: (
        4,
        np.array(
            [
                [1, 1, 1, 1],
                [1, 0, 0, 1],
                [1, 0, 0, 1],
                [1, 1, 1, 1],
            ],
            dtype=bool,
        ),
    ),
    3: (
        1,
        np.array(
            [
                [0, 1, 1, 0],
                [1, 0, 0, 1],
                [1, 0, 0, 1],
                [0, 1, 1, 0],
            ],
            dtype=bool,
        ),
    ),
    5: (
        6,
        np.array(
            [
                [1, 1, 0, 0],
                [1, 1, 0, 0],
                [0, 0, 1, 1],
                [0, 0, 1, 1],
            ],
            dtype=bool,
        ),
    ),
    8: (
        7,
        np.array(
            [
                [1, 0, 0, 1],
                [0, 1, 1, 0],
                [0, 1, 1, 0],
                [1, 0, 0, 1],
            ],
            dtype=bool,
        ),
    ),
}


def _has_source_glyph(g: np.ndarray, r0: int, c0: int, color: int, mask: np.ndarray) -> bool:
    h, w = g.shape
    if r0 < 0 or c0 < 0 or r0 + 4 > h or c0 + 4 > w:
        return False
    patch = g[r0 : r0 + 4, c0 : c0 + 4]
    return bool(np.all(patch[mask] == color) and np.all(patch[~mask] != color))


def _paint_clipped(out: np.ndarray, r0: int, c0: int, color: int, mask: np.ndarray) -> None:
    h, w = out.shape
    for dr in range(4):
        rr = r0 + dr
        if rr < 0 or rr >= h:
            continue
        for dc in range(4):
            cc = c0 + dc
            if 0 <= cc < w and bool(mask[dr, dc]):
                out[rr, cc] = color


def _paint_toroidal(out: np.ndarray, r0: int, c0: int, color: int, mask: np.ndarray) -> None:
    h, w = out.shape
    for dr in range(4):
        for dc in range(4):
            if bool(mask[dr, dc]):
                out[(r0 + dr) % h, (c0 + dc) % w] = color


def _marker_wraps(r0: int, c0: int, mask: np.ndarray, h: int, w: int) -> bool:
    for dr in range(4):
        for dc in range(4):
            if bool(mask[dr, dc]) and (r0 + dr >= h or c0 + dc >= w):
                return True
    return False


def _bbox_cells_toroidal(r0: int, c0: int, h: int, w: int) -> set[tuple[int, int]]:
    return {((r0 + dr) % h, (c0 + dc) % w) for dr in range(4) for dc in range(4)}


def _cyclic_gaps(vals: list[int], size: int) -> list[int]:
    ordered = sorted(vals)
    return [((ordered[(i + 1) % len(ordered)] - ordered[i]) % size) for i in range(len(ordered))]


def _layout_is_irregular(positions: list[tuple[int, int]], size: int) -> bool:
    if len(positions) < 4:
        return False

    rows = sorted({r for r, _ in positions})
    cols = sorted({c for _, c in positions})
    full_product = {(r, c) for r in rows for c in cols}
    if len(rows) >= 2 and len(cols) >= 2 and set(positions) == full_product:
        return False

    row_regular = len(rows) >= 3 and len(set(_cyclic_gaps(rows, size))) == 1
    col_regular = len(cols) >= 3 and len(set(_cyclic_gaps(cols, size))) == 1
    if row_regular and col_regular:
        return False

    return True


def _expected_output(markers: dict[tuple[int, int], int], size: int, toroidal: bool) -> np.ndarray:
    out = np.zeros((size, size), dtype=int)
    for (r, c), marker_color in markers.items():
        glyph_color, mask = GLYPHS[marker_color]
        if toroidal:
            _paint_toroidal(out, r, c, glyph_color, mask)
        else:
            _paint_clipped(out, r, c, glyph_color, mask)
    return out


def _try_place_legend(inp: np.ndarray, used_colors: list[int]) -> bool:
    size = inp.shape[0]
    candidates = list(used_colors)
    random.shuffle(candidates)
    for marker_color in candidates:
        glyph_color, mask = GLYPHS[marker_color]
        for _ in range(100):
            r0 = random.randint(0, size - 5)
            c0 = random.randint(0, size - 5)
            if np.any(inp[r0 : r0 + 5, c0 : c0 + 5] != 0):
                continue
            for dr in range(4):
                for dc in range(4):
                    if bool(mask[dr, dc]):
                        inp[r0 + dr, c0 + dc] = glyph_color
            inp[r0 + 4, c0 + 4] = marker_color
            return True
    return False


def _sample_marker_positions(size: int, marker_count: int) -> list[tuple[int, int]] | None:
    all_positions = [(r, c) for r in range(size) for c in range(size)]
    wrapped_positions = [(r, c) for r, c in all_positions if r >= size - 3 or c >= size - 3]
    interior_positions = [(r, c) for r, c in all_positions if r <= size - 4 and c <= size - 4]
    random.shuffle(wrapped_positions)
    random.shuffle(interior_positions)

    selected: list[tuple[int, int]] = []
    occupied: set[tuple[int, int]] = set()

    wrapped_focus = next(
        ((r, c) for r, c in wrapped_positions if _bbox_cells_toroidal(r, c, size, size).isdisjoint(occupied)),
        None,
    )
    if wrapped_focus is None:
        return None
    selected.append(wrapped_focus)
    occupied |= _bbox_cells_toroidal(wrapped_focus[0], wrapped_focus[1], size, size)

    interior_focus = next(
        ((r, c) for r, c in interior_positions if _bbox_cells_toroidal(r, c, size, size).isdisjoint(occupied)),
        None,
    )
    if interior_focus is None:
        return None
    selected.append(interior_focus)
    occupied |= _bbox_cells_toroidal(interior_focus[0], interior_focus[1], size, size)

    target_wrapped = random.randint(1, min(marker_count - 1, 2 if size == 10 else 4))
    while len(selected) < marker_count:
        current_wrapped = sum(r >= size - 3 or c >= size - 3 for r, c in selected)
        prefer_wrapped = current_wrapped < target_wrapped and random.random() < 0.8
        pools = [wrapped_positions, all_positions] if prefer_wrapped else [all_positions, wrapped_positions]
        choice = None
        for pool in pools:
            shuffled = list(pool)
            random.shuffle(shuffled)
            for r, c in shuffled:
                if (r, c) in selected:
                    continue
                cells = _bbox_cells_toroidal(r, c, size, size)
                if cells.isdisjoint(occupied):
                    choice = (r, c)
                    occupied |= cells
                    break
            if choice is not None:
                break
        if choice is None:
            return None
        selected.append(choice)

    if not _layout_is_irregular(selected, size):
        return None
    return selected


def main(grid: np.ndarray) -> np.ndarray:
    """
    Expand each real marker into its fixed 4x4 glyph on a wraparound grid, even though the markers are scattered irregularly.

    Input structure:
    - Black background with single-pixel markers in colors 2, 3, 5, and/or 8 placed at non-uniform positions rather than on a regular lattice.
    - Some inputs also contain one complete legend entry: a 4x4 sample glyph in its output color plus a same-type label marker at the sample's lower-right offset (+4, +4).

    Rule:
    - Convert each real marker color into the same fixed glyph mapping as the base task: 2->yellow frame, 3->blue ring, 5->magenta diagonal blocks, 8->orange X.
    - Use the marker as the glyph's top-left anchor; whenever glyph pixels run past the bottom or right edge, continue them from the opposite side of the grid instead of clipping them away.
    - Remove any legend glyph from the output and skip the legend's label marker rather than expanding it.

    Key insight:
    - The dictionary from marker color to 4x4 shape is unchanged, but the spatial cue has changed twice: markers must be found one by one because they are randomly scattered, and each glyph must be painted with toroidal coordinates rather than rectangular clipping.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    assert set(np.unique(g)).issubset({0, *GLYPHS.keys(), 1, 4, 6, 7})

    out = np.zeros_like(g)
    seen_marker = False
    for marker_color, (glyph_color, mask) in GLYPHS.items():
        coords = np.argwhere(g == marker_color)
        if coords.size:
            seen_marker = True
        for r, c in coords:
            if _has_source_glyph(g, int(r) - 4, int(c) - 4, glyph_color, mask):
                continue
            _paint_toroidal(out, int(r), int(c), glyph_color, mask)

    assert seen_marker
    return out


def generate_input() -> np.ndarray:
    for _ in range(600):
        size = random.choice([10, 20])
        marker_count = 4 if size == 10 else random.randint(5, 9)
        positions = _sample_marker_positions(size, marker_count)
        if positions is None:
            continue

        used_count = random.randint(1, min(4, marker_count - 1))
        used_colors = random.sample(list(GLYPHS.keys()), used_count)
        focus_color = random.choice(used_colors)

        wrapped_positions = [pos for pos in positions if pos[0] >= size - 3 or pos[1] >= size - 3]
        interior_positions = [pos for pos in positions if pos[0] <= size - 4 and pos[1] <= size - 4]
        if not wrapped_positions or not interior_positions:
            continue

        wrapped_focus = random.choice(wrapped_positions)
        interior_focus = random.choice(interior_positions)
        markers: dict[tuple[int, int], int] = {
            wrapped_focus: focus_color,
            interior_focus: focus_color,
        }

        remaining_positions = [pos for pos in positions if pos not in markers]
        random.shuffle(remaining_positions)

        for marker_color in used_colors:
            if marker_color == focus_color or not remaining_positions:
                continue
            markers[remaining_positions.pop()] = marker_color
        for pos in remaining_positions:
            markers[pos] = random.choice(used_colors)

        wrapped_real = sum(
            _marker_wraps(r, c, GLYPHS[marker_color][1], size, size) for (r, c), marker_color in markers.items()
        )
        if wrapped_real == 0:
            continue

        toroidal_out = _expected_output(markers, size, toroidal=True)
        clipped_out = _expected_output(markers, size, toroidal=False)
        if np.array_equal(toroidal_out, clipped_out):
            continue

        if not _layout_is_irregular(list(markers), size):
            continue

        inp = np.zeros((size, size), dtype=int)
        for (r, c), marker_color in markers.items():
            inp[r, c] = marker_color

        if random.random() < 0.6:
            _try_place_legend(inp, used_colors)

        solved = main(inp)
        assert np.array_equal(solved, toroidal_out)
        return inp

    raise AssertionError("generate_input failed after retries")
