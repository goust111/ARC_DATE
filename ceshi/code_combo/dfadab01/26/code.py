# Combo variant of dfadab01 (index 26)
# combo_spec: D1/2+D3/3+D5/1
# Name: 奇偶翻转+中心定位+非对称字形

from __future__ import annotations

import random

import numpy as np


# ---------------------------------------------------------------------------
# Design analysis for the combo variant
# ---------------------------------------------------------------------------
# design:
#   core_change:
#     - 原题把单像素 marker 当作 4x4 字形左上角，并始终画 canonical 朝向的对称字形。
#     - 本融合版同时改三件事：
#       1) marker 表示字形中心，实际绘制左上角为 (r-2, c-2)；
#       2) 只看 marker 所在“真实 marker 网格行”的奇偶性：偶数行保持原样，奇数行左右翻转；
#       3) 四个 4x4 字形全部换成明显非对称的 L/阶梯类 mask，让翻转后的差异肉眼可见。
#   a_concretization_and_trace:
#     input:
#       - 黑底，真实控制像素仍只使用 {2,3,5,8}。
#       - 这些像素排在 3x3 到 5x5 的稀疏规则网格上；每条网格行和每条网格列至少各出现一个 marker。
#       - 可选 legend 仍是一个 canonical 4x4 彩色字形，标签 marker 放在其包围框右下角 (+4,+4)。
#     output:
#       - 对每个真实 marker，先查颜色对应的输出色与 canonical 非对称字形。
#       - 再按该 marker 处于第几条真实 marker 行决定朝向：row_idx 为奇数时左右翻转。
#       - 最后把字形以“中心锚点”放到输出上，即左上角使用 (r-2, c-2)，越界部分裁剪。
#     concrete_trace:
#       - 例：某个灰色 5-marker 位于真实 marker 网格第 1 行、第 2 列，像素坐标是 (6, 9)。
#       - 原题：直接在 (6, 9) 画出 5 对应的固定对称 4x4 字形。
#       - 本题：先认出它属于奇数行，所以对 5 的非对称 canonical mask 做左右翻转；
#         再把翻转后的 4x4 字形放到左上角 (4, 7) = (6-2, 9-2)。
#       - 差异是三层叠加的：锚点先改成中心，再按行奇偶判定朝向，再因为字形非对称而显出翻转效果。
#   b_complexity_calibration:
#     - 若仍用原题对称字形，水平翻转几乎不可见，因此这里强制所有字形都水平非对称。
#     - 规则自由度只保留少量、可观察的两档：even row / odd row；不再叠加列方向规则，避免样例过载。
#     - 生成器强制至少有一个奇数行 marker，且至少有一对同色 marker 分布在偶数行与奇数行，
#       让解题者能看到“同一种字形在不同网格行朝向不同”。
#   c_implicit_assumptions_review:
#     original_main_dependencies:
#       - 原代码假设 marker 坐标就是 glyph 左上角；本变体失效，必须统一改成中心锚点减 2。
#       - 原代码假设 mask 与位置无关；本变体失效，绘制前要先依据真实 marker 行索引决定是否翻转。
#       - 原代码按颜色独立扫描 marker；本变体若要判断行奇偶，必须先收集全部真实 marker，再做全局行编号。
#       - 原 `_has_source_glyph()` 依赖 legend 使用 canonical mask；本变体继续沿用，所以 generator
#         也必须始终把 legend 画成 canonical 朝向，不能按中心锚点或奇偶翻转去画 legend。
#     new_solver_assumptions:
#       - 真实 marker 行索引取自“排序后的唯一行坐标”，因此每条 intended lattice row 必须至少有一个 marker。
#       - legend 标签只会出现在 canonical legend 的右下角 (+4,+4)。
#       - 非对称 glyph 的水平镜像必须与 canonical 不同，否则 odd row 规则会退化。
#     generator_guarantees:
#       - 占位矩阵先覆盖所有行/列，再补随机空位，所以不会缺整行或整列。
#       - 至少一条 odd row 有 marker，且至少有一对跨 even/odd 行的同色 marker。
#       - legend 只在空白区画 canonical glyph + 标签，避免伪造新的 marker 行。
#   d_placement_legality:
#     - 实际输出 bbox 是 [r-2:r+2) x [c-2:c+2)，所以 marker 行列间距仍保持 4..6，保证字形不重叠。
#     - 网格尺寸允许边缘裁剪，但所有 marker 像素本身都落在画布内。
#     - legend 搜索时要求 4x4 glyph 和标签点都为空，避免与真实 marker 或其他 legend 像素冲突。
#   e_visual_observability:
#     - 同色 marker 在不同真实 marker 行里会出现同一家族但左右相反的非对称图案，规律可以直接看出来。
#     - 中心锚点通过“字形围住 marker 位置而不是从 marker 向右下展开”体现，边缘裁剪样本会进一步强化这一点。
#     - 行奇偶使用的是可见的 marker 行序，而不是颜色值或隐藏排序，不依赖人类不可见因素。
#   f_cleanliness:
#     - 仍然只有黑底、单像素 marker、可选单个 legend，层级和原题一样克制。
#     - 非对称字形依旧限制在 4x4、小像素数，避免大块实心区域造成脏乱。
#   g_color_rule_inference:
#     - 颜色映射仍是固定字典，不引入额外的颜色逻辑。
#     - 颜色只决定“是哪一种字形/输出色”，行位置只决定“是否左右翻转”，两个因素职责清晰。


GLYPHS: dict[int, tuple[int, np.ndarray]] = {
    2: (
        4,
        np.array(
            [
                [1, 1, 1, 0],
                [1, 0, 0, 0],
                [1, 0, 1, 1],
                [0, 0, 0, 1],
            ],
            dtype=bool,
        ),
    ),
    3: (
        1,
        np.array(
            [
                [0, 1, 1, 0],
                [1, 0, 0, 0],
                [1, 1, 0, 0],
                [0, 0, 1, 1],
            ],
            dtype=bool,
        ),
    ),
    5: (
        6,
        np.array(
            [
                [1, 1, 0, 0],
                [0, 1, 0, 0],
                [0, 1, 1, 0],
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
                [0, 1, 0, 0],
                [0, 0, 1, 0],
                [0, 1, 1, 1],
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


def _orient_mask(mask: np.ndarray, row_idx: int) -> np.ndarray:
    return np.fliplr(mask) if row_idx % 2 == 1 else mask


def _extract_real_markers(g: np.ndarray) -> list[tuple[int, int, int]]:
    markers: list[tuple[int, int, int]] = []
    for marker_color, (glyph_color, mask) in GLYPHS.items():
        for r, c in np.argwhere(g == marker_color):
            rr = int(r)
            cc = int(c)
            if _has_source_glyph(g, rr - 4, cc - 4, glyph_color, mask):
                continue
            markers.append((rr, cc, int(marker_color)))
    return markers


def _row_ids(markers: list[tuple[int, int, int]]) -> dict[int, int]:
    return {int(r): idx for idx, r in enumerate(sorted({int(r) for r, _c, _mc in markers}))}


def _render_markers(
    shape: tuple[int, int],
    markers: list[tuple[int, int, int]],
    *,
    mirrored: bool,
    centered: bool,
) -> np.ndarray:
    out = np.zeros(shape, dtype=int)
    if not markers:
        return out

    row_ids = _row_ids(markers)
    for r, c, marker_color in markers:
        glyph_color, base_mask = GLYPHS[int(marker_color)]
        mask = _orient_mask(base_mask, row_ids[int(r)]) if mirrored else base_mask
        anchor_r = int(r) - 2 if centered else int(r)
        anchor_c = int(c) - 2 if centered else int(c)
        _paint_clipped(out, anchor_r, anchor_c, int(glyph_color), mask)
    return out


def _behavior_metrics(inp: np.ndarray) -> dict[str, int]:
    markers = _extract_real_markers(inp)
    rows = _row_ids(markers)
    combo = _render_markers(inp.shape, markers, mirrored=True, centered=True)
    no_flip = _render_markers(inp.shape, markers, mirrored=False, centered=True)
    no_center = _render_markers(inp.shape, markers, mirrored=True, centered=False)

    odd_row_markers = sum(1 for r, _c, _mc in markers if rows[int(r)] % 2 == 1)
    visible_flip_markers = sum(
        1
        for r, _c, marker_color in markers
        if rows[int(r)] % 2 == 1 and not np.array_equal(_orient_mask(GLYPHS[int(marker_color)][1], rows[int(r)]), GLYPHS[int(marker_color)][1])
    )
    return {
        "marker_count": len(markers),
        "odd_row_markers": int(odd_row_markers),
        "visible_flip_markers": int(visible_flip_markers),
        "flip_diff_cells": int(np.count_nonzero(combo != no_flip)),
        "center_diff_cells": int(np.count_nonzero(combo != no_center)),
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Replace each control pixel with a non-symmetric 4x4 glyph centered on that pixel, with odd marker rows mirrored.

    Input structure:
    - Black background with single-pixel markers in colors 2, 3, 5, and/or 8 arranged on a sparse regular marker lattice.
    - Some inputs also contain one legend entry: a canonical 4x4 glyph in its output color plus a same-type label marker
      at the legend bbox's lower-right offset (+4,+4).

    Rule:
    - Ignore the legend glyph and skip its label marker, exactly as in the original task.
    - For each real marker, look up its fixed output color and canonical non-symmetric 4x4 glyph.
    - Determine the marker's row index within the sorted set of real marker rows; odd row indices flip the glyph
      horizontally, even row indices keep it unchanged. Then paint the glyph with top-left corner at (r-2, c-2),
      so the marker acts as the glyph center, clipping at the grid boundary.

    Key insight:
    - The input still uses one control pixel per glyph, but that pixel is now a center anchor, and orientation is not fixed:
      the same marker color produces a left-right mirrored glyph whenever it appears on an odd marker row.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    assert set(np.unique(g)).issubset({0, *GLYPHS.keys(), 1, 4, 6, 7})

    real_markers = _extract_real_markers(g)
    assert real_markers
    return _render_markers(g.shape, real_markers, mirrored=True, centered=True)


# --------------- generator (output-first centered mirrored glyph substitution) ---------------


def _build_occupancy(rng: random.Random, rows: int, cols: int) -> np.ndarray:
    occ = np.zeros((rows, cols), dtype=bool)

    # Force a visible even-row/odd-row pair in the same column.
    occ[0, 0] = True
    occ[1, 0] = True

    for r in range(rows):
        occ[r, rng.randrange(cols)] = True
    for c in range(cols):
        occ[rng.randrange(rows), c] = True

    min_fill = max(rows + cols - 1, int(np.ceil(0.7 * rows * cols)))
    target_fill = rng.randint(min_fill, rows * cols)
    vacant = [(r, c) for r in range(rows) for c in range(cols) if not occ[r, c]]
    rng.shuffle(vacant)
    for r, c in vacant[: max(0, target_fill - int(occ.sum()))]:
        occ[r, c] = True
    return occ


def _place_legend(rng: random.Random, grid: np.ndarray, legend_marker_color: int) -> bool:
    glyph_color, mask = GLYPHS[int(legend_marker_color)]
    h, w = grid.shape
    if h < 5 or w < 5:
        return False

    for _ in range(250):
        r0 = rng.randint(0, h - 5)
        c0 = rng.randint(0, w - 5)
        needed = [(r0 + 4, c0 + 4)]
        for dr in range(4):
            for dc in range(4):
                if bool(mask[dr, dc]):
                    needed.append((r0 + dr, c0 + dc))
        if all(int(grid[r, c]) == 0 for r, c in needed):
            for dr in range(4):
                for dc in range(4):
                    if bool(mask[dr, dc]):
                        grid[r0 + dr, c0 + dc] = int(glyph_color)
            grid[r0 + 4, c0 + 4] = int(legend_marker_color)
            return True
    return False


def _pick_origin(rng: random.Random, slack: int) -> int:
    if slack <= 0:
        return 0
    if slack <= 2:
        return rng.randint(0, slack)
    if rng.random() < 0.55:
        return rng.randint(0, 2)
    return rng.randint(0, slack)


def _assign_marker_colors(
    rng: random.Random,
    occupied_cells: list[tuple[int, int]],
    marker_types: list[int],
) -> dict[tuple[int, int], int]:
    assignments: dict[tuple[int, int], int] = {}
    focus_color = rng.choice(marker_types)
    forced_cells = [(0, 0), (1, 0)]
    for cell in forced_cells:
        assignments[cell] = int(focus_color)

    remaining = [cell for cell in occupied_cells if cell not in assignments]
    rng.shuffle(remaining)

    other_types = [int(color) for color in marker_types if int(color) != int(focus_color)]
    for color in other_types:
        if not remaining:
            break
        assignments[remaining.pop()] = int(color)

    for cell in remaining:
        assignments[cell] = int(rng.choice(marker_types))
    return assignments


def _generate_case(rng: random.Random) -> tuple[np.ndarray, np.ndarray, dict[str, int]]:
    rows = rng.randint(3, 5)
    cols = rng.randint(3, 5)

    row_step = rng.randint(4, 6)
    col_steps = [rng.randint(4, 6) for _ in range(cols - 1)]
    row_span = (rows - 1) * row_step
    col_span = sum(col_steps)

    h = rng.randint(max(10, row_span + 3), row_span + 8)
    w = rng.randint(max(10, col_span + 3), col_span + 8)
    top = _pick_origin(rng, h - 1 - row_span)
    left = _pick_origin(rng, w - 1 - col_span)

    row_positions = [top + i * row_step for i in range(rows)]
    col_positions = [left]
    for step in col_steps:
        col_positions.append(col_positions[-1] + step)

    occ = _build_occupancy(rng, rows, cols)
    assert bool(np.all(occ.any(axis=1)))
    assert bool(np.all(occ.any(axis=0)))
    occupied_cells = [(r, c) for r in range(rows) for c in range(cols) if bool(occ[r, c])]
    marker_types = rng.sample(sorted(GLYPHS), k=rng.randint(2, 4))
    assignments = _assign_marker_colors(rng, occupied_cells, marker_types)

    inp = np.zeros((h, w), dtype=int)
    real_markers: list[tuple[int, int, int]] = []
    for rr, cc in occupied_cells:
        r = row_positions[rr]
        c = col_positions[cc]
        marker_color = assignments[(rr, cc)]
        inp[int(r), int(c)] = int(marker_color)
        real_markers.append((int(r), int(c), int(marker_color)))

    if rng.random() < 0.65:
        _place_legend(rng, inp, rng.choice(sorted({mc for _r, _c, mc in real_markers})))

    expected = _render_markers(inp.shape, real_markers, mirrored=True, centered=True)
    metrics = _behavior_metrics(inp)

    assert metrics["odd_row_markers"] > 0
    assert metrics["visible_flip_markers"] > 0
    assert metrics["flip_diff_cells"] > 0
    assert metrics["center_diff_cells"] > 0
    assert assignments[(0, 0)] == assignments[(1, 0)]
    return inp, expected, metrics


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(500):
        inp, expected, metrics = _generate_case(rng)
        got = main(inp)
        if np.array_equal(got, expected) and metrics["flip_diff_cells"] > 0 and metrics["center_diff_cells"] > 0:
            return inp
    raise RuntimeError("failed to generate a valid centered mirrored-glyph sample")
