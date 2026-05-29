# Combo variant of dfadab01 (index 8)
# combo_spec: D7/1+D8/1
# Name: 对角线排列+邻近色决定字形色
"""
design_analysis:
  concretization:
    preserved:
      - 四种输入标记色 2/3/5/8 仍然各自绑定一个固定的 4x4 字形轮廓。
      - 输出仍以标记像素作为字形左上角来绘制；solver 也仍兼容原题可能出现的 old-style legend 标签像素。
      - 黑底、单像素控制标记、4x4 裁剪绘制这些原题骨架保持不变。
    changed:
      - D7/1: 真正的控制标记不再形成规则矩形点阵，而是全部落在同一条主对角线或反对角线上，表现为阶梯状链条。
      - D8/1: 输出颜色不再由标记自身决定，而是由该标记沿这条对角链最近的另一个真实标记的颜色决定；
        再通过原始映射 2->4, 3->1, 5->6, 8->7 转成实际绘制色。
      - 因为对角链上的 Manhattan 距离等于链上步长的两倍，最近邻可直接从前后相邻台阶的间距比较中读出。
    trace_example:
      input_chain:
        - 主对角线上有四个标记，位置标量为 [0, 4, 9, 15]，对应颜色 [2, 3, 2, 5]。
      original:
        - 四个位置都会直接按自身颜色映射成 [4, 1, 4, 6]，形状和颜色都由自己决定。
      combo:
        - 第 1 个标记只靠近第 2 个，所以画 2 的轮廓但涂成 3 的映射色 1。
        - 第 2 个标记离第 1 个更近，所以画 3 的轮廓但涂成 2 的映射色 4。
        - 第 3 个标记离第 2 个更近，所以它也是 2 的轮廓，但这次涂成 3 的映射色 1。
        - 第 4 个标记只靠近第 3 个，所以画 5 的轮廓但涂成 2 的映射色 4。
      difference:
        - 原题是“自己的颜色同时决定轮廓和输出色”。
        - 融合后是“自己的颜色决定轮廓，对角链上最近邻的颜色决定输出色”。
  complexity_calibration:
    - 若所有台阶间距都相等，则中间标记会同时拥有两个等距邻居，规则容易退化成隐藏 tie-break；
      因此生成器只使用 4/5/6 的非均匀步长模板，并强制每个内部点至少一侧更近。
    - 若所有重复形状收到同一种邻近色，视觉上又会像原题固定配色；因此每个样例都强制出现
      “同一种标记色重复两次，但它们最近邻颜色不同”的现象。
    - 标记数量固定为 4，既足够出现重复形状与多种传色，又不会把一维链关系做得过乱。
  implicit_assumption_review:
    original_main:
      - 原 `GLYPHS` 把轮廓和输出色绑在一起；融合后必须拆成 `GLYPH_MASKS` 与 `PAINT_FROM_MARKER` 两层。
      - 原循环默认每个彩色像素都能独立扩张；现在必须先剔除 legend 标签，再整体识别对角链。
      - 原 legend 检测只需检查“该标签色对应的那一种字形色”；融合后 legend 样例若存在，其 4x4 字形色
        可能来自邻近标记，因此要接受任意合法字形色 1/4/6/7。
    new_main:
      - 所有真实标记必须共线于一条主对角线或反对角线；生成器显式保证这一点。
      - 对角链按行排序后，相邻标记的行差必须等于列差绝对值，这样链上相邻间隔才是可见且可比较的。
      - 若某个内部标记左右两侧间距完全相等，solver 只接受两侧邻居颜色一致的情形；生成器直接避免这种平局。
      - 至少要有 2 个真实标记，否则“最近另一个标记”无定义；生成器固定放 4 个。
  placement_legality:
    - 画布固定为 20x20，标记的左上角限制在 0..16，使 4x4 字形完整落在画布内，不产生额外裁剪噪声。
    - 相邻台阶在行列两个方向上都至少相隔 4，因此 4x4 扩张后的字形 bbox 不会互相重叠。
    - 主对角线使用 `(r0+s, c0+s)`，反对角线使用 `(r0+s, c0-s)`，并根据模板最大标量选择安全偏移量。
  observability_and_cleanliness:
    - 输入一眼能看出“所有控制点排成一条对角阶梯”；输出则会出现同形异色，迫使解题者放弃原题的固定配色假设。
    - 最近邻只由可数的几何距离决定，不依赖颜色 id 排序、扫描顺序或其他不可视因素。
    - 生成器不放 legend，避免静态样例字形与动态传色规则混杂。
  generator_checks:
    - output-first: 先采样对角链位置与颜色，再用同一套链式规则渲染 expected，并验证 `main(input) == expected`。
    - 退化指标:
      - changed_fraction = donor 色与自身色不同的标记占比，要求至少 1/2。
      - repeated_shape_multicolor = 存在一种标记色重复出现且对应 donor 颜色至少两种，要求为真。
      - diagonal_integrity = 所有真实标记满足同一条主对角线或反对角线，要求为真。
"""

from __future__ import annotations

import random

import numpy as np


GLYPH_MASKS: dict[int, np.ndarray] = {
    2: np.array(
        [
            [1, 1, 1, 1],
            [1, 0, 0, 1],
            [1, 0, 0, 1],
            [1, 1, 1, 1],
        ],
        dtype=bool,
    ),
    3: np.array(
        [
            [0, 1, 1, 0],
            [1, 0, 0, 1],
            [1, 0, 0, 1],
            [0, 1, 1, 0],
        ],
        dtype=bool,
    ),
    5: np.array(
        [
            [1, 1, 0, 0],
            [1, 1, 0, 0],
            [0, 0, 1, 1],
            [0, 0, 1, 1],
        ],
        dtype=bool,
    ),
    8: np.array(
        [
            [1, 0, 0, 1],
            [0, 1, 1, 0],
            [0, 1, 1, 0],
            [1, 0, 0, 1],
        ],
        dtype=bool,
    ),
}

PAINT_FROM_MARKER: dict[int, int] = {
    2: 4,
    3: 1,
    5: 6,
    8: 7,
}

MARKER_COLORS = tuple(GLYPH_MASKS.keys())
GLYPH_COLORS = tuple(PAINT_FROM_MARKER.values())
GRID_SIZE = 20
MAX_TOP_LEFT = GRID_SIZE - 4

# Strictly increasing chain coordinates; non-uniform gaps keep nearest neighbors unique.
CHAIN_TEMPLATES: tuple[tuple[int, ...], ...] = (
    (0, 4, 9, 15),
    (0, 6, 10, 15),
    (1, 5, 11, 15),
    (0, 4, 10, 15),
)


def _has_source_glyph(g: np.ndarray, r0: int, c0: int, color: int, mask: np.ndarray) -> bool:
    h, w = g.shape
    if r0 < 0 or c0 < 0 or r0 + 4 > h or c0 + 4 > w:
        return False
    patch = g[r0 : r0 + 4, c0 : c0 + 4]
    return bool(np.all(patch[mask] == color) and np.all(patch[~mask] != color))


def _is_legend_label(g: np.ndarray, r: int, c: int, marker_color: int) -> bool:
    mask = GLYPH_MASKS[marker_color]
    for glyph_color in GLYPH_COLORS:
        if _has_source_glyph(g, r - 4, c - 4, glyph_color, mask):
            return True
    return False


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


def _collect_real_markers(g: np.ndarray) -> list[tuple[int, int, int]]:
    markers: list[tuple[int, int, int]] = []
    for r, c in np.argwhere(np.isin(g, MARKER_COLORS)):
        rr = int(r)
        cc = int(c)
        marker_color = int(g[rr, cc])
        if _is_legend_label(g, rr, cc, marker_color):
            continue
        markers.append((rr, cc, marker_color))
    return markers


def _diagonal_kind(markers: list[tuple[int, int, int]]) -> str:
    diffs = {r - c for r, c, _ in markers}
    if len(diffs) == 1:
        return "main"
    sums = {r + c for r, c, _ in markers}
    if len(sums) == 1:
        return "anti"
    raise AssertionError("real markers must form one diagonal staircase")


def _sorted_diagonal_markers(markers: list[tuple[int, int, int]]) -> tuple[str, list[tuple[int, int, int]]]:
    kind = _diagonal_kind(markers)
    ordered = sorted(markers, key=lambda item: (item[0], item[1]))
    for (r0, c0, _), (r1, c1, _) in zip(ordered, ordered[1:], strict=False):
        gap = r1 - r0
        assert gap > 0, "diagonal chain cannot reuse a row"
        if kind == "main":
            assert c1 - c0 == gap, "main diagonal markers must rise with equal row/col step"
        else:
            assert c0 - c1 == gap, "anti-diagonal markers must rise with opposite col step"
    return kind, ordered


def _donor_marker_colors(ordered_markers: list[tuple[int, int, int]]) -> list[int]:
    assert len(ordered_markers) >= 2
    donor_colors: list[int] = []
    rows = [r for r, _, _ in ordered_markers]
    for i, (_, _, marker_color) in enumerate(ordered_markers):
        if i == 0:
            donor_colors.append(ordered_markers[1][2])
            continue
        if i == len(ordered_markers) - 1:
            donor_colors.append(ordered_markers[-2][2])
            continue
        prev_gap = rows[i] - rows[i - 1]
        next_gap = rows[i + 1] - rows[i]
        assert prev_gap >= 4 and next_gap >= 4
        if prev_gap < next_gap:
            donor_colors.append(ordered_markers[i - 1][2])
        elif next_gap < prev_gap:
            donor_colors.append(ordered_markers[i + 1][2])
        else:
            prev_color = ordered_markers[i - 1][2]
            next_color = ordered_markers[i + 1][2]
            assert prev_color == next_color, "equal diagonal gaps require same-color donors"
            donor_colors.append(prev_color)
        assert marker_color in MARKER_COLORS
    return donor_colors


def _render_output(shape: tuple[int, int], markers: list[tuple[int, int, int]]) -> np.ndarray:
    out = np.zeros(shape, dtype=int)
    _, ordered_markers = _sorted_diagonal_markers(markers)
    donor_colors = _donor_marker_colors(ordered_markers)
    for (r, c, marker_color), donor_marker_color in zip(ordered_markers, donor_colors, strict=True):
        _paint_clipped(out, r, c, PAINT_FROM_MARKER[donor_marker_color], GLYPH_MASKS[marker_color])
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Expand each marker into its fixed glyph shape, with color borrowed from the nearest step on a diagonal chain.

    Input structure:
    - Black background with single-pixel markers in colors 2, 3, 5, and/or 8.
    - The real markers form one staircase along either a main diagonal or an anti-diagonal.
    - Some inputs may still contain an old legend: a 4x4 sample glyph plus a matching label marker one step down-right from that glyph.

    Rule:
    - A marker's own color chooses which 4x4 glyph mask to draw.
    - Ignore legend label markers, sort the real markers along their common diagonal, and for each marker compare the two adjacent diagonal gaps.
    - The nearest other marker donates the output color through the original palette map 2->4, 3->1, 5->6, 8->7; then paint that glyph at the marker's top-left corner.

    Key insight:
    - The diagonal arrangement is not decorative: it turns the scene into a one-dimensional staircase, so each glyph keeps its own shape but inherits color from the closest neighboring step rather than from itself.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    assert set(np.unique(g)).issubset({0, *MARKER_COLORS, *GLYPH_COLORS})

    markers = _collect_real_markers(g)
    assert len(markers) >= 2, "need at least two real markers for neighbor-color transfer"
    return _render_output(g.shape, markers)


def _build_chain_positions(
    scalars: tuple[int, ...],
    main_diagonal: bool,
    row_offset: int,
    col_offset: int,
) -> list[tuple[int, int]]:
    positions: list[tuple[int, int]] = []
    for s in scalars:
        if main_diagonal:
            pos = (row_offset + s, col_offset + s)
        else:
            pos = (row_offset + s, col_offset - s)
        positions.append(pos)
    assert len(set(positions)) == len(positions)
    assert all(0 <= r <= MAX_TOP_LEFT and 0 <= c <= MAX_TOP_LEFT for r, c in positions)
    return positions


def _donor_indices_from_scalars(scalars: tuple[int, ...]) -> list[int]:
    donors: list[int] = []
    for i in range(len(scalars)):
        if i == 0:
            donors.append(1)
            continue
        if i == len(scalars) - 1:
            donors.append(len(scalars) - 2)
            continue
        prev_gap = scalars[i] - scalars[i - 1]
        next_gap = scalars[i + 1] - scalars[i]
        assert prev_gap >= 4 and next_gap >= 4
        if prev_gap < next_gap:
            donors.append(i - 1)
        elif next_gap < prev_gap:
            donors.append(i + 1)
        else:
            raise AssertionError("generator templates must avoid equal nearest-neighbor gaps")
    return donors


def _sample_colors(donor_indices: list[int]) -> list[int]:
    n = len(donor_indices)
    for _ in range(400):
        colors = [random.choice(MARKER_COLORS) for _ in range(n)]
        if len(set(colors)) < 2:
            continue
        changed = sum(colors[i] != colors[donor_indices[i]] for i in range(n))
        if changed * 2 < n:
            continue
        repeated_shape_multicolor = False
        for own_color in MARKER_COLORS:
            idxs = [i for i, color in enumerate(colors) if color == own_color]
            if len(idxs) < 2:
                continue
            donor_colors = {colors[donor_indices[i]] for i in idxs}
            if len(donor_colors) >= 2:
                repeated_shape_multicolor = True
                break
        if not repeated_shape_multicolor:
            continue
        paint_colors = {PAINT_FROM_MARKER[colors[donor_indices[i]]] for i in range(n)}
        if len(paint_colors) < 2:
            continue
        return colors
    raise RuntimeError("failed to sample a non-degenerate diagonal color assignment")


def generate_input() -> np.ndarray:
    for _ in range(200):
        scalars = random.choice(CHAIN_TEMPLATES)
        main_diagonal = bool(random.getrandbits(1))
        max_scalar = max(scalars)
        row_offset = random.randint(0, MAX_TOP_LEFT - max_scalar)
        if main_diagonal:
            col_offset = random.randint(0, MAX_TOP_LEFT - max_scalar)
        else:
            col_offset = random.randint(max_scalar, MAX_TOP_LEFT)
        positions = _build_chain_positions(scalars, main_diagonal, row_offset, col_offset)
        donor_indices = _donor_indices_from_scalars(scalars)
        colors = _sample_colors(donor_indices)

        grid = np.zeros((GRID_SIZE, GRID_SIZE), dtype=int)
        markers: list[tuple[int, int, int]] = []
        for (r, c), marker_color in zip(positions, colors, strict=True):
            grid[r, c] = marker_color
            markers.append((r, c, marker_color))

        expected = _render_output(grid.shape, markers)
        actual = main(grid)
        assert np.array_equal(actual, expected)
        return grid

    raise RuntimeError("failed to construct a valid diagonal-transfer dfadab01 input")
