# Combo variant of dfadab01 (index 13)
# combo_spec: D2/0+D8/0
# Name: 标记色保持+固定偏移色
from __future__ import annotations

import random

import numpy as np


"""
design_analysis:
  concretization_and_trace:
    - 原题里每种 marker 同时决定两件事：4x4 字形轮廓，以及固定的输出颜色。
      例如 marker 3 总是画成蓝色菱形环，marker 8 总是画成橙色 X。
    - 本融合版把这两件事拆开：
      - shape 仍由 marker 色决定，沿用原题的颜色->轮廓字典。
      - paint color 改成 `(marker_color + 2) % 10`。
    - 具体小例子：
      - 输入某处有一个 `3`。原题会画颜色 `1` 的菱形环；现在会画同一个菱形环，
        但颜色变成 `5`。
      - 输入某处有一个 `5`。原题会画颜色 `6` 的对角块；现在仍画对角块，
        但颜色变成 `7`。
      - 输入某处有一个 `8`。原题会画颜色 `7` 的 X；现在颜色变成 `0`，
        所以 X 虽然“逻辑上存在”，但在黑底上完全不可见。
    - 差异集中在 `marker_color -> glyph_color` 这一环，`marker_color -> glyph_mask`
      保持原样。
  complexity_calibration:
    - 不能退化成“看起来还是原题”的样本：因为 marker `2` 恰好仍映射到颜色 `4`，
      如果样本只含 2，融合行为就不可见。
    - 生成器因此强制每题都包含 `8`，以及至少一种 `{3, 5}`。
      这样同一题里既能看到“可见的改色”，也能看到“8 变黑后消失”的特殊情况。
    - 颜色种类仍限制在原题 4 种 marker，视觉复杂度没有失控；输出可见色只会来自
      `{4, 5, 7}`，再加上黑色背景，规律足够集中。
  implicit_assumption_review:
    - 原 `GLYPHS` 把 `(glyph_color, mask)` 绑在一起；融合后这个前提失效，
      必须拆成“固定 mask”与“按公式算出的输出色”。
    - 原 legend 检测依赖“label 色唯一决定 sample 色”；这里依然可以用
      `label -> (label+2)%10` 检测可见 legend，但 marker `8` 的 sample 色是黑色，
      输入里将不可见，因此 solver 不能从空白里恢复出这种 legend。
    - 原扫描 `g == marker_color` 在本题里更危险，因为颜色 `5` 既可能是 marker，
      也可能是 marker `3` 的 sample glyph 颜色。为避免把 sample 像素误当 marker，
      solver 先收集整块 legend 占用区域，再从 marker 候选里剔除。
  placement_legality:
    - 生成器使用 20x20 画布和 2x3 到 5x5 的规则 lattice，行列间距只取 4 或 5，
      所以所有 4x4 展开彼此不重叠。
    - 起始行列允许贴边，因而仍保留原题的裁剪现象；但 spacing >= 4 保证即使裁剪，
      不同 marker 的理论 4x4 bbox 也不会互相覆盖。
    - 为了避免“不可见的黑色 legend”破坏可观察性，生成器不放 legend。
      main() 仍支持跳过可见 legend（即输出色非黑时的 sample+label）。
  observability_and_tidiness:
    - 解题者在少量样例里能稳定看到三件事：
      - 相同 marker 仍对应相同几何形状。
      - `3` 和 `5` 的输出色不再是原题颜色。
      - `8` marker 在输出中系统性消失。
    - 输入只保留稀疏单像素 marker，没有 sample 区和主区域叠压，画面比带黑色 legend
      的版本更整洁。
  fusion_metrics:
    - `shifted_ratio`: 输出颜色与原题固定配色不同的 marker 占比；生成器要求至少 0.5。
    - `invisible_count`: marker `8` 的个数；生成器要求至少 1。
    - `visible_shift_count`: 属于 `{3, 5}` 的 marker 个数；生成器要求至少 1。
  new_main_assumptions:
    - 若输入含 legend，则它只能是“可见 legend”；即 sample 色不能是黑色。
    - 所有真实 marker 仍是单像素，且与其他真实 marker 的间距至少为 4；
      generate_input() 通过规则 lattice 保证这一点。
    - 不存在使用颜色 `1` 或 `6` 的旧题 sample；本融合版输入只允许
      `{0, 2, 3, 4, 5, 7, 8}`。
"""


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

MARKER_COLORS: tuple[int, ...] = tuple(GLYPH_MASKS)
ORIGINAL_GLYPH_COLORS: dict[int, int] = {2: 4, 3: 1, 5: 6, 8: 7}
GRID_SIZE = 20


def _shift_color(marker_color: int) -> int:
    return (marker_color + 2) % 10


VISIBLE_GLYPH_COLORS = tuple(sorted({_shift_color(color) for color in MARKER_COLORS if _shift_color(color) != 0}))
ALLOWED_COLORS = {0, *MARKER_COLORS, *VISIBLE_GLYPH_COLORS}


def _has_source_glyph(g: np.ndarray, r0: int, c0: int, color: int, mask: np.ndarray) -> bool:
    if color == 0:
        return False
    h, w = g.shape
    if r0 < 0 or c0 < 0 or r0 + 4 > h or c0 + 4 > w:
        return False
    patch = g[r0 : r0 + 4, c0 : c0 + 4]
    return bool(np.all(patch[mask] == color) and np.all(patch[~mask] != color))


def _collect_legend_cells(g: np.ndarray) -> set[tuple[int, int]]:
    legend_cells: set[tuple[int, int]] = set()
    for marker_color, mask in GLYPH_MASKS.items():
        legend_color = _shift_color(marker_color)
        if legend_color == 0:
            continue
        for r, c in np.argwhere(g == marker_color):
            rr = int(r)
            cc = int(c)
            if not _has_source_glyph(g, rr - 4, cc - 4, legend_color, mask):
                continue
            for dr in range(4):
                for dc in range(4):
                    legend_cells.add((rr - 4 + dr, cc - 4 + dc))
            legend_cells.add((rr, cc))
    return legend_cells


def _collect_real_markers(g: np.ndarray) -> list[tuple[int, int, int]]:
    legend_cells = _collect_legend_cells(g)
    markers: list[tuple[int, int, int]] = []
    for r, c in np.argwhere(np.isin(g, MARKER_COLORS)):
        rr = int(r)
        cc = int(c)
        if (rr, cc) in legend_cells:
            continue
        markers.append((rr, cc, int(g[rr, cc])))
    return markers


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


def _render_output(shape: tuple[int, int], markers: list[tuple[int, int, int]]) -> np.ndarray:
    out = np.zeros(shape, dtype=int)
    for r, c, marker_color in markers:
        _paint_clipped(out, r, c, _shift_color(marker_color), GLYPH_MASKS[marker_color])
    return out


def _fusion_metrics_from_markers(markers: list[tuple[int, int, int]]) -> dict[str, float]:
    total = len(markers)
    assert total > 0
    invisible_count = sum(marker_color == 8 for _, _, marker_color in markers)
    visible_shift_count = sum(marker_color in (3, 5) for _, _, marker_color in markers)
    shifted_count = sum(_shift_color(marker_color) != ORIGINAL_GLYPH_COLORS[marker_color] for _, _, marker_color in markers)
    distinct_visible_output_colors = len({_shift_color(marker_color) for _, _, marker_color in markers if _shift_color(marker_color) != 0})
    return {
        "marker_count": float(total),
        "invisible_count": float(invisible_count),
        "visible_shift_count": float(visible_shift_count),
        "shifted_ratio": shifted_count / total,
        "distinct_visible_output_colors": float(distinct_visible_output_colors),
    }


def fusion_metrics(grid: np.ndarray) -> dict[str, float]:
    g = np.asarray(grid, dtype=int)
    return _fusion_metrics_from_markers(_collect_real_markers(g))


def main(grid: np.ndarray) -> np.ndarray:
    """
    Replace each marker by its original 4x4 glyph shape, but recolor that glyph with a fixed +2 shift.

    Input structure:
    - Black background with single-pixel markers in colors 2, 3, 5, and/or 8.
    - Some grids may also include an optional visible legend entry: a 4x4 sample glyph in the shifted output color, with its label marker one cell below-right of the sample.

    Rule:
    - A marker's color still selects which 4x4 glyph mask to draw, using the same shape dictionary as the base task.
    - The glyph's fill color is no longer read from a fixed lookup table; it is `(marker_color + 2) mod 10`.
    - Paint the glyph with its top-left corner at the marker position, clip anything beyond the grid edge, and remove any visible legend sample plus its label marker from the output.

    Key insight:
    - Shape and color have been decoupled: the marker color still names the geometry, while the displayed color is a uniform arithmetic offset, causing color 8 markers to disappear into the black background.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    assert set(map(int, np.unique(g))).issubset(ALLOWED_COLORS)

    markers = _collect_real_markers(g)
    assert markers, "expected at least one real marker"
    return _render_output(g.shape, markers)


def _sample_positions() -> list[tuple[int, int]]:
    for _ in range(200):
        rows = random.randint(2, 5)
        cols = random.randint(3, 5)
        if rows * cols < 6:
            continue
        row_spacing = random.randint(4, 5)
        col_spacing = random.randint(4, 5)
        max_row_start = GRID_SIZE - 1 - row_spacing * (rows - 1)
        max_col_start = GRID_SIZE - 1 - col_spacing * (cols - 1)
        if max_row_start < 0 or max_col_start < 0:
            continue
        start_r = random.randint(0, max_row_start)
        start_c = random.randint(0, max_col_start)
        row_positions = [start_r + i * row_spacing for i in range(rows)]
        col_positions = [start_c + i * col_spacing for i in range(cols)]
        return [(r, c) for r in row_positions for c in col_positions]
    raise RuntimeError("failed to sample a legal lattice")


def _sample_used_colors() -> list[int]:
    used_count = random.choices([3, 4], weights=[3, 2], k=1)[0]
    used_colors = {8, random.choice([3, 5])}
    remaining = [color for color in MARKER_COLORS if color not in used_colors]
    while len(used_colors) < used_count:
        used_colors.add(random.choice(remaining))
        remaining = [color for color in MARKER_COLORS if color not in used_colors]
    return sorted(used_colors)


def _sample_marker_colors(fill_count: int, used_colors: list[int]) -> list[int]:
    assert fill_count >= 2 * len(used_colors)
    colors = [color for color in used_colors for _ in range(2)]
    extras_needed = fill_count - len(colors)
    if extras_needed > 0:
        weights = {2: 2, 3: 3, 5: 3, 8: 1}
        population = [color for color in used_colors for _ in range(weights[color])]
        colors.extend(random.choices(population, k=extras_needed))
    random.shuffle(colors)
    return colors


def generate_input() -> np.ndarray:
    for _ in range(400):
        cells = _sample_positions()
        used_colors = _sample_used_colors()
        if len(cells) < 2 * len(used_colors):
            continue
        min_fill = max((3 * len(cells) + 4) // 5, 2 * len(used_colors))
        if min_fill > len(cells):
            continue
        fill_count = random.randint(min_fill, len(cells))
        chosen_cells = random.sample(cells, fill_count)
        marker_colors = _sample_marker_colors(fill_count, used_colors)

        grid = np.zeros((GRID_SIZE, GRID_SIZE), dtype=int)
        markers: list[tuple[int, int, int]] = []
        for (r, c), marker_color in zip(chosen_cells, marker_colors, strict=True):
            grid[r, c] = marker_color
            markers.append((r, c, marker_color))

        metrics = _fusion_metrics_from_markers(markers)
        if metrics["invisible_count"] < 1:
            continue
        if metrics["visible_shift_count"] < 1:
            continue
        if metrics["shifted_ratio"] < 0.5:
            continue
        if metrics["distinct_visible_output_colors"] < 2:
            continue

        expected = _render_output(grid.shape, markers)
        actual = main(grid)
        assert np.array_equal(actual, expected)
        return grid

    raise RuntimeError("failed to construct a non-degenerate combo input")
