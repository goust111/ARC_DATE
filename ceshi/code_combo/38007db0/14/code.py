# Combo variant of 38007db0 (index 14)
# combo_spec: D6/0+D7/2
# Name: 混合尺寸渐变差异

from __future__ import annotations

"""
design_analysis:
  variant_goal:
    keep:
      - 任务核心仍是“每个横向格子行里找出唯一不同的 cell，并抽到单列输出”
      - 颜色角色仍固定为 lattice/bg/sprite 三色
    change:
      - D6/0: 不同行允许不同 cell 尺寸，且竖向格线只在本行 band 内出现
      - D7/2: 第 1 行 unique 相对本行 base 恰好多 1 个像素翻转，第 2 行多 2 个，依次递增
    rationale:
      - solver 不能再依赖全局统一列切分
      - solver 在每行找到 odd-one-out 之后，还应能看出 top-down 的差异计数梯度

  a_concretization_and_trace:
    concrete_pixels:
      - 顶部 band 可能是 5x5 cell，只有这一条带里有对齐的局部竖线；odd cell 比其余 cell 多 1 个翻转像素
      - 第二个 band 可能是 6x6 cell，局部竖线更稀疏；odd cell 比本行多数 cell 多 2 个翻转像素
      - 第三个 band 再次可能回到 5x5，但 odd cell 这次与本行 base 相差 3 像素
      - 输出仍是这些 odd cell 自上而下堆成单列，保留各 band 高度
    trace_example:
      - 原题: 先找全图 uniform rows/cols，再用同一个 cell_w 切整张图
      - D6 后: 只能全局找横线；列切分必须在每个 band 内重算，因为 5x5 与 6x6 的局部竖线不会全图对齐
      - D7 后: 对每个 band 找到多数 cell 与唯一 cell 后，可直接比较两者像素差，得到 1,2,3,... 的递增序列
      - 融合差异: “怎么切 cell” 与 “怎么验证 odd cell” 不再独立；只有先做 band-local 切分，才能稳定测出每行递增的差异像素数

  b_complexity_calibration:
    - 尺寸只允许 5 和 6 两档，肉眼能看出大小变化，但不会引入连续尺度爆炸
    - 行数保持 3-4，因此递增差异只会落在 1-4，既明显又不会把 3x3 / 4x4 inner area 用满
    - 列数保持 3-4，多数项始终至少有 2 个副本，odd-one-out 无歧义
    - 每题强制同时出现两种尺寸，避免退化回原题

  c_implicit_assumption_audit:
    old_main_dependencies:
      - 全图 uniform columns 就是竖向 lattice lines
      - 所有 c_slices 宽度相同，因此只需一个 global cell_w
      - 输出宽度可以直接写成 cell_w + 2
      - unique 与 base 的差异计数只需落在 1-3 的松约束内
    why_invalid_now:
      - 混合尺寸时，竖线只在 band 内完整出现，全图 uniform column 往往只剩 0 个
      - 不同行 cell_w 可能不同，global cell_w 会把其它 band 切坏
      - 输出宽度必须取所有抽取 cell 的最大 strip 宽度
      - 4 行样例时，梯度差异必须允许到 4，而不是原始生成器的 1-3
    new_main_assumptions:
      - 只有水平分隔线必须横跨全图且为纯 lattice 色
      - band 内完整为 lattice 的列就是该 band 的局部竖线
      - 同一 band 内所有 cells 尺寸一致且为正方形，所以 cell slice 的宽度等于 band 高度
      - 每行恰有一个 minority signature，其余 cells 全部相同
      - 从上到下 unique-vs-base 的像素差必须正好是 1,2,3,...,R

  d_placement_legality:
    - 总高度由各 band 高度之和再加上下边界横线组成，不会越界
    - 每个 band 的局部 strip 宽度单独计算，再放进全图最大宽度内
    - 局部竖线只画在对应 band 的上下横线之间，互不重叠
    - cell 通过 paste 放入各自 slice，始终完整落在 lattice 框内

  e_visual_observability:
    - 人类能直接看到有的 band 更高更宽，有的 band 更矮更窄
    - 每行里多数 cell 完全相同，odd cell 通过与同排其它 cell 对比即可看出
    - 若顺着行往下看，odd cell 相对本行 base 的改动会逐步从 1 像素增加到 2、3、4 像素

  f_cleanliness:
    - 只保留 3 种颜色，不叠加额外标记或计数提示
    - band 之间由整行 lattice 横线分隔，局部竖线只服务本 band，视觉分层清楚
    - 左右剩余区域仅用纯 bg 留白，不制造额外伪对象

  g_inferability:
    - 变化不依赖颜色数值大小或坐标排序之外的隐藏元信息
    - 递增模式由“每行 odd cell 相对同排多数 cell 的差异越来越多”直接体现
    - 即便解题者只抓住 odd-one-out 主规则，输出也能推得出来；梯度模式作为额外稳定结构，避免生成退化成普通混合尺寸题

  generator_contract_for_solver:
    - 行数 3-4，列数 3-4
    - 至少一行 5x5，至少一行 6x6
    - 同一行内所有 non-unique cells 完全相同，且恰有 1 个 unique cell
    - 第 i 行 unique 相对本行 base 恰好翻转 i 个 inner pixels（i 从 1 开始）
    - 输出为单列 stacked lattice，宽度取被抽取 cells 中最大者

  post_edit_main_audit:
    - _find_lattice_and_row_slices 假设存在全宽纯色横线；generator 用 full-width row lines 保证
    - _extract_row_cells 假设 band 内局部竖线贯穿整段高度，且 square cell 宽=高；generator 的 stacked builder 保证
    - _pick_unique_cell 假设每行 signature 分布是 (N-1) 个相同 + 1 个不同；generator 逐行直接构造
    - main 末尾的梯度断言假设 diff_counts == [1..R]；generator 按行号精确翻转像素数来保证
"""

from collections import Counter
from pathlib import Path
import math
import sys

import numpy as np

SOLVER_DIR = Path(__file__).resolve().parents[4] / "solvers"
if str(SOLVER_DIR) not in sys.path:
    sys.path.append(str(SOLVER_DIR))

from arc_utils import paste


def _find_lattice_and_row_slices(g: np.ndarray) -> tuple[int, list[tuple[int, int]]]:
    h, _ = g.shape
    uniform_rows = [r for r in range(h) if len(set(map(int, g[r, :]))) == 1]
    assert uniform_rows
    lattice = int(Counter(int(g[r, 0]) for r in uniform_rows).most_common(1)[0][0])
    row_lines = [int(r) for r in uniform_rows if int(g[r, 0]) == int(lattice)]
    row_slices = [(row_lines[i] + 1, row_lines[i + 1]) for i in range(len(row_lines) - 1)]
    row_slices = [(int(a), int(b)) for a, b in row_slices if int(b) > int(a)]
    assert row_slices
    return int(lattice), row_slices


def _extract_row_cells(g: np.ndarray, lattice: int, ra: int, rb: int) -> list[np.ndarray]:
    band = g[int(ra) : int(rb), :]
    band_h = int(rb - ra)
    line_cols = [c for c in range(g.shape[1]) if np.all(band[:, c] == int(lattice))]
    assert len(line_cols) >= 4
    col_slices = [(line_cols[i] + 1, line_cols[i + 1]) for i in range(len(line_cols) - 1)]
    col_slices = [(int(a), int(b)) for a, b in col_slices if int(b) - int(a) == int(band_h)]
    assert len(col_slices) >= 3
    return [band[:, int(ca) : int(cb)].copy() for ca, cb in col_slices]


def _pick_unique_cell(cells: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray, int]:
    sigs = [cell.tobytes() for cell in cells]
    counts = Counter(sigs)
    majority_sig, majority_count = counts.most_common(1)[0]
    assert int(majority_count) == len(cells) - 1
    assert len(counts) == 2
    unique_indices = [i for i, sig in enumerate(sigs) if int(counts[sig]) == 1]
    assert len(unique_indices) == 1
    base_idx = next(i for i, sig in enumerate(sigs) if sig == majority_sig)
    base_cell = cells[int(base_idx)]
    unique_cell = cells[int(unique_indices[0])]
    diff_count = int(np.count_nonzero(base_cell != unique_cell))
    assert diff_count > 0
    return base_cell, unique_cell, diff_count


def _build_stacked_lattice(rows: list[list[np.ndarray]], lattice: int, bg: int) -> np.ndarray:
    assert rows and all(row for row in rows)
    row_sizes = [int(row[0].shape[0]) for row in rows]
    strip_widths = [int(len(row) * (size + 1) + 1) for row, size in zip(rows, row_sizes)]
    height = int(sum(row_sizes) + len(row_sizes) + 1)
    width = int(max(strip_widths))
    grid = np.full((height, width), int(bg), dtype=int)

    top = 0
    grid[top, :] = int(lattice)
    for row, size, strip_w in zip(rows, row_sizes, strip_widths):
        offset = int((width - strip_w) // 2)
        bottom = int(top + size + 1)
        grid[bottom, :] = int(lattice)
        for j in range(len(row) + 1):
            c = int(offset + j * (size + 1))
            grid[top : bottom + 1, c] = int(lattice)
        for j, cell in enumerate(row):
            paste(grid, cell, top + 1, offset + 1 + j * (size + 1))
        top = bottom
    return grid


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fused solver for D6/0 + D7/2.

    1. Use full-width uniform rows to split the canvas into horizontal bands.
    2. Within each band, find band-local vertical lattice lines and recover square cells.
    3. Find the per-row odd cell by majority vote.
    4. Verify that the unique-vs-base pixel differences increase as 1,2,3,...
    5. Stack the unique cells into a single-column lattice.
    """
    g = np.array(grid, dtype=int)
    lattice, row_slices = _find_lattice_and_row_slices(g)

    bg_candidates = [int(x) for x in g.flatten() if int(x) != int(lattice)]
    assert bg_candidates
    bg = int(Counter(bg_candidates).most_common(1)[0][0])

    chosen_cells: list[np.ndarray] = []
    diff_counts: list[int] = []
    for ra, rb in row_slices:
        cells = _extract_row_cells(g, int(lattice), int(ra), int(rb))
        _, unique_cell, diff_count = _pick_unique_cell(cells)
        chosen_cells.append(unique_cell.copy())
        diff_counts.append(int(diff_count))

    assert diff_counts == list(range(1, len(diff_counts) + 1))
    return _build_stacked_lattice([[cell] for cell in chosen_cells], int(lattice), int(bg))


def _make_cell(inner: np.ndarray, bg: int) -> np.ndarray:
    size = int(inner.shape[0] + 2)
    cell = np.full((size, size), int(bg), dtype=int)
    paste(cell, inner, 1, 1)
    return cell


def _make_base_inner(rng: np.random.Generator, inner_size: int, bg: int, sp: int) -> np.ndarray:
    inner_area = int(inner_size * inner_size)
    min_fill = math.ceil(0.4 * inner_area)
    max_fill = math.floor(0.9 * inner_area)
    fill = int(rng.integers(min_fill, max_fill + 1))
    inner = np.full((inner_size, inner_size), int(bg), dtype=int)
    on_idx = rng.choice(inner_area, size=fill, replace=False)
    inner.flat[on_idx] = int(sp)
    return inner


def _flip_exact_count(
    rng: np.random.Generator,
    base_inner: np.ndarray,
    bg: int,
    sp: int,
    flip_count: int,
) -> np.ndarray:
    area = int(base_inner.size)
    assert 1 <= int(flip_count) <= area
    idxs = rng.choice(area, size=int(flip_count), replace=False)
    variant = base_inner.copy()
    for idx in idxs:
        variant.flat[int(idx)] = int(sp) if int(variant.flat[int(idx)]) == int(bg) else int(bg)
    return variant


def _row_sizes_for_attempt(rng: np.random.Generator, n_rows: int) -> list[int]:
    row_sizes = [5, 6]
    while len(row_sizes) < int(n_rows):
        row_sizes.append(int(rng.choice([5, 6])))
    rng.shuffle(row_sizes)
    assert len(set(row_sizes)) == 2
    return [int(x) for x in row_sizes]


def _gen_attempt() -> np.ndarray | None:
    rng = np.random.default_rng()

    lattice, bg, sp = map(int, rng.choice(10, size=3, replace=False))
    n_rows = int(rng.integers(3, 5))
    n_cols = int(rng.integers(3, 5))
    row_sizes = _row_sizes_for_attempt(rng, int(n_rows))

    base_cells: list[np.ndarray] = []
    unique_cells: list[np.ndarray] = []
    for row_idx, cell_size in enumerate(row_sizes, start=1):
        inner_size = int(cell_size - 2)
        base_inner = _make_base_inner(rng, int(inner_size), int(bg), int(sp))
        unique_inner = _flip_exact_count(rng, base_inner, int(bg), int(sp), int(row_idx))
        base_cells.append(_make_cell(base_inner, int(bg)))
        unique_cells.append(_make_cell(unique_inner, int(bg)))

    input_rows: list[list[np.ndarray]] = []
    for base_cell, unique_cell in zip(base_cells, unique_cells):
        odd_col = int(rng.integers(0, n_cols))
        row = [base_cell.copy() for _ in range(n_cols)]
        row[odd_col] = unique_cell.copy()
        input_rows.append(row)

    inp = _build_stacked_lattice(input_rows, int(lattice), int(bg))
    out = _build_stacked_lattice([[cell.copy()] for cell in unique_cells], int(lattice), int(bg))
    return inp if np.array_equal(main(inp), out) else None


def generate_input() -> np.ndarray:
    for _ in range(300):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input after 300 attempts")
