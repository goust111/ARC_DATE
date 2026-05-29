# Variant of 38007db0
# Dimension: D6 — 格子结构参数：单元格尺寸 (meta)
# Change: 所有单元格为统一的5x5或6x6正方形... → 混合尺寸行

from __future__ import annotations

"""
design_analysis:
  variant_goal:
    keep: 每个横向格子行里都只有 1 个奇异 cell，输出仍然是把每行奇异 cell 抽成单列
    change: 不再要求整张图所有 cell 同尺寸；现在每个横向条带自己决定 cell size in {5, 6}
    rationale: 方格仍是正方形，但只在“同一行内”统一，跨行允许 5x5/6x6 混合

  a_concretization_and_trace:
    input_pixels:
      - 保留 1px lattice color 的横线，并让这些横线横跨整张画布，继续作为分行依据
      - 每个横向条带内部是一个局部格子行；竖线只在该条带内部出现，不再全图对齐
      - 不同行的格子行可宽度不同，因此会水平居中，左右留下 bg padding
      - 同一条带内所有 cell 尺寸一致，且 cell 仍有 1px bg 内边框和 sp/bg sprite
    output_pixels:
      - 仍保留同样的横向条带数和对应高度
      - 每个条带只保留该行唯一的奇异 cell
      - 输出宽度取所有被抽取 cell 中的最大宽度；较窄条带仍水平居中
    trace_example:
      - 原逻辑: 先用全图 uniform columns 找到全局 col_lines，再按固定 cell_w 切所有行
      - 新逻辑: 先用全图 uniform rows 找到 row bands；对每个 band 单独找“整列全是 L”的局部竖线，再切该 band 内的 cells
      - 差异: 列切分不再是一次性的全局常量，而是随 band 改变

  b_complexity_calibration:
    - 尺寸只允许两档 5 或 6，复杂度可见但不至于失控
    - 每题强制同时出现两种尺寸，避免退化回原题
    - 同一条带内仍保持 3-4 列，保证“多数 vs 唯一”清晰
    - 只改尺寸与局部格线，不引入额外颜色或新物体语义

  c_implicit_assumption_audit:
    old_main_dependencies:
      - row_lines/col_lines 都可由“全图 uniform 行列”直接获得
      - cell_w = c_slices[0] 的宽度，默认全行全图共享
      - 输出宽度固定为 cell_w + 2，默认每行相同
      - 将 chosen_cells 直接贴到同一列宽中，不考虑行间宽度差
    why_invalid_now:
      - 混合尺寸行下，全图通常只有横线是全宽 uniform；内部竖线只在局部条带内存在
      - 不同行 cell_w 不同，读取 c_slices[0] 会把其它行切错
      - 输出若无差别地用同一个 cell_w，会截断较宽行或给较窄行错误边界
    new_main_assumptions:
      - 只有水平分隔线必须横跨全图且颜色一致
      - 每个 band 内的 lattice 竖线形成一个单独的等间距序列
      - band 内 cell 不含 lattice color，因此“整列全是 L”只会出现在真实竖线
      - generate_input 必须保证这些条件

  d_placement_legality:
    - 画布高度 = 各 band 高度之和 + 横线数量
    - 每个 band 的局部格子宽度单独计算，再在全图最大宽度内水平居中
    - paste 时 cell 完整落在 band 内，不重叠、不越界
    - 局部竖线贯穿该 band 的上下横线，交点自然形成格子拐点

  e_visual_observability:
    - 人类能直接看到某些横向条带更高更宽，某些更矮更窄
    - 输入到输出的核心规则仍清楚: 每条带选出唯一不同的那格
    - 多尺度变化不是隐藏元数据，而是肉眼可见的格子尺度变化

  f_cleanliness:
    - 只使用 3 种颜色: lattice/bg/sp
    - 左右 padding 仅为纯 bg，不叠加额外标记
    - 局部格子行居中摆放，版面整齐，减少“悬空”感

  g_color_inference:
    - 颜色角色不变：L=格线，bg=底色，sp=图案
    - 不使用颜色值大小、排序或额外颜色映射

  generator_contract_for_new_solver:
    - 至少一行 5x5，至少一行 6x6
    - 每个 band 内有 3-4 个 cells，且恰有 1 个 unique
    - unique 通过 1-3 个 inner 像素翻转获得，保证与该行 base 不同
    - band 内所有非 unique cells 完全相同，确保 odd-one-out 无歧义
"""

from collections import Counter
from pathlib import Path
import sys
import math
import numpy as np

SOLVER_DIR = Path(__file__).resolve().parents[5] / "solvers"
if str(SOLVER_DIR) not in sys.path:
    sys.path.append(str(SOLVER_DIR))

from arc_utils import paste


def _find_lattice_and_row_slices(g: np.ndarray) -> tuple[int, list[tuple[int, int]]]:
    H, _ = g.shape
    uniform_rows = [r for r in range(H) if len(set(map(int, g[r, :]))) == 1]
    assert uniform_rows
    L = int(Counter(int(g[r, 0]) for r in uniform_rows).most_common(1)[0][0])
    row_lines = [r for r in uniform_rows if int(g[r, 0]) == int(L)]
    r_slices = [(row_lines[i] + 1, row_lines[i + 1]) for i in range(len(row_lines) - 1)]
    r_slices = [(int(a), int(b)) for a, b in r_slices if int(b) > int(a)]
    assert r_slices
    return int(L), r_slices


def _extract_row_cells(g: np.ndarray, L: int, ra: int, rb: int) -> list[np.ndarray]:
    band = g[int(ra) : int(rb), :]
    band_h = int(rb - ra)
    line_cols = [c for c in range(g.shape[1]) if np.all(band[:, c] == int(L))]
    assert len(line_cols) >= 4
    c_slices = [(line_cols[i] + 1, line_cols[i + 1]) for i in range(len(line_cols) - 1)]
    c_slices = [(int(a), int(b)) for a, b in c_slices if int(b) - int(a) == int(band_h)]
    assert len(c_slices) >= 3
    return [band[:, int(ca) : int(cb)] for ca, cb in c_slices]


def _build_stacked_lattice(rows: list[list[np.ndarray]], lattice: int, bg: int) -> np.ndarray:
    assert rows and all(row for row in rows)
    row_sizes = [int(row[0].shape[0]) for row in rows]
    strip_widths = [int(len(row) * (size + 1) + 1) for row, size in zip(rows, row_sizes)]
    H = int(sum(row_sizes) + len(row_sizes) + 1)
    W = int(max(strip_widths))
    grid = np.full((H, W), int(bg), dtype=int)

    top = 0
    grid[top, :] = int(lattice)
    for row, size, strip_w in zip(rows, row_sizes, strip_widths):
        offset = int((W - strip_w) // 2)
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
    Extract the unique cell from each lattice row into a single-column lattice.

    Variant D6:
    - Horizontal lattice separators still span the whole canvas.
    - Vertical lattice separators are row-local, because each row may use a different square cell size.
    - Therefore row segmentation is global, but column segmentation must be recomputed per row band.
    """
    g = np.array(grid, dtype=int)
    L, r_slices = _find_lattice_and_row_slices(g)

    bg_candidates = [int(x) for x in g.flatten() if int(x) != int(L)]
    assert bg_candidates
    bg = int(Counter(bg_candidates).most_common(1)[0][0])

    chosen_cells: list[np.ndarray] = []
    for ra, rb in r_slices:
        cells = _extract_row_cells(g, int(L), int(ra), int(rb))
        sigs = [cell.tobytes() for cell in cells]
        cnt = Counter(sigs)
        uniq_idx = next((j for j, s in enumerate(sigs) if int(cnt[s]) == 1), None)
        assert uniq_idx is not None
        chosen_cells.append(cells[int(uniq_idx)].copy())

    return _build_stacked_lattice([[cell] for cell in chosen_cells], int(L), int(bg))


def _make_cell(inner: np.ndarray, bg: int) -> np.ndarray:
    s = int(inner.shape[0] + 2)
    cell = np.full((s, s), int(bg), dtype=int)
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


def _flip_variant(
    rng: np.random.Generator,
    base_inner: np.ndarray,
    bg: int,
    sp: int,
    used_signatures: set[bytes],
) -> np.ndarray | None:
    area = int(base_inner.size)
    all_indices = np.arange(area)
    for _ in range(40):
        flip_count = int(rng.integers(1, min(3, area) + 1))
        idxs = rng.choice(all_indices, size=flip_count, replace=False)
        variant = base_inner.copy()
        for idx in idxs:
            variant.flat[int(idx)] = int(sp) if int(variant.flat[int(idx)]) == int(bg) else int(bg)
        sig = variant.tobytes()
        if sig != base_inner.tobytes() and sig not in used_signatures:
            used_signatures.add(sig)
            return variant
    return None


def _gen_attempt() -> np.ndarray | None:
    rng = np.random.default_rng()

    lattice, bg, sp = map(int, rng.choice(10, size=3, replace=False))
    n_rows = int(rng.integers(3, 5))
    n_cols = int(rng.integers(3, 5))

    row_sizes = [5, 6]
    while len(row_sizes) < n_rows:
        row_sizes.append(int(rng.choice([5, 6])))
    rng.shuffle(row_sizes)
    assert len(set(row_sizes)) == 2

    base_cells: list[np.ndarray] = []
    unique_cells: list[np.ndarray] = []
    for cell_size in row_sizes:
        inner_size = int(cell_size - 2)
        base_inner = _make_base_inner(rng, inner_size, int(bg), int(sp))
        used_signatures = {base_inner.tobytes()}
        unique_inner = _flip_variant(rng, base_inner, int(bg), int(sp), used_signatures)
        if unique_inner is None:
            return None
        base_cells.append(_make_cell(base_inner, int(bg)))
        unique_cells.append(_make_cell(unique_inner, int(bg)))

    in_rows: list[list[np.ndarray]] = []
    for base_cell, unique_cell in zip(base_cells, unique_cells):
        odd_col = int(rng.integers(0, n_cols))
        row = [base_cell.copy() for _ in range(n_cols)]
        row[odd_col] = unique_cell.copy()
        in_rows.append(row)

    inp = _build_stacked_lattice(in_rows, int(lattice), int(bg))
    out = _build_stacked_lattice([[cell.copy()] for cell in unique_cells], int(lattice), int(bg))
    return inp if np.array_equal(main(inp), out) else None


def generate_input() -> np.ndarray:
    for _ in range(300):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input after 300 attempts")
