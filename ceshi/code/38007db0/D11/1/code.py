# Variant of 38007db0
# Dimension: D11 — 格子线宽度 (meta)
# Change: 格子线宽度为1像素 → 不均匀格子线（行线与列线宽度不同）

from __future__ import annotations

from collections import Counter
import itertools
import math

import numpy as np

from arc_utils import paste


# Design analysis for the D11 variant
# -----------------------------------
# spec_variant:
#   task: Per-row odd-one-out extraction from a lattice grid into a single-column lattice.
#   preserved_rule:
#     - 每一行的多个单元格中，只有一个单元格与该行其它单元格不同。
#     - 输出仍然按行提取这些奇异单元格，拼成单列格子。
#   changed_dimension:
#     - 原题所有格线都是 1 像素。
#     - solver 兼容 (row_w, col_w) 为 (1, 2) 或 (2, 1)。
#     - 为避免样本退化回旧 solver 也能稳定处理的情形，generator 固定生成更关键的 (1, 2)：
#       行线 1 像素、列线 2 像素。
#   concrete_pixel_shape:
#     - 单元格仍然是 SxS 的正方形，S in {5, 6}。
#     - 格子由“连续格线带”分隔，而不是单条像素线：
#       行方向的格线是若干连续的全 L 行；列方向的格线是若干连续的全 L 列。
#     - 例如 row_w=1, col_w=2 时，输入宽度为 cols*S + (cols+1)*2，高度为 rows*S + (rows+1)*1。
#   behavior_trace:
#     - 原 solver 的关键切片写法是 `(line_i + 1, next_line_i)`，隐含“每条格线只有 1 像素宽”。
#     - 若列线宽度为 2，连续两列都属于格线。原写法会把后一列格线误当成单元格内容的一部分，
#       导致 cell slice 左边界/右边界错位，整行 odd-one-out 比较失真。
#     - 新逻辑先把连续格线索引分组成 band，例如列 band `(0,2), (7,9), (14,16)`，
#       再用相邻 band 之间的空隙 `(2,7), (9,14)` 作为单元格切片。
#   complexity_calibration:
#     - 只引入一个核心变化：非均匀格线宽度。
#     - 宽度档位只保留 1 和 2；generator 固定使用最能触发差异的 (1, 2)，避免一半样本对旧 solver 近似等价。
#   implicit_assumption_audit:
#     - `row_lines = [r for r ... uniform row]`：仍成立，因为格线带中的每一行都是全 L。
#     - `col_lines = [c for c ... uniform col]`：仍成立，因为格线带中的每一列都是全 L。
#     - `r_slices = [(row_lines[i]+1, row_lines[i+1])]`：不再成立；改为 band 终点到下一 band 起点。
#     - `outW = cell_w + 2`：不再成立；改为 `cell_w + 2*col_w`。
#     - `out[:,0] = L`, `out[:,-1] = L`：不再成立；需写入左右两个竖向 band，而非单列。
#   placement_legality:
#     - generator 统一使用固定 `row_w` / `col_w` 生成整张格子，所有 cells 放在 band 间空隙中。
#     - 单元格起点是 `row_w + r*(cell_size + row_w)` 与 `col_w + c*(cell_size + col_w)`，不会越界或重叠。
#   visual_observability:
#     - 在 2-4 个样例里，人类可以直接看到“横线细、竖线粗”。
#     - 主规则仍是按行提取奇异单元格，所以变体只增加解析难度，不引入额外不可见条件。
#   visual_cleanliness:
#     - 颜色仍然只有 lattice / bg / sp 三种，结构与原题一致，不会出现额外叠层。
#   color_reasoning:
#     - 无颜色映射变化；颜色只承担原题中的格线、背景、图案三种角色。
#   new_main_assumptions_and_generator_guarantees:
#     - `main()` 现在假设：所有横向格线 band 宽度一致、所有纵向格线 band 宽度一致、二者不同。
#     - generator 明确只生成 `(1,2)`，从而保证这些假设成立且变体稳定可观察。


def _group_consecutive(indices: list[int]) -> list[tuple[int, int]]:
    if not indices:
        return []
    bands: list[tuple[int, int]] = []
    start = prev = int(indices[0])
    for idx in indices[1:]:
        idx = int(idx)
        if idx == prev + 1:
            prev = idx
            continue
        bands.append((start, prev + 1))
        start = prev = idx
    bands.append((start, prev + 1))
    return bands


def _band_slices(bands: list[tuple[int, int]]) -> list[tuple[int, int]]:
    return [(int(bands[i][1]), int(bands[i + 1][0])) for i in range(len(bands) - 1) if int(bands[i + 1][0]) > int(bands[i][1])]


def _parse_lattice(g: np.ndarray) -> tuple[int, list[tuple[int, int]], list[tuple[int, int]], list[tuple[int, int]], list[tuple[int, int]], int, int]:
    H, W = g.shape

    row_lines = [r for r in range(H) if len(set(map(int, g[r, :]))) == 1]
    col_lines = [c for c in range(W) if len(set(map(int, g[:, c]))) == 1]
    assert row_lines and col_lines

    vals = [int(g[r, 0]) for r in row_lines] + [int(g[0, c]) for c in col_lines]
    L = int(Counter(vals).most_common(1)[0][0])

    row_bands = _group_consecutive(sorted(r for r in row_lines if int(g[r, 0]) == int(L)))
    col_bands = _group_consecutive(sorted(c for c in col_lines if int(g[0, c]) == int(L)))
    assert row_bands and col_bands

    row_widths = {int(b - a) for a, b in row_bands}
    col_widths = {int(b - a) for a, b in col_bands}
    assert len(row_widths) == 1 and len(col_widths) == 1
    row_w = next(iter(row_widths))
    col_w = next(iter(col_widths))
    assert row_w != col_w

    r_slices = _band_slices(row_bands)
    c_slices = _band_slices(col_bands)
    assert r_slices and c_slices

    cell_heights = {int(b - a) for a, b in r_slices}
    cell_widths = {int(b - a) for a, b in c_slices}
    assert len(cell_heights) == 1 and len(cell_widths) == 1

    return L, row_bands, col_bands, r_slices, c_slices, row_w, col_w


def main(grid: np.ndarray) -> np.ndarray:
    """
    Extract the unique cell from each lattice row into a single-column lattice.

    D11 variant:
    - Row gridlines and column gridlines have different thicknesses.
    - The solver must parse contiguous lattice bands rather than assuming 1-pixel lines.
    """
    g = np.array(grid, dtype=int)
    H, _W = g.shape

    L, row_bands, _col_bands, r_slices, c_slices, _row_w, col_w = _parse_lattice(g)

    bg_candidates = [int(x) for x in g.flatten() if int(x) != int(L)]
    assert bg_candidates
    bg = int(Counter(bg_candidates).most_common(1)[0][0])

    cell_h = int(r_slices[0][1] - r_slices[0][0])
    cell_w = int(c_slices[0][1] - c_slices[0][0])

    chosen_cells: list[np.ndarray] = []
    for ra, rb in r_slices:
        cells = [g[int(ra) : int(rb), int(ca) : int(cb)] for ca, cb in c_slices]
        sigs = [cell.tobytes() for cell in cells]
        cnt = Counter(sigs)
        uniq_idx = next((j for j, sig in enumerate(sigs) if int(cnt[sig]) == 1), None)
        assert uniq_idx is not None
        chosen_cells.append(cells[int(uniq_idx)])

    outW = int(cell_w + 2 * col_w)
    out = np.full((H, outW), int(bg), dtype=int)
    out[:, :col_w] = int(L)
    out[:, outW - col_w :] = int(L)
    for a, b in row_bands:
        out[int(a) : int(b), :] = int(L)

    for i, (ra, rb) in enumerate(r_slices):
        assert rb - ra == cell_h
        out[int(ra) : int(rb), int(col_w) : int(col_w + cell_w)] = chosen_cells[int(i)]

    return out


def _make_cell(inner: np.ndarray, bg: int) -> np.ndarray:
    s = inner.shape[0] + 2
    cell = np.full((s, s), int(bg), dtype=int)
    paste(cell, inner, 1, 1)
    return cell


def _build_lattice(cells: list[list[np.ndarray]], lattice: int, bg: int, row_w: int, col_w: int) -> np.ndarray:
    rows = len(cells)
    cols = len(cells[0])
    cell_size = cells[0][0].shape[0]

    H = rows * cell_size + (rows + 1) * int(row_w)
    W = cols * cell_size + (cols + 1) * int(col_w)
    grid = np.full((H, W), int(bg), dtype=int)

    for r in range(rows + 1):
        r0 = r * (cell_size + int(row_w))
        grid[r0 : r0 + int(row_w), :] = int(lattice)
    for c in range(cols + 1):
        c0 = c * (cell_size + int(col_w))
        grid[:, c0 : c0 + int(col_w)] = int(lattice)

    for r, row in enumerate(cells):
        for c, cell in enumerate(row):
            rr = int(row_w + r * (cell_size + row_w))
            cc = int(col_w + c * (cell_size + col_w))
            paste(grid, cell, rr, cc)
    return grid


def _gen_attempt() -> np.ndarray | None:
    rng = np.random.default_rng()

    lattice, bg, sp = map(int, rng.choice(10, size=3, replace=False))
    cell_size = int(rng.choice([5, 6]))
    n_rows = int(rng.integers(3, 5))
    n_cols = int(rng.integers(3, 5))
    row_w, col_w = 1, 2

    inner_size = cell_size - 2
    inner_area = inner_size * inner_size
    min_fill = math.ceil(0.4 * inner_area)
    max_fill = math.floor(0.9 * inner_area)
    fill = int(rng.integers(min_fill, max_fill + 1))

    base_inner = np.full((inner_size, inner_size), int(bg), dtype=int)
    on_idx = rng.choice(inner_area, size=fill, replace=False)
    base_inner.flat[on_idx] = int(sp)

    unique_cells: list[np.ndarray] = []
    used_flip_sets: set[tuple[int, ...]] = set()
    for _row in range(n_rows):
        flip_count = int(rng.integers(1, 4))
        candidates = list(itertools.combinations(range(inner_area), flip_count))
        available = [flip_set for flip_set in candidates if flip_set not in used_flip_sets]
        if not available:
            return None
        flip_set = available[int(rng.integers(len(available)))]
        used_flip_sets.add(flip_set)

        variant_inner = base_inner.copy()
        for idx in flip_set:
            variant_inner.flat[int(idx)] = sp if int(variant_inner.flat[int(idx)]) == int(bg) else bg
        unique_cells.append(_make_cell(variant_inner, bg))

    out = _build_lattice([[cell] for cell in unique_cells], lattice, bg, row_w, col_w)

    base_cell = _make_cell(base_inner, bg)
    in_cells: list[list[np.ndarray]] = []
    for row in range(n_rows):
        odd_col = int(rng.integers(0, n_cols))
        row_cells = [base_cell.copy() for _ in range(n_cols)]
        row_cells[odd_col] = unique_cells[row].copy()
        in_cells.append(row_cells)
    inp = _build_lattice(in_cells, lattice, bg, row_w, col_w)

    return inp if np.array_equal(main(inp), out) else None


def generate_input() -> np.ndarray:
    for _ in range(300):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input after 300 attempts")
