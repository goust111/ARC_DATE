# Variant of 38007db0
# Dimension: D13 — 干扰机制 (meta)
# Change: 无干扰项 — 每行恰好一个唯一项，其余完全相同... → 多个不同项但只选特定一个

from __future__ import annotations

from collections import Counter
import itertools
import math

import numpy as np

from arc_utils import paste

"""
design_analysis:
  source_task: none
  variant_goal:
    core_change: >
      每个晶格行不再只有一个异常格，而是有两个都不同于多数底图的单元格；
      输出只保留其中“与多数底图像素差异更小”的那个。
    preserved_parts:
      - 仍然是规则晶格
      - 仍然只有 3 种颜色: 晶格色 L、背景色 bg、精灵色 sp
      - 输出仍然是按原行顺序压成单列晶格

  a_concretization_and_trace:
    concrete_pixels:
      - 每个 cell 仍然是 SxS 正方形，外圈是 1 像素 bg padding，中间是 (S-2)x(S-2) 精灵图案
      - 每一行中，多数 cell 都是同一个 base sprite
      - 另外有两个单例变体:
        - near variant: 相对 base 只翻转 1-2 个 inner 像素
        - far variant: 相对 base 翻转 3-5 个 inner 像素，且严格多于 near
      - 输出保留 near variant
    flow_trace_example:
      original_row:
        counts: {base: 3, odd: 1}
        solver_action: 找到唯一出现一次的 odd，直接输出
      variant_row:
        counts: {base: 2, near: 1, far: 1}
        new_solver_action:
          1: 找到该行出现次数最多的 base
          2: 对两个非 base 单元分别计算与 base 的像素汉明距离
          3: 选距离较小的 near 输出
      difference: >
        原题依赖“唯一 singleton”，变体则要求先恢复多数底图，再比较两个干扰项与底图的接近程度。

  b_complexity_calibration:
    why_not_too_simple:
      - 每行固定两个干扰项，视觉上能明确看到“多了一个异常格”
      - 差异量只分为少量档位: near=1-2，far=3-5，便于人眼归纳“选更像底图的那个”
      - 列数提高到 4-5，确保 base 在每行重复至少 2 次，真正形成多数参照
    ambiguity_control:
      - far 的翻转数严格大于 near，避免距离并列
      - 两个干扰项都与 base 不同，且彼此不同

  c_implicit_assumption_review:
    old_main_dependencies:
      - old: 用 Counter 找“出现 1 次的签名”
        status: 失效，因为现在每行有两个 singleton
      - old: 任意 singleton 都是目标
        status: 失效，因为现在两个 singleton 只有一个该被输出
      - old: 列数 3-4 即可
        status: 需适配，3 列时会变成 1/1/1 无多数，因此生成器改为 4-5 列
      - old: base 不需要显式识别
        status: 失效，新 solver 必须先定位多数底图
    new_main_assumptions:
      - 每行存在唯一最多出现的 cell signature，且它就是 base
      - 每行恰好有两个非 base cell
      - 这两个 cell 到 base 的距离严格不同
    generator_guarantees:
      - n_cols in {4, 5}，所以 base 计数至少为 2
      - 每行正好放入 2 个特殊格，其他全是 base
      - near/far 的翻转数分离，保证距离不相等

  d_placement_legality:
    lattice:
      - 仍用统一 1 像素网格线隔开所有 cell
      - 所有 cell 尺寸一致，因此不会越界或重叠
    row_content:
      - 每行只替换两个不同列位置
      - 其余列复制同一个 base cell
      - 输出直接由选中的 near cells 组成单列晶格

  e_visual_observability:
    inference_story: >
      解题者能先看到每行有两个不一样的格子，而输出总是选那个“只改了一点点”的格子；
      另一个异常格改动更多，因此不会与规则混淆。
    forbidden_signals_avoided:
      - 不使用颜色编号大小做排序
      - 不使用隐藏坐标顺序做选择
      - 只依赖肉眼可见的像素差异数量

  f_visual_cleanliness:
    cleanliness:
      - 仍是单层晶格，没有额外标记叠加
      - 每个变体只是同一底图上的少量像素翻转，局部差异清晰且不杂乱

  g_color_inference:
    color_use:
      - 仍然只有三色
      - 颜色角色不变化；变化只发生在 sprite 像素的开/关翻转
      - 因此人类无需额外学习颜色映射

  h_generator_validation:
    output_first: >
      先构造每行应输出的 near variant 单列，再反推输入，并用 main(input)==output 校验。
    anti_regression_metric: >
      额外统计每行两个干扰项的距离对 (near_dist, far_dist) 以及
      “若按原题 first-singleton 逻辑会不会选错”，确认样本不是退化回原题。

  i_main_generator_contract:
    main_requires:
      - 晶格线可由整行/整列纯色识别
      - 每行 base 是唯一最高频签名
      - 两个非 base 签名距离可比较且最小者唯一
    generator_satisfies:
      - 颜色选择保证 lattice/bg/sp 两两不同
      - 行列切片规则与原题完全兼容
      - 每行专门构造 near/far 两个距离档位
"""


def _extract_lattice_structure(g: np.ndarray) -> tuple[int, list[int], list[int], list[tuple[int, int]], list[tuple[int, int]]]:
    H, W = g.shape
    row_lines = [r for r in range(H) if len(set(map(int, g[r, :]))) == 1]
    col_lines = [c for c in range(W) if len(set(map(int, g[:, c]))) == 1]
    assert row_lines and col_lines

    vals = [int(g[r, 0]) for r in row_lines] + [int(g[0, c]) for c in col_lines]
    lattice = int(Counter(vals).most_common(1)[0][0])
    row_lines = sorted([r for r in row_lines if int(g[r, 0]) == lattice])
    col_lines = sorted([c for c in col_lines if int(g[0, c]) == lattice])

    r_slices = [(int(row_lines[i]) + 1, int(row_lines[i + 1])) for i in range(len(row_lines) - 1)]
    c_slices = [(int(col_lines[j]) + 1, int(col_lines[j + 1])) for j in range(len(col_lines) - 1)]
    r_slices = [(a, b) for a, b in r_slices if int(b) > int(a)]
    c_slices = [(a, b) for a, b in c_slices if int(b) > int(a)]
    assert r_slices and c_slices
    return lattice, row_lines, col_lines, r_slices, c_slices


def _cell_distance(a: np.ndarray, b: np.ndarray) -> int:
    return int(np.count_nonzero(a != b))


def main(grid: np.ndarray) -> np.ndarray:
    """
    Variant rule:
    - In each lattice row, one sprite is the majority base pattern.
    - Exactly two cells differ from that base.
    - Extract the one whose pixel difference from the base is smaller.
    """
    g = np.array(grid, dtype=int)
    H, _W = g.shape

    lattice, row_lines, _col_lines, r_slices, c_slices = _extract_lattice_structure(g)
    cell_w = int(c_slices[0][1] - c_slices[0][0])

    bg_candidates = [int(x) for x in g.flatten() if int(x) != lattice]
    assert bg_candidates
    bg = int(Counter(bg_candidates).most_common(1)[0][0])

    chosen_cells: list[np.ndarray] = []
    for ra, rb in r_slices:
        cells = [g[int(ra):int(rb), int(ca):int(cb)] for ca, cb in c_slices]
        sigs = [cell.tobytes() for cell in cells]
        cnt = Counter(sigs)
        top = cnt.most_common()
        assert len(top) >= 3
        base_sig, base_count = top[0]
        assert int(base_count) >= 2
        assert len(top) == 3
        assert top[1][1] == 1 and top[2][1] == 1

        base_cell = next(cell for cell, sig in zip(cells, sigs) if sig == base_sig)
        candidates = []
        for cell, sig in zip(cells, sigs):
            if sig != base_sig:
                candidates.append((_cell_distance(cell, base_cell), cell))

        assert len(candidates) == 2
        candidates.sort(key=lambda item: item[0])
        assert candidates[0][0] < candidates[1][0]
        chosen_cells.append(candidates[0][1])

    out_w = int(cell_w + 2)
    out = np.full((H, out_w), int(bg), dtype=int)
    out[:, 0] = int(lattice)
    out[:, -1] = int(lattice)
    for r in row_lines:
        out[int(r), :] = int(lattice)

    for i, (ra, rb) in enumerate(r_slices):
        out[int(ra):int(rb), 1:1 + int(cell_w)] = chosen_cells[int(i)]

    return out


def _make_cell(inner: np.ndarray, bg: int) -> np.ndarray:
    s = inner.shape[0] + 2
    cell = np.full((s, s), int(bg), dtype=int)
    paste(cell, inner, 1, 1)
    return cell


def _build_lattice(cells: list[list[np.ndarray]], lattice: int, bg: int) -> np.ndarray:
    rows = len(cells)
    cols = len(cells[0])
    cell_size = cells[0][0].shape[0]
    grid = np.full((rows * (cell_size + 1) + 1, cols * (cell_size + 1) + 1), int(bg), dtype=int)
    for r in range(rows + 1):
        grid[r * (cell_size + 1), :] = int(lattice)
    for c in range(cols + 1):
        grid[:, c * (cell_size + 1)] = int(lattice)
    for r, row in enumerate(cells):
        for c, cell in enumerate(row):
            paste(grid, cell, 1 + r * (cell_size + 1), 1 + c * (cell_size + 1))
    return grid


def _flip_variant(base_inner: np.ndarray, flip_set: tuple[int, ...], bg: int, sp: int) -> np.ndarray:
    variant = base_inner.copy()
    for idx in flip_set:
        variant.flat[int(idx)] = int(sp) if int(variant.flat[int(idx)]) == int(bg) else int(bg)
    return variant


def _choose_flip_set(
    rng: np.random.Generator,
    inner_area: int,
    flip_count: int,
    used_flip_sets: set[tuple[int, ...]],
) -> tuple[int, ...] | None:
    candidates = list(itertools.combinations(range(inner_area), int(flip_count)))
    available = [fs for fs in candidates if fs not in used_flip_sets]
    if not available:
        return None
    return tuple(available[int(rng.integers(len(available)))])


def _gen_attempt() -> np.ndarray | None:
    rng = np.random.default_rng()

    lattice, bg, sp = map(int, rng.choice(10, size=3, replace=False))
    cell_size = int(rng.choice([5, 6]))
    n_rows = int(rng.integers(3, 5))
    n_cols = int(rng.integers(4, 6))

    inner_size = cell_size - 2
    inner_area = inner_size * inner_size
    min_fill = math.ceil(0.4 * inner_area)
    max_fill = math.floor(0.9 * inner_area)
    fill = int(rng.integers(min_fill, max_fill + 1))

    base_inner = np.full((inner_size, inner_size), int(bg), dtype=int)
    on_idx = rng.choice(inner_area, size=fill, replace=False)
    base_inner.flat[on_idx] = int(sp)

    used_flip_sets: set[tuple[int, ...]] = set()
    near_cells: list[np.ndarray] = []
    far_cells: list[np.ndarray] = []
    for _row in range(n_rows):
        near_count = int(rng.choice([1, 2]))
        far_options = [k for k in [3, 4, 5] if k <= inner_area and k >= near_count + 2]
        if not far_options:
            return None
        far_count = int(rng.choice(far_options))

        near_flip = _choose_flip_set(rng, inner_area, near_count, used_flip_sets)
        if near_flip is None:
            return None
        used_flip_sets.add(near_flip)

        far_flip = _choose_flip_set(rng, inner_area, far_count, used_flip_sets)
        if far_flip is None:
            return None
        used_flip_sets.add(far_flip)

        near_inner = _flip_variant(base_inner, near_flip, bg, sp)
        far_inner = _flip_variant(base_inner, far_flip, bg, sp)
        near_cells.append(_make_cell(near_inner, bg))
        far_cells.append(_make_cell(far_inner, bg))

    out = _build_lattice([[cell] for cell in near_cells], lattice, bg)

    base_cell = _make_cell(base_inner, bg)
    in_cells: list[list[np.ndarray]] = []
    for row in range(n_rows):
        row_cells = [base_cell.copy() for _ in range(n_cols)]
        special_cols = list(map(int, rng.choice(n_cols, size=2, replace=False)))
        row_cells[special_cols[0]] = near_cells[row].copy()
        row_cells[special_cols[1]] = far_cells[row].copy()
        in_cells.append(row_cells)
    inp = _build_lattice(in_cells, lattice, bg)

    return inp if np.array_equal(main(inp), out) else None


def generate_input() -> np.ndarray:
    for _ in range(300):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input after 300 attempts")
