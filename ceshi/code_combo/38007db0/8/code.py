# Combo variant of 38007db0 (index 8)
# combo_spec: D1/0+D13/0
# Name: 列对齐多干扰检测

from __future__ import annotations

from collections import Counter
import itertools
import math

import numpy as np

from arc_utils import paste

"""
design_analysis:
  summary:
    原题是“每行一个奇异项，抽成单列”。本融合版同时叠加两种改动:
    1. D1/0: 扫描方向改成按列，输出改成单行。
    2. D13/0: 每列不再只有一个异常格，而是有两个都偏离基准格；只选与基准差异更小的那个。

  a_concretization_and_trace:
    concrete_scene:
      - 输入仍然是规则晶格，gridline 宽 1，cell 为 5x5 或 6x6 正方形。
      - 每个 cell 内部有 1 像素 bg padding，中间是一个由 sp/bg 组成的 sprite。
      - 整张图共享同一个 base sprite。
      - 现在每一列中:
        - 多数 cell 都是 base
        - 恰好 2 个干扰 cell:
          - near: 相对 base 只翻 1-2 个 inner 像素
          - far: 相对 base 翻 3-5 个 inner 像素，且严格大于 near
      - 输出是单行晶格，每列保留该列的 near cell。
    trace_example:
      - 原题 4x4 cells:
        每一行形如 [base, base, odd, base]，solver 横向扫描每行，找唯一 singleton，输出 4x1。
      - 仅 D1/0:
        每一列形如 [base, odd, base, base]，solver 纵向扫描每列，找唯一 singleton，输出 1x4。
      - 本融合版:
        每一列形如 [base, near, base, far] 或 [far, base, near, base]。
        solver 不能再“见到 singleton 就拿走”，而是要:
          1. 先识别该列多数出现的 base
          2. 找出两个非 base cell
          3. 比较它们与 base 的像素汉明距离
          4. 取距离更小的 near，放到输出对应列
      - 差异:
        扫描轴从行切到列，判定逻辑从“唯一出现一次”切到“两个候选里选更接近基准的那个”。

  b_complexity_calibration:
    controls:
      - 颜色仍严格限制为 3 色: lattice / bg / sp。
      - near 只允许 1-2 次翻转，far 只允许 3-5 次翻转，且 far > near。
      - cell 行数提升到 4-5，保证一列里放 2 个干扰项后，base 仍有明确多数。
      - cell 列数取 4-5，让“按列抽取”有足够重复证据。
    why_not_too_simple:
      - 视觉上每列明显有两个异常格，不会退化成 D1/0 的单异常列。
      - 距离档位只有少数几档，人眼能观察到“输出总是选改动更少的那个”。
    anti_ambiguity:
      - near/far 翻转数严格分离，不允许并列。
      - 每列两个干扰格都与 base 不同，且彼此不同。

  c_implicit_assumption_review:
    original_main_assumptions_broken:
      - 原 `for row in rows` 假设“每行必有且仅有一个 singleton”:
        失效，因为现在目标结构发生在列上，而且每列有两个 singleton。
      - 原 `uniq_idx = ... cnt[s] == 1` 假设“任意 singleton 都是目标”:
        失效，因为两个 singleton 只有 near 该被输出。
      - 原输出构造假设“保留原高度，压成单列”:
        失效，因为现在要保留原宽度，压成单行。
    new_main_assumptions:
      - 每列存在唯一最高频签名，且它是 base。
      - 每列恰好存在两个非 base cell。
      - 这两个候选到 base 的距离严格不同。
      - 晶格线仍可由整行/整列纯色识别。
    generator_guarantees:
      - n_rows in {4, 5}，所以每列 base 个数 = n_rows - 2，至少为 2。
      - 每列只放两个特殊格，near/far 分属不同 row。
      - near_count in {1, 2}，far_count in {3, 4, 5} 且 far_count > near_count。

  d_placement_legality:
    placement:
      - 所有 cell 仍通过统一 lattice 分隔，不存在重叠。
      - near/far 只替换已有晶格槽位中的两个不同 row 位置，不会越界。
      - 输出直接由各列 near cells 组成单行晶格。

  e_visual_observability:
    inference_story:
      - 解题者会看到: 不是每行一个怪格，而是每列有两个怪格。
      - 输出总是从那两个怪格里选更像 base 的那个，且放到同一列的位置上。
      - 生成器额外强制至少一整行完全是 base，避免样例还能被误读成“按行处理”。
    forbidden_signals_avoided:
      - 不依赖颜色编号大小。
      - 不依赖 near/far 的上下顺序。
      - 不依赖隐藏的坐标奇偶性；只依赖肉眼可见的像素差异数量和列对齐关系。

  f_visual_cleanliness:
    cleanliness:
      - 图中只有一层规则晶格和局部像素翻转，没有额外标记覆盖。
      - near/far 都是同一 base 的轻微变体，比较关系清楚，不会显得杂乱。

  g_color_inference:
    color_logic:
      - 没有新的颜色映射规则。
      - 颜色角色始终固定: lattice 负责分隔，bg/sp 负责 sprite 的关/开像素。
      - 因此推理压力集中在“列扫描 + 选更近者”，而不是颜色语义。

  h_generator_validation:
    output_first:
      - 先构造每列应输出的 near cells 作为单行输出。
      - 再把每列的 near/far 插回输入。
      - 最后用 `main(inp) == out` 做闭环校验。
    anti_regression_metric:
      - 统计每列距离对 `(near_dist, far_dist)`，要求总是形如 (1|2, 3|4|5) 且前者严格更小。
      - 统计每行含有的特殊格数量，要求至少一行为 0，证明样例确实强调列而不是行。

  i_main_generator_contract:
    main_requires:
      - 晶格切片规则有效。
      - 每列签名分布正好是 `base^(n_rows-2) + near + far`。
      - 选中项等于该列到 base 距离最小的特殊格。
    generator_satisfies:
      - 颜色三元组两两不同。
      - 所有 near/far 变体都由 base 通过少量像素翻转得到。
      - 每列两个干扰项的距离严格可比较。
"""


def _extract_lattice_structure(
    g: np.ndarray,
) -> tuple[int, list[int], list[int], list[tuple[int, int]], list[tuple[int, int]]]:
    h, w = g.shape
    row_lines = [r for r in range(h) if len(set(map(int, g[r, :]))) == 1]
    col_lines = [c for c in range(w) if len(set(map(int, g[:, c]))) == 1]
    assert row_lines and col_lines

    vals = [int(g[r, 0]) for r in row_lines] + [int(g[0, c]) for c in col_lines]
    lattice = int(Counter(vals).most_common(1)[0][0])
    row_lines = sorted(r for r in row_lines if int(g[r, 0]) == lattice)
    col_lines = sorted(c for c in col_lines if int(g[0, c]) == lattice)

    r_slices = [(row_lines[i] + 1, row_lines[i + 1]) for i in range(len(row_lines) - 1)]
    c_slices = [(col_lines[j] + 1, col_lines[j + 1]) for j in range(len(col_lines) - 1)]
    r_slices = [(a, b) for a, b in r_slices if b > a]
    c_slices = [(a, b) for a, b in c_slices if b > a]
    assert r_slices and c_slices
    return lattice, row_lines, col_lines, r_slices, c_slices


def _cell_distance(a: np.ndarray, b: np.ndarray) -> int:
    return int(np.count_nonzero(a != b))


def _build_single_row_output(
    chosen_cells: list[np.ndarray],
    lattice: int,
    bg: int,
    cell_h: int,
    width: int,
    col_lines: list[int],
    c_slices: list[tuple[int, int]],
) -> np.ndarray:
    out_h = int(cell_h + 2)
    out = np.full((out_h, width), int(bg), dtype=int)
    out[0, :] = int(lattice)
    out[-1, :] = int(lattice)
    for c in col_lines:
        out[:, int(c)] = int(lattice)
    for j, (ca, cb) in enumerate(c_slices):
        out[1 : 1 + int(cell_h), int(ca) : int(cb)] = chosen_cells[int(j)]
    return out


def _column_choice(cells: list[np.ndarray]) -> tuple[np.ndarray, int, int]:
    sigs = [cell.tobytes() for cell in cells]
    cnt = Counter(sigs)
    top = cnt.most_common()
    assert len(top) == 3

    base_sig, base_count = top[0]
    assert int(base_count) == len(cells) - 2
    assert top[1][1] == 1 and top[2][1] == 1

    base_cell = next(cell for cell, sig in zip(cells, sigs) if sig == base_sig)
    candidates = [(_cell_distance(cell, base_cell), cell) for cell, sig in zip(cells, sigs) if sig != base_sig]
    assert len(candidates) == 2

    candidates.sort(key=lambda item: item[0])
    near_dist, near_cell = candidates[0]
    far_dist, _far_cell = candidates[1]
    assert near_dist < far_dist
    return near_cell, int(near_dist), int(far_dist)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fused rule:
    - Scan lattice columns, not rows.
    - In each column, identify the majority base cell.
    - Exactly two cells differ from that base.
    - Extract the one with smaller pixel distance to the base.
    - Return all extracted cells in a single-row lattice.
    """
    g = np.array(grid, dtype=int)
    _h, w = g.shape

    lattice, _row_lines, col_lines, r_slices, c_slices = _extract_lattice_structure(g)
    cell_h = int(r_slices[0][1] - r_slices[0][0])

    bg_candidates = [int(x) for x in g.flatten() if int(x) != lattice]
    assert bg_candidates
    bg = int(Counter(bg_candidates).most_common(1)[0][0])

    chosen_cells: list[np.ndarray] = []
    for ca, cb in c_slices:
        cells = [g[int(ra) : int(rb), int(ca) : int(cb)] for ra, rb in r_slices]
        chosen, _near_dist, _far_dist = _column_choice(cells)
        chosen_cells.append(chosen)

    return _build_single_row_output(chosen_cells, lattice, bg, cell_h, w, col_lines, c_slices)


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


def _choose_special_rows(
    rng: np.random.Generator,
    n_rows: int,
    n_cols: int,
) -> tuple[np.ndarray, np.ndarray] | None:
    for _ in range(240):
        near_rows = rng.integers(0, n_rows, size=n_cols)
        far_rows = rng.integers(0, n_rows, size=n_cols)
        for col in range(n_cols):
            if int(far_rows[col]) == int(near_rows[col]):
                options = [r for r in range(n_rows) if r != int(near_rows[col])]
                far_rows[col] = int(options[int(rng.integers(len(options)))])

        row_special_counts = np.zeros(n_rows, dtype=int)
        for col in range(n_cols):
            row_special_counts[int(near_rows[col])] += 1
            row_special_counts[int(far_rows[col])] += 1

        if int(np.min(row_special_counts)) != 0:
            continue
        if int(np.max(row_special_counts)) < 3:
            continue
        return near_rows.astype(int), far_rows.astype(int)
    return None


def _behavior_signature(grid: np.ndarray) -> dict[str, object]:
    g = np.array(grid, dtype=int)
    _lattice, _row_lines, _col_lines, r_slices, c_slices = _extract_lattice_structure(g)

    row_special_counts = [0 for _ in r_slices]
    distance_pairs: list[tuple[int, int]] = []
    for ca, cb in c_slices:
        cells = [g[int(ra) : int(rb), int(ca) : int(cb)] for ra, rb in r_slices]
        sigs = [cell.tobytes() for cell in cells]
        cnt = Counter(sigs)
        base_sig = cnt.most_common(1)[0][0]
        base_cell = next(cell for cell, sig in zip(cells, sigs) if sig == base_sig)
        dists = []
        for row_idx, (cell, sig) in enumerate(zip(cells, sigs)):
            if sig != base_sig:
                row_special_counts[row_idx] += 1
                dists.append(_cell_distance(cell, base_cell))
        dists.sort()
        distance_pairs.append((int(dists[0]), int(dists[1])))

    return {
        "row_special_counts": tuple(int(x) for x in row_special_counts),
        "distance_pairs": tuple(distance_pairs),
    }


def _gen_attempt() -> np.ndarray | None:
    rng = np.random.default_rng()

    lattice, bg, sp = map(int, rng.choice(10, size=3, replace=False))
    cell_size = int(rng.choice([5, 6]))
    n_rows = int(rng.integers(4, 6))
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
    for _col in range(n_cols):
        near_count = int(rng.choice([1, 2]))
        far_options = [k for k in [3, 4, 5] if k <= inner_area and k > near_count]
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

    chosen_rows = _choose_special_rows(rng, n_rows, n_cols)
    if chosen_rows is None:
        return None
    near_rows, far_rows = chosen_rows

    out = _build_lattice([near_cells], lattice, bg)

    base_cell = _make_cell(base_inner, bg)
    in_cells = [[base_cell.copy() for _ in range(n_cols)] for _ in range(n_rows)]
    for col in range(n_cols):
        in_cells[int(near_rows[col])][col] = near_cells[col].copy()
        in_cells[int(far_rows[col])][col] = far_cells[col].copy()
    inp = _build_lattice(in_cells, lattice, bg)

    signature = _behavior_signature(inp)
    if min(signature["row_special_counts"]) != 0:
        return None
    if not all(near < far for near, far in signature["distance_pairs"]):
        return None

    return inp if np.array_equal(main(inp), out) else None


def generate_input() -> np.ndarray:
    for _ in range(300):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input after 300 attempts")
