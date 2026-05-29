# Variant of 1ae2feb7
# Dimension: D3 — 射线源对象类型 (src)
# Change: 水平单色色条——分隔条一侧的水平连续同色像素段，每行1-2个色条，紧邻分隔条排列... → 单像素控制点

"""
Design analysis for the D3 variant
==================================

a) 具体化与行为追踪
-------------------
- 保留原题的核心舞台：一根近乎贯穿全高的竖直分隔列，把网格分成 source side 和 target side。
- 变化只放在“射线源对象”上：
  原题每个活跃行在 source side 上放 1-2 条水平色条，色条长度就是周期。
  变体中每个活跃行只放 1 个紧贴分隔条的单像素控制点；其颜色仍是涂色颜色，
  但周期改为该像素所在行的 1-based 行号，即 `step = r + 1`。
  例如第 3 行（0-based 的 r=2）上的控制点表示步长 3。
- 输出仍在同一行上向另一侧做周期性打点，从分隔条外第一格开始，
  位置为 `0, step, 2*step, ...`，并保持“已有颜色不覆盖”的规则。
- 小例子：
  若分隔列在 c=5，source side 在左，且第 3 行 `g[2,4] = 7`，
  原题若这里是长度 1 的色条，则目标侧会在 c=6,7,8,... 全部涂成 7；
  变体里同样是单像素，但因为它位于第 3 行，所以只会涂 c=6,9,12,...
  差异从“看长度”变成“看位置”。

b) 复杂度校准
-------------
- 如果允许同一行出现多个控制点，它们共享同一个行号步长，容易互相退化成无意义覆盖；
  因此这里固定为“每个活跃行恰好 1 个控制点”，把自由度集中在行位置和颜色上。
- 为避免样例过于稀疏导致规律不可见，生成器把活跃行限制在较靠上的若干行，
  并保证 target side 足够长，使得大多数控制点能产生至少两个周期落点。
- 活跃行仍保持至少 1 行空隔，延续原题的整洁布局，避免相邻控制点让“行号编码”难以观察。

c) 原 solver 的隐式假设审查
---------------------------
- `_find_sep_col(...)` 假设存在一列近乎满高且非背景色一致的分隔列。
  该假设仍成立；生成器继续构造这种分隔列，并且最多只缺一个端点行。
- 原 `bars_on_left = left_mass >= right_mass` 依赖“输入的 target side 为空”。
  该假设也保留：输入里只有 source side 的控制点，target side 全为背景。
- 原来 `_iter_runs_1d(...)` + `length = c1 - c0 + 1` 假设源对象是水平连续段，
  且“长度”就是周期。变体后这个假设失效，因为所有控制点的水平长度都恒为 1；
  若仍沿用原代码，会把所有行错误地当成步长 1。
- 原来按“离分隔条距离”排序是为了解决同一行 1-2 个色条的覆盖优先级。
  变体中每行只允许一个紧邻分隔条的控制点，因此这一层排序不再需要。

d) 放置合法性
-------------
- 控制点只放在 `sep_col-1` 或 `sep_col+1`，保证“紧邻分隔条”且不会越界。
- 活跃行来自带间隔的采样，因此不同控制点不会重叠。
- 分隔列缺口只允许出现在非活跃的顶行或底行，不会和控制点同排造成识别歧义。
- 生成器约束 `step <= ext_side_len - 1`，使每个控制点在目标侧至少能出现两个落点，
  从而既合法又可观察。

e) 视觉可观察性
----------------
- 解题者肉眼能直接看到：source side 不再是长条，而是单点；而输出的疏密由点所在的行决定。
- 同一网格中会出现多个不同步长的行，例如第 1 行满铺、第 3 行每隔 3 个涂一次、
  第 5 行每隔 5 个涂一次。这个规律不依赖颜色 id，只依赖肉眼可见的位置。
- 生成器强制至少有一个较小步长的行，以避免所有输出都只剩单个点，导致规则不可归纳。

f) 可视化整洁度
---------------
- source side 只占分隔条旁边一列，其他区域保持空白，因此控制点与目标侧周期点阵的角色清晰。
- 不使用额外标记或多层叠加；每行仅一类源对象，减少视觉噪声。

g) 颜色变化可推断性
-------------------
- 颜色只做“控制点颜色 = 输出颜色”的直接继承，不引入隐藏映射。
- 每个活跃行使用互异颜色，便于把“哪个控制点产生了哪一行的点阵”直接对应起来。

9) 新 solver 的隐式假设审查
---------------------------
- solver 现在假设 source side 上真正有用的信号只出现在分隔条相邻列。
  生成器保证除了这些控制点外，source side 其余位置为空。
- solver 假设步长完全由行号决定，和颜色、列位置无关。
  生成器保证控制点列固定紧邻分隔条，避免列位置被误读成规则的一部分。
- solver 假设每个活跃行至多一个控制点。
  生成器用逐行唯一放置保证这一点。
"""

from __future__ import annotations

import numpy as np

from arc_utils import colors_present, get_bg
from arc_utils import get_bg as _bg_color


def _find_sep_col(g: np.ndarray, bg: int) -> tuple[int | None, int | None]:
    h, w = g.shape
    for c in range(w):
        col = g[:, c]
        vals = col[col != bg]
        if vals.size >= h - 1 and (vals == vals[0]).all():
            return c, int(vals[0])
    return None, None


def _source_side_is_left(g: np.ndarray, sep_col: int, bg: int) -> bool:
    left_mass = int(np.sum(g[:, :sep_col] != bg)) if sep_col > 0 else 0
    right_mass = int(np.sum(g[:, sep_col + 1 :] != bg)) if sep_col + 1 < g.shape[1] else 0
    return left_mass >= right_mass


def _iter_control_points(
    g: np.ndarray, sep_col: int, controls_on_left: bool, bg: int
) -> list[tuple[int, int]]:
    h, w = g.shape
    src_col = sep_col - 1 if controls_on_left else sep_col + 1
    if not (0 <= src_col < w):
        return []

    pts: list[tuple[int, int]] = []
    for r in range(h):
        color = int(g[r, src_col])
        if color != bg:
            pts.append((r, color))
    return pts


def _simulate_variant(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = _bg_color(g)

    sep_col, _sep_color = _find_sep_col(g, bg)
    if sep_col is None:
        return g.copy()

    controls_on_left = _source_side_is_left(g, sep_col, bg)
    out = g.copy()

    for r, color in _iter_control_points(g, sep_col, controls_on_left, bg):
        step = int(r) + 1
        if controls_on_left:
            for cc in range(sep_col + 1, w, step):
                if int(out[r, cc]) == bg:
                    out[r, cc] = color
        else:
            for cc in range(sep_col - 1, -1, -step):
                if int(out[r, cc]) == bg:
                    out[r, cc] = color

    return out


def main(grid: np.ndarray) -> np.ndarray:
    return _simulate_variant(grid)


# --------------- generator (D3 control-pixel variant) ---------------

H_MIN, H_MAX = 8, 13
W_MIN, W_MAX = 15, 20

CTRL_SIDE_MIN, CTRL_SIDE_MAX = 3, 12
EXT_SIDE_MIN, EXT_SIDE_MAX = 6, 14

PALETTE_MIN, PALETTE_MAX = 5, 8

ACTIVE_MIN, ACTIVE_MAX = 2, 4
STEP_ROW_MAX = 6

P_SEP_COLUMN_GAP = 0.2

MAX_ATTEMPTS = 400


def _choose_active_rows(rng: np.random.Generator, row_count: int, count: int) -> list[int]:
    """Pick `count` row indices from `[0, row_count)` with pairwise spacing >= 2."""
    base = np.sort(rng.choice(row_count - count + 1, size=count, replace=False))
    return list(map(int, base + np.arange(count)))


def _sample_palette(rng: np.random.Generator) -> tuple[int, list[int]]:
    sep_color = int(rng.integers(1, 10))
    ctrl_pool = [c for c in range(1, 10) if c != sep_color]
    size = int(rng.integers(PALETTE_MIN, min(PALETTE_MAX, len(ctrl_pool)) + 1))
    palette = list(map(int, rng.choice(ctrl_pool, size=size, replace=False)))
    return sep_color, palette


def _place_controls(
    inp: np.ndarray,
    active_rows: list[int],
    sep_col: int,
    controls_on_left: bool,
    row_colors: list[int],
) -> None:
    src_col = sep_col - 1 if controls_on_left else sep_col + 1
    for row, color in zip(active_rows, row_colors):
        inp[row, src_col] = int(color)


def _gen_attempt() -> np.ndarray | None:
    rng = np.random.default_rng()

    h = int(rng.integers(H_MIN, H_MAX + 1))
    w = int(rng.integers(W_MIN, W_MAX + 1))

    lo = max(CTRL_SIDE_MIN, w - 1 - EXT_SIDE_MAX)
    hi = min(CTRL_SIDE_MAX, w - 1 - EXT_SIDE_MIN)
    if lo > hi:
        return None
    ctrl_side_len = int(rng.integers(lo, hi + 1))
    ext_side_len = w - 1 - ctrl_side_len

    controls_on_left = bool(rng.integers(0, 2) == 0)
    sep_col = ctrl_side_len if controls_on_left else (w - 1 - ctrl_side_len)

    max_active_row = min(STEP_ROW_MAX, h - 1, ext_side_len - 2)
    if max_active_row < 2:
        return None
    candidate_row_count = max_active_row + 1
    feasible_max_active = min(ACTIVE_MAX, (candidate_row_count + 1) // 2)
    if feasible_max_active < ACTIVE_MIN:
        return None

    active_count = int(rng.integers(ACTIVE_MIN, feasible_max_active + 1))
    active_rows = _choose_active_rows(rng, candidate_row_count, active_count)
    if min(active_rows) > 2:
        return None

    sep_color, palette = _sample_palette(rng)
    row_colors = list(map(int, rng.choice(palette, size=active_count, replace=False)))

    inp = np.zeros((h, w), dtype=int)
    inp[:, sep_col] = sep_color

    active_set = set(active_rows)
    terminal_inactive = [r for r in (0, h - 1) if r not in active_set]
    if terminal_inactive and rng.random() < P_SEP_COLUMN_GAP:
        gap_row = int(rng.choice(terminal_inactive))
        inp[gap_row, sep_col] = 0

    _place_controls(inp, active_rows, sep_col, controls_on_left, row_colors)

    out = _simulate_variant(inp)

    if get_bg(inp) != 0:
        return None
    if sep_color not in colors_present(inp, exclude=[0]):
        return None
    if np.array_equal(inp, out):
        return None
    if not np.array_equal(main(inp), out):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate valid input after {MAX_ATTEMPTS} attempts")
