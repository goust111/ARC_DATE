# Combo variant of 1ae2feb7 (index 33)
# combo_spec: D3/0+D4/2+D11/0
# Name: 单像素有限射线延迟起始

"""
Design analysis for the combo variant
=====================================

a) 具体化与行为追踪
-------------------
- 舞台仍是原题的竖直分隔列：它把网格分成 source side 和 ext side。
- D3/0 把原来的水平色条替换成“紧邻分隔条的单个彩色像素”。颜色仍直接决定输出颜色，
  但周期不再由条长给出，而是由像素所在行的 1-based 行号给出：`step = r + 1`。
- D11/0 再把起点从 ext side 的第一个格子推迟到一整步之后，所以目标位置不再是
  `0, step, 2*step, ...`，而是 `step, 2*step, 3*step, ...`。
- D4/2 最后把“延伸到边缘”改成“最多只尝试 step 个位置”，也就是一个控制点既携带步长，
  也携带最大射击次数。
- 例子：若分隔列在 `c=4`，控制点在第 3 行（0-based `r=2`，颜色 7），则
  原题若把它看成长度 1 的条，会从 `c=5` 起持续铺满；
  D3 单变体会在 `c=5,8,11,...` 打点；
  本融合版只会尝试 `c=8,11,14` 这 3 个位置，并且已有颜色仍不覆盖。
- 更强的展示行是顶行：`r=0 -> step=1`。此时融合版不会涂紧邻分隔条的格子，而只会在
  第二个 ext 格子涂 1 个点；这同时把“延迟起始”和“有限次数”都直接暴露出来。

b) 复杂度校准
-------------
- 若允许同一行多个控制点，它们会共享相同的行号步长，规则会退化成难以解释的覆盖竞争；
  因此固定为“每个活跃行至多 1 个控制点”。
- 仅靠顶行 `step=1` 虽然能强力展示 D4+D11，但不足以让人看出“行号编码步长”，
  所以生成器还要求至少另一行能产生 2 个以上的落点，用来展示稀疏周期。
- 为避免大量高步长行完全打不到 ext side，活跃行只从较靠上的候选行中抽样，并约束
  `step <= ext_side_len - 1`。

c) 原 solver 的隐式假设审查
---------------------------
- 原 `_find_sep_col(...)` 假设存在一列近乎满高、同色的分隔列。该假设保留。
- 原 `left_mass >= right_mass` 依赖“输入里只有 source side 有信号，ext side 全空”。
  融合版仍保留这个结构，因此仍可用。
- 原 `_iter_runs_1d(...)` 与 `length = c1 - c0 + 1` 假设 source object 是水平连续段；
  D3/0 后这条假设失效，因为有效对象变成单像素，水平长度恒为 1，真正的步长来自行号。
- 原 `range(..., step=length)` 默认从位置 0 开始且一直跑到边缘；D11/0 与 D4/2 都打破了
  这个前提，新的目标枚举必须改成“从 step 偏移开始，最多尝试 step 次”。
- 原来按“离分隔条距离”排序是为了解决同一行 1-2 条色条的优先级；现在每行最多一个控制点，
  行间互不竞争，排序层可以删掉。

d) 放置合法性
-------------
- 控制点只放在 `sep_col-1` 或 `sep_col+1`，始终紧邻分隔条，不越界。
- 活跃行两两至少间隔 1 个空行，因此控制点彼此不相邻，输出也更易辨认。
- 生成器限制候选行满足 `r + 1 <= ext_side_len - 1`，保证每个控制点至少有一个合法落点。
- 分隔列的可选缺口只允许出现在非活跃的端点行，避免和控制点同排造成识别歧义。

e) 视觉可观察性
----------------
- 输入一眼就能看出 source object 已从“条”变成“点”，这是 D3/0 的外显变化。
- 每个活跃行在输出里都留下“分隔条旁先空一段，再开始规律打点”的特征，这是 D11/0。
- 顶行控制点保证至少有一行只出现 1 个落点，从而让 D4/2 与“延伸到边缘”的旧行为明显不同。
- 另一条展示行会给出 2 个或更多周期点，避免样例只剩零星孤点而看不出步长规律。

f) 可视化整洁度
---------------
- source side 只有分隔条旁的一列可能非空，其他 source 区域全部留白。
- 每行只对应一个颜色和一个步长，不引入同一行多层覆盖；“不覆盖已有颜色”的规则仍保留，
  但在生成样本中通常不会触发复杂竞争。

g) 颜色变化可推断性
-------------------
- 颜色完全直传：控制点是什么颜色，输出点就是什么颜色。
- 每个活跃行使用互异颜色，便于把输入单点和输出点列直接对应起来。

9) 新 solver 的隐式假设审查
---------------------------
- solver 假设真正的控制信号只会出现在分隔条相邻列；生成器保证 source side 其余格子为空。
- solver 假设每个活跃行至多一个控制点；生成器逐行唯一放置。
- solver 假设步长与最大尝试次数都等于 `r + 1`；生成器通过控制点行号直接编码这一点。
- solver 假设 ext side 在输入中全为空，这样 side-detection 和输出打点都不会混淆；
  生成器始终满足该条件。
"""

from __future__ import annotations

import itertools

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
    g: np.ndarray,
    sep_col: int,
    controls_on_left: bool,
    bg: int,
) -> list[tuple[int, int]]:
    src_col = sep_col - 1 if controls_on_left else sep_col + 1
    if not (0 <= src_col < g.shape[1]):
        return []

    points: list[tuple[int, int]] = []
    for r in range(g.shape[0]):
        color = int(g[r, src_col])
        if color != bg:
            points.append((r, color))
    return points


def _iter_targets(
    sep_col: int,
    step: int,
    controls_on_left: bool,
    w: int,
    *,
    delayed: bool,
    limited: bool,
):
    if step <= 0:
        return

    offset = step if delayed else 0
    start = sep_col + 1 + offset if controls_on_left else sep_col - 1 - offset
    delta = step if controls_on_left else -step

    if limited:
        for shot in range(step):
            cc = start + shot * delta
            if 0 <= cc < w:
                yield cc
            else:
                break
        return

    cc = start
    while 0 <= cc < w:
        yield cc
        cc += delta


def _simulate_control_rays(
    grid: np.ndarray,
    *,
    delayed: bool,
    limited: bool,
) -> np.ndarray:
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
        for cc in _iter_targets(
            sep_col,
            step,
            controls_on_left,
            w,
            delayed=delayed,
            limited=limited,
        ):
            if int(out[r, cc]) == bg:
                out[r, cc] = color

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Project control-pixel rays with delayed starts and finite ammunition.

    Input structure:
    - A near-full-height vertical separator column.
    - On exactly one side of the separator, a sparse set of single colored pixels,
      each touching the separator on its row.
    - The opposite side is empty background in the input.

    Rule:
    - Each control pixel on row `r` defines `step = r + 1`.
    - Paint on the opposite side using that pixel's color at offsets
      `step, 2*step, 3*step, ...` from the separator, not at offset 0.
    - Stop after attempting exactly `step` target positions, and never overwrite
      an already colored cell.

    Key insight:
    - The single pixel carries both the color and the firing schedule: its row
      encodes the spacing, the first mark is one full spacing away, and the total
      number of attempted marks is also that same spacing value.
    """
    return _simulate_control_rays(grid, delayed=True, limited=True)


def _main_delayed_to_edge(grid: np.ndarray) -> np.ndarray:
    """D3 + D11 only: delayed start, but still extends to the edge."""
    return _simulate_control_rays(grid, delayed=True, limited=False)


def _main_nondelayed_limited(grid: np.ndarray) -> np.ndarray:
    """D3 + D4 only: limited shots, but starts immediately past the separator."""
    return _simulate_control_rays(grid, delayed=False, limited=True)


# --------------- generator (combo variant) ---------------

H_MIN, H_MAX = 8, 13
W_MIN, W_MAX = 15, 20

CTRL_SIDE_MIN, CTRL_SIDE_MAX = 3, 12
EXT_SIDE_MIN, EXT_SIDE_MAX = 8, 14

PALETTE_MIN, PALETTE_MAX = 5, 8

ACTIVE_MIN, ACTIVE_MAX = 2, 4
STEP_ROW_MAX = 6

P_SEP_COLUMN_GAP = 0.2

MAX_ATTEMPTS = 400


def _sample_palette(rng: np.random.Generator) -> tuple[int, list[int]]:
    sep_color = int(rng.integers(1, 10))
    ctrl_pool = [c for c in range(1, 10) if c != sep_color]
    size = int(rng.integers(PALETTE_MIN, min(PALETTE_MAX, len(ctrl_pool)) + 1))
    palette = list(map(int, rng.choice(ctrl_pool, size=size, replace=False)))
    return sep_color, palette


def _choose_active_rows(
    rng: np.random.Generator,
    candidate_row_count: int,
    count: int,
) -> list[int]:
    valid = [
        combo
        for combo in itertools.combinations(range(candidate_row_count), count)
        if combo[0] == 0 and all(b - a >= 2 for a, b in zip(combo, combo[1:]))
    ]
    if not valid:
        raise ValueError("no valid active-row layout")
    return list(map(int, valid[int(rng.integers(0, len(valid)))]))


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


def _visible_fused_count(step: int, ext_side_len: int) -> int:
    if step <= 0 or ext_side_len <= step:
        return 0
    return min(step, (ext_side_len - 1) // step)


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

    max_feasible = len(range(0, candidate_row_count, 2))
    feasible_max_active = min(ACTIVE_MAX, max_feasible)
    if feasible_max_active < ACTIVE_MIN:
        return None

    active_count = int(rng.integers(ACTIVE_MIN, feasible_max_active + 1))
    active_rows = _choose_active_rows(rng, candidate_row_count, active_count)

    # Keep one showcase row at the top (step=1), and require at least one other row
    # to produce multiple visible dots so the row-coded spacing is observable.
    if active_rows[0] != 0:
        return None
    visible_counts = [_visible_fused_count(r + 1, ext_side_len) for r in active_rows]
    if any(count <= 0 for count in visible_counts):
        return None
    if max(visible_counts[1:], default=0) < 2:
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

    out = main(inp)

    if get_bg(inp) != 0:
        return None
    if sep_color not in colors_present(inp, exclude=[0]):
        return None
    if np.array_equal(inp, out):
        return None
    if not np.array_equal(out, _simulate_control_rays(inp, delayed=True, limited=True)):
        return None
    if np.array_equal(out, _main_delayed_to_edge(inp)):
        return None
    if np.array_equal(out, _main_nondelayed_limited(inp)):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate valid input after {MAX_ATTEMPTS} attempts")
