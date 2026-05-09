# Variant of 16de56c4
# Dimension: D19 — 符号编码 (symbol_instructions)
# Change: 无符号编码——所有规则参数直接从点的空间位置推导... → 控制行

from __future__ import annotations

from collections import defaultdict
import random

import numpy as np

from arc_utils import get_bg

"""
design_analysis:
  variant_core:
    - 保留原题的“每条射线线段由两个同色 marker 决定周期、aligned companion 决定截断范围、unaligned bystander 保持不变”。
    - D19 的唯一核心改动是：最终填充色不再由该线上的 marker / companion 颜色直接给出，而是由第一行或第一列上的控制符号给出。
    - 若第一行有控制符号，则它是 control row：位置 (0,c) 的颜色指定第 c 列的填充色，对应射线从 bottom 向上发射。
    - 若第一列有控制符号，则它是 control column：位置 (r,0) 的颜色指定第 r 行的填充色，对应射线从 right 向左发射。
    - 控制线本身在输出中保持不变，不参与被填充。
  concrete_trace:
    - 原题一条带 companion 的行可能是：marker 在位置 10 和 7，companion 在 4，于是步长=3，输出颜色=companion 色，填充位置 4/7/10。
    - 变体里同一条线仍然从 10 和 7 读出步长=3，仍然用 4 作为截断端点，但若该线的 control cell 是颜色 2，则输出改为用颜色 2 填充 4/7/10。
    - 差异不在几何，而在“颜色来源”从局部点改成了全局控制线。
  complexity_calibration:
    - 不引入更高阶符号表，也不让控制符号同时编码步长/范围，避免自由度过多。
    - 控制色只承担“line index -> paint color”的单一职责；周期和 companion 机制完全沿用原题，因此复杂度与参考任务 88e364bc 的“全局图例提供参数、局部对象执行机制”一致。
    - 为了让 D19 肉眼可见，control colors 与 local object colors 使用不相交色集；因此输出颜色明显来自控制线，而非局部点本身。
  implicit_assumptions_review:
    - 原 solver 通过四条边上的非背景计数找唯一 shooting edge；D19 后 top/left 可能被控制线占据，这个前提失效，因此必须先识别 control row / control column。
    - 原 solver 在整条线扫描所有非背景点；D19 下 control cell 也是非背景点，但它不是 marker/companion，所以必须从局部周期分析里排除位置 0。
    - 原 solver 的 paint 取 marker 或 companion；D19 下这个前提失效，paint 必须改为 control cell 的颜色。
    - 原 solver 可以填到整条线两端；D19 下位置 0 是指令位，哪怕命中节拍也不能覆盖。
  placement_legality:
    - 为避免 control cell 与 shooting-edge marker 重叠，row-wise 情况只使用 right edge，column-wise 情况只使用 bottom edge。
    - 活动行/列不使用索引 0，因此非控制边不会意外长出密集噪声，control row / control column 可稳定识别。
    - step 上限改为 line_len-2，保证第二个 marker 不会落到控制线；companion / bystander 也禁止落到控制线或 shooting edge 以外的冲突位置。
  visual_observability:
    - 输入里能直接看到一条顶边或左边的“颜色清单”，输出中各列/各行的填充色与该清单逐项对应。
    - 同一张图里复用 2-4 个 control colors，并允许多个活动行/列共享同一控制色，给人类足够样本支持全局映射。
    - 局部 marker / companion 颜色全部来自另一组颜色，因此“为什么输出换成另一种颜色”只有控制线能解释。
  cleanliness:
    - 每条活动线最多只有 3 个局部点，再加 1 个控制符号；没有多层叠压。
    - inactive lines 的 control cell 为背景，避免控制线本身过度拥挤。
  color_inference:
    - 颜色变化不是按颜色 id 做不可见运算，而是按可见位置映射：控制线上的第 i 个符号控制第 i 条线的最终颜色。
    - 输出色集小而重复，易于从 2-3 个样例中归纳。
  new_solver_assumptions:
    - 恰有一种控制模式成立：第一行或第一列之一携带至少 4 个非背景控制符号，且比另一条控制候选更密。
    - 所有活动线都由非背景 control cell 显式声明；没有“局部有 marker 但控制线没声明”的线。
    - control colors 与局部 marker/companion/bystander colors 不相交，因此不会把控制符号误认成局部重复色。
"""


# --- Configuration ---
_ROWS_RANGE = (7, 12)
_COLS_RANGE = (9, 21)
_STEP_RANGE = (1, 5)
_CONTROL_COLORS = [1, 2, 3, 4]
_OBJECT_COLORS = [5, 6, 7, 8, 9]

_P_MARKER_ONLY = 0.40
_P_COMPANION = 0.35

_MAX_RETRIES = 400


def _detect_control_mode(g: np.ndarray, bg: int) -> str:
    """Return 'row' for a top control row or 'col' for a left control column."""
    h, w = g.shape
    top_count = sum(1 for c in range(1, w) if int(g[0, c]) != bg)
    left_count = sum(1 for r in range(1, h) if int(g[r, 0]) != bg)

    assert max(top_count, left_count) >= 4, (
        f"Need a visible control strip on the first row/column, got top={top_count}, left={left_count}"
    )
    assert top_count != left_count, (
        f"Ambiguous control strip: top={top_count}, left={left_count}"
    )
    return 'row' if top_count > left_count else 'col'


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill periodic rays whose paint color is dictated by a control strip.

    Input structure:
    - bg is the most common color. The first row or first column is a dedicated
      control strip rather than ordinary data.
    - A non-bg cell on the control strip activates one column/row and assigns
      that line's fill color. Top control row means active columns with markers
      shooting from the bottom edge; left control column means active rows with
      markers shooting from the right edge.
    - On each active line away from the control strip, exactly one local color
      repeats twice; those two marker dots define the step.
    - An optional third local dot is either aligned to the same beat
      (companion: it limits the fill span) or off-beat (bystander: unchanged).

    Rule:
    - Detect whether the symbols live on the first row or first column.
    - For every non-bg control cell, inspect its matching row/column excluding
      the control position, recover the marker pair, and derive the beat from
      their gap.
    - If another local dot lies on that beat, fill only the span between the
      marker pair and that aligned stopper; otherwise fill the full line except
      the control strip.
    - Paint every filled beat position with the control-strip color for that
      line, while bystanders remain untouched.

    Key insight: the repeated local dots still encode only geometry (period and
    optional stopping range), but the first-row/first-column symbols globally
    remap each active line to its final paint color.
    """
    g = np.array(grid, dtype=int)
    out = g.copy()
    bg = int(get_bg(g))
    h, w = g.shape

    control_mode = _detect_control_mode(g, bg)
    by_row = control_mode == 'col'
    num_lines = h if by_row else w
    line_len = w if by_row else h

    def cell(line: int, pos: int) -> tuple[int, int]:
        return (line, pos) if by_row else (pos, line)

    def control_color(line: int) -> int:
        return int(g[line, 0]) if by_row else int(g[0, line])

    for li in range(1, num_lines):
        paint = control_color(li)
        colors: dict[int, list[int]] = defaultdict(list)
        for p in range(1, line_len):
            v = int(g[cell(li, p)])
            if v != bg:
                colors[v].append(p)

        if paint == bg:
            assert not colors, (
                f"Inactive line {li} contains local dots {dict(colors)} without a control symbol"
            )
            continue

        assert colors, f"Active line {li} has a control symbol but no local marker dots"
        repeating = {c: ps for c, ps in colors.items() if len(ps) >= 2}
        assert len(repeating) == 1, (
            f"Active line {li} must have exactly one repeating local color, got {dict(colors)}"
        )

        marker, mpos = next(iter(repeating.items()))
        assert len(mpos) == 2, f"Marker on line {li} must appear exactly twice, got {mpos}"
        a, b = sorted(mpos)
        step = b - a
        offset = a % step

        aligned: list[tuple[int, list[int]]] = []
        for oc, ops in colors.items():
            if oc == marker:
                continue
            if any(p % step == offset for p in ops):
                aligned.append((oc, ops))

        assert len(aligned) <= 1, f"Line {li} has multiple aligned companions: {aligned}"
        if aligned:
            _, ops = aligned[0]
            assert len(ops) == 1, f"Companion on line {li} must be a single dot, got {ops}"
            lo, hi = min(a, b, ops[0]), max(a, b, ops[0])
        else:
            lo, hi = 1, line_len - 1

        for p in range(lo, hi + 1):
            if p % step != offset:
                continue
            out[cell(li, p)] = paint

    return out


def _spaced_indices(n: int, total: int) -> list[int]:
    """Choose n indices in [0, total) with gaps >= 1."""
    available = total - (n - 1)
    if available < n:
        return []
    picks = sorted(random.sample(range(available), n))
    return [picks[i] + i for i in range(n)]


def _choose_layout() -> tuple[str, int, int, list[int]] | None:
    """Pick control mode, grid shape, and active line indices."""
    control_mode = random.choice(('row', 'col'))

    if control_mode == 'row':
        h = random.randint(*_ROWS_RANGE)
        w = random.randint(*_COLS_RANGE)
        max_active = min(9, w // 2)  # choose among columns 1..w-1 with a blank gap between indices
        if max_active < 4:
            return None
        n_active = random.randint(4, max_active)
        raw = _spaced_indices(n_active, w - 1)
        if not raw:
            return None
        return control_mode, h, w, [x + 1 for x in raw]

    h = random.randint(max(8, _ROWS_RANGE[0]), _ROWS_RANGE[1])
    w = random.randint(*_COLS_RANGE)
    max_active = min(6, h // 2)  # choose among rows 1..h-1 with a blank gap between indices
    if max_active < 4:
        return None
    n_active = random.randint(4, max_active)
    raw = _spaced_indices(n_active, h - 1)
    if not raw:
        return None
    return control_mode, h, w, [x + 1 for x in raw]


def _assign_control_colors(active_lines: list[int]) -> dict[int, int]:
    """Assign 2-4 repeated control colors across active lines."""
    k = random.randint(2, min(4, len(active_lines) - 1))
    chosen = random.sample(_CONTROL_COLORS, k)
    bag = chosen + [random.choice(chosen) for _ in range(len(active_lines) - k)]
    random.shuffle(bag)
    return dict(zip(active_lines, bag))


def _build_local_dots(line_len: int, marker_color: int) -> tuple[list[tuple[int, int]], str]:
    """
    Build local dots for one active line.

    Returns a list of (pos, color) excluding the control position and a line-type label.
    """
    step = random.randint(_STEP_RANGE[0], min(_STEP_RANGE[1], line_len - 2))
    edge_pos = line_len - 1
    inner_pos = edge_pos - step
    offset = inner_pos % step
    dots = [(edge_pos, marker_color), (inner_pos, marker_color)]

    roll = random.random()
    other_colors = [c for c in _OBJECT_COLORS if c != marker_color]

    if roll < _P_MARKER_ONLY:
        return dots, 'marker_only'

    if roll < _P_MARKER_ONLY + _P_COMPANION:
        beats = list(range(inner_pos - step, 0, -step))
        if beats and other_colors:
            dots.append((random.choice(beats), random.choice(other_colors)))
            return dots, 'companion'
        return dots, 'marker_only'

    non_beat = [
        p for p in range(1, line_len - 1)
        if p not in (inner_pos,) and p % step != offset
    ]
    if non_beat and other_colors:
        dots.append((random.choice(non_beat), random.choice(other_colors)))
        return dots, 'bystander'
    return dots, 'marker_only'


def _gen_attempt() -> np.ndarray | None:
    plan = _choose_layout()
    if plan is None:
        return None
    control_mode, h, w, active_lines = plan
    by_row = control_mode == 'col'

    def cell(line: int, pos: int) -> tuple[int, int]:
        return (line, pos) if by_row else (pos, line)

    inp = np.zeros((h, w), dtype=int)
    control_map = _assign_control_colors(active_lines)

    for li in active_lines:
        paint = control_map[li]
        if by_row:
            inp[li, 0] = paint
        else:
            inp[0, li] = paint

        marker_color = random.choice(_OBJECT_COLORS)
        dots, _ = _build_local_dots(w if by_row else h, marker_color)
        for pos, color in dots:
            r, c = cell(li, pos)
            if inp[r, c] != 0:
                return None
            inp[r, c] = color

    # The unused corner should stay blank so the control strip is visually clean.
    inp[0, 0] = 0

    try:
        out = main(inp)
    except Exception:
        return None
    if np.array_equal(inp, out):
        return None

    return inp


def generate_input() -> np.ndarray:
    for _ in range(_MAX_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"Failed to generate valid input after {_MAX_RETRIES} attempts")
