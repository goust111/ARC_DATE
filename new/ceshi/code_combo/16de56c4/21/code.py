# Combo variant of 16de56c4 (index 21)
# combo_spec: D9/0+D19/2
# Name: 颜色标记边+控制行

from __future__ import annotations

from collections import defaultdict
import random

import numpy as np

from arc_utils import get_bg

"""
design_analysis:
  variant_core:
    - 保留原题的局部几何机制：每条活动线仍由两枚同色 marker 决定周期，aligned companion 只决定截断范围，off-beat bystander 保持不变。
    - D9/0 与 D19/2 的融合点在于：局部 marker 的颜色不再各线自定，而是整题共享一个“边标记色”；该颜色只在真实发射边形成一串同色端点，用来先识别 shooting edge。
    - 填充颜色则完全来自第一行或第一列的 control strip。于是“边标记色”只编码方向，“控制行/列”只编码每条线最终用什么颜色填。
  concrete_trace:
    - 例：若顶部是 control row，列 5 的 control cell 是颜色 A；底边同一列有边标记色 T，向上第 3 格还有一个 T，于是步长=3。
    - 若该列再有一个颜色 B 的 companion 落在同一节拍上，则输出只在 companion 与两枚 T 的包围区间内按节拍填充，填充色不是 B，而是 control row 给出的 A。
    - 与原题相比，原题会把这条线涂成 marker 色或 companion 色；融合后必须先看 T 知道从哪条边射，再看 control row/column 才知道输出颜色。
  complexity_calibration:
    - 只引入两个全局符号层：一个全局边标记色、一个一维 control strip；不让 control strip 同时编码步长或 companion 身份，避免自由度过多。
    - control colors 在每题内只取 2-4 种并重复出现，便于从少量样例归纳“第 i 个控制格控制第 i 条线”。
    - edge tag / control / local-object 三组颜色严格分离，防止“输出为什么变色”落回不可解释的颜色运算。
  implicit_assumptions_review:
    - 原 solver 依赖“边界非背景数量最多的边就是 shooting edge”；融合后 top/left 的 control strip 会与真实发射边打平，因此必须改成识别边标记色形成的统一边带。
    - 原 solver 允许从整条线读取重复色；融合后位置 0 是控制位，不能参与 marker/companion 解析，也不能被填充覆盖。
    - 原 solver 允许每条线的 repeating color 各不相同；融合后 repeating color 被固定为全局边标记色，因此 solver 需要把“tag color 恰出现两次”当成线级约束。
    - 原 solver 的 paint 来源于局部 marker/companion；融合后 paint 必须来自 control strip，companion 仅决定终止范围。
  placement_legality:
    - 为避免 control strip 与 shooting edge 重叠，只生成两种布局：top control row + bottom shooting edge，或 left control column + right shooting edge。
    - 活动行/列索引都从 1 开始，保证控制角落独立，局部对象不会压到控制 strip。
    - step 上限为 line_len-2，确保第二枚 marker 不会落到控制 strip；companion / bystander 也只允许放在 1..line_len-2 的局部区间内。
  visual_observability:
    - 输入上能同时看到两条不同职责的边：一条是第一行/列的彩色控制带，另一条是真实发射边上的同色 tag 点串。
    - 输出中所有被填位置都变成 control strip 的颜色，而不是局部 tag/companion 的颜色，因此 D19 的作用肉眼可见。
    - 发射边不是“点最多”的那条边，因为 control strip 会制造计数干扰；这让 D9 的颜色标记也成为可观察的必要线索。
  cleanliness:
    - 每条活动线最多 4 个非背景点：1 个 control cell，2 个 tag markers，外加 0/1 个 companion 或 bystander。
    - inactive lines 在 control strip 上保持背景，避免整条边过密难以阅读。
  fusion_metrics:
    - 退化检查 1：按原题“边界点数最多”规则，样本必须不能唯一选出真实 shooting edge；本生成器通过让 control strip 与 shooting edge 至少打平来保证这一点。
    - 退化检查 2：输出主色必须来自 control strip 而非局部点；本生成器通过三组颜色分离保证这一点，且每个活动线都会把 tag markers 改写成 control color。
  new_solver_assumptions:
    - 恰有一种边界满足“该边所有非背景端点同色，且对应活动线各自恰有两枚该色 marker”。
    - 该边与 control strip 声明的活动线集合完全一致。
    - 每条活动线至多有一个额外局部点，它要么 aligned 当 companion，要么 off-beat 当 bystander。
"""


_ROWS_RANGE = (7, 12)
_COLS_RANGE = (9, 21)
_STEP_RANGE = (1, 5)

_P_MARKER_ONLY = 0.40
_P_COMPANION = 0.35

_MAX_RETRIES = 500
_EDGES = ("top", "bottom", "left", "right")
_NON_BG = list(range(1, 10))


def _line_geometry(edge: str, h: int, w: int) -> tuple[bool, int, int, int]:
    by_row = edge in ("left", "right")
    num_lines = h if by_row else w
    line_len = w if by_row else h
    edge_pos = 0 if edge in ("left", "top") else line_len - 1
    return by_row, num_lines, line_len, edge_pos


def _detect_shooting_edge(g: np.ndarray, bg: int) -> tuple[str, int, list[int]]:
    """Return (edge, tag_color, active_lines) from the color-marked shooting edge."""
    h, w = g.shape
    candidates: list[tuple[str, int, list[int]]] = []

    for edge in _EDGES:
        by_row, num_lines, line_len, edge_pos = _line_geometry(edge, h, w)

        def cell(line: int, pos: int) -> tuple[int, int]:
            return (line, pos) if by_row else (pos, line)

        edge_cells = []
        for li in range(1, num_lines):
            v = int(g[cell(li, edge_pos)])
            if v != bg:
                edge_cells.append((li, v))

        if len(edge_cells) < 4:
            continue

        edge_colors = {v for _, v in edge_cells}
        if len(edge_colors) != 1:
            continue
        tag_color = next(iter(edge_colors))
        active_lines = [li for li, _ in edge_cells]

        valid = True
        for li in active_lines:
            tag_positions = [p for p in range(1, line_len) if int(g[cell(li, p)]) == tag_color]
            if len(tag_positions) != 2 or edge_pos not in tag_positions:
                valid = False
                break
        if valid:
            candidates.append((edge, tag_color, active_lines))

    assert len(candidates) == 1, f"Expected exactly one color-marked shooting edge, got {candidates}"
    return candidates[0]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill periodic rays whose shooting side is marked by a dedicated border color
    and whose paint color is dictated by a control strip.

    Input structure:
    - bg is the most common color. The first row or first column is a control
      strip: each non-bg cell there activates one column/row and supplies that
      line's eventual paint color.
    - The real shooting edge is not the busiest border. Instead, one border
      carries a uniform tag color; every active line has exactly one tag cell on
      that edge plus one interior tag cell of the same color, and their gap is
      the step.
    - Each active line may also contain one extra local dot. If it lands on the
      same beat as the tag pair it is a companion; otherwise it is a bystander.

    Rule:
    - First identify the shooting edge by finding the border whose non-bg cells
      all share one tag color and whose corresponding lines each contain exactly
      two copies of that tag.
    - Read the control strip on the first row/column to decide which lines are
      active and what color each active line should paint with.
    - On every active line, recover the step from the two tag markers. An
      aligned extra dot limits the fill to the span between itself and the tag
      pair; without one, the beat fills the whole line except the control cell.
    - Paint all filled beat positions with the control-strip color. Bystanders
      stay unchanged.

    Key insight: one symbolic channel answers “which edge fires?” via the shared
    tag color on the border, while a second symbolic channel answers “what color
    does this line paint?” via the first-row/first-column control strip.
    """
    g = np.array(grid, dtype=int)
    out = g.copy()
    bg = int(get_bg(g))
    h, w = g.shape

    edge, tag_color, tagged_lines = _detect_shooting_edge(g, bg)
    assert edge in ("bottom", "right"), f"Fused layout expects bottom/right shooting, got {edge}"

    by_row, num_lines, line_len, _ = _line_geometry(edge, h, w)
    control_mode = "col" if by_row else "row"

    def cell(line: int, pos: int) -> tuple[int, int]:
        return (line, pos) if by_row else (pos, line)

    def control_color(line: int) -> int:
        return int(g[line, 0]) if control_mode == "col" else int(g[0, line])

    controlled_lines = [li for li in range(1, num_lines) if control_color(li) != bg]
    assert controlled_lines == tagged_lines, (
        f"Control strip and color-marked edge disagree: control={controlled_lines}, tag={tagged_lines}"
    )

    for li in range(1, num_lines):
        paint = control_color(li)
        colors: dict[int, list[int]] = defaultdict(list)
        for p in range(1, line_len):
            v = int(g[cell(li, p)])
            if v != bg:
                colors[v].append(p)

        if paint == bg:
            assert not colors, f"Inactive line {li} contains local dots {dict(colors)}"
            continue

        marker_positions = sorted(colors.get(tag_color, []))
        assert len(marker_positions) == 2, (
            f"Active line {li} must contain exactly two tag markers, got {marker_positions}"
        )
        a, b = marker_positions
        step = b - a
        offset = a % step

        other_positions = [
            p
            for color, ps in colors.items()
            if color != tag_color
            for p in ps
        ]
        assert len(other_positions) <= 1, f"Active line {li} has too many extra dots: {dict(colors)}"

        if other_positions and other_positions[0] % step == offset:
            lo, hi = min(a, b, other_positions[0]), max(a, b, other_positions[0])
        else:
            lo, hi = 1, line_len - 1

        for p in range(lo, hi + 1):
            if p % step == offset:
                out[cell(li, p)] = paint

    return out


def _spaced_indices(n: int, total: int) -> list[int]:
    """Choose n indices from [0, total) with at least one blank between picks."""
    available = total - (n - 1)
    if available < n:
        return []
    picks = sorted(random.sample(range(available), n))
    return [picks[i] + i for i in range(n)]


def _choose_layout() -> tuple[str, int, int, list[int]] | None:
    """Return (mode, h, w, active_lines); mode=row -> top control row, mode=col -> left control col."""
    control_mode = random.choice(("row", "col"))

    if control_mode == "row":
        h = random.randint(_ROWS_RANGE[0], _ROWS_RANGE[1])
        w = random.randint(_COLS_RANGE[0], _COLS_RANGE[1])
        max_active = min(9, w // 2)
        if max_active < 4:
            return None
        n_active = random.randint(4, max_active)
        raw = _spaced_indices(n_active, w - 1)
        if not raw:
            return None
        return control_mode, h, w, [x + 1 for x in raw]

    h = random.randint(max(8, _ROWS_RANGE[0]), _ROWS_RANGE[1])
    w = random.randint(_COLS_RANGE[0], _COLS_RANGE[1])
    max_active = min(6, h // 2)
    if max_active < 4:
        return None
    n_active = random.randint(4, max_active)
    raw = _spaced_indices(n_active, h - 1)
    if not raw:
        return None
    return control_mode, h, w, [x + 1 for x in raw]


def _assign_palettes(active_lines: list[int]) -> tuple[int, dict[int, int], list[int]]:
    """Pick disjoint color roles: one edge-tag color, repeated control colors, and object colors."""
    palette = random.sample(_NON_BG, len(_NON_BG))
    tag_color = palette[0]
    n_control = random.randint(2, min(4, len(active_lines) - 1))
    control_palette = palette[1:1 + n_control]
    object_palette = palette[1 + n_control:]
    assert len(object_palette) >= 3

    bag = control_palette + [random.choice(control_palette) for _ in range(len(active_lines) - n_control)]
    random.shuffle(bag)
    control_map = dict(zip(active_lines, bag))
    return tag_color, control_map, object_palette


def _build_local_dots(line_len: int, tag_color: int, object_colors: list[int]) -> tuple[list[tuple[int, int]], str]:
    """
    Build local dots for one active line.

    Returns [(pos, color), ...] with bottom/right shooting: pos 0 is reserved for the control strip.
    """
    step = random.randint(_STEP_RANGE[0], min(_STEP_RANGE[1], line_len - 2))
    edge_pos = line_len - 1
    inner_pos = edge_pos - step
    offset = inner_pos % step
    dots = [(edge_pos, tag_color), (inner_pos, tag_color)]

    roll = random.random()
    if roll < _P_MARKER_ONLY:
        return dots, "marker_only"

    if roll < _P_MARKER_ONLY + _P_COMPANION:
        beats = list(range(inner_pos - step, 0, -step))
        if beats:
            dots.append((random.choice(beats), random.choice(object_colors)))
            return dots, "companion"
        return dots, "marker_only"

    non_beat = [
        p for p in range(1, line_len - 1)
        if p != inner_pos and p % step != offset
    ]
    if non_beat:
        dots.append((random.choice(non_beat), random.choice(object_colors)))
        return dots, "bystander"
    return dots, "marker_only"


def _edge_counts(inp: np.ndarray) -> dict[str, int]:
    h, w = inp.shape
    return {
        "top": sum(1 for c in range(w) if int(inp[0, c]) != 0),
        "bottom": sum(1 for c in range(w) if int(inp[h - 1, c]) != 0),
        "left": sum(1 for r in range(h) if int(inp[r, 0]) != 0),
        "right": sum(1 for r in range(h) if int(inp[r, w - 1]) != 0),
    }


def _gen_attempt() -> np.ndarray | None:
    plan = _choose_layout()
    if plan is None:
        return None
    control_mode, h, w, active_lines = plan
    by_row = control_mode == "col"
    line_len = w if by_row else h

    def cell(line: int, pos: int) -> tuple[int, int]:
        return (line, pos) if by_row else (pos, line)

    inp = np.zeros((h, w), dtype=int)
    tag_color, control_map, object_colors = _assign_palettes(active_lines)

    extra_types: list[str] = []
    for li in active_lines:
        if control_mode == "col":
            inp[li, 0] = control_map[li]
        else:
            inp[0, li] = control_map[li]

        dots, kind = _build_local_dots(line_len, tag_color, object_colors)
        extra_types.append(kind)
        for pos, color in dots:
            r, c = cell(li, pos)
            if inp[r, c] != 0:
                return None
            inp[r, c] = color

    inp[0, 0] = 0

    # Old edge-count detection must not uniquely identify the real shooting edge.
    counts = _edge_counts(inp)
    max_count = max(counts.values())
    winners = [edge for edge, cnt in counts.items() if cnt == max_count]
    expected_edge = "right" if control_mode == "col" else "bottom"
    if winners == [expected_edge]:
        return None

    # Ensure the control strip is actually doing visible work.
    if len(set(control_map.values())) < 2:
        return None
    if all(kind == "marker_only" for kind in extra_types):
        # Keep some range-limiting or bystander evidence in most samples.
        return None

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
