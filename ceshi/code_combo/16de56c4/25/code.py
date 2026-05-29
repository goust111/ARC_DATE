# Combo variant of 16de56c4 (index 25)
# combo_spec: D3/0+D6/3+D15/0
# Name: 交替填充+碰撞变色+多伴随点

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
import random
import sys

import numpy as np

_HERE = Path(__file__).resolve()
for _parent in _HERE.parents:
    if (_parent / "solvers" / "arc_utils.py").exists():
        sys.path.append(str(_parent / "solvers"))
        break

from arc_utils import get_bg


# Design analysis for the combo variant
# -------------------------------------
# a) concretization_and_trace:
# - 保留原题的外壳：唯一 shooting edge 决定方向；每条活跃线仍只有一种 marker
#   颜色出现两次，其中一枚 marker 在射击边上，另一枚在内部，二者间距定义 step。
# - 融合后不再是“单 companion 单色区间”。例如 left-edge、step=2 的一行：
#   marker A 在 0/2，companions B/C 在 6/12，bystander D 在 9。
#   beat 位置是 0,2,4,6,8,10,12,14,16。
#   原题若只看第一个 aligned companion，会把 0..6 全刷成 B 并停止。
#   纯碰撞变色会得到 A,A,A,B,B,B,C,C,C...
#   本融合版则得到 A,B,A,B,C,B,C,C,C...：
#   0..6 这段在“当前色 A”和“下一 companion 色 B”之间交替；
#   到 6 发生碰撞后，当前色切到 B；接着 8..12 在 B/C 间交替；
#   最后一次碰撞后没有“下一 companion”，于是余下 beat 全保持当前色 C。
# - 差异点有三层同时出现：同一条射线上有多次颜色切换；每次切换前不是纯色而是
#   ABAB 条纹；最后一次碰撞后还会继续向前，而不是在 companion 处停止。
#
# b) complexity_calibration:
# - companion 数量限制为 2 或 3，而不是任意多个；marker-only 行仍保留，便于和
#   复杂行形成对照，但每个样本都会强制出现多条 multi-companion 行。
# - 为了让交替规律肉眼可见，第一段 companion 必须落在从射击边数起的奇数 beat
#   索引 3/5/7... 上；后续 companions 与前一个切换点的 beat 距离也必须为奇数。
#   这样每个 companion 自己所在的格子都会保持自己的输入颜色，局部规律清楚。
# - 每个网格只用 5-7 种非背景色，且同一条线内部的 marker / companions /
#   bystander 全部不同，避免颜色身份混淆。
#
# c) implicit_assumption_audit:
# - 原 solver 在发现第一个 aligned companion 后就 `break`，这在多 companion
#   变体下会丢掉后续切换点，因此必须改为收集并排序全部 aligned companions。
# - 原 solver 依赖 `[lo, hi]` 无方向区间。融合后这不成立：D6/3 要求碰撞后继续，
#   D15/0 要求多段顺序，D3/0 还要求按从射击边出发的 beat 顺序交替着色。
# - 原写入条件只覆盖 `bg/marker`。融合后 segment 的终点 companion 自身要保持其
#   颜色，后续 segment 还会以“上一 companion 色”作为当前色，所以写入逻辑必须
#   显式允许当前色/下一 companion 色。
# - 新规则引入了新的隐式假设：相邻切换点之间的 beat 距离必须是奇数，且最后一个
#   companion 后面至少还要留 1 个 beat，solver 会断言这些前提，generator 保证。
#
# d) placement_legality:
# - 活跃线仍两两至少隔 1 行/列，因此不同射线不会重叠。
# - companions 直接从 beat 序列里选，索引 >= 3，且不使用最后一个 beat，这样不会
#   越界，也能保证最后一次碰撞后确实还能继续。
# - bystander 只放在内部 off-beat 空位，不会与 markers/companions 重合，也不会
#   被误识别成 aligned companion。
#
# e) visual_observability:
# - 解题者可以先从两枚 marker 读出 step，再从 aligned 单点读出“这里会发生切换”。
#   因为每个 companion 都保留自己的颜色，样例里可以直接看到：
#   “到这个点以前是当前色/它的颜色交替；过了它之后，它变成新的当前色”。
# - generator 强制至少两条 multi-companion 行，并拒绝那些与“纯碰撞分段”或
#   “纯多 companion 吸收分段”输出相同的样本，确保融合行为真正可见。
#
# f) visual_cleanliness:
# - 每条活跃线仍只有一条稀疏 beat scaffold；复杂性来自沿同一直线的颜色时序变化，
#   而不是叠多个对象层。可选 bystander 最多 1 个，且不在 beat 上，画面不会糊成一团。
#
# g) color_change_inferability:
# - 颜色变化完全由局部几何结构决定，不依赖颜色编号大小、排序或任何人类不可视因素。
#   “下一 companion 的颜色”就是下一段交替里出现的新颜色；“最后一次碰撞后的颜色”
#   就是最后一个 companion 的颜色。颜色映射线索直接写在输入中。
#
# h) anti_degeneration_policy:
# - 生成器要求至少 2 条 multi-companion 行、至少 5 个总 companion，并要求组合
#   输出同时区别于“纯碰撞变色”基线和“纯多 companion 吸收分段”基线。
#   这避免了样本退化成只体现 D6/3 或只体现 D15/0 的情形。
#
# i) audited_new_solver_assumptions:
# - 每条非空线恰有 1 种重复颜色，且只重复 2 次；generator 通过“每条线颜色全 distinct，
#   只有 marker 重复”来保证。
# - aligned companions 若存在，则数量只能是 2 或 3，且都位于 beat 索引 >= 2 的位置；
#   generator 直接从 beat 列表构造，不允许别的情况。
# - 按组合规则，所有切换点之间的 beat 距离都是奇数，且最后一个切换点之后至少还留
#   1 个 beat；solver 对此断言，generator 通过索引选择函数保证。


def _shooting_edge(g: np.ndarray, bg: int) -> str:
    H, W = g.shape
    edge_counts = {
        "top": sum(1 for c in range(W) if g[0, c] != bg),
        "bottom": sum(1 for c in range(W) if g[H - 1, c] != bg),
        "left": sum(1 for r in range(H) if g[r, 0] != bg),
        "right": sum(1 for r in range(H) if g[r, W - 1] != bg),
    }
    max_count = max(edge_counts.values())
    assert max_count > 0, "No non-bg dots on any edge"
    winners = [edge for edge, count in edge_counts.items() if count == max_count]
    assert len(winners) == 1, f"Ambiguous shooting edge: {winners} (counts={edge_counts})"
    return winners[0]


def _line_axes(edge: str, shape: tuple[int, int]):
    H, W = shape
    by_row = edge in ("left", "right")
    num_lines = H if by_row else W
    line_len = W if by_row else H

    def cell(line: int, pos: int) -> tuple[int, int]:
        return (line, pos) if by_row else (pos, line)

    return by_row, num_lines, line_len, cell


def _edge_params(edge: str, line_len: int) -> tuple[int, int]:
    if edge in ("left", "top"):
        return 0, 1
    return line_len - 1, -1


def _ordered_beats(edge: str, line_len: int, step: int) -> list[int]:
    edge_pos, travel_dir = _edge_params(edge, line_len)
    stop = line_len if travel_dir == 1 else -1
    return list(range(edge_pos, stop, travel_dir * step))


def _iter_line_colors(g: np.ndarray, bg: int, line: int, line_len: int, cell):
    colors: dict[int, list[int]] = defaultdict(list)
    for pos in range(line_len):
        value = int(g[cell(line, pos)])
        if value != bg:
            colors[value].append(pos)
    return colors


def _extract_line_spec(
    colors: dict[int, list[int]],
    edge: str,
    line_len: int,
    edge_pos: int,
    travel_dir: int,
    li: int,
    by_row: bool,
):
    axis = "row" if by_row else "col"
    repeating = {color: positions for color, positions in colors.items() if len(positions) >= 2}
    assert len(repeating) == 1, (
        f"Line {li} ({axis}) must have exactly one repeating marker color: {dict(colors)}"
    )

    marker, mpos = next(iter(repeating.items()))
    assert len(mpos) == 2, (
        f"Line {li} ({axis}) marker {marker} has {len(mpos)} dots (expected 2): {mpos}"
    )
    assert edge_pos in mpos, (
        f"Line {li} ({axis}) marker {marker} must touch the shooting edge: {mpos}"
    )

    step = abs(mpos[1] - mpos[0])
    assert step > 0, f"Line {li} ({axis}) has coincident marker dots: {mpos}"
    expected_inner = edge_pos + travel_dir * step
    assert expected_inner in mpos, (
        f"Line {li} ({axis}) marker {marker} must have its second dot one step inward: {mpos}"
    )

    beats = _ordered_beats(edge, line_len, step)
    beat_to_index = {pos: idx for idx, pos in enumerate(beats)}
    aligned_companions: list[tuple[int, int, int]] = []
    for color, positions in colors.items():
        if color == marker:
            continue
        assert len(positions) == 1, (
            f"Line {li} ({axis}) non-marker color {color} appears multiple times: {positions}"
        )
        pos = positions[0]
        beat_idx = beat_to_index.get(pos)
        if beat_idx is not None and beat_idx >= 2:
            aligned_companions.append((beat_idx, pos, color))

    aligned_companions.sort()
    if aligned_companions:
        assert 2 <= len(aligned_companions) <= 3, (
            f"Line {li} ({axis}) has {len(aligned_companions)} aligned companions "
            f"(expected 2 or 3): {aligned_companions}"
        )
        prev_idx = 0
        for beat_idx, _pos, _color in aligned_companions:
            assert beat_idx > prev_idx, (
                f"Line {li} ({axis}) companions must move away from the edge: {aligned_companions}"
            )
            assert (beat_idx - prev_idx) % 2 == 1, (
                f"Line {li} ({axis}) companion gaps must be odd in beat space: {aligned_companions}"
            )
            prev_idx = beat_idx
        assert prev_idx < len(beats) - 1, (
            f"Line {li} ({axis}) must continue after the last companion: {aligned_companions}"
        )

    return marker, step, beats, aligned_companions


def _paint_combo_line(
    out: np.ndarray,
    bg: int,
    cell,
    line: int,
    marker: int,
    beats: list[int],
    companions: list[tuple[int, int, int]],
):
    if not companions:
        for pos in beats:
            rc = cell(line, pos)
            if int(out[rc]) in {bg, marker}:
                out[rc] = marker
        return

    current_color = marker
    prev_idx = 0
    for collision_idx, _collision_pos, next_color in companions:
        start = prev_idx if prev_idx == 0 else prev_idx + 1
        allowed = {bg, marker, current_color, next_color}
        for beat_idx in range(start, collision_idx + 1):
            rc = cell(line, beats[beat_idx])
            if int(out[rc]) not in allowed:
                continue
            if (beat_idx - prev_idx) % 2 == 0:
                out[rc] = current_color
            else:
                out[rc] = next_color
        current_color = next_color
        prev_idx = collision_idx

    for beat_idx in range(prev_idx + 1, len(beats)):
        rc = cell(line, beats[beat_idx])
        if int(out[rc]) in {bg, marker, current_color}:
            out[rc] = current_color


def _paint_collision_line(
    out: np.ndarray,
    bg: int,
    cell,
    line: int,
    marker: int,
    beats: list[int],
    companions: list[tuple[int, int, int]],
):
    current_color = marker
    start_idx = 0
    for collision_idx, _collision_pos, next_color in companions:
        for beat_idx in range(start_idx, collision_idx):
            rc = cell(line, beats[beat_idx])
            if int(out[rc]) in {bg, marker, current_color}:
                out[rc] = current_color
        current_color = next_color
        start_idx = collision_idx

    for beat_idx in range(start_idx, len(beats)):
        rc = cell(line, beats[beat_idx])
        if int(out[rc]) in {bg, marker, current_color}:
            out[rc] = current_color


def _paint_segment_line(
    out: np.ndarray,
    bg: int,
    cell,
    line: int,
    marker: int,
    beats: list[int],
    companions: list[tuple[int, int, int]],
):
    if not companions:
        for pos in beats:
            rc = cell(line, pos)
            if int(out[rc]) in {bg, marker}:
                out[rc] = marker
        return

    prev_idx = 0
    for collision_idx, _collision_pos, paint in companions:
        start = prev_idx if prev_idx == 0 else prev_idx + 1
        for beat_idx in range(start, collision_idx + 1):
            rc = cell(line, beats[beat_idx])
            if int(out[rc]) in {bg, marker, paint}:
                out[rc] = paint
        prev_idx = collision_idx


def _render(grid: np.ndarray, mode: str) -> np.ndarray:
    g = np.array(grid, dtype=int)
    out = g.copy()
    bg = int(get_bg(g))
    edge = _shooting_edge(g, bg)
    by_row, num_lines, line_len, cell = _line_axes(edge, g.shape)
    edge_pos, travel_dir = _edge_params(edge, line_len)

    for li in range(num_lines):
        colors = _iter_line_colors(g, bg, li, line_len, cell)
        if not colors:
            continue
        marker, _step, beats, companions = _extract_line_spec(
            colors, edge, line_len, edge_pos, travel_dir, li, by_row
        )
        if mode == "combo":
            _paint_combo_line(out, bg, cell, li, marker, beats, companions)
        elif mode == "collision":
            _paint_collision_line(out, bg, cell, li, marker, beats, companions)
        elif mode == "segment":
            _paint_segment_line(out, bg, cell, li, marker, beats, companions)
        else:
            raise ValueError(f"Unknown render mode: {mode}")

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fire periodic rays from one unique edge and let multiple companions drive alternating recolor segments.

    Input structure:
    - The background is the most common color and all informative cells are sparse dots.
    - Exactly one boundary has the strict maximum number of non-background dots; that boundary is the
      shooting edge and determines whether rays travel along rows or along columns.
    - Every active shooting-direction line contains exactly one marker color with exactly 2 dots:
      one on the shooting edge and one step inward. Their gap defines the periodic beat.
    - Extra singleton dots on that line are either 2 or 3 aligned companions on the same beat sequence,
      ordered away from the edge, or an off-beat bystander that is not part of the ray.

    Rule:
    - Detect the shooting edge, recover the ordered beat positions from the 2 marker dots, and sort all
      aligned companions in the direction of travel.
    - If a line has no aligned companions, paint every beat on that line with the marker color.
    - If a line has aligned companions, then before each collision the beats alternate between the current
      ray color and the upcoming companion color, starting from the current color at the segment anchor.
      When the ray reaches that companion, the companion's color becomes the new current color.
    - After the last companion there is no new alternation partner, so the remaining beats continue in the
      last adopted current color. Off-beat bystanders remain unchanged.

    Key insight: the base task's geometry is still read from the shooting edge and the repeated marker pair,
    but aligned dots are now an ordered sequence of collision events. Each upcoming companion both provides
    the second stripe color for the current segment and becomes the new carrying color for the next segment.
    """
    return _render(grid, "combo")


_ROWS_RANGE = (7, 12)
_COLS_RANGE = (9, 21)
_ACTIVE_LINES_RANGE = (4, 9)
_STEP_RANGE = (1, 5)
_ALL_COLORS = list(range(1, 10))
_EDGES = ("top", "bottom", "left", "right")

_MULTI_COMPANION_COUNTS = (2, 3)
_PALETTE_SIZE_RANGE = (5, 7)
_P_MARKER_ONLY = 0.20
_P_BYSTANDER_ONLY = 0.20
_P_MULTI_BYSTANDER = 0.30

_MAX_RETRIES = 500


def _choose_dims(edge: str, n_active: int):
    min_num_lines = 2 * n_active - 1
    by_row = edge in ("left", "right")
    if by_row:
        lo = max(_ROWS_RANGE[0], min_num_lines)
        if lo > _ROWS_RANGE[1]:
            return None
        H = random.randint(lo, _ROWS_RANGE[1])
        W = random.randint(*_COLS_RANGE)
        return H, W, H, W

    lo = max(_COLS_RANGE[0], min_num_lines)
    if lo > _COLS_RANGE[1]:
        return None
    W = random.randint(lo, _COLS_RANGE[1])
    H = random.randint(*_ROWS_RANGE)
    return H, W, W, H


def _spaced_indices(n: int, total: int) -> list[int]:
    available = total - (n - 1)
    if available < n:
        return []
    picks = sorted(random.sample(range(available), n))
    return [picks[i] + i for i in range(n)]


def _count_other_edge_dots(
    dots_rc: list[tuple[int, int]], edge: str, H: int, W: int
) -> dict[str, int]:
    boundary = {
        "top": lambda r, c: r == 0,
        "bottom": lambda r, c: r == H - 1,
        "left": lambda r, c: c == 0,
        "right": lambda r, c: c == W - 1,
    }
    counts: dict[str, int] = {}
    for other_edge in _EDGES:
        if other_edge == edge:
            continue
        check = boundary[other_edge]
        counts[other_edge] = sum(1 for r, c in dots_rc if check(r, c))
    return counts


def _valid_steps(line_len: int, n_companions: int) -> list[int]:
    upper = min(_STEP_RANGE[1], line_len - 1)
    steps = []
    for step in range(_STEP_RANGE[0], upper + 1):
        beat_count = len(_ordered_beats("left", line_len, step))
        if n_companions == 0:
            steps.append(step)
            continue
        if beat_count >= n_companions + 4:
            steps.append(step)
    return steps


def _choose_companion_indices(beat_count: int, n_companions: int) -> list[int] | None:
    solutions: list[list[int]] = []

    def dfs(prev_idx: int, chosen: list[int]) -> None:
        remaining = n_companions - len(chosen)
        if remaining == 0:
            if prev_idx < beat_count - 1:
                solutions.append(chosen.copy())
            return

        for idx in range(prev_idx + 1, beat_count - 1):
            if not chosen and idx < 3:
                continue
            if (idx - prev_idx) % 2 == 0:
                continue
            min_last_idx = idx + (remaining - 1)
            if min_last_idx >= beat_count - 1:
                continue
            chosen.append(idx)
            dfs(idx, chosen)
            chosen.pop()

    dfs(0, [])
    if not solutions:
        return None

    def score(sol: list[int]) -> tuple[int, int, int]:
        prev = 0
        long_gaps = 0
        total_gap = 0
        for idx in sol:
            gap = idx - prev
            if gap >= 3:
                long_gaps += 1
            total_gap += gap
            prev = idx
        tail = beat_count - 1 - sol[-1]
        return long_gaps, tail, total_gap

    ranked = sorted(solutions, key=score, reverse=True)
    best = score(ranked[0])
    candidates = [sol for sol in ranked if score(sol) >= (best[0] - 1, 0, 0)]
    return random.choice(candidates)


def _available_bystander_positions(
    edge: str,
    line_len: int,
    step: int,
    occupied: set[int],
) -> list[int]:
    beat_set = set(_ordered_beats(edge, line_len, step))
    return [
        pos
        for pos in range(1, line_len - 1)
        if pos not in occupied and pos not in beat_set
    ]


def _line_kind_sequence(n_active: int) -> list[str]:
    # Keep multi-companion lines dominant, but always leave room for simple
    # contrast branches so the training examples expose all observable cases.
    if n_active <= 4:
        kinds = ["multi", "multi", "multi", random.choice(("marker_only", "bystander_only"))]
    else:
        kinds = ["multi", "multi", "multi", "marker_only", "bystander_only"]

    while len(kinds) < n_active:
        roll = random.random()
        if roll < 0.45:
            kinds.append("multi")
        elif roll < 0.725:
            kinds.append("marker_only")
        else:
            kinds.append("bystander_only")
    random.shuffle(kinds)
    return kinds


def _build_line_dots(
    edge: str,
    line_len: int,
    palette: list[int],
    kind: str,
    prefer_three: bool,
):
    multi = kind == "multi"
    companion_options = []
    if multi:
        for n_companions in _MULTI_COMPANION_COUNTS:
            if len(_valid_steps(line_len, n_companions)) > 0:
                companion_options.append(n_companions)
        if not companion_options:
            return None
        if prefer_three and 3 in companion_options:
            n_companions = 3
        else:
            n_companions = random.choice(companion_options)
    else:
        n_companions = 0

    valid_steps = _valid_steps(line_len, n_companions)
    if not valid_steps:
        return None
    step = random.choice(valid_steps)

    beats = _ordered_beats(edge, line_len, step)
    edge_pos, _travel_dir = _edge_params(edge, line_len)
    inner_pos = beats[1]

    add_bystander = kind == "bystander_only" or (multi and random.random() < _P_MULTI_BYSTANDER)
    needed_colors = 1 + n_companions + (1 if add_bystander else 0)
    if len(palette) < needed_colors:
        return None
    chosen_colors = random.sample(palette, needed_colors)
    marker_color = chosen_colors[0]

    dots = [(edge_pos, marker_color), (inner_pos, marker_color)]
    occupied = {edge_pos, inner_pos}
    meta: dict[str, int | list[int] | str] = {
        "kind": kind,
        "step": step,
        "n_companions": n_companions,
        "companion_indices": [],
    }

    if n_companions:
        indices = _choose_companion_indices(len(beats), n_companions)
        if indices is None:
            return None
        companion_colors = chosen_colors[1 : 1 + n_companions]
        for beat_idx, color in zip(indices, companion_colors):
            pos = beats[beat_idx]
            dots.append((pos, color))
            occupied.add(pos)
        meta["companion_indices"] = indices

    if add_bystander:
        bystander_color = chosen_colors[-1]
        candidates = _available_bystander_positions(edge, line_len, step, occupied)
        if not candidates:
            if kind == "bystander_only":
                return None
        else:
            dots.append((random.choice(candidates), bystander_color))

    return dots, meta


def _fusion_deltas(inp: np.ndarray) -> tuple[int, int]:
    combo = main(inp)
    collision = _render(inp, "collision")
    segment = _render(inp, "segment")
    return int(np.count_nonzero(combo != collision)), int(np.count_nonzero(combo != segment))


def _gen_attempt() -> np.ndarray | None:
    n_active = random.randint(*_ACTIVE_LINES_RANGE)
    edge = random.choice(_EDGES)
    by_row = edge in ("left", "right")

    dims = _choose_dims(edge, n_active)
    if dims is None:
        return None
    H, W, num_lines, line_len = dims

    active_lines = _spaced_indices(n_active, num_lines)
    if not active_lines:
        return None

    palette_size = random.randint(*_PALETTE_SIZE_RANGE)
    palette = random.sample(_ALL_COLORS, palette_size)
    kinds = _line_kind_sequence(n_active)
    prefer_three_lines = set()
    multi_candidates = [li for li, kind in zip(active_lines, kinds) if kind == "multi"]
    if multi_candidates:
        prefer_three_lines = set(
            random.sample(multi_candidates, k=min(2, len(multi_candidates)))
        )

    def cell(line: int, pos: int) -> tuple[int, int]:
        return (line, pos) if by_row else (pos, line)

    inp = np.zeros((H, W), dtype=int)
    all_rc: list[tuple[int, int]] = []
    shooting_edge_dots = 0
    multi_lines = 0
    total_companions = 0
    three_companion_lines = 0

    for li, kind in zip(active_lines, kinds):
        built = _build_line_dots(edge, line_len, palette, kind, li in prefer_three_lines)
        if built is None:
            return None
        dots, meta = built

        for pos, color in dots:
            r, c = cell(li, pos)
            if inp[r, c] != 0:
                return None
            inp[r, c] = color
            all_rc.append((r, c))

        shooting_edge_dots += 1
        n_companions = int(meta["n_companions"])
        if n_companions >= 2:
            multi_lines += 1
            total_companions += n_companions
        if n_companions == 3:
            three_companion_lines += 1

    if multi_lines < 2 or total_companions < 5:
        return None
    if three_companion_lines == 0 and total_companions < 6:
        return None

    other_counts = _count_other_edge_dots(all_rc, edge, H, W)
    if any(count >= shooting_edge_dots for count in other_counts.values()):
        return None

    try:
        out = main(inp)
        delta_collision, delta_segment = _fusion_deltas(inp)
    except Exception:
        return None

    if np.array_equal(inp, out):
        return None
    if delta_collision < 4 or delta_segment < 4:
        return None

    return inp


def generate_input() -> np.ndarray:
    for _ in range(_MAX_RETRIES):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError(f"Failed to generate valid input after {_MAX_RETRIES} attempts")
