# Combo variant of 16de56c4 (index 6)
# combo_spec: D6/3+D12/2
# Name: 碰撞变色+反射传播

from __future__ import annotations

from collections import defaultdict
import random

import numpy as np

from arc_utils import get_bg


# Design analysis for the combo variant
# -------------------------------------
# a) concretization_and_trace:
# - 原题每条活跃线都只做一次“区间填充”判断：若存在 aligned companion，则 marker
#   与 companion 之间的整段 beat 被 companion 色统一覆盖，并在 companion 处结束。
# - 这次融合把 aligned companion 改成“反射检查点 + 变色源”：
#   1) 射线仍从唯一射击边出发，marker 的两个点定义 beat。
#   2) 命中某个未触发 companion 时，方向立刻反转，当前颜料切换成该 companion 色。
#   3) 若回到射击边但后方还有未触发 companion，则再向外反射一次，形成 Z 字往返。
# - 具体例子（left edge, step=2, beats=[0,2,4,6,8,10]）：
#   input: 1@0, 1@2, 6@4, 7@8, 3@10。
#   原题只会把 [0,4] 的 beat 刷成 6，后面的 7/3 只是输入点。
#   融合后轨迹是：
#     0 -> 2(M) -> 4(hit 6, reverse)
#     -> 2(6) -> 0(6, source reflect because 7/3 unused)
#     -> 2 -> 4 -> 6(6) -> 8(hit 7, reverse)
#     -> ... -> 0(source reflect because 3 unused)
#     -> ... -> 10(hit 3, reverse) -> ... -> 0(stop)
#   所以最终 beat 颜色分段出现 6 / 7 / 3，而不是原题那种单段覆盖。
#
# b) complexity_calibration:
# - 仍保留“每条线只有一组重复 marker”这一核心线索，避免对象识别歧义。
# - companion 数量限制在 0-3 个；这足够产生多次反射和多段颜色，但不会把线变成
#   难以肉眼追踪的长模拟。
# - generator 强制至少一条 3-companion 线，并检查输出中确实出现 >=3 种 beat 色，
#   防止融合退化成原题或退化成单次碰撞。
#
# c) implicit_assumption_audit:
# - 原题假设“第一个 aligned companion 决定一个无方向区间”，因此只需要 lo/hi。
#   融合后失效，必须显式枚举从射击边出发的 beat 顺序。
# - 原题假设“一个活跃线最多一个 aligned companion”；融合后允许多个 singleton
#   companion，solver 必须维护 unused checkpoint 集合。
# - 原题默认一次填充即可结束；融合后需要 direction、current_color 和回边继续发射
#   的状态，直到所有 companion 都触发且轨迹回到源边。
# - 若允许后续颜色自由覆写整条路径，会抹掉早期分段；因此 solver 只重涂 bg 或
#   原 marker 色，已形成的 companion 色段保留不变。
#
# d) placement_legality:
# - 每条活跃线恰有一个重复 marker 色，且两个 marker 必须在 edge beat 的前两拍。
# - 所有 companion 都是 singleton，位于同一 beat 序列且严格在 inner marker 之后。
# - bystander 只能放在非 beat 的内部位置，避免边界计数歧义，也避免压到轨迹上。
#
# e) visual_observability:
# - 解题者能先看出“两个同色点给出步长”，再注意到 aligned 单点不是终点而是反射点：
#   输出不会只出现一个 companion 色，而会按碰撞顺序生成多段颜色。
# - generator 会同时产出 plain / bystander 线和 multi-bounce 线，让对比更明显：
#   有的线仍像原题那样单色前进，有的线则明显出现多次折返后的分段多色轨迹。
#
# f) visual_cleanliness:
# - 所有变化都限制在各自的稀疏算术级数上；活跃线之间仍保留至少 1 行/列空隔。
# - companion 与 bystander 不重叠，且 bystander 不在 beat 上，所以不会与真实轨迹混淆。
#
# g) color_inference:
# - 颜色变化完全由空间结构触发：命中哪个 aligned companion，就继承哪个颜色。
# - 颜色编号大小、排序、是否重复于其他线都不参与规则；同色可跨线复用，但单线内的
#   颜色推进顺序只由碰撞位置决定。


def _edge_params(edge: str, line_len: int) -> tuple[int, int]:
    if edge in ("left", "top"):
        return 0, 1
    return line_len - 1, -1


def _ordered_beats(edge: str, line_len: int, step: int) -> list[int]:
    edge_pos, travel_dir = _edge_params(edge, line_len)
    stop = line_len if travel_dir == 1 else -1
    return list(range(edge_pos, stop, travel_dir * step))


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


def _line_profile(
    g: np.ndarray,
    line: int,
    *,
    by_row: bool,
    edge: str,
    bg: int,
) -> dict | None:
    line_len = g.shape[1] if by_row else g.shape[0]
    edge_pos, travel_dir = _edge_params(edge, line_len)

    def cell(pos: int) -> tuple[int, int]:
        return (line, pos) if by_row else (pos, line)

    colors: dict[int, list[int]] = defaultdict(list)
    for pos in range(line_len):
        value = int(g[cell(pos)])
        if value != bg:
            colors[value].append(pos)

    if not colors:
        return None

    repeating = {color: positions for color, positions in colors.items() if len(positions) >= 2}
    assert len(repeating) == 1, (
        f"Line {line} ({'row' if by_row else 'col'}) must have exactly one repeating color: {dict(colors)}"
    )
    marker, mpos = next(iter(repeating.items()))
    assert len(mpos) == 2, (
        f"Line {line} ({'row' if by_row else 'col'}) marker {marker} "
        f"has {len(mpos)} dots (expected 2): {mpos}"
    )
    assert edge_pos in mpos, (
        f"Line {line} ({'row' if by_row else 'col'}) marker {marker} "
        f"must touch the shooting edge {edge_pos}: {mpos}"
    )

    step = abs(mpos[1] - mpos[0])
    assert step >= 1
    expected_inner = edge_pos + travel_dir * step
    assert expected_inner in mpos, (
        f"Line {line} ({'row' if by_row else 'col'}) marker {marker} "
        f"must have its second dot one beat inward: {mpos}"
    )

    beats = _ordered_beats(edge, line_len, step)
    beat_to_idx = {pos: idx for idx, pos in enumerate(beats)}

    companions: list[tuple[int, int]] = []
    bystanders: list[tuple[int, int]] = []
    for color, positions in colors.items():
        if color == marker:
            continue
        assert len(positions) == 1, (
            f"Line {line} ({'row' if by_row else 'col'}) non-marker color {color} "
            f"must be singleton: {positions}"
        )
        pos = positions[0]
        beat_idx = beat_to_idx.get(pos)
        if beat_idx is not None and beat_idx >= 2:
            companions.append((beat_idx, int(color)))
        else:
            bystanders.append((pos, int(color)))

    companions.sort(key=lambda item: item[0])
    return {
        "marker": int(marker),
        "step": int(step),
        "beats": beats,
        "companions": companions,
        "bystanders": bystanders,
    }


def _simulate_line(
    out: np.ndarray,
    line: int,
    profile: dict,
    *,
    by_row: bool,
    bg: int,
) -> None:
    marker = int(profile["marker"])
    beats: list[int] = list(profile["beats"])
    unused = {int(idx): int(color) for idx, color in profile["companions"]}

    def cell(pos: int) -> tuple[int, int]:
        return (line, pos) if by_row else (pos, line)

    cur_idx = 0
    direction = 1
    current_color = marker
    max_steps = max(1, 4 * len(beats) * (len(unused) + 1))

    for _ in range(max_steps):
        next_idx = cur_idx + direction
        if not (0 <= next_idx < len(beats)):
            direction *= -1
            next_idx = cur_idx + direction
        cur_idx = next_idx
        rc = cell(beats[cur_idx])

        if cur_idx in unused:
            hit_color = unused.pop(cur_idx)
            out[rc] = hit_color
            current_color = hit_color
            direction *= -1
        else:
            cur_value = int(out[rc])
            if cur_value == bg or cur_value == marker:
                out[rc] = current_color

        if cur_idx == 0 and direction == -1 and not unused:
            return

    raise AssertionError(f"Line {line} did not terminate under combo reflection: {profile}")


def _baseline_main(grid: np.ndarray) -> np.ndarray:
    g = np.array(grid, dtype=int)
    out = g.copy()
    bg = int(get_bg(g))
    edge = _shooting_edge(g, bg)
    by_row = edge in ("left", "right")
    num_lines = g.shape[0] if by_row else g.shape[1]
    line_len = g.shape[1] if by_row else g.shape[0]

    def cell(line: int, pos: int) -> tuple[int, int]:
        return (line, pos) if by_row else (pos, line)

    for li in range(num_lines):
        colors: dict[int, list[int]] = defaultdict(list)
        for pos in range(line_len):
            value = int(g[cell(li, pos)])
            if value != bg:
                colors[value].append(pos)

        if not colors:
            continue

        repeating = {color: positions for color, positions in colors.items() if len(positions) >= 2}
        assert len(repeating) == 1
        marker, mpos = next(iter(repeating.items()))
        assert len(mpos) == 2
        a, b = sorted(mpos)
        step = b - a
        offset = a % step

        companion = None
        for other_color, positions in colors.items():
            if other_color == marker:
                continue
            if any(pos % step == offset for pos in positions):
                companion = other_color
                break

        if companion is not None:
            paint = companion
            all_pos = mpos + colors[companion]
            lo, hi = min(all_pos), max(all_pos)
        else:
            paint = marker
            lo, hi = 0, line_len - 1

        for pos in range(lo, hi + 1):
            if pos % step != offset:
                continue
            rc = cell(li, pos)
            cur = int(out[rc])
            if cur == bg or cur == marker:
                out[rc] = paint

    return out


def _combo_signature(grid: np.ndarray) -> dict[str, int]:
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))
    edge = _shooting_edge(g, bg)
    by_row = edge in ("left", "right")
    num_lines = g.shape[0] if by_row else g.shape[1]

    out = main(g)
    baseline = _baseline_main(g)

    def cell(line: int, pos: int) -> tuple[int, int]:
        return (line, pos) if by_row else (pos, line)

    active_lines = 0
    total_companions = 0
    zero_companion_lines = 0
    single_companion_lines = 0
    double_companion_lines = 0
    multi_companion_lines = 0
    three_companion_lines = 0
    bystander_lines = 0
    changed_vs_original_lines = 0
    three_color_output_lines = 0

    for li in range(num_lines):
        profile = _line_profile(g, li, by_row=by_row, edge=edge, bg=bg)
        if profile is None:
            continue
        active_lines += 1
        n_comp = len(profile["companions"])
        total_companions += n_comp
        if n_comp == 0:
            zero_companion_lines += 1
        if n_comp == 1:
            single_companion_lines += 1
        if n_comp == 2:
            double_companion_lines += 1
        if n_comp >= 2:
            multi_companion_lines += 1
        if n_comp >= 3:
            three_companion_lines += 1
        if profile["bystanders"]:
            bystander_lines += 1

        beats = list(profile["beats"])
        out_colors = {int(out[cell(li, pos)]) for pos in beats if int(out[cell(li, pos)]) != bg}
        if len(out_colors) >= 3:
            three_color_output_lines += 1

        if any(int(out[cell(li, pos)]) != int(baseline[cell(li, pos)]) for pos in beats):
            changed_vs_original_lines += 1

    diff_cells = int(np.count_nonzero(out != baseline))
    return {
        "active_lines": active_lines,
        "total_companions": total_companions,
        "zero_companion_lines": zero_companion_lines,
        "single_companion_lines": single_companion_lines,
        "double_companion_lines": double_companion_lines,
        "multi_companion_lines": multi_companion_lines,
        "three_companion_lines": three_companion_lines,
        "bystander_lines": bystander_lines,
        "three_color_output_lines": three_color_output_lines,
        "changed_vs_original_lines": changed_vs_original_lines,
        "diff_cells_vs_original": diff_cells,
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Propagate periodic rays from one unique edge; aligned companions reflect the ray and hand off its color.

    Input structure:
    - The background is the dominant color, with sparse non-bg dots arranged on a subset of rows or columns.
    - Exactly one boundary has strictly more non-bg dots than the other three; that boundary is the shooting edge.
    - Every active shooting-direction line contains exactly one repeated marker color with 2 dots:
      one on the shooting edge and one one beat inward, so their gap reveals the step.
    - Extra singleton dots on the same beat are aligned companions; extra singleton dots off that beat are bystanders.

    Rule:
    - Find the unique shooting edge. Left/right edges fire across rows; top/bottom edges fire down columns.
    - On each active line, enumerate beat positions in the order reached from the shooting edge.
    - Launch one ray from the edge marker. When it reaches an unused aligned companion, it immediately reverses
      direction and the active paint color becomes that companion's color; reaching a line end also reflects it.
    - The ray keeps bouncing until every aligned companion on that line has been triggered and the path comes back
      to the shooting edge. Only background cells and original marker cells may be repainted; bystanders stay fixed.

    Key insight: the marker pair still encodes a modular beat, but aligned singleton dots are no longer one-shot
    walls. They are ordered color-changing bounce checkpoints, so the solver has to follow the ray through several
    passes and preserve earlier color segments instead of collapsing the line to a single interval fill.
    """
    g = np.array(grid, dtype=int)
    out = g.copy()
    bg = int(get_bg(g))
    edge = _shooting_edge(g, bg)
    by_row = edge in ("left", "right")
    num_lines = g.shape[0] if by_row else g.shape[1]

    for li in range(num_lines):
        profile = _line_profile(g, li, by_row=by_row, edge=edge, bg=bg)
        if profile is None:
            continue
        _simulate_line(out, li, profile, by_row=by_row, bg=bg)

    return out


# --------------- generator ---------------

_ROWS_RANGE = (7, 12)
_COLS_RANGE = (9, 21)
_ACTIVE_LINES_RANGE = (4, 9)
_STEP_RANGE = (1, 5)
_MAX_COMPANIONS = 3
_COLORS = list(range(1, 10))
_EDGES = ("top", "bottom", "left", "right")

_P_BYSTANDER = 0.35
_MAX_RETRIES = 500


def _choose_dims(edge: str, n_active: int) -> tuple[int, int, int, int] | None:
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
    dots_rc: list[tuple[int, int]],
    edge: str,
    H: int,
    W: int,
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


def _valid_steps(line_len: int, n_companions: int, *, need_nonbeat: bool = False) -> list[int]:
    upper = min(_STEP_RANGE[1], line_len - 1)
    steps = []
    min_beats = max(2 + n_companions, 6 if n_companions >= 3 else 0)
    for step in range(_STEP_RANGE[0], upper + 1):
        beats = _ordered_beats("left", line_len, step)
        beat_count = len(beats)
        if beat_count < min_beats:
            continue
        if need_nonbeat:
            beat_set = set(beats)
            if not any(pos not in beat_set for pos in range(1, line_len - 1)):
                continue
        steps.append(step)
    return steps


def _choose_companion_indices(beat_count: int, n_companions: int) -> list[int] | None:
    candidates = list(range(2, beat_count))
    if len(candidates) < n_companions:
        return None

    if n_companions <= 1:
        return sorted(random.sample(candidates, n_companions))

    gap = 2
    available = beat_count - 2 - gap * (n_companions - 1)
    if available >= n_companions:
        picks = sorted(random.sample(range(available), n_companions))
        return [2 + picks[i] + gap * i for i in range(n_companions)]

    return sorted(random.sample(candidates, n_companions))


def _line_plan_sequence(n_active: int) -> list[tuple[int, bool]]:
    plans = [(0, True), (1, False), (2, False), (3, False)]
    while len(plans) < n_active:
        n_comp = random.choices([1, 2, 3, 0], weights=[2, 4, 3, 1])[0]
        add_bystander = random.random() < _P_BYSTANDER
        plans.append((n_comp, add_bystander))
    random.shuffle(plans)
    return plans[:n_active]


def _build_line_dots(
    edge: str,
    line_len: int,
    marker_color: int,
    n_companions: int,
    add_bystander: bool,
) -> tuple[list[tuple[int, int]], dict[str, int]] | None:
    valid_steps = _valid_steps(line_len, n_companions, need_nonbeat=add_bystander)
    if not valid_steps:
        return None

    step = random.choice(valid_steps)
    edge_pos, travel_dir = _edge_params(edge, line_len)
    inner_pos = edge_pos + travel_dir * step
    beats = _ordered_beats(edge, line_len, step)
    companion_indices = _choose_companion_indices(len(beats), n_companions)
    if companion_indices is None:
        return None

    free_colors = [color for color in _COLORS if color != marker_color]
    if len(free_colors) < n_companions:
        return None

    dots = [(edge_pos, marker_color), (inner_pos, marker_color)]
    companion_colors = random.sample(free_colors, n_companions)
    for idx, color in zip(companion_indices, companion_colors):
        dots.append((beats[idx], color))

    used_colors = {marker_color, *companion_colors}
    if add_bystander:
        other_colors = [color for color in _COLORS if color not in used_colors]
        beat_set = set(beats)
        non_beats = [pos for pos in range(1, line_len - 1) if pos not in beat_set]
        if not other_colors or not non_beats:
            return None
        dots.append((random.choice(non_beats), random.choice(other_colors)))

    return dots, {
        "step": step,
        "beat_count": len(beats),
        "n_companions": n_companions,
    }


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

    def cell(line: int, pos: int) -> tuple[int, int]:
        return (line, pos) if by_row else (pos, line)

    inp = np.zeros((H, W), dtype=int)
    all_rc: list[tuple[int, int]] = []
    shooting_edge_dots = 0

    for li, (n_companions, add_bystander) in zip(active_lines, _line_plan_sequence(n_active)):
        marker_color = random.choice(_COLORS)
        built = _build_line_dots(edge, line_len, marker_color, n_companions, add_bystander)
        if built is None:
            return None
        dots, _meta = built

        for pos, color in dots:
            r, c = cell(li, pos)
            if inp[r, c] != 0:
                return None
            inp[r, c] = color
            all_rc.append((r, c))

        shooting_edge_dots += 1

    other_counts = _count_other_edge_dots(all_rc, edge, H, W)
    if any(count >= shooting_edge_dots for count in other_counts.values()):
        return None

    try:
        out = main(inp)
    except Exception:
        return None

    if np.array_equal(inp, out):
        return None

    sig = _combo_signature(inp)
    if sig["zero_companion_lines"] == 0:
        return None
    if sig["single_companion_lines"] == 0:
        return None
    if sig["double_companion_lines"] == 0:
        return None
    if sig["three_companion_lines"] == 0:
        return None
    if sig["bystander_lines"] == 0:
        return None
    if sig["three_color_output_lines"] == 0:
        return None
    if sig["changed_vs_original_lines"] == 0:
        return None
    if sig["diff_cells_vs_original"] < 3:
        return None
    if sig["total_companions"] < max(4, n_active):
        return None

    return inp


def generate_input() -> np.ndarray:
    for _ in range(_MAX_RETRIES):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError(f"Failed to generate valid input after {_MAX_RETRIES} attempts")
