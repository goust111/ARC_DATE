# Combo variant of 16de56c4 (index 14)
# combo_spec: D6/3+D7/2
# Name: 碰撞变色+伴随点变色

from __future__ import annotations

from collections import defaultdict
import random

import numpy as np

from arc_utils import get_bg


# Design analysis for the combo variant
# -------------------------------------
# concretization_and_trace:
# - 每条活跃行/列仍由一对同色 marker dot 定义 step；其中一个 marker 固定落在唯一
#   射击边上，射线从该边朝网格内部按 beat 前进。
# - 融合前的三种相关行为分别是：
#   1) 原题 companion: 把 marker 到 companion 之间的 beat 全部刷成 companion 色并停止。
#   2) D6/3: companion 变成“碰撞后继续前进”的换色点，collision beat 与后续 beat 都刷成
#      companion 色。
#   3) D7/2: companion 仍是终点，但 companion 自己反而变成 marker 色。
# - 融合后要同时满足“继续前进”和“碰撞即互换”：
#   例子（left edge, step=3）输入位置为 0:M, 3:M, 9:C。
#   原题输出: 0,3,6,9 -> C，之后停止。
#   D6/3 输出: 0,3,6 -> M，9,12,15 -> C。
#   融合输出: 0,3,6,9 -> M，12,15 -> C。
#   也就是 companion 位置本身完成颜色互换，射线从下一个 beat 开始改用 companion 色。
#
# complexity_calibration:
# - 仍保持每条线最多 3 个输入点：2 个 marker + 0/1 个 extra。这样观察者只需区分
#   marker-only、collision、bystander 三类局部结构，不会引入新的对象类别爆炸。
# - generator 强制每题至少出现 2 条 collision line，并且至少各出现 1 条 marker-only /
#   bystander 线；这样 2-4 个训练样例里稳定能同时看到融合分支与对照分支。
# - collision line 还要求碰撞后至少还有 1 个 beat，否则“继续前进”看不见，题目会退化成
#   只剩 companion 变色。
# - 不是所有第三个点都会触发变色：只有 aligned companion 才会触发，off-beat bystander
#   明确保留原色，marker-only 则提供“无第三点时继续用 marker 色”的基线。
#
# implicit_assumption_audit:
# - 原题使用无方向区间 [lo, hi]。融合后这个前提失效，因为“碰撞前”“碰撞后”依赖从射击边
#   出发的顺序；right/bottom 发射时尤其不能再只看 min/max。
# - 原题和 D6/3 都允许 collision beat 直接变成 companion 色；融合后不成立，solver 必须
#   显式识别 companion_pos，并先把该格改成 marker 色。
# - 原题对 aligned companion 的作用是“终止+覆色”；融合后变成“交换+继续”，所以 generator
#   必须保证 companion 在 inner marker 之后，并且不是最后一个 beat。
# - solver 现在额外假设：每条非空线恰有一个重复色、该重复色恰有两个点、edge marker 与
#   inward marker 相隔一个 step、aligned extra 最多一个。下面的 generator 会逐项构造并验证。
#
# placement_legality:
# - 活跃线之间保留至少 1 行/列空隔，避免不同线的对象在视觉上挤在一起。
# - collision companion 只能放在 beats[2:-1]，既保证它位于 inner marker 之后，也保证后面
#   还留有 continuation beat。
# - bystander 只能放在非 beat 的内部位置；这既避免覆盖轨迹，也避免额外边界点破坏唯一射击边。
# - 所有放置都在单条线内部完成，不会跨线重叠；写入前也会检查目标格仍为背景。
#
# visual_observability_and_cleanliness:
# - 融合效果在单条 collision 线上会同时出现三段证据：前段 marker 色、collision 格自身仍为
#   marker 色、collision 之后的尾段变成 companion 色。这比单独 D6/3 或 D7/2 更强，可从
#   2-3 个样例里直接看出“撞上去发生互换但射线没停”。
# - 颜色变化完全由局部结构触发，不依赖颜色编号大小、扫描顺序等不可视因素；同一种颜色可在
#   不同行复用，但谁接管谁只由该行的 aligned companion 决定。
# - 网格仍保持稀疏：每条线最多 3 个输入点，输出只在一维 beat 轨迹上着色，避免多层覆盖导致
#   视觉混乱。


def _detect_edge(g: np.ndarray, bg: int) -> str:
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


def _edge_params(edge: str, line_len: int) -> tuple[int, int]:
    if edge in ("left", "top"):
        return 0, 1
    return line_len - 1, -1


def _ordered_beats(edge: str, line_len: int, step: int) -> list[int]:
    edge_pos, travel_dir = _edge_params(edge, line_len)
    stop = line_len if travel_dir == 1 else -1
    return list(range(edge_pos, stop, travel_dir * step))


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fire periodic rays from one unique border; aligned companion dots swap colors on impact and
    the ray keeps traveling in the new color.

    Input structure:
    - The background is the dominant color, with sparse non-background dots on a subset of rows
      or columns.
    - Exactly one border has strictly more non-background dots than the other three; that border
      is the shooting edge and determines the ray direction.
    - Every active shooting-direction line contains one repeated marker color with exactly 2 dots:
      one on the shooting edge and one one-step inward, so their gap defines the beat spacing.
    - Some active lines also contain one differently colored aligned companion farther along the
      same beat sequence; other extra dots, if present, are off-beat bystanders.

    Rule:
    - Identify the unique shooting edge; left/right edges fire across rows, top/bottom edges fire
      down columns.
    - On each active line, recover the ordered beat sequence from the 2 marker dots.
    - Without an aligned companion, paint every beat on that line with the marker color.
    - With an aligned companion, beats up to and including the collision stay marker-colored, the
      companion cell itself flips to the marker color, and all later beats switch to the companion
      color while off-beat bystanders remain unchanged.

    Key insight: the repeated marker pair still determines the period, but the aligned third dot is
    no longer a stopping wall. It is a direction-sensitive swap event: the collision cell receives
    the incoming marker color, and only subsequent beats inherit the companion color.
    """
    g = np.array(grid, dtype=int)
    out = g.copy()
    bg = int(get_bg(g))
    H, W = g.shape

    edge = _detect_edge(g, bg)
    by_row = edge in ("left", "right")
    num_lines = H if by_row else W
    line_len = W if by_row else H
    edge_pos, travel_dir = _edge_params(edge, line_len)

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
        assert len(repeating) == 1, (
            f"Line {li} ({'row' if by_row else 'col'}) must have exactly one repeating color: "
            f"{dict(colors)}"
        )
        marker, marker_positions = next(iter(repeating.items()))
        assert len(marker_positions) == 2, (
            f"Line {li} ({'row' if by_row else 'col'}) marker {marker} "
            f"has {len(marker_positions)} dots (expected 2): {marker_positions}"
        )
        assert edge_pos in marker_positions, (
            f"Line {li} ({'row' if by_row else 'col'}) marker {marker} "
            f"must touch the shooting edge: {marker_positions}"
        )

        step = abs(marker_positions[1] - marker_positions[0])
        expected_inner = edge_pos + travel_dir * step
        assert expected_inner in marker_positions, (
            f"Line {li} ({'row' if by_row else 'col'}) marker {marker} "
            f"must have its second dot one step inward: {marker_positions}"
        )

        beats = _ordered_beats(edge, line_len, step)
        beat_to_index = {pos: idx for idx, pos in enumerate(beats)}

        aligned_hits: list[tuple[int, int, int]] = []
        for other_color, positions in colors.items():
            if other_color == marker:
                continue
            for pos in positions:
                beat_idx = beat_to_index.get(pos)
                if beat_idx is not None and beat_idx >= 2:
                    aligned_hits.append((beat_idx, pos, other_color))

        assert len(aligned_hits) <= 1, (
            f"Line {li} ({'row' if by_row else 'col'}) has multiple aligned companions: "
            f"{aligned_hits}"
        )

        collision_idx = None
        collision_pos = None
        companion_color = None
        if aligned_hits:
            collision_idx, collision_pos, companion_color = aligned_hits[0]

        for beat_idx, pos in enumerate(beats):
            rc = cell(li, pos)

            if collision_idx is not None and beat_idx == collision_idx:
                out[rc] = marker
                continue

            if collision_idx is not None and beat_idx > collision_idx:
                paint = int(companion_color)
            else:
                paint = int(marker)

            cur = int(out[rc])
            if cur == bg or cur == marker:
                out[rc] = paint

    return out


_ROWS_RANGE = (7, 12)
_COLS_RANGE = (9, 21)
_ACTIVE_LINES_RANGE = (4, 9)
_STEP_RANGE = (1, 5)
_COLORS = list(range(1, 10))
_EDGES = ("top", "bottom", "left", "right")

_P_MARKER_ONLY = 0.20
_P_COLLISION = 0.60

_MAX_RETRIES = 400


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


def _valid_steps(line_len: int, kind: str) -> list[int]:
    upper = min(_STEP_RANGE[1], line_len - 1)
    steps = []
    for step in range(_STEP_RANGE[0], upper + 1):
        beats = _ordered_beats("left", line_len, step)
        if kind == "collision" and len(beats) < 4:
            continue
        if kind == "bystander" and step == 1:
            continue
        steps.append(step)
    return steps


def _line_kind_sequence(n_active: int) -> list[str]:
    kinds = ["collision", "collision", "marker_only", "bystander"]
    while len(kinds) < n_active:
        roll = random.random()
        if roll < _P_MARKER_ONLY:
            kinds.append("marker_only")
        elif roll < _P_MARKER_ONLY + _P_COLLISION:
            kinds.append("collision")
        else:
            kinds.append("bystander")
    random.shuffle(kinds)
    return kinds


def _build_line_dots(
    edge: str,
    line_len: int,
    marker_color: int,
    kind: str,
) -> tuple[list[tuple[int, int]], dict[str, int | str]] | None:
    valid_steps = _valid_steps(line_len, kind)
    if not valid_steps:
        return None

    step = random.choice(valid_steps)
    edge_pos, travel_dir = _edge_params(edge, line_len)
    inner_pos = edge_pos + travel_dir * step
    beats = _ordered_beats(edge, line_len, step)
    beat_set = set(beats)

    dots = [(edge_pos, marker_color), (inner_pos, marker_color)]
    meta: dict[str, int | str] = {"kind": kind, "step": step, "beat_count": len(beats)}
    other_colors = [color for color in _COLORS if color != marker_color]

    if kind == "collision":
        if len(beats) < 4 or not other_colors:
            return None
        companion_pos = random.choice(beats[2:-1])
        companion_color = random.choice(other_colors)
        dots.append((companion_pos, companion_color))
        meta["companion_pos"] = companion_pos
        meta["companion_color"] = companion_color
        meta["post_collision_beats"] = len(beats) - beats.index(companion_pos) - 1
        return dots, meta

    if kind == "bystander" and other_colors:
        non_beats = [
            pos
            for pos in range(1, line_len - 1)
            if pos not in beat_set and pos not in (edge_pos, inner_pos)
        ]
        if non_beats:
            dots.append((random.choice(non_beats), random.choice(other_colors)))
            return dots, meta

    meta["kind"] = "marker_only"
    return dots, meta


def _combo_behavior_metrics(inp: np.ndarray, out: np.ndarray) -> dict[str, int]:
    bg = int(get_bg(inp))
    edge = _detect_edge(inp, bg)
    by_row = edge in ("left", "right")
    num_lines = inp.shape[0] if by_row else inp.shape[1]
    line_len = inp.shape[1] if by_row else inp.shape[0]
    edge_pos, travel_dir = _edge_params(edge, line_len)

    collision_lines = 0
    swapped_collision_cells = 0
    tail_recolored_lines = 0
    combo_signature_lines = 0

    for li in range(num_lines):
        line_in = inp[li, :] if by_row else inp[:, li]
        line_out = out[li, :] if by_row else out[:, li]

        colors: dict[int, list[int]] = defaultdict(list)
        for pos, value in enumerate(line_in):
            value = int(value)
            if value != bg:
                colors[value].append(pos)
        if not colors:
            continue

        repeating = {color: positions for color, positions in colors.items() if len(positions) >= 2}
        if len(repeating) != 1:
            continue
        marker, marker_positions = next(iter(repeating.items()))
        if len(marker_positions) != 2 or edge_pos not in marker_positions:
            continue

        step = abs(marker_positions[1] - marker_positions[0])
        if edge_pos + travel_dir * step not in marker_positions:
            continue

        beats = _ordered_beats(edge, line_len, step)
        beat_to_index = {pos: idx for idx, pos in enumerate(beats)}
        aligned_hits: list[tuple[int, int, int]] = []
        for other_color, positions in colors.items():
            if other_color == marker:
                continue
            for pos in positions:
                beat_idx = beat_to_index.get(pos)
                if beat_idx is not None and beat_idx >= 2:
                    aligned_hits.append((beat_idx, pos, other_color))
        if len(aligned_hits) != 1:
            continue

        collision_idx, collision_pos, companion_color = aligned_hits[0]
        post_positions = beats[collision_idx + 1 :]
        if not post_positions:
            continue

        collision_lines += 1
        if int(line_out[collision_pos]) == marker:
            swapped_collision_cells += 1

        if any(int(line_out[pos]) == companion_color for pos in post_positions):
            tail_recolored_lines += 1

        if (
            int(line_out[collision_pos]) == marker
            and all(int(line_out[pos]) == marker for pos in beats[: collision_idx + 1])
            and all(int(line_out[pos]) == companion_color for pos in post_positions)
        ):
            combo_signature_lines += 1

    return {
        "collision_lines": collision_lines,
        "swapped_collision_cells": swapped_collision_cells,
        "tail_recolored_lines": tail_recolored_lines,
        "combo_signature_lines": combo_signature_lines,
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
    kind_counts = defaultdict(int)

    for li, kind in zip(active_lines, _line_kind_sequence(n_active)):
        marker_color = random.choice(_COLORS)
        built = _build_line_dots(edge, line_len, marker_color, kind)
        if built is None:
            return None
        dots, meta = built

        for pos, color in dots:
            r, c = cell(li, pos)
            if inp[r, c] != 0:
                return None
            inp[r, c] = color
            all_rc.append((r, c))

        actual_kind = str(meta["kind"])
        kind_counts[actual_kind] += 1
        if actual_kind == "collision" and int(meta["post_collision_beats"]) < 1:
            return None

    if kind_counts["collision"] < 2:
        return None
    if kind_counts["marker_only"] < 1 or kind_counts["bystander"] < 1:
        return None

    shooting_edge_dots = n_active
    other_counts = _count_other_edge_dots(all_rc, edge, H, W)
    if any(count >= shooting_edge_dots for count in other_counts.values()):
        return None

    try:
        out = main(inp)
    except Exception:
        return None

    if np.array_equal(inp, out):
        return None

    metrics = _combo_behavior_metrics(inp, out)
    if metrics["combo_signature_lines"] == 0:
        return None

    return inp


def generate_input() -> np.ndarray:
    for _ in range(_MAX_RETRIES):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError(f"Failed to generate valid input after {_MAX_RETRIES} attempts")
