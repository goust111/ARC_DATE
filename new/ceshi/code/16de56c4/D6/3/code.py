# Variant of 16de56c4
# Dimension: D6 — 颜色配对规则 (trajectory)
# Change: 伴随点与标记点在同一模(mod step)类中，伴随点颜色覆盖标记色成为填充色... → 碰撞变色

from __future__ import annotations

from collections import defaultdict
import random

import numpy as np

from arc_utils import get_bg


# Design analysis for the D6 variant
# ----------------------------------
# concretization_and_trace:
# - 每条活跃行/列仍由一对同色 marker dot 定义周期 step，并且其中一个 marker
#   固定落在唯一射击边上，射线沿着该边朝网格内部的方向前进。
# - 原题里 aligned companion 会把整段 [marker, companion] 的 beat 统一刷成
#   companion 色，并在 companion 处停止；方向在原题中并不重要，因为只看数轴区间。
# - 现在 aligned companion 改成“碰撞换色点”：在 companion 之前的 beat 继续用
#   marker 色；从 companion 这个 beat 开始，后续 beat 改用 companion 色并继续向前。
# - 具体例子（left edge, step=3）：
#   input 在位置 0:M, 3:M, 9:C。
#   原题输出会把 0,3,6,9 都刷成 C，并且不再继续。
#   变体输出会把 0,3,6 刷成 M，把 9,12,15,... 刷成 C。
# - 因此 solver 必须按“从射击边开始的 beat 顺序”处理，而不能像原题那样只看
#   [lo, hi] 数值区间；否则 right/bottom 发射时会把前后段弄反。
#
# complexity_calibration:
# - 仍保持每条线最多 3 个输入点（2 marker + 0/1 extra），避免对象类型暴涨。
# - extra dot 仍只有两类：aligned companion 或 off-beat bystander。这样人类仍能
#   先读出周期，再判断 aligned 的点是“换色点”而不是“终点”。
# - 为了让变体在少量样例里可见，generator 强制至少生成一条 collision line，
#   且该 companion 后面至少还留 1 个 beat 继续前进，保证输出出现分段颜色。
#
# implicit_assumption_audit:
# - 原题 `lo, hi = min(all_ps), max(all_ps)` 默认 companion 的作用是定义一个无方向
#   的闭区间；变体下这个前提失效，必须改成有序 beat 序列。
# - 原题 `paint = companion` 会把 marker 也整体覆写成 companion；变体下只能在
#   collision 之后换色，collision 之前的 marker 段必须保留 marker 色。
# - 原题对 right/bottom 没有显式方向处理也能工作；变体下必须显式恢复 edge_pos
#   和 travel_dir，否则“碰撞前/后”无法判定。
#
# placement_legality_and_visual_cleanliness:
# - aligned companion 必须位于 inner marker 之后的同一 beat 序列上，且后面至少
#   还要有一个 beat，保证“碰撞后继续行进”真实发生。
# - bystander 只放在非 beat 的内部位置，避免和射击边/对边计数纠缠，也避免把
#   视觉重点从周期轨迹上分散出去。
# - 仍保留活跃线之间至少 1 行/列的空隔，使分段轨迹彼此清楚。
#
# color_inference:
# - 颜色变化完全由局部结构触发：同一条线中，aligned extra dot 的颜色接管后续 beat。
# - 不依赖颜色编号大小、排序或任何不可见属性；同一种颜色可在不同线里重复出现，
#   但“谁接管谁”只由碰撞位置决定，样例中可通过结构直接读出。


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
    Shoot periodic rays from one unique edge; aligned companion dots recolor the ray mid-flight.

    Input structure:
    - bg is the dominant color, with sparse non-bg dots on a subset of rows or columns.
    - Exactly one boundary has the most non-bg dots; that boundary is the shooting edge.
    - Every active shooting-direction line contains one repeated marker color with exactly 2 dots:
      one on the shooting edge and one one-step inward, so their gap defines the beat spacing.
    - Some active lines also contain one differently colored aligned companion farther along the same
      beat sequence; others instead contain one off-beat bystander dot.

    Rule:
    - Identify the shooting edge; left/right edges fire across rows, top/bottom edges fire down columns.
    - On each active line, recover the step from the 2 marker dots and enumerate beat positions in the
      order the ray encounters them from the shooting edge.
    - Without an aligned companion, every beat on that line is painted in the marker color.
    - With an aligned companion, beats before the collision stay marker-colored; the companion beat and
      all later beats switch to the companion color, while off-beat bystanders remain untouched.

    Key insight: the marker pair still sets a modular beat, but the aligned companion is no longer a
    stopping wall; it is a direction-sensitive color-change event, so the beat order from the edge matters.
    """
    g = np.array(grid, dtype=int)
    out = g.copy()
    bg = int(get_bg(g))
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
    edge = winners[0]

    by_row = edge in ("left", "right")
    num_lines = H if by_row else W
    line_len = W if by_row else H
    edge_pos, travel_dir = _edge_params(edge, line_len)

    def cell(line: int, pos: int) -> tuple[int, int]:
        return (line, pos) if by_row else (pos, line)

    for li in range(num_lines):
        colors: dict[int, list[int]] = defaultdict(list)
        for p in range(line_len):
            value = int(g[cell(li, p)])
            if value != bg:
                colors[value].append(p)

        if not colors:
            continue

        repeating = {color: positions for color, positions in colors.items() if len(positions) >= 2}
        assert len(repeating) == 1, (
            f"Line {li} ({'row' if by_row else 'col'}) must have exactly one repeating color: {dict(colors)}"
        )
        marker, mpos = next(iter(repeating.items()))
        assert len(mpos) == 2, (
            f"Line {li} ({'row' if by_row else 'col'}) marker {marker} "
            f"has {len(mpos)} dots (expected 2): {mpos}"
        )
        assert edge_pos in mpos, (
            f"Line {li} ({'row' if by_row else 'col'}) marker {marker} "
            f"must touch the shooting edge: {mpos}"
        )

        step = abs(mpos[1] - mpos[0])
        expected_inner = edge_pos + travel_dir * step
        assert expected_inner in mpos, (
            f"Line {li} ({'row' if by_row else 'col'}) marker {marker} "
            f"must have its second dot one step inward: {mpos}"
        )

        beats = _ordered_beats(edge, line_len, step)
        beat_to_index = {pos: idx for idx, pos in enumerate(beats)}
        aligned_hits: list[tuple[int, int, int]] = []
        for other_color, positions in colors.items():
            if other_color == marker:
                continue
            for pos in positions:
                beat_idx = beat_to_index.get(pos)
                if beat_idx is not None:
                    aligned_hits.append((beat_idx, pos, other_color))

        aligned_hits = [hit for hit in aligned_hits if hit[0] >= 2]
        assert len(aligned_hits) <= 1, (
            f"Line {li} ({'row' if by_row else 'col'}) has multiple aligned companions: {aligned_hits}"
        )

        collision_idx = None
        companion_color = None
        if aligned_hits:
            collision_idx, _collision_pos, companion_color = aligned_hits[0]

        for beat_idx, pos in enumerate(beats):
            if collision_idx is not None and beat_idx >= collision_idx:
                paint = int(companion_color)
                allowed = {bg, marker, int(companion_color)}
            else:
                paint = int(marker)
                allowed = {bg, marker}

            rc = cell(li, pos)
            cur = int(out[rc])
            if cur in allowed:
                out[rc] = paint

    return out


_ROWS_RANGE = (7, 12)
_COLS_RANGE = (9, 21)
_ACTIVE_LINES_RANGE = (4, 9)
_STEP_RANGE = (1, 5)
_COLORS = list(range(1, 10))
_EDGES = ("top", "bottom", "left", "right")

_P_MARKER_ONLY = 0.25
_P_COLLISION = 0.55

_MAX_RETRIES = 300


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


def _valid_steps(line_len: int, require_continuation: bool) -> list[int]:
    upper = min(_STEP_RANGE[1], line_len - 1)
    steps = []
    for step in range(_STEP_RANGE[0], upper + 1):
        beat_count = len(_ordered_beats("left", line_len, step))
        if require_continuation and beat_count < 4:
            continue
        steps.append(step)
    return steps


def _line_kind_sequence(n_active: int) -> list[str]:
    kinds = ["collision"]
    if n_active >= 2:
        kinds.append(random.choice(["marker_only", "bystander"]))
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
    edge: str, line_len: int, marker_color: int, kind: str
) -> tuple[list[tuple[int, int]], dict[str, int | str]] | None:
    require_continuation = kind == "collision"
    valid_steps = _valid_steps(line_len, require_continuation=require_continuation)
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
    collision_lines = 0

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

        shooting_edge_dots += 1
        if meta["kind"] == "collision":
            collision_lines += 1
            if int(meta["post_collision_beats"]) < 1:
                return None

    if collision_lines == 0:
        return None

    other_counts = _count_other_edge_dots(all_rc, edge, H, W)
    if any(count >= shooting_edge_dots for count in other_counts.values()):
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
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError(f"Failed to generate valid input after {_MAX_RETRIES} attempts")
