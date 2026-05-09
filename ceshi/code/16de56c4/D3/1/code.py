# Variant of 16de56c4
# Dimension: D3 — 轨迹填充模式 (trajectory)
# Change: 等间距连续填充：在beat位置(pos % step == offset)全部填充相同颜色... → 周期性图案重复

from __future__ import annotations

from collections import defaultdict
import random

import numpy as np

from arc_utils import get_bg


# Design analysis
# spec:
#   original_invariant_kept:
#     - 仍然先靠四条边上的非背景点数量识别唯一射击边。
#     - left/right 边对应按行发射，top/bottom 边对应按列发射。
#     - 每条活跃线仍然有且只有一种 marker 颜色出现两次，并且其中一个点落在射击边。
#     - 两个 marker 的间距继续定义 step。
#   d3_change:
#     - 原题在 step 的同余类上做“单色 beat 填充”。
#     - 现在把“从边界 marker 到第二个 marker 之前的 step 个格子”视为一个可见样本单元。
#     - 样本单元里保留 marker 色，并额外放入一个 companion 色，其余槽位是背景。
#     - 输出时从射击边开始，将这个长度为 step 的单元整段周期性平铺到整条射线。
#   concrete_trace:
#     - 例：left edge, step=4，输入某行前 5 格为 [M, C, 0, 0, M]，其中最后一个 M 是第二个 marker。
#     - 原逻辑会把位置 0,4,8,... 都刷成同一种颜色。
#     - 新逻辑把前 4 格样本 [M, C, 0, 0] 当作 motif，输出整行为 [M, C, 0, 0, M, C, 0, 0, ...]。
#     - 若远处某个非 motif 槽位上有 bystander，且该相位在 motif 中是背景，则它保持不变。
#   complexity_calibration:
#     - 为了避免退化回原题，generator 强制每条活跃线都有一个单元内 companion。
#     - step 收紧到 3-5，这样 motif 至少含有 marker、companion 和一个背景槽位，视觉上能明确看出“短周期图案”而不是单色等距点列。
#     - bystander 仍然保留，但只允许落在 motif 的背景相位，避免与平铺图案冲突。
#   implicit_assumptions_in_main:
#     - 每条活跃线恰好有一个颜色满足“出现两次且其中一个点在射击边”；generator 保证这一点。
#     - 第二个 marker 是下一次重复的起点，因此样本单元取 dist=0..step-1，而 dist=step 必须再次出现 marker。
#     - 第一周期内部除 marker/companion 外没有 bystander；否则 solver 会误把它当 motif 一部分。generator 明确禁止。
#     - 非第一周期出现的额外非背景点只能放在 motif 的背景相位；solver 在背景相位不写值，因此它们会保留。
#   placement_legality:
#     - companion 只放在 1..step-1 的内部相位，并额外要求它在输出中至少重复两次，保证“pattern repeat”真实发生。
#     - 活跃线索引继续留出至少 1 格间隔，减少视觉串扰。
#     - 仍然检查非射击边上的边界点数量，确保射击边唯一可识别。
#   observability_and_cleanliness:
#     - 解题者能直接看到“两枚同色 marker 给出长度”，再从其间样本读出 motif。
#     - 输出中每条活跃线只出现一种简短重复单元，不叠加多套规则；不同线彼此平行且稀疏，整体不杂乱。
#   color_inference:
#     - 颜色不依赖 id 排序或数值运算，只依赖在第一周期中的可见位置角色。
#     - companion 色既在输入样本中出现，也会在输出中沿线重复，因此颜色变化有直接结构线索。


def main(grid: np.ndarray) -> np.ndarray:
    """
    Repeat a short sampled trajectory pattern that shoots from a unique grid edge.

    Input structure:
    - bg = most common color. Sparse non-bg dots lie on several shooting-direction lines.
    - Exactly one border edge has strictly the most non-bg dots; that edge is the shooter.
    - On each active line, one marker color appears exactly twice: once on the shooting edge
      and once one period inward. Their gap gives the step.
    - Any additional dot inside that first period is part of the sample motif; extra dots
      outside the first period are bystanders.

    Rule:
    - Determine the shooting edge; left/right propagate along rows, top/bottom along columns.
    - For each active line, read the first period from the shooting edge up to but not
      including the second marker. This length-step slice is the motif.
    - Tile that motif periodically across the full line, anchored at the shooting edge.
    - Motif background slots stay untouched, so bystanders placed on those slots remain as-is.

    Key insight: the two same-colored marker dots do not merely define a beat spacing anymore;
    together they frame one complete repeat unit, and the line is solved by copying that unit
    outward from the shooting edge.
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
    winners = [e for e, cnt in edge_counts.items() if cnt == max_count]
    assert len(winners) == 1, f"Ambiguous shooting edge: {winners} (counts={edge_counts})"
    edge = winners[0]

    by_row = edge in ("left", "right")
    num_lines = H if by_row else W
    line_len = W if by_row else H
    edge_pos = 0 if edge in ("left", "top") else line_len - 1

    def cell(line: int, pos: int) -> tuple[int, int]:
        return (line, pos) if by_row else (pos, line)

    def pos_from_dist(dist: int) -> int:
        return dist if edge_pos == 0 else line_len - 1 - dist

    for li in range(num_lines):
        colors: dict[int, list[int]] = defaultdict(list)
        for p in range(line_len):
            v = int(g[cell(li, p)])
            if v != bg:
                colors[v].append(p)

        if not colors:
            continue

        markers = [(color, sorted(ps)) for color, ps in colors.items() if len(ps) == 2 and edge_pos in ps]
        assert len(markers) == 1, (
            f"Line {li} ({'row' if by_row else 'col'}) expected one edge marker, got {markers} "
            f"from {dict(colors)}"
        )
        marker, mpos = markers[0]
        inner_pos = next(p for p in mpos if p != edge_pos)
        step = abs(inner_pos - edge_pos)
        assert step >= 1, f"Line {li} has zero step"

        motif = [int(g[cell(li, pos_from_dist(dist))]) for dist in range(step)]
        assert motif[0] == marker, (
            f"Line {li} motif must start with marker {marker}, got {motif[0]} (motif={motif})"
        )
        assert int(g[cell(li, pos_from_dist(step))]) == marker, (
            f"Line {li} missing repeated marker at distance {step}"
        )

        for dist in range(line_len):
            color = int(motif[dist % step])
            if color == bg:
                continue
            out[cell(li, pos_from_dist(dist))] = color

    return out


# --------------- generator ---------------

_ROWS_RANGE = (7, 12)
_COLS_RANGE = (9, 21)
_ACTIVE_LINES_RANGE = (4, 9)
_STEP_RANGE = (3, 5)
_COLORS = list(range(1, 10))
_EDGES = ("top", "bottom", "left", "right")

_P_BYSTANDER = 0.45
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
    counts = {}
    for other in _EDGES:
        if other == edge:
            continue
        check = boundary[other]
        counts[other] = sum(1 for r, c in dots_rc if check(r, c))
    return counts


def _build_line_dots(
    edge: str, line_len: int, marker_color: int, companion_color: int, step: int
) -> tuple[list[tuple[int, int]], list[int]]:
    edge_at_zero = edge in ("left", "top")

    def pos_from_dist(dist: int) -> int:
        return dist if edge_at_zero else line_len - 1 - dist

    motif = [0] * step
    motif[0] = marker_color

    repeatable_phases = [phase for phase in range(1, step) if phase + step < line_len]
    if not repeatable_phases:
        return [], []
    companion_phase = random.choice(repeatable_phases)
    motif[companion_phase] = companion_color

    dots = [
        (pos_from_dist(0), marker_color),
        (pos_from_dist(step), marker_color),
        (pos_from_dist(companion_phase), companion_color),
    ]

    if random.random() < _P_BYSTANDER:
        bystander_candidates = [
            dist
            for dist in range(step + 1, line_len)
            if motif[dist % step] == 0
        ]
        other_colors = [c for c in _COLORS if c not in (marker_color, companion_color)]
        if bystander_candidates and other_colors:
            dist = random.choice(bystander_candidates)
            dots.append((pos_from_dist(dist), random.choice(other_colors)))

    return dots, motif


def _gen_attempt() -> np.ndarray | None:
    n_active = random.randint(*_ACTIVE_LINES_RANGE)
    edge = random.choice(_EDGES)
    by_row = edge in ("left", "right")

    dims = _choose_dims(edge, n_active)
    if dims is None:
        return None
    H, W, num_lines, line_len = dims

    max_step = min(_STEP_RANGE[1], line_len - 2)
    if max_step < _STEP_RANGE[0]:
        return None

    active_lines = _spaced_indices(n_active, num_lines)
    if not active_lines:
        return None

    def cell(line: int, pos: int) -> tuple[int, int]:
        return (line, pos) if by_row else (pos, line)

    inp = np.zeros((H, W), dtype=int)
    all_rc: list[tuple[int, int]] = []
    shooting_edge_dots = 0
    total_companion_repeats = 0
    bystander_lines = 0

    for li in active_lines:
        marker_color, companion_color = random.sample(_COLORS, 2)
        step = random.randint(_STEP_RANGE[0], max_step)

        dots, motif = _build_line_dots(edge, line_len, marker_color, companion_color, step)
        if not dots:
            return None

        for pos, color in dots:
            r, c = cell(li, pos)
            if inp[r, c] != 0:
                return None
            inp[r, c] = color
            all_rc.append((r, c))

        shooting_edge_dots += 1
        total_companion_repeats += sum(
            1 for dist in range(line_len) if motif[dist % step] == companion_color
        )
        bystander_lines += int(len(dots) == 4)

    other_counts = _count_other_edge_dots(all_rc, edge, H, W)
    if any(cnt >= shooting_edge_dots for cnt in other_counts.values()):
        return None

    try:
        out = main(inp)
    except Exception:
        return None

    if np.array_equal(inp, out):
        return None

    # Quantitative anti-degeneration checks:
    # every line contributes a repeated companion phase, so overall output must
    # contain more companion copies than active lines.
    if total_companion_repeats <= n_active:
        return None
    if bystander_lines == 0:
        return None

    return inp


def generate_input() -> np.ndarray:
    for _ in range(_MAX_RETRIES):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError(f"Failed to generate valid input after {_MAX_RETRIES} attempts")
