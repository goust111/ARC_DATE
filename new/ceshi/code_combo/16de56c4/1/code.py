# Combo variant of 16de56c4 (index 1)
# combo_spec: D1/0+D3/0
# Name: GCD步长+交替颜色填充

from __future__ import annotations

from collections import defaultdict
from math import gcd
import random

import numpy as np

from arc_utils import get_bg


# Design analysis
# ---------------
# a) 具体化与行为追踪:
#    - 原题每条活跃射线只有 2 个同色 marker，因此步长就是那两个点的差；
#      如果有同 beat 的 companion，则整个受限区间都会被 companion 颜色单色覆盖。
#    - 现在每条活跃线都有 3-5 个同色 marker，而且这些 marker 虽然都落在同一
#      个 beat 格子上，但相邻可见间距不必相等。真实步长要取所有相邻 gap 的 GCD。
#    - 融合后的 companion 线不再是单色覆盖，而是在 beat 位置按
#      marker/companion/marker/companion... 交替着色，且相位从 shooting edge
#      上的 marker 开始计数。
#    - 具体例子（left-shooting row）:
#        输入 marker A 位于列 [0, 4, 6]，gap 为 [4, 2]，所以 step = gcd(4, 2) = 2。
#        若 companion B 位于列 10，则原题会把 0,2,4,6,8,10 全部刷成 B；
#        本融合变体会把这些 beat 列刷成 A,B,A,B,A,B。
#      差异同时体现在“需要从多点抽 GCD”与“companion 线呈现 ABAB 条纹”。
#
# b) 复杂度校准:
#    - 所有活跃线都强制使用非均匀 marker 间隔，因此不会退化成“只看前两个点差值”。
#    - 仍保持每条线只有一个重复颜色，并且最多只有一个额外单点（companion 或
#      bystander），避免引入多重匹配歧义。
#    - 至少两条线是 companion 线，并且 companion 放在奇数 beat 上，这样输出中
#      会出现肉眼可见的 ABAB 带状结构，而不是只在局部产生一次颜色翻转。
#
# c) 隐式假设审查:
#    - 原代码假设 `len(mpos) == 2`，融合后必须改成 `len(mpos) >= 3`，并验证每个
#      marker 位置共享同一个模类。
#    - 原代码的 `step = b - a` 失效，必须改成对相邻 marker gap 求 GCD。
#    - 原代码 companion 线只会把 `bg` 或 `marker` 改写成同一颜色；现在交替填充时
#      companion 自己所在的 beat 也可能需要被写回，因此 companion 线的覆盖白名单
#      必须包含 `{bg, marker, companion}`。
#    - 交替相位不能只依赖排序后的第一个点；right/bottom 发射时，真正的 beat 0
#      是边界上的 marker，所以必须显式找出 edge_anchor。
#    - 由于 generator 保证每条线至多一个 aligned 非 marker 颜色，`companion`
#      选择仍然无歧义；若出现多个 aligned 颜色，solver 会 assert。
#
# d) 放置合法性:
#    - generator 先选 shooting edge，再在互不相邻的活跃线索引上放置对象，因此不
#      会有射线之间的像素重叠。
#    - marker 位置通过 beat index 构造，保证都在边界内且互不重复；companion 只会
#      放到 outermost marker 更远处的同 beat 位置；bystander 只从 off-beat 位置采样。
#    - 额外通过“winning edge dot count 严格最大”校验，防止其他边界与 shooting edge
#      打平。
#
# e) 视觉可观察性:
#    - GCD 变化是可见的：每条活跃线的第一个可见 gap 都大于真实 step，并且至少还有
#      一个更小的同模 gap，所以输出会在一些 marker 中间补出新的 beat 点。
#    - 交替颜色也是可见的：companion 线至少延伸到第 3 个奇数 beat，使输出中出现
#      至少 4 个 beat 的 A/B/A/B 序列；这比单个 companion 端点更容易从 2-4 个样例
#      归纳出来。
#
# f) 可视化整洁度:
#    - 每条线最多两种角色颜色（marker 与额外点颜色），整体 palette 也限制在 3-6
#      种颜色内，避免大面积杂色。
#    - bystander 只出现在 off-beat 位置，不会和填充条纹重合；companion 线的额外点
#      落在条纹终点，结构清晰。
#
# g) 变化的可推断性:
#    - 颜色变化完全由结构决定，而不是颜色编号大小等人类不可见信息：
#      1. 重复色 = marker。
#      2. 与 marker 同 beat 的单点 = companion，其颜色参与 ABAB 交替。
#      3. 不同 beat 的单点 = bystander，保持不变。
#    - generator 强制至少两条“非均匀 GCD + companion 条纹”融合线，确保样例中有
#      足够重复证据支持这两个条件的联合推断。
#
# Post-change solver assumptions reviewed against the generator:
#    - 每条活跃线恰有一个重复颜色，且 marker 数量在 3-5: guaranteed.
#    - 每条活跃线的 marker gap GCD 为 1-5 之间的正整数，且可由可见 gap 恢复: guaranteed.
#    - aligned 的非 marker 颜色最多一个；若存在，则位于 marker 最远端之后: guaranteed.
#    - 至少两条 companion 线同时也是 GCD-critical 线，因此融合行为不是表面拼接: guaranteed.


def _gcd_of_gaps(positions: list[int]) -> int:
    gaps = [b - a for a, b in zip(positions, positions[1:])]
    assert gaps, f"Need at least two positions to define gaps: {positions}"
    step = gaps[0]
    for gap in gaps[1:]:
        step = gcd(step, gap)
    return step


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


def _iter_line_colors(g: np.ndarray, bg: int, line: int, line_len: int, cell):
    colors: dict[int, list[int]] = defaultdict(list)
    for p in range(line_len):
        value = int(g[cell(line, p)])
        if value != bg:
            colors[value].append(p)
    return colors


def _paint_for_beat(marker: int, companion: int, edge_anchor: int, step: int, pos: int) -> int:
    beat_idx = abs(pos - edge_anchor) // step
    return marker if beat_idx % 2 == 0 else companion


def main(grid: np.ndarray) -> np.ndarray:
    """
    Shoot periodic rays from one dominant edge, infer the beat by GCD, and stripe companion spans.

    Input structure:
    - The background is the most common color and the grid contains sparse non-background dots.
    - Exactly one boundary has strictly more non-background dots than the other three; that side is
      the shooting edge and determines whether rays run across rows or across columns.
    - Every active shooting-direction line contains exactly one repeated marker color with 3 or more
      dots. Those marker dots lie on one hidden beat lattice, but their visible gaps may differ.
    - A line may also contain one singleton dot of another color: if it shares the marker beat it is
      a companion, otherwise it is an off-beat bystander.

    Rule:
    - Find the unique shooting edge, then examine each row/column in that firing direction.
    - On each active line, sort the repeated marker positions and compute `step` as the GCD of all
      adjacent marker gaps. The marker that lies on the shooting edge is beat index 0 and anchors
      the alternation phase.
    - Without a companion, fill every beat position on the whole line with the marker color.
    - With a same-beat companion, fill only the span covered by the marker set and the companion,
      but color successive beat positions alternately as marker, companion, marker, companion, ...
      starting from the shooting-edge marker.
    - Off-beat bystanders are preserved.

    Key insight: the solver must first recover a hidden lattice from several unequal marker gaps,
    then use the boundary-side marker to orient an ABAB stripe pattern. The companion no longer
    turns the whole ray into one solid color; it both bounds the ray and supplies the alternating
    second color.
    """
    g = np.array(grid, dtype=int)
    out = g.copy()
    bg = int(get_bg(g))
    edge = _shooting_edge(g, bg)
    by_row, num_lines, line_len, cell = _line_axes(edge, g.shape)

    for li in range(num_lines):
        colors = _iter_line_colors(g, bg, li, line_len, cell)
        if not colors:
            continue

        repeating = {color: sorted(posns) for color, posns in colors.items() if len(posns) >= 2}
        assert len(repeating) == 1, (
            f"Line {li} ({'row' if by_row else 'col'}) has repeating colors "
            f"{list(repeating)}; expected exactly one"
        )
        marker, mpos = next(iter(repeating.items()))
        assert len(mpos) >= 3, (
            f"Line {li} ({'row' if by_row else 'col'}) marker {marker} "
            f"has {len(mpos)} dots (expected >= 3): {mpos}"
        )

        edge_anchor = min(mpos) if edge in ("left", "top") else max(mpos)
        assert edge_anchor in (0, line_len - 1), (
            f"Line {li} ({'row' if by_row else 'col'}) lacks a marker on the shooting edge: {mpos}"
        )

        step = _gcd_of_gaps(mpos)
        assert step > 0, f"Non-positive step on line {li}: {mpos}"
        offset = edge_anchor % step
        assert all(p % step == offset for p in mpos), (
            f"Marker positions on line {li} do not share one beat class: {mpos}"
        )

        aligned_others = [
            other_color
            for other_color, other_positions in colors.items()
            if other_color != marker and any(p % step == offset for p in other_positions)
        ]
        assert len(aligned_others) <= 1, (
            f"Line {li} ({'row' if by_row else 'col'}) has multiple companions: {aligned_others}"
        )
        companion = aligned_others[0] if aligned_others else None

        if companion is None:
            overwrite = {bg, marker}
            lo, hi = 0, line_len - 1
            for p in range(lo, hi + 1):
                if p % step != offset:
                    continue
                rc = cell(li, p)
                if int(out[rc]) in overwrite:
                    out[rc] = marker
            continue

        all_positions = mpos + colors[companion]
        lo, hi = min(all_positions), max(all_positions)
        overwrite = {bg, marker, companion}
        for p in range(lo, hi + 1):
            if p % step != offset:
                continue
            rc = cell(li, p)
            if int(out[rc]) not in overwrite:
                continue
            out[rc] = _paint_for_beat(marker, companion, edge_anchor, step, p)

    return out


# --------------- generator (gcd-step + alternating-fill combo) ---------------

_ROWS_RANGE = (7, 12)
_COLS_RANGE = (9, 21)
_ACTIVE_LINES_RANGE = (4, 9)
_STEP_RANGE = (1, 5)
_MARKER_COUNT_RANGE = (3, 5)
_ALL_COLORS = list(range(1, 10))
_EDGES = ("top", "bottom", "left", "right")

_P_MARKER_ONLY = 0.18
_P_COMPANION = 0.62

_MIN_COMPANION_LINES = 2
_MIN_FUSED_LINES = 2
_PALETTE_SIZE_RANGE = (3, 6)
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


def _choose_line_types(n_active: int) -> list[str]:
    assert n_active >= 4, f"Need at least 4 active lines for branch coverage, got {n_active}"

    # Expose all solver branches in every sample so humans can observe:
    # - unlimited marker-only fill,
    # - bounded ABAB companion fill,
    # - off-beat bystander preservation.
    kinds = ["companion"] * _MIN_COMPANION_LINES + ["marker_only", "bystander"]
    for _ in range(n_active - len(kinds)):
        roll = random.random()
        if roll < _P_MARKER_ONLY:
            kinds.append("marker_only")
        elif roll < _P_MARKER_ONLY + _P_COMPANION:
            kinds.append("companion")
        else:
            kinds.append("bystander")
    random.shuffle(kinds)
    return kinds


def _build_marker_indices(max_beat: int) -> list[int] | None:
    feasible_counts = [
        count
        for count in range(_MARKER_COUNT_RANGE[0], _MARKER_COUNT_RANGE[1] + 1)
        if max_beat >= count
    ]
    if not feasible_counts:
        return None

    count = random.choice(feasible_counts)
    gap_count = count - 1

    increments = [2] + [1] * (gap_count - 1)
    capacities = [1] + [2] * (gap_count - 1)
    protected = random.randrange(1, gap_count)
    capacities[protected] = 0

    spare = min(max_beat - sum(increments), sum(capacities))
    extra = random.randint(0, spare)
    while extra > 0:
        choices = [i for i, cap in enumerate(capacities) if cap > 0]
        idx = random.choice(choices)
        increments[idx] += 1
        capacities[idx] -= 1
        extra -= 1

    indices = [0]
    for inc in increments:
        indices.append(indices[-1] + inc)
    return indices


def _positions_from_indices(edge: str, line_len: int, step: int, beat_indices: list[int]) -> list[int]:
    if edge in ("left", "top"):
        return [idx * step for idx in beat_indices]
    edge_pos = line_len - 1
    return [edge_pos - idx * step for idx in beat_indices]


def _valid_steps(line_len: int, kind: str) -> list[int]:
    upper = min(_STEP_RANGE[1], line_len - 1)
    valid = []
    for step in range(_STEP_RANGE[0], upper + 1):
        max_beat = (line_len - 1) // step
        marker_indices = _build_marker_indices(max_beat)
        if marker_indices is None:
            continue
        if kind == "companion":
            later_odd_beats = [idx for idx in range(marker_indices[-1] + 1, max_beat + 1) if idx % 2 == 1]
            if not later_odd_beats:
                continue
        if kind == "bystander" and step == 1:
            continue
        valid.append(step)
    return valid


def _build_line_dots(
    edge: str,
    line_len: int,
    marker_color: int,
    palette: list[int],
    step: int,
    kind: str,
):
    max_beat = (line_len - 1) // step
    for _ in range(30):
        marker_indices = _build_marker_indices(max_beat)
        if marker_indices is None:
            return None

        marker_positions = _positions_from_indices(edge, line_len, step, marker_indices)
        dots = [(pos, marker_color) for pos in marker_positions]
        meta = {
            "kind": kind,
            "step": step,
            "marker_positions": marker_positions,
            "companion_index": None,
        }

        other_colors = [color for color in palette if color != marker_color]
        if not other_colors:
            other_colors = [color for color in _ALL_COLORS if color != marker_color]
        if not other_colors:
            return None

        if kind == "companion":
            later_odd_beats = [idx for idx in range(marker_indices[-1] + 1, max_beat + 1) if idx % 2 == 1]
            if not later_odd_beats:
                continue
            beat_idx = random.choice(later_odd_beats)
            companion_pos = _positions_from_indices(edge, line_len, step, [beat_idx])[0]
            dots.append((companion_pos, random.choice(other_colors)))
            meta["companion_index"] = beat_idx
            return dots, meta

        if kind == "bystander":
            offset = marker_positions[0] % step
            non_beat = [
                pos
                for pos in range(line_len)
                if pos % step != offset and pos not in marker_positions
            ]
            if not non_beat:
                continue
            dots.append((random.choice(non_beat), random.choice(other_colors)))
            return dots, meta

        return dots, meta

    return None


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
    counts = {}
    for other_edge in _EDGES:
        if other_edge == edge:
            continue
        check = boundary[other_edge]
        counts[other_edge] = sum(1 for r, c in dots_rc if check(r, c))
    return counts


def _line_metrics(values: list[int]) -> dict[str, int | bool]:
    colors: dict[int, list[int]] = defaultdict(list)
    for pos, value in enumerate(values):
        if value != 0:
            colors[value].append(pos)

    repeating = {color: posns for color, posns in colors.items() if len(posns) >= 2}
    assert len(repeating) == 1
    marker_positions = sorted(next(iter(repeating.values())))
    gaps = [b - a for a, b in zip(marker_positions, marker_positions[1:])]
    step = _gcd_of_gaps(marker_positions)
    offset = marker_positions[0] % step
    aligned_others = [
        color
        for color, posns in colors.items()
        if len(posns) == 1 and posns[0] % step == offset
    ]
    return {
        "step": step,
        "gcd_critical": len(set(gaps)) > 1,
        "has_companion": bool(aligned_others),
    }


def _fusion_score(meta: list[dict]) -> tuple[int, int]:
    fused_lines = 0
    beat_mass = 0
    for item in meta:
        gaps = [b - a for a, b in zip(item["marker_positions"], item["marker_positions"][1:])]
        gcd_critical = len(set(gaps)) > 1
        if item["kind"] == "companion" and gcd_critical:
            fused_lines += 1
            beat_mass += int(item["companion_index"]) + 1
    return fused_lines, beat_mass


def _gen_attempt() -> np.ndarray | None:
    n_active = random.randint(*_ACTIVE_LINES_RANGE)
    edge = random.choice(_EDGES)
    dims = _choose_dims(edge, n_active)
    if dims is None:
        return None
    H, W, num_lines, line_len = dims

    active_lines = _spaced_indices(n_active, num_lines)
    if not active_lines:
        return None

    palette_size = random.randint(*_PALETTE_SIZE_RANGE)
    palette = random.sample(_ALL_COLORS, palette_size)
    line_types = _choose_line_types(n_active)

    by_row = edge in ("left", "right")

    def cell(line: int, pos: int) -> tuple[int, int]:
        return (line, pos) if by_row else (pos, line)

    inp = np.zeros((H, W), dtype=int)
    all_rc: list[tuple[int, int]] = []
    meta: list[dict] = []
    shooting_edge_dots = 0

    for li, kind in zip(active_lines, line_types):
        valid_steps = _valid_steps(line_len, kind)
        if not valid_steps:
            return None

        marker_color = random.choice(palette)
        step = random.choice(valid_steps)
        built = _build_line_dots(edge, line_len, marker_color, palette, step, kind)
        if built is None:
            return None
        dots, line_meta = built
        meta.append(line_meta)

        seen_positions = set()
        for pos, color in dots:
            if pos in seen_positions:
                return None
            seen_positions.add(pos)
            r, c = cell(li, pos)
            if inp[r, c] != 0:
                return None
            inp[r, c] = color
            all_rc.append((r, c))

        shooting_edge_dots += 1

    other_counts = _count_other_edge_dots(all_rc, edge, H, W)
    if any(count >= shooting_edge_dots for count in other_counts.values()):
        return None

    fused_lines, beat_mass = _fusion_score(meta)
    if fused_lines < _MIN_FUSED_LINES:
        return None
    if beat_mass < 10:
        return None

    metrics = []
    for li in active_lines:
        values = [int(inp[cell(li, p)]) for p in range(line_len)]
        metrics.append(_line_metrics(values))

    if not all(item["gcd_critical"] for item in metrics):
        return None
    if sum(1 for item in metrics if item["has_companion"]) < _MIN_COMPANION_LINES:
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
