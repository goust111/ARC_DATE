# Combo variant of 16de56c4 (index 20)
# combo_spec: D6/3+D16/1
# Name: 碰撞变色+伴随点外延

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
import random
import sys

import numpy as np

try:
    from arc_utils import get_bg
except ModuleNotFoundError:
    sys.path.append(str(Path(__file__).resolve().parents[5] / "solvers"))
    from arc_utils import get_bg


# Design analysis for combo D6/3 + D16/1
# ---------------------------------------
# a) concretization_and_trace:
# - 每条活跃行/列仍由 2 个同色 marker dot 定义，且其中一个 marker 必须落在唯一
#   shooting edge 上；另一个 marker 恰好沿射线方向前进 1 个 step。
# - 融合前的原题把 aligned companion 当成“终点墙”：
#   例：left edge, line_len=13, marker 色 2 在位置 0 和 3，companion 色 6 在位置 9。
#   原题输出会把 beat 0,3,6,9 全刷成 6，并在 9 处结束，12 保持背景。
# - 融合后，companion 变成“中途变色站”而不是终点：
#   同一例子输出应为 0,3,6 保持/填成 2，9,12 改为 6。
# - D6/3 提供的是“碰撞后换色”这件事，D16/1 提供的是“碰撞后继续向终点边延伸”。
#   两者叠加后不能再用无方向 span 推理，而必须按从 shooting edge 出发的 beat 顺序
#   做分段着色。
#
# b) complexity_calibration:
# - 每条活跃线仍只允许 2 个 marker 加 0/1 个 extra dot，保持对象文法紧凑。
# - extra dot 只分两类：aligned companion 或 off-beat bystander，避免再引入更多对象。
# - 为了让融合行为在 2-4 个样例中可见，generator 强制至少一条 fusion line：
#   aligned companion 至少在第 3 个 beat，且其后至少保留 1 个 beat，这样样例里
#   会同时出现“碰撞前 marker 段”和“碰撞后延伸的 companion 段”。
#
# c) implicit_assumption_audit:
# - 原题 `lo, hi = min(all_ps), max(all_ps)` 默认 companion 只定义一个无方向闭区间；
#   融合后这个假设失效，因为 right/bottom 发射时“之前/之后”与数值大小相反。
# - 原题 `paint = companion` 会把 marker 到 companion 的整段都覆成 companion 色；
#   融合后碰撞前段必须保留 marker 色，只在 companion beat 起切换颜色。
# - 原题即使不恢复 travel_dir 也能工作；融合后若不显式构造有序 beat 序列，就无法区分
#   哪些位置属于碰撞前、哪些属于碰撞后。
# - 生成器因此必须保证 companion 真的是“沿射线方向、位于 inner marker 之后”的
#   aligned dot，而不是任意同模位置。
#
# d) placement_legality:
# - step 取值限制在 1..min(5, line_len-2)，保证第二个 marker 是内部点，不会直接落在
#   opposite edge 上。
# - fusion line 的 companion 只从 beats[2:-1] 中选，保证不与 marker 重叠，且之后仍有
#   至少一个 beat 可继续外延。
# - bystander 只从内部非 beat 位置采样，避免和周期轨迹重叠，也避免影响边界计数。
# - 活跃行/列索引保持至少 1 个空隔，因此不同射线在视觉上分开。
#
# e) visual_observability:
# - 人类应能从局部几何读出规则：先看到 edge+inner marker 定义 step，再看到同 beat 上
#   更远处的异色 dot 充当“滤色片”，之后的 beat 颜色改变且继续前进。
# - 仅有 marker-only 线会退化成原题的整线周期填充，因此每个样例至少保留一条 fusion line，
#   同时至少保留一条非 fusion line 作对照，方便观察“有 companion / 无 companion”的差异。
# - 进一步保证每题都能看到 marker-only 与 bystander 两种非 fusion 对照线，避免训练样例只
#   覆盖“延伸”而忽略“旁观者不参与”的分支。
#
# f) cleanliness:
# - 同一条线最多 3 个输入点，输出虽会形成周期轨迹，但轨迹之间互不重叠。
# - bystander 永远不在 beat 上，因此不会与填充结果混成同一种视觉信号。
# - companion 不落在终点 beat，避免“只是把末端点换色”这种不明显样例。
#
# g) inferable_change_and_color_logic:
# - 颜色变化完全由结构触发：谁在同一 beat 上、并且更靠射线前方，谁就接管后续颜色。
# - 不依赖颜色编号大小、排序或黑色=0 之外的不可视属性。
# - 任务 9 的 solver-generator 契约：
#   solver 假设每条非空线恰有一个 repeating marker 色、该色恰有 2 个点、其中一个在
#   shooting edge 上，aligned extra 至多一个且位于第 3 个 beat 或之后；generator 全部
#   显式保证这些条件。


def _edge_params(edge: str, line_len: int) -> tuple[int, int]:
    if edge in ("left", "top"):
        return 0, 1
    return line_len - 1, -1


def _ordered_beats(edge: str, line_len: int, step: int) -> list[int]:
    edge_pos, travel_dir = _edge_params(edge, line_len)
    stop = line_len if travel_dir == 1 else -1
    return list(range(edge_pos, stop, travel_dir * step))


def _paint_line(
    out: np.ndarray,
    bg: int,
    edge: str,
    line: int,
    line_len: int,
    marker_color: int,
    step: int,
    companion: tuple[int, int] | None,
) -> None:
    by_row = edge in ("left", "right")
    beats = _ordered_beats(edge, line_len, step)
    collision_pos = companion[0] if companion is not None else None
    collision_color = companion[1] if companion is not None else None
    collision_idx = beats.index(collision_pos) if collision_pos is not None else None

    def cell(pos: int) -> tuple[int, int]:
        return (line, pos) if by_row else (pos, line)

    for beat_idx, pos in enumerate(beats):
        if collision_idx is None or beat_idx < collision_idx:
            paint = int(marker_color)
            allowed = {int(bg), int(marker_color)}
        else:
            paint = int(collision_color)
            allowed = {int(bg), int(marker_color), int(collision_color)}

        rc = cell(pos)
        cur = int(out[rc])
        if cur in allowed:
            out[rc] = paint


def main(grid: np.ndarray) -> np.ndarray:
    """
    Shoot periodic rays from one unique edge; aligned companion dots recolor the ray and let it keep extending.

    Input structure:
    - The background is the dominant color, with sparse non-bg dots on several rows or columns.
    - Exactly one boundary has strictly more non-bg dots than the other three; that boundary is the shooting edge.
    - Each active shooting-direction line contains exactly one repeated marker color with 2 dots:
      one on the shooting edge and one step inward, so their gap fixes the periodic beat.
    - Some lines also contain one differently colored aligned companion farther along that same beat;
      other extras are off-beat bystanders and must remain unchanged.

    Rule:
    - Identify the shooting edge; left/right edges fire across rows, top/bottom edges fire down columns.
    - On each active line, recover the step from the marker pair and enumerate beat positions in travel order
      starting at the shooting edge.
    - Without an aligned companion, paint every beat on that line in the marker color.
    - With an aligned companion, beats before the collision keep the marker color, while the companion beat and
      every later beat toward the far edge switch to the companion color and continue filling outward.

    Key insight:
    - The marker pair still defines a modular rhythm, but the aligned foreign dot is no longer a wall. It is a
      direction-sensitive color-switch event, so solving requires ordered beat traversal rather than an undirected span.
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
        assert step > 0, f"Line {li} marker {marker} has zero step"
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
                    aligned_hits.append((beat_idx, pos, int(other_color)))

        aligned_hits = [hit for hit in aligned_hits if hit[0] >= 2]
        assert len(aligned_hits) <= 1, (
            f"Line {li} ({'row' if by_row else 'col'}) has multiple aligned companions: {aligned_hits}"
        )

        companion = None
        if aligned_hits:
            collision_idx, collision_pos, collision_color = aligned_hits[0]
            assert collision_idx < len(beats) - 1, (
                f"Line {li} ({'row' if by_row else 'col'}) companion must leave room to extend: "
                f"{aligned_hits[0]}"
            )
            companion = (int(collision_pos), int(collision_color))

        _paint_line(out, bg, edge, li, line_len, int(marker), int(step), companion)

    return out


_ROWS_RANGE = (7, 12)
_COLS_RANGE = (9, 21)
_ACTIVE_LINES_RANGE = (4, 9)
_STEP_RANGE = (1, 5)
_EDGES = ("top", "bottom", "left", "right")

_P_MARKER_ONLY = 0.25
_P_BYSTANDER = 0.25
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


def _valid_steps(line_len: int, require_fusion: bool) -> list[int]:
    upper = min(_STEP_RANGE[1], line_len - 2)
    if upper < _STEP_RANGE[0]:
        return []

    steps = []
    for step in range(_STEP_RANGE[0], upper + 1):
        beat_count = len(_ordered_beats("left", line_len, step))
        if require_fusion and beat_count < 4:
            continue
        steps.append(step)
    return steps


def _line_kind_sequence(n_active: int) -> list[str]:
    kinds = ["fusion"]
    if n_active >= 2:
        kinds.append("marker_only")
    if n_active >= 3:
        kinds.append("bystander")
    while len(kinds) < n_active:
        roll = random.random()
        if roll < _P_MARKER_ONLY:
            kinds.append("marker_only")
        elif roll < _P_MARKER_ONLY + _P_BYSTANDER:
            kinds.append("bystander")
        else:
            kinds.append("fusion")
    random.shuffle(kinds)
    return kinds


def _active_count_bounds(edge: str) -> tuple[int, int] | None:
    line_cap = _ROWS_RANGE[1] if edge in ("left", "right") else _COLS_RANGE[1]
    max_active = min(_ACTIVE_LINES_RANGE[1], (line_cap + 1) // 2)
    min_active = _ACTIVE_LINES_RANGE[0]
    if min_active > max_active:
        return None
    return min_active, max_active


def _build_line(
    edge: str,
    line_len: int,
    marker_color: int,
    palette: list[int],
    kind: str,
) -> tuple[list[tuple[int, int]], dict[str, object]] | None:
    valid_steps = _valid_steps(line_len, require_fusion=(kind == "fusion"))
    if not valid_steps:
        return None

    if kind == "bystander":
        valid_steps = [step for step in valid_steps if step >= 2]
        if not valid_steps:
            kind = "marker_only"
            valid_steps = _valid_steps(line_len, require_fusion=False)
            if not valid_steps:
                return None

    step = random.choice(valid_steps)
    edge_pos, travel_dir = _edge_params(edge, line_len)
    inner_pos = edge_pos + travel_dir * step
    beats = _ordered_beats(edge, line_len, step)
    beat_set = set(beats)
    other_colors = [color for color in palette if color != marker_color]

    dots = [(edge_pos, marker_color), (inner_pos, marker_color)]
    meta: dict[str, object] = {
        "kind": kind,
        "marker_color": int(marker_color),
        "step": int(step),
        "companion": None,
        "post_collision_beats": 0,
    }

    if kind == "fusion":
        if len(beats) < 4 or not other_colors:
            return None
        companion_pos = random.choice(beats[2:-1])
        companion_color = random.choice(other_colors)
        dots.append((companion_pos, companion_color))
        meta["companion"] = (int(companion_pos), int(companion_color))
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


def _original_style_output(grid: np.ndarray) -> np.ndarray:
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
    edge = max(edge_counts, key=edge_counts.get)
    by_row = edge in ("left", "right")
    num_lines = H if by_row else W
    line_len = W if by_row else H

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
        if len(repeating) != 1:
            continue
        marker, mpos = next(iter(repeating.items()))
        a, b = sorted(mpos)
        step = b - a
        offset = a % step

        companion = None
        companion_positions: list[int] = []
        for other_color, positions in colors.items():
            if other_color == marker:
                continue
            aligned = [pos for pos in positions if pos % step == offset]
            if aligned:
                companion = int(other_color)
                companion_positions = aligned
                break

        if companion is None:
            paint = int(marker)
            lo, hi = 0, line_len - 1
        else:
            paint = companion
            all_positions = list(mpos) + companion_positions
            lo, hi = min(all_positions), max(all_positions)

        for pos in range(lo, hi + 1):
            if pos % step != offset:
                continue
            rc = cell(li, pos)
            cur = int(out[rc])
            if cur == bg or cur == marker:
                out[rc] = paint

    return out


def _gen_attempt() -> np.ndarray | None:
    feasible_edges = [edge for edge in _EDGES if _active_count_bounds(edge) is not None]
    if not feasible_edges:
        return None

    edge = random.choice(feasible_edges)
    active_bounds = _active_count_bounds(edge)
    if active_bounds is None:
        return None
    n_active = random.randint(*active_bounds)
    by_row = edge in ("left", "right")

    dims = _choose_dims(edge, n_active)
    if dims is None:
        return None
    H, W, num_lines, line_len = dims

    active_lines = _spaced_indices(n_active, num_lines)
    if not active_lines:
        return None

    palette_size = random.randint(2, 8)
    palette = random.sample(list(range(1, 10)), palette_size)
    line_kinds = _line_kind_sequence(n_active)

    def cell(line: int, pos: int) -> tuple[int, int]:
        return (line, pos) if by_row else (pos, line)

    inp = np.zeros((H, W), dtype=int)
    expected = np.zeros((H, W), dtype=int)
    all_rc: list[tuple[int, int]] = []
    shooting_edge_dots = 0
    fusion_lines = 0
    non_fusion_lines = 0
    lines_with_tail = 0

    for li, kind in zip(active_lines, line_kinds):
        marker_color = random.choice(palette)
        built = _build_line(edge, line_len, marker_color, palette, kind)
        if built is None:
            return None
        dots, meta = built

        for pos, color in dots:
            r, c = cell(li, pos)
            if inp[r, c] != 0:
                return None
            inp[r, c] = color
            expected[r, c] = color
            all_rc.append((r, c))

        _paint_line(expected, 0, edge, li, line_len, int(meta["marker_color"]), int(meta["step"]), meta["companion"])

        shooting_edge_dots += 1
        if meta["kind"] == "fusion":
            fusion_lines += 1
            if int(meta["post_collision_beats"]) < 1:
                return None
            lines_with_tail += 1
        else:
            non_fusion_lines += 1

    if fusion_lines == 0 or non_fusion_lines == 0 or lines_with_tail == 0:
        return None

    other_counts = _count_other_edge_dots(all_rc, edge, H, W)
    if any(count >= shooting_edge_dots for count in other_counts.values()):
        return None

    try:
        out = main(inp)
    except Exception:
        return None

    if not np.array_equal(out, expected):
        return None
    if np.array_equal(inp, out):
        return None
    if np.array_equal(out, _original_style_output(inp)):
        return None

    return inp


def generate_input() -> np.ndarray:
    for _ in range(_MAX_RETRIES):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError(f"Failed to generate valid input after {_MAX_RETRIES} attempts")
