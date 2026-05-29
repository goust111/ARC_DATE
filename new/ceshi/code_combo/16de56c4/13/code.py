# Combo variant of 16de56c4 (index 13)
# combo_spec: D2/3+D6/3
# Name: 双向射击+碰撞变色

from __future__ import annotations

from collections import defaultdict
from functools import lru_cache
import random

import numpy as np

from arc_utils import get_bg


# Design analysis for the combo variant
# a) 具体化与行为追踪:
#    原题每条活跃线只有一个射击边，因此只需恢复一组 marker pair，再决定 companion
#    是“截断并改色”还是不存在。这里改成同一轴的两侧同时开火，每条活跃行/列都带有
#    两组 marker pair。小例子（按行，长度 16）：
#    - 左源: 0:A, 2:A, 4:C
#    - 右源: 12:B, 15:B, 9:D
#    左射线 beat 为 0,2,4,6,8,10,12,14；在 4 处碰撞后由 A 改成 C。
#    右射线 beat 为 15,12,9,6,3,0；在 9 处碰撞后由 B 改成 D。
#    原题会只认一个边，不会出现对射；D2 单变体会有对射但颜色不分段；D6 单变体会
#    有分段改色但只有单向。融合后位置 6 同时被两边 claim，且此时左右都已经变色，
#    因而要在“谁先到/是否同时到”的几何规则上，再叠加“该源当前颜色是否已切换”。
# b) 复杂度校准:
#    如果两侧 step 完全相同，则某一伴随点会同时落在两侧 beat 上，solver 无法仅靠
#    结构判断它属于哪一边。所以 generator 不做完全对称，而是只选那些：
#    - 存在至少一个共享 beat 作为真正的相遇区；
#    - 左 companion 只对齐左源，右 companion 只对齐右源；
#    - 且相遇区位于两个 companion 之后，能看见“两边先变色再汇聚”的效果。
#    同时每条线最多只加两个 companion 或一个 bystander，避免对象类型暴涨。
# c) 隐式假设审查:
#    - 原题 `winners` 必须唯一；融合后单边不再唯一，改成 left+right vs top+bottom
#      两个边对的计数比较。
#    - 原题默认一条线只有一个 repeating color；这里每条非空线必须正好有两组、
#      且分别贴住相对两端边界。
#    - 原题把 companion 解释成无方向的区间终点；现在必须显式枚举“从边界向内”的
#      beat 顺序，因为 companion 只会影响之后的 beat。
#    - 原题可边扫边写；这里不行，因为两边会抢同一格，必须先汇总 claim，再按
#      travel hops 和固定 tie priority 统一裁决。
# d) 放置合法性:
#    generator 只生成水平轴案例，因为在原始高度上限 12、step 上限 5 的条件下，
#    很难稳定构造“左右各有专属 companion，且二者都在相遇前触发”的纵向样例。
#    但 solver 仍保留对列向输入的通用处理。生成时所有活跃行放在内部行，避免 top/
#    bottom 计数干扰；companion 只放在各自专属 beat，bystander 放在两侧 beat 之外。
# e) 视觉可观察性:
#    输入里每条活跃线都有左右两端 marker，解题者很容易先看出“双向发射”；至少一条
#    线会同时出现左右 companion，且相遇区发生在二者之后，输出会明显呈现
#    “左端原色 -> 左 companion 色 …… 右 companion 色 -> 右端原色”的向心分段。
#    冲突裁决只看距离和固定的右侧优先，不依赖颜色编号等不可视因素。
# f) 可视化整洁度:
#    活跃行之间保留至少一行空隔，且只在宽网格中生成；这样每条线都像一条独立的
#    轨道。额外点数量被限制在 0/1/2 个，避免中心区被过多提示色淹没。
# g) 变化的可推断性:
#    颜色变化完全由局部结构决定：某 companion 落在某一侧自己的 beat 上，就从
#    那一格开始接管该侧后续轨迹。两个 companion 分属两侧且位置可见，因此少量样例
#    即可归纳“哪一个点触发哪一端变色”，不会出现无提示的大规模颜色映射。


def _ordered_beats(edge_pos: int, sign: int, line_len: int, step: int) -> list[int]:
    stop = line_len if sign > 0 else -1
    return list(range(edge_pos, stop, sign * step))


def main(grid: np.ndarray) -> np.ndarray:
    """
    Launch two opposing periodic rays on each active line; companions recolor each ray mid-flight.

    Input structure:
    - The background is the dominant color, and sparse non-bg dots lie on a subset of rows or columns.
    - One opposite-edge pair is denser than the other: left+right selects row-wise processing, while
      top+bottom selects column-wise processing.
    - Every non-empty active line contains two distinct marker colors, one anchored to each opposite edge;
      each marker color appears exactly twice on that line, with the edge dot and the one-step-inward dot
      defining that side's step.
    - Optional extra dots are either companions aligned to exactly one source beyond its inner marker,
      or bystanders aligned to neither source.

    Rule:
    - First decide whether rows or columns are active by comparing opposite-edge-pair boundary counts.
    - On each active line, recover the left/top source and the right/bottom source from the repeated
      colors touching the corresponding edges, then enumerate each source's beat positions in travel order.
    - A source without a companion paints every one of its beats in its marker color; a source with a
      companion keeps the marker color before the collision and switches to the companion color from the
      collision beat onward.
    - If both sources claim the same cell, the one that reaches it in fewer step-hops wins; exact ties
      are broken in favor of the right/bottom source.

    Key insight:
    The puzzle is no longer "find one shooting edge", but "find the active axis, then run two directed
    beat sequences that may both change color before meeting".  The visible output depends on combining
    direction-sensitive recoloring with an explicit geometric conflict rule at shared beat cells.
    """
    g = np.array(grid, dtype=int)
    out = g.copy()
    bg = int(get_bg(g))
    H, W = g.shape

    row_pair = sum(1 for r in range(H) if g[r, 0] != bg) + sum(1 for r in range(H) if g[r, W - 1] != bg)
    col_pair = sum(1 for c in range(W) if g[0, c] != bg) + sum(1 for c in range(W) if g[H - 1, c] != bg)
    pair_counts = {"rows": row_pair, "cols": col_pair}
    max_count = max(pair_counts.values())
    assert max_count > 0, "No non-bg dots on any boundary pair"
    winners = [axis for axis, count in pair_counts.items() if count == max_count]
    assert len(winners) == 1, f"Ambiguous shooting axis: {winners} (counts={pair_counts})"

    by_row = winners[0] == "rows"
    num_lines = H if by_row else W
    line_len = W if by_row else H
    if by_row:
        sides = (
            {"name": "left", "edge_pos": 0, "sign": 1, "tie_priority": 0},
            {"name": "right", "edge_pos": line_len - 1, "sign": -1, "tie_priority": 1},
        )
    else:
        sides = (
            {"name": "top", "edge_pos": 0, "sign": 1, "tie_priority": 0},
            {"name": "bottom", "edge_pos": line_len - 1, "sign": -1, "tie_priority": 1},
        )

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

        sources: list[dict[str, int | list[int]]] = []
        marker_colors: set[int] = set()
        for side in sides:
            candidates = []
            for color, positions in colors.items():
                ps = sorted(int(p) for p in positions)
                if len(ps) != 2 or side["edge_pos"] not in ps:
                    continue
                inner = ps[0] if ps[1] == side["edge_pos"] else ps[1]
                if inner == side["edge_pos"]:
                    continue
                candidates.append((int(color), ps))

            assert len(candidates) == 1, (
                f"Line {li} ({'row' if by_row else 'col'}) needs exactly one {side['name']} marker pair: "
                f"{candidates}, all colors={dict(colors)}"
            )
            marker_color, positions = candidates[0]
            marker_colors.add(marker_color)
            sources.append({"marker_color": marker_color, "positions": positions, **side})

        assert len(marker_colors) == 2, (
            f"Line {li} ({'row' if by_row else 'col'}) must have two distinct marker colors: {sources}"
        )

        claims: dict[int, tuple[int, int, int]] = {}
        for source in sources:
            edge_pos = int(source["edge_pos"])
            sign = int(source["sign"])
            tie_priority = int(source["tie_priority"])
            marker_color = int(source["marker_color"])
            marker_positions = sorted(int(p) for p in source["positions"])
            inner = max(marker_positions) if sign > 0 else min(marker_positions)
            step = abs(inner - edge_pos)
            assert step > 0, f"Zero step on line {li} for source {source}"

            beats = _ordered_beats(edge_pos, sign, line_len, step)
            beat_to_index = {pos: idx for idx, pos in enumerate(beats)}
            aligned_hits: list[tuple[int, int, int]] = []
            for color, positions in colors.items():
                if color in marker_colors:
                    continue
                for pos in positions:
                    beat_idx = beat_to_index.get(int(pos))
                    if beat_idx is None or beat_idx < 2:
                        continue
                    aligned_hits.append((beat_idx, int(pos), int(color)))

            assert len(aligned_hits) <= 1, (
                f"Line {li} ({'row' if by_row else 'col'}) has ambiguous companion for {source['name']}: "
                f"{aligned_hits}, all colors={dict(colors)}"
            )

            collision_idx = None
            collision_color = None
            if aligned_hits:
                collision_idx, _collision_pos, collision_color = aligned_hits[0]

            for beat_idx, pos in enumerate(beats):
                paint = marker_color
                if collision_idx is not None and beat_idx >= collision_idx:
                    paint = int(collision_color)
                travel = beat_idx
                prev = claims.get(pos)
                if prev is None or travel < prev[0] or (travel == prev[0] and tie_priority > prev[1]):
                    claims[pos] = (travel, tie_priority, paint)

        for pos, (_travel, _priority, paint) in claims.items():
            out[cell(li, pos)] = int(paint)

    return out


_ROWS_RANGE = (9, 12)
_COLS_RANGE = (9, 21)
_ACTIVE_LINES_RANGE = (4, 5)
_STEP_RANGE = (1, 5)
_COLORS = list(range(1, 10))

_MAX_RETRIES = 500


def _spaced_interior_indices(n: int, total: int) -> list[int]:
    interior = total - 2
    available = interior - (n - 1)
    if available < n:
        return []
    picks = sorted(random.sample(range(available), n))
    return [1 + picks[i] + i for i in range(n)]


@lru_cache(maxsize=None)
def _pattern_catalog(line_len: int) -> tuple[dict[str, object], ...]:
    patterns: list[dict[str, object]] = []
    for left_step in range(_STEP_RANGE[0], min(_STEP_RANGE[1], line_len - 2) + 1):
        for right_step in range(_STEP_RANGE[0], min(_STEP_RANGE[1], line_len - 2) + 1):
            left_inner = left_step
            right_inner = line_len - 1 - right_step
            if left_inner >= right_inner:
                continue

            left_beats = _ordered_beats(0, 1, line_len, left_step)
            right_beats = _ordered_beats(line_len - 1, -1, line_len, right_step)
            left_idx = {pos: idx for idx, pos in enumerate(left_beats)}
            right_idx = {pos: idx for idx, pos in enumerate(right_beats)}
            overlap = sorted(set(left_beats) & set(right_beats))
            interior_overlap = [pos for pos in overlap if 0 < pos < line_len - 1]
            if not interior_overlap:
                continue

            left_only = [pos for pos in left_beats if left_idx[pos] >= 2 and pos not in right_idx]
            right_only = [pos for pos in right_beats if right_idx[pos] >= 2 and pos not in left_idx]
            dual_choices = [
                (left_companion, right_companion, meet)
                for left_companion in left_only
                for right_companion in right_only
                for meet in interior_overlap
                if left_idx[meet] > left_idx[left_companion] and right_idx[meet] > right_idx[right_companion]
            ]
            left_choices = [
                left_companion
                for left_companion in left_only
                if any(left_idx[meet] > left_idx[left_companion] for meet in interior_overlap)
            ]
            right_choices = [
                right_companion
                for right_companion in right_only
                if any(right_idx[meet] > right_idx[right_companion] for meet in interior_overlap)
            ]
            occupied = {0, left_inner, right_inner, line_len - 1}
            beat_union = set(left_beats) | set(right_beats)
            bystander_choices = [
                pos for pos in range(1, line_len - 1) if pos not in occupied and pos not in beat_union
            ]

            patterns.append(
                {
                    "left_step": left_step,
                    "right_step": right_step,
                    "left_inner": left_inner,
                    "right_inner": right_inner,
                    "interior_overlap": interior_overlap,
                    "dual_choices": dual_choices,
                    "left_choices": left_choices,
                    "right_choices": right_choices,
                    "bystander_choices": bystander_choices,
                }
            )

    return tuple(patterns)


def _valid_widths() -> list[int]:
    widths = []
    for width in range(max(_COLS_RANGE[0], 9), _COLS_RANGE[1] + 1):
        if any(pattern["dual_choices"] for pattern in _pattern_catalog(width)):
            widths.append(width)
    return widths


_VALID_WIDTHS = _valid_widths()


def _choose_dims(n_active: int) -> tuple[int, int, int, int] | None:
    min_rows = max(_ROWS_RANGE[0], 2 * n_active + 1)
    if min_rows > _ROWS_RANGE[1] or not _VALID_WIDTHS:
        return None
    H = random.randint(min_rows, _ROWS_RANGE[1])
    W = random.choice(_VALID_WIDTHS)
    return H, W, H, W


def _line_kind_sequence(n_active: int) -> list[str]:
    kinds = ["dual_collision"]
    if n_active >= 2:
        kinds.append(random.choice(["left_collision", "right_collision"]))
    if n_active >= 3:
        kinds.append(random.choice(["marker_only", "bystander"]))
    while len(kinds) < n_active:
        kinds.append(
            random.choices(
                ["dual_collision", "left_collision", "right_collision", "marker_only", "bystander"],
                weights=[0.30, 0.18, 0.18, 0.18, 0.16],
                k=1,
            )[0]
        )
    random.shuffle(kinds)
    return kinds


def _eligible_patterns(line_len: int, kind: str) -> list[dict[str, object]]:
    patterns = list(_pattern_catalog(line_len))
    if kind == "dual_collision":
        return [pattern for pattern in patterns if pattern["dual_choices"]]
    if kind == "left_collision":
        return [pattern for pattern in patterns if pattern["left_choices"]]
    if kind == "right_collision":
        return [pattern for pattern in patterns if pattern["right_choices"]]
    if kind == "bystander":
        return [pattern for pattern in patterns if pattern["bystander_choices"]]
    if kind == "marker_only":
        return patterns
    return []


def _build_line_dots(
    line_len: int, palette: list[int], kind: str
) -> tuple[list[tuple[int, int]], dict[str, object]] | None:
    patterns = _eligible_patterns(line_len, kind)
    if not patterns:
        return None
    pattern = random.choice(patterns)
    left_step = int(pattern["left_step"])
    right_step = int(pattern["right_step"])
    left_inner = int(pattern["left_inner"])
    right_inner = int(pattern["right_inner"])

    need_colors = 2
    if kind in {"left_collision", "right_collision", "bystander"}:
        need_colors = 3
    elif kind == "dual_collision":
        need_colors = 4
    if len(palette) < need_colors:
        return None

    chosen = random.sample(palette, need_colors)
    left_color = chosen[0]
    right_color = chosen[1]
    dots = [
        (0, left_color),
        (left_inner, left_color),
        (right_inner, right_color),
        (line_len - 1, right_color),
    ]
    meta: dict[str, object] = {
        "kind": kind,
        "left_step": left_step,
        "right_step": right_step,
        "interior_overlap": list(pattern["interior_overlap"]),
    }

    if kind == "dual_collision":
        left_companion, right_companion, meet = random.choice(list(pattern["dual_choices"]))
        dots.append((int(left_companion), chosen[2]))
        dots.append((int(right_companion), chosen[3]))
        meta["left_companion"] = int(left_companion)
        meta["right_companion"] = int(right_companion)
        meta["post_change_meet"] = int(meet)
        return dots, meta

    if kind == "left_collision":
        left_companion = int(random.choice(list(pattern["left_choices"])))
        dots.append((left_companion, chosen[2]))
        meta["left_companion"] = left_companion
        return dots, meta

    if kind == "right_collision":
        right_companion = int(random.choice(list(pattern["right_choices"])))
        dots.append((right_companion, chosen[2]))
        meta["right_companion"] = right_companion
        return dots, meta

    if kind == "bystander":
        bystander = int(random.choice(list(pattern["bystander_choices"])))
        dots.append((bystander, chosen[2]))
        meta["bystander"] = bystander

    return dots, meta


def _axis_counts(inp: np.ndarray) -> dict[str, int]:
    bg = int(get_bg(inp))
    H, W = inp.shape
    return {
        "rows": sum(1 for r in range(H) if inp[r, 0] != bg) + sum(1 for r in range(H) if inp[r, W - 1] != bg),
        "cols": sum(1 for c in range(W) if inp[0, c] != bg) + sum(1 for c in range(W) if inp[H - 1, c] != bg),
    }


def _core_behavior_score(inp: np.ndarray) -> tuple[int, int]:
    bg = int(get_bg(inp))
    H, W = inp.shape
    score = 0
    active = 0
    for r in range(H):
        colors: dict[int, list[int]] = defaultdict(list)
        for c in range(W):
            value = int(inp[r, c])
            if value != bg:
                colors[value].append(c)
        if not colors:
            continue
        active += 1
        repeats = [(color, sorted(pos)) for color, pos in colors.items() if len(pos) == 2]
        if len(repeats) != 2:
            continue
        left_pair = next((pair for pair in repeats if 0 in pair[1]), None)
        right_pair = next((pair for pair in repeats if W - 1 in pair[1]), None)
        if left_pair is None or right_pair is None:
            continue
        left_step = left_pair[1][1]
        right_step = (W - 1) - right_pair[1][0]
        left_beats = set(_ordered_beats(0, 1, W, left_step))
        right_beats = set(_ordered_beats(W - 1, -1, W, right_step))
        left_extras = [pos for color, positions in colors.items() if color not in {left_pair[0], right_pair[0]} for pos in positions]
        left_only_hit = any(pos in left_beats and pos not in right_beats for pos in left_extras)
        right_only_hit = any(pos in right_beats and pos not in left_beats for pos in left_extras)
        if left_only_hit and right_only_hit:
            score += 1
    return score, active


def _gen_attempt() -> np.ndarray | None:
    n_active = random.randint(*_ACTIVE_LINES_RANGE)
    dims = _choose_dims(n_active)
    if dims is None:
        return None
    H, W, num_lines, _line_len = dims
    active_lines = _spaced_interior_indices(n_active, num_lines)
    if not active_lines:
        return None

    inp = np.zeros((H, W), dtype=int)
    palette = random.sample(_COLORS, random.randint(5, 7))
    dual_lines = 0

    for li, kind in zip(active_lines, _line_kind_sequence(n_active)):
        built = _build_line_dots(W, palette, kind)
        if built is None:
            return None
        dots, meta = built
        for pos, color in dots:
            if inp[li, pos] != 0:
                return None
            inp[li, pos] = color
        if meta["kind"] == "dual_collision":
            dual_lines += 1

    if dual_lines == 0:
        return None

    axis_counts = _axis_counts(inp)
    if axis_counts["rows"] <= axis_counts["cols"]:
        return None

    core_score, active = _core_behavior_score(inp)
    if core_score == 0 or active != n_active:
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
