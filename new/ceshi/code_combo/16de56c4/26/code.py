# Combo variant of 16de56c4 (index 26)
# combo_spec: D7/0+D8/0
# Name: 标记点擦除+紧凑裁剪

from __future__ import annotations

from collections import defaultdict
import random

import numpy as np

from arc_utils import get_bg


# Design analysis for the combo variant
# -------------------------------------
# a) 具体化与行为追踪
# - 原题先识别唯一射击边，再按该方向逐行/逐列解码：每条活跃线上有一对同色 marker，
#   两点间距就是 step；若同 beat 上还有异色 companion，则该颜色接管轨迹并把填充范围
#   裁到 marker 与 companion 的跨度；否则 marker 颜色沿整条线周期填充。
# - D7/0 把 marker 从“要被重涂的实体”改成“只负责发指令的编码点”。例如左边发射、
#   step=3、marker 在位置 0/3、companion 在 9：原题输出 beat 0/3/6/9 都变成 companion
#   色；融合后先只保留非 marker beat，因此只新增位置 6，companion 位置 9 保持，随后
#   位置 0/3 擦成背景。
# - D8/0 再对 D7/0 的结果做最小非背景外接框裁剪。继续上例，如果所有活跃行都来自左边，
#   且 marker 擦除后最早幸存内容从列 6 开始，那么整张输出会整体去掉左侧空白，只保留
#   剩余轨迹与旁观者形成的紧凑矩形。
# - 关键差异不是“先裁剪还是先擦除”可交换：若先按原题保留 marker 再裁剪，bbox 仍会被
#   射击边上的 marker 锁在原边界；只有先擦掉 marker，裁剪边界才会真正后退。
# b) 复杂度校准
# - 若允许 step 太大或 companion 离 marker 太近，擦除后可能只剩输入里本来就有的点，视觉上
#   会退化成“删两个点”。因此 generator 只允许至少留下一个新增轨迹格的步长/companion。
# - 若活跃线碰到首尾行(列)，tight crop 在垂直于射线的方向上就不明显；本变体把活跃线放在
#   内部并保持间隔，让裁剪在至少两个维度都可见。
# - companion 与 bystander 仍保留，但 bystander 被限制在最终轨迹 bbox 内，不去扩张裁剪边界，
#   避免视觉重点从“marker 消失导致尺寸收缩”转移到随机孤点。
# c) 隐式假设审查
# - 原 solver 的 `out = g.copy()` 仍可用，但原来的写入条件 `cur == bg or cur == marker`
#   依赖 marker 会被保留为涂色结果；融合后该前提失效，marker 位置必须跳过填充并在末尾强制
#   设回背景。
# - 原 solver 默认输出尺寸与输入相同；加入 D8/0 后，求解的真正结果是“先得到整幅未裁剪画布，
#   再按非背景 bbox 切片”。因此需要把“解码”和“裁剪”拆成两个步骤。
# - companion 检测仍依赖“每条非空线恰有一组 repeated marker，且至多一个 aligned foreign
#   dot 会作为 companion”。generator 继续保证这一点，否则同线多重复色会引入处理顺序歧义。
# - 新增的 crop 假设输出至少还剩一个非背景格。generator 用步长与 companion 距离约束保证
#   每条活跃线在擦除 marker 后仍有可见残留，因此整图不会被裁成空。
# d) 放置合法性
# - 活跃线采用带空隙的内部索引，避免线与线在 bbox 边界处堆叠，也保证裁剪后有清晰留白被移除。
# - 每条线仅放两枚 marker 和至多一个第三点；third dot 绝不与 marker 重合，也不越界。
# - bystander 只允许落在该线最终轨迹 span 的内部非 beat 位置，既不碰射击边，也不碰 opposite
#   edge，从而不会破坏唯一射击边判定或把 bbox 又撑回去。
# e) 视觉可观察性
# - 多个样例里都会出现同一种现象：输入的重复点对消失，输出只剩它们“计算”出来的周期轨迹，
#   而且整张输出比输入更小。人类可以直接从几组样例看出“边缘上的那对点只是说明书”。
# - 裁剪规则完全由可见几何决定：保留所有非背景，再取最小包围框；不依赖颜色编号、排序号或
#   不可视元数据。
# f) 可视化整洁度
# - 轨迹都沿平行的独立行/列展开，彼此不交叉；裁剪只去掉纯背景边带，不引入额外符号。
# - bystander 数量保持很少，且不放在 bbox 外沿，所以画面仍以“几条平行周期轨迹”作为主结构。
# g) 变化的可推断性
# - 颜色逻辑没有新增隐藏映射：无 companion 时轨迹用 marker 色，有 companion 时轨迹用
#   companion 色，bystander 原样保留。新增变化只在“marker 消失”和“结果裁剪”两点。
# 新 main() 的新隐式假设审查
# - 假设 1：每条非空 shooting line 仍能从输入中找到一对且仅一对 marker。generator 保证总点数
#   只有 2 或 3 个，不会制造同线多重复色。
# - 假设 2：tight crop 之后仍能看出射线方向。generator 保证至少 4 条活跃线，并让它们共享同一
#   射击边，因此裁剪后仍有重复平行结构支撑规律。
# - 假设 3：融合不是退化成“原题输出再裁剪”。generator 显式比较“原题输出裁剪版”和融合输出，
#   并要求二者 shape 不同，确保 marker 擦除真的改变了 bbox。
# - 假设 4：crop 以输入背景色为准。由于输出只会复用输入已有非背景颜色，不会生成新的背景色，
#   这个假设成立；generator 也不制造整图被完全擦空的情况。


_ROWS_RANGE = (7, 12)
_COLS_RANGE = (9, 21)
_ACTIVE_LINES_RANGE = (4, 9)
_STEP_RANGE = (1, 5)
_COLORS = list(range(1, 10))
_EDGES = ("top", "bottom", "left", "right")

_P_MARKER_ONLY = 0.35
_P_COMPANION = 0.45

_MAX_RETRIES = 400


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


def _trace_uncropped(grid: np.ndarray, erase_markers: bool) -> np.ndarray:
    g = np.array(grid, dtype=int)
    out = g.copy()
    bg = int(get_bg(g))
    H, W = g.shape

    edge = _detect_edge(g, bg)
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
        assert repeating, (
            f"Line {li} ({'row' if by_row else 'col'}) has non-bg dots "
            f"{dict(colors)} but no repeating color"
        )

        for marker, mpos in repeating.items():
            assert len(mpos) == 2, (
                f"Line {li} ({'row' if by_row else 'col'}) marker {marker} "
                f"has {len(mpos)} dots (expected 2): {mpos}"
            )
            a, b = sorted(mpos)
            step = b - a
            offset = a % step

            companion = None
            for other_color, other_positions in colors.items():
                if other_color == marker:
                    continue
                if any(pos % step == offset for pos in other_positions):
                    companion = other_color
                    break

            if companion is not None:
                paint = companion
                all_positions = mpos + colors[companion]
                lo, hi = min(all_positions), max(all_positions)
            else:
                paint = marker
                lo, hi = 0, line_len - 1

            marker_positions = set(mpos)
            for p in range(lo, hi + 1):
                if p % step != offset:
                    continue
                if erase_markers and p in marker_positions:
                    continue
                rc = cell(li, p)
                cur = int(out[rc])
                if erase_markers:
                    if cur == bg:
                        out[rc] = paint
                else:
                    if cur == bg or cur == marker:
                        out[rc] = paint

            if erase_markers:
                for p in marker_positions:
                    out[cell(li, p)] = bg

    return out


def _tight_crop(grid: np.ndarray, bg: int) -> np.ndarray:
    mask = grid != bg
    assert bool(mask.any()), "Tight crop requires at least one non-bg cell"
    rows = np.flatnonzero(mask.any(axis=1))
    cols = np.flatnonzero(mask.any(axis=0))
    return grid[rows[0] : rows[-1] + 1, cols[0] : cols[-1] + 1]


def _bbox_of_nonbg(grid: np.ndarray, bg: int) -> tuple[int, int, int, int]:
    mask = grid != bg
    assert bool(mask.any())
    rows = np.flatnonzero(mask.any(axis=1))
    cols = np.flatnonzero(mask.any(axis=0))
    return int(rows[0]), int(cols[0]), int(rows[-1]), int(cols[-1])


def _solve_original_uncropped(grid: np.ndarray) -> np.ndarray:
    return _trace_uncropped(grid, erase_markers=False)


def _solve_variant_uncropped(grid: np.ndarray) -> np.ndarray:
    return _trace_uncropped(grid, erase_markers=True)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Decode periodic rays from one dominant edge, erase the encoding dots, then crop tightly.

    Input structure:
    - Background is the most common color, with sparse non-background dots elsewhere.
    - Exactly one border has the highest non-background count; that border determines the
      shooting direction: left/right act across rows, top/bottom act across columns.
    - Each active shooting line contains exactly one repeated color with two dots; those two
      dots are marker instructions, and their separation is the step.
    - A different-colored dot on the same modular beat is a companion that clips the ray and
      provides the paint color; a different-colored off-beat dot is only a bystander.

    Rule:
    - Find the unique shooting edge and inspect every row or column parallel to that direction.
    - For each marker pair, compute the beat from the gap between the two markers.
      Without a companion, paint every beat on the full line in the marker color.
      With a companion, paint only beat cells inside the marker/companion span in the
      companion color.
    - The marker cells themselves are not kept: after the ray is decoded, both marker dots are
      erased back to background, while companions and bystanders remain.
    - Finally crop the result to the smallest rectangle containing all non-background cells.

    Key insight: the repeated dots are temporary instructions, not permanent output. Removing
    them changes which cells define the final bounding box, so the tight crop is determined by
    the surviving trajectory rather than by the original edge markers.
    """
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))
    traced = _solve_variant_uncropped(g)
    return _tight_crop(traced, bg)


def _choose_dims(edge: str, n_active: int):
    by_row = edge in ("left", "right")
    min_num_lines = 2 * n_active + 1
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


def _spaced_interior_indices(n: int, total: int) -> list[int]:
    interior = total - 2
    available = interior - (n - 1)
    if available < n:
        return []
    picks = sorted(random.sample(range(available), n))
    return [1 + picks[i] + i for i in range(n)]


def _valid_steps(line_len: int) -> list[int]:
    hi = min(_STEP_RANGE[1], line_len - 1)
    return [step for step in range(_STEP_RANGE[0], hi + 1) if 2 * step <= line_len - 1]


def _line_geometry(edge: str, line_len: int, step: int) -> dict[str, object]:
    if edge in ("left", "top"):
        edge_pos = 0
        inner_pos = step
        trail_beats = list(range(2 * step, line_len, step))
        companion_beats = list(range(3 * step, line_len, step))
    else:
        edge_pos = line_len - 1
        inner_pos = line_len - 1 - step
        trail_beats = list(range(line_len - 1 - 2 * step, -1, -step))
        companion_beats = list(range(line_len - 1 - 3 * step, -1, -step))

    assert trail_beats, "Need at least one surviving non-marker beat"
    offset = edge_pos % step
    lo = min(trail_beats)
    hi = max(trail_beats)
    safe_bystanders = [
        pos
        for pos in range(lo, hi + 1)
        if pos % step != offset and pos not in (0, line_len - 1)
    ]
    return {
        "edge_pos": edge_pos,
        "inner_pos": inner_pos,
        "offset": offset,
        "trail_beats": trail_beats,
        "companion_beats": companion_beats,
        "safe_bystanders": safe_bystanders,
    }


def _build_line_dots(edge: str, line_len: int, marker_color: int, step: int):
    geom = _line_geometry(edge, line_len, step)
    dots = [
        (int(geom["edge_pos"]), marker_color),
        (int(geom["inner_pos"]), marker_color),
    ]

    other_colors = [color for color in _COLORS if color != marker_color]
    companion_beats = list(geom["companion_beats"])
    safe_bystanders = list(geom["safe_bystanders"])

    roll = random.random()
    if roll < _P_MARKER_ONLY:
        line_type = "marker_only"
    elif roll < _P_MARKER_ONLY + _P_COMPANION and companion_beats and other_colors:
        line_type = "companion"
    elif safe_bystanders and other_colors:
        line_type = "bystander"
    elif companion_beats and other_colors:
        line_type = "companion"
    else:
        line_type = "marker_only"

    if line_type == "companion":
        dots.append((random.choice(companion_beats), random.choice(other_colors)))
    elif line_type == "bystander":
        dots.append((random.choice(safe_bystanders), random.choice(other_colors)))

    return dots, line_type


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
    for other_edge in _EDGES:
        if other_edge == edge:
            continue
        check = boundary[other_edge]
        counts[other_edge] = sum(1 for r, c in dots_rc if check(r, c))
    return counts


def _feasible_active_counts(edge: str) -> list[int]:
    by_row = edge in ("left", "right")
    max_lines = _ROWS_RANGE[1] if by_row else _COLS_RANGE[1]
    lo, hi = _ACTIVE_LINES_RANGE
    return [n for n in range(lo, hi + 1) if 2 * n + 1 <= max_lines]


def _gen_attempt() -> np.ndarray | None:
    edge = random.choice(_EDGES)
    feasible_counts = _feasible_active_counts(edge)
    if not feasible_counts:
        return None
    n_active = random.choice(feasible_counts)
    by_row = edge in ("left", "right")

    dims = _choose_dims(edge, n_active)
    if dims is None:
        return None
    H, W, num_lines, line_len = dims

    active_lines = _spaced_interior_indices(n_active, num_lines)
    if not active_lines:
        return None

    valid_steps = _valid_steps(line_len)
    if not valid_steps:
        return None

    def cell(line: int, pos: int) -> tuple[int, int]:
        return (line, pos) if by_row else (pos, line)

    inp = np.zeros((H, W), dtype=int)
    shooting_edge_dots = 0
    all_rc: list[tuple[int, int]] = []
    line_types: list[str] = []

    for li in active_lines:
        marker_color = random.choice(_COLORS)
        step = random.choice(valid_steps)
        dots, line_type = _build_line_dots(edge, line_len, marker_color, step)

        for pos, color in dots:
            r, c = cell(li, pos)
            if inp[r, c] != 0:
                return None
            inp[r, c] = color
            all_rc.append((r, c))

        shooting_edge_dots += 1
        line_types.append(line_type)

    other_counts = _count_other_edge_dots(all_rc, edge, H, W)
    if any(count >= shooting_edge_dots for count in other_counts.values()):
        return None

    try:
        variant_uncropped = _solve_variant_uncropped(inp)
        original_uncropped = _solve_original_uncropped(inp)
        out = main(inp)
    except Exception:
        return None

    if np.array_equal(inp, variant_uncropped):
        return None
    if out.shape == inp.shape:
        return None

    bg = int(get_bg(inp))
    cropped_original = _tight_crop(original_uncropped, bg)
    if cropped_original.shape == out.shape:
        return None

    erased_markers = 0
    gained_trail = 0
    for (r, c), value in np.ndenumerate(inp):
        out_value = int(variant_uncropped[r, c])
        if int(value) != 0 and out_value == 0:
            erased_markers += 1
        if int(value) == 0 and out_value != 0:
            gained_trail += 1
    if erased_markers < 2 * n_active or gained_trail == 0:
        return None

    vr0, vc0, vr1, vc1 = _bbox_of_nonbg(variant_uncropped, bg)
    or0, oc0, or1, oc1 = _bbox_of_nonbg(original_uncropped, bg)
    if edge == "left" and vc0 == 0:
        return None
    if edge == "right" and vc1 == W - 1:
        return None
    if edge == "top" and vr0 == 0:
        return None
    if edge == "bottom" and vr1 == H - 1:
        return None
    if edge == "left" and vc0 <= oc0:
        return None
    if edge == "right" and vc1 >= oc1:
        return None
    if edge == "top" and vr0 <= or0:
        return None
    if edge == "bottom" and vr1 >= or1:
        return None

    if "companion" not in line_types and n_active >= 5 and random.random() < 0.6:
        return None
    if "bystander" not in line_types and random.random() < 0.2:
        return None

    return inp


def generate_input() -> np.ndarray:
    for _ in range(_MAX_RETRIES):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError(f"Failed to generate valid input after {_MAX_RETRIES} attempts")
