# Combo variant of db695cfb (index 25)
# combo_spec: D1/0+D2/0+D16/2
# Name: 直线路径+平行延伸+单方向

from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np

_SOLVER_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_SOLVER_DIR) not in sys.path:
    sys.path.append(str(_SOLVER_DIR))

from arc_utils import get_bg


# Design analysis for the combo variant
# =====================================
# a) Concrete behavior and trace:
# - 原题先用蓝色端点补一条 45 度对角线，再让对角线上的品红点沿垂直对角方向双向发射。
# - 本融合把三处关键逻辑一起改掉：
#   1. 主路径改为水平或垂直直线。
#   2. 品红延伸不再转 90 度，而是与主路径同轴平行。
#   3. 每个品红只发一条线，方向由它相对主线中点的位置决定。
# - 我把“方向由位置决定”具体化为中点分侧规则：
#   水平段上，位于中点左侧的触发点只向左延伸，位于右侧的只向右延伸；
#   垂直段上，位于中点上侧的只向上延伸，位于下侧的只向下延伸。
# - 小例子：若蓝色端点在 (4,2) 与 (4,8)，品红在 (4,3) 与 (4,7)，
#   原题会补对角线，不适用；D1 单变体会补第 4 行并从两枚品红各自向上和向下打竖线；
#   本融合则补第 4 行，(4,3) 只向左沿第 4 行延伸，(4,7) 只向右沿第 4 行延伸，
#   于是可见新增的品红只会出现在蓝段两端外侧。
#
# b) Complexity calibration:
# - 由于“平行延伸 + on-path 触发”意味着射线与主线共线，如果触发点太靠近端点或者端点贴边，
#   新规则会退化成几乎看不到品红新增像素。因此生成器强制：
#   - 线段端点在延伸轴上不贴边；
#   - 每个有效线段至少有 1 个非中点触发；
#   - 整个样本里至少出现 2 个可见的单向方向类别。
# - 段数控制在 2-3，干扰点适度减少，避免视觉上只剩一大片同轴覆盖。
#
# c) Implicit-assumption audit against the original main():
# - 原 `diag_path()` 假设两个端点一定在同一条 45 度对角线上；融合后必须改成 same-row / same-column。
# - 原“选择 line_color”的条件是最长同色对角配对；现在要改成最长同色水平/垂直配对。
# - 原分组依据 `r-c` 和 `r+c`；现在必须按 row / column 分组，否则连不到目标段。
# - 原触发方向是两条垂直路径；D2 使其变成与主段平行，D16 再把双向改成单向，
#   所以 `draw_segment()` 不能再给每个触发点登记一对方向，只能登记由位置决定的一条。
# - 平行延伸与主段共线，旧题里“射线天然不碰主线”的前提不成立；solver 仍需保留
#   `only fill bg cells` 语义，避免覆盖蓝段、触发点和其他对象。
#
# d) Placement legality:
# - 生成器仍采用 output-first：先摆真线段和触发点，再计算输出中会被补出来的蓝线和品红线。
# - 由于平行单向射线会经过主线内部，所以合法性检查只对 `ray_cells - path_cells`
#   的“主线之外新增部分”做碰撞判定。
# - 蓝色干扰点不能与任何真蓝端点共享行、列或对角线，否则 solver 可能错误地选到别的极值配对。
# - 品红干扰点避免与已有品红同排或同列，降低“触发色被误判成主线色”的风险；
#   最终还会做一次严格检查，确保蓝色拥有唯一最长的轴对齐同色配对。
#
# e) Visual observability:
# - 规则完全由可见结构决定：蓝点的轴对齐端点、品红点在主线上的位置、以及它们相对中点的左右/上下关系。
# - 不使用颜色编号、输入顺序或不可见元数据。
# - 生成器拒绝与两个弱基线等价的样本：
#   - `straight + perpendicular + two-sided`
#   - `straight + parallel + two-sided`
#   这样可以确保训练样本确实体现“平行”与“单方向”两项融合变化。
#
# f) Visual cleanliness:
# - 段路径、段外可见射线、干扰点彼此不重叠。
# - 线段数量和触发数量受控，避免大量同轴覆盖造成读图混乱。
#
# g) New assumptions introduced by the fused rule:
# - 中点分侧要求触发点不要正好落在主线几何中点，否则方向会变得不够可读；生成器显式排除中点触发。
# - solver 允许任意输入里同一触发点被多条主线激活，并用 set 收集方向；生成样本则保证不会依赖这种边角情况。
# - 为了让单向平行延伸可见，生成器保证至少有一部分射线会越过端点继续进入背景区。


def _straight_path(p: tuple[int, int], q: tuple[int, int]) -> list[tuple[int, int]]:
    if p[0] == q[0]:
        lo, hi = sorted((p[1], q[1]))
        return [(p[0], c) for c in range(lo, hi + 1)]
    if p[1] == q[1]:
        lo, hi = sorted((p[0], q[0]))
        return [(r, p[1]) for r in range(lo, hi + 1)]
    raise ValueError("points must share a row or column")


def _single_parallel_dir(
    pos: tuple[int, int],
    p: tuple[int, int],
    q: tuple[int, int],
) -> tuple[int, int]:
    if p[0] == q[0]:
        mid2 = p[1] + q[1]
        return (0, -1) if 2 * pos[1] < mid2 else (0, 1)
    mid2 = p[0] + q[0]
    return (-1, 0) if 2 * pos[0] < mid2 else (1, 0)


def _best_straight_pair_len(pts: list[tuple[int, int]]) -> int:
    best = -1
    for i in range(len(pts)):
        r0, c0 = pts[i]
        for j in range(i + 1, len(pts)):
            r1, c1 = pts[j]
            if (r0 == r1) == (c0 == c1):
                continue
            best = max(best, abs(r1 - r0) + abs(c1 - c0))
    return best


def main(grid: np.ndarray) -> np.ndarray:
    """
    Complete same-color straight endpoint pairs, then let on-path trigger pixels extend a single same-axis ray chosen by side of the midpoint.

    Input structure:
    - The background is the most frequent color. One non-background color appears as endpoint pairs that share a row or a column, and trigger-colored pixels may lie strictly between those endpoints on the same straight path.

    Rule:
    - Fill each full horizontal or vertical segment between the extreme same-color endpoints.
    - Whenever the filled segment passes through a non-line pixel, keep that trigger pixel and extend exactly one ray in the trigger's own color, parallel to the segment: triggers on the left or upper half extend left or up, while triggers on the right or lower half extend right or down, continuing to the grid boundary without overwriting existing non-background cells.

    Key insight:
    - The trigger no longer fires a symmetric perpendicular cross. It stays tied to the straight carrier path and uses its visible position relative to the segment midpoint to choose one same-axis direction.
    """

    grid = np.asarray(grid, dtype=int)
    h, w = grid.shape
    out = grid.copy()

    bg = get_bg(grid)

    non_bg = [(r, c, int(grid[r, c])) for r in range(h) for c in range(w) if int(grid[r, c]) != bg]
    assert non_bg, "grid must contain at least one non-background pixel"

    by_color: dict[int, list[tuple[int, int]]] = {}
    for r, c, col in non_bg:
        by_color.setdefault(col, []).append((r, c))

    best = None  # (dist, color)
    for col, pts in by_color.items():
        if len(pts) < 2:
            continue
        for i in range(len(pts)):
            r0, c0 = pts[i]
            for j in range(i + 1, len(pts)):
                r1, c1 = pts[j]
                if (r0 == r1) == (c0 == c1):
                    continue
                dist = abs(r1 - r0) + abs(c1 - c0)
                cand = (dist, col)
                if best is None or cand > best:
                    best = cand

    assert best is not None, "at least one straight-aligned same-color pair must exist"

    _, line_color = best
    pts = by_color[int(line_color)]

    triggers: dict[tuple[int, int, int], set[tuple[int, int]]] = {}

    row_groups: dict[int, list[tuple[int, int]]] = {}
    col_groups: dict[int, list[tuple[int, int]]] = {}
    for r, c in pts:
        row_groups.setdefault(r, []).append((r, c))
        col_groups.setdefault(c, []).append((r, c))

    def draw_segment(p: tuple[int, int], q: tuple[int, int]) -> None:
        path = _straight_path(p, q)
        for r, c in path:
            v = int(out[r, c])
            if v == bg:
                out[r, c] = int(line_color)
            elif v != int(line_color):
                key = (r, c, v)
                triggers.setdefault(key, set()).add(_single_parallel_dir((r, c), p, q))

    for group in row_groups.values():
        if len(group) >= 2:
            group = sorted(group, key=lambda rc: rc[1])
            draw_segment(group[0], group[-1])

    for group in col_groups.values():
        if len(group) >= 2:
            group = sorted(group, key=lambda rc: rc[0])
            draw_segment(group[0], group[-1])

    for (r0, c0, tcol), dirs in triggers.items():
        for ddr, ddc in dirs:
            r, c = r0 + ddr, c0 + ddc
            while 0 <= r < h and 0 <= c < w:
                if int(out[r, c]) == bg:
                    out[r, c] = tcol
                r += ddr
                c += ddc

    return out


# --------------- generator (c) ---------------

# === CFG: canvas and structural bounds ===
_H_RANGE = (8, 25)
_W_RANGE = (9, 21)
_LINE_COLOR = 1
_TRIGGER_COLOR = 6
_SEGMENT_COUNT = (2, 3)
_SEG_LEN_RANGE = (5, 16)
_TRIGGERS_PER_SEG = (1, 2)

# === CFG: distractor counts ===
_UNPAIRED_LINE_PIXELS = (0, 1)
_OFF_SEG_TRIGGERS = (0, 2)

# === CFG: retry limits ===
_SEGMENT_PLACE_RETRIES = 80
_DISTRACTOR_RETRIES = 50
_GEN_RETRIES = 500


def _shares_axis_or_diagonal(p: tuple[int, int], q: tuple[int, int]) -> bool:
    return p[0] == q[0] or p[1] == q[1] or abs(p[0] - q[0]) == abs(p[1] - q[1])


def _visible_side_buckets(path: list[tuple[int, int]]) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    p, q = path[0], path[-1]
    negative: list[tuple[int, int]] = []
    positive: list[tuple[int, int]] = []
    if p[0] == q[0]:
        mid2 = p[1] + q[1]
        for rc in path[1:-1]:
            cur2 = 2 * rc[1]
            if cur2 < mid2:
                negative.append(rc)
            elif cur2 > mid2:
                positive.append(rc)
    else:
        mid2 = p[0] + q[0]
        for rc in path[1:-1]:
            cur2 = 2 * rc[0]
            if cur2 < mid2:
                negative.append(rc)
            elif cur2 > mid2:
                positive.append(rc)
    return negative, positive


def _compute_single_ray_cells(
    trigger_specs: list[tuple[tuple[int, int], tuple[int, int]]],
    H: int,
    W: int,
) -> set[tuple[int, int]]:
    cells: set[tuple[int, int]] = set()
    for (tr, tc), (ddr, ddc) in trigger_specs:
        r, c = tr + ddr, tc + ddc
        while 0 <= r < H and 0 <= c < W:
            cells.add((r, c))
            r += ddr
            c += ddc
    return cells


def _build_expected(inp: np.ndarray, bg: int, segments) -> np.ndarray:
    H, W = inp.shape
    expected = inp.copy()
    for _, _, trigger_specs, path in segments:
        for r, c in path:
            if expected[r, c] == bg:
                expected[r, c] = _LINE_COLOR
        for (tr, tc), (ddr, ddc) in trigger_specs:
            r, c = tr + ddr, tc + ddc
            while 0 <= r < H and 0 <= c < W:
                if expected[r, c] == bg:
                    expected[r, c] = _TRIGGER_COLOR
                r += ddr
                c += ddc
    return expected


def _extract_solver_segments(inp: np.ndarray):
    H, W = inp.shape
    bg = get_bg(inp)
    by_color: dict[int, list[tuple[int, int]]] = {}
    for r in range(H):
        for c in range(W):
            col = int(inp[r, c])
            if col == bg:
                continue
            by_color.setdefault(col, []).append((r, c))

    best = None
    for col, pts in by_color.items():
        if len(pts) < 2:
            continue
        for i in range(len(pts)):
            r0, c0 = pts[i]
            for j in range(i + 1, len(pts)):
                r1, c1 = pts[j]
                if (r0 == r1) == (c0 == c1):
                    continue
                cand = (abs(r1 - r0) + abs(c1 - c0), col)
                if best is None or cand > best:
                    best = cand

    assert best is not None
    _, line_color = best
    pts = by_color[int(line_color)]

    segments = []
    row_groups: dict[int, list[tuple[int, int]]] = {}
    col_groups: dict[int, list[tuple[int, int]]] = {}
    for r, c in pts:
        row_groups.setdefault(r, []).append((r, c))
        col_groups.setdefault(c, []).append((r, c))

    for group in row_groups.values():
        if len(group) >= 2:
            group = sorted(group, key=lambda rc: rc[1])
            p, q = group[0], group[-1]
            path = _straight_path(p, q)
            triggers = [(r, c, int(inp[r, c])) for r, c in path if int(inp[r, c]) not in (bg, line_color)]
            segments.append((p, q, triggers, path))

    for group in col_groups.values():
        if len(group) >= 2:
            group = sorted(group, key=lambda rc: rc[0])
            p, q = group[0], group[-1]
            path = _straight_path(p, q)
            triggers = [(r, c, int(inp[r, c])) for r, c in path if int(inp[r, c]) not in (bg, line_color)]
            segments.append((p, q, triggers, path))

    return bg, int(line_color), segments


def _baseline_output_from_input(inp: np.ndarray, mode: str) -> np.ndarray:
    H, W = inp.shape
    out = inp.copy()
    bg, line_color, segments = _extract_solver_segments(inp)

    for p, q, triggers, path in segments:
        for r, c in path:
            if int(out[r, c]) == bg:
                out[r, c] = line_color

        if p[0] == q[0]:
            if mode == "perp2":
                dirs = [(-1, 0), (1, 0)]
            elif mode == "parallel2":
                dirs = [(0, -1), (0, 1)]
            else:
                dirs = None
        else:
            if mode == "perp2":
                dirs = [(0, -1), (0, 1)]
            elif mode == "parallel2":
                dirs = [(-1, 0), (1, 0)]
            else:
                dirs = None

        for r0, c0, tcol in triggers:
            use_dirs = [_single_parallel_dir((r0, c0), p, q)] if mode == "single" else dirs
            for dr, dc in use_dirs:
                r, c = r0 + dr, c0 + dc
                while 0 <= r < H and 0 <= c < W:
                    if int(out[r, c]) == bg:
                        out[r, c] = tcol
                    r += dr
                    c += dc

    return out


def _blue_is_unique_best(inp: np.ndarray, bg: int) -> bool:
    by_color: dict[int, list[tuple[int, int]]] = {}
    H, W = inp.shape
    for r in range(H):
        for c in range(W):
            col = int(inp[r, c])
            if col == bg:
                continue
            by_color.setdefault(col, []).append((r, c))
    blue_best = _best_straight_pair_len(by_color.get(_LINE_COLOR, []))
    if blue_best < 0:
        return False
    for col, pts in by_color.items():
        if col == _LINE_COLOR:
            continue
        if _best_straight_pair_len(pts) >= blue_best:
            return False
    return True


def _try_place_segment(
    orient: str,
    seg_len: int,
    H: int,
    W: int,
    occupied: set[tuple[int, int]],
    drawn: set[tuple[int, int]],
    line_positions: list[tuple[int, int]],
):
    for _ in range(_SEGMENT_PLACE_RETRIES):
        if orient == "h":
            if W - seg_len - 3 < 0:
                return None
            r0 = random.randint(0, H - 1)
            c0 = random.randint(1, W - seg_len - 2)
            p = (r0, c0)
            q = (r0, c0 + seg_len)
        else:
            if H - seg_len - 3 < 0:
                return None
            r0 = random.randint(1, H - seg_len - 2)
            c0 = random.randint(0, W - 1)
            p = (r0, c0)
            q = (r0 + seg_len, c0)

        path = _straight_path(p, q)
        if any(rc in occupied or rc in drawn for rc in path):
            continue

        ep0, ep1 = path[0], path[-1]
        if any(_shares_axis_or_diagonal(ep0, lp) or _shares_axis_or_diagonal(ep1, lp) for lp in line_positions):
            continue

        neg_side, pos_side = _visible_side_buckets(path)
        if not neg_side or not pos_side:
            continue

        return path
    return None


def _choose_trigger_specs(path: list[tuple[int, int]]):
    neg_side, pos_side = _visible_side_buckets(path)
    sides = [neg_side, pos_side]
    n_triggers = random.randint(_TRIGGERS_PER_SEG[0], _TRIGGERS_PER_SEG[1])
    if n_triggers == 2:
        picks = [random.choice(neg_side), random.choice(pos_side)]
    else:
        chosen_side = random.choice(sides)
        picks = [random.choice(chosen_side)]
    specs = [(pos, _single_parallel_dir(pos, path[0], path[-1])) for pos in picks]
    return specs


def _gen_attempt() -> np.ndarray | None:
    H = random.randint(*_H_RANGE)
    W = random.randint(*_W_RANGE)
    bg = random.choice([c for c in range(10) if c not in (_LINE_COLOR, _TRIGGER_COLOR)])

    max_seg_len = min(_SEG_LEN_RANGE[1], max(H - 3, 0), max(W - 3, 0))
    if max_seg_len < _SEG_LEN_RANGE[0]:
        return None

    n_segments = random.randint(*_SEGMENT_COUNT)
    occupied: set[tuple[int, int]] = set()
    drawn: set[tuple[int, int]] = set()
    line_positions: list[tuple[int, int]] = []
    trigger_positions: list[tuple[int, int]] = []
    segments = []
    visible_dirs: set[tuple[int, int]] = set()
    total_extra_cells = 0
    has_single_trigger_segment = False

    for _ in range(n_segments):
        orient = random.choice(["h", "v"])
        seg_len = random.randint(_SEG_LEN_RANGE[0], max_seg_len)

        path = _try_place_segment(orient, seg_len, H, W, occupied, drawn, line_positions)
        if path is None:
            continue

        trigger_specs = _choose_trigger_specs(path)
        path_cells = set(path)
        ray_cells = _compute_single_ray_cells(trigger_specs, H, W)
        extra_ray_cells = ray_cells - path_cells
        if not extra_ray_cells:
            continue
        if any(rc in occupied or rc in drawn for rc in extra_ray_cells):
            continue

        ep0, ep1 = path[0], path[-1]
        segments.append((ep0, ep1, trigger_specs, path))
        occupied.update([ep0, ep1])
        occupied.update(pos for pos, _ in trigger_specs)
        drawn.update(set(path[1:-1]) - {pos for pos, _ in trigger_specs})
        drawn.update(extra_ray_cells)
        line_positions.extend([ep0, ep1])
        trigger_positions.extend(pos for pos, _ in trigger_specs)
        visible_dirs.update(direction for _, direction in trigger_specs)
        total_extra_cells += len(extra_ray_cells)
        has_single_trigger_segment = has_single_trigger_segment or len(trigger_specs) == 1

    if len(segments) < 2 or len(visible_dirs) < 2 or total_extra_cells == 0 or not has_single_trigger_segment:
        return None

    inp = np.full((H, W), bg, dtype=int)
    for ep0, ep1, trigger_specs, _ in segments:
        inp[ep0] = _LINE_COLOR
        inp[ep1] = _LINE_COLOR
        for pos, _ in trigger_specs:
            inp[pos] = _TRIGGER_COLOR

    all_line_positions = list(line_positions)
    for _ in range(random.randint(*_UNPAIRED_LINE_PIXELS)):
        for _ in range(_DISTRACTOR_RETRIES):
            pos = (random.randint(0, H - 1), random.randint(0, W - 1))
            if pos in occupied or pos in drawn:
                continue
            if any(_shares_axis_or_diagonal(pos, lp) for lp in all_line_positions):
                continue
            inp[pos] = _LINE_COLOR
            occupied.add(pos)
            all_line_positions.append(pos)
            break

    all_trigger_positions = list(trigger_positions)
    for _ in range(random.randint(*_OFF_SEG_TRIGGERS)):
        for _ in range(_DISTRACTOR_RETRIES):
            pos = (random.randint(0, H - 1), random.randint(0, W - 1))
            if pos in occupied or pos in drawn:
                continue
            if any(pos[0] == tr or pos[1] == tc for tr, tc in all_trigger_positions):
                continue
            inp[pos] = _TRIGGER_COLOR
            occupied.add(pos)
            all_trigger_positions.append(pos)
            break

    if not _blue_is_unique_best(inp, bg):
        return None

    expected = _build_expected(inp, bg, segments)
    try:
        result = main(inp)
    except Exception:
        return None

    if np.array_equal(result, _baseline_output_from_input(inp, "parallel2")):
        return None
    if np.array_equal(result, _baseline_output_from_input(inp, "perp2")):
        return None

    if np.array_equal(result, expected):
        return inp
    return None


def generate_input() -> np.ndarray:
    for _ in range(_GEN_RETRIES):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError(f"Failed to generate valid input after {_GEN_RETRIES} attempts")
