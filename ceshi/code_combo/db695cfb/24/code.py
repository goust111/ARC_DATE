# Combo variant of db695cfb (index 24)
# combo_spec: D4/3+D15/1
# Name: 棋盘背景+不可见主路径

from collections import Counter
import random

import numpy as np


"""
design_analysis:
  fused_summary:
    original_task: 同色端点确定 45 度对角段；沿段遇到触发色像素时，向垂直对角方向发射双向射线；主段本身会被连续涂色。
    fused_change:
      - D4/3: 主段几何路径仍然存在，但不再把中间背景格涂成蓝色，只保留蓝端点和路径上的品红触发点可见。
      - D15/1: 背景改成双色棋盘，只有其中一种背景色属于“可绘制区”；品红射线传播时会穿过另一种背景色但不落笔。
    integration_goal: 解题者既要先从端点和触发点反推一条不可见主对角线，又要从背景纹理判断哪些格子能承接品红射线。

  concrete_form_and_trace:
    example_setup:
      - 背景是 2x2 宏格组成的棋盘，例如 AABB / AABB / BBAA / BBAA / ...
      - 可绘制背景色为 A
      - 蓝端点在 (1,1) 和 (5,5)，中间的品红触发点在 (3,3)
      - 该主段是反斜杠方向，因此品红射线沿正斜杠方向延伸
    original_behavior:
      - 原题会把 (1,1) 到 (5,5) 的整条对角线都补蓝
      - 再从 (3,3) 沿两条垂直对角线连续补品红到边界
    fused_behavior:
      - 新题不补蓝色主段，中间格继续保持棋盘背景，只能从端点与触发点推断隐藏路径
      - 品红射线仍按原几何路线传播，但只有落在 A 背景格上时才着色；落到 B 背景格时保持背景，因此形成规则断续
    visible_difference:
      - 原题输出含有明显蓝色实线；融合后蓝色只剩孤立端点
      - 原题品红是实线；融合后品红沿同一几何射线呈跳格节奏

  complexity_calibration:
    non_degenerate_checkerboard:
      - 标准 1x1 棋盘在任何对角步进下都会停留在同一种格色上，D15 的“只在一种底色上绘制”会退化成不可见约束。
      - 为了保留“棋盘背景”外观且让射线真的出现间断，这里采用 2x2 宏格棋盘，并且只生成反斜杠主段，使垂直射线沿正斜杠方向穿越不同宏格。
      - 进一步约束端点/触发点落在同一相位类上，这样每一步射线都会在“可绘制 / 不可绘制”之间交替，断续现象稳定可见。
    controlled_freedom:
      - 背景始终只有两种颜色，且可绘制区只有一种，避免颜色映射过多
      - 主段统一用一种方向，减少隐藏路径与背景约束叠加后的歧义
      - 每段固定至少一个触发器，保证输出确实体现融合特征

  implicit_assumption_audit:
    original_main_assumptions_broken:
      - "get_bg(grid)" 假设背景只有一种颜色；现在必须恢复两种主背景色和它们的棋盘布局。
      - "if v == bg: out[r, c] = line_color" 假设主路径上所有背景格都应补蓝；D4/3 下这一步必须完全删除。
      - "elif v != line_color" 原来把一切非蓝色路径像素都当触发器；现在要排除两种背景色，避免把另一种背景误判成触发器。
      - ray loop 中 "if out[r, c] == bg" 假设所有背景都可绘制；现在只能在推断出的 active_bg 上着色。
    new_solver_assumptions:
      - 输入中频数最高的两种颜色确实是棋盘背景色。
      - 背景遵循固定的 2x2 棋盘块布局，可从可见背景格高分匹配恢复。
      - 真正功能性的非背景像素都放在 active_bg 上，因此可用它们的位置反推出哪种背景是可绘制区。
      - 线色的有效成对端点不会与其他同色像素形成意外更长的对角配对。
    generator_guarantees:
      - 背景总面积远大于前景，确保“前两大频数颜色”稳定就是背景
      - 所有端点、触发器与干扰点都只放在 active_bg 上
      - 任何隐藏主段都不穿过其他输入像素，避免制造非预期触发器
      - 最长对角同色对严格属于蓝端点颜色，防止 solver 误选线色

  placement_legality:
    segments:
      - 所有真实主段都是反斜杠方向，并完整落在网格内
      - 端点之间的整条几何路径被预留，不允许与其他段路径、可见射线或干扰像素重叠
      - 触发器只放在段内部，并筛到能产生可见品红射线的位置
    collisions:
      - visible ray cells 不能覆盖任何输入像素
      - hidden path cells 虽不着色，但也不能穿过别的对象，否则会在 solver 中被读成额外触发器
    oracle:
      - 生成器继续采用 output-first / solver-checked 流程，用 main(inp) 与显式期望输出比对

  visual_observability:
    human_cues:
      - 输入先给出两色棋盘背景，使“背景类别”本身可见
      - 蓝端点和品红触发器排成一条 45 度隐藏线，提示存在不可见主路径
      - 输出新增的品红像素只落在其中一种棋盘底色上，并呈规律跳格
    why_two_to_four_examples_are_enough:
      - 多个样例会重复出现“蓝线不补、只剩端点”和“品红只在某类棋盘格上续写”的稳定模式
      - 背景实际颜色 id 每题随机，因此规律必须从结构读出，而不是从数字读出

  cleanliness_and_inference:
    cleanliness:
      - 减少段数和干扰点数量，避免隐藏路径与多束射线叠得过密
      - 主段不可见反而降低了蓝/品红覆盖层数，视觉上比把所有元素都画出来更干净
    inference_risk_control:
      - 不引入额外颜色角色；仍然只有线色、触发色、两种背景色
      - 不依赖颜色编号排序；active_bg 由功能像素所处的棋盘格类型决定
      - 用固定方向主段来换取稳定的射线断续模式，避免有些样例看得出、有些样例看不出的退化情况
"""


def _diag_path(p: tuple[int, int], q: tuple[int, int]) -> list[tuple[int, int]]:
    r0, c0 = p
    r1, c1 = q
    dr = 1 if r1 > r0 else -1
    dc = 1 if c1 > c0 else -1
    steps = abs(r1 - r0)
    return [(r0 + k * dr, c0 + k * dc) for k in range(steps + 1)]


def _top_two_colors(grid: np.ndarray) -> tuple[int, int]:
    counts = Counter(map(int, grid.flatten()))
    top = counts.most_common(2)
    assert len(top) == 2, "grid must contain two dominant background colors"
    return int(top[0][0]), int(top[1][0])


def _checker_color(
    r: int,
    c: int,
    bg_pair: tuple[int, int],
    row_phase: int,
    col_phase: int,
) -> int:
    idx = (((r + row_phase) // 2) + ((c + col_phase) // 2)) % 2
    return bg_pair[idx]


def _infer_checker_layout(
    grid: np.ndarray,
    bg0: int,
    bg1: int,
) -> tuple[tuple[int, int], int, int]:
    bg_set = {bg0, bg1}
    best = None  # (score, bg_pair, row_phase, col_phase)
    for bg_pair in ((bg0, bg1), (bg1, bg0)):
        for row_phase in (0, 1):
            for col_phase in (0, 1):
                score = 0
                for r in range(grid.shape[0]):
                    for c in range(grid.shape[1]):
                        v = int(grid[r, c])
                        if v in bg_set and v == _checker_color(r, c, bg_pair, row_phase, col_phase):
                            score += 1
                cand = (score, bg_pair, row_phase, col_phase)
                if best is None or cand > best:
                    best = cand
    assert best is not None
    _, bg_pair, row_phase, col_phase = best
    return bg_pair, row_phase, col_phase


def _best_diagonal_color(non_bg: list[tuple[int, int, int]]) -> tuple[int, int]:
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
                dr = r1 - r0
                dc = c1 - c0
                if dr == 0 or abs(dr) != abs(dc):
                    continue
                cand = (abs(dr), col)
                if best is None or cand > best:
                    best = cand

    assert best is not None, "at least one diagonal-aligned same-color pair must exist"
    return best[1], best[0]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recover a hidden diagonal connection between matching endpoints, then emit checkerboard-filtered perpendicular rays from trigger pixels without ever painting the main path itself.

    Input structure:
    - The canvas is a two-color 2x2-block checkerboard. One non-background color appears as paired diagonal endpoints, and differently colored trigger pixels may lie on the same hidden diagonal. All functional pixels sit on one of the two checkerboard colors.

    Rule:
    - Find the same-color diagonal pair with the longest span and connect the extremes of every diagonal group of that color geometrically, but keep that connecting path invisible.
    - Whenever that hidden path passes through a differently colored pixel, emit the usual perpendicular diagonal rays in the trigger color, painting only those ray cells whose checkerboard background belongs to the active background color class.

    Key insight:
    - The blue segment is no longer a visible object; it is only an inferred scaffold defined by endpoint alignment. The visible transformation is therefore the magenta ray pattern, whose dashed rhythm is determined jointly by the hidden diagonal and by which checkerboard color is allowed to receive paint.
    """

    grid = np.asarray(grid, dtype=int)
    h, w = grid.shape
    out = grid.copy()

    bg0, bg1 = _top_two_colors(grid)
    bg_colors = {bg0, bg1}
    bg_pair, row_phase, col_phase = _infer_checker_layout(grid, bg0, bg1)

    non_bg = [(r, c, int(grid[r, c])) for r in range(h) for c in range(w) if int(grid[r, c]) not in bg_colors]
    assert non_bg, "grid must contain at least one non-background pixel"

    line_color, _ = _best_diagonal_color(non_bg)
    line_pts = [(r, c) for r, c, col in non_bg if col == line_color]
    active_bg = Counter(
        _checker_color(r, c, bg_pair, row_phase, col_phase) for r, c, _ in non_bg
    ).most_common(1)[0][0]

    triggers: dict[tuple[int, int, int], set[tuple[int, int]]] = {}

    groups_a: dict[int, list[tuple[int, int]]] = {}
    groups_b: dict[int, list[tuple[int, int]]] = {}
    for r, c in line_pts:
        groups_a.setdefault(r - c, []).append((r, c))
        groups_b.setdefault(r + c, []).append((r, c))

    def trace_hidden_segment(p: tuple[int, int], q: tuple[int, int]) -> None:
        path = _diag_path(p, q)
        dr = 1 if q[0] > p[0] else -1
        dc = 1 if q[1] > p[1] else -1
        perp_dirs = [(dr, -dc), (-dr, dc)]
        for r, c in path:
            v = int(grid[r, c])
            if v not in bg_colors and v != line_color:
                key = (r, c, v)
                triggers.setdefault(key, set()).update(perp_dirs)

    for g in groups_a.values():
        if len(g) >= 2:
            g = sorted(g)
            trace_hidden_segment(g[0], g[-1])

    for g in groups_b.values():
        if len(g) >= 2:
            g = sorted(g)
            trace_hidden_segment(g[0], g[-1])

    for (r0, c0, tcol), dirs in triggers.items():
        for ddr, ddc in dirs:
            r, c = r0 + ddr, c0 + ddc
            while 0 <= r < h and 0 <= c < w:
                if (
                    int(out[r, c]) in bg_colors
                    and _checker_color(r, c, bg_pair, row_phase, col_phase) == active_bg
                ):
                    out[r, c] = tcol
                r += ddr
                c += ddc

    return out


# --------------- generator (c) ---------------

# === CFG: canvas and structural bounds ===
_H_RANGE = (10, 24)
_W_RANGE = (10, 22)
_LINE_COLOR = 1
_TRIGGER_COLOR = 6
_SEGMENT_COUNT = (1, 3)
_SEG_LEN_RANGE = (4, 14)
_TRIGGERS_PER_SEG = (1, 1)

# === CFG: distractor counts ===
_UNPAIRED_LINE_PIXELS = (0, 1)
_OFF_DIAG_TRIGGERS = (0, 1)

# === CFG: retry limits ===
_SEGMENT_PLACE_RETRIES = 120
_DISTRACTOR_RETRIES = 80
_GEN_RETRIES = 500


def _make_checkerboard(
    h: int,
    w: int,
    bg_pair: tuple[int, int],
    row_phase: int,
    col_phase: int,
) -> np.ndarray:
    grid = np.zeros((h, w), dtype=int)
    for r in range(h):
        for c in range(w):
            grid[r, c] = _checker_color(r, c, bg_pair, row_phase, col_phase)
    return grid


def _is_active_cell(
    r: int,
    c: int,
    active_bg: int,
    bg_pair: tuple[int, int],
    row_phase: int,
    col_phase: int,
) -> bool:
    return _checker_color(r, c, bg_pair, row_phase, col_phase) == active_bg


def _same_backslash_phase(r: int, c: int, row_phase: int, col_phase: int) -> bool:
    return ((r + row_phase) % 2) == ((c + col_phase) % 2)


def _compute_visible_ray_cells(
    trigger_positions: list[tuple[int, int]],
    perp_dirs: list[tuple[int, int]],
    h: int,
    w: int,
    active_bg: int,
    bg_pair: tuple[int, int],
    row_phase: int,
    col_phase: int,
) -> set[tuple[int, int]]:
    cells = set()
    for tr, tc in trigger_positions:
        for ddr, ddc in perp_dirs:
            r, c = tr + ddr, tc + ddc
            while 0 <= r < h and 0 <= c < w:
                if _is_active_cell(r, c, active_bg, bg_pair, row_phase, col_phase):
                    cells.add((r, c))
                r += ddr
                c += ddc
    return cells


def _visible_ray_cells_from_trigger(
    trigger: tuple[int, int],
    perp_dirs: list[tuple[int, int]],
    h: int,
    w: int,
    active_bg: int,
    bg_pair: tuple[int, int],
    row_phase: int,
    col_phase: int,
) -> set[tuple[int, int]]:
    return _compute_visible_ray_cells(
        [trigger],
        perp_dirs,
        h,
        w,
        active_bg,
        bg_pair,
        row_phase,
        col_phase,
    )


def _build_expected(
    inp: np.ndarray,
    active_bg: int,
    bg_pair: tuple[int, int],
    row_phase: int,
    col_phase: int,
    segments,
) -> np.ndarray:
    h, w = inp.shape
    expected = inp.copy()
    for _, _, trigger_positions, _, perp_dirs in segments:
        for tr, tc in trigger_positions:
            for ddr, ddc in perp_dirs:
                r, c = tr + ddr, tc + ddc
                while 0 <= r < h and 0 <= c < w:
                    if (
                        expected[r, c] in bg_pair
                        and _is_active_cell(r, c, active_bg, bg_pair, row_phase, col_phase)
                    ):
                        expected[r, c] = _TRIGGER_COLOR
                    r += ddr
                    c += ddc
    return expected


def _try_place_segment(
    seg_len: int,
    h: int,
    w: int,
    active_bg: int,
    bg_pair: tuple[int, int],
    row_phase: int,
    col_phase: int,
    line_positions: set[tuple[int, int]],
    path_cells: set[tuple[int, int]],
    drawn: set[tuple[int, int]],
    used_backslash_keys: set[int],
):
    for _ in range(_SEGMENT_PLACE_RETRIES):
        if h - 1 - seg_len < 0 or w - 1 - seg_len < 0:
            return None
        r0 = random.randint(0, h - 1 - seg_len)
        c0 = random.randint(0, w - 1 - seg_len)
        ep0 = (r0, c0)
        ep1 = (r0 + seg_len, c0 + seg_len)
        diag_key = ep0[0] - ep0[1]
        if diag_key in used_backslash_keys:
            continue
        if not _same_backslash_phase(r0, c0, row_phase, col_phase):
            continue
        if not _is_active_cell(r0, c0, active_bg, bg_pair, row_phase, col_phase):
            continue
        path = _diag_path(ep0, ep1)
        if any(not _is_active_cell(r, c, active_bg, bg_pair, row_phase, col_phase) for r, c in path):
            continue
        if any(rc in path_cells or rc in drawn for rc in path):
            continue
        if any(abs(ep0[0] - lr) == abs(ep0[1] - lc) for lr, lc in line_positions):
            continue
        if any(abs(ep1[0] - lr) == abs(ep1[1] - lc) for lr, lc in line_positions):
            continue
        return path
    return None


def _best_color_is_line(inp: np.ndarray, bg_pair: tuple[int, int]) -> bool:
    bg_set = set(bg_pair)
    non_bg = [(r, c, int(inp[r, c])) for r in range(inp.shape[0]) for c in range(inp.shape[1]) if int(inp[r, c]) not in bg_set]
    try:
        color, _ = _best_diagonal_color(non_bg)
    except AssertionError:
        return False
    return color == _LINE_COLOR


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(*_H_RANGE)
    w = random.randint(*_W_RANGE)

    bg_choices = [c for c in range(10) if c not in (_LINE_COLOR, _TRIGGER_COLOR)]
    bg_pair = tuple(random.sample(bg_choices, 2))
    row_phase = random.randint(0, 1)
    col_phase = random.randint(0, 1)
    active_bg = random.choice(bg_pair)

    max_seg_len = min(h - 1, w - 1, _SEG_LEN_RANGE[1])
    if max_seg_len < _SEG_LEN_RANGE[0]:
        return None

    n_segments = random.randint(*_SEGMENT_COUNT)
    path_cells: set[tuple[int, int]] = set()
    occupied: set[tuple[int, int]] = set()
    line_positions: set[tuple[int, int]] = set()
    drawn: set[tuple[int, int]] = set()
    used_backslash_keys: set[int] = set()
    segments = []

    for _ in range(n_segments):
        seg_len = random.randint(_SEG_LEN_RANGE[0], max_seg_len)
        path = _try_place_segment(
            seg_len,
            h,
            w,
            active_bg,
            bg_pair,
            row_phase,
            col_phase,
            line_positions,
            path_cells,
            drawn,
            used_backslash_keys,
        )
        if path is None:
            continue

        ep0, ep1 = path[0], path[-1]
        perp_dirs = [(1, -1), (-1, 1)]
        trigger_candidates = []
        for tr in path[1:-1]:
            visible = _visible_ray_cells_from_trigger(
                tr,
                perp_dirs,
                h,
                w,
                active_bg,
                bg_pair,
                row_phase,
                col_phase,
            )
            if visible and not any(rc in occupied for rc in visible) and not any(rc in drawn for rc in visible):
                trigger_candidates.append((tr, visible))
        if not trigger_candidates:
            continue

        n_triggers = min(random.randint(*_TRIGGERS_PER_SEG), len(trigger_candidates))
        chosen = random.sample(trigger_candidates, n_triggers)
        trigger_positions = [tr for tr, _ in chosen]
        ray_cells = set().union(*(visible for _, visible in chosen))
        if any(rc in occupied for rc in ray_cells) or any(rc in drawn for rc in ray_cells):
            continue

        segments.append((ep0, ep1, trigger_positions, path, perp_dirs))
        path_cells.update(path)
        occupied.update([ep0, ep1])
        occupied.update(trigger_positions)
        line_positions.update([ep0, ep1])
        drawn.update(ray_cells)
        used_backslash_keys.add(ep0[0] - ep0[1])

    if not segments:
        return None

    inp = _make_checkerboard(h, w, bg_pair, row_phase, col_phase)
    for ep0, ep1, trigger_positions, _, _ in segments:
        inp[ep0] = _LINE_COLOR
        inp[ep1] = _LINE_COLOR
        for tp in trigger_positions:
            inp[tp] = _TRIGGER_COLOR

    all_line_positions = list(line_positions)
    for _ in range(random.randint(*_UNPAIRED_LINE_PIXELS)):
        for _ in range(_DISTRACTOR_RETRIES):
            r = random.randint(0, h - 1)
            c = random.randint(0, w - 1)
            if not _is_active_cell(r, c, active_bg, bg_pair, row_phase, col_phase):
                continue
            if (r, c) in occupied or (r, c) in drawn or (r, c) in path_cells:
                continue
            if any(abs(r - lr) == abs(c - lc) for lr, lc in all_line_positions):
                continue
            inp[r, c] = _LINE_COLOR
            occupied.add((r, c))
            line_positions.add((r, c))
            all_line_positions.append((r, c))
            break

    for _ in range(random.randint(*_OFF_DIAG_TRIGGERS)):
        for _ in range(_DISTRACTOR_RETRIES):
            r = random.randint(0, h - 1)
            c = random.randint(0, w - 1)
            if not _is_active_cell(r, c, active_bg, bg_pair, row_phase, col_phase):
                continue
            if (r, c) in occupied or (r, c) in drawn or (r, c) in path_cells:
                continue
            inp[r, c] = _TRIGGER_COLOR
            occupied.add((r, c))
            break

    if not _best_color_is_line(inp, bg_pair):
        return None

    try:
        result = main(inp)
    except Exception:
        return None

    expected = _build_expected(inp, active_bg, bg_pair, row_phase, col_phase, segments)
    if not np.array_equal(result, expected):
        return None

    if np.array_equal(result, inp):
        return None

    return inp


def generate_input() -> np.ndarray:
    for _ in range(_GEN_RETRIES):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError(f"Failed to generate valid input after {_GEN_RETRIES} attempts")
