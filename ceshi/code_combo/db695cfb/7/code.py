# Combo variant of db695cfb (index 7)
# combo_spec: D7/0+D17/0
# Name: 多色触发+图例区域

import random

import numpy as np

from arc_utils import get_bg


_LINE_COLOR = 1
_LEGEND_BG = 0
_LEGEND_SYMBOL = 5
_LEGEND_W = 4

_RULE_DIAG = "diag_perp"
_RULE_ORTH = "orth_cross"
_ALL_RULES = (_RULE_DIAG, _RULE_ORTH)
_TRIGGER_COLOR_POOL = (2, 3, 6, 8)

# ---------------------------------------------------------------------------
# Design analysis for the combo variant
#
# concrete_trace:
#   fused_scene:
#     - 左侧是一条宽 4 的图例带，每个 3x3 图例块用“中心彩色像素 + 灰色符号”
#       指示一种触发色的射线规则。
#     - 右侧主区域保留原题结构：蓝色端点成对落在 45 度对角线上；这些对角线内部
#       可能出现多种颜色的触发像素。
#   example_walkthrough:
#     - 假设某个样本的图例显示：绿色中心配 X 角符号，红色中心配十字符号。
#     - 原题会把路径上的所有非蓝像素都视为“斜向垂直对角触发器”。
#     - 融合后要先读图例：绿色表示“沿垂直对角方向发射”，红色表示“沿上下左右
#       四个正交方向发射”。
#     - 当蓝色对角线补齐时，路径上若碰到绿色像素，就长出一对斜对角射线；若碰到
#       红色像素，就长出十字形正交射线。
#   difference:
#     - D7 贡献了“同一路径上的不同颜色触发不同几何行为”。
#     - D17 贡献了“颜色到行为的映射不再固定写死，而是由边缘图例在当前样本内定义”。
#
# complexity_calibration:
#   - 只保留两类规则：垂直对角射线、正交十字射线。这样视觉差异足够明显，同时
#     不会引入过多档位导致 2-4 个样例无法归纳。
#   - 每个样本都强制至少两个真实生效的触发色，并且两种规则都真的出现在蓝线
#     路径上，避免图例或多色机制沦为摆设。
#   - 触发色从小型颜色池里抽样，且图例中的颜色-规则对应会在样本间随机置换；
#     这使“先读图例”成为必要步骤，而不是退化成固定颜色记忆题。
#
# implicit_assumption_review:
#   - 原 main() 会扫描全图挑选“最长同色对角对”作为线色；图例引入后，这种做法
#     会让图例色误参与匹配。这里改为直接只在主区域寻找固定蓝色 1。
#   - 原 draw_segment() 见到路径上的任意非蓝像素都用同一套垂直对角规则；融合后
#     必须先通过图例把颜色解码成规则，再按规则分配方向集合。
#   - 原射线可以画到整张网格；现在左侧图例带是说明区，不允许被主任务输出覆盖，
#     因此发射时只填充主区域列 `c >= 4` 的背景格。
#   - 新 solver 的额外假设是：图例块形状合法、图例至少定义两种触发色、所有蓝色
#     端点都在主区域内。生成器会显式保证这些前提。
#
# placement_legality:
#   - 图例带完全保留给图例，不放主区域对象。
#   - 每条蓝色对角线都必须整体落在主区域内；其内部补线格与所有射线格都不能与
#     已有输入像素或先前预留的输出格重叠。
#   - 为防 solver 把不同线段误合并，任何新蓝色端点都不能与既有蓝色像素形成额外
#     的 45 度同线配对；无配对蓝色干扰点也遵守这个约束。
#
# visual_observability:
#   - X 角图例和十字图例在 3x3 上形状差异很大，且输出几何分别是斜向与正交，
#     人类可以肉眼把图例符号和结果对应起来。
#   - 同一输入里至少两种触发色都实际生效，因此解题者能同时观察“颜色区分”和
#     “图例查表”两个维度，而不是靠跨样例猜固定配色。
#
# visual_cleanliness:
#   - 生成器拒绝射线与射线、射线与待补蓝线的重叠，避免输出中心堆成不可读的网。
#   - 图例带与主区域严格分栏，少量离线路径触发色只作为静态干扰项存在，不会压过
#     主结构。
#
# generator_contract:
#   - `main()` 依赖的关键前提是：图例带宽固定为 4；图例块按 3 行堆叠；至少两种
#     图例色都在真实蓝线路径上出现；所有可能生效的触发色都来自图例。
#   - `generate_input()` 通过固定图例模板、首条线段强制双触发、以及输出优先的
#     无碰撞采样来满足这些假设。
# ---------------------------------------------------------------------------


def _diag_path(p: tuple[int, int], q: tuple[int, int]) -> list[tuple[int, int]]:
    r0, c0 = p
    r1, c1 = q
    dr = 1 if r1 > r0 else -1
    dc = 1 if c1 > c0 else -1
    steps = abs(r1 - r0)
    return [(r0 + k * dr, c0 + k * dc) for k in range(steps + 1)]


def _dirs_for_mode(mode: str, dr: int, dc: int) -> list[tuple[int, int]]:
    if mode == _RULE_DIAG:
        return [(dr, -dc), (-dr, dc)]
    if mode == _RULE_ORTH:
        return [(-1, 0), (1, 0), (0, -1), (0, 1)]
    raise AssertionError(f"unknown legend mode: {mode}")


def _paint_legend(grid: np.ndarray, rule_colors: list[int], rule_modes: list[str]) -> None:
    grid[:, :_LEGEND_W] = _LEGEND_BG
    for i, (color, mode) in enumerate(zip(rule_colors, rule_modes)):
        r0 = 3 * i
        grid[r0 + 1, 1] = int(color)
        if mode == _RULE_DIAG:
            grid[r0, 0] = _LEGEND_SYMBOL
            grid[r0, 2] = _LEGEND_SYMBOL
            grid[r0 + 2, 0] = _LEGEND_SYMBOL
            grid[r0 + 2, 2] = _LEGEND_SYMBOL
        elif mode == _RULE_ORTH:
            grid[r0, 1] = _LEGEND_SYMBOL
            grid[r0 + 1, 0] = _LEGEND_SYMBOL
            grid[r0 + 1, 2] = _LEGEND_SYMBOL
            grid[r0 + 2, 1] = _LEGEND_SYMBOL
        else:
            raise AssertionError(f"unknown legend mode: {mode}")


def _read_legend(grid: np.ndarray, bg: int) -> dict[int, str]:
    h, w = grid.shape
    assert w > _LEGEND_W, "grid must leave room for a main field"

    rules: dict[int, str] = {}
    for r0 in range(0, h - 2, 3):
        center = int(grid[r0 + 1, 1])
        if center in (bg, _LEGEND_BG, _LEGEND_SYMBOL, _LINE_COLOR):
            continue

        left = int(grid[r0 + 1, 0]) == _LEGEND_SYMBOL
        right = int(grid[r0 + 1, 2]) == _LEGEND_SYMBOL
        up = int(grid[r0, 1]) == _LEGEND_SYMBOL
        down = int(grid[r0 + 2, 1]) == _LEGEND_SYMBOL
        corners = (
            int(grid[r0, 0]) == _LEGEND_SYMBOL
            and int(grid[r0, 2]) == _LEGEND_SYMBOL
            and int(grid[r0 + 2, 0]) == _LEGEND_SYMBOL
            and int(grid[r0 + 2, 2]) == _LEGEND_SYMBOL
        )

        if corners and not (left or right or up or down):
            rules[center] = _RULE_DIAG
        elif left and right and up and down and not corners:
            rules[center] = _RULE_ORTH
        else:
            raise AssertionError("legend block is malformed")

    assert len(rules) >= 2, "legend must define at least two trigger rules"
    return rules


def main(grid: np.ndarray) -> np.ndarray:
    """
    Connect blue diagonal endpoint pairs, then extend each on-path trigger by the geometry encoded for its color in the legend strip.

    Input structure:
    - A narrow legend strip occupies the left edge. Each 3x3 legend glyph has one trigger color at its center plus a gray shape that specifies that color's ray behavior.
    - In the main field to the right, blue pixels still appear as endpoint pairs on 45-degree diagonals, and multiple trigger colors may lie strictly inside those diagonal routes.

    Rule:
    - Decode the legend to map each trigger color to either perpendicular-diagonal emission or orthogonal cross emission.
    - Fill every blue diagonal segment between its extreme endpoints; whenever that filled path passes through a trigger color listed in the legend, emit rays of that trigger's color in the legend-prescribed directions until the grid boundary, without painting over the legend strip or other non-background pixels.

    Key insight:
    - The hard part is now a two-stage lookup: color alone is insufficient, and being on the blue path alone is insufficient; the solver must combine the path test with the legend's per-color instruction table.
    """

    grid = np.asarray(grid, dtype=int)
    h, w = grid.shape
    out = grid.copy()
    bg = get_bg(grid)

    legend_rules = _read_legend(grid, bg)

    pts = [(r, c) for r in range(h) for c in range(_LEGEND_W, w) if int(grid[r, c]) == _LINE_COLOR]
    assert len(pts) >= 2, "at least one blue diagonal pair must exist"

    groups_a: dict[int, list[tuple[int, int]]] = {}
    groups_b: dict[int, list[tuple[int, int]]] = {}
    for r, c in pts:
        groups_a.setdefault(r - c, []).append((r, c))
        groups_b.setdefault(r + c, []).append((r, c))

    triggers: dict[tuple[int, int, int], set[tuple[int, int]]] = {}

    def draw_segment(p: tuple[int, int], q: tuple[int, int]) -> None:
        path = _diag_path(p, q)
        dr = 1 if q[0] > p[0] else -1
        dc = 1 if q[1] > p[1] else -1
        for r, c in path:
            v = int(out[r, c])
            if v == bg:
                out[r, c] = _LINE_COLOR
            elif c >= _LEGEND_W and v != _LINE_COLOR and v in legend_rules:
                mode = legend_rules[v]
                triggers.setdefault((r, c, v), set()).update(_dirs_for_mode(mode, dr, dc))

    for group in groups_a.values():
        if len(group) >= 2:
            group = sorted(group)
            draw_segment(group[0], group[-1])

    for group in groups_b.values():
        if len(group) >= 2:
            group = sorted(group)
            draw_segment(group[0], group[-1])

    for (r0, c0, tcol), dirs in triggers.items():
        for ddr, ddc in dirs:
            r, c = r0 + ddr, c0 + ddc
            while 0 <= r < h and 0 <= c < w:
                if c >= _LEGEND_W and int(out[r, c]) == bg:
                    out[r, c] = tcol
                r += ddr
                c += ddc

    return out


# --------------- generator (c) ---------------

_H_RANGE = (10, 25)
_W_RANGE = (14, 21)
_SEGMENT_COUNT = (1, 4)
_SEG_LEN_RANGE = (3, 16)
_EXTRA_TRIGGERS_PER_SEG = (0, 2)
_RULE_COUNT = 2

_UNPAIRED_LINE_PIXELS = (0, 2)
_OFF_DIAG_TRIGGERS = (0, 4)

_SEGMENT_PLACE_RETRIES = 90
_DISTRACTOR_RETRIES = 60
_GEN_RETRIES = 400


def _has_diagonal_alignment(point: tuple[int, int], others: set[tuple[int, int]]) -> bool:
    r, c = point
    return any(abs(r - rr) == abs(c - cc) for rr, cc in others)


def _compute_ray_cells(
    trigger_specs: list[tuple[int, int, int, str]],
    dc: int,
    H: int,
    W: int,
) -> set[tuple[int, int]]:
    cells: set[tuple[int, int]] = set()
    for tr, tc, _, mode in trigger_specs:
        for ddr, ddc in _dirs_for_mode(mode, 1, dc):
            r, c = tr + ddr, tc + ddc
            while 0 <= r < H and 0 <= c < W:
                if c >= _LEGEND_W:
                    cells.add((r, c))
                r += ddr
                c += ddc
    return cells


def _build_expected(
    inp: np.ndarray,
    bg: int,
    segments: list[
        tuple[
            tuple[int, int],
            tuple[int, int],
            list[tuple[int, int, int, str]],
            list[tuple[int, int]],
            int,
        ]
    ],
) -> np.ndarray:
    H, W = inp.shape
    expected = inp.copy()
    for _, _, trigger_specs, path, dc in segments:
        for r, c in path:
            if expected[r, c] == bg:
                expected[r, c] = _LINE_COLOR
        for tr, tc, tcol, mode in trigger_specs:
            for ddr, ddc in _dirs_for_mode(mode, 1, dc):
                r, c = tr + ddr, tc + ddc
                while 0 <= r < H and 0 <= c < W:
                    if c >= _LEGEND_W and expected[r, c] == bg:
                        expected[r, c] = tcol
                    r += ddr
                    c += ddc
    return expected


def _try_place_segment(
    slope_type: str,
    seg_len: int,
    H: int,
    W: int,
    occupied: set[tuple[int, int]],
    drawn: set[tuple[int, int]],
    line_points: set[tuple[int, int]],
) -> tuple[list[tuple[int, int]], int] | None:
    for _ in range(_SEGMENT_PLACE_RETRIES):
        if slope_type == "backslash":
            if H - 1 - seg_len < 0 or W - 1 - seg_len < _LEGEND_W:
                return None
            r0 = random.randint(0, H - 1 - seg_len)
            c0 = random.randint(_LEGEND_W, W - 1 - seg_len)
            dc = 1
        else:
            if H - 1 - seg_len < 0 or _LEGEND_W + seg_len > W - 1:
                return None
            r0 = random.randint(0, H - 1 - seg_len)
            c0 = random.randint(_LEGEND_W + seg_len, W - 1)
            dc = -1

        path = [(r0 + k, c0 + k * dc) for k in range(seg_len + 1)]
        ep0, ep1 = path[0], path[-1]
        if any(c < _LEGEND_W for _, c in path):
            continue
        if any(rc in occupied or rc in drawn for rc in path):
            continue
        if _has_diagonal_alignment(ep0, line_points) or _has_diagonal_alignment(ep1, line_points):
            continue
        return path, dc
    return None


def _choose_trigger_specs(
    path_interior: list[tuple[int, int]],
    rule_colors: list[int],
    legend_rules: dict[int, str],
    require_both: bool,
) -> list[tuple[int, int, int, str]] | None:
    if require_both:
        if len(path_interior) < 2:
            return None
        n_triggers = random.randint(2, min(3, len(path_interior)))
        positions = random.sample(path_interior, n_triggers)
        colors = list(rule_colors)
        while len(colors) < n_triggers:
            colors.append(random.choice(rule_colors))
        random.shuffle(colors)
        return [(r, c, color, legend_rules[color]) for (r, c), color in zip(positions, colors)]

    max_triggers = min(_EXTRA_TRIGGERS_PER_SEG[1], len(path_interior))
    if max_triggers == 0:
        return []
    n_triggers = random.randint(_EXTRA_TRIGGERS_PER_SEG[0], max_triggers)
    if n_triggers == 0:
        return []
    positions = random.sample(path_interior, n_triggers)
    return [(r, c, color, legend_rules[color]) for (r, c), color in zip(positions, random.choices(rule_colors, k=n_triggers))]


def _gen_attempt() -> np.ndarray | None:
    H = random.randint(*_H_RANGE)
    W = random.randint(*_W_RANGE)

    if H < 3 * _RULE_COUNT or W <= _LEGEND_W + _SEG_LEN_RANGE[0]:
        return None

    bg_choices = [c for c in range(10) if c not in (_LINE_COLOR, _LEGEND_BG, _LEGEND_SYMBOL)]
    bg = random.choice(bg_choices)

    rule_colors = random.sample([c for c in _TRIGGER_COLOR_POOL if c != bg], _RULE_COUNT)
    rule_modes = list(_ALL_RULES)
    random.shuffle(rule_modes)
    legend_rules = dict(zip(rule_colors, rule_modes))

    inp = np.full((H, W), bg, dtype=int)
    _paint_legend(inp, rule_colors, rule_modes)

    n_segments = random.randint(*_SEGMENT_COUNT)
    max_seg_len = min(H - 1, W - _LEGEND_W - 1, _SEG_LEN_RANGE[1])
    if max_seg_len < _SEG_LEN_RANGE[0]:
        return None

    occupied: set[tuple[int, int]] = set()
    drawn: set[tuple[int, int]] = set()
    line_points: set[tuple[int, int]] = set()
    segments: list[
        tuple[
            tuple[int, int],
            tuple[int, int],
            list[tuple[int, int, int, str]],
            list[tuple[int, int]],
            int,
        ]
    ] = []

    for _ in range(n_segments):
        slope_type = random.choice(["backslash", "slash"])
        seg_len = random.randint(_SEG_LEN_RANGE[0], min(max_seg_len, _SEG_LEN_RANGE[1]))
        result = _try_place_segment(slope_type, seg_len, H, W, occupied, drawn, line_points)
        if result is None:
            continue

        path, dc = result
        path_set = set(path)
        path_interior = path[1:-1]
        trigger_specs = _choose_trigger_specs(path_interior, rule_colors, legend_rules, require_both=len(segments) == 0)
        if trigger_specs is None:
            continue

        ray_cells = _compute_ray_cells(trigger_specs, dc, H, W)
        if any(rc in occupied or rc in drawn or rc in path_set for rc in ray_cells):
            continue

        ep0, ep1 = path[0], path[-1]
        trigger_positions = {(tr, tc) for tr, tc, _, _ in trigger_specs}
        segments.append((ep0, ep1, trigger_specs, path, dc))
        occupied.update([ep0, ep1])
        occupied.update(trigger_positions)
        line_points.update([ep0, ep1])
        drawn.update(set(path_interior) - trigger_positions)
        drawn.update(ray_cells)

    if not segments:
        return None

    live_modes = {mode for _, _, trigger_specs, _, _ in segments for _, _, _, mode in trigger_specs}
    if live_modes != set(_ALL_RULES):
        return None

    for ep0, ep1, trigger_specs, _, _ in segments:
        inp[ep0] = _LINE_COLOR
        inp[ep1] = _LINE_COLOR
        for tr, tc, color, _ in trigger_specs:
            inp[tr, tc] = int(color)

    for _ in range(random.randint(*_UNPAIRED_LINE_PIXELS)):
        for _ in range(_DISTRACTOR_RETRIES):
            r = random.randint(0, H - 1)
            c = random.randint(_LEGEND_W, W - 1)
            candidate = (r, c)
            if candidate in occupied or candidate in drawn:
                continue
            if _has_diagonal_alignment(candidate, line_points):
                continue
            inp[candidate] = _LINE_COLOR
            occupied.add(candidate)
            line_points.add(candidate)
            break

    for _ in range(random.randint(*_OFF_DIAG_TRIGGERS)):
        for _ in range(_DISTRACTOR_RETRIES):
            r = random.randint(0, H - 1)
            c = random.randint(_LEGEND_W, W - 1)
            candidate = (r, c)
            if candidate in occupied or candidate in drawn:
                continue
            inp[candidate] = random.choice(rule_colors)
            occupied.add(candidate)
            break

    try:
        result = main(inp)
    except Exception:
        return None

    expected = _build_expected(inp, bg, segments)
    if np.array_equal(result, expected):
        return inp
    return None


def generate_input() -> np.ndarray:
    for _ in range(_GEN_RETRIES):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError(f"Failed to generate valid input after {_GEN_RETRIES} attempts")
