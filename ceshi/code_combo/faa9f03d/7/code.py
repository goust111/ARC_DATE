# Combo variant of faa9f03d (index 7)
# combo_spec: D3/0+D5/2
# Name: 三种标记颜色 + 交叉处棋盘格交替

from __future__ import annotations

import random

import numpy as np

from arc_utils import components_mask, get_bg


"""
design_analysis:
  a_concretization_and_trace:
    original_scene:
      - 线网由 1 像素宽的正交折线组成；红色单像素补缺口，黄色单像素把一条臂翻转并延伸到边界。
      - 两色逻辑交叉时，最终只显示一种颜色：黄色射线优先，否则按“修复前碎片度更高者胜出”。
    fused_scene:
      - 保留红/黄两类标记，再加入粉色单像素。粉色不只是补线，还会给其所属颜色在一整串交叉里赋予“优先色”身份。
      - 但 D5/2 把“优先色全覆盖交叉”改成了“交叉格按全局棋盘格奇偶交替显示两色”。
      - 具体地，若交叉点坐标满足 `(r + c) % 2 == 0`，则显示该交叉的优先色；否则显示另一条穿过该点的线色。
      - 粉色放在一条直干道上，solver 先把它修回原线色，再沿该直干道收集所有可达交叉；这些交叉共同构成一个棋盘格交替家族。
    concrete_trace:
      - 例：橙线是一条竖直主干，依次与青色蛇形线在 4 个位置相交。原题会让青色因碎片更多而在 4 个交叉点全部可见。
      - 现在若橙线主干上有一个粉色标记，solver 会先修复该粉点，再顺着橙线找到这 4 个交叉。
      - 输出不再是“4 个交叉全变橙色”，而是按坐标奇偶交替：偶格显示橙，奇格显示青，于是同一列交叉形成橙/青/橙/青 的可见节奏。
      - 与原题的区别出现在两个环节：粉色改变了“谁是优先色”，而棋盘格规则又把“优先色”从全覆盖改成了交替显影。
  b_complexity_calibration:
    - 只保留两条逻辑线色，让“交叉处两色交替”在视觉上干净，不被第三条穿越线打乱。
    - 每个样例都强制至少 3 个同家族交叉，且这些交叉沿主干按奇偶交替，避免退化成单交叉或同奇偶重复而看不出棋盘格。
    - 每个样例都至少有 1 个粉色标记；红/黄再以少量模式与之组合，保证三种标记的重复支撑存在但不过载。
  c_implicit_assumption_review_of_main:
    - 原 solver 默认“所有非黄单像素只影响补线，不影响交叉决策”；融合后粉色还会产生交叉家族优先权，因此必须显式追踪粉色目标家族。
    - 原 solver 在 `len(colors_here) > 1` 时直接选单一胜者；融合后只有真正的正交交叉才能套用棋盘格，其它纯填充重叠仍沿用原单色消歧。
    - 原 `marker_brackets` 把所有非背景单像素都可视作补线标记；融合后必须把黄标排除，否则黄标会错误参与 gap-bracket 推断。
    - 原 pink 单变体只要求“唯一交叉”；这里改成“同一直干道上的一组交叉”，于是 `trace` 逻辑必须从收集单点改成收集整族，并假定该族只涉及同一对颜色。
    - 棋盘格决策依赖一个完全可见的空间规则 `(r+c)%2`，不依赖颜色 id；generator 因而必须保证目标交叉族里同时出现两种 parity。
  d_placement_legality:
    - 粉色只放在会经过至少 3 个交叉的直主干上，且离最近交叉正好 2 格，中间保留 1 个同色像素，避免与交叉/端点/转角重叠。
    - 黄色仍只放在端点外侧；其旧臂与新射线都不允许碰到任何交叉族，避免“黄射线全覆盖”与“棋盘格交替”在同一点争优先级。
    - 红色仍放在转角或端点，并与粉/黄以及彼此保持单像素孤立，不越界、不相邻、不覆盖交叉。
  e_visual_observability:
    - 输入中直接能看到三种孤立标记色：红用于普通补线，黄只出现在边界延伸方向附近，粉固定出现在靠近一串交叉的直干道上。
    - 输出中最显眼的新现象是：一整串交叉不再统一显示同一色，而是按网格棋盘格交替显色；粉色所在那条线恰好提供“哪一色是优先色”的局部线索。
    - 由于每个样例都至少有 3 个同家族交叉，人类从 2-4 个样例里能看到重复的交替节奏，而不是把变化误读成噪声。
  f_visual_tidiness:
    - 只保留两条逻辑线色，交叉家族沿一条直主干排开，避免多色多层覆盖造成的拥挤。
    - 粉色家族与黄色射线强制分离；红标也不落在交叉附近，因此同一区域不会同时叠加三种新机制。
    - 交叉列/行至少有 3 个点且间隔 >=2，棋盘格节奏清晰可读。
  g_inferability_of_changes:
    - 颜色变化不是任意映射，而是伴随固定结构线索：粉色总在某条直干道上，且该干道对应的交叉族在输出里呈交替显色。
    - “优先色”不靠颜色数值排序决定，而由粉色的归属线决定；“交替方式”不靠隐藏计数，而由全局棋盘格坐标决定。
  generator_guarantees_for_new_main:
    - 样本始终包含至少 1 个粉色标记，并且它追踪到的交叉家族大小 >= 3，且该家族包含两种 parity。
    - 同一点不会同时受黄射线和粉色家族控制；于是 solver 的优先级顺序 `yellow > pink-family parity > original single-color tie-break` 有构造性支撑。
    - 所有粉色标记都位于直线段上，沿该段可达的交叉只涉及同一对颜色；solver 的 family trace 不会遇到多义目标。
    - 在这些输入上，原始 solver 会把交叉全判给碎片更高的颜色，而融合 solver 会在粉色家族上输出交替图案，因此可通过差异统计验证没有退化回原任务。
"""


DIRS4 = ((-1, 0), (1, 0), (0, -1), (0, 1))

RED = 2
YELLOW = 4
PINK = 6
KNOWN_MARKERS = {RED, YELLOW, PINK}


def _pts(mask: np.ndarray) -> set[tuple[int, int]]:
    return {tuple(map(int, rc)) for rc in np.argwhere(mask)}


def _axis_support(mask: set[tuple[int, int]], r: int, c: int, axis: int) -> bool:
    if axis == 0:
        return (r, c - 1) in mask or (r, c + 1) in mask
    return (r - 1, c) in mask or (r + 1, c) in mask


def _perp_support(mask: set[tuple[int, int]], r: int, c: int, axis: int) -> bool:
    if axis == 0:
        return (r - 1, c) in mask or (r + 1, c) in mask
    return (r, c - 1) in mask or (r, c + 1) in mask


def _component_count(shape: tuple[int, int], pts: set[tuple[int, int]]) -> int:
    mask = np.zeros(shape, dtype=bool)
    for r, c in pts:
        mask[r, c] = True
    return len(components_mask(mask, connectivity=4))


def _neighbors(pt: tuple[int, int], H: int, W: int) -> list[tuple[int, int]]:
    r, c = pt
    return [(r + dr, c + dc) for dr, dc in DIRS4 if 0 <= r + dr < H and 0 <= c + dc < W]


def _segment(a: tuple[int, int], b: tuple[int, int]) -> list[tuple[int, int]]:
    r0, c0 = a
    r1, c1 = b
    if r0 == r1:
        step = 1 if c1 >= c0 else -1
        return [(r0, c) for c in range(c0, c1 + step, step)]
    step = 1 if r1 >= r0 else -1
    return [(r, c0) for r in range(r0, r1 + step, step)]


def _path(points: list[tuple[int, int]]) -> list[tuple[int, int]]:
    cells: list[tuple[int, int]] = []
    for a, b in zip(points, points[1:]):
        seg = _segment(a, b)
        cells.extend(seg[1:] if cells else seg)
    return cells


def _dirs_in(pts: set[tuple[int, int]], pt: tuple[int, int]) -> list[tuple[int, int]]:
    r, c = pt
    return [(dr, dc) for dr, dc in DIRS4 if (r + dr, c + dc) in pts]


def _orientation(pts: set[tuple[int, int]], pt: tuple[int, int]) -> str | None:
    dirs = _dirs_in(pts, pt)
    if len(dirs) != 2 or (dirs[0][0] + dirs[1][0], dirs[0][1] + dirs[1][1]) != (0, 0):
        return None
    return "v" if dirs[0][0] else "h"


def _axis_dirs(mask: set[tuple[int, int]], pt: tuple[int, int]) -> tuple[tuple[int, int], tuple[int, int]] | None:
    r, c = pt
    horiz = (r, c - 1) in mask or (r, c + 1) in mask
    vert = (r - 1, c) in mask or (r + 1, c) in mask
    if horiz and not vert:
        return ((0, -1), (0, 1))
    if vert and not horiz:
        return ((-1, 0), (1, 0))
    return None


def _checker_prefers_priority(pt: tuple[int, int]) -> bool:
    return (pt[0] + pt[1]) % 2 == 0


def _crossing_other(colors_here: list[int], priority: int) -> int:
    assert len(colors_here) == 2 and priority in colors_here
    return colors_here[0] if colors_here[1] == priority else colors_here[1]


def _is_actual_crossing(
    masks: dict[int, set[tuple[int, int]]],
    pt: tuple[int, int],
    colors_here: list[int],
) -> bool:
    if len(colors_here) != 2:
        return False
    orientations = [_orientation(masks[color], pt) for color in colors_here]
    return None not in orientations and sorted(orientations) == ["h", "v"]


def _trace_pink_family(
    marker: tuple[int, int],
    color: int,
    masks: dict[int, set[tuple[int, int]]],
    line_colors: set[int],
    shape: tuple[int, int],
) -> set[tuple[int, int]]:
    H, W = shape
    dirs = _axis_dirs(masks[color], marker)
    assert dirs is not None, f"pink marker at {marker} is not on a straight repaired run"

    family: set[tuple[int, int]] = set()
    other_colors: set[int] = set()
    for dr, dc in dirs:
        r, c = marker[0] + dr, marker[1] + dc
        while 0 <= r < H and 0 <= c < W and (r, c) in masks[color]:
            owners = [other for other in line_colors if (r, c) in masks[other]]
            if len(owners) > 1:
                assert len(owners) == 2 and color in owners, f"pink family at {marker} hit non-binary crossing {owners} at {(r, c)}"
                family.add((r, c))
                other_colors.add(_crossing_other(owners, color))
            nr, nc = r + dr, c + dc
            if not (0 <= nr < H and 0 <= nc < W and (nr, nc) in masks[color]):
                break
            r, c = nr, nc

    assert family, f"pink marker at {marker} does not reach any crossing"
    assert len(other_colors) == 1, f"pink family at {marker} reaches multiple opponent colors: {other_colors}"
    return family


def main(grid: np.ndarray) -> np.ndarray:
    """
    Repair a thin colored line network by recoloring singleton markers, filling rectilinear gaps, redirecting yellow-controlled arms, and rendering pink-prioritized crossings as a checkerboard alternation.

    Input structure:
    - A rectangular grid with a dominant background and 1-cell-thick axis-aligned line fragments.
    - Marker colors are singleton pixels whose 4-connected components all have size 1.
    - Red singletons mark missing line cells, yellow singletons sit just beyond an endpoint to redirect an arm, and pink singletons lie on a straight run near a family of crossings whose priority color must change.

    Rule:
    - Assign every singleton marker to the nearest visible line color in row/column line-of-sight, preferring a seen cell that already has support along that same axis.
    - Treat red and pink markers as restored line pixels, then fill gaps between facing same-color cells in a row or column; process yellow markers by erasing the old opposite arm beyond the pivot endpoint and extending the line through the marker to the boundary.
    - At true orthogonal crossings, first determine a priority color (yellow ray fully wins if present; otherwise pink marks can override the default fragmentation winner), then display the priority color only on even checkerboard parity cells and the other crossing color on odd parity cells.

    Key insight: the singleton symbols still tell you how to reconstruct the hidden line geometry, but the pink symbol no longer selects a single visible crossing cell. Instead it tags one repaired straight trunk, and every crossing reachable along that trunk participates in a global checkerboard alternation where pink's color is the preferred phase.
    """
    g = np.asarray(grid, dtype=int)
    bg = int(get_bg(g))
    H, W = g.shape

    colors = set(map(int, g.ravel())) - {bg}
    assert colors, "expected foreground line and marker colors"

    marker_colors: set[int] = set()
    for color in colors:
        comps = components_mask(g == color, connectivity=4)
        assert comps, "foreground color must have at least one component"
        if max(len(comp) for comp in comps) == 1:
            marker_colors.add(int(color))

    assert marker_colors <= KNOWN_MARKERS, f"unexpected marker colors: {marker_colors}"
    line_colors = colors - marker_colors
    assert line_colors, "expected at least one non-marker line color"
    assert marker_colors, "expected singleton marker colors"

    original = {color: _pts(g == color) for color in line_colors}
    assignment: dict[tuple[int, int], int] = {}
    assign_dir: dict[tuple[int, int], tuple[int, int]] = {}

    marker_mask = np.isin(g, list(marker_colors))
    for r0, c0 in _pts(marker_mask):
        candidates: list[tuple[int, int, int, int, int]] = []
        for dr, dc in DIRS4:
            r, c = r0 + dr, c0 + dc
            dist = 1
            while 0 <= r < H and 0 <= c < W:
                value = int(g[r, c])
                if value in line_colors:
                    axis = 0 if dr == 0 else 1
                    supported = _axis_support(original[value], r, c, axis)
                    candidates.append((dist, 0 if supported else 1, value, dr, dc))
                    break
                if value == bg or value in marker_colors:
                    r += dr
                    c += dc
                    dist += 1
                    continue
                break

        assert candidates, f"marker at {(r0, c0)} cannot see a line color"
        candidates.sort()
        _dist, _unsupported, color, dr, dc = candidates[0]
        assignment[(r0, c0)] = int(color)
        assign_dir[(r0, c0)] = (int(dr), int(dc))

    masks = {color: set(pts) for color, pts in original.items()}
    repair_marker_pts = {color: set() for color in line_colors}
    pink_marker_pts = {color: set() for color in line_colors}
    for pt, color in assignment.items():
        if int(g[pt]) == YELLOW:
            continue
        masks[color].add(pt)
        repair_marker_pts[color].add(pt)
        if int(g[pt]) == PINK:
            pink_marker_pts[color].add(pt)

    comp_count = {color: _component_count(g.shape, masks[color]) for color in line_colors}

    def has_axis_support(color: int, r: int, c: int, axis: int) -> bool:
        return (r, c) in repair_marker_pts[color] or _axis_support(masks[color], r, c, axis)

    def should_fill(color: int, a: tuple[int, int], b: tuple[int, int], axis: int) -> bool:
        r1, c1 = a
        r2, c2 = b
        gap = c2 - c1 - 1 if axis == 0 else r2 - r1 - 1
        if gap <= 0:
            return False

        a_axis = has_axis_support(color, r1, c1, axis)
        b_axis = has_axis_support(color, r2, c2, axis)
        if gap == 1:
            if a_axis or b_axis:
                return True
            a_perp = _perp_support(masks[color], r1, c1, axis)
            b_perp = _perp_support(masks[color], r2, c2, axis)
            return not (a_perp and b_perp)

        if (r1, c1) in repair_marker_pts[color] and (r2, c2) in repair_marker_pts[color]:
            return False
        return a_axis and b_axis

    for color in sorted(line_colors):
        additions: set[tuple[int, int]] = set()
        for r in range(H):
            cols = sorted(c for rr, c in masks[color] if rr == r)
            for c1, c2 in zip(cols, cols[1:]):
                if should_fill(color, (r, c1), (r, c2), axis=0):
                    additions.update((r, c) for c in range(c1 + 1, c2))

        for c in range(W):
            rows = sorted(r for r, cc in masks[color] if cc == c)
            for r1, r2 in zip(rows, rows[1:]):
                if should_fill(color, (r1, c), (r2, c), axis=1):
                    additions.update((r, c) for r in range(r1 + 1, r2))

        masks[color].update(additions)

    control_ray = {color: set() for color in line_colors}
    if YELLOW in marker_colors:
        for control in _pts(g == YELLOW):
            color = assignment[control]
            dr, dc = assign_dir[control]

            r, c = control[0] + dr, control[1] + dc
            pivot: tuple[int, int] | None = None
            while 0 <= r < H and 0 <= c < W:
                if (r, c) in masks[color]:
                    pivot = (r, c)
                    break
                r += dr
                c += dc
            assert pivot is not None, f"yellow control at {control} has no assigned endpoint"

            old_start = (pivot[0] + dr, pivot[1] + dc)
            if old_start in masks[color]:
                stack = [old_start]
                old_arm: set[tuple[int, int]] = set()
                while stack:
                    q = stack.pop()
                    if q == pivot or q in old_arm or q not in masks[color]:
                        continue
                    old_arm.add(q)
                    qr, qc = q
                    for ndr, ndc in DIRS4:
                        stack.append((qr + ndr, qc + ndc))

                assert old_arm, f"yellow control at {control} found no old arm to erase"
                for q in old_arm:
                    masks[color].discard(q)
                    repair_marker_pts[color].discard(q)
                    pink_marker_pts[color].discard(q)
                    control_ray[color].discard(q)

            pr, pc = -dr, -dc
            r, c = pivot
            while 0 <= r < H and 0 <= c < W:
                masks[color].add((r, c))
                control_ray[color].add((r, c))
                r += pr
                c += pc

    claims: dict[tuple[int, int], list[int]] = {}
    for color, pts in masks.items():
        for pt in pts:
            claims.setdefault(pt, []).append(int(color))

    pink_priority: dict[tuple[int, int], int] = {}
    if PINK in marker_colors:
        for marker in _pts(g == PINK):
            color = assignment[marker]
            family = _trace_pink_family(marker, color, masks, line_colors, g.shape)
            parities = {(r + c) % 2 for r, c in family}
            assert len(parities) >= 2, f"pink family at {marker} does not span both checkerboard parities"
            for crossing in family:
                colors_here = claims.get(crossing, [])
                assert _is_actual_crossing(masks, crossing, colors_here), f"pink family hit non-crossing at {crossing}"
                prev = pink_priority.get(crossing)
                assert prev in (None, color), f"conflicting pink priorities at {crossing}: {prev} vs {color}"
                pink_priority[crossing] = color

    out = np.full_like(g, bg)

    def marker_brackets(color: int, pt: tuple[int, int]) -> bool:
        r, c = pt
        same_row = [mc for mr, mc in repair_marker_pts[color] if mr == r]
        if any(mc < c for mc in same_row) and any(mc > c for mc in same_row):
            return True
        same_col = [mr for mr, mc in repair_marker_pts[color] if mc == c]
        return any(mr < r for mr in same_col) and any(mr > r for mr in same_col)

    def default_priority(colors_here: list[int], pt: tuple[int, int]) -> int:
        original_crossing = [color for color in colors_here if int(g[pt]) == color]
        if original_crossing:
            return min(colors_here, key=lambda color: (-comp_count[color], color))
        bracketed = [color for color in colors_here if marker_brackets(color, pt)]
        if bracketed:
            return min(bracketed, key=lambda color: (-comp_count[color], color))
        return min(colors_here)

    for pt, colors_here in claims.items():
        forced = {color for color in colors_here if pt in control_ray[color]}
        if forced:
            assert len(forced) == 1, f"ambiguous yellow-control rays at {pt}: {forced}"
            out[pt] = next(iter(forced))
            continue

        if len(colors_here) == 1:
            out[pt] = colors_here[0]
            continue

        if _is_actual_crossing(masks, pt, colors_here):
            priority = pink_priority.get(pt, default_priority(colors_here, pt))
            other = _crossing_other(colors_here, priority)
            out[pt] = priority if _checker_prefers_priority(pt) else other
            continue

        out[pt] = default_priority(colors_here, pt)

    for color in line_colors:
        for comp in components_mask(out == color, connectivity=4):
            pts_set = set(map(tuple, comp))
            for r, c in pts_set:
                nbrs = [(nr, nc) for nr, nc in [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)] if (nr, nc) in pts_set]
                if len(nbrs) != 1:
                    continue
                on_boundary = r == 0 or r == H - 1 or c == 0 or c == W - 1
                if on_boundary:
                    nr, nc = nbrs[0]
                    dr, dc = nr - r, nc - c
                    if not ((r in (0, H - 1)) and (c in (0, W - 1))):
                        perp = (
                            (r == 0 and dr == 1)
                            or (r == H - 1 and dr == -1)
                            or (c == 0 and dc == 1)
                            or (c == W - 1 and dc == -1)
                        )
                        assert perp, f"boundary endpoint ({r},{c}) color {color} not perpendicular"
                    continue
                adj_other = any(
                    0 <= r + dr < H and 0 <= c + dc < W and int(out[r + dr, c + dc]) not in (bg, color)
                    for dr, dc in DIRS4
                )
                assert adj_other, f"dangling endpoint ({r},{c}) color {color}: not on boundary or crossing"

    color_cells = {c: set() for c in line_colors}
    for r in range(H):
        for c in range(W):
            v = int(out[r, c])
            if v in line_colors:
                color_cells[v].add((r, c))
    crossing_colors = set()
    for r in range(H):
        for c in range(W):
            a = int(out[r, c])
            if a not in line_colors:
                continue
            for b in line_colors - {a}:
                if (((r, c - 1) in color_cells[b] and (r, c + 1) in color_cells[b]) or ((r - 1, c) in color_cells[b] and (r + 1, c) in color_cells[b])):
                    crossing_colors.add(a)
                    crossing_colors.add(b)
    for color in line_colors:
        assert color in crossing_colors, f"color {color} has no crossing with any other color"

    return out


# --------------- generator (combo variant) ---------------

# --- CFG: grid & palette ---
H = W = 12
BG = 0
LINE_PALETTE = [1, 3, 5, 7, 8, 9]

# --- CFG: polyline shape ---
POLYLINE_PIECE_WEIGHTS = [5, 4, 2, 1]
MIN_SPAN_LEN = 4
FULL_SPAN_PROB = 0.65
ARM_MIN_OFFSET = 2

# --- CFG: serpentine & crossings ---
CROSSING_COUNT_OPTIONS = [3, 4]
CROSSING_COUNT_WEIGHTS = [3, 2]
SERPENTINE_MINOR_LO = 6
SERPENTINE_MINOR_HI = 5
CROSS_MINOR_MARGIN = 2
MIN_FAMILY_CROSSINGS = 3
MAX_CROSSINGS = 12

# --- CFG: markers ---
MARKER_MODE_WEIGHTS = [3, 3, 2, 3]  # pink / red+pink / yellow+pink / all
YELLOW_COUNT_RANGE = (1, 1)
RED_COUNT_RANGE = (2, 5)
PINK_COUNT = 1
MIN_CELLS_FOR_RED = 2

# --- CFG: output validation ---
COMPONENTS_PER_COLOR = (2, 4)
TOTAL_COMPONENTS = (4, 6)
FILL_RANGE = (33, 55)

# --- CFG: retry budgets ---
SPACED_VALUES_RETRIES = 80
GEN_RETRIES = 2000


def _is_turn_or_endpoint(pts: set[tuple[int, int]], pt: tuple[int, int]) -> bool:
    dirs = _dirs_in(pts, pt)
    if len(dirs) == 1:
        return True
    return len(dirs) == 2 and (dirs[0][0] + dirs[1][0], dirs[0][1] + dirs[1][1]) != (0, 0)


def _gen_component_count(pts: set[tuple[int, int]]) -> int:
    if not pts:
        return 0
    mask = np.zeros((H, W), dtype=bool)
    for r, c in pts:
        mask[r, c] = True
    return len(components_mask(mask, connectivity=4))


def _on_boundary(r: int, c: int) -> bool:
    return r == 0 or r == H - 1 or c == 0 or c == W - 1


def _valid_path(cells: list[tuple[int, int]]) -> bool:
    pts = set(cells)
    if len(pts) != len(cells) or len(pts) < 2:
        return False
    for r, c in pts:
        if not (0 <= r < H and 0 <= c < W):
            return False
        if len(_dirs_in(pts, (r, c))) > 2:
            return False
    if _gen_component_count(pts) != 1:
        return False
    for r, c in pts:
        if not _on_boundary(r, c):
            continue
        for dr, dc in _dirs_in(pts, (r, c)):
            nr, nc = r + dr, c + dc
            if (r in (0, H - 1) and nr == r) or (c in (0, W - 1) and nc == c):
                return False
        nbrs = _dirs_in(pts, (r, c))
        if len(nbrs) != 1:
            continue
        if (r in (0, H - 1)) and (c in (0, W - 1)):
            continue
        dr, dc = nbrs[0]
        if not ((r == 0 and dr == 1) or (r == H - 1 and dr == -1) or (c == 0 and dc == 1) or (c == W - 1 and dc == -1)):
            return False
    for r, c in pts:
        if len(_dirs_in(pts, (r, c))) == 1 and not _on_boundary(r, c):
            return False
    return True


def _rc(ax: int, major: int, minor: int) -> tuple[int, int]:
    return (major, minor) if ax == 0 else (minor, major)


def _dim(ax: int) -> int:
    return H if ax == 0 else W


def _random_span(limit: int) -> tuple[int, int]:
    if random.random() < 0.5:
        return 0, random.randint(MIN_SPAN_LEN, limit - 1)
    return random.randint(0, limit - MIN_SPAN_LEN - 1), limit - 1


def _build_arm(ax: int, main_coord: int, cross_lo: int, cross_hi: int, is_start: bool) -> list[tuple[int, int]]:
    dim_minor = _dim(1 - ax)
    arm_choices = [v for v in range(1, dim_minor - 1) if abs(v - main_coord) >= ARM_MIN_OFFSET]
    if not arm_choices:
        return []
    minor = random.choice(arm_choices)
    if is_start:
        if cross_lo <= 1:
            return []
        arm_major = random.randint(1, cross_lo - 1)
        return [_rc(ax, 0, minor), _rc(ax, arm_major, minor), _rc(ax, arm_major, main_coord)]
    if cross_hi >= _dim(ax) - 2:
        return []
    arm_major = random.randint(cross_hi + 1, _dim(ax) - 2)
    return [_rc(ax, arm_major, main_coord), _rc(ax, arm_major, minor), _rc(ax, _dim(ax) - 1, minor)]


def _spaced_values_checker(limit: int, count: int) -> list[int] | None:
    choices = list(range(1, limit - 1))
    for _ in range(SPACED_VALUES_RETRIES):
        vals = sorted(random.sample(choices, count))
        if not all(b - a >= 2 for a, b in zip(vals, vals[1:])):
            continue
        if not all((b - a) % 2 == 1 for a, b in zip(vals, vals[1:])):
            continue
        return vals
    return None


def _serpentine_path(count: int) -> tuple[list[tuple[int, int]], list[tuple[int, int]]] | None:
    ax = random.randint(0, 1)
    lanes = _spaced_values_checker(_dim(ax), count)
    if lanes is None:
        return None
    if random.random() < 0.5:
        m0, m1 = 1, random.randint(SERPENTINE_MINOR_LO, _dim(1 - ax) - 2)
    else:
        m0, m1 = random.randint(1, SERPENTINE_MINOR_HI), _dim(1 - ax) - 2
    cross_minor = random.randint(m0 + CROSS_MINOR_MARGIN, m1 - CROSS_MINOR_MARGIN)

    start_at_m0 = random.random() < 0.5
    pts = [_rc(ax, lanes[0], m0 if start_at_m0 else m1)]
    fwd = start_at_m0
    for i, lane in enumerate(lanes):
        end_m = m1 if fwd else m0
        if pts[-1] != _rc(ax, lane, end_m):
            pts.append(_rc(ax, lane, end_m))
        if i + 1 < len(lanes):
            pts.append(_rc(ax, lanes[i + 1], end_m))
            fwd = not fwd

    if not _on_boundary(*pts[0]):
        r0, c0 = pts[0]
        if ax == 0:
            pts.insert(0, (r0, 0 if c0 == m0 else W - 1))
        else:
            pts.insert(0, (0 if r0 == m0 else H - 1, c0))
    if not _on_boundary(*pts[-1]):
        rn, cn = pts[-1]
        if ax == 0:
            pts.append((rn, 0 if cn == m0 else W - 1))
        else:
            pts.append((0 if rn == m0 else H - 1, cn))

    cells = _path(pts)
    crossings = [_rc(ax, lane, cross_minor) for lane in lanes]
    parities = {(r + c) % 2 for r, c in crossings}
    if len(parities) < 2:
        return None
    return (cells, crossings) if _valid_path(cells) else None


def _path_crossing_all(crossings: list[tuple[int, int]]) -> list[tuple[int, int]] | None:
    rows = {r for r, _ in crossings}
    cols = {c for _, c in crossings}
    if len(cols) == 1:
        ax, main_coord, cross_vals = 0, next(iter(cols)), rows
    elif len(rows) == 1:
        ax, main_coord, cross_vals = 1, next(iter(rows)), cols
    else:
        return None

    lo, hi = min(cross_vals), max(cross_vals)
    if lo <= 1 or hi >= _dim(ax) - 2:
        return None

    pts = [_rc(ax, 0, main_coord), _rc(ax, _dim(ax) - 1, main_coord)]
    flavor = random.choice(("straight", "start_arm", "end_arm", "both_arms"))
    if flavor in ("start_arm", "both_arms"):
        arm = _build_arm(ax, main_coord, lo, hi, is_start=True)
        if arm:
            pts = arm + pts[1:]
    if flavor in ("end_arm", "both_arms"):
        arm = _build_arm(ax, main_coord, lo, hi, is_start=False)
        if arm:
            pts = pts[:-1] + arm
    cells = _path(pts)
    return cells if _valid_path(cells) else None


def _can_add(logical: dict[int, set[tuple[int, int]]], color: int, cells: list[tuple[int, int]]) -> bool:
    if not _valid_path(cells):
        return False
    pts = set(cells)
    for pt in pts:
        if pt in logical[color]:
            return False
        if any(nb in logical[color] for nb in _neighbors(pt, H, W)):
            return False
        if sum(pt in pset for pset in logical.values()) >= 2:
            return False
    return True


def _crossing_cells(logical: dict[int, set[tuple[int, int]]]) -> list[tuple[int, int]]:
    cells = set().union(*logical.values())
    return [pt for pt in cells if sum(pt in logical[color] for color in logical) > 1]


def _crossing_map(logical: dict[int, set[tuple[int, int]]]) -> dict[tuple[int, int], list[int]]:
    return {pt: [color for color in logical if pt in logical[color]] for pt in _crossing_cells(logical)}


def _valid_crossings(logical: dict[int, set[tuple[int, int]]], dominant: int) -> bool:
    crossings = _crossing_cells(logical)
    if not MIN_FAMILY_CROSSINGS <= len(crossings) <= MAX_CROSSINGS:
        return False
    parities = {(r + c) % 2 for r, c in crossings}
    if len(parities) < 2:
        return False
    for pt in crossings:
        colors = [color for color in logical if pt in logical[color]]
        if len(colors) != 2 or dominant not in colors:
            return False
        orientations = [_orientation(logical[color], pt) for color in colors]
        if None in orientations or sorted(orientations) != ["h", "v"]:
            return False
    return True


def _ray_from(pivot: tuple[int, int], direction: tuple[int, int]) -> set[tuple[int, int]]:
    out = set()
    r, c = pivot
    dr, dc = direction
    while 0 <= r < H and 0 <= c < W:
        out.add((r, c))
        r += dr
        c += dc
    return out


def _yellow_candidates(
    logical: dict[int, set[tuple[int, int]]],
    paths: dict[int, list[set[tuple[int, int]]]],
    crossings: set[tuple[int, int]],
) -> list[tuple[int, tuple[int, int], tuple[int, int], set[tuple[int, int]], set[tuple[int, int]]]]:
    candidates = []
    occupied = set().union(*logical.values())
    for color, comps in paths.items():
        for comp in comps:
            for pivot in comp:
                if pivot in crossings or not _is_turn_or_endpoint(comp, pivot):
                    continue
                for out_dir in DIRS4:
                    mr, mc = pivot[0] + out_dir[0], pivot[1] + out_dir[1]
                    if not (0 <= mr < H and 0 <= mc < W) or (mr, mc) not in comp:
                        continue
                    target_ray = _ray_from(pivot, out_dir)
                    if not target_ray.issubset(comp):
                        continue
                    if target_ray & crossings:
                        continue
                    old_dir = (-out_dir[0], -out_dir[1])
                    old_cells: set[tuple[int, int]] = set()
                    r, c = pivot[0] + old_dir[0], pivot[1] + old_dir[1]
                    while 0 <= r < H and 0 <= c < W and (r, c) not in occupied:
                        old_cells.add((r, c))
                        r += old_dir[0]
                        c += old_dir[1]
                    if not old_cells:
                        continue
                    old_len = random.randint(1, len(old_cells))
                    old_arm = set(list(_ray_from((pivot[0] + old_dir[0], pivot[1] + old_dir[1]), old_dir))[:old_len])
                    if old_arm and old_arm.issubset(old_cells):
                        candidates.append((color, pivot, out_dir, target_ray, old_arm))
    random.shuffle(candidates)
    return candidates


def _choose_red_markers(
    logical: dict[int, set[tuple[int, int]]],
    paths: dict[int, list[set[tuple[int, int]]]],
    blocked: set[tuple[int, int]],
    count: int,
) -> list[tuple[int, tuple[int, int]]]:
    candidates = [(color, pt) for color, comps in paths.items() for comp in comps for pt in comp if pt not in blocked and _is_turn_or_endpoint(comp, pt)]
    random.shuffle(candidates)
    chosen: list[tuple[int, tuple[int, int]]] = []
    used: set[tuple[int, int]] = set(blocked)
    per_color_left = {color: len(logical[color]) for color in logical}
    for color, pt in candidates:
        if len(chosen) == count:
            break
        if per_color_left[color] <= MIN_CELLS_FOR_RED:
            continue
        if any(nb in used for nb in [pt] + _neighbors(pt, H, W)):
            continue
        chosen.append((color, pt))
        used.add(pt)
        per_color_left[color] -= 1
    return chosen if len(chosen) == count else []


def _trace_pink_family_logical(marker: tuple[int, int], color: int, logical: dict[int, set[tuple[int, int]]]) -> set[tuple[int, int]]:
    dirs = _axis_dirs(logical[color], marker)
    if dirs is None:
        return set()
    family: set[tuple[int, int]] = set()
    other_colors: set[int] = set()
    for dr, dc in dirs:
        r, c = marker[0] + dr, marker[1] + dc
        while 0 <= r < H and 0 <= c < W and (r, c) in logical[color]:
            owners = [other for other, pts in logical.items() if (r, c) in pts]
            if len(owners) > 1:
                if len(owners) != 2 or color not in owners:
                    return set()
                family.add((r, c))
                other_colors.add(_crossing_other(owners, color))
            nr, nc = r + dr, c + dc
            if not (0 <= nr < H and 0 <= nc < W and (nr, nc) in logical[color]):
                break
            r, c = nr, nc
    if len(other_colors) != 1:
        return set()
    return family


def _pink_candidates(
    logical: dict[int, set[tuple[int, int]]],
    pink_color: int,
    blocked: set[tuple[int, int]],
) -> list[tuple[int, tuple[int, int], tuple[tuple[int, int], ...]]]:
    crossings = sorted([pt for pt, colors in _crossing_map(logical).items() if pink_color in colors])
    if len(crossings) < MIN_FAMILY_CROSSINGS:
        return []
    if len({(r + c) % 2 for r, c in crossings}) < 2:
        return []

    orient = _orientation(logical[pink_color], crossings[0])
    if orient is None:
        return []
    dirs = [(0, -1), (0, 1)] if orient == "h" else [(-1, 0), (1, 0)]

    candidates: list[tuple[int, tuple[int, int], tuple[tuple[int, int], ...]]] = []
    for cross in crossings:
        for dr, dc in dirs:
            mid = (cross[0] + dr, cross[1] + dc)
            marker = (cross[0] + 2 * dr, cross[1] + 2 * dc)
            if not (0 <= marker[0] < H and 0 <= marker[1] < W):
                continue
            if mid not in logical[pink_color] or marker not in logical[pink_color]:
                continue
            if _orientation(logical[pink_color], marker) != orient:
                continue
            if marker in blocked or any(nb in blocked for nb in [marker] + _neighbors(marker, H, W)):
                continue
            family = _trace_pink_family_logical(marker, pink_color, logical)
            if family != set(crossings):
                continue
            candidates.append((pink_color, marker, tuple(crossings)))
    random.shuffle(candidates)
    return candidates


def _build_logical() -> tuple[dict[int, set[tuple[int, int]]], dict[int, list[set[tuple[int, int]]]], int, int] | None:
    desired_crossings = random.choices(CROSSING_COUNT_OPTIONS, weights=CROSSING_COUNT_WEIGHTS)[0]
    colors = random.sample(LINE_PALETTE, 2)
    dominant = random.choice(colors)
    crossing_color = colors[0] if colors[1] == dominant else colors[1]
    logical: dict[int, set[tuple[int, int]]] = {c: set() for c in colors}
    paths: dict[int, list[set[tuple[int, int]]]] = {c: [] for c in colors}

    built = _serpentine_path(desired_crossings)
    if built is None:
        return None
    dom_cells, forced_crossings = built
    if not _can_add(logical, dominant, dom_cells):
        return None
    logical[dominant].update(dom_cells)
    paths[dominant].append(set(dom_cells))

    cross_cells = _path_crossing_all(forced_crossings)
    if cross_cells is None or not _can_add(logical, crossing_color, cross_cells):
        return None
    logical[crossing_color].update(cross_cells)
    paths[crossing_color].append(set(cross_cells))

    if not _valid_crossings(logical, dominant):
        return None
    return logical, paths, dominant, crossing_color


def _render_target(
    logical: dict[int, set[tuple[int, int]]],
    comp_count: dict[int, int],
    yellow_rays: dict[int, set[tuple[int, int]]],
    pink_forced: dict[tuple[int, int], int],
) -> np.ndarray:
    out = np.full((H, W), BG, dtype=int)
    for pt in set().union(*logical.values()):
        colors_here = [color for color in logical if pt in logical[color]]
        forced = [color for color in colors_here if pt in yellow_rays.get(color, set())]
        if forced:
            out[pt] = forced[0]
        elif len(colors_here) == 1:
            out[pt] = colors_here[0]
        else:
            priority = pink_forced.get(pt, min(colors_here, key=lambda color: (-comp_count[color], color)))
            other = _crossing_other(colors_here, priority)
            out[pt] = priority if _checker_prefers_priority(pt) else other
    return out


def _marker_isolation_ok(grid: np.ndarray) -> bool:
    for color in (RED, YELLOW, PINK):
        for pt in map(tuple, np.argwhere(grid == color)):
            if any(grid[nb] == color for nb in _neighbors(pt, H, W)):
                return False
    return True


def _derive_input(
    logical: dict[int, set[tuple[int, int]]],
    paths: dict[int, list[set[tuple[int, int]]]],
    dominant: int,
    pink_color: int,
) -> tuple[np.ndarray, np.ndarray] | None:
    crossings = set(_crossing_cells(logical))
    crossing_map = _crossing_map(logical)
    mode = random.choices(["pink", "red_pink", "yellow_pink", "all"], weights=MARKER_MODE_WEIGHTS)[0]
    yellow_count = random.randint(*YELLOW_COUNT_RANGE) if mode in ("yellow_pink", "all") else 0
    red_count = random.randint(*RED_COUNT_RANGE) if mode in ("red_pink", "all") else 0

    pink_infos: list[tuple[int, tuple[int, int], tuple[tuple[int, int], ...]]] = []
    pink_blocked: set[tuple[int, int]] = set()
    for color, marker, family in _pink_candidates(logical, pink_color, pink_blocked):
        pink_infos.append((color, marker, family))
        pink_blocked.add(marker)
        pink_blocked.update(_neighbors(marker, H, W))
        for cross in family:
            pink_blocked.update(_segment(marker, cross))
        if len(pink_infos) == PINK_COUNT:
            break
    if len(pink_infos) != PINK_COUNT:
        return None

    pink_forced = {cross: color for color, _marker, family in pink_infos for cross in family}
    if len({(r + c) % 2 for r, c in pink_forced}) < 2:
        return None

    blocked = set(crossings)
    for _color, marker, family in pink_infos:
        blocked.add(marker)
        blocked.update(_neighbors(marker, H, W))
        for cross in family:
            blocked.update(_segment(marker, cross))

    yellow_infos = []
    if yellow_count:
        for cand in _yellow_candidates(logical, paths, crossings):
            color, pivot, out_dir, target_ray, old_arm = cand
            marker = (pivot[0] + out_dir[0], pivot[1] + out_dir[1])
            if marker in blocked or any(nb in blocked for nb in _neighbors(marker, H, W)):
                continue
            if any((old & old_arm) or (ray & old_arm) or (ray & target_ray) for _, _, _, ray, old in yellow_infos):
                continue
            yellow_infos.append(cand)
            blocked.update(target_ray | old_arm | {marker})
            if len(yellow_infos) == yellow_count:
                break
        if len(yellow_infos) != yellow_count:
            return None

    red_markers = _choose_red_markers(logical, paths, blocked, red_count) if red_count else []
    if red_count and not red_markers:
        return None

    inp = np.full((H, W), BG, dtype=int)
    comp_count_masks: dict[int, set[tuple[int, int]]] = {c: set() for c in logical}

    crossing_display: dict[tuple[int, int], int] = {}
    for pt, colors in crossing_map.items():
        assert dominant in colors and pink_color in colors and len(colors) == 2
        priority = pink_forced.get(pt, dominant)
        target_color = priority if _checker_prefers_priority(pt) else _crossing_other(colors, priority)
        crossing_display[pt] = _crossing_other(colors, target_color)

    removed_for_yellow: dict[int, set[tuple[int, int]]] = {c: set() for c in logical}
    old_arms: dict[int, set[tuple[int, int]]] = {c: set() for c in logical}
    yellow_rays: dict[int, set[tuple[int, int]]] = {c: set() for c in logical}
    yellow_markers: list[tuple[int, tuple[int, int]]] = []
    for color, pivot, out_dir, target_ray, old_arm in yellow_infos:
        marker = (pivot[0] + out_dir[0], pivot[1] + out_dir[1])
        removed_for_yellow[color].update(target_ray - {pivot})
        old_arms[color].update(old_arm)
        yellow_rays[color].update(target_ray)
        yellow_markers.append((color, marker))

    red_by_pt = {pt: color for color, pt in red_markers}
    yellow_by_pt = {pt: color for color, pt in yellow_markers}
    pink_by_pt = {marker: color for color, marker, _family in pink_infos}

    for color, pts in logical.items():
        for pt in pts:
            if pt in removed_for_yellow[color]:
                continue
            if pt in crossings and crossing_display[pt] != color:
                continue
            if pt in red_by_pt or pt in yellow_by_pt or pt in pink_by_pt:
                continue
            inp[pt] = color
            comp_count_masks[color].add(pt)
    for color, pts in old_arms.items():
        for pt in pts:
            if inp[pt] != BG:
                return None
            inp[pt] = color
            comp_count_masks[color].add(pt)

    for pt, color in red_by_pt.items():
        inp[pt] = RED
        comp_count_masks[color].add(pt)
    for pt, color in pink_by_pt.items():
        inp[pt] = PINK
        comp_count_masks[color].add(pt)
    for pt, _color in yellow_by_pt.items():
        inp[pt] = YELLOW

    if not _marker_isolation_ok(inp):
        return None
    for pt, color in yellow_by_pt.items():
        if not any(inp[nb] == color for nb in _neighbors(pt, H, W)):
            return None
    for marker, color in pink_by_pt.items():
        family = _trace_pink_family_logical(marker, color, logical)
        if family != set(next(fam for col, mk, fam in pink_infos if mk == marker and col == color)):
            return None

    comp_count = {color: _gen_component_count(pts) for color, pts in comp_count_masks.items()}
    target = _render_target(logical, comp_count, yellow_rays, pink_forced)
    target_without_pink = _render_target(logical, comp_count, yellow_rays, {})

    for pt in crossings:
        if target[pt] == inp[pt]:
            return None
    if not any(target[pt] != target_without_pink[pt] for pt in pink_forced):
        return None
    if not all(target_without_pink[pt] == (dominant if _checker_prefers_priority(pt) else pink_color) for pt in pink_forced):
        return None

    colors = sorted(logical)
    counts = {c: len(components_mask(target == c, connectivity=4)) for c in colors}
    if not all(COMPONENTS_PER_COLOR[0] <= v <= COMPONENTS_PER_COLOR[1] for v in counts.values()):
        return None
    if not TOTAL_COMPONENTS[0] <= sum(counts.values()) <= TOTAL_COMPONENTS[1]:
        return None
    if not FILL_RANGE[0] <= int(np.count_nonzero(target)) <= FILL_RANGE[1]:
        return None

    try:
        solved = main(inp)
    except Exception:
        return None
    return (inp, target) if np.array_equal(solved, target) else None


def _gen_attempt() -> np.ndarray | None:
    built = _build_logical()
    if built is None:
        return None
    logical, paths, dominant, pink_color = built
    derived = _derive_input(logical, paths, dominant, pink_color)
    return derived[0] if derived is not None else None


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("failed to generate a valid faa9f03d combo input")
