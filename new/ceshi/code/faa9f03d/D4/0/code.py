# Variant of faa9f03d
# Dimension: D4 — 标记到线色的分配规则 (transform)
# Change: 标记沿四个正交方向扫描，分配给最近可见的线色，优先选择沿该轴有邻接支撑的端点
#   → 分配给最远可见线色而非最近
#
# Design analysis
# source_task: None
# variant_spec:
#   core_change:
#     what_changes:
#       - solver 里唯一必须反转的是标记候选排序的距离主键: 从较小 dist 优先改成较大 dist 优先
#       - 同距离时仍保留“沿该轴有邻接支撑的端点优先”的旧次序, 避免把 D4 变成另一个维度
#     immediate consequence:
#       - 如果红标记仍像原题那样直接替换某条线上的原像素, 那么该颜色必然在距离 1 处可见
#       - 这种输入上“最近”和“最远”要么相同, 要么最远会错误跳去别色, 因而 D4 视觉上不可观察
#   concretization_and_trace:
#     input_objects:
#       - 输出仍是 12x12 上的单像素宽正交折线网络, 颜色之间允许交叉
#       - 黄色标记仍放在线端点旁, 因为它必须先看到相邻枢轴端点才可执行翻臂
#       - 红色标记改成放在“长缺口内部”的引导点: 同色线段在左右或上下更远处可见, 同时某个别色线段在垂直方向更近可见
#     micro_example:
#       - 假设某蓝色水平线原本覆盖 (5,2)..(5,8), 现在输入中保留 (5,2)(5,3)(5,7)(5,8), 删除 (5,4)(5,6), 在 (5,5) 放红标记
#       - 若同列上方 (4,5) 有一条绿色竖线, 则红标记看到: 上方绿 dist=1, 左右蓝 dist=2
#       - 原规则会把红标记分给绿, 新规则会分给蓝; 然后蓝色再用 gap-fill 恢复整段水平线
#   complexity_calibration:
#     safeguards:
#       - 红标记至少 2 个, 并且每个都要求 nearest_color != farthest_color, 让样例中有重复支撑
#       - 黄色标记最多 1 个且只在 farthest 仍能选回本色时才保留, 避免其邻接距离 1 的旧性质掩盖 D4
#       - 仍保留 2-4 条线色、交叉、碎片度裁决与黄色翻臂, 所以任务没有被简化成单一路径修补
#   implicit_assumption_audit:
#     solver_lines_touched:
#       - marker candidate collection: 旧代码默认 candidates.sort() 后取第一项即最近; 变体下必须显式按 -dist 排
#       - assign_dir for yellow: 仍依赖“被选中的那条可见线”所在方向, 所以黄色生成必须保证 farthest 也指向正确枢轴
#       - has_axis_support / should_fill: 这些逻辑不依赖 nearest/farthest, 只要求 marker 最终被分给正确颜色
#       - crossing disambiguation: 依赖 comp_count 是“输入修补前, 但把 marker 视作其所属线色后”的碎片数; 新 generator 继续维护这一点
#   placement_legality:
#     red_marker_constraints:
#       - 标记点必须位于逻辑输出的一条直线段上, 不是 crossing
#       - 标记两侧都要保留更远的同色锚点, 且锚点外侧还要有同色支撑, 这样 solver 的 should_fill 才会补回缺口
#       - 被删除的缺口单元不能撞到 crossing、黄色射线、旧臂或别的标记邻域
#     yellow_marker_constraints:
#       - 仍沿原任务定义, 放在未来新射线的第一格
#       - 只接受在最终输入上 farthest assignment 仍回到本色的候选
#   visual_observability:
#     why_human_can_see_it:
#       - 红标记附近会稳定出现“近处别色, 远处本色”的对视结构
#       - 同一输入里至少重复两次这个结构, 解题者能从 2-4 例中归纳出“不是找最近, 而是找最远”
#       - 不使用颜色 id 排序来制造变化; 颜色只作为线色身份, 决策线索来自可见距离和几何支撑
#   visual_cleanliness:
#     choices:
#       - 标记之间保持隔离, 红标记只选单条直段内部, 避免 turn / crossing / yellow 三层叠加在同一小区域
#       - 黄色数量受限, 保留原题风格但不让画面过乱
#   new_main_assumptions_review:
#     assumptions_enforced_by_generator:
#       - 每个红标记在最终输入上都存在至少一个最近别色候选和至少一个更远的本色候选
#       - 每个黄色标记在最终输入上 farthest assignment 仍能回到自己的枢轴颜色
#       - 红标记删出的长缺口两端都有轴向支撑, 因此 fill 规则可恢复输出
#   final_generator_strategy:
#     - 实现上不再重写整套布局器, 而是把原任务 generator 当作候选分布
#     - 只保留那些在新 solver 下可解, 且至少含有 2 枚真正触发 nearest!=farthest 的红标记的样本
#     - 这样保留了原题的大部分视觉与结构复杂度, 同时确保 D4 变化确实在训练样本里发生

from __future__ import annotations

import itertools
import importlib.util
import random

import numpy as np

from arc_utils import components_mask, get_bg


DIRS4 = ((-1, 0), (1, 0), (0, -1), (0, 1))


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


def _visible_candidates(
    grid: np.ndarray,
    pt: tuple[int, int],
) -> list[tuple[int, int, int, int, int]]:
    g = np.asarray(grid, dtype=int)
    line_colors = set(map(int, g.ravel())) - {BG, 2, 4}
    original = {color: _pts(g == color) for color in line_colors}
    H, W = g.shape
    r0, c0 = pt
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
            if value in (BG, 2, 4):
                r += dr
                c += dc
                dist += 1
                continue
            break
    return candidates


def _pick_visible(
    candidates: list[tuple[int, int, int, int, int]],
    *,
    farthest: bool,
) -> tuple[int, int, int, int, int] | None:
    if not candidates:
        return None
    if farthest:
        return min(candidates, key=lambda x: (-x[0], x[1], x[2], x[3], x[4]))
    return min(candidates, key=lambda x: (x[0], x[1], x[2], x[3], x[4]))


def main(grid: np.ndarray) -> np.ndarray:
    """
    Repair a one-cell-thick colored line network by recoloring singleton markers, filling rectilinear gaps, and redirecting yellow-marked arms using the farthest visible line color.

    Input structure:
    - A 12x12-style rectangular grid with one dominant background color.
    - Non-marker colors form one-cell-thick horizontal/vertical polyline fragments that may cross.
    - Marker colors are exactly those whose 4-connected components are all singleton pixels.
    - Red singletons mark missing line cells inside repairable gaps; yellow singletons sit beside endpoints that must flip to the marked side.

    Rule:
    - For every marker, scan in the four orthogonal directions and assign it to the farthest visible line color, still preferring a candidate that has support along that scan axis when distances tie.
    - After markers are recolored conceptually, each line color fills the orthogonal gaps between consecutive facing cells when the endpoint-support tests allow it.
    - A yellow control uses its assigned direction to find the pivot endpoint, deletes the old arm on the opposite side, and paints a new straight ray from the pivot through the control cell to the boundary.
    - At overlaps, yellow-control rays override everything else; otherwise crossings are resolved by pre-repair fragmentation, with lower color id breaking ties among equally filled claims.

    Key insight:
    The singleton pixels are not attached to the nearest visible stroke anymore. Red markers are placed so that a closer distractor color is visible, but the intended line color is farther away on the same row or column; assigning markers by farthest line-of-sight is what makes the gap-filling and yellow arm redirection reconstruct the target network.
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
        _dist, _unsupported, color, dr, dc = min(
            candidates,
            key=lambda x: (-x[0], x[1], x[2], x[3], x[4]),
        )
        assignment[(r0, c0)] = int(color)
        assign_dir[(r0, c0)] = (int(dr), int(dc))

    masks = {color: set(pts) for color, pts in original.items()}
    marker_pts = {color: set() for color in line_colors}
    for pt, color in assignment.items():
        masks[color].add(pt)
        marker_pts[color].add(pt)

    comp_count = {color: _component_count(g.shape, masks[color]) for color in line_colors}

    def has_axis_support(color: int, r: int, c: int, axis: int) -> bool:
        return (r, c) in marker_pts[color] or _axis_support(masks[color], r, c, axis)

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

        if (r1, c1) in marker_pts[color] and (r2, c2) in marker_pts[color]:
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
    if 4 in marker_colors:
        for control in _pts(g == 4):
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
                    marker_pts[color].discard(q)
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

    out = np.full_like(g, bg)

    def marker_brackets(color: int, pt: tuple[int, int]) -> bool:
        r, c = pt
        same_row = [mc for mr, mc in marker_pts[color] if mr == r]
        if any(mc < c for mc in same_row) and any(mc > c for mc in same_row):
            return True
        same_col = [mr for mr, mc in marker_pts[color] if mc == c]
        return any(mr < r for mr in same_col) and any(mr > r for mr in same_col)

    for pt, colors_here in claims.items():
        forced = {color for color in colors_here if pt in control_ray[color]}
        if forced:
            assert len(forced) == 1, f"ambiguous yellow-control rays at {pt}: {forced}"
            out[pt] = next(iter(forced))
            continue

        if len(colors_here) == 1:
            out[pt] = colors_here[0]
            continue

        original_crossing = [color for color in colors_here if int(g[pt]) == color]
        if original_crossing:
            out[pt] = min(colors_here, key=lambda color: (-comp_count[color], color))
        else:
            bracketed = [color for color in colors_here if marker_brackets(color, pt)]
            if bracketed:
                out[pt] = min(bracketed, key=lambda color: (-comp_count[color], color))
            else:
                out[pt] = min(colors_here)

    for color in line_colors:
        for comp in components_mask(out == color, connectivity=4):
            pts_set = set(map(tuple, comp))
            for r, c in pts_set:
                nbrs = [
                    (nr, nc)
                    for nr, nc in [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]
                    if (nr, nc) in pts_set
                ]
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
                    0 <= r + dr < H
                    and 0 <= c + dc < W
                    and int(out[r + dr, c + dc]) not in (bg, color)
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
                if (
                    ((r, c - 1) in color_cells[b] and (r, c + 1) in color_cells[b])
                    or ((r - 1, c) in color_cells[b] and (r + 1, c) in color_cells[b])
                ):
                    crossing_colors.add(a)
                    crossing_colors.add(b)
    for color in line_colors:
        assert color in crossing_colors, f"color {color} has no crossing with any other color"

    return out


# --------------- generator (o5x, D4 variant) ---------------

H = W = 12
BG = 0
LINE_PALETTE = [1, 3, 5, 6, 7, 8, 9]

POLYLINE_PIECE_WEIGHTS = [5, 4, 2, 1]
MIN_SPAN_LEN = 4
FULL_SPAN_PROB = 0.65
ARM_MIN_OFFSET = 2

CROSSING_COUNT_WEIGHTS = [3, 4]
SERPENTINE_MINOR_LO = 6
SERPENTINE_MINOR_HI = 5
CROSS_MINOR_MARGIN = 2
MAX_COLORS = 4
MAX_CROSSINGS = 12

MARKER_MODE_WEIGHTS = [1]  # generator focuses on red markers so D4 is always visible
YELLOW_COUNT_RANGE = (0, 0)
RED_COUNT_RANGE = (2, 2)
RED_GAP_SIDE_RANGE = (2, 4)
RED_CANDIDATE_LIMIT = 14

COMPONENTS_PER_COLOR = (1, 4)
TOTAL_COMPONENTS = (4, 6)
FILL_RANGE = (33, 55)

SPACED_VALUES_RETRIES = 60
EXTRA_POLYLINE_RETRIES = 80
GEN_RETRIES = 800


def _neighbors(pt: tuple[int, int]) -> list[tuple[int, int]]:
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
        if not (
            (r == 0 and dr == 1)
            or (r == H - 1 and dr == -1)
            or (c == 0 and dc == 1)
            or (c == W - 1 and dc == -1)
        ):
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


def _build_polyline_waypoints(ax: int) -> list[tuple[int, int]]:
    pieces = random.choices(
        range(1, len(POLYLINE_PIECE_WEIGHTS) + 1),
        weights=POLYLINE_PIECE_WEIGHTS,
    )[0]
    a0, a1 = _random_span(_dim(ax))
    pieces = min(pieces, max(1, a1 - a0))
    minors = random.sample(range(1, _dim(1 - ax) - 1), pieces)
    bends = sorted(random.sample(range(a0 + 1, a1), pieces - 1)) if pieces > 1 else []
    pts = [_rc(ax, a0, minors[0])]
    for i, bend in enumerate(bends):
        pts.append(_rc(ax, bend, minors[i]))
        pts.append(_rc(ax, bend, minors[i + 1]))
    pts.append(_rc(ax, a1, minors[-1]))
    if not _on_boundary(*pts[0]):
        r0, c0 = pts[0]
        if ax == 0:
            pts.insert(0, (r0, random.choice([0, W - 1])))
        else:
            pts.insert(0, (random.choice([0, H - 1]), c0))
    if not _on_boundary(*pts[-1]):
        rn, cn = pts[-1]
        if ax == 0:
            pts.append((rn, random.choice([0, W - 1])))
        else:
            pts.append((random.choice([0, H - 1]), cn))
    return pts


def _build_arm(
    ax: int,
    main_coord: int,
    cross_lo: int,
    cross_hi: int,
    is_start: bool,
) -> list[tuple[int, int]]:
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
    return [
        _rc(ax, arm_major, main_coord),
        _rc(ax, arm_major, minor),
        _rc(ax, _dim(ax) - 1, minor),
    ]


def _path_through(cell: tuple[int, int], orient: str) -> list[tuple[int, int]] | None:
    r, c = cell
    ax = 0 if orient == "v" else 1
    major = r if ax == 0 else c
    minor = c if ax == 0 else r
    if major == 0 or major == _dim(ax) - 1:
        return None
    lo = 0 if random.random() < FULL_SPAN_PROB else random.randint(0, major - 1)
    hi = _dim(ax) - 1 if random.random() < FULL_SPAN_PROB else random.randint(major + 1, _dim(ax) - 1)
    pts = [_rc(ax, lo, minor), _rc(ax, hi, minor)]
    flavor = random.choice(("straight", "start_arm", "end_arm", "both_arms"))
    if flavor in ("start_arm", "both_arms") and lo > 0:
        arm = _build_arm(ax, minor, lo, hi, is_start=True)
        if arm:
            pts = arm + pts[1:]
    if flavor in ("end_arm", "both_arms") and hi < _dim(ax) - 1:
        arm = _build_arm(ax, minor, lo, hi, is_start=False)
        if arm:
            pts = pts[:-1] + arm
    if not _on_boundary(*pts[0]):
        pts[0] = _rc(ax, 0, minor)
    if not _on_boundary(*pts[-1]):
        pts[-1] = _rc(ax, _dim(ax) - 1, minor)
    cells = _path(pts)
    return cells if _valid_path(cells) and _orientation(set(cells), cell) == orient else None


def _spaced_values(limit: int, count: int) -> list[int] | None:
    choices = list(range(1, limit - 1))
    for _ in range(SPACED_VALUES_RETRIES):
        vals = sorted(random.sample(choices, count))
        if all(b - a >= 2 for a, b in zip(vals, vals[1:])):
            return vals
    return None


def _serpentine_path(count: int) -> tuple[list[tuple[int, int]], list[tuple[int, int]]] | None:
    ax = random.randint(0, 1)
    lanes = _spaced_values(_dim(ax), count)
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
        if any(nb in logical[color] for nb in _neighbors(pt)):
            return False
        if sum(pt in pset for pset in logical.values()) >= 2:
            return False
    return True


def _crossing_cells(logical: dict[int, set[tuple[int, int]]]) -> list[tuple[int, int]]:
    cells = set().union(*logical.values())
    return [pt for pt in cells if sum(pt in logical[color] for color in logical) > 1]


def _valid_crossings(logical: dict[int, set[tuple[int, int]]], dominant: int) -> bool:
    crossings = _crossing_cells(logical)
    if not 1 <= len(crossings) <= MAX_CROSSINGS:
        return False
    for pt in crossings:
        colors = [color for color in logical if pt in logical[color]]
        if len(colors) != 2 or dominant not in colors:
            return False
        orientations = [_orientation(logical[color], pt) for color in colors]
        if None in orientations or sorted(orientations) != ["h", "v"]:
            return False
    return True


def _ray_from(pivot: tuple[int, int], direction: tuple[int, int]) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    r, c = pivot
    dr, dc = direction
    while 0 <= r < H and 0 <= c < W:
        out.append((r, c))
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
                    target_ray = set(_ray_from(pivot, out_dir))
                    if not target_ray.issubset(comp):
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
                    ray = _ray_from((pivot[0] + old_dir[0], pivot[1] + old_dir[1]), old_dir)
                    old_arm = set(ray[:old_len])
                    if old_arm and old_arm.issubset(old_cells):
                        candidates.append((color, pivot, out_dir, target_ray, old_arm))
    random.shuffle(candidates)
    return candidates


def _find_crossing_point(logical: dict[int, set[tuple[int, int]]]) -> tuple[tuple[int, int], str] | None:
    candidates = []
    for pts in logical.values():
        for r, c in pts:
            orient = _orientation(pts, (r, c))
            if orient is None:
                continue
            perp = "v" if orient == "h" else "h"
            major = r if perp == "v" else c
            if major <= 1 or major >= (H if perp == "v" else W) - 2:
                continue
            candidates.append(((r, c), perp))
    random.shuffle(candidates)
    return candidates[0] if candidates else None


def _build_logical() -> tuple[dict[int, set[tuple[int, int]]], dict[int, list[set[tuple[int, int]]]], int] | None:
    desired_crossings = random.choices(
        range(1, len(CROSSING_COUNT_WEIGHTS) + 1),
        weights=CROSSING_COUNT_WEIGHTS,
    )[0]
    if min(MAX_COLORS, TOTAL_COMPONENTS[1] - desired_crossings) < 4:
        return None
    n_colors = 4
    colors = random.sample(LINE_PALETTE, n_colors)
    dominant = random.choice(colors)
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

    crossing_color = random.choice([c for c in colors if c != dominant])
    cross_cells = _path_crossing_all(forced_crossings)
    if cross_cells is None or not _can_add(logical, crossing_color, cross_cells):
        return None
    logical[crossing_color].update(cross_cells)
    paths[crossing_color].append(set(cross_cells))

    for color in [c for c in colors if c not in (dominant, crossing_color)]:
        cells = None
        for _ in range(EXTRA_POLYLINE_RETRIES):
            cp = _find_crossing_point(logical)
            if cp is None:
                break
            trial = _path_through(cp[0], cp[1])
            if trial is not None and _can_add(logical, color, trial):
                cells = trial
                break
        if cells is None:
            return None
        logical[color].update(cells)
        paths[color].append(set(cells))

    if not _valid_crossings(logical, dominant):
        return None
    return logical, paths, dominant


def _render_target(
    logical: dict[int, set[tuple[int, int]]],
    comp_count: dict[int, int],
    yellow_rays: dict[int, set[tuple[int, int]]],
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
            out[pt] = min(colors_here, key=lambda color: (-comp_count[color], color))
    return out


def _marker_isolation_ok(grid: np.ndarray) -> bool:
    for color in (2, 4):
        for pt in map(tuple, np.argwhere(grid == color)):
            if any(grid[nb] == color for nb in _neighbors(pt)):
                return False
    return True


def _crossing_display(
    logical: dict[int, set[tuple[int, int]]],
    dominant: int,
    crossings: set[tuple[int, int]],
) -> dict[tuple[int, int], int] | None:
    display: dict[tuple[int, int], int] = {}
    for pt in crossings:
        colors = [color for color in logical if pt in logical[color]]
        if dominant not in colors or len(colors) != 2:
            return None
        display[pt] = colors[0] if colors[1] == dominant else colors[1]
    return display


def _render_base_grid(
    logical: dict[int, set[tuple[int, int]]],
    crossings: set[tuple[int, int]],
    crossing_display: dict[tuple[int, int], int],
    removed_for_yellow: dict[int, set[tuple[int, int]]],
    old_arms: dict[int, set[tuple[int, int]]],
    red_removed: dict[int, set[tuple[int, int]]],
) -> np.ndarray | None:
    grid = np.full((H, W), BG, dtype=int)
    for color, pts in logical.items():
        for pt in pts:
            if pt in removed_for_yellow[color] or pt in red_removed[color]:
                continue
            if pt in crossings and crossing_display[pt] != color:
                continue
            if grid[pt] not in (BG, color):
                return None
            grid[pt] = color
    for color, pts in old_arms.items():
        for pt in pts:
            if grid[pt] != BG:
                return None
            grid[pt] = color
    return grid


def _yellow_ok(
    working: np.ndarray,
    color: int,
    marker: tuple[int, int],
) -> bool:
    nearest = _pick_visible(_visible_candidates(working, marker), farthest=False)
    farthest = _pick_visible(_visible_candidates(working, marker), farthest=True)
    return nearest is not None and farthest is not None and nearest[2] == color and farthest[2] == color


def _ray_along(
    pts: set[tuple[int, int]],
    pt: tuple[int, int],
    direction: tuple[int, int],
) -> list[tuple[int, int]]:
    ray: list[tuple[int, int]] = []
    r, c = pt
    dr, dc = direction
    while (r + dr, c + dc) in pts:
        r += dr
        c += dc
        ray.append((r, c))
    return ray


def _choose_yellow_markers(
    logical: dict[int, set[tuple[int, int]]],
    paths: dict[int, list[set[tuple[int, int]]]],
    crossings: set[tuple[int, int]],
    working: np.ndarray,
    blocked: set[tuple[int, int]],
    count: int,
) -> list[tuple[int, tuple[int, int], tuple[int, int], set[tuple[int, int]], set[tuple[int, int]]]]:
    chosen = []
    for cand in _yellow_candidates(logical, paths, crossings):
        color, pivot, out_dir, target_ray, old_arm = cand
        marker = (pivot[0] + out_dir[0], pivot[1] + out_dir[1])
        if len(chosen) == count:
            break
        if marker in blocked or any(nb in blocked for nb in _neighbors(marker)):
            continue
        if any(working[pt] != color for pt in target_ray - {pivot}):
            continue
        if any(working[pt] != BG for pt in old_arm):
            continue
        trial = working.copy()
        for pt in target_ray - {pivot}:
            trial[pt] = BG
        for pt in old_arm:
            trial[pt] = color
        trial[marker] = 4
        if not _yellow_ok(trial, color, marker):
            continue
        chosen.append(cand)
        working[:] = trial
        blocked.update(target_ray | old_arm | {marker})
        blocked.update(_neighbors(marker))
    return chosen if len(chosen) == count else []


def _list_red_candidates(
    logical: dict[int, set[tuple[int, int]]],
    working: np.ndarray,
    blocked: set[tuple[int, int]],
    crossings: set[tuple[int, int]],
) -> list[tuple[int, int, tuple[int, int], set[tuple[int, int]]]]:
    candidates: list[tuple[int, int, tuple[int, int], set[tuple[int, int]]]] = []
    for color, pts in logical.items():
        for pt in pts:
            if pt in blocked or pt in crossings or int(working[pt]) != color:
                continue
            if any(nb in blocked for nb in _neighbors(pt)):
                continue
            dirs = _dirs_in(pts, pt)
            if len(dirs) != 2:
                continue
            rays = [_ray_along(pts, pt, direction) for direction in dirs]
            ranges = []
            for ray in rays:
                hi = min(RED_GAP_SIDE_RANGE[1], len(ray) - 1)
                if hi < RED_GAP_SIDE_RANGE[0]:
                    ranges = []
                    break
                ranges.append(range(RED_GAP_SIDE_RANGE[0], hi + 1))
            if not ranges:
                continue
            for dists in itertools.product(*ranges):
                gap: set[tuple[int, int]] = set()
                valid = True
                for ray, dist in zip(rays, dists):
                    arm_gap = ray[: dist - 1]
                    anchor = ray[dist - 1]
                    support = ray[dist]
                    if any(q in blocked or q in crossings for q in arm_gap):
                        valid = False
                        break
                    if any(int(working[q]) != color for q in arm_gap):
                        valid = False
                        break
                    if int(working[anchor]) != color or int(working[support]) != color:
                        valid = False
                        break
                    gap.update(arm_gap)
                if not valid or not gap:
                    continue
                trial = working.copy()
                for q in gap:
                    trial[q] = BG
                trial[pt] = 2
                nearest = _pick_visible(_visible_candidates(trial, pt), farthest=False)
                farthest = _pick_visible(_visible_candidates(trial, pt), farthest=True)
                if nearest is None or farthest is None:
                    continue
                if farthest[2] != color or nearest[2] == color:
                    continue
                score = farthest[0] - nearest[0]
                candidates.append((score, color, pt, gap))
    random.shuffle(candidates)
    candidates.sort(key=lambda x: x[0], reverse=True)
    return candidates[:RED_CANDIDATE_LIMIT]


def _derive_input(
    logical: dict[int, set[tuple[int, int]]],
    paths: dict[int, list[set[tuple[int, int]]]],
    dominant: int,
) -> tuple[np.ndarray, np.ndarray] | None:
    crossings = set(_crossing_cells(logical))
    yellow_count = 0
    red_count = random.randint(*RED_COUNT_RANGE)

    crossing_display = _crossing_display(logical, dominant, crossings)
    if crossing_display is None:
        return None

    removed_for_yellow: dict[int, set[tuple[int, int]]] = {c: set() for c in logical}
    old_arms: dict[int, set[tuple[int, int]]] = {c: set() for c in logical}
    red_removed: dict[int, set[tuple[int, int]]] = {c: set() for c in logical}
    yellow_rays: dict[int, set[tuple[int, int]]] = {c: set() for c in logical}

    working = _render_base_grid(
        logical,
        crossings,
        crossing_display,
        removed_for_yellow,
        old_arms,
        red_removed,
    )
    if working is None:
        return None

    blocked = set(crossings)
    yellow_infos = _choose_yellow_markers(logical, paths, crossings, working, blocked, yellow_count) if yellow_count else []
    if yellow_count and len(yellow_infos) != yellow_count:
        return None

    yellow_markers: list[tuple[int, tuple[int, int]]] = []
    for color, pivot, out_dir, target_ray, old_arm in yellow_infos:
        marker = (pivot[0] + out_dir[0], pivot[1] + out_dir[1])
        removed_for_yellow[color].update(target_ray - {pivot})
        old_arms[color].update(old_arm)
        yellow_rays[color].update(target_ray)
        yellow_markers.append((color, marker))

    red_candidates = _list_red_candidates(logical, working, blocked, crossings)
    if len(red_candidates) < red_count:
        return None

    for combo in itertools.combinations(red_candidates, red_count):
        combo_colors = {color for _score, color, _pt, _gap in combo}
        if len(combo_colors) < 2:
            continue

        combo_removed: dict[int, set[tuple[int, int]]] = {c: set() for c in logical}
        conflict = False
        for _score, color, pt, gap in combo:
            if pt in combo_removed[color] or any(pt in combo_removed[other] for other in combo_removed):
                conflict = True
                break
            for other_score, other_color, other_pt, other_gap in combo:
                if (color, pt, tuple(sorted(gap))) == (other_color, other_pt, tuple(sorted(other_gap))):
                    continue
                if pt in other_gap or other_pt in gap:
                    conflict = True
                    break
            if conflict:
                break
            combo_removed[color].update(gap)
        if conflict:
            continue

        base = _render_base_grid(
            logical,
            crossings,
            crossing_display,
            removed_for_yellow,
            old_arms,
            combo_removed,
        )
        if base is None:
            continue

        inp = base.copy()
        red_infos = [(color, pt, gap) for _score, color, pt, gap in combo]
        for _color, pt in yellow_markers:
            inp[pt] = 4
        for _color, pt, _gap in red_infos:
            inp[pt] = 2

        if not _marker_isolation_ok(inp):
            continue

        yellow_ok = True
        for color, pt in yellow_markers:
            if not any(inp[nb] == color for nb in _neighbors(pt)):
                yellow_ok = False
                break
            nearest = _pick_visible(_visible_candidates(inp, pt), farthest=False)
            farthest = _pick_visible(_visible_candidates(inp, pt), farthest=True)
            if nearest is None or farthest is None or farthest[2] != color:
                yellow_ok = False
                break
        if not yellow_ok:
            continue

        red_effect_count = 0
        for color, pt, _gap in red_infos:
            nearest = _pick_visible(_visible_candidates(inp, pt), farthest=False)
            farthest = _pick_visible(_visible_candidates(inp, pt), farthest=True)
            if nearest is None or farthest is None:
                red_effect_count = -1
                break
            if farthest[2] != color or nearest[2] == color:
                red_effect_count = -1
                break
            red_effect_count += 1
        if red_effect_count < 2:
            continue

        comp_count_masks = {c: _pts(inp == c) for c in logical}
        for color, pt in yellow_markers:
            comp_count_masks[color].add(pt)
        for color, pt, _gap in red_infos:
            comp_count_masks[color].add(pt)

        comp_count = {color: _gen_component_count(pts) for color, pts in comp_count_masks.items()}
        target = _render_target(logical, comp_count, yellow_rays)

        if any(target[pt] == inp[pt] for pt in crossings):
            continue

        colors = sorted(logical)
        counts = {c: len(components_mask(target == c, connectivity=4)) for c in colors}
        if not all(COMPONENTS_PER_COLOR[0] <= v <= COMPONENTS_PER_COLOR[1] for v in counts.values()):
            continue
        if not TOTAL_COMPONENTS[0] <= sum(counts.values()) <= TOTAL_COMPONENTS[1]:
            continue
        if not FILL_RANGE[0] <= int(np.count_nonzero(target)) <= FILL_RANGE[1]:
            continue

        try:
            solved = main(inp)
        except Exception:
            continue
        if np.array_equal(solved, target):
            return inp, target

    return None


_ORIGINAL_MODULE = None


def _original_module():
    global _ORIGINAL_MODULE
    if _ORIGINAL_MODULE is None:
        spec = importlib.util.spec_from_file_location(
            "faa9f03d_original",
            "/home/lixinyu/snap/new/arc2_solver_select/faa9f03d.py",
        )
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)
        _ORIGINAL_MODULE = module
    return _ORIGINAL_MODULE


def _d4_effect_count(inp: np.ndarray, out: np.ndarray | None = None) -> int:
    out = main(inp) if out is None else out
    count = 0
    for pt in map(tuple, np.argwhere(inp == 2)):
        target_color = int(out[pt])
        nearest = _pick_visible(_visible_candidates(inp, pt), farthest=False)
        farthest = _pick_visible(_visible_candidates(inp, pt), farthest=True)
        if nearest is None or farthest is None:
            continue
        if farthest[2] == target_color and nearest[2] != target_color:
            count += 1
    return count


def _gen_attempt() -> np.ndarray | None:
    orig = _original_module()
    inp = orig.generate_input()
    try:
        out = main(inp)
    except Exception:
        return None
    return inp if _d4_effect_count(inp, out) >= 2 else None


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("failed to generate a valid faa9f03d D4 input")
