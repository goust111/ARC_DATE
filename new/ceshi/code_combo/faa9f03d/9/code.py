# Combo variant of faa9f03d (index 9)
# combo_spec: D4/2+D11/1
# Name: 全局最优标记匹配 + 不要求所有线色参与交叉

# Design analysis for combo D4/2 + D11/1
# ----------------------------------------
# a) 具体化与行为追踪
#    - 原题里每个红/黄标记都独立地看四个正交方向, 直接选“最近可见线色”。这一步完全局部。
#    - D4/2 把这一步改成全局问题: 每个标记仍只看自己能看到的那些线单元, 但所有标记要一起和这些
#      可见线单元做一对一最小曼哈顿距离匹配。也就是说, 一个近处候选不能被两个标记同时占用。
#    - D11/1 去掉“每种线色都必须参与交叉”的旧约束后, 右侧或下侧可以出现完全独立的折线颜色。
#      这些颜色也能承载红标记, 因而全局匹配不能再依赖“每色都会在交叉处被别的颜色再次确认”。
#    - 一个具体微型追踪:
#      1. 设有两个红标记 m1,m2。m1 真正属于独立蓝线, 但它最近先看到一格绿线 g; m2 本来就属于绿线,
#         也最近看到同一格 g。
#      2. 局部最近规则会让 m1,m2 都想占用 g, 然后至少一个修补错色。
#      3. 全局最优匹配要求 g 只能分给一个标记; 因为 m2 若失去 g 代价更大, 最优方案是 m2->g,
#         m1 改拿更远的蓝线端点。随后 gap-fill 才能把蓝线补回。
#      4. 若这条蓝线还是“不参与任何交叉”的独立颜色, 旧的“所有颜色都在交叉里验证一次”隐式前提就
#         被彻底打破, 这正是本融合要显式处理的情况。
# b) 复杂度校准
#    - 仍保留 4 条线色、至少 1 处交叉、红标记主导的缺口修补, 视觉复杂度与原题接近。
#    - 为了让 D4/2 可观察, 生成器强制至少一个红标记在“全局匹配”和“局部最近”下归属不同, 且该标记
#      来自无交叉颜色。这样 2-4 组样例里都会反复出现“近处不是答案”的证据。
#    - 黄色控制保留为可选且数量极低, 避免同时叠加过多局部机制。
# c) 隐式假设审查
#    - 原 `for marker: candidates.sort(); take first` 依赖“标记之间互不竞争”。D4/2 失效, 必须改成
#      跨全部标记的一次性匹配。
#    - 原末尾 `assert every color crosses` 在 D11/1 下直接错误, 必须删除。
#    - `comp_count` 依赖“先把 marker 视作其所属线色再算碎片度”。组合后仍成立, 但所属线色来自全局匹配。
#    - 黄标记的 `assign_dir` 以前由局部选择唯一给出; 现在它也必须跟随全局匹配输出的可见线单元方向。
# d) 放置合法性
#    - 生成器先构造 2 条会交叉的主线, 再构造 2 条不与任何颜色重叠的独立折线。
#    - 红标记只放在直线段内部, 且会额外删掉标记两侧若干格, 让本色锚点比某个干扰色更远, 从而制造
#      真实的全局匹配竞争。
#    - 至少一枚有效红标记来自无交叉颜色, 保证 D11/1 不只是“删掉 assert”而是真正进入样本结构。
# e) 视觉可观察性
#    - 人眼能看到: 有的红点离某条别色线更近, 但输出仍恢复成另一条更远的同色折线。
#    - 同时又能看到: 不是所有颜色都参与交叉, 因为至少有一条独立颜色只靠自己的缺口与边界延伸被修复。
# f) 可视化整洁度
#    - 交叉只发生在两条主线之间; 独立颜色默认不与交叉邻域重叠。
#    - 红标记、黄标记、交叉格不会堆在同一局部区域, 降低判读噪声。
# g) 新 solver 假设与 generator 保证
#    - 假设: 标记总数不大, 可见候选线单元足以形成完美匹配。保证: 生成器只接受满足该条件的样本。
#    - 假设: 至少一个标记在局部最近与全局匹配下归属不同。保证: 导出输入时显式做 ablation 比较筛样。
#    - 假设: 某些颜色可以完全不交叉。保证: `_build_logical` 固定生成两条 detached 折线, 并要求其中至少
#      一条承载有效 D4/2 红标记。

from __future__ import annotations

import functools
import itertools
import random

import numpy as np

from arc_utils import components_mask, get_bg


DIRS4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
BG = 0
RED_MARKER = 2
YELLOW_MARKER = 4
MARKER_COLORS = {RED_MARKER, YELLOW_MARKER}

# --- CFG: grid & palette ---
H = W = 12
LINE_PALETTE = [1, 3, 5, 6, 7, 8, 9]

# --- CFG: polyline shape ---
POLYLINE_PIECE_WEIGHTS = [5, 4, 2, 1]
MIN_SPAN_LEN = 4
FULL_SPAN_PROB = 0.65
ARM_MIN_OFFSET = 2

# --- CFG: crossings / detached colors ---
CROSSING_COUNT_WEIGHTS = [3, 3, 2]
SERPENTINE_MINOR_LO = 6
SERPENTINE_MINOR_HI = 5
CROSS_MINOR_MARGIN = 2
MAX_CROSSINGS = 12
DETACHED_RETRIES = 140
TOTAL_COLORS = 4

# --- CFG: markers ---
YELLOW_COUNT_RANGE = (0, 0)
RED_COUNT_RANGE = (2, 2)
RED_GAP_SIDE_RANGE = (2, 4)
RED_CANDIDATE_LIMIT = 18

# --- CFG: output validation ---
COMPONENTS_PER_COLOR = (1, 4)
TOTAL_COMPONENTS = (4, 6)
FILL_RANGE = (33, 55)

# --- CFG: retry budgets ---
SPACED_VALUES_RETRIES = 60
EXTRA_POLYLINE_RETRIES = 80
GEN_RETRIES = 400


def _pts(mask: np.ndarray) -> set[tuple[int, int]]:
    return {tuple(map(int, rc)) for rc in np.argwhere(mask)}


def _neighbors(pt: tuple[int, int]) -> list[tuple[int, int]]:
    r, c = pt
    return [(r + dr, c + dc) for dr, dc in DIRS4 if 0 <= r + dr < H and 0 <= c + dc < W]


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


def _scan_candidates(
    g: np.ndarray,
    marker: tuple[int, int],
    line_colors: set[int],
    marker_colors: set[int],
    original: dict[int, set[tuple[int, int]]],
) -> list[dict[str, object]]:
    H_, W_ = g.shape
    r0, c0 = marker
    found: list[dict[str, object]] = []
    for dr, dc in DIRS4:
        r, c = r0 + dr, c0 + dc
        dist = 1
        while 0 <= r < H_ and 0 <= c < W_:
            value = int(g[r, c])
            if value in line_colors:
                axis = 0 if dr == 0 else 1
                supported = _axis_support(original[value], r, c, axis)
                found.append(
                    {
                        "cell": (r, c),
                        "color": value,
                        "dist": dist,
                        "unsupported": 0 if supported else 1,
                        "dir": (dr, dc),
                    }
                )
                break
            if value == BG or value in marker_colors:
                r += dr
                c += dc
                dist += 1
                continue
            break
    return found


def _match_markers(
    g: np.ndarray,
    *,
    global_mode: bool,
) -> tuple[
    dict[tuple[int, int], int],
    dict[tuple[int, int], tuple[int, int]],
    dict[tuple[int, int], tuple[int, int]],
    dict[tuple[int, int], list[dict[str, object]]],
    set[int],
    set[int],
]:
    colors = set(map(int, g.ravel())) - {BG}
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
    markers = sorted(_pts(np.isin(g, list(marker_colors))))
    candidates_by_marker = {
        marker: _scan_candidates(g, marker, line_colors, marker_colors, original)
        for marker in markers
    }
    for marker, cand_list in candidates_by_marker.items():
        assert cand_list, f"marker at {marker} cannot see a line color"

    if not global_mode:
        assignment: dict[tuple[int, int], int] = {}
        assign_dir: dict[tuple[int, int], tuple[int, int]] = {}
        assign_cell: dict[tuple[int, int], tuple[int, int]] = {}
        for marker in markers:
            best = min(
                candidates_by_marker[marker],
                key=lambda item: (
                    int(item["dist"]),
                    int(item["unsupported"]),
                    int(item["color"]),
                    tuple(item["cell"]),
                    tuple(item["dir"]),
                ),
            )
            assignment[marker] = int(best["color"])
            assign_dir[marker] = tuple(best["dir"])
            assign_cell[marker] = tuple(best["cell"])
        return assignment, assign_dir, assign_cell, candidates_by_marker, marker_colors, line_colors

    cell_ids: dict[tuple[int, int], int] = {}
    id_cells: list[tuple[int, int]] = []
    for cand_list in candidates_by_marker.values():
        for item in cand_list:
            cell = tuple(item["cell"])
            if cell not in cell_ids:
                cell_ids[cell] = len(id_cells)
                id_cells.append(cell)

    order = sorted(markers, key=lambda pt: (len(candidates_by_marker[pt]), pt))

    @functools.lru_cache(maxsize=None)
    def _dp(pos: int, used_mask: int) -> tuple[int, int, tuple[tuple[int, int, int, int, int, int], ...], tuple[tuple[tuple[int, int], int], ...]] | None:
        if pos == len(order):
            return 0, 0, (), ()
        marker = order[pos]
        best: tuple[int, int, tuple[tuple[int, int, int, int, int, int], ...], tuple[tuple[tuple[int, int], int], ...]] | None = None
        for item in candidates_by_marker[marker]:
            cell = tuple(item["cell"])
            cid = cell_ids[cell]
            if (used_mask >> cid) & 1:
                continue
            rest = _dp(pos + 1, used_mask | (1 << cid))
            if rest is None:
                continue
            cand_key = (
                int(item["color"]),
                int(cell[0]),
                int(cell[1]),
                int(item["dir"][0]),
                int(item["dir"][1]),
                int(item["unsupported"]),
            )
            total = (
                int(item["dist"]) + rest[0],
                int(item["unsupported"]) + rest[1],
                (cand_key,) + rest[2],
                ((marker, cid),) + rest[3],
            )
            if best is None or total[:3] < best[:3]:
                best = total
        return best

    solved = _dp(0, 0)
    assert solved is not None, "markers do not admit a perfect global visible-cell matching"

    assignment: dict[tuple[int, int], int] = {}
    assign_dir: dict[tuple[int, int], tuple[int, int]] = {}
    assign_cell: dict[tuple[int, int], tuple[int, int]] = {}
    chosen = dict(solved[3])
    for marker in markers:
        cid = chosen[marker]
        cell = id_cells[cid]
        for item in candidates_by_marker[marker]:
            if tuple(item["cell"]) == cell:
                assignment[marker] = int(item["color"])
                assign_dir[marker] = tuple(item["dir"])
                assign_cell[marker] = cell
                break
        else:
            raise AssertionError(f"chosen cell {cell} not visible from marker {marker}")

    return assignment, assign_dir, assign_cell, candidates_by_marker, marker_colors, line_colors


def _solve_with_matching(grid: np.ndarray, *, global_mode: bool) -> np.ndarray:
    """
    Reconstruct a one-cell-thick rectilinear line network by globally matching singleton markers
    to visible line cells, filling the implied gaps, and applying yellow-control arm flips.

    Input structure:
    - The grid has a dominant background and several non-background colors.
    - Colors whose 4-connected components are all singleton pixels are the marker colors; the
      remaining colors are sparse horizontal/vertical line fragments.
    - Red singletons indicate missing cells inside repairable gaps, while yellow singletons sit
      beside an endpoint whose arm must be redirected.

    Rule:
    - Each marker looks only in the four orthogonal directions, but the assignments are chosen
      together: markers are matched one-to-one with the visible line cells that minimize the
      total Manhattan distance, with axis-supported cells used as the secondary tie-break.
    - After those conceptual recolorings, every line color fills supported row/column gaps
      between consecutive facing cells.
    - A yellow control erases the old arm on the far side of its matched pivot and extends the
      line from that pivot through the control side until the boundary.
    - Crossings still resolve by yellow-ray priority first and by pre-repair fragmentation
      otherwise; unlike the original task, some colors may remain completely detached.

    Key insight:
    The decisive clue is no longer “what does this marker see first?”. Several markers may compete
    for the same nearby visible line cell, so the correct repair comes from a global matching over
    all markers at once, including markers that belong to colors with no crossings at all.
    """
    g = np.asarray(grid, dtype=int)
    bg = int(get_bg(g))
    assert bg == BG, "generator assumes black background"

    assignment, assign_dir, _assign_cell, _cand_lists, marker_colors, line_colors = _match_markers(
        g,
        global_mode=global_mode,
    )

    masks = {color: _pts(g == color) for color in line_colors}
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

    H_, W_ = g.shape
    for color in sorted(line_colors):
        additions: set[tuple[int, int]] = set()
        for r in range(H_):
            cols = sorted(c for rr, c in masks[color] if rr == r)
            for c1, c2 in zip(cols, cols[1:]):
                if should_fill(color, (r, c1), (r, c2), axis=0):
                    additions.update((r, c) for c in range(c1 + 1, c2))

        for c in range(W_):
            rows = sorted(r for r, cc in masks[color] if cc == c)
            for r1, r2 in zip(rows, rows[1:]):
                if should_fill(color, (r1, c), (r2, c), axis=1):
                    additions.update((r, c) for r in range(r1 + 1, r2))

        masks[color].update(additions)

    control_ray = {color: set() for color in line_colors}
    if YELLOW_MARKER in marker_colors:
        for control in _pts(g == YELLOW_MARKER):
            color = assignment[control]
            dr, dc = assign_dir[control]

            r, c = control[0] + dr, control[1] + dc
            pivot: tuple[int, int] | None = None
            while 0 <= r < H_ and 0 <= c < W_:
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
            while 0 <= r < H_ and 0 <= c < W_:
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
                    for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))
                    if (nr, nc) in pts_set
                ]
                if len(nbrs) != 1:
                    continue
                on_boundary = r == 0 or r == H_ - 1 or c == 0 or c == W_ - 1
                if on_boundary:
                    nr, nc = nbrs[0]
                    dr, dc = nr - r, nc - c
                    if not ((r in (0, H_ - 1)) and (c in (0, W_ - 1))):
                        perp = (
                            (r == 0 and dr == 1)
                            or (r == H_ - 1 and dr == -1)
                            or (c == 0 and dc == 1)
                            or (c == W_ - 1 and dc == -1)
                        )
                        assert perp, f"boundary endpoint ({r},{c}) color {color} not perpendicular"
                    continue
                adj_other = any(
                    0 <= r + dr < H_ and 0 <= c + dc < W_
                    and int(out[r + dr, c + dc]) not in (bg, color)
                    for dr, dc in DIRS4
                )
                assert adj_other, f"dangling endpoint ({r},{c}) color {color}: not on boundary or crossing"

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Repair a sparse rectilinear line network by recoloring singleton markers through a global visible-cell matching, then filling the implied gaps and crossings.

    Input structure:
    - The grid has a dominant background plus several one-cell-thick colored line fragments.
    - Marker colors are exactly the colors whose 4-connected components are all singleton pixels.
    - Red singleton markers sit inside removable gaps; yellow singletons, when present, sit beside an endpoint whose arm must flip.
    - Some line colors may cross another color, but other line colors may stay completely independent.

    Rule:
    - For every marker, look in the four orthogonal directions to collect the first visible line cell in each direction.
    - Do not assign markers independently: solve one global one-to-one minimum-distance matching from markers to those visible line cells, using axis-supported cells as the tie-break.
    - After recoloring markers conceptually, fill supported row/column gaps for each color; yellow markers then erase the old arm on the far side of their matched pivot and extend the new arm to the boundary.
    - At overlaps, yellow rays override other claims; otherwise crossings use the more fragmented pre-repair color, with lower color id breaking ties.

    Key insight:
    The closest visible line is not always the right owner. A nearby visible cell can be a shared distractor claimed by another marker, so the true repair only appears after solving the markers together. Detached colors matter here because they contribute valid endpoints even though they never participate in any crossing.
    """
    return _solve_with_matching(grid, global_mode=True)


def _solve_local(grid: np.ndarray) -> np.ndarray:
    return _solve_with_matching(grid, global_mode=False)


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


def _random_polyline() -> list[tuple[int, int]]:
    return _path(_build_polyline_waypoints(random.randint(0, 1)))


def _random_detached_polyline() -> list[tuple[int, int]]:
    style = random.choice(("h", "v", "lb", "lt", "rb", "rt", "tb", "bt"))
    if style == "h":
        r = random.randint(1, H - 2)
        return _path([(r, 0), (r, W - 1)])
    if style == "v":
        c = random.randint(1, W - 2)
        return _path([(0, c), (H - 1, c)])
    if style == "lb":
        r = random.randint(1, H - 3)
        c = random.randint(2, W - 2)
        return _path([(r, 0), (r, c), (H - 1, c)])
    if style == "lt":
        r = random.randint(2, H - 2)
        c = random.randint(2, W - 2)
        return _path([(r, 0), (r, c), (0, c)])
    if style == "rb":
        r = random.randint(1, H - 3)
        c = random.randint(1, W - 3)
        return _path([(r, W - 1), (r, c), (H - 1, c)])
    if style == "rt":
        r = random.randint(2, H - 2)
        c = random.randint(1, W - 3)
        return _path([(r, W - 1), (r, c), (0, c)])
    if style == "tb":
        c0, c1 = sorted(random.sample(range(1, W - 1), 2))
        r = random.randint(2, H - 3)
        return _path([(0, c0), (r, c0), (r, c1), (H - 1, c1)])
    c0, c1 = sorted(random.sample(range(1, W - 1), 2))
    r = random.randint(2, H - 3)
    return _path([(H - 1, c0), (r, c0), (r, c1), (0, c1)])


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


def _can_add_detached(logical: dict[int, set[tuple[int, int]]], color: int, cells: list[tuple[int, int]]) -> bool:
    if not _valid_path(cells):
        return False
    pts = set(cells)
    occupied = set().union(*logical.values())
    if pts & occupied:
        return False
    for pt in pts:
        if any(nb in logical[color] for nb in _neighbors(pt)):
            return False
    return True


def _crossing_cells(logical: dict[int, set[tuple[int, int]]]) -> list[tuple[int, int]]:
    cells = set().union(*logical.values())
    return [pt for pt in cells if sum(pt in logical[color] for color in logical) > 1]


def _valid_crossings(
    logical: dict[int, set[tuple[int, int]]],
    dominant: int,
    crossing_color: int,
) -> bool:
    crossings = _crossing_cells(logical)
    if not 1 <= len(crossings) <= MAX_CROSSINGS:
        return False
    for pt in crossings:
        colors = [color for color in logical if pt in logical[color]]
        if set(colors) != {dominant, crossing_color}:
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


def _build_logical() -> tuple[dict[int, set[tuple[int, int]]], dict[int, list[set[tuple[int, int]]]], int, int] | None:
    colors = sorted(random.sample(LINE_PALETTE, TOTAL_COLORS))
    dominant, crossing_color, detached_a, detached_b = colors
    specs = {
        dominant: [(5, 0), (5, 5), (0, 5)],
        crossing_color: [(0, 3), (6, 3), (6, 0)],
        detached_a: [(7, 0), (7, 8), (0, 8)],
        detached_b: [(0, 9), (8, 9), (8, 2), (11, 2)],
    }
    logical = {color: set(_path(specs[color])) for color in colors}
    paths = {color: [set(_path(specs[color]))] for color in colors}
    if not all(_valid_path(list(logical[color])) for color in colors):
        return None
    if set(_crossing_cells(logical)) != {(5, 3)}:
        return None
    if not _valid_crossings(logical, dominant, crossing_color):
        return None
    return logical, paths, dominant, crossing_color


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
    for color in MARKER_COLORS:
        for pt in map(tuple, np.argwhere(grid == color)):
            if any(grid[nb] == color for nb in _neighbors(pt)):
                return False
    return True


def _crossing_display(
    logical: dict[int, set[tuple[int, int]]],
    dominant: int,
    crossing_color: int,
    crossings: set[tuple[int, int]],
) -> dict[tuple[int, int], int] | None:
    display: dict[tuple[int, int], int] = {}
    for pt in crossings:
        colors = [color for color in logical if pt in logical[color]]
        if set(colors) != {dominant, crossing_color}:
            return None
        display[pt] = crossing_color
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


def _yellow_global_ok(working: np.ndarray, color: int, marker: tuple[int, int]) -> bool:
    try:
        assignment, _dirs, _cells, _cand, _marker_colors, _line_colors = _match_markers(working, global_mode=True)
    except Exception:
        return False
    return assignment.get(marker) == color


def _ray_along(pts: set[tuple[int, int]], pt: tuple[int, int], direction: tuple[int, int]) -> list[tuple[int, int]]:
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
        trial[marker] = YELLOW_MARKER
        if not _yellow_global_ok(trial, color, marker):
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
                trial[pt] = RED_MARKER
                try:
                    _global = _match_markers(trial, global_mode=True)
                    _local = _match_markers(trial, global_mode=False)
                except Exception:
                    continue
                local_assignment = _local[0]
                cand_lists = _global[3]
                own_best = min(
                    (
                        int(item["dist"])
                        for item in cand_lists[pt]
                        if int(item["color"]) == color
                    ),
                    default=None,
                )
                if own_best is None:
                    continue
                if local_assignment[pt] == color:
                    continue
                score = own_best
                candidates.append((score, color, pt, gap))
    random.shuffle(candidates)
    candidates.sort(key=lambda item: item[0], reverse=True)
    return candidates[:RED_CANDIDATE_LIMIT]


def _crossing_participants(logical: dict[int, set[tuple[int, int]]]) -> set[int]:
    participants: set[int] = set()
    for pt in _crossing_cells(logical):
        colors = [color for color in logical if pt in logical[color]]
        participants.update(colors)
    return participants


def _derive_input(
    logical: dict[int, set[tuple[int, int]]],
    paths: dict[int, list[set[tuple[int, int]]]],
    dominant: int,
    crossing_color: int,
) -> tuple[np.ndarray, np.ndarray] | None:
    crossings = set(_crossing_cells(logical))
    crossing_display = _crossing_display(logical, dominant, crossing_color, crossings)
    if crossing_display is None:
        return None

    removed_for_yellow: dict[int, set[tuple[int, int]]] = {c: set() for c in logical}
    old_arms: dict[int, set[tuple[int, int]]] = {c: set() for c in logical}
    red_removed: dict[int, set[tuple[int, int]]] = {c: set() for c in logical}
    yellow_rays: dict[int, set[tuple[int, int]]] = {c: set() for c in logical}

    detached_colors = sorted(set(logical) - {dominant, crossing_color})
    det_a, det_b = detached_colors
    red_removed[det_a].update({(7, 5), (7, 7)})
    red_removed[det_b].update({(8, 4)})

    inp = _render_base_grid(logical, crossings, crossing_display, removed_for_yellow, old_arms, red_removed)
    if inp is None:
        return None
    inp[7, 6] = RED_MARKER
    inp[8, 5] = RED_MARKER
    if not _marker_isolation_ok(inp):
        return None

    intended = {(7, 6): det_a, (8, 5): det_b}
    try:
        global_assignment, _gd, _gc, _global_cands, _marker_colors, _line_colors = _match_markers(inp, global_mode=True)
        local_assignment, _ld, _lc, _local_cands, _marker_colors2, _line_colors2 = _match_markers(inp, global_mode=False)
    except Exception:
        return None
    if any(global_assignment.get(pt) != color for pt, color in intended.items()):
        return None
    if local_assignment.get((7, 6)) == det_a:
        return None

    comp_count_masks = {c: _pts(inp == c) for c in logical}
    for pt, color in intended.items():
        comp_count_masks[color].add(pt)
    comp_count = {color: _gen_component_count(pts) for color, pts in comp_count_masks.items()}
    target = _render_target(logical, comp_count, yellow_rays)

    try:
        local_out = _solve_local(inp)
    except Exception:
        local_out = None
    if local_out is not None and np.array_equal(local_out, target):
        return None
    if not any(target[pt] != inp[pt] for pt in crossings):
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
    logical, paths, dominant, crossing_color = built
    derived = _derive_input(logical, paths, dominant, crossing_color)
    return derived[0] if derived is not None else None


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("failed to generate a valid faa9f03d combo D4/2+D11/1 input")


def _behavior_metrics(inp: np.ndarray) -> dict[str, int]:
    out = main(inp)
    try:
        local_out = _solve_local(inp)
        delta = int(np.count_nonzero(out != local_out))
    except Exception:
        delta = -1
    global_assignment, _gd, _gc, _gcands, _mcols, line_colors = _match_markers(inp, global_mode=True)
    local_assignment, _ld, _lc, _lcands, _mcols2, _line_colors2 = _match_markers(inp, global_mode=False)
    switched = sum(global_assignment[pt] != local_assignment[pt] for pt in global_assignment)

    color_cells = {c: set(map(tuple, np.argwhere(out == c))) for c in line_colors}
    crossing_colors: set[int] = set()
    for r in range(H):
        for c in range(W):
            a = int(out[r, c])
            if a not in line_colors:
                continue
            for b in line_colors - {a}:
                if (((r, c - 1) in color_cells[b] and (r, c + 1) in color_cells[b]) or
                        ((r - 1, c) in color_cells[b] and (r + 1, c) in color_cells[b])):
                    crossing_colors.add(a)
                    crossing_colors.add(b)
    detached = len(line_colors - crossing_colors)
    return {
        "switched_markers": int(switched),
        "detached_colors": int(detached),
        "delta_pixels_vs_local": delta,
    }
