# Combo variant of 89565ca0 (index 29)
# combo_spec: D6/1+D7/2
# Name: 互补色填充+颜色重映射

from __future__ import annotations

import math
import random
import sys
from pathlib import Path

import numpy as np

_SOLVER_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_SOLVER_DIR) not in sys.path:
    sys.path.append(str(_SOLVER_DIR))

from arc_utils import bbox, colors_present, crop_bbox, get_bg, label_mask


"""
design_analysis:
  a_concretization_and_trace:
    - 原题示例追踪：若三个框架的房间数按升序是 1,3,5，原输出是三行条带，前缀分别用原框架色，
      后缀统一用噪声色补到宽度 5。
    - 融合后同一个输入先做完全相同的房间计数和排序，但第 1/2/3 行不再保留原框架色，而是固定改成
      排名色序列的前 3 个颜色；每行剩余部分再用该新颜色的互补色填满。
    - 具体小例子：排序后若原行色是 [7,8,9]、房间数是 [1,3,5]，原题输出会近似
      [7 5 5 5 5; 8 8 8 5 5; 9 9 9 9 9]；本题改为
      [1 9 9 9 9; 2 2 2 8 8; 3 3 3 3 3]。差异集中在两个环节：条带主色取决于排序位置，
      填充色取决于新主色的互补关系，而不是输入噪声色。
  b_complexity_calibration:
    - 颜色重映射只用固定前缀色序列，控制自由度为 5 档，避免“每题随机颜色表”导致不可归纳。
    - 互补规则固定为 ARC 非黑色上的镜像配对 1<->9, 2<->8, 3<->7, 4<->6, 5<->5。
      生成器刻意不把 5 用作输出色，这样互补总是可见的双色对。
    - 为了让两层变化都肉眼可见，生成器固定输入噪声色为 5，并拒绝“只有一行出现可见填充”以及
      “排序后原框架色恰好等于目标排名色”的退化样本。
  c_implicit_assumption_audit:
    - _room_count_for_color() 仍假设每个真实框架颜色在输入中形成一个可由 bbox 裁切并重建的单个线框；
      颜色变换只发生在输出，输入保持该假设。
    - main() 原先把 rooms==0 的唯一颜色当作输出填充值；融合后这个前提只用于识别噪声色，
      不再用于着色输出，因此 generator 只需继续保证恰有一种零房间颜色存在。
    - _cover_order() 假设同房间数对象若并列，必须靠可见覆盖关系唯一决定顺序；生成器仍强制 tie 对重叠，
      并限制最多一对 tie，维持唯一拓扑序。
    - 新增假设是“输出行数不超过固定排名色序列长度”；生成器把 frame_count 保持在 3-5，
      与长度 5 的排名色序列一致。
  d_placement_legality:
    - 框架仍在 20-30 的网格内放置，所有 bbox 都不越界。
    - 原任务允许合法重叠来表达覆盖顺序；这里保留这种受控重叠，只在生成器指定的 pair 上重定位，
      不引入新的碰撞类型。
    - 每个候选样本都用新 main() 回算校验，失败即丢弃。
  e_visual_observability:
    - 解题者能直接看到：输入里的框架颜色与输出条带颜色脱钩，但输出从上到下总复用同一套位置色序列。
    - 右侧填充不再统一，而是随行色变化，并且总与左侧主色形成固定互补对。
    - 固定灰色噪声只出现在输入，不再出现在输出，能帮助人类排除“还是用噪声色补齐”的旧规则。
  f_visual_cleanliness:
    - 输出每行始终只有两种颜色，且分界点正好是房间数，不引入额外标记。
    - 输入端继续沿用原题的稀疏噪声和有限重叠范围，避免把框架识别本身搅乱。
  g_inferability:
    - 颜色变化由强结构支撑：排序位置决定主色，主色再决定互补填充色。
    - 生成器偏向 4-5 个框架，并要求至少两行短于最大宽度，从而在单个样例内通常能看到多个
      “主色/互补色”对，也能跨样例重复看到前缀位置色序列。
"""


_OUTPUT_COLOR_SEQUENCE = (1, 2, 3, 4, 6)
_INPUT_FRAME_COLOR_POOL = (1, 2, 3, 4, 6, 7, 8, 9)
_NOISE_COLOR = 5
_COMPLEMENT = {
    1: 9,
    2: 8,
    3: 7,
    4: 6,
    5: 5,
    6: 4,
    7: 3,
    8: 2,
    9: 1,
}


def _complement_color(color: int) -> int:
    return int(_COMPLEMENT[int(color)])


def _count_enclosed_regions(walls: np.ndarray) -> int:
    labels, n_labels = label_mask(~walls, connectivity=4)
    h, w = walls.shape
    count = 0
    for k in range(1, n_labels + 1):
        coords = np.argwhere(labels == k)
        assert coords.size > 0
        touches_edge = (
            np.any(coords[:, 0] == 0)
            or np.any(coords[:, 0] == h - 1)
            or np.any(coords[:, 1] == 0)
            or np.any(coords[:, 1] == w - 1)
        )
        if not touches_edge:
            count += 1
    return int(count)


def _reconstruct_walls(mask: np.ndarray) -> np.ndarray:
    h, _w = mask.shape
    col_counts = mask.sum(axis=0)
    v_threshold = max(3, math.ceil(0.45 * h))
    vcols = [int(c) for c, n in enumerate(col_counts) if int(n) >= v_threshold]
    if len(vcols) < 2:
        return mask.copy()

    row_counts = mask.sum(axis=1)
    h_threshold = max(3, len(vcols) + 1)
    hrows = [int(r) for r, n in enumerate(row_counts) if int(n) >= h_threshold]
    assert len(hrows) >= 2

    walls = np.zeros_like(mask, dtype=bool)

    for r in hrows:
        for c0, c1 in zip(vcols, vcols[1:]):
            segment = mask[r, c0 : c1 + 1]
            length = int(c1 - c0 + 1)
            hits = int(segment.sum())
            dense = hits >= max(2, math.ceil(0.45 * length))
            interior_evidence = length > 2 and hits >= 2 and bool(segment[1:-1].any())
            if dense or interior_evidence:
                walls[r, c0 : c1 + 1] = True

    for c in vcols:
        for r0, r1 in zip(hrows, hrows[1:]):
            segment = mask[r0 : r1 + 1, c]
            length = int(r1 - r0 + 1)
            hits = int(segment.sum())
            if hits >= max(2, math.ceil(0.45 * length)):
                walls[r0 : r1 + 1, c] = True

    return walls


def _room_count_for_color(g: np.ndarray, color: int) -> int:
    b = bbox(g == int(color))
    assert b is not None
    mask = crop_bbox(g == int(color), b).astype(bool)
    walls = _reconstruct_walls(mask)
    return _count_enclosed_regions(walls)


def _walls_for_color(g: np.ndarray, color: int) -> np.ndarray:
    b = bbox(g == int(color))
    assert b is not None
    r0, c0, r1, c1 = b
    mask = crop_bbox(g == int(color), b).astype(bool)
    walls = _reconstruct_walls(mask)
    out = np.zeros_like(g, dtype=bool)
    out[r0 : r1 + 1, c0 : c1 + 1] = walls
    return out


def _cover_order(
    g: np.ndarray, group: list[tuple[int, int, np.ndarray]]
) -> list[tuple[int, int, np.ndarray]]:
    if len(group) == 1:
        return group

    by_color = {int(item[0]): item for item in group}
    above: dict[int, set[int]] = {color: set() for color, _rooms, _walls in group}
    indegree: dict[int, int] = {color: 0 for color, _rooms, _walls in group}

    for i, (ca, _ra, wa) in enumerate(group):
        for cb, _rb, wb in group[i + 1 :]:
            overlap = wa & wb
            if not bool(overlap.any()):
                continue
            visible = set(int(v) for v in np.unique(g[overlap]))
            top_colors = visible & {int(ca), int(cb)}
            assert len(top_colors) == 1
            top = top_colors.pop()
            under = int(cb) if top == int(ca) else int(ca)
            if top not in above[under]:
                above[under].add(top)
                indegree[top] += 1

    ordered: list[tuple[int, int, np.ndarray]] = []
    remaining = set(by_color)
    while remaining:
        ready = [color for color in remaining if indegree[color] == 0]
        assert len(ready) == 1
        color = ready[0]
        remaining.remove(color)
        ordered.append(by_color[color])
        for nxt in above[color]:
            indegree[nxt] -= 1
    return ordered


def _sorted_drawings(g: np.ndarray) -> tuple[list[tuple[int, int, np.ndarray]], int]:
    counts: list[tuple[int, int, np.ndarray]] = []
    for color in colors_present(g, exclude={0}):
        rooms = _room_count_for_color(g, int(color))
        walls = _walls_for_color(g, int(color))
        counts.append((int(color), int(rooms), walls))

    drawings = [(color, rooms, walls) for color, rooms, walls in counts if rooms > 0]
    pad_colors = [color for color, rooms, _walls in counts if rooms == 0]
    assert drawings
    assert len(drawings) <= len(_OUTPUT_COLOR_SEQUENCE)
    assert len(pad_colors) == 1

    ordered: list[tuple[int, int, np.ndarray]] = []
    for rooms in sorted({rooms for _color, rooms, _walls in drawings}):
        group = [item for item in drawings if item[1] == rooms]
        ordered.extend(_cover_order(g, group))
    return ordered, int(pad_colors[0])


def _paint_ranked_output(room_counts_in_order: list[int]) -> np.ndarray:
    assert room_counts_in_order
    assert len(room_counts_in_order) <= len(_OUTPUT_COLOR_SEQUENCE)
    width = max(int(v) for v in room_counts_in_order)
    out = np.zeros((len(room_counts_in_order), width), dtype=int)
    for r, rooms in enumerate(room_counts_in_order):
        bar_color = int(_OUTPUT_COLOR_SEQUENCE[r])
        fill_color = _complement_color(bar_color)
        out[r, :] = fill_color
        out[r, : int(rooms)] = bar_color
    return out


def _paint_legacy_output(
    room_counts_in_order: list[int], frame_colors_in_order: list[int]
) -> np.ndarray:
    width = max(int(v) for v in room_counts_in_order)
    out = np.full((len(room_counts_in_order), width), _NOISE_COLOR, dtype=int)
    for r, (rooms, color) in enumerate(zip(room_counts_in_order, frame_colors_in_order)):
        out[r, : int(rooms)] = int(color)
    return out


def _paint_rank_only_output(room_counts_in_order: list[int]) -> np.ndarray:
    width = max(int(v) for v in room_counts_in_order)
    out = np.full((len(room_counts_in_order), width), _NOISE_COLOR, dtype=int)
    for r, rooms in enumerate(room_counts_in_order):
        out[r, : int(rooms)] = int(_OUTPUT_COLOR_SEQUENCE[r])
    return out


def _paint_complement_only_output(
    room_counts_in_order: list[int], frame_colors_in_order: list[int]
) -> np.ndarray:
    width = max(int(v) for v in room_counts_in_order)
    out = np.zeros((len(room_counts_in_order), width), dtype=int)
    for r, (rooms, color) in enumerate(zip(room_counts_in_order, frame_colors_in_order)):
        out[r, :] = _complement_color(int(color))
        out[r, : int(rooms)] = int(color)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reduce each damaged partition frame to a ranked bar, then recolor the bars by rank and fill the suffix with complementary colors.

    Input structure:
    - Black background with several colored axis-aligned line frames, plus one scattered non-frame noise color.
    - Real frames may overlap and may lose some wall pixels to that noise, but each frame still forms a recoverable rectangular wall lattice.
    - Every real frame encloses one or more black rooms; the noise color encloses none.

    Rule:
    - Reconstruct each non-black color's wall grid from dense horizontal and vertical evidence, then count its enclosed 4-connected black regions.
    - Discard the unique zero-room color as noise and sort the true frames by room count, breaking ties by bottom-to-top cover order.
    - Ignore the original frame colors in the output: row k instead uses the k-th fixed ranking color, and the remainder of that row is filled by the complement of that new color.

    Key insight:
    - The input colors matter only long enough to recover which frame has which room count and rank; once that ordering is known, the output colors come from a fixed rank palette plus a deterministic complement map.
    """
    g = np.asarray(grid, dtype=int)
    bg = int(get_bg(g))
    assert bg == 0

    ordered, _noise_color = _sorted_drawings(g)
    room_counts_in_order = [int(rooms) for _color, rooms, _walls in ordered]
    return _paint_ranked_output(room_counts_in_order)


# --------------- generator (Codex) ---------------

# ---- CFG: Tunable generation parameters ----
_FRAME_COUNT_OPTIONS = (3, 4, 4, 5, 5, 5)
_ROOM_COUNT_RANGE = (1, 6)
_TIE_PROBABILITY = 0.5
_GRID_SIZE_BY_FRAMES = {
    5: (25, 30, 25, 30, 16, 18),
    4: (22, 30, 22, 30, 18, 22),
    3: (20, 30, 20, 30, 22, 25),
}
_CELL_GAP_RANGE = (3, 13)
_NOISE_FRACTION_RANGE = (0.04, 0.09)
_WALL_DAMAGE_FRACTION = 0.25
_EXTRA_OVERLAP_RANGE = (0, 3)
_OVERLAP_RETRY = 100
_GEN_RETRIES = 400
_COLOR_ASSIGN_RETRIES = 200


def _random_gaps(
    n: int, min_gap: int, max_gap: int, max_total: int
) -> list[int] | None:
    if n * min_gap > max_total:
        return None
    gaps = [min_gap] * n
    remaining = max_total - sum(gaps)
    for i in range(n):
        extra = min(max_gap - gaps[i], remaining)
        if extra > 0:
            add = random.randint(0, extra)
            gaps[i] += add
            remaining -= add
    random.shuffle(gaps)
    return gaps


def _build_frame(target_rooms: int, max_h: int = 22, max_w: int = 25) -> np.ndarray | None:
    if target_rooms == 1:
        gap_h = random.randint(_CELL_GAP_RANGE[0], max(_CELL_GAP_RANGE[0], max_h - 2))
        gap_w = random.randint(_CELL_GAP_RANGE[0], max(_CELL_GAP_RANGE[0], max_w - 2))
        h, w = gap_h + 2, gap_w + 2
        walls = np.zeros((h, w), dtype=bool)
        walls[0, :] = True
        walls[h - 1, :] = True
        walls[:, 0] = True
        walls[:, w - 1] = True
        return walls

    possible_configs = []
    for nv in range(2, 6):
        for nh in range(2, 5):
            cells = (nv - 1) * (nh - 1)
            if target_rooms <= cells <= target_rooms + 4:
                possible_configs.append((nv, nh, cells))
    if not possible_configs:
        for nv in range(2, 7):
            for nh in range(2, 6):
                cells = (nv - 1) * (nh - 1)
                if cells >= target_rooms:
                    possible_configs.append((nv, nh, cells))
    if not possible_configs:
        return None

    n_vcols, n_hrows, total_cells = random.choice(possible_configs)
    n_row_gaps = n_hrows - 1
    n_col_gaps = n_vcols - 1
    min_gap, max_gap = _CELL_GAP_RANGE

    if min_gap * n_row_gaps > max_h - 1 or min_gap * n_col_gaps > max_w - 1:
        return None

    row_gaps = _random_gaps(n_row_gaps, min_gap, max_gap, max_h - 1)
    col_gaps = _random_gaps(n_col_gaps, min_gap, max_gap, max_w - 1)
    if row_gaps is None or col_gaps is None:
        return None

    vcol_positions = [0]
    for gap in col_gaps:
        vcol_positions.append(vcol_positions[-1] + gap)
    hrow_positions = [0]
    for gap in row_gaps:
        hrow_positions.append(hrow_positions[-1] + gap)

    frame_h = hrow_positions[-1] + 1
    frame_w = vcol_positions[-1] + 1
    walls = np.zeros((frame_h, frame_w), dtype=bool)
    for r in hrow_positions:
        walls[r, :] = True
    for c in vcol_positions:
        walls[:, c] = True

    cells_to_merge = total_cells - target_rooms
    if cells_to_merge == 0:
        return walls

    parent = list(range(total_cells))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x: int, y: int) -> bool:
        px, py = find(x), find(y)
        if px == py:
            return False
        parent[px] = py
        return True

    def cell_id(cr: int, cc: int) -> int:
        return cr * (n_vcols - 1) + cc

    internal_segments = []
    for hi in range(1, n_hrows - 1):
        r = hrow_positions[hi]
        for ci in range(n_vcols - 1):
            c0, c1 = vcol_positions[ci], vcol_positions[ci + 1]
            internal_segments.append(
                ("h", r, c0, c1, cell_id(hi - 1, ci), cell_id(hi, ci))
            )
    for ci in range(1, n_vcols - 1):
        c = vcol_positions[ci]
        for hi in range(n_hrows - 1):
            r0, r1 = hrow_positions[hi], hrow_positions[hi + 1]
            internal_segments.append(
                ("v", c, r0, r1, cell_id(hi, ci - 1), cell_id(hi, ci))
            )

    random.shuffle(internal_segments)
    merged = 0
    for seg in internal_segments:
        if merged >= cells_to_merge:
            break
        if seg[0] == "h":
            _, r, c0, c1, ca, cb = seg
            if find(ca) != find(cb):
                union(ca, cb)
                walls[r, c0 + 1 : c1] = False
                merged += 1
        else:
            _, c, r0, r1, ca, cb = seg
            if find(ca) != find(cb):
                union(ca, cb)
                walls[r0 + 1 : r1, c] = False
                merged += 1

    if merged < cells_to_merge:
        return None
    if _count_enclosed_regions(walls) != target_rooms:
        return None
    return walls


def _force_overlap(positions, i, j, frames, grid_h, grid_w) -> None:
    fi_h, fi_w = frames[i][1].shape
    fj_h, fj_w = frames[j][1].shape
    ri, ci = positions[i]
    r0_min = max(0, ri - fj_h + 2)
    r0_max = min(grid_h - fj_h, ri + fi_h - 2)
    c0_min = max(0, ci - fj_w + 2)
    c0_max = min(grid_w - fj_w, ci + fi_w - 2)
    if r0_min > r0_max or c0_min > c0_max:
        return
    for _ in range(_OVERLAP_RETRY):
        rj = random.randint(r0_min, r0_max)
        cj = random.randint(c0_min, c0_max)
        if (
            max(ri, rj) <= min(ri + fi_h - 1, rj + fj_h - 1)
            and max(ci, cj) <= min(ci + fi_w - 1, cj + fj_w - 1)
        ):
            positions[j] = (rj, cj)
            return


def _render_frames(frames, positions, render_order, grid_h, grid_w):
    grid = np.zeros((grid_h, grid_w), dtype=int)
    for idx in render_order:
        r0, c0 = positions[idx]
        wall_arr = frames[idx][1]
        color = frames[idx][2]
        rs, cs = np.where(wall_arr)
        grid[rs + r0, cs + c0] = color

    frame_mask = np.zeros((grid_h, grid_w), dtype=bool)
    for idx in render_order:
        r0, c0 = positions[idx]
        wall_arr = frames[idx][1]
        color = frames[idx][2]
        rs, cs = np.where(wall_arr)
        abs_rs, abs_cs = rs + r0, cs + c0
        visible = grid[abs_rs, abs_cs] == color
        frame_mask[abs_rs[visible], abs_cs[visible]] = True

    return grid, frame_mask


def _scatter_noise(grid, frame_mask, noise_color, grid_h, grid_w) -> None:
    total_cells = grid_h * grid_w
    noise_fraction = random.uniform(*_NOISE_FRACTION_RANGE)
    noise_count = int(total_cells * noise_fraction)

    nonwall_rs, nonwall_cs = np.where(~frame_mask)
    perm = np.random.permutation(len(nonwall_rs))
    n_nonwall = min(noise_count, len(perm))
    sel = perm[:n_nonwall]
    grid[nonwall_rs[sel], nonwall_cs[sel]] = noise_color
    placed = n_nonwall

    wall_rs, wall_cs = np.where(frame_mask)
    n_wall = len(wall_rs)
    max_damage = min(noise_count - placed, int(_WALL_DAMAGE_FRACTION * n_wall))
    if max_damage > 0:
        damage_perm = np.random.permutation(n_wall)[:max_damage]
        grid[wall_rs[damage_perm], wall_cs[damage_perm]] = noise_color


def _ordered_indices_from_render(room_counts: list[int], render_order: list[int]) -> list[int]:
    render_rank = {idx: pos for pos, idx in enumerate(render_order)}
    ordered = []
    for rooms in sorted(set(room_counts)):
        group = [i for i, count in enumerate(room_counts) if count == rooms]
        group.sort(key=lambda idx: render_rank[idx])
        ordered.extend(group)
    return ordered


def _assign_frame_colors(ordered_indices: list[int]) -> list[int] | None:
    frame_count = len(ordered_indices)
    target_colors = list(_OUTPUT_COLOR_SEQUENCE[:frame_count])
    for _ in range(_COLOR_ASSIGN_RETRIES):
        colors_in_order = random.sample(list(_INPUT_FRAME_COLOR_POOL), frame_count)
        if all(
            int(colors_in_order[pos]) != int(target_colors[pos])
            for pos in range(frame_count)
        ):
            assigned = [0] * frame_count
            for pos, idx in enumerate(ordered_indices):
                assigned[idx] = int(colors_in_order[pos])
            return assigned
    return None


def _gen_attempt() -> np.ndarray | None:
    frame_count = random.choice(_FRAME_COUNT_OPTIONS)
    assert frame_count <= len(_OUTPUT_COLOR_SEQUENCE)

    available = list(range(_ROOM_COUNT_RANGE[0], _ROOM_COUNT_RANGE[1] + 1))
    allow_tie = random.random() < _TIE_PROBABILITY and frame_count >= 3
    if allow_tie:
        distinct_needed = frame_count - 1
        if distinct_needed > len(available):
            return None
        chosen = random.sample(available, distinct_needed)
        room_counts = chosen + [random.choice(chosen)]
    else:
        if frame_count > len(available):
            return None
        room_counts = random.sample(available, frame_count)
    random.shuffle(room_counts)

    max_rooms = max(room_counts)
    if sum(int(v) < max_rooms for v in room_counts) < 2:
        return None

    cfg = _GRID_SIZE_BY_FRAMES.get(frame_count, _GRID_SIZE_BY_FRAMES[3])
    h_min, h_max, w_min, w_max, f_max_h, f_max_w = cfg
    grid_h = random.randint(h_min, h_max)
    grid_w = random.randint(w_min, w_max)

    frames: list[tuple[int, np.ndarray, int]] = []
    for target_rooms in room_counts:
        frame_data = _build_frame(target_rooms, max_h=f_max_h, max_w=f_max_w)
        if frame_data is None:
            return None
        fh, fw = frame_data.shape
        if fh > grid_h - 2 or fw > grid_w - 2:
            return None
        frames.append((int(target_rooms), frame_data, -1))

    positions = [None] * frame_count
    for idx in sorted(range(frame_count), key=lambda i: -frames[i][1].size):
        fh, fw = frames[idx][1].shape
        positions[idx] = (random.randint(0, grid_h - fh), random.randint(0, grid_w - fw))

    tied_pairs = [
        (i, j)
        for i in range(frame_count)
        for j in range(i + 1, frame_count)
        if room_counts[i] == room_counts[j]
    ]
    for i, j in tied_pairs:
        _force_overlap(positions, i, j, frames, grid_h, grid_w)

    non_tied = [
        (i, j)
        for i in range(frame_count)
        for j in range(i + 1, frame_count)
        if room_counts[i] != room_counts[j]
    ]
    n_extra = random.randint(*_EXTRA_OVERLAP_RANGE)
    if n_extra > 0 and non_tied:
        random.shuffle(non_tied)
        for i, j in non_tied[: min(n_extra, len(non_tied))]:
            _force_overlap(positions, i, j, frames, grid_h, grid_w)

    render_order = list(range(frame_count))
    random.shuffle(render_order)
    ordered_indices = _ordered_indices_from_render(room_counts, render_order)
    frame_colors = _assign_frame_colors(ordered_indices)
    if frame_colors is None:
        return None

    frames = [
        (int(room_counts[idx]), frames[idx][1], int(frame_colors[idx]))
        for idx in range(frame_count)
    ]
    grid, frame_mask = _render_frames(frames, positions, render_order, grid_h, grid_w)
    _scatter_noise(grid, frame_mask, _NOISE_COLOR, grid_h, grid_w)

    try:
        output = main(grid)
    except Exception:
        return None

    room_counts_in_order = [int(room_counts[idx]) for idx in ordered_indices]
    frame_colors_in_order = [int(frame_colors[idx]) for idx in ordered_indices]
    expected = _paint_ranked_output(room_counts_in_order)
    legacy = _paint_legacy_output(room_counts_in_order, frame_colors_in_order)
    rank_only = _paint_rank_only_output(room_counts_in_order)
    complement_only = _paint_complement_only_output(
        room_counts_in_order, frame_colors_in_order
    )
    if (
        np.array_equal(output, legacy)
        or np.array_equal(output, rank_only)
        or np.array_equal(output, complement_only)
    ):
        return None

    visible_fills = {
        int(expected[r, rooms])
        for r, rooms in enumerate(room_counts_in_order)
        if int(rooms) < expected.shape[1]
    }
    if len(visible_fills) < 2:
        return None

    return grid if np.array_equal(output, expected) else None


def generate_input() -> np.ndarray:
    for _ in range(_GEN_RETRIES):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError(f"Failed to generate valid input after {_GEN_RETRIES} attempts")
