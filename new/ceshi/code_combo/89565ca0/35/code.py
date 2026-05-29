# Combo variant of 89565ca0 (index 35)
# combo_spec: D5/3+D6/0+D7/0
# Name: 点阵编码+黑色填充+双倍长度

from __future__ import annotations

import math
import random
import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[4]
if str(_ROOT) not in sys.path:
    sys.path.append(str(_ROOT))

from arc_utils import bbox, colors_present, crop_bbox, get_bg, label_mask


"""
design_analysis:
  a_concretization_and_trace:
    - 原题里，一个房间数为 3 的框架会输出成长度 3 的连续色条，右边再用噪声色补齐。
    - D7 先把“应显示的长度”从 3 改成 6；D5 再把这 6 个连续格改成 6 个等间距点；
      D6 最后把所有空位统一改成黑色，而不是噪声色。
    - 所以同一个例子在融合后不再是 [c c c p p p]，而会变成
      [c 0 c 0 c 0 c 0 c 0 c]：共有 6 个同色点，列坐标固定为 0,2,4,6,8,10。
    - 输入侧逻辑不变：仍然先重建每个彩色线框、统计封闭房间数、丢掉唯一的零房间噪声色、
      然后按房间数升序并用覆盖关系打破并列。
  b_complexity_calibration:
    - 只保留一个全局倍率 2，以及一个固定间距 1 的点阵模板，自由度很少，便于从 2-4 个样例归纳。
    - 视觉差异足够明显：连续条带消失，输出改成稀疏点列，而且总点数始终比原题多一倍。
    - 最大输出宽度从原题的 max_rooms 变成 4 * max_rooms - 1，上限 23，仍然远小于输入网格尺度，
      不会造成拥挤或和别的编码混淆。
  c_implicit_assumption_audit:
    - _room_count_for_color() 和 _walls_for_color() 只依赖输入中的线框几何，不依赖输出编码；融合后仍成立。
    - rooms == 0 仍被用来识别唯一噪声色，但该颜色不再参与输出填充；因此 generator 只需继续保证
      恰有一种非黑色会重建出 0 个房间。
    - _cover_order() 仍要求同房间数对象能通过真实重叠关系得到唯一拓扑序；生成器继续只允许最多一组 tie，
      并强制 tie 对重叠，维持该假设。
    - 原 main() 中真正依赖“长度 == 房间数、填充色 == 噪声色”的只有输出宽度和逐行写出逻辑。
      融合后必须同时改成“点数 == 2 * 房间数、背景 == 黑色”，否则会退化回单独某一维的旧行为。
  d_placement_legality:
    - 输入生成仍沿用原题的放置约束：所有框架完整落在 20-30 的黑底画布内，可受控重叠但不越界。
    - 重叠只发生在生成器选择的少量 pair 上；噪声仍只以散点形式覆盖背景或部分墙像素，不引入新对象。
    - 每个候选输入都要经过新 main() 回算并和期望输出逐像素比对，不合法样本直接丢弃。
  e_visual_observability:
    - 人类能直接看到两层变化：同一房间数对应的输出不再是实心条，而是隔一格一个点；同时点数比原题条长多一倍。
    - 黑色空位在行内和行尾同时可见，能支持“背景改黑”这一规律，而不是误以为仍然在用输入噪声色填充。
    - 排序规则保持不变，所以样例之间唯一新增的可学内容就是稀疏点阵和 2 倍缩放，信号集中。
  f_visual_cleanliness:
    - 输出每行只包含一种框架色和黑色，两者职责清晰；没有第三种颜色参与编码。
    - 点阵列坐标固定为偶数列，行与行之间完全独立，不会出现多对象共享同一输出区域的情况。
    - 输入端继续使用原题已经验证过的噪声比例和重叠范围，避免把“能否看出框架”变成新的难点。
  g_inferability:
    - 没有新增颜色映射，颜色仍然一一继承自对应框架，降低推断负担。
    - 关键控制变量只有房间数；房间数越大，点越多，而且永远是 2 倍关系。这个单调关系在少量样例中可重复观察。
    - 生成器天然会产生至少一个非最大行，因此黑色行尾补位可见；而房间数通常包含大于 1 的行，因此稀疏点间隔也可见。
"""


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


def _cover_order(g: np.ndarray, group: list[tuple[int, int, np.ndarray]]) -> list[tuple[int, int, np.ndarray]]:
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


def _encoded_dot_count(rooms: int) -> int:
    return 2 * int(rooms)


def _output_width(max_rooms: int) -> int:
    return 2 * _encoded_dot_count(max_rooms) - 1


def _render_dot_rows(rows: list[tuple[int, int]]) -> np.ndarray:
    max_rooms = max(rooms for _color, rooms in rows)
    out = np.zeros((len(rows), _output_width(max_rooms)), dtype=int)
    for r, (color, rooms) in enumerate(rows):
        cols = np.arange(0, 2 * _encoded_dot_count(rooms), 2, dtype=int)
        out[r, cols] = int(color)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Convert each colored partitioned frame into a sparse dot row whose number of dots is twice its enclosed-region count.

    Input structure:
    - Black background with several axis-aligned colored line grids plus one extra non-black scattered noise color.
    - Each real colored object is a rectangular wall framework that encloses one or more black rooms.
    - Noise and overlaps may hide parts of those walls, but enough rectilinear evidence remains to reconstruct them.

    Rule:
    - For each non-black color, rebuild its horizontal and vertical wall segments from dense row and column evidence.
    - Count the 4-connected black regions fully enclosed by the reconstructed walls.
    - Treat the unique color with zero enclosed regions as noise only; exclude it from the output objects.
    - Sort the real frame colors by room count ascending, using cover order only to break ties.
    - For a frame with R rooms, output one row on a black background with 2 x R same-color dots placed at columns
      0, 2, 4, ...; all remaining cells stay black.

    Key insight:
    - The hard part is still recovering each frame's stable room count despite damage and overlap. The fused variant
      changes only the destination code: a room count is first doubled, then shown as an equally spaced dot tally on black.
    """
    g = np.asarray(grid, dtype=int)
    bg = int(get_bg(g))
    assert bg == 0

    counts: list[tuple[int, int, np.ndarray]] = []
    for color in colors_present(g, exclude={bg}):
        rooms = _room_count_for_color(g, int(color))
        walls = _walls_for_color(g, int(color))
        counts.append((int(color), int(rooms), walls))

    drawings = [(color, rooms, walls) for color, rooms, walls in counts if rooms > 0]
    pad_colors = [color for color, rooms, _walls in counts if rooms == 0]
    assert drawings
    assert len(pad_colors) == 1

    ordered: list[tuple[int, int, np.ndarray]] = []
    for rooms in sorted({rooms for _color, rooms, _walls in drawings}):
        group = [item for item in drawings if item[1] == rooms]
        ordered.extend(_cover_order(g, group))

    return _render_dot_rows([(color, rooms) for color, rooms, _walls in ordered])


# --------------- generator (codex) ---------------

# ---- CFG: Tunable generation parameters ----
_FRAME_COUNT_RANGE = (3, 5)
_ROOM_COUNT_RANGE = (1, 6)
_TIE_PROBABILITY = 0.35
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
_GEN_RETRIES = 300


def _random_gaps(n: int, min_gap: int, max_gap: int, max_total: int) -> list[int] | None:
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
            internal_segments.append(("h", r, c0, c1, cell_id(hi - 1, ci), cell_id(hi, ci)))
    for ci in range(1, n_vcols - 1):
        c = vcol_positions[ci]
        for hi in range(n_hrows - 1):
            r0, r1 = hrow_positions[hi], hrow_positions[hi + 1]
            internal_segments.append(("v", c, r0, r1, cell_id(hi, ci - 1), cell_id(hi, ci)))

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


def _force_overlap(
    positions: list[tuple[int, int] | None],
    i: int,
    j: int,
    frames: list[tuple[int, int, np.ndarray]],
    grid_h: int,
    grid_w: int,
) -> None:
    fi_h, fi_w = frames[i][2].shape
    fj_h, fj_w = frames[j][2].shape
    ri, ci = positions[i]
    assert ri is not None and ci is not None
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


def _render_frames(
    frames: list[tuple[int, int, np.ndarray]],
    positions: list[tuple[int, int]],
    render_order: list[int],
    grid_h: int,
    grid_w: int,
) -> tuple[np.ndarray, np.ndarray]:
    grid = np.zeros((grid_h, grid_w), dtype=int)
    for idx in render_order:
        r0, c0 = positions[idx]
        wall_arr = frames[idx][2]
        color = frames[idx][0]
        rs, cs = np.where(wall_arr)
        grid[rs + r0, cs + c0] = color

    frame_mask = np.zeros((grid_h, grid_w), dtype=bool)
    for idx in render_order:
        r0, c0 = positions[idx]
        wall_arr = frames[idx][2]
        color = frames[idx][0]
        rs, cs = np.where(wall_arr)
        abs_rs, abs_cs = rs + r0, cs + c0
        visible = grid[abs_rs, abs_cs] == color
        frame_mask[abs_rs[visible], abs_cs[visible]] = True

    return grid, frame_mask


def _scatter_noise(
    grid: np.ndarray,
    frame_mask: np.ndarray,
    noise_color: int,
    grid_h: int,
    grid_w: int,
) -> None:
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


def _expected_output(frame_info: list[tuple[int, int, int]]) -> np.ndarray:
    rows = [(color, room_count) for room_count, _order_rank, color in frame_info]
    return _render_dot_rows(rows)


def _gen_attempt() -> np.ndarray | None:
    frame_count = random.randint(*_FRAME_COUNT_RANGE)

    available = list(range(_ROOM_COUNT_RANGE[0], _ROOM_COUNT_RANGE[1] + 1))
    allow_tie = random.random() < _TIE_PROBABILITY and frame_count >= 3
    if allow_tie:
        distinct_needed = frame_count - 1
        chosen = random.sample(available, min(distinct_needed, len(available)))
        if len(chosen) < distinct_needed:
            return None
        room_counts = chosen + [random.choice(chosen)]
    else:
        if frame_count > len(available):
            return None
        room_counts = random.sample(available, frame_count)
    random.shuffle(room_counts)

    chosen_colors = random.sample(list(range(1, 10)), frame_count + 1)
    frame_colors = chosen_colors[:frame_count]
    noise_color = chosen_colors[frame_count]

    cfg = _GRID_SIZE_BY_FRAMES.get(frame_count, _GRID_SIZE_BY_FRAMES[3])
    h_min, h_max, w_min, w_max, f_max_h, f_max_w = cfg
    grid_h = random.randint(h_min, h_max)
    grid_w = random.randint(w_min, w_max)

    frames: list[tuple[int, int, np.ndarray]] = []
    for color, target_rooms in zip(frame_colors, room_counts):
        frame_data = _build_frame(target_rooms, max_h=f_max_h, max_w=f_max_w)
        if frame_data is None:
            return None
        fh, fw = frame_data.shape
        if fh > grid_h - 2 or fw > grid_w - 2:
            return None
        frames.append((color, target_rooms, frame_data))

    positions: list[tuple[int, int] | None] = [None] * frame_count
    for idx in sorted(range(frame_count), key=lambda i: -frames[i][2].size):
        fh, fw = frames[idx][2].shape
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

    final_positions = [pos for pos in positions if pos is not None]
    assert len(final_positions) == frame_count

    render_order = list(range(frame_count))
    random.shuffle(render_order)

    grid, frame_mask = _render_frames(frames, final_positions, render_order, grid_h, grid_w)
    _scatter_noise(grid, frame_mask, noise_color, grid_h, grid_w)

    try:
        output = main(grid)
    except Exception:
        return None

    frame_info = sorted(
        (room_counts[i], render_order.index(i), frame_colors[i]) for i in range(frame_count)
    )
    expected = _expected_output(frame_info)
    return grid if np.array_equal(output, expected) else None


def generate_input() -> np.ndarray:
    for _ in range(_GEN_RETRIES):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError(f"Failed to generate valid input after {_GEN_RETRIES} attempts")
