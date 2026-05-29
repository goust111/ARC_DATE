# Combo variant of 89565ca0 (index 14)
# combo_spec: D1/4+D7/1
# Name: 降序排列+反向映射

from __future__ import annotations

import math
import random

import numpy as np

from arc_utils import bbox, colors_present, crop_bbox, get_bg, label_mask


# Design analysis for combo D1/4 + D7/1
# --------------------------------------
# combo_summary:
#   - 保持原题输入几何不变：黑底上有 3-5 个彩色矩形线框，线框之间可重叠，并有 1 种非黑噪声色散落在背景和墙上。
#   - 仍先重建每种颜色的矩形墙，再统计每个真实线框的封闭房间数 rooms。
#   - 组合后的输出同时改两处：
#       1. D1/4: 行顺序按 rooms 降序，房间最多的线框排最上面。
#       2. D7/1: 条长改为 inv_len = max_rooms - rooms + 1，房间越多条反而越短。
#
# behavior_trace:
#   - 取一个具体样例，三个真实线框的房间数分别为 [2, 4, 5]。
#   - 原题：按升序排成 [2, 4, 5]，条长直接是 [2, 4, 5]。
#   - 仅 D7/1：顺序仍是 [2, 4, 5]，但条长变为 [4, 2, 1]。
#   - 本融合：先按降序得到 [5, 4, 2]，再映射条长为 [1, 2, 4]。
#   - 可见差异不只是“长度反转”，还包括“颜色行次序反转”；顶行既代表最多房间，又是最短条。
#   - 这个例子也暴露原题一个不再成立的隐式性质：输出宽度不再等于 max_rooms，此时宽度应为 4 而不是 5。
#
# complexity_calibration:
#   - 变体没有引入新对象、隐藏标记或依赖颜色 id 的排序；所有变化都来自可见的房间数比较。
#   - 为避免退化成原题，生成器继续让每题至少有 3 个真实框，且房间数不可能全相等，因此组合输出与原题输出必不同。
#   - ties 仍然最多只出现一对，既保留覆盖关系这个原题线索，又不会让降序排序后的输出难以读懂。
#
# implicit_assumptions_review:
#   - `colors_present(..., exclude={bg})`: 仍假设每个非背景色要么是一个可重建框，要么是唯一噪声色；生成器保持 1 个 noise color。
#   - `_room_count_for_color` / `_walls_for_color`: 仍依赖“每个真实框至少保留足够多的墙像素”；生成器继续限制噪声破坏比例。
#   - `drawings = rooms > 0`, `pad_colors = rooms == 0`: 仍要求恰有一种颜色重建后无封闭区域；生成器不改变这一点。
#   - 同房间数组仍调用 `_cover_order`，其前提是 tied frames 之间存在可判断上下层的墙像素重叠；因此生成器对 tied pair
#     不只要求 bbox overlap，而是显式要求 wall overlap，否则直接放弃该次生成。
#   - 输出阶段不再能写成 `out[r, :rooms] = color`，也不能把宽度写成 `max(rooms)`；必须先按降序排，再按反向长度写条。
#
# placement_validity:
#   - 所有框仍完全放在 20-30 x 20-30 的网格内。
#   - 额外非 tied overlap 仍可有 0-3 对；tied overlap 则强制存在真实墙重叠，保证 solver 的 tie-break 可用。
#   - 最终仍通过 `main()` 作为 oracle 校验，确保输入构造与 solver 的隐式前提一致。
#
# visual_observability:
#   - 人眼在 2-4 个样例中能看见：房间更多的框被排到更上方，但输出条却更短，整列形成自上而下逐渐变长的楔形。
#   - 这种规律只依赖可见结构复杂度，不依赖颜色编号或坐标奇偶等不可见因素。
#
# cleanliness_and_inference:
#   - 输入仍保持原题的整洁风格：只有线框、重叠和单色噪声三种视觉元素。
#   - 输出没有额外标记，只有颜色顺序和条长同时变化；规律集中、样式稳定，便于从少量样例归纳出“降序排列+反向映射”。


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


def _inverse_bar_length(rooms: int, max_rooms: int) -> int:
    return int(max_rooms - rooms + 1)


def _order_drawings_desc(
    g: np.ndarray, drawings: list[tuple[int, int, np.ndarray]]
) -> list[tuple[int, int, np.ndarray]]:
    ordered: list[tuple[int, int, np.ndarray]] = []
    for rooms in sorted({rooms for _color, rooms, _walls in drawings}, reverse=True):
        group = [item for item in drawings if item[1] == rooms]
        ordered.extend(_cover_order(g, group))
    return ordered


def main(grid: np.ndarray) -> np.ndarray:
    """
    Convert each reconstructed frame into a horizontal bar sequence that runs from the most-roomed frame at the top to the least-roomed frame at the bottom, while reversing the room-count-to-length mapping.

    Input structure:
    - A black background contains several colored rectilinear frame drawings plus one extra non-background color used only as scattered noise/padding.
    - Real frames may overlap one another and may have some of their wall pixels replaced by the noise color.
    - Each genuine frame is still a rectangular wall layout that encloses one or more 4-connected background rooms after reconstruction.

    Rule:
    - For each non-background color, rebuild candidate horizontal and vertical wall segments from dense row/column evidence, then count the enclosed background regions.
    - The only color whose reconstructed walls enclose zero rooms is the noise/pad color; every other non-background color is a true frame.
    - Order the true frames by room count in descending order, breaking ties by bottom-to-top cover order among overlapping reconstructions.
    - Let `max_rooms` be the largest recovered room count. Output one row per ordered frame, but draw that row with length `max_rooms - rooms + 1`, padding the remainder with the noise color.

    Key insight:
    - The scene still asks you to recover each frame's room count despite overlap and wall damage, but the output now combines two inversions at once: the richest frame moves to the top while also shrinking to the shortest bar.
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

    pad = int(pad_colors[0])
    ordered = _order_drawings_desc(g, drawings)

    max_rooms = max(rooms for _color, rooms, _walls in ordered)
    bar_lengths = [_inverse_bar_length(rooms, max_rooms) for _color, rooms, _walls in ordered]
    width = max(bar_lengths)

    out = np.full((len(ordered), width), pad, dtype=int)
    for r, ((color, _rooms, _walls), bar_len) in enumerate(zip(ordered, bar_lengths)):
        out[r, :bar_len] = int(color)
    return out


# --------------- generator (combo-adapted) ---------------

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
_GEN_RETRIES = 600


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

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x, y):
        px, py = find(x), find(y)
        if px == py:
            return False
        parent[px] = py
        return True

    def cell_id(cr, cc):
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


def _wall_overlap_exists(
    pos_a: tuple[int, int], wall_a: np.ndarray, pos_b: tuple[int, int], wall_b: np.ndarray
) -> bool:
    ra, ca = pos_a
    rb, cb = pos_b
    ha, wa = wall_a.shape
    hb, wb = wall_b.shape

    r0 = max(ra, rb)
    c0 = max(ca, cb)
    r1 = min(ra + ha, rb + hb)
    c1 = min(ca + wa, cb + wb)
    if r0 >= r1 or c0 >= c1:
        return False

    a_slice = wall_a[r0 - ra : r1 - ra, c0 - ca : c1 - ca]
    b_slice = wall_b[r0 - rb : r1 - rb, c0 - cb : c1 - cb]
    return bool((a_slice & b_slice).any())


def _force_overlap(
    positions, i, j, frames, grid_h, grid_w, require_wall_overlap: bool = False
) -> bool:
    fi_h, fi_w = frames[i][2].shape
    fj_h, fj_w = frames[j][2].shape
    ri, ci = positions[i]
    r0_min = max(0, ri - fj_h + 2)
    r0_max = min(grid_h - fj_h, ri + fi_h - 2)
    c0_min = max(0, ci - fj_w + 2)
    c0_max = min(grid_w - fj_w, ci + fi_w - 2)
    if r0_min > r0_max or c0_min > c0_max:
        return False

    candidates = [(rj, cj) for rj in range(r0_min, r0_max + 1) for cj in range(c0_min, c0_max + 1)]
    random.shuffle(candidates)
    for rj, cj in candidates:
        if require_wall_overlap and not _wall_overlap_exists(
            (ri, ci), frames[i][2], (rj, cj), frames[j][2]
        ):
            continue
        positions[j] = (rj, cj)
        return True
    return False


def _render_frames(frames, positions, render_order, grid_h, grid_w):
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


def _scatter_noise(grid, frame_mask, noise_color, grid_h, grid_w):
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


def _expected_output(frame_info: list[tuple[int, int, int]], noise_color: int) -> np.ndarray:
    ordered = sorted(frame_info, key=lambda item: (-item[0], item[1]))
    max_rooms = max(rc for rc, _cover_rank, _color in ordered)
    bar_lengths = [_inverse_bar_length(rc, max_rooms) for rc, _cover_rank, _color in ordered]
    width = max(bar_lengths)
    expected = np.full((len(ordered), width), noise_color, dtype=int)
    for r, ((_, _cover_rank, color), bar_len) in enumerate(zip(ordered, bar_lengths)):
        expected[r, :bar_len] = color
    return expected


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
    if len(set(room_counts)) < 2:
        return None

    chosen_colors = random.sample(list(range(1, 10)), frame_count + 1)
    frame_colors = chosen_colors[:frame_count]
    noise_color = chosen_colors[frame_count]

    cfg = _GRID_SIZE_BY_FRAMES.get(frame_count, _GRID_SIZE_BY_FRAMES[3])
    h_min, h_max, w_min, w_max, f_max_h, f_max_w = cfg
    grid_h = random.randint(h_min, h_max)
    grid_w = random.randint(w_min, w_max)

    frames = []
    for color, target_rooms in zip(frame_colors, room_counts):
        frame_data = _build_frame(target_rooms, max_h=f_max_h, max_w=f_max_w)
        if frame_data is None:
            return None
        fh, fw = frame_data.shape
        if fh > grid_h - 2 or fw > grid_w - 2:
            return None
        frames.append((color, target_rooms, frame_data))

    positions = [None] * frame_count
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
        if not _force_overlap(positions, i, j, frames, grid_h, grid_w, require_wall_overlap=True):
            return None

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
            _force_overlap(positions, i, j, frames, grid_h, grid_w, require_wall_overlap=False)

    render_order = list(range(frame_count))
    random.shuffle(render_order)

    grid, frame_mask = _render_frames(frames, positions, render_order, grid_h, grid_w)
    _scatter_noise(grid, frame_mask, noise_color, grid_h, grid_w)

    try:
        output = main(grid)
    except Exception:
        return None

    frame_info = [
        (room_counts[i], render_order.index(i), frame_colors[i]) for i in range(frame_count)
    ]
    expected = _expected_output(frame_info, noise_color)

    return grid if np.array_equal(output, expected) else None


def generate_input() -> np.ndarray:
    for _ in range(_GEN_RETRIES):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError(f"Failed to generate valid input after {_GEN_RETRIES} attempts")
