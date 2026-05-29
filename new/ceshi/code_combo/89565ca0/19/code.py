from __future__ import annotations

# Combo variant of 89565ca0 (index 19)
# combo_spec: D4/1+D6/1
# Name: 多色噪声+互补色填充
#
# Design analysis
# =============================================================================
# a) 具体化与行为追踪
# - 原题输入里只有 1 种散落噪声色, 输出每行在真实长度右侧也统一用这 1 种噪声色补齐。
# - 融合后输入里有 2-3 种噪声色, 它们都可能落在框外背景、框内空房间、以及部分墙线上; 输出则不再复用任意
#   噪声色, 而是对每个框架行单独使用固定互补映射 `c -> 9-c` 来填充右侧尾部。
# - 小例子: 若三个框架颜色与房间数分别是 `(2,2)`, `(3,4)`, `(8,1)`, 输入里另外有噪声色 `{5,6}`。
#   原题输出会是:
#     [8,5,5,5]
#     [2,2,5,5]
#     [3,3,3,3]
#   融合后输出改成:
#     [8,1,1,1]
#     [2,2,7,7]
#     [3,3,3,3]
#   差异点有两个: 解题者先要把 2-3 个零房间颜色都识别为噪声, 再要看出每行尾部来自该行主色的互补色。
#
# b) 复杂度校准
# - 多色噪声若放得太少, 会退化成“几乎只有一种噪声”; 放得太多又会破坏墙体重建。生成器固定使用 2-3 色,
#   总噪声仍控制在 4-9%, 且每种噪声至少出现若干像素。
# - 互补映射若允许任意置换, 人类很难从少量样例归纳。这里采用固定且全任务一致的 `c -> 9-c`, 并将框架色
#   限制在 `1..8`, 使互补色也都落在非黑色集合中。
# - 为了让 D6/1 肉眼可见, 每个样例至少有一个非最大房间数的框架, 因而至少一行真的会出现互补尾部填充。
#
# c) 隐式假设审查（逐段针对 main）
# - `rooms > 0` 仍可区分真实框架与噪声色, 但“零房间颜色只有一个”这一旧前提失效, 因此改成要求
#   `2 <= len(noise_colors) <= 3`。
# - 旧版输出依赖单个 `pad` 色初始化整张输出网格; 融合后必须逐行决定尾部颜色, 因而改成每行先填 `9-color`
#   再覆盖左侧 `rooms` 个主色像素。
# - 原始 `_cover_order()` 假设重叠位置能直接看到唯一顶层框架色。多噪声时该点可能被噪声覆盖, 因此采用
#   D4/1 的更稳健版本: 在重叠重建墙区域中比较两种框架色的可见命中数, 并由生成器保护这类证据像素。
# - `get_bg(grid) == 0`、基于 bbox 裁切、基于稠密行列恢复墙体的前提均保持不变; 生成器继续控制墙损伤比例,
#   并用 solver-oracle 回验。
#
# d) 放置合法性
# - 每个框架仍先独立构造成轴对齐 1px 线框, 再在 20-30 尺寸画布内随机放置, 全部都保证不越界。
# - 框架之间允许有限 bbox 重叠, 但噪声采样不会额外制造新的几何对象; 保护的 overlap-evidence 像素不参与
#   墙体损伤, 所以 tied room-count 的上下覆盖关系仍可恢复。
#
# e) 视觉可观察性
# - 输入侧的变化是显眼的: 画面不再只有一个散色, 而是 2-3 种明显不同的离散噪声颜色, 且都不形成房间。
# - 输出侧的变化也显眼: 右侧尾部不再整表同色, 而是随着行首框架色改变而改变, 且遵循固定互补对
#   `(1,8) (2,7) (3,6) (4,5)`。
# - 人类从几组样例可以先看出“左侧长度=房间数”的旧主线, 再注意到“尾部颜色并不是输入噪声色, 而是行首颜色
#   的固定对色”。
#
# f) 可视化整洁度
# - 虽然输入颜色更多, 但噪声仍是稀疏散点, 主体几何仍由少量粗线框组成, 没有新增纹理或高频标记。
# - 输出仍是简单水平条阵列, 只是每行尾色不同, 因此不会造成结构混乱。
#
# g) 变化的可推断性
# - 互补规则不是依赖数值排序或黑色特殊地位的不可见规则, 而是固定 palette 对称 `c -> 9-c`, 在多行上重复出现。
# - 为避免“输出尾色恰好等于输入噪声色”而退化回原题观感, 生成器还要求噪声色同时避开所有框架色及其互补色。
# - 生成器拒绝任何“只体现 D4 而没体现 D6”的退化样本: 会检查互补输出与 D4-only 的统一噪声填充输出确实不同。
# - 同时也拒绝任何“只剩一种零房间颜色”的样本, 确保 D4/1 的多噪声输入特征真实存在。
#
# 新增 solver 假设与 generator 保证
# - 假设 1: 所有真实框架色都在 `1..8`, 因而其互补色 `9-c` 仍是可见非黑色。generator 只从 `1..8` 选框架色。
# - 假设 2: 所有噪声色都重建为 0 个房间, 且数量恰为 2 或 3。generator 对每个样本显式验证。
# - 假设 3: 等房间数的重叠框架在重建墙交集上保留了可判定的上下覆盖证据。generator 保护这些像素不被噪声覆盖。
# - 假设 4: 至少一行存在真实尾部填充, 从而互补规则被激活。generator 通过多框架和非全等房间数配置保证这一点.

import math
import random

import numpy as np

from arc_utils import bbox, colors_present, crop_bbox, get_bg, label_mask


COMPLEMENT_MAP = {c: 9 - c for c in range(10)}
FRAME_COLOR_POOL = list(range(1, 9))
NOISE_COLOR_POOL = list(range(1, 10))


def _comp(color: int) -> int:
    return int(COMPLEMENT_MAP[int(color)])


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
            overlap_vals = g[overlap]
            hits_a = int(np.count_nonzero(overlap_vals == int(ca)))
            hits_b = int(np.count_nonzero(overlap_vals == int(cb)))
            assert hits_a != hits_b
            assert hits_a > 0 or hits_b > 0
            top = int(ca) if hits_a > hits_b else int(cb)
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


def _solve(grid: np.ndarray, *, use_complement_fill: bool) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    bg = int(get_bg(g))
    assert bg == 0

    counts: list[tuple[int, int, np.ndarray]] = []
    for color in colors_present(g, exclude={bg}):
        rooms = _room_count_for_color(g, int(color))
        walls = _walls_for_color(g, int(color))
        counts.append((int(color), int(rooms), walls))

    drawings = [(color, rooms, walls) for color, rooms, walls in counts if rooms > 0]
    noise_colors = [color for color, rooms, _walls in counts if rooms == 0]
    assert drawings
    assert 2 <= len(noise_colors) <= 3

    ordered: list[tuple[int, int, np.ndarray]] = []
    for rooms in sorted({rooms for _color, rooms, _walls in drawings}):
        group = [item for item in drawings if item[1] == rooms]
        ordered.extend(_cover_order(g, group))

    width = max(rooms for _color, rooms, _walls in ordered)
    out = np.zeros((len(ordered), width), dtype=int)
    for r, (color, rooms, _walls) in enumerate(ordered):
        fill_color = _comp(color) if use_complement_fill else int(noise_colors[0])
        out[r, :] = int(fill_color)
        out[r, :rooms] = int(color)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Convert each colored partitioned frame into a horizontal bar whose length is its enclosed-room count, with row-specific complement padding.

    Input structure:
    - Black background with several axis-aligned colored line frames that partition their interiors into 1-6 rooms.
    - The scene also contains 2-3 scattered noise colors, and those noise pixels may appear both on open background cells and on top of visible wall pixels.
    - True frame colors are exactly the non-background colors whose reconstructed wall layouts still enclose at least one region.

    Rule:
    - For every non-background color, reconstruct its rectilinear wall layout from dense row and column evidence and count its enclosed 4-connected regions.
    - Treat the positive-room colors as real frames, sort them by room count, and break equal-count ties by bottom-to-top cover order recovered from overlaps.
    - Output one row per frame: paint the first `rooms` cells with the frame color, then fill the remaining cells in that row with the fixed complement color `9-c` of that same frame color.

    Key insight:
    - The input-side challenge is now to ignore several zero-room distractor colors instead of a single pad color, while the output-side challenge is to notice that trailing cells are not noise at all: each row uses the complement of its own frame color.
    """
    return _solve(grid, use_complement_fill=True)


# --------------- generator (gpt-5) ---------------

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
_NOISE_COLOR_COUNT_RANGE = (2, 3)
_MIN_NOISE_PIXELS_PER_COLOR = 4
_WALL_DAMAGE_FRACTION = 0.25
_DAMAGE_SHARE_RANGE = (0.18, 0.38)
_EXTRA_OVERLAP_RANGE = (0, 3)
_OVERLAP_RETRY = 100
_GEN_RETRIES = 500


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


def _force_overlap(positions, i, j, frames, grid_h, grid_w):
    fi_h, fi_w = frames[i][2].shape
    fj_h, fj_w = frames[j][2].shape
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


def _absolute_wall_masks(frames, positions, grid_h, grid_w) -> list[np.ndarray]:
    masks: list[np.ndarray] = []
    for idx in range(len(frames)):
        mask = np.zeros((grid_h, grid_w), dtype=bool)
        r0, c0 = positions[idx]
        wall_arr = frames[idx][2]
        rs, cs = np.where(wall_arr)
        mask[rs + r0, cs + c0] = True
        masks.append(mask)
    return masks


def _bbox_union_mask(frames, positions, grid_h, grid_w) -> np.ndarray:
    mask = np.zeros((grid_h, grid_w), dtype=bool)
    for idx, (_color, _rooms, wall_arr) in enumerate(frames):
        r0, c0 = positions[idx]
        h, w = wall_arr.shape
        mask[r0 : r0 + h, c0 : c0 + w] = True
    return mask


def _render_frames(frames, positions, render_order, grid_h, grid_w):
    grid = np.zeros((grid_h, grid_w), dtype=int)
    wall_masks = _absolute_wall_masks(frames, positions, grid_h, grid_w)
    order_rank = {idx: pos for pos, idx in enumerate(render_order)}

    for idx in render_order:
        color = frames[idx][0]
        mask = wall_masks[idx]
        grid[mask] = color

    frame_mask = np.zeros((grid_h, grid_w), dtype=bool)
    for idx in render_order:
        color = frames[idx][0]
        visible = wall_masks[idx] & (grid == int(color))
        frame_mask |= visible

    protected_overlap_mask = np.zeros((grid_h, grid_w), dtype=bool)
    for i in range(len(frames)):
        for j in range(i + 1, len(frames)):
            overlap = wall_masks[i] & wall_masks[j]
            if not bool(overlap.any()):
                continue
            top_idx = i if order_rank[i] > order_rank[j] else j
            top_color = int(frames[top_idx][0])
            protected_overlap_mask |= overlap & (grid == top_color)

    bbox_mask = _bbox_union_mask(frames, positions, grid_h, grid_w)
    return grid, frame_mask, protected_overlap_mask, bbox_mask


def _noise_color_counts(total_noise: int, noise_colors: list[int]) -> dict[int, int] | None:
    n = len(noise_colors)
    if total_noise < n * _MIN_NOISE_PIXELS_PER_COLOR:
        return None

    counts = [_MIN_NOISE_PIXELS_PER_COLOR] * n
    remaining = total_noise - n * _MIN_NOISE_PIXELS_PER_COLOR
    for _ in range(remaining):
        counts[random.randrange(n)] += 1
    random.shuffle(counts)
    return {int(color): int(count) for color, count in zip(noise_colors, counts)}


def _sample_coords(coords: np.ndarray, count: int) -> np.ndarray:
    assert count <= len(coords)
    if count == 0:
        return np.empty((0, 2), dtype=int)
    perm = np.random.permutation(len(coords))[:count]
    return coords[perm]


def _scatter_noise(
    grid: np.ndarray,
    frame_mask: np.ndarray,
    protected_overlap_mask: np.ndarray,
    bbox_mask: np.ndarray,
    noise_colors: list[int],
):
    grid_h, grid_w = grid.shape
    total_cells = grid_h * grid_w
    total_noise = int(total_cells * random.uniform(*_NOISE_FRACTION_RANGE))
    counts_by_color = _noise_color_counts(total_noise, noise_colors)
    if counts_by_color is None:
        raise AssertionError("Noise budget too small for multiple visible colors")

    damage_pool = np.argwhere(frame_mask & ~protected_overlap_mask)
    nonwall_pool = np.argwhere(~frame_mask)
    inside_nonwall = np.argwhere(bbox_mask & ~frame_mask)
    outside_nonwall = np.argwhere(~bbox_mask & ~frame_mask)

    max_damage = min(
        total_noise - len(noise_colors),
        int(_WALL_DAMAGE_FRACTION * len(damage_pool)),
        int(total_noise * random.uniform(*_DAMAGE_SHARE_RANGE)),
    )
    max_damage = max(0, max_damage)
    damage_count = min(max_damage, len(damage_pool))
    background_count = total_noise - damage_count
    assert background_count >= len(noise_colors)
    assert background_count <= len(nonwall_pool)

    chosen_background: list[np.ndarray] = []
    reserved = np.zeros((grid_h, grid_w), dtype=bool)

    if len(inside_nonwall) > 0:
        inside_pick = _sample_coords(inside_nonwall, 1)
        chosen_background.append(inside_pick)
        reserved[inside_pick[:, 0], inside_pick[:, 1]] = True
    if len(outside_nonwall) > 0:
        outside_pick = _sample_coords(outside_nonwall, 1)
        if not reserved[outside_pick[:, 0], outside_pick[:, 1]].all():
            chosen_background.append(outside_pick)
            reserved[outside_pick[:, 0], outside_pick[:, 1]] = True

    remaining_background = background_count - sum(len(chunk) for chunk in chosen_background)
    remaining_nonwall = np.argwhere((~frame_mask) & (~reserved))
    chosen_background.append(_sample_coords(remaining_nonwall, remaining_background))
    background_coords = np.concatenate(chosen_background, axis=0)
    assert len(background_coords) == background_count

    damage_coords = _sample_coords(damage_pool, damage_count)
    all_coords = np.concatenate([background_coords, damage_coords], axis=0)
    perm = np.random.permutation(len(all_coords))
    all_coords = all_coords[perm]

    color_sequence: list[int] = []
    for color in noise_colors:
        color_sequence.extend([int(color)] * int(counts_by_color[color]))
    random.shuffle(color_sequence)
    assert len(color_sequence) == len(all_coords)

    assigned = np.zeros((grid_h, grid_w), dtype=bool)
    bg_perm = np.random.permutation(len(background_coords))
    for idx, color in enumerate(noise_colors):
        r, c = background_coords[bg_perm[idx]]
        grid[r, c] = int(color)
        assigned[r, c] = True
        color_sequence.remove(int(color))

    remaining_coords = [tuple(map(int, rc)) for rc in all_coords if not assigned[rc[0], rc[1]]]
    random.shuffle(remaining_coords)
    for (r, c), color in zip(remaining_coords, color_sequence):
        grid[r, c] = int(color)


def _expected_output(frame_info: list[tuple[int, int, int]], width: int) -> np.ndarray:
    out = np.zeros((len(frame_info), width), dtype=int)
    for r, (rooms, _order, color) in enumerate(frame_info):
        out[r, :] = _comp(color)
        out[r, :rooms] = int(color)
    return out


def _pick_frame_and_noise_colors(frame_count: int, noise_count: int) -> tuple[list[int], list[int]] | None:
    if frame_count > len(FRAME_COLOR_POOL):
        return None
    for _ in range(200):
        frame_colors = random.sample(FRAME_COLOR_POOL, frame_count)
        forbidden_noise = set(frame_colors) | {_comp(color) for color in frame_colors}
        available_noise = [c for c in NOISE_COLOR_POOL if c not in forbidden_noise]
        if len(available_noise) < noise_count:
            continue
        noise_colors = random.sample(available_noise, noise_count)
        return frame_colors, noise_colors
    return None


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

    noise_count = random.randint(*_NOISE_COLOR_COUNT_RANGE)
    picked = _pick_frame_and_noise_colors(frame_count, noise_count)
    if picked is None:
        return None
    frame_colors, noise_colors = picked

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

    grid, frame_mask, protected_overlap_mask, bbox_mask = _render_frames(
        frames, positions, render_order, grid_h, grid_w
    )
    try:
        _scatter_noise(grid, frame_mask, protected_overlap_mask, bbox_mask, noise_colors)
    except Exception:
        return None

    try:
        output = main(grid)
    except Exception:
        return None

    frame_info = sorted(
        [(room_counts[i], render_order.index(i), frame_colors[i]) for i in range(frame_count)]
    )
    max_rooms = max(rc for rc, _, _ in frame_info)
    expected = _expected_output(frame_info, max_rooms)

    if not np.array_equal(output, expected):
        return None

    direct_d4_only = _solve(grid, use_complement_fill=False)
    if np.array_equal(output, direct_d4_only):
        return None

    inferred_noise = [
        color for color in colors_present(grid, exclude={0}) if _room_count_for_color(grid, color) == 0
    ]
    if set(inferred_noise) != set(noise_colors):
        return None
    if not (2 <= len(inferred_noise) <= 3):
        return None
    if set(noise_colors) & {_comp(color) for color in frame_colors}:
        return None

    trailing_changed = 0
    trailing_fill_colors: set[int] = set()
    shorter_rows = 0
    for r, (rooms, _order, color) in enumerate(frame_info):
        if rooms < max_rooms:
            trailing = output[r, rooms:]
            if not np.all(trailing == _comp(color)):
                return None
            trailing_changed += int(len(trailing))
            trailing_fill_colors.add(int(_comp(color)))
            shorter_rows += 1
    if trailing_changed == 0:
        return None
    if shorter_rows < 2:
        return None
    if len(trailing_fill_colors) < 2:
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(_GEN_RETRIES):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError(f"Failed to generate valid combo input after {_GEN_RETRIES} attempts")
