# Variant of 89565ca0
# Dimension: D7 — 变换：房间数到条长的映射 (transform)
# Change: 直接映射：色条长度 = 封闭区域数量... → 长度 = 最大房间数 - 当前房间数 + 1（取反映射）

from __future__ import annotations

import math
import random

import numpy as np

from arc_utils import bbox, colors_present, crop_bbox, get_bg, label_mask


# Design analysis for the D7 variant
# ----------------------------------
# variant_spec:
#   preserved_input:
#     - 输入仍是黑底上的多个彩色矩形线框，每个线框由水平/垂直墙构成，噪声色会散落在背景和墙上。
#     - 每个真实线框经重建后仍对应一个 1..6 的封闭房间数；房间数升序排序、同房间数仍按覆盖关系打破平局。
#   changed_output:
#     - 原题输出第 r 行的彩条长度等于该线框的房间数 rooms。
#     - 变体改为先求所有真实线框的最大房间数 max_rooms，再输出长度
#       inv_len = max_rooms - rooms + 1。
#     - 因而房间最多的框架变成最短条，房间最少的框架变成最长条；排序不变，只反转“计数到长度”的映射。
#   behavior_trace:
#     - 例：三个框房间数分别为 [1, 3, 4]，升序后颜色顺序不变。
#     - 原题条长为 [1, 3, 4]。
#     - 本变体中 max_rooms = 4，所以条长变为 [4, 2, 1]。
#     - 差异只发生在输出阶段；输入构造、墙重建、噪声识别、覆盖平局规则都保持原状。
#   complexity_calibration:
#     - 不引入新的对象类别，也不依赖颜色 id、绝对位置等不可见信号。
#     - 反转映射只依赖同一张图里对象之间“谁房间更多”的相对关系，人类可从 2-4 个样例中观察到：
#       排在上面的少房间框反而得到更长条，排在下面的多房间框反而更短。
#     - 为避免退化成原题，生成器仍随机产生 3-5 个框且常含多个不同房间数；在这些情况下新旧输出必然不同。
#   implicit_assumptions_review:
#     - colors_present / _room_count_for_color / _walls_for_color 只假设每个非背景色要么是一个可重建线框，要么是噪声色；
#       D7 不改变这一点。
#     - drawings = rooms > 0, pad_colors = rooms == 0 依赖“恰有一种噪声色没有封闭区域”；生成器继续保证此条件。
#     - _cover_order 只在同房间数组内使用，依赖有真实像素重叠时可从可见颜色判断上下层；D7 不影响。
#     - 原题 `out[r, :rooms] = color` 隐含“条长直接等于 rooms”；这里必须替换为反向长度，
#       否则 solver 会悄悄退化回原题。
#     - 原题 `width = max(rooms...)` 隐含“最大条长就是最大房间数”；变体中应改为
#       `max(inverted_lengths)`，虽然当最小房间数为 1 时两者数值相同，但语义已不同。
#   placement_validity:
#     - 生成器仍沿用原来的 output-first / oracle-check 模式：先随机房间数、建框、放置、制造重叠和噪声，再用 main 校验。
#     - 由于 D7 不改输入几何，原放置合法性约束仍足够：框不越界，允许 0-3 对重叠，墙保留率足以重建。
#   visual_observability:
#     - 输出中每一行的颜色顺序和原题一致，但条长与房间数方向相反，这是直接可见的主要线索。
#     - 生成器不固定某一种房间数组合，因此少房间长条 / 多房间短条会在不同样例中重复出现。
#   cleanliness:
#     - 不增加额外标记，不改变输入噪声密度和叠放风格，避免在视觉上引入与 D7 无关的新混乱。


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


def _inverse_bar_length(rooms: int, max_rooms: int) -> int:
    return int(max_rooms - rooms + 1)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Convert each reconstructed frame into a horizontal bar whose length is inversely related to its room count within that scene.

    Input structure:
    - Black background containing several colored rectilinear frame drawings plus one scattered non-background noise color.
    - Real drawings may overlap each other and may lose some wall pixels to the noise color.
    - Each true drawing is still a rectangular wall layout enclosing one or more background rooms after reconstruction.

    Rule:
    - For every non-background color, rebuild candidate horizontal and vertical wall segments from dense row/column evidence and count its enclosed 4-connected background regions.
    - The color whose reconstruction encloses zero rooms is the padding/noise color; all other colors are genuine frames.
    - Order genuine frames by increasing room count, breaking ties by bottom-to-top cover order among overlapping reconstructions.
    - Let max_rooms be the largest room count among the ordered frames. Output each row as that frame's color repeated `max_rooms - rooms + 1` times, then pad the rest of the row with the noise color.

    Key insight:
    - The perceptual challenge is unchanged: recover each frame's room count despite overlap and damage. The variant only flips how that recovered count is turned into bar length, so more rooms now mean a shorter bar rather than a longer one.
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
    ordered: list[tuple[int, int, np.ndarray]] = []
    for rooms in sorted({rooms for _color, rooms, _walls in drawings}):
        group = [item for item in drawings if item[1] == rooms]
        ordered.extend(_cover_order(g, group))

    max_rooms = max(rooms for _color, rooms, _walls in ordered)
    bar_lengths = [_inverse_bar_length(rooms, max_rooms) for _color, rooms, _walls in ordered]
    width = max(bar_lengths)

    out = np.full((len(ordered), width), pad, dtype=int)
    for r, ((color, _rooms, _walls), bar_len) in enumerate(zip(ordered, bar_lengths)):
        out[r, :bar_len] = int(color)
    return out


# --------------- generator (claude-opus-4-6, adapted for D7) ---------------

# ---- CFG: Tunable generation parameters ----
_FRAME_COUNT_RANGE = (3, 5)           # number of colored frames per puzzle
_ROOM_COUNT_RANGE = (1, 6)            # rooms per frame
_TIE_PROBABILITY = 0.35               # chance of one tied room-count pair
_GRID_SIZE_BY_FRAMES = {              # (h_min, h_max, w_min, w_max, frame_max_h, frame_max_w)
    5: (25, 30, 25, 30, 16, 18),
    4: (22, 30, 22, 30, 18, 22),
    3: (20, 30, 20, 30, 22, 25),
}
_CELL_GAP_RANGE = (3, 13)             # min/max gap between grid lines within a frame
_NOISE_FRACTION_RANGE = (0.04, 0.09)  # fraction of total cells as noise
_WALL_DAMAGE_FRACTION = 0.25          # max fraction of wall pixels replaced by noise
_EXTRA_OVERLAP_RANGE = (0, 3)         # non-tied overlap pairs to force
_OVERLAP_RETRY = 100                  # attempts to place overlapping frame
_GEN_RETRIES = 300                    # outer retry limit


def _random_gaps(n: int, min_gap: int, max_gap: int, max_total: int) -> list[int] | None:
    """Generate n gaps each in [min_gap, max_gap] with sum <= max_total."""
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
    """Build a boolean wall array with exactly target_rooms enclosed regions."""
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
    """Try to reposition frame j so it overlaps frame i. Modifies positions in place."""
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


def _render_frames(frames, positions, render_order, grid_h, grid_w):
    """Render frame walls onto grid; return grid and set of visible frame pixel coords."""
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
    """Place noise pixels on non-wall cells and damage some wall cells."""
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
    max_rooms = max(rc for rc, _cover_rank, _color in frame_info)
    bar_lengths = [_inverse_bar_length(rc, max_rooms) for rc, _cover_rank, _color in frame_info]
    width = max(bar_lengths)
    expected = np.full((len(frame_info), width), noise_color, dtype=int)
    for r, ((rc, _cover_rank, color), bar_len) in enumerate(zip(frame_info, bar_lengths)):
        expected[r, :bar_len] = color
    return expected


def _gen_attempt() -> np.ndarray | None:
    """Single attempt to generate a valid puzzle grid."""
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

    grid, frame_mask = _render_frames(frames, positions, render_order, grid_h, grid_w)
    _scatter_noise(grid, frame_mask, noise_color, grid_h, grid_w)

    try:
        output = main(grid)
    except Exception:
        return None

    frame_info = sorted(
        [(room_counts[i], render_order.index(i), frame_colors[i]) for i in range(frame_count)]
    )
    expected = _expected_output(frame_info, noise_color)

    return grid if np.array_equal(output, expected) else None


def generate_input() -> np.ndarray:
    for _ in range(_GEN_RETRIES):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError(f"Failed to generate valid input after {_GEN_RETRIES} attempts")
