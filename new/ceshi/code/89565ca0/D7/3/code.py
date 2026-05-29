# Variant of 89565ca0
# Dimension: D7 — 变换：房间数到条长的映射 (transform)
# Change: 直接映射：色条长度 = 封闭区域数量... → 色条长度 = 框架内的墙壁线段数

from __future__ import annotations

import math
import random

import numpy as np

from arc_utils import bbox, colors_present, crop_bbox, get_bg, label_mask

# Design analysis for the D7 variant
# a) Concretization and behavior trace:
#    The visible input still contains several colored rectilinear frames on black,
#    damaged by one scattered noise color and by frame overlaps. The difference is
#    only in what the output bars measure. A "wall segment" here means one straight
#    horizontal or vertical span between two adjacent frame intersections after the
#    frame has been reconstructed. Example: a 3-vertical-line by 2-horizontal-line
#    frame has 7 segments when intact (top/bottom split into 2 each, plus 3 vertical
#    spans). If its middle vertical divider is removed, the original task outputs a
#    bar of length 1 because there is one room, while this variant outputs a bar of
#    length 6 because six independent wall segments remain.
# b) Complexity calibration:
#    Counting segments is not visually obvious if every sample keeps room order and
#    segment order perfectly aligned. The generator therefore chooses target segment
#    counts directly and rejects easy cases where room-count relations and
#    segment-count relations are indistinguishable across all frame pairs.
# c) Assumption audit against the original main():
#    - Original _room_count_for_color assumed "real drawing iff enclosed rooms > 0".
#      That no longer holds; now we detect real drawings by successfully recovering
#      a frame-like grid with at least four boundary-touching segments.
#    - Original sorting and output width used room counts. Those uses are replaced by
#      segment counts, but the overlap-based tiebreaker still applies unchanged.
#    - Original pad-color detection used room_count == 0. Now it uses segment_count
#      == 0, so the generator must ensure the scattered noise cannot masquerade as a
#      valid frame after reconstruction; output-first validation enforces that.
# d) Placement legality:
#    Frames still stay axis-aligned, in-bounds, and may overlap only by bbox
#    intersection. Ties are forced to overlap so cover order remains observable and
#    solver tiebreaks stay well defined.
# e) Visual observability:
#    The rule stays human-visible because people can count wall pieces directly from
#    the line grids. We also preserve repeated segment counts in some samples so the
#    overlap tiebreaker is still exercised without relying on color ids.
# f) Visual cleanliness:
#    We keep the original sparse black background, 1-pixel walls, and moderate noise.
#    No extra markers are added, so the only new burden is counting wall pieces on
#    the existing frame geometry.
# g) Color inference:
#    Colors are unchanged. They label which frame each output bar belongs to; color
#    ids do not participate in any ordering rule.


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


def _extract_grid_lines(mask: np.ndarray) -> tuple[list[int], list[int]]:
    h, _w = mask.shape
    col_counts = mask.sum(axis=0)
    v_threshold = max(3, math.ceil(0.45 * h))
    vcols = [int(c) for c, n in enumerate(col_counts) if int(n) >= v_threshold]
    if len(vcols) < 2:
        return [], []

    row_counts = mask.sum(axis=1)
    h_threshold = max(3, len(vcols) + 1)
    hrows = [int(r) for r, n in enumerate(row_counts) if int(n) >= h_threshold]
    if len(hrows) < 2:
        return [], []
    return hrows, vcols


def _reconstruct_walls(mask: np.ndarray) -> tuple[np.ndarray, list[int], list[int]]:
    hrows, vcols = _extract_grid_lines(mask)
    if len(hrows) < 2 or len(vcols) < 2:
        return np.zeros_like(mask, dtype=bool), hrows, vcols

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

    return walls, hrows, vcols


def _touches_all_bbox_sides(walls: np.ndarray) -> bool:
    return bool(
        walls.shape[0] >= 2
        and walls.shape[1] >= 2
        and walls[0, :].any()
        and walls[-1, :].any()
        and walls[:, 0].any()
        and walls[:, -1].any()
    )


def _count_wall_segments(walls: np.ndarray, hrows: list[int], vcols: list[int]) -> int:
    count = 0
    for r in hrows:
        for c0, c1 in zip(vcols, vcols[1:]):
            if bool(walls[r, c0 : c1 + 1].all()):
                count += 1
    for c in vcols:
        for r0, r1 in zip(hrows, hrows[1:]):
            if bool(walls[r0 : r1 + 1, c].all()):
                count += 1
    return int(count)


def _frame_signature_for_color(g: np.ndarray, color: int) -> tuple[int, np.ndarray, int]:
    b = bbox(g == int(color))
    assert b is not None
    r0, c0, r1, c1 = b
    mask = crop_bbox(g == int(color), b).astype(bool)
    walls, hrows, vcols = _reconstruct_walls(mask)
    if len(hrows) < 2 or len(vcols) < 2 or not _touches_all_bbox_sides(walls):
        return 0, np.zeros_like(g, dtype=bool), 0

    segments = _count_wall_segments(walls, hrows, vcols)
    if segments < 4:
        return 0, np.zeros_like(g, dtype=bool), 0

    full_walls = np.zeros_like(g, dtype=bool)
    full_walls[r0 : r1 + 1, c0 : c1 + 1] = walls
    rooms = _count_enclosed_regions(walls)
    return int(segments), full_walls, int(rooms)


def _cover_order(
    g: np.ndarray, group: list[tuple[int, int, np.ndarray]]
) -> list[tuple[int, int, np.ndarray]]:
    if len(group) == 1:
        return group

    by_color = {int(item[0]): item for item in group}
    above: dict[int, set[int]] = {color: set() for color, _metric, _walls in group}
    indegree: dict[int, int] = {color: 0 for color, _metric, _walls in group}

    for i, (ca, _ma, wa) in enumerate(group):
        for cb, _mb, wb in group[i + 1 :]:
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


def main(grid: np.ndarray) -> np.ndarray:
    """
    Convert each colored partitioned frame into a horizontal bar whose length is its number of reconstructed wall segments.

    Input structure:
    - Black background with several axis-aligned colored line frames.
    - One extra non-background color appears as scattered noise and as damage over some wall pixels.
    - Each real frame is a rectilinear outer box with optional internal dividers, and frames may overlap one another.

    Rule:
    - For each non-background color, recover the frame's horizontal and vertical walls from dense row/column evidence inside that color's bbox.
    - Count the independent straight wall spans between adjacent intersections in the reconstructed grid.
    - The unique color with zero enclosed rooms is the pad/noise color.
    - Sort real frames by enclosed-room count, preserving the original overlap tiebreak, then use wall-segment count only for bar length.

    Key insight:
    - The noisy color should be ignored as geometry, while each true frame still reveals a stable rectilinear skeleton after reconstruction; room count still determines row order, but wall-segment count determines bar length.
    """
    g = np.asarray(grid, dtype=int)
    bg = int(get_bg(g))
    assert bg == 0

    counts: list[tuple[int, int, int, np.ndarray]] = []
    for color in colors_present(g, exclude={bg}):
        segments, walls, rooms = _frame_signature_for_color(g, int(color))
        counts.append((int(color), int(segments), int(rooms), walls))

    drawings = [(color, segments, rooms, walls) for color, segments, rooms, walls in counts if rooms > 0]
    pad_colors = [color for color, _segments, rooms, _walls in counts if rooms == 0]
    assert drawings
    assert len(pad_colors) == 1

    pad = int(pad_colors[0])
    ordered: list[tuple[int, int, int, np.ndarray]] = []
    segments_by_color = {color: segments for color, segments, _rooms, _walls in drawings}
    for rooms in sorted({rooms for _color, _segments, rooms, _walls in drawings}):
        group = [(color, room_count, walls) for color, _segments, room_count, walls in drawings if room_count == rooms]
        for color, room_count, walls in _cover_order(g, group):
            ordered.append((color, segments_by_color[color], room_count, walls))

    width = max(segments for _color, segments, _rooms, _walls in ordered)

    out = np.full((len(ordered), width), pad, dtype=int)
    for r, (color, segments, _rooms, _walls) in enumerate(ordered):
        out[r, :segments] = int(color)
    return out


# --------------- generator (D7 variant) ---------------

# ---- CFG: Tunable generation parameters ----
_FRAME_COUNT_RANGE = (3, 5)
_ROOM_COUNT_RANGE = (1, 6)
_TIE_PROBABILITY = 0.35
_GRID_SIZE_BY_FRAMES = {
    5: (25, 30, 25, 30, 16, 18),
    4: (22, 30, 22, 30, 18, 22),
    3: (20, 30, 20, 30, 22, 25),
}
_CELL_GAP_RANGE = (3, 8)
_NOISE_FRACTION_RANGE = (0.03, 0.06)
_WALL_DAMAGE_FRACTION = 0.12
_EXTRA_OVERLAP_RANGE = (0, 0)
_OVERLAP_RETRY = 100
_GEN_RETRIES = 400
_MAX_SEGMENT_BAR = 15


def _segment_value_catalog() -> tuple[int, ...]:
    values: set[int] = set()
    for n_vcols in range(2, 6):
        for n_hrows in range(2, 5):
            total_cells = (n_vcols - 1) * (n_hrows - 1)
            full_segments = n_hrows * (n_vcols - 1) + n_vcols * (n_hrows - 1)
            for rooms in range(_ROOM_COUNT_RANGE[0], min(_ROOM_COUNT_RANGE[1], total_cells) + 1):
                removed = total_cells - rooms
                values.add(full_segments - removed)
    return tuple(sorted(v for v in values if v <= _MAX_SEGMENT_BAR))


_SEGMENT_COUNT_VALUES = _segment_value_catalog()


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


def _build_frame(
    target_segments: int, max_h: int = 22, max_w: int = 25
) -> tuple[np.ndarray, int] | None:
    possible_configs: list[tuple[int, int, int]] = []
    for n_vcols in range(2, 6):
        for n_hrows in range(2, 5):
            total_cells = (n_vcols - 1) * (n_hrows - 1)
            full_segments = n_hrows * (n_vcols - 1) + n_vcols * (n_hrows - 1)
            removed = full_segments - target_segments
            room_count = total_cells - removed
            if not (_ROOM_COUNT_RANGE[0] <= room_count <= min(_ROOM_COUNT_RANGE[1], total_cells)):
                continue
            if not (0 <= removed <= total_cells - 1):
                continue
            min_gap, _max_gap = _CELL_GAP_RANGE
            if min_gap * (n_hrows - 1) > max_h - 1 or min_gap * (n_vcols - 1) > max_w - 1:
                continue
            possible_configs.append((n_vcols, n_hrows, room_count))

    if not possible_configs:
        return None

    n_vcols, n_hrows, room_count = random.choice(possible_configs)
    total_cells = (n_vcols - 1) * (n_hrows - 1)
    removed_needed = total_cells - room_count
    min_gap, max_gap = _CELL_GAP_RANGE

    row_gaps = _random_gaps(n_hrows - 1, min_gap, max_gap, max_h - 1)
    col_gaps = _random_gaps(n_vcols - 1, min_gap, max_gap, max_w - 1)
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

    internal_segments: list[tuple[str, int, int, int, int, int]] = []
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
    removed = 0
    for seg in internal_segments:
        if removed >= removed_needed:
            break
        if seg[0] == "h":
            _, r, c0, c1, ca, cb = seg
            if union(ca, cb):
                walls[r, c0 + 1 : c1] = False
                removed += 1
        else:
            _, c, r0, r1, ca, cb = seg
            if union(ca, cb):
                walls[r0 + 1 : r1, c] = False
                removed += 1

    if removed < removed_needed:
        return None

    calc_segments = _count_wall_segments(walls, hrow_positions, vcol_positions)
    if calc_segments != target_segments:
        return None
    if _count_enclosed_regions(walls) != room_count:
        return None
    return walls, int(room_count)


def _boxes_intersect(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    return max(a[0], b[0]) <= min(a[2], b[2]) and max(a[1], b[1]) <= min(a[3], b[3])


def _overlap_area(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> int:
    if not _boxes_intersect(a, b):
        return 0
    h = min(a[2], b[2]) - max(a[0], b[0]) + 1
    w = min(a[3], b[3]) - max(a[1], b[1]) + 1
    return int(h * w)


def _frame_bbox(position: tuple[int, int], wall_arr: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = position
    h, w = wall_arr.shape
    return r0, c0, r0 + h - 1, c0 + w - 1


def _place_frames_without_overlap(frames, grid_h: int, grid_w: int):
    positions = [None] * len(frames)
    for idx in sorted(range(len(frames)), key=lambda i: -frames[i][2].size):
        fh, fw = frames[idx][2].shape
        placed = False
        for _ in range(160):
            candidate = (random.randint(0, grid_h - fh), random.randint(0, grid_w - fw))
            cb = _frame_bbox(candidate, frames[idx][2])
            if all(
                positions[j] is None
                or not _boxes_intersect(cb, _frame_bbox(positions[j], frames[j][2]))
                for j in range(len(frames))
            ):
                positions[idx] = candidate
                placed = True
                break
        if not placed:
            return None
    return positions


def _force_corner_overlap(positions, i: int, j: int, frames, grid_h: int, grid_w: int) -> bool:
    ri, ci = positions[i]
    fi_h, fi_w = frames[i][2].shape
    fj_h, fj_w = frames[j][2].shape
    candidates = [
        (ri + fi_h - 1, ci + fi_w - 1),
        (ri + fi_h - 1, ci - fj_w + 1),
        (ri - fj_h + 1, ci + fi_w - 1),
        (ri - fj_h + 1, ci - fj_w + 1),
    ]
    random.shuffle(candidates)
    ib = _frame_bbox(positions[i], frames[i][2])
    for rj, cj in candidates:
        if not (0 <= rj <= grid_h - fj_h and 0 <= cj <= grid_w - fj_w):
            continue
        jb = _frame_bbox((rj, cj), frames[j][2])
        if _overlap_area(ib, jb) != 1:
            continue
        if any(
            k not in {i, j}
            and positions[k] is not None
            and _boxes_intersect(jb, _frame_bbox(positions[k], frames[k][2]))
            for k in range(len(frames))
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


def _metric_relation_differs(room_counts: list[int], segment_counts: list[int]) -> bool:
    n = len(room_counts)
    for i in range(n):
        for j in range(i + 1, n):
            room_rel = (room_counts[i] > room_counts[j]) - (room_counts[i] < room_counts[j])
            seg_rel = (segment_counts[i] > segment_counts[j]) - (segment_counts[i] < segment_counts[j])
            if room_rel != seg_rel:
                return True
    return False


def _gen_attempt() -> np.ndarray | None:
    frame_count = random.randint(*_FRAME_COUNT_RANGE)

    available = list(_SEGMENT_COUNT_VALUES)
    allow_tie = random.random() < _TIE_PROBABILITY and frame_count >= 3
    if allow_tie:
        distinct_needed = frame_count - 1
        chosen = random.sample(available, min(distinct_needed, len(available)))
        if len(chosen) < distinct_needed:
            return None
        segment_counts = chosen + [random.choice(chosen)]
    else:
        if frame_count > len(available):
            return None
        segment_counts = random.sample(available, frame_count)
    random.shuffle(segment_counts)

    chosen_colors = random.sample(list(range(1, 10)), frame_count + 1)
    frame_colors = chosen_colors[:frame_count]
    noise_color = chosen_colors[frame_count]

    cfg = _GRID_SIZE_BY_FRAMES.get(frame_count, _GRID_SIZE_BY_FRAMES[3])
    h_min, h_max, w_min, w_max, f_max_h, f_max_w = cfg
    grid_h = random.randint(max(h_min, h_max - 4), h_max)
    grid_w = random.randint(max(w_min, w_max - 4), w_max)
    max_frame_h = min(f_max_h, grid_h)
    max_frame_w = min(f_max_w, grid_w)

    frames = []
    room_counts: list[int] = []
    for color, target_segments in zip(frame_colors, segment_counts):
        built = _build_frame(target_segments, max_h=max_frame_h, max_w=max_frame_w)
        if built is None:
            return None
        frame_data, room_count = built
        frames.append((color, target_segments, frame_data))
        room_counts.append(room_count)

    if not _metric_relation_differs(room_counts, segment_counts):
        return None

    positions = _place_frames_without_overlap(frames, grid_h, grid_w)
    if positions is None:
        return None

    tied_pairs = [
        (i, j)
        for i in range(frame_count)
        for j in range(i + 1, frame_count)
        if segment_counts[i] == segment_counts[j]
    ]
    for i, j in tied_pairs:
        if not _force_corner_overlap(positions, i, j, frames, grid_h, grid_w):
            return None

    non_tied = [
        (i, j)
        for i in range(frame_count)
        for j in range(i + 1, frame_count)
        if segment_counts[i] != segment_counts[j]
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

    recovered_drawings: list[tuple[int, int, int, int]] = []
    for idx, color in enumerate(frame_colors):
        segments, _walls, rooms = _frame_signature_for_color(grid, int(color))
        if rooms <= 0 or segments <= 0 or segments not in _SEGMENT_COUNT_VALUES:
            return None
        recovered_drawings.append((idx, int(color), int(rooms), int(segments)))

    pad_candidates = []
    for color in colors_present(grid, exclude={0}):
        segments, _walls, _rooms = _frame_signature_for_color(grid, int(color))
        if segments == 0:
            pad_candidates.append(int(color))
    if pad_candidates != [int(noise_color)]:
        return None
    if not any(rooms != segments for _idx, _color, rooms, segments in recovered_drawings):
        return None

    try:
        output = main(grid)
    except Exception:
        return None

    frame_info = sorted(
        (rooms, render_order.index(idx), segments, color)
        for idx, color, rooms, segments in recovered_drawings
    )
    max_segments = max(segments for _rooms, _order_idx, segments, _color in frame_info)
    expected = np.full((frame_count, max_segments), noise_color, dtype=int)
    for r, (_rooms, _order_idx, segments, color) in enumerate(frame_info):
        expected[r, :segments] = color

    return grid if np.array_equal(output, expected) else None


def generate_input() -> np.ndarray:
    for _ in range(_GEN_RETRIES):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError(f"Failed to generate valid input after {_GEN_RETRIES} attempts")
