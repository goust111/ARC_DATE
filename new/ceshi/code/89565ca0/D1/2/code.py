# Variant of 89565ca0
# Dimension: D1 — 排序依据：封闭区域数量 (match_cond)
# Change: 按每个框架的封闭4-连通背景区域数量升序排列... → 按框架到网格中心的距离排序

from __future__ import annotations

import math
import random

import numpy as np

from arc_utils import bbox, colors_present, crop_bbox, get_bg, label_mask


# Design analysis
# ---
# variant:
#   source_baseline:
#     original_task: "89565ca0 sorts output rows by room count and uses overlap only as a tie-break."
#     reference_task: "291dc1e1 shows a location-driven ordering rule: compute object positions from the visible grid and sort by distance/proximity."
#   a_concretization_and_trace:
#     input_objects: "The visible objects remain colored rectilinear frames with 1px walls, damaged by one scattered noise color."
#     changed_behavior: "The bar length still equals the number of enclosed rooms, but row order now follows frame-center distance to the whole-grid center."
#     concrete_trace:
#       old: "If red has 2 rooms and blue has 4 rooms, red must appear before blue even when blue is closest to the middle of the board."
#       new: "If blue's bbox center is closest to the board center, blue appears first even if its bar is longer than red's."
#       difference: "Ordering is now spatial rather than histogram-based, so output bar lengths are generally not monotone."
#   b_complexity_calibration:
#     kept: "Frame reconstruction, enclosed-region counting, one noise color, and bar-length encoding are unchanged."
#     adapted: "Forced overlap is removed because it only supported the old tie-break and makes bbox-center reading harder."
#     visibility_guard: "Generator rejects samples whose distance order would also yield nondecreasing room counts, so the new rule is actually visible."
#   c_implicit_assumption_audit:
#     old_main_dependencies:
#       - "Grouping by room count assumed room count was the primary key."
#       - "_cover_order assumed overlaps were informative and needed for ties."
#     new_main_dependencies:
#       - "Each true frame still reconstructs to a non-empty wall mask."
#       - "Reconstructed wall bbox approximates the real frame bbox well enough to recover its center."
#       - "Noise color still has zero enclosed rooms and is therefore identifiable as pad color."
#     generator_response:
#       - "Frames are non-overlapping, so bbox center is not distorted by occlusion."
#       - "The four outer corners of every frame are protected from wall damage, keeping bbox extrema visible."
#   d_placement_legality:
#     policy: "Frames are placed fully inside the grid with at least a 1-cell bbox gap, so they neither overlap nor touch."
#     reason: "This preserves clear object boundaries and keeps center-distance comparisons human-readable."
#   e_visual_observability:
#     human_signal: "Examples show bars whose lengths are scrambled with respect to row order; solvers must notice near-center frames rise to the top."
#     direction_choice: "Ordering is fixed to near-to-far to avoid hidden branch ambiguity."
#   f_visual_cleanliness:
#     cleanup: "No forced overlaps; noise remains sparse; frame count and frame sizes are tuned so multiple frames can coexist clearly."
#   g_color_inference:
#     note: "Colors are not remapped; they only label which frame each output bar belongs to."
#   h_variant_metric:
#     measured_property: "Distance-sorted room sequence should contain at least one inversion, proving the sample is not behaving like the original room-sorted task."
#   i_new_main_assumptions_checked_against_generator:
#     - "Distance ties are avoided by requiring a minimum gap between sorted distances."
#     - "Since tiebreaks should be irrelevant, row/col fallback keys exist only for determinism, not as semantic content."


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


def _frame_sort_features(walls: np.ndarray, shape: tuple[int, int]) -> tuple[float, float, float]:
    b = bbox(walls)
    assert b is not None
    r0, c0, r1, c1 = b
    center_r = (float(r0) + float(r1)) / 2.0
    center_c = (float(c0) + float(c1)) / 2.0
    grid_r = (float(shape[0]) - 1.0) / 2.0
    grid_c = (float(shape[1]) - 1.0) / 2.0
    distance = math.hypot(center_r - grid_r, center_c - grid_c)
    return distance, center_r, center_c


def main(grid: np.ndarray) -> np.ndarray:
    """
    Convert each colored partitioned frame into a horizontal bar whose length is its number of enclosed regions.

    Input structure:
    - Black background with several axis-aligned colored line frames.
    - One extra non-background color appears as scattered noise and may replace some frame-wall pixels.
    - Each real frame still encloses one or more background rooms after rectilinear wall reconstruction.

    Rule:
    - For each non-background color, rebuild the frame walls from dense row/column evidence and count its enclosed 4-connected background regions.
    - The color with zero enclosed regions is the noise/pad color.
    - For every real frame, measure the Euclidean distance from its reconstructed bbox center to the whole-grid center.
    - Output one row per real frame, ordered from nearest to farthest; each row is that frame color repeated by its room count and padded with the noise color.

    Key insight:
    - Room counting still determines bar length, but spatial position now determines row order: the decisive signal is how close each reconstructed frame sits to the center of the board.
    """
    g = np.asarray(grid, dtype=int)
    bg = int(get_bg(g))
    assert bg == 0

    counts: list[tuple[int, int, np.ndarray, float, float, float]] = []
    for color in colors_present(g, exclude={bg}):
        rooms = _room_count_for_color(g, int(color))
        walls = _walls_for_color(g, int(color))
        dist, center_r, center_c = _frame_sort_features(walls, g.shape)
        counts.append((int(color), int(rooms), walls, float(dist), float(center_r), float(center_c)))

    drawings = [item for item in counts if item[1] > 0]
    pad_colors = [color for color, rooms, _walls, _dist, _cr, _cc in counts if rooms == 0]
    assert drawings
    assert len(pad_colors) == 1

    pad = int(pad_colors[0])
    ordered = sorted(drawings, key=lambda item: (item[3], item[4], item[5], item[0]))
    width = max(rooms for _color, rooms, _walls, _dist, _cr, _cc in ordered)

    out = np.full((len(ordered), width), pad, dtype=int)
    for r, (color, rooms, _walls, _dist, _cr, _cc) in enumerate(ordered):
        out[r, :rooms] = int(color)
    return out


# --------------- generator ---------------

# ---- CFG: Tunable generation parameters ----
_FRAME_COUNT_RANGE = (3, 5)
_ROOM_COUNT_RANGE = (1, 6)
_TIE_PROBABILITY = 0.35
_GRID_SIZE_BY_FRAMES = {
    5: (28, 30, 28, 30, 9, 10),
    4: (26, 30, 26, 30, 10, 11),
    3: (24, 30, 24, 30, 11, 12),
}
_CELL_GAP_RANGE = (2, 5)
_NOISE_FRACTION_RANGE = (0.04, 0.09)
_WALL_DAMAGE_FRACTION = 0.18
_MIN_BOX_GAP = 0
_MIN_DISTANCE_GAP = 1.0
_MIN_DISTANCE_SPREAD = 4.0
_PLACEMENT_RETRY = 250
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


def _build_frame(target_rooms: int, max_h: int, max_w: int) -> np.ndarray | None:
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


def _expanded_box(pos: tuple[int, int], shape: tuple[int, int], gap: int) -> tuple[int, int, int, int]:
    r0, c0 = pos
    h, w = shape
    return r0 - gap, c0 - gap, r0 + h - 1 + gap, c0 + w - 1 + gap


def _boxes_overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    return not (a[2] < b[0] or b[2] < a[0] or a[3] < b[1] or b[3] < a[1])


def _place_frames(frames: list[tuple[int, int, np.ndarray]], grid_h: int, grid_w: int) -> list[tuple[int, int]] | None:
    slot_centers = [
        (0.50, 0.50),
        (0.18, 0.64),
        (0.68, 0.82),
        (0.82, 0.22),
        (0.12, 0.12),
    ]
    order = sorted(range(len(frames)), key=lambda i: -frames[i][2].size)

    for _ in range(_PLACEMENT_RETRY):
        chosen_slots = random.sample(slot_centers, len(frames))
        positions: list[tuple[int, int] | None] = [None] * len(frames)
        placed_boxes: list[tuple[int, int, int, int]] = []
        ok = True

        for idx, (slot_r, slot_c) in zip(order, chosen_slots):
            fh, fw = frames[idx][2].shape
            target_r = slot_r * float(grid_h - 1) - (float(fh) - 1.0) / 2.0
            target_c = slot_c * float(grid_w - 1) - (float(fw) - 1.0) / 2.0
            jitter_r = random.randint(-2, 2)
            jitter_c = random.randint(-2, 2)
            r0 = int(round(target_r + jitter_r))
            c0 = int(round(target_c + jitter_c))
            r0 = max(0, min(grid_h - fh, r0))
            c0 = max(0, min(grid_w - fw, c0))
            candidate = _expanded_box((r0, c0), (fh, fw), _MIN_BOX_GAP)
            if any(_boxes_overlap(candidate, prev) for prev in placed_boxes):
                ok = False
                break
            positions[idx] = (r0, c0)
            placed_boxes.append(candidate)

        if ok and all(pos is not None for pos in positions):
            return [(int(r), int(c)) for r, c in positions if r is not None and c is not None]

    return None


def _frame_distance(position: tuple[int, int], walls: np.ndarray, grid_h: int, grid_w: int) -> tuple[float, float, float]:
    r0, c0 = position
    h, w = walls.shape
    center_r = float(r0) + (float(h) - 1.0) / 2.0
    center_c = float(c0) + (float(w) - 1.0) / 2.0
    grid_r = (float(grid_h) - 1.0) / 2.0
    grid_c = (float(grid_w) - 1.0) / 2.0
    distance = math.hypot(center_r - grid_r, center_c - grid_c)
    return distance, center_r, center_c


def _render_frames(
    frames: list[tuple[int, int, np.ndarray]],
    positions: list[tuple[int, int]],
    grid_h: int,
    grid_w: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    grid = np.zeros((grid_h, grid_w), dtype=int)
    frame_mask = np.zeros((grid_h, grid_w), dtype=bool)
    protected_mask = np.zeros((grid_h, grid_w), dtype=bool)

    for idx, (color, _rooms, wall_arr) in enumerate(frames):
        r0, c0 = positions[idx]
        rs, cs = np.where(wall_arr)
        abs_rs = rs + r0
        abs_cs = cs + c0
        grid[abs_rs, abs_cs] = int(color)
        frame_mask[abs_rs, abs_cs] = True

        h, w = wall_arr.shape
        protected_mask[r0, c0 : c0 + w] = True
        protected_mask[r0 + h - 1, c0 : c0 + w] = True
        protected_mask[r0 : r0 + h, c0] = True
        protected_mask[r0 : r0 + h, c0 + w - 1] = True

    return grid, frame_mask, protected_mask


def _scatter_noise(
    grid: np.ndarray,
    frame_mask: np.ndarray,
    protected_mask: np.ndarray,
    noise_color: int,
) -> None:
    grid_h, grid_w = grid.shape
    total_cells = grid_h * grid_w
    noise_fraction = random.uniform(*_NOISE_FRACTION_RANGE)
    noise_count = int(total_cells * noise_fraction)

    nonwall_rs, nonwall_cs = np.where(~frame_mask)
    if len(nonwall_rs) > 0:
        perm = np.random.permutation(len(nonwall_rs))
        n_nonwall = min(noise_count, len(perm))
        sel = perm[:n_nonwall]
        grid[nonwall_rs[sel], nonwall_cs[sel]] = int(noise_color)
    else:
        n_nonwall = 0

    damage_candidates = frame_mask & (~protected_mask)
    wall_rs, wall_cs = np.where(damage_candidates)
    n_wall = len(wall_rs)
    max_damage = min(noise_count - n_nonwall, int(_WALL_DAMAGE_FRACTION * n_wall))
    if max_damage > 0:
        damage_perm = np.random.permutation(n_wall)[:max_damage]
        grid[wall_rs[damage_perm], wall_cs[damage_perm]] = int(noise_color)


def _distance_order_has_room_inversion(ordered_rooms: list[int]) -> bool:
    return any(int(ordered_rooms[i]) > int(ordered_rooms[i + 1]) for i in range(len(ordered_rooms) - 1))


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
        frame_data = _build_frame(int(target_rooms), max_h=f_max_h, max_w=f_max_w)
        if frame_data is None:
            return None
        fh, fw = frame_data.shape
        if fh > grid_h or fw > grid_w:
            return None
        frames.append((int(color), int(target_rooms), frame_data))

    positions = _place_frames(frames, grid_h, grid_w)
    if positions is None:
        return None

    frame_info = []
    for idx, (color, rooms, wall_arr) in enumerate(frames):
        distance, center_r, center_c = _frame_distance(positions[idx], wall_arr, grid_h, grid_w)
        frame_info.append((idx, float(distance), float(center_r), float(center_c), int(color), int(rooms)))

    ordered_by_distance = sorted(frame_info, key=lambda item: (item[1], item[2], item[3], item[4]))
    ordered_distances = [item[1] for item in ordered_by_distance]
    if ordered_distances[-1] - ordered_distances[0] < _MIN_DISTANCE_SPREAD:
        return None
    if any((b - a) < _MIN_DISTANCE_GAP for a, b in zip(ordered_distances, ordered_distances[1:])):
        return None

    ordered_rooms = [item[5] for item in ordered_by_distance]
    if not _distance_order_has_room_inversion(ordered_rooms):
        return None

    grid, frame_mask, protected_mask = _render_frames(frames, positions, grid_h, grid_w)
    _scatter_noise(grid, frame_mask, protected_mask, noise_color)

    try:
        output = main(grid)
    except Exception:
        return None

    max_rooms = max(item[5] for item in ordered_by_distance)
    expected = np.full((frame_count, max_rooms), int(noise_color), dtype=int)
    for r, (_idx, _dist, _cr, _cc, color, rooms) in enumerate(ordered_by_distance):
        expected[r, :rooms] = int(color)

    return grid if np.array_equal(output, expected) else None


def generate_input() -> np.ndarray:
    for _ in range(_GEN_RETRIES):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError(f"Failed to generate valid input after {_GEN_RETRIES} attempts")
