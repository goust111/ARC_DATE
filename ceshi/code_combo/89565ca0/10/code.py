# Combo variant of 89565ca0 (index 10)
# combo_spec: D1/2+D9/0
# Name: 距离排序+禁止重叠

from __future__ import annotations

import math
import random

import numpy as np

from arc_utils import bbox, colors_present, crop_bbox, get_bg, label_mask


# Design analysis
# ---
# variant:
#   source_baseline:
#     original_task: "89565ca0 reconstructs each colored frame, uses enclosed-room count as the primary output order, and only uses overlap to break ties."
#     fused_change: "D1/2 replaces count-based ordering with distance-to-grid-center ordering, while D9/0 forbids frame overlap entirely."
#   a_concretization_and_trace:
#     input_objects: "The input still shows several colored rectilinear frames plus one scattered noise color, but frame bboxes never overlap in this combo."
#     concrete_trace:
#       old: "A 2-room frame had to appear above a 4-room frame even if the 4-room frame sat closest to the center."
#       fused: "The 4-room frame appears first when its bbox center is nearest the board center; its bar is still length 4, only the row order changes."
#       difference: "Bar lengths encode room counts exactly as before, but row order is now spatial and the input no longer contains any occlusion clues."
#   b_complexity_calibration:
#     kept: "Wall reconstruction, room counting, one noise color, and bar padding all remain, so the task is not reduced to simple object sorting."
#     removed: "Overlap cover-order reasoning is intentionally deleted; otherwise D9/0 would be contradicted."
#     visibility_guard: "The generator rejects samples whose distance order would also be nondecreasing by room count, so the new spatial rule is visible."
#   c_implicit_assumption_audit:
#     original_main_dependencies:
#       - "Grouping by room count assumed room count was the semantic sort key."
#       - "_cover_order assumed equal-room frames overlapped and had a unique bottom-to-top relation."
#     fused_main_dependencies:
#       - "Each true frame must still reconstruct to a stable bbox whose center is readable after noise damage."
#       - "The noise color must still yield zero enclosed rooms so it remains identifiable as the pad color."
#       - "Distance ties should be rare; deterministic row/col/color fallbacks exist but are not meant to carry semantic load."
#     generator_response:
#       - "Frames are placed with non-overlapping bboxes and a clear gap, so no center estimate is corrupted by occlusion."
#       - "Every outer border is protected from noise replacement, preserving bbox extrema for distance measurement."
#       - "Accepted instances require a minimum spacing between sorted distances."
#   d_placement_legality:
#     policy: "Every frame is fully inside the grid and expanded bboxes with a 1-cell moat may not overlap."
#     effect: "This guarantees the D9/0 non-overlap rule and also keeps objects visually separable."
#   e_visual_observability:
#     human_signal: "Across examples, the top output row comes from the frame nearest the global center even when that row is not the shortest bar."
#     clarity: "Because frames never cover one another, solvers can visually compare centers without reasoning about hidden walls."
#   f_visual_cleanliness:
#     cleanup: "Frame sizes are reduced relative to the original generator so 3-5 objects fit with spacing; noise stays sparse and cannot erase full borders."
#   g_color_inference:
#     note: "Colors still label identities only; no color remapping is introduced, avoiding an extra latent rule."
#   h_variant_metric:
#     measured_property: "Accepted samples must contain at least one inversion in the room-count sequence induced by distance order."
#   i_new_main_assumptions_checked_against_generator:
#     - "main() sorts by (distance, center_r, center_c, color); generate_input() enforces distance gaps so the first key dominates."
#     - "main() expects one zero-room color; generate_input() uses a single noise color and validates the full expected output via the solver."


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
    - Black background with several axis-aligned colored line frames that do not overlap one another.
    - One extra non-background color appears as scattered noise and may replace some wall pixels.
    - Each true frame is a rectangular partitioned outline whose enclosed rooms survive rectilinear reconstruction.

    Rule:
    - For each non-background color, reconstruct the frame walls from dense row/column evidence and count the enclosed 4-connected background regions.
    - The color with zero enclosed regions is the noise/pad color.
    - For each real frame, measure the Euclidean distance from its reconstructed bbox center to the whole-grid center.
    - Output one row per real frame, ordered from nearest to farthest; fill each row with that frame color for its room count, then pad with the noise color.

    Key insight:
    - Room counting still determines bar length, but the row order is now purely spatial: the decisive cue is which intact frame sits closest to the center of the board.
    """
    g = np.asarray(grid, dtype=int)
    bg = int(get_bg(g))
    assert bg == 0

    counts: list[tuple[int, int, np.ndarray]] = []
    for color in colors_present(g, exclude={bg}):
        rooms = _room_count_for_color(g, int(color))
        walls = _walls_for_color(g, int(color))
        counts.append((int(color), int(rooms), walls))

    drawings: list[tuple[int, int, np.ndarray, float, float, float]] = []
    pad_colors: list[int] = []
    for color, rooms, walls in counts:
        if rooms == 0:
            pad_colors.append(int(color))
            continue
        dist, center_r, center_c = _frame_sort_features(walls, g.shape)
        drawings.append((int(color), int(rooms), walls, float(dist), float(center_r), float(center_c)))

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
    5: (29, 30, 29, 30, 8, 9),
    4: (27, 30, 27, 30, 9, 10),
    3: (24, 30, 24, 30, 11, 12),
}
_ROOM_COUNT_MAX_BY_FRAMES = {3: 6, 4: 5, 5: 4}
_CELL_GAP_RANGE = (2, 5)
_NOISE_FRACTION_RANGE = (0.04, 0.09)
_WALL_DAMAGE_FRACTION = 0.18
_MIN_BOX_GAP = 1
_MIN_DISTANCE_GAP = 1.0
_MIN_DISTANCE_SPREAD_BY_FRAMES = {3: 4.0, 4: 3.5, 5: 3.0}
_PLACEMENT_RETRY = 300
_GEN_RETRIES = 600
_FRAME_COUNT_CYCLE = [3, 4, 5]
_TARGET_FRAME_RETRIES = 240
_SLOT_LAYOUTS: dict[int, list[list[tuple[float, float]]]] = {
    3: [
        [(0.50, 0.50), (0.18, 0.18), (0.78, 0.78)],
        [(0.48, 0.52), (0.22, 0.50), (0.84, 0.20)],
    ],
    4: [
        [(0.50, 0.50), (0.24, 0.50), (0.50, 0.84), (0.16, 0.16)],
        [(0.50, 0.48), (0.50, 0.24), (0.84, 0.50), (0.84, 0.84)],
    ],
    5: [
        [(0.50, 0.50), (0.24, 0.50), (0.50, 0.84), (0.84, 0.22), (0.16, 0.16)],
        [(0.50, 0.48), (0.50, 0.24), (0.84, 0.50), (0.22, 0.84), (0.84, 0.84)],
    ],
}
_generate_counter = 0


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
    slot_layouts = _SLOT_LAYOUTS[len(frames)]
    order = sorted(range(len(frames)), key=lambda i: -frames[i][2].size)
    jitter = {3: 2, 4: 1, 5: 0}[len(frames)]

    for _ in range(_PLACEMENT_RETRY):
        chosen_slots = list(random.choice(slot_layouts))
        random.shuffle(chosen_slots)
        positions: list[tuple[int, int] | None] = [None] * len(frames)
        placed_boxes: list[tuple[int, int, int, int]] = []
        ok = True

        for idx, (slot_r, slot_c) in zip(order, chosen_slots):
            fh, fw = frames[idx][2].shape
            target_r = slot_r * float(grid_h - 1) - (float(fh) - 1.0) / 2.0
            target_c = slot_c * float(grid_w - 1) - (float(fw) - 1.0) / 2.0
            jitter_r = random.randint(-jitter, jitter)
            jitter_c = random.randint(-jitter, jitter)
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


def _gen_attempt(preferred_frame_count: int | None = None) -> np.ndarray | None:
    if preferred_frame_count is None:
        frame_count = random.randint(*_FRAME_COUNT_RANGE)
    else:
        frame_count = int(preferred_frame_count)

    room_hi = min(_ROOM_COUNT_RANGE[1], _ROOM_COUNT_MAX_BY_FRAMES.get(frame_count, _ROOM_COUNT_RANGE[1]))
    available = list(range(_ROOM_COUNT_RANGE[0], room_hi + 1))
    must_tie = frame_count > len(available)
    allow_tie = must_tie or (random.random() < _TIE_PROBABILITY and frame_count >= 3)
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
    min_distance_spread = _MIN_DISTANCE_SPREAD_BY_FRAMES.get(frame_count, 4.0)
    if ordered_distances[-1] - ordered_distances[0] < min_distance_spread:
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
    global _generate_counter
    target = _FRAME_COUNT_CYCLE[_generate_counter % len(_FRAME_COUNT_CYCLE)]
    _generate_counter += 1

    for _ in range(_TARGET_FRAME_RETRIES):
        result = _gen_attempt(preferred_frame_count=target)
        if result is not None:
            return result

    for fallback in _FRAME_COUNT_CYCLE:
        if fallback == target:
            continue
        for _ in range(max(1, (_GEN_RETRIES - _TARGET_FRAME_RETRIES) // (len(_FRAME_COUNT_CYCLE) - 1))):
            result = _gen_attempt(preferred_frame_count=fallback)
            if result is not None:
                return result
    raise RuntimeError(f"Failed to generate valid input after {_GEN_RETRIES} attempts")
