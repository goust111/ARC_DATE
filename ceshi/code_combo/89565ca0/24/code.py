# Combo variant of 89565ca0 (index 24)
# combo_spec: D1/2+D5/3
# Name: 距离排序+点阵编码

from __future__ import annotations

import math
import random
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from arc_utils import bbox, colors_present, crop_bbox, get_bg, label_mask


# Design analysis
# ---------------
# combo_variant:
#   summary: "Keep the original frame reconstruction and room counting, but fuse a spatial
#     ordering rule with a sparse destination encoding."
#   a_concretization_and_trace:
#     example:
#       - "Suppose three frames have room counts [4, 1, 3] and reconstructed centers whose
#         distances to the board center are [2.0, 6.2, 4.4]."
#       - "Original task: sort by room count -> rows correspond to counts [1, 3, 4], and each
#         row is a solid left-anchored bar."
#       - "D1/2 effect: sort by distance instead -> rows correspond to counts [4, 3, 1] because
#         the 4-room frame is closest, then the 3-room frame, then the 1-room frame."
#       - "D5/3 effect: encode each count as equally spaced dots, so the 4-room row becomes
#         [C, pad, C, pad, C, pad, C] instead of [C, C, C, C]."
#       - "Fusion difference: both row order and per-row geometry change. The output is not
#         monotone in room count, and rows are sparse rather than solid."
#   b_complexity_calibration:
#     kept:
#       - "Input objects are still colored rectilinear frames damaged by one scattered noise color."
#       - "Room count still comes only from reconstructed wall topology."
#     changed:
#       - "Ordering is now spatial, so the generator must make centers visually readable."
#       - "Encoding is now sparse, so width becomes 2*max_rooms-1 rather than max_rooms."
#     safeguards:
#       - "Frames are generated compactly and placed without overlap so bbox centers remain legible."
#       - "Samples are rejected unless distance order disagrees with ascending room count, making
#         the spatial rule observable rather than latent."
#   c_implicit_assumption_audit:
#     original_lines_broken:
#       - "Grouping by room count and resolving ties by overlap is no longer semantic; distance is
#         the primary key now."
#       - "Writing a contiguous prefix of color cells would silently revert the new destination rule."
#     new_main_assumptions:
#       - "Every real frame reconstructs to a valid wall mask whose bbox still approximates the
#         true frame extent."
#       - "The only zero-room color is the scattered noise color."
#       - "Distance ties are rare enough that a deterministic fallback (row, col, color) is only a
#         final stabilizer, not hidden task logic."
#     generator_response:
#       - "Outer frame boundaries are protected from noise damage so bbox extrema survive."
#       - "Placement enforces non-overlap and a gap, preventing occlusion-driven bbox distortion."
#       - "Sorted center distances must be separated by a minimum gap."
#   d_placement_legality:
#     policy: "Every frame bbox is fully inside the grid, expanded bboxes do not overlap, and a
#       one-cell safety gap is preserved between them."
#     reason: "The fusion relies on visible spatial position, so touching or overlapping frames
#       would create ambiguous centers and clutter."
#   e_visual_observability:
#     human_signal:
#       - "Across examples, the top output row is often a long dotted row rather than the smallest
#         count, so a solver can infer that proximity to the board center overrides count order."
#       - "Within each row, the count is still visible because dots are equally spaced at columns
#         0, 2, 4, ..."
#     branch_choice: "Ordering is fixed to near-to-far; we avoid a hidden near/far branch."
#   f_visual_cleanliness:
#     cleanup:
#       - "No frame overlap, bounded noise damage, and padding-only gaps in the output keep both
#         input and output easy to parse."
#       - "Only two colors appear in each output row: the frame color and the padding/noise color."
#   g_color_inference:
#     note: "There is no remapping. A frame keeps its own color in the output; the noise color
#       remains the universal pad/gap color."
#   h_non_degeneracy_metric:
#     metric: "Count adjacent inversions in the room-count sequence after distance sorting."
#     target: "At least one inversion per generated sample, so the D1 ordering cannot collapse to
#       the original ascending-count order."
#   i_new_main_assumptions_checked_against_generator:
#       - "main() sorts by reconstructed-center distance, so generate_input() guarantees distinct
#         distances with a minimum spread."
#       - "main() renders dots at even columns, so generate_input() uses room counts >= 1 and
#         multiple frames, ensuring the sparse encoding is visible."


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
    return float(distance), float(center_r), float(center_c)


def _dot_output_width(max_rooms: int) -> int:
    return int(2 * max_rooms - 1)


def _render_dot_rows(rows: list[tuple[int, int]], pad: int) -> np.ndarray:
    max_rooms = max(rooms for _color, rooms in rows)
    out = np.full((len(rows), _dot_output_width(max_rooms)), int(pad), dtype=int)
    for r, (color, rooms) in enumerate(rows):
        cols = np.arange(0, 2 * int(rooms), 2, dtype=int)
        out[r, cols] = int(color)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Convert each colored partitioned frame into a sparse row of evenly spaced dots whose count equals the frame's enclosed-room count.

    Input structure:
    - Black background with several axis-aligned colored line frames.
    - One extra non-background color appears only as scattered noise/padding and may replace some frame-wall pixels.
    - Each real frame is still a rectilinear partition that encloses one or more background rooms after wall reconstruction.

    Rule:
    - For each non-background color, reconstruct the frame walls from dense row/column evidence and count its enclosed 4-connected background regions.
    - The only color with zero enclosed regions is the noise/pad color.
    - For every real frame, compute the Euclidean distance from its reconstructed bbox center to the whole-grid center.
    - Sort real frames from nearest to farthest from the board center.
    - Output one row per frame, filling the row with the pad color except at columns 0, 2, 4, ... where one same-color dot is placed for each enclosed room.

    Key insight:
    - Counting and ordering are decoupled in the fused task: room count decides how many dots a row contains, while spatial proximity to the board center decides which colored row comes first.
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
    return _render_dot_rows([(color, rooms) for color, rooms, _walls, _dist, _cr, _cc in ordered], pad)


# --------------- generator ---------------

# Compact frames are deliberate here: the fused rule needs multiple clearly separated objects
# so center-distance ordering is obvious and stable after noise damage.
_FRAME_COUNT_OPTIONS = (3, 4, 5)
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
_MIN_BOX_GAP = 1
_MIN_DISTANCE_GAP = 1.0
_MIN_DISTANCE_SPREAD = 4.0
_PLACEMENT_RETRY = 300
_GEN_RETRIES = 600
_GENERATE_INPUT_CALLS = 0


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
    center_slot = (0.50, 0.50)
    outer_slots = [
        (0.18, 0.66),
        (0.70, 0.82),
        (0.82, 0.22),
        (0.14, 0.14),
    ]
    order = sorted(range(len(frames)), key=lambda i: -frames[i][2].size)

    for _ in range(_PLACEMENT_RETRY):
        chosen_slots = [center_slot] + random.sample(outer_slots, len(frames) - 1)
        random.shuffle(chosen_slots)
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


def _frame_distance(
    position: tuple[int, int],
    walls: np.ndarray,
    grid_h: int,
    grid_w: int,
) -> tuple[float, float, float]:
    r0, c0 = position
    h, w = walls.shape
    center_r = float(r0) + (float(h) - 1.0) / 2.0
    center_c = float(c0) + (float(w) - 1.0) / 2.0
    grid_r = (float(grid_h) - 1.0) / 2.0
    grid_c = (float(grid_w) - 1.0) / 2.0
    distance = math.hypot(center_r - grid_r, center_c - grid_c)
    return float(distance), float(center_r), float(center_c)


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


def _expected_output(frame_info: list[tuple[int, float, float, float, int, int]], noise_color: int) -> np.ndarray:
    rows = [(color, rooms) for _idx, _dist, _cr, _cc, color, rooms in frame_info]
    return _render_dot_rows(rows, noise_color)


def _gen_attempt(frame_count: int | None = None) -> np.ndarray | None:
    if frame_count is None:
        frame_count = random.choice(_FRAME_COUNT_OPTIONS)
    frame_count = int(frame_count)

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

    expected = _expected_output(ordered_by_distance, int(noise_color))
    return grid if np.array_equal(output, expected) else None


def generate_input() -> np.ndarray:
    global _GENERATE_INPUT_CALLS
    start = _GENERATE_INPUT_CALLS % len(_FRAME_COUNT_OPTIONS)
    target_counts = [
        _FRAME_COUNT_OPTIONS[(start + offset) % len(_FRAME_COUNT_OPTIONS)]
        for offset in range(len(_FRAME_COUNT_OPTIONS))
    ]
    _GENERATE_INPUT_CALLS += 1

    per_target_retries = max(1, _GEN_RETRIES // len(target_counts))
    for target in target_counts:
        for _ in range(per_target_retries):
            result = _gen_attempt(frame_count=int(target))
            if result is not None:
                return result
    raise RuntimeError(f"Failed to generate valid input after {_GEN_RETRIES} attempts")
