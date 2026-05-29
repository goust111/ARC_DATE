# Combo variant of 89565ca0 (index 17)
# combo_spec: D2/1+D9/0+D13/0
# Name: 位置消歧+无重叠+全部不同

from __future__ import annotations

import math
import random

import numpy as np

from arc_utils import bbox, colors_present, crop_bbox, get_bg, label_mask


# Design analysis for the combo variant
# -------------------------------------
# concrete_variant:
# - The scene still contains several colored rectilinear frames on black, plus one extra non-black
#   noise color scattered across the grid and replacing some wall pixels.
# - Each genuine frame still maps to one horizontal output bar whose length equals its enclosed-room
#   count.
# - This combo removes every overlap relation and also forbids equal room counts. The D2 position
#   tiebreak remains part of the abstract rule, but because all counts are unique it never fires.
#
# behavior_trace:
# - Original task: if blue and red were both 3-room frames, the solver grouped them by room count
#   and used overlap visibility to decide which one was underneath and should appear first. The
#   generator therefore had to create tied pairs and often overlapping bboxes.
# - D2 alone would replace that with row-major bbox ordering among tied frames.
# - This combo goes further: imagine three disjoint frames with room counts 1, 3, 5. The solver now
#   counts rooms and outputs bars in length order immediately. There is no equal-count group to
#   inspect and no overlap evidence to read. The only surviving use of position is as a dormant
#   fallback for out-of-spec ties.
#
# complexity_calibration:
# - Removing overlap and ties risks making the family visually too bland, so the generator keeps
#   3-5 frames, preserves random internal partitions, and keeps light wall damage plus scattered
#   noise. The puzzle is still about reconstructing damaged roomed frames, not merely spotting clean
#   rectangles.
# - At the same time, free variation is constrained: room counts come from a small 1-6 range, all
#   frame colors remain distinct, and frames are separated by blank space. This keeps instances easy
#   to parse by eye and avoids accidental ambiguity.
#
# implicit_assumptions_audit:
# - Original `_cover_order()` assumed every equal-room group could be ordered from visible overlap.
#   D9/0 breaks that premise completely, so the cover-DAG path is removed.
# - `bbox(g == color)` now assumes each frame color appears only on one non-overlapping frame and
#   that wall damage does not erase an entire outer row or column. The generator uses one color per
#   frame and modest damage rates, so the visible bbox stays stable.
# - Filtering with `rooms > 0` assumes the scattered noise color does not accidentally reconstruct
#   into a closed frame. The generator validates every sample through `main()` and rejects failures.
# - Sorting by `(rooms, row, col)` adds a new fallback assumption: if a tie ever existed, the bbox
#   anchor would be a human-visible quantity. That is satisfied automatically because frames do not
#   overlap and their visible bboxes are unobscured.
#
# placement_legality:
# - Frames must stay within bounds and their bounding boxes must be pairwise disjoint. The generator
#   enforces this with shelf-style packing plus 1-3 cells of separation, so non-overlap is built in
#   rather than inferred afterward.
#
# visual_observability_and_cleanliness:
# - A human can see that frames are spatially separate and that no stacked/covered structure exists.
#   The examples therefore advertise the simplified regime directly.
# - The noise color remains visibly different in role because it is fragmented across the board,
#   while each true frame forms a coherent rectilinear scaffold. No hidden property such as numeric
#   color id participates in the ordering.
#
# inferability_of_change:
# - The important observable change is not a new output permutation on tied inputs; it is the
#   systematic absence of both ties and overlap. Across a few examples, a solver can infer that the
#   task has collapsed to counting rooms and ordering by those unique counts alone.


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


def _frame_anchor(g: np.ndarray, color: int) -> tuple[int, int]:
    b = bbox(g == int(color))
    assert b is not None
    return int(b[0]), int(b[1])


# Convert each colored partition frame into a horizontal bar whose length equals its enclosed-room count.
#
# Input structure:
# - Black background with several colored rectilinear frame drawings.
# - One extra non-black color appears as scattered noise and also replaces some wall pixels.
# - Genuine frames are pairwise non-overlapping, and each one encloses a unique number of rooms.
#
# Rule:
# - For each non-background color, rebuild the frame's wall scaffold from dense row/column evidence inside its visible bbox.
# - Count the 4-connected background regions fully enclosed by that reconstructed scaffold.
# - The color whose reconstruction yields zero rooms is the padding/noise color.
# - Sort true frames by room count ascending; if a tie ever occurred, break it by the bbox's top-left position in row-major order.
#
# Key insight:
# - Because the frames are separated and their room counts are all different, the task reduces to reliable room counting; the position rule exists only as a fallback and overlap reasoning disappears entirely.
def main(grid: np.ndarray) -> np.ndarray:
    """
    Convert each colored partition frame into a horizontal bar whose length equals its enclosed-room count.

    Input structure:
    - Black background with several colored rectilinear frame drawings.
    - One extra non-black color appears as scattered noise and also replaces some wall pixels.
    - Genuine frames are pairwise non-overlapping, and each one encloses a unique number of rooms.

    Rule:
    - For each non-background color, rebuild the frame's wall scaffold from dense row/column evidence inside its visible bbox.
    - Count the 4-connected background regions fully enclosed by that reconstructed scaffold.
    - The color whose reconstruction yields zero rooms is the padding/noise color.
    - Sort true frames by room count ascending; if a tie ever occurred, break it by the bbox's top-left position in row-major order.

    Key insight:
    - Because the frames are separated and their room counts are all different, the task reduces to reliable room counting; the position rule exists only as a fallback and overlap reasoning disappears entirely.
    """
    g = np.asarray(grid, dtype=int)
    bg = int(get_bg(g))
    assert bg == 0

    counts: list[tuple[int, int, tuple[int, int]]] = []
    for color in colors_present(g, exclude={bg}):
        c = int(color)
        rooms = _room_count_for_color(g, c)
        anchor = _frame_anchor(g, c)
        counts.append((c, int(rooms), anchor))

    drawings = [(color, rooms, anchor) for color, rooms, anchor in counts if rooms > 0]
    pad_colors = [color for color, rooms, _anchor in counts if rooms == 0]
    assert drawings
    assert len(pad_colors) == 1

    pad = int(pad_colors[0])
    ordered = sorted(drawings, key=lambda item: (int(item[1]), int(item[2][0]), int(item[2][1])))
    width = max(rooms for _color, rooms, _anchor in ordered)

    out = np.full((len(ordered), width), pad, dtype=int)
    for r, (color, rooms, _anchor) in enumerate(ordered):
        out[r, :rooms] = int(color)
    return out


# --------------- generator (gpt-5) ---------------

# ---- CFG: Tunable generation parameters ----
_FRAME_COUNT_RANGE = (3, 5)
_ROOM_COUNT_RANGE = (1, 6)
_GRID_SIZE_BY_FRAMES = {
    5: (27, 30, 27, 30, 12, 14),
    4: (24, 30, 24, 30, 13, 16),
    3: (20, 30, 20, 30, 16, 18),
}
_CELL_GAP_RANGE = (3, 8)
_FRAME_SEPARATION_RANGE = (1, 3)
_NOISE_FRACTION_RANGE = (0.03, 0.07)
_WALL_DAMAGE_FRACTION = 0.18
_PACKING_RETRY = 100
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
        if h > max_h or w > max_w:
            return None
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


def _random_partition(total: int, parts: int) -> list[int]:
    bins = [0] * parts
    for _ in range(total):
        bins[random.randrange(parts)] += 1
    return bins


def _boxes_overlap(
    pos_a: tuple[int, int],
    shape_a: tuple[int, int],
    pos_b: tuple[int, int],
    shape_b: tuple[int, int],
) -> bool:
    ra0, ca0 = pos_a
    ha, wa = shape_a
    rb0, cb0 = pos_b
    hb, wb = shape_b
    return (
        max(ra0, rb0) <= min(ra0 + ha - 1, rb0 + hb - 1)
        and max(ca0, cb0) <= min(ca0 + wa - 1, cb0 + wb - 1)
    )


def _pack_frames_nonoverlapping(
    frames: list[tuple[int, int, np.ndarray]],
    grid_h: int,
    grid_w: int,
) -> list[tuple[int, int]] | None:
    for _ in range(_PACKING_RETRY):
        order = list(range(len(frames)))
        random.shuffle(order)
        gap_h = random.randint(*_FRAME_SEPARATION_RANGE)
        gap_v = random.randint(*_FRAME_SEPARATION_RANGE)

        rows: list[tuple[list[int], int, int]] = []
        current: list[int] = []
        current_width = 0
        current_height = 0

        for idx in order:
            fh, fw = frames[idx][2].shape
            needed = fw if not current else current_width + gap_h + fw
            if current and needed > grid_w:
                rows.append((current, current_width, current_height))
                current = [idx]
                current_width = int(fw)
                current_height = int(fh)
            else:
                if not current:
                    current = [idx]
                    current_width = int(fw)
                    current_height = int(fh)
                else:
                    current.append(idx)
                    current_width = int(needed)
                    current_height = max(int(current_height), int(fh))

        if current:
            rows.append((current, int(current_width), int(current_height)))

        total_h = sum(row_h for _indices, _row_w, row_h in rows) + gap_v * (len(rows) - 1)
        if total_h > grid_h:
            continue

        positions: list[tuple[int, int] | None] = [None] * len(frames)
        extra_v = grid_h - total_h
        v_slack = _random_partition(extra_v, len(rows) + 1)
        y = v_slack[0]
        valid = True

        for row_idx, (indices, row_w, row_h) in enumerate(rows):
            extra_h = grid_w - row_w
            x = random.randint(0, extra_h)
            for k, idx in enumerate(indices):
                fh, fw = frames[idx][2].shape
                top = y + random.randint(0, row_h - fh)
                positions[idx] = (int(top), int(x))
                x += int(fw)
                if k + 1 < len(indices):
                    x += gap_h
            y += row_h
            if row_idx + 1 < len(rows):
                y += gap_v + v_slack[row_idx + 1]

        if not valid or any(pos is None for pos in positions):
            continue

        packed = [tuple(map(int, pos)) for pos in positions if pos is not None]
        assert len(packed) == len(frames)
        if _count_overlap_pairs(packed, frames) != 0:
            continue
        return packed

    return None


def _count_overlap_pairs(
    positions: list[tuple[int, int]],
    frames: list[tuple[int, int, np.ndarray]],
) -> int:
    overlap_pairs = 0
    for i in range(len(frames)):
        for j in range(i + 1, len(frames)):
            if _boxes_overlap(positions[i], frames[i][2].shape, positions[j], frames[j][2].shape):
                overlap_pairs += 1
    return int(overlap_pairs)


def _render_frames(
    frames: list[tuple[int, int, np.ndarray]],
    positions: list[tuple[int, int]],
    grid_h: int,
    grid_w: int,
) -> tuple[np.ndarray, np.ndarray]:
    grid = np.zeros((grid_h, grid_w), dtype=int)
    frame_mask = np.zeros((grid_h, grid_w), dtype=bool)

    for idx, (color, _rooms, wall_arr) in enumerate(frames):
        r0, c0 = positions[idx]
        rs, cs = np.where(wall_arr)
        abs_rs = rs + r0
        abs_cs = cs + c0
        grid[abs_rs, abs_cs] = int(color)
        frame_mask[abs_rs, abs_cs] = True

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
    if len(nonwall_rs) > 0 and noise_count > 0:
        perm = np.random.permutation(len(nonwall_rs))
        n_nonwall = min(noise_count, len(perm))
        sel = perm[:n_nonwall]
        grid[nonwall_rs[sel], nonwall_cs[sel]] = int(noise_color)
        placed = int(n_nonwall)
    else:
        placed = 0

    wall_rs, wall_cs = np.where(frame_mask)
    n_wall = len(wall_rs)
    max_damage = min(noise_count - placed, int(_WALL_DAMAGE_FRACTION * n_wall))
    if max_damage > 0:
        damage_sel = np.random.permutation(n_wall)[:max_damage]
        grid[wall_rs[damage_sel], wall_cs[damage_sel]] = int(noise_color)


def _gen_attempt(
    return_meta: bool = False,
) -> np.ndarray | tuple[np.ndarray, dict[str, object]] | None:
    frame_count = random.randint(*_FRAME_COUNT_RANGE)
    room_counts = random.sample(list(range(_ROOM_COUNT_RANGE[0], _ROOM_COUNT_RANGE[1] + 1)), frame_count)
    random.shuffle(room_counts)

    chosen_colors = random.sample(list(range(1, 10)), frame_count + 1)
    frame_colors = chosen_colors[:frame_count]
    noise_color = int(chosen_colors[frame_count])

    cfg = _GRID_SIZE_BY_FRAMES.get(frame_count, _GRID_SIZE_BY_FRAMES[3])
    h_min, h_max, w_min, w_max, f_max_h, f_max_w = cfg
    grid_h = random.randint(h_min, h_max)
    grid_w = random.randint(w_min, w_max)

    frames: list[tuple[int, int, np.ndarray]] = []
    for color, target_rooms in zip(frame_colors, room_counts):
        frame_data = _build_frame(int(target_rooms), max_h=int(f_max_h), max_w=int(f_max_w))
        if frame_data is None:
            return None
        fh, fw = frame_data.shape
        if fh > grid_h or fw > grid_w:
            return None
        frames.append((int(color), int(target_rooms), frame_data))

    positions = _pack_frames_nonoverlapping(frames, grid_h, grid_w)
    if positions is None:
        return None
    if _count_overlap_pairs(positions, frames) != 0:
        return None

    grid, frame_mask = _render_frames(frames, positions, grid_h, grid_w)
    _scatter_noise(grid, frame_mask, noise_color, grid_h, grid_w)

    try:
        output = main(grid)
    except Exception:
        return None

    frame_info = sorted(
        [
            (room_counts[i], positions[i][0], positions[i][1], frame_colors[i])
            for i in range(frame_count)
        ],
        key=lambda item: (int(item[0]), int(item[1]), int(item[2])),
    )
    max_rooms = max(rc for rc, _r0, _c0, _color in frame_info)
    expected = np.full((frame_count, max_rooms), noise_color, dtype=int)
    for r, (rc, _r0, _c0, color) in enumerate(frame_info):
        expected[r, :rc] = int(color)

    if not np.array_equal(output, expected):
        return None

    if return_meta:
        multiplicities = {int(rc): int(room_counts.count(rc)) for rc in set(room_counts)}
        return grid, {
            "positions": [tuple(map(int, pos)) for pos in positions],
            "room_counts": list(map(int, room_counts)),
            "frame_colors": list(map(int, frame_colors)),
            "noise_color": int(noise_color),
            "overlap_pairs": int(_count_overlap_pairs(positions, frames)),
            "max_room_count_multiplicity": int(max(multiplicities.values())),
            "sorted_room_counts": sorted(map(int, room_counts)),
            "position_tiebreak_used": False,
        }
    return grid


def generate_input() -> np.ndarray:
    for _ in range(_GEN_RETRIES):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError(f"Failed to generate valid input after {_GEN_RETRIES} attempts")
