# Combo variant of 89565ca0 (index 22)
# combo_spec: D1/0+D2/1+D5/1
# Name: 面积排序+位置消歧+面积编码

from __future__ import annotations

import math
import random
from itertools import combinations

import numpy as np

from arc_utils import bbox, colors_present, crop_bbox, get_bg, label_mask


# Design analysis for the combo variant
# -------------------------------------
# a) Concrete behavior and trace:
# - The input family stays close to the source task: several colored rectilinear frames on black,
#   plus one scattered non-black noise color that also damages some wall pixels.
# - Each real frame still has to be reconstructed and its enclosed-room count still matters, but only
#   for the destination encoding: a frame with N rooms becomes an N x N solid square.
# - The changed ranking signal is spatial rather than topological. Example:
#   frame A: rooms=2, bbox=8x10 => area 80
#   frame B: rooms=5, bbox=8x10 => area 80
#   frame C: rooms=3, bbox=7x9  => area 63
#   Original task order would be C? no: original sorts by rooms, so A(2), C(3), B(5), and each output
#   item is one horizontal bar.
#   Combo order is C first because 63 < 80, then A/B are tied on area so whichever frame sits higher
#   (or, if on the same row, further left) comes first. Their outputs are 3x3, 2x2, and 5x5 squares,
#   stacked vertically rather than one-row bars.
# - This means the solver must combine two recovered properties per object: bbox area for ordering and
#   room count for square size.
# b) Complexity calibration:
# - To make D2 observable, every generated example contains exactly one equal-area pair among true
#   frames. Those two frames have different room counts, so the tie-break is visible in the output.
# - To make D1 observable, the generator rejects samples where sorting by area (with the forced
#   row-major tie-break) would yield the same color order as sorting by room count.
# - To keep the puzzle readable, the generator uses exact-size frame construction and avoids frame-frame
#   overlap. The original overlap mechanic is no longer needed once depth ordering is removed, and
#   avoiding it keeps bbox extent visually inspectable.
# c) Implicit assumption audit:
# - The old solver assumed the primary sort key and the output-length key were both `rooms`; that is no
#   longer true. We must split "ordering stats" from "encoding stats".
# - The old tie-resolution code assumed an overlap-derived cover DAG. That assumption breaks completely
#   once ties are resolved geometrically, so cover-order logic is removed rather than repurposed.
# - Area sorting introduces a new requirement: the visible bbox of each frame must match its intended
#   outer rectangle. The generator guarantees this by protecting all four frame corners from noise
#   damage and by keeping frames disjoint.
# - Position tie-breaking assumes equal-area frames have distinct row-major anchors. The generator
#   explicitly places the tied pair either on different rows or on the same row with different columns.
# d) Placement legality:
# - All frames are placed fully in bounds.
# - Frames are pairwise non-overlapping in this combo generator, so no frame can hide another's bbox
#   extrema or top-left anchor.
# - Noise is only a color replacement process; it never creates a new "true frame" because the oracle
#   rejects any sample where the solver disagrees.
# e) Visual observability:
# - From 2-4 examples, a human can see that output order no longer follows square side length: a larger
#   square can appear above a smaller one if its input frame has a smaller bbox area.
# - The equal-area pair is visible because their frames occupy different row-major positions while
#   producing different square sizes. That exposes the position tie-break directly in pixel space.
# f) Visual cleanliness:
# - Stacked left-aligned squares keep the destination regular.
# - Disjoint source frames plus protected corners keep the bbox signal legible even with scattered noise.
# g) New solver assumptions and generator guarantees:
# - `main()` assumes each true frame remains reconstructible from dense row/column evidence and that its
#   visible bbox corners survive. The generator enforces both with exact wall synthesis, limited wall
#   damage, protected corners, and oracle validation.


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


def _frame_signature(g: np.ndarray, color: int) -> tuple[int, int, int, tuple[int, int]]:
    color_mask = g == int(color)
    b = bbox(color_mask)
    assert b is not None
    r0, c0, r1, c1 = b
    local = crop_bbox(color_mask, b).astype(bool)
    walls = _reconstruct_walls(local)
    rooms = _count_enclosed_regions(walls)
    area = int(r1 - r0 + 1) * int(c1 - c0 + 1)
    return int(color), int(rooms), int(area), (int(r0), int(c0))


def _encode_room_squares(
    ordered: list[tuple[int, int, int, tuple[int, int]]], pad: int
) -> np.ndarray:
    width = max(rooms for _color, rooms, _area, _anchor in ordered)
    height = sum(rooms for _color, rooms, _area, _anchor in ordered)
    out = np.full((height, width), int(pad), dtype=int)
    r0 = 0
    for color, rooms, _area, _anchor in ordered:
        out[r0 : r0 + rooms, :rooms] = int(color)
        r0 += rooms
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Convert each noisy colored room-frame into a stacked solid square, ordered by recovered frame area rather than by the square side itself.

    Input structure:
    - Black background with several colored axis-aligned line-grid frames.
    - One extra non-black color appears as scattered noise and also replaces some wall pixels.
    - Each true frame still encloses one or more background rooms inside a rectangular outer boundary.

    Rule:
    - For each non-background color, reconstruct its rectilinear wall scaffold from dense row and column evidence inside that color's visible bbox.
    - Count the fully enclosed 4-connected background regions to get that frame's room count.
    - The lone color with zero enclosed regions is the padding/noise color, not a real frame.
    - Sort the real frames by bbox area ascending; if two areas tie, compare the bbox top-left corners in row-major order.
    - Render each sorted frame as an N x N solid block of its own color, where N is its room count, stacking the blocks top-to-bottom and padding unused cells with the noise color.

    Key insight:
    - The destination size still comes from the recovered room topology, but row order now comes from spatial footprint, with equal-footprint cases resolved by visible position. Solving therefore requires combining two independent object properties instead of reusing one statistic for everything.
    """
    g = np.asarray(grid, dtype=int)
    bg = int(get_bg(g))
    assert bg == 0

    stats: list[tuple[int, int, int, tuple[int, int]]] = []
    for color in colors_present(g, exclude={bg}):
        stats.append(_frame_signature(g, int(color)))

    drawings = [(color, rooms, area, anchor) for color, rooms, area, anchor in stats if rooms > 0]
    pad_colors = [color for color, rooms, _area, _anchor in stats if rooms == 0]
    assert drawings
    assert len(pad_colors) == 1

    pad = int(pad_colors[0])
    ordered = sorted(drawings, key=lambda item: (int(item[2]), int(item[3][0]), int(item[3][1])))
    return _encode_room_squares(ordered, pad)


# --------------- generator (gpt-5, combo adapted) ---------------

# ---- CFG: Tunable generation parameters ----
_FRAME_COUNT_RANGE = (3, 5)
_ROOM_COUNT_RANGE = (1, 6)
_GRID_SIZE_BY_FRAMES = {
    5: (25, 30, 25, 30, 16, 18),
    4: (22, 30, 22, 30, 18, 22),
    3: (20, 30, 20, 30, 22, 25),
}
_CELL_GAP_RANGE = (3, 13)
_NOISE_FRACTION_RANGE = (0.04, 0.09)
_WALL_DAMAGE_FRACTION = 0.25
_TIE_SAME_ROW_PROBABILITY = 0.35
_PLACEMENT_RETRY = 250
_SPEC_RETRIES = 300
_GEN_RETRIES = 500
_NEXT_TIE_MODE = "same_row"


def _random_gaps_exact(n: int, min_gap: int, max_gap: int, total: int) -> list[int] | None:
    if n * min_gap > total or n * max_gap < total:
        return None
    gaps = [min_gap] * n
    remaining = total - n * min_gap
    order = list(range(n))
    random.shuffle(order)
    extra_cap = max_gap - min_gap
    for i, idx in enumerate(order[:-1]):
        slots_left = len(order) - i - 1
        min_add = max(0, remaining - slots_left * extra_cap)
        max_add = min(extra_cap, remaining)
        add = random.randint(min_add, max_add)
        gaps[idx] += add
        remaining -= add
    last = order[-1]
    gaps[last] += remaining
    if gaps[last] > max_gap:
        return None
    return gaps


def _feasible_configs(target_rooms: int, target_h: int, target_w: int) -> list[tuple[int, int, int]]:
    configs = []
    min_gap, max_gap = _CELL_GAP_RANGE
    for nv in range(2, 6):
        for nh in range(2, 5):
            cells = (nv - 1) * (nh - 1)
            if not (target_rooms <= cells <= target_rooms + 4):
                continue
            n_row_gaps = nh - 1
            n_col_gaps = nv - 1
            if not (n_row_gaps * min_gap <= target_h - 1 <= n_row_gaps * max_gap):
                continue
            if not (n_col_gaps * min_gap <= target_w - 1 <= n_col_gaps * max_gap):
                continue
            configs.append((nv, nh, cells))
    if configs:
        return configs

    for nv in range(2, 7):
        for nh in range(2, 6):
            cells = (nv - 1) * (nh - 1)
            if cells < target_rooms:
                continue
            n_row_gaps = nh - 1
            n_col_gaps = nv - 1
            if not (n_row_gaps * min_gap <= target_h - 1 <= n_row_gaps * max_gap):
                continue
            if not (n_col_gaps * min_gap <= target_w - 1 <= n_col_gaps * max_gap):
                continue
            configs.append((nv, nh, cells))
    return configs


def _build_frame(target_rooms: int, target_h: int, target_w: int) -> np.ndarray | None:
    if target_h < 4 or target_w < 4:
        return None
    if target_rooms == 1:
        walls = np.zeros((target_h, target_w), dtype=bool)
        walls[0, :] = True
        walls[target_h - 1, :] = True
        walls[:, 0] = True
        walls[:, target_w - 1] = True
        return walls

    possible_configs = _feasible_configs(target_rooms, target_h, target_w)
    if not possible_configs:
        return None

    n_vcols, n_hrows, total_cells = random.choice(possible_configs)
    min_gap, max_gap = _CELL_GAP_RANGE
    row_gaps = _random_gaps_exact(n_hrows - 1, min_gap, max_gap, target_h - 1)
    col_gaps = _random_gaps_exact(n_vcols - 1, min_gap, max_gap, target_w - 1)
    if row_gaps is None or col_gaps is None:
        return None

    vcol_positions = [0]
    for gap in col_gaps:
        vcol_positions.append(vcol_positions[-1] + gap)
    hrow_positions = [0]
    for gap in row_gaps:
        hrow_positions.append(hrow_positions[-1] + gap)

    walls = np.zeros((target_h, target_w), dtype=bool)
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


def _count_inversions(values: list[int]) -> int:
    count = 0
    for i in range(len(values)):
        for j in range(i + 1, len(values)):
            if values[i] > values[j]:
                count += 1
    return count


def _candidate_dims(max_h: int, max_w: int) -> list[tuple[int, int]]:
    dims = []
    for h in range(6, max_h + 1):
        for w in range(6, max_w + 1):
            area = h * w
            if 36 <= area <= 140:
                dims.append((h, w))
    random.shuffle(dims)
    return dims


def _choose_frame_specs(
    room_counts: list[int],
    max_h: int,
    max_w: int,
    grid_h: int,
    grid_w: int,
) -> tuple[dict[int, tuple[int, int, int]], tuple[int, int], tuple[int, int]] | None:
    rooms = list(room_counts)
    compat: dict[int, list[tuple[int, int, int]]] = {room: [] for room in rooms}
    for h, w in _candidate_dims(max_h, max_w):
        if h > grid_h - 2 or w > grid_w - 2:
            continue
        for room in rooms:
            if _feasible_configs(room, h, w):
                compat[room].append((h, w, h * w))

    if any(not compat[room] for room in rooms):
        return None

    min_inversions = 1 if len(rooms) == 3 else 2
    pair_candidates: list[tuple[int, int, int]] = []
    for ra, rb in combinations(rooms, 2):
        areas_a = {area for _h, _w, area in compat[ra]}
        areas_b = {area for _h, _w, area in compat[rb]}
        common = list(areas_a & areas_b)
        random.shuffle(common)
        for area in common:
            pair_candidates.append((int(ra), int(rb), int(area)))
    random.shuffle(pair_candidates)

    for _ in range(_SPEC_RETRIES):
        if not pair_candidates:
            break
        tie_room_a, tie_room_b, tie_area = random.choice(pair_candidates)
        opts_a = [item for item in compat[tie_room_a] if item[2] == tie_area]
        opts_b = [item for item in compat[tie_room_b] if item[2] == tie_area]
        if not opts_a or not opts_b:
            continue

        spec_by_room: dict[int, tuple[int, int, int]] = {
            int(tie_room_a): random.choice(opts_a),
            int(tie_room_b): random.choice(opts_b),
        }
        used_areas = {int(tie_area)}
        remaining = [room for room in rooms if room not in {tie_room_a, tie_room_b}]
        random.shuffle(remaining)
        ok = True
        for room in remaining:
            options = [item for item in compat[room] if item[2] not in used_areas]
            if not options:
                ok = False
                break
            spec_by_room[int(room)] = random.choice(options)
            used_areas.add(int(spec_by_room[int(room)][2]))
        if not ok:
            continue

        total_area = sum(spec[2] for spec in spec_by_room.values())
        if total_area > int(0.50 * grid_h * grid_w):
            continue

        area_order_prefix = sorted(
            [
                (spec_by_room[room][2], int(room))
                for room in rooms
                if room not in {tie_room_a, tie_room_b}
            ]
        )
        tied_below = [room for area, room in area_order_prefix if area < tie_area]
        tied_above = [room for area, room in area_order_prefix if area > tie_area]

        seq_ab = tied_below + [int(tie_room_a), int(tie_room_b)] + tied_above
        seq_ba = tied_below + [int(tie_room_b), int(tie_room_a)] + tied_above
        inv_ab = _count_inversions(seq_ab)
        inv_ba = _count_inversions(seq_ba)
        if max(inv_ab, inv_ba) < min_inversions:
            continue

        preferred_order = (
            (int(tie_room_a), int(tie_room_b))
            if inv_ab >= inv_ba
            else (int(tie_room_b), int(tie_room_a))
        )
        return spec_by_room, (int(tie_room_a), int(tie_room_b)), preferred_order
    return None


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


def _place_tied_pair(
    positions: list[tuple[int, int] | None],
    frames: list[tuple[int, int, int, np.ndarray]],
    first_idx: int,
    second_idx: int,
    grid_h: int,
    grid_w: int,
    target_mode: str | None = None,
) -> str | None:
    if target_mode is not None:
        modes = [str(target_mode)]
    else:
        modes = ["same_row", "different_row"]
        if random.random() >= _TIE_SAME_ROW_PROBABILITY:
            modes.reverse()

    h1, w1 = frames[first_idx][3].shape
    h2, w2 = frames[second_idx][3].shape

    for mode in modes:
        for _ in range(_PLACEMENT_RETRY):
            if mode == "same_row":
                needed_w = w1 + 1 + w2
                if needed_w > grid_w or max(h1, h2) > grid_h:
                    break
                row = random.randint(0, grid_h - max(h1, h2))
                c1 = random.randint(0, grid_w - needed_w)
                c2 = random.randint(c1 + w1 + 1, grid_w - w2)
                p1 = (row, c1)
                p2 = (row, c2)
            else:
                needed_h = h1 + 1 + h2
                if needed_h > grid_h or max(w1, w2) > grid_w:
                    break
                r1 = random.randint(0, grid_h - needed_h)
                r2 = random.randint(r1 + h1 + 1, grid_h - h2)
                c1 = random.randint(0, grid_w - w1)
                c2 = random.randint(0, grid_w - w2)
                p1 = (r1, c1)
                p2 = (r2, c2)
            if not _boxes_overlap(p1, (h1, w1), p2, (h2, w2)):
                positions[first_idx] = p1
                positions[second_idx] = p2
                return mode
    return None


def _random_nonoverlap_position(
    frame_arr: np.ndarray,
    existing: list[tuple[int, int]],
    existing_shapes: list[tuple[int, int]],
    grid_h: int,
    grid_w: int,
) -> tuple[int, int] | None:
    h, w = frame_arr.shape
    for _ in range(_PLACEMENT_RETRY):
        r0 = random.randint(0, grid_h - h)
        c0 = random.randint(0, grid_w - w)
        if all(
            not _boxes_overlap((r0, c0), (h, w), pos, shape)
            for pos, shape in zip(existing, existing_shapes)
        ):
            return (r0, c0)
    return None


def _frame_corners(pos: tuple[int, int], shape: tuple[int, int]) -> set[tuple[int, int]]:
    r0, c0 = pos
    h, w = shape
    return {
        (int(r0), int(c0)),
        (int(r0), int(c0 + w - 1)),
        (int(r0 + h - 1), int(c0)),
        (int(r0 + h - 1), int(c0 + w - 1)),
    }


def _render_frames(
    frames: list[tuple[int, int, int, np.ndarray]],
    positions: list[tuple[int, int]],
    render_order: list[int],
    grid_h: int,
    grid_w: int,
) -> tuple[np.ndarray, np.ndarray]:
    grid = np.zeros((grid_h, grid_w), dtype=int)
    frame_mask = np.zeros((grid_h, grid_w), dtype=bool)
    for idx in render_order:
        r0, c0 = positions[idx]
        wall_arr = frames[idx][3]
        color = frames[idx][0]
        rs, cs = np.where(wall_arr)
        abs_rs, abs_cs = rs + r0, cs + c0
        grid[abs_rs, abs_cs] = color
        frame_mask[abs_rs, abs_cs] = True
    return grid, frame_mask


def _scatter_noise(
    grid: np.ndarray,
    frame_mask: np.ndarray,
    noise_color: int,
    protected_points: set[tuple[int, int]],
) -> None:
    grid_h, grid_w = grid.shape
    total_cells = grid_h * grid_w
    noise_fraction = random.uniform(*_NOISE_FRACTION_RANGE)
    noise_count = int(total_cells * noise_fraction)

    nonwall_rs, nonwall_cs = np.where(~frame_mask)
    perm = np.random.permutation(len(nonwall_rs))
    n_nonwall = min(noise_count, len(perm))
    sel = perm[:n_nonwall]
    grid[nonwall_rs[sel], nonwall_cs[sel]] = int(noise_color)
    placed = n_nonwall

    protected_mask = np.zeros_like(frame_mask, dtype=bool)
    for r, c in protected_points:
        protected_mask[int(r), int(c)] = True

    damageable = frame_mask & ~protected_mask
    wall_rs, wall_cs = np.where(damageable)
    n_wall = len(wall_rs)
    max_damage = min(noise_count - placed, int(_WALL_DAMAGE_FRACTION * n_wall))
    if max_damage > 0:
        damage_perm = np.random.permutation(n_wall)[:max_damage]
        grid[wall_rs[damage_perm], wall_cs[damage_perm]] = int(noise_color)


def _expected_output(
    ordered: list[tuple[int, int, int, int, int]],
    noise_color: int,
) -> np.ndarray:
    width = max(rooms for _area, _r0, _c0, rooms, _color in ordered)
    height = sum(rooms for _area, _r0, _c0, rooms, _color in ordered)
    out = np.full((height, width), int(noise_color), dtype=int)
    r0 = 0
    for _area, _row, _col, rooms, color in ordered:
        out[r0 : r0 + rooms, :rooms] = int(color)
        r0 += rooms
    return out


def _gen_attempt(
    target_tie_mode: str | None = None,
    return_meta: bool = False,
) -> np.ndarray | tuple[np.ndarray, dict[str, object]] | None:
    frame_count = random.randint(*_FRAME_COUNT_RANGE)
    room_counts = random.sample(list(range(_ROOM_COUNT_RANGE[0], _ROOM_COUNT_RANGE[1] + 1)), frame_count)
    random.shuffle(room_counts)

    chosen_colors = random.sample(list(range(1, 10)), frame_count + 1)
    frame_colors = chosen_colors[:frame_count]
    noise_color = chosen_colors[frame_count]

    cfg = _GRID_SIZE_BY_FRAMES.get(frame_count, _GRID_SIZE_BY_FRAMES[3])
    h_min, h_max, w_min, w_max, f_max_h, f_max_w = cfg
    grid_h = random.randint(h_min, h_max)
    grid_w = random.randint(w_min, w_max)

    spec_choice = _choose_frame_specs(room_counts, f_max_h, f_max_w, grid_h, grid_w)
    if spec_choice is None:
        return None
    spec_by_room, tie_rooms, preferred_order = spec_choice

    frames: list[tuple[int, int, int, np.ndarray]] = []
    for color, rooms in zip(frame_colors, room_counts):
        target_h, target_w, area = spec_by_room[int(rooms)]
        frame_data = _build_frame(int(rooms), target_h=target_h, target_w=target_w)
        if frame_data is None:
            return None
        frames.append((int(color), int(rooms), int(area), frame_data))

    tie_indices = [idx for idx, rooms in enumerate(room_counts) if rooms in tie_rooms]
    assert len(tie_indices) == 2
    idx_by_room = {int(room_counts[idx]): int(idx) for idx in tie_indices}
    first_idx = idx_by_room[int(preferred_order[0])]
    second_idx = idx_by_room[int(preferred_order[1])]

    positions: list[tuple[int, int] | None] = [None] * frame_count
    tie_mode = _place_tied_pair(
        positions,
        frames,
        first_idx,
        second_idx,
        grid_h,
        grid_w,
        target_mode=target_tie_mode,
    )
    if tie_mode is None:
        return None

    placed_positions = [positions[first_idx], positions[second_idx]]
    placed_shapes = [frames[first_idx][3].shape, frames[second_idx][3].shape]
    for idx in sorted(range(frame_count), key=lambda i: -frames[i][2]):
        if positions[idx] is not None:
            continue
        pos = _random_nonoverlap_position(frames[idx][3], placed_positions, placed_shapes, grid_h, grid_w)
        if pos is None:
            return None
        positions[idx] = pos
        placed_positions.append(pos)
        placed_shapes.append(frames[idx][3].shape)

    assert all(pos is not None for pos in positions)
    final_positions = [tuple(map(int, pos)) for pos in positions]

    protected_points: set[tuple[int, int]] = set()
    for idx, pos in enumerate(final_positions):
        protected_points |= _frame_corners(pos, frames[idx][3].shape)

    render_order = list(range(frame_count))
    random.shuffle(render_order)
    grid, frame_mask = _render_frames(frames, final_positions, render_order, grid_h, grid_w)
    _scatter_noise(grid, frame_mask, noise_color, protected_points)

    try:
        output = main(grid)
    except Exception:
        return None

    ordered = sorted(
        [
            (frames[i][2], final_positions[i][0], final_positions[i][1], frames[i][1], frames[i][0])
            for i in range(frame_count)
        ],
        key=lambda item: (int(item[0]), int(item[1]), int(item[2])),
    )
    expected = _expected_output(ordered, noise_color)
    if not np.array_equal(output, expected):
        return None

    room_sorted_colors = [color for rooms, color in sorted((frames[i][1], frames[i][0]) for i in range(frame_count))]
    combo_colors = [color for _area, _row, _col, _rooms, color in ordered]
    if room_sorted_colors == combo_colors:
        return None

    tied_area_groups = {}
    for area, row, col, rooms, color in ordered:
        tied_area_groups.setdefault(int(area), []).append((int(row), int(col), int(rooms), int(color)))
    tie_group_sizes = sorted(len(group) for group in tied_area_groups.values() if len(group) > 1)
    if tie_group_sizes != [2]:
        return None

    if return_meta:
        return grid, {
            "room_counts": list(map(int, room_counts)),
            "frame_colors": list(map(int, frame_colors)),
            "noise_color": int(noise_color),
            "positions": final_positions,
            "areas": [int(frames[i][2]) for i in range(frame_count)],
            "ordered_rooms": [int(rooms) for _area, _row, _col, rooms, _color in ordered],
            "combo_colors": combo_colors,
            "room_sorted_colors": room_sorted_colors,
            "tie_rooms": tuple(map(int, tie_rooms)),
            "preferred_order": tuple(map(int, preferred_order)),
            "tie_mode": tie_mode,
        }
    return grid


def generate_input() -> np.ndarray:
    global _NEXT_TIE_MODE

    target_tie_mode = _NEXT_TIE_MODE
    for _ in range(_GEN_RETRIES):
        result = _gen_attempt(target_tie_mode=target_tie_mode)
        if result is not None:
            _NEXT_TIE_MODE = "different_row" if target_tie_mode == "same_row" else "same_row"
            return result
    raise RuntimeError(f"Failed to generate valid input after {_GEN_RETRIES} attempts")
