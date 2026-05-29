# Variant of 89565ca0
# Dimension: D2 — 消歧规则：覆盖顺序 (match_cond)
# Change: 房间数相同时，被覆盖在下方的框架排在前面（底部优先）... → 按框架位置（行优先）消歧

from __future__ import annotations

import math
import random

import numpy as np

from arc_utils import bbox, colors_present, crop_bbox, get_bg, label_mask


# Design analysis for the D2 variant
# ----------------------------------
# concrete_variant:
# - The input still contains several noisy colored rectilinear frames on black.
# - Each real frame still maps to one horizontal output bar whose length equals its room count.
# - Only the tie-breaker changes: if two frames have the same room count, the one whose bbox starts
#   higher in the grid comes first; if those top rows match, the leftmost one comes first.
#
# behavior_trace:
# - Original task: two 3-room frames could appear in output order because one visibly sits under
#   the other at an overlap. The generator therefore had to force overlaps for tied pairs, and the
#   solver had to infer a bottom-to-top cover DAG.
# - Variant: the same two 3-room frames are compared by their top-left positions instead. A tied
#   pair can now be disjoint. Example: blue 3-room frame at (4, 14), red 3-room frame at (11, 3).
#   Original logic would need an overlap to decide the order; new logic outputs blue before red
#   because 4 < 11. In a same-row example, (6, 4) precedes (6, 17) because 4 < 17.
#
# complexity_calibration:
# - To make the new rule observable from a few examples, every generated puzzle includes exactly one
#   tied room-count pair. Without that, many samples would be identical to the original task.
# - The tied pair is sometimes arranged on the same row so the column tie-break branch is exercised.
# - Non-tied overlaps remain possible so the puzzle still looks like the original family rather than
#   collapsing into a trivial non-overlapping counting task.
#
# implicit_assumptions_audit:
# - Old main() assumed every equal-room group could be linearly ordered by visible overlap; that is
#   why `_cover_order()` asserted a unique topological next node. This is invalid once tied frames
#   may be disjoint, so the group-ordering step must use geometric anchors instead.
# - Sorting by position introduces a new assumption: each tied frame's top-left anchor must remain
#   visible despite noise and unrelated overlaps. The generator guarantees this by protecting that
#   pixel from wall damage and by never placing/moving another frame over it.
# - Expected-output construction can no longer depend on `render_order.index(i)`; it must use the
#   actual frame placements `(row, col)` because rendering order is now irrelevant for ties.
#
# placement_legality:
# - Tied frames are placed with distinct row-major anchors, within bounds, and never on the same
#   top-left pixel. Other frames may overlap, but no frame is allowed to cover a protected anchor.
# - This keeps the solver's bbox-based ordering stable while preserving the original noisy overlap
#   style elsewhere.
#
# visual_observability_and_cleanliness:
# - The changed rule is human-visible because tied bars always occur and their corresponding frames
#   are spatially comparable directly in the input.
# - The generator avoids using hidden properties such as color id for sorting, and it protects only
#   a single anchor pixel per tied frame, so the images remain legible rather than heavily marked.


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


def _position_order(
    group: list[tuple[int, int, np.ndarray, tuple[int, int]]],
) -> list[tuple[int, int, np.ndarray, tuple[int, int]]]:
    return sorted(group, key=lambda item: (int(item[3][0]), int(item[3][1])))


def main(grid: np.ndarray) -> np.ndarray:
    """
    Convert each colored partition frame into a horizontal bar whose length equals its enclosed-room count.

    Input structure:
    - Black background with several colored rectilinear frame drawings.
    - One extra non-black color appears as scattered noise and also damages some frame-wall pixels.
    - Real objects may overlap, but each genuine drawing is still a rectangular wall layout enclosing one or more rooms.

    Rule:
    - For each non-background color, rebuild its wall scaffold from dense row/column evidence inside that color's visible bbox.
    - Count how many 4-connected background regions are fully enclosed by the reconstructed walls.
    - The color with zero enclosed regions is the padding/noise color rather than a true frame.
    - Sort true frames by room count ascending; when room counts tie, compare the visible bbox anchors in row-major order (topmost first, then leftmost).

    Key insight:
    - Room count is recovered from the rectilinear wall pattern, while tie resolution no longer depends on overlap depth at all: it comes from where the frame sits in the grid.
    """
    g = np.asarray(grid, dtype=int)
    bg = int(get_bg(g))
    assert bg == 0

    counts: list[tuple[int, int, np.ndarray, tuple[int, int]]] = []
    for color in colors_present(g, exclude={bg}):
        rooms = _room_count_for_color(g, int(color))
        walls = _walls_for_color(g, int(color))
        b = bbox(g == int(color))
        assert b is not None
        counts.append((int(color), int(rooms), walls, (int(b[0]), int(b[1]))))

    drawings = [(color, rooms, walls, anchor) for color, rooms, walls, anchor in counts if rooms > 0]
    pad_colors = [color for color, rooms, _walls, _anchor in counts if rooms == 0]
    assert drawings
    assert len(pad_colors) == 1

    pad = int(pad_colors[0])
    ordered: list[tuple[int, int, np.ndarray, tuple[int, int]]] = []
    for rooms in sorted({rooms for _color, rooms, _walls, _anchor in drawings}):
        group = [item for item in drawings if item[1] == rooms]
        ordered.extend(_position_order(group))

    width = max(rooms for _color, rooms, _walls, _anchor in ordered)

    out = np.full((len(ordered), width), pad, dtype=int)
    for r, (color, rooms, _walls, _anchor) in enumerate(ordered):
        out[r, :rooms] = int(color)
    return out


# --------------- generator (gpt-5) ---------------

# ---- CFG: Tunable generation parameters ----
_FRAME_COUNT_RANGE = (3, 5)           # number of colored frames per puzzle
_ROOM_COUNT_RANGE = (1, 6)            # rooms per frame
_TIE_PROBABILITY = 1.0                # always create one tied pair so the new D2 rule is visible
_TIE_SAME_ROW_PROBABILITY = 0.35      # sometimes exercise the secondary column tie-break
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
_PLACEMENT_RETRY = 150                # attempts for special placement subroutines
_GEN_RETRIES = 500                    # outer retry limit


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


def _covers_point(pos: tuple[int, int], shape: tuple[int, int], point: tuple[int, int]) -> bool:
    r0, c0 = pos
    h, w = shape
    pr, pc = point
    return r0 <= pr < r0 + h and c0 <= pc < c0 + w


def _hits_protected_anchor(
    pos: tuple[int, int],
    shape: tuple[int, int],
    protected_points: set[tuple[int, int]],
) -> bool:
    return any(_covers_point(pos, shape, point) for point in protected_points)


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
    frames: list[tuple[int, int, np.ndarray]],
    tie_pair: tuple[int, int],
    grid_h: int,
    grid_w: int,
) -> tuple[set[tuple[int, int]], str] | None:
    i, j = tie_pair
    modes = ["same_row", "different_row"] if random.random() < _TIE_SAME_ROW_PROBABILITY else ["different_row", "same_row"]

    for mode in modes:
        for _ in range(_PLACEMENT_RETRY):
            if mode == "same_row":
                left_idx, right_idx = (i, j) if random.random() < 0.5 else (j, i)
                h_left, w_left = frames[left_idx][2].shape
                h_right, w_right = frames[right_idx][2].shape
                needed_w = w_left + 1 + w_right
                if needed_w > grid_w or max(h_left, h_right) > grid_h:
                    break
                row = random.randint(0, grid_h - max(h_left, h_right))
                left_col = random.randint(0, grid_w - needed_w)
                right_col = random.randint(left_col + w_left + 1, grid_w - w_right)
                positions[left_idx] = (row, left_col)
                positions[right_idx] = (row, right_col)
            else:
                top_idx, bottom_idx = (i, j) if random.random() < 0.5 else (j, i)
                h_top, w_top = frames[top_idx][2].shape
                h_bottom, w_bottom = frames[bottom_idx][2].shape
                needed_h = h_top + 1 + h_bottom
                if needed_h > grid_h or max(w_top, w_bottom) > grid_w:
                    break
                top_row = random.randint(0, grid_h - needed_h)
                bottom_row = random.randint(top_row + h_top + 1, grid_h - h_bottom)
                top_col = random.randint(0, grid_w - w_top)
                bottom_col = random.randint(0, grid_w - w_bottom)
                positions[top_idx] = (top_row, top_col)
                positions[bottom_idx] = (bottom_row, bottom_col)

            assert positions[i] is not None and positions[j] is not None
            protected = {
                (int(positions[i][0]), int(positions[i][1])),
                (int(positions[j][0]), int(positions[j][1])),
            }
            return protected, mode
    return None


def _random_position_avoiding_points(
    frame_arr: np.ndarray,
    grid_h: int,
    grid_w: int,
    protected_points: set[tuple[int, int]],
) -> tuple[int, int] | None:
    h, w = frame_arr.shape
    for _ in range(_PLACEMENT_RETRY):
        r0 = random.randint(0, grid_h - h)
        c0 = random.randint(0, grid_w - w)
        if not _hits_protected_anchor((r0, c0), frame_arr.shape, protected_points):
            return r0, c0
    return None


def _force_overlap(
    positions: list[tuple[int, int] | None],
    i: int,
    j: int,
    frames: list[tuple[int, int, np.ndarray]],
    grid_h: int,
    grid_w: int,
    protected_points: set[tuple[int, int]],
) -> bool:
    """Try to reposition frame j so it overlaps frame i without covering protected anchors."""
    fi_h, fi_w = frames[i][2].shape
    fj_h, fj_w = frames[j][2].shape
    ri, ci = positions[i]
    r0_min = max(0, ri - fj_h + 2)
    r0_max = min(grid_h - fj_h, ri + fi_h - 2)
    c0_min = max(0, ci - fj_w + 2)
    c0_max = min(grid_w - fj_w, ci + fi_w - 2)
    if r0_min > r0_max or c0_min > c0_max:
        return False
    for _ in range(_OVERLAP_RETRY):
        rj = random.randint(r0_min, r0_max)
        cj = random.randint(c0_min, c0_max)
        if _hits_protected_anchor((rj, cj), frames[j][2].shape, protected_points):
            continue
        if _boxes_overlap((ri, ci), frames[i][2].shape, (rj, cj), frames[j][2].shape):
            positions[j] = (rj, cj)
            return True
    return False


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


def _scatter_noise(grid, frame_mask, noise_color, grid_h, grid_w, protected_points):
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

    protected_mask = np.zeros((grid_h, grid_w), dtype=bool)
    for r, c in protected_points:
        protected_mask[int(r), int(c)] = True

    damageable = frame_mask & ~protected_mask
    wall_rs, wall_cs = np.where(damageable)
    n_wall = len(wall_rs)
    max_damage = min(noise_count - placed, int(_WALL_DAMAGE_FRACTION * n_wall))
    if max_damage > 0:
        damage_perm = np.random.permutation(n_wall)[:max_damage]
        grid[wall_rs[damage_perm], wall_cs[damage_perm]] = noise_color


def _gen_attempt(return_meta: bool = False) -> np.ndarray | tuple[np.ndarray, dict[str, object]] | None:
    """Single attempt to generate a valid puzzle grid."""
    frame_count = random.randint(*_FRAME_COUNT_RANGE)

    available = list(range(_ROOM_COUNT_RANGE[0], _ROOM_COUNT_RANGE[1] + 1))
    allow_tie = random.random() < _TIE_PROBABILITY and frame_count >= 3
    if allow_tie:
        distinct_needed = frame_count - 1
        chosen = random.sample(available, min(distinct_needed, len(available)))
        if len(chosen) < distinct_needed:
            return None
        duplicated_rooms = random.choice(chosen)
        room_counts = chosen + [duplicated_rooms]
    else:
        if frame_count > len(available):
            return None
        duplicated_rooms = None
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

    positions: list[tuple[int, int] | None] = [None] * frame_count
    protected_points: set[tuple[int, int]] = set()
    tie_mode = "none"
    tie_pair: tuple[int, int] | None = None

    if duplicated_rooms is not None:
        tied_indices = [idx for idx, rooms in enumerate(room_counts) if rooms == duplicated_rooms]
        assert len(tied_indices) == 2
        tie_pair = (int(tied_indices[0]), int(tied_indices[1]))
        placed = _place_tied_pair(positions, frames, tie_pair, grid_h, grid_w)
        if placed is None:
            return None
        protected_points, tie_mode = placed

    for idx in sorted(range(frame_count), key=lambda i: -frames[i][2].size):
        if positions[idx] is not None:
            continue
        pos = _random_position_avoiding_points(frames[idx][2], grid_h, grid_w, protected_points)
        if pos is None:
            return None
        positions[idx] = pos

    assert all(pos is not None for pos in positions)

    tied_set = set() if tie_pair is None else set(tie_pair)
    non_tied = [
        (i, j)
        for i in range(frame_count)
        for j in range(i + 1, frame_count)
        if room_counts[i] != room_counts[j] and i not in tied_set and j not in tied_set
    ]
    n_extra = random.randint(*_EXTRA_OVERLAP_RANGE)
    if n_extra > 0 and non_tied:
        random.shuffle(non_tied)
        for i, j in non_tied[: min(n_extra, len(non_tied))]:
            mover_i, mover_j = (i, j) if random.random() < 0.5 else (j, i)
            _force_overlap(positions, mover_i, mover_j, frames, grid_h, grid_w, protected_points)

    render_order = list(range(frame_count))
    random.shuffle(render_order)

    grid, frame_mask = _render_frames(frames, positions, render_order, grid_h, grid_w)
    _scatter_noise(grid, frame_mask, noise_color, grid_h, grid_w, protected_points)

    try:
        output = main(grid)
    except Exception:
        return None

    frame_info = sorted(
        [(room_counts[i], positions[i][0], positions[i][1], frame_colors[i]) for i in range(frame_count)],
        key=lambda item: (int(item[0]), int(item[1]), int(item[2])),
    )
    max_rooms = max(rc for rc, _r0, _c0, _color in frame_info)
    expected = np.full((frame_count, max_rooms), noise_color, dtype=int)
    for r, (rc, _r0, _c0, color) in enumerate(frame_info):
        expected[r, :rc] = color

    if not np.array_equal(output, expected):
        return None

    if return_meta:
        return grid, {
            "positions": [tuple(map(int, pos)) for pos in positions],
            "room_counts": list(map(int, room_counts)),
            "frame_colors": list(map(int, frame_colors)),
            "noise_color": int(noise_color),
            "tie_pair": tie_pair,
            "tie_mode": tie_mode,
            "render_order": list(map(int, render_order)),
        }
    return grid


def generate_input() -> np.ndarray:
    for _ in range(_GEN_RETRIES):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError(f"Failed to generate valid input after {_GEN_RETRIES} attempts")
