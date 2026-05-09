# Combo variant of 89565ca0 (index 13)
# combo_spec: D5/0+D7/2
# Name: 垂直色条+颜色重映射

from __future__ import annotations

import math
import random

import numpy as np

from arc_utils import bbox, colors_present, crop_bbox, get_bg, label_mask

# Design analysis:
# - Concrete behavior trace:
#   Original task sorts frames by room count and emits one left-filled horizontal row per frame,
#   reusing the frame's own color. In this combo, the room-count extraction and tie-breaking stay
#   the same, but the encoding rotates into columns: if sorted room counts are [1, 2, 4], the
#   old output was three rows with lengths 1/2/4, while the new output is a max_rooms x 3 grid
#   whose columns have heights 1/2/4. Colors also stop following the source frames: column 0 uses
#   the first fixed rank color, column 1 the second, etc. This makes the new rule visibly different
#   in both geometry and color semantics.
# - Complexity calibration / observability:
#   The remap uses a fixed short palette prefix indexed by sorted position, not a per-instance
#   random permutation. That keeps the rule learnable from 2-4 examples because the same left-to-
#   right rank colors recur across samples. The generator also avoids frequent accidental matches
#   between "sorted input colors" and "rank colors", so the color-transfer behavior is visible.
# - Assumption audit:
#   _reconstruct_walls, _room_count_for_color and _cover_order still rely on the original frame
#   geometry assumptions: one true noise color has zero rooms, every real frame has >=1 room, and
#   tied room-count frames overlap enough to reveal a unique bottom-first cover order. Those remain
#   valid here because only the output encoding changes. The new assumption is that frame_count
#   never exceeds the fixed rank-palette length; the generator enforces 3-5 frames against 5 rank
#   colors, and it keeps the noise color outside the rank palette to preserve clear padding.
# - Placement legality / cleanliness:
#   Frames stay in-bounds, tie pairs are forced to overlap for deterministic cover-order recovery,
#   and non-tied overlaps remain optional. Output columns are separated by full columns, so the new
#   vertical encoding is visually cleaner than drawing extra markers or overlays.


_RANK_COLORS = (1, 2, 3, 4, 5)


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


def _ordered_drawings(
    g: np.ndarray,
) -> tuple[list[tuple[int, int, np.ndarray]], int]:
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

    ordered: list[tuple[int, int, np.ndarray]] = []
    for rooms in sorted({rooms for _color, rooms, _walls in drawings}):
        group = [item for item in drawings if item[1] == rooms]
        ordered.extend(_cover_order(g, group))
    return ordered, int(pad_colors[0])


def _build_output(ordered: list[tuple[int, int, np.ndarray]], pad: int) -> np.ndarray:
    assert len(ordered) <= len(_RANK_COLORS)
    height = max(rooms for _color, rooms, _walls in ordered)
    width = len(ordered)
    out = np.full((height, width), pad, dtype=int)
    for c, (_src_color, rooms, _walls) in enumerate(ordered):
        out[:rooms, c] = _RANK_COLORS[c]
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Convert each reconstructed frame into a vertical rank-colored bar whose height is its room count.

    Input structure:
    - Black background with several axis-aligned colored frame grids plus one scattered noise color.
    - Real frames may be partially damaged by that noise and may overlap one another.
    - Each true frame still forms one or more enclosed background rooms after rectilinear reconstruction.

    Rule:
    - Recover each non-noise frame by rebuilding dense horizontal and vertical wall segments, then count its enclosed 4-connected rooms.
    - Treat the unique non-background color with zero enclosed rooms as the padding/noise color.
    - Sort frames by room count ascending; if two frames tie, order them bottom-first according to which reconstructed frame is covered by the other.
    - Emit one vertical bar per sorted frame: column height equals the room count, the column color is reassigned from a fixed rank palette by sorted position, and the remaining cells are filled with the noise color.

    Key insight:
    - The geometric evidence in the input still determines only the ordering and the bar heights; the output colors are no longer copied from the source frames, so the solver must separate "which object is this?" from "which rank slot does it occupy?".
    """
    g = np.asarray(grid, dtype=int)
    ordered, pad = _ordered_drawings(g)
    return _build_output(ordered, pad)


# --------------- generator (combo variant, output-first) ---------------

_FRAME_COUNT_RANGE = (3, 5)
_ROOM_COUNT_RANGE = (1, 6)
_TIE_PROBABILITY = 0.35
_GRID_SIZE_BY_FRAMES = {
    5: (30, 36, 30, 36, 14, 16),
    4: (26, 34, 26, 34, 16, 18),
    3: (22, 30, 22, 30, 20, 22),
}
_GAP_RANGE_BY_FRAMES = {
    5: (3, 8),
    4: (3, 10),
    3: (3, 13),
}
_NOISE_FRACTION_RANGE = (0.03, 0.07)
_WALL_DAMAGE_FRACTION = 0.12
_OVERLAP_RETRY = 100
_GEN_RETRIES = 300
_NOISE_COLOR_CHOICES = (6, 7, 8, 9)


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
    target_rooms: int,
    max_h: int = 22,
    max_w: int = 25,
    gap_range: tuple[int, int] = (3, 13),
) -> np.ndarray | None:
    min_gap, max_gap = gap_range
    if target_rooms == 1:
        gap_h = random.randint(min_gap, min(max_gap, max_h - 2))
        gap_w = random.randint(min_gap, min(max_gap, max_w - 2))
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


def _rectangles_overlap(
    a: tuple[int, int, int, int], b: tuple[int, int, int, int], margin: int = 0
) -> bool:
    ar0, ac0, ah, aw = a
    br0, bc0, bh, bw = b
    return not (
        ar0 + ah - 1 + margin < br0
        or br0 + bh - 1 + margin < ar0
        or ac0 + aw - 1 + margin < bc0
        or bc0 + bw - 1 + margin < ac0
    )


def _partial_overlap_starts(anchor: int, anchor_len: int, moving_len: int, limit: int) -> list[int]:
    starts = []
    max_start = limit - moving_len
    if max_start < 0:
        return starts
    lo_before = max(0, anchor - moving_len + 2)
    hi_before = min(anchor - 1, max_start)
    if lo_before <= hi_before:
        starts.extend(range(lo_before, hi_before + 1))
    lo_after = anchor + 1
    hi_after = min(max_start, anchor + anchor_len - 2)
    if lo_after <= hi_after:
        starts.extend(range(lo_after, hi_after + 1))
    return starts


def _find_non_overlapping_position(
    wall_arr: np.ndarray,
    occupied: list[tuple[int, int, int, int]],
    grid_h: int,
    grid_w: int,
    margin: int = 1,
    attempts: int = 400,
) -> tuple[int, int] | None:
    fh, fw = wall_arr.shape
    for _ in range(attempts):
        r0 = random.randint(0, grid_h - fh)
        c0 = random.randint(0, grid_w - fw)
        rect = (r0, c0, fh, fw)
        if not any(_rectangles_overlap(rect, other, margin=margin) for other in occupied):
            return (r0, c0)
    return None


def _find_forced_overlap_position(
    anchor_pos: tuple[int, int],
    anchor_wall: np.ndarray,
    moving_wall: np.ndarray,
    occupied: list[tuple[int, int, int, int]],
    grid_h: int,
    grid_w: int,
) -> tuple[int, int] | None:
    ar0, ac0 = anchor_pos
    ah, aw = anchor_wall.shape
    mh, mw = moving_wall.shape
    candidates: list[tuple[int, int]] = []

    row_starts = _partial_overlap_starts(ar0, ah, mh, grid_h)
    col_starts = _partial_overlap_starts(ac0, aw, mw, grid_w)

    for r0 in row_starts:
        candidates.append((r0, ac0))
        right_aligned = ac0 + aw - mw
        if 0 <= right_aligned <= grid_w - mw:
            candidates.append((r0, right_aligned))
    for c0 in col_starts:
        candidates.append((ar0, c0))
        bottom_aligned = ar0 + ah - mh
        if 0 <= bottom_aligned <= grid_h - mh:
            candidates.append((bottom_aligned, c0))

    random.shuffle(candidates)
    for r0, c0 in candidates[: _OVERLAP_RETRY * 4]:
        if not (0 <= r0 <= grid_h - mh and 0 <= c0 <= grid_w - mw):
            continue
        rect = (r0, c0, mh, mw)
        if any(_rectangles_overlap(rect, other, margin=1) for other in occupied):
            continue
        overlap_h = min(ar0 + ah, r0 + mh) - max(ar0, r0)
        overlap_w = min(ac0 + aw, c0 + mw) - max(ac0, c0)
        if overlap_h >= 2 and overlap_w >= 1:
            return (r0, c0)
    return None


def _choose_tie_pair(room_counts: list[int]) -> tuple[int, int] | None:
    for i in range(len(room_counts)):
        for j in range(i + 1, len(room_counts)):
            if room_counts[i] == room_counts[j]:
                return (i, j)
    return None


def _place_frames(
    wall_arrays: list[np.ndarray],
    room_counts: list[int],
    grid_h: int,
    grid_w: int,
) -> list[tuple[int, int]] | None:
    frame_count = len(wall_arrays)
    positions: list[tuple[int, int] | None] = [None] * frame_count
    occupied: list[tuple[int, int, int, int]] = []

    tie_pair = _choose_tie_pair(room_counts)
    place_order = sorted(range(frame_count), key=lambda i: -wall_arrays[i].size)

    if tie_pair is not None:
        a, b = tie_pair
        if wall_arrays[b].size > wall_arrays[a].size:
            a, b = b, a
        anchor_pos = _find_non_overlapping_position(wall_arrays[a], occupied, grid_h, grid_w, margin=0)
        if anchor_pos is None:
            return None
        positions[a] = anchor_pos
        ah, aw = wall_arrays[a].shape
        occupied.append((anchor_pos[0], anchor_pos[1], ah, aw))

        overlap_pos = _find_forced_overlap_position(
            anchor_pos, wall_arrays[a], wall_arrays[b], occupied=[], grid_h=grid_h, grid_w=grid_w
        )
        if overlap_pos is None:
            return None
        positions[b] = overlap_pos
        bh, bw = wall_arrays[b].shape
        occupied.append((overlap_pos[0], overlap_pos[1], bh, bw))
        place_order = [idx for idx in place_order if idx not in {a, b}]

    for idx in place_order:
        pos = _find_non_overlapping_position(wall_arrays[idx], occupied, grid_h, grid_w)
        if pos is None:
            return None
        positions[idx] = pos
        fh, fw = wall_arrays[idx].shape
        occupied.append((pos[0], pos[1], fh, fw))

    assert all(pos is not None for pos in positions)
    return [(int(r), int(c)) for r, c in positions]


def _render_frames(frame_colors, wall_arrays, positions, render_order, grid_h, grid_w):
    grid = np.zeros((grid_h, grid_w), dtype=int)
    for idx in render_order:
        r0, c0 = positions[idx]
        wall_arr = wall_arrays[idx]
        color = frame_colors[idx]
        rs, cs = np.where(wall_arr)
        grid[rs + r0, cs + c0] = color

    frame_mask = np.zeros((grid_h, grid_w), dtype=bool)
    for idx in render_order:
        r0, c0 = positions[idx]
        wall_arr = wall_arrays[idx]
        color = frame_colors[idx]
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


def _ordered_indices(room_counts: list[int], render_order: list[int]) -> list[int]:
    return [
        idx
        for _rooms, _depth, idx in sorted(
            (room_counts[i], render_order.index(i), i) for i in range(len(room_counts))
        )
    ]


def _choose_frame_colors(frame_count: int, noise_color: int, ordered_indices: list[int]) -> list[int] | None:
    available = [c for c in range(1, 10) if c != noise_color]
    best: list[int] | None = None
    best_matches = frame_count + 1
    for _ in range(200):
        candidate = random.sample(available, frame_count)
        matches = sum(candidate[idx] == _RANK_COLORS[pos] for pos, idx in enumerate(ordered_indices))
        if matches < best_matches:
            best = candidate
            best_matches = matches
        if matches == 0:
            return candidate
    return best if best is not None and best_matches <= 1 else None


def _expected_output(room_counts: list[int], render_order: list[int], noise_color: int) -> np.ndarray:
    ordered_indices = _ordered_indices(room_counts, render_order)
    ordered = [(0, room_counts[idx], np.zeros((1, 1), dtype=bool)) for idx in ordered_indices]
    return _build_output(ordered, noise_color)


def _gen_attempt() -> np.ndarray | None:
    frame_count = random.randint(*_FRAME_COUNT_RANGE)
    assert frame_count <= len(_RANK_COLORS)

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

    cfg = _GRID_SIZE_BY_FRAMES.get(frame_count, _GRID_SIZE_BY_FRAMES[3])
    h_min, h_max, w_min, w_max, f_max_h, f_max_w = cfg
    gap_range = _GAP_RANGE_BY_FRAMES.get(frame_count, _GAP_RANGE_BY_FRAMES[3])
    grid_h = random.randint(h_min, h_max)
    grid_w = random.randint(w_min, w_max)

    wall_arrays = []
    for target_rooms in room_counts:
        frame_data = _build_frame(
            target_rooms,
            max_h=f_max_h,
            max_w=f_max_w,
            gap_range=gap_range,
        )
        if frame_data is None:
            return None
        fh, fw = frame_data.shape
        if fh > grid_h - 2 or fw > grid_w - 2:
            return None
        wall_arrays.append(frame_data)

    positions = _place_frames(wall_arrays, room_counts, grid_h, grid_w)
    if positions is None:
        return None

    render_order = list(range(frame_count))
    random.shuffle(render_order)
    noise_color = random.choice(_NOISE_COLOR_CHOICES)
    ordered_indices = _ordered_indices(room_counts, render_order)
    frame_colors = _choose_frame_colors(frame_count, noise_color, ordered_indices)
    if frame_colors is None:
        return None

    grid, frame_mask = _render_frames(frame_colors, wall_arrays, positions, render_order, grid_h, grid_w)
    _scatter_noise(grid, frame_mask, noise_color, grid_h, grid_w)

    try:
        output = main(grid)
    except Exception:
        return None

    expected = _expected_output(room_counts, render_order, noise_color)
    return grid if np.array_equal(output, expected) else None


def generate_input() -> np.ndarray:
    for _ in range(_GEN_RETRIES):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError(f"Failed to generate valid input after {_GEN_RETRIES} attempts")
