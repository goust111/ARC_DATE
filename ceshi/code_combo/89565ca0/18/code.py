# Combo variant of 89565ca0 (index 18)
# combo_spec: D1/3+D15/2
# Name: 按网格线数排序+仅水平分隔

from __future__ import annotations

import math
import random
from collections import Counter

import numpy as np

from arc_utils import bbox, colors_present, crop_bbox, get_bg, label_mask


"""
Design analysis for combo D1/3 + D15/2
======================================

a) Concrete behavior:
   - Each real object is no longer a full 2D grid. It is now a rectangle with only two vertical side walls and several
     horizontal divider rows, so the rooms stack as horizontal strips.
   - The output bars still encode room count by length, but row order now uses `grid-line count = 2 side walls +
     recognizable horizontal line rows`.
   - To keep the two dimensions from collapsing into the same ranking, some internal horizontal dividers contain one
     wide opening. That opening merges two adjacent strip-rooms, so `room_count` drops while the divider row is still
     visually present and still counted as a line.
   - Small trace:
       frame A: 6 total lines, 3 rooms
       frame B: 7 total lines, 2 rooms
     Original task would order by rooms as B then A. This combo orders by lines as A then B, while still drawing bars
     of lengths 3 and 2 respectively.

b) Complexity calibration:
   - If every divider were full-width, then `rooms = horizontal_lines - 1` and the new sort key would always match the
     old one. That would make D1 ineffective inside D15.
   - The generator therefore forces at least one divider-opening frame per puzzle and at least one pairwise inversion
     between room ranking and line ranking.
   - Line counts stay in a compact observable range 4-9, and room counts stay in 1-6, so the examples remain easy to
     compare by eye.

c) Implicit assumptions audit:
   - The original `_reconstruct_walls()` assumed arbitrary internal vertical partitions. That assumption is replaced by
     a stronger one: exactly two dense vertical boundary columns plus several dense horizontal rows.
   - Sorting can no longer reuse the bar length because length still encodes rooms while ordering uses line count.
   - Tie breaking now applies to equal line counts rather than equal room counts, so the generator must ensure such
     pairs actually overlap whenever they are created.
   - Noise detection via `rooms == 0` still holds because the scattered color does not reconstruct into a closed frame
     with two side walls and horizontal separators.

d) Placement legality:
   - Every frame is built inside its own bbox, then placed fully in-bounds on the global grid.
   - Overlaps are allowed, but equal-line-count pairs are explicitly forced to overlap so cover-order inference is
     well-defined instead of ambiguous.
   - The generator validates each sample with the solver oracle and rejects any placement whose recovered statistics
     drift away from the intended combo behavior.

e) Visual observability:
   - A human can count horizontal divider rows directly, and can also see that some dividers are broken, which explains
     why bar length is not just "number of rows".
   - Because at least one sample-level inversion is enforced, the training pairs expose that the order depends on line
     count rather than room count.
   - No invisible metadata such as color ids or creation order is used as a primary rule.

f) Visual cleanliness:
   - The objects remain axis-aligned 1px frames on black, with only one noise color added.
   - Internal structure is simpler than the original task because there are no internal vertical walls, so even with a
     few overlaps the objects stay readable.
   - Divider openings are single contiguous gaps rather than many scattered deletions, which keeps the room-merging cue
     visually crisp.

g) New solver assumptions checked against generation:
   - `_detect_line_positions()` assumes the true frame has two tall columns that survive damage. The generator preserves
     at least 55% of every side wall and uses moderate overlap/noise.
   - `_reconstruct_walls()` assumes any intentional divider gap is much wider than random damage gaps. The generator
     uses wide openings and the reconstructor only preserves the longest significant internal zero-run.
   - `_cover_order()` assumes equal-key frames have an overlap relation that induces a unique bottom-first order. The
     generator either forces that overlap or rejects the sample.
"""


_REPAIR_GAP_MAX = 2


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


def _fill_short_gaps(arr: np.ndarray, max_gap: int) -> np.ndarray:
    out = arr.copy()
    i = 0
    while i < len(out):
        if out[i]:
            i += 1
            continue
        j = i
        while j < len(out) and not out[j]:
            j += 1
        if i > 0 and j < len(out) and (j - i) <= max_gap:
            out[i:j] = True
        i = j
    return out


def _zero_runs(arr: np.ndarray) -> list[tuple[int, int]]:
    runs: list[tuple[int, int]] = []
    i = 0
    while i < len(arr):
        if arr[i]:
            i += 1
            continue
        j = i
        while j < len(arr) and not arr[j]:
            j += 1
        runs.append((i, j - 1))
        i = j
    return runs


def _detect_line_positions(mask: np.ndarray) -> tuple[list[int], list[int]]:
    h, w = mask.shape
    if h < 2 or w < 2:
        return [], []

    col_counts = mask.sum(axis=0)
    v_threshold = max(3, math.ceil(0.45 * h))
    dense_cols = [int(c) for c, n in enumerate(col_counts) if int(n) >= v_threshold]
    if len(dense_cols) < 2:
        return [], []

    left = int(dense_cols[0])
    right = int(dense_cols[-1])
    if left >= right:
        return [], []

    span = right - left + 1
    row_counts = mask[:, left : right + 1].sum(axis=1)
    h_threshold = max(3, math.ceil(0.45 * span))
    hrows = [int(r) for r, n in enumerate(row_counts) if int(n) >= h_threshold]
    if len(hrows) < 2:
        return [left, right], []
    return [left, right], hrows


def _reconstruct_walls(mask: np.ndarray) -> np.ndarray:
    vcols, hrows = _detect_line_positions(mask)
    if len(vcols) < 2 or len(hrows) < 2:
        return mask.copy()

    left, right = int(vcols[0]), int(vcols[-1])
    top, bottom = int(hrows[0]), int(hrows[-1])
    span = right - left + 1
    min_open_gap = max(3, math.ceil(0.16 * span))
    min_side_hits = max(2, math.ceil(0.18 * span))

    walls = np.zeros_like(mask, dtype=bool)
    walls[top : bottom + 1, left] = True
    walls[top : bottom + 1, right] = True

    for r in hrows:
        row = mask[r, left : right + 1]
        repaired = _fill_short_gaps(row, _REPAIR_GAP_MAX)
        if r == top or r == bottom:
            walls[r, left : right + 1] = True
            continue

        internal_gaps = [run for run in _zero_runs(repaired) if run[0] > 0 and run[1] < span - 1]
        if not internal_gaps:
            walls[r, left : right + 1] = True
            continue

        gap_start, gap_end = max(internal_gaps, key=lambda run: run[1] - run[0] + 1)
        gap_len = int(gap_end - gap_start + 1)
        left_hits = int(repaired[:gap_start].sum())
        right_hits = int(repaired[gap_end + 1 :].sum())

        if gap_len >= min_open_gap and left_hits >= min_side_hits and right_hits >= min_side_hits:
            walls[r, left : left + gap_start] = True
            walls[r, left + gap_end + 1 : right + 1] = True
        else:
            walls[r, left : right + 1] = True

    return walls


def _count_grid_lines(walls: np.ndarray) -> int:
    vcols, hrows = _detect_line_positions(walls)
    if len(vcols) < 2 or len(hrows) < 2:
        return 0
    return int(len(vcols) + len(hrows))


def _frame_stats_for_color(g: np.ndarray, color: int) -> tuple[int, int, np.ndarray]:
    b = bbox(g == int(color))
    assert b is not None
    mask = crop_bbox(g == int(color), b).astype(bool)
    walls = _reconstruct_walls(mask)
    rooms = _count_enclosed_regions(walls)
    line_count = _count_grid_lines(walls)
    return int(rooms), int(line_count), walls


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
    g: np.ndarray, group: list[tuple[int, int, int, np.ndarray]]
) -> list[tuple[int, int, int, np.ndarray]]:
    if len(group) == 1:
        return group

    by_color = {int(item[0]): item for item in group}
    above: dict[int, set[int]] = {color: set() for color, _rooms, _lines, _walls in group}
    indegree: dict[int, int] = {color: 0 for color, _rooms, _lines, _walls in group}

    for i, (ca, _ra, _la, wa) in enumerate(group):
        for cb, _rb, _lb, wb in group[i + 1 :]:
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

    ordered: list[tuple[int, int, int, np.ndarray]] = []
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
    Convert each colored strip-partitioned frame into a horizontal bar whose length is its room count, but order bars by total grid-line count.

    Input structure:
    - Black background containing several colored rectangular frames plus one scattered noise color.
    - Each true frame has only two vertical side walls; all internal structure is horizontal, so rooms appear as stacked strips.
    - Some horizontal dividers may contain one wide opening, and overlaps/noise can erase parts of the visible walls.

    Rule:
    - For each non-black color, reconstruct the frame as two dense side walls together with the recognizable horizontal divider rows.
    - Count total grid lines as `#vertical side walls + #horizontal line rows`; this is the sorting key.
    - Count enclosed 4-connected background regions on the reconstructed frame; this room count sets the horizontal bar length.
    - The only color with zero enclosed rooms is the pad/noise color; sort real frames by line count, breaking equal keys by inferred cover order.

    Key insight:
    - In this combo, the visible scaffold is simpler to read than the room topology: divider rows can stay visible even when a wide opening merges two rows into one room, so line count and room count must be inferred separately.
    """
    g = np.asarray(grid, dtype=int)
    bg = int(get_bg(g))
    assert bg == 0

    stats: list[tuple[int, int, int, np.ndarray]] = []
    for color in colors_present(g, exclude={bg}):
        rooms, line_count, _walls_local = _frame_stats_for_color(g, int(color))
        walls = _walls_for_color(g, int(color))
        stats.append((int(color), int(rooms), int(line_count), walls))

    drawings = [(color, rooms, line_count, walls) for color, rooms, line_count, walls in stats if rooms > 0]
    pad_colors = [color for color, rooms, _line_count, _walls in stats if rooms == 0]
    assert drawings
    assert len(pad_colors) == 1

    pad = int(pad_colors[0])
    ordered: list[tuple[int, int, int, np.ndarray]] = []
    for line_count in sorted({line_count for _color, _rooms, line_count, _walls in drawings}):
        group = [item for item in drawings if item[2] == line_count]
        ordered.extend(_cover_order(g, group))

    width = max(rooms for _color, rooms, _line_count, _walls in ordered)
    out = np.full((len(ordered), width), pad, dtype=int)
    for r, (color, rooms, _line_count, _walls) in enumerate(ordered):
        out[r, :rooms] = int(color)
    return out


# --------------- generator (combo variant) ---------------

# ---- CFG: Tunable generation parameters ----
_FRAME_COUNT_RANGE = (3, 5)
_ROOM_COUNT_RANGE = (1, 6)
_LINE_COUNT_VALUES = (4, 5, 6, 7, 8, 9)
_TIE_PROBABILITY = 0.45
_GRID_SIZE_BY_FRAMES = {
    5: (25, 30, 25, 30, 16, 18),
    4: (22, 30, 22, 30, 18, 22),
    3: (20, 30, 20, 30, 22, 25),
}
_ROW_GAP_RANGE = (3, 5)
_FRAME_WIDTH_RANGE = (10, 18)
_OPENING_FRACTION_RANGE = (0.22, 0.32)
_NOISE_FRACTION_RANGE = (0.035, 0.075)
_WALL_DAMAGE_FRACTION = 0.18
_EXTRA_OVERLAP_RANGE = (0, 1)
_OVERLAP_RETRY = 120
_SPEC_ATTEMPTS = 250
_GEN_RETRIES = 500

_FEASIBLE_ROOM_COUNTS_BY_LINES: dict[int, tuple[int, ...]] = {
    4: (1,),
    5: (1, 2),
    6: (1, 2, 3),
    7: (1, 2, 3, 4),
    8: (1, 2, 3, 4, 5),
    9: (1, 2, 3, 4, 5, 6),
}


def _random_gaps(n: int, min_gap: int, max_gap: int, max_total: int) -> list[int] | None:
    if n * min_gap > max_total:
        return None
    gaps = [min_gap] * n
    remaining = max_total - sum(gaps)
    order = list(range(n))
    random.shuffle(order)
    for i in order:
        extra = min(max_gap - gaps[i], remaining)
        if extra > 0:
            add = random.randint(0, extra)
            gaps[i] += add
            remaining -= add
    random.shuffle(gaps)
    return gaps


def _build_frame(target_rooms: int, target_lines: int, max_h: int = 22, max_w: int = 25) -> np.ndarray | None:
    n_hrows = target_lines - 2
    n_bands = n_hrows - 1
    if target_rooms < 1 or target_rooms > n_bands:
        return None

    row_gaps = _random_gaps(n_bands, _ROW_GAP_RANGE[0], _ROW_GAP_RANGE[1], max_h - 1)
    if row_gaps is None:
        return None

    hrow_positions = [0]
    for gap in row_gaps:
        hrow_positions.append(hrow_positions[-1] + gap)

    frame_h = hrow_positions[-1] + 1
    min_w = min(_FRAME_WIDTH_RANGE[0], max_w)
    max_frame_w = min(_FRAME_WIDTH_RANGE[1], max_w)
    if frame_h > max_h or min_w > max_frame_w:
        return None

    frame_w = random.randint(min_w, max_frame_w)
    walls = np.zeros((frame_h, frame_w), dtype=bool)
    walls[:, 0] = True
    walls[:, frame_w - 1] = True
    for r in hrow_positions:
        walls[r, :] = True

    openings_needed = n_bands - target_rooms
    internal_rows = hrow_positions[1:-1]
    if openings_needed > len(internal_rows):
        return None

    chosen_rows = set(random.sample(internal_rows, openings_needed))
    min_open = 3
    max_open = max(min_open, int(math.floor(_OPENING_FRACTION_RANGE[1] * frame_w)))
    min_side_margin = 2
    if frame_w - 2 * min_side_margin <= min_open:
        return None

    for r in chosen_rows:
        open_len_lo = max(min_open, int(math.ceil(_OPENING_FRACTION_RANGE[0] * frame_w)))
        open_len_hi = min(max_open, frame_w - 2 * min_side_margin - 1)
        if open_len_lo > open_len_hi:
            return None
        gap_len = random.randint(open_len_lo, open_len_hi)
        gap_start = random.randint(min_side_margin, frame_w - min_side_margin - gap_len)
        walls[r, gap_start : gap_start + gap_len] = False

    if _count_enclosed_regions(walls) != target_rooms:
        return None
    if _count_grid_lines(walls) != target_lines:
        return None
    return walls


def _force_overlap(positions, i, j, frames, grid_h, grid_w) -> bool:
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
        if max(ri, rj) <= min(ri + fi_h - 1, rj + fj_h - 1) and max(ci, cj) <= min(ci + fi_w - 1, cj + fj_w - 1):
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


def _has_line_room_inversion(room_counts: list[int], line_counts: list[int]) -> bool:
    for i in range(len(room_counts)):
        for j in range(i + 1, len(room_counts)):
            if (line_counts[i] < line_counts[j] and room_counts[i] > room_counts[j]) or (
                line_counts[i] > line_counts[j] and room_counts[i] < room_counts[j]
            ):
                return True
    return False


def _sample_specs(frame_count: int) -> list[tuple[int, int]] | None:
    for _ in range(_SPEC_ATTEMPTS):
        allow_tie = random.random() < _TIE_PROBABILITY and frame_count >= 3
        if allow_tie:
            distinct_needed = frame_count - 1
            chosen = random.sample(list(_LINE_COUNT_VALUES), distinct_needed)
            line_counts = chosen + [random.choice(chosen)]
        else:
            line_counts = random.sample(list(_LINE_COUNT_VALUES), frame_count)
        random.shuffle(line_counts)

        room_counts = [random.choice(_FEASIBLE_ROOM_COUNTS_BY_LINES[line_count]) for line_count in line_counts]
        if len(set(room_counts)) < 2:
            continue
        if max(Counter(room_counts).values()) > 2:
            continue
        if not _has_line_room_inversion(room_counts, line_counts):
            continue
        if not any(room_count == line_count - 3 for room_count, line_count in zip(room_counts, line_counts)):
            continue
        if not any(room_count < line_count - 3 for room_count, line_count in zip(room_counts, line_counts)):
            continue
        return list(zip(room_counts, line_counts))
    return None


def _gen_attempt() -> np.ndarray | None:
    frame_count = random.randint(*_FRAME_COUNT_RANGE)
    specs = _sample_specs(frame_count)
    if specs is None:
        return None

    chosen_colors = random.sample(list(range(1, 10)), frame_count + 1)
    frame_colors = chosen_colors[:frame_count]
    noise_color = chosen_colors[frame_count]

    cfg = _GRID_SIZE_BY_FRAMES.get(frame_count, _GRID_SIZE_BY_FRAMES[3])
    h_min, h_max, w_min, w_max, f_max_h, f_max_w = cfg
    grid_h = random.randint(h_min, h_max)
    grid_w = random.randint(w_min, w_max)

    frames = []
    room_counts = []
    line_counts = []
    for color, (target_rooms, target_lines) in zip(frame_colors, specs):
        frame_data = _build_frame(target_rooms, target_lines, max_h=f_max_h, max_w=f_max_w)
        if frame_data is None:
            return None
        fh, fw = frame_data.shape
        if fh > grid_h - 2 or fw > grid_w - 2:
            return None
        frames.append((color, target_rooms, frame_data, target_lines))
        room_counts.append(target_rooms)
        line_counts.append(target_lines)

    positions = [None] * frame_count
    for idx in sorted(range(frame_count), key=lambda i: -frames[i][2].size):
        fh, fw = frames[idx][2].shape
        positions[idx] = (random.randint(0, grid_h - fh), random.randint(0, grid_w - fw))

    tied_pairs = [(i, j) for i in range(frame_count) for j in range(i + 1, frame_count) if line_counts[i] == line_counts[j]]
    for i, j in tied_pairs:
        if not _force_overlap(positions, i, j, frames, grid_h, grid_w):
            return None

    non_tied = [(i, j) for i in range(frame_count) for j in range(i + 1, frame_count) if line_counts[i] != line_counts[j]]
    n_extra = random.randint(*_EXTRA_OVERLAP_RANGE)
    if n_extra > 0 and non_tied:
        random.shuffle(non_tied)
        for i, j in non_tied[: min(n_extra, len(non_tied))]:
            _force_overlap(positions, i, j, frames, grid_h, grid_w)

    render_order = list(range(frame_count))
    random.shuffle(render_order)

    grid, frame_mask = _render_frames(frames, positions, render_order, grid_h, grid_w)
    _scatter_noise(grid, frame_mask, noise_color, grid_h, grid_w)

    recovered = {}
    for color in colors_present(grid, exclude={0}):
        rooms, line_count, _walls = _frame_stats_for_color(grid, int(color))
        recovered[int(color)] = (int(rooms), int(line_count))

    if set(recovered) != set(frame_colors) | {noise_color}:
        return None

    pad_colors = [color for color, (rooms, _line_count) in recovered.items() if rooms == 0]
    if len(pad_colors) != 1 or pad_colors[0] != noise_color:
        return None

    recovered_drawings = []
    recovered_room_counts = []
    recovered_line_counts = []
    for idx, color in enumerate(frame_colors):
        rooms, line_count = recovered[int(color)]
        if rooms != room_counts[idx] or line_count != line_counts[idx]:
            return None
        recovered_drawings.append((idx, int(color), rooms, line_count))
        recovered_room_counts.append(rooms)
        recovered_line_counts.append(line_count)

    if not _has_line_room_inversion(recovered_room_counts, recovered_line_counts):
        return None
    if not any(rooms < line_count - 3 for rooms, line_count in zip(recovered_room_counts, recovered_line_counts)):
        return None

    try:
        output = main(grid)
    except Exception:
        return None

    frame_info = sorted(
        (line_count, render_order.index(i), rooms, color)
        for i, color, rooms, line_count in recovered_drawings
    )
    max_rooms = max(rooms for _i, _color, rooms, _line_count in recovered_drawings)
    expected = np.full((frame_count, max_rooms), noise_color, dtype=int)
    for r, (_line_count, _render_rank, rooms, color) in enumerate(frame_info):
        expected[r, :rooms] = color

    return grid if np.array_equal(output, expected) else None


def generate_input() -> np.ndarray:
    for _ in range(_GEN_RETRIES):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError(f"Failed to generate valid input after {_GEN_RETRIES} attempts")
