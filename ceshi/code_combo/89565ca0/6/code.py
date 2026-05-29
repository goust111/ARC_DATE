# Combo variant of 89565ca0 (index 6)
# combo_spec: D4/0+D6/0
# Name: 无噪声+黑色填充

from __future__ import annotations

import random

import numpy as np

from arc_utils import bbox, colors_present, crop_bbox, get_bg, label_mask


# Design analysis for D4/0 + D6/0
# 1. Concrete behavior trace:
#    - Original: each true frame color is mixed with one scattered noise color; some wall pixels are damaged; output rows
#      are padded with that same noise color.
#    - Fused variant: the input keeps only intact colored frames on black background, so there is no decoy color to
#      detect and no wall reconstruction step to perform. The output still encodes room counts as horizontal bars, but
#      the unused cells are black because black is now both the input background and the natural pad color.
#    - Example: if a blue frame has 2 rooms and a red frame has 4 rooms, the original could output [blue, blue, noise, noise]
#      and [red, red, red, red]. The fused task instead outputs [blue, blue, 0, 0] and [red, red, red, red].
# 2. Complexity calibration:
#    - Removing noise and black-padding both simplify the task. To avoid replacing one ambiguity with another, the
#      generator uses distinct room counts and non-overlapping frames, so sorting is still visible and never depends on
#      hidden color ids or arbitrary generation order.
# 3. Hidden-assumption audit for main():
#    - Iterating `colors_present(..., exclude={0})` assumes every non-black color is a real frame color; generator enforces
#      that by removing noise entirely.
#    - Cropping `bbox(g == color)` assumes one connected frame per color and no overlap between different colors; generator
#      places each frame once and forbids overlap.
#    - Sorting only by room count assumes there are no ties; generator samples unique room counts.
#    - Padding with black assumes the background is black; generator renders only on black background and `main()` asserts it.
# 4. Placement legality:
#    - Frames are shelf-packed with fixed gaps, so they never overlap, never leave the grid, and remain visually separated.
# 5. Visual observability and cleanliness:
#    - The change is obvious from samples: the noisy speckles are gone, the frames are clean, and the output padding is
#      visibly black rather than a fifth color. Using spacing between frames keeps the scene legible.
# 6. Inferability:
#    - No hidden color remapping remains. A human only needs to count rooms and notice that empty output cells match the
#      background color.


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


def _room_count_for_color(g: np.ndarray, color: int) -> int:
    b = bbox(g == int(color))
    assert b is not None
    mask = crop_bbox(g == int(color), b).astype(bool)
    return _count_enclosed_regions(mask)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Convert each clean colored frame into a horizontal bar whose length is its enclosed-room count.

    Input structure:
    - Black background with several non-overlapping axis-aligned colored line grids.
    - Each non-black color appears in exactly one intact rectangular frame with 1-pixel walls.
    - There are no scattered noise pixels and no damaged wall segments.

    Rule:
    - For each non-black color, crop its frame and count the 4-connected black regions fully enclosed by that color.
    - Sort the frame colors by enclosed-region count in ascending order.
    - Output one row per frame; fill the first `rooms` cells with the frame color and pad the rest with black up to the maximum room count.

    Key insight:
    - Once the noise color disappears, every remaining non-black color is already a complete frame, so the task reduces to direct room counting plus black background padding.
    """
    g = np.asarray(grid, dtype=int)
    bg = int(get_bg(g))
    assert bg == 0

    counts: list[tuple[int, int]] = []
    for color in colors_present(g, exclude={bg}):
        rooms = _room_count_for_color(g, int(color))
        assert rooms > 0
        counts.append((int(color), int(rooms)))

    assert counts
    assert len({rooms for _color, rooms in counts}) == len(counts)
    ordered = sorted(counts, key=lambda item: item[1])
    width = max(rooms for _color, rooms in ordered)

    out = np.zeros((len(ordered), width), dtype=int)
    for r, (color, rooms) in enumerate(ordered):
        out[r, :rooms] = int(color)
    return out


# --------------- generator (combo variant) ---------------

_FRAME_COUNT_RANGE = (3, 5)
_ROOM_COUNT_RANGE = (1, 6)
_GRID_MIN = 20
_GRID_MAX = 30
_FRAME_MAX_BY_COUNT = {
    5: (9, 11),
    4: (11, 13),
    3: (13, 15),
}
_CELL_GAP_BY_COUNT = {
    5: (2, 4),
    4: (2, 5),
    3: (3, 6),
}
_FRAME_GAP = 2
_GEN_RETRIES = 300


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


def _build_frame(target_rooms: int, max_h: int, max_w: int, gap_range: tuple[int, int]) -> np.ndarray | None:
    min_gap, max_gap = gap_range
    if target_rooms == 1:
        if max_h < min_gap + 2 or max_w < min_gap + 2:
            return None
        gap_h = random.randint(min_gap, max(min_gap, max_h - 2))
        gap_w = random.randint(min_gap, max(min_gap, max_w - 2))
        h, w = gap_h + 2, gap_w + 2
        walls = np.zeros((h, w), dtype=bool)
        walls[0, :] = True
        walls[h - 1, :] = True
        walls[:, 0] = True
        walls[:, w - 1] = True
        return walls

    possible_configs: list[tuple[int, int, int]] = []
    for nv in range(2, 6):
        for nh in range(2, 5):
            cells = (nv - 1) * (nh - 1)
            if target_rooms <= cells <= target_rooms + 3:
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
    merged = 0
    for seg in internal_segments:
        if merged >= cells_to_merge:
            break
        if seg[0] == "h":
            _, r, c0, c1, ca, cb = seg
            if union(ca, cb):
                walls[r, c0 + 1 : c1] = False
                merged += 1
        else:
            _, c, r0, r1, ca, cb = seg
            if union(ca, cb):
                walls[r0 + 1 : r1, c] = False
                merged += 1

    if merged < cells_to_merge:
        return None
    if _count_enclosed_regions(walls) != target_rooms:
        return None
    return walls


def _pack_frames(
    frames: list[tuple[int, int, np.ndarray]],
    gap: int,
) -> tuple[list[tuple[int, int]], int, int] | None:
    order = sorted(range(len(frames)), key=lambda i: (-frames[i][2].shape[0], -frames[i][2].shape[1]))
    widths_to_try = range(_GRID_MIN, _GRID_MAX + 1)

    best: tuple[list[tuple[int, int]], int, int] | None = None
    best_area: int | None = None
    for width_limit in widths_to_try:
        positions: list[tuple[int, int] | None] = [None] * len(frames)
        cur_r = 0
        cur_c = 0
        row_h = 0
        used_w = 0
        feasible = True
        for idx in order:
            h, w = frames[idx][2].shape
            if w > width_limit:
                feasible = False
                break
            if cur_c > 0 and cur_c + w > width_limit:
                cur_r += row_h + gap
                cur_c = 0
                row_h = 0
            if cur_r + h > _GRID_MAX:
                feasible = False
                break
            positions[idx] = (cur_r, cur_c)
            cur_c += w + gap
            row_h = max(row_h, h)
            used_w = max(used_w, cur_c - gap)
        if not feasible:
            continue
        total_h = cur_r + row_h
        total_w = used_w
        if total_h > _GRID_MAX or total_w > _GRID_MAX:
            continue
        area = total_h * total_w
        if best is None or area < int(best_area):
            best = ([pos for pos in positions if pos is not None], total_h, total_w)
            best_area = area
    return best


def _render_frames(
    frames: list[tuple[int, int, np.ndarray]],
    positions: list[tuple[int, int]],
    grid_h: int,
    grid_w: int,
) -> np.ndarray:
    grid = np.zeros((grid_h, grid_w), dtype=int)
    for (color, _rooms, wall_arr), (r0, c0) in zip(frames, positions):
        rs, cs = np.where(wall_arr)
        grid[rs + r0, cs + c0] = color
    return grid


def _gen_attempt() -> np.ndarray | None:
    frame_count = random.randint(*_FRAME_COUNT_RANGE)
    room_counts = random.sample(
        list(range(_ROOM_COUNT_RANGE[0], _ROOM_COUNT_RANGE[1] + 1)),
        frame_count,
    )
    frame_colors = random.sample(list(range(1, 10)), frame_count)

    max_h, max_w = _FRAME_MAX_BY_COUNT[frame_count]
    gap_range = _CELL_GAP_BY_COUNT[frame_count]

    frames: list[tuple[int, int, np.ndarray]] = []
    for color, target_rooms in zip(frame_colors, room_counts):
        wall_arr = _build_frame(target_rooms, max_h=max_h, max_w=max_w, gap_range=gap_range)
        if wall_arr is None:
            return None
        frames.append((color, target_rooms, wall_arr))

    packed = _pack_frames(frames, gap=_FRAME_GAP)
    if packed is None:
        return None
    packed_positions, used_h, used_w = packed

    grid_h = random.randint(max(_GRID_MIN, used_h), _GRID_MAX)
    grid_w = random.randint(max(_GRID_MIN, used_w), _GRID_MAX)
    offset_r = random.randint(0, grid_h - used_h)
    offset_c = random.randint(0, grid_w - used_w)
    positions = [(r + offset_r, c + offset_c) for r, c in packed_positions]

    grid = _render_frames(frames, positions, grid_h, grid_w)

    expected_info = sorted((rooms, color) for color, rooms, _walls in frames)
    max_rooms = max(rooms for rooms, _color in expected_info)
    expected = np.zeros((frame_count, max_rooms), dtype=int)
    for r, (rooms, color) in enumerate(expected_info):
        expected[r, :rooms] = color

    try:
        output = main(grid)
    except Exception:
        return None
    return grid if np.array_equal(output, expected) else None


def generate_input() -> np.ndarray:
    for _ in range(_GEN_RETRIES):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError(f"Failed to generate valid input after {_GEN_RETRIES} attempts")
