# Combo variant of 89565ca0 (index 30)
# combo_spec: D4/0+D12/0+D15/2
# Name: 无噪声+无损墙壁+仅水平分隔

from __future__ import annotations

import random

import numpy as np


# Design analysis for the fused variant
# -------------------------------------
# a) Concrete behavior trace:
#    - Original task: each color could be either a real frame or the scattered noise color.
#      Solver had to rebuild partially damaged vertical/horizontal walls, count enclosed regions,
#      then use overlap cover-order to break equal-count ties.
#    - Fused task: there is no scattered noise at all, walls are fully intact, and each frame
#      contains only horizontal separators. A small example is a color-2 rectangle with top/bottom
#      walls and one internal full-width horizontal line; originally this would have been one case
#      among many grid-like patterns, but here it deterministically means 2 stacked rooms.
#    - Difference: room counting no longer needs wall reconstruction or overlap reasoning. The
#      decisive visible cue is simply the number of horizontal bands inside each intact rectangle.
#
# b) Complexity calibration:
#    - Removing noise and vertical separators greatly simplifies the puzzle, so the generator keeps
#      3-5 frames with different room counts and varied widths/heights to preserve a nontrivial sort.
#    - Equal room counts would become visually ambiguous because the original overlap-based tiebreak
#      is gone, so the generator avoids ties. The solver still has a visible reading-order fallback
#      for robustness on off-distribution inputs.
#
# c) Implicit assumption audit against the original main():
#    - "One non-background color is noise" no longer holds, so every non-black color must be treated
#      as a real frame.
#    - "Walls must be reconstructed from 45% evidence" no longer holds; using that logic here would
#      be unnecessary and could even hallucinate structure that is not part of the new rule.
#    - "Ties are resolved by overlap cover order" no longer holds because placement forbids overlap.
#      We therefore sort by room count and use top-left reading order only as a deterministic fallback.
#
# d) Placement legality:
#    - Generator places frame bboxes with a one-cell black moat, so frames do not overlap, do not
#      touch, and always remain inside the grid.
#
# e) Visual observability:
#    - A human can count stacked horizontal compartments directly. The absence of noise and the lack
#      of vertical inner walls are both obvious from 2-4 examples.
#
# f) Visual cleanliness:
#    - No overlapping shapes, no damaged walls, and only one object type per color keeps the input
#      uncluttered. Output uses black padding, which is now the only neutral color left.
#
# g) New solver assumptions guaranteed by the generator:
#    - Each non-black color appears in exactly one frame.
#    - Each frame's bbox is a single intact rectangle with only horizontal full-width separators.
#    - Room counts are unique in generated samples, though main() remains deterministic if they are not.


_FRAME_COUNT_RANGE = (3, 5)
_ROOM_COUNT_VALUES = (1, 2, 3, 4, 5, 6)
_GRID_SIZE_BY_FRAMES = {
    3: (22, 28, 22, 28),
    # Slightly larger canvases keep 4/5-frame cases as reachable as 3-frame cases,
    # improving generator branch coverage without changing the visible rule.
    4: (27, 33, 27, 33),
    5: (31, 36, 31, 36),
}
_ROOM_BAND_HEIGHT_RANGE = (1, 3)
_FRAME_INNER_WIDTH_RANGE = (4, 10)
_PLACEMENT_MARGIN = 1
_PLACEMENT_RETRIES = 400
_GEN_RETRIES = 300


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _crop(grid: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = box
    return grid[r0 : r1 + 1, c0 : c1 + 1].copy()


def _colors_present(grid: np.ndarray, exclude: set[int] | None = None) -> list[int]:
    colors = set(int(v) for v in np.unique(grid))
    if exclude is not None:
        colors -= set(int(v) for v in exclude)
    return sorted(colors)


def _count_enclosed_regions(walls: np.ndarray) -> int:
    h, w = walls.shape
    open_cells = ~walls
    seen = np.zeros((h, w), dtype=bool)
    count = 0

    for r in range(h):
        for c in range(w):
            if seen[r, c] or not open_cells[r, c]:
                continue

            stack = [(r, c)]
            seen[r, c] = True
            touches_edge = False
            while stack:
                cr, cc = stack.pop()
                if cr == 0 or cr == h - 1 or cc == 0 or cc == w - 1:
                    touches_edge = True
                for nr, nc in ((cr - 1, cc), (cr + 1, cc), (cr, cc - 1), (cr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and open_cells[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))

            if not touches_edge:
                count += 1

    return count


def _horizontal_wall_rows(mask: np.ndarray) -> list[int]:
    return [int(r) for r in range(mask.shape[0]) if bool(mask[r, :].all())]


def _room_count_for_color(grid: np.ndarray, color: int) -> tuple[int, tuple[int, int, int, int]]:
    box = _bbox(grid == int(color))
    assert box is not None
    mask = _crop(grid == int(color), box)
    h, w = mask.shape
    assert h >= 3 and w >= 3

    full_rows = _horizontal_wall_rows(mask)
    assert full_rows[0] == 0 and full_rows[-1] == h - 1
    assert len(full_rows) >= 2
    assert mask[:, 0].all() and mask[:, w - 1].all()
    assert not mask[:, 1 : w - 1].all(axis=0).any()

    rooms = len(full_rows) - 1
    assert rooms == _count_enclosed_regions(mask)
    return int(rooms), box


def main(grid: np.ndarray) -> np.ndarray:
    """
    Turn each intact horizontally partitioned frame into a left-aligned color bar whose length equals its room count.

    Input structure:
    - Black background with several non-overlapping colored rectangular frames.
    - Every frame has complete outer walls and only full-width horizontal internal separators.
    - There is no noise color and no wall damage, so each non-black color corresponds to exactly one real frame.

    Rule:
    - For each non-black color, crop its bounding box and count how many enclosed horizontal bands it contains.
    - This count is the number of rooms, equivalently the number of full-width wall rows minus one.
    - Sort frames by increasing room count; if a tie ever appears, break it by top-to-bottom then left-to-right position.
    - Output one row per frame, filling the first `rooms` cells with the frame color and padding the rest with black.

    Key insight:
    - Once noise, damage, and vertical inner walls are removed, the puzzle reduces to recognizing stacked compartments:
      each intact horizontal separator adds one more enclosed band, so the output is just the sorted room counts rendered as bars.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    bg = 0

    frames: list[tuple[int, int, tuple[int, int, int, int]]] = []
    for color in _colors_present(g, exclude={bg}):
        rooms, box = _room_count_for_color(g, int(color))
        frames.append((int(color), int(rooms), box))

    assert frames
    frames.sort(key=lambda item: (item[1], item[2][0], item[2][1]))
    width = max(rooms for _color, rooms, _box in frames)

    out = np.zeros((len(frames), width), dtype=int)
    for r, (color, rooms, _box) in enumerate(frames):
        out[r, :rooms] = int(color)
    return out


def _make_horizontal_frame(room_count: int) -> np.ndarray:
    band_heights = [random.randint(*_ROOM_BAND_HEIGHT_RANGE) for _ in range(room_count)]
    inner_width = random.randint(*_FRAME_INNER_WIDTH_RANGE)

    h = sum(band_heights) + room_count + 1
    w = inner_width + 2
    walls = np.zeros((h, w), dtype=bool)

    walls[:, 0] = True
    walls[:, w - 1] = True

    boundary_rows = [0]
    cursor = 0
    for band_h in band_heights:
        cursor += band_h + 1
        boundary_rows.append(cursor)

    for r in boundary_rows:
        walls[r, :] = True

    assert len(boundary_rows) == room_count + 1
    assert _count_enclosed_regions(walls) == room_count
    return walls


def _place_frames(frame_masks: list[np.ndarray], grid_h: int, grid_w: int) -> list[tuple[int, int]] | None:
    positions: list[tuple[int, int] | None] = [None] * len(frame_masks)
    occupied = np.zeros((grid_h, grid_w), dtype=bool)

    order = sorted(range(len(frame_masks)), key=lambda i: frame_masks[i].shape[0] * frame_masks[i].shape[1], reverse=True)
    for idx in order:
        h, w = frame_masks[idx].shape
        if h > grid_h or w > grid_w:
            return None

        placed = False
        for _ in range(_PLACEMENT_RETRIES):
            r0 = random.randint(0, grid_h - h)
            c0 = random.randint(0, grid_w - w)
            rr0 = max(0, r0 - _PLACEMENT_MARGIN)
            cc0 = max(0, c0 - _PLACEMENT_MARGIN)
            rr1 = min(grid_h, r0 + h + _PLACEMENT_MARGIN)
            cc1 = min(grid_w, c0 + w + _PLACEMENT_MARGIN)
            if occupied[rr0:rr1, cc0:cc1].any():
                continue
            occupied[rr0:rr1, cc0:cc1] = True
            positions[idx] = (r0, c0)
            placed = True
            break

        if not placed:
            return None

    return [(int(r), int(c)) for r, c in positions if r is not None and c is not None]


def _render(frames: list[tuple[int, int, np.ndarray]], positions: list[tuple[int, int]], grid_h: int, grid_w: int) -> np.ndarray:
    grid = np.zeros((grid_h, grid_w), dtype=int)
    for (color, _rooms, walls), (r0, c0) in zip(frames, positions):
        rs, cs = np.where(walls)
        grid[rs + r0, cs + c0] = int(color)
    return grid


def _expected_output(frames: list[tuple[int, int, np.ndarray]], positions: list[tuple[int, int]]) -> np.ndarray:
    info = []
    for (color, rooms, _walls), (r0, c0) in zip(frames, positions):
        info.append((int(color), int(rooms), int(r0), int(c0)))
    info.sort(key=lambda item: (item[1], item[2], item[3]))

    width = max(rooms for _color, rooms, _r0, _c0 in info)
    out = np.zeros((len(info), width), dtype=int)
    for row, (color, rooms, _r0, _c0) in enumerate(info):
        out[row, :rooms] = int(color)
    return out


def _variant_metrics(grid: np.ndarray) -> tuple[int, int]:
    colors = _colors_present(grid, exclude={0})
    n_frames = len(colors)
    internal_vertical_columns = 0
    for color in colors:
        box = _bbox(grid == color)
        assert box is not None
        mask = _crop(grid == color, box)
        if mask.shape[1] > 2:
            internal_vertical_columns += int(mask[:, 1:-1].all(axis=0).sum())
    return n_frames, internal_vertical_columns


def _gen_attempt() -> np.ndarray | None:
    frame_count = random.randint(*_FRAME_COUNT_RANGE)
    room_counts = random.sample(_ROOM_COUNT_VALUES, frame_count)
    colors = random.sample(list(range(1, 10)), frame_count)

    frames: list[tuple[int, int, np.ndarray]] = []
    for color, rooms in zip(colors, room_counts):
        walls = _make_horizontal_frame(int(rooms))
        frames.append((int(color), int(rooms), walls))

    h_min, h_max, w_min, w_max = _GRID_SIZE_BY_FRAMES[frame_count]
    grid_h = random.randint(h_min, h_max)
    grid_w = random.randint(w_min, w_max)
    positions = _place_frames([walls for _color, _rooms, walls in frames], grid_h, grid_w)
    if positions is None or len(positions) != len(frames):
        return None

    grid = _render(frames, positions, grid_h, grid_w)
    expected = _expected_output(frames, positions)

    try:
        output = main(grid)
    except Exception:
        return None

    if not np.array_equal(output, expected):
        return None

    n_frames, internal_vertical_columns = _variant_metrics(grid)
    if n_frames != frame_count or internal_vertical_columns != 0:
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(_GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"Failed to generate valid input after {_GEN_RETRIES} attempts")
