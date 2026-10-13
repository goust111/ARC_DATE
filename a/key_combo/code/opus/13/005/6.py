from __future__ import annotations

import random

import numpy as np


BG = 0
FRAME_COLORS = (1, 2, 3, 4)
GREY = 5
MAGENTA = 6
ORANGE = 7

GEN_RETRIES = 400
PATTERN_RETRIES = 200
LAYOUT_RETRIES = 120


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _colors_present(grid: np.ndarray, exclude: set[int] | None = None) -> list[int]:
    colors = sorted(int(v) for v in np.unique(grid))
    if exclude:
        colors = [c for c in colors if c not in exclude]
    return colors


def _full_frame(grid: np.ndarray, color: int, r0: int, c0: int, r1: int, c1: int) -> bool:
    return (
        np.all(grid[r0, c0 : c1 + 1] == color)
        and np.all(grid[r1, c0 : c1 + 1] == color)
        and np.all(grid[r0 : r1 + 1, c0] == color)
        and np.all(grid[r0 : r1 + 1, c1] == color)
    )


def _rectangular_frames(grid: np.ndarray) -> list[tuple[int, int, int, int, int]]:
    bg = _get_bg(grid)
    frames: list[tuple[int, int, int, int, int]] = []
    for color in _colors_present(grid, exclude={bg, GREY, MAGENTA, ORANGE}):
        rows = np.flatnonzero(np.any(grid == color, axis=1))
        cols = np.flatnonzero(np.any(grid == color, axis=0))
        for i, r0 in enumerate(rows):
            for r1 in rows[i + 1 :]:
                if r1 - r0 < 2:
                    continue
                interior_h = int(r1 - r0 - 1)
                for j, c0 in enumerate(cols):
                    for c1 in cols[j + 1 :]:
                        if c1 - c0 < 2:
                            continue
                        interior_w = int(c1 - c0 - 1)
                        if interior_h != interior_w:
                            continue
                        if _full_frame(grid, int(color), int(r0), int(c0), int(r1), int(c1)):
                            frames.append((int(color), int(r0), int(c0), int(r1), int(c1)))
    return frames


def _segment_length(grid: np.ndarray) -> int:
    coords = np.argwhere(grid == ORANGE)
    assert coords.size > 0, "expected an orange anti-diagonal segment"
    diag_sums = {int(r + c) for r, c in coords}
    assert len(diag_sums) == 1, "orange cells must lie on one anti-diagonal"
    cols = sorted(int(c) for _r, c in coords)
    assert len(cols) == len(set(cols)), "segment must have one cell per column"
    assert cols == list(range(cols[0], cols[-1] + 1)), "segment columns must be contiguous"
    return len(cols)


def _quadrant_rotations(pattern: np.ndarray) -> np.ndarray:
    return np.block(
        [
            [np.rot90(pattern, 2), np.rot90(pattern, -1)],
            [np.rot90(pattern, 1), pattern],
        ]
    )


def main(grid: np.ndarray) -> np.ndarray:
    """Use the advanced anti-diagonal length as a frame index, then crop and rotate-tile that frame's contents.

    Input structure:
    - A black background contains several complete one-pixel-thick monochrome rectangular frames.
    - Each frame has a square interior filled with a grey/magenta pattern, and there is one orange segment on a single anti-diagonal.

    Rule:
    - Measure the orange anti-diagonal segment length L and conceptually advance it to the next length L+1 on the same anti-diagonal.
    - Find all complete rectangular frames, sort them by interior area from small to large, and select frame number L+1.
    - Crop the selected frame's strict interior pattern P and build a 2x2 output of rotated copies:
      upper-left rot180(P), upper-right rot90 clockwise, lower-left rot90 counterclockwise, lower-right unchanged.

    Key insight:
    - The orange segment length is a hidden pointer: after the segment would grow by one, that new length becomes the index into the area-sorted frame list.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    length = _segment_length(g)
    target_index = length + 1

    frames = _rectangular_frames(g)
    assert frames, "expected at least one complete rectangular frame"
    frames_sorted = sorted(frames, key=lambda fr: (fr[3] - fr[1] - 1) * (fr[4] - fr[2] - 1))
    assert 1 <= target_index <= len(frames_sorted), "segment points outside the frame list"

    _color, r0, c0, r1, c1 = frames_sorted[target_index - 1]
    pattern = g[r0 + 1 : r1, c0 + 1 : c1].copy()
    return _quadrant_rotations(pattern)


def _pattern_ok(pattern: np.ndarray) -> bool:
    if pattern.ndim != 2 or pattern.shape[0] != pattern.shape[1]:
        return False
    area = int(pattern.size)
    non_bg = int(np.count_nonzero(pattern != BG))
    if non_bg < 3 or non_bg >= area:
        return False
    colors = {int(v) for v in np.unique(pattern)} - {BG}
    if colors != {GREY, MAGENTA}:
        return False
    rotations = [np.rot90(pattern, k) for k in range(4)]
    keys = {rot.tobytes() for rot in rotations}
    return len(keys) == 4


def _random_pattern(size: int) -> np.ndarray | None:
    area = size * size
    min_count = max(3, area // 3)
    max_count = max(min_count, area - 2)
    for _ in range(PATTERN_RETRIES):
        pattern = np.zeros((size, size), dtype=int)
        count = random.randint(min_count, max_count)
        cells = random.sample(range(area), count)
        grey_count = random.randint(1, count - 1)
        grey_cells = set(random.sample(cells, grey_count))
        for idx in cells:
            r, c = divmod(idx, size)
            pattern[r, c] = GREY if idx in grey_cells else MAGENTA
        if _pattern_ok(pattern):
            return pattern
    return None


def _paint_frame(grid: np.ndarray, r0: int, c0: int, color: int, pattern: np.ndarray) -> None:
    size = int(pattern.shape[0])
    r1 = r0 + size + 1
    c1 = c0 + size + 1
    grid[r0 : r1 + 1, c0] = color
    grid[r0 : r1 + 1, c1] = color
    grid[r0, c0 : c1 + 1] = color
    grid[r1, c0 : c1 + 1] = color
    grid[r0 + 1 : r1, c0 + 1 : c1] = pattern


def _place_frames(
    grid_size: int, sizes: list[int]
) -> list[tuple[int, int, int, int]] | None:
    occupied = np.zeros((grid_size, grid_size), dtype=bool)
    placements: dict[int, tuple[int, int, int, int]] = {}
    for size in sorted(sizes, reverse=True):
        outer = size + 2
        candidates = [
            (r0, c0)
            for r0 in range(grid_size - outer + 1)
            for c0 in range(grid_size - outer + 1)
        ]
        random.shuffle(candidates)
        placed = False
        for r0, c0 in candidates:
            r1 = r0 + outer - 1
            c1 = c0 + outer - 1
            rr0 = max(0, r0 - 1)
            cc0 = max(0, c0 - 1)
            rr1 = min(grid_size - 1, r1 + 1)
            cc1 = min(grid_size - 1, c1 + 1)
            if occupied[rr0 : rr1 + 1, cc0 : cc1 + 1].any():
                continue
            occupied[rr0 : rr1 + 1, cc0 : cc1 + 1] = True
            placements[size] = (r0, c0, r1, c1)
            placed = True
            break
        if not placed:
            return None
    return [placements[size] for size in sizes]


def _segment_cells(diag_sum: int, c0: int, length: int) -> list[tuple[int, int]]:
    return [(diag_sum - c, c) for c in range(c0, c0 + length)]


def _valid_segment_placements(
    grid_size: int, length: int, blocked: np.ndarray
) -> list[list[tuple[int, int]]]:
    choices: list[list[tuple[int, int]]] = []
    corner_band = 6
    for diag_sum in range(2 * grid_size - 1):
        for c0 in range(grid_size):
            cur = _segment_cells(diag_sum, c0, length)
            nxt = _segment_cells(diag_sum, c0 + length, length + 1)
            all_cells = cur + nxt
            if not all(0 <= r < grid_size and 0 <= c < grid_size for r, c in all_cells):
                continue
            if not all(c <= corner_band and r >= grid_size - 1 - corner_band for r, c in cur):
                continue
            if any(blocked[r, c] for r, c in all_cells):
                continue
            choices.append(cur)
    return choices


def _attempt_generate() -> np.ndarray | None:
    frame_count = random.choice((3, 4))
    size_pool = list(range(3, 8 if frame_count == 3 else 7))
    sizes = sorted(random.sample(size_pool, frame_count))
    max_pattern_size = max(sizes)
    if 2 * max_pattern_size > 30:
        return None

    min_grid = max(20, max_pattern_size + 10)
    grid_size = random.randint(min_grid, 30)

    placements = None
    for _ in range(LAYOUT_RETRIES):
        placements = _place_frames(grid_size, sizes)
        if placements is not None:
            break
    if placements is None:
        return None

    patterns: list[np.ndarray] = []
    for size in sizes:
        pattern = _random_pattern(size)
        if pattern is None:
            return None
        patterns.append(pattern)

    grid = np.zeros((grid_size, grid_size), dtype=int)
    frame_colors = random.sample(FRAME_COLORS, frame_count)
    for (r0, c0, _r1, _c1), color, pattern in zip(placements, frame_colors, patterns):
        _paint_frame(grid, r0, c0, color, pattern)

    blocked = grid != BG
    max_length = min(4, frame_count - 1)
    lengths = list(range(1, max_length + 1))
    random.shuffle(lengths)
    segment = None
    for length in lengths:
        candidates = _valid_segment_placements(grid_size, length, blocked)
        if not candidates:
            continue
        segment = random.choice(candidates)
        break
    if segment is None:
        return None

    for r, c in segment:
        grid[r, c] = ORANGE

    frames = _rectangular_frames(grid)
    if len(frames) != frame_count:
        return None

    areas = [(r1 - r0 - 1) * (c1 - c0 - 1) for _color, r0, c0, r1, c1 in frames]
    if sorted(areas) != [size * size for size in sizes]:
        return None

    try:
        out = main(grid)
    except Exception:
        return None
    if out.ndim != 2 or max(out.shape) > 30:
        return None
    expected = _quadrant_rotations(patterns[len(segment)])
    if not np.array_equal(out, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _attempt_generate()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
