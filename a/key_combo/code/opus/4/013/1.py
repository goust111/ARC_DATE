from __future__ import annotations

import random

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5
PINK = 6
ORANGE = 7
TEAL = 8
MAROON = 9

MAIN_COLORS = (BLUE, RED, YELLOW)
DISTRACTOR_COLORS = (GREEN, PINK, ORANGE, TEAL, MAROON)
DIAGONAL_DIRS = ((1, 1), (1, -1))
GENERATION_RETRIES = 500


def _row_has_foreground(row: np.ndarray) -> bool:
    return bool(np.any((row != BLACK) & (row != GREY)))


def _dominant_row_color(row: np.ndarray) -> int | None:
    counts: dict[int, int] = {}
    first_pos: dict[int, int] = {}
    for idx, value in enumerate(map(int, row)):
        if value in (BLACK, GREY):
            continue
        counts[value] = counts.get(value, 0) + 1
        first_pos.setdefault(value, idx)
    if not counts:
        return None

    best_color = None
    best_count = -1
    best_pos = len(row)
    for color, count in counts.items():
        pos = first_pos[color]
        if count > best_count or (count == best_count and pos < best_pos):
            best_color = color
            best_count = count
            best_pos = pos
    return best_color


def _invert_row(row: np.ndarray, dominant: int) -> np.ndarray:
    out = np.full_like(row, int(dominant))
    out[row == int(dominant)] = BLACK
    out[row == GREY] = GREY
    return out


def _segment_coords(start_r: int, start_c: int, length: int, dc: int) -> list[tuple[int, int]]:
    return [(int(start_r) + i, int(start_c) + i * int(dc)) for i in range(int(length))]


def _extract_diagonal_segments(grid: np.ndarray) -> list[dict[str, object]]:
    h, w = map(int, grid.shape)
    segments: list[dict[str, object]] = []

    for dr, dc in DIAGONAL_DIRS:
        for r in range(h):
            for c in range(w):
                if int(grid[r, c]) != GREY:
                    continue
                prev_r, prev_c = r - dr, c - dc
                if 0 <= prev_r < h and 0 <= prev_c < w and int(grid[prev_r, prev_c]) == GREY:
                    continue

                coords: list[tuple[int, int]] = []
                rr, cc = r, c
                while 0 <= rr < h and 0 <= cc < w and int(grid[rr, cc]) == GREY:
                    coords.append((int(rr), int(cc)))
                    rr += dr
                    cc += dc

                if len(coords) >= 2:
                    segments.append(
                        {
                            "coords": coords,
                            "length": len(coords),
                            "direction": (dr, dc),
                        }
                    )
    return segments


def _segment_sort_key(segment: dict[str, object]) -> tuple[int, int, int, int]:
    coords = segment["coords"]
    assert isinstance(coords, list) and coords
    start_r, start_c = coords[0]
    dr, dc = segment["direction"]
    direction_rank = 0 if dc == 1 else 1
    return (-int(segment["length"]), int(start_r), int(start_c), direction_rank)


def _apply_selected_rows(grid: np.ndarray, selected_rows: set[int]) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    out = g.copy()
    for r in sorted(selected_rows):
        dominant = _dominant_row_color(g[r])
        if dominant is None:
            continue
        out[r] = _invert_row(g[r], dominant)
    out[out == GREY] = BLACK
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Use the unique longest grey diagonal segment to choose which rows undergo dominant-color inversion.

    Input structure:
    - A black background grid contains sparse grey pixels and several rows with colored foreground pixels.
    - Each non-empty foreground row has one dominant non-black, non-grey color and may include up to two distractor pixels of other colors.
    - The grey pixels form maximal continuous diagonal segments in either diagonal direction.

    Rule:
    - Enumerate every maximal grey diagonal segment in both diagonal directions and choose the longest one.
    - Collect the set of row indices touched by that segment.
    - In each touched row that has foreground pixels, turn dominant-color pixels black and recolor every other non-grey cell in that row to the dominant color.
    - Leave all other rows unchanged, then erase every grey marker pixel to black.

    Key insight:
    - The grey diagonals are only a hidden row-selection mechanism, and only the rows crossed by the single longest segment are inverted.
    """
    g = np.asarray(grid, dtype=int)
    segments = _extract_diagonal_segments(g)
    if not segments:
        out = g.copy()
        out[out == GREY] = BLACK
        return out

    segments.sort(key=_segment_sort_key)
    longest = segments[0]
    selected_rows = {int(r) for r, _c in longest["coords"]}
    return _apply_selected_rows(g, selected_rows)


def _random_segment_start(height: int, width: int, length: int, dc: int) -> tuple[int, int]:
    start_r = random.randint(0, height - length)
    if dc == 1:
        start_c = random.randint(0, width - length)
    else:
        start_c = random.randint(length - 1, width - 1)
    return start_r, start_c


def _can_place_segment(
    coords: list[tuple[int, int]],
    shape: tuple[int, int],
    occupied: set[tuple[int, int]],
    forbidden_rows: set[int],
) -> bool:
    h, w = map(int, shape)
    coord_set = set(coords)
    for r, c in coords:
        if not (0 <= r < h and 0 <= c < w):
            return False
        if r in forbidden_rows or (r, c) in occupied:
            return False
        for dr, dc in DIAGONAL_DIRS:
            nr, nc = r + dr, c + dc
            if (nr, nc) in occupied and (nr, nc) not in coord_set:
                return False
            nr, nc = r - dr, c - dc
            if (nr, nc) in occupied and (nr, nc) not in coord_set:
                return False
    return True


def _free_intervals(width: int, blocked_cols: set[int]) -> list[tuple[int, int]]:
    intervals: list[tuple[int, int]] = []
    start = None
    for c in range(width):
        if c in blocked_cols:
            if start is not None:
                intervals.append((start, c - 1))
                start = None
            continue
        if start is None:
            start = c
    if start is not None:
        intervals.append((start, width - 1))
    return intervals


def _paint_foreground_row(grid: np.ndarray, row: int, blocked_cols: set[int]) -> bool:
    width = int(grid.shape[1])
    intervals = _free_intervals(width, blocked_cols)
    if not intervals:
        return False

    max_run = max(end - start + 1 for start, end in intervals)
    if max_run < 3:
        return False

    main_len = random.randint(3, min(6, max_run))
    candidates = [(start, end) for start, end in intervals if end - start + 1 >= main_len]
    if not candidates:
        return False

    start, end = random.choice(candidates)
    run_start = random.randint(start, end - main_len + 1)
    main_cols = list(range(run_start, run_start + main_len))
    main_color = random.choice(MAIN_COLORS)
    grid[row, main_cols] = main_color

    remaining = [c for c in range(width) if c not in blocked_cols and c not in main_cols]
    max_distractors = min(2, len(remaining))
    if max_distractors == 0:
        distractor_count = 0
    else:
        distractor_count = random.choices([0, 1, 2], weights=[3, 3, 2], k=1)[0]
        distractor_count = min(distractor_count, max_distractors)

    if distractor_count:
        positions = random.sample(remaining, distractor_count)
        colors = random.sample(DISTRACTOR_COLORS, distractor_count)
        for c, color in zip(positions, colors):
            grid[row, c] = int(color)
    return True


def _gen_attempt() -> np.ndarray | None:
    height = random.randint(12, 18)
    width = random.randint(12, 18)
    grid = np.full((height, width), BLACK, dtype=int)

    n_segments = random.randint(2, 4)
    longest_len = random.randint(4, min(7, height, width))
    longest_dc = random.choice((1, -1))
    longest_start = _random_segment_start(height, width, longest_len, longest_dc)
    longest_coords = _segment_coords(longest_start[0], longest_start[1], longest_len, longest_dc)
    longest_rows = [r for r, _c in longest_coords]
    longest_cols_by_row = {r: c for r, c in longest_coords}

    selected_fg_count = random.randint(2, min(4, longest_len))
    selected_foreground_rows = set(random.sample(longest_rows, selected_fg_count))

    extra_candidates = [r for r in range(height) if r not in set(longest_rows)]
    random.shuffle(extra_candidates)
    extra_count = random.randint(1, min(3, len(extra_candidates)))
    extra_rows = set(extra_candidates[:extra_count])

    for row in sorted(selected_foreground_rows | extra_rows):
        blocked = {longest_cols_by_row[row]} if row in longest_cols_by_row else set()
        if not _paint_foreground_row(grid, row, blocked):
            return None

    for r, c in longest_coords:
        if int(grid[r, c]) != BLACK:
            return None
        grid[r, c] = GREY

    occupied = set(longest_coords)
    forbidden_rows = set(longest_rows) | extra_rows
    shorter_lengths = [random.randint(2, longest_len - 1) for _ in range(n_segments - 1)]

    for length in shorter_lengths:
        placed = False
        for _ in range(120):
            dc = random.choice((1, -1))
            start_r, start_c = _random_segment_start(height, width, length, dc)
            coords = _segment_coords(start_r, start_c, length, dc)
            if not _can_place_segment(coords, grid.shape, occupied, forbidden_rows):
                continue
            for r, c in coords:
                grid[r, c] = GREY
            occupied.update(coords)
            placed = True
            break
        if not placed:
            return None

    segments = _extract_diagonal_segments(grid)
    if len(segments) != n_segments:
        return None

    segments.sort(key=_segment_sort_key)
    if int(segments[0]["length"]) != longest_len:
        return None
    if sum(int(seg["length"]) == longest_len for seg in segments) != 1:
        return None
    if set(segments[0]["coords"]) != set(longest_coords):
        return None

    flipped_rows = set(longest_rows)
    if sum(_row_has_foreground(grid[r]) for r in flipped_rows) < 2:
        return None

    expected = _apply_selected_rows(grid, flipped_rows)
    actual = main(grid)
    if not np.array_equal(actual, expected):
        return None

    for r in extra_rows:
        if not np.array_equal(actual[r], grid[r]):
            return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
