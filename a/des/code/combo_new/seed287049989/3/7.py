import random

import numpy as np


def _background_color(grid: np.ndarray) -> int:
    values, counts = np.unique(np.asarray(grid, dtype=int), return_counts=True)
    return int(values[np.argmax(counts)])


def _extract_runs(values: np.ndarray, bg: int) -> list[tuple[int, int, int]]:
    runs: list[tuple[int, int, int]] = []
    i = 0
    n = int(values.size)
    while i < n:
        color = int(values[i])
        if color == bg:
            i += 1
            continue
        start = i
        while i + 1 < n and int(values[i + 1]) == color:
            i += 1
        runs.append((start, i, color))
        i += 1
    return runs


def _segment_priority(start: int, end: int, axis_len: int) -> int:
    return min(int(start), int(axis_len - 1 - end))


def _render_from_segments(
    grid: np.ndarray,
    vertical_segments: list[tuple[int, int, int]],
    horizontal_segments: list[tuple[int, int, int]],
) -> np.ndarray:
    out = np.asarray(grid, dtype=int).copy()
    h, w = out.shape

    vertical_by_col: dict[int, tuple[int, int]] = {}
    for start, end, color in vertical_segments:
        priority = _segment_priority(start, end, w)
        for col in range(start, end + 1):
            vertical_by_col[int(col)] = (int(color), int(priority))

    horizontal_by_row: dict[int, tuple[int, int]] = {}
    for start, end, color in horizontal_segments:
        priority = _segment_priority(start, end, h)
        for row in range(start, end + 1):
            horizontal_by_row[int(row)] = (int(color), int(priority))

    for row in range(1, h - 1):
        for col in range(1, w - 1):
            vertical = vertical_by_col.get(col)
            horizontal = horizontal_by_row.get(row)
            if vertical is None and horizontal is None:
                continue
            if vertical is None:
                out[row, col] = horizontal[0]
                continue
            if horizontal is None:
                out[row, col] = vertical[0]
                continue
            if vertical[1] <= horizontal[1]:
                out[row, col] = vertical[0]
            else:
                out[row, col] = horizontal[0]

    return out


def _parse_segments(grid: np.ndarray) -> tuple[int, list[tuple[int, int, int]], list[tuple[int, int, int]]]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    h, w = g.shape
    assert h >= 3 and w >= 3, "grid must have an interior"

    bg = _background_color(g)
    top_runs = _extract_runs(g[0, :], bg)
    bottom_runs = _extract_runs(g[h - 1, :], bg)
    left_runs = _extract_runs(g[:, 0], bg)
    right_runs = _extract_runs(g[:, w - 1], bg)

    assert top_runs == bottom_runs, "top and bottom edge segments must match"
    assert left_runs == right_runs, "left and right edge segments must match"

    return bg, top_runs, left_runs


def main(grid: np.ndarray) -> np.ndarray:
    """Project edge-encoded horizontal and vertical bands through the interior, resolving crossings by corner proximity.

    Input structure:
    - The background is the dominant color, and the only non-background pixels lie on the outer border.
    - Matching colored segments on the top and bottom edges encode vertical line ranges and colors.
    - Matching colored segments on the left and right edges encode horizontal line ranges and colors.
    - Each segment is a contiguous same-color run, and different runs are visually distinct.

    Rule:
    - For each top/bottom segment, fill every covered column from row 1 through row H-2 with that segment color.
    - For each left/right segment, fill every covered row from column 1 through column W-2 with that segment color.
    - At an intersection, compare the two crossing segments by the smaller endpoint-to-corner distance along their edge.
    - The line whose segment sits closer to a corner wins the crossing; if the distances tie, the vertical line wins.

    Key insight:
    - The border bands encode both the line placement and a hidden crossing priority: edge segments nearer to a corner dominate farther ones.
    """
    g = np.asarray(grid, dtype=int)
    _bg, vertical_segments, horizontal_segments = _parse_segments(g)
    return _render_from_segments(g, vertical_segments, horizontal_segments)


_GEN_ATTEMPTS = 400
_GEN_SIZE_MIN = 12
_GEN_SIZE_MAX = 20
_GEN_EXTRA_SEGMENT_MAX = 2
_GEN_GAP = 1
_GEN_BRANCHES = ("vertical", "horizontal", "tie")


def _interval_conflicts(start: int, end: int, existing: list[tuple[int, int, int]]) -> bool:
    for other_start, other_end, _color in existing:
        if not (end < other_start - _GEN_GAP or start > other_end + _GEN_GAP):
            return True
    return False


def _place_interval_with_priority(
    axis_len: int,
    priority: int,
    existing: list[tuple[int, int, int]],
    color: int,
) -> tuple[int, int, int] | None:
    max_len = min(5, axis_len - 2)
    if max_len < 1:
        return None

    choices: list[tuple[int, int, int]] = []

    start = priority
    max_end = min(axis_len - 2, axis_len - 1 - priority)
    if start <= max_end:
        for end in range(start, min(max_end, start + max_len - 1) + 1):
            if not _interval_conflicts(start, end, existing):
                choices.append((start, end, color))

    end = axis_len - 1 - priority
    min_start = max(priority, end - max_len + 1, 1)
    if min_start <= end:
        for start in range(min_start, end + 1):
            if not _interval_conflicts(start, end, existing):
                choices.append((start, end, color))

    if not choices:
        return None
    return random.choice(choices)


def _place_random_interval(
    axis_len: int,
    existing: list[tuple[int, int, int]],
    color: int,
) -> tuple[int, int, int] | None:
    max_priority = (axis_len - 1) // 2
    priorities = list(range(1, max_priority + 1))
    random.shuffle(priorities)
    for priority in priorities:
        interval = _place_interval_with_priority(axis_len, priority, existing, color)
        if interval is not None:
            return interval
    return None


def _dominance_ok(grid: np.ndarray, bg: int) -> bool:
    bg_count = int(np.sum(grid == bg))
    other_counts = [int(np.sum(grid == color)) for color in np.unique(grid) if int(color) != bg]
    return not other_counts or bg_count > max(other_counts)


def _independent_expected(
    grid: np.ndarray,
    vertical_segments: list[tuple[int, int, int]],
    horizontal_segments: list[tuple[int, int, int]],
) -> np.ndarray:
    return _render_from_segments(grid, vertical_segments, horizontal_segments)


def _branch_present(
    vertical_segments: list[tuple[int, int, int]],
    horizontal_segments: list[tuple[int, int, int]],
    h: int,
    w: int,
    branch: str,
) -> bool:
    vertical_priorities = [_segment_priority(start, end, w) for start, end, _ in vertical_segments]
    horizontal_priorities = [_segment_priority(start, end, h) for start, end, _ in horizontal_segments]
    if branch == "vertical":
        return any(vp < hp for vp in vertical_priorities for hp in horizontal_priorities)
    if branch == "horizontal":
        return any(vp > hp for vp in vertical_priorities for hp in horizontal_priorities)
    return any(vp == hp for vp in vertical_priorities for hp in horizontal_priorities)


def _build_input_grid(
    h: int,
    w: int,
    bg: int,
    vertical_segments: list[tuple[int, int, int]],
    horizontal_segments: list[tuple[int, int, int]],
) -> np.ndarray:
    grid = np.full((h, w), bg, dtype=int)
    for start, end, color in vertical_segments:
        grid[0, start : end + 1] = color
        grid[h - 1, start : end + 1] = color
    for start, end, color in horizontal_segments:
        grid[start : end + 1, 0] = color
        grid[start : end + 1, w - 1] = color
    return grid


def _candidate_pairs(max_v_priority: int, max_h_priority: int, branch: str) -> list[tuple[int, int]]:
    pairs = []
    for vp in range(1, max_v_priority + 1):
        for hp in range(1, max_h_priority + 1):
            if branch == "vertical" and vp < hp:
                pairs.append((vp, hp))
            elif branch == "horizontal" and vp > hp:
                pairs.append((vp, hp))
            elif branch == "tie" and vp == hp:
                pairs.append((vp, hp))
    return pairs


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(_GEN_SIZE_MIN, _GEN_SIZE_MAX)
    w = random.randint(_GEN_SIZE_MIN, _GEN_SIZE_MAX)
    bg = random.randint(0, 9)
    branch = random.choice(_GEN_BRANCHES)

    max_v_priority = (w - 1) // 2
    max_h_priority = (h - 1) // 2
    priority_pairs = _candidate_pairs(max_v_priority, max_h_priority, branch)
    if not priority_pairs:
        return None

    extra_v = random.randint(0, _GEN_EXTRA_SEGMENT_MAX)
    extra_h = random.randint(0, _GEN_EXTRA_SEGMENT_MAX)
    total_colors = 2 + extra_v + extra_h
    palette = [color for color in range(10) if color != bg]
    if total_colors > len(palette):
        return None
    colors = random.sample(palette, total_colors)

    vertical_segments: list[tuple[int, int, int]] = []
    horizontal_segments: list[tuple[int, int, int]] = []

    focus_v_priority, focus_h_priority = random.choice(priority_pairs)
    focus_vertical = _place_interval_with_priority(w, focus_v_priority, vertical_segments, colors[0])
    focus_horizontal = _place_interval_with_priority(h, focus_h_priority, horizontal_segments, colors[1])
    if focus_vertical is None or focus_horizontal is None:
        return None
    vertical_segments.append(focus_vertical)
    horizontal_segments.append(focus_horizontal)

    color_index = 2
    for _ in range(extra_v):
        interval = _place_random_interval(w, vertical_segments, colors[color_index])
        if interval is None:
            break
        vertical_segments.append(interval)
        color_index += 1
    for _ in range(extra_h):
        interval = _place_random_interval(h, horizontal_segments, colors[color_index])
        if interval is None:
            break
        horizontal_segments.append(interval)
        color_index += 1

    vertical_segments.sort()
    horizontal_segments.sort()
    grid = _build_input_grid(h, w, bg, vertical_segments, horizontal_segments)

    if not _dominance_ok(grid, bg):
        return None
    if not _branch_present(vertical_segments, horizontal_segments, h, w, branch):
        return None

    expected = _independent_expected(grid, vertical_segments, horizontal_segments)
    try:
        actual = main(grid)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(_GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid edge-band piercing-lines puzzle")
