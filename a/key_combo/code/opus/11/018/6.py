from __future__ import annotations

import random

import numpy as np


BG = 0
RED = 2
GREY = 5

UP_COLORS = (3, 1, 4)
DOWN_COLORS = (6, 7, 8)

GRID_MIN = 15
GRID_MAX = 22
BAR_LENGTH_CHOICES = (4, 5, 6, 6)
GEN_ATTEMPTS = 400


def _contiguous_runs(values: np.ndarray) -> list[tuple[int, int]]:
    idx = np.flatnonzero(values)
    if idx.size == 0:
        return []
    runs: list[tuple[int, int]] = []
    start = int(idx[0])
    prev = int(idx[0])
    for value in idx[1:]:
        value = int(value)
        if value != prev + 1:
            runs.append((start, prev))
            start = value
        prev = value
    runs.append((start, prev))
    return runs


def _find_red_bar(g: np.ndarray) -> tuple[int, int]:
    red_rows = np.unique(np.argwhere(g == RED)[:, 0])
    assert len(red_rows) == 1, "expected exactly one red horizontal segment"
    bar_row = int(red_rows[0])
    row = g[bar_row]
    red_runs = _contiguous_runs(row == RED)
    assert len(red_runs) == 1, "red pixels must form one contiguous run"
    c0, c1 = red_runs[0]
    assert c0 == 0, "red segment must be left-anchored"
    bar_len = c1 - c0 + 1
    assert np.count_nonzero(g == RED) == bar_len, "red foreground must be only the base segment"
    return bar_row, int(bar_len)


def _segment_index(distances: list[int], d: int) -> int:
    idx = 0
    for obstacle_d in distances:
        if d > obstacle_d:
            idx += 1
    return idx


def _paint_side(
    out: np.ndarray,
    *,
    bar_row: int,
    bar_len: int,
    obstacle_distances: list[int],
    direction: int,
    palette: tuple[int, ...],
) -> None:
    height, width = out.shape
    for d in range(1, height + width):
        r = bar_row + direction * d
        if not (0 <= r < height):
            break

        if direction < 0:
            fill_width = bar_len + d
        else:
            fill_width = bar_len - d
            if fill_width <= 0:
                break

        if d in obstacle_distances:
            continue

        color = palette[min(_segment_index(obstacle_distances, d), len(palette) - 1)]
        out[r, : min(fill_width, width)] = color


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill a distance-grown wedge around a left red bar, but recolor each obstacle-separated segment differently.

    Input structure:
    - The grid has a black background, one red horizontal segment, and several grey horizontal obstacle segments.
    - The red segment is contiguous, starts at the left edge, and sits around the vertical middle of the grid.
    - Grey obstacle rows appear above and below the red segment and fully cut across the wedge width for their row.

    Rule:
    - Preserve the red segment and every grey obstacle segment.
    - Above the red segment, row distance d uses a left-aligned wedge width of bar_len + d; below it, width is bar_len - d and stops when non-positive.
    - Obstacle rows themselves are never filled; instead they split each side into near-to-far segments. Upper segments are colored green, then blue, then yellow. Lower segments are colored pink, then orange, then teal.

    Key insight:
    - The wedge geometry depends only on distance from the red base bar, while the colors depend only on how grey obstacle rows partition that same wedge into consecutive bands.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bar_row, bar_len = _find_red_bar(g)
    out = g.copy()

    upper_obstacles = sorted(int(bar_row - r) for r in np.flatnonzero(np.any(g[:bar_row] == GREY, axis=1)))
    lower_obstacles = sorted(int(r + 1) for r in np.flatnonzero(np.any(g[bar_row + 1 :] == GREY, axis=1)))

    _paint_side(
        out,
        bar_row=bar_row,
        bar_len=bar_len,
        obstacle_distances=upper_obstacles,
        direction=-1,
        palette=UP_COLORS,
    )
    _paint_side(
        out,
        bar_row=bar_row,
        bar_len=bar_len,
        obstacle_distances=lower_obstacles,
        direction=1,
        palette=DOWN_COLORS,
    )
    out[bar_row, :bar_len] = RED
    return out


def _positive_partition(total: int, parts: int) -> list[int]:
    assert total >= parts >= 1
    cuts = sorted(random.sample(range(1, total), parts - 1)) if parts > 1 else []
    prev = 0
    out = []
    for cut in cuts + [total]:
        out.append(cut - prev)
        prev = cut
    return out


def _obstacle_distances(max_distance: int, n_segments: int) -> list[int]:
    n_obstacles = n_segments - 1
    fill_rows = max_distance - n_obstacles
    lengths = _positive_partition(fill_rows, n_segments)
    distances: list[int] = []
    d = 0
    for i, seg_len in enumerate(lengths[:-1]):
        d += seg_len
        distances.append(d + 1)
        d += 1
    return distances


def _render_obstacles(g: np.ndarray, bar_row: int, bar_len: int, distances: list[int], direction: int) -> None:
    height, width = g.shape
    for d in distances:
        r = bar_row + direction * d
        if direction < 0:
            wedge_width = bar_len + d
        else:
            wedge_width = bar_len - d
        assert 0 <= r < height
        assert wedge_width > 0
        max_extra = max(0, width - wedge_width)
        seg_len = wedge_width + random.randint(0, min(4, max_extra))
        g[r, :seg_len] = GREY


def _choose_profile() -> tuple[int, int, int, int, int, int] | None:
    bar_len = random.choice(BAR_LENGTH_CHOICES)

    up_segments = random.choice((2, 3))
    down_choices = [2]
    if bar_len >= 6:
        down_choices.append(3)
    down_segments = random.choice(down_choices)

    min_bar_row = 2 * up_segments - 1
    if min_bar_row > GRID_MAX - 5:
        return None
    bar_row = random.randint(min_bar_row, min(10, GRID_MAX - 5))

    min_lower_d = 2 * down_segments - 1
    max_lower_d = bar_len - 1
    if min_lower_d > max_lower_d:
        return None
    lower_dmax = random.randint(min_lower_d, max_lower_d)

    min_width = bar_len + bar_row
    width_lo = max(GRID_MIN, min_width)
    width_hi = min(GRID_MAX, min_width + 6)
    if width_lo > width_hi:
        return None
    width = random.randint(width_lo, width_hi)

    min_height = bar_row + 1 + lower_dmax
    height_lo = max(GRID_MIN, min_height)
    height_hi = min(GRID_MAX, min_height + 4)
    if height_lo > height_hi:
        return None
    height = random.randint(height_lo, height_hi)
    return height, width, bar_row, bar_len, up_segments, down_segments


def _gen_attempt() -> np.ndarray | None:
    profile = _choose_profile()
    if profile is None:
        return None
    height, width, bar_row, bar_len, up_segments, down_segments = profile

    g = np.full((height, width), BG, dtype=int)
    g[bar_row, :bar_len] = RED

    upper_obstacles = _obstacle_distances(bar_row, up_segments)
    lower_dmax = min(height - bar_row - 1, bar_len - 1)
    lower_obstacles = _obstacle_distances(lower_dmax, down_segments)

    _render_obstacles(g, bar_row, bar_len, upper_obstacles, -1)
    _render_obstacles(g, bar_row, bar_len, lower_obstacles, 1)

    upper_rows = np.flatnonzero(np.any(g[:bar_row] == GREY, axis=1))
    lower_rows = np.flatnonzero(np.any(g[bar_row + 1 :] == GREY, axis=1))
    if len(upper_rows) != up_segments - 1 or len(lower_rows) != down_segments - 1:
        return None

    try:
        out = main(g)
    except AssertionError:
        return None

    if np.array_equal(out, g):
        return None

    used_upper = {int(out[bar_row - d, 0]) for d in range(1, bar_row + 1) if bar_row - d >= 0 and d not in upper_obstacles}
    used_lower = {
        int(out[bar_row + d, 0])
        for d in range(1, lower_dmax + 1)
        if bar_row + d < height and d not in lower_obstacles
    }
    if len(used_upper) != up_segments or len(used_lower) != down_segments:
        return None
    return g


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
