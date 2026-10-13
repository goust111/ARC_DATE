from __future__ import annotations

import random

import numpy as np


BACKGROUND = 0
COLOR_POOL = tuple(range(1, 10))
MIN_LINES_PER_DIRECTION = 3
MAX_LINES_PER_DIRECTION = 5
MIN_SIZE = 9
MAX_SIZE = 30
GENERATION_ATTEMPTS = 300


def _full_non_bg_rows(g: np.ndarray, bg: int) -> list[int]:
    return [int(r) for r in range(g.shape[0]) if bool(np.all(g[r, :] != bg))]


def _full_non_bg_cols(g: np.ndarray, bg: int) -> list[int]:
    return [int(c) for c in range(g.shape[1]) if bool(np.all(g[:, c] != bg))]


def _single_visible_color(values: np.ndarray, bg: int) -> int:
    colors = [int(v) for v in np.unique(values) if int(v) != int(bg)]
    assert len(colors) == 1, "line color must be visible on its non-crossing cells"
    return int(colors[0])


def _line_colors(g: np.ndarray, h_rows: list[int], v_cols: list[int], bg: int) -> tuple[list[int], list[int]]:
    height, width = g.shape
    v_col_set = set(v_cols)
    h_row_set = set(h_rows)
    non_v_cols = [c for c in range(width) if c not in v_col_set]
    non_h_rows = [r for r in range(height) if r not in h_row_set]
    assert non_v_cols, "expected at least one non-vertical column"
    assert non_h_rows, "expected at least one non-horizontal row"

    h_colors = [_single_visible_color(g[r, non_v_cols], bg) for r in h_rows]
    v_colors = [_single_visible_color(g[non_h_rows, c], bg) for c in v_cols]
    all_colors = h_colors + v_colors
    assert len(set(all_colors)) == len(all_colors), "every thread must have a distinct color"
    return h_colors, v_colors


def main(grid: np.ndarray) -> np.ndarray:
    """Repair the single wrong over-under crossing in a woven set of colored threads.

    Input structure:
    - The background is black.
    - Several distinct-colored horizontal threads span the full grid width, and
      several distinct-colored vertical threads span the full grid height.
    - At each horizontal/vertical crossing, the visible color is exactly one of
      the two crossing thread colors.

    Rule:
    - Identify full horizontal and vertical threads and their colors from the
      uninterrupted cells between crossings.
    - Count, for every thread, how often its color is visible at its crossings.
      A strict majority means that thread is normally on top.
    - Find the unique crossing where the shown thread is normally underneath and
      the hidden thread is normally on top, then recolor that pixel to the hidden
      top thread's color.
    - Leave all other pixels unchanged.

    Key insight:
    - A crossing pixel encodes an occlusion relation; the one wrong pixel is
      exposed only after the normal layer of each thread is inferred by majority.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    bg = BACKGROUND

    h_rows = _full_non_bg_rows(g, bg)
    v_cols = _full_non_bg_cols(g, bg)
    assert h_rows and v_cols, "expected horizontal and vertical full-length threads"

    h_colors, v_colors = _line_colors(g, h_rows, v_cols, bg)
    h_index = {r: i for i, r in enumerate(h_rows)}
    v_index = {c: j for j, c in enumerate(v_cols)}
    assert len(h_index) == len(h_rows)
    assert len(v_index) == len(v_cols)

    h_top_counts = [0] * len(h_rows)
    v_top_counts = [0] * len(v_cols)
    visible_is_h: dict[tuple[int, int], bool] = {}

    for r in h_rows:
        i = h_index[r]
        for c in v_cols:
            j = v_index[c]
            value = int(g[r, c])
            if value == h_colors[i]:
                h_top_counts[i] += 1
                visible_is_h[(i, j)] = True
            elif value == v_colors[j]:
                v_top_counts[j] += 1
                visible_is_h[(i, j)] = False
            else:
                raise AssertionError("crossing must show one of the two thread colors")

    h_normally_top: list[bool] = []
    v_normally_top: list[bool] = []
    v_total = len(v_cols)
    h_total = len(h_rows)
    for count in h_top_counts:
        assert 2 * count != v_total, "horizontal thread layer must have a strict majority"
        h_normally_top.append(2 * count > v_total)
    for count in v_top_counts:
        assert 2 * count != h_total, "vertical thread layer must have a strict majority"
        v_normally_top.append(2 * count > h_total)

    repairs: list[tuple[int, int, int]] = []
    for i, r in enumerate(h_rows):
        for j, c in enumerate(v_cols):
            h_top = h_normally_top[i]
            v_top = v_normally_top[j]
            assert h_top != v_top, "each crossing must have exactly one normally top thread"
            if visible_is_h[(i, j)]:
                if (not h_top) and v_top:
                    repairs.append((r, c, v_colors[j]))
            else:
                if h_top and (not v_top):
                    repairs.append((r, c, h_colors[i]))

    assert len(repairs) == 1, "expected exactly one wrong crossing"
    out = g.copy()
    r, c, color = repairs[0]
    out[r, c] = color
    return out


# --------------- generator ---------------


def _max_spaced_count(size: int) -> int:
    return max(0, (size - 1) // 2)


def _spaced_positions(size: int, count: int) -> list[int]:
    usable = size - 2
    assert 0 < count <= (usable + 1) // 2
    compressed = sorted(random.sample(range(usable - count + 1), count))
    return [int(1 + value + i) for i, value in enumerate(compressed)]


def _render_threads(
    height: int,
    width: int,
    h_rows: list[int],
    v_cols: list[int],
    h_colors: list[int],
    v_colors: list[int],
    horizontal_over: bool,
) -> np.ndarray:
    grid = np.full((height, width), BACKGROUND, dtype=int)
    for r, color in zip(h_rows, h_colors):
        grid[r, :] = int(color)
    for c, color in zip(v_cols, v_colors):
        grid[:, c] = int(color)
    for i, r in enumerate(h_rows):
        for j, c in enumerate(v_cols):
            grid[r, c] = int(h_colors[i] if horizontal_over else v_colors[j])
    return grid


def _sample_counts(height: int, width: int) -> tuple[int, int] | None:
    max_h = min(MAX_LINES_PER_DIRECTION, _max_spaced_count(height))
    max_v = min(MAX_LINES_PER_DIRECTION, _max_spaced_count(width))
    pairs = [
        (h_count, v_count)
        for h_count in range(MIN_LINES_PER_DIRECTION, max_h + 1)
        for v_count in range(MIN_LINES_PER_DIRECTION, max_v + 1)
        if h_count + v_count <= len(COLOR_POOL)
    ]
    if not pairs:
        return None
    return random.choice(pairs)


def _gen_attempt() -> np.ndarray | None:
    height = random.randint(MIN_SIZE, MAX_SIZE)
    width = random.randint(MIN_SIZE, MAX_SIZE)
    counts = _sample_counts(height, width)
    if counts is None:
        return None
    h_count, v_count = counts

    h_rows = _spaced_positions(height, h_count)
    v_cols = _spaced_positions(width, v_count)
    colors = random.sample(COLOR_POOL, h_count + v_count)
    h_colors = [int(c) for c in colors[:h_count]]
    v_colors = [int(c) for c in colors[h_count:]]
    horizontal_over = bool(random.getrandbits(1))

    expected = _render_threads(height, width, h_rows, v_cols, h_colors, v_colors, horizontal_over)
    inp = expected.copy()
    bad_i = random.randrange(h_count)
    bad_j = random.randrange(v_count)
    inp[h_rows[bad_i], v_cols[bad_j]] = int(v_colors[bad_j] if horizontal_over else h_colors[bad_i])

    if np.array_equal(inp, expected):
        return None
    try:
        actual = main(inp)
    except AssertionError:
        return None
    if not np.array_equal(actual, expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid woven-thread anomaly input")
