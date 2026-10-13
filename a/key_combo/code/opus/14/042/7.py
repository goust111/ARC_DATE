from __future__ import annotations

import numpy as np


BG = 0
MARKER = 5
LEFT = -1
RIGHT = 1
BAR_COLORS = (1, 2, 3, 4, 6, 7, 8, 9)

MIN_BARS = 4
MAX_BARS = 6
MIN_HEIGHT = 3
MAX_HEIGHT = 15

GEN_HEIGHT_RANGE = (17, 24)
GEN_ATTEMPTS = 600


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= r < shape[0] and 0 <= c < shape[1]


def _parse_bars(grid: np.ndarray) -> list[dict[str, int]]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    h, w = g.shape
    assert h >= 5 and w >= 7, "grid too small"

    bars: list[dict[str, int]] = []
    bar_cells: set[tuple[int, int]] = set()
    marker_cells: set[tuple[int, int]] = set()

    for c in range(w):
        color = int(g[h - 1, c])
        if color in (BG, MARKER):
            continue

        r = h - 1
        while r >= 0 and int(g[r, c]) == color:
            r -= 1
        top = r + 1
        height = h - top

        assert MIN_HEIGHT <= height <= h - 1, "invalid bar height"
        assert np.all(g[top:h, c] == color), "bar must be contiguous and bottom-aligned"
        assert not np.any((g[:top, c] != BG) & (g[:top, c] != MARKER)), "unexpected extra pixels above a bar"

        dirs = []
        if top > 0 and c > 0 and int(g[top - 1, c - 1]) == MARKER:
            dirs.append(LEFT)
        if top > 0 and c + 1 < w and int(g[top - 1, c + 1]) == MARKER:
            dirs.append(RIGHT)
        assert len(dirs) == 1, "each bar must have exactly one diagonal marker"
        direction = dirs[0]
        marker_c = c + direction
        marker = (top - 1, marker_c)

        assert marker not in marker_cells, "markers must be unique"
        marker_cells.add(marker)

        for rr in range(top, h):
            cell = (rr, c)
            assert cell not in bar_cells, "bars must not overlap"
            bar_cells.add(cell)

        bars.append(
            {
                "col": c,
                "top": top,
                "height": height,
                "color": color,
                "direction": direction,
            }
        )

    assert MIN_BARS <= len(bars) <= MAX_BARS, "expected 4-6 bars"
    bars.sort(key=lambda bar: bar["col"])

    cols = [bar["col"] for bar in bars]
    heights = [bar["height"] for bar in bars]
    colors = [bar["color"] for bar in bars]
    assert len(set(cols)) == len(cols), "bar columns must be unique"
    assert len(set(heights)) == len(heights), "bar heights must be distinct"
    assert len(set(colors)) == len(colors), "bar colors must be distinct"
    for a, b in zip(cols, cols[1:]):
        gap = b - a - 1
        assert 1 <= gap <= 3, "input bars must be separated by 1-3 blank columns"

    actual_markers = {tuple(map(int, rc)) for rc in np.argwhere(g == MARKER)}
    assert actual_markers == marker_cells, "grey pixels must be exactly the direction markers"

    actual_bar_cells = {
        tuple(map(int, rc))
        for rc in np.argwhere((g != BG) & (g != MARKER))
    }
    assert actual_bar_cells == bar_cells, "non-grey foreground pixels must belong to bars"
    return bars


def _render_layout(
    bars_in_draw_order: list[dict[str, int]],
    target_cols: list[int],
    shape: tuple[int, int],
) -> tuple[np.ndarray, list[dict[str, int]], list[dict[str, int | str]]]:
    h, w = shape
    out = np.zeros((h, w), dtype=int)
    static_bar_mask = np.zeros((h, w), dtype=bool)
    placed: list[dict[str, int]] = []

    for bar, col in zip(bars_in_draw_order, target_cols):
        top = h - bar["height"]
        assert 0 <= top < h
        assert 0 <= col < w
        out[top:h, col] = bar["color"]
        static_bar_mask[top:h, col] = True
        placed.append(
            {
                "height": bar["height"],
                "color": bar["color"],
                "direction": bar["direction"],
                "col": col,
                "top": top,
            }
        )

    traces: list[dict[str, int | str]] = []
    for bar in placed:
        r = bar["top"]
        c = bar["col"]
        dc = bar["direction"]
        color = bar["color"]
        length = 0
        stop_kind = ""

        while True:
            nr = r - 1
            nc = c + dc
            if not _in_bounds((h, w), nr, nc):
                stop_kind = "boundary"
                break
            if int(out[nr, nc]) != BG:
                stop_kind = "collision_bar" if static_bar_mask[nr, nc] else "collision_line"
                break
            out[nr, nc] = color
            r, c = nr, nc
            length += 1

        traces.append(
            {
                "color": color,
                "height": bar["height"],
                "stop": stop_kind,
                "length": length,
            }
        )

    return out, placed, traces


def _sorted_bars(bars: list[dict[str, int]]) -> list[dict[str, int]]:
    return sorted(bars, key=lambda bar: bar["height"])


def _render_sorted_output(grid: np.ndarray) -> tuple[np.ndarray, list[dict[str, int]], list[dict[str, int | str]]]:
    bars = _sorted_bars(_parse_bars(grid))
    target_cols = [2 * idx for idx in range(len(bars))]
    return _render_layout(bars, target_cols, grid.shape)


def _render_in_place_height_order(grid: np.ndarray) -> np.ndarray:
    bars = _sorted_bars(_parse_bars(grid))
    cols = [bar["col"] for bar in bars]
    out, _placed, _traces = _render_layout(bars, cols, grid.shape)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reorder the bottom-aligned bars by height, then draw each bar's directed diagonal until it collides.

    Input structure:
    - The background is black.
    - There are 4-6 one-column colored vertical bars, all bottom-aligned and all with different heights.
    - Each bar has exactly one grey pixel diagonally above its top cell, on either the upper-left or upper-right,
      which marks whether that bar should draw a left-up or right-up diagonal.

    Rule:
    - Measure every bar's height, ignoring its grey marker.
    - Reorder the bars into ascending height and place them at columns 0, 2, 4, ... with one blank column between neighbors,
      keeping their colors, heights, bottom alignment, and stored marker directions.
    - Remove all grey markers.
    - In ascending-height order, draw a same-colored diagonal from each reordered bar's top cell in its marked direction.
      Paint only empty cells, and stop before the first step that would hit another reordered bar, hit an already drawn diagonal,
      or leave the grid.

    Key insight:
    - The grey pixels do not connect objects directly; they encode diagonal direction, while the mandatory height sort changes
      the spatial layout and therefore changes where each later collision happens.
    """
    out, _placed, _traces = _render_sorted_output(np.asarray(grid, dtype=int))
    return out


def _sample_sorted_specs(rng: np.random.Generator, h: int, n: int) -> list[dict[str, int]] | None:
    max_height = min(MAX_HEIGHT, h - 2)
    if max_height < 8:
        return None

    h1_min = 6
    h1_max = max_height - (n - 2)
    if h1_max < h1_min:
        return None

    h1 = int(rng.integers(h1_min, h1_max + 1))
    h0 = int(rng.integers(MIN_HEIGHT, h1 - 1))
    if h0 > h1 - 2:
        h0 = h1 - 2
    if h0 < MIN_HEIGHT:
        return None

    base_heights = [h0, h1, h1 + 1, h1 + 2]
    extra_count = n - 4
    extras: list[int] = []
    if extra_count:
        pool = list(range(h1 + 3, max_height + 1))
        if len(pool) < extra_count:
            return None
        extras = sorted(int(v) for v in rng.choice(pool, size=extra_count, replace=False))

    heights = base_heights + extras
    directions = [RIGHT, RIGHT, LEFT, LEFT]
    while len(directions) < n:
        directions.append(int(rng.choice((LEFT, RIGHT))))
    directions[-1] = LEFT

    colors = [int(v) for v in rng.choice(BAR_COLORS, size=n, replace=False)]
    return [
        {
            "height": height,
            "direction": direction,
            "color": color,
        }
        for height, direction, color in zip(heights, directions, colors)
    ]


def _sample_input_columns(rng: np.random.Generator, n: int) -> tuple[int, list[int]] | None:
    gaps = [int(rng.integers(1, 4)) for _ in range(n - 1)]
    used_without_extra = 2 + n + sum(gaps)
    w_min = max(13, used_without_extra)
    w_max = min(20, used_without_extra + 4)
    if w_min > w_max:
        return None

    w = int(rng.integers(w_min, w_max + 1))
    extra = w - used_without_extra
    left_margin = 1 + int(rng.integers(0, extra + 1))
    cols = [left_margin]
    for gap in gaps:
        cols.append(cols[-1] + 1 + gap)
    if cols[-1] >= w - 1:
        return None
    return w, cols


def _build_input(
    shape: tuple[int, int],
    ordered_bars: list[dict[str, int]],
    cols: list[int],
) -> np.ndarray:
    h, w = shape
    grid = np.zeros((h, w), dtype=int)
    for bar, col in zip(ordered_bars, cols):
        top = h - bar["height"]
        grid[top:h, col] = bar["color"]
        grid[top - 1, col + bar["direction"]] = MARKER
    return grid


def _semantic_checks(grid: np.ndarray) -> bool:
    try:
        bars = _parse_bars(grid)
        input_heights = [bar["height"] for bar in bars]
        if input_heights == sorted(input_heights):
            return False

        out, placed, traces = _render_sorted_output(grid)
        if not np.array_equal(out, main(grid)):
            return False
        if np.array_equal(grid, out):
            return False
        if np.array_equal(out, _render_in_place_height_order(grid)):
            return False

        if {bar["direction"] for bar in placed} != {LEFT, RIGHT}:
            return False

        stops = {str(trace["stop"]) for trace in traces}
        if not {"boundary", "collision_bar", "collision_line"}.issubset(stops):
            return False
        positive_lengths = [int(trace["length"]) for trace in traces if int(trace["length"]) > 0]
        if len(positive_lengths) < 3:
            return False
        if sum(positive_lengths) < len(traces):
            return False

        sorted_cols = [bar["col"] for bar in placed]
        if sorted_cols != list(range(0, 2 * len(placed), 2)):
            return False

        return True
    except AssertionError:
        return False


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    h = int(rng.integers(GEN_HEIGHT_RANGE[0], GEN_HEIGHT_RANGE[1] + 1))
    n = int(rng.integers(MIN_BARS, MAX_BARS + 1))

    sorted_specs = _sample_sorted_specs(rng, h, n)
    if sorted_specs is None:
        return None

    width_and_cols = _sample_input_columns(rng, n)
    if width_and_cols is None:
        return None
    w, cols = width_and_cols

    perm = list(range(n))
    rng.shuffle(perm)
    if perm == list(range(n)):
        return None

    input_bars = [sorted_specs[idx] for idx in perm]
    grid = _build_input((h, w), input_bars, cols)
    if not _semantic_checks(grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 7")
