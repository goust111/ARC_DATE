import random

import numpy as np


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _find_baseline(grid: np.ndarray, bg: int) -> tuple[int, int]:
    best = None
    h, w = grid.shape
    for r in range(h):
        row = grid[r]
        if np.all(row == row[0]) and int(row[0]) != bg:
            score = w
            cand = (score, r, int(row[0]))
            if best is None or cand > best:
                best = cand
    assert best is not None, "expected a full-width non-background baseline"
    return int(best[1]), int(best[2])


def _extract_bars(grid: np.ndarray, bg: int, base_r: int, base_color: int) -> list[dict[str, int]]:
    h, w = grid.shape
    bars: list[dict[str, int]] = []
    for c in range(w):
        colors = {int(v) for v in grid[:base_r, c] if int(v) != bg}
        assert len(colors) <= 1, "each slot may contain at most one bar color"
        if not colors:
            continue
        color = colors.pop()
        assert color != base_color, "bar color must differ from baseline"
        rows = np.where(grid[:base_r, c] == color)[0]
        assert rows.size > 0
        r0 = int(rows[0])
        r1 = int(rows[-1])
        assert r1 == base_r - 1, "bar must touch the baseline from above"
        assert rows.size == r1 - r0 + 1, "bar must be vertically contiguous"
        assert np.all(grid[r0 : r1 + 1, c] == color)
        bars.append({"col": c, "color": color, "height": int(rows.size)})
    assert len(bars) >= 2, "expected multiple bars"
    return bars


def main(grid: np.ndarray) -> np.ndarray:
    """Sort vertical bars by height and redistribute them across fixed colored slots."""
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    h, _w = g.shape
    bg = _get_bg(g)
    base_r, base_color = _find_baseline(g, bg)
    assert 1 <= base_r < h
    bars = _extract_bars(g, bg, base_r, base_color)

    slots = sorted(bars, key=lambda bar: bar["col"])
    shapes = sorted(bars, key=lambda bar: (bar["height"], bar["col"]))

    out = np.full_like(g, bg)
    out[base_r, :] = base_color
    for slot, shape in zip(slots, shapes):
        out[base_r - shape["height"] : base_r, slot["col"]] = slot["color"]
    return out


_COLORS = tuple(range(10))
_BAR_COUNT_RANGE = (3, 7)
_GRID_HEIGHT_RANGE = (7, 18)
_GRID_WIDTH_RANGE = (7, 20)
_MAX_ATTEMPTS = 400


def _sample_columns(width: int, count: int) -> list[int]:
    cols = random.sample(range(width), count)
    cols.sort()
    return cols


def _render_scene(
    height: int,
    width: int,
    bg: int,
    base_r: int,
    base_color: int,
    cols: list[int],
    slot_colors: list[int],
    bar_heights: list[int],
) -> np.ndarray:
    grid = np.full((height, width), bg, dtype=int)
    grid[base_r, :] = base_color
    for c, color, bar_h in zip(cols, slot_colors, bar_heights):
        grid[base_r - bar_h : base_r, c] = color
    return grid


def _oracle_output(
    height: int,
    width: int,
    bg: int,
    base_r: int,
    base_color: int,
    cols: list[int],
    slot_colors: list[int],
    bar_heights: list[int],
) -> np.ndarray:
    return _render_scene(
        height,
        width,
        bg,
        base_r,
        base_color,
        cols,
        slot_colors,
        sorted(bar_heights),
    )


def _gen_attempt() -> np.ndarray | None:
    height = random.randint(*_GRID_HEIGHT_RANGE)
    width = random.randint(*_GRID_WIDTH_RANGE)
    max_bars = min(_BAR_COUNT_RANGE[1], width)
    if max_bars < _BAR_COUNT_RANGE[0]:
        return None
    count = random.randint(_BAR_COUNT_RANGE[0], max_bars)

    base_r = random.randint(2, height - 2)
    bg = random.choice(_COLORS)
    remaining = [c for c in _COLORS if c != bg]
    base_color = random.choice(remaining)
    remaining = [c for c in remaining if c != base_color]
    if len(remaining) < count:
        return None
    slot_colors = random.sample(remaining, count)
    cols = _sample_columns(width, count)

    height_choices = list(range(1, base_r + 1))
    if len(height_choices) >= count and random.random() < 0.6:
        bar_heights = random.sample(height_choices, count)
    else:
        bar_heights = [random.choice(height_choices) for _ in range(count)]
    if len(set(bar_heights)) == 1:
        idx = random.randrange(count)
        alternatives = [v for v in height_choices if v != bar_heights[idx]]
        if not alternatives:
            return None
        bar_heights[idx] = random.choice(alternatives)

    if bar_heights == sorted(bar_heights):
        order = list(range(count))
        random.shuffle(order)
        if order == list(range(count)):
            order[0], order[-1] = order[-1], order[0]
        bar_heights = [bar_heights[i] for i in order]

    inp = _render_scene(height, width, bg, base_r, base_color, cols, slot_colors, bar_heights)
    expected = _oracle_output(height, width, bg, base_r, base_color, cols, slot_colors, bar_heights)
    if np.array_equal(inp, expected):
        return None
    try:
        actual = main(inp)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(_MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
