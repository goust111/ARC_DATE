import numpy as np


BACKGROUND = 7  # orange
COLOR_POOL = [c for c in range(10) if c != BACKGROUND]
MIN_BARS = 3
MAX_BARS = 8
MIN_SIZE = 6
MAX_SIZE = 18
MAX_GENERATION_ATTEMPTS = 300
_RNG = np.random.default_rng()


def _extract_bars(grid: np.ndarray) -> tuple[int, list[dict[str, int]]]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    height, width = map(int, g.shape)

    non_bg_cols = [c for c in range(width) if np.any(g[:, c] != BACKGROUND)]
    assert non_bg_cols, "expected at least one bar"
    start = non_bg_cols[0]
    end = non_bg_cols[-1]
    assert non_bg_cols == list(range(start, end + 1)), "bars must occupy consecutive columns"

    bars: list[dict[str, int]] = []
    for col in range(start, end + 1):
        column = g[:, col]
        non_bg_rows = np.flatnonzero(column != BACKGROUND)
        assert non_bg_rows.size > 0, "every bar column must be nonempty"
        top = int(non_bg_rows[0])
        color = int(column[top])
        assert color != BACKGROUND
        assert np.all(column[:top] == BACKGROUND), "bars must extend upward from the bottom"
        assert np.all(column[top:] == color), "each bar must be monochrome and contiguous"
        bars.append({"col": col, "color": color, "height": height - top})

    colors = [bar["color"] for bar in bars]
    assert len(colors) == len(set(colors)), "bar colors must be distinct"
    return height, bars


def main(grid: np.ndarray) -> np.ndarray:
    """Rotate bar colors right by one slot and bar heights left by one slot."""
    g = np.asarray(grid, dtype=int)
    height, bars = _extract_bars(g)

    out = np.full_like(g, BACKGROUND)
    count = len(bars)
    for i, bar in enumerate(bars):
        color = bars[(i - 1) % count]["color"]
        new_height = bars[(i + 1) % count]["height"]
        out[height - new_height : height, bar["col"]] = color
    return out


def _render_grid(height: int, width: int, start: int, colors: np.ndarray, heights: np.ndarray) -> np.ndarray:
    grid = np.full((height, width), BACKGROUND, dtype=int)
    for offset, (color, bar_height) in enumerate(zip(colors, heights, strict=False)):
        col = start + offset
        grid[height - int(bar_height) : height, col] = int(color)
    return grid


def _sample_heights(height: int, count: int) -> np.ndarray | None:
    choices = np.arange(1, height + 1)
    if count > len(choices):
        return None
    heights = _RNG.choice(choices, size=count, replace=False)
    _RNG.shuffle(heights)
    return heights.astype(int)


def _gen_attempt() -> np.ndarray | None:
    bar_count = int(_RNG.integers(MIN_BARS, MAX_BARS + 1))
    height = int(_RNG.integers(max(MIN_SIZE, bar_count), MAX_SIZE + 1))
    width_min = max(MIN_SIZE, bar_count)
    width = int(_RNG.integers(width_min, MAX_SIZE + 1))
    if width < bar_count:
        return None

    max_start = width - bar_count
    start = int(_RNG.integers(0, max_start + 1))
    colors = _RNG.choice(COLOR_POOL, size=bar_count, replace=False).astype(int)
    heights = _sample_heights(height, bar_count)
    if heights is None:
        return None

    grid = _render_grid(height, width, start, colors, heights)
    try:
        out = main(grid)
    except AssertionError:
        return None

    if np.array_equal(grid, out):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
