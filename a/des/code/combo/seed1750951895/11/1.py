import numpy as np

from arc_utils import get_bg


def _find_baseline(g: np.ndarray, bg: int) -> tuple[int, int]:
    h, _ = g.shape
    candidates: list[tuple[int, int]] = []
    for r in range(h):
        row = g[r]
        if int(row[0]) != bg and bool(np.all(row == row[0])):
            candidates.append((r, int(row[0])))
    assert candidates, "expected a full-width non-background baseline"
    baseline_row, baseline_color = candidates[-1]
    assert baseline_row == h - 1, "baseline must be on the bottom row"
    return baseline_row, baseline_color


def _extract_bars(g: np.ndarray, bg: int, baseline_row: int) -> list[dict[str, int]]:
    bars: list[dict[str, int]] = []
    for c in range(g.shape[1]):
        above = g[:baseline_row, c]
        non_bg_rows = np.flatnonzero(above != bg)
        if non_bg_rows.size == 0:
            continue

        r0 = int(non_bg_rows[0])
        assert bool(np.all(above[:r0] == bg)), "bar column must be empty above the bar"
        color = int(above[r0])
        assert color != bg
        assert bool(np.all(above[r0:] == color)), "each bar must be a solid contiguous column"

        bars.append(
            {
                "col": c,
                "color": color,
                "height": baseline_row - r0,
            }
        )

    assert len(bars) >= 2, "expected at least two bars"
    assert len({bar["col"] for bar in bars}) == len(bars)
    assert len({bar["color"] for bar in bars}) == len(bars), "bar colors must be distinct"
    assert len({bar["height"] for bar in bars}) == len(bars), "bar heights must be distinct"
    return bars


def main(grid: np.ndarray) -> np.ndarray:
    """Sort vertical bar heights ascending and redraw them into the original left-to-right slots.

    Input structure:
    - Black background.
    - A full-width colored baseline on the bottom row.
    - Several one-cell-wide vertical bars grow upward from the baseline, each in its own column.

    Rule:
    - Read the bar slots from left to right by column.
    - Sort the source bar heights from shortest to tallest.
    - Put the i-th shortest height into the i-th slot, but keep that slot's original color.
    - Leave the baseline unchanged.

    Key insight:
    - Position controls color identity, while the sorted heights are reassigned independently.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg = int(get_bg(g))
    assert bg == 0, "background must be black"
    baseline_row, baseline_color = _find_baseline(g, bg)
    bars = _extract_bars(g, bg, baseline_row)
    bars.sort(key=lambda bar: bar["col"])

    sorted_heights = sorted(int(bar["height"]) for bar in bars)
    out = np.full_like(g, bg)
    out[baseline_row, :] = baseline_color
    for bar, height in zip(bars, sorted_heights):
        out[baseline_row - height : baseline_row, bar["col"]] = int(bar["color"])
    return out


# --------------- generator ---------------

_RNG = np.random.default_rng()

BACKGROUND_COLOR = 0
GRID_HEIGHT_RANGE = (7, 16)
BAR_COUNT_RANGE = (3, 7)
MARGIN_RANGE = (0, 4)
COL_GAP_RANGE = (1, 5)
COLOR_POOL = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=int)
MAX_GENERATION_ATTEMPTS = 300


def _render_grid(
    height: int,
    width: int,
    baseline_color: int,
    cols: list[int],
    colors: list[int],
    heights: list[int],
) -> np.ndarray:
    grid = np.full((height, width), BACKGROUND_COLOR, dtype=int)
    grid[height - 1, :] = int(baseline_color)
    for col, color, bar_height in zip(cols, colors, heights):
        grid[height - 1 - int(bar_height) : height - 1, int(col)] = int(color)
    return grid


def _expected_output(
    height: int,
    width: int,
    baseline_color: int,
    cols: list[int],
    colors: list[int],
    heights: list[int],
) -> np.ndarray:
    return _render_grid(height, width, baseline_color, cols, colors, sorted(heights))


def _sample_columns(n: int) -> tuple[int, list[int]] | None:
    left_margin = int(_RNG.integers(*MARGIN_RANGE))
    right_margin = int(_RNG.integers(*MARGIN_RANGE))
    gaps = [int(_RNG.integers(*COL_GAP_RANGE)) for _ in range(n - 1)]

    cols = [left_margin]
    for gap in gaps:
        cols.append(cols[-1] + gap)

    width = cols[-1] + 1 + right_margin
    if width < 6 or width > 30:
        return None
    return width, cols


def _assign_heights(sorted_heights: list[int]) -> list[int]:
    mode = int(_RNG.integers(3))
    if mode == 0:
        return list(sorted_heights)
    if mode == 1:
        return list(reversed(sorted_heights))

    while True:
        perm = _RNG.permutation(len(sorted_heights))
        heights = [sorted_heights[int(i)] for i in perm]
        if heights != sorted(sorted_heights) and heights != list(reversed(sorted_heights)):
            return heights


def _gen_attempt() -> np.ndarray | None:
    n = int(_RNG.integers(*BAR_COUNT_RANGE))
    height = int(_RNG.integers(*GRID_HEIGHT_RANGE))
    if height - 1 < n:
        return None

    width_and_cols = _sample_columns(n)
    if width_and_cols is None:
        return None
    width, cols = width_and_cols

    palette = _RNG.choice(COLOR_POOL, size=n + 1, replace=False)
    baseline_color = int(palette[0])
    slot_colors = [int(color) for color in palette[1:]]

    sorted_heights = sorted(int(v) for v in _RNG.choice(np.arange(1, height), size=n, replace=False))
    heights = _assign_heights(sorted_heights)

    grid = _render_grid(height, width, baseline_color, cols, slot_colors, heights)
    expected = _expected_output(height, width, baseline_color, cols, slot_colors, heights)
    if np.array_equal(main(grid), expected):
        return grid
    return None


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
