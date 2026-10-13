import numpy as np

from arc_utils import find_components, get_bg


def _extract_bars(g: np.ndarray, bg: int) -> list[dict[str, int]]:
    height = g.shape[0]
    components = find_components(g, bg=bg, connectivity=4, monochromatic=True)
    assert len(components) >= 2

    bars: list[dict[str, int]] = []
    for comp in components:
        r0, c0, r1, c1 = map(int, comp["bbox"])
        color = int(comp["color"])
        area = int(comp["area"])

        assert c0 == c1
        assert r1 == height - 1
        assert area == r1 - r0 + 1
        assert bool(np.all(g[r0 : r1 + 1, c0] == color))

        bars.append({"color": color, "col": c0, "height": area})

    bars.sort(key=lambda bar: bar["col"])
    assert len({bar["col"] for bar in bars}) == len(bars)
    return bars


def main(grid: np.ndarray) -> np.ndarray:
    """Rotate each vertical bar's color one slot right and its height one slot left.

    Input structure
    - A dominant background surrounds several monochrome one-cell-wide vertical bars.
    - Every bar is bottom-aligned and occupies a distinct column.
    - The ordered bar slots are read from left to right.

    Rule
    - Keep the grid shape and the bar columns unchanged.
    - At slot i, use the color from the previous slot cyclically.
    - At slot i, use the height from the next slot cyclically.

    Key insight
    - Color and height are independent cyclic attributes of the ordered bar slots:
      colors rotate right while heights rotate left.
    """
    g = np.asarray(grid, dtype=int)
    height = g.shape[0]
    bg = get_bg(g)
    bars = _extract_bars(g, bg)

    out = np.full_like(g, bg)
    count = len(bars)
    for i, bar in enumerate(bars):
        color = bars[(i - 1) % count]["color"]
        bar_height = bars[(i + 1) % count]["height"]
        assert 1 <= bar_height <= height
        out[height - bar_height : height, bar["col"]] = color

    return out

# --------------- generator (o5x) ---------------

_RNG = np.random.default_rng()

# Canvas/bar bounds.
BAR_COUNT_MIN, BAR_COUNT_MAX_EXCLUSIVE = 3, 6
BACKGROUND_COLOR, MIN_BAR_HEIGHT = 7, 1
BAR_COL_OFFSET, BAR_COL_STRIDE = 1, 2
# Color and transform sampling.
COLOR_POOL = [0, 1, 2, 3, 4, 5, 6, 8, 9]
OUTPUT_COLOR_SHIFT, OUTPUT_HEIGHT_SHIFT = 1, -1
# Retry limit.
MAX_GENERATION_ATTEMPTS = 300


def _render_bars(n: int, colors: np.ndarray, heights: np.ndarray) -> np.ndarray:
    side = BAR_COL_STRIDE * n + BAR_COL_OFFSET
    grid = np.full((side, side), BACKGROUND_COLOR, dtype=int)
    for i, (color, bar_height) in enumerate(zip(colors, heights)):
        col = BAR_COL_STRIDE * i + BAR_COL_OFFSET
        grid[side - int(bar_height) : side, col] = int(color)
    return grid


def _expected_output(n: int, colors: np.ndarray, heights: np.ndarray) -> np.ndarray:
    out_colors = np.roll(colors, OUTPUT_COLOR_SHIFT)
    out_heights = np.roll(heights, OUTPUT_HEIGHT_SHIFT)
    return _render_bars(n, out_colors, out_heights)


def _gen_attempt() -> np.ndarray | None:
    n = int(_RNG.integers(BAR_COUNT_MIN, BAR_COUNT_MAX_EXCLUSIVE))
    side = BAR_COL_STRIDE * n + BAR_COL_OFFSET
    colors = _RNG.choice(COLOR_POOL, size=n, replace=False)
    heights = _RNG.integers(MIN_BAR_HEIGHT, side, size=n)
    if len(set(map(int, heights))) == 1:
        idx = int(_RNG.integers(n))
        choices = [h for h in range(MIN_BAR_HEIGHT, side) if h != int(heights[idx])]
        heights[idx] = int(_RNG.choice(choices))

    grid = _render_bars(n, colors, heights)
    if np.array_equal(main(grid), _expected_output(n, colors, heights)):
        return grid
    return None


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
