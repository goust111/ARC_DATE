import numpy as np


BACKGROUND = 0
GREY = 5
BAR_COLORS = np.array([1, 2, 3, 4, 6, 7, 8, 9], dtype=int)
MAX_GENERATION_ATTEMPTS = 400
_RNG = np.random.default_rng()
_GEN_CALL_COUNT = 0


def _background_color(g: np.ndarray) -> int:
    values, counts = np.unique(g, return_counts=True)
    return int(values[int(np.argmax(counts))])


def _indicator_lengths(g: np.ndarray) -> tuple[int, int]:
    width = int(g.shape[1])

    k1 = 0
    while k1 < width and int(g[0, k1]) == GREY:
        k1 += 1

    k2 = 0
    c = width - 1
    while c >= 0 and int(g[0, c]) == GREY:
        k2 += 1
        c -= 1

    assert k1 >= 1 and k2 >= 1, "expected grey indicators in both top corners"
    assert k1 + k2 < width, "top indicators must be separated by background"
    return int(k1), int(k2)


def _assert_indicator_mask(g: np.ndarray, k1: int, k2: int) -> None:
    expected = np.zeros(g.shape, dtype=bool)
    expected[0, :k1] = True
    expected[0, g.shape[1] - k2 :] = True
    assert np.array_equal(g == GREY, expected), "grey should appear only in the two indicators"


def _extract_bars(g: np.ndarray, bg: int) -> list[dict[str, int]]:
    height, width = map(int, g.shape)
    fg = (g != bg) & (g != GREY)
    bars: list[dict[str, int]] = []

    for col in range(width):
        rows = np.flatnonzero(fg[:, col])
        if rows.size == 0:
            continue

        top = int(rows[0])
        bottom = int(rows[-1])
        color_values = set(map(int, g[rows, col]))
        assert len(color_values) == 1, "each vertical bar must be monochrome"
        assert bottom == height - 1, "vertical bars must be bottom-aligned"
        assert top > 0, "bars must leave the indicator row clear"
        assert np.array_equal(rows, np.arange(top, height)), "bar cells must be contiguous"

        bars.append(
            {
                "col": int(col),
                "color": int(next(iter(color_values))),
                "height": int(height - top),
            }
        )

    assert len(bars) >= 3, "expected at least three ordered bar slots"
    assert len({bar["color"] for bar in bars}) == len(bars), "bar colors must be unique"
    assert len({bar["height"] for bar in bars}) == len(bars), "bar heights must be unique"
    assert int(fg.sum()) == sum(bar["height"] for bar in bars), "unexpected foreground outside bars"
    return bars


def main(grid: np.ndarray) -> np.ndarray:
    """Shift bar colors by the left key and bar heights by the right key in opposite directions.

    Input structure:
    - A background grid contains bottom-aligned, one-cell-wide vertical bars in ordered slots.
    - Bar colors are all different, bar heights are all different, and the top row is clear
      except for two grey horizontal indicators touching the upper-left and upper-right corners.
    - The left indicator length is k1 and the right indicator length is k2.

    Rule:
    - Keep the grid shape, slot columns, background, and both grey indicators unchanged.
    - At slot i, draw the color from slot i-k1 cyclically.
    - At slot i, draw the height from slot i+k2 cyclically.

    Key insight:
    - Color and height are two independent cyclic streams over the same ordered bar slots;
      the two corner indicators encode their separate, opposite shift distances.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    height, width = map(int, g.shape)
    assert height >= 4 and width >= 5, "grid too small for keyed vertical bars"

    bg = _background_color(g)
    assert bg != GREY, "background must be distinct from the grey indicators"
    k1, k2 = _indicator_lengths(g)
    _assert_indicator_mask(g, k1, k2)
    bars = _extract_bars(g, bg)

    out = np.full(g.shape, bg, dtype=int)
    count = len(bars)
    assert k1 % count != 0 and k2 % count != 0, "indicator shifts must affect the bars"

    for i, bar in enumerate(bars):
        color = bars[(i - k1) % count]["color"]
        bar_height = bars[(i + k2) % count]["height"]
        out[height - bar_height : height, bar["col"]] = color

    out[0, :k1] = GREY
    out[0, width - k2 :] = GREY
    return out


def _render(
    height: int,
    width: int,
    cols: np.ndarray,
    colors: np.ndarray,
    heights: np.ndarray,
    k1: int,
    k2: int,
) -> np.ndarray:
    grid = np.full((height, width), BACKGROUND, dtype=int)
    grid[0, :k1] = GREY
    grid[0, width - k2 :] = GREY
    for col, color, bar_height in zip(cols, colors, heights):
        top = height - int(bar_height)
        assert top > 0
        grid[top:height, int(col)] = int(color)
    return grid


def _expected_output(
    height: int,
    width: int,
    cols: np.ndarray,
    colors: np.ndarray,
    heights: np.ndarray,
    k1: int,
    k2: int,
) -> np.ndarray:
    shifted_colors = np.roll(colors, int(k1))
    shifted_heights = np.roll(heights, -int(k2))
    return _render(height, width, cols, shifted_colors, shifted_heights, k1, k2)


def _choose_indicator_pair(n: int, mode: int, rng: np.random.Generator) -> tuple[int, int]:
    limit = min(n - 1, 5)
    all_pairs = [
        (k1, k2)
        for k1 in range(1, limit + 1)
        for k2 in range(1, limit + 1)
        if k1 % n != 0 and k2 % n != 0 and (k1 + k2) % n != 0
    ]

    if mode == 0:
        candidates = [(1, 1)] if (1, 1) in all_pairs else []
    elif mode == 1:
        candidates = [(k1, k2) for k1, k2 in all_pairs if k1 > 1 and k2 == 1]
    elif mode == 2:
        candidates = [(k1, k2) for k1, k2 in all_pairs if k1 == 1 and k2 > 1]
    else:
        candidates = [(k1, k2) for k1, k2 in all_pairs if k1 > 1 and k2 > 1 and k1 != k2]

    if not candidates:
        candidates = all_pairs
    assert candidates
    return candidates[int(rng.integers(0, len(candidates)))]


def _sample_spaced_columns(width: int, n: int, rng: np.random.Generator) -> np.ndarray | None:
    choices = np.arange(1, width - 1, dtype=int)
    if choices.size < n:
        return None

    for _ in range(200):
        cols = np.sort(rng.choice(choices, size=n, replace=False))
        if np.all(np.diff(cols) >= 2):
            return cols.astype(int)
    return None


def _passes_semantic_checks(
    inp: np.ndarray,
    expected: np.ndarray,
    colors: np.ndarray,
    heights: np.ndarray,
    k1: int,
    k2: int,
) -> bool:
    n = len(colors)
    if np.array_equal(inp, expected):
        return False
    if np.array_equal(np.roll(colors, k1), colors):
        return False
    if np.array_equal(np.roll(heights, -k2), heights):
        return False
    if (k1 + k2) % n == 0:
        return False
    if k1 != 1 and np.array_equal(np.roll(colors, k1), np.roll(colors, 1)):
        return False
    if k2 != 1 and np.array_equal(np.roll(heights, -k2), np.roll(heights, -1)):
        return False
    try:
        actual = main(inp)
    except Exception:
        return False
    return bool(np.array_equal(actual, expected))


def _gen_attempt(mode: int) -> np.ndarray | None:
    rng = _RNG
    n = int(rng.integers(4, 8))
    k1, k2 = _choose_indicator_pair(n, mode, rng)

    min_height = max(n + 4, 9)
    height = int(rng.integers(min_height, 21))
    height_choices = np.arange(1, height - 1, dtype=int)
    if height_choices.size < n:
        return None
    heights = rng.choice(height_choices, size=n, replace=False).astype(int)

    min_width = max(2 * n + 3, k1 + k2 + 5, 12)
    max_width = min(30, min_width + 10)
    width = int(rng.integers(min_width, max_width + 1))
    cols = _sample_spaced_columns(width, n, rng)
    if cols is None:
        return None

    colors = rng.choice(BAR_COLORS, size=n, replace=False).astype(int)
    inp = _render(height, width, cols, colors, heights, k1, k2)
    expected = _expected_output(height, width, cols, colors, heights, k1, k2)
    if not _passes_semantic_checks(inp, expected, colors, heights, k1, k2):
        return None
    return inp


def generate_input() -> np.ndarray:
    global _GEN_CALL_COUNT
    mode = _GEN_CALL_COUNT % 4
    _GEN_CALL_COUNT += 1

    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt(mode)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid double-key rotation input")
