import numpy as np


BG_COLOR = 0


def _extract_bars(grid: np.ndarray) -> list[dict[str, int]]:
    """Return bars in input row order."""
    bars: list[dict[str, int]] = []
    height, width = map(int, grid.shape)
    assert height > 0 and width > 1

    for row in range(height):
        cols = np.flatnonzero(grid[row] != BG_COLOR)
        if cols.size == 0:
            continue

        length = int(cols.size)
        assert np.array_equal(cols, np.arange(length)), "each bar must be left-aligned"
        color = int(grid[row, 0])
        assert color != BG_COLOR
        assert np.all(grid[row, :length] == color), "each bar must be monochrome"
        assert np.all(grid[row, length:] == BG_COLOR), "bar row must contain only one segment"

        bars.append({"row": row, "color": color, "length": length})

    assert bars, "expected at least one horizontal bar"
    return bars


def _render_sorted_bars(shape: tuple[int, int], bars: list[dict[str, int]]) -> np.ndarray:
    height, width = map(int, shape)
    out = np.full((height, width), BG_COLOR, dtype=int)

    # Python's sort is stable, so equal-length bars keep their input order.
    ordered = sorted(bars, key=lambda bar: bar["length"])
    assert len(ordered) <= height

    for out_row, bar in enumerate(ordered):
        length = int(bar["length"])
        color = int(bar["color"])
        assert 1 <= length < width + 1
        out[out_row, :length] = color

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """Sort left-aligned horizontal bars by increasing length.

    Input structure
    - The background is black.
    - Each non-empty row contains one horizontal one-pixel-thick bar.
    - Every bar is left-aligned and monochrome.

    Rule
    - Ignore bar colors and original row positions for ranking.
    - Sort the bars from shortest to longest by length.
    - Redraw them on the same-size canvas from the top row downward.
    - If two bars have the same length, keep their input order.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    bars = _extract_bars(g)
    return _render_sorted_bars(g.shape, bars)


# --------------- generator ---------------

_RNG = np.random.default_rng()

HEIGHT_RANGE = (4, 12)
WIDTH_RANGE = (5, 16)
BAR_COUNT_RANGE = (2, 6)
COLOR_POOL = np.arange(1, 10)
TIE_PROBABILITY = 0.5
PACKED_ROWS_PROBABILITY = 0.35
MAX_GENERATION_ATTEMPTS = 300


def _sample_rows(height: int, count: int) -> list[int]:
    if _RNG.random() < PACKED_ROWS_PROBABILITY:
        start = int(_RNG.integers(0, height - count + 1))
        return list(range(start, start + count))
    return sorted(map(int, _RNG.choice(height, size=count, replace=False)))


def _sample_lengths(count: int, max_len: int, allow_ties: bool) -> list[int]:
    choices = np.arange(1, max_len + 1)
    if not allow_ties:
        return list(map(int, _RNG.choice(choices, size=count, replace=False)))

    if count == 2:
        length = int(_RNG.choice(choices))
        return [length, length]

    base = list(map(int, _RNG.choice(choices, size=count - 1, replace=False)))
    duplicate = int(_RNG.choice(base))
    lengths = base + [duplicate]
    _RNG.shuffle(lengths)
    assert len(set(lengths)) < len(lengths)
    return lengths


def _draw_input(shape: tuple[int, int], rows: list[int], colors: list[int], lengths: list[int]) -> np.ndarray:
    grid = np.full(shape, BG_COLOR, dtype=int)
    for row, color, length in zip(rows, colors, lengths):
        grid[row, :length] = color
    return grid


def _expected_output(shape: tuple[int, int], rows: list[int], colors: list[int], lengths: list[int]) -> np.ndarray:
    bars = [
        {"row": row, "color": color, "length": length}
        for row, color, length in zip(rows, colors, lengths)
    ]
    bars.sort(key=lambda bar: bar["row"])
    return _render_sorted_bars(shape, bars)


def _gen_attempt() -> np.ndarray | None:
    height = int(_RNG.integers(HEIGHT_RANGE[0], HEIGHT_RANGE[1] + 1))
    width = int(_RNG.integers(WIDTH_RANGE[0], WIDTH_RANGE[1] + 1))
    max_count = min(BAR_COUNT_RANGE[1], height, width - 1, len(COLOR_POOL))
    if max_count < BAR_COUNT_RANGE[0]:
        return None

    count = int(_RNG.integers(BAR_COUNT_RANGE[0], max_count + 1))
    allow_ties = bool(_RNG.random() < TIE_PROBABILITY)

    rows = _sample_rows(height, count)
    colors = list(map(int, _RNG.choice(COLOR_POOL, size=count, replace=False)))
    lengths = _sample_lengths(count, width - 1, allow_ties)

    inp = _draw_input((height, width), rows, colors, lengths)
    expected = _expected_output((height, width), rows, colors, lengths)

    if np.array_equal(inp, expected):
        return None

    try:
        actual = main(inp)
    except Exception:
        return None

    if not np.array_equal(actual, expected):
        return None

    has_tie = len(set(lengths)) < len(lengths)
    if has_tie != allow_ties:
        return None

    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
