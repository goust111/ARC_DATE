from __future__ import annotations

import hashlib
import random

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREY = 5


GEN_ROWS_RANGE = (14, 20)
GEN_COLS_RANGE = (17, 24)
GEN_BAR_COUNT_RANGE = (5, 7)
GEN_MIN_HEIGHT = 5
GEN_MIN_BAR_GAP = 2
GEN_RETRY_LIMIT = 800

_GEN_CALL_COUNT = 0
_SEEN_OUTPUT_SIGNATURES: set[str] = set()


def _background_color(grid: np.ndarray) -> int:
    values = np.asarray(grid, dtype=int).ravel()
    return int(np.bincount(values).argmax())


def _grid_signature(grid: np.ndarray) -> str:
    return hashlib.md5(grid.tobytes() + str(tuple(grid.shape)).encode()).hexdigest()


def _extract_bars(grid: np.ndarray) -> list[dict[str, int]]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    bg = _background_color(g)
    colors = sorted(set(map(int, g.ravel())) - {bg})
    assert len(colors) == 1, "expected exactly one foreground color"
    fg = colors[0]

    mask = g == fg
    cols = np.flatnonzero(mask.any(axis=0))
    assert cols.size >= 2, "expected multiple bars"

    bars: list[dict[str, int]] = []
    for c in cols:
        if c > 0:
            assert not mask[:, c - 1].any(), "bars must be separated by at least one empty column"
        if c + 1 < g.shape[1]:
            assert not mask[:, c + 1].any(), "bars must be separated by at least one empty column"

        rows = np.flatnonzero(mask[:, c])
        assert rows.size > 0, "occupied column must contain one bar"
        r0, r1 = int(rows[0]), int(rows[-1])
        assert int(rows.size) == r1 - r0 + 1, "each bar must be vertically solid"
        assert np.all(g[r0 : r1 + 1, c] == fg), "bar cells must be monochromatic"
        assert np.count_nonzero(mask[:, c]) == rows.size, "no holes allowed inside a bar"
        bars.append({"r0": r0, "r1": r1, "c": int(c), "height": int(rows.size)})

    bars.sort(key=lambda bar: bar["c"])
    return bars


def _extreme_bars(bars: list[dict[str, int]]) -> tuple[dict[str, int], dict[str, int]]:
    heights = [bar["height"] for bar in bars]
    tallest_height = max(heights)
    shortest_height = min(heights)
    tallest = [bar for bar in bars if bar["height"] == tallest_height]
    shortest = [bar for bar in bars if bar["height"] == shortest_height]
    assert len(tallest) == 1 and len(shortest) == 1, "tallest and shortest bars must be unique"
    return tallest[0], shortest[0]


def _kept_rows(bar: dict[str, int], *, start_from_top: bool) -> list[int]:
    r0, r1 = bar["r0"], bar["r1"]
    if start_from_top:
        return [r for r in range(r0, r1 + 1) if (r - r0) % 2 == 0]
    return [r for r in range(r0, r1 + 1) if (r1 - r) % 2 == 0]


def _render_output(
    shape: tuple[int, int],
    bg: int,
    tallest: dict[str, int],
    shortest: dict[str, int],
    *,
    tallest_from_top: bool,
    shortest_from_top: bool,
) -> np.ndarray:
    out = np.full(shape, bg, dtype=int)
    for row in _kept_rows(tallest, start_from_top=tallest_from_top):
        out[row, tallest["c"]] = BLUE
    for row in _kept_rows(shortest, start_from_top=shortest_from_top):
        out[row, shortest["c"]] = RED
    return out


def _oracle_output(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    bg = _background_color(g)
    bars = _extract_bars(g)
    tallest, shortest = _extreme_bars(bars)
    return _render_output(
        g.shape,
        bg,
        tallest,
        shortest,
        tallest_from_top=True,
        shortest_from_top=False,
    )


def main(grid: np.ndarray) -> np.ndarray:
    """
    Keep only the tallest and shortest vertical bars, recolor them blue/red, and punch opposite-phase holes through them.

    Input structure:
    - A black background contains 5-7 separated one-cell-wide vertical bars in a single foreground color.
    - Each bar is a solid vertical segment in its own column, with at least one empty column between neighboring bars.
    - Bar heights are distinct, and the unique tallest and shortest bars remain identifiable even when bar tops and bottoms are not aligned.

    Rule:
    - Rank the input bars by height and erase every bar except the unique tallest and unique shortest.
    - Recolor the tallest surviving bar blue and keep every other pixel starting from its top cell.
    - Recolor the shortest surviving bar red and keep every other pixel starting from its bottom cell.
    - Leave all other cells black.

    Key insight:
    - Height ranking does two jobs at once: it decides which bars survive, and it also decides the punching phase, with the tallest bar anchored from the top and the shortest bar anchored from the bottom.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    bg = _background_color(g)
    bars = _extract_bars(g)
    tallest, shortest = _extreme_bars(bars)
    return _render_output(
        g.shape,
        bg,
        tallest,
        shortest,
        tallest_from_top=True,
        shortest_from_top=False,
    )


def _choose_columns(width: int, count: int) -> list[int] | None:
    candidates = list(range(1, width - 1))
    for _ in range(200):
        cols = sorted(random.sample(candidates, count))
        if all(b - a >= GEN_MIN_BAR_GAP for a, b in zip(cols, cols[1:])):
            return cols
    return None


def _choose_heights(rows: int, count: int) -> tuple[int, list[int], int] | None:
    max_height = rows - 2
    if max_height < GEN_MIN_HEIGHT + count - 1:
        return None

    shortest_choices = [
        h for h in range(GEN_MIN_HEIGHT + 1, max_height - (count - 2) + 1) if h % 2 == 0
    ]
    random.shuffle(shortest_choices)
    for shortest in shortest_choices:
        tallest_choices = [h for h in range(shortest + count - 1, max_height + 1) if h % 2 == 0]
        random.shuffle(tallest_choices)
        for tallest in tallest_choices:
            middle_pool = list(range(shortest + 1, tallest))
            if len(middle_pool) < count - 2:
                continue
            middles = random.sample(middle_pool, count - 2)
            return shortest, middles, tallest
    return None


def _extreme_indices(count: int, mode: int) -> tuple[int, int]:
    interior = list(range(1, count - 1))
    all_indices = list(range(count))

    if mode == 0:
        tallest_idx = random.randrange(0, count - 1)
        shortest_idx = random.randrange(tallest_idx + 1, count)
        return tallest_idx, shortest_idx
    if mode == 1:
        shortest_idx = random.randrange(0, count - 1)
        tallest_idx = random.randrange(shortest_idx + 1, count)
        return tallest_idx, shortest_idx
    if mode == 2:
        tallest_idx = random.choice(interior)
        shortest_idx = random.choice([idx for idx in all_indices if idx != tallest_idx])
        return tallest_idx, shortest_idx

    shortest_idx = random.choice(interior)
    tallest_idx = random.choice([idx for idx in all_indices if idx != shortest_idx])
    return tallest_idx, shortest_idx


def _choose_tops(rows: int, heights: list[int]) -> list[int] | None:
    tallest_idx = int(np.argmax(heights))
    shortest_idx = int(np.argmin(heights))

    for _ in range(400):
        tops = [random.randint(0, rows - height) for height in heights]
        bottoms = [top + height - 1 for top, height in zip(tops, heights)]

        if len(set(tops)) < 2 or len(set(bottoms)) < 2:
            continue
        if tops[tallest_idx] <= min(tops):
            continue
        if bottoms[shortest_idx] >= max(bottoms):
            continue
        return tops
    return None


def _build_candidate(mode: int) -> np.ndarray | None:
    rows = random.randint(*GEN_ROWS_RANGE)
    width = random.randint(*GEN_COLS_RANGE)
    count = random.randint(*GEN_BAR_COUNT_RANGE)

    cols = _choose_columns(width, count)
    if cols is None:
        return None

    height_parts = _choose_heights(rows, count)
    if height_parts is None:
        return None
    shortest, middles, tallest = height_parts
    tallest_idx, shortest_idx = _extreme_indices(count, mode)

    heights = [0] * count
    heights[tallest_idx] = tallest
    heights[shortest_idx] = shortest
    remaining_indices = [idx for idx in range(count) if idx not in (tallest_idx, shortest_idx)]
    random.shuffle(middles)
    for idx, height in zip(remaining_indices, middles):
        heights[idx] = height

    tops = _choose_tops(rows, heights)
    if tops is None:
        return None

    grid = np.full((rows, width), BLACK, dtype=int)
    for c, height, top in zip(cols, heights, tops):
        grid[top : top + height, c] = GREY
    return grid


def _validate_candidate(grid: np.ndarray) -> bool:
    try:
        bars = _extract_bars(grid)
        if not (GEN_BAR_COUNT_RANGE[0] <= len(bars) <= GEN_BAR_COUNT_RANGE[1]):
            return False

        heights = [bar["height"] for bar in bars]
        if min(heights) < GEN_MIN_HEIGHT or len(set(heights)) != len(heights):
            return False
        if any(height % 2 == 1 for height in (min(heights), max(heights))):
            return False

        cols = [bar["c"] for bar in bars]
        if any(b - a < GEN_MIN_BAR_GAP for a, b in zip(cols, cols[1:])):
            return False

        tops = [bar["r0"] for bar in bars]
        bottoms = [bar["r1"] for bar in bars]
        tallest_idx = int(np.argmax(heights))
        shortest_idx = int(np.argmin(heights))
        if len(set(tops)) < 2 or len(set(bottoms)) < 2:
            return False
        if tops[tallest_idx] <= min(tops):
            return False
        if bottoms[shortest_idx] >= max(bottoms):
            return False

        tallest, shortest = _extreme_bars(bars)
        oracle = _oracle_output(grid)
        if not np.array_equal(main(grid), oracle):
            return False

        wrong_shortest_from_top = _render_output(
            grid.shape,
            BLACK,
            tallest,
            shortest,
            tallest_from_top=True,
            shortest_from_top=True,
        )
        wrong_tallest_from_bottom = _render_output(
            grid.shape,
            BLACK,
            tallest,
            shortest,
            tallest_from_top=False,
            shortest_from_top=False,
        )
        if np.array_equal(oracle, wrong_shortest_from_top):
            return False
        if np.array_equal(oracle, wrong_tallest_from_bottom):
            return False

        plain_extremes = np.full(grid.shape, BLACK, dtype=int)
        plain_extremes[tallest["r0"] : tallest["r1"] + 1, tallest["c"]] = BLUE
        plain_extremes[shortest["r0"] : shortest["r1"] + 1, shortest["c"]] = RED
        if np.array_equal(oracle, plain_extremes):
            return False

        out_sig = _grid_signature(oracle)
        if out_sig in _SEEN_OUTPUT_SIGNATURES:
            return False
        return True
    except AssertionError:
        return False


def generate_input() -> np.ndarray:
    global _GEN_CALL_COUNT

    for _ in range(GEN_RETRY_LIMIT):
        mode = _GEN_CALL_COUNT % 4
        grid = _build_candidate(mode)
        if grid is None:
            continue
        if not _validate_candidate(grid):
            continue

        _SEEN_OUTPUT_SIGNATURES.add(_grid_signature(_oracle_output(grid)))
        _GEN_CALL_COUNT += 1
        return grid

    raise RuntimeError("failed to generate a valid non-degenerate input")
