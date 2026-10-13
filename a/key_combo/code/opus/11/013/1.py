from __future__ import annotations

import random

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5
PINK = 6

STAMP_CYCLE = (BLUE, RED, GREEN, YELLOW, PINK)


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _extract_bars(grid: np.ndarray, bg: int) -> list[dict[str, int]]:
    fg_colors = {int(v) for v in np.unique(grid) if int(v) != bg}
    assert fg_colors == {GREY}, "expected only grey vertical bars on the background"

    bars: list[dict[str, int]] = []
    seen = np.zeros_like(grid, dtype=bool)
    for c in range(grid.shape[1]):
        rows = np.flatnonzero(grid[:, c] == GREY)
        if rows.size == 0:
            continue
        r0 = int(rows[0])
        r1 = int(rows[-1])
        height = int(rows.size)
        assert height == r1 - r0 + 1, "bar pixels must be vertically contiguous"
        assert np.all(grid[r0 : r1 + 1, c] == GREY)
        seen[r0 : r1 + 1, c] = True
        bars.append({"top": r0, "bottom": r1, "col": c, "height": height})

    assert len(bars) >= 1, "expected at least one bar"
    assert np.array_equal(seen, grid == GREY), "foreground must consist only of 1-pixel-wide bars"

    heights = [bar["height"] for bar in bars]
    assert len(set(heights)) == len(heights), "bar heights must be distinct"
    return bars


def main(grid: np.ndarray) -> np.ndarray:
    """Replace each grey vertical bar with a ranked stamp whose arm color follows a fixed cycle.

    Input structure
    - The grid has a black background and 4-7 separated grey 1-pixel-wide vertical bars.
    - Each bar is solid, all bar heights are distinct, and each top pixel acts as the future stamp center.

    Rule
    - Rank the grey bars globally by height from shortest to tallest.
    - Remove every original grey bar.
    - For rank k, choose the arm color from the cycle [blue, red, green, yellow, pink] using ((k-1) mod 5).
    - At the original top pixel, draw a 3x3 stamp with grey diagonals, the chosen color on the four orthogonal neighbors, and a black center; discard out-of-bounds stamp cells.

    Key insight
    - The local stamp color cannot be decided from a single bar or its position; it is determined only after computing the global height ranking and cycling through the palette.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    bg = _get_bg(g)
    bars = _extract_bars(g, bg)

    ranked = sorted(bars, key=lambda bar: bar["height"])
    out = np.full(g.shape, bg, dtype=int)
    h, w = g.shape

    for rank, bar in enumerate(ranked, start=1):
        color = STAMP_CYCLE[(rank - 1) % len(STAMP_CYCLE)]
        r = bar["top"]
        c = bar["col"]

        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                rr = r + dr
                cc = c + dc
                if not (0 <= rr < h and 0 <= cc < w):
                    continue
                if dr == 0 and dc == 0:
                    out[rr, cc] = bg
                elif abs(dr) == 1 and abs(dc) == 1:
                    out[rr, cc] = GREY
                else:
                    out[rr, cc] = color
    return out


# --------------- generator ---------------

HEIGHT_RANGE = tuple(range(2, 9))
GRID_HEIGHT_RANGE = (12, 20)
GRID_WIDTH_RANGE = (16, 26)
BAR_COUNT_CHOICES = (4, 5, 6, 7)
BAR_COUNT_WEIGHTS = (1, 1, 2, 2)
MIN_COLUMN_GAP = 3
GEN_RETRIES = 400


def _weighted_bar_count() -> int:
    return int(random.choices(BAR_COUNT_CHOICES, weights=BAR_COUNT_WEIGHTS, k=1)[0])


def _sample_columns(width: int, count: int) -> list[int]:
    base_span = count + (MIN_COLUMN_GAP - 1) * (count - 1)
    extra = width - base_span
    assert extra >= 0

    edge_mode = random.choices(
        ["none", "left", "right", "both"],
        weights=[4, 2, 2, 2],
        k=1,
    )[0]
    leading = 0 if edge_mode in ("left", "both") else None
    trailing = 0 if edge_mode in ("right", "both") else None

    slot_count = count + 1
    extras = np.random.multinomial(extra, [1.0 / slot_count] * slot_count).astype(int)
    if leading is not None:
        extras[0] = 0
    if trailing is not None:
        extras[-1] = 0
    remaining = extra - int(extras.sum())
    if remaining > 0:
        allowed = list(range(slot_count))
        if leading is not None:
            allowed.remove(0)
        if trailing is not None:
            allowed.remove(slot_count - 1)
        for _ in range(remaining):
            extras[random.choice(allowed)] += 1

    cols: list[int] = []
    col = int(extras[0])
    cols.append(col)
    for i in range(1, count):
        col += 1 + (MIN_COLUMN_GAP - 1) + int(extras[i])
        cols.append(col)
    assert cols[-1] <= width - 1 - int(extras[-1])
    assert all(cols[i + 1] - cols[i] >= MIN_COLUMN_GAP for i in range(count - 1))
    return cols


def _expected_output(spec: list[dict[str, int]], shape: tuple[int, int]) -> np.ndarray:
    out = np.zeros(shape, dtype=int)
    ranked = sorted(spec, key=lambda bar: bar["height"])
    h, w = shape
    for rank, bar in enumerate(ranked, start=1):
        color = STAMP_CYCLE[(rank - 1) % len(STAMP_CYCLE)]
        r = bar["top"]
        c = bar["col"]
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                rr = r + dr
                cc = c + dc
                if 0 <= rr < h and 0 <= cc < w:
                    if dr == 0 and dc == 0:
                        out[rr, cc] = BLACK
                    elif abs(dr) == 1 and abs(dc) == 1:
                        out[rr, cc] = GREY
                    else:
                        out[rr, cc] = color
    return out


def _gen_attempt() -> np.ndarray | None:
    height = random.randint(*GRID_HEIGHT_RANGE)
    width = random.randint(*GRID_WIDTH_RANGE)
    bar_count = _weighted_bar_count()
    if width < bar_count + (MIN_COLUMN_GAP - 1) * (bar_count - 1):
        return None

    heights = random.sample(HEIGHT_RANGE, bar_count)
    cols = _sample_columns(width, bar_count)

    grid = np.zeros((height, width), dtype=int)
    bars: list[dict[str, int]] = []
    for col, bar_height in zip(cols, heights):
        top = random.randint(1, height - bar_height)
        bottom = top + bar_height - 1
        grid[top : bottom + 1, col] = GREY
        bars.append({"top": top, "bottom": bottom, "col": col, "height": bar_height})

    if np.count_nonzero(grid == GREY) != sum(heights):
        return None

    expected = _expected_output(bars, grid.shape)
    try:
        actual = main(grid)
    except Exception:
        return None

    if not np.array_equal(actual, expected):
        return None

    output_colors = {int(v) for v in np.unique(actual) if int(v) != BLACK}
    if len(output_colors) < 3:
        return None

    if bar_count >= 6 and len(output_colors & {BLUE, RED, GREEN, YELLOW, PINK}) < 5:
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid ranked-stamp input")
