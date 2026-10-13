from __future__ import annotations

import itertools

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
GREY = 5

BAR_COUNT = 4
GEN_ATTEMPTS = 500
HEIGHT_RANGE = (13, 18)
WIDTH_RANGE = (11, 15)
BAR_HEIGHTS = np.arange(3, 9)
MARKER_COUNT_RANGE = range(2, 11)

_RNG = np.random.default_rng()


def _neighbors8(r: int, c: int):
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            yield r + dr, c + dc


def _all_nonadjacent_column_sets(width: int, count: int) -> list[tuple[int, ...]]:
    return [
        cols
        for cols in itertools.combinations(range(width), count)
        if all(b - a >= 2 for a, b in zip(cols, cols[1:]))
    ]


def _extract_bars(grid: np.ndarray) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    bars: list[dict] = []
    for c in range(g.shape[1]):
        rows = np.flatnonzero(g[:, c] == RED)
        if rows.size == 0:
            continue
        assert np.all(np.diff(rows) == 1), "each red bar must be one contiguous vertical segment"
        bars.append(
            {
                "col": int(c),
                "top": int(rows[0]),
                "bottom": int(rows[-1]),
                "height": int(rows.size),
            }
        )
    assert bars, "expected at least one red bar"
    return bars


def _render_compact_bars(ordered_bars: list[dict]) -> np.ndarray:
    max_height = max(bar["height"] for bar in ordered_bars)
    out = np.full((max_height, len(ordered_bars)), BLACK, dtype=int)

    for out_c, bar in enumerate(ordered_bars):
        height = int(bar["height"])
        keep_red = (height + 1) // 2
        recolor_blue = height // 2
        start_r = max_height - height
        out[start_r : start_r + keep_red, out_c] = RED
        if recolor_blue:
            out[start_r + keep_red : start_r + keep_red + recolor_blue, out_c] = BLUE
    return out


def _analyze_input(grid: np.ndarray) -> dict:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    assert max(g.shape) <= 30, "ARC grids must stay within 30x30"

    colors = set(map(int, np.unique(g)))
    assert colors <= {BLACK, RED, GREEN, GREY}, "unexpected input colors"

    grey_rows = [r for r in range(g.shape[0]) if np.all(g[r, :] == GREY)]
    assert grey_rows == [g.shape[0] - 2], "expected one full-width grey reference line on the second-last row"

    bars = sorted(_extract_bars(g), key=lambda bar: bar["col"])
    assert 4 <= len(bars) <= 6, "expected 4-6 red bars"
    assert len({bar["height"] for bar in bars}) == len(bars), "bar heights must be distinct"
    assert all(bars[i + 1]["col"] - bars[i]["col"] >= 2 for i in range(len(bars) - 1)), "bar columns must be separated"
    assert all(bar["bottom"] < grey_rows[0] for bar in bars), "bars must stay above the grey reference line"

    markers = [tuple(map(int, rc)) for rc in np.argwhere(g == GREEN)]
    assert markers, "expected at least one green marker"
    bar_cols = {bar["col"] for bar in bars}
    for r, c in markers:
        assert c not in bar_cols, "green markers cannot share a column with a bar"
        for nr, nc in _neighbors8(r, c):
            if 0 <= nr < g.shape[0] and 0 <= nc < g.shape[1]:
                if (nr, nc) != (r, c):
                    assert int(g[nr, nc]) == BLACK, "green markers must be isolated single pixels"

    return {
        "bars": bars,
        "marker_count": len(markers),
        "parity_even": len(markers) % 2 == 0,
    }


def _solve_from_analysis(info: dict) -> np.ndarray:
    ordered = sorted(info["bars"], key=lambda bar: bar["height"], reverse=not info["parity_even"])
    return _render_compact_bars(ordered)


def main(grid: np.ndarray) -> np.ndarray:
    """Count green markers to choose bar-sort direction, then half-recolor and compress the red bars.

    Input structure:
    - The grid has a black background, 4-6 one-column red vertical bars with distinct heights, and isolated green singleton markers.
    - A full-width grey horizontal reference line sits on the second-last row; it is only a visual divider.
    - Green markers never share a column with a red bar, and each red bar is a single contiguous segment entirely above the grey line.

    Rule:
    - Count all green markers; if that count is even, sort the red bars by height ascending, otherwise descending.
    - For each red bar, keep its upper ceil(height/2) cells red and recolor its lower floor(height/2) cells blue.
    - Build a new output whose width is the number of bars and whose height is the tallest bar, placing the sorted bars left to right with bottom alignment.
    - Omit the green markers, grey line, and all original input positions from the output.

    Key insight:
    - The green pixels matter only through their count parity, which secretly flips the sort direction before the per-bar half-recolor and final compression.
    """
    info = _analyze_input(np.asarray(grid, dtype=int))
    return _solve_from_analysis(info)


def _candidate_marker_cells(grid: np.ndarray, bar_cols: set[int], grey_row: int) -> list[tuple[int, int]]:
    g = np.asarray(grid, dtype=int)
    cells: list[tuple[int, int]] = []
    for r in range(g.shape[0]):
        for c in range(g.shape[1]):
            if r >= grey_row - 1:
                continue
            if c in bar_cols or int(g[r, c]) != BLACK:
                continue
            isolated = True
            for nr, nc in _neighbors8(r, c):
                if 0 <= nr < g.shape[0] and 0 <= nc < g.shape[1] and int(g[nr, nc]) != BLACK:
                    isolated = False
                    break
            if isolated:
                cells.append((r, c))
    _RNG.shuffle(cells)
    return cells


def _place_markers(
    grid: np.ndarray,
    bar_cols: set[int],
    grey_row: int,
    marker_count: int,
) -> bool:
    candidates = _candidate_marker_cells(grid, bar_cols, grey_row)
    chosen: list[tuple[int, int]] = []
    for r, c in candidates:
        if all(max(abs(r - rr), abs(c - cc)) >= 2 for rr, cc in chosen):
            chosen.append((r, c))
            if len(chosen) == marker_count:
                for rr, cc in chosen:
                    grid[rr, cc] = GREEN
                return True
    return False


def _alternative_output(bars: list[dict], mode: str, even: bool) -> np.ndarray:
    if mode == "left_to_right":
        ordered = list(bars)
    elif mode == "top":
        ordered = sorted(bars, key=lambda bar: bar["top"], reverse=not even)
    elif mode == "bottom":
        ordered = sorted(bars, key=lambda bar: bar["bottom"], reverse=not even)
    else:
        raise ValueError(mode)
    return _render_compact_bars(ordered)


def _valid_counterexample_structure(grid: np.ndarray) -> bool:
    info = _analyze_input(grid)
    bars = info["bars"]
    even = bool(info["parity_even"])
    intended = _solve_from_analysis(info)

    heights_by_col = [bar["height"] for bar in bars]
    if heights_by_col == sorted(heights_by_col) or heights_by_col == sorted(heights_by_col, reverse=True):
        return False

    if np.array_equal(intended, _solve_from_analysis({"bars": bars, "parity_even": not even})):
        return False

    for mode in ("left_to_right", "top", "bottom"):
        if np.array_equal(intended, _alternative_output(bars, mode, even)):
            return False

    return True


def _gen_attempt() -> np.ndarray | None:
    height = int(_RNG.integers(HEIGHT_RANGE[0], HEIGHT_RANGE[1] + 1))
    width = int(_RNG.integers(WIDTH_RANGE[0], WIDTH_RANGE[1] + 1))
    column_sets = _all_nonadjacent_column_sets(width, BAR_COUNT)
    if not column_sets:
        return None

    grid = np.full((height, width), BLACK, dtype=int)
    grey_row = height - 2
    grid[grey_row, :] = GREY

    bar_cols = list(column_sets[int(_RNG.integers(len(column_sets)))])
    heights = list(map(int, _RNG.choice(BAR_HEIGHTS, size=BAR_COUNT, replace=False)))
    _RNG.shuffle(heights)

    tops: list[int] = []
    bottoms: list[int] = []
    for col, bar_height in zip(bar_cols, heights):
        bottom = int(_RNG.integers(bar_height - 1, grey_row))
        top = bottom - bar_height + 1
        tops.append(top)
        bottoms.append(bottom)
        grid[top : bottom + 1, col] = RED

    if len(set(tops)) == 1 or len(set(bottoms)) == 1:
        return None

    want_even = bool(_RNG.integers(0, 2) == 0)
    parity_counts = [n for n in MARKER_COUNT_RANGE if (n % 2 == 0) == want_even]
    _RNG.shuffle(parity_counts)
    placed = False
    for marker_count in parity_counts:
        trial = grid.copy()
        if _place_markers(trial, set(bar_cols), grey_row, int(marker_count)):
            grid = trial
            placed = True
            break
    if not placed:
        return None

    if not _valid_counterexample_structure(grid):
        return None

    expected = _solve_from_analysis(_analyze_input(grid))
    if not np.array_equal(main(grid), expected):
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 8")
