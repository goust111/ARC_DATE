from __future__ import annotations

import random

import numpy as np


BG = 0
BLUE = 1
RED = 2
GREEN = 3
GREY = 5

GEN_ATTEMPTS = 400


def _runs_of_color(row: np.ndarray, color: int) -> list[tuple[int, int]]:
    runs: list[tuple[int, int]] = []
    start: int | None = None
    for c, value in enumerate(row):
        if int(value) == int(color):
            if start is None:
                start = c
        elif start is not None:
            runs.append((int(start), int(c - 1)))
            start = None
    if start is not None:
        runs.append((int(start), int(len(row) - 1)))
    return runs


def _find_barrier(grid: np.ndarray) -> tuple[int, int, int]:
    best: tuple[int, int, int] | None = None
    best_len = -1
    for r in range(grid.shape[0]):
        for c0, c1 in _runs_of_color(grid[r], GREY):
            run_len = int(c1 - c0 + 1)
            if run_len > best_len:
                best_len = run_len
                best = (int(r), int(c0), int(c1))
    assert best is not None and best_len >= max(5, grid.shape[1] // 2)
    return best


def _find_reference(grid: np.ndarray, barrier_row: int) -> tuple[int, int]:
    refs: list[tuple[int, int]] = []
    for r in range(barrier_row):
        for c in range(grid.shape[1] - 1):
            if int(grid[r, c]) == BLUE and int(grid[r, c + 1]) == RED:
                refs.append((int(r), int(c)))
    assert refs, "expected a top-row blue-red reference bar"
    refs.sort()
    return refs[0]


def _read_two_color_bar(row: np.ndarray) -> tuple[int, tuple[int, int]] | None:
    cols = [int(c) for c in np.flatnonzero(row != BG)]
    if len(cols) != 2 or cols[1] != cols[0] + 1:
        return None
    left = cols[0]
    colors = (int(row[left]), int(row[left + 1]))
    if set(colors) != {BLUE, RED}:
        return None
    return int(left), colors


def _find_upper_bars(
    grid: np.ndarray,
    reference_row: int,
    barrier_row: int,
    barrier_span: tuple[int, int],
) -> list[dict]:
    c0, c1 = barrier_span
    bars: list[dict] = []
    for r in range(reference_row + 1, barrier_row):
        found = _read_two_color_bar(grid[r])
        if found is None:
            continue
        left, colors = found
        assert c0 <= left and left + 1 <= c1
        bars.append({"row": int(r), "left": int(left), "colors": colors})
    assert bars, "expected at least one upper two-color bar"
    return bars


def _find_lower_markers(
    grid: np.ndarray,
    barrier_row: int,
    barrier_span: tuple[int, int],
) -> list[tuple[int, int]]:
    c0, c1 = barrier_span
    markers: list[tuple[int, int]] = []
    for r in range(barrier_row + 1, grid.shape[0]):
        cols = [int(c) for c in np.flatnonzero(grid[r] == GREEN)]
        if not cols:
            continue
        assert len(cols) == 1 and np.count_nonzero(grid[r] != BG) == 1
        col = cols[0]
        assert c0 <= col <= c1
        markers.append((int(r), int(col)))
    assert markers, "expected at least one lower marker"
    return markers


def _pair_bars_and_markers(bars: list[dict], markers: list[tuple[int, int]]) -> list[dict]:
    pairs: list[dict] = []
    unused = list(markers)
    for bar in bars:
        left = int(bar["left"])
        candidates = [marker for marker in unused if marker[1] in (left, left + 1)]
        assert len(candidates) == 1, "each bar should match one marker in its column range"
        marker = candidates[0]
        unused.remove(marker)
        pairs.append(
            {
                "bar_row": int(bar["row"]),
                "bar_left": int(left),
                "colors": tuple(map(int, bar["colors"])),
                "marker_row": int(marker[0]),
                "marker_col": int(marker[1]),
            }
        )
    assert not unused and len(pairs) == len(bars)
    return pairs


def _paint_bar(grid: np.ndarray, row: int, left: int, colors: tuple[int, int]) -> None:
    grid[int(row), int(left)] = int(colors[0])
    grid[int(row), int(left + 1)] = int(colors[1])


def _spec_output(
    shape: tuple[int, int],
    barrier_row: int,
    barrier_left: int,
    barrier_right: int,
    reference_row: int,
    reference_left: int,
    pairs: list[dict],
    *,
    align: bool = True,
) -> np.ndarray:
    out = np.full(shape, BG, dtype=int)
    out[barrier_row, barrier_left : barrier_right + 1] = GREY
    _paint_bar(out, reference_row, reference_left, (BLUE, RED))
    ref_cols = {BLUE: int(reference_left), RED: int(reference_left + 1)}

    for pair in pairs:
        out[barrier_row - 1, int(pair["marker_col"])] = GREEN

    for pair in pairs:
        colors = tuple(map(int, pair["colors"]))
        left = int(pair["bar_left"])
        target_left = left if (not align or colors[0] == BLUE) else ref_cols[colors[0]]
        _paint_bar(out, barrier_row + 1, target_left, colors)

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Swap each cross-barrier two-color bar with its marker, then align only the swapped-down reversed bar to the top reference columns.

    Input structure:
    - Black background with one long gray horizontal barrier segment across the middle.
    - Above the barrier, the topmost blue-red adjacent pair is the fixed reference bar.
    - Between the reference row and the barrier, several rows each contain one horizontal blue/red two-pixel bar.
    - Below the barrier, the same number of rows each contain one green marker pixel whose column lies inside exactly one upper bar's two-column span.

    Rule:
    - Pair every upper two-color bar with the unique lower marker whose column is one of that bar's two columns.
    - Swap sides across the barrier: erase the original bar and marker, move the marker to the row just above the barrier at its original column, and move the bar to the row just below the barrier at its original horizontal position.
    - After the swap, inspect each moved-down bar. If its left pixel has the same color as the reference bar's left pixel, keep it in place. Otherwise shift the whole bar horizontally so that its left pixel lands in the reference column of that same color.
    - Keep the barrier and the reference bar unchanged; every vacated cell becomes background.

    Key insight:
    - The upper bars are not aligned while they remain above the barrier; alignment activates only after the swap creates a new bar directly governed by the reference row.
    """
    g = np.asarray(grid, dtype=int)
    barrier_row, barrier_left, barrier_right = _find_barrier(g)
    reference_row, reference_left = _find_reference(g, barrier_row)
    bars = _find_upper_bars(g, reference_row, barrier_row, (barrier_left, barrier_right))
    markers = _find_lower_markers(g, barrier_row, (barrier_left, barrier_right))
    pairs = _pair_bars_and_markers(bars, markers)

    out = g.copy()
    for pair in pairs:
        out[int(pair["bar_row"]), int(pair["bar_left"])] = BG
        out[int(pair["bar_row"]), int(pair["bar_left"] + 1)] = BG
        out[int(pair["marker_row"]), int(pair["marker_col"])] = BG

    ref_cols = {BLUE: int(reference_left), RED: int(reference_left + 1)}
    ref_left_color = int(g[reference_row, reference_left])

    for pair in pairs:
        out[barrier_row - 1, int(pair["marker_col"])] = GREEN

    for pair in pairs:
        colors = tuple(map(int, pair["colors"]))
        left = int(pair["bar_left"])
        target_left = left if colors[0] == ref_left_color else ref_cols[colors[0]]
        _paint_bar(out, barrier_row + 1, target_left, colors)

    return out


def _well_spaced(lefts: list[int], min_gap: int = 3) -> bool:
    ordered = sorted(lefts)
    return all(int(b - a) >= int(min_gap) for a, b in zip(ordered, ordered[1:]))


def _sample_left_layout(
    width: int,
    barrier_left: int,
    barrier_right: int,
    count: int,
) -> tuple[int, list[int], int] | None:
    candidates = list(range(barrier_left, barrier_right))
    for _ in range(200):
        ref_left = random.randint(barrier_left + 1, barrier_right - 2)
        random.shuffle(candidates)
        chosen: list[int] = []
        for left in candidates:
            if all(abs(int(left - prev)) >= 3 for prev in chosen):
                chosen.append(int(left))
            if len(chosen) == count:
                break
        if len(chosen) != count:
            continue

        reversed_target = int(ref_left + 1)
        valid_reversed = [
            int(left)
            for left in chosen
            if int(left) != reversed_target
            and all(abs(int(other - reversed_target)) >= 3 for other in chosen if other != left)
        ]
        if not valid_reversed:
            continue
        reversed_left = random.choice(valid_reversed)
        return int(ref_left), sorted(chosen), int(reversed_left)
    return None


def _build_input_from_spec(
    height: int,
    width: int,
    barrier_row: int,
    barrier_left: int,
    barrier_right: int,
    reference_row: int,
    reference_left: int,
    pairs: list[dict],
) -> np.ndarray:
    grid = np.full((height, width), BG, dtype=int)
    grid[barrier_row, barrier_left : barrier_right + 1] = GREY
    _paint_bar(grid, reference_row, reference_left, (BLUE, RED))
    for pair in pairs:
        _paint_bar(grid, int(pair["bar_row"]), int(pair["bar_left"]), tuple(map(int, pair["colors"])))
        grid[int(pair["marker_row"]), int(pair["marker_col"])] = GREEN
    return grid


def _semantic_checks(
    inp: np.ndarray,
    out: np.ndarray,
    barrier_row: int,
    reference_left: int,
    pairs: list[dict],
) -> bool:
    if np.array_equal(inp, out):
        return False

    if np.count_nonzero(inp[barrier_row - 1] != BG) != 0:
        return False
    if np.count_nonzero(inp[barrier_row + 1] != BG) != 0:
        return False

    reversed_pairs = [pair for pair in pairs if tuple(pair["colors"]) == (RED, BLUE)]
    same_pairs = [pair for pair in pairs if tuple(pair["colors"]) == (BLUE, RED)]
    if len(reversed_pairs) != 1 or not same_pairs:
        return False

    reversed_pair = reversed_pairs[0]
    if int(reversed_pair["bar_left"]) == int(reference_left + 1):
        return False

    final_lefts = [
        int(reference_left + 1) if tuple(pair["colors"]) == (RED, BLUE) else int(pair["bar_left"])
        for pair in pairs
    ]
    if not _well_spaced(final_lefts):
        return False

    swap_only = _spec_output(
        inp.shape,
        barrier_row,
        int(np.flatnonzero(inp[barrier_row] == GREY)[0]),
        int(np.flatnonzero(inp[barrier_row] == GREY)[-1]),
        int(np.flatnonzero((inp == BLUE).any(axis=1))[0]),
        reference_left,
        pairs,
        align=False,
    )
    if np.array_equal(swap_only, out):
        return False

    marker_row = barrier_row - 1
    if np.count_nonzero(out[marker_row] == GREEN) != len(pairs):
        return False

    bar_row = barrier_row + 1
    if np.count_nonzero(np.isin(out[bar_row], [BLUE, RED])) != 2 * len(pairs):
        return False

    return True


def _gen_attempt() -> np.ndarray | None:
    height = random.randint(12, 16)
    width = random.randint(12, 18)

    barrier_row = random.randint(max(5, height // 2 - 1), min(height - 4, height // 2 + 1))
    barrier_left = random.randint(0, 1)
    barrier_right = width - 1 - random.randint(0, 1)
    if barrier_right - barrier_left + 1 < 8:
        return None

    ref_row_max = min(3, barrier_row - 4)
    if ref_row_max < 1:
        return None
    reference_row = random.randint(1, ref_row_max)

    upper_rows = list(range(reference_row + 1, barrier_row - 1))
    lower_rows = list(range(barrier_row + 2, height))
    max_pairs = min(4, len(upper_rows), len(lower_rows), max(0, (barrier_right - barrier_left + 3) // 3))
    if max_pairs < 2:
        return None
    count = random.randint(2, max_pairs)

    layout = _sample_left_layout(width, barrier_left, barrier_right, count)
    if layout is None:
        return None
    reference_left, bar_lefts, reversed_left = layout

    upper_pick = sorted(random.sample(upper_rows, count))
    lower_pick = sorted(random.sample(lower_rows, count))
    random.shuffle(bar_lefts)

    pairs: list[dict] = []
    for bar_row, marker_row, bar_left in zip(upper_pick, lower_pick, bar_lefts):
        colors = (RED, BLUE) if int(bar_left) == int(reversed_left) else (BLUE, RED)
        marker_col = int(bar_left + random.randint(0, 1))
        pairs.append(
            {
                "bar_row": int(bar_row),
                "bar_left": int(bar_left),
                "colors": colors,
                "marker_row": int(marker_row),
                "marker_col": int(marker_col),
            }
        )

    grid = _build_input_from_spec(
        height,
        width,
        barrier_row,
        barrier_left,
        barrier_right,
        reference_row,
        reference_left,
        pairs,
    )
    expected = _spec_output(
        grid.shape,
        barrier_row,
        barrier_left,
        barrier_right,
        reference_row,
        reference_left,
        pairs,
    )

    try:
        produced = main(grid)
    except Exception:
        return None

    if not np.array_equal(produced, expected):
        return None
    if not _semantic_checks(grid, produced, barrier_row, reference_left, pairs):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid cross-barrier swap-alignment puzzle")
