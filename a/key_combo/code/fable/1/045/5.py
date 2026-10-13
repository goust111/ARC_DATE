from __future__ import annotations

import numpy as np


class Color:
    """Local subset of the shared ARC palette ids from arc_utils."""

    BLACK = 0
    BLUE = 1
    GREY = 5
    GRAY = 5


def _settle_sand(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    counts = np.count_nonzero(g == Color.GREY, axis=0)
    settled = np.full((h, w), Color.BLACK, dtype=int)
    for c, count in enumerate(counts):
        count = int(count)
        if count > 0:
            settled[h - count :, c] = Color.GREY
    return settled


def _prefix_sum(mask: np.ndarray) -> np.ndarray:
    padded = np.pad(mask.astype(np.int32), ((1, 0), (1, 0)), constant_values=0)
    return padded.cumsum(axis=0).cumsum(axis=1)


def _rect_sum(prefix: np.ndarray, r0: int, c0: int, side: int) -> int:
    r1 = int(r0 + side)
    c1 = int(c0 + side)
    return int(prefix[r1, c1] - prefix[r0, c1] - prefix[r1, c0] + prefix[r0, c0])


def _select_black_squares(grid: np.ndarray) -> list[tuple[int, int, int]]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    black = g == Color.BLACK
    prefix = _prefix_sum(black)
    occupied = np.zeros((h, w), dtype=bool)
    chosen: list[tuple[int, int, int]] = []

    for side in range(min(h, w), 1, -1):
        full_area = side * side
        for r0 in range(h - side + 1):
            for c0 in range(w - side + 1):
                if _rect_sum(prefix, r0, c0, side) != full_area:
                    continue
                if occupied[r0 : r0 + side, c0 : c0 + side].any():
                    continue
                occupied[r0 : r0 + side, c0 : c0 + side] = True
                chosen.append((int(r0), int(c0), int(side)))

    return chosen


def _paint_selected_squares(grid: np.ndarray, squares: list[tuple[int, int, int]]) -> np.ndarray:
    out = np.asarray(grid, dtype=int).copy()
    for r0, c0, side in squares:
        out[r0 : r0 + side, c0 : c0 + side] = Color.BLUE
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Let the grey sand settle, then paint the largest non-overlapping black squares blue.

    Input structure:
    - The grid is a black background containing only grey single-pixel sand grains.
    - The bottom border acts as the only floor; there are no other obstacles or frames.
    - Multiple grains may share a column but can begin at arbitrary heights.

    Rule:
    - In each column, drop every grey grain straight down until all grains in that column
      form one settled stack on the bottom edge.
    - In the settled grid, find every all-black square of side at least 2.
    - Process candidate squares from larger side length to smaller, keeping a square only
      if it does not overlap any previously kept square, and repaint every kept square blue.

    Key insight:
    - The searchable square regions do not exist in the input itself; they appear only
      after the grey grains have first been collapsed into bottom stacks.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    assert set(map(int, np.unique(g))) <= {Color.BLACK, Color.GREY}, "input must contain only black and grey"

    settled = _settle_sand(g)
    squares = _select_black_squares(settled)
    return _paint_selected_squares(settled, squares)


def _random_composition(total: int, parts: int, rng: np.random.Generator) -> list[int]:
    shares = [0] * parts
    for _ in range(int(total)):
        shares[int(rng.integers(parts))] += 1
    return shares


def _choose_sizes(rng: np.random.Generator) -> list[int]:
    roll = float(rng.random())
    if roll < 0.25:
        sizes = [int(rng.integers(2, 8))]
    elif roll < 0.65:
        k = int(rng.integers(2, 4))
        pool = np.arange(2, 8, dtype=int)
        rng.shuffle(pool)
        sizes = [int(x) for x in pool[:k]]
    else:
        k = int(rng.integers(2, 4))
        base = int(rng.integers(2, 8))
        sizes = [base] * k
        if k == 3 and float(rng.random()) < 0.5:
            other = int(rng.integers(2, 8))
            while other == base:
                other = int(rng.integers(2, 8))
            sizes[-1] = other
    if max(sizes) == 2 and float(rng.random()) < 0.7:
        sizes[0] = 3
    sizes_arr = np.array(sizes, dtype=int)
    rng.shuffle(sizes_arr)
    return [int(x) for x in sizes_arr]


def _build_settled_case(
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[tuple[int, int]], list[tuple[int, int, int]]] | None:
    sizes = _choose_sizes(rng)
    h = int(max(sizes) + rng.integers(2, 4))

    gap_count = len(sizes) + 1
    min_w = int(sum(sizes) + gap_count)
    max_extra = min(30 - min_w, 7)
    if max_extra < 0:
        return None
    extra = int(rng.integers(max_extra + 1))
    gap_widths = [1 + add for add in _random_composition(extra, gap_count, rng)]

    w = int(sum(sizes) + sum(gap_widths))
    black_heights = rng.choice(np.array([0, 1], dtype=int), size=w, p=np.array([0.25, 0.75]))
    segments: list[tuple[int, int]] = []
    c = gap_widths[0]
    for i, side in enumerate(sizes):
        side = int(side)
        segments.append((c, side))
        black_heights[c : c + side] = side
        c += side + gap_widths[i + 1]
    assert c == w

    settled = np.full((h, w), Color.BLACK, dtype=int)
    counts = h - black_heights.astype(int)
    for col, count in enumerate(counts):
        count = int(count)
        if count > 0:
            settled[h - count :, col] = Color.GREY

    expected = [(0, int(start), int(side)) for start, side in segments]
    expected.sort(key=lambda item: (-item[2], item[0], item[1]))
    if _select_black_squares(settled) != expected:
        return None

    out = _paint_selected_squares(settled, expected)
    return settled, out, counts, segments, expected


def _forced_rows(
    rng: np.random.Generator, segments: list[tuple[int, int]]
) -> dict[int, list[int]]:
    forced: dict[int, list[int]] = {}
    for start, side in segments:
        n_cols = 1 if side == 2 else int(rng.integers(1, min(3, side) + 1))
        cols = rng.choice(np.arange(start, start + side, dtype=int), size=n_cols, replace=False)
        for col in np.atleast_1d(cols):
            forced.setdefault(int(col), []).append(int(rng.integers(0, side)))
    return forced


def _scatter_input(
    rng: np.random.Generator,
    counts: np.ndarray,
    forced: dict[int, list[int]],
    h: int,
) -> np.ndarray | None:
    w = int(counts.shape[0])
    grid = np.full((h, w), Color.BLACK, dtype=int)
    weights = np.linspace(float(h), 1.0, h) ** 2

    for col, count in enumerate(counts):
        count = int(count)
        if count == 0:
            continue

        must_rows = sorted(set(int(r) for r in forced.get(col, [])))
        if len(must_rows) > count:
            return None

        available = [r for r in range(h) if r not in must_rows]
        need = count - len(must_rows)
        rows = list(must_rows)
        if need > 0:
            if need == len(available):
                rows.extend(available)
            else:
                p = np.array([weights[r] for r in available], dtype=float)
                p /= p.sum()
                chosen = rng.choice(np.array(available, dtype=int), size=need, replace=False, p=p)
                rows.extend(int(r) for r in chosen)

        grid[np.array(rows, dtype=int), col] = Color.GREY

    return grid


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    built = _build_settled_case(rng)
    if built is None:
        return None
    settled, expected_out, counts, segments, _expected = built
    h, _w = settled.shape

    forced = _forced_rows(rng, segments)
    inp = _scatter_input(rng, counts, forced, h)
    if inp is None:
        return None

    if np.array_equal(inp, settled):
        return None
    if not np.array_equal(np.count_nonzero(inp == Color.GREY, axis=0), counts):
        return None
    if not np.array_equal(_settle_sand(inp), settled):
        return None

    if not any(np.any(inp[:side, start : start + side] == Color.GREY) for start, side in segments):
        return None

    same_spots = int(np.count_nonzero((inp == Color.GREY) & (settled == Color.GREY)))
    total_sand = int(np.count_nonzero(settled == Color.GREY))
    if total_sand >= 6 and same_spots * 5 > total_sand * 4:
        return None

    if np.array_equal(_paint_selected_squares(inp, _select_black_squares(inp)), expected_out):
        return None

    try:
        actual_out = main(inp)
    except AssertionError:
        return None
    if not np.array_equal(actual_out, expected_out):
        return None

    return inp


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(400):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid sand-settling square-cavity input")
