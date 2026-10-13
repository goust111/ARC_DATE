from __future__ import annotations

from collections import Counter

import numpy as np


BG = 0
PALETTE = np.arange(1, 10, dtype=int)
MAX_GENERATION_ATTEMPTS = 400
_RNG = np.random.default_rng()


def _most_common_color(grid: np.ndarray) -> int:
    counts = Counter(map(int, np.asarray(grid).reshape(-1)))
    return int(counts.most_common(1)[0][0])


def _row_runs(row: np.ndarray, bg: int) -> list[tuple[int, int]]:
    runs: list[tuple[int, int]] = []
    c = 0
    width = int(row.shape[0])
    while c < width:
        if int(row[c]) == int(bg):
            c += 1
            continue
        start = c
        while c < width and int(row[c]) != int(bg):
            c += 1
        runs.append((start, c - 1))
    return runs


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    height, width = map(int, mask.shape)
    seen = np.zeros((height, width), dtype=bool)
    out: list[list[tuple[int, int]]] = []

    for r in range(height):
        for c in range(width):
            if not bool(mask[r, c]) or bool(seen[r, c]):
                continue
            stack = [(r, c)]
            seen[r, c] = True
            cells: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                cells.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < height and 0 <= nc < width and bool(mask[nr, nc]) and not bool(seen[nr, nc]):
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            out.append(cells)
    return out


def _legend_colors(grid: np.ndarray, bg: int) -> list[int]:
    runs = _row_runs(grid[0], bg)
    assert len(runs) == 1, "expected one top-edge legend run"
    c0, c1 = runs[0]
    legend = [int(v) for v in grid[0, c0 : c1 + 1]]
    assert legend and len(set(legend)) == len(legend), "legend colors must be pairwise distinct"
    return legend


def _count_color_rectangles(field: np.ndarray, color: int) -> int:
    total = 0
    for comp in _components(field == int(color)):
        rows = [r for r, _ in comp]
        cols = [c for _, c in comp]
        r0, r1 = min(rows), max(rows)
        c0, c1 = min(cols), max(cols)
        bbox = field[r0 : r1 + 1, c0 : c1 + 1]
        assert np.all(bbox == int(color)), "expected solid monochrome rectangles"
        assert len(comp) == bbox.size, "expected rectangular components"
        total += 1
    return total


def main(grid: np.ndarray) -> np.ndarray:
    """
    Filter the top legend string to colors whose rectangle count equals their legend index.

    Input structure
    - A background grid with one horizontal legend string on the top edge.
    - The legend is a contiguous run of pairwise distinct foreground colors.
    - Below it are pairwise separated solid monochrome rectangles whose colors all come from the legend.

    Rule
    - Read the legend from left to right, numbering its positions from 1.
    - For each legend color, count how many rectangles of that color appear in the field below the legend.
    - Keep exactly the colors whose count equals their legend position index, and output them as a single horizontal row in the original legend order.

    Key insight
    - The legend is self-referential: each color's spatial index is a claim about how many field rectangles of that color should exist.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.shape[0] >= 2, "expected a 2D grid with a top legend row"

    bg = _most_common_color(g)
    legend = _legend_colors(g, bg)
    field = g[1:, :]

    kept = [
        color
        for index, color in enumerate(legend, start=1)
        if _count_color_rectangles(field, color) == index
    ]
    return np.asarray([kept], dtype=int)


RECT_SIZES_SMALL = ((1, 1), (1, 2), (2, 1), (2, 2), (1, 3), (3, 1))
RECT_SIZES_MEDIUM = RECT_SIZES_SMALL + ((2, 3), (3, 2))
RECT_SIZES_LARGE = RECT_SIZES_MEDIUM + ((1, 4), (4, 1), (3, 3))


def _pick_weighted(values: list[int], weights: list[int]) -> int:
    probs = np.asarray(weights, dtype=float)
    probs /= probs.sum()
    return int(_RNG.choice(np.asarray(values, dtype=int), p=probs))


def _sample_counts(legend_len: int) -> list[int] | None:
    for _ in range(200):
        matches = _RNG.random(legend_len) < 0.45
        if not bool(matches.any()):
            matches[int(_RNG.integers(legend_len))] = True
        if bool(matches.all()):
            matches[int(_RNG.integers(legend_len))] = False

        counts: list[int] = []
        for index in range(1, legend_len + 1):
            if bool(matches[index - 1]):
                counts.append(index)
                continue
            choices = [v for v in range(0, legend_len + 1) if v != index]
            weights = [4 if v == 0 else 3 if abs(v - index) == 1 else 1 for v in choices]
            counts.append(_pick_weighted(choices, weights))

        total_rects = sum(counts)
        positive_colors = sum(count > 0 for count in counts)
        if 4 <= total_rects <= 13 and positive_colors >= 2 and any(count == i for i, count in enumerate(counts, start=1)):
            return counts
    return None


def _sample_rectangles(legend: list[int], counts: list[int]) -> list[tuple[int, int, int]]:
    total_rects = sum(counts)
    if total_rects >= 10:
        size_choices = RECT_SIZES_SMALL
    elif total_rects >= 7:
        size_choices = RECT_SIZES_MEDIUM
    else:
        size_choices = RECT_SIZES_LARGE

    rects: list[tuple[int, int, int]] = []
    for color, count in zip(legend, counts):
        for _ in range(count):
            h, w = size_choices[int(_RNG.integers(len(size_choices)))]
            rects.append((int(color), int(h), int(w)))

    if rects and max(h * w for _, h, w in rects) == 1:
        idx = int(_RNG.integers(len(rects)))
        color, _h, _w = rects[idx]
        new_h, new_w = RECT_SIZES_MEDIUM[int(_RNG.integers(1, len(RECT_SIZES_MEDIUM)))]
        rects[idx] = (color, int(new_h), int(new_w))

    order = np.argsort([-h * w for _, h, w in rects])
    return [rects[int(i)] for i in order]


def _sample_canvas(legend_len: int, rects: list[tuple[int, int, int]]) -> tuple[int, int] | None:
    total_area = sum(h * w for _, h, w in rects)
    max_h = max(h for _, h, _ in rects)
    max_w = max(w for _, _, w in rects)
    height_low = max(10, max_h + 3, int(np.ceil(np.sqrt(total_area))) + 4)
    width_low = max(legend_len + 2, max_w + 2, int(np.ceil(np.sqrt(total_area))) + 5)
    if height_low > 30 or width_low > 30:
        return None
    height_high = min(30, height_low + 8)
    width_high = min(30, width_low + 10)
    height = int(_RNG.integers(height_low, height_high + 1))
    width = int(_RNG.integers(width_low, width_high + 1))
    return height, width


def _place_rectangles(
    shape: tuple[int, int],
    legend: list[int],
    rects: list[tuple[int, int, int]],
) -> np.ndarray | None:
    height, width = map(int, shape)
    grid = np.full((height, width), BG, dtype=int)
    legend_left = int(_RNG.integers(0, width - len(legend) + 1))
    grid[0, legend_left : legend_left + len(legend)] = np.asarray(legend, dtype=int)

    blocked = np.zeros((height, width), dtype=bool)
    blocked[:2, :] = True
    blocked[0, legend_left : legend_left + len(legend)] = True

    for color, rect_h, rect_w in rects:
        candidates: list[tuple[int, int]] = []
        for r0 in range(2, height - rect_h + 1):
            for c0 in range(0, width - rect_w + 1):
                r1 = r0 + rect_h - 1
                c1 = c0 + rect_w - 1
                rr0 = max(0, r0 - 1)
                rr1 = min(height - 1, r1 + 1)
                cc0 = max(0, c0 - 1)
                cc1 = min(width - 1, c1 + 1)
                if not bool(blocked[rr0 : rr1 + 1, cc0 : cc1 + 1].any()):
                    candidates.append((r0, c0))
        if not candidates:
            return None
        r0, c0 = candidates[int(_RNG.integers(len(candidates)))]
        grid[r0 : r0 + rect_h, c0 : c0 + rect_w] = int(color)
        blocked[max(0, r0 - 1) : min(height, r0 + rect_h + 1), max(0, c0 - 1) : min(width, c0 + rect_w + 1)] = True

    return grid


def _expected_output(legend: list[int], counts: list[int]) -> np.ndarray:
    kept = [color for i, (color, count) in enumerate(zip(legend, counts), start=1) if count == i]
    return np.asarray([kept], dtype=int)


def _gen_attempt() -> np.ndarray | None:
    legend_len = int(_RNG.integers(3, 7))
    legend = [int(v) for v in _RNG.choice(PALETTE, size=legend_len, replace=False)]
    counts = _sample_counts(legend_len)
    if counts is None:
        return None

    rects = _sample_rectangles(legend, counts)
    shape = _sample_canvas(legend_len, rects)
    if shape is None:
        return None

    grid = _place_rectangles(shape, legend, rects)
    if grid is None:
        return None
    if int(np.count_nonzero(grid != BG)) * 2 >= int(grid.size):
        return None

    expected = _expected_output(legend, counts)
    try:
        actual = main(grid)
    except Exception:
        return None
    if np.array_equal(actual, expected):
        return grid
    return None


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid self-referential legend input")
