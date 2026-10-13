from __future__ import annotations

import random

import numpy as np

from arc_utils import Color, get_bg


GREY = Color.GREY


def _find_segments(line: np.ndarray, color: int) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    i = 1
    while i < len(line):
        if int(line[i]) != color:
            i += 1
            continue
        j = i
        while j + 1 < len(line) and int(line[j + 1]) == color:
            j += 1
        spans.append((i, j))
        i = j + 1
    return spans


def _find_span(index: int, spans: list[tuple[int, int]]) -> tuple[int, int] | None:
    for start, end in spans:
        if start <= index <= end:
            return start, end
    return None


def _paint_rectangles(
    shape: tuple[int, int],
    bg: int,
    row_spans: list[tuple[int, int]],
    col_spans: list[tuple[int, int]],
    markers: list[tuple[int, int, int]],
) -> np.ndarray:
    out = np.full(shape, bg, dtype=int)
    for r, c, color in markers:
        row_span = _find_span(r, row_spans)
        col_span = _find_span(c, col_spans)
        if row_span is None or col_span is None:
            continue
        r0, r1 = row_span
        c0, c1 = col_span
        out[r0 : r1 + 1, c0 : c1 + 1] = color
    return out


def main(input_grid: np.ndarray) -> np.ndarray:
    """
    Reconstruct solid rectangles from their top-row and left-column grey projections.

    A colored singleton authorizes the rectangle formed by the left-edge row span
    covering its row and the top-edge column span covering its column. All shadows
    and all singletons are erased in the output.
    """
    grid = np.asarray(input_grid, dtype=int)
    bg = get_bg(grid)
    row_spans = _find_segments(grid[:, 0], GREY)
    col_spans = _find_segments(grid[0], GREY)

    markers = [
        (int(r), int(c), int(grid[r, c]))
        for r, c in np.argwhere((grid != bg) & (grid != GREY))
        if r > 0 and c > 0
    ]
    return _paint_rectangles(grid.shape, bg, row_spans, col_spans, markers)


def _sample_segments(axis_len: int, count: int) -> list[tuple[int, int]]:
    usable = axis_len - 3
    base = 2 * count + (count - 1)
    if usable < base:
        raise RuntimeError("axis too short for the requested number of spans")

    lengths = [2] * count
    gaps = [0] + [1] * (count - 1) + [0]
    extra = usable - base
    for _ in range(extra):
        bucket = random.randrange(len(lengths) + len(gaps))
        if bucket < len(lengths):
            lengths[bucket] += 1
        else:
            gaps[bucket - len(lengths)] += 1

    spans: list[tuple[int, int]] = []
    cursor = 2 + gaps[0]
    for i, length in enumerate(lengths):
        spans.append((cursor, cursor + length - 1))
        cursor += length + gaps[i + 1]
    return spans


def _choose_pairs(n_rows: int, n_cols: int) -> list[tuple[int, int]]:
    all_pairs = [(ri, ci) for ri in range(n_rows) for ci in range(n_cols)]
    target = random.randint(2, min(6, len(all_pairs)))
    pairs: set[tuple[int, int]] = set()

    if n_rows >= 1 and n_cols >= 2 and random.random() < 0.5:
        ri = random.randrange(n_rows)
        for ci in random.sample(range(n_cols), 2):
            pairs.add((ri, ci))
    elif n_rows >= 2 and n_cols >= 1:
        ci = random.randrange(n_cols)
        for ri in random.sample(range(n_rows), 2):
            pairs.add((ri, ci))

    while len(pairs) < target:
        pairs.add(random.choice(all_pairs))
    return list(pairs)


def _is_clear(cell: tuple[int, int], occupied: set[tuple[int, int]]) -> bool:
    r, c = cell
    return all(max(abs(r - rr), abs(c - cc)) > 1 for rr, cc in occupied)


def generate_input() -> np.ndarray:
    for _ in range(400):
        height = random.randint(12, 22)
        width = random.randint(12, 22)
        bg = random.choice([c for c in Color.ALL_COLORS if c != GREY])
        palette = [c for c in Color.ALL_COLORS if c not in {bg, GREY}]

        n_row_spans = random.randint(2, min(4, (height - 2) // 3))
        n_col_spans = random.randint(2, min(4, (width - 2) // 3))
        row_spans = _sample_segments(height, n_row_spans)
        col_spans = _sample_segments(width, n_col_spans)
        pairs = _choose_pairs(n_row_spans, n_col_spans)

        grid = np.full((height, width), bg, dtype=int)
        for c0, c1 in col_spans:
            grid[0, c0 : c1 + 1] = GREY
        for r0, r1 in row_spans:
            grid[r0 : r1 + 1, 0] = GREY

        occupied: set[tuple[int, int]] = set()
        markers: list[tuple[int, int, int]] = []
        colors = random.sample(palette, len(pairs))
        for (ri, ci), color in zip(pairs, colors):
            r0, r1 = row_spans[ri]
            c0, c1 = col_spans[ci]
            cells = [(r, c) for r in range(r0, r1 + 1) for c in range(c0, c1 + 1)]
            random.shuffle(cells)
            marker_cell = next((cell for cell in cells if _is_clear(cell, occupied)), None)
            if marker_cell is None:
                break
            occupied.add(marker_cell)
            markers.append((marker_cell[0], marker_cell[1], color))
            grid[marker_cell] = color
        else:
            row_cover = {r for r0, r1 in row_spans for r in range(r0, r1 + 1)}
            col_cover = {c for c0, c1 in col_spans for c in range(c0, c1 + 1)}
            distractor_goal = random.randint(1, 4)
            distractor_cells = [
                (r, c)
                for r in range(1, height)
                for c in range(1, width)
                if (r not in row_cover or c not in col_cover)
            ]
            random.shuffle(distractor_cells)
            for r, c in distractor_cells:
                if distractor_goal == 0:
                    break
                if not _is_clear((r, c), occupied):
                    continue
                occupied.add((r, c))
                grid[r, c] = random.choice(palette)
                distractor_goal -= 1

            if len(markers) >= 2:
                return grid

    raise RuntimeError("failed to generate a valid ARC instance")
