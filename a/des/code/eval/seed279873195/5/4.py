from __future__ import annotations

import random

import numpy as np


BG = 0
COLUMN_COLOR = 1
ROW_COLOR = 2
FILL_COLOR = 3
SEPARATOR_COLOR = 5

GRID_SIZE_OPTIONS = (3, 4)
CELL_SIZE_RANGE = (3, 5)
MISSING_COUNT_OPTIONS = (2, 3)
MAX_GENERATION_ATTEMPTS = 400


def _full_line_positions(grid: np.ndarray, color: int, axis: int) -> list[int]:
    if axis == 0:
        return [r for r in range(grid.shape[0]) if bool(np.all(grid[r, :] == int(color)))]
    return [c for c in range(grid.shape[1]) if bool(np.all(grid[:, c] == int(color)))]


def _separator_info(grid: np.ndarray) -> tuple[int, list[int], list[int]]:
    candidates: list[tuple[int, list[int], list[int]]] = []
    for color in map(int, np.unique(grid)):
        rows = _full_line_positions(grid, color, axis=0)
        cols = _full_line_positions(grid, color, axis=1)
        if len(rows) >= 2 and len(cols) >= 2:
            candidates.append((int(color), rows, cols))

    assert candidates, "expected a separator lattice color"
    candidates.sort(key=lambda item: (len(item[1]) + len(item[2]), item[0]), reverse=True)
    sep_color, sep_rows, sep_cols = candidates[0]
    return sep_color, sorted(sep_rows), sorted(sep_cols)


def _cell_spans(lines: list[int], length: int) -> list[tuple[int, int]]:
    boundaries = [-1, *sorted(lines), int(length)]
    spans: list[tuple[int, int]] = []
    for start, stop in zip(boundaries, boundaries[1:], strict=False):
        lo = int(start) + 1
        hi = int(stop)
        if lo < hi:
            spans.append((lo, hi))
    assert spans, "expected at least one cell span"
    sizes = {hi - lo for lo, hi in spans}
    assert len(sizes) == 1, "all cells must have the same size"
    return spans


def _cell_view(grid: np.ndarray, row_span: tuple[int, int], col_span: tuple[int, int]) -> np.ndarray:
    r0, r1 = row_span
    c0, c1 = col_span
    return grid[r0:r1, c0:c1]


def _majority_color(cell: np.ndarray) -> int:
    values, counts = np.unique(cell, return_counts=True)
    return int(values[np.argmax(counts)])


def _mask_signature(mask: np.ndarray) -> tuple[tuple[int, int], ...]:
    return tuple((int(r), int(c)) for r, c in zip(*np.where(mask), strict=False))


def _pick_fill_color(grid: np.ndarray, row_spans: list[tuple[int, int]], col_spans: list[tuple[int, int]], bg: int) -> int:
    vals: list[int] = []
    for row_span in row_spans[1:]:
        for col_span in col_spans[1:]:
            cell = _cell_view(grid, row_span, col_span)
            vals.extend(map(int, cell[cell != int(bg)]))
    if vals:
        values, counts = np.unique(vals, return_counts=True)
        return int(values[np.argmax(counts)])

    used = set(map(int, np.unique(grid)))
    for color in range(10):
        if color not in used:
            return int(color)
    return FILL_COLOR


def main(grid: np.ndarray) -> np.ndarray:
    """
    Complete blank interior cells by intersecting row-title and column-title masks.

    The grid is a spreadsheet-like lattice of equal MxM cells separated by 1-pixel
    grey lines. The first row holds blue column-title patterns, the first column
    holds red row-title patterns, and each filled interior cell shows the pixelwise
    AND of its row-title and column-title masks in green.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    _sep_color, sep_rows, sep_cols = _separator_info(g)
    row_spans = _cell_spans(sep_rows, g.shape[0])
    col_spans = _cell_spans(sep_cols, g.shape[1])
    assert len(row_spans) == len(col_spans) >= 2, "expected a square cell lattice with headers"

    bg = _majority_color(_cell_view(g, row_spans[0], col_spans[0]))
    fill_color = _pick_fill_color(g, row_spans, col_spans, bg)

    row_masks = [_cell_view(g, row_span, col_spans[0]) != int(bg) for row_span in row_spans[1:]]
    col_masks = [_cell_view(g, row_spans[0], col_span) != int(bg) for col_span in col_spans[1:]]

    out = g.copy()
    for i, row_mask in enumerate(row_masks, start=1):
        for j, col_mask in enumerate(col_masks, start=1):
            cell = _cell_view(out, row_spans[i], col_spans[j])
            if bool(np.any(cell != int(bg))):
                continue
            cell[row_mask & col_mask] = int(fill_color)
    return out


def _sample_title_masks(count: int, cell_size: int) -> tuple[list[np.ndarray], list[np.ndarray]] | None:
    cells = [(r, c) for r in range(cell_size) for c in range(cell_size)]
    max_core = min(2, cell_size)
    core_size = random.randint(1, max_core)
    core = set(random.sample(cells, core_size))

    remaining = [pos for pos in cells if pos not in core]
    pool_size = random.randint(max(3, count), min(len(remaining), max(4, count + 1)))
    feature_pool = random.sample(remaining, pool_size)

    max_extra = min(len(feature_pool), max(2, cell_size))
    subset_cache: list[set[tuple[int, int]]] = []
    seen_subsets: set[frozenset[tuple[int, int]]] = set()
    while len(subset_cache) < 2 * count:
        extra_count = random.randint(1, max_extra)
        subset = frozenset(random.sample(feature_pool, extra_count))
        if subset in seen_subsets:
            continue
        seen_subsets.add(subset)
        subset_cache.append(set(subset))

    random.shuffle(subset_cache)
    row_subsets = subset_cache[:count]
    col_subsets = subset_cache[count:]

    row_masks = []
    col_masks = []
    for subset in row_subsets:
        mask = np.zeros((cell_size, cell_size), dtype=bool)
        for r, c in core | subset:
            mask[r, c] = True
        row_masks.append(mask)
    for subset in col_subsets:
        mask = np.zeros((cell_size, cell_size), dtype=bool)
        for r, c in core | subset:
            mask[r, c] = True
        col_masks.append(mask)

    row_sigs = {_mask_signature(mask) for mask in row_masks}
    col_sigs = {_mask_signature(mask) for mask in col_masks}
    if len(row_sigs) != count or len(col_sigs) != count:
        return None

    intersections = [row_masks[i] & col_masks[j] for i in range(count) for j in range(count)]
    if not all(bool(mask.any()) for mask in intersections):
        return None

    unique_intersections = {_mask_signature(mask) for mask in intersections}
    min_unique = max(4, count + 1)
    if len(unique_intersections) < min_unique:
        return None

    if all(int(mask.sum()) == core_size for mask in intersections):
        return None

    for i in range(count):
        row_variants = {_mask_signature(row_masks[i] & col_masks[j]) for j in range(count)}
        col_variants = {_mask_signature(row_masks[j] & col_masks[i]) for j in range(count)}
        if len(row_variants) < 2 or len(col_variants) < 2:
            return None

    return row_masks, col_masks


def _sample_missing_cells(count: int, blank_count: int) -> list[tuple[int, int]]:
    cells = [(r, c) for r in range(count) for c in range(count)]
    for _ in range(100):
        chosen = random.sample(cells, blank_count)
        rows = {r for r, _c in chosen}
        cols = {c for _r, c in chosen}
        if len(rows) >= min(2, blank_count) and len(cols) >= min(2, blank_count):
            return chosen
    return random.sample(cells, blank_count)


def _layout_spans(count: int, cell_size: int) -> tuple[list[tuple[int, int]], list[int]]:
    spans: list[tuple[int, int]] = []
    separators: list[int] = []
    pos = 0
    separators.append(pos)
    for _ in range(count + 1):
        start = pos + 1
        stop = start + cell_size
        spans.append((start, stop))
        pos = stop
        separators.append(pos)
    return spans, separators


def _paint_mask(grid: np.ndarray, row_span: tuple[int, int], col_span: tuple[int, int], mask: np.ndarray, color: int) -> None:
    cell = _cell_view(grid, row_span, col_span)
    cell[mask] = int(color)


def _build_example() -> tuple[np.ndarray, np.ndarray] | None:
    count = random.choice(GRID_SIZE_OPTIONS)
    cell_size = random.randint(*CELL_SIZE_RANGE)
    blank_count = random.choice(MISSING_COUNT_OPTIONS)

    sampled = _sample_title_masks(count, cell_size)
    if sampled is None:
        return None
    row_masks, col_masks = sampled

    spans, separators = _layout_spans(count, cell_size)
    side = separators[-1] + 1
    inp = np.full((side, side), BG, dtype=int)
    out = inp.copy()

    inp[separators, :] = SEPARATOR_COLOR
    inp[:, separators] = SEPARATOR_COLOR
    out[separators, :] = SEPARATOR_COLOR
    out[:, separators] = SEPARATOR_COLOR

    for i, mask in enumerate(col_masks, start=1):
        _paint_mask(inp, spans[0], spans[i], mask, COLUMN_COLOR)
        _paint_mask(out, spans[0], spans[i], mask, COLUMN_COLOR)
    for i, mask in enumerate(row_masks, start=1):
        _paint_mask(inp, spans[i], spans[0], mask, ROW_COLOR)
        _paint_mask(out, spans[i], spans[0], mask, ROW_COLOR)

    blank_cells = set(_sample_missing_cells(count, blank_count))
    for r in range(count):
        for c in range(count):
            combined = row_masks[r] & col_masks[c]
            _paint_mask(out, spans[r + 1], spans[c + 1], combined, FILL_COLOR)
            if (r, c) not in blank_cells:
                _paint_mask(inp, spans[r + 1], spans[c + 1], combined, FILL_COLOR)

    if np.array_equal(inp, out):
        return None
    if not np.array_equal(main(inp), out):
        return None
    return inp, out


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        built = _build_example()
        if built is None:
            continue
        inp, _out = built
        return inp
    raise RuntimeError("failed to generate a valid ARC input")
