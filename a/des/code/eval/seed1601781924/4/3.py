import random
from itertools import combinations

import numpy as np


BG = 0
SEP = 5
MAROON = 9
WORK_COLORS = (1, 2, 3, 4, 6, 7, 8)

MAX_GRID_SIZE = 30
MIN_CELL_SIZE = 5
MAX_CELL_SIZE = 7
GENERATION_RETRIES = 400
CELL_RETRIES = 120

ALL_SINGLE_KEYS = [(color,) for color in WORK_COLORS]
ALL_PAIR_KEYS = list(combinations(WORK_COLORS, 2))


def _true_spans(mask: np.ndarray) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    start = None
    for idx, flag in enumerate(mask.tolist()):
        if flag and start is None:
            start = idx
        elif not flag and start is not None:
            spans.append((start, idx - 1))
            start = None
    if start is not None:
        spans.append((start, len(mask) - 1))
    return spans


def _neighbors4(r: int, c: int, h: int, w: int) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        rr, cc = r + dr, c + dc
        if 0 <= rr < h and 0 <= cc < w:
            out.append((rr, cc))
    return out


def _paste(grid: np.ndarray, patch: np.ndarray, r0: int, c0: int) -> None:
    h, w = patch.shape
    grid[r0 : r0 + h, c0 : c0 + w] = patch


def _split_cells(grid: np.ndarray) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    row_bands = _true_spans(~np.all(grid == SEP, axis=1))
    col_bands = _true_spans(~np.all(grid == SEP, axis=0))
    assert len(row_bands) >= 2, "expected a palette row and at least one content row"
    assert len(col_bands) >= 1, "expected at least one column of cells"
    return row_bands, col_bands


def _palette_entry(cell: np.ndarray) -> tuple[frozenset[int], int]:
    h, _w = cell.shape
    split = h // 2
    assert 0 < split < h, "palette cells must have both an upper and lower half"

    upper = cell[:split]
    lower = cell[split:]

    key_pixels = [int(v) for v in upper.flat if int(v) not in (BG, SEP)]
    key_colors = frozenset(key_pixels)
    assert 1 <= len(key_colors) <= 2, "palette key must contain one or two colors"
    assert len(key_pixels) == len(key_colors), "palette key colors must be distinct singleton pixels"

    value_colors = {int(v) for v in lower.flat if int(v) not in (BG, SEP)}
    assert len(value_colors) == 1, "palette value area must use exactly one color"
    return key_colors, next(iter(value_colors))


def _signal_key(cell: np.ndarray) -> frozenset[int]:
    signal_pixels = [int(v) for v in cell.flat if int(v) not in (BG, SEP, MAROON)]
    signal_colors = frozenset(signal_pixels)
    assert 1 <= len(signal_colors) <= 2, "content cell must contain one or two signal pixels"
    assert len(signal_pixels) == len(signal_colors), "signal colors must be distinct"
    return signal_colors


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor each maroon foreground shape by matching its content-cell signal colors
    against the palette row above it, then clear the signal pixels.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "grid must be 2D"

    row_bands, col_bands = _split_cells(g)
    palette_row = row_bands[0]

    palette: dict[frozenset[int], int] = {}
    pr0, pr1 = palette_row
    for c0, c1 in col_bands:
        key, value = _palette_entry(g[pr0 : pr1 + 1, c0 : c1 + 1])
        assert key not in palette, "palette keys must be unique"
        palette[key] = value

    out = g.copy()
    for r0, r1 in row_bands[1:]:
        for c0, c1 in col_bands:
            cell = g[r0 : r1 + 1, c0 : c1 + 1]
            shape_mask = cell == MAROON
            assert np.any(shape_mask), "each content cell must contain a maroon foreground shape"
            signal_mask = (cell != BG) & (cell != SEP) & (cell != MAROON)
            key = _signal_key(cell)
            assert key in palette, "every content key must appear in the palette row"

            out_cell = out[r0 : r1 + 1, c0 : c1 + 1]
            out_cell[shape_mask] = palette[key]
            out_cell[signal_mask] = BG

    return out


def _sample_band_sizes(count: int) -> list[int] | None:
    limit = MAX_GRID_SIZE - (count + 1)
    for _ in range(100):
        sizes = [random.randint(MIN_CELL_SIZE, MAX_CELL_SIZE) for _ in range(count)]
        if sum(sizes) <= limit:
            return sizes
    return None


def _cell_spans(sizes: list[int]) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    pos = 1
    for size in sizes:
        spans.append((pos, pos + size - 1))
        pos += size + 1
    return spans


def _non_touching_positions(candidates: list[tuple[int, int]], count: int) -> list[tuple[int, int]] | None:
    for _ in range(80):
        pool = candidates[:]
        random.shuffle(pool)
        chosen: list[tuple[int, int]] = []
        for r, c in pool:
            if all(max(abs(r - rr), abs(c - cc)) > 1 for rr, cc in chosen):
                chosen.append((r, c))
                if len(chosen) == count:
                    return chosen
    return None


def _make_palette_cell(height: int, width: int, key: tuple[int, ...], value: int) -> np.ndarray | None:
    cell = np.full((height, width), BG, dtype=int)
    split = height // 2

    top_candidates = [(r, c) for r in range(split) for c in range(width)]
    key_positions = _non_touching_positions(top_candidates, len(key))
    if key_positions is None:
        return None

    key_colors = list(key)
    random.shuffle(key_colors)
    for (r, c), color in zip(key_positions, key_colors):
        cell[r, c] = color

    square_sizes = [size for size in (2, 3) if size <= height - split and size <= width]
    if not square_sizes:
        return None
    square_size = random.choice(square_sizes)
    r0 = random.randint(split, height - square_size)
    c0 = random.randint(0, width - square_size)
    cell[r0 : r0 + square_size, c0 : c0 + square_size] = value
    return cell


def _random_connected_mask(height: int, width: int) -> np.ndarray | None:
    margin = 1 if height >= 5 and width >= 5 else 0
    cells = [(r, c) for r in range(margin, height - margin) for c in range(margin, width - margin)]
    if len(cells) < 3:
        return None

    max_size = min(6, len(cells) - 1)
    if max_size < 3:
        return None

    target_size = random.randint(3, max_size)
    allowed = set(cells)
    shape = {random.choice(cells)}

    while len(shape) < target_size:
        frontier: list[tuple[int, int]] = []
        for r, c in shape:
            for rr, cc in _neighbors4(r, c, height, width):
                if (rr, cc) in allowed and (rr, cc) not in shape:
                    frontier.append((rr, cc))
        if not frontier:
            return None
        shape.add(random.choice(frontier))

    mask = np.zeros((height, width), dtype=bool)
    for r, c in shape:
        mask[r, c] = True
    return mask


def _signal_candidates(shape_mask: np.ndarray) -> list[tuple[int, int]]:
    height, width = shape_mask.shape
    blocked = shape_mask.copy()
    for r, c in np.argwhere(shape_mask):
        for rr, cc in _neighbors4(int(r), int(c), height, width):
            blocked[rr, cc] = True
    return [(r, c) for r in range(height) for c in range(width) if not blocked[r, c]]


def _make_content_cell(height: int, width: int, key: tuple[int, ...], value: int) -> tuple[np.ndarray, np.ndarray] | None:
    for _ in range(CELL_RETRIES):
        shape_mask = _random_connected_mask(height, width)
        if shape_mask is None:
            continue

        signal_positions = _non_touching_positions(_signal_candidates(shape_mask), len(key))
        if signal_positions is None:
            continue

        inp = np.full((height, width), BG, dtype=int)
        out = np.full((height, width), BG, dtype=int)
        inp[shape_mask] = MAROON
        out[shape_mask] = value

        signal_colors = list(key)
        random.shuffle(signal_colors)
        for (r, c), color in zip(signal_positions, signal_colors):
            inp[r, c] = color
        return inp, out
    return None


def _choose_palette_entries(n_cols: int) -> list[tuple[tuple[int, ...], int]]:
    if n_cols == 3:
        n_pairs = random.choice((1, 2))
    else:
        n_pairs = random.choice((1, 2, 3))
    n_singles = n_cols - n_pairs

    keys = random.sample(ALL_SINGLE_KEYS, n_singles) + random.sample(ALL_PAIR_KEYS, n_pairs)
    random.shuffle(keys)

    values: list[int] = []
    for key in keys:
        options = [color for color in WORK_COLORS if color not in key]
        values.append(random.choice(options))

    if len(set(values)) == 1 and len(values) > 1:
        idx = random.randrange(len(values))
        alternatives = [color for color in WORK_COLORS if color not in keys[idx] and color != values[idx]]
        values[idx] = random.choice(alternatives)

    return list(zip(keys, values))


def _gen_attempt() -> np.ndarray | None:
    n_cols = random.randint(3, 4)
    n_content_rows = random.randint(1, 3)
    row_sizes = _sample_band_sizes(1 + n_content_rows)
    col_sizes = _sample_band_sizes(n_cols)
    if row_sizes is None or col_sizes is None:
        return None

    height = sum(row_sizes) + len(row_sizes) + 1
    width = sum(col_sizes) + len(col_sizes) + 1
    if height > MAX_GRID_SIZE or width > MAX_GRID_SIZE:
        return None

    row_spans = _cell_spans(row_sizes)
    col_spans = _cell_spans(col_sizes)
    palette_entries = _choose_palette_entries(n_cols)

    inp = np.full((height, width), SEP, dtype=int)
    out = np.full((height, width), SEP, dtype=int)

    pr0, _pr1 = row_spans[0]
    for col_idx, (c0, _c1) in enumerate(col_spans):
        cell = _make_palette_cell(row_sizes[0], col_sizes[col_idx], *palette_entries[col_idx])
        if cell is None:
            return None
        _paste(inp, cell, pr0, c0)
        _paste(out, cell, pr0, c0)

    base_order = list(range(n_cols))
    for row_idx in range(n_content_rows):
        entry_order = base_order[:]
        random.shuffle(entry_order)
        r0, _r1 = row_spans[row_idx + 1]
        for col_idx, (c0, _c1) in enumerate(col_spans):
            key, value = palette_entries[entry_order[col_idx]]
            pair = _make_content_cell(row_sizes[row_idx + 1], col_sizes[col_idx], key, value)
            if pair is None:
                return None
            cell_in, cell_out = pair
            _paste(inp, cell_in, r0, c0)
            _paste(out, cell_out, r0, c0)

    try:
        if np.array_equal(main(inp), out):
            return inp
    except AssertionError:
        return None
    return None


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid palette-signal ARC input")
