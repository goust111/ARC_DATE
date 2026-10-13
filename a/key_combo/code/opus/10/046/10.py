import itertools
import random

import numpy as np


BG = 0
SEP = 5
FG_COLORS = (1, 2, 3, 4)
CELL_SIZE = 3
LATTICE_SIZE = 4
GRID_SIZE = LATTICE_SIZE * CELL_SIZE + (LATTICE_SIZE - 1)
KEEP_COUNT_RANGE = (4, 8)
GEN_ATTEMPTS = 300


def _bands(size: int, separators: list[int]) -> list[tuple[int, int]]:
    bands = []
    start = 0
    for cut in separators:
        if start < cut:
            bands.append((start, cut - 1))
        start = cut + 1
    if start < size:
        bands.append((start, size - 1))
    return bands


def _is_connected(mask: np.ndarray) -> bool:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return False
    seen = {tuple(map(int, coords[0]))}
    stack = [tuple(map(int, coords[0]))]
    while stack:
        r, c = stack.pop()
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nr = r + dr
            nc = c + dc
            if 0 <= nr < mask.shape[0] and 0 <= nc < mask.shape[1] and mask[nr, nc]:
                nxt = (nr, nc)
                if nxt not in seen:
                    seen.add(nxt)
                    stack.append(nxt)
    return len(seen) == int(mask.sum())


def _shape_library() -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray]]:
    touch, shallow, deep = [], [], []
    for bits in itertools.product((0, 1), repeat=CELL_SIZE * CELL_SIZE):
        if not 2 <= sum(bits) <= 5:
            continue
        mask = np.array(bits, dtype=bool).reshape(CELL_SIZE, CELL_SIZE)
        if not _is_connected(mask):
            continue
        top = int(np.argwhere(mask)[:, 0].min())
        if top == 0:
            touch.append(mask)
        elif top == 1:
            shallow.append(mask)
        else:
            deep.append(mask)
    assert touch and shallow and deep
    return touch, shallow, deep


TOUCH_SHAPES, SHALLOW_ERASE_SHAPES, DEEP_ERASE_SHAPES = _shape_library()


def _separator_bands(grid: np.ndarray) -> tuple[int, list[tuple[int, int]], list[tuple[int, int]]]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    row_full = {int(g[r, 0]) for r in range(h) if np.all(g[r, :] == g[r, 0])}
    col_full = {int(g[0, c]) for c in range(w) if np.all(g[:, c] == g[0, c])}
    sep_candidates = row_full & col_full
    assert len(sep_candidates) == 1, "expected exactly one separator color"
    sep = next(iter(sep_candidates))

    sep_rows = [r for r in range(h) if np.all(g[r, :] == sep)]
    sep_cols = [c for c in range(w) if np.all(g[:, c] == sep)]
    row_bands = _bands(h, sep_rows)
    col_bands = _bands(w, sep_cols)

    assert len(row_bands) == len(col_bands) == LATTICE_SIZE, "expected a 4x4 cell lattice"
    assert all(r1 - r0 + 1 == CELL_SIZE for r0, r1 in row_bands), "unexpected cell height"
    assert all(c1 - c0 + 1 == CELL_SIZE for c0, c1 in col_bands), "unexpected cell width"
    return sep, row_bands, col_bands


def _cell_mask(cell: np.ndarray, sep: int) -> np.ndarray:
    return (cell != BG) & (cell != sep)


def _top_foreground_row(cell: np.ndarray, sep: int) -> int:
    mask = _cell_mask(cell, sep)
    assert np.any(mask), "each cell must contain a foreground glyph"
    return int(np.argwhere(mask)[:, 0].min())


def main(grid: np.ndarray) -> np.ndarray:
    """
    Decode separator-lattice cells into keep-or-erase instructions and compact the kept cell interiors.

    Input structure:
    - A single-pixel grey separator lattice splits the grid into a 4x4 array of 3x3 interior cells.
    - Each cell contains one 4-connected, single-color glyph made of 2 to 5 non-black pixels.
    - Glyph colors vary, but only the glyph's vertical placement inside its own cell matters.

    Rule:
    - Detect the full-row and full-column separator lines and use them to recover the 4x4 cell lattice.
    - For each 3x3 cell, find the topmost row occupied by the non-black, non-separator glyph.
    - If that topmost glyph row is the cell's first interior row, keep the whole 3x3 cell; otherwise erase it.
    - Concatenate all kept 3x3 cell interiors in row-major order with no gaps to form a 3-row output.

    Key insight:
    - The glyph shape and color are distractors: the only encoded bit is whether the glyph reaches the top boundary of its cell interior.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    sep, row_bands, col_bands = _separator_bands(g)

    kept = []
    for r0, r1 in row_bands:
        for c0, c1 in col_bands:
            cell = g[r0 : r1 + 1, c0 : c1 + 1]
            if _top_foreground_row(cell, sep) == 0:
                kept.append(cell.copy())

    assert kept, "at least one cell must be kept"
    return np.concatenate(kept, axis=1)


def _tile_from_mask(mask: np.ndarray, color: int) -> np.ndarray:
    tile = np.zeros((CELL_SIZE, CELL_SIZE), dtype=int)
    tile[mask] = int(color)
    return tile


def _sample_keep_count() -> int:
    return random.randint(*KEEP_COUNT_RANGE)


def _sample_colors(n: int) -> list[int]:
    colors = list(FG_COLORS)
    random.shuffle(colors)
    sampled = colors[: min(3, n)]
    while len(sampled) < n:
        sampled.append(random.choice(FG_COLORS))
    random.shuffle(sampled)
    return sampled


def _expected_output(records: list[dict]) -> np.ndarray:
    kept_tiles = [record["tile"] for record in records if record["keep"]]
    assert kept_tiles
    return np.concatenate(kept_tiles, axis=1)


def _validate_semantics(grid: np.ndarray, records: list[dict], expected: np.ndarray) -> None:
    assert grid.shape == (GRID_SIZE, GRID_SIZE)
    sep, row_bands, col_bands = _separator_bands(grid)
    assert sep == SEP
    assert len(records) == LATTICE_SIZE * LATTICE_SIZE

    keep_tops = 0
    erase_tops = []
    idx = 0
    for r0, r1 in row_bands:
        for c0, c1 in col_bands:
            cell = grid[r0 : r1 + 1, c0 : c1 + 1]
            mask = _cell_mask(cell, sep)
            colors = np.unique(cell[mask])
            assert len(colors) == 1, "glyph must use a single visible color"
            assert 2 <= int(mask.sum()) <= 5, "glyph size must stay within 2..5"
            assert _is_connected(mask), "glyph must be 4-connected"
            assert np.array_equal(cell, records[idx]["tile"])
            top = _top_foreground_row(cell, sep)
            if top == 0:
                keep_tops += 1
            else:
                erase_tops.append(top)
            idx += 1

    assert KEEP_COUNT_RANGE[0] <= keep_tops <= KEEP_COUNT_RANGE[1]
    assert len(erase_tops) >= 3
    assert all(top in (1, 2) for top in erase_tops)
    assert 1 in erase_tops, "non-touching examples must include row-1 starts"
    if any(record["deep"] for record in records):
        assert 2 in erase_tops, "deep non-touching branch must survive in the rendered grid"

    actual = main(grid)
    assert np.array_equal(actual, expected), "solver output must match the independently built expected output"


def _generate_attempt() -> tuple[np.ndarray, np.ndarray] | None:
    grid = np.zeros((GRID_SIZE, GRID_SIZE), dtype=int)
    for i in range(1, LATTICE_SIZE):
        cut = i * CELL_SIZE + (i - 1)
        grid[cut, :] = SEP
        grid[:, cut] = SEP

    keep_count = _sample_keep_count()
    total_cells = LATTICE_SIZE * LATTICE_SIZE
    erase_count = total_cells - keep_count
    assert erase_count >= 3

    keep_slots = set(random.sample(range(total_cells), keep_count))
    deep_count = 0
    if erase_count >= 4 and random.random() < 0.65:
        deep_count = random.randint(1, min(3, erase_count - 2))

    erase_shape_types = [2] * deep_count + [1] * (erase_count - deep_count)
    random.shuffle(erase_shape_types)
    colors = _sample_colors(total_cells)

    records = []
    erase_idx = 0
    cell_index = 0
    for ri in range(LATTICE_SIZE):
        for ci in range(LATTICE_SIZE):
            if cell_index in keep_slots:
                mask = random.choice(TOUCH_SHAPES)
                keep = True
                deep = False
            else:
                shape_type = erase_shape_types[erase_idx]
                erase_idx += 1
                if shape_type == 2:
                    mask = random.choice(DEEP_ERASE_SHAPES)
                    deep = True
                else:
                    mask = random.choice(SHALLOW_ERASE_SHAPES)
                    deep = False
                keep = False

            tile = _tile_from_mask(mask, colors[cell_index])
            r0 = ri * (CELL_SIZE + 1)
            c0 = ci * (CELL_SIZE + 1)
            grid[r0 : r0 + CELL_SIZE, c0 : c0 + CELL_SIZE] = tile
            records.append({"tile": tile, "keep": keep, "deep": deep})
            cell_index += 1

    expected = _expected_output(records)
    try:
        _validate_semantics(grid, records, expected)
    except AssertionError:
        return None
    return grid, expected


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        generated = _generate_attempt()
        if generated is not None:
            grid, _expected = generated
            return grid
    raise RuntimeError("failed to generate a valid separator-glyph erase puzzle")
