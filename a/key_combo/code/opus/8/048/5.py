from __future__ import annotations

import numpy as np


BG = 0
MASK_COLOR = 5
MASK_SIZE = 4
SOURCE_COLORS = (1, 2, 3, 4, 7)
MIN_GENERATION_ATTEMPTS = 400


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _component_count(mask: np.ndarray) -> int:
    h, w = mask.shape
    seen: set[tuple[int, int]] = set()
    count = 0
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or (r, c) in seen:
                continue
            count += 1
            stack = [(r, c)]
            seen.add((r, c))
            while stack:
                cr, cc = stack.pop()
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and (nr, nc) not in seen:
                        seen.add((nr, nc))
                        stack.append((nr, nc))
    return count


def _extract_mask_and_sources(grid: np.ndarray) -> tuple[np.ndarray, list[tuple[int, int]]]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    h, w = g.shape
    assert 5 <= h <= 30 and 9 <= w <= 30, "grid must fit ARC limits and contain the 4x4 mask plus source area"

    mask = g[:MASK_SIZE, :MASK_SIZE]
    assert set(map(int, np.unique(mask))).issubset({BG, MASK_COLOR}), "top-left 4x4 area must be a binary grey/black mask"
    assert int(np.count_nonzero(mask == MASK_COLOR)) >= 4, "mask must contain at least four grey cells"

    assert np.all(g[1:MASK_SIZE, MASK_SIZE:] == BG), "rows 1..3 outside the mask must be black"
    assert np.all(g[MASK_SIZE:, :] == BG), "all rows below the source row and mask rows must start empty"
    assert np.all(g[:, MASK_SIZE] == BG), "column 4 is a spacer between the mask and the sources"

    source_cols = [c for c in range(MASK_SIZE + 1, w) if g[0, c] != BG]
    assert source_cols, "expected at least one source pixel on the top row"
    assert all((c - 5) % 2 == 0 for c in source_cols), "source pixels must sit on columns 5,7,9,..."
    assert all(int(g[0, c]) != MASK_COLOR for c in source_cols), "grey is reserved for the binary mask"
    assert len({int(g[0, c]) for c in source_cols}) == len(source_cols), "source colors must be distinct"

    top_row_allowed = set(source_cols) | set(range(MASK_SIZE))
    for c in range(w):
        if c in top_row_allowed:
            continue
        assert int(g[0, c]) == BG, "top row may contain only the mask area and separated source pixels"

    return mask == MASK_COLOR, [(c, int(g[0, c])) for c in source_cols]


def _oracle_output(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    mask, sources = _extract_mask_and_sources(g)

    out = np.zeros((h, w), dtype=int)
    out[:MASK_SIZE, :MASK_SIZE] = g[:MASK_SIZE, :MASK_SIZE]

    for c, color in sources:
        out[0, c] = color
        for r in range(1, h):
            if mask[r % MASK_SIZE, c % MASK_SIZE]:
                out[r, c] = color

    return out


def _solid_fill_output(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    _mask, sources = _extract_mask_and_sources(g)

    out = np.zeros((h, w), dtype=int)
    out[:MASK_SIZE, :MASK_SIZE] = g[:MASK_SIZE, :MASK_SIZE]
    for c, color in sources:
        out[:, c] = color
    out[:MASK_SIZE, :MASK_SIZE] = g[:MASK_SIZE, :MASK_SIZE]
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill each source column downward while a tiled 4x4 binary mask periodically gates which cells may pass.

    Input structure:
    - A mostly black grid contains a 4x4 grey/black binary mask in the top-left corner.
    - Row 0 contains several separated source pixels at columns 5,7,9,..., each with a distinct non-grey color.
    - Outside the mask block and those top-row source pixels, the input is black.

    Rule:
    - Interpret grey cells in the top-left 4x4 block as 1 and black cells there as 0, then tile that 4x4 mask periodically across the full canvas.
    - For every source pixel, propagate its color downward along its own column.
    - A target cell in that column is painted iff the tiled mask value at `(row mod 4, col mod 4)` is 1; otherwise it stays black.
    - Preserve the 4x4 mask block exactly, and keep each source pixel itself in its original color regardless of the mask.

    Key insight:
    - The final stripes are the intersection of two independent signals: source columns decide which color travels downward, while the repeated 4x4 mask decides exactly which rows in each column are open or blocked.
    """
    return _oracle_output(grid)


def _sample_connected_mask(rng: np.random.Generator, count: int) -> np.ndarray | None:
    mask = np.zeros((MASK_SIZE, MASK_SIZE), dtype=bool)
    start = (int(rng.integers(0, MASK_SIZE)), int(rng.integers(0, MASK_SIZE)))
    cells = {start}
    while len(cells) < count:
        frontier = []
        for r, c in sorted(cells):
            for nr, nc in _neighbors4(r, c):
                if 0 <= nr < MASK_SIZE and 0 <= nc < MASK_SIZE and (nr, nc) not in cells:
                    frontier.append((nr, nc))
        if not frontier:
            return None
        cells.add(frontier[int(rng.integers(0, len(frontier)))])
    for r, c in cells:
        mask[r, c] = True
    return mask


def _sample_disconnected_mask(rng: np.random.Generator, count: int) -> np.ndarray:
    mask = np.zeros((MASK_SIZE, MASK_SIZE), dtype=bool)
    idx = rng.choice(MASK_SIZE * MASK_SIZE, size=count, replace=False)
    mask.flat[idx] = True
    return mask


def _mask_is_semantic(mask: np.ndarray, require_connected: bool) -> bool:
    if mask.shape != (MASK_SIZE, MASK_SIZE):
        return False
    ones = int(mask.sum())
    if not (6 <= ones <= 10):
        return False

    components = _component_count(mask)
    if require_connected and components != 1:
        return False
    if not require_connected and components < 2:
        return False

    phase1 = tuple(int(v) for v in mask[:, 1])
    phase3 = tuple(int(v) for v in mask[:, 3])
    if phase1 == phase3:
        return False
    if sum(phase1) in {0, 4} or sum(phase3) in {0, 4}:
        return False

    row_patterns = {tuple(int(v) for v in row) for row in mask}
    col_patterns = {tuple(int(v) for v in mask[:, c]) for c in range(MASK_SIZE)}
    if len(row_patterns) < 2 or len(col_patterns) < 2:
        return False

    return True


def _sample_mask(rng: np.random.Generator) -> np.ndarray:
    require_connected = bool(rng.integers(0, 2))
    for _ in range(200):
        count = int(rng.integers(6, 11))
        if require_connected:
            mask = _sample_connected_mask(rng, count)
            if mask is None:
                continue
        else:
            mask = _sample_disconnected_mask(rng, count)
        if _mask_is_semantic(mask, require_connected=require_connected):
            return mask
    raise RuntimeError("failed to sample a non-degenerate 4x4 mask")


def _allowed_source_columns(width: int) -> list[int]:
    return [c for c in range(5, width, 2)]


def _choose_source_columns(rng: np.random.Generator, width: int) -> list[int]:
    allowed = _allowed_source_columns(width)
    assert len(allowed) >= 4

    phase1 = [c for c in allowed if c % 4 == 1]
    phase3 = [c for c in allowed if c % 4 == 3]
    cols = [
        phase1[int(rng.integers(0, len(phase1)))],
        phase3[int(rng.integers(0, len(phase3)))],
    ]

    remaining = [c for c in allowed if c not in cols]
    count = 4 if len(allowed) == 4 else int(rng.integers(4, min(5, len(allowed)) + 1))
    if count > len(cols):
        extra = rng.choice(np.array(remaining, dtype=int), size=count - len(cols), replace=False)
        cols.extend(int(v) for v in extra)
    cols.sort()
    return cols


def _build_input(mask: np.ndarray, height: int, width: int, columns: list[int], colors: list[int]) -> np.ndarray:
    grid = np.zeros((height, width), dtype=int)
    grid[:MASK_SIZE, :MASK_SIZE] = np.where(mask, MASK_COLOR, BG)
    for c, color in zip(columns, colors):
        grid[0, c] = color
    return grid


def _column_pattern(mask: np.ndarray, height: int, col: int) -> tuple[int, ...]:
    return tuple(int(mask[r % MASK_SIZE, col % MASK_SIZE]) for r in range(1, height))


def _semantic_checks(inp: np.ndarray, out: np.ndarray) -> bool:
    g = np.asarray(inp, dtype=int)
    h, _w = g.shape
    mask, sources = _extract_mask_and_sources(g)

    if not np.array_equal(out, _oracle_output(g)):
        return False
    if not np.array_equal(out[:MASK_SIZE, :MASK_SIZE], g[:MASK_SIZE, :MASK_SIZE]):
        return False
    if np.array_equal(out, _solid_fill_output(g)):
        return False

    phases = {c % MASK_SIZE for c, _color in sources}
    if phases != {1, 3}:
        return False
    if _column_pattern(mask, h, 5) == _column_pattern(mask, h, 7):
        return False

    for c, color in sources:
        trail = out[1:, c]
        if not np.any(trail == color):
            return False
        if not np.any(trail == BG):
            return False
        expected = np.array([color if mask[r % MASK_SIZE, c % MASK_SIZE] else BG for r in range(1, h)], dtype=int)
        if not np.array_equal(trail, expected):
            return False

    used_cols = {c for c, _color in sources}
    for c in range(g.shape[1]):
        if c in used_cols:
            continue
        if c < MASK_SIZE:
            if not np.all(out[MASK_SIZE:, c] == BG):
                return False
        else:
            if not np.all(out[:, c] == BG):
                return False

    return True


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()

    for _ in range(MIN_GENERATION_ATTEMPTS):
        height = int(rng.integers(16, 25))
        width = int(rng.integers(14, 25))
        mask = _sample_mask(rng)
        columns = _choose_source_columns(rng, width)
        colors = [int(v) for v in rng.choice(np.array(SOURCE_COLORS, dtype=int), size=len(columns), replace=False)]
        inp = _build_input(mask, height, width, columns, colors)
        out = _oracle_output(inp)
        if _semantic_checks(inp, out):
            return inp

    raise RuntimeError("failed to generate a non-degenerate periodic-mask column-fill input")
