from __future__ import annotations

import random
from collections import Counter, deque

import numpy as np


BACKGROUND = 0
PATTERN_COLOR = 1
SEPARATOR_COLOR = 9

CELL_COUNT_CHOICES = (3, 5)
CELL_SIZE_CHOICES = (4, 5)
PATTERN_AREA_RANGE = (3, 6)
GENERATION_RETRIES = 500
MODE_SEQUENCE = ["quadrant", "axis_row", "axis_col"]

_seen_output_signatures: set[tuple[tuple[int, int], bytes]] = set()
_mode_cursor = 0


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def _sign(value: int) -> int:
    if value < 0:
        return -1
    if value > 0:
        return 1
    return 0


def _regions_between(size: int, separator_lines: list[int]) -> list[tuple[int, int]]:
    regions: list[tuple[int, int]] = []
    start = 0
    for line in separator_lines:
        assert start < line < size - 1, "separator lines must split non-empty regions"
        regions.append((start, line - 1))
        start = line + 1
    assert start < size, "expected a trailing region after the final separator"
    regions.append((start, size - 1))
    return regions


def _line_layout(grid: np.ndarray) -> tuple[int, list[tuple[int, int]], list[tuple[int, int]], int]:
    g = np.asarray(grid, dtype=int)
    bg = _get_bg(g)
    h, w = g.shape

    candidates: list[tuple[int, list[int], list[int]]] = []
    for color in np.unique(g):
        color = int(color)
        if color == bg:
            continue
        rows = [r for r in range(h) if np.all(g[r, :] == color)]
        cols = [c for c in range(w) if np.all(g[:, c] == color)]
        if rows and cols:
            candidates.append((color, rows, cols))

    assert len(candidates) == 1, "expected exactly one separator color spanning rows and columns"
    separator, row_lines, col_lines = candidates[0]
    row_regions = _regions_between(h, row_lines)
    col_regions = _regions_between(w, col_lines)

    row_sizes = {r1 - r0 + 1 for r0, r1 in row_regions}
    col_sizes = {c1 - c0 + 1 for c0, c1 in col_regions}
    assert len(row_sizes) == 1 and len(col_sizes) == 1, "all cells must have equal size"
    assert len(row_regions) % 2 == 1 and len(col_regions) % 2 == 1, "cell grid must have odd counts"
    return separator, row_regions, col_regions, bg


def _pattern_forms(block: np.ndarray) -> list[np.ndarray]:
    return [
        np.asarray(block, dtype=int),
        np.fliplr(block).copy(),
        np.flipud(block).copy(),
        np.rot90(block, 2).copy(),
    ]


def _is_4_connected(mask: np.ndarray) -> bool:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return False
    start = tuple(map(int, coords[0]))
    seen = {start}
    queue: deque[tuple[int, int]] = deque([start])
    while queue:
        r, c = queue.popleft()
        for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
            if 0 <= nr < mask.shape[0] and 0 <= nc < mask.shape[1] and mask[nr, nc] and (nr, nc) not in seen:
                seen.add((nr, nc))
                queue.append((nr, nc))
    return len(seen) == int(mask.sum())


def _validate_source_block(block: np.ndarray) -> None:
    mask = np.asarray(block == PATTERN_COLOR)
    area = int(mask.sum())
    assert PATTERN_COLOR in block, "expected a non-empty source pattern"
    assert area in range(PATTERN_AREA_RANGE[0], PATTERN_AREA_RANGE[1] + 1), "unexpected pattern area"
    assert _is_4_connected(mask), "source pattern must be 4-connected"

    rows = np.where(mask.any(axis=1))[0]
    cols = np.where(mask.any(axis=0))[0]
    assert len(rows) >= 2 and len(cols) >= 2, "pattern must span at least two rows and two cols"

    forms = _pattern_forms(block)
    assert len({form.tobytes() for form in forms}) == 4, "all four orientations must be distinguishable"


def _extract_source(grid: np.ndarray) -> tuple[np.ndarray, tuple[int, int], list[tuple[int, int]], list[tuple[int, int]], int, int]:
    separator, row_regions, col_regions, bg = _line_layout(grid)
    g = np.asarray(grid, dtype=int)
    populated: list[tuple[tuple[int, int], np.ndarray]] = []

    for rr, (r0, r1) in enumerate(row_regions):
        for cc, (c0, c1) in enumerate(col_regions):
            block = g[r0 : r1 + 1, c0 : c1 + 1]
            assert not np.any(block == separator), "separator color cannot appear inside a cell"
            non_bg = (block != bg)
            if np.any(non_bg):
                assert np.all(block[non_bg] == PATTERN_COLOR), "only the blue pattern may populate a cell"
                populated.append(((rr, cc), block.copy()))

    assert len(populated) == 1, "expected exactly one populated cell"
    source_cell, source_block = populated[0]
    rc = len(row_regions) // 2
    cc = len(col_regions) // 2
    assert source_cell != (rc, cc), "the source cell cannot be the center cell"
    _validate_source_block(source_block)
    return source_block, source_cell, row_regions, col_regions, separator, bg


def _transform_for_cell(block: np.ndarray, source_cell: tuple[int, int], target_cell: tuple[int, int], center: tuple[int, int]) -> np.ndarray:
    src_dr = _sign(source_cell[0] - center[0])
    src_dc = _sign(source_cell[1] - center[1])
    dst_dr = _sign(target_cell[0] - center[0])
    dst_dc = _sign(target_cell[1] - center[1])

    out = block.copy()
    if src_dr != 0 and dst_dr == -src_dr:
        out = np.flipud(out)
    if src_dc != 0 and dst_dc == -src_dc:
        out = np.fliplr(out)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """Tile the unique populated cell into every separator-delimited cell, mirroring by quadrant relative to the grid center and the source cell.

    Input structure:
    - The grid has black background and maroon full separator rows and columns that split it into equal rectangular cells.
    - The cell grid has odd row and column counts, so there is a unique center cell.
    - Exactly one non-center cell contains a blue 4-connected asymmetric pattern; every other cell is empty.

    Rule:
    - Detect the separator lattice and locate the single populated source cell.
    - Copy that cell-sized pattern into every cell while keeping all separator lines unchanged.
    - For each destination cell, flip vertically if it lies on the opposite side of the center row from the source, flip horizontally if it lies on the opposite side of the center column from the source, and apply both flips when both sides differ.

    Key insight:
    - The same tile is repeated everywhere, but its orientation is determined by each cell's side of the separator-grid center relative to the source cell, so tiling and mirroring are coupled through the coarse cell coordinates.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"

    source_block, source_cell, row_regions, col_regions, separator, _bg = _extract_source(g)
    center = (len(row_regions) // 2, len(col_regions) // 2)

    out = g.copy()
    for rr, (r0, r1) in enumerate(row_regions):
        for cc, (c0, c1) in enumerate(col_regions):
            out[r0 : r1 + 1, c0 : c1 + 1] = _transform_for_cell(source_block, source_cell, (rr, cc), center)

    for r in range(out.shape[0]):
        if np.all(g[r, :] == separator):
            assert np.all(out[r, :] == separator)
    for c in range(out.shape[1]):
        if np.all(g[:, c] == separator):
            assert np.all(out[:, c] == separator)
    return out


def _grow_connected_coords(height: int, width: int, area: int) -> list[tuple[int, int]]:
    start = (random.randrange(height), random.randrange(width))
    coords = {start}
    while len(coords) < area:
        frontier = []
        for r, c in coords:
            for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                if 0 <= nr < height and 0 <= nc < width and (nr, nc) not in coords:
                    frontier.append((nr, nc))
        coords.add(random.choice(frontier))
    return sorted(coords)


def _generate_source_block(cell_h: int, cell_w: int) -> np.ndarray | None:
    for _ in range(200):
        block = np.zeros((cell_h, cell_w), dtype=int)
        area = random.randint(*PATTERN_AREA_RANGE)
        for r, c in _grow_connected_coords(cell_h, cell_w, area):
            block[r, c] = PATTERN_COLOR

        if block.sum() == 0:
            continue
        if block.tobytes() != min(form.tobytes() for form in _pattern_forms(block)):
            continue
        try:
            _validate_source_block(block)
        except AssertionError:
            continue
        return block
    return None


def _separator_lines(cell_count: int, cell_size: int) -> list[int]:
    return [i * cell_size + (i - 1) for i in range(1, cell_count)]


def _cell_origin(index: int, cell_size: int) -> int:
    return index * (cell_size + 1)


def _candidate_source_cells(rows: int, cols: int, mode: str) -> list[tuple[int, int]]:
    rc = rows // 2
    cc = cols // 2
    if mode == "axis_row":
        return [(rc, c) for c in range(cols) if c != cc]
    if mode == "axis_col":
        return [(r, cc) for r in range(rows) if r != rc]
    if mode == "quadrant":
        return [(r, c) for r in range(rows) for c in range(cols) if r != rc and c != cc]
    raise ValueError(f"unknown mode: {mode}")


def _infer_source_mode(source_cell: tuple[int, int], rows: int, cols: int) -> str:
    rc = rows // 2
    cc = cols // 2
    if source_cell[0] == rc:
        return "axis_row"
    if source_cell[1] == cc:
        return "axis_col"
    return "quadrant"


def _render_input(cell_rows: int, cell_cols: int, cell_h: int, cell_w: int, source_cell: tuple[int, int], source_block: np.ndarray) -> np.ndarray:
    height = cell_rows * cell_h + (cell_rows - 1)
    width = cell_cols * cell_w + (cell_cols - 1)
    grid = np.full((height, width), BACKGROUND, dtype=int)

    for r in _separator_lines(cell_rows, cell_h):
        grid[r, :] = SEPARATOR_COLOR
    for c in _separator_lines(cell_cols, cell_w):
        grid[:, c] = SEPARATOR_COLOR

    r0 = _cell_origin(source_cell[0], cell_h)
    c0 = _cell_origin(source_cell[1], cell_w)
    grid[r0 : r0 + cell_h, c0 : c0 + cell_w] = source_block
    return grid


def _oracle_accepts(inp: np.ndarray, expected_mode: str) -> bool:
    try:
        source_block, source_cell, row_regions, col_regions, separator, bg = _extract_source(inp)
        out = main(inp)
    except AssertionError:
        return False

    if bg != BACKGROUND or separator != SEPARATOR_COLOR:
        return False
    if _infer_source_mode(source_cell, len(row_regions), len(col_regions)) != expected_mode:
        return False
    if out.shape != inp.shape or max(inp.shape) > 30 or max(out.shape) > 30:
        return False

    center = (len(row_regions) // 2, len(col_regions) // 2)
    seen_forms: set[bytes] = set()
    for rr, (r0, r1) in enumerate(row_regions):
        for cc, (c0, c1) in enumerate(col_regions):
            block = out[r0 : r1 + 1, c0 : c1 + 1]
            expect = _transform_for_cell(source_block, source_cell, (rr, cc), center)
            if not np.array_equal(block, expect):
                return False
            seen_forms.add(block.tobytes())

    if expected_mode == "quadrant":
        required_forms = {form.tobytes() for form in _pattern_forms(source_block)}
    elif expected_mode == "axis_row":
        required_forms = {source_block.tobytes(), np.fliplr(source_block).copy().tobytes()}
    else:
        required_forms = {source_block.tobytes(), np.flipud(source_block).copy().tobytes()}
    if not required_forms.issubset(seen_forms):
        return False
    return True


def _gen_attempt(mode: str) -> np.ndarray | None:
    cell_rows = random.choice(CELL_COUNT_CHOICES)
    cell_cols = random.choice(CELL_COUNT_CHOICES)
    cell_h = random.choice(CELL_SIZE_CHOICES)
    cell_w = random.choice(CELL_SIZE_CHOICES)

    candidates = _candidate_source_cells(cell_rows, cell_cols, mode)
    source_cell = random.choice(candidates)

    source_block = _generate_source_block(cell_h, cell_w)
    if source_block is None:
        return None

    inp = _render_input(cell_rows, cell_cols, cell_h, cell_w, source_cell, source_block)
    if not _oracle_accepts(inp, mode):
        return None
    return inp


def _next_mode() -> str:
    global _mode_cursor
    mode = MODE_SEQUENCE[_mode_cursor % len(MODE_SEQUENCE)]
    _mode_cursor += 1
    return mode


def generate_input() -> np.ndarray:
    target_mode = _next_mode()
    for _ in range(GENERATION_RETRIES):
        grid = _gen_attempt(target_mode)
        if grid is None:
            continue
        out = main(grid)
        signature = (tuple(map(int, out.shape)), out.tobytes())
        if signature in _seen_output_signatures:
            continue
        _seen_output_signatures.add(signature)
        return grid
    raise RuntimeError("failed to generate a valid input for puzzle 6")
