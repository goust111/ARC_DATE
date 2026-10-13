import random
from collections import Counter

import numpy as np


TEAL = 8
MAGENTA = 6
GREEN = 3
RED = 2
CELL = 5


def _mode_color(grid: np.ndarray) -> int:
    counts = Counter(map(int, grid.reshape(-1)))
    return counts.most_common(1)[0][0]


def _separator_rows(grid: np.ndarray, bg: int) -> tuple[int, int, int]:
    rows: list[int] = []
    colors: list[int] = []
    for r in range(grid.shape[0]):
        row_color = int(grid[r, 0])
        if row_color != bg and np.all(grid[r] == row_color):
            rows.append(r)
            colors.append(row_color)

    assert len(rows) == 2, "expected two full-width separator rows"
    assert colors[0] == colors[1], "separator rows must have the same color"
    first, second = rows
    assert first > 0 and second > first + 1 and second < grid.shape[0] - 1
    return first, second, colors[0]


def _glyph_signature(tile: np.ndarray, bg: int) -> tuple[int, tuple[tuple[bool, ...], ...]] | None:
    assert tile.shape == (CELL, CELL)
    colors = [int(v) for v in np.unique(tile) if int(v) != bg]
    if not colors:
        return None

    assert len(colors) == 1, "each glyph cell must use one foreground color"
    color = colors[0]

    outer = np.ones((CELL, CELL), dtype=bool)
    outer[1:4, 1:4] = False
    assert np.all(tile[outer] == bg), "glyph color must be confined to the 3x3 interior"

    inner = tile[1:4, 1:4]
    assert np.all((inner == bg) | (inner == color))
    shape = inner == color
    assert bool(np.any(shape)), "glyph interior shape may not be empty"
    return color, tuple(tuple(bool(x) for x in row) for row in shape)


def _parse_reference(
    band: np.ndarray, bg: int
) -> tuple[dict[tuple[int, tuple[tuple[bool, ...], ...]], tuple[int, int]], int, int]:
    h, w = band.shape
    assert h % CELL == 0 and w % CELL == 0
    rows, cols = h // CELL, w // CELL
    assert rows >= 2 and cols >= 2

    signatures: dict[tuple[int, tuple[tuple[bool, ...], ...]], tuple[int, int]] = {}
    for r in range(rows):
        for c in range(cols):
            tile = band[r * CELL : (r + 1) * CELL, c * CELL : (c + 1) * CELL]
            sig = _glyph_signature(tile, bg)
            assert sig is not None, "every reference-table cell must contain a glyph"
            assert sig not in signatures, "reference glyph signatures must be unique"
            signatures[sig] = (r, c)
    return signatures, rows, cols


def _query_signatures(
    band: np.ndarray, bg: int
) -> list[tuple[int, tuple[tuple[bool, ...], ...]]]:
    h, w = band.shape
    assert h % CELL == 0 and w % CELL == 0

    out = []
    for r in range(h // CELL):
        for c in range(w // CELL):
            tile = band[r * CELL : (r + 1) * CELL, c * CELL : (c + 1) * CELL]
            sig = _glyph_signature(tile, bg)
            if sig is not None:
                out.append(sig)
    assert out, "query band must contain at least one glyph"
    return out


def _forms_complete_line(positions: set[tuple[int, int]], rows: int, cols: int) -> bool:
    if not positions:
        return False

    rs = {r for r, _ in positions}
    cs = {c for _, c in positions}
    full_row = len(positions) == cols and len(rs) == 1 and cs == set(range(cols))
    full_col = len(positions) == rows and len(cs) == 1 and rs == set(range(rows))
    return full_row or full_col


def main(grid: np.ndarray) -> np.ndarray:
    """
    Mark the looked-up reference glyph cells and answer whether they form one complete row or column.

    Input structure:
    - A Teal background grid is split into three horizontal bands by two full-width Magenta separator rows.
    - The top band is a query row made of 5x5 glyph cells; a glyph is identified by its color and its 3x3 interior mask.
    - The middle band is a rectangular table of the same 5x5 glyph cells with unique color-and-shape signatures.
    - The bottom band is an initially blank Teal judgement area.

    Rule:
    - Match every query glyph to the reference-table cell with the same foreground color and 3x3 interior shape.
    - Fill the 3x3 interior of each matched reference cell with Green.
    - If the matched cells are exactly all cells of one reference-table row or one reference-table column, fill the query and judgement bands with Green.
    - Otherwise leave the query band unchanged and fill the judgement band with Red.

    Key insight:
    - The lookup positions are only an intermediate result; the bottom band encodes the yes/no test of whether those positions make a complete aligned row or column.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0

    bg = _mode_color(g)
    sep_a, sep_b, _sep_color = _separator_rows(g, bg)

    query = g[:sep_a, :]
    reference = g[sep_a + 1 : sep_b, :]
    judgement = g[sep_b + 1 :, :]
    assert judgement.size > 0 and np.all(judgement == bg)

    ref_signatures, table_rows, table_cols = _parse_reference(reference, bg)
    query_sigs = _query_signatures(query, bg)

    matched: set[tuple[int, int]] = set()
    for sig in query_sigs:
        assert sig in ref_signatures, "each query glyph must appear in the reference table"
        pos = ref_signatures[sig]
        assert pos not in matched, "query glyphs must refer to distinct reference cells"
        matched.add(pos)

    out = g.copy()
    ref_r0 = sep_a + 1
    for r, c in matched:
        rr = ref_r0 + r * CELL + 1
        cc = c * CELL + 1
        out[rr : rr + 3, cc : cc + 3] = GREEN

    if _forms_complete_line(matched, table_rows, table_cols):
        out[:sep_a, :] = GREEN
        out[sep_b + 1 :, :] = GREEN
    else:
        out[sep_b + 1 :, :] = RED
    return out


# --------------- generator ---------------

GLYPH_COLORS = (0, 1, 4, 5, 7, 9)
TOP_HEIGHT = CELL
BOTTOM_HEIGHT_RANGE = (2, 3)
TABLE_ROWS_RANGE = (2, 4)
TABLE_COLS_RANGE = (3, 6)
MASK_SIZE_RANGE = (2, 6)
GENERATION_ATTEMPTS = 300


def _neighbors3(r: int, c: int) -> list[tuple[int, int]]:
    out = []
    for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        rr, cc = r + dr, c + dc
        if 0 <= rr < 3 and 0 <= cc < 3:
            out.append((rr, cc))
    return out


def _random_mask() -> tuple[tuple[bool, ...], ...]:
    target = random.randint(*MASK_SIZE_RANGE)
    starts = [(1, 1), (1, 1), (1, 1), *[(r, c) for r in range(3) for c in range(3)]]
    cells = {random.choice(starts)}

    while len(cells) < target:
        frontier = sorted({nb for r, c in cells for nb in _neighbors3(r, c)} - cells)
        cells.add(random.choice(frontier))

    mask = np.zeros((3, 3), dtype=bool)
    for r, c in cells:
        mask[r, c] = True

    mask = np.rot90(mask, random.randrange(4))
    if random.random() < 0.5:
        mask = np.fliplr(mask)
    if random.random() < 0.5:
        mask = np.flipud(mask)
    return tuple(tuple(bool(x) for x in row) for row in mask)


def _make_signatures(count: int) -> list[tuple[int, tuple[tuple[bool, ...], ...]]] | None:
    signatures: list[tuple[int, tuple[tuple[bool, ...], ...]]] = []
    seen = set()
    for _ in range(count):
        for _attempt in range(500):
            sig = (random.choice(GLYPH_COLORS), _random_mask())
            if sig not in seen:
                signatures.append(sig)
                seen.add(sig)
                break
        else:
            return None
    return signatures


def _tile(sig: tuple[int, tuple[tuple[bool, ...], ...]]) -> np.ndarray:
    color, mask_tuple = sig
    mask = np.array(mask_tuple, dtype=bool)
    tile = np.full((CELL, CELL), TEAL, dtype=int)
    tile[1:4, 1:4][mask] = color
    return tile


def _paste(grid: np.ndarray, patch: np.ndarray, r: int, c: int) -> None:
    grid[r : r + patch.shape[0], c : c + patch.shape[1]] = patch


def _sample_non_line_positions(rows: int, cols: int) -> list[tuple[int, int]] | None:
    all_positions = [(r, c) for r in range(rows) for c in range(cols)]
    for _ in range(500):
        count = random.randint(2, cols)
        chosen = set(random.sample(all_positions, count))
        if not _forms_complete_line(chosen, rows, cols):
            return list(chosen)
    return None


def _expected_output(inp: np.ndarray, rows: int, cols: int, positions: list[tuple[int, int]]) -> np.ndarray:
    out = inp.copy()
    sep_a = TOP_HEIGHT
    sep_b = TOP_HEIGHT + 1 + rows * CELL
    for r, c in set(positions):
        rr = sep_a + 1 + r * CELL + 1
        cc = c * CELL + 1
        out[rr : rr + 3, cc : cc + 3] = GREEN

    if _forms_complete_line(set(positions), rows, cols):
        out[:sep_a, :] = GREEN
        out[sep_b + 1 :, :] = GREEN
    else:
        out[sep_b + 1 :, :] = RED
    return out


def _gen_attempt() -> np.ndarray | None:
    rows = random.randint(*TABLE_ROWS_RANGE)
    cols = random.randint(max(TABLE_COLS_RANGE[0], rows), TABLE_COLS_RANGE[1])
    bottom_h = random.randint(*BOTTOM_HEIGHT_RANGE)

    height = TOP_HEIGHT + 1 + rows * CELL + 1 + bottom_h
    width = cols * CELL
    if height > 30 or width > 30:
        return None

    signatures = _make_signatures(rows * cols)
    if signatures is None:
        return None

    sig_by_pos = {
        (r, c): signatures[r * cols + c]
        for r in range(rows)
        for c in range(cols)
    }

    aligned = random.random() < 0.5
    if aligned:
        if random.random() < 0.5:
            rr = random.randrange(rows)
            positions = [(rr, c) for c in range(cols)]
        else:
            cc = random.randrange(cols)
            positions = [(r, cc) for r in range(rows)]
    else:
        positions = _sample_non_line_positions(rows, cols)
        if positions is None:
            return None

    grid = np.full((height, width), TEAL, dtype=int)
    sep_a = TOP_HEIGHT
    ref_r0 = sep_a + 1
    sep_b = ref_r0 + rows * CELL
    grid[sep_a, :] = MAGENTA
    grid[sep_b, :] = MAGENTA

    for r in range(rows):
        for c in range(cols):
            _paste(grid, _tile(sig_by_pos[(r, c)]), ref_r0 + r * CELL, c * CELL)

    query_sigs = [sig_by_pos[pos] for pos in positions]
    random.shuffle(query_sigs)
    slots = random.sample(range(cols), len(query_sigs))
    for slot, sig in zip(slots, query_sigs):
        _paste(grid, _tile(sig), 0, slot * CELL)

    expected = _expected_output(grid, rows, cols, positions)
    try:
        actual = main(grid)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for seed2047415201/7/1")
