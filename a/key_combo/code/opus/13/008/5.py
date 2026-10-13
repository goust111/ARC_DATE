import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
TEAL = 8
MAROON = 9

HEADER_COLOR_POOL = (BLUE, RED, GREEN, TEAL, MAROON)
GRID_HEIGHT_RANGE = (12, 16)
GRID_WIDTH_RANGE = (12, 16)
ROW_COUNT_OPTIONS = (2, 3)
COLUMN_SEGMENT_PROB = 0.55
MAX_GENERATION_ATTEMPTS = 400


def _extract_header(top_row: np.ndarray) -> list[int]:
    header = [int(value) for value in top_row if int(value) != YELLOW]
    assert len(header) >= 3, "expected a nontrivial top-row header"
    assert all(color != BLACK for color in header), "header cannot contain black"
    distinct = set(header)
    assert 3 <= len(distinct) <= 5, "header must use 3-5 colors"
    return header


def _detect_segments(grid: np.ndarray) -> tuple[list[int], list[int]]:
    h, w = map(int, grid.shape)
    rows = [r for r in range(1, h - 1) if int(grid[r, 0]) == YELLOW and int(grid[r, w - 1]) == YELLOW]
    cols = [c for c in range(1, w - 1) if int(grid[0, c]) == YELLOW and int(grid[h - 1, c]) == YELLOW]

    assert 2 <= len(rows) <= 3, "expected 2-3 row segments"
    assert len(cols) <= 1, "expected at most one column segment"

    expected_markers = {(r, 0) for r in rows}
    expected_markers |= {(r, w - 1) for r in rows}
    expected_markers |= {(0, c) for c in cols}
    expected_markers |= {(h - 1, c) for c in cols}

    actual_markers = {(int(r), int(c)) for r, c in np.argwhere(grid == YELLOW)}
    assert actual_markers == expected_markers, "yellow pixels must be exactly the edge markers"
    return rows, cols


def _segment_colors(values: np.ndarray) -> set[int]:
    colors = {int(value) for value in values}
    assert YELLOW not in colors and BLACK not in colors, "segment interiors must be non-black and non-yellow"
    assert len(colors) == 2, "each segment interior must use exactly two colors"
    return colors


def _swap_colors_in_place(values: np.ndarray, color_a: int, color_b: int) -> None:
    mask_a = values == int(color_a)
    mask_b = values == int(color_b)
    values[mask_a] = int(color_b)
    values[mask_b] = int(color_a)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Swap colors along opposite-edge segments using cyclic pairs read from the top header.

    Input structure:
    - The background is black.
    - The top row is a non-black header sequence, except that an optional yellow top-edge marker may replace one header cell.
    - Yellow pixels appear only as opposite-edge markers: 2-3 left/right row pairs and optionally 1 top/bottom column pair.
    - Each marked row or column segment has a strict interior filled with exactly two non-black colors drawn from the header palette.

    Rule:
    - Remove the yellow marker cells from the top row and read the remaining header sequence left-to-right.
    - List all marked row segments by increasing row, then any marked column segment by increasing column.
    - For segment i, take header colors at cyclic positions (2i mod L) and ((2i+1) mod L) to form that segment's swap pair.
    - Process the segments in that order, swapping only those two colors inside the strict interior of the segment; keep markers, the header, and all other cells unchanged.

    Key insight:
    - The yellow markers tell you where the active segments are, while the top header tells you which two colors each segment should swap; the segment order bridges those two clues.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    h, w = map(int, g.shape)
    assert h >= 4 and w >= 4, "grid is too small for edge segments"

    header = _extract_header(g[0])
    header_colors = set(header)
    rows, cols = _detect_segments(g)

    for r in rows:
        interior_colors = _segment_colors(g[r, 1:-1])
        assert interior_colors <= header_colors, "row segment uses colors outside the header palette"
    for c in cols:
        interior_colors = _segment_colors(g[1:-1, c])
        assert interior_colors <= header_colors, "column segment uses colors outside the header palette"

    segments = [("row", r) for r in rows] + [("col", c) for c in cols]
    out = g.copy()
    header_len = len(header)

    for i, (axis, index) in enumerate(segments):
        color_a = int(header[(2 * i) % header_len])
        color_b = int(header[(2 * i + 1) % header_len])
        if axis == "row":
            _swap_colors_in_place(out[index, 1:-1], color_a, color_b)
        else:
            _swap_colors_in_place(out[1:-1, index], color_a, color_b)

    return out


def _build_two_color_pattern(
    length: int,
    color_a: int,
    color_b: int,
    rng: np.random.Generator,
    forced: dict[int, int] | None = None,
) -> np.ndarray:
    forced = forced or {}
    assert length >= 2, "two-color pattern needs room for both colors"

    max_runs = min(6, length)
    run_count = int(rng.integers(3, max_runs + 1)) if length >= 3 else 2
    cut_count = run_count - 1
    if cut_count > 0:
        cuts = sorted(int(x) for x in rng.choice(np.arange(1, length), size=cut_count, replace=False))
    else:
        cuts = []

    segments = []
    start = 0
    for stop in cuts + [length]:
        segments.append(stop - start)
        start = stop

    start_color = int(color_a if rng.random() < 0.5 else color_b)
    alt = [start_color, int(color_b if start_color == color_a else color_a)]
    values = np.empty(length, dtype=int)
    pos = 0
    for i, seg_len in enumerate(segments):
        values[pos : pos + seg_len] = alt[i % 2]
        pos += seg_len

    for idx, color in forced.items():
        values[int(idx)] = int(color)

    present = {int(v) for v in values}
    if present != {int(color_a), int(color_b)}:
        missing = int(color_a if color_a not in present else color_b)
        fallback = [i for i in range(length) if i not in forced]
        target = fallback[0] if fallback else 0
        values[int(target)] = missing

    assert {int(v) for v in values} == {int(color_a), int(color_b)}
    return values


def _build_header_sequence(segment_pairs: list[tuple[int, int]], header_len: int) -> list[int]:
    base: list[int] = []
    for color_a, color_b in segment_pairs:
        base.extend([int(color_a), int(color_b)])
    assert len(base) >= 4
    return [int(base[i % len(base)]) for i in range(header_len)]


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    h = int(rng.integers(GRID_HEIGHT_RANGE[0], GRID_HEIGHT_RANGE[1] + 1))
    w = int(rng.integers(GRID_WIDTH_RANGE[0], GRID_WIDTH_RANGE[1] + 1))

    row_count = int(rng.choice(ROW_COUNT_OPTIONS))
    has_col = bool(rng.random() < COLUMN_SEGMENT_PROB)
    col_count = 1 if has_col else 0
    segment_count = row_count + col_count

    distinct_needed = segment_count + 1
    palette = [int(c) for c in rng.choice(np.array(HEADER_COLOR_POOL), size=distinct_needed, replace=False)]
    anchor = palette[0]
    partners = palette[1:]
    segment_pairs = [(anchor, partner) for partner in partners]

    header_len = w - col_count
    if header_len < 2 * segment_count:
        return None

    grid = np.zeros((h, w), dtype=int)

    row_choices = np.arange(1, h - 1)
    rows = sorted(int(r) for r in rng.choice(row_choices, size=row_count, replace=False))
    col = None
    if has_col:
        col = int(rng.integers(1, w - 1))

    header = _build_header_sequence(segment_pairs, header_len)
    header_idx = 0
    for c in range(w):
        if col is not None and c == col:
            grid[0, c] = YELLOW
        else:
            grid[0, c] = int(header[header_idx])
            header_idx += 1

    for idx, r in enumerate(rows):
        grid[r, 0] = YELLOW
        grid[r, w - 1] = YELLOW
        color_a, color_b = segment_pairs[idx]
        forced = {}
        if col is not None:
            forced[col - 1] = color_a
        grid[r, 1:-1] = _build_two_color_pattern(w - 2, color_a, color_b, rng, forced=forced)

    if col is not None:
        grid[h - 1, col] = YELLOW
        color_a, color_b = segment_pairs[-1]
        forced = {r - 1: color_a for r in rows}
        column_values = _build_two_color_pattern(h - 2, color_a, color_b, rng, forced=forced)
        grid[1:-1, col] = column_values

    if col is not None:
        for r in rows:
            assert grid[r, col] == anchor, "row/column overlap must keep the shared anchor color"

    try:
        out = main(grid)
    except Exception:
        return None

    if np.array_equal(grid, out):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
