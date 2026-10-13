from __future__ import annotations

import numpy as np


BLACK = 0
GREY = 5


def _parse_barcode_row(row: np.ndarray) -> list[tuple[int, int]]:
    """Extract contiguous color runs from one barcode row, ignoring outer black padding."""
    active = np.flatnonzero(row != BLACK)
    assert active.size > 0, "each active row must contain a barcode"

    start = int(active[0])
    end = int(active[-1])
    assert bool(np.all(row[start : end + 1] != BLACK)), "barcode rows cannot contain internal black gaps"

    segments: list[tuple[int, int]] = []
    c = start
    while c <= end:
        color = int(row[c])
        d = c + 1
        while d <= end and int(row[d]) == color:
            d += 1
        segments.append((color, d - c))
        c = d

    assert segments, "expected at least one segment"
    return segments


def _extract_barcodes(grid: np.ndarray) -> tuple[list[int], np.ndarray]:
    """Return the shared segment colors and the per-row segment lengths."""
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"

    rows = [g[r] for r in range(g.shape[0]) if np.any(g[r] != BLACK)]
    assert rows, "expected at least one non-empty barcode row"

    parsed = [_parse_barcode_row(row) for row in rows]
    seg_count = len(parsed[0])
    colors = [int(color) for color, _length in parsed[0]]

    for segments in parsed[1:]:
        assert len(segments) == seg_count, "every row must have the same number of segments"
        assert [int(color) for color, _length in segments] == colors, "segment colors must align by position"

    lengths = np.array([[int(length) for _color, length in segments] for segments in parsed], dtype=int)
    assert bool(np.all(lengths > 0))
    return colors, lengths


def _render_output(colors: list[int], lengths: np.ndarray) -> np.ndarray:
    """Stack each column's row-specific heights bottom-up with grey separators."""
    n_rows, n_cols = map(int, lengths.shape)
    col_heights = lengths.sum(axis=0) + max(0, n_rows - 1)
    out_height = int(col_heights.max())
    out = np.full((out_height, n_cols), BLACK, dtype=int)

    for col, color in enumerate(colors):
        cursor = out_height - 1
        for row_idx in range(n_rows):
            block_height = int(lengths[row_idx, col])
            out[cursor - block_height + 1 : cursor + 1, col] = int(color)
            cursor -= block_height
            if row_idx != n_rows - 1:
                out[cursor, col] = GREY
                cursor -= 1

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Decode row-wise color barcodes into stacked vertical columns.

    Input structure
    - Background is black.
    - Each non-empty row contains one complete barcode: contiguous horizontal
      color segments with no internal gaps.
    - Every row has the same number of segments, and segment colors match by
      position while segment lengths may differ.

    Rule
    - Read the shared segment colors and the per-row segment lengths.
    - Build one output column per segment position.
    - In each output column, stack the rows' segment lengths from bottom to top
      in input-row order.
    - Insert a single grey separator cell between consecutive row blocks.

    Key insight
    - The horizontal run length of a segment is reinterpreted as a vertical
      block height in the corresponding output column.
    """
    colors, lengths = _extract_barcodes(np.asarray(grid, dtype=int))
    return _render_output(colors, lengths)


# --------------- generator ---------------

_RNG = np.random.default_rng()

GEN_MIN_ROWS = 2
GEN_MAX_ROWS = 5
GEN_MIN_SEGMENTS = 2
GEN_MAX_SEGMENTS = 5
GEN_MIN_LENGTH = 1
GEN_MAX_LENGTH = 6
GEN_MAX_EXTRA_RIGHT_PAD = 3
GEN_RETRIES = 300
GEN_PALETTE = tuple(color for color in range(10) if color not in (BLACK, GREY))


def _render_input(colors: list[int], lengths: np.ndarray, extra_right_pad: int) -> np.ndarray:
    row_widths = lengths.sum(axis=1)
    width = int(row_widths.max()) + int(extra_right_pad)
    grid = np.full((lengths.shape[0], width), BLACK, dtype=int)

    for row_idx in range(lengths.shape[0]):
        cursor = 0
        for col_idx, color in enumerate(colors):
            block_width = int(lengths[row_idx, col_idx])
            grid[row_idx, cursor : cursor + block_width] = int(color)
            cursor += block_width

    return grid


def _sample_lengths(n_rows: int, n_cols: int) -> np.ndarray:
    lengths = _RNG.integers(GEN_MIN_LENGTH, GEN_MAX_LENGTH + 1, size=(n_rows, n_cols))
    if np.all(lengths == lengths[0]):
        r = int(_RNG.integers(0, n_rows))
        c = int(_RNG.integers(0, n_cols))
        choices = [v for v in range(GEN_MIN_LENGTH, GEN_MAX_LENGTH + 1) if v != int(lengths[r, c])]
        lengths[r, c] = int(_RNG.choice(choices))
    return lengths.astype(int, copy=False)


def _valid_lengths(lengths: np.ndarray) -> bool:
    if len({tuple(map(int, row)) for row in lengths}) < 2:
        return False
    varying_columns = sum(len(set(map(int, lengths[:, c]))) > 1 for c in range(lengths.shape[1]))
    if varying_columns == 0:
        return False
    return True


def _gen_attempt() -> np.ndarray | None:
    n_rows = int(_RNG.integers(GEN_MIN_ROWS, GEN_MAX_ROWS + 1))
    n_cols = int(_RNG.integers(GEN_MIN_SEGMENTS, GEN_MAX_SEGMENTS + 1))
    colors = _RNG.choice(np.array(GEN_PALETTE, dtype=int), size=n_cols, replace=False).tolist()
    lengths = _sample_lengths(n_rows, n_cols)
    if not _valid_lengths(lengths):
        return None

    grid = _render_input(colors, lengths, extra_right_pad=int(_RNG.integers(0, GEN_MAX_EXTRA_RIGHT_PAD + 1)))
    expected = _render_output(colors, lengths)
    if not np.array_equal(main(grid), expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid barcode grid")
