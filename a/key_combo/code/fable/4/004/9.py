from __future__ import annotations

import numpy as np


BACKGROUND_COLOR = 0
NON_BACKGROUND_COLORS = np.arange(1, 10, dtype=int)
MIN_K = 3
MAX_K = 29
DIRECTIONS = ((0, 1), (1, 0), (0, -1), (-1, 0))
GENERATION_RETRIES = 300

_RNG = np.random.default_rng()


def _spiral_cells(colors: np.ndarray) -> list[tuple[int, int, int, int]]:
    cells: list[tuple[int, int, int, int]] = []
    row, col = 0, -1
    occupied: set[tuple[int, int]] = set()

    for segment_index, color in enumerate(colors):
        dr, dc = DIRECTIONS[segment_index % len(DIRECTIONS)]
        length = segment_index + 1
        for _ in range(length):
            row += dr
            col += dc
            coord = (row, col)
            assert coord not in occupied, "spiral path should not overlap itself"
            occupied.add(coord)
            cells.append((row, col, int(color), segment_index))

    return cells


def _render_spiral(colors: np.ndarray) -> np.ndarray:
    cells = _spiral_cells(colors)
    rows = np.array([row for row, _col, _color, _seg in cells], dtype=int)
    cols = np.array([col for _row, col, _color, _seg in cells], dtype=int)
    row0, row1 = int(rows.min()), int(rows.max())
    col0, col1 = int(cols.min()), int(cols.max())

    out = np.full((row1 - row0 + 1, col1 - col0 + 1), BACKGROUND_COLOR, dtype=int)
    for row, col, color, _segment_index in cells:
        rr = row - row0
        cc = col - col0
        assert out[rr, cc] == BACKGROUND_COLOR
        out[rr, cc] = int(color)

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Unfold a single-column color seed into an increasing clockwise spiral polyline.

    Input structure:
    - The input is a K by 1 column with K >= 3.
    - Every entry is a non-background color, read from top to bottom as the segment
      color sequence.
    - The background color does not appear in the input seed.

    Rule:
    - Draw segment 1 to the right with length 1 and the first seed color.
    - For each following seed color, turn 90 degrees clockwise, increase the segment
      length by 1, and draw from the adjacent cell after the previous segment end.
    - Return the tight bounding rectangle of all drawn cells, with undrawn cells left
      as background.

    Key insight:
    - The column is only a generator for a new object: the hidden clockwise direction
      cycle and the increasing segment lengths determine both the spiral and its
      output canvas.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    height, width = map(int, g.shape)
    assert width == 1, "expected a single-column input"
    assert MIN_K <= height <= MAX_K, "seed length must fit the ARC output bound"
    assert np.all(g != BACKGROUND_COLOR), "seed colors must be non-background"

    colors = g[:, 0].astype(int)
    return _render_spiral(colors)


def _sample_color_sequence(length: int) -> np.ndarray:
    colors = np.empty(length, dtype=int)
    palette = NON_BACKGROUND_COLORS.copy()
    _RNG.shuffle(palette)

    colors[0] = int(palette[0])
    for i in range(1, length):
        choices = NON_BACKGROUND_COLORS[NON_BACKGROUND_COLORS != colors[i - 1]]
        colors[i] = int(_RNG.choice(choices))

    min_unique = min(length, 4)
    if len(set(map(int, colors))) < min_unique:
        return _sample_color_sequence(length)
    return colors


def _passes_semantic_checks(grid: np.ndarray) -> bool:
    try:
        if grid.ndim != 2 or grid.shape[1] != 1:
            return False
        if not (MIN_K <= grid.shape[0] <= MAX_K):
            return False
        colors = grid[:, 0].astype(int)
        if np.any(colors == BACKGROUND_COLOR):
            return False
        if np.any(colors[:-1] == colors[1:]):
            return False
        if len(set(map(int, colors))) < min(len(colors), 4):
            return False

        cells = _spiral_cells(colors)
        if len({(row, col) for row, col, _color, _seg in cells}) != len(cells):
            return False

        expected_count = len(colors) * (len(colors) + 1) // 2
        if len(cells) != expected_count:
            return False

        for segment_index in range(len(colors)):
            segment_cells = [cell for cell in cells if cell[3] == segment_index]
            if len(segment_cells) != segment_index + 1:
                return False
            dr, dc = DIRECTIONS[segment_index % len(DIRECTIONS)]
            for (row_a, col_a, _color_a, _seg_a), (row_b, col_b, _color_b, _seg_b) in zip(
                segment_cells, segment_cells[1:]
            ):
                if (row_b - row_a, col_b - col_a) != (dr, dc):
                    return False
            if any(color != int(colors[segment_index]) for _row, _col, color, _seg in segment_cells):
                return False

        out = main(grid)
        if out.ndim != 2 or max(out.shape) > 30:
            return False
        return np.count_nonzero(out != BACKGROUND_COLOR) == expected_count
    except Exception:
        return False


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_RETRIES):
        length = int(_RNG.integers(MIN_K, MAX_K + 1))
        colors = _sample_color_sequence(length)
        grid = colors.reshape(-1, 1).astype(int)
        if _passes_semantic_checks(grid):
            return grid
    raise RuntimeError("failed to generate a valid increasing spiral seed")
