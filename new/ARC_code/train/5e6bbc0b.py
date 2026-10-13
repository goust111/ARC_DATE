import random

import numpy as np
from arc_utils import Color


def _checker_colors(g: np.ndarray, marker_row: int, marker_col: int) -> tuple[int, int]:
    marker_parity = (marker_row + marker_col) % 2
    moving_values = []
    bg_values = []
    for r in range(g.shape[0]):
        for c in range(g.shape[1]):
            if r == marker_row and c == marker_col:
                continue
            if (r + c) % 2 == marker_parity:
                moving_values.append(int(g[r, c]))
            else:
                bg_values.append(int(g[r, c]))

    assert moving_values and bg_values
    assert len(set(moving_values)) == 1
    assert len(set(bg_values)) == 1
    moving_color = moving_values[0]
    bg_color = bg_values[0]
    assert moving_color != bg_color
    return moving_color, bg_color


def _trail_indices(sources: list[int], final_positions: list[int], marker_index: int) -> set[int]:
    assert len(sources) == len(final_positions)
    final_set = set(final_positions)
    swept = set()
    for src, dst in zip(sources, final_positions):
        if src == marker_index:
            assert dst == marker_index
            continue
        swept.update(range(min(src, dst), max(src, dst) + 1))
    return swept - final_set


def main(grid: np.ndarray) -> np.ndarray:
    """Compact the checkerboard's moving color toward the edge marked by the teal pixel, leaving a maroon swept trail on the marker line.

    Input structure
    - A blue/black checkerboard with exactly one teal marker replacing a blue pixel on a non-corner edge.
    - The marker's checkerboard parity identifies the moving color (blue); the opposite parity is background (black).

    Rule
    - If the marker is on the left or right edge, compact moving-color pixels in every row toward that side.
    - If the marker is on the top or bottom edge, compact moving-color pixels in every column toward that side.
    - Preserve the teal marker at its edge cell.
    - On the marker's row or column, color cells swept by shifted moving pixels maroon unless the final compacted line occupies them.

    Key insight
    The marker selects both the compression direction and the checkerboard color class that moves; maroon records the movement trail only along that marked line.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    height, width = g.shape

    marker_color = Color.TEAL
    trail_color = Color.MAROON
    marker_cells = np.argwhere(g == marker_color)
    assert marker_cells.shape == (1, 2)
    marker_row, marker_col = map(int, marker_cells[0])

    on_top = marker_row == 0
    on_bottom = marker_row == height - 1
    on_left = marker_col == 0
    on_right = marker_col == width - 1
    assert sum([on_top, on_bottom, on_left, on_right]) == 1

    moving_color, bg_color = _checker_colors(g, marker_row, marker_col)
    assert moving_color == Color.BLUE
    assert bg_color == Color.BLACK
    assert trail_color not in (moving_color, bg_color, marker_color)

    out = np.full_like(g, bg_color)

    if on_left or on_right:
        for r in range(height):
            count = int(np.count_nonzero(g[r, :] == moving_color))
            if r == marker_row:
                count += 1
            assert 0 < count <= width
            final_positions = list(range(count)) if on_left else list(range(width - count, width))
            out[r, final_positions] = moving_color

        out[marker_row, marker_col] = marker_color
        sources = [c for c in range(width) if g[marker_row, c] == moving_color or c == marker_col]
        final_positions = list(range(len(sources))) if on_left else list(range(width - len(sources), width))
        for c in _trail_indices(sources, final_positions, marker_col):
            out[marker_row, c] = trail_color
    else:
        for c in range(width):
            count = int(np.count_nonzero(g[:, c] == moving_color))
            if c == marker_col:
                count += 1
            assert 0 < count <= height
            final_positions = list(range(count)) if on_top else list(range(height - count, height))
            out[final_positions, c] = moving_color

        out[marker_row, marker_col] = marker_color
        sources = [r for r in range(height) if g[r, marker_col] == moving_color or r == marker_row]
        final_positions = list(range(len(sources))) if on_top else list(range(height - len(sources), height))
        for r in _trail_indices(sources, final_positions, marker_row):
            out[r, marker_col] = trail_color

    return out

# --------------- generator (att2) ---------------

# Canvas and retry bounds.
GRID_SIZE_RANGE = (3, 8)
EDGE_MARKER_PADDING = 1
MAX_GENERATION_ATTEMPTS = 300
# Layout and color choices.
TOP, BOTTOM, LEFT, RIGHT = EDGE_NAMES = ("top", "bottom", "left", "right")
MOVING_COLOR, BG_COLOR = Color.BLUE, Color.BLACK
MARKER_COLOR, TRAIL_COLOR = Color.TEAL, Color.MAROON

def _choose_marker(edge: str, h: int, w: int) -> tuple[int, int]:
    """Sample a non-corner marker coordinate on the selected edge."""
    lo = EDGE_MARKER_PADDING
    if edge == TOP:
        return 0, random.randint(lo, w - lo - 1)
    if edge == BOTTOM:
        return h - 1, random.randint(lo, w - lo - 1)
    if edge == LEFT:
        return random.randint(lo, h - lo - 1), 0
    return random.randint(lo, h - lo - 1), w - 1

def _make_checkerboard(h: int, w: int, marker_r: int, marker_c: int) -> np.ndarray:
    """Build the parity grid whose moving-color class contains the marker."""
    marker_parity = (marker_r + marker_c) % 2
    rr, cc = np.indices((h, w))
    grid = np.where((rr + cc) % 2 == marker_parity, MOVING_COLOR, BG_COLOR).astype(int)
    grid[marker_r, marker_c] = MARKER_COLOR
    return grid

def _packed_positions(length: int, count: int, toward_start: bool) -> list[int]:
    """Return occupied indices after compaction toward one side of a line."""
    return list(range(count)) if toward_start else list(range(length - count, length))

def _expected_output(grid: np.ndarray, marker_r: int, marker_c: int) -> np.ndarray:
    """Render the compacted checkerboard oracle for one generated input."""
    h, w = grid.shape
    out = np.full_like(grid, BG_COLOR)

    if marker_c == 0 or marker_c == w - 1:
        to_left = marker_c == 0
        counts = np.count_nonzero(grid == MOVING_COLOR, axis=1)
        counts[marker_r] += 1
        for r in range(h):
            out[r, _packed_positions(w, int(counts[r]), to_left)] = MOVING_COLOR

        sources = [c for c, value in enumerate(grid[marker_r, :]) if value == MOVING_COLOR or c == marker_c]
        finals = _packed_positions(w, len(sources), to_left)
        for c in _trail_indices(sources, finals, marker_c):
            out[marker_r, c] = TRAIL_COLOR
    else:
        to_top = marker_r == 0
        counts = np.count_nonzero(grid == MOVING_COLOR, axis=0)
        counts[marker_c] += 1
        for c in range(w):
            out[_packed_positions(h, int(counts[c]), to_top), c] = MOVING_COLOR

        sources = [r for r, value in enumerate(grid[:, marker_c]) if value == MOVING_COLOR or r == marker_r]
        finals = _packed_positions(h, len(sources), to_top)
        for r in _trail_indices(sources, finals, marker_r):
            out[r, marker_c] = TRAIL_COLOR

    out[marker_r, marker_c] = MARKER_COLOR
    return out

def _gen_attempt() -> np.ndarray | None:
    """Sample one candidate grid and keep it only if the solver matches the oracle."""
    h = random.randint(*GRID_SIZE_RANGE)
    w = random.randint(*GRID_SIZE_RANGE)
    edge = random.choice(EDGE_NAMES)
    marker_r, marker_c = _choose_marker(edge, h, w)
    grid = _make_checkerboard(h, w, marker_r, marker_c)

    expected = _expected_output(grid, marker_r, marker_c)
    if not np.array_equal(main(grid), expected):
        return None
    return grid

def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
