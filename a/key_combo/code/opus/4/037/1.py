import random
from collections import deque

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5
PINK = 6
ORANGE = 7
TEAL = 8
MAROON = 9

GRID_CELLS = 4
SEP_WIDTH = 1
PATTERN_COLORS = (BLUE, GREEN, YELLOW, PINK, ORANGE, TEAL, MAROON)
MARKER_POSITIONS = [
    (r, c)
    for r in range(1, GRID_CELLS)
    for c in range(1, GRID_CELLS)
    if (r, c) != (GRID_CELLS - 1, GRID_CELLS - 1)
]
MAX_GEN_ATTEMPTS = 400


def _bands(size: int, separators: list[int]) -> list[tuple[int, int]]:
    bands = []
    start = 0
    for cut in separators:
        if start >= cut:
            raise AssertionError("separator rows/cols must be distinct")
        bands.append((start, cut))
        start = cut + 1
    if start >= size:
        raise AssertionError("missing trailing cell band")
    bands.append((start, size))
    return bands


def _full_color_rows(grid: np.ndarray) -> dict[int, list[int]]:
    out: dict[int, list[int]] = {}
    for r in range(grid.shape[0]):
        color = int(grid[r, 0])
        if np.all(grid[r, :] == color):
            out.setdefault(color, []).append(r)
    return out


def _full_color_cols(grid: np.ndarray) -> dict[int, list[int]]:
    out: dict[int, list[int]] = {}
    for c in range(grid.shape[1]):
        color = int(grid[0, c])
        if np.all(grid[:, c] == color):
            out.setdefault(color, []).append(c)
    return out


def _parse_lattice(grid: np.ndarray) -> tuple[int, list[tuple[int, int]], list[tuple[int, int]], list[list[np.ndarray]]]:
    g = np.asarray(grid, dtype=int)
    if g.ndim != 2:
        raise AssertionError("expected a 2D grid")

    row_map = _full_color_rows(g)
    col_map = _full_color_cols(g)
    candidates = []
    for color in sorted(set(row_map) & set(col_map)):
        rows = row_map[color]
        cols = col_map[color]
        if len(rows) != GRID_CELLS - 1 or len(cols) != GRID_CELLS - 1:
            continue
        row_bands = _bands(g.shape[0], rows)
        col_bands = _bands(g.shape[1], cols)
        if len(row_bands) != GRID_CELLS or len(col_bands) != GRID_CELLS:
            continue
        row_sizes = {r1 - r0 for r0, r1 in row_bands}
        col_sizes = {c1 - c0 for c0, c1 in col_bands}
        if len(row_sizes) != 1 or len(col_sizes) != 1:
            continue
        if min(row_sizes) < 2 or min(col_sizes) < 2:
            continue
        candidates.append((color, row_bands, col_bands))

    if len(candidates) != 1:
        raise AssertionError("expected exactly one separator lattice color")

    sep, row_bands, col_bands = candidates[0]
    cells: list[list[np.ndarray]] = []
    for r0, r1 in row_bands:
        row_cells = []
        for c0, c1 in col_bands:
            row_cells.append(g[r0:r1, c0:c1].copy())
        cells.append(row_cells)
    return int(sep), row_bands, col_bands, cells


def _find_marker(cells: list[list[np.ndarray]]) -> tuple[int, int]:
    marker_positions = []
    for r in range(GRID_CELLS):
        for c in range(GRID_CELLS):
            cell = cells[r][c]
            if np.all(cell == RED):
                marker_positions.append((r, c))
    if len(marker_positions) != 1:
        raise AssertionError("expected exactly one solid red marker cell")
    return marker_positions[0]


def _blank_marker(cells: list[list[np.ndarray]], marker_rc: tuple[int, int]) -> list[list[np.ndarray]]:
    out = [[cell.copy() for cell in row] for row in cells]
    mr, mc = marker_rc
    out[mr][mc][:] = BLACK
    return out


def _roll_cells(cells: list[list[np.ndarray]], shift_r: int, shift_c: int) -> list[list[np.ndarray]]:
    h = len(cells)
    w = len(cells[0])
    out = [[None for _ in range(w)] for _ in range(h)]
    for r in range(h):
        for c in range(w):
            out[(r + shift_r) % h][(c + shift_c) % w] = cells[r][c].copy()
    return out


def _stitch_cells(cells: list[list[np.ndarray]]) -> np.ndarray:
    row_blocks = [np.hstack(row) for row in cells]
    return np.vstack(row_blocks)


def main(input_grid: np.ndarray) -> np.ndarray:
    """
    Decode the solid red control cell as a lattice shift, blank it, then cyclically reorder the cell grid without separators.

    Input structure:
    - A full-row and full-column grey separator lattice divides the grid into a 4 by 4 array of equal square cells.
    - Exactly one cell is filled entirely with red and serves as the control marker.
    - Every other cell contains a small non-black colored pattern on a black background.

    Rule:
    - Detect the separator rows and columns and split the input into the 4 by 4 cell array.
    - Find the unique solid red cell and read its lattice coordinates as (R, C), using zero-based row and column indices.
    - Replace that marker cell by an all-black cell, then cyclically shift the whole cell array downward by R and rightward by C.
    - Remove the separator lines and output the stitched rearranged cells.

    Key insight:
    - The pure red cell is not data to keep; its lattice position is the only control signal, and the shift must be applied at cell granularity after parsing the separator grid.
    """
    _sep, _row_bands, _col_bands, cells = _parse_lattice(input_grid)
    marker_rc = _find_marker(cells)
    blanked = _blank_marker(cells, marker_rc)
    rolled = _roll_cells(blanked, marker_rc[0], marker_rc[1])
    return _stitch_cells(rolled)


def _neighbors4(r: int, c: int, h: int, w: int):
    for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        rr = r + dr
        cc = c + dc
        if 0 <= rr < h and 0 <= cc < w:
            yield rr, cc


def _connected_mask(cell_size: int, area: int) -> set[tuple[int, int]]:
    start = (random.randrange(cell_size), random.randrange(cell_size))
    cells = {start}
    while len(cells) < area:
        frontier = []
        for r, c in cells:
            for rr, cc in _neighbors4(r, c, cell_size, cell_size):
                if (rr, cc) not in cells:
                    frontier.append((rr, cc))
        if not frontier:
            raise AssertionError("unexpected empty growth frontier")
        cells.add(random.choice(frontier))
    return cells


def _pattern_signature(cell: np.ndarray) -> bytes:
    return cell.astype(np.int16, copy=False).tobytes()


def _make_pattern_cell(cell_size: int, used: set[bytes]) -> np.ndarray:
    max_area = min(5, cell_size * cell_size - 1)
    for _ in range(200):
        area = random.randint(2, max_area)
        coords = list(_connected_mask(cell_size, area))
        random.shuffle(coords)

        n_colors = random.randint(1, min(3, area, len(PATTERN_COLORS)))
        palette = random.sample(PATTERN_COLORS, n_colors)
        colors = [palette[i % n_colors] for i in range(area)]
        random.shuffle(colors)

        cell = np.full((cell_size, cell_size), BLACK, dtype=int)
        for (r, c), color in zip(coords, colors):
            cell[r, c] = color

        sig = _pattern_signature(cell)
        if sig in used:
            continue
        used.add(sig)
        return cell
    raise RuntimeError("failed to generate a unique pattern cell")


def _render_input(cells: list[list[np.ndarray]], sep_color: int) -> np.ndarray:
    cell_size = cells[0][0].shape[0]
    size = GRID_CELLS * cell_size + (GRID_CELLS - 1) * SEP_WIDTH
    grid = np.full((size, size), sep_color, dtype=int)
    stride = cell_size + SEP_WIDTH
    for r in range(GRID_CELLS):
        for c in range(GRID_CELLS):
            r0 = r * stride
            c0 = c * stride
            grid[r0 : r0 + cell_size, c0 : c0 + cell_size] = cells[r][c]
    return grid


def _foreground_component_sizes(mask: np.ndarray) -> list[int]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    sizes = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            area = 0
            while q:
                rr, cc = q.popleft()
                area += 1
                for nr, nc in _neighbors4(rr, cc, h, w):
                    if mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            sizes.append(area)
    return sizes


def _assert_semantics(inp: np.ndarray) -> None:
    sep, row_bands, col_bands, cells = _parse_lattice(inp)
    assert sep == GREY, "generated separator must stay grey"
    cell_h = row_bands[0][1] - row_bands[0][0]
    cell_w = col_bands[0][1] - col_bands[0][0]
    assert cell_h == cell_w
    assert 2 <= cell_h <= 6
    assert len(row_bands) == GRID_CELLS and len(col_bands) == GRID_CELLS

    marker_rc = _find_marker(cells)
    assert marker_rc in MARKER_POSITIONS

    non_marker_sigs = set()
    for r in range(GRID_CELLS):
        for c in range(GRID_CELLS):
            cell = cells[r][c]
            if (r, c) == marker_rc:
                assert np.all(cell == RED)
                continue
            assert not np.all(cell == RED)
            sig = _pattern_signature(cell)
            assert sig not in non_marker_sigs, "non-marker cells must be unique"
            non_marker_sigs.add(sig)
            comps = _foreground_component_sizes(cell != BLACK)
            assert comps and len(comps) == 1
            assert 2 <= comps[0] <= 5

    blanked = _blank_marker(cells, marker_rc)
    out = main(inp)
    expected_shape = (GRID_CELLS * cell_h, GRID_CELLS * cell_w)
    assert out.shape == expected_shape

    no_roll = _stitch_cells(blanked)
    assert not np.array_equal(out, no_roll), "shift must matter"

    keep_marker_roll = _stitch_cells(_roll_cells(cells, marker_rc[0], marker_rc[1]))
    assert not np.array_equal(out, keep_marker_roll), "blanking the marker must matter"

    all_shifted = {}
    for dr in range(GRID_CELLS):
        for dc in range(GRID_CELLS):
            shifted = _stitch_cells(_roll_cells(blanked, dr, dc))
            all_shifted[(dr, dc)] = shifted
    for shift, shifted in all_shifted.items():
        if shift == marker_rc:
            assert np.array_equal(shifted, out)
        else:
            assert not np.array_equal(shifted, out), "marker position must uniquely determine the shift"


def generate_input() -> np.ndarray:
    for _ in range(MAX_GEN_ATTEMPTS):
        cell_size = random.randint(3, 6)
        used: set[bytes] = set()
        marker_rc = random.choice(MARKER_POSITIONS)

        cells: list[list[np.ndarray]] = []
        for r in range(GRID_CELLS):
            row_cells = []
            for c in range(GRID_CELLS):
                if (r, c) == marker_rc:
                    row_cells.append(np.full((cell_size, cell_size), RED, dtype=int))
                else:
                    row_cells.append(_make_pattern_cell(cell_size, used))
            cells.append(row_cells)

        inp = _render_input(cells, GREY)
        try:
            _assert_semantics(inp)
        except AssertionError:
            continue
        out = main(inp)
        if inp.ndim != 2 or out.ndim != 2:
            continue
        if max(inp.shape) > 30 or max(out.shape) > 30:
            continue
        return inp
    raise RuntimeError("failed to generate a valid ARC input")
