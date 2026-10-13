from __future__ import annotations

from collections import Counter
import random

import numpy as np


BG = 0
SEPARATOR = 5
MARKER = 6
ARROW_COLORS = [1, 2, 3, 4, 7, 8, 9]

DIRS: dict[str, tuple[int, int]] = {
    "U": (-1, 0),
    "D": (1, 0),
    "L": (0, -1),
    "R": (0, 1),
}

ARROW_MASKS: dict[str, tuple[tuple[int, ...], ...]] = {
    "U": ((1, 1, 1), (0, 1, 0), (0, 1, 0)),
    "D": ((0, 1, 0), (0, 1, 0), (1, 1, 1)),
    "L": ((1, 0, 0), (1, 1, 1), (1, 0, 0)),
    "R": ((0, 0, 1), (1, 1, 1), (0, 0, 1)),
}

MASK_TO_DIR = {mask: direction for direction, mask in ARROW_MASKS.items()}


def _full_line_indices(g: np.ndarray, color: int, axis: int) -> list[int]:
    if axis == 0:
        return [int(r) for r in np.where(np.all(g == color, axis=1))[0]]
    return [int(c) for c in np.where(np.all(g == color, axis=0))[0]]


def _constant_region_size(lines: list[int], limit: int) -> int | None:
    if len(lines) < 2 or lines[0] != 0 or lines[-1] != limit - 1:
        return None
    gaps = [lines[i + 1] - lines[i] - 1 for i in range(len(lines) - 1)]
    if min(gaps) < 3 or len(set(gaps)) != 1:
        return None
    return int(gaps[0])


def _find_partition(g: np.ndarray) -> tuple[int, list[int], list[int], int, int]:
    h, w = g.shape
    candidates: list[tuple[int, list[int], list[int], int, int]] = []
    for color in map(int, np.unique(g)):
        row_lines = _full_line_indices(g, color, axis=0)
        col_lines = _full_line_indices(g, color, axis=1)
        cell_h = _constant_region_size(row_lines, h)
        cell_w = _constant_region_size(col_lines, w)
        if cell_h is not None and cell_w is not None:
            candidates.append((int(color), row_lines, col_lines, cell_h, cell_w))

    assert len(candidates) == 1, "expected one full separator lattice"
    return candidates[0]


def _background_color(g: np.ndarray, sep: int) -> int:
    vals = [int(v) for v in g.ravel() if int(v) != int(sep)]
    assert vals, "expected non-separator cells"
    return Counter(vals).most_common(1)[0][0]


def _region_bounds(row_lines: list[int], col_lines: list[int], rr: int, cc: int) -> tuple[int, int, int, int]:
    r0 = int(row_lines[rr] + 1)
    r1 = int(row_lines[rr + 1])
    c0 = int(col_lines[cc] + 1)
    c1 = int(col_lines[cc + 1])
    return r0, c0, r1, c1


def _locate_region(row_lines: list[int], col_lines: list[int], r: int, c: int) -> tuple[int, int]:
    row = next((i for i in range(len(row_lines) - 1) if row_lines[i] < r < row_lines[i + 1]), None)
    col = next((j for j in range(len(col_lines) - 1) if col_lines[j] < c < col_lines[j + 1]), None)
    assert row is not None and col is not None, "marker must be inside a region"
    return int(row), int(col)


def _decode_arrow(block: np.ndarray, bg: int) -> tuple[str, int]:
    mask = (block != int(bg)) & (block != MARKER)
    colors = sorted(set(map(int, block[mask])))
    assert len(colors) == 1, "each region should contain one arrow color"
    color = int(colors[0])

    coords = np.argwhere(block == color)
    assert coords.shape[0] > 0, "arrow must be nonempty"
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    cropped = block[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1] == color
    symbol = tuple(tuple(int(v) for v in row) for row in cropped.astype(int))
    assert symbol in MASK_TO_DIR, "unknown arrow glyph"
    return MASK_TO_DIR[symbol], color


def _decode_regions(
    g: np.ndarray, row_lines: list[int], col_lines: list[int], bg: int
) -> dict[tuple[int, int], tuple[str, int]]:
    arrows: dict[tuple[int, int], tuple[str, int]] = {}
    for rr in range(len(row_lines) - 1):
        for cc in range(len(col_lines) - 1):
            r0, c0, r1, c1 = _region_bounds(row_lines, col_lines, rr, cc)
            direction, color = _decode_arrow(g[r0:r1, c0:c1], bg)
            arrows[(rr, cc)] = (direction, color)
    return arrows


def _trace_chain(
    start: tuple[int, int], arrows: dict[tuple[int, int], tuple[str, int]], n_rows: int, n_cols: int
) -> tuple[list[tuple[int, int]], list[int]]:
    path = [start]
    incoming = [MARKER]
    seen = {start}
    current = start

    while True:
        direction, color = arrows[current]
        dr, dc = DIRS[direction]
        nxt = (current[0] + dr, current[1] + dc)
        if not (0 <= nxt[0] < n_rows and 0 <= nxt[1] < n_cols):
            break
        assert nxt not in seen, "the pointer chain should be acyclic"
        path.append(nxt)
        incoming.append(int(color))
        seen.add(nxt)
        current = nxt

    return path, incoming


def main(grid: np.ndarray) -> np.ndarray:
    """
    Follow the arrow-room chain and shift the arrow colors forward into visited rooms.

    Input structure:
    - A one-pixel separator lattice divides the grid into equal rectangular rooms.
    - Every room contains one small 3x3 colored arrow pointing up, down, left, or right.
    - Exactly one room also contains a single magenta marker pixel, which marks the chain start.

    Rule:
    - Starting in the marked room, repeatedly follow the room arrow to an adjacent room until an arrow points outside
      the lattice.
    - Clear every room interior to the background while preserving the separator lattice.
    - Fill the start room with magenta, and fill each later visited room with the arrow color of the predecessor room.
    - Leave all unvisited room interiors cleared to background.

    Key insight:
    - The arrows are pointers, not paint labels for their own rooms; the visible color is shifted one step along the
      pointer chain, and only rooms reached from the marked start keep any fill.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    sep, row_lines, col_lines, _cell_h, _cell_w = _find_partition(g)
    bg = _background_color(g, sep)

    marker_coords = np.argwhere(g == MARKER)
    assert marker_coords.shape == (1, 2), "expected one magenta start marker"
    mr, mc = map(int, marker_coords[0])
    start = _locate_region(row_lines, col_lines, mr, mc)

    n_rows = len(row_lines) - 1
    n_cols = len(col_lines) - 1
    arrows = _decode_regions(g, row_lines, col_lines, bg)
    path, incoming = _trace_chain(start, arrows, n_rows, n_cols)

    out = g.copy()
    for rr in range(n_rows):
        for cc in range(n_cols):
            r0, c0, r1, c1 = _region_bounds(row_lines, col_lines, rr, cc)
            out[r0:r1, c0:c1] = bg

    for (rr, cc), color in zip(path, incoming):
        r0, c0, r1, c1 = _region_bounds(row_lines, col_lines, rr, cc)
        out[r0:r1, c0:c1] = int(color)

    return out


# --------------- generator ---------------

GEN_RETRIES = 400


def _layout_candidates() -> list[tuple[int, int, int, int]]:
    out = []
    for n_rows in range(2, 5):
        for n_cols in range(2, 5):
            if n_rows * n_cols < 6:
                continue
            for cell_h in range(5, 8):
                for cell_w in range(5, 8):
                    h = n_rows * cell_h + n_rows + 1
                    w = n_cols * cell_w + n_cols + 1
                    if h <= 30 and w <= 30:
                        out.append((n_rows, n_cols, cell_h, cell_w))
    return out


def _snake_cells(n_rows: int, n_cols: int) -> list[tuple[int, int]]:
    cells: list[tuple[int, int]] = []
    for rr in range(n_rows):
        cols = range(n_cols) if rr % 2 == 0 else range(n_cols - 1, -1, -1)
        for cc in cols:
            cells.append((rr, cc))
    return cells


def _transform_cell(cell: tuple[int, int], n_rows: int, n_cols: int, rot: int, flip: bool) -> tuple[int, int]:
    r, c = cell
    h, w = n_rows, n_cols
    for _ in range(rot):
        r, c = c, h - 1 - r
        h, w = w, h
    if flip:
        c = w - 1 - c
    return int(r), int(c)


def _oriented_snake(n_rows: int, n_cols: int) -> list[tuple[int, int]]:
    rot = random.randrange(4)
    flip = random.random() < 0.5
    base_h, base_w = (n_cols, n_rows) if rot % 2 else (n_rows, n_cols)
    base = _snake_cells(base_h, base_w)
    cells = [_transform_cell(cell, base_h, base_w, rot, flip) for cell in base]
    if random.random() < 0.5:
        cells = list(reversed(cells))
    assert len(set(cells)) == n_rows * n_cols
    assert all(0 <= r < n_rows and 0 <= c < n_cols for r, c in cells)
    return cells


def _is_boundary(cell: tuple[int, int], n_rows: int, n_cols: int) -> bool:
    r, c = cell
    return r == 0 or c == 0 or r == n_rows - 1 or c == n_cols - 1


def _path_directions(path: list[tuple[int, int]]) -> list[str]:
    directions = []
    for (r0, c0), (r1, c1) in zip(path, path[1:]):
        delta = (r1 - r0, c1 - c0)
        directions.append(next(name for name, step in DIRS.items() if step == delta))
    return directions


def _sample_path(n_rows: int, n_cols: int) -> list[tuple[int, int]] | None:
    total = n_rows * n_cols
    min_len = 3
    max_len = total - 1
    cells = _oriented_snake(n_rows, n_cols)
    boundary_indices = [i for i, cell in enumerate(cells) if _is_boundary(cell, n_rows, n_cols)]
    random.shuffle(boundary_indices)

    for end in boundary_indices:
        possible_lengths = [length for length in range(min_len, max_len + 1) if end - length + 1 >= 0]
        random.shuffle(possible_lengths)
        for length in possible_lengths:
            start = end - length + 1
            path = cells[start : end + 1]
            if len(path) != length or len(set(path)) != length:
                continue
            dirs = _path_directions(path)
            if len(set(dirs)) < 2:
                continue
            return path
    return None


def _direction_between(a: tuple[int, int], b: tuple[int, int]) -> str:
    delta = (b[0] - a[0], b[1] - a[1])
    return next(name for name, step in DIRS.items() if step == delta)


def _outward_direction(cell: tuple[int, int], n_rows: int, n_cols: int) -> str:
    r, c = cell
    choices = []
    if r == 0:
        choices.append("U")
    if r == n_rows - 1:
        choices.append("D")
    if c == 0:
        choices.append("L")
    if c == n_cols - 1:
        choices.append("R")
    assert choices, "terminal cell must be on boundary"
    return random.choice(choices)


def _grid_lines(count: int, cell_size: int) -> list[int]:
    return [i * (cell_size + 1) for i in range(count + 1)]


def _draw_arrow(grid: np.ndarray, rr: int, cc: int, cell_h: int, cell_w: int, direction: str, color: int) -> None:
    r0 = rr * (cell_h + 1) + 1
    c0 = cc * (cell_w + 1) + 1
    top = r0 + (cell_h - 3) // 2
    left = c0 + (cell_w - 3) // 2
    mask = ARROW_MASKS[direction]
    for dr in range(3):
        for dc in range(3):
            if mask[dr][dc]:
                grid[top + dr, left + dc] = int(color)


def _room_bg_cells(grid: np.ndarray, rr: int, cc: int, cell_h: int, cell_w: int) -> list[tuple[int, int]]:
    r0 = rr * (cell_h + 1) + 1
    c0 = cc * (cell_w + 1) + 1
    cells = []
    for dr in range(cell_h):
        for dc in range(cell_w):
            r, c = r0 + dr, c0 + dc
            if int(grid[r, c]) == BG and (dr in (0, cell_h - 1) or dc in (0, cell_w - 1)):
                cells.append((r, c))
    return cells


def _assign_arrows_and_colors(
    n_rows: int, n_cols: int, path: list[tuple[int, int]]
) -> tuple[dict[tuple[int, int], str], dict[tuple[int, int], int]]:
    directions: dict[tuple[int, int], str] = {}
    colors: dict[tuple[int, int], int] = {}

    for rr in range(n_rows):
        for cc in range(n_cols):
            directions[(rr, cc)] = random.choice(tuple(DIRS))
            colors[(rr, cc)] = random.choice(ARROW_COLORS)

    for current, nxt in zip(path, path[1:]):
        directions[current] = _direction_between(current, nxt)
    directions[path[-1]] = _outward_direction(path[-1], n_rows, n_cols)

    previous = None
    for cell in path:
        choices = [color for color in ARROW_COLORS if color != previous]
        colors[cell] = random.choice(choices)
        previous = colors[cell]

    return directions, colors


def _build_input(
    n_rows: int,
    n_cols: int,
    cell_h: int,
    cell_w: int,
    path: list[tuple[int, int]],
    directions: dict[tuple[int, int], str],
    colors: dict[tuple[int, int], int],
) -> np.ndarray:
    h = n_rows * cell_h + n_rows + 1
    w = n_cols * cell_w + n_cols + 1
    grid = np.full((h, w), BG, dtype=int)

    for r in _grid_lines(n_rows, cell_h):
        grid[r, :] = SEPARATOR
    for c in _grid_lines(n_cols, cell_w):
        grid[:, c] = SEPARATOR

    for rr in range(n_rows):
        for cc in range(n_cols):
            _draw_arrow(grid, rr, cc, cell_h, cell_w, directions[(rr, cc)], colors[(rr, cc)])

    start = path[0]
    marker_options = _room_bg_cells(grid, start[0], start[1], cell_h, cell_w)
    assert marker_options, "expected a free marker pixel in the start room"
    mr, mc = random.choice(marker_options)
    grid[mr, mc] = MARKER
    return grid


def _expected_output(
    inp: np.ndarray,
    n_rows: int,
    n_cols: int,
    cell_h: int,
    cell_w: int,
    path: list[tuple[int, int]],
    colors: dict[tuple[int, int], int],
) -> np.ndarray:
    out = inp.copy()
    for rr in range(n_rows):
        for cc in range(n_cols):
            r0 = rr * (cell_h + 1) + 1
            c0 = cc * (cell_w + 1) + 1
            out[r0 : r0 + cell_h, c0 : c0 + cell_w] = BG

    incoming = [MARKER] + [colors[cell] for cell in path[:-1]]
    for (rr, cc), color in zip(path, incoming):
        r0 = rr * (cell_h + 1) + 1
        c0 = cc * (cell_w + 1) + 1
        out[r0 : r0 + cell_h, c0 : c0 + cell_w] = int(color)
    return out


def _semantic_non_degenerate(
    n_rows: int,
    n_cols: int,
    path: list[tuple[int, int]],
    colors: dict[tuple[int, int], int],
) -> bool:
    if len(path) < 3 or len(path) >= n_rows * n_cols:
        return False
    if len(set(_path_directions(path))) < 2:
        return False
    if not _is_boundary(path[-1], n_rows, n_cols):
        return False

    incoming = [MARKER] + [colors[cell] for cell in path[:-1]]
    for cell, fill in zip(path, incoming):
        if colors[cell] == fill:
            return False
    return True


def _gen_attempt() -> np.ndarray | None:
    n_rows, n_cols, cell_h, cell_w = random.choice(_layout_candidates())
    path = _sample_path(n_rows, n_cols)
    if path is None:
        return None

    directions, colors = _assign_arrows_and_colors(n_rows, n_cols, path)
    if not _semantic_non_degenerate(n_rows, n_cols, path, colors):
        return None

    inp = _build_input(n_rows, n_cols, cell_h, cell_w, path, directions, colors)
    expected = _expected_output(inp, n_rows, n_cols, cell_h, cell_w, path, colors)

    try:
        got = main(inp)
    except Exception:
        return None
    if not np.array_equal(got, expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid pointer-chain input")
