import random
from collections import Counter

import numpy as np


_DIRS = {
    "U": (-1, 0),
    "D": (1, 0),
    "L": (0, -1),
    "R": (0, 1),
}
_DIR_ORDER = ("U", "D", "L", "R")


def _arrow_mask(cell_size: int, direction: str) -> np.ndarray:
    """Return the centered arrow mask for a square cell."""
    mid = cell_size // 2
    mask = np.zeros((cell_size, cell_size), dtype=bool)
    if direction == "U":
        mask[1:cell_size - 1, mid] = True
        mask[0, mid - 1:mid + 2] = True
    elif direction == "D":
        mask[1:cell_size - 1, mid] = True
        mask[cell_size - 1, mid - 1:mid + 2] = True
    elif direction == "L":
        mask[mid, 1:cell_size - 1] = True
        mask[mid - 1:mid + 2, 0] = True
    elif direction == "R":
        mask[mid, 1:cell_size - 1] = True
        mask[mid - 1:mid + 2, cell_size - 1] = True
    else:
        raise ValueError(f"unknown direction: {direction}")
    return mask


def _segment_ranges(size: int, separators: list[int]) -> list[tuple[int, int]]:
    """Convert separator indices into non-separator [start, end) cell spans."""
    bounds = [-1, *sorted(set(separators)), size]
    ranges: list[tuple[int, int]] = []
    for left, right in zip(bounds, bounds[1:]):
        gap = right - left - 1
        if gap == 0:
            assert left == -1 or right == size, "zero-width gaps are only valid at optional borders"
            continue
        assert gap > 0, "separator rows/cols must be strictly increasing"
        ranges.append((left + 1, right))
    assert ranges, "expected at least one cell band"
    lengths = {end - start for start, end in ranges}
    assert len(lengths) == 1, "all cell bands must have the same size"
    band = next(iter(lengths))
    assert all(ranges[i + 1][0] - ranges[i][0] == band + 1 for i in range(len(ranges) - 1)), "bands must be split by single separators"
    return ranges


def _find_partition(grid: np.ndarray) -> tuple[int, list[tuple[int, int]], list[tuple[int, int]], int]:
    """Find the separator color and the equal-size cell partition."""
    h, w = grid.shape
    candidates: list[tuple[int, list[tuple[int, int]], list[tuple[int, int]], int]] = []
    for color in map(int, np.unique(grid)):
        full_rows = np.flatnonzero(np.all(grid == color, axis=1)).tolist()
        full_cols = np.flatnonzero(np.all(grid == color, axis=0)).tolist()
        if not full_rows or not full_cols:
            continue
        try:
            row_ranges = _segment_ranges(h, full_rows)
            col_ranges = _segment_ranges(w, full_cols)
        except AssertionError:
            continue
        if len(row_ranges) < 2 or len(col_ranges) < 2:
            continue
        cell_size = row_ranges[0][1] - row_ranges[0][0]
        if cell_size != col_ranges[0][1] - col_ranges[0][0]:
            continue
        if cell_size < 5 or cell_size % 2 == 0:
            continue
        candidates.append((color, row_ranges, col_ranges, cell_size))
    assert len(candidates) == 1, "expected exactly one separator grid"
    return candidates[0]


def _infer_background(grid: np.ndarray, separator: int) -> int:
    """Background is the most common non-separator color."""
    values, counts = np.unique(grid[grid != separator], return_counts=True)
    assert values.size > 0, "expected non-separator cells"
    return int(values[int(counts.argmax())])


def _parse_cell(cell: np.ndarray, bg: int, separator: int, cell_size: int) -> tuple[int, str] | None:
    """Return (arrow_color, direction) for a filled cell, or None for an empty one."""
    colors = set(map(int, np.unique(cell))) - {bg}
    if not colors:
        return None
    assert separator not in colors, "separator pixels cannot appear inside a cell"
    assert len(colors) == 1, "each cell must contain at most one arrow color"
    color = colors.pop()
    mask = cell == color
    for direction in _DIR_ORDER:
        if np.array_equal(mask, _arrow_mask(cell_size, direction)):
            return int(color), direction
    raise AssertionError("cell contents must be empty or a centered arrow")


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor the directed arrow chain so every visited arrow matches the unique start color.

    Input structure:
    - The background is black and thin red separator lines split the canvas into equal square cells.
    - Each cell is either empty or contains one centered monochrome arrow pointing up, down, left, or right.
    - Exactly one arrow color appears in only one cell; that uniquely colored arrow is the start.

    Rule:
    - Find the start cell and remember its arrow color as the propagation color.
    - Follow the start arrow into the neighboring cell, then keep following each reached cell's arrow.
    - Stop when the next step would leave the board or would enter an empty cell.
    - Recolor every arrow on that visited chain, including the start arrow, to the propagation color.

    Key insight:
    - The arrows are not independent decorations: they define a single directed walk whose cells must be recolored as one chain.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "grid must be 2D"

    separator, row_ranges, col_ranges, cell_size = _find_partition(g)
    bg = _infer_background(g, separator)
    nr, nc = len(row_ranges), len(col_ranges)

    cells: list[list[tuple[int, str] | None]] = [[None for _ in range(nc)] for _ in range(nr)]
    color_counts: Counter[int] = Counter()
    for rr, (r0, r1) in enumerate(row_ranges):
        for cc, (c0, c1) in enumerate(col_ranges):
            info = _parse_cell(g[r0:r1, c0:c1], bg, separator, cell_size)
            cells[rr][cc] = info
            if info is not None:
                color_counts[info[0]] += 1

    assert color_counts, "expected at least one arrow"
    unique_colors = [color for color, count in color_counts.items() if count == 1]
    assert len(unique_colors) == 1, "expected exactly one unique start color"
    propagation_color = unique_colors[0]

    starts = [
        (rr, cc)
        for rr in range(nr)
        for cc in range(nc)
        if cells[rr][cc] is not None and cells[rr][cc][0] == propagation_color
    ]
    assert len(starts) == 1, "expected exactly one start cell"

    path: list[tuple[int, int]] = []
    seen: set[tuple[int, int]] = set()
    row, col = starts[0]
    while 0 <= row < nr and 0 <= col < nc and (row, col) not in seen:
        info = cells[row][col]
        if info is None:
            break
        path.append((row, col))
        seen.add((row, col))
        dr, dc = _DIRS[info[1]]
        next_row = row + dr
        next_col = col + dc
        if not (0 <= next_row < nr and 0 <= next_col < nc):
            break
        if cells[next_row][next_col] is None:
            break
        row, col = next_row, next_col

    out = g.copy()
    for rr, cc in path:
        r0, r1 = row_ranges[rr]
        c0, c1 = col_ranges[cc]
        color, _direction = cells[rr][cc]
        patch = out[r0:r1, c0:c1]
        patch[patch == color] = propagation_color
        out[r0:r1, c0:c1] = patch

    return out


# --------------- generator (arc_chain_arrows) ---------------

GEN_BG = 0
GEN_SEPARATOR = 2
GEN_CELL_SIZES = (5, 7)
GEN_ARROW_COLORS = [1, 3, 4, 5, 6, 7, 8, 9]
GEN_MAX_GENERATION_ATTEMPTS = 400


def _neighbors(cell: tuple[int, int], rows: int, cols: int) -> list[tuple[int, int]]:
    r, c = cell
    out = []
    if r > 0:
        out.append((r - 1, c))
    if r + 1 < rows:
        out.append((r + 1, c))
    if c > 0:
        out.append((r, c - 1))
    if c + 1 < cols:
        out.append((r, c + 1))
    return out


def _direction_between(src: tuple[int, int], dst: tuple[int, int]) -> str:
    dr = dst[0] - src[0]
    dc = dst[1] - src[1]
    for direction, delta in _DIRS.items():
        if delta == (dr, dc):
            return direction
    raise ValueError("cells are not adjacent")


def _outward_directions(cell: tuple[int, int], rows: int, cols: int) -> list[str]:
    r, c = cell
    dirs = []
    if r == 0:
        dirs.append("U")
    if r == rows - 1:
        dirs.append("D")
    if c == 0:
        dirs.append("L")
    if c == cols - 1:
        dirs.append("R")
    return dirs


def _build_backward_path(
    end: tuple[int, int],
    length: int,
    rows: int,
    cols: int,
    forbidden: set[tuple[int, int]] | None = None,
) -> list[tuple[int, int]] | None:
    """Build a self-avoiding path of the requested length that ends at `end`."""
    blocked = set() if forbidden is None else set(forbidden)
    path = [end]
    used = {end}

    def dfs() -> bool:
        if len(path) == length:
            return True
        choices = _neighbors(path[0], rows, cols)
        random.shuffle(choices)
        for nxt in choices:
            if nxt in used or nxt in blocked:
                continue
            path.insert(0, nxt)
            used.add(nxt)
            if dfs():
                return True
            used.remove(nxt)
            path.pop(0)
        return False

    return path if dfs() else None


def _cell_ranges(count: int, cell_size: int) -> list[tuple[int, int]]:
    ranges = []
    cursor = 0
    for idx in range(count):
        ranges.append((cursor, cursor + cell_size))
        cursor += cell_size
        if idx + 1 < count:
            cursor += 1
    return ranges


def _render_scene(
    arrows: list[list[tuple[int, str] | None]],
    cell_size: int,
    bg: int,
    separator: int,
) -> np.ndarray:
    rows = len(arrows)
    cols = len(arrows[0])
    row_ranges = _cell_ranges(rows, cell_size)
    col_ranges = _cell_ranges(cols, cell_size)
    height = row_ranges[-1][1]
    width = col_ranges[-1][1]
    grid = np.full((height, width), bg, dtype=int)

    cursor = cell_size
    while cursor < height:
        grid[cursor, :] = separator
        cursor += cell_size + 1
    cursor = cell_size
    while cursor < width:
        grid[:, cursor] = separator
        cursor += cell_size + 1

    for rr, (r0, r1) in enumerate(row_ranges):
        for cc, (c0, c1) in enumerate(col_ranges):
            info = arrows[rr][cc]
            if info is None:
                continue
            color, direction = info
            patch = grid[r0:r1, c0:c1]
            patch[_arrow_mask(cell_size, direction)] = color
            grid[r0:r1, c0:c1] = patch
    return grid


def _trace_path(arrows: list[list[tuple[int, str] | None]], start: tuple[int, int]) -> list[tuple[int, int]]:
    rows = len(arrows)
    cols = len(arrows[0])
    path: list[tuple[int, int]] = []
    seen: set[tuple[int, int]] = set()
    r, c = start
    while 0 <= r < rows and 0 <= c < cols and (r, c) not in seen:
        info = arrows[r][c]
        if info is None:
            break
        path.append((r, c))
        seen.add((r, c))
        dr, dc = _DIRS[info[1]]
        nr = r + dr
        nc = c + dc
        if not (0 <= nr < rows and 0 <= nc < cols):
            break
        if arrows[nr][nc] is None:
            break
        r, c = nr, nc
    return path


def _copy_arrows(arrows: list[list[tuple[int, str] | None]]) -> list[list[tuple[int, str] | None]]:
    return [[cell if cell is None else (cell[0], cell[1]) for cell in row] for row in arrows]


def _gen_attempt() -> np.ndarray | None:
    cell_size = random.choice(GEN_CELL_SIZES)
    max_cells = 31 // (cell_size + 1)
    rows = random.randint(3, max_cells)
    cols = random.randint(3, max_cells)
    termination = random.choice(["empty", "boundary"])
    max_path = min(6, rows * cols - 1)
    if max_path < 3:
        return None
    path_length = random.randint(3, max_path)

    start_color, common_color = random.sample(GEN_ARROW_COLORS, 2)
    path: list[tuple[int, int]] | None = None
    last_direction = ""
    reserved_empty: tuple[int, int] | None = None

    if termination == "boundary":
        boundary_cells = [
            (r, c)
            for r in range(rows)
            for c in range(cols)
            if r in {0, rows - 1} or c in {0, cols - 1}
        ]
        random.shuffle(boundary_cells)
        for end in boundary_cells:
            candidate = _build_backward_path(end, path_length, rows, cols)
            if candidate is None:
                continue
            exit_dirs = _outward_directions(end, rows, cols)
            if not exit_dirs:
                continue
            path = candidate
            last_direction = random.choice(exit_dirs)
            break
    else:
        all_cells = [(r, c) for r in range(rows) for c in range(cols)]
        random.shuffle(all_cells)
        for target in all_cells:
            end_choices = _neighbors(target, rows, cols)
            random.shuffle(end_choices)
            for end in end_choices:
                candidate = _build_backward_path(end, path_length, rows, cols, forbidden={target})
                if candidate is None:
                    continue
                path = candidate
                last_direction = _direction_between(end, target)
                reserved_empty = target
                break
            if path is not None:
                break

    if path is None:
        return None

    arrows: list[list[tuple[int, str] | None]] = [[None for _ in range(cols)] for _ in range(rows)]
    for idx, cell in enumerate(path):
        direction = _direction_between(cell, path[idx + 1]) if idx + 1 < len(path) else last_direction
        color = start_color if idx == 0 else common_color
        arrows[cell[0]][cell[1]] = (color, direction)

    free_cells = [
        (r, c)
        for r in range(rows)
        for c in range(cols)
        if arrows[r][c] is None and (reserved_empty is None or (r, c) != reserved_empty)
    ]
    max_distractors = min(5, len(free_cells))
    distractor_count = random.randint(0, max_distractors)
    for cell in random.sample(free_cells, distractor_count):
        arrows[cell[0]][cell[1]] = (common_color, random.choice(_DIR_ORDER))

    arrow_colors = Counter(
        color
        for row in arrows
        for cell in row
        if cell is not None
        for color in [cell[0]]
    )
    if arrow_colors[start_color] != 1 or arrow_colors[common_color] < 2:
        return None

    inp = _render_scene(arrows, cell_size, GEN_BG, GEN_SEPARATOR)
    expected_arrows = _copy_arrows(arrows)
    for r, c in _trace_path(arrows, path[0]):
        _color, direction = expected_arrows[r][c]
        expected_arrows[r][c] = (start_color, direction)
    expected = _render_scene(expected_arrows, cell_size, GEN_BG, GEN_SEPARATOR)

    if np.array_equal(inp, expected):
        return None
    if inp.shape[0] > 30 or inp.shape[1] > 30 or expected.shape[0] > 30 or expected.shape[1] > 30:
        return None

    try:
        solved = main(inp)
    except Exception:
        return None
    if not np.array_equal(solved, expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid chain-arrow ARC input")
