from __future__ import annotations

import random

import numpy as np


BG = 0
SEP = 5
MARK = 9
COLOR_POOL = (1, 2, 3, 4, 6, 7, 8)
CORNER_ORDER = ("tl", "tr", "br", "bl")
CORNER_TO_ROT90 = {
    "tl": 0,  # 0 degrees
    "tr": 3,  # 90 clockwise
    "br": 2,  # 180
    "bl": 1,  # 270 clockwise
}
MAX_GENERATION_ATTEMPTS = 400


def _regions_between(size: int, separator_lines: list[int]) -> list[tuple[int, int]]:
    regions: list[tuple[int, int]] = []
    start = 0
    for line in sorted(separator_lines):
        assert start < line < size - 1, "separator lines must split non-empty regions"
        regions.append((start, line - 1))
        start = line + 1
    assert start < size, "expected a trailing region after the separators"
    regions.append((start, size - 1))
    return regions


def _parse_separator_grid(grid: np.ndarray) -> tuple[list[tuple[int, int]], list[tuple[int, int]], int]:
    height, width = map(int, grid.shape)
    row_lines = [r for r in range(height) if np.all(grid[r, :] == SEP)]
    col_lines = [c for c in range(width) if np.all(grid[:, c] == SEP)]
    assert row_lines and col_lines, "expected full separator rows and columns"

    row_regions = _regions_between(height, row_lines)
    col_regions = _regions_between(width, col_lines)
    assert len(row_regions) == len(col_regions), "expected a square cell board"
    assert len(row_regions) in (3, 4), "expected a 3x3 or 4x4 board"

    cell_heights = {r1 - r0 + 1 for r0, r1 in row_regions}
    cell_widths = {c1 - c0 + 1 for c0, c1 in col_regions}
    assert len(cell_heights) == 1 and len(cell_widths) == 1, "expected equal-sized cells"
    cell_h = next(iter(cell_heights))
    cell_w = next(iter(cell_widths))
    assert cell_h == cell_w, "rotation requires square cells"
    return row_regions, col_regions, int(cell_h)


def _corner_name(local_r: int, local_c: int, size: int) -> str | None:
    corner_map = {
        (0, 0): "tl",
        (0, size - 1): "tr",
        (size - 1, size - 1): "br",
        (size - 1, 0): "bl",
    }
    return corner_map.get((int(local_r), int(local_c)))


def _corner_coord(corner: str, size: int) -> tuple[int, int]:
    if corner == "tl":
        return 0, 0
    if corner == "tr":
        return 0, size - 1
    if corner == "br":
        return size - 1, size - 1
    if corner == "bl":
        return size - 1, 0
    raise ValueError(f"unknown corner: {corner}")


def _locate_source_and_targets(
    grid: np.ndarray,
    row_regions: list[tuple[int, int]],
    col_regions: list[tuple[int, int]],
    cell_size: int,
) -> tuple[tuple[int, int, np.ndarray], list[tuple[int, int, str]]]:
    source: tuple[int, int, np.ndarray] | None = None
    targets: list[tuple[int, int, str]] = []

    for row_idx, (r0, r1) in enumerate(row_regions):
        for col_idx, (c0, c1) in enumerate(col_regions):
            cell = grid[r0 : r1 + 1, c0 : c1 + 1]
            active = np.argwhere((cell != BG) & (cell != SEP))
            marks = np.argwhere(cell == MARK)
            tile_pixels = np.argwhere((cell != BG) & (cell != SEP) & (cell != MARK))

            if len(tile_pixels) > 0:
                assert source is None, "expected exactly one populated tile cell"
                assert len(active) >= 2, "source tile must contain multiple visible pixels"
                source = (row_idx, col_idx, cell.copy())
                continue

            assert len(active) == 1 and len(marks) == 1, "each empty cell must contain exactly one corner mark"
            mark_r, mark_c = map(int, marks[0])
            corner = _corner_name(mark_r, mark_c, cell_size)
            assert corner is not None, "mark must sit in a cell corner"
            targets.append((row_idx, col_idx, corner))

    assert source is not None, "expected one source tile"
    assert len(targets) == len(row_regions) * len(col_regions) - 1, "all other cells must be marked targets"
    return source, targets


def _render_cell(tile: np.ndarray, corner: str) -> np.ndarray:
    out = np.rot90(tile, CORNER_TO_ROT90[corner]).copy()
    mark_r, mark_c = _corner_coord(corner, out.shape[0])
    out[mark_r, mark_c] = MARK
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Rotate the single source tile into every marked separator-grid cell using the mark corner as the rotation code.

    Input structure:
    - A black background contains grey full-width and full-height separator lines that partition the grid into a 3x3 or 4x4 board of equal square cells.
    - Exactly one cell contains a multi-color non-grey tile pattern.
    - Every other cell is otherwise black except for one maroon mark pixel placed in exactly one corner.

    Rule:
    - Use the grey lines to recover the cell board and extract the unique populated cell as the source tile, keeping its full square cell contents.
    - For each marked empty cell, read the mark corner as a rotation code: top-left -> 0 degrees, top-right -> 90 degrees clockwise, bottom-right -> 180 degrees, bottom-left -> 270 degrees clockwise.
    - Stamp the rotated source tile into that cell, then restore the maroon mark on top; keep the source cell and all grey separator lines unchanged.

    Key insight:
    - The corner mark is both the destination indicator and the orientation code, so the solver must use the separator-defined cell frame before rotating the source tile.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"

    row_regions, col_regions, cell_size = _parse_separator_grid(g)
    source, targets = _locate_source_and_targets(g, row_regions, col_regions, cell_size)
    source_row, source_col, tile = source

    out = g.copy()
    out[
        row_regions[source_row][0] : row_regions[source_row][1] + 1,
        col_regions[source_col][0] : col_regions[source_col][1] + 1,
    ] = tile

    for row_idx, col_idx, corner in targets:
        r0, r1 = row_regions[row_idx]
        c0, c1 = col_regions[col_idx]
        out[r0 : r1 + 1, c0 : c1 + 1] = _render_cell(tile, corner)

    return out


def _neighbors4_in_square(r: int, c: int, size: int) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        rr = r + dr
        cc = c + dc
        if 0 <= rr < size and 0 <= cc < size:
            out.append((rr, cc))
    return out


def _connected_shape(size: int, pixel_count: int) -> list[tuple[int, int]] | None:
    start = (random.randrange(size), random.randrange(size))
    coords = {start}
    frontier = set(_neighbors4_in_square(start[0], start[1], size))
    while len(coords) < pixel_count:
        frontier -= coords
        if not frontier:
            return None
        chosen = random.choice(tuple(frontier))
        coords.add(chosen)
        frontier.update(_neighbors4_in_square(chosen[0], chosen[1], size))
    return sorted(coords)


def _tile_ok(tile: np.ndarray) -> bool:
    used_colors = sorted(int(c) for c in np.unique(tile) if int(c) != BG)
    if not (2 <= len(used_colors) <= 3):
        return False
    if SEP in used_colors or MARK in used_colors:
        return False
    if np.count_nonzero(tile == BG) == 0:
        return False

    rotations = [np.rot90(tile, k) for k in range(4)]
    if len({rot.tobytes() for rot in rotations}) != 4:
        return False

    rendered = {corner: _render_cell(tile, corner) for corner in CORNER_ORDER}
    if len({cell.tobytes() for cell in rendered.values()}) != 4:
        return False

    for corner in CORNER_ORDER:
        variants = {
            np.rot90(tile, k).copy().tobytes(): k for k in range(4)
        }
        correct = _render_cell(tile, corner)
        matches = 0
        for k in range(4):
            wrong = np.rot90(tile, k).copy()
            mr, mc = _corner_coord(corner, tile.shape[0])
            wrong[mr, mc] = MARK
            matches += int(np.array_equal(correct, wrong))
        if matches != 1:
            return False

    for corner in ("tr", "br", "bl"):
        plain = tile.copy()
        mr, mc = _corner_coord(corner, tile.shape[0])
        plain[mr, mc] = MARK
        if np.array_equal(plain, rendered[corner]):
            return False

    return True


def _generate_tile(cell_size: int) -> np.ndarray | None:
    area = cell_size * cell_size
    min_pixels = 4 if cell_size == 3 else 5
    max_pixels = 6 if cell_size == 3 else 8

    for _ in range(200):
        pixel_count = random.randint(min_pixels, min(max_pixels, area - 1))
        coords = _connected_shape(cell_size, pixel_count)
        if coords is None:
            continue

        n_colors = random.randint(2, 3)
        colors = random.sample(COLOR_POOL, n_colors)
        order = coords[:]
        random.shuffle(order)

        tile = np.zeros((cell_size, cell_size), dtype=int)
        for idx, (r, c) in enumerate(order):
            if idx < n_colors:
                color = colors[idx]
            else:
                color = random.choice(colors)
            tile[r, c] = color

        if _tile_ok(tile):
            return tile
    return None


def _build_separator_grid(board_size: int, cell_size: int) -> tuple[np.ndarray, list[tuple[int, int]], list[tuple[int, int]]]:
    side = board_size * cell_size + (board_size - 1)
    grid = np.full((side, side), BG, dtype=int)
    for idx in range(1, board_size):
        sep = idx * cell_size + (idx - 1)
        grid[sep, :] = SEP
        grid[:, sep] = SEP

    row_regions = _regions_between(side, [r for r in range(side) if np.all(grid[r, :] == SEP)])
    col_regions = _regions_between(side, [c for c in range(side) if np.all(grid[:, c] == SEP)])
    return grid, row_regions, col_regions


def _render_expected(
    grid: np.ndarray,
    row_regions: list[tuple[int, int]],
    col_regions: list[tuple[int, int]],
    source_pos: tuple[int, int],
    tile: np.ndarray,
    target_corners: dict[tuple[int, int], str],
) -> np.ndarray:
    out = grid.copy()
    sr, sc = source_pos
    r0, r1 = row_regions[sr]
    c0, c1 = col_regions[sc]
    out[r0 : r1 + 1, c0 : c1 + 1] = tile
    for (rr, cc), corner in target_corners.items():
        r0, r1 = row_regions[rr]
        c0, c1 = col_regions[cc]
        out[r0 : r1 + 1, c0 : c1 + 1] = _render_cell(tile, corner)
    return out


def _semantic_case_ok(
    inp: np.ndarray,
    expected: np.ndarray,
    source_pos: tuple[int, int],
    target_corners: dict[tuple[int, int], str],
) -> bool:
    try:
        solved = main(inp)
    except AssertionError:
        return False
    if not np.array_equal(solved, expected):
        return False

    row_regions, col_regions, cell_size = _parse_separator_grid(inp)
    (src_r, src_c, tile), targets = _locate_source_and_targets(inp, row_regions, col_regions, cell_size)
    if (src_r, src_c) != source_pos:
        return False
    if {corner for _, _, corner in targets} != set(CORNER_ORDER):
        return False

    for (rr, cc), corner in target_corners.items():
        r0, r1 = row_regions[rr]
        c0, c1 = col_regions[cc]
        block = solved[r0 : r1 + 1, c0 : c1 + 1]
        if not np.array_equal(block, _render_cell(tile, corner)):
            return False

    naive = inp.copy()
    for (rr, cc), corner in target_corners.items():
        r0, r1 = row_regions[rr]
        c0, c1 = col_regions[cc]
        plain = tile.copy()
        mr, mc = _corner_coord(corner, tile.shape[0])
        plain[mr, mc] = MARK
        naive[r0 : r1 + 1, c0 : c1 + 1] = plain
    if np.array_equal(naive, expected):
        return False

    return True


def _gen_attempt() -> np.ndarray | None:
    board_size = random.choice((3, 4))
    cell_size = random.choice((3, 4))
    tile = _generate_tile(cell_size)
    if tile is None:
        return None

    grid, row_regions, col_regions = _build_separator_grid(board_size, cell_size)
    source_pos = (random.randrange(board_size), random.randrange(board_size))

    all_cells = [(r, c) for r in range(board_size) for c in range(board_size)]
    target_cells = [cell for cell in all_cells if cell != source_pos]
    corners = list(CORNER_ORDER)
    while len(corners) < len(target_cells):
        corners.append(random.choice(CORNER_ORDER))
    random.shuffle(corners)
    target_corners = dict(zip(target_cells, corners))

    sr0, sr1 = row_regions[source_pos[0]]
    sc0, sc1 = col_regions[source_pos[1]]
    grid[sr0 : sr1 + 1, sc0 : sc1 + 1] = tile

    for (rr, cc), corner in target_corners.items():
        r0, r1 = row_regions[rr]
        c0, c1 = col_regions[cc]
        mark_r, mark_c = _corner_coord(corner, cell_size)
        grid[r0 + mark_r, c0 + mark_c] = MARK

    expected = _render_expected(grid, row_regions, col_regions, source_pos, tile, target_corners)
    if not _semantic_case_ok(grid, expected, source_pos, target_corners):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        candidate = _gen_attempt()
        if candidate is not None:
            return candidate
    raise RuntimeError("failed to generate a valid rotation-coded tile input")
