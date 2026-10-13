from __future__ import annotations

import random
from itertools import combinations

import numpy as np


BG = 0
SEPARATOR = 5
ERROR = 6

GRID_SIDE = 3
CELL_SIZE_CHOICES = [4, 5, 6, 7]
EMPTY_COUNT_CHOICES = [1, 2]
PATTERN_AREA_RANGE = (4, 8)
PATTERN_COLOR_COUNT_CHOICES = [2, 3]
PATTERN_COLOR_POOL = [1, 2, 3, 4, 7, 8, 9]
PATTERN_RETRIES = 200
GENERATION_RETRIES = 400


def _regions_between(length: int, lines: list[int]) -> list[tuple[int, int]]:
    bounds = [-1] + sorted(int(x) for x in lines) + [int(length)]
    regions: list[tuple[int, int]] = []
    for left, right in zip(bounds, bounds[1:]):
        if right - left > 1:
            regions.append((left + 1, right - 1))
    return regions


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    (r0, c0), (r1, c1) = coords.min(axis=0), coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _crop_bbox(grid: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = bbox
    return grid[r0 : r1 + 1, c0 : c1 + 1].copy()


def _split_separator_grid(grid: np.ndarray) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    height, width = map(int, grid.shape)
    row_lines = [r for r in range(height) if np.all(grid[r, :] == SEPARATOR)]
    col_lines = [c for c in range(width) if np.all(grid[:, c] == SEPARATOR)]
    row_regions = _regions_between(height, row_lines)
    col_regions = _regions_between(width, col_lines)

    assert len(row_lines) == GRID_SIDE - 1 and len(col_lines) == GRID_SIDE - 1
    assert len(row_regions) == GRID_SIDE and len(col_regions) == GRID_SIDE
    assert len({r1 - r0 + 1 for r0, r1 in row_regions}) == 1
    assert len({c1 - c0 + 1 for c0, c1 in col_regions}) == 1
    return row_regions, col_regions


def _extract_cell(
    grid: np.ndarray,
    row_regions: list[tuple[int, int]],
    col_regions: list[tuple[int, int]],
    row_idx: int,
    col_idx: int,
) -> np.ndarray:
    r0, r1 = row_regions[row_idx]
    c0, c1 = col_regions[col_idx]
    return grid[r0 : r1 + 1, c0 : c1 + 1].copy()


def _write_cell(
    grid: np.ndarray,
    row_regions: list[tuple[int, int]],
    col_regions: list[tuple[int, int]],
    row_idx: int,
    col_idx: int,
    tile: np.ndarray,
) -> None:
    r0, r1 = row_regions[row_idx]
    c0, c1 = col_regions[col_idx]
    assert tile.shape == (r1 - r0 + 1, c1 - c0 + 1)
    grid[r0 : r1 + 1, c0 : c1 + 1] = tile


def _rotate_cw(tile: np.ndarray, quarter_turns: int) -> np.ndarray:
    return np.rot90(tile, k=-(quarter_turns % 4)).copy()


def _rotation_from_position(pos: tuple[int, int], source_pos: tuple[int, int]) -> int:
    return (abs(int(pos[0]) - int(source_pos[0])) + abs(int(pos[1]) - int(source_pos[1]))) % 4


def _matches_with_wildcard(observed: np.ndarray, candidate: np.ndarray) -> bool:
    if observed.shape != candidate.shape:
        return False
    mask = observed != ERROR
    return np.array_equal(observed[mask], candidate[mask])


def _tile_signature(tile: np.ndarray) -> tuple[tuple[int, int], bytes]:
    return tuple(map(int, tile.shape)), tile.tobytes()


def _board_shape(cell_size: int) -> tuple[int, int]:
    side = GRID_SIDE * cell_size + (GRID_SIDE - 1)
    return side, side


def _blank_board(cell_size: int) -> np.ndarray:
    height, width = _board_shape(cell_size)
    grid = np.full((height, width), BG, dtype=int)
    for idx in range(1, GRID_SIDE):
        line = idx * cell_size + (idx - 1)
        grid[line, :] = SEPARATOR
        grid[:, line] = SEPARATOR
    return grid


def _neighbors4(r: int, c: int, height: int, width: int) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        rr = r + dr
        cc = c + dc
        if 0 <= rr < height and 0 <= cc < width:
            out.append((rr, cc))
    return out


def _random_connected_coords(height: int, width: int, count: int) -> list[tuple[int, int]] | None:
    if count > height * width:
        return None

    start = (random.randrange(height), random.randrange(width))
    coords = {start}
    frontier = set(_neighbors4(start[0], start[1], height, width))

    while len(coords) < count:
        frontier = {coord for coord in frontier if coord not in coords}
        if not frontier:
            return None
        chosen = random.choice(tuple(frontier))
        coords.add(chosen)
        frontier.update(_neighbors4(chosen[0], chosen[1], height, width))

    return list(coords)


def _find_corrupted_tile(source_tile: np.ndarray, quarter_turns: int) -> np.ndarray | None:
    target = _rotate_cw(source_tile, quarter_turns)
    fg_coords = [tuple(map(int, rc)) for rc in np.argwhere(target != BG)]
    size_order = [1, 2]
    random.shuffle(size_order)

    for n_errors in size_order:
        if len(fg_coords) < n_errors:
            continue
        candidate_sets = list(combinations(fg_coords, n_errors))
        random.shuffle(candidate_sets)
        for error_cells in candidate_sets:
            corrupted = target.copy()
            for r, c in error_cells:
                corrupted[r, c] = ERROR
            matches = [
                turns
                for turns in range(4)
                if _matches_with_wildcard(corrupted, _rotate_cw(source_tile, turns))
            ]
            if matches == [quarter_turns]:
                return corrupted
    return None


def _generate_source_tile(cell_size: int) -> np.ndarray | None:
    for _ in range(PATTERN_RETRIES):
        n_pixels = random.randint(PATTERN_AREA_RANGE[0], min(PATTERN_AREA_RANGE[1], cell_size * cell_size))
        n_colors = random.choice(PATTERN_COLOR_COUNT_CHOICES)
        if n_colors > n_pixels:
            continue

        box_h = random.randint(2, cell_size)
        box_w = random.randint(2, cell_size)
        if box_h * box_w < n_pixels:
            continue

        coords = _random_connected_coords(box_h, box_w, n_pixels)
        if coords is None:
            continue

        raw = np.full((box_h, box_w), BG, dtype=int)
        colors = random.sample(PATTERN_COLOR_POOL, n_colors)
        color_assignments = colors + [random.choice(colors) for _ in range(n_pixels - n_colors)]
        random.shuffle(color_assignments)
        for (r, c), color in zip(coords, color_assignments):
            raw[r, c] = color

        raw_bbox = _bbox(raw != BG)
        assert raw_bbox is not None
        sprite = _crop_bbox(raw, raw_bbox)
        sprite_h, sprite_w = map(int, sprite.shape)
        tile = np.full((cell_size, cell_size), BG, dtype=int)
        top = random.randint(0, cell_size - sprite_h)
        left = random.randint(0, cell_size - sprite_w)
        tile[top : top + sprite_h, left : left + sprite_w] = sprite

        if len({_tile_signature(_rotate_cw(tile, turns)) for turns in range(4)}) != 4:
            continue
        if any(_find_corrupted_tile(tile, turns) is None for turns in range(4)):
            continue
        return tile
    return None


def _render_solution(source_tile: np.ndarray, source_pos: tuple[int, int]) -> np.ndarray:
    cell_size = int(source_tile.shape[0])
    out = _blank_board(cell_size)
    row_regions, col_regions = _split_separator_grid(out)
    for row_idx in range(GRID_SIDE):
        for col_idx in range(GRID_SIDE):
            turns = _rotation_from_position((row_idx, col_idx), source_pos)
            _write_cell(out, row_regions, col_regions, row_idx, col_idx, _rotate_cw(source_tile, turns))
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Infer the clean tile in a separator grid, rotate it by cell position, and use it to repair or fill every cell.

    Input structure:
    - Grey separator lines split the canvas into a 3x3 grid of equal square cells on a black background.
    - Exactly one non-empty cell contains the full multi-color source tile with no magenta errors.
    - Other non-empty cells contain rotated copies of that tile, but one or two foreground pixels have been replaced by magenta; some cells may be empty.

    Rule:
    - Find the unique non-empty cell without magenta and treat that whole cell as the source tile at position (r_s, c_s).
    - For every cell (r, c), its correct contents are the source tile rotated clockwise by ((|r-r_s| + |c-c_s|) mod 4) quarter-turns.
    - Replace each cell with its correct rotation, which both fixes magenta error pixels and fills empty cells, while leaving the grey separator lines unchanged.

    Key insight:
    - The corrupted copies are only partial witnesses: once the one clean tile is found, the Manhattan-distance position rule determines every required rotation and therefore the full repaired board.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0

    row_regions, col_regions = _split_separator_grid(g)
    cells: dict[tuple[int, int], np.ndarray] = {}
    source_candidates: list[tuple[int, int]] = []

    for row_idx in range(GRID_SIDE):
        for col_idx in range(GRID_SIDE):
            tile = _extract_cell(g, row_regions, col_regions, row_idx, col_idx)
            cells[(row_idx, col_idx)] = tile
            if np.any(tile != BG) and not np.any(tile == ERROR):
                source_candidates.append((row_idx, col_idx))

    assert len(source_candidates) == 1
    source_pos = source_candidates[0]
    source_tile = cells[source_pos]
    assert len({_tile_signature(_rotate_cw(source_tile, turns)) for turns in range(4)}) == 4

    out = g.copy()
    for row_idx in range(GRID_SIDE):
        for col_idx in range(GRID_SIDE):
            turns = _rotation_from_position((row_idx, col_idx), source_pos)
            expected = _rotate_cw(source_tile, turns)
            observed = cells[(row_idx, col_idx)]

            if np.any(observed != BG):
                if np.any(observed == ERROR):
                    matches = [
                        candidate_turns
                        for candidate_turns in range(4)
                        if _matches_with_wildcard(observed, _rotate_cw(source_tile, candidate_turns))
                    ]
                    assert matches == [turns]
                else:
                    assert np.array_equal(observed, expected)

            _write_cell(out, row_regions, col_regions, row_idx, col_idx, expected)

    return out


def _gen_attempt() -> np.ndarray | None:
    cell_size = random.choice(CELL_SIZE_CHOICES)
    source_tile = _generate_source_tile(cell_size)
    if source_tile is None:
        return None

    positions = [(r, c) for r in range(GRID_SIDE) for c in range(GRID_SIDE)]
    source_pos = random.choice(positions)
    other_positions = [pos for pos in positions if pos != source_pos]
    all_turns = {_rotation_from_position(pos, source_pos) for pos in other_positions}

    empty_count = random.choice(EMPTY_COUNT_CHOICES)
    empty_cells: set[tuple[int, int]] | None = None
    for _ in range(100):
        trial = set(random.sample(other_positions, empty_count))
        shown_turns = {_rotation_from_position(pos, source_pos) for pos in other_positions if pos not in trial}
        if shown_turns == all_turns:
            empty_cells = trial
            break
    if empty_cells is None:
        return None

    grid = _blank_board(cell_size)
    row_regions, col_regions = _split_separator_grid(grid)
    _write_cell(grid, row_regions, col_regions, source_pos[0], source_pos[1], source_tile)

    for pos in other_positions:
        if pos in empty_cells:
            continue
        turns = _rotation_from_position(pos, source_pos)
        corrupted = _find_corrupted_tile(source_tile, turns)
        if corrupted is None:
            return None
        _write_cell(grid, row_regions, col_regions, pos[0], pos[1], corrupted)

    expected = _render_solution(source_tile, source_pos)
    try:
        actual = main(grid)
    except AssertionError:
        return None

    if not np.array_equal(actual, expected):
        return None
    if np.array_equal(actual, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
