from __future__ import annotations

import random

import numpy as np


BLACK = 0
RED = 2
BLOCK_SIZE = 3

DIRS = [(-1, 0), (0, 1), (1, 0), (0, -1)]
DIR_TO_OFFSET = {
    0: (0, 1),
    1: (1, 2),
    2: (2, 1),
    3: (1, 0),
}
OFFSET_TO_DIR = {offset: direction for direction, offset in DIR_TO_OFFSET.items()}

SEP_CHOICES = [1, 3, 4, 5, 6, 7, 8, 9]
MARKER_CHOICES = [1, 3, 4, 6, 7, 8, 9]

BLOCK_ROWS_RANGE = (3, 7)
BLOCK_COLS_RANGE = (3, 7)
PATH_BLOCKS_MIN = 5
PATH_BLOCKS_MAX = 10
GEN_ATTEMPTS = 400
PATH_ATTEMPTS = 120

TURN_LEFT = -1
TURN_STRAIGHT = 0
TURN_RIGHT = 1
TURN_REQUIREMENTS = [
    {TURN_LEFT, TURN_STRAIGHT, TURN_RIGHT},
    {TURN_LEFT, TURN_STRAIGHT},
    {TURN_STRAIGHT, TURN_RIGHT},
    {TURN_LEFT, TURN_RIGHT},
]


def _regular_intervals(separators: list[int], length: int) -> list[tuple[int, int]]:
    assert separators, "expected at least one separator line"
    bounds = [-1, *separators, int(length)]
    intervals = [(bounds[i] + 1, bounds[i + 1]) for i in range(len(bounds) - 1)]
    sizes = {end - start for start, end in intervals}
    assert sizes == {BLOCK_SIZE}, "expected a regular 3x3 block grid"
    return intervals


def _separator_layout(grid: np.ndarray) -> tuple[list[tuple[int, int]], list[tuple[int, int]], int]:
    row_candidates = [r for r in range(grid.shape[0]) if np.all(grid[r] == grid[r, 0]) and int(grid[r, 0]) != BLACK]
    col_candidates = [c for c in range(grid.shape[1]) if np.all(grid[:, c] == grid[0, c]) and int(grid[0, c]) != BLACK]
    row_intervals = _regular_intervals(row_candidates, grid.shape[0])
    col_intervals = _regular_intervals(col_candidates, grid.shape[1])
    sep_colors = {int(grid[r, 0]) for r in row_candidates} | {int(grid[0, c]) for c in col_candidates}
    assert len(sep_colors) == 1, "separator rows and columns must share one color"
    return row_intervals, col_intervals, next(iter(sep_colors))


def _turn_kind(prev_dir: int, next_dir: int) -> int:
    delta = (int(next_dir) - int(prev_dir)) % 4
    if delta == 0:
        return TURN_STRAIGHT
    if delta == 1:
        return TURN_RIGHT
    if delta == 3:
        return TURN_LEFT
    raise AssertionError("backward code is forbidden")


def _parse_blocks(grid: np.ndarray) -> tuple[np.ndarray, np.ndarray, int]:
    row_intervals, col_intervals, sep_color = _separator_layout(grid)
    marker_dirs = np.full((len(row_intervals), len(col_intervals)), -1, dtype=int)
    marker_colors = np.full(marker_dirs.shape, -1, dtype=int)

    for br, (r0, r1) in enumerate(row_intervals):
        for bc, (c0, c1) in enumerate(col_intervals):
            block = grid[r0:r1, c0:c1]
            coords = np.argwhere(block != BLACK)
            assert coords.shape == (1, 2), "each block must contain exactly one marker pixel"
            rr, cc = map(int, coords[0])
            assert (rr, cc) in OFFSET_TO_DIR, "marker must sit one step from block center"
            marker_dirs[br, bc] = OFFSET_TO_DIR[(rr, cc)]
            marker_colors[br, bc] = int(block[rr, cc])

    red_blocks = np.argwhere(marker_colors == RED)
    assert red_blocks.shape == (1, 2), "expected exactly one red start block"
    return marker_dirs, marker_colors, sep_color


def _trace_path(marker_dirs: np.ndarray, marker_colors: np.ndarray) -> tuple[list[tuple[int, int]], list[int]]:
    starts = np.argwhere(marker_colors == RED)
    assert starts.shape == (1, 2)
    pos = tuple(map(int, starts[0]))
    visited: list[tuple[int, int]] = []
    seen: set[tuple[int, int]] = set()
    turns: list[int] = []
    current_dir = int(marker_dirs[pos])
    rows, cols = marker_dirs.shape

    while True:
        assert pos not in seen, "path cannot revisit a block"
        seen.add(pos)
        visited.append(pos)

        nr = pos[0] + DIRS[current_dir][0]
        nc = pos[1] + DIRS[current_dir][1]
        if not (0 <= nr < rows and 0 <= nc < cols):
            return visited, turns

        pos = (nr, nc)
        next_code = int(marker_dirs[pos])
        turn = _turn_kind(current_dir, next_code)
        turns.append(turn)
        current_dir = (current_dir + turn) % 4


def _render_output(marker_colors: np.ndarray, visited: list[tuple[int, int]], sep_color: int) -> np.ndarray:
    rows, cols = marker_colors.shape
    height = rows * BLOCK_SIZE + (rows - 1)
    width = cols * BLOCK_SIZE + (cols - 1)
    out = np.zeros((height, width), dtype=int)

    for r in range(3, height, 4):
        out[r, :] = sep_color
    for c in range(3, width, 4):
        out[:, c] = sep_color

    for br, bc in visited:
        r0 = br * (BLOCK_SIZE + 1)
        c0 = bc * (BLOCK_SIZE + 1)
        out[r0 : r0 + BLOCK_SIZE, c0 : c0 + BLOCK_SIZE] = int(marker_colors[br, bc])
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Follow the block-level turning chain from the unique red start and solid-fill only the visited blocks.

    Input structure:
    - The grid is partitioned by one-color separator rows and columns into 3x3 blocks.
    - Every block contains exactly one non-black marker pixel, placed one cell above, below, left, or right of the block center.
    - Exactly one marker is red, and that red marker identifies the start block.

    Rule:
    - In the start block, the red marker's offset from the center gives the absolute first movement direction.
    - After entering any later block, read that block's marker offset relative to the current movement direction: same offset means straight, left-offset means turn left, right-offset means turn right, and the backward case never appears.
    - Keep walking from block to adjacent block until the next step would leave the block grid.
    - Preserve the separator lines; fill each visited block solid with its own marker color; clear every unvisited block to black.

    Key insight:
    - The red block is the only absolute anchor; every later marker must be interpreted as a turn code chained from the previous movement, while marker colors themselves are decoys except for the unique red start.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    marker_dirs, marker_colors, sep_color = _parse_blocks(g)
    visited, _turns = _trace_path(marker_dirs, marker_colors)
    return _render_output(marker_colors, visited, sep_color)


def _neighbors(cell: tuple[int, int], rows: int, cols: int) -> list[tuple[tuple[int, int], int]]:
    r, c = cell
    out = []
    for direction, (dr, dc) in enumerate(DIRS):
        nr, nc = r + dr, c + dc
        if 0 <= nr < rows and 0 <= nc < cols:
            out.append(((nr, nc), direction))
    random.shuffle(out)
    return out


def _search_path(
    rows: int,
    cols: int,
    target_blocks: int,
    required_turns: set[int],
) -> tuple[list[tuple[int, int]], list[int], list[int]] | None:
    starts = [(r, c) for r in range(rows) for c in range(cols)]
    random.shuffle(starts)

    def dfs(
        path: list[tuple[int, int]],
        codes: list[int],
        seen_turns: set[int],
        used: set[tuple[int, int]],
    ) -> tuple[list[tuple[int, int]], list[int], list[int]] | None:
        current = path[-1]
        if len(path) == target_blocks:
            outward = []
            for direction, (dr, dc) in enumerate(DIRS):
                nr, nc = current[0] + dr, current[1] + dc
                if 0 <= nr < rows and 0 <= nc < cols:
                    continue
                if codes and direction == (codes[-1] + 2) % 4:
                    continue
                outward.append(direction)
            random.shuffle(outward)
            for exit_dir in outward:
                final_turns = set(seen_turns)
                relations = []
                if codes:
                    relations = [_turn_kind(codes[i - 1], codes[i]) for i in range(1, len(codes))]
                    final_turns = set(relations)
                    final_turns.add(_turn_kind(codes[-1], exit_dir))
                if not required_turns.issubset(final_turns):
                    continue
                all_codes = codes + [exit_dir]
                all_relations = [_turn_kind(all_codes[i - 1], all_codes[i]) for i in range(1, len(all_codes))]
                if len(set(all_relations)) < 2:
                    continue
                return path[:], all_codes, all_relations
            return None

        candidates = _neighbors(current, rows, cols)
        scored = []
        for next_cell, direction in candidates:
            if next_cell in used:
                continue
            score = random.random()
            if codes:
                rel = _turn_kind(codes[-1], direction)
                if rel in required_turns - seen_turns:
                    score -= 1.0
            scored.append((score, next_cell, direction))
        scored.sort(key=lambda item: item[0])

        for _score, next_cell, direction in scored:
            next_turns = set(seen_turns)
            if codes:
                next_turns.add(_turn_kind(codes[-1], direction))
            remaining_opportunities = target_blocks - (len(path) + 1) + 1
            if len(required_turns - next_turns) > remaining_opportunities:
                continue
            used.add(next_cell)
            path.append(next_cell)
            codes.append(direction)
            found = dfs(path, codes, next_turns, used)
            if found is not None:
                return found
            codes.pop()
            path.pop()
            used.remove(next_cell)
        return None

    for start in starts:
        result = dfs([start], [], set(), {start})
        if result is not None:
            return result
    return None


def _random_path_plan(rows: int, cols: int) -> tuple[list[tuple[int, int]], list[int], list[int]] | None:
    total = rows * cols
    max_blocks = min(PATH_BLOCKS_MAX, total - 1)
    if max_blocks < PATH_BLOCKS_MIN:
        return None

    requirements = TURN_REQUIREMENTS[:]
    random.shuffle(requirements)
    lengths = list(range(PATH_BLOCKS_MIN, max_blocks + 1))
    random.shuffle(lengths)

    for _ in range(PATH_ATTEMPTS):
        target_blocks = random.choice(lengths)
        viable = [req for req in requirements if len(req) <= target_blocks - 1]
        required_turns = random.choice(viable)
        result = _search_path(rows, cols, target_blocks, required_turns)
        if result is not None:
            return result
    return None


def _choose_separator_and_palette() -> tuple[int, list[int]]:
    sep_color = random.choice(SEP_CHOICES)
    marker_palette = [color for color in MARKER_CHOICES if color != sep_color]
    return sep_color, marker_palette


def _sample_colors(total_blocks: int, path: list[tuple[int, int]], palette: list[int]) -> np.ndarray:
    colors = np.random.choice(palette, size=total_blocks, replace=True).astype(int)
    if len(path) > 1:
        while len(set(map(int, colors[: len(path) - 1]))) < 2:
            colors = np.random.choice(palette, size=total_blocks, replace=True).astype(int)
    return colors


def _build_input(
    rows: int,
    cols: int,
    path: list[tuple[int, int]],
    codes: list[int],
    sep_color: int,
    palette: list[int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    marker_dirs = np.random.randint(0, 4, size=(rows, cols), dtype=int)
    marker_colors = np.random.choice(palette, size=(rows, cols), replace=True).astype(int)

    flat_colors = _sample_colors(rows * cols, path, palette)
    marker_colors[:, :] = flat_colors.reshape(rows, cols)

    for idx, (br, bc) in enumerate(path):
        marker_dirs[br, bc] = codes[idx]
        if idx == 0:
            marker_colors[br, bc] = RED
        else:
            marker_colors[br, bc] = int(random.choice(palette))

    height = rows * BLOCK_SIZE + (rows - 1)
    width = cols * BLOCK_SIZE + (cols - 1)
    grid = np.zeros((height, width), dtype=int)

    for r in range(3, height, 4):
        grid[r, :] = sep_color
    for c in range(3, width, 4):
        grid[:, c] = sep_color

    for br in range(rows):
        for bc in range(cols):
            rr, cc = DIR_TO_OFFSET[int(marker_dirs[br, bc])]
            r0 = br * (BLOCK_SIZE + 1)
            c0 = bc * (BLOCK_SIZE + 1)
            grid[r0 + rr, c0 + cc] = int(marker_colors[br, bc])

    return grid, marker_dirs, marker_colors


def _validate_generated(
    grid: np.ndarray,
    expected_path: list[tuple[int, int]],
    expected_turns: list[int],
    expected_colors: np.ndarray,
    expected_sep: int,
) -> bool:
    try:
        marker_dirs, marker_colors, sep_color = _parse_blocks(grid)
        if sep_color != expected_sep:
            return False
        if not np.array_equal(marker_colors, expected_colors):
            return False
        path, turns = _trace_path(marker_dirs, marker_colors)
        if path != expected_path:
            return False
        if turns != expected_turns:
            return False
        if len(path) >= marker_dirs.size:
            return False
        if len(set(turns)) < 2:
            return False
        red_count = int(np.count_nonzero(grid == RED))
        if red_count != 1:
            return False
        expected_out = _render_output(expected_colors, expected_path, expected_sep)
        produced_out = main(grid)
        if not np.array_equal(produced_out, expected_out):
            return False
        visited_colors = [int(expected_colors[br, bc]) for br, bc in expected_path]
        if len(set(visited_colors)) < 2:
            return False
    except Exception:
        return False
    return True


def _generate_one() -> np.ndarray | None:
    rows = random.randint(*BLOCK_ROWS_RANGE)
    cols = random.randint(*BLOCK_COLS_RANGE)
    if rows * BLOCK_SIZE + (rows - 1) > 30 or cols * BLOCK_SIZE + (cols - 1) > 30:
        return None

    plan = _random_path_plan(rows, cols)
    if plan is None:
        return None
    path, codes, turns = plan

    sep_color, palette = _choose_separator_and_palette()
    grid, _dirs, colors = _build_input(rows, cols, path, codes, sep_color, palette)
    if not _validate_generated(grid, path, turns, colors, sep_color):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _generate_one()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid turning-pointer ARC sample")
