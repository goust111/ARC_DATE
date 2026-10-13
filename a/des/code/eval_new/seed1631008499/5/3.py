from __future__ import annotations

import random

import numpy as np


CELL_SIZE = 5
CELL_CENTER = 2
LINE_WIDTH = 1
GRID_CHOICES = (4, 5)
BG_COLOR = 0
GREY_COLOR = 5
SEED_COLORS = (1, 2, 3, 4, 7)
EQUAL_OVERLAP_COUNTS = (3, 4)
UNEQUAL_OVERLAP_PAIRS = (
    (1, 4),
    (4, 1),
    (2, 3),
    (3, 2),
    (2, 4),
    (4, 2),
    (3, 4),
    (4, 3),
)
MAX_GENERATION_ATTEMPTS = 400


def _non_line_blocks(length: int, lines: set[int]) -> list[list[int]]:
    blocks: list[list[int]] = []
    current: list[int] = []
    for idx in range(length):
        if idx in lines:
            if current:
                blocks.append(current)
                current = []
            continue
        current.append(idx)
    if current:
        blocks.append(current)
    return blocks


def _detect_geometry(grid: np.ndarray) -> tuple[int, list[int], list[int], list[list[int]], list[list[int]]]:
    height, width = map(int, grid.shape)
    candidates: list[tuple[int, int, list[int], list[int], list[list[int]], list[list[int]]]] = []

    for color in map(int, np.unique(grid)):
        line_rows = [r for r in range(height) if bool(np.all(grid[r, :] == color))]
        line_cols = [c for c in range(width) if bool(np.all(grid[:, c] == color))]
        if not line_rows or not line_cols:
            continue

        row_blocks = _non_line_blocks(height, set(line_rows))
        col_blocks = _non_line_blocks(width, set(line_cols))
        if (
            row_blocks
            and col_blocks
            and len(row_blocks) == len(col_blocks)
            and len(row_blocks) >= 2
            and all(len(block) == CELL_SIZE for block in row_blocks)
            and all(len(block) == CELL_SIZE for block in col_blocks)
        ):
            score = len(line_rows) + len(line_cols)
            candidates.append((score, color, line_rows, line_cols, row_blocks, col_blocks))

    assert candidates, "expected a uniform grid of full-row/full-column separators"
    candidates.sort(key=lambda item: item[0], reverse=True)
    _score, grey, line_rows, line_cols, row_blocks, col_blocks = candidates[0]
    return int(grey), line_rows, line_cols, row_blocks, col_blocks


def _dominant_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _extract_cells(board: np.ndarray, bg: int, cell_count: int) -> tuple[np.ndarray, np.ndarray]:
    counts = np.zeros((cell_count, cell_count), dtype=int)
    colors = np.full((cell_count, cell_count), -1, dtype=int)

    for cell_r in range(cell_count):
        for cell_c in range(cell_count):
            r0 = cell_r * CELL_SIZE
            c0 = cell_c * CELL_SIZE
            block = board[r0 : r0 + CELL_SIZE, c0 : c0 + CELL_SIZE]
            values = [int(v) for v in block.reshape(-1) if int(v) != bg]
            if not values:
                continue
            assert len(set(values)) == 1, "every non-empty cell must contain same-color seeds"
            assert 1 <= len(values) <= 4, "seed count must be between 1 and 4"
            counts[cell_r, cell_c] = len(values)
            colors[cell_r, cell_c] = int(values[0])

    return counts, colors


def _paint_crosses(counts: np.ndarray, colors: np.ndarray, bg: int) -> np.ndarray:
    cell_count = int(counts.shape[0])
    board_size = cell_count * CELL_SIZE
    out = np.full((board_size, board_size), bg, dtype=int)
    best_len = np.full((board_size, board_size), -1, dtype=int)
    best_native = np.full((board_size, board_size), -1, dtype=int)

    for cell_r in range(cell_count):
        for cell_c in range(cell_count):
            arm_len = int(counts[cell_r, cell_c])
            if arm_len <= 0:
                continue

            color = int(colors[cell_r, cell_c])
            center_r = cell_r * CELL_SIZE + CELL_CENTER
            center_c = cell_c * CELL_SIZE + CELL_CENTER

            pixels = [(center_r, center_c)]
            for step in range(1, arm_len + 1):
                pixels.extend(
                    [
                        (center_r - step, center_c),
                        (center_r + step, center_c),
                        (center_r, center_c - step),
                        (center_r, center_c + step),
                    ]
                )

            for rr, cc in pixels:
                if not (0 <= rr < board_size and 0 <= cc < board_size):
                    continue
                native = int((rr // CELL_SIZE, cc // CELL_SIZE) == (cell_r, cell_c))
                if arm_len > best_len[rr, cc] or (
                    arm_len == best_len[rr, cc] and native > best_native[rr, cc]
                ):
                    out[rr, cc] = color
                    best_len[rr, cc] = arm_len
                    best_native[rr, cc] = native

    return out


def _expand_logical_board(board: np.ndarray, bg: int, grey: int) -> np.ndarray:
    cell_count = int(board.shape[0] // CELL_SIZE)
    size = cell_count * CELL_SIZE + (cell_count - 1) * LINE_WIDTH
    out = np.full((size, size), bg, dtype=int)

    for k in range(1, cell_count):
        line_idx = k * CELL_SIZE + (k - 1) * LINE_WIDTH
        out[line_idx, :] = grey
        out[:, line_idx] = grey

    pitch = CELL_SIZE + LINE_WIDTH
    for cell_r in range(cell_count):
        for cell_c in range(cell_count):
            r0 = cell_r * pitch
            c0 = cell_c * pitch
            src_r0 = cell_r * CELL_SIZE
            src_c0 = cell_c * CELL_SIZE
            out[r0 : r0 + CELL_SIZE, c0 : c0 + CELL_SIZE] = board[
                src_r0 : src_r0 + CELL_SIZE, src_c0 : src_c0 + CELL_SIZE
            ]

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Grow center crosses inside a grey-divided cell board based on the seed count in each cell.

    Input structure:
    - Black background fills the usable area between uniform grey separator rows and columns.
    - The grey lines split the board into an N x N array of cells, each with a 5x5 usable interior.
    - A non-empty cell contains 1 to 4 seed pixels of one color; empty cells contain none.

    Rule:
    - For every non-empty cell, count its seed pixels and use that count as the arm length of a
      plus sign centered in the cell's middle pixel.
    - Paint the center and the four arms in the seed color; when an arm crosses a separator,
      skip the grey line and keep extending through the neighboring usable cells.
    - If multiple crosses reach the same usable pixel, the cross from the larger seed count wins;
      if the counts are equal, keep the color of the cross whose home cell contains that pixel.
    - Remove the original seed placement and keep all grey separator lines unchanged.

    Key insight:
    - The number of seeds is both the growth length and the overlap priority, so the solver must
      read each 5x5 cell structurally rather than interpreting colors numerically.
    """
    g = np.asarray(grid, dtype=int)
    grey, line_rows, line_cols, row_blocks, col_blocks = _detect_geometry(g)
    usable_rows = [r for block in row_blocks for r in block]
    usable_cols = [c for block in col_blocks for c in block]

    board = g[np.ix_(usable_rows, usable_cols)]
    bg = _dominant_color(board)
    cell_count = len(row_blocks)
    counts, colors = _extract_cells(board, bg, cell_count)
    painted = _paint_crosses(counts, colors, bg)

    out = np.full_like(g, bg)
    out[line_rows, :] = grey
    out[:, line_cols] = grey
    out[np.ix_(usable_rows, usable_cols)] = painted
    return out


def _neighbors4(cell: tuple[int, int], size: int) -> list[tuple[int, int]]:
    r, c = cell
    out: list[tuple[int, int]] = []
    for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        rr, cc = r + dr, c + dc
        if 0 <= rr < size and 0 <= cc < size:
            out.append((rr, cc))
    return out


def _group_is_clear(cells: list[tuple[int, int]], occupied: set[tuple[int, int]], size: int) -> bool:
    cell_set = set(cells)
    for cell in cells:
        if cell in occupied:
            return False
        for neighbor in _neighbors4(cell, size):
            if neighbor in occupied and neighbor not in cell_set:
                return False
    return True


def _pair_candidates(size: int, occupied: set[tuple[int, int]], orient: str) -> list[list[tuple[int, int]]]:
    candidates: list[list[tuple[int, int]]] = []
    for r in range(size):
        for c in range(size):
            if orient == "h" and c + 1 < size:
                cells = [(r, c), (r, c + 1)]
            elif orient == "v" and r + 1 < size:
                cells = [(r, c), (r + 1, c)]
            else:
                continue
            if _group_is_clear(cells, occupied, size):
                candidates.append(cells)
    return candidates


def _single_candidates(size: int, occupied: set[tuple[int, int]]) -> list[tuple[int, int]]:
    return [
        (r, c)
        for r in range(size)
        for c in range(size)
        if _group_is_clear([(r, c)], occupied, size)
    ]


def _sample_seed_positions(count: int) -> list[tuple[int, int]]:
    cells = [(r, c) for r in range(CELL_SIZE) for c in range(CELL_SIZE)]
    for _ in range(200):
        picked = random.sample(cells, count)
        rows = {r for r, _ in picked}
        cols = {c for _, c in picked}
        if count >= 2 and (len(rows) == 1 or len(cols) == 1):
            continue
        if count >= 2 and (2, 2) in picked and all(r == 2 or c == 2 for r, c in picked):
            continue
        return picked
    return random.sample(cells, count)


def _adjacent_overlap_types(counts: np.ndarray) -> tuple[bool, bool, int]:
    equal_overlap = False
    unequal_overlap = False
    edge_count = 0
    size = int(counts.shape[0])

    for r in range(size):
        for c in range(size):
            here = int(counts[r, c])
            if here <= 0:
                continue
            for rr, cc in ((r + 1, c), (r, c + 1)):
                if rr >= size or cc >= size:
                    continue
                other = int(counts[rr, cc])
                if other <= 0:
                    continue
                edge_count += 1
                if here + other >= CELL_SIZE:
                    if here == other:
                        equal_overlap = True
                    else:
                        unequal_overlap = True

    return equal_overlap, unequal_overlap, edge_count


def _build_input(counts: np.ndarray, colors: np.ndarray) -> np.ndarray:
    size = int(counts.shape[0])
    board = np.full((size * CELL_SIZE, size * CELL_SIZE), BG_COLOR, dtype=int)

    for cell_r in range(size):
        for cell_c in range(size):
            count = int(counts[cell_r, cell_c])
            if count <= 0:
                continue
            color = int(colors[cell_r, cell_c])
            r0 = cell_r * CELL_SIZE
            c0 = cell_c * CELL_SIZE
            for rr, cc in _sample_seed_positions(count):
                board[r0 + rr, c0 + cc] = color

    return _expand_logical_board(board, BG_COLOR, GREY_COLOR)


def _generate_scene() -> tuple[np.ndarray, np.ndarray]:
    size = random.choice(GRID_CHOICES)
    counts = np.zeros((size, size), dtype=int)
    colors = np.full((size, size), -1, dtype=int)
    occupied: set[tuple[int, int]] = set()

    equal_orient = random.choice(("h", "v"))
    equal_candidates = _pair_candidates(size, occupied, equal_orient)
    if not equal_candidates:
        equal_candidates = _pair_candidates(size, occupied, "v" if equal_orient == "h" else "h")
    if not equal_candidates:
        raise RuntimeError("no room for equal-overlap pair")
    equal_cells = random.choice(equal_candidates)
    equal_count = random.choice(EQUAL_OVERLAP_COUNTS)
    for cell, color in zip(equal_cells, random.sample(SEED_COLORS, 2), strict=False):
        counts[cell] = equal_count
        colors[cell] = color
        occupied.add(cell)

    unequal_orient = random.choice(("h", "v"))
    unequal_candidates = _pair_candidates(size, occupied, unequal_orient)
    if not unequal_candidates:
        unequal_candidates = _pair_candidates(size, occupied, "v" if unequal_orient == "h" else "h")
    if not unequal_candidates:
        raise RuntimeError("no room for unequal-overlap pair")
    unequal_cells = random.choice(unequal_candidates)
    unequal_counts = random.choice(UNEQUAL_OVERLAP_PAIRS)
    for cell, count, color in zip(unequal_cells, unequal_counts, random.sample(SEED_COLORS, 2), strict=False):
        counts[cell] = count
        colors[cell] = color
        occupied.add(cell)

    missing_counts = [count for count in range(1, CELL_SIZE) if count not in set(map(int, counts.reshape(-1)))]
    random.shuffle(missing_counts)
    for missing in missing_counts:
        candidates = _single_candidates(size, occupied)
        if not candidates:
            raise RuntimeError("no room to cover all seed-count branches")
        cell = random.choice(candidates)
        counts[cell] = missing
        colors[cell] = random.choice(SEED_COLORS)
        occupied.add(cell)

    for _ in range(random.randint(0, 2)):
        candidates = _single_candidates(size, occupied)
        if not candidates:
            break
        cell = random.choice(candidates)
        counts[cell] = random.randint(1, 4)
        colors[cell] = random.choice(SEED_COLORS)
        occupied.add(cell)

    equal_overlap, unequal_overlap, edge_count = _adjacent_overlap_types(counts)
    assert equal_overlap and unequal_overlap, "generator must include both overlap branches"
    assert edge_count == 2, "only the intended adjacent pairs should remain"
    assert set(range(1, CELL_SIZE)).issubset(set(map(int, counts.reshape(-1)))), "need all counts 1-4"
    assert np.any(counts == 0), "need some empty cells"
    return counts, colors


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        try:
            counts, colors = _generate_scene()
            inp = _build_input(counts, colors)
            expected_board = _paint_crosses(counts, colors, BG_COLOR)
            expected = _expand_logical_board(expected_board, BG_COLOR, GREY_COLOR)
            if np.array_equal(main(inp), expected):
                return inp
        except Exception:
            continue
    raise RuntimeError("failed to generate a valid cross-growth puzzle input")
