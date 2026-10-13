from __future__ import annotations

import random
from typing import Iterable

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GRAY = 5
LATIN_COLORS = (BLUE, RED, YELLOW, GRAY)
BASE_LATIN = np.array(
    [
        [0, 1, 2, 3],
        [1, 2, 3, 0],
        [2, 3, 0, 1],
        [3, 0, 1, 2],
    ],
    dtype=int,
)
MAX_ATTEMPTS = 400


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _flood_fill(grid: np.ndarray, r: int, c: int, color: int) -> None:
    h, w = grid.shape
    start_r = int(r)
    start_c = int(c)
    if not (0 <= start_r < h and 0 <= start_c < w):
        return
    old_color = int(grid[start_r, start_c])
    new_color = int(color)
    if old_color == new_color:
        return
    stack = [(start_r, start_c)]
    while stack:
        rr, cc = stack.pop()
        if not (0 <= rr < h and 0 <= cc < w):
            continue
        if int(grid[rr, cc]) != old_color:
            continue
        grid[rr, cc] = new_color
        stack.append((rr - 1, cc))
        stack.append((rr + 1, cc))
        stack.append((rr, cc - 1))
        stack.append((rr, cc + 1))


def _parse_partition(grid: np.ndarray) -> tuple[tuple[int, int, int, int], list[int], list[int]]:
    green_mask = grid == GREEN
    frame = _bbox(green_mask)
    assert frame is not None, "expected a green divider network"
    r0, c0, r1, c1 = frame
    sub = grid[r0 : r1 + 1, c0 : c1 + 1]
    row_lines = [r0 + rr for rr in range(sub.shape[0]) if np.all(sub[rr, :] == GREEN)]
    col_lines = [c0 + cc for cc in range(sub.shape[1]) if np.all(sub[:, cc] == GREEN)]
    assert len(row_lines) == 5 and len(col_lines) == 5, "expected a 4x4 divider lattice"
    expected = np.zeros_like(sub, dtype=bool)
    expected[[rr - r0 for rr in row_lines], :] = True
    expected[:, [cc - c0 for cc in col_lines]] = True
    assert np.array_equal(sub == GREEN, expected), "green pixels must be full divider lines"
    return frame, row_lines, col_lines


def _extract_board(grid: np.ndarray, row_lines: list[int], col_lines: list[int]) -> np.ndarray:
    board = np.zeros((4, 4), dtype=int)
    for rr in range(4):
        for cc in range(4):
            r_start = row_lines[rr] + 1
            r_stop = row_lines[rr + 1]
            c_start = col_lines[cc] + 1
            c_stop = col_lines[cc + 1]
            region = grid[r_start:r_stop, c_start:c_stop]
            assert region.size > 0, "region interiors must be non-empty"
            colors = set(map(int, np.unique(region)))
            assert colors <= {BLACK, *LATIN_COLORS}, "unexpected region color"
            non_black = [color for color in colors if color != BLACK]
            assert len(non_black) <= 1, "a region must be either blank or monochromatic"
            if non_black:
                fill_color = non_black[0]
                assert np.all(region == fill_color), "filled regions must be fully painted"
                board[rr, cc] = fill_color
            else:
                assert np.all(region == BLACK), "blank regions must be fully black"
    return board


def _candidate_colors(board: np.ndarray, r: int, c: int) -> list[int]:
    if int(board[r, c]) != BLACK:
        return []
    forbidden = set(int(v) for v in board[r, :] if int(v) != BLACK)
    forbidden |= set(int(board[rr, c]) for rr in range(4) if int(board[rr, c]) != BLACK)
    return [color for color in LATIN_COLORS if color not in forbidden]


def _propagate(board: np.ndarray) -> tuple[np.ndarray | None, list[list[tuple[int, int, int]]]]:
    work = np.array(board, dtype=int, copy=True)
    rounds: list[list[tuple[int, int, int]]] = []
    while True:
        placements: list[tuple[int, int, int]] = []
        for rr in range(4):
            for cc in range(4):
                if int(work[rr, cc]) != BLACK:
                    continue
                candidates = _candidate_colors(work, rr, cc)
                if not candidates:
                    return None, rounds
                if len(candidates) == 1:
                    placements.append((rr, cc, candidates[0]))
        if not placements:
            return work, rounds
        for rr, cc, color in placements:
            work[rr, cc] = color
        rounds.append(placements)


def _solve_unique(board: np.ndarray, limit: int = 2) -> tuple[int, np.ndarray | None]:
    work, _rounds = _propagate(board)
    if work is None:
        return 0, None
    if np.all(work != BLACK):
        return 1, work

    best_cell: tuple[int, int] | None = None
    best_candidates: list[int] | None = None
    for rr in range(4):
        for cc in range(4):
            if int(work[rr, cc]) != BLACK:
                continue
            candidates = _candidate_colors(work, rr, cc)
            if not candidates:
                return 0, None
            if best_candidates is None or len(candidates) < len(best_candidates):
                best_cell = (rr, cc)
                best_candidates = candidates

    assert best_cell is not None and best_candidates is not None
    total = 0
    first_solution = None
    rr, cc = best_cell
    for color in best_candidates:
        nxt = work.copy()
        nxt[rr, cc] = color
        count, solution = _solve_unique(nxt, limit=limit - total)
        if count:
            if first_solution is None:
                first_solution = solution
            total += count
            if total >= limit:
                return total, first_solution
    return total, first_solution


def _random_latin_board(rng: random.Random) -> np.ndarray:
    row_perm = list(range(4))
    col_perm = list(range(4))
    sym_perm = list(range(4))
    rng.shuffle(row_perm)
    rng.shuffle(col_perm)
    rng.shuffle(sym_perm)
    board = BASE_LATIN[row_perm][:, col_perm]
    if rng.random() < 0.5:
        board = board.T
    mapping = {idx: LATIN_COLORS[sym_perm[idx]] for idx in range(4)}
    return np.vectorize(mapping.get)(board).astype(int)


def _random_sizes(rng: random.Random) -> tuple[list[int], list[int]]:
    while True:
        heights = [rng.randint(2, 4) for _ in range(4)]
        widths = [rng.randint(2, 4) for _ in range(4)]
        if len(set(heights)) >= 2 and len(set(widths)) >= 2:
            return heights, widths


def _random_clues(solution: np.ndarray, rng: random.Random) -> np.ndarray | None:
    indices = list(range(16))
    rng.shuffle(indices)
    clue_count = rng.randint(8, 10)
    clue_mask = np.zeros((4, 4), dtype=bool)
    for idx in indices[:clue_count]:
        rr, cc = divmod(idx, 4)
        clue_mask[rr, cc] = True

    if set(int(v) for v in solution[clue_mask]) != set(LATIN_COLORS):
        return None

    partial = np.where(clue_mask, solution, BLACK)
    initial_ambiguous = any(
        len(_candidate_colors(partial, rr, cc)) > 1
        for rr in range(4)
        for cc in range(4)
        if int(partial[rr, cc]) == BLACK
    )
    if not initial_ambiguous:
        return None

    propagated, rounds = _propagate(partial)
    if propagated is None or not np.array_equal(propagated, solution):
        return None
    if len(rounds) < 2:
        return None

    missing_colors = set(int(v) for v in solution[~clue_mask])
    if len(missing_colors) < 3:
        return None
    if max(np.count_nonzero(~clue_mask, axis=0)) < 2:
        return None
    if max(np.count_nonzero(~clue_mask, axis=1)) < 2:
        return None

    count, solved = _solve_unique(partial)
    if count != 1 or solved is None or not np.array_equal(solved, solution):
        return None
    return clue_mask


def _render_board(
    solution: np.ndarray,
    clue_mask: np.ndarray,
    heights: list[int],
    widths: list[int],
    margins: tuple[int, int, int, int],
) -> np.ndarray:
    top, bottom, left, right = margins
    inner_h = sum(heights) + 5
    inner_w = sum(widths) + 5
    grid = np.full((top + inner_h + bottom, left + inner_w + right), BLACK, dtype=int)

    row_lines = [top]
    cursor = top
    for height in heights:
        cursor += height + 1
        row_lines.append(cursor)
    col_lines = [left]
    cursor = left
    for width in widths:
        cursor += width + 1
        col_lines.append(cursor)

    for rr in row_lines:
        grid[rr, left : left + inner_w] = GREEN
    for cc in col_lines:
        grid[top : top + inner_h, cc] = GREEN

    for rr in range(4):
        for cc in range(4):
            if not clue_mask[rr, cc]:
                continue
            r_start = row_lines[rr] + 1
            r_stop = row_lines[rr + 1]
            c_start = col_lines[cc] + 1
            c_stop = col_lines[cc + 1]
            grid[r_start:r_stop, c_start:c_stop] = int(solution[rr, cc])
    return grid


def _validate_generated_instance(inp: np.ndarray, solution: np.ndarray) -> None:
    assert inp.ndim == 2 and max(inp.shape) <= 30
    _frame, row_lines, col_lines = _parse_partition(inp)
    board = _extract_board(inp, row_lines, col_lines)
    count, solved = _solve_unique(board)
    assert count == 1 and solved is not None
    assert np.array_equal(solved, solution)
    out = main(inp)
    expected = _render_board(solution, np.ones((4, 4), dtype=bool), [row_lines[i + 1] - row_lines[i] - 1 for i in range(4)], [col_lines[i + 1] - col_lines[i] - 1 for i in range(4)], (row_lines[0], inp.shape[0] - row_lines[-1] - 1, col_lines[0], inp.shape[1] - col_lines[-1] - 1))
    assert np.array_equal(out, expected)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Resolve a 4x4 irregular region Latin square and flood-fill the blank regions with their forced colors.

    Input structure:
    - A black background surrounds one green rectangular divider network.
    - Inside that network, five horizontal and five vertical green lines partition the interior into a 4x4 grid of rectangular regions with uneven widths and heights.
    - Some regions are already fully filled with one of blue, red, yellow, or gray, while the remaining enclosed regions are entirely black.

    Rule:
    - Parse the green divider network to recover the 4x4 region topology and each region's row and column.
    - Read the pre-filled region colors as clues for a 4-symbol Latin square: each region row and each region column must contain every clue color exactly once.
    - Repeatedly eliminate colors from each blank region using its region row and region column until the unique completion is determined.
    - Flood-fill every blank enclosed region with its deduced color, preserving the green dividers, background, and any already filled regions.

    Key insight:
    - The colors do not live on individual pixels but on irregular enclosed regions, so the divider topology must be decoded first; only then can row-column uniqueness determine which color each black compartment must receive.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"
    _frame, row_lines, col_lines = _parse_partition(g)
    board = _extract_board(g, row_lines, col_lines)
    count, solved = _solve_unique(board)
    assert count == 1 and solved is not None, "puzzle must have a unique completion"

    out = g.copy()
    for rr in range(4):
        for cc in range(4):
            if int(board[rr, cc]) != BLACK:
                continue
            seed_r = row_lines[rr] + 1
            seed_c = col_lines[cc] + 1
            _flood_fill(out, seed_r, seed_c, int(solved[rr, cc]))
    return out


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(MAX_ATTEMPTS):
        solution = _random_latin_board(rng)
        clue_mask = _random_clues(solution, rng)
        if clue_mask is None:
            continue
        heights, widths = _random_sizes(rng)
        margins = (
            rng.randint(1, 3),
            rng.randint(1, 3),
            rng.randint(1, 3),
            rng.randint(1, 3),
        )
        inp = _render_board(solution, clue_mask, heights, widths, margins)
        _validate_generated_instance(inp, solution)
        return inp
    raise RuntimeError("failed to generate a valid irregular Latin flood-fill puzzle")
