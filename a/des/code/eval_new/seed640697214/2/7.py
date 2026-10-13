from __future__ import annotations

from collections import Counter

import numpy as np


BG_COLOR = 0
SEPARATOR_COLOR = 4
JUDGE_COLOR = 9
VOTE_PALETTE = (1, 2, 3, 5, 6, 7, 8)
MAX_GENERATION_ATTEMPTS = 300


def _regular_spans(lines: np.ndarray, size: int) -> list[slice] | None:
    bounds = sorted({int(x) for x in np.asarray(lines, dtype=int).tolist()})
    if not bounds:
        return None

    if bounds[0] != 0:
        bounds = [-1] + bounds
    if bounds[-1] != size - 1:
        bounds = bounds + [size]
    else:
        bounds = bounds + [size]

    spans: list[slice] = []
    gap_sizes: list[int] = []
    for left, right in zip(bounds, bounds[1:]):
        gap = int(right - left - 1)
        if gap <= 0:
            continue
        spans.append(slice(int(left + 1), int(right)))
        gap_sizes.append(gap)

    if len(spans) < 2 or len(set(gap_sizes)) != 1:
        return None
    return spans


def _candidate_lattices(board: np.ndarray) -> list[tuple[int, list[slice], list[slice], int, int]]:
    candidates: list[tuple[int, list[slice], list[slice], int, int]] = []
    for color in np.unique(board):
        row_lines = np.flatnonzero(np.all(board == int(color), axis=1))
        col_lines = np.flatnonzero(np.all(board == int(color), axis=0))
        row_spans = _regular_spans(row_lines, board.shape[0])
        col_spans = _regular_spans(col_lines, board.shape[1])
        if row_spans is None or col_spans is None:
            continue
        candidates.append((int(color), row_spans, col_spans, len(row_lines), len(col_lines)))
    return candidates


def _outside_values(grid: np.ndarray, board_h: int, board_w: int) -> np.ndarray:
    if board_h < grid.shape[0]:
        return np.asarray(grid[board_h:, :], dtype=int).reshape(-1)
    if board_w < grid.shape[1]:
        return np.asarray(grid[:, board_w:], dtype=int).reshape(-1)
    return np.empty(0, dtype=int)


def _infer_bg_and_judge(
    grid: np.ndarray,
    board_h: int,
    board_w: int,
    separator: int,
    row_spans: list[slice],
    col_spans: list[slice],
) -> tuple[int, int]:
    outside = _outside_values(grid, board_h, board_w)
    if outside.size:
        outside_counts = Counter(map(int, outside.tolist()))
        bg = int(outside_counts.most_common(1)[0][0])
        judge_candidates = [
            int(color)
            for color, count in outside_counts.items()
            if int(color) != int(bg) and int(color) != int(separator) and int(count) == 1
        ]
        if judge_candidates:
            return bg, judge_candidates[0]

    board = np.asarray(grid[:board_h, :board_w], dtype=int)
    interior = np.concatenate(
        [board[row_span, col_span].reshape(-1) for row_span in row_spans for col_span in col_spans]
    )
    interior_counts = Counter(map(int, interior.tolist()))
    interior_counts.pop(int(separator), None)
    assert interior_counts, "expected non-separator cells inside lattice blocks"
    bg = int(interior_counts.most_common(1)[0][0])

    global_counts = Counter(map(int, grid.reshape(-1).tolist()))
    judge_candidates = [
        int(color)
        for color, count in global_counts.items()
        if int(color) != int(bg) and int(color) != int(separator) and int(count) == 1
    ]
    assert judge_candidates, "failed to identify referee color"
    return bg, judge_candidates[0]


def _find_layout(grid: np.ndarray) -> tuple[int, int, int, list[slice], list[slice], int, int]:
    h, w = map(int, grid.shape)
    options = [(h, w)]
    if h >= 3:
        options.append((h - 1, w))
    if w >= 3:
        options.append((h, w - 1))

    best: tuple[int, int, int, int, list[slice], list[slice]] | None = None
    for board_h, board_w in options:
        board = np.asarray(grid[:board_h, :board_w], dtype=int)
        outside = _outside_values(grid, board_h, board_w)
        outside_counts = Counter(map(int, outside.tolist()))
        outside_bg = int(outside_counts.most_common(1)[0][0]) if outside_counts else None
        judge_count = sum(
            1
            for color, count in outside_counts.items()
            if int(color) != int(outside_bg) and int(count) == 1
        )

        for separator, row_spans, col_spans, n_rows, n_cols in _candidate_lattices(board):
            score = 100 * len(row_spans) * len(col_spans) + 10 * (n_rows + n_cols)
            if outside.size:
                score += 50
            if outside_bg is not None and int(outside_bg) != int(separator):
                score += 20
            score += 20 * judge_count

            if best is None or score > best[0]:
                best = (score, int(board_h), int(board_w), int(separator), row_spans, col_spans)

    assert best is not None, "failed to detect rectangular lattice"
    _, board_h, board_w, separator, row_spans, col_spans = best
    bg, judge = _infer_bg_and_judge(grid, board_h, board_w, separator, row_spans, col_spans)
    return board_h, board_w, separator, row_spans, col_spans, bg, judge


def main(grid: np.ndarray) -> np.ndarray:
    """
    Resolve each lattice block by majority vote, then mark the losing vote positions with the referee color.

    Input structure:
    - A one-pixel separator color forms evenly spaced horizontal and vertical lattice lines,
      creating equal-sized rectangular blocks.
    - Each block interior contains background cells plus vote pixels of two foreground colors.
    - A single referee-color pixel sits in the last row or last column outside the lattice.

    Rule:
    - Detect the lattice and the outside strip that reveals the referee color.
    - For each block, count the non-background vote colors found in that block.
    - If one color has a unique highest count, fill the whole block with that color and repaint
      every original non-winning vote position to the referee color.
    - If the highest vote count is tied, fill the entire block with the referee color.
    - Leave the separator lines and the outside strip unchanged.

    Key insight:
    - The visible majority decides the new block fill, but the original locations of the losing
      votes still matter because they must be remembered and specially marked after the repaint.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    board_h, board_w, separator, row_spans, col_spans, bg, judge = _find_layout(g)
    out = g.copy()

    for row_span in row_spans:
        for col_span in col_spans:
            block = np.asarray(g[row_span, col_span], dtype=int)
            counts = Counter(map(int, block.reshape(-1).tolist()))
            counts.pop(int(bg), None)
            counts.pop(int(separator), None)
            counts.pop(int(judge), None)

            if not counts:
                continue

            max_count = max(counts.values())
            winners = [int(color) for color, count in counts.items() if int(count) == int(max_count)]

            patch = out[row_span, col_span]
            if len(winners) != 1:
                patch[:] = int(judge)
                continue

            winner = winners[0]
            patch[:] = int(winner)
            for color in counts:
                if int(color) == int(winner):
                    continue
                patch[block == int(color)] = int(judge)

    return out


def _block_slice(index: int, block_size: int) -> slice:
    start = 1 + index * (block_size + 1)
    return slice(int(start), int(start + block_size))


def _fill_block(block: np.ndarray, tie_case: bool) -> None:
    area = int(block.size)
    colors = [int(c) for c in np.random.choice(VOTE_PALETTE, size=2, replace=False)]
    positions = np.arange(area)
    np.random.shuffle(positions)

    if tie_case:
        max_each = max(1, (area - 1) // 2)
        each = int(np.random.randint(1, max_each + 1))
        count_a = each
        count_b = each
    else:
        max_minor = max(1, (area - 2) // 3)
        count_b = int(np.random.randint(1, max_minor + 1))
        max_major = area - count_b - 1
        min_major = count_b + 1
        count_a = int(np.random.randint(min_major, max_major + 1))

    block[:] = BG_COLOR
    rows, cols = np.unravel_index(positions, block.shape)
    for idx in range(count_a):
        block[int(rows[idx]), int(cols[idx])] = int(colors[0])
    for idx in range(count_a, count_a + count_b):
        block[int(rows[idx]), int(cols[idx])] = int(colors[1])


def _gen_attempt() -> np.ndarray | None:
    n_block_rows = int(np.random.randint(2, 5))
    n_block_cols = int(np.random.randint(2, 5))
    block_h = int(np.random.randint(3, 7))
    block_w = int(np.random.randint(3, 7))

    board_h = n_block_rows * block_h + (n_block_rows + 1)
    board_w = n_block_cols * block_w + (n_block_cols + 1)
    if max(board_h, board_w) > 29:
        return None

    board = np.full((board_h, board_w), SEPARATOR_COLOR, dtype=int)
    block_total = n_block_rows * n_block_cols
    tie_count = block_total // 2 if np.random.rand() < 0.5 else (block_total + 1) // 2
    tie_flags = np.array([True] * tie_count + [False] * (block_total - tie_count), dtype=bool)
    np.random.shuffle(tie_flags)

    flag_index = 0
    for br in range(n_block_rows):
        row_span = _block_slice(br, block_h)
        for bc in range(n_block_cols):
            col_span = _block_slice(bc, block_w)
            _fill_block(board[row_span, col_span], bool(tie_flags[flag_index]))
            flag_index += 1

    if np.random.rand() < 0.5:
        grid = np.full((board_h + 1, board_w), BG_COLOR, dtype=int)
        grid[:board_h, :] = board
        judge_col = int(np.random.randint(0, board_w))
        grid[board_h, judge_col] = JUDGE_COLOR
    else:
        grid = np.full((board_h, board_w + 1), BG_COLOR, dtype=int)
        grid[:, :board_w] = board
        judge_row = int(np.random.randint(0, board_h))
        grid[judge_row, board_w] = JUDGE_COLOR

    try:
        out = main(grid)
    except AssertionError:
        return None

    if out.shape != grid.shape:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid lattice-vote puzzle")
