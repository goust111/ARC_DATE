from __future__ import annotations

from collections import Counter
from itertools import combinations
import random

import numpy as np


BLACK = 0
BLUE = 1
GREY = 5
ORANGE = 7
PALETTE_COLORS = (2, 3, 4, 6, 8, 9)
MAX_ATTEMPTS = 400


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _cell_separator_indices(cell_size: int, n: int) -> list[int]:
    return [k * (cell_size + 1) + cell_size for k in range(n - 1)]


def _infer_layout(grid: np.ndarray) -> tuple[int, int, int]:
    grey_mask = grid == GREY
    row_counts = np.count_nonzero(grey_mask, axis=1)
    col_counts = np.count_nonzero(grey_mask, axis=0)

    row_peak = int(row_counts.max())
    col_peak = int(col_counts.max())
    assert row_peak > 0 and col_peak > 0

    sep_rows = np.flatnonzero(row_counts == row_peak)
    sep_cols = np.flatnonzero(col_counts == col_peak)
    assert len(sep_rows) >= 2 and len(sep_cols) >= 2

    n = len(sep_rows) + 1
    assert len(sep_cols) + 1 == n

    row_steps = np.diff(sep_rows)
    col_steps = np.diff(sep_cols)
    assert np.all(row_steps == row_steps[0])
    assert np.all(col_steps == col_steps[0])

    cell_size = int(sep_rows[0])
    assert cell_size == int(sep_cols[0])
    assert int(row_steps[0]) == cell_size + 1
    assert int(col_steps[0]) == cell_size + 1

    board_size = n * cell_size + (n - 1)
    assert board_size == row_peak == col_peak
    assert grid.shape[0] >= board_size and grid.shape[1] > board_size
    return int(n), int(cell_size), int(board_size)


def _extract_patterns(board: np.ndarray, n: int, cell_size: int) -> tuple[np.ndarray, np.ndarray]:
    step = cell_size + 1
    patterns: dict[tuple[int, ...], np.ndarray] = {}
    cell_sigs: list[tuple[int, ...]] = []

    for r in range(n):
        for c in range(n):
            r0 = r * step
            c0 = c * step
            patch = board[r0 : r0 + cell_size, c0 : c0 + cell_size].copy()
            sig = tuple(map(int, patch.ravel()))
            patterns.setdefault(sig, patch)
            cell_sigs.append(sig)

    counts = Counter(cell_sigs)
    assert len(counts) == 2
    minority_sig, majority_sig = sorted(counts.items(), key=lambda item: item[1])
    return patterns[minority_sig[0]], patterns[majority_sig[0]]


def _extract_palette(grid: np.ndarray, board_size: int, n: int) -> np.ndarray:
    side = grid[:, board_size:]
    palette_mask = (~np.isin(side, [BLACK, GREY]))
    b = _bbox(palette_mask)
    assert b is not None
    r0, c0, r1, c1 = b
    assert c0 == c1
    assert r1 - r0 + 1 == n
    palette = side[r0 : r1 + 1, c0].copy()
    assert len(set(map(int, palette))) == n
    return palette


def _render_board(
    majority: np.ndarray,
    minority_mask: np.ndarray,
    active_mask: np.ndarray,
    palette: np.ndarray,
) -> np.ndarray:
    n = int(active_mask.shape[0])
    cell_size = int(majority.shape[0])
    step = cell_size + 1
    board_size = n * cell_size + (n - 1)
    board = np.zeros((board_size, board_size), dtype=int)

    for idx in _cell_separator_indices(cell_size, n):
        board[idx, :] = GREY
        board[:, idx] = GREY

    recolored = [np.where(minority_mask, int(color), BLACK).astype(int) for color in palette]
    for r in range(n):
        for c in range(n):
            r0 = r * step
            c0 = c * step
            patch = recolored[r] if active_mask[r, c] else majority
            board[r0 : r0 + cell_size, c0 : c0 + cell_size] = patch
    return board


def main(grid: np.ndarray) -> np.ndarray:
    """
    Complete the fractal cell grid by expanding the minority blueprint and recoloring it row-by-row from the palette strip.

    Input structure:
    - The left side is an `N x N` grid of equal square cells separated by grey full-row and full-column divider lines.
    - Most cells share one black/orange interior pattern, while 2 to 3 existing minority cells share a different black/blue pattern.
    - To the right of the cell grid, a separate vertical `N x 1` palette strip contains one distinct color per palette row.

    Rule:
    - Infer the cell size and divider layout from the grey separator lines.
    - Count exact cell-interior patterns to identify the majority pattern and the minority pattern.
    - Treat the minority pattern's non-black pixels as an `N x N` binary blueprint of which cell positions should be minority cells.
    - For every blueprint-active position in row `r`, place the minority pattern there and recolor all of its non-black pixels with the `r`-th palette color.
    - Keep blueprint-inactive positions as the majority pattern, preserve the grey separators, and drop the palette strip from the output.

    Key insight:
    - The minority cell is both a miniature placement map and a reusable stamp: its shape chooses the target cell positions, while the separate palette provides the final color independently by cell-row.
    """
    g = np.asarray(grid, dtype=int)
    n, cell_size, board_size = _infer_layout(g)
    board = g[:board_size, :board_size]

    minority, majority = _extract_patterns(board, n, cell_size)
    assert minority.shape == (n, n)
    minority_mask = minority != BLACK
    assert int(minority_mask.sum()) >= 2

    palette = _extract_palette(g, board_size, n)
    return _render_board(majority, minority_mask, minority_mask, palette)


def _build_blueprint(n: int) -> np.ndarray | None:
    target_count = random.randint(n + 1, min(n * n - 1, 2 * n))
    cols = random.sample(range(n), n)
    active = {(r, cols[r]) for r in range(n)}

    remaining = [(r, c) for r in range(n) for c in range(n) if (r, c) not in active]
    extra = target_count - n
    active.update(random.sample(remaining, extra))

    row_counts = Counter(r for r, _ in active)
    col_counts = Counter(c for _, c in active)
    if max(row_counts.values()) < 2 or max(col_counts.values()) < 2:
        return None

    mask = np.zeros((n, n), dtype=bool)
    for r, c in active:
        mask[r, c] = True
    return mask


def _majority_templates(n: int) -> list[np.ndarray]:
    rr, cc = np.indices((n, n))
    center = n // 2
    return [
        rr == cc,
        rr + cc == n - 1,
        (rr == center) | (cc == center),
        (rr == 0) | (rr == n - 1) | (cc == 0) | (cc == n - 1),
        (rr + cc) % 2 == 0,
        (cc % 2) == 0,
        (rr % 2) == 0,
        ((rr == 0) | (rr == n - 1)) & ((cc == 0) | (cc == n - 1)),
    ]


def _choose_majority_mask(n: int, minority_mask: np.ndarray) -> np.ndarray | None:
    candidates: list[np.ndarray] = []
    for base in _majority_templates(n):
        for k in range(4):
            mask = np.rot90(base, k)
            candidates.append(mask)
            candidates.append(np.fliplr(mask))

    unique: list[np.ndarray] = []
    seen: set[tuple[int, ...]] = set()
    for mask in candidates:
        sig = tuple(map(int, mask.ravel()))
        if sig in seen:
            continue
        seen.add(sig)
        if np.array_equal(mask, minority_mask):
            continue
        orange_count = int(mask.sum())
        if 2 <= orange_count <= n * n - 2:
            unique.append(mask.astype(bool))
    if not unique:
        return None
    return random.choice(unique)


def _choose_given_positions(active_mask: np.ndarray) -> list[tuple[int, int]] | None:
    active = [tuple(map(int, rc)) for rc in np.argwhere(active_mask)]
    random.shuffle(active)
    for size in random.sample([2, 3], 2):
        if len(active) < size:
            continue
        valid = [
            subset
            for subset in combinations(active, size)
            if len({r for r, _ in subset}) >= 2 and len({c for _, c in subset}) >= 2
        ]
        if valid:
            return list(random.choice(valid))
    return None


def _render_input(
    majority: np.ndarray,
    minority_mask: np.ndarray,
    given_positions: list[tuple[int, int]],
    palette: np.ndarray,
    palette_top: int,
    gap: int,
) -> np.ndarray:
    n = int(majority.shape[0])
    board = _render_board(majority, minority_mask, np.zeros((n, n), dtype=bool), palette)
    step = n + 1

    minority = np.where(minority_mask, BLUE, BLACK).astype(int)
    for r, c in given_positions:
        r0 = r * step
        c0 = c * step
        board[r0 : r0 + n, c0 : c0 + n] = minority

    grid = np.zeros((board.shape[0], board.shape[1] + gap + 1), dtype=int)
    grid[:, : board.shape[1]] = board
    grid[palette_top : palette_top + n, -1] = palette
    return grid


def _gen_attempt() -> np.ndarray | None:
    n = random.choice([3, 4])
    blueprint = _build_blueprint(n)
    if blueprint is None:
        return None

    majority_mask = _choose_majority_mask(n, blueprint)
    if majority_mask is None:
        return None

    given_positions = _choose_given_positions(blueprint)
    if given_positions is None:
        return None

    palette = np.array(random.sample(PALETTE_COLORS, n), dtype=int)
    palette_top = random.randint(0, n * n - 1)
    palette_top = min(palette_top, (n * n + n - 1) - n)
    gap = random.randint(1, 2)

    majority = np.where(majority_mask, ORANGE, BLACK).astype(int)
    inp = _render_input(majority, blueprint, given_positions, palette, palette_top, gap)
    expected = _render_board(majority, blueprint, blueprint, palette)

    if inp.ndim != 2 or expected.ndim != 2:
        return None
    if max(inp.shape) > 30 or max(expected.shape) > 30:
        return None

    try:
        actual = main(inp)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 3")
