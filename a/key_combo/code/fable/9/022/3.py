from __future__ import annotations

import random
from typing import Iterable

import numpy as np


BLACK = 0
UNKNOWN = -1
GEN_ATTEMPTS = 400


def _open_runs(size: int, separator_lines: Iterable[int]) -> list[tuple[int, int]]:
    separator_set = set(map(int, separator_lines))
    runs: list[tuple[int, int]] = []
    start: int | None = None
    for idx in range(size):
        if idx in separator_set:
            if start is not None:
                runs.append((start, idx - 1))
                start = None
        elif start is None:
            start = idx
    if start is not None:
        runs.append((start, size - 1))
    return runs


def _separator_candidates(grid: np.ndarray) -> list[tuple[int, np.ndarray, np.ndarray]]:
    height, width = map(int, grid.shape)
    candidates: list[tuple[int, np.ndarray, np.ndarray]] = []
    for color in np.unique(grid):
        color = int(color)
        if color == BLACK:
            continue
        row_lines = np.flatnonzero(np.all(grid == color, axis=1))
        col_lines = np.flatnonzero(np.all(grid == color, axis=0))
        if len(row_lines) == 0 or len(col_lines) == 0:
            continue

        row_runs = _open_runs(height, row_lines)
        col_runs = _open_runs(width, col_lines)
        if not row_runs or not col_runs:
            continue
        if len(row_runs) * len(col_runs) < 2:
            continue
        if len({r1 - r0 + 1 for r0, r1 in row_runs}) != 1:
            continue
        if len({c1 - c0 + 1 for c0, c1 in col_runs}) != 1:
            continue

        expected = np.zeros_like(grid, dtype=bool)
        expected[row_lines, :] = True
        expected[:, col_lines] = True
        if not np.array_equal(grid == color, expected):
            continue
        candidates.append((color, row_lines, col_lines))
    return candidates


def _infer_template(blocks: list[np.ndarray]) -> np.ndarray:
    cell_h, cell_w = map(int, blocks[0].shape)
    template = np.full((cell_h, cell_w), UNKNOWN, dtype=int)
    for block in blocks:
        mask = block != BLACK
        known = template[mask]
        values = block[mask]
        if np.any((known != UNKNOWN) & (known != values)):
            raise AssertionError("same-parity cells disagree on a visible pixel")
        template[mask] = values
    return template


def main(grid: np.ndarray) -> np.ndarray:
    """
    Restore a checkerboard dual-wallpaper with blacked-out cells by parity and mirror.

    Input structure:
    - One separator color forms complete horizontal and vertical lines that partition the canvas into equal rectangular cells.
    - Cells split into two classes by `(cell_row + cell_col) % 2`; within each class, all non-black pixels match at the same offsets.
    - The two class patterns are horizontal mirrors of each other.
    - Black marks missing data inside some cells, and exactly one whole cell is fully black.

    Rule:
    - Keep the separator lines unchanged and treat black only as missing pixels.
    - Infer the latent template of each parity class from the visible same-parity cells.
    - Use the left-right mirror relation between the two classes to fill any template positions still hidden, then write the completed template back into every damaged cell.

    Key insight:
    - The separator lattice gives cell coordinates, so restoration must use cell parity rather than nearest neighbors; the all-black cell is recoverable only through the hidden mirror link between opposite parities.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"

    separator_hits = _separator_candidates(g)
    assert len(separator_hits) == 1, "expected a unique separator color"
    separator_color, row_lines, col_lines = separator_hits[0]
    row_cells = _open_runs(g.shape[0], row_lines)
    col_cells = _open_runs(g.shape[1], col_lines)
    assert row_cells and col_cells, "separator grid must leave open cells"

    cell_h = row_cells[0][1] - row_cells[0][0] + 1
    cell_w = col_cells[0][1] - col_cells[0][0] + 1
    cells: dict[tuple[int, int], np.ndarray] = {}
    full_black = []
    for rr, (r0, r1) in enumerate(row_cells):
        for cc, (c0, c1) in enumerate(col_cells):
            block = g[r0 : r1 + 1, c0 : c1 + 1].copy()
            assert block.shape == (cell_h, cell_w)
            assert not np.any(block == separator_color), "separator color must stay on separator lines"
            cells[(rr, cc)] = block
            if np.all(block == BLACK):
                full_black.append((rr, cc))

    assert len(full_black) == 1, "expected exactly one fully black cell"

    parity_blocks = {0: [], 1: []}
    for (rr, cc), block in cells.items():
        parity_blocks[(rr + cc) % 2].append(block)

    templates = {parity: _infer_template(blocks) for parity, blocks in parity_blocks.items()}

    changed = True
    while changed:
        changed = False
        for parity in (0, 1):
            other = 1 - parity
            mirrored = np.fliplr(templates[other])
            mask = (templates[parity] == UNKNOWN) & (mirrored != UNKNOWN)
            if np.any(mask):
                templates[parity][mask] = mirrored[mask]
                changed = True

    for parity in (0, 1):
        assert not np.any(templates[parity] == UNKNOWN), "mirror relation must fully determine both templates"
    assert np.array_equal(templates[0], np.fliplr(templates[1])), "parity templates must be mirrors"

    out = g.copy()
    for rr, (r0, r1) in enumerate(row_cells):
        for cc, (c0, c1) in enumerate(col_cells):
            parity = (rr + cc) % 2
            block = cells[(rr, cc)]
            template = templates[parity]
            mask = block == BLACK
            if np.any(mask):
                restored = block.copy()
                restored[mask] = template[mask]
                out[r0 : r1 + 1, c0 : c1 + 1] = restored

    return out


def _pattern_candidates(pattern: np.ndarray) -> list[tuple[int, int]]:
    cell_h, cell_w = map(int, pattern.shape)
    coords = []
    for r in range(cell_h):
        for c in range(cell_w):
            if pattern[r, c] != pattern[r, cell_w - 1 - c]:
                coords.append((r, c))
    return coords


def _build_pattern(cell_h: int, cell_w: int, palette: list[int]) -> np.ndarray | None:
    base = palette[0]
    accents = palette[1:]
    pattern = np.full((cell_h, cell_w), base, dtype=int)

    target = random.randint(4, min(cell_h * cell_w - 1, 9))
    coords = set()
    left_bias = max(1, cell_w // 2)
    while len(coords) < target:
        r = random.randint(0, cell_h - 1)
        if random.random() < 0.7:
            c = random.randint(0, left_bias - 1)
        else:
            c = random.randint(0, cell_w - 1)
        coords.add((r, c))

    if len({r for r, _c in coords}) < 2 or len({c for _r, c in coords}) < 2:
        return None

    accent_list = list(accents)
    while len(accent_list) < len(coords):
        accent_list.append(random.choice(accents))
    random.shuffle(accent_list)
    for (r, c), color in zip(coords, accent_list):
        pattern[r, c] = color

    if np.array_equal(pattern, np.fliplr(pattern)):
        return None
    if len(np.unique(pattern)) < 3:
        return None
    if not _pattern_candidates(pattern):
        return None
    return pattern


def _scattered_holes(pattern: np.ndarray) -> list[tuple[int, int]] | None:
    candidates = _pattern_candidates(pattern)
    if len(candidates) < 2:
        return None
    hole_count = random.randint(2, min(6, len(candidates)))
    holes = random.sample(candidates, hole_count)
    if len({r for r, _c in holes}) < 2 and pattern.shape[0] > 2:
        return None
    return holes


def _render_board(
    pattern_a: np.ndarray,
    pattern_b: np.ndarray,
    separator_color: int,
    orientation: str,
) -> tuple[np.ndarray, dict[tuple[int, int], tuple[int, int]]]:
    cell_h, cell_w = map(int, pattern_a.shape)
    if orientation == "horizontal":
        board = np.full((cell_h + 2, cell_w * 3 + 4), separator_color, dtype=int)
        starts = {(0, 0): (1, 1), (0, 1): (1, cell_w + 2), (0, 2): (1, 2 * cell_w + 3)}
        board[1 : 1 + cell_h, 1 : 1 + cell_w] = pattern_a
        board[1 : 1 + cell_h, cell_w + 2 : 2 * cell_w + 2] = pattern_b
        board[1 : 1 + cell_h, 2 * cell_w + 3 : 3 * cell_w + 3] = pattern_a
    else:
        board = np.full((cell_h * 3 + 4, cell_w + 2), separator_color, dtype=int)
        starts = {(0, 0): (1, 1), (1, 0): (cell_h + 2, 1), (2, 0): (2 * cell_h + 3, 1)}
        board[1 : 1 + cell_h, 1 : 1 + cell_w] = pattern_a
        board[cell_h + 2 : 2 * cell_h + 2, 1 : 1 + cell_w] = pattern_b
        board[2 * cell_h + 3 : 3 * cell_h + 3, 1 : 1 + cell_w] = pattern_a
    return board, starts


def _fill_cell(board: np.ndarray, start: tuple[int, int], block: np.ndarray) -> None:
    r0, c0 = start
    h, w = block.shape
    board[r0 : r0 + h, c0 : c0 + w] = block


def _wrong_center_copy(out: np.ndarray, starts: dict[tuple[int, int], tuple[int, int]], pattern_a: np.ndarray) -> np.ndarray:
    candidate = out.copy()
    center_idx = (0, 1) if (0, 1) in starts else (1, 0)
    _fill_cell(candidate, starts[center_idx], pattern_a)
    return candidate


def _make_example() -> tuple[np.ndarray, np.ndarray]:
    orientation = random.choice(["horizontal", "vertical"])
    cell_h = random.randint(4, 8)
    cell_w = random.randint(4, 8)

    colors = random.sample(range(1, 10), random.randint(4, 5))
    separator_color = colors[0]
    palette = colors[1:]

    pattern_a = _build_pattern(cell_h, cell_w, palette)
    if pattern_a is None:
        raise ValueError("pattern construction failed")
    pattern_b = np.fliplr(pattern_a)

    board, starts = _render_board(pattern_a, pattern_b, separator_color, orientation)
    out = board.copy()
    inp = board.copy()

    center_idx = (0, 1) if orientation == "horizontal" else (1, 0)
    edge_choices = [(0, 0), (0, 2)] if orientation == "horizontal" else [(0, 0), (2, 0)]
    partial_idx = random.choice(edge_choices)
    intact_idx = edge_choices[1] if partial_idx == edge_choices[0] else edge_choices[0]

    holes = _scattered_holes(pattern_a)
    if holes is None:
        raise ValueError("hole sampling failed")

    partial_block = pattern_a.copy()
    for r, c in holes:
        partial_block[r, c] = BLACK
    _fill_cell(inp, starts[partial_idx], partial_block)
    _fill_cell(inp, starts[intact_idx], pattern_a)
    _fill_cell(inp, starts[center_idx], np.zeros_like(pattern_b))

    if np.array_equal(partial_block, pattern_a):
        raise ValueError("partial cell must contain holes")
    if np.array_equal(_wrong_center_copy(out, starts, pattern_a), out):
        raise ValueError("center mirror step became degenerate")
    return inp, out


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        try:
            inp, out = _make_example()
        except ValueError:
            continue

        try:
            solved = main(inp)
        except Exception:
            continue

        center_idx = (0, 1) if inp.shape[0] < inp.shape[1] else (1, 0)
        if center_idx == (0, 1):
            row_cells, col_cells = 1, 3
        else:
            row_cells, col_cells = 3, 1
        if row_cells * col_cells != 3:
            continue
        if not np.array_equal(solved, out):
            continue
        if max(inp.shape) > 30 or max(out.shape) > 30:
            continue
        if np.array_equal(np.unique(inp), np.array([BLACK])):
            continue
        return inp
    raise RuntimeError("failed to generate a valid odd-even dual-wallpaper input")
