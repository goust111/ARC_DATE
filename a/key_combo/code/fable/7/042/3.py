from __future__ import annotations

import random

import numpy as np


SPIRAL_ORDER: tuple[tuple[int, int], ...] = (
    (0, 0),
    (0, 1),
    (0, 2),
    (1, 2),
    (2, 2),
    (2, 1),
    (2, 0),
    (1, 0),
    (1, 1),
)

AMBIGUOUS_MISSING_INDICES = (0, 4, 8)
MAX_GENERATION_ATTEMPTS = 400


def _rotate_cw(tile: np.ndarray, quarter_turns: int) -> np.ndarray:
    return np.rot90(tile, -int(quarter_turns) % 4)


def _is_solid(tile: np.ndarray) -> bool:
    return int(np.unique(tile).size) == 1


def _cell_slice(index: int, cell_size: int) -> tuple[slice, slice]:
    br, bc = SPIRAL_ORDER[index]
    r0 = br * (cell_size + 1)
    c0 = bc * (cell_size + 1)
    return slice(r0, r0 + cell_size), slice(c0, c0 + cell_size)


def _extract_cells(grid: np.ndarray) -> tuple[list[np.ndarray], int]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    h, w = g.shape
    assert h == w, "board must be square so rotated tiles keep the same shape"
    assert (h - 2) % 3 == 0, "board height must match 3 square cells plus two separators"
    assert (w - 2) % 3 == 0, "board width must match 3 square cells plus two separators"

    cell_size = (h - 2) // 3
    assert cell_size >= 2, "tiles must be at least 2x2"

    sep_rows = (cell_size, 2 * cell_size + 1)
    sep_cols = (cell_size, 2 * cell_size + 1)
    for r in sep_rows:
        assert np.all(g[r, :] == g[r, 0]), "separator rows must be monochrome"
    for c in sep_cols:
        assert np.all(g[:, c] == g[0, c]), "separator cols must be monochrome"

    cells: list[np.ndarray] = []
    for index in range(9):
        rs, cs = _cell_slice(index, cell_size)
        cells.append(g[rs, cs].copy())
    return cells, cell_size


def _solve_missing_tile(cells: list[np.ndarray]) -> np.ndarray:
    solid_count = sum(_is_solid(cell) for cell in cells)
    assert solid_count == 1, "expected exactly one placeholder cell"

    solutions: list[tuple[int, np.ndarray]] = []
    for missing_index in range(9):
        visible_indices = [idx for idx in range(9) if idx != missing_index]
        for anchor_index in visible_indices:
            anchor = cells[anchor_index]
            if _is_solid(anchor):
                continue

            for idx in visible_indices:
                expected = _rotate_cw(anchor, idx - anchor_index)
                if not np.array_equal(cells[idx], expected):
                    break
            else:
                predicted = _rotate_cw(anchor, missing_index - anchor_index)
                if _is_solid(predicted):
                    continue
                solutions.append((missing_index, predicted))

    unique: list[tuple[int, np.ndarray]] = []
    for missing_index, tile in solutions:
        if any(
            old_missing == missing_index and np.array_equal(old_tile, tile)
            for old_missing, old_tile in unique
        ):
            continue
        unique.append((missing_index, tile))

    assert len(unique) == 1, "expected one unique missing-tile reconstruction"
    return unique[0][1]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recover the missing tile from a 3x3 spiral-ordered rotation progression.

    Input structure:
    - A square board is partitioned by one-pixel separator lines into a 3x3 set of equal square cells.
    - Eight cells contain rotated copies of the same non-4-fold-symmetric tile, and one cell is a solid placeholder color.
    - The intended reading order is the clockwise spiral: top-left, across the top row, around the outer ring, then the center.

    Rule:
    - Extract the nine square cells and interpret them in the fixed spiral order.
    - Consecutive spiral cells advance by one 90-degree clockwise rotation of the tile.
    - Find the unique placeholder position whose omission makes all visible cells consistent with that progression, then emit the missing rotated tile alone.

    Key insight:
    - The rotation step is attached to the hidden spiral order rather than to board coordinates, so the missing tile must be inferred from relative rotations along the spiral.
    """
    cells, _cell_size = _extract_cells(grid)
    return _solve_missing_tile(cells)


def _random_template(cell_size: int, colors: list[int]) -> np.ndarray:
    bg = colors[0]
    accents = colors[1:]
    coords = [(r, c) for r in range(cell_size) for c in range(cell_size)]

    for _ in range(300):
        tile = np.full((cell_size, cell_size), bg, dtype=int)

        for color in accents:
            block_h = random.randint(1, max(1, cell_size // 2))
            block_w = random.randint(1, max(1, cell_size // 2))
            r0 = random.randint(0, cell_size - block_h)
            c0 = random.randint(0, cell_size - block_w)
            tile[r0 : r0 + block_h, c0 : c0 + block_w] = color

        extra_points = random.sample(coords, random.randint(cell_size, cell_size * 2))
        for r, c in extra_points:
            tile[r, c] = random.choice(accents)

        if np.unique(tile).size < 3:
            continue
        if any(np.all(tile[r, :] == tile[r, 0]) for r in range(cell_size)):
            continue
        if any(np.all(tile[:, c] == tile[0, c]) for c in range(cell_size)):
            continue

        rotations = [_rotate_cw(tile, k) for k in range(4)]
        if len({rot.tobytes() for rot in rotations}) != 4:
            continue
        return tile

    raise RuntimeError("failed to generate a tile with four distinct rotations")


def _render_board(
    cell_size: int,
    separator_color: int,
    placeholder_color: int,
    base_tile: np.ndarray,
    start_rotation: int,
    missing_index: int,
) -> tuple[np.ndarray, np.ndarray]:
    board = np.full((3 * cell_size + 2, 3 * cell_size + 2), separator_color, dtype=int)
    expected = _rotate_cw(base_tile, start_rotation + missing_index)

    for index in range(9):
        rs, cs = _cell_slice(index, cell_size)
        if index == missing_index:
            board[rs, cs] = placeholder_color
        else:
            board[rs, cs] = _rotate_cw(base_tile, start_rotation + index)
    return board, expected


def _generate_attempt() -> np.ndarray | None:
    cell_size = random.randint(4, 8)
    palette = random.sample(range(10), 6)
    separator_color, placeholder_color = palette[:2]
    template_colors = palette[2:]

    base_tile = _random_template(cell_size, template_colors)
    start_rotation = random.randrange(4)
    missing_index = random.choice(AMBIGUOUS_MISSING_INDICES)
    board, expected = _render_board(
        cell_size=cell_size,
        separator_color=separator_color,
        placeholder_color=placeholder_color,
        base_tile=base_tile,
        start_rotation=start_rotation,
        missing_index=missing_index,
    )

    try:
        actual = main(board)
    except AssertionError:
        return None
    if not np.array_equal(actual, expected):
        return None
    return board


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        board = _generate_attempt()
        if board is not None:
            return board
    raise RuntimeError("failed to generate a valid spiral-rotation completion input")
