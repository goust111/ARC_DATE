from __future__ import annotations

import random

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
GREY = 5

TILE = 5
STEP = TILE + 1
BOARD_H = TILE * 2 + 1
BOARD_W = TILE * 3 + 2
SEPARATOR_ROW = TILE
SEPARATOR_COLS = (TILE, TILE + STEP)

GEN_RETRIES = 400
FRAGMENT_RETRIES = 200
PAIR_RETRIES = 200
PADDING_RANGE = (0, 2)


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _neighbors4(r: int, c: int) -> list[tuple[int, int]]:
    return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    seen = np.zeros(mask.shape, dtype=bool)
    parts: list[list[tuple[int, int]]] = []
    h, w = mask.shape
    for r, c in np.argwhere(mask):
        r = int(r)
        c = int(c)
        if seen[r, c]:
            continue
        stack = [(r, c)]
        seen[r, c] = True
        comp: list[tuple[int, int]] = []
        while stack:
            cr, cc = stack.pop()
            comp.append((cr, cc))
            for nr, nc in _neighbors4(cr, cc):
                if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                    seen[nr, nc] = True
                    stack.append((nr, nc))
        parts.append(comp)
    return parts


def _is_connected(mask: np.ndarray) -> bool:
    return mask.any() and len(_components(mask)) == 1


def _rot180(mask: np.ndarray) -> np.ndarray:
    return np.rot90(mask, 2)


def _complete_mask(mask: np.ndarray) -> np.ndarray:
    return mask | _rot180(mask)


def _tile_origin(tile_r: int, tile_c: int) -> tuple[int, int]:
    return tile_r * STEP, tile_c * STEP


def _extract_board(grid: np.ndarray) -> np.ndarray:
    grey = np.argwhere(grid == GREY)
    assert grey.size > 0, "expected grey separator lines"
    r0, c0 = grey.min(axis=0)
    r1, c1 = grey.max(axis=0)
    board = grid[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1].copy()
    assert board.shape == (BOARD_H, BOARD_W), f"unexpected board shape {board.shape}"

    expected_sep = np.zeros(board.shape, dtype=bool)
    expected_sep[SEPARATOR_ROW, :] = True
    for col in SEPARATOR_COLS:
        expected_sep[:, col] = True
    assert np.all(board[expected_sep] == GREY), "separator lines must be grey"
    assert not np.any(board[~expected_sep] == GREY), "grey should only appear on separators"
    return board


def _extract_tiles(board: np.ndarray) -> list[list[np.ndarray]]:
    tiles: list[list[np.ndarray]] = []
    for tile_r in range(2):
        row_tiles = []
        for tile_c in range(3):
            r0, c0 = _tile_origin(tile_r, tile_c)
            row_tiles.append(board[r0 : r0 + TILE, c0 : c0 + TILE].copy())
        tiles.append(row_tiles)
    return tiles


def _mask_from_tile(tile: np.ndarray, color: int) -> np.ndarray:
    allowed = {BLACK, int(color)}
    values = set(map(int, np.unique(tile)))
    assert values.issubset(allowed), f"unexpected tile colors {values}"
    mask = tile == int(color)
    assert 3 <= int(mask.sum()) <= 8, "fragment area must be 3-8"
    assert _is_connected(mask), "fragment must be 4-connected"
    return mask


def _render_output(top_masks: list[np.ndarray], bottom_masks: list[np.ndarray], *, complete: bool) -> np.ndarray:
    out = np.full((TILE, BOARD_W), BLACK, dtype=int)
    for col in SEPARATOR_COLS:
        out[:, col] = GREY

    for tile_c in range(3):
        top = _complete_mask(top_masks[tile_c]) if complete else top_masks[tile_c]
        bottom = _complete_mask(bottom_masks[tile_c]) if complete else bottom_masks[tile_c]
        xor_mask = np.logical_xor(top, bottom)
        tile = np.full((TILE, TILE), BLACK, dtype=int)
        tile[xor_mask] = GREEN
        _r0, c0 = _tile_origin(0, tile_c)
        out[:, c0 : c0 + TILE] = tile
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Complete each tile fragment by 180-degree rotation, then XOR the completed masks columnwise.

    Input structure:
    - The grid contains a 2x3 array of 5x5 tiles on a black background, separated by grey single-cell lines.
    - Each upper tile contains one red 4-connected fragment and each lower tile contains one blue 4-connected fragment.
    - Every fragment is part of a shape that becomes 180-degree rotationally symmetric inside its own tile.

    Rule:
    - For each tile, rotate its fragment by 180 degrees around the tile center and union the rotated pixels with the original mask.
    - For each column, compare the completed upper and lower masks cell by cell.
    - Output one row of three 5x5 tiles with the grey vertical separators preserved, painting green exactly where one completed mask has foreground and the other does not.

    Key insight:
    - The missing half of each fragment is not supplied elsewhere; it must be reconstructed in place before the columnwise XOR is computed.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    board = _extract_board(g)
    tiles = _extract_tiles(board)

    top_masks = [_mask_from_tile(tile, RED) for tile in tiles[0]]
    bottom_masks = [_mask_from_tile(tile, BLUE) for tile in tiles[1]]
    return _render_output(top_masks, bottom_masks, complete=True)


def _random_connected_fragment() -> np.ndarray:
    for _ in range(FRAGMENT_RETRIES):
        target = random.randint(3, 8)
        start = (random.randrange(TILE), random.randrange(TILE))
        cells = {start}
        while len(cells) < target:
            frontier = sorted(
                {
                    nbr
                    for r, c in cells
                    for nbr in _neighbors4(r, c)
                    if 0 <= nbr[0] < TILE and 0 <= nbr[1] < TILE and nbr not in cells
                }
            )
            if not frontier:
                break
            if random.random() < 0.4:
                pivot = random.choice(tuple(cells))
                local = [
                    nbr
                    for nbr in _neighbors4(*pivot)
                    if 0 <= nbr[0] < TILE and 0 <= nbr[1] < TILE and nbr not in cells
                ]
                cells.add(random.choice(local if local else frontier))
            else:
                cells.add(random.choice(frontier))

        if len(cells) != target:
            continue

        mask = np.zeros((TILE, TILE), dtype=bool)
        for r, c in cells:
            mask[r, c] = True

        full = _complete_mask(mask)
        if np.array_equal(mask, full):
            continue
        if not _is_connected(full):
            continue
        full_area = int(full.sum())
        if not (4 <= full_area <= 12):
            continue
        if int(mask.sum()) * 10 < 4 * full_area:
            continue
        return mask
    raise RuntimeError("failed to generate a valid fragment")


def _sample_column_pair() -> tuple[np.ndarray, np.ndarray]:
    for _ in range(PAIR_RETRIES):
        top = _random_connected_fragment()
        bottom = _random_connected_fragment()
        top_full = _complete_mask(top)
        bottom_full = _complete_mask(bottom)
        raw_xor = np.logical_xor(top, bottom)
        full_xor = np.logical_xor(top_full, bottom_full)

        if np.array_equal(top_full, bottom_full):
            continue
        if not np.any(top_full & ~top):
            continue
        if not np.any(bottom_full & ~bottom):
            continue
        if np.array_equal(raw_xor, full_xor):
            continue
        if int(full_xor.sum()) < 2:
            continue
        return top, bottom
    raise RuntimeError("failed to generate a non-degenerate column pair")


def _assemble_board(top_masks: list[np.ndarray], bottom_masks: list[np.ndarray]) -> np.ndarray:
    board = np.full((BOARD_H, BOARD_W), BLACK, dtype=int)
    board[SEPARATOR_ROW, :] = GREY
    for col in SEPARATOR_COLS:
        board[:, col] = GREY

    for tile_c, mask in enumerate(top_masks):
        r0, c0 = _tile_origin(0, tile_c)
        board[r0 : r0 + TILE, c0 : c0 + TILE][mask] = RED
    for tile_c, mask in enumerate(bottom_masks):
        r0, c0 = _tile_origin(1, tile_c)
        board[r0 : r0 + TILE, c0 : c0 + TILE][mask] = BLUE
    return board


def _pad_board(board: np.ndarray) -> np.ndarray:
    top = random.randint(*PADDING_RANGE)
    bottom = random.randint(*PADDING_RANGE)
    left = random.randint(*PADDING_RANGE)
    right = random.randint(*PADDING_RANGE)
    grid = np.full((board.shape[0] + top + bottom, board.shape[1] + left + right), BLACK, dtype=int)
    grid[top : top + board.shape[0], left : left + board.shape[1]] = board
    return grid


def _semantic_validate(grid: np.ndarray) -> None:
    board = _extract_board(grid)
    tiles = _extract_tiles(board)
    top_masks = [_mask_from_tile(tile, RED) for tile in tiles[0]]
    bottom_masks = [_mask_from_tile(tile, BLUE) for tile in tiles[1]]

    for top, bottom in zip(top_masks, bottom_masks):
        top_full = _complete_mask(top)
        bottom_full = _complete_mask(bottom)
        assert int(top.sum()) * 10 >= 4 * int(top_full.sum())
        assert int(bottom.sum()) * 10 >= 4 * int(bottom_full.sum())
        assert np.any(top_full & ~top), "top fragment should require completion"
        assert np.any(bottom_full & ~bottom), "bottom fragment should require completion"
        assert not np.array_equal(top_full, bottom_full), "column XOR must be non-empty"
        assert not np.array_equal(np.logical_xor(top, bottom), np.logical_xor(top_full, bottom_full))

    expected = _render_output(top_masks, bottom_masks, complete=True)
    shortcut = _render_output(top_masks, bottom_masks, complete=False)
    actual = main(grid)
    assert np.array_equal(actual, expected), "main must follow the stated rule"
    assert not np.array_equal(actual, shortcut), "output must depend on completion, not raw fragments"
    assert len({np.logical_xor(_complete_mask(t), _complete_mask(b)).tobytes() for t, b in zip(top_masks, bottom_masks)}) >= 2


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        pairs = [_sample_column_pair() for _ in range(3)]
        top_masks = [top for top, _bottom in pairs]
        bottom_masks = [bottom for _top, bottom in pairs]

        if len({mask.tobytes() for mask in top_masks + bottom_masks}) < 4:
            continue

        board = _assemble_board(top_masks, bottom_masks)
        grid = _pad_board(board)
        try:
            _semantic_validate(grid)
        except AssertionError:
            continue
        return grid
    raise RuntimeError("failed to generate a valid puzzle instance")
