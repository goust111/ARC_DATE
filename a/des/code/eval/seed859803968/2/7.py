from __future__ import annotations

import random

import numpy as np


BLACK = 0
GREEN = 3
YELLOW = 4
BOARD_COLOR_POOL = (1, 2, 5, 6, 7, 8, 9)

BOARD_H_RANGE = (6, 16)
BOARD_W_RANGE = (6, 16)
MAX_GENERATION_ATTEMPTS = 400
MAX_HOLE_PLACEMENT_ATTEMPTS = 120


def _checkerboard(height: int, width: int, even_color: int, odd_color: int) -> np.ndarray:
    rows, cols = np.indices((int(height), int(width)))
    return np.where((rows + cols) % 2 == 0, int(even_color), int(odd_color)).astype(int)


def _reconstruct_checkerboard(board: np.ndarray) -> np.ndarray:
    board = np.asarray(board, dtype=int)
    colors = sorted(set(map(int, np.unique(board))) - {BLACK})
    assert len(colors) == 2, "board must contain exactly two visible checker colors"

    visible = board != BLACK
    assert bool(np.any(visible)), "at least one checker cell must remain visible"

    rows, cols = np.indices(board.shape)
    parity_even = (rows + cols) % 2 == 0
    candidates: list[np.ndarray] = []
    for even_color, odd_color in ((colors[0], colors[1]), (colors[1], colors[0])):
        filled = np.where(parity_even, int(even_color), int(odd_color)).astype(int)
        if bool(np.array_equal(filled[visible], board[visible])):
            candidates.append(filled)

    assert len(candidates) == 1, "visible cells must determine the checker phase uniquely"
    return candidates[0]


def _translate(board: np.ndarray, dx: int, dy: int) -> np.ndarray:
    board = np.asarray(board, dtype=int)
    height, width = map(int, board.shape)
    out = np.full((height, width), BLACK, dtype=int)
    if int(dx) >= width or int(dy) >= height:
        return out
    out[int(dy) :, int(dx) :] = board[: height - int(dy), : width - int(dx)]
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Restore the erased checkerboard, then shift it by counts encoded in the clue bands.

    Input:
    - row 0 is a green top band with some yellow clue pixels,
    - col 0 is a green left band with some yellow clue pixels,
    - the interior is a two-color checkerboard with black rectangular holes.

    Output:
    - the restored checkerboard interior shifted right by the number of top-band
      yellow pixels and down by the number of left-band yellow pixels,
    - cropped back to the interior size and black-filled where content shifts in.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.shape[0] >= 2 and g.shape[1] >= 2
    assert bool(np.all(np.isin(g[0], (GREEN, YELLOW))))
    assert bool(np.all(np.isin(g[:, 0], (GREEN, YELLOW))))

    dx = int(np.count_nonzero(g[0, 1:] == YELLOW))
    dy = int(np.count_nonzero(g[1:, 0] == YELLOW))

    board = g[1:, 1:]
    restored = _reconstruct_checkerboard(board)
    return _translate(restored, dx=dx, dy=dy)


def _sample_shift(limit: int) -> int:
    assert int(limit) >= 1
    if int(limit) == 1 or random.random() < 0.35:
        return 0
    return random.randint(1, int(limit) - 1)


def _can_place_rect(mask: np.ndarray, r0: int, c0: int, height: int, width: int) -> bool:
    h, w = map(int, mask.shape)
    rr0 = max(0, int(r0) - 1)
    cc0 = max(0, int(c0) - 1)
    rr1 = min(h, int(r0) + int(height) + 1)
    cc1 = min(w, int(c0) + int(width) + 1)
    return not bool(mask[rr0:rr1, cc0:cc1].any())


def _hole_count_options(height: int, width: int) -> int:
    area = int(height) * int(width)
    if area >= 140:
        return 4
    if area >= 90:
        return 3
    return 2


def _sample_hole_mask(height: int, width: int) -> np.ndarray | None:
    mask = np.zeros((int(height), int(width)), dtype=bool)
    hole_target = random.randint(1, _hole_count_options(height, width))
    hole_area_limit = int(0.38 * int(height) * int(width))
    placed = 0
    covered = 0

    for _ in range(MAX_HOLE_PLACEMENT_ATTEMPTS):
        if placed == hole_target:
            break

        rect_h = random.randint(1, max(1, int(height) // 2))
        rect_w = random.randint(1, max(1, int(width) // 2))
        area = rect_h * rect_w
        if covered + area > hole_area_limit:
            continue

        r0 = random.randint(0, int(height) - rect_h)
        c0 = random.randint(0, int(width) - rect_w)
        if not _can_place_rect(mask, r0, c0, rect_h, rect_w):
            continue

        mask[r0 : r0 + rect_h, c0 : c0 + rect_w] = True
        covered += area
        placed += 1

    if placed != hole_target:
        return None
    if not bool(mask.any()):
        return None
    return mask


def _build_input(
    board_h: int,
    board_w: int,
    even_color: int,
    odd_color: int,
    dx: int,
    dy: int,
    hole_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    full_board = _checkerboard(board_h, board_w, even_color, odd_color)
    board_with_holes = full_board.copy()
    board_with_holes[np.asarray(hole_mask, dtype=bool)] = BLACK

    grid = np.full((int(board_h) + 1, int(board_w) + 1), GREEN, dtype=int)
    grid[1:, 1:] = board_with_holes
    grid[0, 0] = GREEN

    if dx:
        for c in random.sample(range(1, int(board_w) + 1), int(dx)):
            grid[0, c] = YELLOW
    if dy:
        for r in random.sample(range(1, int(board_h) + 1), int(dy)):
            grid[r, 0] = YELLOW

    return grid, full_board


def _valid_generated_case(grid: np.ndarray, full_board: np.ndarray, dx: int, dy: int) -> bool:
    interior = grid[1:, 1:]
    visible_colors = set(map(int, np.unique(interior))) - {BLACK}
    if len(visible_colors) != 2:
        return False
    if not bool(np.any(interior == BLACK)):
        return False

    expected = _translate(full_board, dx=dx, dy=dy)
    try:
        actual = main(grid)
    except Exception:
        return False
    return bool(np.array_equal(actual, expected))


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        board_h = random.randint(*BOARD_H_RANGE)
        board_w = random.randint(*BOARD_W_RANGE)
        even_color, odd_color = random.sample(BOARD_COLOR_POOL, 2)
        dx = _sample_shift(board_w)
        dy = _sample_shift(board_h)

        hole_mask = _sample_hole_mask(board_h, board_w)
        if hole_mask is None:
            continue

        grid, full_board = _build_input(
            board_h=board_h,
            board_w=board_w,
            even_color=even_color,
            odd_color=odd_color,
            dx=dx,
            dy=dy,
            hole_mask=hole_mask,
        )
        if _valid_generated_case(grid, full_board, dx=dx, dy=dy):
            return grid

    raise RuntimeError("failed to generate a valid checkerboard-shift task instance")
