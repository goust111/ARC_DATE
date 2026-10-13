import random

import numpy as np

BLACK = 0
GREY = 5


def _legend(grid: np.ndarray):
    cells = []
    start = None
    for c in range(grid.shape[1]):
        if grid[-1, c] != GREY and start is None:
            start = c
        if (c == grid.shape[1] - 1 or grid[-1, c + 1] == GREY) and start is not None:
            cells.append(int(grid[-1, start]))
            start = None
    return cells


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill each room by counting its door gaps and looking up the matching legend color.

    Input structure:
    - Grey walls partition the upper part of the grid into rectangular rooms.
    - Some wall pixels are missing, forming 1-cell doors between adjacent rooms.
    - The bottom row is a grey-separated color legend whose k-th colored cell corresponds to door count k.

    Rule:
    - Read the bottom legend left-to-right.
    - Detect wall rows and wall columns in the room area above the legend separator.
    - For each rectangular room, count black door cells on its four surrounding wall segments and fill its black interior with the legend color for that count.

    Key insight:
    - The bottom row is a position-based lookup table, and the same missing wall pixel is counted separately for both rooms it connects.
    """
    out = grid.copy()
    legend = _legend(grid)
    room_w = int(np.max(np.sum(grid[:-2, :] == GREY, axis=1)))
    room_area = grid[:-2, :room_w]
    wall_rows = [r for r in range(room_area.shape[0]) if np.count_nonzero(room_area[r, :] == GREY) >= room_area.shape[1] - 3]
    wall_cols = [c for c in range(room_area.shape[1]) if np.count_nonzero(room_area[:, c] == GREY) >= room_area.shape[0] - 3]
    for r0, r1 in zip(wall_rows, wall_rows[1:]):
        for c0, c1 in zip(wall_cols, wall_cols[1:]):
            if r1 - r0 <= 1 or c1 - c0 <= 1:
                continue
            doors = 0
            doors += int(np.count_nonzero(grid[r0, c0 + 1:c1] == BLACK))
            doors += int(np.count_nonzero(grid[r1, c0 + 1:c1] == BLACK))
            doors += int(np.count_nonzero(grid[r0 + 1:r1, c0] == BLACK))
            doors += int(np.count_nonzero(grid[r0 + 1:r1, c1] == BLACK))
            if doors <= 0 or doors > len(legend):
                continue
            room = out[r0 + 1:r1, c0 + 1:c1]
            room[room == BLACK] = legend[doors - 1]
    return out


def generate_input() -> np.ndarray:
    rng = random.Random()
    legend_colors = [1, 2, 3, 4, 6]
    for _ in range(400):
        room_rows = rng.randint(2, 3)
        room_cols = rng.randint(2, 3)
        hs = [rng.randint(2, 4) for _ in range(room_rows)]
        ws = [rng.randint(2, 4) for _ in range(room_cols)]
        room_h = sum(hs) + room_rows + 1
        room_w = sum(ws) + room_cols + 1
        legend_n = 4
        total_w = max(room_w, 2 * legend_n + 1)
        total_h = room_h + 2
        if total_h > 30 or total_w > 30:
            continue
        grid = np.zeros((total_h, total_w), dtype=int)
        row_lines = [0]
        acc = 0
        for hh in hs:
            acc += hh + 1
            row_lines.append(acc)
        col_lines = [0]
        acc = 0
        for ww in ws:
            acc += ww + 1
            col_lines.append(acc)
        for r in row_lines:
            grid[r, :room_w] = GREY
        for c in col_lines:
            grid[:room_h, c] = GREY
        grid[room_h, :total_w] = GREY
        for i, color in enumerate(legend_colors[:legend_n]):
            c0 = 1 + 2 * i
            grid[-1, c0] = color
            if c0 + 1 < total_w:
                grid[-1, c0 + 1] = color
            if c0 - 1 >= 0:
                grid[-1, c0 - 1] = GREY
            if c0 + 2 < total_w:
                grid[-1, c0 + 2] = GREY
        # create doors
        for c in col_lines[1:-1]:
            for ridx in range(room_rows):
                if rng.random() < 0.8:
                    r0 = row_lines[ridx] + 1
                    r1 = row_lines[ridx + 1] - 1
                    rr = rng.randint(r0, r1)
                    grid[rr, c] = BLACK
        for r in row_lines[1:-1]:
            for cidx in range(room_cols):
                if rng.random() < 0.8:
                    c0 = col_lines[cidx] + 1
                    c1 = col_lines[cidx + 1] - 1
                    cc = rng.randint(c0, c1)
                    grid[r, cc] = BLACK
        # ensure at least one door per room
        for ridx in range(room_rows):
            for cidx in range(room_cols):
                r0, r1 = row_lines[ridx], row_lines[ridx + 1]
                c0, c1 = col_lines[cidx], col_lines[cidx + 1]
                doors = (
                    np.count_nonzero(grid[r0, c0 + 1:c1] == BLACK)
                    + np.count_nonzero(grid[r1, c0 + 1:c1] == BLACK)
                    + np.count_nonzero(grid[r0 + 1:r1, c0] == BLACK)
                    + np.count_nonzero(grid[r0 + 1:r1, c1] == BLACK)
                )
                if doors == 0:
                    rr = rng.randint(r0 + 1, r1 - 1)
                    grid[rr, c1] = BLACK
        return grid
    raise RuntimeError("failed to generate task 14/2")
