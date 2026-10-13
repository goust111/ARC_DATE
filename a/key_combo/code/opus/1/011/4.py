import random

import numpy as np


def _find_panel(grid):
    h, w = grid.shape
    for r in range(h - 7):
        for c in range(w - 7):
            block = grid[r:r + 8, c:c + 8]
            if np.all(block != 0):
                return r, c
    raise ValueError("panel not found")


def _find_frame(grid):
    h, w = grid.shape
    for r in range(h - 5):
        for c in range(w - 5):
            b = grid[r:r + 6, c:c + 6]
            if b[0, 0] or b[0, 5] or b[5, 0] or b[5, 5]:
                continue
            if np.any(b[1:5, 1:5] != 0):
                continue
            sides = (b[0, 1:5], b[5, 1:5], b[1:5, 0], b[1:5, 5])
            if all(np.count_nonzero(x) == 4 and len(set(map(int, x))) == 1 for x in sides):
                return r, c
    raise ValueError("frame not found")


def main(grid: np.ndarray) -> np.ndarray:
    grid = np.array(grid, copy=True)
    pr, pc = _find_panel(grid)
    fr, fc = _find_frame(grid)
    source = grid[pr:pr + 4, pc:pc + 4]
    pieces = [source[:2, :2], source[:2, 2:], source[2:, :2], source[2:, 2:]]
    out = np.zeros_like(grid)
    out[fr:fr + 6, fc:fc + 6] = grid[fr:fr + 6, fc:fc + 6]
    side_color = {
        "top": int(grid[fr, fc + 1]),
        "bottom": int(grid[fr + 5, fc + 1]),
        "left": int(grid[fr + 1, fc]),
        "right": int(grid[fr + 1, fc + 5]),
    }
    for piece in pieces:
        color = int(piece[0, 0])
        side = next(k for k, v in side_color.items() if v == color)
        if side == "top":
            rr, cc = fr - 2, fc + 2
        elif side == "bottom":
            rr, cc = fr + 6, fc + 2
        elif side == "left":
            rr, cc = fr + 2, fc - 2
        else:
            rr, cc = fr + 2, fc + 6
        for r in range(2):
            for c in range(2):
                if 0 <= rr + r < out.shape[0] and 0 <= cc + c < out.shape[1]:
                    out[rr + r, cc + c] = piece[r, c]
    return out


def generate_input() -> np.ndarray:
    grid = np.zeros((22, 22), dtype=int)
    s = np.zeros((4, 4), dtype=int)
    layout = [1, 2, 3, 4]
    random.shuffle(layout)
    s[:2, :2] = layout[0]
    s[:2, 2:] = layout[1]
    s[2:, :2] = layout[2]
    s[2:, 2:] = layout[3]
    panel = np.block([[s, np.fliplr(s)], [np.flipud(s), np.flipud(np.fliplr(s))]])
    pr, pc = random.choice([(1, 1), (1, 12), (12, 1)])
    grid[pr:pr + 8, pc:pc + 8] = panel
    frame_spots = [(2, 11), (11, 2), (12, 12), (9, 2), (14, 8), (8, 14)]
    chosen = None
    for fr, fc in random.sample(frame_spots, len(frame_spots)):
        if grid[fr:fr + 6, fc:fc + 6].sum() == 0 and fr >= 2 and fc >= 2 and fr + 8 <= 22 and fc + 8 <= 22:
            chosen = (fr, fc)
            break
    if chosen is None:
        fr, fc = 11, 11
        grid[fr:fr + 6, fc:fc + 6] = 0
    else:
        fr, fc = chosen
    grid[fr, fc + 1:fc + 5] = 2
    grid[fr + 5, fc + 1:fc + 5] = 3
    grid[fr + 1:fr + 5, fc] = 1
    grid[fr + 1:fr + 5, fc + 5] = 4
    return grid
