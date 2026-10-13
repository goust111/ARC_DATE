import random
import numpy as np


def _erase_diagonal_noise(arr):
    out = arr.copy()
    while True:
        remove = []
        red = set(map(tuple, np.argwhere(out == 2)))
        for r, c in red:
            orth = any((r + dr, c + dc) in red for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)))
            diag = any((r + dr, c + dc) in red for dr, dc in ((1, 1), (1, -1), (-1, 1), (-1, -1)))
            if diag and not orth:
                remove.append((r, c))
        if not remove:
            return out
        for r, c in remove:
            out[r, c] = 0


def _frame_bbox(grid):
    pos = np.argwhere(grid == 5)
    return tuple(pos.min(axis=0)) + tuple(pos.max(axis=0))


def main(grid: np.ndarray) -> np.ndarray:
    out = _erase_diagonal_noise(grid)
    fr0, fc0, fr1, fc1 = _frame_bbox(out)
    green = np.argwhere(out[fr0 + 1:fr1, fc0 + 1:fc1] == 3)
    if len(green) == 0:
        return out
    gr0, gc0 = green.min(axis=0)
    gr1, gc1 = green.max(axis=0)
    budget_w = gc1 - gc0 + 1
    budget_h = gr1 - gr0 + 1
    h, w = out.shape
    for r in range(h):
        reds = [c for c in range(w) if out[r, c] == 2]
        for a, b in zip(reds, reds[1:]):
            gap = b - a - 1
            if 0 < gap <= budget_w and np.all(out[r, a + 1:b] == 0):
                out[r, a + 1:b] = 2
    for c in range(w):
        reds = [r for r in range(h) if out[r, c] == 2]
        for a, b in zip(reds, reds[1:]):
            gap = b - a - 1
            if 0 < gap <= budget_h and np.all(out[a + 1:b, c] == 0):
                out[a + 1:b, c] = 2
    out[fr0:fr1 + 1, fc0:fc1 + 1] = grid[fr0:fr1 + 1, fc0:fc1 + 1]
    return out


def _draw_frame(grid, top, left, height, width):
    grid[top, left:left + width] = 5
    grid[top + height - 1, left:left + width] = 5
    grid[top:top + height, left] = 5
    grid[top:top + height, left + width - 1] = 5


def _place_noise(grid):
    candidates = [(2, 12), (3, 13), (10, 2), (11, 3), (12, 4), (4, 15), (5, 16)]
    random.shuffle(candidates)
    for r, c in candidates[:random.randint(2, 4)]:
        if grid[r, c] == 0:
            grid[r, c] = 2


def generate_input() -> np.ndarray:
    grid = np.zeros((18, 18), dtype=int)
    top, left = random.randint(1, 2), random.randint(1, 2)
    fh, fw = random.randint(5, 7), random.randint(5, 7)
    _draw_frame(grid, top, left, fh, fw)
    pat = random.choice([
        [(1, 1), (1, 2), (2, 1), (2, 2), (2, 3)],
        [(1, 1), (2, 1), (3, 1), (3, 2)],
        [(1, 1), (1, 2), (1, 3), (2, 2), (3, 2)],
    ])
    for dr, dc in pat:
        if top + dr < top + fh - 1 and left + dc < left + fw - 1:
            grid[top + dr, left + dc] = 3
    rows = [11, 13]
    gaps = [random.randint(1, 3), random.randint(4, 5)]
    for row, gap in zip(rows, gaps):
        start = random.randint(1, 3)
        mid = start + random.randint(2, 3)
        end = mid + gap + random.randint(2, 3)
        grid[row, start:mid] = 2
        grid[row, mid + gap:end] = 2
    cols = [12, 15]
    gaps = [random.randint(1, 2), random.randint(3, 4)]
    for col, gap in zip(cols, gaps):
        start = random.randint(1, 3)
        mid = start + random.randint(2, 3)
        end = mid + gap + random.randint(2, 3)
        grid[start:mid, col] = 2
        grid[mid + gap:end, col] = 2
    _place_noise(grid)
    return grid
