import random
import numpy as np

BLACK = 0
RED = 2


def main(grid: np.ndarray) -> np.ndarray:
    grid = np.array(grid, dtype=int)
    h, w = grid.shape
    for r in range(h):
        if np.all(grid[r] == RED):
            top = grid[:r]
            bottom = grid[r + 1 :]
            out = np.zeros_like(bottom)
            for i in range(bottom.shape[0]):
                a = top[top.shape[0] - 1 - i]
                b = bottom[i]
                row = np.where(a != BLACK, a, b)
                out[i] = np.where((a != BLACK) & (b != BLACK), a, row)
            return out
    for c in range(w):
        if np.all(grid[:, c] == RED):
            left = grid[:, :c]
            right = grid[:, c + 1 :]
            out = np.zeros_like(right)
            for i in range(right.shape[1]):
                a = left[:, left.shape[1] - 1 - i]
                b = right[:, i]
                col = np.where(a != BLACK, a, b)
                out[:, i] = np.where((a != BLACK) & (b != BLACK), a, col)
            return out
    raise ValueError("missing fold axis")


def generate_input() -> np.ndarray:
    n = random.choice([10, 12, 14])
    grid = np.zeros((n, n), dtype=int)
    colors = [1, 3, 4, 5, 6, 7, 8, 9]
    for _ in range(random.randint(8, 18)):
        r, c = random.randrange(n), random.randrange(n)
        grid[r, c] = random.choice(colors)
    if random.random() < 0.5:
        axis = n // 2
        grid[axis, :] = RED
    else:
        axis = n // 2
        grid[:, axis] = RED
    return grid
