import random
import numpy as np


MASKS = {
    (1, 1, 1, 1, 1, 1, 1, 1, 1): 2,
    (0, 1, 0, 1, 1, 1, 0, 1, 0): 1,
    (1, 0, 0, 1, 0, 0, 1, 1, 1): 3,
    (1, 0, 0, 0, 1, 0, 0, 0, 1): 4,
    (1, 1, 1, 1, 0, 1, 1, 1, 1): 8,
}


def _panel_positions(n, corner):
    for i in range(n // 3 + 1):
        r, c = 3 * i, 3 * i
        if corner == 1:
            r, c = 3 * i, n - 3 - 3 * i
        elif corner == 2:
            r, c = n - 3 - 3 * i, n - 3 - 3 * i
        elif corner == 3:
            r, c = n - 3 - 3 * i, 3 * i
        if 0 <= r <= n - 3 and 0 <= c <= n - 3:
            yield r, c


def main(grid: np.ndarray) -> np.ndarray:
    n = grid.shape[0]
    best = []
    for corner in range(4):
        panels = []
        for r, c in _panel_positions(n, corner):
            block = grid[r:r + 3, c:c + 3]
            if np.any(block == 5):
                key = tuple(int(v == 5) for v in block.ravel())
                panels.append(MASKS.get(key, 9))
            else:
                break
        if len(panels) > len(best):
            best = panels
    colors = best or [2]
    out = np.zeros_like(grid)
    layers = (n + 1) // 2
    for layer in range(layers):
        color = colors[min(layer, len(colors) - 1)]
        out[layer:n - layer, layer] = color
        out[layer:n - layer, n - 1 - layer] = color
        out[layer, layer:n - layer] = color
        out[n - 1 - layer, layer:n - layer] = color
    return out


def generate_input() -> np.ndarray:
    k = random.choice([3, 4])
    n = 3 * k + 1
    if n % 2 == 0:
        n += 1
    grid = np.zeros((n, n), dtype=int)
    corner = random.randrange(4)
    keys = random.sample(list(MASKS.keys()), k)
    for (r, c), key in zip(_panel_positions(n, corner), keys):
        block = np.array(key).reshape(3, 3)
        grid[r:r + 3, c:c + 3][block == 1] = 5
    return grid
