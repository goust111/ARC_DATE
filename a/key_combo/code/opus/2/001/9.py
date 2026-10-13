import random

import numpy as np


def main(grid: np.ndarray) -> np.ndarray:
    grid = np.asarray(grid)
    panels = [grid[0:5, i * 6 : i * 6 + 5] for i in range(4)]
    mask_area = grid[-10:, 0:10]
    quads = [mask_area[0:5, 0:5], mask_area[0:5, 5:10], mask_area[5:10, 0:5], mask_area[5:10, 5:10]]
    layers = []
    for i, quad in enumerate(quads):
        density = int(np.count_nonzero(quad == 6))
        if density == 0:
            continue
        clipped = np.where((quad == 6) & (panels[i] != 0), panels[i], 0)
        layers.append((density, i, clipped))
    out = np.zeros((5, 5), dtype=int)
    for _, _, layer in sorted(layers, key=lambda x: -x[0]):
        out[layer != 0] = layer[layer != 0]
    return out


def _panel_pattern(color):
    mask = np.zeros((5, 5), dtype=int)
    cells = random.sample([(r, c) for r in range(5) for c in range(5)], random.randint(5, 8))
    for r, c in cells:
        mask[r, c] = color
    return mask


def generate_input() -> np.ndarray:
    grid = np.zeros((17, 23), dtype=int)
    for sep in (5, 11, 17):
        grid[0:5, sep] = 5
    colors = [1, 2, 3, 4]
    random.shuffle(colors)
    panels = []
    for i, color in enumerate(colors):
        panel = _panel_pattern(color)
        panels.append(panel)
        grid[0:5, i * 6 : i * 6 + 5] = panel

    counts = random.sample(range(3, 10), 3) + [0]
    random.shuffle(counts)
    for i, count in enumerate(counts):
        rr = 7 if i < 2 else 12
        cc = 0 if i % 2 == 0 else 5
        if count:
            cells = random.sample([(r, c) for r in range(5) for c in range(5)], count)
            for r, c in cells:
                grid[rr + r, cc + c] = 6
    return grid
