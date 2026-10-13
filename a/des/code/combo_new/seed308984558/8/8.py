import random

import numpy as np


BLACK = 0
YELLOW = 4
COLORS = [1, 2, 3, 4, 6, 7, 8, 9]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Convert each bottom tower's layer count and top color into a target-side bar.

    Input structure:
    - The lower part of the grid contains bottom-aligned one-column towers made of
      stacked contiguous color layers.
    - The upper part contains single target pixels whose color matches exactly one
      tower's bottom layer color.
    - The rest of the canvas is black.

    Rule:
    - For each tower, count the number of contiguous color layers and record the
      tower's topmost color and bottom-layer color.
    - Match the tower to the unique target pixel with the same bottom-layer color.
    - Draw a vertical bar of height equal to the layer count starting at the target
      cell and extending downward, colored by the tower's top color.
    - Remove all source towers from the output.

    Key insight:
    - The bottom color is only a lookup key, while the output uses two other tower
      properties: number of layers for height and topmost color for paint.
    """
    h, w = grid.shape
    out = grid.copy()
    towers = []
    for c in range(w):
        if grid[h - 1, c] == BLACK:
            continue
        r = h - 1
        cells = []
        while r >= 0 and grid[r, c] != BLACK:
            cells.append((r, int(grid[r, c])))
            r -= 1
        cells.reverse()
        if len(cells) < 2:
            continue
        layers = 1
        for i in range(1, len(cells)):
            if cells[i][1] != cells[i - 1][1]:
                layers += 1
        top_color = cells[0][1]
        bottom_color = cells[-1][1]
        towers.append((c, layers, top_color, bottom_color, len(cells)))
        out[h - len(cells) : h, c] = BLACK

    for c, layers, top_color, bottom_color, _ in towers:
        locs = np.argwhere(out == bottom_color)
        locs = [tuple(x) for x in locs if x[0] < h - 1]
        if len(locs) != 1:
            continue
        r0, c0 = locs[0]
        for k in range(layers):
            rr = r0 + k
            if rr < h:
                out[rr, c0] = top_color
    return out


def generate_input() -> np.ndarray:
    while True:
        h = random.randint(16, 26)
        w = random.randint(12, 22)
        grid = np.zeros((h, w), dtype=int)
        tower_cols = sorted(random.sample(range(1, w - 1), random.randint(2, 4)))
        base_colors = random.sample(COLORS, len(tower_cols))
        target_rows = random.sample(range(1, h // 2), len(tower_cols))
        target_cols = random.sample(range(1, w - 1), len(tower_cols))
        for col, base_color, tr, tc in zip(tower_cols, base_colors, target_rows, target_cols):
            layers = random.randint(2, 4)
            layer_colors = [base_color]
            while len(layer_colors) < layers:
                nxt = random.choice(COLORS)
                if nxt != layer_colors[-1]:
                    layer_colors.append(nxt)
            heights = [random.randint(1, 2) for _ in range(layers)]
            total_h = sum(heights)
            if total_h >= h // 2:
                total_h = h // 2 - 1
            r = h - 1
            for color, lh in zip(layer_colors, heights):
                for _ in range(lh):
                    grid[r, col] = color
                    r -= 1
            grid[tr, tc] = base_color
        return grid
