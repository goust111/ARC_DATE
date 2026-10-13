import random
import numpy as np
from collections import Counter


BG = 0
GREY = 5


def main(grid: np.ndarray) -> np.ndarray:
    """
    Find the odd-colored cell in each row of a grey grid and repaint the other cell borders with that color.

    Input structure:
    - Grey lines partition the grid into equal cells.
    - Every cell contains one small monochrome figure on a black background.
    - In each row, exactly one figure has a color different from the others.

    Rule:
    - Determine the odd-color cell in each row.
    - Keep its border grey.
    - Repaint the full borders of the other cells in that row to the odd figure's color.
    - Leave all figures unchanged.

    Key insight:
    - The odd cell serves as a color donor; the transformed elements are the neighboring
      cell frames, not the odd figure itself.
    """
    out = grid.copy()
    rows = [r for r in range(grid.shape[0]) if np.all(grid[r] == GREY)]
    cols = [c for c in range(grid.shape[1]) if np.all(grid[:, c] == GREY)]
    rs = [-1] + rows + [grid.shape[0]]
    cs = [-1] + cols + [grid.shape[1]]
    for i in range(len(rs) - 1):
        cell_colors = []
        for j in range(len(cs) - 1):
            cell = grid[rs[i] + 1 : rs[i + 1], cs[j] + 1 : cs[j + 1]]
            colors = [int(c) for c in np.unique(cell) if c != BG]
            cell_colors.append(colors[0] if colors else BG)
        freq = Counter(cell_colors)
        odd_color = min(freq, key=freq.get)
        odd_idx = cell_colors.index(odd_color)
        for j in range(len(cs) - 1):
            if j == odd_idx:
                continue
            r0, r1 = rs[i] + 1, rs[i + 1] - 1
            c0, c1 = cs[j] + 1, cs[j + 1] - 1
            ra0, ra1 = max(0, r0 - 1), min(out.shape[0] - 1, r1 + 1)
            ca0, ca1 = max(0, c0 - 1), min(out.shape[1] - 1, c1 + 1)
            out[ra0, ca0 : ca1 + 1] = odd_color
            out[ra1, ca0 : ca1 + 1] = odd_color
            out[ra0 : ra1 + 1, ca0] = odd_color
            out[ra0 : ra1 + 1, ca1] = odd_color
    return out


def generate_input() -> np.ndarray:
    rows = random.randint(2, 3); cols = random.randint(3, 4); cell = 4
    H = rows * cell + rows + 1; W = cols * cell + cols + 1
    grid = np.zeros((H, W), dtype=int)
    for r in range(0, H, cell + 1):
        grid[r, :] = GREY
    for c in range(0, W, cell + 1):
        grid[:, c] = GREY
    base_colors = random.sample([1, 2, 3, 4, 6, 7, 8, 9], rows * 2)
    idx = 0
    for i in range(rows):
        common = base_colors[idx]; odd = base_colors[idx + 1]; idx += 2
        odd_j = random.randint(0, cols - 1)
        for j in range(cols):
            color = odd if j == odd_j else common
            r = 1 + i * (cell + 1)
            c = 1 + j * (cell + 1)
            grid[r + 1, c + 1 : c + 3] = color
            grid[r + 2, c + 2] = color
    return grid
