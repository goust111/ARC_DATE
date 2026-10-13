import random

import numpy as np


BG = 0
GREY = 5
DIV = 9
CORNER_TO_ROT = {(0, 0): 0, (0, 2): 3, (2, 2): 2, (2, 0): 1}


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor each 3x3 tile from the right legend and rotate it from its colored corner mark.

    Input structure:
    - A full-height maroon divider splits the grid into a left tile panel and a right color legend.
    - The left panel is a regular grid of grey 3x3 squares separated by one black cell.
    - Each square has exactly one non-grey colored corner pixel, and the right panel has one legend pixel per tile position.

    Rule:
    - For tile (i, j), recolor all grey pixels to the legend color at (i, j) on the right panel while keeping the corner marker color.
    - Rotate the whole 3x3 tile based on which corner contains the marker.
    - Output only the transformed left-panel tile layout with the original spacing.

    Key insight:
    - Position in the right panel chooses the fill color, while the marked corner independently encodes the rotation angle.
    """
    g = np.array(grid, dtype=int)
    split = int(np.where(np.all(g == DIV, axis=0))[0][0])
    left = g[:, :split]
    right = g[:, split + 1 :]
    tile_pos = []
    for r in range(left.shape[0] - 2):
        for c in range(left.shape[1] - 2):
            block = left[r : r + 3, c : c + 3]
            if np.count_nonzero(block != BG) == 9 and np.count_nonzero(block == GREY) == 8:
                tile_pos.append((r, c))
    rows = sorted(set(r for r, _ in tile_pos))
    cols = sorted(set(c for _, c in tile_pos))
    legend_pts = [(r, c) for r in range(right.shape[0]) for c in range(right.shape[1]) if right[r, c] != BG]
    lrows = sorted(set(r for r, _ in legend_pts))
    lcols = sorted(set(c for _, c in legend_pts))
    out = np.zeros((len(rows) * 4 - 1, len(cols) * 4 - 1), dtype=int)
    for i, r in enumerate(rows):
        for j, c in enumerate(cols):
            tile = left[r : r + 3, c : c + 3].copy()
            marker = [(x, y) for x in [0, 2] for y in [0, 2] if tile[x, y] != GREY][0]
            fill = int(right[lrows[i], lcols[j]])
            tile[tile == GREY] = fill
            tile = np.rot90(tile, CORNER_TO_ROT[marker])
            rr, cc = i * 4, j * 4
            out[rr : rr + 3, cc : cc + 3] = tile
    return out


def generate_input() -> np.ndarray:
    m, n = random.randint(2, 3), random.randint(2, 3)
    left_h, left_w = m * 4 - 1, n * 4 - 1
    right_h, right_w = max(m + 1, 4), max(n + 1, 4)
    h = max(left_h, right_h)
    w = left_w + 1 + right_w
    g = np.zeros((h, w), dtype=int)
    g[:, left_w] = DIV
    colors = [1, 2, 3, 4]
    for i in range(m):
        for j in range(n):
            r, c = i * 4, j * 4
            g[r : r + 3, c : c + 3] = GREY
            mr, mc = random.choice([(0, 0), (0, 2), (2, 2), (2, 0)])
            g[r + mr, c + mc] = random.choice(colors)
    lrows = random.sample(range(right_h), m)
    lcols = random.sample(range(right_w), n)
    lrows.sort()
    lcols.sort()
    for i in range(m):
        for j in range(n):
            g[lrows[i], left_w + 1 + lcols[j]] = random.choice(colors)
    return g
