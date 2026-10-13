import random
import numpy as np


def main(grid: np.ndarray) -> np.ndarray:
    tile_pos = np.argwhere((grid == 4) | (grid == 7))
    tr0, tc0 = tile_pos.min(axis=0)
    tr1, tc1 = tile_pos.max(axis=0)
    tile = grid[tr0:tr1 + 1, tc0:tc1 + 1]
    mask_w = tc0
    half_h = grid.shape[0] // 2
    upper = grid[:half_h, :mask_w]
    lower = grid[half_h:half_h * 2, :mask_w]
    both_bg = (upper == 0) & (lower == 0)
    counts = both_bg.sum(axis=0)
    out = np.zeros((half_h, mask_w), dtype=int)
    th, tw = tile.shape
    mirrored = tile[:, ::-1]
    for r in range(half_h):
        for c in range(mask_w):
            src = mirrored if counts[c] % 2 else tile
            out[r, c] = src[r % th, c % tw]
    return out


def generate_input() -> np.ndarray:
    for _ in range(200):
        mh = random.randint(4, 6)
        mask_w = random.choice([6, 9, 12])
        grid = np.zeros((mh * 2, mask_w + 3), dtype=int)
        upper_fg = np.random.random((mh, mask_w)) < random.uniform(0.25, 0.45)
        lower_fg = np.random.random((mh, mask_w)) < random.uniform(0.25, 0.45)
        grid[:mh, :mask_w][upper_fg] = 1
        grid[mh:mh * 2, :mask_w][lower_fg] = 2
        both_bg = (~upper_fg) & (~lower_fg)
        parities = both_bg.sum(axis=0) % 2
        if len(set(parities.tolist())) < 2:
            continue
        tile = np.random.choice([4, 7], size=(3, 3))
        if np.array_equal(tile, tile[:, ::-1]):
            tile[0, 0] = 4
            tile[0, 2] = 7
        grid[-3:, mask_w:mask_w + 3] = tile
        return grid
    return generate_input()
