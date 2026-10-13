import random

import numpy as np


def main(grid: np.ndarray) -> np.ndarray:
    """
    Parse:
        Interpret the input as an N x N binary mask with black as 0 and blue as
        1, where N is between 3 and 5.
    Analyze:
        For each mask cell, combine its binary value with the parity of its
        block coordinates to decide which block pattern should be emitted.
    Transform:
        Assemble an N x N array of N x N blocks using the specified lookup:
        positive mask, inverted mask, full black, or full blue.
    Return:
        Produce the full Kronecker-style expanded ndarray of shape (N*N, N*N).
    """
    grid = np.asarray(grid, dtype=int)
    mask = (grid == 1).astype(int)
    n = grid.shape[0]
    out = np.zeros((n * n, n * n), dtype=int)
    positive = mask.copy()
    inverted = 1 - positive
    full_black = np.zeros_like(mask)
    full_blue = np.ones_like(mask)

    for r in range(n):
        for c in range(n):
            if mask[r, c] == 0 and (r + c) % 2 == 0:
                block = positive
            elif mask[r, c] == 0 and (r + c) % 2 == 1:
                block = inverted
            elif mask[r, c] == 1 and (r + c) % 2 == 0:
                block = full_black
            else:
                block = full_blue
            out[r * n : (r + 1) * n, c * n : (c + 1) * n] = block
    return out


def generate_input() -> np.ndarray:
    for _ in range(5000):
        n = random.randint(3, 5)
        grid = np.random.randint(0, 2, size=(n, n), dtype=int)
        if int((grid == 0).sum()) >= 3 and int((grid == 1).sum()) >= 3:
            return grid
    raise RuntimeError("failed to generate a valid task-6 input")
