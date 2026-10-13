import random

import numpy as np


def _infer_seed_size(payload):
    ys, xs = np.where(payload != 0)
    extent = int(max(ys.max(), xs.max())) + 1 if len(ys) else 3
    if extent <= 3:
        return 3
    return 4


def main(grid: np.ndarray) -> np.ndarray:
    top = grid[0]
    left = grid[:, 0]
    h_offset = int(np.where(top == 1)[0][0]) - 1
    v_offset = int(np.where(left == 2)[0][0]) - 1
    payload = grid[1:, 1:]
    ph, pw = payload.shape
    s = _infer_seed_size(payload)
    seed = payload[:s, :s].copy()
    out = np.zeros_like(payload)
    for ty in range(ph // s):
        for tx in range(pw // s):
            tile = np.roll(seed, shift=(ty * v_offset) % s, axis=0)
            tile = np.roll(tile, shift=(tx * h_offset) % s, axis=1)
            out[ty * s : (ty + 1) * s, tx * s : (tx + 1) * s] = tile
    return out


def generate_input() -> np.ndarray:
    s = random.choice([3, 4])
    reps_y = random.randint(2, 4)
    reps_x = random.randint(2, 4)
    ph, pw = s * reps_y, s * reps_x
    grid = np.zeros((ph + 1, pw + 1), dtype=int)
    grid[0, :] = 5
    grid[:, 0] = 5
    h_offset = random.randrange(s)
    v_offset = random.randrange(s)
    grid[0, 1 + h_offset] = 1
    grid[1 + v_offset, 0] = 2
    seed = np.zeros((s, s), dtype=int)
    for r in range(s):
        for c in range(s):
            if random.random() < 0.35:
                seed[r, c] = 4
    seed[0, 0] = 4
    seed[s - 1, random.randrange(s)] = 4
    seed[random.randrange(s), s - 1] = 4
    grid[1 : 1 + s, 1 : 1 + s] = seed
    return grid
