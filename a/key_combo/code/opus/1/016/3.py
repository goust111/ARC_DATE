import random
import numpy as np


def _rot(arr, k):
    return np.rot90(arr, -(k % 4))


def main(grid: np.ndarray) -> np.ndarray:
    out = grid.copy()
    corners = [(0, 0), (0, grid.shape[1] - 1), (grid.shape[0] - 1, grid.shape[1] - 1), (grid.shape[0] - 1, 0)]
    mark_idx = next(i for i, p in enumerate(corners) if grid[p] == 2)
    out[corners[mark_idx]] = 0
    pts = np.argwhere((out != 0) & (out != 2))
    r0, c0 = pts.min(axis=0)
    r1, c1 = pts.max(axis=0)
    center = ((r0 + r1) / 2, (c0 + c1) / 2)
    near_idx = min(range(4), key=lambda i: abs(center[0] - corners[i][0]) + abs(center[1] - corners[i][1]))
    k = (mark_idx - near_idx) % 4
    crop = out[r0:r1 + 1, c0:c1 + 1].copy()
    out[r0:r1 + 1, c0:c1 + 1] = 0
    rotated = _rot(crop, k)
    if rotated.shape[0] != rotated.shape[1]:
        size = max(rotated.shape)
        padded = np.zeros((size, size), dtype=rotated.dtype)
        pr = (size - rotated.shape[0]) // 2
        pc = (size - rotated.shape[1]) // 2
        padded[pr:pr + rotated.shape[0], pc:pc + rotated.shape[1]] = rotated
        rotated = padded
    rr0 = int(round(center[0] - (rotated.shape[0] - 1) / 2))
    cc0 = int(round(center[1] - (rotated.shape[1] - 1) / 2))
    color_vals = [int(v) for v in rotated.ravel() if v != 0]
    color = color_vals[0] if color_vals else 1
    full = rotated.copy()
    n = full.shape[0]
    for r, c in np.argwhere(rotated != 0):
        positions = [(int(r), int(c))]
        x, y = int(r), int(c)
        for _ in range(3):
            x, y = y, n - 1 - x
            positions.append((x, y))
        for x, y in positions:
            if 0 <= x < full.shape[0] and 0 <= y < full.shape[1]:
                full[x, y] = color
    for r in range(full.shape[0]):
        for c in range(full.shape[1]):
            if full[r, c] != 0:
                rr, cc = rr0 + r, cc0 + c
                if 0 <= rr < out.shape[0] and 0 <= cc < out.shape[1]:
                    out[rr, cc] = full[r, c]
    return out


def _complete_pattern():
    pts = {(0, 2), (1, 1), (1, 2), (2, 0), (2, 1), (2, 2)}
    full = set()
    for r, c in pts:
        x, y = r, c
        for _ in range(4):
            full.add((x, y))
            x, y = y, 4 - x
    return list(full)


def generate_input() -> np.ndarray:
    grid = np.zeros((13, 13), dtype=int)
    color = random.choice([1, 3, 4])
    top, left = random.choice([(2, 4), (4, 2), (5, 6), (6, 3)])
    pts = _complete_pattern()
    remove = set(random.sample(pts, random.randint(2, 4)))
    for r, c in pts:
        if (r, c) not in remove:
            grid[top + r, left + c] = color
    corner = random.randrange(4)
    positions = [(0, 0), (0, 12), (12, 12), (12, 0)]
    grid[positions[corner]] = 2
    return grid
