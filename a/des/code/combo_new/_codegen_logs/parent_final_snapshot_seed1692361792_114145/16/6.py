import random
import numpy as np

BLACK = 0
RED = 2
GREEN = 3
GREY = 5


def main(grid: np.ndarray) -> np.ndarray:
    grid = np.array(grid, dtype=int)
    sep_cols = [c for c in range(grid.shape[1]) if np.all(grid[:, c] == GREY)]
    left = grid[:, : sep_cols[0]]
    mid = grid[:, sep_cols[0] + 1 : sep_cols[1]]
    out = grid.copy()
    out[:, sep_cols[0] + 1 : sep_cols[1]] = mid
    right = np.full_like(mid, BLACK)
    only_left = (left != BLACK) & (mid == BLACK)
    only_mid = (mid != BLACK) & (left == BLACK)
    right[only_left] = RED
    right[only_mid] = GREEN
    out[:, sep_cols[1] + 1 :] = right
    return out


def random_mask(h: int, w: int, n: int) -> np.ndarray:
    cells = [(random.randrange(h), random.randrange(w))]
    seen = {cells[0]}
    while len(cells) < n:
        r, c = random.choice(cells)
        dr, dc = random.choice([(1, 0), (-1, 0), (0, 1), (0, -1)])
        nr, nc = r + dr, c + dc
        if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in seen:
            seen.add((nr, nc))
            cells.append((nr, nc))
    mask = np.zeros((h, w), dtype=int)
    for r, c in cells:
        mask[r, c] = 1
    return mask


def generate_input() -> np.ndarray:
    ph = random.randint(8, 13)
    pw = random.randint(5, 8)
    color = random.choice([1, 4, 6, 7, 8, 9])
    base = random_mask(ph, pw, random.randint(max(6, ph), min(ph * pw - 2, ph + pw + 6)))
    other = base.copy()
    edits = random.randint(2, max(2, min(6, ph * pw // 5)))
    for _ in range(edits):
        r, c = random.randrange(ph), random.randrange(pw)
        other[r, c] ^= 1
    if not np.any(base != other):
        other[random.randrange(ph), random.randrange(pw)] ^= 1
    grid = np.zeros((ph, pw * 3 + 2), dtype=int)
    grid[:, pw] = GREY
    grid[:, 2 * pw + 1] = GREY
    grid[:, :pw][base == 1] = color
    grid[:, pw + 1 : 2 * pw + 1][other == 1] = color
    return grid
