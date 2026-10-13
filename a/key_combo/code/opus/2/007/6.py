import random
import numpy as np


def main(grid: np.ndarray) -> np.ndarray:
    out = np.zeros_like(grid)
    controls = []
    for color in (1, 3, 4, 9):
        r, c = np.argwhere(grid == color)[0]
        controls.append((int(r), int(c), color))
    pr, pc = map(int, np.argwhere(grid == 2)[0])
    top = min(controls, key=lambda x: x[0])
    bottom = max(controls, key=lambda x: x[0])
    left = min(controls, key=lambda x: x[1])
    right = max(controls, key=lambda x: x[1])
    tr, br, lc, rc = top[0], bottom[0], left[1], right[1]

    out[tr, lc:rc + 1] = top[2]
    out[br, lc:rc + 1] = bottom[2]
    out[tr + 1:br, lc] = left[2]
    out[tr + 1:br, rc] = right[2]
    out[pr, pc] = 2

    diag = np.zeros_like(grid, dtype=bool)
    for dr, dc in ((-1, -1), (-1, 1), (1, -1), (1, 1)):
        r, c = pr + dr, pc + dc
        while tr < r < br and lc < c < rc:
            out[r, c] = 5
            diag[r, c] = True
            r += dr
            c += dc

    for r in range(tr + 1, br):
        for c in range(lc + 1, rc):
            if (r, c) == (pr, pc) or diag[r, c]:
                continue
            dr, dc = r - pr, c - pc
            if dr < 0 and abs(dc) < -dr:
                out[r, c] = top[2]
            elif dr > 0 and abs(dc) < dr:
                out[r, c] = bottom[2]
            elif dc < 0 and abs(dr) < -dc:
                out[r, c] = left[2]
            elif dc > 0 and abs(dr) < dc:
                out[r, c] = right[2]
            else:
                out[r, c] = 5
    return out


def generate_input() -> np.ndarray:
    for _ in range(1000):
        h = w = random.randint(13, 16)
        tr = random.randint(1, 3)
        br = random.randint(10, h - 2)
        lc = random.randint(1, 3)
        rc = random.randint(10, w - 2)
        if br - tr <= 5 or rc - lc <= 5:
            continue
        grid = np.zeros((h, w), dtype=int)
        colors = random.sample([1, 3, 4, 9], 4)
        pos = [
            (tr, random.randint(lc + 2, rc - 2)),
            (br, random.randint(lc + 2, rc - 2)),
            (random.randint(tr + 2, br - 2), lc),
            (random.randint(tr + 2, br - 2), rc),
        ]
        for (r, c), color in zip(pos, colors):
            grid[r, c] = color
        pr = random.randint(tr + 2, br - 2)
        pc = random.randint(lc + 2, rc - 2)
        if pr == (tr + br) // 2 or pc == (lc + rc) // 2:
            continue
        grid[pr, pc] = 2
        return grid
    raise RuntimeError("failed to generate input")
