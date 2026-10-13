import random
import numpy as np


def _frames(grid):
    h, w = grid.shape
    frames = []
    for r in range(h - 2):
        for c in range(w - 2):
            block = grid[r:r + 3, c:c + 3]
            border = [int(block[i, j]) for i in range(3) for j in range(3) if (i, j) != (1, 1)]
            center = int(block[1, 1])
            if len(set(border)) == 1 and border[0] not in (0, 5) and center not in (0, 5, border[0]):
                frames.append((c, r, center))
    return sorted(frames)


def main(grid: np.ndarray) -> np.ndarray:
    grid = np.array(grid, copy=True)
    h, w = grid.shape
    frames = _frames(grid)
    palette = [center for _, _, center in frames]
    gray_pts = np.argwhere(grid == 5)
    marker_col = int(gray_pts[:, 1].min())
    marker_rows = sorted(int(r) for r, c in gray_pts if c == marker_col)
    height = 1
    if marker_rows:
        height = 0
        expected = marker_rows[0]
        for r in marker_rows:
            if r == expected:
                height += 1
                expected += 1
            else:
                break

    out = np.zeros_like(grid)
    out[grid == 5] = 5
    for r in range(h):
        idx = (r // height) % len(palette)
        out[r, w - 1] = palette[idx]
    return out


def _can_place(grid, top, left):
    h, w = grid.shape
    if top < 0 or left < 0 or top + 3 > h or left + 3 > w - 1:
        return False
    area = grid[max(0, top - 1):min(h, top + 4), max(0, left - 1):min(w, left + 4)]
    return np.all(area == 0)


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(1000):
        h, w = 15, 18
        grid = np.zeros((h, w), dtype=int)
        marker_h = rng.randint(2, 4)
        grid[:marker_h, 0] = 5
        n = rng.randint(3, 5)
        border = 1
        centers = rng.sample([2, 3, 4, 6, 8, 9], n)
        placed = []
        for center in centers:
            ok = False
            for _try in range(200):
                top = rng.randint(1, h - 4)
                left = rng.randint(2, w - 5)
                if _can_place(grid, top, left):
                    grid[top:top + 3, left:left + 3] = border
                    grid[top + 1, left + 1] = center
                    placed.append((top, left))
                    ok = True
                    break
            if not ok:
                break
        if len(placed) != n:
            continue
        for _d in range(rng.randint(3, 7)):
            for _try in range(100):
                r = rng.randrange(h)
                c = rng.randrange(1, w - 1)
                if grid[r, c] != 0:
                    continue
                near = False
                for rr, cc in placed:
                    if rr - 1 <= r <= rr + 3 and cc - 1 <= c <= cc + 3:
                        near = True
                        break
                if not near:
                    grid[r, c] = border
                    break
        return grid
    return generate_input()
