import random
import numpy as np

BLACK = 0
GREY = 5
YELLOW = 4

TURN = {(-1, -1): "L", (-1, 1): "R", (1, -1): "S", (1, 1): "U"}


def main(grid: np.ndarray) -> np.ndarray:
    grid = np.array(grid, dtype=int)
    h, w = grid.shape
    out = grid.copy()
    endpoints = []
    for r in range(h):
        for c in range(w):
            if grid[r, c] not in (BLACK, GREY, YELLOW):
                endpoints.append((r, c, grid[r, c]))
    endpoints.sort()
    signals = {}
    for r in range(1, h - 1):
        for c in range(1, w - 1):
            if grid[r, c] == GREY:
                nbrs = [grid[r + dr, c + dc] == GREY for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1))]
                if all(nbrs):
                    for dr, dc in TURN:
                        if grid[r + dr, c + dc] == YELLOW:
                            signals[(r, c)] = TURN[(dr, dc)]
                            break
    for sr, sc, color in endpoints:
        nbrs = [(sr + dr, sc + dc) for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)) if 0 <= sr + dr < h and 0 <= sc + dc < w and grid[sr + dr, sc + dc] == GREY]
        if not nbrs:
            continue
        nr, nc = nbrs[0]
        dr, dc = nr - sr, nc - sc
        prev = (sr, sc)
        cur = (nr, nc)
        seen = set()
        while True:
            if cur in seen:
                break
            seen.add(cur)
            out[cur] = color
            r, c = cur
            if (r, c) in signals:
                action = signals[(r, c)]
                if action == "L":
                    dr, dc = -dc, dr
                elif action == "R":
                    dr, dc = dc, -dr
                elif action == "U":
                    dr, dc = -dr, -dc
                nr, nc = r + dr, c + dc
            else:
                nxts = []
                for rr, cc in ((r + 1, c), (r - 1, c), (r, c + 1), (r, c - 1)):
                    if 0 <= rr < h and 0 <= cc < w and (rr, cc) != prev and grid[rr, cc] == GREY:
                        nxts.append((rr, cc))
                if not nxts:
                    break
                nr, nc = nxts[0]
                dr, dc = nr - r, nc - c
            if not (0 <= nr < h and 0 <= nc < w):
                break
            if grid[nr, nc] not in (GREY,):
                break
            prev, cur = cur, (nr, nc)
    return out


def generate_input() -> np.ndarray:
    h, w = random.randint(13, 21), random.randint(13, 21)
    r = random.randint(4, h - 5)
    c = random.randint(4, w - 5)
    grid = np.zeros((h, w), dtype=int)
    grid[r, 1 : w - 1] = GREY
    grid[1 : h - 1, c] = GREY
    grid[r, 0] = random.choice([1, 2, 3, 6])
    grid[r, w - 1] = random.choice([4, 7, 8, 9])
    grid[0, c] = random.choice([1, 2, 3, 6, 7, 8, 9])
    grid[h - 1, c] = random.choice([1, 2, 3, 6, 7, 8, 9])
    dr, dc = random.choice(list(TURN))
    grid[r + dr, c + dc] = YELLOW
    return grid
