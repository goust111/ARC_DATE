import numpy as np
from collections import deque

BG = 0
GREEN = 3
BLUE = 1
RED = 2
YELLOW = 4
ORANGE = 7


def _flood_fill(mask: np.ndarray, sr: int, sc: int, vis: np.ndarray):
    h, w = mask.shape
    q = deque([(sr, sc)])
    vis[sr, sc] = True
    comp = []
    while q:
        r, c = q.popleft()
        comp.append((r, c))
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not vis[nr, nc]:
                vis[nr, nc] = True
                q.append((nr, nc))
    return comp


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid)
    top = g[:10, :10]
    bot = g[10:, :10]

    out = np.zeros((10, 10), dtype=np.int64)
    top_fg = top != BG
    bot_fg = bot != BG

    inter = top_fg & bot_fg
    out[inter] = RED
    out[top_fg & ~bot_fg] = GREEN
    out[bot_fg & ~top_fg] = BLUE

    touch = np.zeros_like(inter, dtype=bool)
    for r in range(10):
        for c in range(10):
            if not inter[r, c]:
                continue
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                nr, nc = r + dr, c + dc
                if 0 <= nr < 10 and 0 <= nc < 10 and not inter[nr, nc]:
                    touch[nr, nc] = True

    for r in range(10):
        for c in range(10):
            if touch[r, c]:
                if out[r, c] == GREEN:
                    out[r, c] = YELLOW
                elif out[r, c] == BLUE:
                    out[r, c] = ORANGE
                elif out[r, c] == BG:
                    out[r, c] = BG

    return out


def _rand_blob(rng: np.random.Generator, h: int, w: int, n: int):
    cells = set()
    r = int(rng.integers(0, h))
    c = int(rng.integers(0, w))
    cells.add((r, c))
    frontier = [(r, c)]
    while len(cells) < n:
        rr, cc = frontier[int(rng.integers(0, len(frontier)))]
        dr, dc = rng.choice([(-1, 0), (1, 0), (0, -1), (0, 1)])
        nr, nc = rr + dr, cc + dc
        if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in cells:
            cells.add((nr, nc))
            frontier.append((nr, nc))
    return cells


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    g = np.zeros((20, 20), dtype=np.int64)

    top_half = int(rng.integers(8, 21))
    bot_half = int(rng.integers(8, 21))

    top_cells = _rand_blob(rng, 10, 10, top_half)
    bot_cells = _rand_blob(rng, 10, 10, bot_half)

    # force overlap at least 3 pixels
    common = rng.choice(list(top_cells), size=3, replace=False)
    for r, c in common:
        bot_cells.add((r, c))

    for r, c in top_cells:
        g[r, c] = GREEN

    for r, c in bot_cells:
        g[10 + r, c] = BLUE

    return g
