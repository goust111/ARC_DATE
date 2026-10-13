import random
from collections import Counter

import numpy as np


def _infer_period(grid):
    h, w = grid.shape
    known = grid[(grid != 5) & (grid != 6)]
    colors = {int(x) for x in known if x != 0}
    for area in range(1, h * w + 1):
        for pr in range(1, h + 1):
            if area % pr:
                continue
            pc = area // pr
            if pc < 1 or pc > w:
                continue
            tile = [[None for _ in range(pc)] for _ in range(pr)]
            ok = True
            for r in range(h):
                for c in range(w):
                    val = int(grid[r, c])
                    if val in (5, 6):
                        continue
                    rr, cc = r % pr, c % pc
                    if tile[rr][cc] is None:
                        tile[rr][cc] = val
                    elif tile[rr][cc] != val:
                        ok = False
                        break
                if not ok:
                    break
            if ok and all(tile[r][c] is not None for r in range(pr) for c in range(pc)):
                if colors and len({tile[r][c] for r in range(pr) for c in range(pc)}) >= min(2, len(colors)):
                    return pr, pc, np.array(tile, dtype=int)
    raise ValueError("no consistent period")


def main(grid: np.ndarray) -> np.ndarray:
    """Repair a noisy periodic tile, then stamp the repaired tile at gray control cells.

    Input structure: A small rectangular periodic two-color pattern contains magenta
    missing cells and gray control cells. Gray and magenta are excluded when the
    period is inferred.
    Rule: Infer the smallest rectangular period from the remaining pattern cells,
    repair only the magenta cells, keep gray cells, and copy the repaired source
    grid into the large output block corresponding to each gray control cell.
    Key insight: Gray cells have two roles: they are ignored as pattern evidence
    but later act as the stamp placement mask.
    """
    grid = np.asarray(grid, dtype=int)
    h, w = grid.shape
    pr, pc, tile = _infer_period(grid)
    source = grid.copy()
    for r, c in zip(*np.where(source == 6)):
        source[r, c] = tile[r % pr, c % pc]

    out = np.zeros((h * h, w * w), dtype=int)
    for i in range(h):
        for j in range(w):
            if source[i, j] == 5:
                out[i * h:(i + 1) * h, j * w:(j + 1) * w] = source
    return out


def generate_input() -> np.ndarray:
    colors = random.sample([1, 2, 3, 4, 7, 8, 9], 2)
    for _ in range(200):
        h = random.choice([4, 5])
        w = random.choice([4, 5])
        pr = random.choice([1, 2])
        pc = random.choice([1, 2])
        if pr == pc == 1:
            pc = 2
        tile = np.zeros((pr, pc), dtype=int)
        for r in range(pr):
            for c in range(pc):
                tile[r, c] = colors[(r + c + random.randrange(2)) % 2]
        if len(set(tile.ravel())) < 2:
            continue
        grid = np.fromfunction(lambda r, c: tile[(r.astype(int) % pr), (c.astype(int) % pc)], (h, w), dtype=int)

        cells = [(r, c) for r in range(h) for c in range(w)]
        random.shuffle(cells)
        gray_count = random.randint(1, 3)
        gray = set(cells[:gray_count])
        candidates = cells[gray_count:]
        magenta_count = random.randint(1, min(4, len(candidates)))
        magenta = set(candidates[:magenta_count])
        trial = grid.copy()
        for r, c in gray:
            trial[r, c] = 5
        for r, c in magenta:
            trial[r, c] = 6

        ok = True
        for rr in range(pr):
            for cc in range(pc):
                seen = [trial[r, c] for r in range(h) for c in range(w)
                        if r % pr == rr and c % pc == cc and trial[r, c] not in (5, 6)]
                if not seen or len(set(int(x) for x in seen)) != 1:
                    ok = False
        if ok:
            return trial.astype(int)
    return np.array([[1, 3, 6, 5], [3, 1, 3, 1], [1, 3, 1, 3], [5, 1, 3, 6]], dtype=int)
