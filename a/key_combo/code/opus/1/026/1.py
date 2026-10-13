import random
import numpy as np


def main(grid: np.ndarray) -> np.ndarray:
    g = np.array(grid, copy=True)
    h, w = g.shape
    gray = grid == 5
    hits = {}

    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if color in (0, 5):
                continue
            if r not in (0, h - 1) and c not in (0, w - 1):
                continue
            if r == 0:
                dr, dc = 1, 0
            elif r == h - 1:
                dr, dc = -1, 0
            elif c == 0:
                dr, dc = 0, 1
            else:
                dr, dc = 0, -1

            rr, cc = r + dr, c + dc
            dist = 1
            while 0 <= rr < h and 0 <= cc < w:
                if gray[rr, cc]:
                    old = hits.get((rr, cc))
                    if old is None or dist < old[0]:
                        hits[(rr, cc)] = (dist, color)
                    break
                rr += dr
                cc += dc
                dist += 1

    for (r, c), (_, color) in hits.items():
        g[r, c] = color
    return g


def _connected_shape(n: int) -> set[tuple[int, int]]:
    cells = {(0, 0)}
    while len(cells) < n:
        r, c = random.choice(tuple(cells))
        dr, dc = random.choice(((1, 0), (-1, 0), (0, 1), (0, -1)))
        cells.add((r + dr, c + dc))
    min_r = min(r for r, _ in cells)
    min_c = min(c for _, c in cells)
    return {(r - min_r, c - min_c) for r, c in cells}


def _ray_hits(grid: np.ndarray) -> list[tuple[tuple[int, int], int]]:
    h, w = grid.shape
    ans = []
    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if color in (0, 5):
                continue
            if r == 0:
                dr, dc = 1, 0
            elif r == h - 1:
                dr, dc = -1, 0
            elif c == 0:
                dr, dc = 0, 1
            elif c == w - 1:
                dr, dc = 0, -1
            else:
                continue
            rr, cc = r + dr, c + dc
            dist = 1
            while 0 <= rr < h and 0 <= cc < w:
                if grid[rr, cc] == 5:
                    ans.append(((rr, cc), dist))
                    break
                rr += dr
                cc += dc
                dist += 1
    return ans


def generate_input() -> np.ndarray:
    colors = [1, 2, 3, 4, 7]
    for _ in range(500):
        h = random.randint(15, 18)
        w = random.randint(15, 18)
        grid = np.zeros((h, w), dtype=int)
        shape = _connected_shape(random.randint(8, 15))
        sh = max(r for r, _ in shape) + 1
        sw = max(c for _, c in shape) + 1
        if sh > h - 8 or sw > w - 8:
            continue
        top = random.randint(4, h - sh - 4)
        left = random.randint(4, w - sw - 4)
        cells = {(top + r, left + c) for r, c in shape}
        for r, c in cells:
            grid[r, c] = 5

        by_col = {}
        by_row = {}
        for r, c in cells:
            by_col.setdefault(c, []).append(r)
            by_row.setdefault(r, []).append(c)
        if len(by_col) < 2 or len(by_row) < 2:
            continue

        markers = []
        for side in ("top", "bottom"):
            cols = list(by_col)
            random.shuffle(cols)
            for c in cols[: random.randint(2, min(4, len(cols)))]:
                r = 0 if side == "top" else h - 1
                markers.append((r, c, random.choice(colors)))
        for side in ("left", "right"):
            rows = list(by_row)
            random.shuffle(rows)
            for r in rows[: random.randint(2, min(4, len(rows)))]:
                c = 0 if side == "left" else w - 1
                markers.append((r, c, random.choice(colors)))

        if len({(r, c) for r, c, _ in markers}) != len(markers):
            continue
        for r, c, color in markers:
            grid[r, c] = color

        seen: dict[tuple[int, int], set[int]] = {}
        ok = True
        for hit, dist in _ray_hits(grid):
            seen.setdefault(hit, set())
            if dist in seen[hit]:
                ok = False
                break
            seen[hit].add(dist)
        if ok and any(len(v) > 1 for v in seen.values()):
            return grid
    return grid
