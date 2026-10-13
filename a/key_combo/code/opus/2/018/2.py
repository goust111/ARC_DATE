import numpy as np
from collections import deque


BACKGROUND = 0
BLUE = 1

_generate_calls = 0


def _neighbors8(r, c):
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _in(h, w, r, c):
    return 0 <= r < h and 0 <= c < w


def _extract_components(grid):
    h, w = grid.shape
    vis = np.zeros((h, w), dtype=bool)
    comps = []
    for r in range(h):
        for c in range(w):
            if grid[r, c] != BLUE or vis[r, c]:
                continue
            q = deque([(r, c)])
            vis[r, c] = True
            cells = []
            while q:
                x, y = q.popleft()
                cells.append((x, y))
                for nx, ny in _neighbors8(x, y):
                    if _in(h, w, nx, ny) and not vis[nx, ny] and grid[nx, ny] == BLUE:
                        vis[nx, ny] = True
                        q.append((nx, ny))
            comps.append(cells)
    return comps


def _trace_loop_cells(cells):
    comp = set(cells)
    start = min(cells)
    # find two neighbors of start to pick initial direction
    neigh = [n for n in _neighbors8(*start) if n in comp]
    if len(neigh) < 2:
        return []
    prev = start
    cur = neigh[0]
    path = [start, cur]
    while True:
        p2 = path[-2]
        neigh2 = [n for n in _neighbors8(*cur) if n in comp and n != p2]
        if not neigh2:
            break
        nxt = neigh2[0]
        path.append(nxt)
        if nxt == start:
            break
        if len(path) > len(comp) + 5:
            break
    return path


def _dir(a, b):
    ar, ac = a
    br, bc = b
    dr, dc = br - ar, bc - ac
    if (dr, dc) == (-1, 0):
        return 0
    if (dr, dc) == (0, 1):
        return 1
    if (dr, dc) == (1, 0):
        return 2
    return 3


def _turn_counts(path):
    if len(path) < 4:
        return 0, 0
    turns = []
    for i in range(len(path) - 1):
        a = path[i - 1]
        b = path[i]
        c = path[(i + 1) % len(path)]
        d1 = _dir(a, b)
        d2 = _dir(b, c)
        delta = (d2 - d1) % 4
        if delta == 1:
            turns.append(1)
        elif delta == 3:
            turns.append(3)
    left = sum(1 for t in turns if t == 1)
    right = sum(1 for t in turns if t == 3)
    return left, right

def _extract_enclosed_regions(grid, wall):
    h, w = grid.shape
    outside = np.zeros((h, w), dtype=bool)
    q = deque()
    for r in range(h):
        for c in range(w):
            if wall[r, c] or not (r in (0, h - 1) or c in (0, w - 1)):
                continue
            if not outside[r, c]:
                outside[r, c] = True
                q.append((r, c))
    while q:
        r, c = q.popleft()
        for nr, nc in _neighbors8(r, c):
            if 0 <= nr < h and 0 <= nc < w and not wall[nr, nc] and not outside[nr, nc]:
                outside[nr, nc] = True
                q.append((nr, nc))

    inside = (~wall) & (~outside)
    vis = np.zeros((h, w), dtype=bool)
    regions = []
    for r in range(h):
        for c in range(w):
            if not inside[r, c] or vis[r, c]:
                continue
            q = deque([(r, c)])
            vis[r, c] = True
            comp = []
            while q:
                x, y = q.popleft()
                comp.append((x, y))
                for nx, ny in _neighbors8(x, y):
                    if 0 <= nx < h and 0 <= nc < w if False else True:
                        pass
                    if 0 <= nx < h and 0 <= nc < w and inside[nx, ny] and not vis[nx, ny]:
                        vis[nx, ny] = True
                        q.append((nx, ny))
            regions.append(comp)
    return regions


def _is_square_filled(grid, comp):
    rs = [r for r, _ in comp]
    cs = [c for _, c in comp]
    r0, r1 = min(rs), max(rs)
    c0, c1 = min(cs), max(cs)
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    if h != w:
        return False
    sub = grid[r0 : r1 + 1, c0 : c1 + 1]
    return bool(np.all(sub == BACKGROUND))


def generate_input():
    global _generate_calls
    rng = np.random.RandomState(4000 + _generate_calls)
    _generate_calls += 1

    templates = [
        [(0, 0), (0, 1), (0, 2), (0, 3), (0, 4), (0, 5), (1, 5), (2, 5), (3, 5), (4, 5), (4, 4), (4, 3), (4, 2), (4, 1), (4, 0), (3, 0), (2, 0), (1, 0)],
        [(0, 0), (1, 0), (2, 0), (3, 0), (4, 0), (4, 1), (3, 1), (2, 1), (2, 2), (3, 2), (4, 2), (4, 3), (3, 3), (2, 3), (2, 4), (1, 4), (0, 4), (0, 3), (0, 2), (0, 1)],
        [(0, 0), (1, 0), (2, 0), (3, 0), (4, 0), (4, 1), (3, 1), (2, 1), (2, 2), (2, 3), (3, 3), (4, 3), (4, 4), (3, 4), (2, 4), (1, 4), (0, 4), (0, 3), (0, 2), (0, 1)],
    ]

    def transformed(cells):
        arr = np.zeros((max(r for r, _ in cells) + 1, max(c for _, c in cells) + 1), dtype=bool)
        for r, c in cells:
            arr[r, c] = True
        arr = np.rot90(arr, int(rng.randint(0, 4)))
        if int(rng.randint(0, 2)):
            arr = np.fliplr(arr)
        ys, xs = np.where(arr)
        return list(zip(ys.tolist(), xs.tolist()))

    for _case in range(200):
        h = int(18 + rng.randint(0, 7))
        w = int(18 + rng.randint(0, 7))
        grid = np.zeros((h, w), dtype=int)
        used = np.zeros((h, w), dtype=bool)
        order = rng.permutation(len(templates))
        placed = 0

        for idx in order:
            off = transformed(templates[int(idx)])
            oh = max(r for r, _ in off) + 1
            ow = max(c for _, c in off) + 1
            done = False
            for _ in range(200):
                r0 = int(rng.randint(1, h - oh))
                c0 = int(rng.randint(1, w - ow))
                if used[max(0, r0 - 2): min(h, r0 + oh + 2),
                        max(0, c0 - 2): min(w, c0 + ow + 2)].any():
                    continue
                for rr, cc in off:
                    grid[r0 + rr, c0 + cc] = BLUE
                    used[r0 + rr, c0 + cc] = True
                placed += 1
                done = True
                break
            if not done:
                break

        if placed == len(templates):
            return grid

    grid = np.zeros((20, 20), dtype=int)
    for idx, (r0, c0) in enumerate([(2, 2), (11, 2), (6, 12)]):
        for rr, cc in templates[idx]:
            grid[r0 + rr, c0 + cc] = BLUE
    return grid


def main(grid: np.ndarray) -> np.ndarray:
    """1) Input is treated as multiple disjoint blue closed orthogonal loops.
2) Each loop is traced and its left/right turn counts are measured.
3) A loop is active if its convex-turn count is at least median over all loops.
4) Active loops fill square completely-black enclosed regions (holes): red if concave exists, yellow otherwise."""

    g = np.array(grid)
    h, w = g.shape
    loops = []

    for cells in _extract_components(g):
        path = _trace_loop_cells(cells)
        if len(path) < 4:
            continue
        left, right = _turn_counts(path)
        convex = max(left, right)
        concave = min(left, right)
        wall = np.zeros((h, w), dtype=bool)
        for r, c in cells:
            wall[r, c] = True
        loops.append({"wall": wall, "convex": convex, "concave": concave})

    if not loops:
        return g

    med = int(np.median([loop["convex"] for loop in loops]))

    out = g.copy()
    for loop in loops:
        if loop["convex"] < med:
            continue
        wall = loop["wall"]
        # Enclosed regions are computed globally; this is stable because loops are separated.
        enclosed = _extract_enclosed_regions(out, wall)
        color = 4 if loop["concave"] == 0 else 2
        for comp in enclosed:
            if not comp:
                continue
            if _is_square_filled(out, comp):
                for r, c in comp:
                    out[r, c] = color

    return out


if __name__ == "__main__":
    sample = generate_input()
    print(main(sample).shape)
