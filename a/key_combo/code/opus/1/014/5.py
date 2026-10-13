import random
from collections import deque
import numpy as np


def _components(grid, color):
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or grid[r, c] != color:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            comp = []
            while q:
                x, y = q.popleft()
                comp.append((x, y))
                for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < h and 0 <= ny < w and not seen[nx, ny] and grid[nx, ny] == color:
                        seen[nx, ny] = True
                        q.append((nx, ny))
            comps.append(comp)
    return comps


def _bbox(coords):
    rs = [r for r, _ in coords]
    cs = [c for _, c in coords]
    return min(rs), min(cs), max(rs), max(cs)


def main(grid: np.ndarray) -> np.ndarray:
    out = np.array(grid, copy=True)
    grays = _components(grid, 5)
    if not grays:
        return out
    occ = max(grays, key=len)
    or0, oc0, or1, oc1 = _bbox(occ)
    mid = grid.shape[1] // 2
    left_pts = [(r, c) for r, c in zip(*np.nonzero(grid[:, :mid]))]
    if not left_pts:
        return out
    sr0, sc0, sr1, sc1 = _bbox(left_pts)
    axis = (sc0 + sc1) / 2
    for r, c in occ:
        mc = int(round(2 * axis - c))
        if 0 <= mc < grid.shape[1]:
            out[r, c] = grid[r, mc]
    sprite = np.array(out[sr0:sr1 + 1, sc0:sc1 + 1], copy=True)
    hidden = []
    for r in range(or0, or1 + 1):
        for c in range(oc0, oc1 + 1):
            if out[r, c] in (2, 3):
                hidden.append((r - sr0, c - sc0, int(out[r, c])))
    if hidden:
        dr, dc, color = hidden[0]
    else:
        pts = np.argwhere((sprite == 2) | (sprite == 3))
        if pts.size == 0:
            return out
        dr, dc = map(int, pts[0])
        color = int(sprite[dr, dc])
    stamp = np.fliplr(sprite) if color == 2 else sprite
    if color == 2:
        dc = stamp.shape[1] - 1 - dc

    for r, c in map(tuple, np.argwhere(grid[:, mid:] == color)):
        tr, tc = int(r), int(c + mid)
        if sr0 <= tr <= sr1 and sc0 <= tc <= sc1:
            continue
        top, left = tr - dr, tc - dc
        for rr in range(stamp.shape[0]):
            for cc in range(stamp.shape[1]):
                if stamp[rr, cc] == 0:
                    continue
                nr, nc = top + rr, left + cc
                if 0 <= nr < out.shape[0] and 0 <= nc < out.shape[1] and (out[nr, nc] == 0 or (nr, nc) == (tr, tc)):
                    out[nr, nc] = stamp[rr, cc]
    return out


def generate_input() -> np.ndarray:
    h = w = 20
    g = np.zeros((h, w), dtype=int)
    sr0, sc0, size = 2, 1, 9
    base = np.zeros((size, size), dtype=int)
    for r in range(size):
        for c in range(size // 2 + 1):
            if random.random() < 0.35 or r in (1, size - 2) or c == 1:
                val = random.choice([1, 4])
                base[r, c] = val
                base[r, size - 1 - c] = val
    color = random.choice([2, 3])
    base[5, 7] = color
    base[5, 1] = color
    g[sr0:sr0 + size, sc0:sc0 + size] = base
    g[sr0 + 4:sr0 + 7, sc0 + 2:sc0 + 5] = 5
    targets = [(7, 15), (11, 14), (5, 17)]
    for r, c in random.sample(targets, random.randint(1, 3)):
        g[r, c] = color
    other = 3 if color == 2 else 2
    for r, c in random.sample([(3, 15), (13, 17), (15, 12)], random.randint(1, 2)):
        g[r, c] = other
    return g
