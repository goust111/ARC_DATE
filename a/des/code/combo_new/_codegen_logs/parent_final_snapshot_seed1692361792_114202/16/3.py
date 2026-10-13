import random
from collections import deque
import numpy as np

BLACK = 0
MAROON = 9


def components(mask: np.ndarray):
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps = []
    for r in range(h):
        for c in range(w):
            if mask[r, c] and not seen[r, c]:
                q = deque([(r, c)])
                seen[r, c] = True
                cells = []
                while q:
                    x, y = q.popleft()
                    cells.append((x, y))
                    for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                        nx, ny = x + dx, y + dy
                        if 0 <= nx < h and 0 <= ny < w and mask[nx, ny] and not seen[nx, ny]:
                            seen[nx, ny] = True
                            q.append((nx, ny))
                comps.append(cells)
    return comps


def bbox(cells):
    rs = [r for r, _ in cells]
    cs = [c for _, c in cells]
    return min(rs), max(rs), min(cs), max(cs)


def side_segments(arr):
    out = {}
    i = 0
    while i < len(arr):
        if arr[i] != BLACK:
            j = i
            while j < len(arr) and arr[j] == arr[i]:
                j += 1
            out[arr[i]] = ((i + j - 1) / 2)
            i = j
        else:
            i += 1
    return out


def main(grid: np.ndarray) -> np.ndarray:
    grid = np.array(grid, dtype=int)
    out = grid.copy()
    top_axes = side_segments(grid[0, 1:])
    left_axes = side_segments(grid[1:, 0])
    rects = []
    mask = (grid != BLACK)
    mask[0, :] = False
    mask[:, 0] = False
    for cells in components(mask):
        color = grid[cells[0]]
        r0, r1, c0, c1 = bbox(cells)
        if np.all(grid[r0 : r1 + 1, c0 : c1 + 1] == color):
            rects.append((-(r1 - r0 + 1) * (c1 - c0 + 1), r0, color, r0, r1, c0, c1))
    rects.sort()
    for _, _, color, r0, r1, c0, c1 in rects:
        if color not in top_axes or color not in left_axes:
            continue
        ax_c = top_axes[color] + 1
        ax_r = left_axes[color] + 1
        hits = {}
        for r in range(r0, r1 + 1):
            for c in range(c0, c1 + 1):
                for vr in (0, 1):
                    for vc in (0, 1):
                        rr = int(round(2 * ax_r - r)) if vr else r
                        cc = int(round(2 * ax_c - c)) if vc else c
                        if 0 <= rr < grid.shape[0] and 0 <= cc < grid.shape[1]:
                            hits[(rr, cc)] = hits.get((rr, cc), 0) + 1
        for (rr, cc), cnt in hits.items():
            out[rr, cc] = MAROON if cnt > 1 else color
    out[0, :] = grid[0, :]
    out[:, 0] = grid[:, 0]
    return out


def generate_input() -> np.ndarray:
    h, w = random.randint(16, 22), random.randint(16, 22)
    grid = np.zeros((h, w), dtype=int)
    shared = random.sample([1, 2, 3, 4, 6, 7, 8], random.randint(2, 4))
    pos = 1
    for color in shared:
        seg = random.randint(2, 4)
        if pos + seg >= w:
            break
        grid[0, pos : pos + seg] = color
        pos += seg + 1
    pos = 1
    for color in shared:
        seg = random.randint(2, 4)
        if pos + seg >= h:
            break
        grid[pos : pos + seg, 0] = color
        pos += seg + 1
    for color in shared:
        rh, rw = random.randint(2, 4), random.randint(2, 4)
        r0 = random.randint(2, h - rh - 1)
        c0 = random.randint(2, w - rw - 1)
        if np.any(grid[r0 - 1 : r0 + rh + 1, c0 - 1 : c0 + rw + 1] != BLACK):
            continue
        grid[r0 : r0 + rh, c0 : c0 + rw] = color
    return grid
