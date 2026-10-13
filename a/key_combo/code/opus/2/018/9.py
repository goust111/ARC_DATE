import numpy as np
from collections import deque


_BG = 0
_BLUE = 1
_RED = 2
_GREEN = 3
_YELLOW = 4
_MARK = 6
_GRAY = 5

_generate_calls = 0


def _bfs_components(grid, color):
    h, w = grid.shape
    vis = np.zeros((h, w), dtype=bool)
    comps = []
    for r in range(h):
        for c in range(w):
            if grid[r, c] != color or vis[r, c]:
                continue
            q = deque([(r, c)])
            vis[r, c] = True
            comp = []
            while q:
                x, y = q.popleft()
                comp.append((x, y))
                for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < h and 0 <= ny < w and not vis[nx, ny] and grid[nx, ny] == color:
                        vis[nx, ny] = True
                        q.append((nx, ny))
            comps.append(comp)
    return comps


def _enclosed_region_count(cell):
    h, w = cell.shape
    loop = cell == _BLUE
    out = np.zeros((h, w), dtype=bool)
    q = deque()
    for r in range(h):
        for c in range(w):
            if r in (0, h - 1) or c in (0, w - 1):
                if not loop[r, c]:
                    out[r, c] = True
                    q.append((r, c))
    while q:
        r, c = q.popleft()
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w and not loop[nr, nc] and not out[nr, nc]:
                out[nr, nc] = True
                q.append((nr, nc))
    inside = np.logical_not(np.logical_or(loop, out))
    vis = np.zeros((h, w), dtype=bool)
    cnt = 0
    for r in range(h):
        for c in range(w):
            if not inside[r, c] or vis[r, c]:
                continue
            cnt += 1
            q = deque([(r, c)])
            vis[r, c] = True
            while q:
                x, y = q.popleft()
                for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < h and 0 <= ny < w and inside[nx, ny] and not vis[nx, ny]:
                        vis[nx, ny] = True
                        q.append((nx, ny))
    return cnt


def generate_input():
    global _generate_calls
    rng = np.random.RandomState(11000 + _generate_calls)
    _generate_calls += 1

    unit = 7
    R = 3
    C = 3
    h = R * unit + (R - 1)
    w = C * unit + (C - 1)
    grid = np.zeros((h, w), dtype=int)

    for r in range(R - 1):
        grid[(r + 1) * unit + r, :] = _GRAY
    for c in range(C - 1):
        grid[:, (c + 1) * unit + c] = _GRAY

    # templates in top-left cell
    grid[0, 0] = _RED
    grid[1, 1] = _GREEN
    grid[2, 2] = _YELLOW
    grid[1, 2] = _GREEN

    # 8 target cells
    for rr in range(R):
        for cc in range(C):
            if rr == 0 and cc == 0:
                continue
            top = rr * (unit + 1)
            left = cc * (unit + 1)
            # rectangular loop
            for x in range(left + 1, left + unit - 1):
                grid[top + 1, x] = _BLUE
                grid[top + unit - 2, x] = _BLUE
            for y in range(top + 1, top + unit - 1):
                grid[y, left + 1] = _BLUE
                grid[y, left + unit - 2] = _BLUE

            # optional internal notch
            if rng.random() < 0.6:
                grid[top + 3, left + 3] = 0

            if rng.random() < 0.65:
                grid[top + 2, left + 2] = _MARK

    return grid


def main(grid: np.ndarray) -> np.ndarray:
    """1) Read three templates from the top-left cell by pixel counts.
2) For each marked target cell, count enclosed background regions inside its blue loop.
3) Select template with area K and stamp at the cell center.
4) Keep markers and structural loops, do not mutate background outside stamp targets."""

    g = np.array(grid)
    h, w = g.shape
    unit = 7

    # template area map
    templates = {}
    for color in (_RED, _GREEN, _YELLOW):
        cells = np.argwhere(g[:unit, :unit] == color)
        if len(cells) == 0:
            continue
        templates[len(cells)] = color
        # record local pattern bbox
    out = g.copy()

    for rr in range(3):
        for cc in range(3):
            if rr == 0 and cc == 0:
                continue
            top = rr * (unit + 1)
            left = cc * (unit + 1)
            sub = g[top : top + unit, left : left + unit]
            if not np.any(sub == _MARK):
                continue
            blue = _bfs_components(sub, _BLUE)
            if not blue:
                continue
            cell = blue[0]
            loop_mask = np.zeros((unit, unit), dtype=bool)
            for r0, c0 in cell:
                loop_mask[r0, c0] = True
            k = _enclosed_region_count(sub)
            if k not in templates:
                continue
            color = templates[k]
            tr = top + (unit // 2)
            tc = left + (unit // 2)
            pr = 0
            pc = 0
            if color == _RED:
                pr, pc = 1, 0
            elif color == _GREEN:
                pr, pc = 0, 1
            else:
                pr, pc = 1, 1
            out[tr + pr - 1, tc + pc - 1] = color

    return out


if __name__ == "__main__":
    print(main(generate_input()).shape)
