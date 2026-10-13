import numpy as np
from collections import deque


_BG = 0
_RED = 2
_BLUE = 1
_GREEN = 3
_GRAY = 5

_generate_calls = 0


def _bfs_region_count(grid):
    h, w = grid.shape
    vis = np.zeros((h, w), dtype=bool)
    cnt = 0
    for r in range(h):
        for c in range(w):
            if grid[r, c] != _BG or vis[r, c]:
                continue
            cnt += 1
            q = deque([(r, c)])
            vis[r, c] = True
            while q:
                x, y = q.popleft()
                for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < h and 0 <= ny < w and grid[nx, ny] == _BG and not vis[nx, ny]:
                        vis[nx, ny] = True
                        q.append((nx, ny))
    return cnt


def generate_input():
    global _generate_calls
    rng = np.random.RandomState(7000 + _generate_calls)
    _generate_calls += 1

    h = 30
    w = 30
    grid = np.zeros((h, w), dtype=int)
    sep = 14
    grid[sep, :] = _GRAY

    # top half barriers creating region count 2..5 (barriers are red vertical walls)
    region_goal = 2 + rng.randint(0, 4)
    num_walls = region_goal - 1
    barrier_cols = rng.choice(np.arange(1, w - 1), size=num_walls, replace=False)
    for cc in barrier_cols:
        grid[:sep, cc] = _RED

    # lower half two-color structure
    outer_t = (r := 18)
    outer_b = 21
    outer_l = 11
    outer_r = 16
    for c in range(outer_l, outer_r + 1):
        grid[outer_t, c] = _BLUE
        grid[outer_b, c] = _BLUE
    for r2 in range(outer_t + 1, outer_b):
        grid[r2, outer_l] = _BLUE
        grid[r2, outer_r] = _BLUE

    # green inner rectangle
    for r2 in range(outer_t + 1, outer_b):
        for c2 in range(outer_l + 1, outer_r):
            if outer_t + 1 < r2 < outer_b - 1 and outer_l + 1 < c2 < outer_r - 1:
                grid[r2, c2] = _GREEN

    return grid


def _draw_rect(out, r0, c0, r1, c1, color):
    h, w = out.shape
    r0 = max(0, r0)
    c0 = max(0, c0)
    r1 = min(h - 1, r1)
    c1 = min(w - 1, c1)
    if r0 > r1 or c0 > c1:
        return
    out[r0, c0:c1 + 1] = color
    out[r1, c0:c1 + 1] = color
    out[r0:r1 + 1, c0] = color
    out[r0:r1 + 1, c1] = color


def main(grid: np.ndarray) -> np.ndarray:
    """1) Count 0-background connected regions in upper half as a code N.
2) Keep lower blue frame+green core as fixed source geometry.
3) Expand outward from the blue frame with N alternating layers (odd green, even blue).
4) Erase top half and separator, return full-grid result with transformed lower-half structure only."""

    g = np.array(grid)
    h, w = g.shape
    sep_rows = np.where(np.any(g == _GRAY, axis=1))[0]
    sep = int(sep_rows[0]) if len(sep_rows) else h // 2

    top = g[:sep, :]
    n_regions = _bfs_region_count(top)

    # find outer blue frame bbox in lower half
    blue_cells = np.argwhere((g == _BLUE) & (np.arange(h)[:, None] >= sep))
    if len(blue_cells) == 0:
        return np.zeros_like(g)
    minr, minc = blue_cells.min(axis=0)
    maxr, maxc = blue_cells.max(axis=0)

    out = np.zeros_like(g)
    # copy lower existing structure unchanged
    out[sep:, :] = 0
    for r in range(minr, maxr + 1):
        for c in range(minc, maxc + 1):
            if g[r, c] == _BLUE:
                out[r, c] = _BLUE
            elif g[r, c] == _GREEN:
                out[r, c] = _GREEN

    for i in range(1, n_regions + 1):
        color = _GREEN if i % 2 == 1 else _BLUE
        _draw_rect(out, minr - i, minc - i, maxr + i, maxc + i, color)

    return out


if __name__ == "__main__":
    print(main(generate_input()).shape)
