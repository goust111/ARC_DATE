import numpy as np
from collections import deque

BG = 0


def _components(grid: np.ndarray, mask: np.ndarray):
    h, w = grid.shape
    vis = np.zeros((h, w), dtype=bool)
    comps = []
    for r in range(h):
        for c in range(w):
            if mask[r, c] and not vis[r, c]:
                color = int(grid[r, c])
                q = deque([(r, c)])
                vis[r, c] = True
                pts = []
                while q:
                    rr, cc = q.popleft()
                    pts.append((rr, cc))
                    for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                        nr, nc = rr + dr, cc + dc
                        if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not vis[nr, nc]:
                            vis[nr, nc] = True
                            q.append((nr, nc))
                comps.append((color, np.asarray(pts, dtype=np.int32)))
    return comps


def _bbox(pts: np.ndarray):
    r0, c0 = pts[:, 0].min(), pts[:, 1].min()
    r1, c1 = pts[:, 0].max(), pts[:, 1].max()
    return int(r0), int(c0), int(r1), int(c1)


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid)
    h, w = g.shape
    out = g.copy()

    # Parse legend in first two rows: left color -> right color.
    legend = {}
    for r in range(2):
        c = 0
        while c + 1 < w:
            a, b = int(g[r, c]), int(g[r, c + 1])
            if a != BG and b != BG:
                legend[a] = b
                c += 2
            else:
                c += 1

    main_area = g[2:, :]
    mask_main = main_area != BG
    comps = _components(main_area, mask_main)
    if not comps:
        return out

    # Map component colors to color key for deterministic ordering.
    comps = [(c, p) for c, p in comps if c in legend]
    comps.sort(key=lambda x: (x[1][:, 0].min(), x[1][:, 1].min()))

    for color, pts in comps:
        fill_color = legend.get(color, BG)
        if fill_color == BG:
            continue

        r0, c0 = pts[:, 0].min(), pts[:, 1].min()
        r1, c1 = pts[:, 0].max(), pts[:, 1].max()
        pts_local = pts.copy()
        pts_local[:, 0] -= r0
        pts_local[:, 1] -= c0

        axis_c = int(pts[:, 1].max())
        # Mirror half shape across its rightmost column (axis).
        full = set((int(r), int(c)) for r, c in pts)
        for rr, cc in pts:
            mc = 2 * axis_c - int(cc)
            if 0 <= mc < g.shape[1]:
                full.add((int(rr), int(mc)))

        # Fill spans between first and last contour pixel on each row.
        for rr in range(r0, r1 + 1):
            row_cols = [c for r, c in full if r == rr]
            if len(row_cols) < 2:
                continue
            left = min(row_cols)
            right = max(row_cols)
            for cc in range(left + 1, right):
                out[rr, cc] = fill_color

        # Erase contour pixels.
        for rr, cc in full:
            out[rr, cc] = BG

    return out


def _random_half_outline(rng: np.random.Generator, h: int, w: int):
    # Width includes axis column.
    axis_w = int(rng.integers(3, 6))
    hh = int(rng.integers(4, 8))
    half = np.zeros((hh, axis_w), dtype=np.int8)
    axis = axis_w - 1
    for r in range(hh):
        left = int(rng.integers(0, axis + 1))
        for c in range(left, axis_w):
            if c >= left:
                half[r, c] = 1
    # Ensure top and bottom continuous horizontal segment by forcing full row until axis.
    if hh >= 2:
        half[0, :] = 1
        half[hh - 1, :] = 1
    return half


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    h = 12
    w = 18
    grid = np.zeros((h, w), dtype=np.int64)

    colors = [1, 2, 3, 7]
    rng.shuffle(colors)

    # Legend in first two rows.
    c = 0
    for color in colors[:3]:
        fill = int(rng.integers(4, 10))
        while fill in colors[:3]:
            fill = int(rng.integers(4, 10))
        grid[0, c] = color
        grid[0, c + 1] = fill
        c += 3

    occupied = np.zeros_like(grid, dtype=bool)
    occupied[:2, :] = True

    n = 2
    for i in range(n):
        color = colors[i]
        half = _random_half_outline(rng, h, w)
        hh, ww = half.shape
        for _ in range(1000):
            r0 = int(rng.integers(2, h - hh))
            c0 = int(rng.integers(0, w - (2 * ww - 1)))
            if occupied[r0 : r0 + hh, c0 : c0 + (2 * ww - 1)].any():
                continue
            rr, cc = np.where(half == 1)
            coords = [(r0 + int(x), c0 + int(y)) for x, y in zip(rr, cc)]
            for x, y in coords:
                grid[x, y] = color
                occupied[x, y] = True
            break

    return grid
