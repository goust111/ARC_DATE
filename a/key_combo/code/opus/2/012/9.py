import numpy as np

_rng = np.random.default_rng(202609)


def _components_of_color(grid, color):
    h, w = grid.shape
    vis = np.zeros_like(grid, dtype=bool)
    comps = []
    for r in range(h):
        for c in range(w):
            if vis[r, c] or grid[r, c] != color:
                continue
            q = [(r, c)]
            vis[r, c] = True
            comp = []
            for cr, cc in q:
                comp.append((cr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not vis[nr, nc] and grid[nr, nc] == color:
                        vis[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _drop_region(region, color):
    h, w = region.shape
    comps = _components_of_color(region, color)
    comps.sort(key=lambda comp: max(r for r, _ in comp), reverse=True)

    out = np.zeros_like(region)
    occ = np.zeros_like(region, dtype=bool)

    for comp in comps:
        maxr = max(r for r, _ in comp)
        shift = h - 1 - maxr
        while shift > 0:
            blocked = False
            for r, c in comp:
                nr = r + shift
                if nr >= h:
                    blocked = True
                    break
                if occ[nr, c] and (nr, c) not in comp:
                    blocked = True
                    break
            if not blocked:
                break
            shift -= 1

        for r, c in comp:
            out[r + shift, c] = color
            occ[r + shift, c] = True

    return out


def _random_object(h, w, size, color, occupancy):
    for _ in range(80):
        start_r = int(_rng.integers(0, h))
        start_c = int(_rng.integers(0, w))
        if occupancy[start_r, start_c] != 0:
            continue

        cells = [(start_r, start_c)]
        for _ in range(max(1, size - 1)):
            rr, cc = cells[int(_rng.integers(0, len(cells)))]
            dr, dc = [(1, 0), (-1, 0), (0, 1), (0, -1)][int(_rng.integers(0, 4))]
            nr, nc = rr + dr, cc + dc
            if not (0 <= nr < h and 0 <= nc < w):
                continue
            if occupancy[nr, nc] != 0 or (nr, nc) in cells:
                continue
            cells.append((nr, nc))
        if len(cells) >= size:
            cells = cells[:size]
            for r, c in cells:
                occupancy[r, c] = color
            return cells
    return []


def generate_input():
    """
    生成 11x7 网格，带分隔线与上下两组对象。

    上半区红、下半区蓝对象在各自区域随机生成后用于重力沉降交集任务。
    """
    H, W = 11, 7
    mid = H // 2

    while True:
        up = np.zeros((mid, W), dtype=int)
        down = np.zeros((H - mid - 1, W), dtype=int)

        upper_objs = int(_rng.integers(2, 4))
        lower_objs = int(_rng.integers(2, 4))

        for _ in range(upper_objs):
            size = int(_rng.integers(2, 6))
            if not _random_object(mid, W, size, 2, up):
                break

        for _ in range(lower_objs):
            size = int(_rng.integers(2, 6))
            if not _random_object(H - mid - 1, W, size, 1, down):
                break

        if not np.any(up) or not np.any(down):
            continue

        if not _components_of_color(up, 2) or not _components_of_color(down, 1):
            continue

        g = np.zeros((H, W), dtype=int)
        g[mid, :] = 5
        g[:mid, :] = up
        g[mid + 1 :, :] = down
        return g


def main(grid):
    """
    输入: 11x7 网格，奇数行灰分隔，红在上蓝在下。

    输出: 上下两区对象分别下坠堆叠后逐像素交集，交叠处为 3。

    算法: 1) 分离上/下区；2) 按各自颜色向下滑动到边界并堆叠；
    3) 对齐后交集置 3。

    复杂度: O(H*W)。
    """
    g = np.array(grid, dtype=int)
    h, w = g.shape
    mid = h // 2

    up = g[:mid, :]
    down = g[mid + 1 :, :]

    up_f = _drop_region(up, 2)
    down_f = _drop_region(down, 1)

    out = np.zeros_like(up_f)
    for r in range(out.shape[0]):
        for c in range(out.shape[1]):
            if up_f[r, c] == 2 and down_f[r, c] == 1:
                out[r, c] = 3

    return out
