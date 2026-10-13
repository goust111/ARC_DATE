import numpy as np
from collections import deque

_rng = np.random.default_rng(202608)


def _components(grid):
    h, w = grid.shape
    vis = np.zeros_like(grid, dtype=bool)
    comps = []
    for r in range(h):
        for c in range(w):
            if vis[r, c] or grid[r, c] == 0:
                continue
            col = int(grid[r, c])
            q = [(r, c)]
            vis[r, c] = True
            comp = []
            for cr, cc in q:
                comp.append((cr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not vis[nr, nc] and grid[nr, nc] == col:
                        vis[nr, nc] = True
                        q.append((nr, nc))
            comps.append((col, comp))
    return comps


def _has_enclosed_hole(grid, comp):
    h, w = grid.shape
    rs = [p[0] for p in comp]
    cs = [p[1] for p in comp]
    r0, r1 = min(rs), max(rs)
    c0, c1 = min(cs), max(cs)
    # 缩小 ROI 到 bbox
    box = np.ones((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    for rr, cc in comp:
        box[rr - r0, cc - c0] = False

    # 边界连通背景
    vis = np.zeros_like(box, dtype=bool)
    q = deque()
    rows, cols = box.shape
    for rr in range(rows):
        for cc in (0, cols - 1):
            if box[rr, cc]:
                box[rr, cc] = box[rr, cc] and False
    for cc in range(cols):
        for rr in (0, rows - 1):
            if box[rr, cc]:
                pass

    q = deque()
    for rr in range(rows):
        for cc in (0, cols - 1):
            if box[rr, cc] and not vis[rr, cc]:
                vis[rr, cc] = True
                q.append((rr, cc))
    for cc in range(cols):
        for rr in (0, rows - 1):
            if box[rr, cc] and not vis[rr, cc]:
                vis[rr, cc] = True
                q.append((rr, cc))

    while q:
        cr, cc = q.popleft()
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nr, nc = cr + dr, cc + dc
            if 0 <= nr < rows and 0 <= nc < cols and box[nr, nc] and not vis[nr, nc]:
                vis[nr, nc] = True
                q.append((nr, nc))

    return np.any(~vis & box)


def generate_input():
    """
    生成 10x8 网格，含 4~6 个 4-12 像素单色对象，闭合/开放对象比例一差一。
    多数类决定镜像方向，少数类被清除。
    """
    H, W = 10, 8
    n_closed = int(_rng.integers(2, 4))
    n_open = n_closed + 1
    total = n_closed + n_open

    while True:
        if total < 4 or total > 6:
            n_closed = int(_rng.integers(2, 4))
            n_open = n_closed + 1
            total = n_closed + n_open
        if total > 6:
            n_closed = 2
            n_open = 2
            total = 4

        grid = np.zeros((H, W), dtype=int)
        colors = [1, 2, 3, 4, 7]
        occupied = np.zeros_like(grid, dtype=bool)

        def place_shape(cells):
            for r, c in cells:
                grid[r, c] = color
                occupied[r, c] = True

        placed = 0

        # 放闭合对象（矩形环）
        for i in range(n_closed):
            success = False
            color = colors[i]
            for _ in range(120):
                h = int(_rng.integers(3, 5))
                w = int(_rng.integers(3, 5))
                if h > H - 2 or w > W - 2:
                    continue
                r0 = int(_rng.integers(1, H - h - 1))
                c0 = int(_rng.integers(1, W - w - 1))
                r1, c1 = r0 + h - 1, c0 + w - 1
                if occupied[r0 : r1 + 1, c0 : c1 + 1].any():
                    continue
                cells = set()
                for rr in range(r0, r1 + 1):
                    cells.add((rr, c0))
                    cells.add((rr, c1))
                for cc in range(c0, c1 + 1):
                    cells.add((r0, cc))
                    cells.add((r1, cc))
                if len(cells) < 4 or len(cells) > 12:
                    continue
                place_shape(cells)
                success = True
                placed += 1
                break
            if not success:
                continue

        # 放开放对象
        for i in range(n_open):
            color = colors[(n_closed + i) % len(colors)]
            for _ in range(120):
                sz = int(_rng.integers(4, 9))
                r = int(_rng.integers(0, H))
                c = int(_rng.integers(0, W))
                if occupied[r, c]:
                    continue
                cells = [(r, c)]
                for _ in range(sz - 1):
                    rr, cc = cells[int(_rng.integers(0, len(cells)))]
                    dr, dc = [ (1,0),(-1,0),(0,1),(0,-1) ][int(_rng.integers(0, 4))]
                    nr, nc = rr + dr, cc + dc
                    if not (0 <= nr < H and 0 <= nc < W):
                        continue
                    if occupied[nr, nc]:
                        continue
                    cells.append((nr, nc))
                    occupied[nr, nc] = True
                if 4 <= len(cells) <= 12:
                    for rr, cc in cells:
                        grid[rr, cc] = color
                    placed += 1
                    break

        # 若对象数不对则重生成
        objs = _components(grid)
        if not (4 <= len(objs) <= 6):
            continue
        return grid


def _mirror_concat(grid, axis):
    if axis == "h":
        return np.concatenate([grid, np.fliplr(grid)], axis=1)
    return np.concatenate([grid, np.flipud(grid)], axis=0)


def main(grid):
    """
    输入: 10x8 由多个对象组成的网格，含封闭对象(有孔)与开放对象。

    输出: 统计对象数量多数类决定镜像轴，擦除少数类，并做该轴方向拼接。

    算法: 1) 4 邻接提取对象并判定其是否存在被包围孔；
    2) N_closed 与 N_open 比较，多数类保留；
    3) 左右或上下镜像拼接保留对象。

    复杂度: O(H*W)。
    """
    g = np.array(grid, dtype=int)
    comps = _components(g)

    info = []
    for col, comp in comps:
        rs = [p[0] for p in comp]
        cs = [p[1] for p in comp]
        r0, r1 = min(rs), max(rs)
        c0, c1 = min(cs), max(cs)
        has_hole = False
        if r0 < r1 and c0 < c1:
            sub = np.ones((r1 - r0 + 1, c1 - c0 + 1), dtype=int)
            for rr, cc in comp:
                sub[rr - r0, cc - c0] = 0
            from collections import deque
            vis = np.zeros_like(sub, dtype=bool)
            q = deque()
            rows, cols = sub.shape
            for rr in range(rows):
                for cc in (0, cols - 1):
                    if sub[rr, cc] == 1 and not vis[rr, cc]:
                        vis[rr, cc] = True
                        q.append((rr, cc))
            for cc in range(cols):
                for rr in (0, rows - 1):
                    if sub[rr, cc] == 1 and not vis[rr, cc]:
                        vis[rr, cc] = True
                        q.append((rr, cc))

            while q:
                cr, cc = q.popleft()
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < rows and 0 <= nc < cols and sub[nr, nc] == 1 and not vis[nr, nc]:
                        vis[nr, nc] = True
                        q.append((nr, nc))

            has_hole = np.any(~vis)
        info.append((col, comp, has_hole))

    closed_cnt = sum(1 for _, _, h in info if h)
    open_cnt = len(info) - closed_cnt
    if closed_cnt > open_cnt:
        keep = {c for _, comp, h in info for c in [0] if h}
        # keep objects list
        keep_cells = set()
        for _, comp, h in info:
            if h:
                for p in comp:
                    keep_cells.add(p)
        out = np.zeros_like(g)
        for r, c in keep_cells:
            out[r, c] = g[r, c]
        return _mirror_concat(out, "h")
    elif open_cnt > closed_cnt:
        keep_cells = set()
        for _, comp, h in info:
            if not h:
                for p in comp:
                    keep_cells.add(p)
        out = np.zeros_like(g)
        for r, c in keep_cells:
            out[r, c] = g[r, c]
        return _mirror_concat(out, "v")
    return g
