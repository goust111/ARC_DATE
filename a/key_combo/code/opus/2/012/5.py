import numpy as np

_rng = np.random.default_rng(202605)


def _simulate_ray(h, w, start, dr, dc):
    path = set()
    r, c = start
    for _ in range(4 * max(h, w) * 4):
        nr = r + dr
        nc = c + dc
        if not (0 <= nr < h):
            dr *= -1
            nr = r + dr
        if not (0 <= nc < w):
            dc *= -1
            nc = c + dc

        r, c = nr, nc
        path.add((r, c))
        if (r, c) == start:
            break
    return path


def _is_collinear(p0, p1, p2):
    return (p1[0] - p0[0]) == (p2[0] - p1[0]) and (p1[1] - p0[1]) == (p2[1] - p1[1])


def _parse_seed(grid):
    pts = np.argwhere(grid == 6)
    pts = [tuple(map(int, p)) for p in pts]
    if len(pts) not in (2, 3):
        return None

    center = ((grid.shape[0] - 1) / 2.0, (grid.shape[1] - 1) / 2.0)

    if len(pts) == 2:
        a, b = sorted(pts)
        dr = b[0] - a[0]
        dc = b[1] - a[1]
        if abs(dr) != 1 or abs(dc) != 1:
            return None
        da = (a[0] - center[0]) ** 2 + (a[1] - center[1]) ** 2
        db = (b[0] - center[0]) ** 2 + (b[1] - center[1]) ** 2
        if da >= db:
            return a, (dr, dc)
        return b, (-dr, -dc)

    a, b, c = sorted(pts)
    if _is_collinear(a, b, c):
        start, end = a, c
    else:
        start, end = a, b
    dr = end[0] - start[0]
    dc = end[1] - start[1]
    if abs(dr) != 1 or abs(dc) != 1:
        return None

    ds = (start[0] - center[0]) ** 2 + (start[1] - center[1]) ** 2
    de = (end[0] - center[0]) ** 2 + (end[1] - center[1]) ** 2
    if ds >= de:
        return start, (dr, dc)
    return end, (-dr, -dc)


def generate_input():
    """
    生成 11x11 的奇数网格，构造对角射线与空洞修复示例。

    规则满足：
    - 生成的射线仅经过 0/4/6；
    - 至少 2 个路径空洞，至少 2 个非路径空洞。
    """
    n = 11
    colors = [1, 2, 3]

    for _ in range(600):
        grid = np.zeros((n, n), dtype=int)

        ori = int(_rng.integers(0, 4))
        if ori == 0:
            seed = [(0, 0), (1, 1)]
            dr, dc = 1, 1
        elif ori == 1:
            seed = [(0, 10), (1, 9)]
            dr, dc = 1, -1
        elif ori == 2:
            seed = [(10, 0), (9, 1)]
            dr, dc = -1, 1
        else:
            seed = [(10, 10), (9, 9)]
            dr, dc = -1, -1

        if int(_rng.integers(0, 2)) == 1:
            extra = (seed[1][0] + dr, seed[1][1] + dc)
            if 0 <= extra[0] < n and 0 <= extra[1] < n:
                seed.append(extra)

        for r, c in seed:
            grid[r, c] = 6

        start = seed[0]
        path = _simulate_ray(n, n, start, dr, dc)

        # 先绘制近似对称前景，但避免沿射线路径染色
        orbits = []
        for r in range((n // 2) + 1):
            for c in range((n // 2) + 1):
                orbit = [(r, c), (r, n - 1 - c), (n - 1 - r, c), (n - 1 - r, n - 1 - c)]
                if any(x in seed for x in orbit):
                    continue
                orbits.append((r, c, orbit))

        if len(orbits) < 15:
            continue

        _rng.shuffle(orbits)
        fill_n = int(_rng.integers(12, 22))
        for _, _, orbit in orbits[:fill_n]:
            col = int(_rng.choice(colors))
            for rr, cc in orbit:
                if (rr, cc) not in path and (rr, cc) not in seed:
                    grid[rr, cc] = col

        # 依据总前景数控制密度
        colored_count = int(np.sum(np.isin(grid, colors)))
        if not (30 <= colored_count <= 50):
            continue

        # 选择并保证至少两个路径空洞可修复
        zeros = [(r, c) for r in range(n) for c in range(n) if grid[r, c] == 0 and (r, c) not in seed]
        path_holes = []
        off_holes = []

        for r, c in zeros:
            orbit = [(r, c), (r, n - 1 - c), (n - 1 - r, c), (n - 1 - r, n - 1 - c)]
            has_anchor = any(grid[rr, cc] in colors for rr, cc in orbit)
            if has_anchor:
                if (r, c) in path:
                    path_holes.append((r, c))
                else:
                    off_holes.append((r, c))

        if len(path_holes) < 2:
            # 在路径上人工制造可修复空洞锚点
            for r, c in path:
                if (r, c) in seed:
                    continue
                orbit = [(r, c), (r, n - 1 - c), (n - 1 - r, c), (n - 1 - r, n - 1 - c)]
                for rr, cc in orbit:
                    if (rr, cc) == (r, c):
                        continue
                    if (rr, cc) not in seed and (rr, cc) not in path:
                        grid[rr, cc] = int(_rng.choice(colors))
                        break
                if (r, c) not in seed and grid[r, c] == 0 and any(grid[rr, cc] in colors for rr, cc in orbit):
                    path_holes.append((r, c))
                if len(path_holes) >= 2:
                    break

        zeros = [(r, c) for r in range(n) for c in range(n) if grid[r, c] == 0 and (r, c) not in seed]
        path_holes = []
        off_holes = []
        for r, c in zeros:
            orbit = [(r, c), (r, n - 1 - c), (n - 1 - r, c), (n - 1 - r, n - 1 - c)]
            if any(grid[rr, cc] in colors for rr, cc in orbit):
                if (r, c) in path:
                    path_holes.append((r, c))
                else:
                    off_holes.append((r, c))

        if len(path_holes) < 2 or len(off_holes) < 2:
            continue

        _rng.shuffle(path_holes)
        _rng.shuffle(off_holes)

        for p in path_holes[:2]:
            grid[p] = 4
        for p in off_holes[:2]:
            grid[p] = 4

        return grid

    # 兜底：返回一组固定但合法的示例
    g = np.zeros((n, n), dtype=int)
    g[0, 0] = 6
    g[1, 1] = 6
    g[0, 1] = 4
    g[1, 0] = 4
    for k, col in enumerate(colors):
        r = 2 + k
        for c in range(2, 6):
            g[r, c] = col
    g[3, 5] = 4
    g[4, 6] = 4
    return g


def main(grid):
    """
    输入: 11x11 网格，含蓝/红/绿前景、黄空洞、品红种子(6)，背景黑。

    输出: 仅对被对角反射射线命中的黄空洞做对称修复，修复色来自其四点轨道中出现的非黄前景色。

    算法: 1) 根据种子识别射线路径（碰到边界反射）；
    2) 对每个黄像素判断是否在路径集合中；
    3) 命中则按反射轨道取行号最小的非黄颜色，否则保持不变。

    复杂度: O(H*W)。
    """
    g = np.array(grid, dtype=int, copy=True)
    h, w = g.shape

    seed_info = _parse_seed(g)
    if seed_info is None:
        return g
    start, (dr, dc) = seed_info

    path = _simulate_ray(h, w, start, dr, dc)
    if not path:
        return g

    for r, c in np.argwhere(g == 4):
        r, c = int(r), int(c)
        if (r, c) not in path:
            continue
        orbit = [
            (r, c),
            (r, w - 1 - c),
            (h - 1 - r, c),
            (h - 1 - r, w - 1 - c),
        ]
        candidates = []
        for rr, cc in orbit:
            v = int(g[rr, cc])
            if v not in (0, 4, 6):
                candidates.append((rr, v))
        if not candidates:
            continue
        g[r, c] = min(candidates, key=lambda x: x[0])[1]

    return g
