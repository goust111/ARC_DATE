import numpy as np

_RNG = np.random.default_rng(20260106)


def _component_positions(mask: np.ndarray):
    h, w = mask.shape
    vis = np.zeros_like(mask, dtype=bool)
    comps = []
    dirs = ((1, 0), (-1, 0), (0, 1), (0, -1))
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or vis[r, c]:
                continue
            q = [(r, c)]
            vis[r, c] = True
            qi = 0
            while qi < len(q):
                cr, cc = q[qi]
                qi += 1
                for dr, dc in dirs:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not vis[nr, nc]:
                        vis[nr, nc] = True
                        q.append((nr, nc))
            comps.append(q)
    return comps


def _bbox(cells):
    rs = [r for r, _ in cells]
    cs = [c for _, c in cells]
    return min(rs), max(rs), min(cs), max(cs)


def main(grid):
    """Input:
    约 17x17：棋盘格起点、若干环形对象、灰色边框面板。

    Returns:
    追踪引导路径选环，按目标环方位旋转面板内容输出。

    Algorithm:
    定位 3x3 棋盘格与路径颜色；跟随颜色前进到达目标环外部端点后确定方位；裁剪面板内区并按 0/90/180/270 顺时针旋转。

    Complexity:
    O(HW) 搜索路径与区域检测。
    """
    g = np.array(grid, dtype=int)
    h, w = g.shape

    # 1) 找棋盘格 (3x3)
    check = None
    guide_start = None
    guide_color = None
    for r in range(h - 2):
        for c in range(w - 2):
            block = g[r : r + 3, c : c + 3]
            s = set(block.flatten())
            if s.issubset({0, 7}) and (0 in s and 7 in s):
                # 简单判定：交替棋盘格至少 3/4 为 7
                if np.sum(block == 7) in (4, 5):
                    check = (r, c)
                    break
        if check is not None:
            break

    if check is None:
        return g.copy()

    cr, cc = check
    checker = {(cr + dr, cc + dc) for dr in range(3) for dc in range(3)}
    for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        nr, nc = cr + dr, cc + dc
        if 0 <= nr < h and 0 <= nc < w and g[nr, nc] not in (0, 7):
            guide_start = (nr, nc)
            guide_color = int(g[nr, nc])
            break
    if guide_start is None:
        return g.copy()

    # 跟踪路径（步长 2）
    path = [guide_start]
    visited = {guide_start}
    cur = guide_start
    while True:
        r, c = cur
        nxt = []
        for dr, dc in ((2, 0), (-2, 0), (0, 2), (0, -2)):
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w and g[nr, nc] == guide_color and (nr, nc) not in visited:
                nxt.append((nr, nc))
        if not nxt:
            break
        cur = sorted(nxt, key=lambda p: abs(p[0] - guide_start[0]) + abs(p[1] - guide_start[1]))[-1]
        visited.add(cur)
        path.append(cur)

    # 找 panel（灰色边框）
    panel = None
    for comp in _component_positions(g == 5):
        r0, r1, c0, c1 = _bbox(comp)
        area = len(comp)
        if area != 2 * (r1 - r0 + c1 - c0):
            continue
        ok = True
        for x in range(c0, c1 + 1):
            if g[r0, x] != 5 or g[r1, x] != 5:
                ok = False
                break
        for y in range(r0, r1 + 1):
            if g[y, c0] != 5 or g[y, c1] != 5:
                ok = False
                break
        if ok and (r1 - r0 + 1) >= 4 and (c1 - c0 + 1) >= 4:
            panel = (r0, r1, c0, c1)
            break

    if panel is None:
        return g.copy()

    pr0, pr1, pc0, pc1 = panel
    prc = (pr0 + pr1) // 2
    pcc = (pc0 + pc1) // 2
    panel_cells = [(r, c) for r in range(pr0 + 1, pr1) for c in range(pc0 + 1, pc1)]

    # 识别 3x3 环形 + 中心标记 9
    rings = []
    for r in range(h - 2):
        for c in range(w - 2):
            center = (r + 1, c + 1)
            if g[center] != 9:
                continue
            border = [
                (r, c), (r, c + 1), (r, c + 2),
                (r + 1, c), (r + 1, c + 2),
                (r + 2, c), (r + 2, c + 1), (r + 2, c + 2),
            ]
            if any(g[x, y] != 6 for x, y in border):
                continue
            holes = [p for p in border + [center] if g[p] == 0]
            # 允许仅一个孔格（门洞）
            if len(holes) != 1:
                continue
            hole = holes[0]
            ring_side = ""
            if hole[0] == r:
                ring_side = "up"
            elif hole[0] == r + 2:
                ring_side = "down"
            elif hole[1] == c:
                ring_side = "left"
            else:
                ring_side = "right"
            # 外门洞为门洞外侧格
            if ring_side == "up":
                door = (r - 1, hole[1])
            elif ring_side == "down":
                door = (r + 3, hole[1])
            elif ring_side == "left":
                door = (hole[0], c - 1)
            else:
                door = (hole[0], c + 3)
            rings.append((center, door, (r0 := r), (c0 := c)))

    endpoint = path[-1] if path else None
    target = None
    if endpoint is not None and rings:
        # 终点附近找门洞外侧格
        for center, door, rr, cc in rings:
            if abs(endpoint[0] - door[0]) + abs(endpoint[1] - door[1]) <= 1:
                target = (center, rr, cc)
                break

    rot = 0
    if target is not None:
        (tr, tc), _, _ = target
        if tr > prc and abs(tr - prc) >= abs(tc - pcc):
            rot = 2
        elif tr < prc and abs(tr - prc) >= abs(tc - pcc):
            rot = 0
        elif tc > pcc:
            rot = -1 % 4
        else:
            rot = 1

    inner = g[pr0 + 1 : pr1, pc0 + 1 : pc1]
    if inner.size == 0:
        return g.copy()

    k = rot % 4
    out = np.rot90(inner, k=k)
    return out


def _draw_panel(g, r0, c0, s=7):
    # 外框
    for x in range(c0, c0 + s):
        g[r0, x] = 5
        g[r0 + s - 1, x] = 5
    for y in range(r0, r0 + s):
        g[y, c0] = 5
        g[y, c0 + s - 1] = 5
    # 内部随机颜色模式
    palette = [1, 2, 3, 7]
    for y in range(r0 + 1, r0 + s - 1):
        for x in range(c0 + 1, c0 + s - 1):
            g[y, x] = int(palette[_RNG.integers(0, len(palette))])


def _place_ring(g, top, left, hole_dir, color):
    # top left 指 3x3 的左上
    g[top + 1, left + 1] = 9
    ring = [(top, left), (top, left + 1), (top, left + 2), (top + 1, left), (top + 1, left + 2), (top + 2, left), (top + 2, left + 1), (top + 2, left + 2)]
    for r, c in ring:
        g[r, c] = 6
    if hole_dir == "up":
        hole = (top, left + 1)
        g[hole] = 0
    elif hole_dir == "down":
        hole = (top + 2, left + 1)
        g[hole] = 0
    elif hole_dir == "left":
        hole = (top + 1, left)
        g[hole] = 0
    else:
        hole = (top + 1, left + 2)
        g[hole] = 0


def generate_input():
    """随机且稳定地生成输入：17x17，含四环框、棋盘格和可达路径。"""
    h = w = 17
    g = np.zeros((h, w), dtype=int)

    _draw_panel(g, 5, 5, s=7)
    _place_ring(g, 1, 7, "down", 2)
    _place_ring(g, 7, 12, "left", 1)
    _place_ring(g, 11, 7, "up", 3)
    _place_ring(g, 7, 1, "right", 4)

    # 棋盘格（左上角）
    g[0:3, 0:3] = np.array([[0, 7, 0], [7, 0, 7], [0, 7, 0]])
    guide_color = 8
    g[0, 3] = guide_color
    g[1, 3] = guide_color

    # 从棋盘相邻点做 2 格步长路径到右侧环的外门
    path = [(0, 3)]
    for _ in range(1, 8):
        r = 0
        c = 3 + 2 * _
        if c < w:
            g[r, c] = guide_color
    for _ in range(4):
        r = 2 + 2 * _
        c = 15
        if r < h:
            g[r, c] = guide_color

    return g
