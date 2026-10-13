import numpy as np
from collections import deque

# RGB-like ARC palette constants used by this task.
BG_COLOR = 0
COLOR_RED = 2      # up
COLOR_BLUE = 1     # down
COLOR_GREEN = 3    # left
COLOR_ORANGE = 7   # right

DIR_FROM_COLOR = {
    COLOR_RED: (-1, 0),
    COLOR_BLUE: (1, 0),
    COLOR_GREEN: (0, -1),
    COLOR_ORANGE: (0, 1),
}


def _neighbors(r, c, h, w):
    if r > 0:
        yield r - 1, c
    if r + 1 < h:
        yield r + 1, c
    if c > 0:
        yield r, c - 1
    if c + 1 < w:
        yield r, c + 1


def _connected_components(mask):
    h, w = mask.shape
    visited = np.zeros_like(mask, dtype=bool)
    comps = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or visited[r, c]:
                continue
            q = deque([(r, c)])
            visited[r, c] = True
            comp = []
            while q:
                x, y = q.popleft()
                comp.append((x, y))
                for nx, ny in _neighbors(x, y, h, w):
                    if mask[nx, ny] and not visited[nx, ny]:
                        visited[nx, ny] = True
                        q.append((nx, ny))
            comps.append(comp)
    return comps


def _trace_cycle(start, comp_set, comp_size, h, w):
    sx, sy = start
    start_neighbors = [p for p in _neighbors(sx, sy, h, w) if p in comp_set]
    if len(start_neighbors) != 2:
        return []

    for first in start_neighbors:
        path = [start]
        visited = {start}
        prev = start
        cur = first

        while len(visited) < comp_size:
            path.append(cur)
            visited.add(cur)
            nxts = [n for n in _neighbors(*cur, h, w) if n in comp_set and n != prev]
            if not nxts:
                break
            nxt = nxts[0]
            if nxt == start:
                break
            if nxt in visited:
                break
            prev, cur = cur, nxt
        else:
            return path

        if cur == start:
            return path
        # Candidate start direction failed.

    return []


def _signed_area2(path):
    if not path:
        return 0
    n = len(path)
    a = 0
    for i in range(n):
        x1, y1 = path[i][1], path[i][0]
        x2, y2 = path[(i + 1) % n][1], path[(i + 1) % n][0]
        a += x1 * y2 - x2 * y1
    return a


def _run_length_instructions(grid, path):
    if not path:
        return []

    instr = []
    run_color = int(grid[path[0]])
    run_len = 1

    for r, c in path[1:]:
        col = int(grid[r, c])
        if col == run_color:
            run_len += 1
        else:
            direction = DIR_FROM_COLOR.get(run_color)
            if direction is not None and run_len > 0:
                instr.append((direction[0], direction[1], run_len))
            run_color = col
            run_len = 1

    direction = DIR_FROM_COLOR.get(run_color)
    if direction is not None and run_len > 0:
        instr.append((direction[0], direction[1], run_len))

    return instr


def _point_in_polygon(x, y, path):
    n = len(path)
    if n < 3:
        return False
    inside = False
    for i in range(n):
        x1, y1 = path[i][1], path[i][0]
        x2, y2 = path[(i + 1) % n][1], path[(i + 1) % n][0]
        if ((y1 > y) != (y2 > y)):
            xinters = (x2 - x1) * (y - y1) / (y2 - y1 + 1e-12) + x1
            if x < xinters:
                inside = not inside
    return inside


def _find_boundaries_and_instructions(grid, bg):
    h, w = grid.shape
    non_bg = grid != bg
    comps = _connected_components(non_bg)
    info = []
    boundary_mask = np.zeros_like(non_bg, dtype=bool)

    for comp in comps:
        if len(comp) < 4:
            continue

        comp_set = set(comp)
        valid = True
        for p in comp:
            d = 0
            for n in _neighbors(*p, h, w):
                if n in comp_set:
                    d += 1
            if d != 2:
                valid = False
                break
        if not valid:
            continue

        start = min(comp, key=lambda p: (p[0], p[1]))
        path = _trace_cycle(start, comp_set, len(comp), h, w)
        if not path or len(path) != len(comp):
            continue

        # Ensure clockwise order (for row-axis downward coordinates, clockwise gives positive area).
        if _signed_area2(path) < 0:
            path = [path[0]] + path[:0:-1]

        instructions = _run_length_instructions(grid, path)
        if not instructions:
            continue

        info.append({
            "path": path,
            "instructions": instructions,
            "seed": None,
        })
        for r, c in comp:
            boundary_mask[r, c] = True

    return info, boundary_mask


def _paint_trace(grid, seed, instructions, boundary_mask):
    h, w = grid.shape
    seed_r, seed_c = seed
    seed_color = int(grid[seed_r, seed_c])

    r, c = seed_r, seed_c
    grid[r, c] = seed_color

    for dr, dc, steps in instructions:
        for _ in range(steps):
            nr, nc = r + dr, c + dc
            if not (0 <= nr < h and 0 <= nc < w):
                break
            if boundary_mask[nr, nc]:
                break
            r, c = nr, nc
            grid[r, c] = seed_color


def main(grid: np.ndarray) -> np.ndarray:
    """
    输入
    读取一张 ARC 风格的二维整数网格。

    输出
    返回与输入同形状的网格；保留边界颜色，只在每个区域内按边界编码的指令
    生长黄种子轨迹。

    规则
    先识别闭合边界环，再从每个闭合区域内非边界种子像素出发，按
    颜色片段长度解码出方向-步数指令（上/下/左/右），并在遇到边界前绘制。

    实现
    1) 以非背景连通块中“每点度数为 2”的像素作为边界环。
    2) 从“顶部最左”像素追踪完整闭合顺序，并按多边形面积修正为顺时针。
    3) 将边界颜色按段转为指令序列，然后给到区域种子像素填色。
    """

    g = np.array(grid, dtype=int, copy=True)
    if g.size == 0:
        return g

    bg = int(np.bincount(g.ravel()).argmax())
    regions, boundary_mask = _find_boundaries_and_instructions(g, bg)
    if not regions:
        return g

    # Find possible seeds: non-bg, non-boundary pixels.
    candidates = np.argwhere((g != bg) & (~boundary_mask))

    for region in regions:
        if region["seed"] is not None:
            continue
        for sr, sc in map(tuple, candidates):
            if g[sr, sc] in DIR_FROM_COLOR:
                continue
            if _point_in_polygon(sc + 0.5, sr + 0.5, region["path"]):
                region["seed"] = (sr, sc)
                break

    for region in regions:
        if region["seed"] is None:
            continue
        _paint_trace(g, region["seed"], region["instructions"], boundary_mask)

    return g


def generate_input() -> np.ndarray:
    """生成随机封闭边界与种子轨迹题目输入，尺寸不超过 30。"""

    rng = np.random.default_rng()
    size = int(rng.integers(18, 31))
    grid = np.full((size, size), BG_COLOR, dtype=int)

    occupied = np.zeros((size, size), dtype=bool)
    drawn = 0
    target_regions = int(rng.integers(1, 4))
    seed_palette = np.array([4, 5, 6, 8, 9], dtype=int)

    for _ in range(target_regions):
        placed = False
        for _ in range(400):
            h = int(rng.integers(5, min(12, size - 2) + 1))
            w = int(rng.integers(5, min(12, size - 2) + 1))
            if h + 2 >= size or w + 2 >= size:
                continue

            top = int(rng.integers(1, size - h - 1))
            left = int(rng.integers(1, size - w - 1))
            bottom = top + h
            right = left + w

            pad = 1
            rs0 = max(0, top - pad)
            rs1 = min(size, bottom + pad + 1)
            cs0 = max(0, left - pad)
            cs1 = min(size, right + pad + 1)
            if occupied[rs0:rs1, cs0:cs1].any():
                continue

            # Encode directions in boundary segment order: right/down/left/up (clockwise).
            grid[top, left : right + 1] = COLOR_ORANGE
            if top + 1 <= bottom - 1:
                grid[top + 1 : bottom, right] = COLOR_BLUE
            grid[bottom, left : right + 1] = COLOR_GREEN
            if top + 1 <= bottom - 1:
                grid[top + 1 : bottom, left] = COLOR_RED

            # Seed in interior and different from boundary/bg colors.
            sr = int(rng.integers(top + 1, bottom))
            sc = int(rng.integers(left + 1, right))
            seed_color = int(rng.choice(seed_palette))
            grid[sr, sc] = seed_color

            occupied[rs0:rs1, cs0:cs1] = True
            drawn += 1
            placed = True
            break

        if not placed:
            continue

    # Safety fallback to guarantee at least one valid region.
    if drawn == 0:
        top = size // 2 - 3
        left = size // 2 - 3
        bottom = top + 5
        right = left + 5
        top = max(1, top)
        left = max(1, left)
        bottom = min(size - 2, top + 5)
        right = min(size - 2, left + 5)

        grid[top, left : right + 1] = COLOR_ORANGE
        if top + 1 <= bottom - 1:
            grid[top + 1 : bottom, right] = COLOR_BLUE
        grid[bottom, left : right + 1] = COLOR_GREEN
        if top + 1 <= bottom - 1:
            grid[top + 1 : bottom, left] = COLOR_RED

        grid[(top + 1 + bottom - 1) // 2, (left + 1 + right - 1) // 2] = 4

    return grid


if __name__ == "__main__":
    g = generate_input()
    out = main(g)
    print(g.shape, out.shape)
