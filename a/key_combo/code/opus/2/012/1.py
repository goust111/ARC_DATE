import numpy as np
from collections import deque

_rng = np.random.default_rng(202601)


def _bfs_path(grid, start, goal):
    h, w = grid.shape
    dist = -np.ones((h, w), dtype=int)
    prev = {}
    q = deque([start])
    dist[start] = 0
    while q:
        r, c = q.popleft()
        if (r, c) == goal:
            break
        nd = dist[r, c] + 1
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nr, nc = r + dr, c + dc
            if not (0 <= nr < h and 0 <= nc < w):
                continue
            if grid[nr, nc] == 3:
                continue
            if dist[nr, nc] != -1:
                continue
            dist[nr, nc] = nd
            prev[(nr, nc)] = (r, c)
            q.append((nr, nc))

    if dist[goal] == -1:
        return None

    path = []
    cur = goal
    while cur != start:
        path.append(cur)
        cur = prev[cur]
    path.append(start)
    path.reverse()
    return path


def generate_input():
    """
    输入 15x15 网格。

    构造方法: 1) 选偶数种子列并绘制周期绿线；2) 设定缺口行；3) 放置左右端点并验通路。

    约束: 两端点不在同一行，生成后存在至少一条通路。

    返回: 仅含绿色种子柱、两红端点、黑底。
    """
    H = W = 15
    while True:
        seed_col = int(_rng.integers(4, 10))
        if seed_col % 2 == 1:
            seed_col -= 1
        seed_row = int(_rng.integers(2, H - 2))

        left_max = max(1, seed_col - 1)
        left_col = int(_rng.integers(1, left_max))
        right_min = min(W - 1, seed_col + 5)
        if right_min >= W - 1:
            continue
        right_col = int(_rng.integers(right_min, W - 1))

        row1 = int(_rng.integers(0, H))
        row2 = int(_rng.integers(0, H))
        if row1 == row2:
            row2 = (row2 + 2) % H if H > 2 else row2

        grid = np.zeros((H, W), dtype=int)
        for c in range(seed_col, W, 2):
            grid[:, c] = 3

        for c in range(seed_col, W, 2):
            grid[seed_row, c] = 0

        grid[row1, left_col] = 2
        grid[row2, right_col] = 2

        if _bfs_path(grid, (row1, left_col), (row2, right_col)) is None:
            continue

        return grid


def main(grid):
    """
    输入: 0 背景、3 绿栅栏、2 两红端点。

    输出: 保留原像素，并在端点之间用 4 标注障碍避让后的四邻接最短路径。

    算法: 1) 从障碍 3 中提取绿柱列；2) BFS 求最短路；3) 染色中间 0 像素为 4。

    复杂度: O(H*W)。
    """
    g = np.array(grid, dtype=int, copy=True)
    h, w = g.shape

    verts = np.argwhere(g == 3)
    if verts.size == 0:
        return g

    reds = np.argwhere(g == 2)
    if reds.shape[0] != 2:
        return g

    start = tuple(int(x) for x in reds[0])
    goal = tuple(int(x) for x in reds[1])

    path = _bfs_path(g, start, goal)
    if path is None:
        return g

    for r, c in path:
        if (r, c) not in (start, goal) and g[r, c] == 0:
            g[r, c] = 4

    return g
