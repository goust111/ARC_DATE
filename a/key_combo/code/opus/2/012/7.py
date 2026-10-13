import numpy as np
from itertools import combinations

_rng = np.random.default_rng(202607)


def _make_quadrant_pattern(size, color):
    mat = np.zeros((size, size), dtype=int)
    k = int(_rng.integers(6, 12))
    cells = []
    r = int(_rng.integers(0, size))
    c = int(_rng.integers(0, size))
    mat[r, c] = color
    cells.append((r, c))
    for _ in range(k - 1):
        rr, cc = cells[int(_rng.integers(0, len(cells)))]
        dirs = [(1, 0), (-1, 0), (0, 1), (0, -1)]
        _rng.shuffle(dirs)
        placed = False
        for dr, dc in dirs:
            nr, nc = rr + dr, cc + dc
            if 0 <= nr < size and 0 <= nc < size and mat[nr, nc] == 0:
                mat[nr, nc] = color
                cells.append((nr, nc))
                placed = True
                break
        if not placed:
            break
    return mat


def generate_input():
    """
    生成 13x13 网格，十字分割为四象限并构造唯一最小差异象限对。

    生成规则: 构造 q0 与 q1 为同色且小差异；其余象限随机，强制 q0-q1 为唯一最小差异。
    """
    while True:
        n = 13
        mid = n // 2
        hq = mid
        base = np.zeros((n, n), dtype=int)
        base[mid, :] = 5
        base[:, mid] = 5

        colors = [1, 2, 3]
        c0 = int(_rng.choice(colors))
        q0 = _make_quadrant_pattern(hq, c0)
        q1 = q0.copy()

        # 确保 q1 与 q0 同色但有差异
        if np.count_nonzero(q1) == 0:
            q1 = q0
        zr, zc = int(_rng.integers(0, hq)), int(_rng.integers(0, hq))
        if q1[zr, zc] == c0:
            q1[zr, zc] = 0
        else:
            q1[zr, zc] = c0

        q2 = _make_quadrant_pattern(hq, int(_rng.choice(colors)))
        q3 = _make_quadrant_pattern(hq, int(_rng.choice(colors)))

        quads = [q0, q1, q2, q3]
        scores = [(int(np.sum(quads[i] != quads[j])), i, j) for i, j in combinations(range(4), 2)]
        m = min(s for s, _, _ in scores)
        ties = [x for x in scores if x[0] == m]

        if m == 0:
            continue
        if not (len(ties) == 1 and ties[0][1] == 0 and ties[0][2] == 1):
            continue

        base[0:hq, 0:hq] = q0
        base[0:hq, mid + 1 :] = q1
        base[hq + 1 :, 0:hq] = q2
        base[hq + 1 :, mid + 1 :] = q3
        return base


def main(grid):
    """
    输入: 灰(5)形成中位十字后，四象限各含前景。

    输出: 取四象限中差异最小的一对象限，输出其前景交集。

    算法: 1) 提取四象限；2) 统计 6 个差异值；3) 最小差异对交集置 3。

    复杂度: O(H*W)。
    """
    g = np.array(grid, dtype=int)
    n, m = g.shape
    if n % 2 == 0 or m % 2 == 0:
        return np.zeros((n // 2, m // 2), dtype=int)

    mid_r, mid_c = n // 2, m // 2

    q_tl = g[:mid_r, :mid_c]
    q_tr = g[:mid_r, mid_c + 1 :]
    q_bl = g[mid_r + 1 :, :mid_c]
    q_br = g[mid_r + 1 :, mid_c + 1 :]

    quads = [q_tl, q_tr, q_bl, q_br]
    scores = []
    for i, j in [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]:
        scores.append((int(np.sum(quads[i] != quads[j])), i, j))

    scores.sort(key=lambda x: (x[0], x[1], x[2]))
    d0, i0, j0 = scores[0]
    if d0 == 0:
        return np.zeros((mid_r, mid_c), dtype=int)

    A, B = quads[i0], quads[j0]
    inter = ((A != 0) & (B != 0)).astype(int)
    out = np.zeros_like(inter, dtype=int)
    out[inter == 1] = 3
    return out
