import numpy as np

_rng = np.random.default_rng(202603)


def _blob(size, color):
    mat = np.zeros((size, size), dtype=int)
    cells = []
    r = int(_rng.integers(0, size))
    c = int(_rng.integers(0, size))
    cells.append((r, c))
    mat[r, c] = color
    target = int(_rng.integers(7, 13))
    for _ in range(target - 1):
        cr, cc = cells[int(_rng.integers(0, len(cells)))]
        dirs = [(1, 0), (-1, 0), (0, 1), (0, -1)]
        _rng.shuffle(dirs)
        moved = False
        for dr, dc in dirs:
            nr, nc = cr + dr, cc + dc
            if 0 <= nr < size and 0 <= nc < size and mat[nr, nc] == 0:
                mat[nr, nc] = color
                cells.append((nr, nc))
                moved = True
                break
        if not moved:
            break
    return mat


def _build():
    n = 13
    mid = n // 2
    hq = mid

    base = np.zeros((n, n), dtype=int)
    base[mid, :] = 5
    base[:, mid] = 5

    colors = [1, 2, 3]

    q0c = int(_rng.choice(colors))
    q0 = _blob(hq, q0c)

    q1 = q0.copy()
    flips = int(_rng.integers(1, hq + 1))
    for _ in range(flips):
        r = int(_rng.integers(0, hq))
        c = int(_rng.integers(0, hq))
        if np.any(q0):
            q1[r, c] = q0c

    # 若仍与 q0 完全一致，做一处开口差异
    if np.array_equal(q1, q0):
        r, c = int(_rng.integers(0, hq)), int(_rng.integers(0, hq))
        q1[r, c] = q0c if q1[r, c] == 0 else 0

    q2 = _blob(hq, int(_rng.choice(colors)))
    q3 = _blob(hq, int(_rng.choice(colors)))

    quads = [q0, q1, q2, q3]
    from itertools import combinations
    diffs = [((np.sum(quads[i] != quads[j])), i, j) for i, j in combinations(range(4), 2)]
    m = min(d[0] for d in diffs)

    if m == 0:
        return None

    # 确保最小对为 (0,1) 且唯一
    ties = [x for x in diffs if x[0] == m]
    if len(ties) != 1 or ties[0][1] != 0 or ties[0][2] != 1:
        return None

    # 同色前景要求：q0 与 q1 使用同一个非零色值
    if np.all(q0[q1 != 0] == q0c) is False:
        return None

    base[0:hq, 0:hq] = q0
    base[0:hq, mid + 1 :] = q1
    base[hq + 1 :, 0:hq] = q2
    base[hq + 1 :, mid + 1 :] = q3

    return base


def generate_input():
    """
    生成 13x13 输入，十字分四象限并保证唯一最小差异象限对。

    方法: 构造 q0 与 q1 只差很小（可见为同色前景），其余象限随机，
    并检查两两差异最小值唯一。
    """
    while True:
        g = _build()
        if g is not None:
            return g


def main(grid):
    """
    输入: 灰(5)十字将 13x13 分为四象限，象限内有单色前景。

    输出: 识别差异像素数最小且唯一的一对象限并输出其前景交集。

    算法: 1) 提取四象限；2) 6 对两两比较；3) 最小差异对逐像素交集，前景置 3。

    复杂度: O(H*W)。
    """
    g = np.array(grid, dtype=int)
    n, m = g.shape
    mid_r, mid_c = n // 2, m // 2

    q_tl = g[:mid_r, :mid_c]
    q_tr = g[:mid_r, mid_c + 1 :]
    q_bl = g[mid_r + 1 :, :mid_c]
    q_br = g[mid_r + 1 :, mid_c + 1 :]

    quads = [q_tl, q_tr, q_bl, q_br]
    pairs = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
    scores = []
    for i, j in pairs:
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
