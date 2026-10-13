import numpy as np

_RNG = np.random.default_rng(20260109)


def main(grid):
    """Input:
    约 14x14，含两个栗色标记点与滑动目标前景色。

    Returns:
    直接滑动到标记旁或遇到灰障碍后按偏转轨迹生长。

    Algorithm:
    对每个非标记前景像素找其对齐标记；若直线路径无灰障碍则搬运到标记旁，否则路径生长并顺时针偏转，步进中留下轨迹。

    Complexity:
    对每个像素常数步骤扫描，O(HW) 时间。
    """
    g = np.array(grid, dtype=int)
    h, w = g.shape

    markers = [(r, c) for r in range(h) for c in range(w) if g[r, c] == 9]
    if len(markers) != 2:
        return g.copy()

    out = g.copy()
    def is_obstacle(r, c):
        return out[r, c] == 5

    def clockwise(d):
        return {( -1, 0): (0, 1), (0, 1): (1, 0), (1, 0): (0, -1), (0, -1): (-1, 0)}[d]

    for r in range(h):
        for c in range(w):
            col = int(out[r, c])
            if col in (0, 5, 9):
                continue

            # 找唯一对齐标记
            cand = []
            for mr, mc in markers:
                if mr == r or mc == c:
                    cand.append((mr, mc))
            if len(cand) != 1:
                continue
            mr, mc = cand[0]
            if mr == r:
                d = (0, 1 if mc > c else -1)
            else:
                d = (1 if mr > r else -1, 0)

            # 检查障碍直路
            blocked = False
            rr, cc = r + d[0], c + d[1]
            while 0 <= rr < h and 0 <= cc < w and (rr, cc) != (mr, mc):
                if is_obstacle(rr, cc):
                    blocked = True
                    break
                rr += d[0]
                cc += d[1]

            if not blocked:
                # 清除原位，搬运到标记旁
                out[r, c] = 0
                tr, tc = mr - d[0], mc - d[1]
                if 0 <= tr < h and 0 <= tc < w and out[tr, tc] == 0:
                    out[tr, tc] = col
                continue

            # 有阻挡：路径生长，保留起点
            cur = (r, c)
            cur_dir = d
            while True:
                rr = cur[0] + cur_dir[0]
                cc = cur[1] + cur_dir[1]
                if not (0 <= rr < h and 0 <= cc < w):
                    break
                if out[rr, cc] in (5, 9):
                    # 顺时针偏转
                    cur_dir = clockwise(cur_dir)
                    rr = cur[0] + cur_dir[0]
                    cc = cur[1] + cur_dir[1]
                    if not (0 <= rr < h and 0 <= cc < w):
                        break
                    if out[rr, cc] in (5, 9):
                        break
                if out[rr, cc] != 0:
                    break
                out[rr, cc] = col
                cur = (rr, cc)

    return out


def generate_input():
    """随机且稳定地生成输入：每个非标记前景像素对齐一个标记，部分被障碍阻挡。"""
    h = w = 14
    g = np.zeros((h, w), dtype=int)
    # 两个标记尽量不共享同一行/列，避免多对一绑定
    def random_marker():
        return int(_RNG.integers(2, h - 2)), int(_RNG.integers(2, w - 2))

    m1 = random_marker()
    m2 = random_marker()
    while m2[0] == m1[0] or m2[1] == m1[1] or abs(m2[0] - m1[0]) < 4:
        m2 = random_marker()

    g[m1] = 9
    g[m2] = 9

    colors = [1, 2, 3, 4]

    # 随机但稳定地为每个标记放一条直达与一条受阻滑动像素
    idx = 0
    for marker in (m1, m2):
        mr, mc = marker
        # 直达：同列（先行）或同行（后行）都行，这里统一同一行
        cr = mr
        c1 = int(_RNG.integers(1, w - 1))
        if c1 == mc:
            c1 = 1 if mc > 1 else 2
        g[cr, c1] = colors[idx % len(colors)]
        idx += 1

        # 受阻：同列且在中间放一个障碍，保证有障碍转折
        cp = mc
        r1 = int(_RNG.integers(2, h - 2))
        if r1 == mr:
            r1 = 3 if mr > 3 else 4
        g[r1, cp] = colors[idx % len(colors)]
        idx += 1
        lo, hi = sorted((r1, mr))
        mid = int((lo + hi) // 2)
        if lo < hi and mid != r1 and mid != mr and 0 <= mid < h:
            g[mid, cp] = 5
            # 若中点在边缘，改用临近点
            if g[mid, cp] == 9 or g[mid, cp] == 0:
                g[mid - 1 if mid - 1 >= 0 else mid + 1, cp] = 5

    # 再加少量随机干扰色块，但保持输入规模与语义约束
    extra = int(_RNG.integers(0, 3))
    for _ in range(extra):
        rr = int(_RNG.integers(0, h))
        cc = int(_RNG.integers(0, w))
        if g[rr, cc] == 0:
            g[rr, cc] = int(_RNG.integers(0, 5))

    return g
