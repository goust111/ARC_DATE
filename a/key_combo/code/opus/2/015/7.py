import numpy as np

_RNG = np.random.default_rng(20260707)
_PALETTE = np.array([1, 2, 3, 4, 5, 6, 7, 8], dtype=int)

def _rng(seed=None):
    return np.random.default_rng(seed) if seed is not None else _RNG


def _corner_pattern(grid, corner):
    h, w = grid.shape
    nz = np.argwhere(grid != 0)
    if nz.size == 0:
        return 0, (0, 0)

    if corner == "tl":
        sel = nz[(nz[:, 0] < h // 2) & (nz[:, 1] < w // 2)]
        if len(sel) == 0:
            return 0, (0, 0)
        s = max(int(sel[:, 0].max()) + 1, int(sel[:, 1].max()) + 1)
        return s, (0, 0)
    if corner == "tr":
        sel = nz[(nz[:, 0] < h // 2) & (nz[:, 1] >= w // 2)]
        if len(sel) == 0:
            return 0, (0, 0)
        s = max(int(sel[:, 0].max()) + 1, int(grid.shape[1] - 1 - sel[:, 1].min()) + 1)
        return s, (0, w - s)
    if corner == "bl":
        sel = nz[(nz[:, 0] >= h // 2) & (nz[:, 1] < w // 2)]
        if len(sel) == 0:
            return 0, (0, 0)
        s = max(int(grid.shape[0] - 1 - sel[:, 0].min()) + 1, int(sel[:, 1].max()) + 1)
        return s, (h - s, 0)
    sel = nz[(nz[:, 0] >= h // 2) & (nz[:, 1] >= w // 2)]
    if len(sel) == 0:
        return 0, (0, 0)
    s = max(int(grid.shape[0] - 1 - sel[:, 0].min()) + 1, int(grid.shape[1] - 1 - sel[:, 1].min()) + 1)
    return s, (h - s, w - s)


def main(grid):
    """
1) 分别读取四个角落区域的有色图案并补齐为同边长方阵S。
2) 按角落对应方式拼成(2S+1)x(2S+1)中间网格：四象限填入原角落图，中心横竖为黑。
3) 统计中间网格中的非黑颜色种数K，作为像素放大倍率。
4) 输出对中间网格做K倍像素重复后的结果。
"""

    inp = np.array(grid, dtype=int)
    h, w = inp.shape

    s_tl, _ = _corner_pattern(inp, "tl")
    s_tr, _ = _corner_pattern(inp, "tr")
    s_bl, _ = _corner_pattern(inp, "bl")
    s_br, _ = _corner_pattern(inp, "br")

    S = max(1, s_tl, s_tr, s_bl, s_br)

    mid = np.zeros((2 * S + 1, 2 * S + 1), dtype=int)

    # tl
    for r in range(min(S, s_tl)):
        for c in range(min(S, s_tl)):
            mid[r, c] = inp[r, c]
    # tr
    for r in range(min(S, s_tr)):
        for c in range(min(S, s_tr)):
            mid[r, S + 1 + c] = inp[r, w - s_tr + c] if s_tr > 0 else 0
    # bl
    for r in range(min(S, s_bl)):
        for c in range(min(S, s_bl)):
            mid[S + 1 + r, c] = inp[h - s_bl + r, c] if s_bl > 0 else 0
    # br
    for r in range(min(S, s_br)):
        for c in range(min(S, s_br)):
            mid[S + 1 + r, S + 1 + c] = inp[h - s_br + r, w - s_br + c] if s_br > 0 else 0

    colors = set(v for v in mid.flatten() if v != 0)
    K = len(colors)
    if K == 0:
        return inp

    return np.repeat(np.repeat(mid, K, axis=0), K, axis=1)


def generate_input(seed=None):
    rng = _rng(seed)
    H = W = 14

    while True:
        k = int(rng.integers(2, 5))
        if rng.random() < 0.6:
            S = int(rng.integers(2, 4))
            if S == 4 and k == 4:
                S = 3
        else:
            S = int(rng.integers(2, 5))
        S = max(2, min(4, S))
        if S == 4 and k == 4:
            continue

        grid = np.zeros((H, W), dtype=int)
        colors = rng.choice(_PALETTE, size=k, replace=False)

        # 角落图案
        for r in range(S):
            for c in range(S):
                col = int(rng.choice(colors))
                grid[r, c] = col
                grid[r, W - S + c] = int(rng.choice(colors))
                grid[H - S + r, c] = int(rng.choice(colors))
                grid[H - S + r, W - S + c] = int(rng.choice(colors))

        return grid


if __name__ == "__main__":
    pass
