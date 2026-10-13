import numpy as np

_RNG = np.random.default_rng(20260707)

def _rng(seed=None):
    return np.random.default_rng(seed) if seed is not None else _RNG


def main(grid):
    """
1) 在每个列中识别单列连续到底边界的竖条对象（蓝/红）。
2) 从每个蓝色竖条按长度跳转到下一列，递归得到链式集合。
3) 链内竖条先整体清空，再按原高度绘制水平横条。
4) 横条按绝对列奇偶着色：偶列原色，奇列交换色，返回最终网格。
"""

    out = np.array(grid, dtype=int, copy=True)
    h, w = out.shape

    bars = {}
    for c in range(w):
        rows = np.where(out[:, c] != 0)[0]
        if rows.size == 0:
            continue
        top = int(rows.min())
        bottom = int(rows.max())
        if bottom != h - 1:
            continue
        if bottom - top + 1 != rows.size:
            continue
        col = out[rows[0], c]
        if col not in (1, 2):
            continue
        bars[c] = {
            "top": top,
            "bottom": bottom,
            "height": bottom - top + 1,
            "color": col,
        }

    chain_cols = set()
    for c, info in bars.items():
        if info["color"] != 1:
            continue
        cur = c
        visited = set()
        while cur in bars and cur not in visited:
            visited.add(cur)
            chain_cols.add(cur)
            cur = cur + bars[cur]["height"]

    original = {c: info.copy() for c, info in bars.items()}

    # 擦除链条
    for c in chain_cols:
        hgt = original[c]["height"]
        top = original[c]["top"]
        out[top : hgt + top, c] = 0

    # 绘制链内横条
    for c in chain_cols:
        hgt = original[c]["height"]
        color = original[c]["color"]
        swapped = 2 if color == 1 else 1
        for x in range(c, min(w, c + hgt)):
            out[h - 1, x] = color if (x % 2 == 0) else swapped

    return out


def generate_input(seed=None):
    rng = _rng(seed)
    h, w = 12, 20

    while True:
        grid = np.zeros((h, w), dtype=int)
        occ = np.zeros((h, w), dtype=bool)

        bars = {}
        # 构造一个蓝色起点链
        placed = 0
        start = int(rng.integers(0, 8))
        c = start
        color = 1
        for _ in range(3):
            height = int(rng.integers(2, 7))
            if c + 2 >= w:
                break
            top = h - height
            if np.any(occ[top:h, c]):
                break
            for rr in range(top, h):
                grid[rr, c] = color
                occ[rr, c] = True
            bars[c] = (color, height)
            placed += 1
            c = c + height
            color = 2 if color == 1 else 1
            if c >= w:
                break

        # 额外随机竖条
        n_extra = int(rng.integers(0, 5))
        while n_extra > 0:
            c = int(rng.integers(0, w))
            if np.any(occ[:, c]):
                continue
            height = int(rng.integers(2, 7))
            top = h - height
            if top < 0:
                continue
            for rr in range(top, h):
                grid[rr, c] = int(rng.integers(1, 3))
                occ[rr, c] = True
            n_extra -= 1

        # 至少一个蓝色条和一个红色条
        if not np.any(grid[:, :] == 1):
            continue
        return grid


if __name__ == "__main__":
    pass
