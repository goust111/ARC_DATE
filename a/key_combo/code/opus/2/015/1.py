import numpy as np

_RNG = np.random.default_rng(20260707)
_PALETTE = np.array([1, 3, 4, 7, 6], dtype=np.int32)
_DIRECTIONS = ["up", "down", "left", "right"]


def _rng(seed=None):
    return np.random.default_rng(seed) if seed is not None else _RNG


def _components(grid):
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps = []
    for r in range(h):
        for c in range(w):
            if grid[r, c] == 0 or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            cells = []
            while stack:
                x, y = stack.pop()
                cells.append((x, y))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = x + dr, y + dc
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and grid[nr, nc] != 0:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(cells)
    return comps


def main(grid):
    """
1. 规则归纳：每个块内颜色数决定其射线/色带长度。
2. 输入假设：黑底网格，单个对象可能包含恰好一个红色指示像素与其它非黑颜色。
3. 处理步骤：
   - 按四邻域提取非零连通块，找出含唯一红点的块。
   - 统计该块非黑且非红颜色种数 K。
   - 沿指示像素朝外延伸 K 格，长度方向以块边界长度为另一维。
4. 输出：返回同尺寸 np.ndarray。
"""

    out = np.array(grid, dtype=int, copy=True)
    for cells in _components(out):
        reds = [(r, c) for r, c in cells if out[r, c] == 2]
        if len(reds) != 1:
            continue
        ir, ic = reds[0]
        rows = [r for r, _ in cells]
        cols = [c for _, c in cells]
        minr, maxr = min(rows), max(rows)
        minc, maxc = min(cols), max(cols)
        if ir == minr:
            direction = "up"
        elif ir == maxr:
            direction = "down"
        elif ic == minc:
            direction = "left"
        elif ic == maxc:
            direction = "right"
        else:
            continue

        k = len({out[r, c] for r, c in cells if out[r, c] not in (0, 2)})
        if k <= 0:
            continue

        if direction in ("up", "down"):
            c0, c1 = minc, maxc + 1
            if direction == "up":
                r0 = ir - k
                if r0 < 0:
                    continue
                for rr in range(r0, ir):
                    out[rr, c0:c1] = 3
            else:
                r0 = ir + 1
                r1 = r0 + k
                if r1 > out.shape[0]:
                    continue
                for rr in range(r0, r1):
                    out[rr, c0:c1] = 3
        else:
            r0, r1 = minr, maxr + 1
            if direction == "left":
                c0 = ic - k
                if c0 < 0:
                    continue
                for cc in range(c0, ic):
                    out[r0:r1, cc] = 3
            else:
                c0 = ic + 1
                c1 = c0 + k
                if c1 > out.shape[1]:
                    continue
                for cc in range(c0, c1):
                    out[r0:r1, cc] = 3

        out[ir, ic] = 4
    return out


def generate_input(seed=None):
    rng = _rng(seed)
    H = W = 15

    while True:
        grid = np.zeros((H, W), dtype=int)
        occ = np.zeros((H, W), dtype=bool)
        n_blocks = int(rng.integers(3, 6))
        placed = 0

        for _ in range(n_blocks):
            ok = False
            for _ in range(300):
                h = int(rng.integers(2, 4))
                w = int(rng.integers(3, 5))
                direction = _DIRECTIONS[int(rng.integers(4))]
                k = int(rng.integers(1, 5))
                colors = rng.choice(_PALETTE, size=k, replace=False)

                if direction == "up":
                    if h + k >= H:
                        continue
                    r0 = int(rng.integers(k, H - h + 1))
                    c0 = int(rng.integers(0, W - w + 1))
                    ext_r0, ext_r1 = r0 - k, r0
                    ext_c0, ext_c1 = c0, c0 + w
                elif direction == "down":
                    if h + k >= H:
                        continue
                    r0 = int(rng.integers(0, H - h - k + 1))
                    c0 = int(rng.integers(0, W - w + 1))
                    ext_r0, ext_r1 = r0 + h, r0 + h + k
                    ext_c0, ext_c1 = c0, c0 + w
                elif direction == "left":
                    if w + k >= W:
                        continue
                    r0 = int(rng.integers(0, H - h + 1))
                    c0 = int(rng.integers(k, W - w + 1))
                    ext_r0, ext_r1 = r0, r0 + h
                    ext_c0, ext_c1 = c0 - k, c0
                else:  # right
                    if w + k >= W:
                        continue
                    r0 = int(rng.integers(0, H - h + 1))
                    c0 = int(rng.integers(0, W - w - k + 1))
                    ext_r0, ext_r1 = r0, r0 + h
                    ext_c0, ext_c1 = c0 + w, c0 + w + k

                block_r0, block_r1 = r0, r0 + h
                block_c0, block_c1 = c0, c0 + w
                rr0 = max(0, min(block_r0, ext_r0) - 1)
                rr1 = min(H, max(block_r1, ext_r1) + 1)
                cc0 = max(0, min(block_c0, ext_c0) - 1)
                cc1 = min(W, max(block_c1, ext_c1) + 1)
                if np.any(occ[rr0:rr1, cc0:cc1]):
                    continue

                if direction == "up":
                    ir = r0
                    ic = int(rng.integers(c0, c0 + w))
                elif direction == "down":
                    ir = r0 + h - 1
                    ic = int(rng.integers(c0, c0 + w))
                elif direction == "left":
                    ir = int(rng.integers(r0, r0 + h))
                    ic = c0
                else:
                    ir = int(rng.integers(r0, r0 + h))
                    ic = c0 + w - 1

                body = [(rr, cc) for rr in range(block_r0, block_r1) for cc in range(block_c0, block_c1)]
                body_no_red = [(rr, cc) for rr, cc in body if (rr, cc) != (ir, ic)]
                if len(body_no_red) < 1:
                    continue

                colors_assign = list(rng.choice(body_no_red, size=k, replace=False))
                for idx, pos in enumerate(colors_assign):
                    grid[pos] = int(colors[idx])
                for pos in body_no_red:
                    if grid[pos] == 0:
                        grid[pos] = int(rng.choice(colors))
                grid[ir, ic] = 2

                if direction in ("up", "down"):
                    grid[ext_r0:ext_r1, ext_c0:ext_c1] = 0
                else:
                    grid[ext_r0:ext_r1, ext_c0:ext_c1] = 0
                occ[rr0:rr1, cc0:cc1] = True
                ok = True
                placed += 1
                break
            if not ok:
                break

        if placed == n_blocks:
            return grid


if __name__ == "__main__":
    pass
