import numpy as np

_rng = np.random.default_rng(202604)


def _cc4(grid, color_mask=None):
    h, w = grid.shape
    vis = np.zeros_like(grid, dtype=bool)
    comps = []
    for r in range(h):
        for c in range(w):
            if vis[r, c]:
                continue
            v = grid[r, c]
            if v == 0 or (color_mask is not None and v not in color_mask):
                continue
            q = [(r, c)]
            vis[r, c] = True
            comp = []
            for cr, cc in q:
                comp.append((cr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not vis[nr, nc] and grid[nr, nc] == v:
                        vis[nr, nc] = True
                        q.append((nr, nc))
            comps.append((v, comp))
    return comps


def _is_rect_frame(cells):
    rs = [p[0] for p in cells]
    cs = [p[1] for p in cells]
    r0, r1 = min(rs), max(rs)
    c0, c1 = min(cs), max(cs)
    if r1 - r0 < 3 or c1 - c0 < 3:
        return None
    expected = 2 * ((r1 - r0 + 1) + (c1 - c0 + 1)) - 4
    border_count = 0
    for r, c in cells:
        if r in (r0, r1) or c in (c0, c1):
            border_count += 1
    if border_count != expected:
        return None
    return (r0, r1, c0, c1)


def _draw_frame(grid, color, r0, c0, h, w):
    r1 = r0 + h - 1
    c1 = c0 + w - 1
    for c in range(c0, c1 + 1):
        grid[r0, c] = color
        grid[r1, c] = color
    for r in range(r0, r1 + 1):
        grid[r, c0] = color
        grid[r, c1] = color
    return (r0, r1, c0, c1)


def generate_input():
    """
    生成 14x14 网格，包含 3~4 个不同色矩形框和黄标记。

    设计为确保至少两个框有边框重叠，以满足部分重叠条件。
    """
    H = W = 14
    colors = [1, 2, 3, 7]

    while True:
        k = int(_rng.integers(3, 5))
        frame_colors = colors[:k]
        _rng.shuffle(frame_colors)

        grid = np.zeros((H, W), dtype=int)
        frames = []
        rects = []

        for idx, fc in enumerate(frame_colors):
            placed = False
            for _ in range(60):
                h = int(_rng.integers(4, 8))
                w = int(_rng.integers(4, 8))

                if idx == 1 and rects:
                    pr0, pr1, pc0, pc1 = rects[0]
                    r0 = max(0, min(H - h, pr0 + int(_rng.integers(-2, 3))))
                    c0 = max(0, min(W - w, pc0 + int(_rng.integers(-2, 3))))
                else:
                    r0 = int(_rng.integers(0, H - h))
                    c0 = int(_rng.integers(0, W - w))

                if r0 < 0 or c0 < 0 or r0 + h > H or c0 + w > W:
                    continue

                r1 = r0 + h - 1
                c1 = c0 + w - 1

                # 避免完全被已有边框遮盖，仍留一些可见像素
                if idx > 0 and np.all(grid[r0:r1 + 1, c0:c1 + 1] == 0):
                    pass

                _draw_frame(grid, fc, r0, c0, h, w)
                rects.append((r0, r1, c0, c1))
                frames.append((fc, r0, r1, c0, c1))
                placed = True
                break

            if not placed:
                break

        if len(frames) != k:
            continue

        # 至少两框有边框重叠
        overlap = False
        for i in range(len(frames)):
            for j in range(i + 1, len(frames)):
                _, a0, a1, a2, a3 = frames[i]
                _, b0, b1, b2, b3 = frames[j]
                if not (a1 < b0 or b1 < a0 or a3 < b2 or b3 < a2):
                    overlap = True
                    break
            if overlap:
                break
        if not overlap:
            continue

        # 严格内部黄点计数互异
        cnts = list(range(1, k + 1))
        _rng.shuffle(cnts)

        for idx, (fc, r0, r1, c0, c1) in enumerate(frames):
            interior = [(r, c) for r in range(r0 + 1, r1) for c in range(c0 + 1, c1) if grid[r, c] == 0]
            if len(interior) < cnts[idx]:
                break
            _rng.shuffle(interior)
            for r, c in interior[:cnts[idx]]:
                grid[r, c] = 4
        else:
            # 干扰项（框外黄点）
            outs = [(r, c) for r in range(H) for c in range(W) if grid[r, c] == 0]
            _rng.shuffle(outs)
            for r, c in outs[:2]:
                grid[r, c] = 4

            return grid


def main(grid):
    """
    输入: 若干单像素厚矩形框（边框色 1/2/3/7）与黄标记(4)。

    输出: 擦除所有黄点后仅保留边框，按“框内黄点数”从少到多依次绘制。

    算法: 1) 识别各矩形框并计算其严格内部黄点数量；
    2) 依据数量排序后绘制边框，后绘制覆盖先绘制；
    3) 忽略框外黄点。

    复杂度: O(H*W + F*P)。
    """
    g = np.array(grid, dtype=int, copy=True)

    comps = _cc4(g, {1, 2, 3, 7})
    frames = []

    for color, comp in comps:
        bbox = _is_rect_frame(comp)
        if bbox is None:
            continue
        r0, r1, c0, c1 = bbox
        cnt = 0
        for r in range(r0 + 1, r1):
            for c in range(c0 + 1, c1):
                if g[r, c] == 4:
                    cnt += 1
        frames.append((cnt, color, r0, r1, c0, c1))

    if not frames:
        return np.zeros(g.shape, dtype=int)

    frames.sort(key=lambda x: x[0])

    out = np.zeros(g.shape, dtype=int)
    for _, color, r0, r1, c0, c1 in frames:
        for c in range(c0, c1 + 1):
            out[r0, c] = color
            out[r1, c] = color
        for r in range(r0, r1 + 1):
            out[r, c0] = color
            out[r, c1] = color
    return out
