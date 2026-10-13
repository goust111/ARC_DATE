import numpy as np

_rng = np.random.default_rng(202606)


def _cc4(grid, mask):
    h, w = grid.shape
    vis = np.zeros_like(grid, dtype=bool)
    comps = []
    for r in range(h):
        for c in range(w):
            if vis[r, c] or grid[r, c] not in mask:
                continue
            color = int(grid[r, c])
            q = [(r, c)]
            vis[r, c] = True
            comp = []
            for cr, cc in q:
                comp.append((cr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not vis[nr, nc] and grid[nr, nc] == color:
                        vis[nr, nc] = True
                        q.append((nr, nc))
            comps.append((color, comp))
    return comps


def _draw_line(a, b, color, out):
    ar, ac = a
    br, bc = b
    if ar == br:
        c0, c1 = min(ac, bc), max(ac, bc)
        out[ar, c0 : c1 + 1] = color
    elif ac == bc:
        r0, r1 = min(ar, br), max(ar, br)
        out[r0 : r1 + 1, ac] = color


def _interior_flood(h, w, fill_color, mask):
    return

def _fill_closed(mask, color):
    # mask 为布尔，1 表示已有线段/轮廓
    h, w = mask.shape
    if not mask.any():
        return np.zeros_like(mask, dtype=bool)
    filled = mask.copy()
    # 找到连通域边界（四邻）
    vis = np.zeros_like(mask, dtype=bool)
    from collections import deque
    q = deque()
    for r in range(h):
        for c in range(w):
            if r not in (0, h - 1) and c not in (0, w - 1):
                continue
            if mask[r, c] == 0 and not vis[r, c]:
                vis[r, c] = True
                q.append((r, c))
                while q:
                    cr, cc = q.popleft()
                    for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                        nr, nc = cr + dr, cc + dc
                        if 0 <= nr < h and 0 <= nc < w and not vis[nr, nc] and mask[nr, nc] == 0:
                            vis[nr, nc] = True
                            q.append((nr, nc))
    inside = ~vis & ~mask
    return inside


def generate_input():
    """
    生成 18x18 网格：2~3 个空心框架与 2~3 个独立精灵体。
    精灵体的像素数与某一框架边框像素数严格一一匹配。
    """
    H = W = 18
    frame_colors = [5, 9, 7]
    sprite_colors = [1, 2, 3]

    while True:
        k = int(_rng.integers(2, 4))
        fc = frame_colors[:k]
        sc = sprite_colors[:k]

        perims = []
        dims = []
        # 产生互异周长
        for _ in range(k):
            while True:
                fh = int(_rng.integers(4, 8))
                fw = int(_rng.integers(4, 8))
                p = 2 * (fh + fw) - 4
                if p not in perims and 3 <= p <= 25:
                    perims.append(p)
                    dims.append((fh, fw))
                    break

        # 目标精灵面积=对应周长
        areas = list(perims)

        grid = np.zeros((H, W), dtype=int)
        frames = []
        used = np.zeros((H, W), dtype=bool)

        for color, (fh, fw), p in zip(fc, dims, perims):
            placed = False
            for _ in range(120):
                r0 = int(_rng.integers(0, H - fh))
                c0 = int(_rng.integers(0, W - fw))
                r1 = r0 + fh - 1
                c1 = c0 + fw - 1
                area_slice = np.s_[r0 : r1 + 1, c0 : c1 + 1]
                pad = np.s_[max(0, r0 - 1) : min(H, r1 + 2), max(0, c0 - 1) : min(W, c1 + 2)]
                if used[area_slice].any():
                    continue
                used[pad] = True
                for r in range(r0, r1 + 1):
                    grid[r, c0] = color
                    grid[r, c1] = color
                for c in range(c0, c1 + 1):
                    grid[r0, c] = color
                    grid[r1, c] = color
                frames.append((color, r0, r1, c0, c1, p))
                placed = True
                break
            if not placed:
                break
        if len(frames) != k:
            continue

        # 精灵体必须置于框架外部
        sprite_targets = []
        frame_mask = np.zeros((H, W), dtype=bool)
        for c, r0, r1, c0, c1, _ in frames:
            frame_mask[r0 : r1 + 1, c0 : c1 + 1] = True

        for color, area in zip(sc, areas):
            placed = False
            for _ in range(600):
                r0 = int(_rng.integers(0, H))
                c0 = int(_rng.integers(0, W))
                if frame_mask[r0, c0] or grid[r0, c0] != 0:
                    continue
                cells = [(r0, c0)]
                q = [(r0, c0)]
                comp = {(r0, c0)}
                while q and len(comp) < area:
                    cr, cc = q.pop(0)
                    moves = [(1, 0), (-1, 0), (0, 1), (0, -1)]
                    _rng.shuffle(moves)
                    for dr, dc in moves:
                        nr, nc = cr + dr, cc + dc
                        if not (0 <= nr < H and 0 <= nc < W):
                            continue
                        if frame_mask[nr, nc] or grid[nr, nc] != 0 or (nr, nc) in comp:
                            continue
                        comp.add((nr, nc))
                        q.append((nr, nc))
                        if len(comp) == area:
                            break
                    if len(comp) == area:
                        break
                if len(comp) != area:
                    continue
                for rr, cc in comp:
                    grid[rr, cc] = color
                sprite_targets.append((color, area, sorted(comp)))
                placed = True
                break
            if not placed:
                break

        if len(sprite_targets) != k:
            continue

        # 确保至少一色精灵在每个框之外且非空
        return grid


def main(grid):
    """
    输入: 18x18，含空心矩形框架(5/7/9)与单色连通精灵体(1/2/3)。

    输出: 找到精灵体像素数 S 与框架边框像素数 P 的唯一一一匹配，
    将每个精灵体缩放映射到配对框架内并绘制。

    算法: 1) 识别框架和精灵体并计算 P、S；
    2) 按 S==P 配对；
    3) 把精灵体边界映射到框内坐标，连线还原并填充闭合区域；
    4) 擦除原始精灵体。

    复杂度: O(H*W + K*H*W)。
    """
    g = np.array(grid, dtype=int, copy=True)
    h, w = g.shape

    frame_parts = _cc4(g, {5, 7, 9})
    frames = []
    for color, comp in frame_parts:
        rs = [p[0] for p in comp]
        cs = [p[1] for p in comp]
        r0, r1 = min(rs), max(rs)
        c0, c1 = min(cs), max(cs)
        perimeter = len(comp)
        if (r1 - r0 + 1) <= 2 or (c1 - c0 + 1) <= 2:
            continue
        # 简单按“边框厚1”检查
        if perimeter != 2 * ((r1 - r0 + 1) + (c1 - c0 + 1)) - 4:
            continue
        frames.append((perimeter, color, r0, r1, c0, c1))

    sprite_parts = _cc4(g, {1, 2, 3})
    sprites = []
    for color, comp in sprite_parts:
        area = len(comp)
        rs = [p[0] for p in comp]
        cs = [p[1] for p in comp]
        r0, r1 = min(rs), max(rs)
        c0, c1 = min(cs), max(cs)
        sprites.append((area, color, r0, r1, c0, c1, comp))

    mapping = []
    used_frames = set()
    used_sprites = set()
    for s in sprites:
        area = s[0]
        hit = [f for f in frames if f[0] == area and f[1] not in used_frames]
        if not hit:
            continue
        frame = hit[0]
        mapping.append((s, frame))
        used_frames.add(frame[1])
        used_sprites.add(s[1])

    out = np.zeros((h, w), dtype=int)
    # 先保留框架
    for _, color, r0, r1, c0, c1 in frames:
        for c in range(c0, c1 + 1):
            out[r0, c] = color
            out[r1, c] = color
        for r in range(r0, r1 + 1):
            out[r, c0] = color
            out[r, c1] = color

    for sprite, frame in mapping:
        s_area, s_col, sr0, sr1, sc0, sc1, s_cells = sprite
        _, f_col, fr0, fr1, fc0, fc1, = frame
        si0, si1 = sr0, sr1
        sj0, sj1 = sc0, sc1
        fi0, fi1 = fr0 + 1, fr1 - 1
        fj0, fj1 = fc0 + 1, fc1 - 1
        if fi0 > fi1 or fj0 > fj1:
            continue

        hs, ws = si1 - si0 + 1, sj1 - sj0 + 1
        hi, wi = fi1 - fi0 + 1, fj1 - fj0 + 1

        # 映射后的坐标集合（按极端点对应规则）
        mapped = {}
        for sr, sc in s_cells:
            tr = fi0 if hs == 1 else fi0 + int(round((sr - si0) * (hi - 1) / (hs - 1)))
            tc = fj0 if ws == 1 else fj0 + int(round((sc - sj0) * (wi - 1) / (ws - 1)))
            tr = int(np.clip(tr, fi0, fi1))
            tc = int(np.clip(tc, fj0, fj1))
            mapped[(sr, sc)] = (tr, tc)

        # 先连线（保持原有形状拓扑）
        for sr, sc in s_cells:
            for dr, dc in ((1, 0), (0, 1)):
                nr, nc = sr + dr, sc + dc
                if (nr, nc) not in mapped:
                    continue
                _draw_line(mapped[(sr, sc)], mapped[(nr, nc)], s_col, out)

        # 封闭区域填充
        if len(s_cells) == 0:
            continue
        fr = np.zeros((h, w), dtype=bool)
        for rr, cc in mapped.values():
            fr[rr, cc] = True
        in_mask = _fill_closed(fr[fi0 : fi1 + 1, fj0 : fj1 + 1], s_col)
        for rr in range(fi0, fi1 + 1):
            for cc in range(fj0, fj1 + 1):
                if in_mask[rr - fi0, cc - fj0]:
                    out[rr, cc] = s_col

    return out
