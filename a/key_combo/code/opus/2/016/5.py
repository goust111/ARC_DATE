import numpy as np

_RNG = np.random.default_rng(20260105)


def _components(mask: np.ndarray):
    h, w = mask.shape
    vis = np.zeros_like(mask, dtype=bool)
    comps = []
    dirs = ((1, 0), (-1, 0), (0, 1), (0, -1))
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or vis[r, c]:
                continue
            q = [(r, c)]
            vis[r, c] = True
            comp = []
            qi = 0
            while qi < len(q):
                cr, cc = q[qi]
                qi += 1
                comp.append((cr, cc))
                for dr, dc in dirs:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not vis[nr, nc]:
                        vis[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def main(grid):
    """Input:
    约 7x7 含边缘颜色标记和若干橙色对象。

    Returns:
    镜像-原始交替拼接 2x2 后按外边接触侧重涂色。

    Algorithm:
    先构造镜像拼图并清除侧标记，再统计每个橙色对象接触外边缘的哪一侧以决定颜色。

    Complexity:
    O(HW) 的组件识别与接触统计。
    """
    g = np.array(grid, dtype=int)
    h, w = g.shape

    lr = np.fliplr(g)
    t = np.concatenate([np.concatenate([lr, g], axis=1), np.concatenate([g, lr], axis=1)], axis=0)

    out = t.copy()
    for color in (1, 2, 3, 4):
        out[out == color] = 0

    mask = out == 7
    color_map = {"top": 2, "right": 1, "bottom": 3, "left": 4}
    dirs = ((-1, 0, "top"), (1, 0, "bottom"), (0, -1, "left"), (0, 1, "right"))
    H, W = out.shape

    for comp in _components(mask):
        counts = {"top": 0, "bottom": 0, "left": 0, "right": 0}
        for r, c in comp:
            for dr, dc, side in dirs:
                nr, nc = r + dr, c + dc
                if 0 <= nr < H and 0 <= nc < W:
                    continue
                counts[side] += 1

        contact = [(v, k) for k, v in counts.items() if v > 0]
        if not contact:
            continue
        max_cnt = max(v for v, _ in contact)
        candidates = [k for v, k in contact if v == max_cnt]
        if "top" in candidates:
            side = "top"
        elif "bottom" in candidates:
            side = "bottom"
        elif "left" in candidates:
            side = "left"
        else:
            side = "right"

        new_color = color_map[side]
        for r, c in comp:
            out[r, c] = new_color

    return out


def _walk(start):
    size = 4
    cells = [start]
    frontier = [start]
    seen = {start}
    while len(cells) < size:
        r, c = frontier[-1]
        cand = []
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nr, nc = r + dr, c + dc
            if 0 <= nr < 7 and 0 <= nc < 7 and (nr, nc) not in seen:
                cand.append((nr, nc))
        if not cand:
            break
        nn = cand[int(_RNG.integers(0, len(cand)))]
        cells.append(nn)
        seen.add(nn)
        frontier.append(nn)
    return cells


def generate_input():
    """随机且稳定地生成输入：7x7，边缘各有一个侧标记，若干橙色对象。"""
    g = np.zeros((7, 7), dtype=int)
    # 边缘标记
    g[0, int(_RNG.integers(1, 6))] = 2
    g[int(_RNG.integers(1, 6)), 6] = 1
    g[6, int(_RNG.integers(1, 6))] = 3
    g[int(_RNG.integers(1, 6)), 0] = 4

    obj_count = int(_RNG.integers(2, 5))
    used = set(zip(*np.where(g == 0)))
    for _ in range(obj_count):
        size = int(_RNG.integers(2, 5))
        # 选取起点
        idx = int(_RNG.integers(0, len(used)))
        r, c = list(used)[idx]
        comp = _walk((r, c))
        if len(comp) < 2:
            continue
        for rr, cc in comp[:size]:
            g[rr, cc] = 7
            used.discard((rr, cc))

    return g
