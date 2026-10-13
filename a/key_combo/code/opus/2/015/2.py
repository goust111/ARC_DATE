import numpy as np

_RNG = np.random.default_rng(20260707)

def _rng(seed=None):
    return np.random.default_rng(seed) if seed is not None else _RNG


def _components(grid, value):
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps = []
    for r in range(h):
        for c in range(w):
            if grid[r, c] != value or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            cells = []
            while stack:
                x, y = stack.pop()
                cells.append((x, y))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = x + dr, y + dc
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and grid[nr, nc] == value:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(cells)
    return comps


def _bbox(cells):
    rs = [r for r, _ in cells]
    cs = [c for _, c in cells]
    return min(rs), max(rs), min(cs), max(cs)


def main(grid):
    """
1) 识别每个蓝色矩形框架并解析其内部区域。
2) 按内区宽高比选择镜像轴（宽框用垂直轴，高框用水平轴）进行镜像补全。
3) 对镜像后框架内所有红色标记沿垂直于镜像轴方向发射绿射线到障碍前停止。
4) 输出网格将所有红标记重涂为黄，其余像素按规则保留。
"""

    out = np.array(grid, dtype=int, copy=True)
    h, w = out.shape
    ray_sources = []

    for comp in _components(out, 1):
        r0, r1, c0, c1 = _bbox(comp)
        if r1 - r0 < 2 or c1 - c0 < 2:
            continue
        ir0, ir1 = r0 + 1, r1 - 1
        ic0, ic1 = c0 + 1, c1 - 1
        ih = ir1 - ir0 + 1
        iw = ic1 - ic0 + 1
        if ih <= 0 or iw <= 0:
            continue

        is_wide = iw > ih
        is_high = ih > iw
        if not (is_wide or is_high):
            continue

        source = [(r, c) for r in range(ir0, ir1 + 1) for c in range(ic0, ic1 + 1) if out[r, c] != 0]
        mirrored = []
        frame_set = set(comp)

        if is_wide:
            axis = (ic0 + ic1) // 2
            for r, c in source:
                mc = 2 * axis - c
                if mc < ic0 or mc > ic1 or (r, mc) in frame_set:
                    continue
                mirrored.append((r, mc, out[r, c]))
            for _, mc, v in mirrored:
                out[_, mc] = 8 if v != 2 else 2
            for r, c in source:
                if out[r, c] == 2:
                    ray_sources.append((r, c, -1, 0))
                    ray_sources.append((r, c, 1, 0))
            for _, mc, v in mirrored:
                if v == 2:
                    ray_sources.append((_, mc, -1, 0))
                    ray_sources.append((_, mc, 1, 0))
        else:
            axis = (ir0 + ir1) // 2
            for r, c in source:
                mr = 2 * axis - r
                if mr < ir0 or mr > ir1 or (mr, c) in frame_set:
                    continue
                mirrored.append((mr, c, out[r, c]))
            for mr, c, v in mirrored:
                out[mr, c] = 8 if v != 2 else 2
            for mr, c, v in mirrored:
                if v == 2:
                    ray_sources.append((mr, c, 0, -1))
                    ray_sources.append((mr, c, 0, 1))

        # 为原始红标记和镜像红标记增加射线源
        for r, c in [(x, y) for x, y in source if out[x, y] == 2]:
            if is_wide:
                ray_sources.append((r, c, -1, 0))
                ray_sources.append((r, c, 1, 0))
            else:
                ray_sources.append((r, c, 0, -1))
                ray_sources.append((r, c, 0, 1))

    seen_sources = set()
    for sr, sc, dr, dc in ray_sources:
        key = (sr, sc, dr, dc)
        if key in seen_sources:
            continue
        seen_sources.add(key)
        nr, nc = sr + dr, sc + dc
        while 0 <= nr < h and 0 <= nc < w:
            if out[nr, nc] != 0:
                break
            out[nr, nc] = 3
            nr += dr
            nc += dc

    red_pos = np.argwhere(out == 2)
    for r, c in red_pos:
        out[r, c] = 4
    return out


def _try_place_frame(grid, occ, rng, r0, c0, ih, iw):
    h, w = grid.shape
    r1 = r0 + ih + 1
    c1 = c0 + iw + 1
    if r1 >= h - 1 or c1 >= w - 1:
        return False
    pad = grid.shape[0]
    if np.any(occ[max(0, r0 - 1):min(h, r1 + 2), max(0, c0 - 1):min(w, c1 + 2)]):
        return False

    for r in range(r0, r1 + 1):
        grid[r, c0] = 1
        grid[r, c1] = 1
    for c in range(c0, c1 + 1):
        grid[r0, c] = 1
        grid[r1, c] = 1

    ir0, ir1 = r0 + 1, r1 - 1
    ic0, ic1 = c0 + 1, c1 - 1
    ih_inner = ih
    iw_inner = iw
    is_wide = iw_inner > ih_inner

    mark_on_left = bool(rng.integers(2) == 0)
    if is_wide:
        axis = (ic0 + ic1) // 2
        if mark_on_left:
            valid_cols = [c for c in range(ic0, axis + 1)]
        else:
            valid_cols = [c for c in range(axis + 1, ic1 + 1)]
        if not valid_cols:
            return False
        mr = int(rng.integers(ir0, ir1 + 1))
        mc = int(rng.choice(valid_cols))
        for r in range(ir0, ir1 + 1):
            for c in valid_cols:
                if rng.random() < 0.45:
                    grid[r, c] = 8
        grid[mr, mc] = 2
    else:
        axis = (ir0 + ir1) // 2
        if mark_on_left:
            valid_rows = [r for r in range(ir0, axis + 1)]
        else:
            valid_rows = [r for r in range(axis + 1, ir1 + 1)]
        if not valid_rows:
            return False
        mr = int(rng.choice(valid_rows))
        mc = int(rng.integers(ic0, ic1 + 1))
        for r in valid_rows:
            for c in range(ic0, ic1 + 1):
                if rng.random() < 0.45:
                    grid[r, c] = 8
        grid[mr, mc] = 2

    occ[max(0, r0 - 1):min(h, r1 + 2), max(0, c0 - 1):min(w, c1 + 2)] = True
    return True


def generate_input(seed=None):
    rng = _rng(seed)
    H = W = 18

    while True:
        grid = np.zeros((H, W), dtype=int)
        occ = np.zeros((H, W), dtype=bool)
        n = int(rng.integers(2, 4))
        placed = 0

        for _ in range(n):
            ok = False
            for _ in range(400):
                is_wide = bool(rng.integers(2) == 0)
                if is_wide:
                    ih = int(rng.integers(5, 9))
                    if ih <= 0:
                        continue
                    iw = int(rng.integers(ih + 2, min(W - 3, ih * 2) + 1))
                else:
                    iw = int(rng.integers(5, 9))
                    if iw <= 0:
                        continue
                    ih = int(rng.integers(iw + 2, min(H - 3, iw * 2) + 1))
                if ih + 2 >= H - 1 or iw + 2 >= W - 1:
                    continue
                r0 = int(rng.integers(1, H - ih - 1))
                c0 = int(rng.integers(1, W - iw - 1))
                if _try_place_frame(grid, occ, rng, r0, c0, ih, iw):
                    placed += 1
                    ok = True
                    break
            if not ok:
                break

        if placed != n:
            continue
        return grid


if __name__ == "__main__":
    pass
