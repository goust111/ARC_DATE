import numpy as np

_RNG = np.random.default_rng(20260707)
_DIRECTIONS = [(-1, 0), (1, 0), (0, -1), (0, 1)]

def _rng(seed=None):
    return np.random.default_rng(seed) if seed is not None else _RNG


def _components(grid, color):
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps = []
    for r in range(h):
        for c in range(w):
            if grid[r, c] != color or seen[r, c]:
                continue
            st = [(r, c)]
            seen[r, c] = True
            cells = []
            while st:
                x, y = st.pop()
                cells.append((x, y))
                for dr, dc in _DIRECTIONS:
                    nr, nc = x + dr, y + dc
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and grid[nr, nc] == color:
                        seen[nr, nc] = True
                        st.append((nr, nc))
            comps.append(cells)
    return comps


def _bbox(cells):
    rs = [r for r, _ in cells]
    cs = [c for _, c in cells]
    return min(rs), max(rs), min(cs), max(cs)


def main(grid):
    """
1) 扫描红色与栗色的接触边，确定每个红色精灵的推移方向。
2) 按到目标边界距离由近到远平移红色精灵，阶段内仅受已推红精灵边界阻挡。
3) 再扫描蓝色精灵，按从下到上顺序执行重力下落，障碍为标记、推后红色和已沉降蓝色。
4) 返回变换后的整幅数组。
"""

    out = np.array(grid, dtype=int, copy=True)
    h, w = out.shape

    red_stages = []
    for cells in _components(out, 2):
        touch_dirs = []
        for r, c in cells:
            for dr, dc in _DIRECTIONS:
                nr, nc = r + dr, c + dc
                if 0 <= nr < h and 0 <= nc < w and out[nr, nc] == 9:
                    touch_dirs.append((dr, dc))
        uniq = set(touch_dirs)
        if len(uniq) != 1:
            continue
        (tr, tc) = next(iter(uniq))
        push_dir = (-tr, -tc)
        minr, maxr, minc, maxc = _bbox(cells)
        if push_dir == (-1, 0):
            dist = minr
        elif push_dir == (1, 0):
            dist = (h - 1) - maxr
        elif push_dir == (0, -1):
            dist = minc
        else:
            dist = (w - 1) - maxc
        red_stages.append((dist, push_dir, cells))

    fixed_red = set()
    red_stages.sort(key=lambda x: x[0])
    for _, push_dir, cells in red_stages:
        moved = set(cells)
        for r, c in moved:
            out[r, c] = 0

        dr, dc = push_dir
        while True:
            nxt = {(r + dr, c + dc) for r, c in moved}
            if any(not (0 <= r < h and 0 <= c < w) for r, c in nxt):
                break
            if any((r, c) in fixed_red for r, c in nxt):
                break
            moved = nxt

        for r, c in moved:
            out[r, c] = 2
        fixed_red.update(moved)

    blue_meta = []
    for cells in _components(out, 1):
        _, maxr, _, _ = _bbox(cells)
        blue_meta.append((maxr, cells))
    blue_meta.sort(reverse=True, key=lambda x: x[0])

    fixed = set((r, c) for r in range(h) for c in range(w) if out[r, c] in (2, 9))
    fixed.update(fixed_red)

    for _, cells in blue_meta:
        moved = set(cells)
        for r, c in moved:
            out[r, c] = 0
        while True:
            nxt = {(r + 1, c) for r, c in moved}
            if any(r >= h for r, _ in nxt):
                break
            if any((r, c) in fixed for r, c in nxt):
                break
            moved = nxt
        for r, c in moved:
            out[r, c] = 1
        fixed.update(moved)

    return out


def _random_shape(rng, start, size, occ, h, w):
    cells = [start]
    frontier = [start]
    occ[start] = True
    while len(cells) < size:
        x, y = frontier[rng.integers(len(frontier))]
        cand = []
        for dr, dc in _DIRECTIONS:
            nx, ny = x + dr, y + dc
            if 0 <= nx < h and 0 <= ny < w and not occ[nx, ny]:
                cand.append((nx, ny))
        if not cand:
            return None
        nx, ny = cand[rng.integers(len(cand))]
        occ[nx, ny] = True
        frontier.append((nx, ny))
        cells.append((nx, ny))
    return cells


def _clear_cells(grid, occ, cells):
    for r, c in cells:
        grid[r, c] = 0
        occ[r, c] = False


def generate_input(seed=None):
    rng = _rng(seed)
    h = w = 14

    while True:
        grid = np.zeros((h, w), dtype=int)
        occ = np.zeros((h, w), dtype=bool)

        # markers
        n_mark = int(rng.integers(1, 4))
        marker_cells = []
        for _ in range(n_mark):
            tries = 0
            while tries < 50:
                r = int(rng.integers(h - 3, h))
                c = int(rng.integers(w))
                if not occ[r, c]:
                    grid[r, c] = 9
                    occ[r, c] = True
                    marker_cells.append((r, c))
                    break
                tries += 1

        if not marker_cells:
            continue

        # reds
        n_red = int(rng.integers(2, 4))
        red_ok = 0
        for _ in range(n_red):
            placed = False
            for _ in range(300):
                marker = marker_cells[int(rng.integers(len(marker_cells)))]
                mr, mc = marker
                dir_idx = int(rng.integers(4))
                dr, dc = _DIRECTIONS[dir_idx]
                sr, sc = mr + dr, mc + dc
                if not (0 <= sr < h and 0 <= sc < w):
                    continue
                if occ[sr, sc]:
                    continue
                size = int(rng.integers(2, 6))
                cells = _random_shape(rng, (sr, sc), size, occ, h, w)
                if cells is None:
                    continue

                touch = []
                for r, c in cells:
                    for ddr, ddc in _DIRECTIONS:
                        nr, nc = r + ddr, c + ddc
                        if 0 <= nr < h and 0 <= nc < w and grid[nr, nc] == 9:
                            touch.append((ddr, ddc))
                if len(set(touch)) != 1:
                    _clear_cells(grid, occ, cells)
                    continue
                for r, c in cells:
                    grid[r, c] = 2
                red_ok += 1
                placed = True
                break
            if not placed:
                continue

        if red_ok < 2:
            continue

        # blues
        n_blue = int(rng.integers(2, 4))
        blue_ok = 0
        for _ in range(n_blue):
            placed = False
            for _ in range(300):
                sr = int(rng.integers(h))
                sc = int(rng.integers(w))
                if occ[sr, sc]:
                    continue
                size = int(rng.integers(2, 6))
                cells = _random_shape(rng, (sr, sc), size, occ, h, w)
                if cells is None:
                    continue

                invalid = False
                for r, c in cells:
                    for dr, dc in _DIRECTIONS:
                        nr, nc = r + dr, c + dc
                        if 0 <= nr < h and 0 <= nc < w and grid[nr, nc] == 9:
                            invalid = True
                            break
                    if invalid:
                        break
                if invalid:
                    _clear_cells(grid, occ, cells)
                    continue

                for r, c in cells:
                    grid[r, c] = 1
                blue_ok += 1
                placed = True
                break
            if not placed:
                continue

        if red_ok >= 2 and blue_ok >= 2:
            return grid


if __name__ == "__main__":
    pass
