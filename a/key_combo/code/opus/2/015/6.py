import numpy as np

_RNG = np.random.default_rng(20260707)

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
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nx, ny = x + dr, y + dc
                    if 0 <= nx < h and 0 <= ny < w and not seen[nx, ny] and grid[nx, ny] == color:
                        seen[nx, ny] = True
                        st.append((nx, ny))
            comps.append(cells)
    return comps


def _parse_group(cells_set):
    rows = [r for r, _ in cells_set]
    cols = [c for _, c in cells_set]
    minr, maxr = min(rows), max(rows)
    minc, maxc = min(cols), max(cols)

    if minr < 0 or minc < 0:
        return None

    for br in range(minr, maxr):
        for bc in range(minc, maxc):
            body = {(br, bc), (br + 1, bc), (br, bc + 1), (br + 1, bc + 1)}
            if not body.issubset(cells_set):
                continue

            # right
            line = []
            cur = (br, bc + 2)
            while cur in cells_set and cur not in body:
                line.append(cur)
                cur = (cur[0], cur[1] + 1)
            if len(line) in (1, 2, 3, 4) and len(cells_set) == len(body | set(line)):
                return {"body": body, "line": set(line), "L": len(line), "dir": (0, 1)}

            # left
            line = []
            cur = (br, bc - 1)
            while cur in cells_set and cur not in body:
                line.append(cur)
                cur = (cur[0], cur[1] - 1)
            if len(line) in (1, 2, 3, 4) and len(cells_set) == len(body | set(line)):
                return {"body": body, "line": set(line), "L": len(line), "dir": (0, -1)}

            # down
            line = []
            cur = (br + 2, bc)
            while cur in cells_set and cur not in body:
                line.append(cur)
                cur = (cur[0] + 1, cur[1])
            if len(line) in (1, 2, 3, 4) and len(cells_set) == len(body | set(line)):
                return {"body": body, "line": set(line), "L": len(line), "dir": (1, 0)}

            # up
            line = []
            cur = (br - 1, bc)
            while cur in cells_set and cur not in body:
                line.append(cur)
                cur = (cur[0] - 1, cur[1])
            if len(line) in (1, 2, 3, 4) and len(cells_set) == len(body | set(line)):
                return {"body": body, "line": set(line), "L": len(line), "dir": (-1, 0)}

    return None


def main(grid):
    """
1) 在每个红色组块中识别 2x2 主体、杆段与方向。
2) 杆长 L=杆段长度(1..4)决定其在顶部调色板中索引的颜色。
3) 将主体连同杆段移除后，主体整体平移 L 格，贴近锚定物方向。
4) 平移后主体填充为对应颜色，其它像素保持不变。
"""

    out = np.array(grid, dtype=int, copy=True)
    h, w = out.shape
    palette = [v for v in out[0, :4] if v != 0]

    for comp in _components(out, 2):
        comp_set = set(comp)
        info = _parse_group(comp_set)
        if info is None:
            continue

        L = info["L"]
        if L < 1 or L > len(palette):
            continue

        body = info["body"]
        line = info["line"]
        dr, dc = info["dir"]

        for r, c in body:
            out[r, c] = 0
        for r, c in line:
            out[r, c] = 0

        br = min(r for r, _ in body)
        bc = min(c for _, c in body)
        nr = br + dr * L
        nc = bc + dc * L

        color = int(palette[L - 1])
        for rr in range(2):
            for cc in range(2):
                r2, c2 = nr + rr, nc + cc
                if 0 <= r2 < h and 0 <= c2 < w:
                    out[r2, c2] = color

    return out


def generate_input(seed=None):
    rng = _rng(seed)
    h = w = 16

    while True:
        grid = np.zeros((h, w), dtype=int)
        occ = np.zeros((h, w), dtype=bool)

        palette_colors = [1, 3, 4, 7, 6]
        cols = list(rng.choice(palette_colors, size=4, replace=False))
        for c in range(4):
            grid[0, c] = cols[c]

        n = int(rng.integers(2, 5))
        length_set = set()

        for _ in range(n):
            placed = False
            for _ in range(300):
                drc = int(rng.integers(4))
                L = int(rng.integers(1, 5))

                if drc in (0, 1):
                    r = int(rng.integers(0, h - 2))
                    c = int(rng.integers(0, w - 1))
                if drc in (2, 3):
                    r = int(rng.integers(0, h - 1))
                    c = int(rng.integers(0, w - 2))

                # 为了更简单，统一以 body 左上角为(br,bc)
                if drc == 0:  # right
                    if r + 2 > h - 1 or c + 3 + L > w - 1:
                        continue
                    body = {(r, c), (r, c + 1), (r + 1, c), (r + 1, c + 1)}
                    line = {(r, c + 2 + t) for t in range(L)}
                    anchor = {(r, c + 2 + L + s) for s in (0, 1)} | {(r + 1, c + 2 + L + s) for s in (0, 1)}
                elif drc == 1:  # left
                    if c - (L + 2) < 0 or c + 1 >= w:
                        continue
                    body = {(r, c), (r, c + 1), (r + 1, c), (r + 1, c + 1)}
                    line = {(r, c - 1 - t) for t in range(L)}
                    anchor = {(r, c - 2 - L + s) for s in (0, 1)} | {(r + 1, c - 2 - L + s) for s in (0, 1)}
                elif drc == 2:  # down
                    if r + 3 + L > h - 1 or c + 1 > w - 1:
                        continue
                    body = {(r, c), (r, c + 1), (r + 1, c), (r + 1, c + 1)}
                    line = {(r + 2 + t, c) for t in range(L)}
                    anchor = {(r + 2 + L + s, c) for s in (0, 1)} | {(r + 2 + L + s, c + 1) for s in (0, 1)}
                else:  # up
                    if r - (L + 2) < 0 or c + 1 > w - 1:
                        continue
                    body = {(r, c), (r, c + 1), (r + 1, c), (r + 1, c + 1)}
                    line = {(r - 1 - t, c) for t in range(L)}
                    anchor = {(r - 2 - L + s, c) for s in (0, 1)} | {(r - 2 - L + s, c + 1) for s in (0, 1)}

                cells = body | line | anchor
                if any(x < 0 or y < 0 or x >= h or y >= w for x, y in cells):
                    continue
                if any(occ[x, y] for x, y in cells):
                    continue

                for x, y in body:
                    grid[x, y] = 2
                for x, y in line:
                    grid[x, y] = 2
                for x, y in anchor:
                    grid[x, y] = 5
                for x, y in cells:
                    occ[x, y] = True

                length_set.add(L)
                placed = True
                break
            if not placed:
                continue

        if len(length_set) >= 2:
            return grid


if __name__ == "__main__":
    pass
