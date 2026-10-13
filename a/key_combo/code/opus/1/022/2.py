import random
import numpy as np


def _components(mask):
    h, w = mask.shape
    seen = np.zeros((h, w), bool)
    comps = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp = []
            while stack:
                x, y = stack.pop()
                comp.append((x, y))
                for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < h and 0 <= ny < w and mask[nx, ny] and not seen[nx, ny]:
                        seen[nx, ny] = True
                        stack.append((nx, ny))
            comps.append(comp)
    return comps


def _line_positions(grid, axis):
    counts = np.sum(grid == 5, axis=axis)
    limit = max(3, int(max(counts) * 0.55))
    return [int(i) for i, n in enumerate(counts) if n >= limit]


def main(grid: np.ndarray) -> np.ndarray:
    grid = np.array(grid, copy=True)
    rows = _line_positions(grid, 1)
    cols = _line_positions(grid, 0)
    if len(rows) < 2 or len(cols) < 2:
        return grid
    out = grid.copy()
    area_color = {2: 2, 3: 1, 4: 3, 5: 4}
    for i in range(len(rows) - 1):
        for j in range(len(cols) - 1):
            r0, r1 = rows[i], rows[i + 1]
            c0, c1 = cols[j], cols[j + 1]
            if r1 <= r0 + 1 or c1 <= c0 + 1:
                continue
            border = []
            border.extend((r0, c) for c in range(c0 + 1, c1))
            border.extend((r1, c) for c in range(c0 + 1, c1))
            border.extend((r, c0) for r in range(r0 + 1, r1))
            border.extend((r, c1) for r in range(r0 + 1, r1))
            intact = all(grid[r, c] == 5 for r, c in border)
            interior = grid[r0 + 1:r1, c0 + 1:c1]
            for comp in _components((interior != 0) & (interior != 5)):
                area = len(comp)
                for rr, cc in comp:
                    out[r0 + 1 + rr, c0 + 1 + cc] = area_color.get(area, 0) if intact else 0
    return out


def _shape(area, rh, cw):
    for _ in range(200):
        cells = {(0, 0)}
        while len(cells) < area:
            r, c = random.choice(tuple(cells))
            dr, dc = random.choice([(1, 0), (-1, 0), (0, 1), (0, -1)])
            cells.add((r + dr, c + dc))
        minr = min(r for r, _ in cells)
        minc = min(c for _, c in cells)
        pts = [(r - minr, c - minc) for r, c in cells]
        if max(r for r, _ in pts) < rh and max(c for _, c in pts) < cw:
            return pts
    shapes = {
        2: [(0, 0), (0, 1)],
        3: [(0, 0), (1, 0), (1, 1)],
        4: [(0, 0), (0, 1), (1, 0), (1, 1)],
        5: [(0, 0), (0, 1), (1, 1), (2, 1), (2, 2)],
    }
    return shapes[area]


def generate_input() -> np.ndarray:
    cell_h = random.randint(3, 4)
    cell_w = random.randint(3, 4)
    nr, nc = random.randint(3, 4), random.randint(3, 4)
    h, w = nr * cell_h + nr + 1, nc * cell_w + nc + 1
    grid = np.zeros((h, w), dtype=int)
    rows = [i * (cell_h + 1) for i in range(nr + 1)]
    cols = [i * (cell_w + 1) for i in range(nc + 1)]
    grid[rows, :] = 5
    grid[:, cols] = 5
    cells = [(i, j) for i in range(nr) for j in range(nc)]
    broken = set(random.sample(cells, max(2, len(cells) // 4)))
    for i, j in broken:
        r0, r1 = rows[i], rows[i + 1]
        c0, c1 = cols[j], cols[j + 1]
        side = random.choice(["top", "bottom", "left", "right"])
        if side in ("top", "bottom"):
            r = r0 if side == "top" else r1
            c = random.randint(c0 + 1, c1 - 1)
        else:
            c = c0 if side == "left" else c1
            r = random.randint(r0 + 1, r1 - 1)
        grid[r, c] = 0
    object_cells = random.sample(cells, min(len(cells), random.randint(6, len(cells))))
    if sum(c not in broken for c in object_cells) < 3:
        object_cells += [c for c in cells if c not in broken][:3]
    if sum(c in broken for c in object_cells) < 2:
        object_cells += [c for c in broken][:2]
    used = set()
    for i, j in object_cells:
        if (i, j) in used:
            continue
        used.add((i, j))
        r0, r1 = rows[i], rows[i + 1]
        c0, c1 = cols[j], cols[j + 1]
        area = random.randint(2, 5)
        pts = _shape(area, cell_h, cell_w)
        rr = random.randint(r0 + 1, r1 - max(r for r, _ in pts) - 1)
        cc = random.randint(c0 + 1, c1 - max(c for _, c in pts) - 1)
        color = random.choice([1, 2, 3, 4])
        for dr, dc in pts:
            grid[rr + dr, cc + dc] = color
    return grid
