import random
import numpy as np


def _black_components(grid):
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps = []
    for r in range(h):
        for c in range(w):
            if grid[r, c] != 0 or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp = []
            touch = False
            while stack:
                x, y = stack.pop()
                comp.append((x, y))
                touch = touch or x in (0, h - 1) or y in (0, w - 1)
                for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < h and 0 <= ny < w and grid[nx, ny] == 0 and not seen[nx, ny]:
                        seen[nx, ny] = True
                        stack.append((nx, ny))
            if not touch:
                comps.append(comp)
    return comps


def main(grid: np.ndarray) -> np.ndarray:
    out = grid.copy()
    cells = []
    for comp in _black_components(grid):
        rs, cs = [p[0] for p in comp], [p[1] for p in comp]
        r0, r1, c0, c1 = min(rs), max(rs), min(cs), max(cs)
        cells.append({"pixels": comp, "box": (r0, r1, c0, c1), "cy": (r0 + r1) / 2, "cx": (c0 + c1) / 2})
    rows = []
    for cell in sorted(cells, key=lambda x: x["cy"]):
        for row in rows:
            if abs(row[0]["cy"] - cell["cy"]) <= 1:
                row.append(cell)
                break
        else:
            rows.append([cell])
    for row in rows:
        control = None
        for cell in row:
            r0, r1, c0, c1 = cell["box"]
            vals = grid[r0:r1 + 1, c0:c1 + 1]
            found = [int(v) for v in np.unique(vals) if v not in (0, 5, 6)]
            if found:
                control = found[0]
                break
        if control is None:
            continue
        for cell in row:
            r0, r1, c0, c1 = cell["box"]
            region = (grid[r0:r1 + 1, c0:c1 + 1] == 6)
            out[r0:r1 + 1, c0:c1 + 1][region] = control
    return out


def generate_input() -> np.ndarray:
    rows = random.randint(3, 4)
    cols = random.randint(3, 4)
    heights = [random.randint(2, 4) for _ in range(rows)]
    widths = [random.randint(2, 4) for _ in range(cols)]
    h = sum(heights) + rows + 3
    w = sum(widths) + cols + 3
    g = np.zeros((h, w), dtype=int)
    rlines = [1]
    for ht in heights:
        rlines.append(rlines[-1] + ht + 1)
    clines = [1]
    for wd in widths:
        clines.append(clines[-1] + wd + 1)
    for r in rlines:
        g[r, clines[0]:clines[-1] + 1] = 5
    for c in clines:
        g[rlines[0]:rlines[-1] + 1, c] = 5
    control_colors = random.sample([1, 2, 3, 4, 7, 8, 9], rows)
    for i in range(rows):
        r0, r1 = rlines[i] + 1, rlines[i + 1] - 1
        control_col = 0
        c0, c1 = clines[control_col] + 1, clines[control_col + 1] - 1
        g[random.randint(r0, r1), random.randint(c0, c1)] = control_colors[i]
        for j in range(1, cols):
            if random.random() < 0.75 or j == 1:
                a, b = clines[j] + 1, clines[j + 1] - 1
                for _ in range(random.randint(1, 3)):
                    g[random.randint(r0, r1), random.randint(a, b)] = 6
    return g
