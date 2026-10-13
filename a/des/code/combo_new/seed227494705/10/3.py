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
            q = [(r, c)]
            seen[r, c] = True
            pts = []
            while q:
                x, y = q.pop()
                pts.append((x, y))
                for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < h and 0 <= ny < w and mask[nx, ny] and not seen[nx, ny]:
                        seen[nx, ny] = True
                        q.append((nx, ny))
            comps.append(pts)
    return comps


def main(grid: np.ndarray) -> np.ndarray:
    """Recolor each vertical panel's object by using maroon marker counts as palette indices.

    Input structure: a black grid with a palette column at the left, full grey
    vertical separators, and panels containing one colored object plus isolated
    maroon count markers.
    Rule: in each panel count the maroon markers, use that count as a one-based
    row index into the left palette, recolor every non-marker object pixel in
    the panel to the selected palette color, and remove the maroon markers.
    Key insight: marker quantity is not a color itself; it is an index into a
    separate visible palette.
    """
    out = grid.copy()
    grey, maroon = 5, 9
    palette = [int(grid[r, 0]) for r in range(grid.shape[0]) if grid[r, 0] not in (0, grey, maroon)]
    sep_cols = [c for c in range(grid.shape[1]) if np.all(grid[:, c] == grey)]
    bounds = sep_cols + [grid.shape[1]]
    prev = 1
    for b in bounds:
        if b <= prev:
            prev = b + 1
            continue
        c1, c2 = prev, b
        panel = grid[:, c1:c2]
        markers = panel == maroon
        k = int(markers.sum())
        if k and k <= len(palette):
            obj = (panel != 0) & (panel != grey) & (panel != maroon)
            out[:, c1:c2][obj] = palette[k - 1]
            out[:, c1:c2][markers] = 0
        prev = b + 1
    return out


def generate_input() -> np.ndarray:
    h = random.randint(10, 16)
    n = random.randint(3, 4)
    panel_w = random.randint(4, 5)
    w = 2 + n * panel_w + n
    grid = np.zeros((h, w), dtype=int)
    palette = random.sample([1, 2, 3, 4, 6, 7, 8], n)
    for i, color in enumerate(palette):
        grid[i + 1, 0] = color
    sep = 1
    for _ in range(n + 1):
        grid[:, sep] = 5
        sep += panel_w + 1
    for p in range(n):
        c1 = 2 + p * (panel_w + 1)
        color = random.choice([x for x in [1, 2, 3, 4, 6, 7, 8] if x not in (9,)])
        top = random.randint(1, h - 5)
        left = c1 + random.randint(0, max(0, panel_w - 3))
        shape = random.choice([[(0, 0), (1, 0), (1, 1)], [(0, 1), (1, 0), (1, 1), (2, 1)], [(0, 0), (0, 1), (1, 1)]])
        for dr, dc in shape:
            if top + dr < h and left + dc < c1 + panel_w:
                grid[top + dr, left + dc] = color
        k = random.randint(1, n)
        placed = 0
        tries = 0
        while placed < k and tries < 200:
            tries += 1
            r = random.randint(0, h - 1)
            c = random.randint(c1, c1 + panel_w - 1)
            if grid[r, c] == 0:
                grid[r, c] = 9
                placed += 1
    return grid
