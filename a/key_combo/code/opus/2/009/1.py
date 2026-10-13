import random
import numpy as np


def _anchors(grid):
    h, w = grid.shape
    found = []
    used = set()
    for r in range(h - 2):
        for c in range(w - 2):
            block = grid[r:r + 3, c:c + 3]
            vals = block.reshape(-1)
            if np.count_nonzero(vals == 5) != 1:
                continue
            colors = [int(x) for x in vals if x not in (0, 5)]
            if len(colors) != 8 or len(set(colors)) != 1:
                continue
            color = colors[0]
            key = (color, r, c)
            if key not in used:
                gr, gc = np.argwhere(block == 5)[0]
                found.append({"color": color, "top": r, "left": c, "gray": (r + gr, c + gc)})
                used.add(key)
    found.sort(key=lambda a: a["left"])
    return found


def main(grid: np.ndarray) -> np.ndarray:
    grid = np.array(grid, copy=True)
    groups = _anchors(grid)
    crops = []
    for a in groups:
        color = a["color"]
        top, left = a["top"], a["left"]
        center = (top + 1, left + 1)
        block_cells = {(r, c) for r in range(top, top + 3) for c in range(left, left + 3)}
        counts = {}
        for r, c in zip(*np.where(grid == color)):
            if (r, c) in block_cells:
                continue
            dr = -1 if r < top else 1 if r > top + 2 else 0
            dc = -1 if c < left else 1 if c > left + 2 else 0
            if (dr, dc) != (0, 0):
                counts[(dr, dc)] = counts.get((dr, dc), 0) + 1

        canvas = np.zeros_like(grid)
        for r, c in block_cells:
            canvas[r, c] = 5 if (r, c) == a["gray"] else color
        for (dr, dc), n in counts.items():
            for i in range(n):
                rr = center[0] + dr * (2 + i)
                cc = center[1] + dc * (2 + i)
                if 0 <= rr < grid.shape[0] and 0 <= cc < grid.shape[1]:
                    canvas[rr, cc] = color

        pts = np.argwhere((canvas == color) | (canvas == 5))
        r0, c0 = pts.min(axis=0)
        r1, c1 = pts.max(axis=0)
        crop = canvas[r0:r1 + 1, c0:c1 + 1]
        gr, gc = np.argwhere(crop == 5)[0]
        crops.append((crop, int(gr), color))

    above = max(gray_row for _, gray_row, _ in crops)
    below = max(crop.shape[0] - gray_row - 1 for crop, gray_row, _ in crops)
    width = sum(crop.shape[1] for crop, _, _ in crops)
    out = np.zeros((above + below + 1, width), dtype=grid.dtype)
    x = 0
    for crop, gray_row, color in crops:
        y = above - gray_row
        part = crop.copy()
        part[part == 5] = color
        out[y:y + part.shape[0], x:x + part.shape[1]] = part
        x += part.shape[1]
    return out


def generate_input() -> np.ndarray:
    rng = random.Random()
    dirs = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]
    for _ in range(1000):
        h, w = 22, 26
        grid = np.zeros((h, w), dtype=int)
        n = rng.choice([3, 4])
        colors = [1, 2, 3, 4][:n]
        lefts = [2, 8, 14, 20][:n]
        rows = [rng.randint(6, 14) for _ in range(n)]
        occupied = set()
        ok = True
        for i, color in enumerate(colors):
            top, left = rows[i], lefts[i]
            gray_pos = (top + (i % 3), left + rng.randrange(3))
            block = {(r, c) for r in range(top, top + 3) for c in range(left, left + 3)}
            if occupied & block:
                ok = False
                break
            for r, c in block:
                grid[r, c] = 5 if (r, c) == gray_pos else color
                occupied.add((r, c))
            center = (top + 1, left + 1)
            chosen = rng.sample(dirs, rng.randint(2, 4))
            total = rng.randint(2, 5)
            rays = [rng.choice(chosen) for _ in range(total)]
            for dr, dc in rays:
                gap = rng.randint(2, 5)
                rr = center[0] + dr * (2 + gap)
                cc = center[1] + dc * (2 + gap)
                if not (0 <= rr < h and 0 <= cc < w) or grid[rr, cc] != 0:
                    ok = False
                    break
                grid[rr, cc] = color
            if not ok:
                break
        if ok:
            return grid
    return generate_input()
