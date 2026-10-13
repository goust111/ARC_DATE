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


def main(grid: np.ndarray) -> np.ndarray:
    grid = np.asarray(grid)
    h, w = grid.shape
    comps = _components(grid != 0)
    records = []
    for comp in comps:
        vals = [int(grid[r, c]) for r, c in comp]
        body = [v for v in vals if v != 5]
        if not body:
            continue
        color = max(set(body), key=body.count)
        markers = [(r, c) for r, c in comp if grid[r, c] == 5]
        if not markers:
            continue
        cells = set(comp)
        count = 0
        for r, c in comp:
            if grid[r, c] == 5:
                continue
            ok = True
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    if (r + dr, c + dc) not in cells:
                        ok = False
            count += int(ok)
        records.append((markers[0][0], color, count))
    records.sort()
    out = np.zeros((h, len(records)), dtype=int)
    for c, (_, color, count) in enumerate(records):
        out[max(0, h - count):h, c] = color
    return out


def generate_input() -> np.ndarray:
    h, w = random.randint(14, 20), random.randint(16, 22)
    grid = np.zeros((h, w), dtype=int)
    colors = random.sample([1, 2, 3, 4, 6], random.randint(3, 5))
    anchors = [(1, 1), (1, w - 7), (h // 2 - 2, 2), (h // 2, w - 8), (h - 7, w // 2 - 3)]
    random.shuffle(anchors)
    marker_rows = random.sample(range(3, h - 3), len(colors))
    for idx, color in enumerate(colors):
        ar, ac = anchors[idx]
        bh, bw = random.choice([(3, 3), (3, 4), (4, 3)])
        for r in range(ar, ar + bh):
            for c in range(ac, ac + bw):
                grid[r, c] = color
        for _ in range(random.randint(1, 3)):
            side = random.choice(["up", "down", "left", "right"])
            if side == "up" and ar > 0:
                grid[ar - 1, random.randint(ac, ac + bw - 1)] = color
            elif side == "down" and ar + bh < h:
                grid[ar + bh, random.randint(ac, ac + bw - 1)] = color
            elif side == "left" and ac > 0:
                grid[random.randint(ar, ar + bh - 1), ac - 1] = color
            elif side == "right" and ac + bw < w:
                grid[random.randint(ar, ar + bh - 1), ac + bw] = color
        mr = marker_rows[idx]
        rr = min(max(ar + 1, mr), ar + bh - 2)
        cc = ac + random.randint(1, bw - 2)
        grid[rr, cc] = 5
    return grid
