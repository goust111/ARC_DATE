import random
from collections import defaultdict, deque
import numpy as np

AZURE = 8


def components(mask: np.ndarray):
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps = []
    for r in range(h):
        for c in range(w):
            if mask[r, c] and not seen[r, c]:
                q = deque([(r, c)])
                seen[r, c] = True
                cells = []
                while q:
                    x, y = q.popleft()
                    cells.append((x, y))
                    for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                        nx, ny = x + dx, y + dy
                        if 0 <= nx < h and 0 <= ny < w and mask[nx, ny] and not seen[nx, ny]:
                            seen[nx, ny] = True
                            q.append((nx, ny))
                comps.append(cells)
    return comps


def bbox(cells):
    rs = [r for r, _ in cells]
    cs = [c for _, c in cells]
    return min(rs), max(rs), min(cs), max(cs)


def main(grid: np.ndarray) -> np.ndarray:
    grid = np.array(grid, dtype=int)
    groups = defaultdict(list)
    for cells in components(grid != AZURE):
        r0, r1, c0, c1 = bbox(cells)
        touch = int(any(r == 0 for r, _ in cells)) + int(any(r == grid.shape[0] - 1 for r, _ in cells))
        touch += int(any(c == 0 for _, c in cells)) + int(any(c == grid.shape[1] - 1 for _, c in cells))
        crop = grid[r0 : r1 + 1, c0 : c1 + 1].copy()
        groups[touch].append((c0, r0, crop))
    order = sorted(groups.items(), key=lambda kv: (-len(kv[1]), -kv[0]))
    strips = []
    for _, items in order:
        items.sort(key=lambda t: (t[0], t[1]))
        height = max(c.shape[0] for _, _, c in items)
        width = sum(c.shape[1] for _, _, c in items) + max(0, len(items) - 1)
        strip = np.full((height, width), AZURE, dtype=int)
        x = 0
        for _, _, crop in items:
            strip[: crop.shape[0], x : x + crop.shape[1]] = crop
            x += crop.shape[1] + 1
        strips.append(strip)
    out_h = sum(s.shape[0] for s in strips) + max(0, len(strips) - 1)
    out_w = max(s.shape[1] for s in strips)
    out = np.full((out_h, out_w), AZURE, dtype=int)
    r = 0
    for strip in strips:
        out[r : r + strip.shape[0], : strip.shape[1]] = strip
        r += strip.shape[0] + 1
    return out


def make_shape(h: int, w: int, color: int) -> np.ndarray:
    mask = np.zeros((h, w), dtype=int)
    cells = [(random.randrange(h), random.randrange(w))]
    mask[cells[0]] = color
    target = random.randint(max(3, h), min(h * w - 1, h * w))
    while np.count_nonzero(mask) < target:
        r, c = random.choice(cells)
        nr, nc = r + random.choice([-1, 1, 0, 0]), c + random.choice([0, 0, -1, 1])
        if 0 <= nr < h and 0 <= nc < w:
            if mask[nr, nc] == 0:
                mask[nr, nc] = color
                cells.append((nr, nc))
    return mask


def generate_input() -> np.ndarray:
    h, w = random.randint(18, 24), random.randint(18, 24)
    grid = np.full((h, w), AZURE, dtype=int)
    colors = [1, 2, 3, 4, 5, 6, 7, 9]
    random.shuffle(colors)
    placed = 0
    attempts = 0
    while placed < random.randint(4, 7) and attempts < 200:
        attempts += 1
        sh, sw = random.randint(2, 5), random.randint(2, 5)
        color = colors[placed % len(colors)]
        mask = make_shape(sh, sw, color)
        touch = random.choice([0, 1, 2])
        if touch == 0:
            r0 = random.randint(1, h - sh - 2)
            c0 = random.randint(1, w - sw - 2)
        elif touch == 1:
            side = random.choice(["top", "bottom", "left", "right"])
            if side == "top":
                r0 = 0
                c0 = random.randint(1, w - sw - 2)
            elif side == "bottom":
                r0 = h - sh
                c0 = random.randint(1, w - sw - 2)
            elif side == "left":
                r0 = random.randint(1, h - sh - 2)
                c0 = 0
            else:
                r0 = random.randint(1, h - sh - 2)
                c0 = w - sw
        else:
            side = random.choice([(0, 0), (0, w - sw), (h - sh, 0), (h - sh, w - sw)])
            r0, c0 = side
        patch = grid[max(0, r0 - 1) : min(h, r0 + sh + 1), max(0, c0 - 1) : min(w, c0 + sw + 1)]
        if np.any(patch != AZURE):
            continue
        for r in range(sh):
            for c in range(sw):
                if mask[r, c]:
                    grid[r0 + r, c0 + c] = mask[r, c]
        placed += 1
    return grid
