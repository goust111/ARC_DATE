import random
import numpy as np


SHAPES = [
    (((0, 0), (0, 1), (1, 0), (1, 1)), 1),
    (((0, 1), (1, 0), (1, 1)), 2),
    (((0, 0), (0, 1), (1, 1)), 3),
    (((0, 0), (1, 0), (1, 1)), 4),
]


def _cover(mask):
    h, w = mask.shape
    used = np.zeros_like(mask, dtype=bool)
    placed = []

    def search():
        rest = np.argwhere(mask & ~used)
        if len(rest) == 0:
            return True
        r, c = map(int, rest[0])
        options = []
        for t, (shape, color) in enumerate(SHAPES):
            for dr, dc in shape:
                ar, ac = r - dr, c - dc
                pts = [(ar + sr, ac + sc) for sr, sc in shape]
                if all(0 <= x < h and 0 <= y < w and mask[x, y] and not used[x, y] for x, y in pts):
                    options.append((t, color, pts))
        for t, color, pts in options:
            for x, y in pts:
                used[x, y] = True
            placed.append((t, color, pts))
            if search():
                return True
            placed.pop()
            for x, y in pts:
                used[x, y] = False
        return False

    if not search():
        raise ValueError("no exact cover")
    return placed


def main(grid: np.ndarray) -> np.ndarray:
    h, w = grid.shape
    top_h = h // 2
    out = grid.copy()
    placed = _cover(grid[:top_h] == 5)
    counts = [0, 0, 0, 0]
    colors = [1, 2, 3, 4]
    out[:top_h][grid[:top_h] == 5] = 0
    for t, color, pts in placed:
        counts[t] += 1
        for r, c in pts:
            out[r, c] = color
    max_t = max(range(4), key=lambda i: counts[i])
    min_t = min(range(4), key=lambda i: counts[i])
    rect_w = counts[max_t]
    rect_h = counts[min_t]
    color = colors[max_t]
    out[top_h:, :] = 0
    r0 = h - rect_h
    out[r0:h, 0:rect_w] = color
    for i in range(min(rect_h, rect_w)):
        out[h - 1 - i, i] = 4
        out[h - 1 - i, rect_w - 1 - i] = 4
    return out


def generate_input() -> np.ndarray:
    for _ in range(2000):
        grid = np.zeros((16, 16), dtype=int)
        coarse = [(r, c) for r in range(4) for c in range(8)]
        start = random.choice(coarse)
        cells = {start}
        while len(cells) < random.randint(9, 13):
            r, c = random.choice(list(cells))
            nr, nc = random.choice([(r + 1, c), (r - 1, c), (r, c + 1), (r, c - 1)])
            if 0 <= nr < 4 and 0 <= nc < 8:
                cells.add((nr, nc))
        type_list = []
        counts = random.sample([1, 2, 3, 4, 5, 6], 4)
        total = len(cells)
        counts = [max(1, round(c * total / sum(counts))) for c in counts]
        while sum(counts) < total:
            counts[counts.index(max(counts))] += 1
        while sum(counts) > total:
            i = counts.index(max(counts))
            if counts[i] > 1:
                counts[i] -= 1
            else:
                break
        if len(set(counts)) < 4:
            continue
        for t, cnt in enumerate(counts):
            type_list.extend([t] * cnt)
        random.shuffle(type_list)
        for (cr, cc), t in zip(cells, type_list):
            shape, _color = SHAPES[t]
            ar, ac = cr * 2, cc * 2
            for dr, dc in shape:
                grid[ar + dr, ac + dc] = 5
        try:
            main(grid)
        except Exception:
            continue
        return grid
    raise RuntimeError("failed to generate input")
