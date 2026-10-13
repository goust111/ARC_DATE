import random
import numpy as np


def main(grid: np.ndarray) -> np.ndarray:
    g = np.array(grid, copy=True)
    h, w = g.shape
    frame = np.argwhere(g == 5)
    if frame.size == 0:
        return g
    r0, c0 = frame.min(axis=0)
    r1, c1 = frame.max(axis=0)

    original = np.array(grid, copy=True)
    paired = set()
    for r in range(h):
        for c in range(w):
            if original[r, c] in (0, 5):
                continue
            mc = w - 1 - c
            if original[r, mc] == original[r, c] and original[r, mc] != 5:
                paired.add((r, c))

    palette = []
    seen = set()
    for r in range(h):
        for c in range(w):
            if (r, c) in paired and original[r, c] not in seen:
                seen.add(int(original[r, c]))
                palette.append(int(original[r, c]))

    for r in range(h):
        for c in range(w):
            if original[r, c] not in (0, 5) and (r, c) not in paired:
                g[r, c] = 0

    inner_h = max(0, r1 - r0 - 1)
    inner_w = max(0, c1 - c0 - 1)
    if not palette or inner_h == 0 or inner_w == 0:
        return g
    for r in range(r0 + 1, r1):
        for c in range(c0 + 1, c1):
            layer = min(r - r0 - 1, r1 - 1 - r, c - c0 - 1, c1 - 1 - c)
            g[r, c] = palette[min(layer, len(palette) - 1)]
    return g


def generate_input() -> np.ndarray:
    h = random.randint(14, 18)
    w = random.choice([14, 16, 18])
    g = np.zeros((h, w), dtype=int)

    inner_h = random.randint(5, 7)
    inner_w = random.randint(5, 7)
    r0 = random.randint(4, max(4, h - inner_h - 3))
    c0 = random.randint(2, max(2, w - inner_w - 4))
    r1 = r0 + inner_h + 1
    c1 = c0 + inner_w + 1
    g[r0:r1 + 1, c0] = 5
    g[r0:r1 + 1, c1] = 5
    g[r0, c0:c1 + 1] = 5
    g[r1, c0:c1 + 1] = 5

    layers = (min(inner_h, inner_w) + 1) // 2
    palette = random.sample([1, 2, 3, 4], layers)
    used = set(map(tuple, np.argwhere(g != 0)))

    left_cols = list(range(1, w // 2 - 1))
    random.shuffle(left_cols)
    rows = list(range(1, r0 - 1)) + list(range(r1 + 2, h - 1))
    random.shuffle(rows)
    placements = []
    for color in palette:
        placed = False
        for r in rows:
            for c in left_cols:
                m = w - 1 - c
                if (r, c) not in used and (r, m) not in used:
                    g[r, c] = color
                    g[r, m] = color
                    used.add((r, c))
                    used.add((r, m))
                    placements.append((r, c))
                    placed = True
                    break
            if placed:
                break

    for _ in range(random.randint(2, 5)):
        for _attempt in range(100):
            r = random.randrange(h)
            c = random.randrange(w)
            if (r, c) in used or g[r, c] != 0:
                continue
            mc = w - 1 - c
            color = random.choice([1, 2, 3, 4])
            if g[r, mc] == color:
                continue
            g[r, c] = color
            used.add((r, c))
            break
    return g
