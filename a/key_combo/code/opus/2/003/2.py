import random
import numpy as np


def _mask_from_points(points):
    pts = np.array(points)
    r0, c0 = pts.min(axis=0)
    r1, c1 = pts.max(axis=0)
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    for r, c in pts:
        mask[r - r0, c - c0] = True
    return (int(r0), int(c0), mask)


def _toposort(colors, edges):
    indeg = {c: 0 for c in colors}
    adj = {c: [] for c in colors}
    for a, b in edges:
        if a != b and b not in adj[a]:
            adj[a].append(b)
            indeg[b] += 1
    order = []
    ready = sorted([c for c in colors if indeg[c] == 0])
    while ready:
        c = ready.pop(0)
        order.append(c)
        for d in adj[c]:
            indeg[d] -= 1
            if indeg[d] == 0:
                ready.append(d)
                ready.sort()
    return order if len(order) == len(colors) else list(colors)


def main(grid: np.ndarray) -> np.ndarray:
    colors = [int(c) for c in np.unique(grid) if c != 0]
    pieces = {}
    max_h = max_w = 0
    for color in colors:
        pts = np.argwhere(grid == color)
        if len(pts):
            r0, c0, mask = _mask_from_points(pts)
            pieces[color] = (r0, c0, mask)
            max_h = max(max_h, mask.shape[0])
            max_w = max(max_w, mask.shape[1])
    full = np.zeros((max_h, max_w), dtype=bool)
    for _, _, mask in pieces.values():
        full[: mask.shape[0], : mask.shape[1]] |= mask
    restored = {}
    for color, (r0, c0, _) in pieces.items():
        restored[color] = {(r0 + r, c0 + c) for r, c in zip(*np.where(full))}
    edges = set()
    for i, a in enumerate(colors):
        for b in colors[i + 1:]:
            for r, c in restored[a] & restored[b]:
                seen = int(grid[r, c]) if 0 <= r < grid.shape[0] and 0 <= c < grid.shape[1] else 0
                if seen == a:
                    edges.add((b, a))
                    break
                if seen == b:
                    edges.add((a, b))
                    break
    order = _toposort(colors, edges)
    n = max(full.shape)
    square = np.zeros((n, n), dtype=bool)
    square[: full.shape[0], : full.shape[1]] = full
    out = np.zeros((2 * n, 2 * n), dtype=int)
    rotations = [square, np.rot90(square, 1), np.rot90(square, 2), np.rot90(square, 3)]
    offsets = [(0, 0), (0, n), (n, 0), (n, n)]
    for color, mask, (ro, co) in zip(order, rotations, offsets):
        out[ro : ro + n, co : co + n][mask] = color
    return out


def _random_template(n):
    mask = np.zeros((n, n), dtype=bool)
    mask[0, random.randrange(n)] = True
    mask[-1, random.randrange(n)] = True
    mask[random.randrange(n), 0] = True
    mask[random.randrange(n), -1] = True
    for _ in range(n * n // 2 + 2):
        mask[random.randrange(n), random.randrange(n)] = True
    return mask


def generate_input() -> np.ndarray:
    h = w = random.randint(13, 16)
    grid = np.zeros((h, w), dtype=int)
    n = random.randint(3, 5)
    template = _random_template(n)
    colors = random.sample([1, 2, 3, 4, 5, 6], 4)
    offsets = [(4, 4), (4, 5), (5, 4), (5, 5)]
    random.shuffle(offsets)
    order = colors[:]
    random.shuffle(order)
    for color in order:
        idx = colors.index(color)
        r0, c0 = offsets[idx]
        keep = template.copy()
        holes = list(zip(*np.where(template)))
        random.shuffle(holes)
        for r, c in holes[: max(1, len(holes) // 3)]:
            if keep.sum() > 5 and not (r in (0, n - 1) or c in (0, n - 1)):
                keep[r, c] = False
        for r, c in zip(*np.where(keep)):
            grid[r0 + r, c0 + c] = color
    return grid
