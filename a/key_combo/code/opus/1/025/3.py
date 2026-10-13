import random
import numpy as np


def _components(mask):
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
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
    out = grid.copy()
    objs = []
    for color in [int(v) for v in np.unique(grid) if v != 0]:
        pts = np.argwhere(grid == color)
        r0, c0 = pts.min(axis=0)
        r1, c1 = pts.max(axis=0)
        mask = grid[r0:r1 + 1, c0:c1 + 1] == color
        objs.append({"color": color, "box": (int(r0), int(r1), int(c0), int(c1)), "mask": mask})
    if not objs:
        return out
    mh = max(o["mask"].shape[0] for o in objs)
    mw = max(o["mask"].shape[1] for o in objs)
    template = np.zeros((mh, mw), dtype=bool)
    for obj in objs:
        m = obj["mask"]
        template[:m.shape[0], :m.shape[1]] |= m
    fills = [3, 4, 2]
    missing = []
    for obj in objs:
        r0, r1, c0, c1 = obj["box"]
        m = obj["mask"]
        need = template[:m.shape[0], :m.shape[1]]
        miss = int(np.count_nonzero(need & ~m))
        missing.append((miss, obj))
        out[r0:r1 + 1, c0:c1 + 1][need] = obj["color"]
    for rank, (_, obj) in enumerate(sorted(missing, key=lambda x: x[0])):
        r0, r1, c0, c1 = obj["box"]
        out[r0 + 1:r1, c0 + 1:c1] = fills[rank]
    return out


def generate_input() -> np.ndarray:
    h = w = random.randint(16, 20)
    size = random.randint(5, 7)
    border = [(0, c) for c in range(size)] + [(size - 1, c) for c in range(size)]
    border += [(r, 0) for r in range(1, size - 1)] + [(r, size - 1) for r in range(1, size - 1)]
    positions = [(1, 1), (1, w - size - 1), (h - size - 1, random.randint(2, w - size - 2))]
    colors = random.sample([1, 3, 4, 6, 8, 9], 3)
    misses = random.sample(range(2, min(7, len(border))), 3)
    g = np.zeros((h, w), dtype=int)
    for (r0, c0), color, miss in zip(positions, colors, misses):
        start = random.randrange(len(border))
        missing = {border[(start + k) % len(border)] for k in range(miss)}
        for r, c in border:
            if (r, c) not in missing:
                g[r0 + r, c0 + c] = color
    return g
