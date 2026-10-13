import random
from collections import deque

import numpy as np


def _components(mask):
    seen = np.zeros(mask.shape, dtype=bool)
    comps = []
    for start in np.argwhere(mask):
        sr, sc = map(int, start)
        if seen[sr, sc]:
            continue
        q = deque([(sr, sc)])
        seen[sr, sc] = True
        comp = []
        while q:
            r, c = q.popleft()
            comp.append((r, c))
            for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                if 0 <= nr < mask.shape[0] and 0 <= nc < mask.shape[1] and mask[nr, nc] and not seen[nr, nc]:
                    seen[nr, nc] = True
                    q.append((nr, nc))
        comps.append(comp)
    return comps


def _bbox(comp):
    rs = [p[0] for p in comp]
    cs = [p[1] for p in comp]
    return min(rs), max(rs) + 1, min(cs), max(cs) + 1


def main(grid: np.ndarray) -> np.ndarray:
    div = max(range(grid.shape[0]), key=lambda r: np.count_nonzero(grid[r] == 5))
    top = grid[:div]
    bottom = grid[div + 1 :]
    out = bottom.copy()
    top_objs = {}
    for comp in _components((top != 0) & (top != 5)):
        r0, r1, c0, c1 = _bbox(comp)
        top_objs[(r0, c0)] = top[r0:r1, c0:c1]

    for comp in _components((bottom != 0) & (bottom != 5)):
        r0, r1, c0, c1 = _bbox(comp)
        b = bottom[r0:r1, c0:c1]
        t = top_objs.get((r0, c0))
        if t is None:
            continue
        colors = sorted(int(x) for x in np.unique(b) if x not in (0, 5))
        if len(colors) != 2:
            continue
        a, z = colors
        top_major = a if np.count_nonzero(t == a) > np.count_nonzero(t == z) else z
        bot_major = a if np.count_nonzero(b == a) > np.count_nonzero(b == z) else z
        if top_major != bot_major:
            nb = b.copy()
            nb[b == a] = z
            nb[b == z] = a
            out[r0:r1, c0:c1] = nb
    return out


def _object(colors, majority):
    a, b = colors
    maj_count = random.choice([5, 6, 7, 8])
    vals = [majority] * maj_count + [b if majority == a else a] * (9 - maj_count)
    random.shuffle(vals)
    return np.array(vals, dtype=int).reshape(3, 3)


def generate_input() -> np.ndarray:
    half = random.choice([6, 7, 8])
    w = random.choice([10, 11, 12])
    grid = np.zeros((2 * half + 1, w), dtype=int)
    grid[half, :] = 5
    candidates = [(r, c) for r in range(half - 2) for c in range(w - 2)]
    random.shuffle(candidates)
    positions = []
    for r, c in candidates:
        if all(abs(r - pr) >= 3 or abs(c - pc) >= 3 for pr, pc in positions):
            positions.append((r, c))
            if len(positions) == random.randint(2, 4):
                break
    if len(positions) < 2:
        positions = [(0, 0), (0, w - 3)]

    colors = random.sample([1, 2, 3, 4, 6, 7, 8, 9], 2)
    flips = [True, False] + [random.choice([True, False]) for _ in positions[2:]]
    random.shuffle(flips)
    for (r, c), flip in zip(positions, flips):
        top_major = random.choice(colors)
        bot_major = colors[1] if colors[0] == top_major else colors[0] if flip else top_major
        grid[r : r + 3, c : c + 3] = _object(colors, top_major)
        grid[half + 1 + r : half + 1 + r + 3, c : c + 3] = _object(colors, bot_major)
    return grid
