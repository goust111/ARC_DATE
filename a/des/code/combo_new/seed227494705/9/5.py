import random
from collections import deque

import numpy as np


BG = 0
GREY = 5
MAROON = 9


def _components(mask):
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            pts = []
            while q:
                cr, cc = q.popleft()
                pts.append((cr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            yield pts


def _split_row(g):
    for r in range(g.shape[0]):
        if np.all(g[r, :] == GREY):
            return r
    return g.shape[0] // 2


def _templates(g, split):
    result = {}
    top = g[:split, :]
    for color in [int(x) for x in np.unique(top) if int(x) not in (BG, GREY, MAROON)]:
        pts = np.argwhere(top == color)
        if pts.size == 0:
            continue
        r0, c0 = pts.min(axis=0)
        r1, c1 = pts.max(axis=0)
        result[color] = (top[r0 : r1 + 1, c0 : c1 + 1] == color)
    return result


def main(grid: np.ndarray) -> np.ndarray:
    """
    Stamp color-matched templates on seeds using obstacle-selected rotation.

    Input structure:
    - A grey divider separates an upper template library from a lower canvas.
    - Each template is a small monochrome shape whose color also appears as canvas seed pixels.
    - Maroon obstacle blocks on the canvas constrain template rotation.

    Rule:
    - Match each seed to the template of the same color.
    - Center each possible 0/90/180/270-degree rotation on the seed and reject rotations overlapping maroon obstacles.
    - Stamp the valid rotation with greatest distance from obstacles, clipping at the canvas boundary and leaving the library unchanged.

    Key insight:
    - Color selects the template; obstacle geometry selects the orientation.
    """
    g = np.asarray(grid, dtype=int)
    split = _split_row(g)
    templates = _templates(g, split)
    out = g.copy()
    canvas = g[split + 1 :, :]
    obstacles = {tuple(map(int, p)) for p in np.argwhere(canvas == MAROON)}
    for color, mask in templates.items():
        seed_pts = [tuple(map(int, p)) for p in np.argwhere(canvas == color)]
        for seed in seed_pts:
            if any(abs(seed[0] - o[0]) + abs(seed[1] - o[1]) == 1 for o in obstacles):
                pass
            best = None
            for k in range(4):
                rot = np.rot90(mask, k)
                ph, pw = rot.shape
                top = seed[0] - ph // 2
                left = seed[1] - pw // 2
                cells = []
                bad = False
                for rr, cc in np.argwhere(rot):
                    r, c = top + int(rr), left + int(cc)
                    if 0 <= r < canvas.shape[0] and 0 <= c < canvas.shape[1]:
                        if (r, c) in obstacles:
                            bad = True
                            break
                        cells.append((r, c))
                if bad:
                    continue
                if obstacles and cells:
                    score = min(abs(r - o[0]) + abs(c - o[1]) for r, c in cells for o in obstacles)
                else:
                    score = 99
                if best is None or score > best[0]:
                    best = (score, cells)
            if best is None:
                continue
            for r, c in best[1]:
                out[split + 1 + r, c] = color
    return out


def _shape(rng, size):
    cells = {(size // 2, size // 2)}
    while len(cells) < rng.randint(3, size + 2):
        r, c = rng.choice(list(cells))
        for nr, nc in rng.sample([(r + 1, c), (r - 1, c), (r, c + 1), (r, c - 1)], 4):
            if 0 <= nr < size and 0 <= nc < size:
                cells.add((nr, nc))
                break
    return cells


def generate_input() -> np.ndarray:
    rng = random.Random()
    h = rng.randint(22, 30)
    w = rng.randint(18, 30)
    split = rng.randint(7, 10)
    g = np.zeros((h, w), dtype=int)
    g[split, :] = GREY
    colors = rng.sample([1, 2, 3, 4, 6, 7, 8], rng.randint(2, 4))
    col = 1
    templates = {}
    for color in colors:
        size = rng.choice([3, 4])
        cells = _shape(rng, size)
        if col + size >= w:
            break
        for r, c in cells:
            g[1 + r, col + c] = color
        templates[color] = (size, cells)
        col += size + 2
    for color in templates:
        sr = rng.randint(split + 3, h - 4)
        sc = rng.randint(3, w - 4)
        g[sr, sc] = color
        for _ in range(rng.randint(1, 3)):
            orow = min(max(sr + rng.choice([-3, -2, 2, 3]), split + 1), h - 2)
            ocol = min(max(sc + rng.choice([-3, -2, 2, 3]), 1), w - 2)
            g[orow : min(h, orow + 2), ocol : min(w, ocol + 2)] = MAROON
    return g
