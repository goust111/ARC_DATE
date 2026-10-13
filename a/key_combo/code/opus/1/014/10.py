import random
from collections import deque
import numpy as np


GLYPH_COLORS = {
    ((0, 1), (1, 1), (2, 1)): 1,
    ((1, 0), (1, 1), (1, 2)): 3,
    ((0, 0), (1, 1), (2, 2)): 7,
    ((0, 2), (1, 1), (2, 0)): 6,
}


def _components(grid, color):
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or grid[r, c] != color:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            comp = []
            while q:
                x, y = q.popleft()
                comp.append((x, y))
                for dx in (-1, 0, 1):
                    for dy in (-1, 0, 1):
                        if dx == 0 and dy == 0:
                            continue
                        nx, ny = x + dx, y + dy
                        if 0 <= nx < h and 0 <= ny < w and not seen[nx, ny] and grid[nx, ny] == color:
                            seen[nx, ny] = True
                            q.append((nx, ny))
            comps.append(comp)
    return comps


def _bbox(comp):
    rs = [r for r, _ in comp]
    cs = [c for _, c in comp]
    return min(rs), min(cs), max(rs), max(cs)


def _glyph_key(grid, cells):
    r0, c0, r1, c1 = _bbox(cells)
    block = grid[r0:r1 + 1, c0:c1 + 1]
    if block.shape != (3, 3):
        pad = np.zeros((3, 3), dtype=int)
        pad[:min(3, block.shape[0]), :min(3, block.shape[1])] = block[:min(3, block.shape[0]), :min(3, block.shape[1])]
        block = pad
    return tuple(sorted(map(tuple, np.argwhere(block == 5).tolist())))


def main(grid: np.ndarray) -> np.ndarray:
    out = np.array(grid, copy=True)
    axes = _components(grid, 2)
    yellows = _components(grid, 4)
    gray = _components(grid, 5)
    for axis in axes:
        ar0, ac0, ar1, ac1 = _bbox(axis)
        if ac0 != ac1:
            continue
        col = ac0
        candidates = []
        for comp in yellows:
            yr0, yc0, yr1, yc1 = _bbox(comp)
            if yr1 >= ar0 - 2 and yr0 <= ar1 + 2 and (yc1 == col - 1 or yc0 == col + 1):
                candidates.append(comp)
        if not candidates:
            continue
        sprite = max(candidates, key=len)
        reflected = [(r, 2 * col - c) for r, c in sprite]
        ref_set = set(reflected)
        glyphs = [comp for comp in gray if any((r, c) in ref_set for r, c in comp)]
        if not glyphs:
            continue
        color = GLYPH_COLORS.get(_glyph_key(grid, max(glyphs, key=len)), 1)
        for r, c in reflected:
            if 0 <= r < out.shape[0] and 0 <= c < out.shape[1]:
                out[r, c] = color
    return out


def generate_input() -> np.ndarray:
    h, w = 15, 18
    g = np.zeros((h, w), dtype=int)
    glyphs = list(GLYPH_COLORS.keys())
    shape_bank = [
        [(0, 1), (1, 1), (1, 2), (2, 1), (2, 2)],
        [(0, 1), (1, 1), (1, 2), (2, 1), (3, 1)],
        [(0, 1), (0, 2), (1, 1), (2, 1), (2, 2)],
        [(0, 1), (1, 1), (2, 1), (2, 2), (3, 2)],
    ]
    axis_cols = random.sample([5, 7, 10, 12], random.randint(1, 2))
    for col in axis_cols:
        r0 = random.randint(2, 8)
        r1 = min(h - 2, r0 + random.randint(3, 5))
        g[r0:r1 + 1, col] = 2
        side = random.choice([-1, 1])
        raw = random.choice(shape_bank)
        shape = [(r, side * c) for r, c in raw]
        sprite = [(r0 + 1 + r, col + dc) for r, dc in shape]
        if any(not (0 <= r < h and 0 <= c < w) for r, c in sprite):
            continue
        for r, c in sprite:
            g[r, c] = 4
        reflected = [(r, 2 * col - c) for r, c in sprite]
        gr0 = min(r for r, _ in reflected)
        gc0 = min(c for _, c in reflected)
        key = random.choice(glyphs)
        for r, c in key:
            rr, cc = gr0 + r, gc0 + c
            if 0 <= rr < h and 0 <= cc < w:
                g[rr, cc] = 5
    return g
