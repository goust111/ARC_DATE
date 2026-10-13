import random

import numpy as np


def _block(grid, br, bc, b):
    r, c = br * (b + 1), bc * (b + 1)
    return grid[r:r + b, c:c + b], r, c


def _infer_sprite(grid, b):
    for br in range(3):
        for bc in range(3):
            block, _, _ = _block(grid, br, bc, b)
            if np.any(block == 1) and not np.any(block == 5):
                pts = [(r, c) for r in range(b) for c in range(b) if block[r, c] == 1]
                cr = sum(r for r, _ in pts) / len(pts)
                cc = sum(c for _, c in pts) / len(pts)
                mid = (b - 1) / 2
                if abs(cr - mid) > abs(cc - mid):
                    direction = (1, 0) if cr > mid else (-1, 0)
                else:
                    direction = (0, 1) if cc > mid else (0, -1)
                return pts, direction
    return [(1, 1), (1, 2), (1, 3), (2, 2)], (0, 1)


def main(grid: np.ndarray) -> np.ndarray:
    grid = np.array(grid, copy=True)
    out = grid.copy()
    b = (grid.shape[0] - 2) // 3
    sprite, direction = _infer_sprite(grid, b)
    states = {}
    for br in range(3):
        for bc in range(3):
            block, r0, c0 = _block(grid, br, bc, b)
            states[(br, bc)] = "sprite" if np.any(block == 1) else "blank"
            if states[(br, bc)] == "blank" and np.any(block == 3):
                gr, gc = next((r, c) for r in range(b) for c in range(b) if block[r, c] == 3)
                out[r0 + gr, c0 + gc] = 2
                dr, dc = direction
                if dr == -1:
                    target = (r0 - 1, c0 + b // 2)
                elif dr == 1:
                    target = (r0 + b, c0 + b // 2)
                elif dc == -1:
                    target = (r0 + b // 2, c0 - 1)
                else:
                    target = (r0 + b // 2, c0 + b)
                tr, tc = target
                if 0 <= tr < out.shape[0] and 0 <= tc < out.shape[1]:
                    out[tr, tc] = 3
    for br in range(3):
        for bc in range(3):
            block, _, _ = _block(out, br, bc, b)
            gray = [(r, c) for r in range(b) for c in range(b) if block[r, c] == 5]
            if states[(br, bc)] == "sprite":
                block[:, :] = 2
            else:
                block[:, :] = 0
                for r, c in sprite:
                    if 0 <= r < b and 0 <= c < b:
                        block[r, c] = 1
            for r, c in gray:
                block[r, c] = 5
    return out


def generate_input() -> np.ndarray:
    b = 4
    grid = np.zeros((3 * b + 2, 3 * b + 2), dtype=int)
    sprite = [(1, 1), (1, 2), (1, 3), (2, 2)]
    cells = [(r, c) for r in range(3) for c in range(3)]
    sprite_cells = set(random.sample(cells, random.randint(3, 5)))
    blank_cells = [p for p in cells if p not in sprite_cells]
    if len(blank_cells) < 3:
        blank_cells = [p for p in cells if p not in list(sprite_cells)[:3]]
        sprite_cells = set(list(sprite_cells)[:3])
    content_cells = random.sample(blank_cells, min(random.randint(2, 3), len(blank_cells)))
    for br, bc in cells:
        block, _, _ = _block(grid, br, bc, b)
        if (br, bc) in sprite_cells:
            for r, c in sprite:
                block[r, c] = 1
        else:
            block[:, :] = 2
            if (br, bc) in content_cells:
                block[b // 2, b // 2] = 3
        if random.random() < 0.25:
            block[random.randrange(b), random.randrange(b)] = 5
    return grid
