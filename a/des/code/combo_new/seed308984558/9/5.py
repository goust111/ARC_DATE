import random
from collections import deque

import numpy as np


BLACK = 0
COLORS = [1, 2, 3, 4, 6, 7, 8, 9]


def _components(grid, color):
    mask = grid == color
    h, w = grid.shape
    seen = np.zeros_like(mask, dtype=bool)
    out = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            cells = []
            while q:
                cr, cc = q.popleft()
                cells.append((cr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            out.append(cells)
    return out


def _bbox(cells):
    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]
    return min(rows), min(cols), max(rows), max(cols)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Stack all solid squares concentrically onto the largest square's position.

    Input structure:
    - The grid contains several disjoint solid colored square components on black.
    - Every square has a distinct color and even side length.

    Rule:
    - Identify all solid square components and sort them by descending area.
    - Use the largest square's center as the common center for the whole output stack.
    - Keep the largest square in place, then overlay each smaller square centered on
      the same point using its own original color.
    - Erase every smaller square from its original location.

    Key insight:
    - The objects are not independent final answers; they encode a size ordering that
      must be collapsed onto one shared center, turning scattered solids into nested
      concentric color layers.
    """
    squares = []
    for color in [int(v) for v in np.unique(grid) if v != BLACK]:
        comp = _components(grid, color)[0]
        r0, c0, r1, c1 = _bbox(comp)
        side = r1 - r0 + 1
        squares.append((side * side, side, color, r0, c0))
    squares.sort(reverse=True)
    out = np.zeros_like(grid)
    _, max_side, _, mr0, mc0 = squares[0]
    for _, side, color, _, _ in squares:
        off = (max_side - side) // 2
        out[mr0 + off : mr0 + off + side, mc0 + off : mc0 + off + side] = color
    return out


def generate_input() -> np.ndarray:
    h = random.randint(16, 24)
    w = random.randint(16, 24)
    grid = np.zeros((h, w), dtype=int)
    sides = sorted(random.sample([2, 4, 6, 8], random.randint(2, 4)), reverse=True)
    colors = random.sample(COLORS, len(sides))
    placed = []
    for side, color in zip(sides, colors):
        for _ in range(100):
            r0 = random.randint(0, h - side)
            c0 = random.randint(0, w - side)
            if all(r0 + side - 1 < pr0 or pr1 < r0 or c0 + side - 1 < pc0 or pc1 < c0 for pr0, pc0, pr1, pc1 in placed):
                grid[r0 : r0 + side, c0 : c0 + side] = color
                placed.append((r0, c0, r0 + side - 1, c0 + side - 1))
                break
    return grid
